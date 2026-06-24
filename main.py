"""
simple_pyworker.py
==================

Vast.ai Serverless ile uyumlu, AMA çok daha basit bir PyWorker backend'i.

NE KORUNDU (Vast'ın autoscaler/routing'i bunları bekliyor, atılamaz):
  - worker_status periyodik push'u (report_addr'a)   -> Metrics._send_metrics_loop
  - delete_requests push'u                            -> Metrics._send_delete_requests_loop
  - /session/create, /session/get, /session/health, /session/end route'ları
  - /pyworker/update (Bearer mtoken ile)
  - pubkey fetch + auth_data.signature doğrulaması    -> __check_signature

NE BASİTLEŞTİRİLDİ:
  - Model-hazır tespiti: log-tailing/regex/state-machine YOK.
    Sadece MODEL_HEALTH_URL'i sabit aralıkla (0.3sn) polluyor, 200 dönünce hazır.
  - Benchmark: GERÇEK BENCHMARK YOK. max_throughput sabit bir değer (bkz. FAKE_MAX_THROUGHPUT).
  - Model isteği: streaming YOK, basit JSON passthrough (request.payload -> model'e POST -> JSON cevap).
  - Devam eden (post-ready) crash detection YOK.
  - İkinci (internal/HTTP-only) webhook listener YOK.

Bu üç eksiği istersen sonradan ekleriz; şimdilik "basit ve çalışır" öncelikli.
"""

import os
import ssl
import json
import time
import base64
import logging
import secrets
import asyncio
from asyncio import sleep
from dataclasses import dataclass, field, asdict
from functools import cached_property
from typing import Optional, Dict, Any, List

from aiohttp import web, ClientSession, ClientTimeout, TCPConnector, ClientResponseError, ClientConnectorError
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("simple_pyworker")
log.info("=== simple_pyworker VERSION 2026-06-24-imza-debug-v3 ===")

# ---------------------------------------------------------------------------
# Sabitler / Config (env var'lardan okunuyor)
# ---------------------------------------------------------------------------

CONTAINER_ID = int(os.environ.get("CONTAINER_ID", "0"))
REPORT_ADDR_LIST = os.environ["REPORT_ADDR"].split(",")
WORKER_PORT = int(os.environ["WORKER_PORT"])
USE_SSL = os.environ.get("USE_SSL", "false").lower() == "true"
UNSECURED = os.environ.get("UNSECURED", "false").lower() == "true"
MASTER_TOKEN = os.environ.get("MASTER_TOKEN", "")

MODEL_HEALTH_URL = os.environ.get("MODEL_HEALTH_URL", "http://127.0.0.1:5000/health")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:5000")

FAKE_MAX_THROUGHPUT = float(os.environ.get("FAKE_MAX_THROUGHPUT", "100.0"))  # benchmark yok, sabit değer
HEALTHCHECK_POLL_INTERVAL = 0.3
METRICS_UPDATE_INTERVAL = 1
DELETE_REQUESTS_INTERVAL = 1
SESSION_GC_INTERVAL = 5.0
DEFAULT_SESSION_LIFETIME = 60.0


def get_url() -> str:
    worker_port_ext = os.environ[f"VAST_TCP_PORT_{WORKER_PORT}"]
    public_ip = os.environ["PUBLIC_IPADDR"]
    return f"http{'s' if USE_SSL else ''}://{public_ip}:{worker_port_ext}"


# ---------------------------------------------------------------------------
# Veri tipleri
# ---------------------------------------------------------------------------

@dataclass
class AuthData:
    cost: str
    endpoint: str
    reqnum: int
    request_idx: int
    signature: str
    url: str

    @staticmethod
    def from_dict(d: dict) -> "AuthData":
        return AuthData(
            cost=d.get("cost"),
            endpoint=d.get("endpoint"),
            reqnum=d.get("reqnum"),
            request_idx=d.get("request_idx"),
            signature=d["signature"],
            url=d.get("url") or get_url(),  # url body'de gelmiyorsa worker'ın kendi adresi (imza bunun üzerine atılıyor)
        )


@dataclass
class Session:
    session_id: str
    auth_data: dict
    lifetime: float
    created_at: float
    expiration: float
    on_close_route: Optional[str] = None
    on_close_payload: Optional[dict] = None


# ---------------------------------------------------------------------------
# Metrics (Vast autoscaler'a worker_status / delete_requests gönderen kısım)
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    mtoken: str = ""
    version: str = "simple-0.1"
    model_is_loaded: bool = False
    model_loading_start: float = field(default_factory=time.time)
    loadtime: Optional[float] = None
    max_throughput: float = 0.0
    error_msg: str = ""
    update_pending: bool = False
    last_metric_update: float = 0.0
    requests_working: Dict[int, Any] = field(default_factory=dict)
    requests_deleting_success: List[int] = field(default_factory=list)
    requests_deleting_failed: List[int] = field(default_factory=list)
    _session: Optional[ClientSession] = field(default=None, init=False, repr=False)

    async def http(self) -> ClientSession:
        if self._session is None:
            self._session = ClientSession(
                timeout=ClientTimeout(total=10),
                connector=TCPConnector(limit=8, limit_per_host=4, force_close=True),
            )
        return self._session

    def model_loaded(self, max_throughput: float) -> None:
        self.loadtime = time.time() - self.model_loading_start
        self.model_is_loaded = True
        self.max_throughput = max_throughput
        self.update_pending = True  # bir sonraki tick'te HEMEN bildir (bkz. daha önceki sohbet)
        log.debug(f"model_is_loaded=True, loadtime={self.loadtime:.3f}s, hemen bildiriliyor")

    def request_start(self, reqnum: int) -> None:
        self.requests_working[reqnum] = True
        self.update_pending = True

    def request_end(self, reqnum: int, success: bool) -> None:
        self.requests_working.pop(reqnum, None)
        if success:
            self.requests_deleting_success.append(reqnum)
        else:
            self.requests_deleting_failed.append(reqnum)
        self.update_pending = True

    async def _send_metrics_loop(self) -> None:
        while True:
            await sleep(METRICS_UPDATE_INTERVAL)
            elapsed = time.time() - self.last_metric_update
            if (not self.model_is_loaded and elapsed >= 10) or self.update_pending or elapsed > 10:
                await self.__send_worker_status()

    async def __send_worker_status(self) -> None:
        data = {
            "id": CONTAINER_ID,
            "mtoken": self.mtoken,
            "version": self.version,
            "loadtime": self.loadtime or 0.0,
            "error_msg": self.error_msg,
            "max_perf": self.max_throughput,
            "num_requests_working": len(self.requests_working),
            "url": get_url(),
        }
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            full_path = addr.rstrip("/") + "/worker_status/"
            try:
                async with session.post(full_path, json=data) as res:
                    res.raise_for_status()
                log.debug(f"worker_status gönderildi (model_is_loaded={self.model_is_loaded}, loadtime={self.loadtime})")
                self.update_pending = False
                self.last_metric_update = time.time()
                return
            except Exception as e:
                log.debug(f"worker_status gönderilemedi ({addr}): {e}")

    async def _send_delete_requests_loop(self) -> None:
        while True:
            await sleep(DELETE_REQUESTS_INTERVAL)
            if self.requests_deleting_success or self.requests_deleting_failed:
                await self.__send_delete_requests()

    async def __send_delete_requests(self) -> None:
        success = self.requests_deleting_success[:]
        failed = self.requests_deleting_failed[:]
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            full_path = addr.rstrip("/") + "/delete_requests/"
            ok = True
            try:
                if success:
                    async with session.post(full_path, json={"worker_id": CONTAINER_ID, "mtoken": self.mtoken, "request_idxs": success, "success": True}) as res:
                        res.raise_for_status()
                if failed:
                    async with session.post(full_path, json={"worker_id": CONTAINER_ID, "mtoken": self.mtoken, "request_idxs": failed, "success": False}) as res:
                        res.raise_for_status()
            except Exception as e:
                log.debug(f"delete_requests gönderilemedi ({addr}): {e}")
                ok = False
            if ok:
                self.requests_deleting_success = [r for r in self.requests_deleting_success if r not in success]
                self.requests_deleting_failed = [r for r in self.requests_deleting_failed if r not in failed]
                return


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Backend:
    def __init__(self) -> None:
        self.metrics = Metrics(mtoken=MASTER_TOKEN)
        self.sessions: Dict[str, Session] = {}
        self._pubkey: Optional[RSA.RsaKey] = None
        self._pubkey_fetch_complete = asyncio.Event()
        self.max_sessions = int(os.environ.get("MAX_SESSIONS", "8"))
        self._last_api_key = None

    @cached_property
    def session(self) -> ClientSession:
        return ClientSession(timeout=ClientTimeout(total=30))

    # --- Hazır olma tespiti (BASİTLEŞTİRİLMİŞ KISIM) -----------------------

    async def wait_for_model_ready(self) -> None:
        timeout = ClientTimeout(total=5)
        while True:
            try:
                async with self.session.get(MODEL_HEALTH_URL, timeout=timeout) as res:
                    if res.status == 200:
                        break
            except Exception:
                pass
            await sleep(HEALTHCHECK_POLL_INTERVAL)

        log.debug("Model /health 200 döndü -> hazır")
        self.metrics.model_loaded(max_throughput=FAKE_MAX_THROUGHPUT)  # gerçek benchmark yok, sabit değer

    # --- Pubkey / imza doğrulama (DOKUNULMADI) ------------------------------

    async def fetch_pubkey(self) -> None:
        if UNSECURED:
            self._pubkey_fetch_complete.set()
            return
        url = REPORT_ADDR_LIST[0].rstrip("/") + "/pubkey/"
        log.debug(f"Pubkey şu adresten çekiliyor: {url}")
        timeout = ClientTimeout(total=10)
        for attempt in range(1, 6):
            try:
                async with ClientSession(timeout=timeout) as client:
                    async with client.get(url) as response:
                        response.raise_for_status()
                        text = await response.text()
                        self._pubkey = RSA.import_key(text)
                        self._pubkey_fetch_complete.set()
                        log.debug(f"Pubkey alındı (deneme {attempt}), key fingerprint: {text.strip()[:40]}...")
                        return
            except Exception as e:
                log.debug(f"Pubkey alınamadı (deneme {attempt}/5): {e}")
                if attempt < 5:
                    await sleep(2 ** (attempt - 1))
        log.error("Pubkey 5 denemede de alınamadı")
        self._pubkey_fetch_complete.set()

    def check_signature(self, auth_data: AuthData) -> bool:
        if UNSECURED:
            return True
        if self._pubkey is None:
            log.error("İmza reddedildi: pubkey yüklenmemiş")
            return False
        message = json.dumps({"url": auth_data.url}, indent=4, sort_keys=True)
        h = SHA256.new(message.encode())
        try:
            sig_bytes = base64.b64decode(auth_data.signature)
        except Exception as e:
            log.error(f"İmza base64 decode edilemedi: {e}")
            return False
        try:
            pkcs1_15.new(self._pubkey).verify(h, sig_bytes)
            return True
        except (ValueError, TypeError):
            log.error(
                f"İmza doğrulaması başarısız. "
                f"sig string uzunluğu={len(auth_data.signature)}, "
                f"decode byte uzunluğu={len(sig_bytes)} (256 bekleniyor), "
                f"message={message!r}"
            )
            return False

    # --- Session route handler'ları (DOKUNULMADI) ---------------------------

    async def session_create_handler(self, request: web.Request) -> web.Response:
        if len(self.sessions) >= self.max_sessions:
            return web.Response(status=429)
        try:
            data = await request.json()
            auth_data = data["auth_data"]
            payload = data.get("payload", {})
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)

        session_id = secrets.token_urlsafe(12)
        now = time.time()
        lifetime = float(payload.get("lifetime", DEFAULT_SESSION_LIFETIME))
        self.sessions[session_id] = Session(
            session_id=session_id,
            auth_data=auth_data,
            lifetime=lifetime,
            created_at=now,
            expiration=now + lifetime,
            on_close_route=payload.get("on_close_route"),
            on_close_payload=payload.get("on_close_payload"),
        )
        return web.json_response({"session_id": session_id, "expiration": now + lifetime}, status=201)

    async def session_get_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id, session_auth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        session = self.sessions.get(session_id)
        if session is None:
            return web.json_response({"error": "not found"}, status=400)
        if session.auth_data != session_auth:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(asdict(session))

    async def session_health_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id, session_auth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        session = self.sessions.get(session_id)
        if session is not None and session.auth_data != session_auth:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"ok": session is not None})

    async def session_end_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            session_id, session_auth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        session = self.sessions.get(session_id)
        if session is None:
            return web.json_response({"error": "not found"}, status=400)
        if session.auth_data != session_auth:
            return web.json_response({"error": "unauthorized"}, status=401)
        del self.sessions[session_id]
        if session.on_close_route:
            try:
                body = dict(session.on_close_payload or {})
                body["session_id"] = session_id
                async with self.session.post(session.on_close_route, json=body):
                    pass
            except Exception as e:
                log.debug(f"on_close_route webhook başarısız: {e}")
        return web.json_response({"ended": True, "removed_session": session_id})

    async def pyworker_update_handler(self, request: web.Request) -> web.Response:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {MASTER_TOKEN}":
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            open("/.force_update", "w").close()
        except Exception as e:
            return web.json_response({"error": f"failed to write file: {e}"}, status=500)
        return web.json_response({"ok": True})

    # --- Model inference route (BASİTLEŞTİRİLMİŞ: streaming yok) ------------

    def _debug_signature_formats(self, auth_data: dict) -> None:
        """İmzanın GERÇEKTE hangi string üzerine atıldığını bulmak için çok sayıda format dener."""
        try:
            sig = base64.b64decode(auth_data["signature"])
        except Exception as e:
            log.info(f"İMZA TEŞHİSİ: signature decode edilemedi: {e}")
            return
        own_url = get_url()
        body_url = auth_data.get("url")

        public_ip = os.environ.get("PUBLIC_IPADDR", "")
        worker_port_ext = os.environ.get(f"VAST_TCP_PORT_{WORKER_PORT}", "")
        internal_port = str(WORKER_PORT)

        # Çok sayıda url varyasyonu üret
        base_urls = set()
        for u in [own_url, body_url]:
            if u:
                base_urls.add(u)
        for scheme in ["http", "https"]:
            for port in [worker_port_ext, internal_port]:
                if public_ip and port:
                    base_urls.add(f"{scheme}://{public_ip}:{port}")
                    base_urls.add(f"{scheme}://{public_ip}:{port}/")
            if public_ip:
                base_urls.add(f"{scheme}://{public_ip}")

        candidates = {}
        for u in base_urls:
            candidates[f"url={u} | i4 sort"] = json.dumps({"url": u}, indent=4, sort_keys=True)
            candidates[f"url={u} | compact sort"] = json.dumps({"url": u}, sort_keys=True)
            candidates[f"url={u} | ham"] = u

        # api_key query parametresinin içindeki msg'i de aday olarak dene
        api_key_raw = getattr(self, "_last_api_key", None)
        if api_key_raw:
            try:
                decoded = base64.b64decode(api_key_raw + "=" * (-len(api_key_raw) % 4))
                api_obj = json.loads(decoded)
                msg = api_obj.get("msg")
                if msg:
                    candidates["api_key.msg ham"] = msg
                    candidates["api_key.msg i4 sort"] = json.dumps(json.loads(msg), indent=4, sort_keys=True)
                    candidates["api_key.msg compact sort"] = json.dumps(json.loads(msg), sort_keys=True)
            except Exception as e:
                log.info(f"api_key decode edilemedi: {e}")
        for fld in ["request_idx", "reqnum", "endpoint", "cost", "__request_id"]:
            if fld in auth_data:
                v = auth_data[fld]
                candidates[f"{fld}: ham"] = str(v)
                candidates[f"{fld}: dumps i4 sort"] = json.dumps({fld: v}, indent=4, sort_keys=True)
        ad_no_sig = {k: v for k, v in auth_data.items() if k != "signature"}
        candidates["auth_data(sig hariç) i4 sort"] = json.dumps(ad_no_sig, indent=4, sort_keys=True)
        candidates["auth_data(sig hariç) compact sort"] = json.dumps(ad_no_sig, sort_keys=True)
        if own_url:
            candidates["url+request_idx i4 sort"] = json.dumps(
                {"url": own_url, "request_idx": auth_data.get("request_idx")}, indent=4, sort_keys=True
            )

        log.info("=== İMZA FORMAT TEŞHİSİ ===")
        found = False
        for label, message in candidates.items():
            try:
                h = SHA256.new(message.encode())
                pkcs1_15.new(self._pubkey).verify(h, sig)
                log.info(f"  ✅ EŞLEŞTİ -> {label} | message={message!r}")
                found = True
            except (ValueError, TypeError):
                log.info(f"  ❌ {label}")
        if not found:
            log.info("  HİÇBİR ADAY EŞLEŞMEDİ")
        log.info("=== TEŞHİS BİTTİ ===")

    async def model_request_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            auth_data = AuthData.from_dict(data["auth_data"])
            payload = data.get("payload", {})
        except Exception as e:
            return web.json_response({"error": f"invalid request: {e}"}, status=422)

        log.debug(f"Gelen auth_data: {data.get('auth_data')}")
        log.debug(f"Gelen query string: {dict(request.query)}")
        log.debug(f"Worker'ın kendi get_url(): {get_url()}")

        self._last_api_key = request.query.get("api_key")
        if not UNSECURED and self._pubkey is not None:
            self._debug_signature_formats(data["auth_data"])

        if not self.check_signature(auth_data):
            return web.Response(status=401)

        reqnum = auth_data.reqnum
        self.metrics.request_start(reqnum)
        try:
            async with self.session.post(f"{MODEL_BASE_URL}{request.path}", json=payload) as res:
                result = await res.json()
                success = res.status == 200
        except Exception as e:
            self.metrics.request_end(reqnum, success=False)
            return web.json_response({"error": str(e)}, status=502)

        self.metrics.request_end(reqnum, success=success)
        return web.json_response(result, status=200 if success else 502)

    # --- Session garbage collection (basit) ----------------------------------

    async def session_gc_loop(self) -> None:
        while True:
            await sleep(SESSION_GC_INTERVAL)
            now = time.time()
            expired = [sid for sid, s in self.sessions.items() if s.expiration < now]
            for sid in expired:
                del self.sessions[sid]

    # --- Her şeyi başlatan ana gather ----------------------------------------

    async def start_tracking(self) -> None:
        await asyncio.gather(
            self.fetch_pubkey(),
            self.wait_for_model_ready(),
            self.metrics._send_metrics_loop(),
            self.metrics._send_delete_requests_loop(),
            self.session_gc_loop(),
        )


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

async def main() -> None:
    backend = Backend()
    app = web.Application()
    app.router.add_post("/session/create", backend.session_create_handler)
    app.router.add_post("/session/get", backend.session_get_handler)
    app.router.add_post("/session/health", backend.session_health_handler)
    app.router.add_post("/session/end", backend.session_end_handler)
    app.router.add_post("/pyworker/update", backend.pyworker_update_handler)
    app.router.add_post("/v1/completions", backend.model_request_handler)
    app.router.add_post("/v1/chat/completions", backend.model_request_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    ssl_context = None
    if USE_SSL:
        log.debug("Getting SSL Certificate from /etc/instance.crt")
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(certfile="/etc/instance.crt", keyfile="/etc/instance.key")
        except Exception as ex:
            raise Exception(f"Failed to get SSL Certificate: {ex}")

    site = web.TCPSite(runner, "0.0.0.0", WORKER_PORT, ssl_context=ssl_context)
    await site.start()
    log.info(f"Sunucu {WORKER_PORT} portunda dinliyor (SSL={'açık' if USE_SSL else 'kapalı'}, basitleştirilmiş PyWorker)")

    await backend.start_tracking()


if __name__ == "__main__":
    asyncio.run(main())
