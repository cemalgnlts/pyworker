"""
worker.py — Llama.cpp için SADE PyWorker (istek tabanlı ölçek)

Felsefe: vast-ai'nin yaptığı gibi log tail etme, benchmark, karmaşık metrik
muhasebesi YOK. Kendi ihtiyacımız: /health 200 dönünce model hazırdır, o kadar.

İstek tabanlı ölçek:
  Her istek SABİT load taşır. perf sabittir (benchmark yok).
  Bir worker'ı dolduran istek sayısı = WORKER_PERF / WORKLOAD_PER_REQUEST.
  örn: WORKER_PERF=100, REQUESTS_PER_WORKER=4 -> her istek 25 load,
       4 istekte cur_load=100=perf (doluluk %100).

Autoscaler hâlâ Vast (REPORT_ADDR=run.vast.ai) olduğu için worker_status ve
delete_requests'in TELDEKİ formatı Vast'ın beklediği gibi tutuluyor.
"""

import os
import ssl
import json
import time
import base64
import asyncio
import logging
import secrets
from asyncio import sleep
from dataclasses import dataclass, asdict
from functools import cached_property, cache
from typing import Optional, Dict, List

from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pyworker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTAINER_ID = int(os.environ.get("CONTAINER_ID", "0"))
REPORT_ADDR_LIST = os.environ["REPORT_ADDR"].split(",")
WORKER_PORT = int(os.environ["WORKER_PORT"])
USE_SSL = os.environ.get("USE_SSL", "false").lower() == "true"
UNSECURED = os.environ.get("UNSECURED", "false").lower() == "true"
MASTER_TOKEN = os.environ.get("MASTER_TOKEN", "")
PYWORKER_VERSION = os.environ.get("PYWORKER_VERSION", "llamacpp-1.0")

# Model sunucusu adresleri — Vast openai şablonunun KULLANDIĞI env isimleri öncelikli:
#   MODEL_SERVER_URL (örn http://127.0.0.1:5000), MODEL_HEALTH_ENDPOINT (örn http://127.0.0.1:1800/health)
# Yoksa LLAMA_ARG_PORT'tan türet, o da yoksa 8080.
_MODEL_PORT = os.environ.get("LLAMA_ARG_PORT", "8080")
MODEL_BASE_URL = (
    os.environ.get("MODEL_SERVER_URL")
    or os.environ.get("MODEL_BASE_URL")
    or f"http://127.0.0.1:{_MODEL_PORT}"
).rstrip("/")
MODEL_HEALTH_URL = (
    os.environ.get("MODEL_HEALTH_ENDPOINT")
    or os.environ.get("MODEL_HEALTH_URL")
    or f"{MODEL_BASE_URL}/health"
)

# === İstek tabanlı ölçek ===
WORKER_PERF = float(os.environ.get("WORKER_PERF", "100"))
REQUESTS_PER_WORKER = float(os.environ.get("REQUESTS_PER_WORKER", "4"))
WORKLOAD_PER_REQUEST = WORKER_PERF / max(REQUESTS_PER_WORKER, 1e-9)  # 100/4 = 25

MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
DEFAULT_SESSION_LIFETIME = 60.0

METRICS_INTERVAL = 1
HEALTH_POLL_INTERVAL = 1.0


@cache
def get_url() -> str:
    ext = os.environ[f"VAST_TCP_PORT_{WORKER_PORT}"]
    ip = os.environ["PUBLIC_IPADDR"]
    return f"http{'s' if USE_SSL else ''}://{ip}:{ext}"


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
    url: str  # imza bunun üzerine atılı; ASLA yeniden üretme

    @staticmethod
    def from_dict(d: dict) -> "AuthData":
        return AuthData(
            cost=d.get("cost"),
            endpoint=d.get("endpoint"),
            reqnum=int(d.get("reqnum", 0)),
            request_idx=int(d.get("request_idx", 0)),
            signature=d["signature"],
            url=d.get("url", ""),
        )


@dataclass
class RequestInfo:
    request_idx: int
    entered_at: float
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    status: str = "Created"


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
# Metrics
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self) -> None:
        self.model_is_loaded = False
        self.loading_start = time.time()
        self.loadtime = 0.0
        self.loadtime_sent = False
        self.max_perf = 0.0
        self.error_msg = ""
        self.working: Dict[int, RequestInfo] = {}
        self.deleting: List[RequestInfo] = []
        self.update_pending = False
        self.last_update = 0.0
        self._http: Optional[ClientSession] = None

    async def http(self) -> ClientSession:
        if self._http is None:
            self._http = ClientSession(
                timeout=ClientTimeout(total=10),
                connector=TCPConnector(limit=8, force_close=True),
            )
        return self._http

    @property
    def cur_load(self) -> float:
        # İstek tabanlı: yük = açık istek sayısı * sabit load
        return len(self.working) * WORKLOAD_PER_REQUEST

    def model_loaded(self, perf: float) -> None:
        self.loadtime = time.time() - self.loading_start
        self.model_is_loaded = True
        self.max_perf = perf
        self.update_pending = True
        log.info(f"Model hazır. loadtime={self.loadtime:.1f}s, perf={perf}, istek/worker={REQUESTS_PER_WORKER}")

    def start(self, info: RequestInfo) -> None:
        self.working[info.request_idx] = info
        self.update_pending = True

    def end(self, info: RequestInfo, success: bool) -> None:
        info.success = success
        info.status = "Success" if success else "Error"
        info.completed_at = time.time()
        self.working.pop(info.request_idx, None)
        self.deleting.append(info)
        self.update_pending = True

    async def send_loop(self) -> None:
        while True:
            await sleep(METRICS_INTERVAL)
            elapsed = time.time() - self.last_update
            if self.update_pending or elapsed > 10:
                await self._send_status()
            if self.deleting:
                await self._send_deletes()

    async def _send_status(self) -> None:
        loadtime = self.loadtime if (self.model_is_loaded and not self.loadtime_sent) else 0.0
        data = {
            "id": CONTAINER_ID,
            "mtoken": MASTER_TOKEN,
            "version": PYWORKER_VERSION,
            "loadtime": loadtime,
            "error_msg": self.error_msg,
            "max_perf": self.max_perf,
            "cur_load": self.cur_load,
            "num_requests_working": len(self.working),
            "working_request_idxs": list(self.working.keys()),
            "url": get_url(),
        }
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            try:
                async with session.post(addr.rstrip("/") + "/worker_status/", json=data) as res:
                    res.raise_for_status()
                if self.model_is_loaded:
                    self.loadtime_sent = True
                self.update_pending = False
                self.last_update = time.time()
                return
            except Exception as e:
                log.debug(f"worker_status gönderilemedi ({addr}): {e}")

    async def _send_deletes(self) -> None:
        snap = list(self.deleting)
        reqs = [
            {
                "request_idx": r.request_idx,
                "success": r.success,
                "status": r.status,
                "entered_queue_at": r.entered_at,
                "work_started_at": r.started_at,
                "work_completed_at": r.completed_at,
            }
            for r in snap
        ]
        data = {"worker_id": CONTAINER_ID, "mtoken": MASTER_TOKEN, "requests": reqs}
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            try:
                async with session.post(addr.rstrip("/") + "/delete_requests/", json=data) as res:
                    res.raise_for_status()
                sent = {r.request_idx for r in snap}
                self.deleting[:] = [r for r in self.deleting if r.request_idx not in sent]
                return
            except Exception as e:
                log.debug(f"delete_requests gönderilemedi ({addr}): {e}")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Backend:
    def __init__(self) -> None:
        self.metrics = Metrics()
        self.sessions: Dict[str, Session] = {}
        self._pubkey: Optional[RSA.RsaKey] = None
        self._sig_format_name: Optional[str] = None

    @cached_property
    def session(self) -> ClientSession:
        # streaming için: toplam timeout yok, sadece bağlantı timeout'u
        return ClientSession(
            timeout=ClientTimeout(total=None, connect=10, sock_read=None),
            connector=TCPConnector(limit=0),
        )

    # --- pubkey / imza ---

    async def fetch_pubkey(self) -> None:
        if UNSECURED:
            return
        url = REPORT_ADDR_LIST[0].rstrip("/") + "/pubkey/"
        for attempt in range(1, 6):
            try:
                async with ClientSession(timeout=ClientTimeout(total=10)) as c:
                    async with c.get(url) as res:
                        res.raise_for_status()
                        self._pubkey = RSA.import_key(await res.text())
                        log.info(f"Pubkey alındı (deneme {attempt})")
                        return
            except Exception as e:
                log.debug(f"Pubkey alınamadı ({attempt}/5): {e}")
                if attempt < 5:
                    await sleep(2 ** (attempt - 1))
        log.error("Pubkey alınamadı")

    def _verify(self, message: str, signature: str) -> bool:
        if not message or not signature:
            return False
        try:
            sig = base64.b64decode(signature + "=" * (-len(signature) % 4))
            pkcs1_15.new(self._pubkey).verify(SHA256.new(message.encode()), sig)
            return True
        except Exception:
            return False

    def _sig_candidates(self, auth_data: dict):
        """auth_data.signature'ın atılmış olabileceği mesaj adayları (en olası ilk sırada)."""
        url = auth_data.get("url", "")
        yield "url i4_sort (kanonik)", json.dumps({"url": url}, indent=4, sort_keys=True)
        yield "url compact", json.dumps({"url": url}, separators=(",", ":"))
        yield "url sort", json.dumps({"url": url}, sort_keys=True)
        yield "url default", json.dumps({"url": url})
        yield "url ham", url
        yield "url(no slash) i4_sort", json.dumps({"url": url.rstrip("/")}, indent=4, sort_keys=True)
        yield "url(slash) i4_sort", json.dumps({"url": url + "/"}, indent=4, sort_keys=True)
        ad = {k: v for k, v in auth_data.items() if k != "signature"}
        yield "auth_data(sig hariç) i4_sort", json.dumps(ad, indent=4, sort_keys=True)

    def check_signature(self, auth_data: dict) -> bool:
        """Adaptif: doğru formatı bulunca bir kez loglar; sonra hep onu dener."""
        if UNSECURED:
            return True
        if self._pubkey is None:
            log.error("İmza reddedildi: pubkey yok")
            return False
        sig = auth_data.get("signature", "")
        # Önce daha önce tutan format
        if self._sig_format_name is not None:
            for name, msg in self._sig_candidates(auth_data):
                if name == self._sig_format_name:
                    if self._verify(msg, sig):
                        return True
                    break  # bilinen format tutmadı -> baştan ara
        # Tüm adayları dene
        for name, msg in self._sig_candidates(auth_data):
            if self._verify(msg, sig):
                if self._sig_format_name != name:
                    log.info(f"✅ İmza formatı bulundu ve sabitlendi: '{name}'")
                    self._sig_format_name = name
                return True
        return False

    def _probe_signature(self, auth_data: dict, query: dict) -> None:
        """Hiçbir format tutmadı: pubkey mi yanlış? api_key'in bilinen imzasıyla test et."""
        if self._pubkey is None:
            log.error("PROBE: pubkey yok"); return

        def verify(msg, sig_b64):
            return self._verify(msg, sig_b64) if msg else False

        # 1) PUBKEY DOĞRU MU? api_key içindeki bilinen msg+signature ile test
        raw = query.get("api_key")
        if raw:
            try:
                obj = json.loads(base64.b64decode(raw + "=" * (-len(raw) % 4)))
                msg, sig = obj.get("msg"), obj.get("signature")
                cands = {
                    "msg ham": msg,
                    "msg json sort": json.dumps(json.loads(msg), sort_keys=True) if msg else None,
                    "msg json compact": json.dumps(json.loads(msg), separators=(",", ":")) if msg else None,
                    "msg json i4 sort": json.dumps(json.loads(msg), indent=4, sort_keys=True) if msg else None,
                }
                hit = [n for n, m in cands.items() if verify(m, sig)]
                if hit:
                    log.error(f"PROBE: ✅ PUBKEY DOĞRU (api_key imzası '{hit}' formatında tuttu). "
                              f"Sorun auth_data formatında -> adaylar arttırılmalı.")
                else:
                    log.error("PROBE: ❌ pubkey api_key imzasını HİÇ tutmadı -> PUBKEY YANLIŞ "
                              "(yanlış /pubkey kaynağı) olabilir.")
            except Exception as e:
                log.error(f"PROBE: api_key çözülemedi (log kırpılmış olabilir): {e}")
        else:
            log.error("PROBE: query'de api_key yok -> pubkey testi yapılamadı")

        # 2) auth_data hangi formatta? (check_signature zaten denedi; burada özet)
        url = auth_data.get("url", "")
        log.error(f"PROBE: ❌ auth_data imzası denenen formatların hiçbirinde tutmadı. url={url!r}")

    # --- model hazır: SADECE /health ---

    async def wait_for_model_ready(self) -> None:
        log.info(f"/health yoklanıyor: {MODEL_HEALTH_URL}")
        while True:
            try:
                async with self.session.get(MODEL_HEALTH_URL, timeout=ClientTimeout(total=5)) as res:
                    if res.status == 200:
                        break
                    log.debug(f"/health status={res.status}, bekleniyor")
            except Exception as e:
                log.debug(f"/health erişilemedi: {e}")
            await sleep(HEALTH_POLL_INTERVAL)
        self.metrics.model_loaded(WORKER_PERF)

    # --- model isteği ---

    async def model_request_handler(self, request: web.Request) -> web.StreamResponse:
        log.info(f"İstek geldi: {request.method} {request.path}")
        try:
            data = await request.json()
            auth = AuthData.from_dict(data["auth_data"])
            payload = data.get("payload", {})
        except Exception as e:
            log.error(f"İstek parse edilemedi: {e}")
            return web.json_response({"error": f"invalid request: {e}"}, status=422)

        log.debug(f"auth.url={auth.url!r} | get_url()={get_url()!r} | request_idx={auth.request_idx}")
        if not self.check_signature(data["auth_data"]):
            self._probe_signature(data["auth_data"], dict(request.query))
            return web.Response(status=401)

        info = RequestInfo(request_idx=auth.request_idx, entered_at=time.time(), started_at=time.time())
        self.metrics.start(info)
        log.info(f"İstek {auth.request_idx} -> modele iletiliyor: {MODEL_BASE_URL}{request.path} (stream={payload.get('stream')})")
        try:
            return await self._forward(request, payload, info)
        except asyncio.CancelledError:
            log.warning(f"İstek {auth.request_idx} iptal edildi (client koptu)")
            self.metrics.end(info, success=False)
            raise
        except Exception as e:
            log.error(f"İstek {auth.request_idx} hata: {e}")
            self.metrics.end(info, success=False)
            return web.json_response({"error": str(e)}, status=502)

    async def _forward(self, request: web.Request, payload: dict, info: RequestInfo) -> web.StreamResponse:
        async with self.session.post(f"{MODEL_BASE_URL}{request.path}", json=payload) as model_res:
            ctype = model_res.content_type or ""
            is_stream = ctype.startswith("text/event-stream") or model_res.headers.get("Transfer-Encoding") == "chunked"
            log.info(f"İstek {info.request_idx} model yanıtı: status={model_res.status} ctype={ctype} stream={is_stream}")

            if is_stream:
                res = web.StreamResponse(status=model_res.status)
                if ctype:
                    res.content_type = ctype
                res.headers["X-Accel-Buffering"] = "no"
                await res.prepare(request)
                async for chunk in model_res.content.iter_any():
                    if chunk:
                        await res.write(chunk)
                await res.write_eof()
                self.metrics.end(info, success=(model_res.status == 200))
                return res

            body = await model_res.read()
            self.metrics.end(info, success=(model_res.status == 200))
            return web.Response(body=body, status=model_res.status, content_type=ctype or None)

    # --- session ---

    async def session_create_handler(self, request: web.Request) -> web.Response:
        if len(self.sessions) >= MAX_SESSIONS:
            return web.Response(status=429)
        try:
            data = await request.json()
            auth_data = data["auth_data"]
            payload = data.get("payload", {})
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        sid = secrets.token_urlsafe(12)
        now = time.time()
        lifetime = float(payload.get("lifetime", DEFAULT_SESSION_LIFETIME))
        self.sessions[sid] = Session(
            session_id=sid, auth_data=auth_data, lifetime=lifetime,
            created_at=now, expiration=now + lifetime,
            on_close_route=payload.get("on_close_route"),
            on_close_payload=payload.get("on_close_payload"),
        )
        return web.json_response({"session_id": sid, "expiration": now + lifetime}, status=201)

    async def session_get_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            sid, sauth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        s = self.sessions.get(sid)
        if s is None:
            return web.json_response({"error": "not found"}, status=400)
        if s.auth_data != sauth:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(asdict(s))

    async def session_health_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            sid, sauth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        s = self.sessions.get(sid)
        if s is not None and s.auth_data != sauth:
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"ok": s is not None})

    async def session_end_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            sid, sauth = data["session_id"], data["session_auth"]
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=422)
        s = self.sessions.get(sid)
        if s is None:
            return web.json_response({"error": "not found"}, status=400)
        if s.auth_data != sauth:
            return web.json_response({"error": "unauthorized"}, status=401)
        del self.sessions[sid]
        if s.on_close_route:
            try:
                body = dict(s.on_close_payload or {})
                body["session_id"] = sid
                async with self.session.post(s.on_close_route, json=body):
                    pass
            except Exception as e:
                log.debug(f"on_close webhook hata: {e}")
        return web.json_response({"ended": True, "removed_session": sid})

    async def pyworker_update_handler(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization", "") != f"Bearer {MASTER_TOKEN}":
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            open("/.force_update", "w").close()
        except Exception as e:
            return web.json_response({"error": f"failed: {e}"}, status=500)
        return web.json_response({"ok": True})

    async def catchall_handler(self, request: web.Request) -> web.Response:
        # Kayıtlı route'lara düşmeyen her istek burada loglanır (teşhis için).
        log.warning(f"EŞLEŞMEYEN istek: {request.method} {request.path}  query={dict(request.query)}")
        return web.Response(status=404)

    async def session_gc_loop(self) -> None:
        while True:
            await sleep(5)
            now = time.time()
            for sid in [s for s, v in self.sessions.items() if v.expiration < now]:
                del self.sessions[sid]

    async def start_tracking(self) -> None:
        await asyncio.gather(
            self.fetch_pubkey(),
            self.wait_for_model_ready(),
            self.metrics.send_loop(),
            self.session_gc_loop(),
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def main() -> None:
    backend = Backend()
    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_post("/session/create", backend.session_create_handler)
    app.router.add_post("/session/get", backend.session_get_handler)
    app.router.add_post("/session/health", backend.session_health_handler)
    app.router.add_post("/session/end", backend.session_end_handler)
    app.router.add_post("/pyworker/update", backend.pyworker_update_handler)
    app.router.add_post("/v1/chat/completions", backend.model_request_handler)
    app.router.add_post("/v1/completions", backend.model_request_handler)
    app.router.add_post("/completion", backend.model_request_handler)
    app.router.add_route("*", "/{tail:.*}", backend.catchall_handler)  # en son; eşleşmeyenleri loglar

    runner = web.AppRunner(app)
    await runner.setup()

    ssl_context = None
    if USE_SSL:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(certfile="/etc/instance.crt", keyfile="/etc/instance.key")

    site = web.TCPSite(runner, "0.0.0.0", WORKER_PORT, ssl_context=ssl_context)
    await site.start()
    log.info(f"PyWorker {WORKER_PORT} portunda dinliyor (SSL={'açık' if USE_SSL else 'kapalı'})")

    await backend.start_tracking()


if __name__ == "__main__":
    asyncio.run(main())
