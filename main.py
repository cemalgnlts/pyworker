"""
worker.py
=========

Vast.ai Serverless ile uyumlu, Llama.cpp (llama-server) backend'i için
SAĞLAMLAŞTIRILMIŞ standalone PyWorker.

ÖNEMLİ: Dosya adı `worker.py` olmalı. start_server.sh şu sırayla arar:
    worker.py  ->  workers/$BACKEND/worker.py  ->  workers/$BACKEND/server.py
ve repodaki worker.py'i `python3 -m worker` ile başlatır. Eski `main.py` adıyla
script onu OTOMATİK başlatmaz.

ÖNCEKİ main.py'e GÖRE DÜZELTİLENLER:
  1. worker_status artık autoscaler'ın beklediği TAM şema ile gönderiliyor
     (cur_load, new_load, rej_load, cur_perf, num_requests_recieved,
      working_request_idxs, additional_disk_usage, cur_capacity, max_capacity).
     Metrikler INTERVAL bazlı: her başarılı gönderimden sonra sıfırlanır.
  2. delete_requests artık DOĞRU formatta:
     {"worker_id","mtoken","requests":[{request_idx,success,status,
       entered_queue_at,work_started_at,work_completed_at}, ...]}
     (eskisi "request_idxs" listesi gönderiyordu; autoscaler tamamlanan
      istekleri temizleyemiyordu -> hayalet yük.)
  3. İmza doğrulama kanonik: SHA256( json.dumps({"url": auth_data.url},
     indent=4, sort_keys=True) ) + PKCS1_15. URL ARTIK YENİDEN ÜRETİLMİYOR;
     imza, autoscaler'ın gönderdiği TAM url string'i üzerine atılıyor.
  4. İSTEK TABANLI ölçek: gerçek benchmark YOK. Her istek sabit load taşır
     (WORKLOAD_PER_REQUEST) ve perf sabittir (WORKER_PERF). Böylece bir worker
     WORKER_PERF / WORKLOAD_PER_REQUEST eşzamanlı istekte dolar.
     örn: WORKER_PERF=100, REQUESTS_PER_WORKER=4 -> her istek 25 load,
     4 istekte cur_load=100=perf (doluluk %100).
  5. Streaming passthrough (llama.cpp "stream": true -> SSE). JSON da desteklenir.
  6. Workload = istekteki max_tokens / n_predict. Tüm load/perf hesapları buna dayanır.
  7. loadtime yalnızca BİR kez gönderilir (autoscaler böyle bekliyor).
  8. (Opsiyonel) MODEL_LOG verilirse, log'da hata satırı görülünce worker
     hatalı işaretlenir -> serverless daha hızlı yeniden planlar.
"""

import os
import ssl
import json
import time
import shutil
import base64
import asyncio
import logging
import secrets
from asyncio import sleep, create_task
from dataclasses import dataclass, field, asdict
from functools import cached_property, cache
from typing import Optional, Dict, Any, List, Set, Union

from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llamacpp_pyworker")

# ---------------------------------------------------------------------------
# Config (env var'lardan)
# ---------------------------------------------------------------------------

CONTAINER_ID = int(os.environ.get("CONTAINER_ID", "0"))
REPORT_ADDR_LIST = os.environ["REPORT_ADDR"].split(",")
WORKER_PORT = int(os.environ["WORKER_PORT"])
USE_SSL = os.environ.get("USE_SSL", "false").lower() == "true"
UNSECURED = os.environ.get("UNSECURED", "false").lower() == "true"
MASTER_TOKEN = os.environ.get("MASTER_TOKEN", "")
PYWORKER_VERSION = os.environ.get("PYWORKER_VERSION", "llamacpp-1.0")

# llama-server (llama.cpp) varsayılan portu 8080. /v1/... ve native /completion sunar.
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL_HEALTH_URL = os.environ.get("MODEL_HEALTH_URL", f"{MODEL_BASE_URL}/health")
MODEL_LOG = os.environ.get("MODEL_LOG", "")  # opsiyonel: hata tespiti için tail edilir

# === İSTEK TABANLI ÖLÇEK (load-based DEĞİL) ===
# Her istek SABİT bir workload taşır; benchmark YOK, perf sabittir.
# Bir worker'ı "dolduran" eşzamanlı istek sayısı = WORKER_PERF / WORKLOAD_PER_REQUEST.
#   örn: WORKER_PERF=100, REQUESTS_PER_WORKER=4  ->  her istek = 25 load.
# Böylece worker 4 eşzamanlı istekte cur_load=100=perf olur (doluluk = %100).
WORKER_PERF = float(os.environ.get("WORKER_PERF", "100"))
REQUESTS_PER_WORKER = float(os.environ.get("REQUESTS_PER_WORKER", "4"))
WORKLOAD_PER_REQUEST = WORKER_PERF / max(REQUESTS_PER_WORKER, 1e-9)  # 100/4 = 25

# Worker'ın kendi SERT tavanı (istek SAYISI). 0 = sınırsız (ölçeği autoscaler yönetir).
# Genelde 0 bırakın; gerçek scale-up kararını endpoint parametreleri verir.
HARD_MAX_CONCURRENT = int(os.environ.get("HARD_MAX_CONCURRENT", "0"))

MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))

HEALTHCHECK_POLL_INTERVAL = 0.5
METRICS_UPDATE_INTERVAL = 1
DELETE_REQUESTS_INTERVAL = 1
SESSION_GC_INTERVAL = 5.0
DEFAULT_SESSION_LIFETIME = 60.0


@cache
def get_url() -> str:
    """Bu worker'ın dışarıdan erişilebilir URL'i. SADECE worker_status.url alanı için."""
    worker_port_ext = os.environ[f"VAST_TCP_PORT_{WORKER_PORT}"]
    public_ip = os.environ["PUBLIC_IPADDR"]
    return f"http{'s' if USE_SSL else ''}://{public_ip}:{worker_port_ext}"


def _disk_used_gb() -> float:
    return shutil.disk_usage("/").used / (2 ** 30)


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
        # url İMZANIN üzerine atıldığı string'tir; ASLA yeniden üretme.
        url = d.get("url")
        if not url:
            # Bu olmamalı; olduysa imza büyük ihtimalle doğrulanmaz. Sadece logla.
            log.error("auth_data.url eksik geldi; imza doğrulaması başarısız olabilir.")
            url = ""
        return AuthData(
            cost=d.get("cost"),
            endpoint=d.get("endpoint"),
            reqnum=int(d.get("reqnum", 0)),
            request_idx=int(d.get("request_idx", 0)),
            signature=d["signature"],
            url=url,
        )


@dataclass
class RequestMetrics:
    request_idx: int
    reqnum: int
    workload: float
    status: str = "Created"
    success: bool = False
    entered_queue_at: float = 0.0
    work_started_at: float = 0.0
    work_completed_at: float = 0.0


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
# Metrics  (autoscaler'a worker_status / delete_requests gönderir)
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self) -> None:
        self.mtoken = MASTER_TOKEN
        self.version = PYWORKER_VERSION

        # sistem
        self.model_is_loaded = False
        self.model_loading_start = time.time()
        self.loadtime: Optional[float] = None
        self.loadtime_sent = False
        self.baseline_disk_gb = _disk_used_gb()
        self.additional_disk_usage = 0.0

        # interval bazlı sayaçlar (her gönderimde sıfırlanır)
        self.workload_received = 0.0
        self.workload_served = 0.0
        self.workload_cancelled = 0.0
        self.workload_errored = 0.0
        self.workload_rejected = 0.0
        self.requests_recieved: Set[int] = set()

        # kalıcı durum
        self.workload_pending = 0.0
        self.max_throughput = 0.0
        self.error_msg: Optional[str] = None
        self.requests_working: Dict[int, RequestMetrics] = {}
        self.requests_deleting: List[RequestMetrics] = []

        self.update_pending = False
        self.last_metric_update = 0.0
        self._session: Optional[ClientSession] = None

    async def http(self) -> ClientSession:
        if self._session is None:
            self._session = ClientSession(
                timeout=ClientTimeout(total=10),
                connector=TCPConnector(limit=8, limit_per_host=4, force_close=True, enable_cleanup_closed=True),
            )
        return self._session

    # ---- türetilmiş ----

    @property
    def cur_load(self) -> float:
        return sum(r.workload for r in self.requests_working.values())

    @property
    def workload_processing(self) -> float:
        return max(self.workload_received - self.workload_cancelled, 0.0)

    @property
    def wait_time(self) -> float:
        if not self.requests_working:
            return 0.0
        return self.cur_load / max(self.max_throughput, 1e-5)

    @property
    def working_request_idxs(self) -> List[int]:
        return [r.request_idx for r in self.requests_working.values()]

    # ---- yaşam döngüsü ----

    def model_loaded(self, max_throughput: float) -> None:
        self.loadtime = time.time() - self.model_loading_start
        self.model_is_loaded = True
        self.max_throughput = max_throughput
        self.update_pending = True
        log.info(f"Model hazır. loadtime={self.loadtime:.1f}s, max_throughput={max_throughput:.1f} tok/s")

    def model_errored(self, error_msg: str) -> None:
        self.error_msg = error_msg
        self.model_is_loaded = True  # autoscaler'ın hata mesajını alabilmesi için
        self.update_pending = True
        log.error(f"Model hatalı: {error_msg}")

    def request_start(self, rm: RequestMetrics) -> None:
        rm.status = "Started"
        self.workload_received += rm.workload
        self.workload_pending += rm.workload
        self.requests_recieved.add(rm.reqnum)
        self.requests_working[rm.request_idx] = rm
        self.update_pending = True

    def request_success(self, rm: RequestMetrics) -> None:
        rm.work_completed_at = time.time()
        rm.status = "Success"
        rm.success = True
        self.workload_served += rm.workload
        self._end(rm)

    def request_errored(self, rm: RequestMetrics, msg: str) -> None:
        rm.work_completed_at = time.time()
        rm.status = "Error"
        rm.success = False
        self.workload_errored += rm.workload
        log.error(f"İstek {rm.request_idx} hata: {msg}")
        self._end(rm)

    def request_cancelled(self, rm: RequestMetrics) -> None:
        rm.work_completed_at = time.time()
        rm.status = "Cancelled"
        rm.success = True
        self.workload_cancelled += rm.workload
        self._end(rm)

    def request_reject(self, rm: RequestMetrics) -> None:
        rm.status = "Rejected"
        rm.success = False
        self.workload_rejected += rm.workload
        self.requests_recieved.add(rm.reqnum)
        self.requests_deleting.append(rm)
        self.update_pending = True

    def _end(self, rm: RequestMetrics) -> None:
        self.workload_pending -= rm.workload
        self.requests_working.pop(rm.request_idx, None)
        self.requests_deleting.append(rm)
        self.update_pending = True

    # ---- gönderim döngüleri ----

    async def send_metrics_loop(self) -> None:
        while True:
            await sleep(METRICS_UPDATE_INTERVAL)
            elapsed = time.time() - self.last_metric_update
            if (not self.model_is_loaded and elapsed >= 10) or self.update_pending or elapsed > 10:
                await self.__send_worker_status()

    async def __send_worker_status(self) -> None:
        self.additional_disk_usage = _disk_used_gb() - self.baseline_disk_gb

        loadtime_to_send = (self.loadtime or 0.0) if not self.loadtime_sent else 0.0

        data = {
            "id": CONTAINER_ID,
            "mtoken": self.mtoken,
            "version": self.version,
            "loadtime": loadtime_to_send,
            "cur_load": self.cur_load,
            "new_load": self.workload_processing,
            "rej_load": self.workload_rejected,
            "max_perf": self.max_throughput,
            "cur_perf": self.workload_served,
            "cur_capacity": 0,
            "max_capacity": 0,
            "error_msg": self.error_msg or "",
            "num_requests_working": len(self.requests_working),
            "num_requests_recieved": len(self.requests_recieved),
            "additional_disk_usage": self.additional_disk_usage,
            "working_request_idxs": self.working_request_idxs,
            "url": get_url(),
        }
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            full_path = addr.rstrip("/") + "/worker_status/"
            try:
                async with session.post(full_path, json=data) as res:
                    res.raise_for_status()
                # başarı: interval sayaçlarını sıfırla
                if self.model_is_loaded and self.loadtime is not None:
                    self.loadtime_sent = True
                self.update_pending = False
                self.last_metric_update = time.time()
                self.__reset_interval_counters()
                return
            except Exception as e:
                log.debug(f"worker_status gönderilemedi ({addr}): {e}")

    def __reset_interval_counters(self) -> None:
        self.workload_received = 0.0
        self.workload_served = 0.0
        self.workload_cancelled = 0.0
        self.workload_errored = 0.0
        self.workload_rejected = 0.0
        self.requests_recieved.clear()

    async def send_delete_requests_loop(self) -> None:
        while True:
            await sleep(DELETE_REQUESTS_INTERVAL)
            if self.requests_deleting:
                await self.__send_delete_requests()

    async def __send_delete_requests(self) -> None:
        snapshot = list(self.requests_deleting)
        if not snapshot:
            return
        requests_payload = [
            {
                "request_idx": r.request_idx,
                "success": r.success,
                "status": r.status,
                "entered_queue_at": r.entered_queue_at,
                "work_started_at": r.work_started_at,
                "work_completed_at": r.work_completed_at,
            }
            for r in snapshot
        ]
        data = {"worker_id": CONTAINER_ID, "mtoken": self.mtoken, "requests": requests_payload}
        session = await self.http()
        for addr in REPORT_ADDR_LIST:
            full_path = addr.rstrip("/") + "/delete_requests/"
            try:
                async with session.post(full_path, json=data) as res:
                    res.raise_for_status()
                sent = {r.request_idx for r in snapshot}
                self.requests_deleting[:] = [r for r in self.requests_deleting if r.request_idx not in sent]
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
        self._pubkey_fetch_complete = asyncio.Event()

    @cached_property
    def session(self) -> ClientSession:
        # Streaming için connector limiti yüksek, timeout uzun.
        return ClientSession(
            timeout=ClientTimeout(total=None, connect=10, sock_read=None),
            connector=TCPConnector(limit=0, enable_cleanup_closed=True),
        )

    # --- pubkey / imza ----------------------------------------------------

    async def fetch_pubkey(self) -> None:
        if UNSECURED:
            self._pubkey_fetch_complete.set()
            return
        url = REPORT_ADDR_LIST[0].rstrip("/") + "/pubkey/"
        timeout = ClientTimeout(total=10)
        for attempt in range(1, 6):
            try:
                async with ClientSession(timeout=timeout) as client:
                    async with client.get(url) as res:
                        res.raise_for_status()
                        text = await res.text()
                        self._pubkey = RSA.import_key(text)
                        self._pubkey_fetch_complete.set()
                        log.info(f"Pubkey alındı (deneme {attempt})")
                        return
            except Exception as e:
                log.debug(f"Pubkey alınamadı ({attempt}/5): {e}")
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
            pkcs1_15.new(self._pubkey).verify(h, base64.b64decode(auth_data.signature))
            return True
        except (ValueError, TypeError):
            log.error(f"İmza doğrulanamadı. message={message!r}")
            return False

    # --- model hazır + benchmark ------------------------------------------

    async def wait_for_model_ready(self) -> None:
        timeout = ClientTimeout(total=5)
        while True:
            try:
                async with self.session.get(MODEL_HEALTH_URL, timeout=timeout) as res:
                    if res.status == 200:
                        break
            except Exception:
                pass
            if self.metrics.error_msg:  # log-tail hata bulduysa bekleme
                return
            await sleep(HEALTHCHECK_POLL_INTERVAL)

        log.info("llama-server /health 200 döndü")

        # pubkey hazır olmadan benchmark perf'i göndermenin anlamı yok
        if not UNSECURED:
            try:
                await asyncio.wait_for(self._pubkey_fetch_complete.wait(), timeout=60)
            except asyncio.TimeoutError:
                self.metrics.model_errored("pubkey fetch zaman aşımı")
                return

        # İstek tabanlı ölçek: benchmark YOK, perf sabit raporlanır.
        self.metrics.model_loaded(max_throughput=WORKER_PERF)

    # --- opsiyonel: model log'undan hata tespiti --------------------------

    async def tail_model_log(self) -> None:
        if not MODEL_LOG:
            return
        error_markers = [
            "INFO exited: ",
            "error loading model",
            "failed to load model",
            "terminate called",
            "CUDA error",
            "out of memory",
        ]
        # dosya gelene kadar bekle
        while not os.path.exists(MODEL_LOG):
            await sleep(1)
        with open(MODEL_LOG, "r", errors="ignore") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    await sleep(0.5)
                    continue
                if any(m in line for m in error_markers) and not self.metrics.model_is_loaded:
                    self.metrics.model_errored(line.strip()[:300])

    # --- workload --------------------------------------------------------

    @staticmethod
    def count_workload(path: str, payload: dict) -> float:
        # İstek tabanlı ölçek: her istek sabit load taşır (token sayısından bağımsız).
        return WORKLOAD_PER_REQUEST

    # --- model isteği ----------------------------------------------------

    async def model_request_handler(self, request: web.Request) -> web.StreamResponse:
        try:
            data = await request.json()
            auth_data = AuthData.from_dict(data["auth_data"])
            payload = data.get("payload", {})
        except Exception as e:
            return web.json_response({"error": f"invalid request: {e}"}, status=422)

        rm = RequestMetrics(
            request_idx=auth_data.request_idx,
            reqnum=auth_data.reqnum,
            workload=self.count_workload(request.path, payload),
            entered_queue_at=time.time(),
        )

        if not self.check_signature(auth_data):
            self.metrics.request_reject(rm)
            return web.Response(status=401)

        if HARD_MAX_CONCURRENT and len(self.metrics.requests_working) >= HARD_MAX_CONCURRENT:
            self.metrics.request_reject(rm)
            return web.Response(status=429)

        self.metrics.request_start(rm)
        rm.work_started_at = time.time()
        try:
            return await self.__forward(request, payload, rm)
        except asyncio.CancelledError:
            self.metrics.request_cancelled(rm)
            raise
        except Exception as e:
            self.metrics.request_errored(rm, str(e))
            return web.json_response({"error": str(e)}, status=502)

    async def __forward(self, request: web.Request, payload: dict, rm: RequestMetrics) -> web.StreamResponse:
        url = f"{MODEL_BASE_URL}{request.path}"
        async with self.session.post(url, json=payload) as model_res:
            ctype = model_res.content_type or ""
            is_stream = (
                ctype.startswith("text/event-stream")
                or ctype in ("application/x-ndjson", "application/jsonl")
                or model_res.headers.get("Transfer-Encoding") == "chunked"
            )

            if is_stream:
                res = web.StreamResponse(status=model_res.status)
                if ctype:
                    res.content_type = ctype
                res.headers["X-Accel-Buffering"] = "no"  # reverse-proxy buffering'i engelle
                await res.prepare(request)
                async for chunk in model_res.content.iter_any():
                    if chunk:
                        await res.write(chunk)
                await res.write_eof()
                if model_res.status == 200:
                    self.metrics.request_success(rm)
                else:
                    self.metrics.request_errored(rm, f"model status {model_res.status}")
                return res

            body = await model_res.read()
            if model_res.status == 200:
                self.metrics.request_success(rm)
            else:
                self.metrics.request_errored(rm, f"model status {model_res.status}")
            return web.Response(body=body, status=model_res.status, content_type=ctype or None)

    # --- session handler'ları --------------------------------------------

    async def session_create_handler(self, request: web.Request) -> web.Response:
        if len(self.sessions) >= MAX_SESSIONS:
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
            session_id=session_id, auth_data=auth_data, lifetime=lifetime,
            created_at=now, expiration=now + lifetime,
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
                log.debug(f"on_close_route webhook hata: {e}")
        return web.json_response({"ended": True, "removed_session": session_id})

    async def pyworker_update_handler(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization", "") != f"Bearer {MASTER_TOKEN}":
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            open("/.force_update", "w").close()
        except Exception as e:
            return web.json_response({"error": f"failed to write file: {e}"}, status=500)
        return web.json_response({"ok": True})

    # --- session GC + ana gather -----------------------------------------

    async def session_gc_loop(self) -> None:
        while True:
            await sleep(SESSION_GC_INTERVAL)
            now = time.time()
            for sid in [s for s, v in self.sessions.items() if v.expiration < now]:
                del self.sessions[sid]

    async def start_tracking(self) -> None:
        await asyncio.gather(
            self.fetch_pubkey(),
            self.wait_for_model_ready(),
            self.tail_model_log(),
            self.metrics.send_metrics_loop(),
            self.metrics.send_delete_requests_loop(),
            self.session_gc_loop(),
        )


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

async def main() -> None:
    backend = Backend()
    app = web.Application(client_max_size=100 * 1024 * 1024)

    app.router.add_post("/session/create", backend.session_create_handler)
    app.router.add_post("/session/get", backend.session_get_handler)
    app.router.add_post("/session/health", backend.session_health_handler)
    app.router.add_post("/session/end", backend.session_end_handler)
    app.router.add_post("/pyworker/update", backend.pyworker_update_handler)

    # llama.cpp endpoint'leri
    app.router.add_post("/v1/chat/completions", backend.model_request_handler)
    app.router.add_post("/v1/completions", backend.model_request_handler)
    app.router.add_post("/completion", backend.model_request_handler)  # native

    runner = web.AppRunner(app)
    await runner.setup()

    ssl_context = None
    if USE_SSL:
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(certfile="/etc/instance.crt", keyfile="/etc/instance.key")
        except Exception as ex:
            raise Exception(f"SSL sertifikası yüklenemedi: {ex}")

    site = web.TCPSite(runner, "0.0.0.0", WORKER_PORT, ssl_context=ssl_context)
    await site.start()
    log.info(f"PyWorker {WORKER_PORT} portunda dinliyor (SSL={'açık' if USE_SSL else 'kapalı'}, backend=llama.cpp)")

    await backend.start_tracking()


if __name__ == "__main__":
    asyncio.run(main())
