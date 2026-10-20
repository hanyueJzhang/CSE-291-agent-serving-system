"""Periodic memory and load sampling for memory.csv.

Sources, each optional (a missing source leaves its columns blank and adds a note):
- GPU memory and utilization from NVML (pynvml)
- llama-server process RSS from psutil; the host-RAM prompt cache (--cache-ram)
  lives here, so RSS can grow during a run even though GPU memory does not
- llama-server /slots: tokens held across all slots
- llama-server /metrics (needs --metrics): requests deferred inside
  llama-server, i.e. queueing that our scheduler cannot see

llama.cpp allocates the whole KV cache at startup, so GPU memory is mostly
flat; slot_tokens_total is the signal for how much of that cache is in use.
Our own running/waiting counts are not sampled: requests.csv timestamps give
them exactly.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable
from urllib.parse import urlparse

import aiohttp

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def find_server_process(base_url: str | None = None):
    """Return the local llama-server psutil.Process, preferring one on base_url's port."""
    if psutil is None:
        return None
    port = str(urlparse(base_url).port or 8080) if base_url else None
    candidates = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
            name = proc.info["name"] or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "llama-server" in name or (cmdline and "llama-server" in cmdline[0]):
            candidates.append((proc, cmdline))
    for proc, cmdline in candidates:
        if port and _port_arg(cmdline) == port:
            return proc
    return candidates[0][0] if len(candidates) == 1 else None


def _port_arg(cmdline: list[str]) -> str:
    for i, arg in enumerate(cmdline):
        if arg == "--port" and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if arg.startswith("--port="):
            return arg.split("=", 1)[1]
    return "8080"  # llama-server's default


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse llama-server's `llamacpp:<name> <value>` lines."""
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("llamacpp:"):
            try:
                values[parts[0][len("llamacpp:"):]] = float(parts[1])
            except ValueError:
                pass
    return values


class MemorySampler:
    def __init__(
        self,
        *,
        clock: Callable[[], float],
        base_url: str | None = None,
        interval_s: float = 0.5,
        server_process: Any = None,
        use_gpu: bool = True,
        gpu_index: int = 0,
    ):
        self.clock = clock
        self.base_url = base_url.rstrip("/") if base_url else None
        self.interval_s = interval_s
        self.server_process = server_process
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.rows: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self._gpu = None
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._http_ok = {"slots": True, "metrics": True}

    # ---- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self.use_gpu:
            self._init_gpu()
        if self.server_process is None:
            self.notes.append("llama-server process not found locally; server_rss_mb is blank")
        if self.base_url:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._task is not None:
            await self._task
        await self.sample()  # one last sample after the workload ends
        if self._session is not None:
            await self._session.close()
        return self.rows

    async def _loop(self) -> None:
        while not self._stop.is_set():
            started = self.clock()
            await self.sample()
            remaining = self.interval_s - (self.clock() - started)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(remaining, 0))
            except asyncio.TimeoutError:
                pass

    # ---- one sample --------------------------------------------------------

    async def sample(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": self.clock()}
        row.update(self._read_gpu())
        row["server_rss_mb"] = self._read_rss()
        if self._session is not None:
            slots, metrics = await asyncio.gather(self._get("slots"), self._get("metrics"))
            if isinstance(slots, list):
                row["slot_tokens_total"] = sum(s.get("n_prompt_tokens", 0) or 0 for s in slots)
            if metrics is not None:
                row["server_requests_deferred"] = parse_prometheus(metrics).get("requests_deferred")
        self.rows.append(row)
        return row

    def _init_gpu(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._gpu = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index))
        except Exception as exc:  # ImportError, NVMLError, ...
            self.notes.append(f"GPU sampling unavailable ({exc!r}); gpu_* columns are blank")

    def _read_gpu(self) -> dict[str, Any]:
        if self._gpu is None:
            return {}
        pynvml, handle = self._gpu
        try:
            return {
                "gpu_mem_used_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
                "gpu_util_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            }
        except Exception:
            return {}

    def _read_rss(self) -> float | None:
        if self.server_process is None:
            return None
        try:
            return self.server_process.memory_info().rss / 2**20
        except Exception:
            return None

    async def _get(self, endpoint: str) -> Any:
        """GET /slots (parsed JSON) or /metrics (text). None if unavailable.

        Each endpoint is dropped after its first failure so a disabled endpoint
        does not cost a request every sample.
        """
        if not self._http_ok[endpoint]:
            return None
        timeout = aiohttp.ClientTimeout(total=max(2 * self.interval_s, 2.0))
        try:
            async with self._session.get(f"{self.base_url}/{endpoint}", timeout=timeout) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                return json.loads(text) if endpoint == "slots" else text
        except asyncio.TimeoutError:
            return None  # server busy; try again next sample
        except Exception as exc:
            if self._http_ok[endpoint]:  # a concurrent sample may have failed first
                self._http_ok[endpoint] = False
                self.notes.append(f"/{endpoint} unavailable ({exc}); its columns are blank")
            return None
