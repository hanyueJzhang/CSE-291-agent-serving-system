"""Run a list of experiments unattended, restarting llama-server before each.

    python -m scripts.sweep scripts/sweeps/phase1_baseline.json
    python -m scripts.sweep scripts/sweeps/phase1_baseline.json --dry-run

For every run in the plan: stop any llama-server, start a fresh one with that
run's server settings (CTX, PARALLEL, EXTRA_ARGS), wait for /health, run
scripts/run_experiment.py, stop the server. A fresh server per run means no
run inherits another's slot KV or host-RAM prompt cache (server RSS).

Results land in results/<plan name>/<run_id>/. A run whose folder already has
requests.csv is skipped, so rerunning the same plan resumes after a crash or
Ctrl+C. A half-written folder is renamed to <run_id>.incomplete-<time>.
Afterwards the plan's runs are summarized into results/<plan name>/sweep.csv,
and the default server is restarted in tmux session "llm" (--no-restore skips).

Plan file (JSON):

    {
      "name": "phase1_baseline",
      "server": {"CTX": 8192, "PARALLEL": 4},          # defaults for every run
      "dimensions": {"phase": 1, "policy": "fifo"},   # recorded metadata
      "args": ["--workload", "workload/tasks/example_mix.json"],
      "runs": [
        {"run_id": "rate_c4_r1", "args": ["--concurrency", "4", "--rate", "1"]},
        {"run_id": "c16", "server": {"CTX": 32768, "PARALLEL": 16},
         "dimensions": {"variant": "large"},
         "args": ["--concurrency", "16"]}
      ]
    }

"server" values become environment variables of setup/start_server.sh, which
reads CTX, PARALLEL, NGL, SERVER_PORT, CACHE_RAM (MiB, 0 = off) and EXTRA_ARGS.
A run's "server" and "dimensions" are merged over the plan's; its "args"
follow the plan's. Dimensions must be scalar JSON values and are preserved in
run_config.json, summary.json and sweep.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVER_CMD = "bash setup/start_server.sh"
DEFAULT_PORT = 8080


def log(message: str) -> None:
    print(f"[sweep {datetime.now():%H:%M:%S}] {message}", flush=True)


# ---- plan ----------------------------------------------------------------------

def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not plan.get("name") or not plan.get("runs"):
        raise ValueError(f"{path}: a plan needs 'name' and a non-empty 'runs' list")
    run_ids = [run.get("run_id") for run in plan["runs"]]
    if not all(run_ids):
        raise ValueError(f"{path}: every run needs a 'run_id'")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"{path}: duplicate run_id")
    for run_id in run_ids:
        if "/" in run_id or run_id.startswith("."):
            raise ValueError(f"{path}: run_id {run_id!r} must be a plain folder name")
    dimension_scopes = [("plan", plan.get("dimensions", {}))]
    dimension_scopes.extend(
        (f"run {run['run_id']}", run.get("dimensions", {})) for run in plan["runs"]
    )
    for label, dimensions in dimension_scopes:
        if not isinstance(dimensions, dict):
            raise ValueError(f"{path}: {label} dimensions must be an object")
        invalid = [key for key, value in dimensions.items()
                   if not isinstance(key, str) or isinstance(value, (dict, list))]
        if invalid:
            raise ValueError(
                f"{path}: {label} dimensions need string keys and scalar JSON values; "
                f"invalid keys: {invalid}"
            )
    return plan


def expand_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Each run with its merged server settings and full run_experiment args."""
    runs = []
    for run in plan["runs"]:
        server = {**plan.get("server", {}), **run.get("server", {})}
        dimensions = {**plan.get("dimensions", {}), **run.get("dimensions", {})}
        runs.append({
            "run_id": run["run_id"],
            "server": {k: str(v) for k, v in server.items()},
            "args": [str(a) for a in [*plan.get("args", []), *run.get("args", [])]],
            "dimensions": dimensions,
        })
    return runs


# ---- server --------------------------------------------------------------------

def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def stop_all_llama_servers(timeout_s: float = 15.0) -> None:
    """Stop every llama-server process of this machine (e.g. one left in tmux)."""
    if psutil is None:
        subprocess.run(["pkill", "-f", "llama-server"], check=False)
        return
    procs = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
            if "llama-server" in (proc.info["name"] or "") or (cmdline and "llama-server" in cmdline[0]):
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for proc in procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class Server:
    """One server process started by the sweep, in its own process group."""

    def __init__(self, cmd: list[str], env: dict[str, str], port: int, log_path: Path):
        self.cmd, self.env, self.port, self.log_path = cmd, env, port, log_path
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.cmd, cwd=REPO_ROOT, env={**os.environ, **self.env},
            stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def wait_ready(self, timeout_s: float) -> str | None:
        """None when /health is 200, else why not."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return f"server exited with code {self.proc.returncode}"
            if _health_ok(self.port):
                return None
            time.sleep(1.0)
        return f"server not ready after {timeout_s:.0f}s"

    def stop(self, timeout_s: float = 15.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        if self.proc is not None:
            self.log_file.close()

    def log_tail(self, lines: int = 15) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def wait_port_free(port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _port_open(port):
            return True
        time.sleep(0.5)
    return False


# ---- sweep ---------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    runs = expand_runs(plan)
    out_dir = Path(args.results_dir) / plan["name"]
    server_cmd = shlex.split(args.server_cmd)

    if args.dry_run:
        for run in runs:
            server = " ".join(f"{k}={v}" for k, v in run["server"].items())
            print(f"{run['run_id']}:\n  server: {server} {args.server_cmd}\n"
                  f"  run:    python -m scripts.run_experiment {' '.join(run['args'])} "
                  f"--run-id {run['run_id']} --results-dir {out_dir}\n"
                  f"  dimensions: {json.dumps(run['dimensions'], sort_keys=True)}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "server_logs").mkdir(exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    log(f"plan {plan['name']}: {len(runs)} runs -> {out_dir}")

    statuses: dict[str, str] = {}
    try:
        for i, run in enumerate(runs, 1):
            statuses[run["run_id"]] = run_one(run, i, len(runs), out_dir, server_cmd, args)
    except KeyboardInterrupt:
        log("interrupted; rerun the same command to resume")
        stop_all_llama_servers()
        return 130
    finally:
        if not args.no_restore:
            restore_default_server()

    done = [out_dir / r["run_id"] for r in runs if (out_dir / r["run_id"] / "requests.csv").exists()]
    log("results: " + ", ".join(f"{k} {v}" for k, v in statuses.items()))
    if done:
        from analysis.summarize import main as summarize_main
        summarize_main([*map(str, done), "--table", str(out_dir / "sweep.csv")])
    failed = [k for k, v in statuses.items() if v == "failed"]
    if failed:
        log(f"failed: {', '.join(failed)}; see {out_dir}/sweep_log.jsonl and server_logs/")
    return 1 if failed else 0


def run_one(run: dict[str, Any], i: int, n: int, out_dir: Path,
            server_cmd: list[str], args: argparse.Namespace) -> str:
    run_id = run["run_id"]
    run_dir = out_dir / run_id
    if (run_dir / "requests.csv").exists():
        log(f"[{i}/{n}] {run_id}: already done, skipping")
        return "skipped"
    if run_dir.exists():
        aside = run_dir.with_name(f"{run_id}.incomplete-{datetime.now():%Y%m%d-%H%M%S}")
        run_dir.rename(aside)
        log(f"[{i}/{n}] {run_id}: moved an incomplete earlier attempt to {aside.name}")

    port = int(run["server"].get("SERVER_PORT", DEFAULT_PORT))
    record: dict[str, Any] = {"run_id": run_id, "server": run["server"], "args": run["args"],
                              "dimensions": run["dimensions"],
                              "started_at": datetime.now().astimezone().isoformat()}
    t0 = time.monotonic()

    log(f"[{i}/{n}] {run_id}: restarting server ({' '.join(f'{k}={v}' for k, v in run['server'].items())})")
    stop_all_llama_servers()
    server = Server(server_cmd, run["server"], port, out_dir / "server_logs" / f"{run_id}.log")
    try:
        if not wait_port_free(port):
            status, why = "failed", f"port {port} still in use after stopping llama-server"
        else:
            server.start()
            why = server.wait_ready(args.ready_timeout_s)
            if why is not None:
                status = "failed"
                log(f"[{i}/{n}] {run_id}: {why}\n{server.log_tail()}")
            else:
                log(f"[{i}/{n}] {run_id}: server ready, running experiment")
                cmd = [sys.executable, "-m", "scripts.run_experiment", *run["args"],
                       "--run-id", run_id, "--results-dir", str(out_dir),
                       "--url", f"http://localhost:{port}",
                       "--dimensions-json", json.dumps(run["dimensions"], separators=(",", ":"))]
                code = subprocess.run(cmd, cwd=REPO_ROOT).returncode
                status = "ok" if code == 0 and (run_dir / "requests.csv").exists() else "failed"
                why = None if status == "ok" else f"run_experiment exited with code {code}"
    finally:
        server.stop()

    record.update({"status": status, "error": why, "duration_s": round(time.monotonic() - t0, 1)})
    with (out_dir / "sweep_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log(f"[{i}/{n}] {run_id}: {status} in {record['duration_s']:.0f}s" + (f" ({why})" if why else ""))
    return status


def restore_default_server() -> None:
    """Leave the team's default server running, as the server guide expects."""
    stop_all_llama_servers()
    try:
        subprocess.run(["tmux", "kill-session", "-t", "llm"], cwd=REPO_ROOT,
                       stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["tmux", "new", "-d", "-s", "llm", DEFAULT_SERVER_CMD], cwd=REPO_ROOT, check=True)
        log("default server restarted in tmux session 'llm'")
    except (OSError, subprocess.CalledProcessError):
        log(f"could not restart the default server; run: tmux new -d -s llm '{DEFAULT_SERVER_CMD}'")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="sweep plan JSON, e.g. scripts/sweeps/phase1_baseline.json")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--dry-run", action="store_true", help="print the runs and exit")
    p.add_argument("--no-restore", action="store_true",
                   help="don't restart the default server in tmux afterwards")
    p.add_argument("--ready-timeout-s", type=float, default=180.0)
    p.add_argument("--server-cmd", default=DEFAULT_SERVER_CMD, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_sweep(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
