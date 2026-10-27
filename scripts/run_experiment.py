"""Run one experiment against the already-running llama-server.

    python -m scripts.run_experiment --workload workload/tasks/example_mix.json --concurrency 4

Writes results/<run_id>/ with run_config.json, requests.csv, tasks.csv,
and memory.csv (format: results/SCHEMA.md). Never starts or
stops llama-server; its settings are read from /props and recorded.

Every prompt starts with a random per-run salt. llama-server keeps prompt
caches across requests (in slots and, by default, in host RAM via
--cache-ram), so without the salt a repeated run of the same workload would
reuse the previous run's KV and look faster than it is. Sharing within a run
(an agent's carried context, a common prefix across tasks) is unaffected.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import platform
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from scheduler import FIFOScheduler
from workload.generator import estimate_peak_context, generate_tasks, load_workload
from workload.llama_client import LlamaClient
from workload.metrics import (
    MEMORY_COLUMNS, REQUEST_COLUMNS, TASK_COLUMNS,
    request_rows, task_rows, write_csv,
)
from workload.runner import run_workload
from workload.sampler import MemorySampler, find_server_process

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEDULERS = {"fifo": FIFOScheduler}


class RunClock:
    """Seconds since reset(); shared by scheduler, client, runner and sampler."""

    def __init__(self):
        self.t0 = time.monotonic()

    def reset(self) -> None:
        self.t0 = time.monotonic()

    def __call__(self) -> float:
        return time.monotonic() - self.t0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", required=True, help="workload JSON, e.g. workload/tasks/example_mix.json")
    p.add_argument("--scheduler", default="fifo", choices=sorted(SCHEDULERS))
    p.add_argument("--concurrency", type=int, required=True, help="max requests in flight")
    p.add_argument("--url", default="http://localhost:8080", help="llama-server base URL")
    p.add_argument("--run-id", help="default: <workload>_<scheduler>_c<N>_<timestamp>")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    # Workload overrides, so sweeps don't need one JSON file per point.
    p.add_argument("--num-tasks", type=int, help="override arrival.num_tasks")
    p.add_argument("--rate", type=float, help="override arrival.rate_per_s")
    p.add_argument("--seed", type=int, help="override the workload seed")
    p.add_argument(
        "--dimensions-json",
        default="{}",
        help="JSON object of experiment dimensions recorded in run_config.json",
    )
    # Client
    p.add_argument("--timeout-s", type=float, default=300.0, help="per-request timeout, from dispatch")
    p.add_argument("--no-ignore-eos", action="store_true", help="let the model stop before max_tokens")
    p.add_argument("--no-cache-prompt", action="store_true", help="ask llama-server not to reuse cached prompt KV")
    # Run
    p.add_argument("--warmup", type=int, default=1, help="unrecorded short requests before the run")
    p.add_argument("--sample-interval-s", type=float, default=0.5)
    p.add_argument("--ready-timeout-s", type=float, default=60.0)
    p.add_argument("--skip-env-check", action="store_true", help="don't run setup/check_env.sh")
    return p.parse_args(argv)


def parse_dimensions(raw: str) -> dict[str, Any]:
    """Parse self-describing experiment dimensions passed by a sweep plan."""
    try:
        dimensions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--dimensions-json is not valid JSON: {exc}") from exc
    if not isinstance(dimensions, dict):
        raise ValueError("--dimensions-json must contain a JSON object")
    invalid = [key for key, value in dimensions.items()
               if not isinstance(key, str) or isinstance(value, (dict, list))]
    if invalid:
        raise ValueError(
            "experiment dimensions must have string keys and scalar JSON values; "
            f"invalid keys: {invalid}"
        )
    return dimensions


# ---- workload ----------------------------------------------------------------

def build_workload(args: argparse.Namespace) -> dict[str, Any]:
    workload = copy.deepcopy(load_workload(args.workload))
    if args.num_tasks is not None:
        workload["arrival"]["num_tasks"] = args.num_tasks
    if args.rate is not None:
        workload["arrival"]["rate_per_s"] = args.rate
    if args.seed is not None:
        workload["seed"] = args.seed
    return workload


def salt_prompts(tasks: list[dict[str, Any]], salt: str) -> None:
    for task in tasks:
        for step in task["steps"]:
            if step["kind"] == "llm":
                step["prompt"] = f"{salt} {step['prompt']}"


# ---- environment and server ------------------------------------------------------

def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_info() -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def pinned_config() -> dict[str, str]:
    """KEY=VALUE pairs from setup/config.env (comments stripped)."""
    values = {}
    path = REPO_ROOT / "setup" / "config.env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


async def _get_json(session: aiohttp.ClientSession, url: str) -> Any:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return None


async def server_info(base_url: str) -> dict[str, Any]:
    """What the running llama-server reports about itself."""
    base_url = base_url.rstrip("/")
    async with aiohttp.ClientSession() as session:
        props, slots = await asyncio.gather(
            _get_json(session, f"{base_url}/props"), _get_json(session, f"{base_url}/slots")
        )
    info: dict[str, Any] = {"url": base_url}
    if props:
        info.update({
            "total_slots": props.get("total_slots"),
            "n_ctx_per_slot": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "model_path": props.get("model_path"),
            "build_info": props.get("build_info"),
            "endpoint_slots": props.get("endpoint_slots"),
            "endpoint_metrics": props.get("endpoint_metrics"),
            "chat_template_sha256": hashlib.sha256(
                (props.get("chat_template") or "").encode()
            ).hexdigest()[:16],
        })
    if isinstance(slots, list):
        info["slots_n_ctx"] = [s.get("n_ctx") for s in slots]
    return info


# ---- run ---------------------------------------------------------------------

async def run_experiment(args: argparse.Namespace) -> Path:
    workload = build_workload(args)
    dimensions = parse_dimensions(args.dimensions_json)
    tasks = generate_tasks(workload)
    salt = f"[run {secrets.token_hex(4)}]"
    salt_prompts(tasks, salt)

    run_id = args.run_id or (
        f"{workload['name']}_{args.scheduler}_c{args.concurrency}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    out_dir = Path(args.results_dir) / run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; pick another --run-id")

    clock = RunClock()
    client = LlamaClient(
        args.url,
        timeout_s=args.timeout_s,
        ignore_eos=not args.no_ignore_eos,
        cache_prompt=not args.no_cache_prompt,
        clock=clock,
    )
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr)

    try:
        await client.wait_ready(timeout_s=args.ready_timeout_s)
        server = await server_info(args.url)
        server_proc = find_server_process(args.url)
        if server_proc is not None:
            try:
                server["cmdline"] = server_proc.cmdline()
            except Exception:
                pass

        total_slots = server.get("total_slots")
        if total_slots and args.concurrency > total_slots:
            warn(f"concurrency {args.concurrency} > llama-server slots {total_slots}: "
                 "extra requests queue inside llama-server, where our queue_s does not see them")
        n_ctx = server.get("n_ctx_per_slot")
        if n_ctx:
            too_big = [t["id"] for t in tasks if estimate_peak_context(t) > 0.95 * n_ctx]
            if too_big:
                warn(f"{len(too_big)} task(s) may not fit the {n_ctx}-token slot context "
                     f"(estimated), e.g. {too_big[:3]}; expect context errors")
        if server.get("endpoint_metrics") is False:
            warn("llama-server /metrics is disabled (--metrics); server_requests_deferred will be blank")
        git = git_info()
        if git["dirty"]:
            warn("repo has uncommitted changes; run_config records the commit, not your edits")

        env_text = None
        if not args.skip_env_check:
            env_text = _run(["bash", "setup/check_env.sh"], timeout=180)

        config = {
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "argv": sys.argv,
            "workload_file": str(args.workload),
            "workload": workload,
            "dimensions": dimensions,
            "num_tasks": len(tasks),
            "prompt_salt": salt,
            "scheduler": None,
            "client": client.config(),
            "server": server,
            "pinned": pinned_config(),
            "repo": git,
            "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                     "python": platform.python_version()},
            "environment": env_text,
            "sample_interval_s": args.sample_interval_s,
            "warmup_requests": args.warmup,
            "warnings": warnings,
        }

        for i in range(args.warmup):
            await client.complete([{"role": "user", "content": f"warmup {salt} {i}"}], max_tokens=8)

        clock.reset()
        scheduler = SCHEDULERS[args.scheduler](args.concurrency, clock=clock)
        config["scheduler"] = scheduler.config()
        out_dir.mkdir(parents=True)
        _write_json(out_dir / "run_config.json", config)

        sampler = MemorySampler(
            clock=clock,
            base_url=args.url,
            interval_s=args.sample_interval_s,
            server_process=server_proc,
        )
        await sampler.start()
        try:
            results = await run_workload(tasks, scheduler, client, clock=clock)
        finally:
            memory = await sampler.stop()
    finally:
        await client.close()

    requests = request_rows(run_id, results)
    task_records = task_rows(run_id, results)
    write_csv(out_dir / "requests.csv", REQUEST_COLUMNS, requests)
    write_csv(out_dir / "tasks.csv", TASK_COLUMNS, task_records)
    write_csv(out_dir / "memory.csv", MEMORY_COLUMNS, memory)

    config.update({
        "status": "done",
        "ended_at": datetime.now().astimezone().isoformat(),
        "sampler_notes": sampler.notes,
    })
    _write_json(out_dir / "run_config.json", config)

    print_check(run_id, out_dir, task_records, requests, memory)
    return out_dir


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def print_check(
    run_id: str,
    out_dir: Path,
    tasks: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    memory: list[dict[str, Any]],
) -> None:
    """A few counts and peaks to judge on the spot whether the run is usable.
    Latency percentiles and throughput are computed from the CSVs in analysis/."""

    def peak(column: str) -> str:
        values = [row[column] for row in memory if row.get(column) is not None]
        return f"{max(values):.0f}" if values else "-"

    counts = {s: sum(t["status"] == s for t in tasks) for s in ("ok", "failed", "timeout")}
    duration = max((t["t_finish"] for t in tasks), default=0.0)
    print(f"\n{run_id}  ->  {out_dir}")
    print(f"  tasks   ok {counts['ok']}/{len(tasks)}   failed {counts['failed']}   "
          f"timeout {counts['timeout']}   duration {duration:.1f}s")
    print(f"  peaks   gpu {peak('gpu_mem_used_mb')} MB   rss {peak('server_rss_mb')} MB   "
          f"slot tokens {peak('slot_tokens_total')}   server deferred {peak('server_requests_deferred')}")
    first_error = next((r["error"] for r in requests if r["error"]), None)
    if first_error:
        print(f"  first error: {first_error}")


def main(argv: list[str] | None = None) -> int:
    asyncio.run(run_experiment(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
