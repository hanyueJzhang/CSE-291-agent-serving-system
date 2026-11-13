"""Offline slot replay: how much agent history could placement alone keep?

    python -m analysis.slot_replay results/phase2_diag/w1_fifo_c4_l95 results/phase2_diag/w1_fifo_c5_l95
    python -m analysis.slot_replay <runs...> --hold 0.5 1 2 5 --json out.json

Takes a run's requests in the order and at the times they actually held a
slot (analysis/slot_trace.py) and replays only the choice of *which* free slot
each request lands on. A slot keeps the context of the last request it served;
a follow-up reuses its own history only if that context is still in the slot
it lands on. Timing, dispatch order and the number of free slots at every
instant stay as measured, so the replay isolates placement (proposal 2.2/2.5).

Policies:
  observed      the slot llama-server actually used (checks the content model:
                predicted reuse should match the measured cached tokens)
  llama         llama.cpp 972b5fc: free slot with the largest prompt-prefix
                similarity above 0.1, else least recently used
  lru           least recently used free slot
  dead_first    own slot if it still holds this task's context; else a slot
                whose owner has finished (largest shared prefix, then LRU);
                else the live slot with the least history to lose
  dead_belady   dead_first, but the live victim is the one whose owner comes
                back last (oracle: uses the owner's measured next arrival)

Hold: for every live slot dead_belady still overwrites, the owner's return
gap g = (owner's next arrival - now). Holding the slot for g instead would
have kept that history at the cost of g slot-seconds idle (and delaying the
request that took the slot). For each threshold tau we report the history
tokens and estimated prefill seconds a "hold if g <= tau" rule would save, and
the slot-seconds it would cost. It is first-order: holding shifts later
timings, which the replay does not model. Prefill seconds use the run's mean
prefill time per uncached token.

Cross-task prefix reuse (Mooncake templates) is estimated from the step-0
prompts, which are regenerated from run_config.json's workload.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .slot_trace import Req, Run, load_run

POLICIES = ("observed", "llama", "lru", "dead_first", "dead_belady")
LLAMA_SIM_THRESHOLD = 0.1       # common.h slot_prompt_similarity default
EPS = 1e-6


@dataclass
class Slot:
    idx: int
    owner: Req | None = None      # request whose context the slot holds
    busy_until: float = -1.0
    last_used: float = -1.0


class Replay:
    def __init__(self, run: Run, policy: str):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choices: {POLICIES}")
        if policy == "observed" and not run.has_slots:
            raise ValueError(f"{run.path}: requests.csv has no slot_id; 'observed' needs a Phase 2 run")
        self.run = run
        self.policy = policy
        self.slots = [Slot(i) for i in range(run.n_slots)]
        self.conflicts = 0          # starts with no free slot (timestamp noise)
        self.live_overwrites: list[tuple[float, int]] = []   # (owner return gap s, history tokens lost)
        self.assigned: dict[Req, int] = {}
        self.reuse: dict[Req, int] = {}

    # ---- content model ---------------------------------------------------------

    def reuse_tokens(self, slot: Slot, r: Req) -> int:
        """Prompt tokens of r already in the slot (common prefix)."""
        owner = slot.owner
        if owner is None:
            return 0
        if owner.task_id == r.task_id:
            # Only the step r extends is still useful; an older step of the same
            # task is a prefix of it, so it counts as far as it goes.
            if r.carried and owner.step_idx < r.step_idx:
                return min(owner.held, r.prompt)
            return 0
        return min(self.run.shared_prefix(owner.task_id, r.task_id), owner.held, r.prompt)

    def is_live(self, slot: Slot, now: float) -> bool:
        """Owner still has a later step that has not started yet."""
        owner = slot.owner
        if owner is None or owner.next is None or not owner.next.carried:
            return False
        return owner.next.start > now + EPS

    # ---- policies --------------------------------------------------------------

    def _lru(self, free: list[Slot]) -> Slot:
        return min(free, key=lambda s: (s.last_used, s.idx))

    def choose(self, free: list[Slot], r: Req, now: float) -> Slot:
        if self.policy == "observed":
            want = [s for s in self.slots if s.idx == r.slot_id]
            return want[0]
        if self.policy == "lru":
            return self._lru(free)
        if self.policy == "llama":
            best, best_sim = None, 0.0
            for s in free:
                if s.owner is None or not r.prompt:
                    continue
                sim = self.reuse_tokens(s, r) / r.prompt
                if sim > best_sim and sim > LLAMA_SIM_THRESHOLD:
                    best, best_sim = s, sim
            return best or self._lru(free)

        own = [s for s in free if s.owner is not None and s.owner.task_id == r.task_id
               and self.reuse_tokens(s, r) > 0]
        if own:
            return max(own, key=lambda s: self.reuse_tokens(s, r))
        dead = [s for s in free if not self.is_live(s, now)]
        if dead:
            return max(dead, key=lambda s: (self.reuse_tokens(s, r), -s.last_used, -s.idx))
        if self.policy == "dead_first":
            return min(free, key=lambda s: (s.owner.held, s.last_used))
        # dead_belady: owner that returns last
        return max(free, key=lambda s: (s.owner.next.t_arrive, -s.idx))

    # ---- run -------------------------------------------------------------------

    def run_all(self) -> "Replay":
        for r in self.run.requests:
            now = r.start
            free = [s for s in self.slots if s.busy_until <= now + EPS]
            if self.policy == "observed":
                slot = self.choose(free, r, now)
                if slot not in free:
                    self.conflicts += 1
            elif free:
                slot = self.choose(free, r, now)
            else:
                self.conflicts += 1
                slot = min(self.slots, key=lambda s: s.busy_until)
            if self.is_live(slot, now) and slot.owner.task_id != r.task_id:
                owner = slot.owner
                self.live_overwrites.append((owner.next.t_arrive - now, owner.next.carried))
            self.reuse[r] = self.reuse_tokens(slot, r)
            self.assigned[r] = slot.idx
            slot.owner = r
            slot.busy_until = r.end
            slot.last_used = r.end
        return self

    # ---- results ---------------------------------------------------------------

    def summary(self, holds: list[float]) -> dict[str, Any]:
        reqs = self.run.requests
        fus = [r for r in reqs if r.is_followup]
        history = sum(r.carried for r in fus)
        kept = sum(min(self.reuse[r], r.carried) for r in fus)
        full = sum(self.reuse[r] >= r.carried for r in fus)
        ms_per_tok = self.run.prefill_ms_per_token()
        lost = history - kept
        out: dict[str, Any] = {
            "run": self.run.path.name,
            "policy": self.policy,
            "slots": self.run.n_slots,
            "requests": len(reqs),
            "follow_ups": len(fus),
            "history_tokens": history,
            "history_kept_pct": 100 * kept / history if history else float("nan"),
            "full_hit_follow_ups_pct": 100 * full / len(fus) if fus else float("nan"),
            "lost_tokens_per_follow_up": lost / len(fus) if fus else 0.0,
            "lost_prefill_s_est": lost * ms_per_tok / 1000,
            "first_step_prefix_tokens": sum(self.reuse[r] for r in reqs if r.prev is None),
            "live_overwrites": len(self.live_overwrites),
            "conflicts": self.conflicts,
        }
        if self.run.has_slots and self.policy != "observed":
            same = sum(self.assigned[r] == r.slot_id for r in reqs)
            out["same_slot_as_observed_pct"] = 100 * same / len(reqs) if reqs else float("nan")
        # Content-model check: what the server actually reused. 'observed' should match it;
        # 'llama' should come close if the replay models llama.cpp's choice well.
        measured = sum(min(r.cached, r.carried) for r in fus)
        out["measured_history_kept_pct"] = 100 * measured / history if history else float("nan")
        if self.policy == "dead_belady":
            out["hold"] = [hold_row(self.live_overwrites, tau, ms_per_tok) for tau in holds]
        return out


def hold_row(overwrites: list[tuple[float, int]], tau: float, ms_per_tok: float) -> dict[str, float]:
    held = [(max(g, 0.0), tok) for g, tok in overwrites if g <= tau]
    tokens = sum(tok for _, tok in held)
    return {
        "tau_s": tau,
        "holds": len(held),
        "tokens_saved": tokens,
        "prefill_s_saved_est": tokens * ms_per_tok / 1000,
        "slot_s_idle": sum(g for g, _ in held),
    }


def analyze(run_dir: Path, policies: tuple[str, ...] = POLICIES, holds: list[float] | None = None,
            load: Callable[[Path], Run] = load_run) -> list[dict[str, Any]]:
    holds = holds if holds is not None else [0.5, 1.0, 2.0, 5.0]
    run = load(run_dir)
    rows = []
    for policy in policies:
        if policy == "observed" and not run.has_slots:
            continue
        rows.append(Replay(run, policy).run_all().summary(holds))
    if rows and run.notes:
        rows[0]["notes"] = run.notes
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders under results/")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--hold", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0],
                        help="hold thresholds tau in seconds")
    parser.add_argument("--json", type=Path, help="also write all rows as JSON")
    args = parser.parse_args(argv)

    all_rows = []
    for run_dir in args.runs:
        if not (run_dir / "requests.csv").is_file():
            print(f"skipping {run_dir}: no requests.csv", file=sys.stderr)
            continue
        rows = analyze(run_dir, tuple(args.policies), args.hold)
        all_rows.extend(rows)
        print(f"\n{run_dir}  ({rows[0]['slots']} slots, {rows[0]['follow_ups']} follow-ups, "
              f"{rows[0]['history_tokens']} history tokens)")
        for note in rows[0].get("notes", []):
            print(f"  note: {note}")
        print(f"  {'policy':12s} {'kept%':>6} {'full%':>6} {'lost/fu':>8} {'lost_s':>7} "
              f"{'live_ow':>7} {'same%':>6} {'confl':>5}")
        for r in rows:
            same = r.get("same_slot_as_observed_pct")
            print(f"  {r['policy']:12s} {r['history_kept_pct']:6.1f} {r['full_hit_follow_ups_pct']:6.1f} "
                  f"{r['lost_tokens_per_follow_up']:8.0f} {r['lost_prefill_s_est']:7.1f} "
                  f"{r['live_overwrites']:7d} {'' if same is None else f'{same:6.1f}':>6} {r['conflicts']:5d}")
        print(f"  {'(measured)':12s} {rows[0]['measured_history_kept_pct']:6.1f}   "
              "<- what the server reused; 'observed' / 'llama' should be close")
        for r in rows:
            for h in r.get("hold", []):
                print(f"  hold tau={h['tau_s']:<4g} holds {h['holds']:4d}  saves {h['tokens_saved']:7d} tok "
                      f"~{h['prefill_s_saved_est']:6.1f} s prefill  costs {h['slot_s_idle']:6.1f} slot-s idle")
    if args.json:
        args.json.write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
