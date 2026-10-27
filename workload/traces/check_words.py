"""Check the trace prompt words against the running llama-server's /tokenize.

    python -m workload.traces.check_words [--url http://localhost:8080]

Trace prompts (mooncake.py) assume every word in words.txt
is one token after a space, and that blocks tokenize the same alone as joined
(including after the run salt). Then a prompt of N words is N tokens and two
prompts sharing k blocks share exactly k * BLOCK_WORDS tokens. Exit 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request

from .mooncake import _words, block_text, scaled_input_tokens, trace_prompt, load_trace

SALT = "[run 1a2b3c4d]"
TRACE = {"path": "workload/traces/toolagent_trace.jsonl"}


def tokenize(url: str, text: str) -> list[int]:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{url}/tokenize", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["tokens"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--pairs", type=int, default=200, help="random block pairs to check at the seam")
    args = p.parse_args(argv)
    url = args.url.rstrip("/")
    failures = []

    words = _words()
    multi = [w for w in words if len(tokenize(url, " " + w)) != 1]
    if multi:
        failures.append(f"{len(multi)} words are not one token: {multi[:10]}")

    rng = random.Random(0)
    salt = tokenize(url, SALT)
    for _ in range(args.pairs):
        a, b = rng.randrange(100_000), rng.randrange(100_000)
        ta, tb = tokenize(url, " " + block_text(a)), tokenize(url, " " + block_text(b))
        joined = tokenize(url, f"{SALT} {block_text(a)} {block_text(b)}")
        if joined != salt + ta + tb:
            failures.append(f"seam differs for blocks {a}, {b}")
            break

    rows, _ = load_trace(TRACE)
    for row in rng.sample(rows, 50):
        n = scaled_input_tokens(row["input_length"])
        got = len(tokenize(url, " " + trace_prompt(row["hash_ids"], n)))
        if got != n:
            failures.append(f"prompt of {n} words is {got} tokens")
            break

    print(f"{len(words)} words, {args.pairs} seams, 50 prompts checked against {url}")
    for failure in failures:
        print(f"FAIL {failure}")
    print("OK" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
