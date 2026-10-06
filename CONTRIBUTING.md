# How we work

1. Never push directly to `main`. Create a branch: `git checkout -b <name>/<topic>` (e.g. `james/fifo-queue`).
2. Open a Pull Request into `main`; at least one teammate reviews before merge.
3. Keep PRs small and focused (one feature or fix).
4. Do not commit model files (`*.gguf`) or big raw logs; `.gitignore` covers them.
5. Changing the pinned model, quantization, or llama.cpp commit (`setup/config.env`) needs agreement from the whole team, since it invalidates earlier results.
6. Every experiment run writes its config next to its data (see `results/SCHEMA.md`) so it can be reproduced.
