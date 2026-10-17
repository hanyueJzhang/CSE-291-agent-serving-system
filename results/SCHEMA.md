# Results schema

Every run writes one folder: `results/<run_id>/` containing

- `run_config.json`: everything needed to reproduce the run.
  - **Setup:** command, workload (after CLI overrides), scheduler, client settings.
  - **Server:** slots, context per slot, model, build, launch command (as reported by llama-server).
  - **Code and machine:** repo commit, `setup/config.env`, `setup/check_env.sh` output.
  - **Run:** start/end time, status, `prompt_salt`.
  - **Check before using the data:** `warnings`, `sampler_notes`.
- `requests.csv`: one row per LLM request (agent step).
- `tasks.csv`: one row per agent task.
- `memory.csv`: periodic memory samples.

All timestamps are seconds since run start (monotonic clock).

## requests.csv
| column | meaning |
|---|---|
| run_id | run identifier |
| task_id | parent task |
| step_idx | step index within the task |
| t_arrive | request became ready (prev step / tool finished) |
| t_dispatch | scheduler sent it to llama-server |
| t_first_token | first token received (streaming) |
| t_finish | response complete |
| prompt_tokens | prompt tokens (from server response) |
| cached_tokens | prompt tokens served from cache, if reported |
| output_tokens | generated tokens (= the step's max_output_tokens) |
| prefill_ms | server-reported prompt eval time (uncached tokens only) |
| decode_ms | server-reported generation time |
| status | ok / error / timeout |
| error | server or client error message |

## tasks.csv
| column | meaning |
|---|---|
| run_id, task_id, task_type | identifiers |
| t_arrive | task arrival |
| t_finish | task ended |
| num_steps | steps completed |
| total_queue_s | sum of queueing time across steps |
| total_tool_s | sum of tool delays |
| status | ok / failed / timeout |

## memory.csv
| column | meaning |
|---|---|
| t | sample time |
| gpu_mem_used_mb, gpu_util_pct | from NVML (pynvml) |
| server_rss_mb | llama-server process RSS |
| slot_tokens_total | tokens held across llama-server slots, i.e. KV cache in use |
| server_requests_deferred | normally 0; > 0 means requests queued inside llama-server, which needs rerun|
