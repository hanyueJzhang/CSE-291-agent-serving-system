# Results schema (DRAFT)

Every run writes one folder: `results/<run_id>/` containing

- `run_config.json`: workload file, scheduler + params, llama.cpp commit, model file, server args (`-c`, `--parallel`, `-ngl`), machine, git commit of this repo.
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
| slot_id | llama-server slot used (if reported) |
| t_arrive | request became ready (prev step / tool finished) |
| t_dispatch | scheduler sent it to llama-server |
| t_first_token | first token received (streaming) |
| t_finish | response complete |
| prompt_tokens | prompt tokens (from server response) |
| cached_tokens | prompt tokens served from cache, if reported |
| output_tokens | generated tokens |
| prefill_ms | server-reported prompt eval time |
| decode_ms | server-reported generation time |
| status | ok / error / timeout / truncated |

## tasks.csv
| column | meaning |
|---|---|
| run_id, task_id, task_type | identifiers |
| t_arrive | task arrival |
| t_first_dispatch | first step dispatched |
| t_finish | last step finished |
| num_steps | steps completed |
| total_queue_s | sum of queueing time across steps |
| total_tool_s | sum of tool delays |
| status | ok / failed / timeout |

## memory.csv
| column | meaning |
|---|---|
| t | sample time |
| gpu_mem_used_mb | from nvidia-smi (or pynvml) |
| server_rss_mb | llama-server process RSS |
| active_requests | in-flight requests at sample time |
| queue_len | requests waiting in our scheduler |
