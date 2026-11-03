# Phase 1 报告修订记录（2026-11-02 ～ 11-03）

本文件记录对 `CSE291A_LLM_Phase_1_Report.pdf` 的评审结论、为此补做的实验和数据，以及可直接粘贴进 LaTeX 的修改段落。所有数字都可以用文末的命令从 `results/` 重新算出。

---

## 1. 评审结论

对照项目 prompt 和 TA 反馈（1：需要真实 trace；2：非调度技术，可选），原稿评为 **88/100（A-）**。

主要优点：
- prompt 要求的指标都报了。
- 瓶颈解释清楚：被挤出的 context 需要重新 prefill。
- 用 Mooncake trace 回应了 TA 第 1 条。
- 发现 KV 显存在启动时预分配，显存占用不随负载变化。

扣分点和处理情况：

| # | 问题 | 处理 | 状态 |
|---|---|---|---|
| 1 | “re-prefill 是很大的代价”没有量化 | 新增 `analysis/reprefill.py`，算出占 GPU 忙碌时间的比例 | ✅ 有数据 |
| 2 | trace 输入缩到 1/4、输出不变，负载偏 decode，可能低估 prefill | 新增输出也缩到 1/4 的对照（`mooncake_out4`） | ✅ 有数据 |
| 3 | 46 个 run 都是 0 失败，failures 这一项没有测到 | 新增 context 溢出实验（`phase1_overflow`）和 `analysis/overflow.py` | ✅ 有数据 |
| 4a | 两处吞吐数对不上：0.900 和 0.839 | 原因是任务数不同（200 对 100），收尾空转占比不同。加脚注说明 | ✅ 文字 |
| 4b | 只有 seed 42 | 关键点补跑 seed 43 和 44（`phase1_repeats`），新增 `analysis/repeats.py` | ✅ 有数据 |
| 5 | Phase 2 机制描述模糊 | 改成具体机制：slot 亲和加 follow-up 优先 | ✅ 文字 |

## 2. 新数据

### 2.1 三个 seed 汇总（mean ± s.d.，括号内依次为 seed 42/43/44）

由 `python -m analysis.repeats` 生成。

| 配置 | tasks/s | task P95 (s) | mean queue (s) | follow-up cached | GPU util | 可避免 re-prefill（占 GPU 忙碌时间） |
|---|---|---|---|---|---|---|
| W1 0.50X | 0.166 ± 0.002 | 19.1 ± 1.3 | 0.42 ± 0.31 | 71.5 ± 4.7% | 60.0 ± 2.2% | 3.6 ± 1.3% |
| W1 0.75X | 0.248 ± 0.002 | 29.9 ± 2.3 | 2.8 ± 1.0 | 51.5 ± 4.6% | 78.5 ± 1.4% | 10.7 ± 1.4% |
| W1 0.95X | 0.306 ± 0.006 | **63.5 ± 22.6** (38.5/69.6/82.4) | 17.3 ± 10.3 | 30.9 ± 6.5% | 88.7 ± 0.2% | 19.2 ± 2.6% |
| W1 0.95X + 8 GiB cache | 0.312 ± 0.004 | 41.5 ± 11.6 (28.7/44.5/51.4) | 8.6 ± 4.8 | 78.2 ± 1.7% | 77.0 ± 1.4% | 2.7 ± 0.6% |
| 合成负载 1 task/s | 0.949 ± 0.043 (0.900/0.975/0.972) | 42.2 ± 4.5 | 9.0 ± 1.4 | 7.4 ± 3.1% | 90.9 ± 1.9% | 10.9 ± 0.6% |
| 合成负载 4 tasks/s | 1.018 ± 0.015 | 149.2 ± 4.2 | 57.9 ± 3.2 | 3.2 ± 0.0% | 93.6 ± 0.2% | 12.5 ± 0.7% |
| W1 输出 /4，550 任务 | 0.599 ± 0.008 | 36.0 ± 18.2 | 9.1 ± 5.9 | 33.2 ± 3.8% | 82.3 ± 1.9% | **37.1 ± 1.9%** |
| W1 输出 /4，550 任务 + 8 GiB cache | 0.606 ± 0.000 | 13.8 ± 0.6 | 1.69 ± 0.26 | 78.0 ± 1.0% | 60.8 ± 0.6% | 6.3 ± 0.7% |

host cache 的配对效果（每个 seed 内开/关 cache 的相对变化，再求 mean ± s.d.）：

| | task P95 | mean queue | tasks/s |
|---|---|---|---|
| W1 0.95X | −33 ± 7%（−25/−36/−38） | −50 ± 3% | +1.8 ± 1.0% |
| W1 输出 /4，550 任务 | −56 ± 18%（−40/−75/−53） | −76 ± 14% | +1.2 ± 1.4% |

要点：
- **seed 42 是三个 seed 里尾延迟最低的。** W1 0.95X 的 P95 是 38.5 s，而另外两个 seed 是 69.6 s 和 82.4 s。不同 seed 抽到的 prefix family 不同，负载的突发程度也不同。吞吐、GPU 利用率和 re-prefill 占比则很稳定，变化只有 2–3%。
- **原稿 0.95X 下“P95 降 25.4%”的结论成立，而且偏保守。** 三个 seed 的配对降幅是 33 ± 7%。
- **re-prefill 占比随负载单调上升**：0.50X 约 4%，0.75X 约 11%，0.95X 约 19%。输出也缩到 1/4 后升到约 37%。

### 2.2 context 溢出（`phase1_overflow`，2,048-token slot，0.4 task/s，c = 4）

由 `python -m analysis.overflow results/phase1_overflow/*/` 生成。这个实验的结果只由任务的 context 大小决定，不需要重复 seed。

| 初始 prompt | 任务结果 | 失败方式 |
|---|---|---|
| 512 | 100/100 ok | 无 |
| 1024 | 100/100 **报告为 ok** | **每个任务最后一步（step 4）的输出都被截断到 slot 上限，但状态仍是 ok** |
| 1536 | 0/100 | step 0 成功；step 2 被拒绝，HTTP 400 “request (2057 tokens) exceeds the available context size” |
| 2048 | 0/100 | step 0 就被拒绝，HTTP 400 |

另外，用同一个脚本检查了原有的 46 个 run，截断数全部为 0。

## 3. LaTeX 修改（按位置排列）

### ① 第 3 节表格，Offered load 一行（换成三个 seed 的数字）

原句中 “from 1 to 4 tasks/s it rises only from 0.900 to 1.030 … Task P95 grows from 12.0 s to 44.4 s at 1 task/s and 146.3 s at 4.” 替换为：
```latex
Throughput tracks offered load up to 0.9 tasks/s, then saturates: from 1 to 4 tasks/s it rises only from $0.95\pm0.04$ to $1.02\pm0.02$ (mean $\pm$ s.d., three seeds), with mean GPU utilization at 89--94\% (66\% at 0.5). Task P95 grows from 12.0\,s to $42\pm5$\,s at 1 task/s and $149\pm4$\,s at 4.
```

### ② 第 3 节表格，Concurrency 一行末尾加脚注（4a）
```latex
\footnote{Concurrency runs use 100 tasks versus 200 in the load sweep; the shorter run's drain tail lowers measured throughput (0.839 vs.\ 0.900 tasks/s at the same setting).}
```

### ③ 第 3 节 “All 46 synthetic and Mooncake runs completed every task, with no failed, errored, or timed-out requests.” 整句替换（第 3 条）
```latex
All 46 synthetic and Mooncake runs completed every task with full-length outputs and no failed or timed-out requests. Failures appear only when a task's own context outgrows its slot, regardless of load. On 2,048-token slots, agent tasks with 512- and 1,024-token initial prompts all report success, yet every 1,024-token task's final response is silently truncated at the slot limit; with 1,536 or 2,048 tokens, llama.cpp rejects the second or first request (HTTP 400) after earlier steps have already used the GPU. Completion counts alone therefore overstate success, and context footprint must be checked at admission.
```

### ④ 第 5 节 host cache 那句（换成三个 seed 的数字，并说明尾延迟的波动）

原句 “On W1, 8 GiB reduces task P95 by 13.9% and 25.4% and mean task queue time by 35.3% and 49.7%, while throughput changes by less than 1%.” 替换为：
```latex
On W1, 8\,GiB reduces task P95 by 13.9\% at $0.75X$ and by $33\pm7\%$ at $0.95X$, and mean task queue time by 35.3\% and $50\pm3\%$, while throughput changes by less than 3\% ($0.95X$ values are paired means over three seeds). Tail latency near saturation depends strongly on which prefix families are sampled: across seeds, W1 task P95 at $0.95X$ ranges from 38.5 to 82.4\,s, while throughput and GPU utilization vary by under 3\%. Figures~1 and~2 show seed 42.
```

### ⑤ 第 6 节最后一句 “Re-prefilling displaced context is thus a large, schedulable cost.” 替换（第 1 条，已用三个 seed 的数字）
```latex
To bound this cost, we attribute to each follow-up the prefill time of uncached tokens that were already part of its history, excluding the 300 newly appended tokens. On W1, this avoidable re-prefill accounts for $11\pm1\%$ of GPU-busy time (makespan $\times$ mean utilization) at $0.75X$ and $19\pm3\%$ at $0.95X$ (three seeds); the 8\,GiB host cache reduces it to $3\pm1\%$. Because per-request prefill can overlap other slots' decoding, these are upper bounds. Re-prefilling displaced context is therefore a bounded but schedulable cost: removing it frees at most about one-fifth of GPU time near saturation, and the larger P95 reductions in Section~5 arise because queueing amplifies this saving.
```

### ⑥ 第 7 节 Limitations 第一句（4b）

原句 “Results cover one machine/model, seed 42, and three workload windows; trace host-cache tests cover W1 only.” 替换为：
```latex
Results cover one machine and model; cited W1 and synthetic results are repeated over seeds 42--44, while W2--W3 use seed 42 only, and trace host-cache tests cover W1 only.
```

### ⑦ 第 7 节 Phase 2 机制（第 5 条）

原句 “Phase 2 will add one policy that uses estimated context footprint and ready follow-up status to reduce re-prefill of displaced context.” 替换为：
```latex
Phase 2 will add one policy, slot-affine follow-up priority: a ready follow-up is dispatched ahead of new tasks and pinned to the slot that holds its history (llama.cpp \texttt{id\_slot}), and new tasks are steered to slots whose owner has finished, with aging to bound new-task delay. Section~6 bounds the expected gain: the policy can recover at most the 11--19\% of GPU-busy time spent on avoidable re-prefill on W1 (37\% when outputs are scaled with inputs; Appendix~B), so we expect benefits mainly in queueing and tail latency near saturation rather than in peak throughput.
```

### ⑧ Appendix B，接在 “Workload characteristics” 那段末尾（第 2 条）
```latex
To test whether this understates prefill effects, we reran W1 with every output length also divided by four (same tasks, prompts, and arrivals). At comparable load (550 tasks, $82\pm2\%$ mean GPU utilization versus $89\%$ for the $0.95X$ baseline), avoidable re-prefill (Section~6) rises from $19\pm3\%$ to $37\pm2\%$ of GPU-busy time. An 8\,GiB host cache cuts it to $6\pm1\%$, lowering mean GPU utilization from 82\% to 61\% and task P95 by $56\pm18\%$ at unchanged throughput (three seeds). Our main results are therefore conservative with respect to re-prefill cost.
```

### ⑨ Reproducibility 一句（可选）

在 “Reproducibility:” 那句末尾加上：
```latex
Re-prefill bounds, overflow classification and seed statistics are produced by \texttt{analysis.reprefill}, \texttt{analysis.overflow} and \texttt{analysis.repeats}.
```

> 篇幅：③、⑤、⑦ 会让正文多出大约 8–10 行，可能超过 2 页。如果超了，优先压缩第 6 节开头 “On the synthetic mix … 146 s.” 那句，以及第 3 节 Tool waits 一行的第二句。

## 4. 还没做的

- **图 1 和图 2 还是 seed 42。** 正文已经说明了这一点（④）。如果要画误差条，图 1 的 W1 曲线可以用 `analysis.repeats --json` 的输出；W2 和 W3 仍然只有 seed 42。
- **W1 0.75X 的 host cache 对照没有重复**，所以 ④ 里 13.9% 和 35.3% 这两个数仍是单 seed 的。

## 5. 复现命令

```bash
python -m analysis.repeats                                    # 2.1 汇总表
python -m analysis.reprefill results/mooncake_baseline/w1_fifo_c4_l{75,95}*/ results/mooncake_out4/w1out4_*/
python -m analysis.overflow results/phase1_overflow/*/       # 2.2
```

相关 PR：#21（re-prefill 分析和两个新实验）、#22（溢出文档修正）、#23（实验结果）、#24（seed 重复的 plan），以及本 PR（seed 重复结果、`analysis/repeats.py` 和本文件）。
