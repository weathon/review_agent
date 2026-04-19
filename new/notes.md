Use LLM to analysis reviews and find problems
消融不用真的做，去掉find human找到的
和cache merger重新直接打分
2026是看质量，所以要混合sample，不看分数

pair wise对比相似主题的文章，一acc一rej看看模型能不能分出来，这个2025还是2026， 2026主要是有点随机，尤其是结果，还是2025
测试cspaper with "published" removed


pred human range，不用linear，而是直接看分布

---

## 2026-04-17  bench 2026 为什么 pred 挤在 4.5-5.5 的调查

走过几个错误假设,最终结论放在最前面。

### 结论

Pipeline 对极端低分/高分论文**系统性地不敢给极端分**,把所有东西往 4.5-5.5 推。与 paper 顺序、sample 大小、时段、Ollama Cloud drift 都无关 — 是 pipeline 本身的 calibration 问题。

支持证据:
- 前 14 篇 vs 后 28 篇, 在 **gt ∈ [3,7] 的子集**上 MAE 完全一致 (0.89),Pearson 都接近 0。pipeline 质量两段一样。
- 残差 std 前 14=1.42, 后 28=1.47。噪声水平一致。
- Pearson 0.19 vs 0.78 的差异**纯粹来自 gt 分布**: 前 14 只有 2/14=14% 是极端 gt (<3 或 >7), 后 28 有 18/28=64%。同样的噪声放在方差大的 gt 上出高 Pearson, 放在窄 gt 上出低 Pearson。
- Per-bin accept rate: pred 57 篇里 32 篇 (53%) 挤在 [5,6), accept rate 47% (近 coin-flip, 无信息)。人类单评分在同 bin 边界分辨率远更高 (score 0→0%, 2→5%, 4→33%, 6→64%, 8→79%, 10→100%)。
- Single-paper rerun (避开并发/顺序效应): Xn33bU71m4 bench 4.5 → single 3.5, ulXBsAYSwU bench 4.0 → single 4.5。两次方向相反, ±0.5-1.0, 属于 LLM sampling variance, 不能系统性"救出"极端分。
- Xn33bU71m4 (gt=1.33) single 3.5, 还是远离 gt。pipeline 读得出 "这是烂文章", 但不敢打 2 以下。

### 走过的错误假设 (按被推翻顺序)

1. **PDF parser contamination** — 约一半 2026 论文前面被 PDF 解析器拉出 "108 109 110 ..." 的行号垃圾,100+ 个 `\b\d{1,3}\s+\d{1,3}...\b` run。Correlation(line_num_runs, |pred-gt|)=0.38, 真的有影响,但单独清 parse 只能贡献 ~+0.5 (VbTLgEUocp clean-parse rerun: 4.5 → 5.0), 解释不了 2.5 分的 gap。 *次要因素, 值得修, 但不是主因。*

2. **2026 bench 是 reviewer-disagreement-heavy subset** — `iclr2026_cspaper/papers/` 只有 203 / 571 的 ratings.csv, 这个 disk subset 的 one-vs-rest reviewer r=0.609 vs full ratings.csv 的 0.722。真的有偏。但不能解释 pipeline 在所有 gt 上都压缩,也不能解释 pred std 恒低。 *真实但非主因。*

3. **小样本 variance** — N=15 vs N=40 看起来故事完全不一样 (std 0.40 → 0.83, 范围 [4.5,5.5] → [2.5,7.0])。曾误以为"跑得多就没事了"。**错的**, 因为同样 pipeline 在 gt∈[3,7] 的 matched subset 上两段 MAE 完全一致, 区别是后 28 碰到更多极端 gt。

4. **Ollama Cloud 时段 drift** — 02:17-02:41 Pearson=0.19, 02:45 后 Pearson 爬升。曾误以为 Ollama Cloud 在早上 2 点前后从 degraded 恢复了。**错的**, 因为 per-gt-bin MAE 两段一致, 不是 model 质量变化, 是 gt 分布变化。

5. **Pipeline session warm-up (冷启动 / CONCURRENCY=8×4 并发冲击)** — 曾假设前 15 篇受 Ollama 模型加载 / prompt cache 未建立 / 32 并发首冲击影响。**错的**, 因为 single-paper rerun (完全没有并发、没有 queue) 也没给出显著不同的分数。

6. **前 14 篇是内容上刚好奇怪** — 曾假设这些 paper 恰好是难判别的。**错的**, 因为这 14 篇 gt 从 1.33 到 7.5 跨度很大, 说明内容本身有强信号, 只是 pipeline 不采用。

### 关键教训

- 单看 Pearson/MAE 容易被 gt 分布骗。**控制 gt 范围后再对比两段子集**, 才能判断 pipeline 质量是否真的不同。
- MAE 看起来 OK (~1.1) 不代表 pipeline 有信号。如果 pred 恒为 5, gt 均值 5, MAE 会很低但 Pearson=0。
- Single-paper rerun 是隔离 concurrency/warm-up 的干净实验, 但一次不够, 要多次统计才能区分 sampling variance 和系统效应。
- 记忆不要急着写, 改几版之后的往往是对的。

### 验证用的 commands

```bash
# rolling window by completion time
grep "Timestamp:" pipeline_whole_2026.log | head; grep -c "Paper:" pipeline_whole_2026.log
# per-gt-bin MAE comparison
python3 -c "..." # 见 /tmp/corr_check*.py

# PDF line-number contamination
grep -cE '^\s*[0-9]{3}\s+[0-9]{3}\s+[0-9]{3}' iclr2026_cspaper/papers/*.txt

# single-paper rerun (no concurrency)
python main.py --single_paper <path>
```

metric.py 里加了 per-bin accept rate (pred vs individual human) — 这是最直观看到 pipeline 压缩的方式。

---

## 2026-04-17 (晚) Pipeline 精简: 砍掉 Spark + Human Finder, Merger 改 Claude SDK + Haiku 子代理

### 改动

1. **砍 Spark agent**: 把它的 "Missing Experiments / Deeper Analysis / Visualizations / Next Steps" 角色合并进 Harsh Critic 的 prompt 里(harsh_critic.md 新增 "Missing Parts and Places to Improve" section, 在 Strengths 和 Overall Assessment 之间)。

2. **砍 Human Finder**: 用日志分析评估了它的实际贡献:
   - 它吃掉单篇 ~55% 的 token (avg input 395k vs 全 pipeline 730k)
   - 它产出的 weakness 里 ~65% 是 noise (40-45% transplanted - 直接从相似论文抄过来; 15-20% generic; 5% hallucinated; 只有 30-35% 是真的 paper-grounded)
   - 这 30-35% paper-grounded 的也大多和 Harsh Critic / Neutral 重叠
   - 唯一独特的产出是 calibration anchor list, 但 Merger 自己就有 search/grep/read 工具可以做
   - **结论**: 整个 agent 删掉, calibration 让 Merger 直接做

3. **Merger 改 Claude SDK + 子代理 calibration_search**:
   - Merger 之前 ~302k input / ~12 turns, 因为每个 search 结果都累积进 context, 反复传 12 次
   - 把 search/read/grep 工具下放给 Haiku 跑的子代理 calibration_search (通过 SDK 的 Task 工具调用)
   - Merger 只看到子代理返回的 short paper-list summary, search 中间结果不污染主上下文
   - 子代理是 *retrieval* 角色 (返回 paper path + 一句话 why match), **不做** calibration reasoning
   - 子代理约束: 一次 call 只查一个 attribute (occasional 两个); 按分数找 anchor 时只要 topic loosely related, 不要完全同 topic (避免重复返回)
   - Merger 限制最多 4 次 calibration_search invocation

4. **Harsh Critic 也走 Claude SDK** (HARSH_MODEL=claude_sdk:claude-sonnet-4-6):
   - SDK CLI 有输入长度限制, 论文不能直接 inline, 给 read_file 工具让它自己读
   - 同 Merger 共用 _run_claude_sdk_query helper, 自动捕获 ResultMessage 里的 cost/usage

5. **Token/cost tracking**:
   - 之前只记 OpenAI agent 的 token, SDK merger 是 N/A
   - 现在每个 SDK agent 单独记 cost (USD) / turns / input / output / cache_read / cache_creation
   - log 里有 `--- Claude SDK Usage ---` section + TOTAL 行
   - run_single_paper 终端也打印这块
   - meta data 上 OpenAI usage 和 SDK usage 分开两个 dict (agent_usages vs sdk_usages)

6. **OpenAI Merger path 临时禁用** (raise NotImplementedError, 代码注释掉但留着): 集中精力优化 SDK 路径。

### 单论文测试结果 (../paper.md, FaceLinkGen)

- Final score: 6.5 / Accept (定性看也合理)
- 总耗时 ~6 min, 总 cost $0.55:
  - Harsh Critic: $0.234, 5 turns, 9.5k output, 43k cache read
  - Merger: $0.315, 9 turns, 8.2k output, 132k cache read
- 比之前 GLM merger 的 raw input ~302k 降到 ~174k (12 + 129k cache_read + 44k cache_create), 主要 win 来自子代理隔离 search context
- 主观看 review 质量也比 GLM 版强 (尤其 weakness 的具体 grounding 和 score calibration 解释)

### 没解决 / 待办

- 主代理子代理共用 MCP server name "merger_fs", print prefix 都显示 "[merger:read_file]", 看不出哪个是主哪个是子, 需要重命名 print label
- Subagent 跑过几次没有被实际 invoke (Merger 直接绕过), 要看是不是 prompt 不够强制 / 或者 Merger 觉得不需要 calibration
- 还没跑 batch 验证整体 MAE/Pearson 在 ICLR2025/2026 上有没有改善, 单论文不能下结论

---

## 2026-04-18  Pearson 从 0.19 跳到 0.83 的原因

ICLR2025 unbalanced, random 200 sample, seed 2545463167。N=19 partial 结果:
- MAE = 0.80
- **Pearson = 0.83** (上次类似 regime 只有 0.19)
- decision_match = 13/19 (68%)

跟之前"pipeline 挤在 4.5-5.5"的病状对比, 这次 pred range 3.0-6.5 已经打开, gt 范围 2.5-8.0 基本覆盖。

### 做对了什么 (按重要性排)

1. **human_reviews 重采样到 7k balanced corpus** (最大 win)。原来 17791 篇集中在 bin 4/5/6 (14k), 极端 bin 几乎没有。 Merger retrieval 每次都只拿到 middle-clustered anchors, 于是 paper 永远被 placed 在 "between middle anchors" → 4.5-5.5。重采样到 bin 距离 max/min 15× → 5× 后, retrieval 能拿到真正 3 分和 8 分的 anchor, merger 才能 place paper 到极端区间。
   - HIGH paper (gt=8) Sonnet merger: 全 17k corpus → 5.0 Reject; 7k balanced → 6.5 Accept。单独这一步就解决了 ceiling 问题。

2. **Harsh Critic / Merger 换成 Claude Sonnet (SDK)** (第二大 win)。同 corpus 同 prompt 下, GLM merger 在 HIGH paper 只能到 5.5 Reject, Sonnet 到 6.5 Accept。GLM 的 filter-Harsh 能力弱, 容易把 parser-artifact 当真 flaw (比如 "Theorem 3 没有 proof" 实际是 appendix 被 parser 删了, GLM 照抄, Sonnet 能识别)。

3. **Neutral Reviewer → Strength Finder** (中等 win)。明确分工: 只找 evidence-backed strengths, 不写 weakness。merger.md 加了"strength 与 major weakness 冲突时 weakness 胜, 去掉 superficial/delusional 的 strength"的规则。这避免了 HIGH paper 里 strength 被 merger 一笔带过。

4. **Merger prompt 加了 "score from anchors, not from how the review feels"** (中等 win)。之前 merger 会自己说"this paper sits between X and Y" 然后给低于 X 的分数 (self-contradict)。现在强调 anchor 是 ground truth, gut feeling 不算。

5. **Parser-artifact rule explicit**: "appendix/references 被 parser 删了, 不要当成 missing" — 让 merger 忽略那类假 weakness。

6. **cut Human Finder + Spark**: 前者占 55% token 但只贡献 30% 真 signal (65% noise); 后者大部分和 Harsh overlap。删掉后 noise 减少, Merger 注意力更集中在真实的 Harsh weaknesses 和 Strength Finder strengths。

7. **Subagent 隔离 calibration retrieval**: Merger 不直接跑 search/grep, 而是派 Haiku subagent (Claude SDK) 或 as_tool (OpenAI SDK) 去查。Merger 上下文里只剩 "anchor paper list + 一句话总结", 不会被 search 结果堆成 300k context。

### 注意事项

- 这次是 **unbalanced random sample**, gt 分布本身就集中 (bin 6 占 33%), 所以 decision accuracy 看起来高一部分是 sample 本身偏 middle。Pearson 0.83 不受这个影响因为是 scale-invariant, 但 MAE 和 decision_match 在 balanced sample 上会比现在难看。
- 这些改动是 **组合拳**, 单独拿任何一个单改动都只能移动 0.5-1 分, 效果不显著。只有多个改动一起上才打开极端分的 ceiling 和 floor。
