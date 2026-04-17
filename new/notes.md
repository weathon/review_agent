Use LLM to analysis reviews and find problems
消融不用真的做，去掉find human找到的
和cache merger重新直接打分
2026是看质量，所以要混合sample，不看分数

pair wise对比相似主题的文章，一acc一rej看看模型能不能分出来，这个2025还是2026， 2026主要是有点随机，尤其是结果，还是2025
测试cspaper with "published" removed

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
