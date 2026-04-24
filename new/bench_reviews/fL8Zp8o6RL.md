## Summary

FTP (FFN Token Pruning) proposes to accelerate the prefilling stage of long-context LLM inference by dynamically skipping Feed-Forward Network (FFN) computations for tokens deemed non-critical via layer-wise attention scores, while preserving pruned token information through residual connections. Profiling demonstrates FFN accounts for >60% of prefilling time. Evaluated on LongBench across six tasks, FTP achieves up to 1.45× TTFT speedup with generally small accuracy drops on Qwen2-7B and larger models, but suffers a catastrophic 35% drop on Llama3-8B Code Completion—a deviation never discussed.

## Strengths

- **Profiling validates FFN as the dominant cost**: Walltime measurements on Llama3-8B and Qwen2-7B (Figure 3, Table) show FFN consumes over 60% of prefilling time, strongly motivating the target.
- **Novel focus on FFN pruning**: Unlike prior work that prunes tokens for attention/KV cache, FTP uniquely targets the FFN module with a layer-wise, attention-driven selection strategy.
- **Clear algorithmic description**: The method is detailed with equations, pseudo-code (Algorithm 1), and a complete implementation using flash attention.
- **Strong results on most models/tasks**: For Qwen2-7B-Instruct, FTP yields 1.2–1.3× speedup with accuracy drops mostly below 6% relative across six LongBench tasks. Larger models (32B, 72B) show similar or better trade-offs.
- **Attention-based selection is essential**: Ablation comparing random vs. attention-based pruning (Table 3) shows random causes catastrophic failure while attention-based retains performance, confirming the design.
- **Quantification of attention concentration**: Figure 6 demonstrates that 60% of tokens carry 95% of attention scores, justifying the high pruning ratios achievable.

## Weaknesses

### Fatal
None. The method is technically sound and results are reproducible from the tables.

### Major
- **Overgeneralized performance claims and ignored outlier**: The paper repeatedly describes accuracy loss as “negligible” (abstract, line 15; Sec. 3.2.1, line 61; Sec. 4.2, line 280) yet Table 1 shows a 35% relative accuracy collapse on Code Completion for Llama3-8B-Instruct (55.17 → 35.91). This massive outlier is never acknowledged, analyzed, or factored into conclusions, severely undermining claims of broad reliability.
- **No hyperparameter sensitivity study**: Critical hyperparameters η (reserve ratio), F (layers kept intact), P (prefix tokens), and N (suffix tokens) are set arbitrarily per model (line 272) with no justification or exploration of their impact on accuracy and speedup. This leaves practitioners without guidance and raises concerns about robustness.
- **Missing per-dataset breakdown**: Results are aggregated at the task level (e.g., Code Completion combines two datasets). The specific dataset driving the Llama3-8B failure is hidden, preventing diagnosis of the failure mode and limiting understanding of where the method works.
- **No evaluation of decoding impact**: By altering hidden states during prefilling, FTP could affect KV cache contents and decoding latency/throughput. The paper reports only TTFT and final accuracy, omitting any decoding metrics—an important gap for practical utility.

### Minor
- **Misleading terminology**: “FFN Token Pruning” suggests tokens are removed from the sequence; in reality, tokens are retained for attention and only bypassed in FFN. A name like “FFN Token Skipping” would be more accurate.
- **Profiling dataset limitation**: The FFN time proportion (Figure 3) is measured solely on TriviaQA; its generalizability to other dataset types is not demonstrated.
- **Overhead scaling unclear**: The additional ~10 ms for recalculating attention weights (Sec. 4.6.1) is noted, but its dependence on context length and model size is not analyzed.

### Trivial
None.

## Nice‑to‑Haves
- Release code and models to ensure reproducibility.
- Study the characteristics of pruned tokens (position, content) to interpret behavior.
- Provide statistical significance (e.g., confidence intervals) across multiple runs.
- Add a theoretical analysis of why attention scores reliably identify prunable tokens.

## Removed Points

These points are flagged to be removed, treat them with caution:

- _The claim that “attention scores typically concentrate on a small proportion of tokens” relies on visualizing only two samples (Figure 5)._ – **Invalid** because Figure 6 offers a systematic quantitative analysis averaged over all layers and samples; the claim is adequately supported.
- _No comparison with a true token‑removal baseline (reducing sequence length)._ – **Invalid** because the paper compares against LLMLingua2 and PyramidInfer, both of which reduce sequence length (LLMLingua2 compresses the prompt; PyramidInfer prunes the KV cache).
- _The assertion that “shallow layers are more sensitive” is stated without quantitative support._ – **Likely invalid** as the paper directs readers to “more analysis” in Appendix 6.1; the appendix (present in the full submission) presumably contains the supporting evidence.

## Novel Insights
None beyond the paper’s own contributions.

## Suggestions
- Add hyperparameter sensitivity curves (vary η, F) on a held‑out validation set.
- Include per‑dataset results to pinpoint the Llama3‑8B Code Completion failure and any other outliers.
- Measure and report decoding latency and throughput to complete the efficiency picture.
- Clarify terminology: use “FFN token skipping” or “selective FFN bypass”.
- Provide a post‑hoc analysis of the Code Completion failure (e.g., attention pattern differences, token types) to bound the method’s applicability.

---
**Calibration anchors** (compared against):  
- FastGen (uNrFpDPMyo.md, avg 8.0, Accept oral): abundant experiments, ablations, code release—much stronger than this paper.  
- StreamingLLM (NG7sS51zVF.md, avg 7.5, Accept): novel phenomenon, comprehensive model coverage, minor weaknesses.  
- Radar (ZTpWOwMrzQ.md, avg 6.6, Accept): theoretical guarantees, competitive results, solid comparisons; one reviewer 3 but overall strong.  
- Writing in the Margins (56mg1JFd3n.md, avg 6.0, Reject): polarizing scores, missing larger models and sophisticated baselines—similar weaknesses but this paper adds overclaim and lacking key analyses.  
- PyramidDrop (5ncdKonxd4.md, avg 3.0, withdrawn): missing ablations, insufficient experiments, no practical speed numbers—this paper is clearly above that level.

**Positioning**: This paper sits below Radar/StreamingLLM due to multiple major gaps (overclaim, no sensitivity study, no decoding evaluation, missing per-dataset analysis). It shares some weaknesses with the rejected WiM but arguably has more severe overgeneralization. Its core idea and main results are sound for Qwen2‑7B and larger models, but the unaddressed catastrophic failure on Llama3‑8B Code Completion plus missing fundamental analyses keep it below the acceptance threshold.

---
MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>