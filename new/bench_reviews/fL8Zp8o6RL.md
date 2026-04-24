## Summary

The paper proposes **FFN Token Pruning (FTP)**, a training-free method to accelerate long-context LLM prefilling by pruning non-critical tokens before the FFN module, which the authors empirically show dominates per-layer walltime (>60%, Figure 3). FTP uses per-layer cumulative attention scores to dynamically select tokens to retain, zeros out the FFN output for pruned tokens, and passes them through unchanged via the residual connection. On LongBench, FTP achieves measured TTFT speedups of 1.2–1.45× on models from 7B to 72B parameters with modest accuracy loss on many tasks, but the evaluation suffers from a structurally misleading baseline comparison and an unacknowledged catastrophic accuracy collapse on Llama3 Code Completion that directly contradicts the paper’s central robustness claim.

## Strengths

- **Targets a genuine, under-explored bottleneck.** The paper empirically establishes that the FFN module consumes 61.3–62.4% of per-layer prefilling walltime (Figure 3), then designs FTP to reduce FFN compute rather than attention or KV-cache size. This is a sensible shift in focus and yields concrete TTFT speedups (1.22–1.30× on Qwen2-7B, 1.37–1.45× on Qwen1.5-32B; Tables 1–2).
- **Dynamic, layer-adaptive pruning with residual preservation.** FTP determines both *which* and *how many* tokens to keep per layer by thresholding cumulative attention scores (Eq. 2–3), adapting to the varying concentration observed across layers (Figure 5). The design to zero out FFN outputs while keeping tokens in the hidden-state stream (Section 3.2, Figure 4) is validated by the catastrophic degradation under random pruning (Table 3: e.g., Llama3 Single-Document QA drops from 37.20 to 11.14).
- **Training-free and low-overhead.** FTP requires no fine-tuning or architectural changes (Section 4.1) and is implementable in a concise PyTorch loop (Algorithm 1). The attention-score recomputation adds only 1–3% to TTFT (Section 4.6.1), confirming that the speedups reflect genuine walltime reductions.

## Weaknesses

### Fatal
None.

### Major
- **Misleading baseline comparison undermines the SOTA claim.** The paper dismisses PyramidInfer’s official implementation (PyramidInfer*) as “fails to accelerate prefilling” because it uses PyTorch-native attention (Section 4.3, Table 1). This conflates backend inefficiency with algorithmic limitation and is used to motivate that prior work yields only “subtle speedup.” While the authors reimplement PyramidInfer with Flash Attention, the reimplemented version outperforms FTP by nearly 20 accuracy points on Llama3 Code Completion (55.24 vs. 35.91) with comparable speedup (1.10× vs. 1.19×; Table 1). The unqualified summary that PyramidInfer suffers “more accuracy degradation” obscures this counterexample and weakens the paper’s positioning against prior work.
- **Cherry-picked robustness claim masks a severe, unacknowledged failure mode.** The abstract and introduction advertise a “1.30% performance drop” and a “negligible decrease in performance,” but these figures are drawn only from the Qwen2-7B-Instruct task average. Table 1 reveals that on Llama3-8B-Instruct Code Completion (RepoBench-P), accuracy collapses from 55.17 to 35.91—a 35% relative drop—with no explanation, diagnosis, per-dataset variance, or warning that FTP can catastrophically degrade performance on standard tasks. This directly contradicts the central claim of general robustness and raises safety concerns for practitioners.

### Minor
- **Unsupported claim of surpassing baseline accuracy.** Section 4.4 asserts that FTP “even surpasses that of the baseline in certain tasks (e.g., Single-Document QA and Synthetic Task),” yet neither Table 1 nor Table 2 contains a single instance where FTP exceeds the baseline on those tasks. This claim appears to be factually unsupported.
- **Heuristic justification gap.** The paper uses attention scores from the last *N* queries as a proxy for FFN importance (Section 3.2.1), but offers no empirical or theoretical argument linking low attention to dispensable FFN updates. Concentration of attention (Figure 5) does not automatically imply that tokens receiving low attention can safely skip the non-linear FFN transformation.
- **Fixed hyperparameters without cross-length justification.** The position-based reserves *P* = 100 and *N* = 50 are held constant across models and sequence lengths ranging from 4k to 32k (Section 4.1). The paper gives no rationale for why these constants are appropriate regardless of input scale.
- **Unexplained baseline behavior.** LLMLingua2 is reported to achieve <1× prefilling speedup even with a compression ratio of 0.2 (Section 4.2), but the paper offers no discussion of compressor overhead or other causes, making this baseline comparison uninformative.

### Trivial
None.

## Nice-to-Haves
- **Position-only truncation baseline.** Compare FTP against a simple baseline that reserves the same number of tokens based only on position (e.g., first *P* + last *N* + a middle chunk) without attention-based selection, to isolate how much benefit comes from attention-based ranking versus static head/tail preservation.
- **Accuracy–speedup curve for the failure task.** Plot accuracy vs. TTFT speedup as *η* varies specifically for Llama3 Code Completion. If the curve shows a cliff rather than graceful degradation, this reveals the method is unsafe for inputs near that task distribution.
- **Statistical reporting.** Report standard deviations or confidence intervals for LongBench scores; with ~200 samples per dataset (500 for code completion), point averages alone are insufficient to support “1.30% drop” claims.
- **Layer-wise importance alignment analysis.** Provide evidence that attention scores in layer *l* actually correlate with a token’s need for the FFN update in layer *l*, strengthening the heuristic’s mechanistic grounding.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **Random ablation preserves first *P* and last *N* tokens:** The paper does not state that the random variant reserves the first *P* and last *N* tokens; it says only that the same number of tokens are pruned randomly. This criticism is unsubstantiated.
- **Unreported tuning differences in reimplemented PyramidInfer:** Speculative and not grounded in any evidence from the paper.
- **Typos, grammar, and formatting nitpicks:** These are parser artifacts from the PDF extraction, not author errors.
- **Missing appendix or proofs:** The parser strips those sections; they exist in the original submission.

## Novel Insights

The observation that FFN dominates prefilling walltime is not new in isolation, but shifting token pruning from the attention/KV-cache axis to the FFN module represents a genuinely under-explored angle. The paper’s most novel design insight is the choice to zero out FFN outputs while preserving pruned tokens in the hidden-state stream via the residual connection, allowing subsequent attention layers to still attend to them. This decouples compute reduction from context eviction in a way that prior prefilling accelerators do not. However, the paper does not fully capitalize on this insight mechanistically—linking attention scores to FFN sensitivity remains an open question that, if addressed, could turn a heuristic into a principled method.

## Suggestions
- Rerun the PyramidInfer comparison with explicitly matched backends and **directly discuss the Llama3 Code Completion outlier** in the main text; if the collapse is systematic, the method cannot claim general robustness without task-specific guardrails or restrictions.
- Add standard-error bars or report per-dataset variance to support quantitative robustness claims, especially given the small sample sizes.
- Provide a position-only truncation ablation to verify that attention-based ranking provides value beyond static head/tail preservation.

## Score and Decision

**Calibration anchors used:**
- **High:** `/home/wg25r/review_agent/human_reviews/yUC8pU508S.md` (avg 6.4, Accept Poster) — praised for thorough empirical analysis, practical training-free implementation, and drastic prefilling speedup. Our paper shares the training-free, prefilling-speedup angle but lacks its thorough baseline matching and suffers a severe accuracy collapse on one task, placing it below this anchor.
- **High:** `/home/wg25r/review_agent/human_reviews/ZTpWOwMrzQ.md` (avg 6.6, Accept Poster) — strong for its theoretical justification and competitive results against streaming and sliding-window attention. Our paper lacks theoretical grounding and has baseline fairness issues, so it sits below this anchor.
- **Medium:** `/home/wg25r/review_agent/human_reviews/SYv9b4juom.md` (avg 5.25, Reject) — criticized for limited experimental scope (fixed sequence length) and insufficient comparisons. Our paper has broader scope but shares similar baseline-concern and accuracy-degradation patterns, placing it near this anchor.
- **Medium:** `/home/wg25r/review_agent/human_reviews/mMmzHS28ht.md` (avg 5.00, Reject) — noted for substantial accuracy degradation at modest compression ratios. Our paper shows a comparable degradation on one task but better overall results, so it is slightly above this anchor.
- **Low:** `/home/wg25r/review_agent/human_reviews/gcEhF4nuYI.md` (avg 3.00, Withdrawn) — flagged for unclear contribution and unfair baseline comparisons (training vs. training-free). Our paper has a clearer contribution and less severe unfairness, so it is well above this anchor.
- **Low:** `/home/wg25r/review_agent/human_reviews/4QWPCTLq20.md` (avg 3.00, Withdrawn) — criticized for confusing presentation, insufficient baselines, and lack of ablations. Our paper is clearer, includes ablations, and has broader experiments, placing it above this anchor.
- **Quality-based:** `/home/wg25r/review_agent/human_reviews/S1qSHSFOew.md` (avg 5.50, Reject) — explicitly called out for overclaiming SOTA robustness and unfair comparisons (A+B+C vs. A+B). Our paper shares a very similar pattern of overclaiming and selective baseline treatment, suggesting a comparable score band.

**Comparison:** The paper has a real, practical contribution and strong results on many configurations, which lifts it above the low-scoring cluster (≤4). However, the structurally misleading dismissal of PyramidInfer based on backend mismatch, the cherry-picked “1.30% drop” figure, and the catastrophic Llama3 Code Completion failure—none of which are adequately explained—are issues that reviewers in the medium band (4.5–5.5) consistently weigh against acceptance. The severity of the robustness gap and the unfair baseline treatment align the paper most closely with S1qSHSFOew (5.5) and mMmzHS28ht (5.0). Because the Llama3 failure is present in the paper’s own main results table and directly undermines the central robustness claim, a score at the lower end of this band is warranted.

**Score:** 5.0

**Decision:** Reject

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>