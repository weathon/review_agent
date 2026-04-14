=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

DLP-LoRA proposes a sentence-level dynamic LoRA fusion system for multi-task LLM inference. A lightweight 4-layer mini-MLP plugin (5M parameters) classifies each input sentence and selects, via top-*p* sampling, which task-specific LoRA adapters to fuse. Combined with a parallel GEMM batching strategy using contiguous HBM allocation, the system reduces multi-LoRA inference overhead to less than 2× the single-LoRA cost. Evaluations span 26 tasks (17 MCQ + 9 QA) across four LLM backbones.

---

## Strengths

- **Concrete inference efficiency advantage over token-level MoE baselines**: Table 7 shows DLP-LoRA achieves 1.20× decoding latency ratio vs. MOLA (10.54×), PESC (3.54×), MoRAL (3.58×), and LoRA-Switch (1.29×) under the same 7-LoRA setting. This is a specific, quantified advantage that directly addresses the bottleneck of token-level gating methods.

- **Lightweight classifier with strong routing accuracy**: The 5M-parameter mini-MLP trains in under 10 minutes and achieves 98.45% task classification accuracy across 26 tasks. Combined with the contiguous-HBM GEMM acceleration, the end-to-end system overhead stays below 2× the backbone latency even at 100 LoRAs (Table 6: 1.83×), a concrete scalability result that many LoRA MoE papers do not provide.

- **Broad backbone evaluation**: Testing across Qwen-2 1.5B/7B, LLaMA-2 7B, and LLaMA-3 8B on 26 tasks provides cross-architecture evidence that the efficiency and performance results are not cherry-picked for a single favorable backbone.

---

## Weaknesses

### Fatal
*None identified. The core efficiency contribution is well-supported.*

### Major

- **No accuracy comparison with Meteora, the paper's primary motivation**: The entire introduction frames Meteora's token-level gating as the key problem to solve, yet Table 7 only compares latency and memory against MOLA, PESC, MoRAL, and LoRA-Switch—not Meteora. There is no experiment showing whether DLP-LoRA matches Meteora's accuracy while achieving its latency advantage. Without this head-to-head, the central efficiency-vs-accuracy trade-off claim cannot be evaluated. This is the most critical empirical gap.

- **Top-*p* threshold never specified**: The threshold *p* is the key hyperparameter governing how many LoRAs are fused per sentence, yet the paper never states its value in the experiments section or appendix. No sensitivity analysis is provided. In the case study (Figure 3), the probabilities shown (50.5%, 49.5%, 100%) suggest *p* must be ≤ 0.495, but since FormFall is shown at 100% yet *not* selected (the paper says AbsNarr and NewsDE are selected), the mechanism is self-contradictory. Specifically, if *p* is a lower-bound threshold and FormFall has 100% probability, it must satisfy the criterion and should always be selected. The case study text (Section 4.3) separately says "NewsDE and FormFall" are selected for the subsequent prompt, yet Figure 3 labels "AbsNarr and NewsDE" as selected. This is a direct inconsistency in both the mechanism's description and its illustration.

- **Sentence boundary detection is not described**: The plugin fires "once the first token of every new sentence is generated" (Figure 1 caption, Section 3.2), but autoregressive generation produces one token at a time. The paper never explains how the system detects that a new sentence has begun during streaming inference. This likely requires buffering until a sentence-ending token appears, introducing latency not accounted for in Table 4. For a paper whose central claim is inference efficiency, this omission is significant.

- **Composite-task baseline is too weak to support the multi-task claim**: Table 3 shows the single combined LoRA (r=64) trained on all 26 tasks barely outperforms the untuned backbone (e.g., for LLaMA-3 8B: 65.98% vs. 65.44%). This indicates the combined LoRA is undertrained, likely because 900 samples × 26 tasks with r=64 provides insufficient capacity. Demonstrating improvement over such a weak multi-task baseline provides little evidence for the composite-task claim. A properly trained combined LoRA, or inclusion of other dynamic routing methods in the composite setting, is needed.

### Minor

- **Inference time inconsistency**: Table 4 shows Qwen-2 1.5B DLP-LoRA (mini-MLP) at 1.12× relative to the backbone, while single LoRA is 1.15×—meaning DLP-LoRA is apparently *faster* than single LoRA despite doing more computation. No explanation is given. The LLaMA-2 7B overhead (1.60× relative to backbone, ~1.52× relative to single LoRA) also contrasts sharply with LLaMA-3 8B (1.11×) for models of similar size; this variation is unexplained. The abstract claim of "1.24 times slower than single LoRA" in Section 3.3 is an average that obscures the 1.52× overhead on LLaMA-2 7B.

- **Mini-MLP input representation is underspecified**: Section 3.1 states the plugin uses "the ALBERT tokenizer," which confirms the 5M parameter count reflects only the MLP itself. However, the paper does not describe how tokenizer outputs (variable-length token ID sequences) are converted into a fixed-size input vector for the MLP. This is essential for reproducibility.

- **GPU hardware is non-standard**: Experiments run on "a single custom-upgraded NVIDIA GTX 2080Ti with 22GB"—the stock 2080Ti has 11GB. This non-standard hardware affects reproducibility and may affect latency comparisons.

- **Notation collision in Equations 4–5**: `w₁` denotes the first token of sentence *S_m* in Eq. 4 and also appears as the first element of the softmax weight set `{w₁, …, w_R}` in Eq. 5. This collision makes Section 3.2 harder to parse.

- **No code or reproducibility checklist**: No code release or hyperparameter table is provided for the mini-MLP (hidden dimensions, learning rate, optimizer). Section 4.1 vaguely states the classifier is "trained on *some* samples"—the exact training data for the classifier is unspecified.

### Tiny

- Table 1 rows are identified by internal dataset shorthand names (AbsNarr, ConParaKC, etc.) that are never decoded in the main paper; this forces readers to the appendix to interpret every result row.

---

## Nice-to-Haves

- **Ablation: sentence-level vs. token-level routing on the same datasets**: The design choice to use sentence-level routing is well-motivated by prior observations, but an empirical comparison (even on a subset of tasks) against a token-level routing method would substantiate the claim that the accuracy-efficiency trade-off favors the sentence-level approach.

- **Failure mode analysis**: No examples or statistics are provided for the ~1.55% of inputs that the mini-MLP misclassifies. A confusion matrix or analysis of which task pairs are most confused would clarify the method's practical risks.

- **LoRA interference analysis**: No experiment tests whether fusing semantically incompatible LoRAs (e.g., machine translation + mathematical reasoning) causes degradation relative to using the correct single LoRA. This would help calibrate when top-*p* fusion helps vs. hurts.

- **Top-*p* vs. top-*k* ablation**: The claim that top-*p* outperforms top-*k* (Section 5) is asserted without experimental comparison. An ablation at matched average adapter count would make this a substantiated design choice.

- **Table 5 framing**: Comparing a fine-tuned small model against a larger untuned model is a well-known and expected result in the field. Replacing LLaMA-2 13B with a fine-tuned version would make this a more informative comparison; as-is, the 19.89 BLEU of LLaMA-2 13B likely reflects instruction formatting failure rather than model capacity.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **"92.34% accuracy is not a meaningful summary"** (Harsh Critic): The figure is arithmetically correct (macro-average across four backbone averages). The critic's point that several tasks saturate at 100% is true but does not constitute a factual error in the paper's claim.

- **"Contribution 3 is not surprising"** (Harsh Critic): This is a subjective argument about novelty framing, not a substantive empirical or methodological failure.

- **"The comparison against Meteora on accuracy is an unfair baseline for DLP-LoRA"** (Harsh Critic, Table 7): The latency comparisons in Table 7 are against different methods and on different numbers of tasks; this is noted but the core criticism (missing Meteora accuracy comparison) is retained as a Major weakness, not removed.

- **"Table 5 is methodologically flawed and should be removed"** (Harsh Critic): Too strong. The comparison is positioned as a discussion point about deployment scenarios, not as a core experimental claim. The concern is retained as a Nice-to-Have.

- **"Test set size of ~90 examples is too small to draw reliable conclusions"** (Harsh Critic): Results are averaged across 10 runs and across 26 diverse tasks. While small per-task test sets are not ideal, this is not uncommon in multi-task benchmarking and the multi-run averaging provides some mitigation. Requesting per-task confidence intervals is above the norm for this setting.

- **"Missing related works"**: Per reviewer instructions, removed entirely.

---

## Novel Insights

The most genuinely novel observation across all three reviews is the tension between the sentence-level routing design choice and the case study's self-contradiction regarding FormFall (100% probability, not selected). If correctly diagnosed, this suggests the top-*p* mechanism may not function as described—or that the case study was not generated by the actual system. This would undermine confidence in the entire fusion mechanism and is worth the authors' attention independent of reviewer skepticism. Beyond this, the efficiency comparison in Table 7 across multiple token-level and sentence-level MoE baselines using a consistent 7-LoRA setup is one of the more carefully controlled efficiency evaluations in recent LoRA-fusion literature, and the sub-linear scaling of latency with LoRA count (Table 6: 50 LoRAs→1.76×, 100 LoRAs→1.83×) is a practically useful datapoint that similar papers have not reported.

---

## Suggestions

1. **Report Meteora accuracy on the same 26 tasks in a unified table**, even if only on a subset. This single experiment would substantially address the most critical gap.
2. **State the value of *p* used** in all experiments and include a sensitivity sweep (e.g., p ∈ {0.3, 0.5, 0.7, 0.9}).
3. **Fix the FormFall inconsistency in Figure 3/Section 4.3**: either correct the probabilities, the list of selected LoRAs, or explain why a 100%-probability LoRA is not selected.
4. **Describe sentence boundary detection explicitly**: specify whether the system uses a punctuation heuristic, a token classifier, or another mechanism, and quantify any buffering latency added.
5. **Provide a hyperparameter table** (mini-MLP hidden size, layers, learning rate, optimizer, training data source and size for the classifier, LoRA rank used).
6. **Clarify the Qwen-2 1.5B anomaly** in Table 4 where DLP-LoRA (1.12×) is faster than single LoRA (1.15×).

---

**Axis evaluations:**
- **Novelty**: Moderate. Sentence-level LoRA routing exists (Polytropon, Flix), but the specific combination with a detached mini-MLP plugin and parallel GEMM acceleration is a distinguishable contribution.
- **Technical soundness**: Below average for ICLR. The methodology description has several underspecified components, a case-study self-contradiction, and an unresolved efficiency anomaly.
- **Empirical support**: Moderate. The efficiency comparison is reasonably thorough; the accuracy evaluation omits the primary competing method and relies on a weak composite baseline.
- **Significance**: Moderate. Efficient multi-LoRA inference at scale is a real deployment problem, and the latency results are practically meaningful.
- **Clarity**: Weak. Multiple notation issues, unexplained figure inconsistencies, and vague descriptions of key components reduce confidence in the work.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
