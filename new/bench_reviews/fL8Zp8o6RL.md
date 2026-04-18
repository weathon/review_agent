Now let me compile the final review, carefully verifying each claimed weakness against the paper.

## Summary

The paper proposes FFN Token Pruning (FTP), a training-free method that accelerates the prefilling stage of long-context LLM inference by selectively bypassing FFN computation for tokens deemed unimportant by attention scores. Pruned tokens retain their hidden states through residual connections, preserving information while reducing computation. Experiments on LongBench across Llama3-8B, Qwen2-7B, Qwen1.5-32B, and Qwen2-72B show TTFT speedups of 1.2–1.45× with moderate accuracy impact.

## Strengths

- **Well-motivated and novel focus on FFN computation.** The paper correctly identifies that FFN accounts for over 60% of inference time per decoder layer during prefilling (Figure 3), which is an underexplored optimization target compared to KV cache compression. The core idea—prune tokens before FFN rather than attention, leveraging residual connections—is simple, clean, and architecturally sound.

- **Convincing ablation on selection strategy.** Table 3 provides a strong internal ablation: random pruning with the same token budget devastates accuracy (e.g., Code Completion drops from 55.17→16.28 on Llama3) while offering nearly identical TTFT savings, clearly validating that the attention-based criterion is doing meaningful work rather than just providing speedup from arbitrary token dropping.

- **Strong scaling results to larger models.** Table 2 demonstrates that FTP achieves 1.31–1.45× TTFT speedups on Qwen1.5-32B and Qwen2-72B, showing the method is not limited to small models and that deeper architectures permit more aggressive pruning, strengthening practical relevance.

- **Compatibility with FlashAttention acknowledged and addressed.** The paper is transparent about the requirement to recompute attention weights (since FlashAttention does not return them) and quantifies the overhead (1–3% of TTFT, Section 4.6.1), though this analysis is limited (see Weaknesses).

## Weaknesses

### Fatal
None.

### Major

- **No end-to-end latency evaluation undermines the core practical claim.** The paper frames its contribution as "efficient long-context LLM inference" (title, abstract: "long-context LLM inference"), yet only reports TTFT speedup. Figure 2 itself shows that prefilling constitutes only 23.7% of total inference time on RepoBench-P, meaning a 1.25× TTFT speedup translates to roughly 1.06× end-to-end speedup on such workloads. Without reporting total inference time (prefilling + decoding), the practical significance of FTP is indeterminate. The paper cannot claim general inference efficiency when it measures only one component, especially when that component's share varies dramatically across tasks.

- **Unfair comparison with PyramidInfer.** The paper's primary baseline, PyramidInfer*, uses PyTorch attention (not FlashAttention), making it artificially slow. The authors' own reimplementation with FlashAttention ("PyramidInfer" without asterisk) shows better TTFT speedup but the paper does not establish that this reimplementation faithfully reproduces the original method's compression schedule or that its hyperparameters are comparably tuned. The paper acknowledges PyramidInfer targets both prefilling and decoding, but reports only TTFT speedup, stacking the comparison metric in FTP's favor. This asymmetric evaluation makes it impossible to determine whether FTP or PyramidInfer provides better overall inference acceleration.

- **Severe accuracy degradation on Code Completion is unaddressed.** On Llama3-8B, Code Completion drops from 55.17 to 35.91 (a 19.26 absolute point / 34.9% relative drop), yet the abstract and introduction claim "only a negligible decrease in performance." This is not negligible for a task category; it suggests FTP may systematically fail on structured tasks where many tokens carry critical syntactic information. The paper averages across tasks and never discusses this failure mode, which undermines the generality claim.

- **LazyLLM is missing from baselines.** LazyLLM (Fu et al., 2024) is directly cited in the paper as a prefilling-focused token pruning method and is the most closely related work. It is discussed in Related Work and Section 1 ("they either yield subtle speedup during the prefilling stage or defer a portion of computations to the decoding phase"), but never evaluated experimentally. Since LazyLLM also targets TTFT acceleration without training, its absence from the comparison table is a significant gap for claims of superiority over prior prefilling methods.

### Minor

- **Hyperparameter selection lacks systematic justification.** The paper sets F=10, η=0.90 (Llama3) and η=0.95 (Qwen2-7B) without explaining how these were chosen. No ablation on F, η, P, or N is provided beyond the Pareto curves in Figure 7, which vary η but do not explore F. The paper claims "dynamically determined" pruning (Section 3.2.1) via η, but η itself is a static per-model hyperparameter.

- **Evaluation is confined to a single benchmark family.** All results come from LongBench with average context lengths of 5K–15K tokens, while the models support 8K–128K. No evaluation at truly long contexts (50K+) or on harder benchmarks (RULER, Needle-in-a-Haystack) is provided. The aggregated "Score" metric also obscures per-dataset variation, making it hard to assess whether some datasets degrade substantially.

- **Attention score recalculation overhead is under-analyzed.** The paper reports 7–15ms overhead (1–3% of TTFT) for two models on one dataset, but does not provide a breakdown by context length, layer, or operation type. Since recomputing softmax attention weights scales quadratically with sequence length, this overhead could grow substantially for 128K context inputs—the regime where FTP matters most.

### Trivial
- Minor typo in the conclusion: "In a addition" should be "In addition" (Page 8).

## Nice-to-Haves

- Report end-to-end latency (prefilling + decoding) alongside TTFT to establish real-world speedup.
- Include LazyLLM as a baseline, since it directly targets prefilling acceleration.
- Add per-dataset results (not just task-averaged scores) to reveal whether Code Completion's degradation is representative of a broader pattern.
- Analyze why code completion degrades significantly and whether adaptive η or task-specific configurations could mitigate it.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No comparison with MoD, LayerSkip, or other module-skipping methods":** These are training-based methods (MoD requires training a router; LayerSkip requires training with layer dropout) that fundamentally differ from FTP's training-free approach. Comparing a training-free method against methods requiring retraining is not a fair or informative comparison. (From Harsh Critic Point 3)

- **"PyramidInfer accelerates decoding too, so it might be better end-to-end":** This is not a valid criticism because it argues FTP should beat baselines on a dimension it does not target. If anything, this would make PyramidInfer *stronger* as a baseline, which means FTP beating it on TTFT is a harder comparison bar to clear, not an unfair one. However, the concern about not reporting end-to-end metrics at all is kept as Major Weakness 1.

- **"Missing MInference, SnapKV, Quest as baselines":** MInference, SnapKV, and Quest primarily target decoding-stage KV cache compression, not prefilling acceleration. They are not appropriate baselines for a method that specifically optimizes TTFT. LazyLLM *is* an appropriate baseline and its absence is noted. (From Spark and Human Finder)

- **"Reproducibility concerns about undisclosed hyperparameters":** The paper specifies all main hyperparameters (P, N, F, η) for each model. Minor tuning details are not a substantive weakness for an empirical systems paper. (From Harsh Critic)

- **"No variance/confidence intervals":** While good practice, single-run evaluation without confidence intervals is the norm for benchmark evaluation in this field (LongBench has 200 samples per dataset). This is a nice-to-have, not a substantive weakness. (From Harsh Critic)

- **"FTP sometimes exceeds baseline accuracy":** The Harsh Critic flags this as needing explanation, but this is likely measurement noise from the aggregated metric and small evaluation sets. It does not indicate a fundamental problem. (From Harsh Critic)

- **"Formatting nitpicks"**: Removed per rules. (From Harsh Critic Section-by-Section Notes)

## Novel Insights

The most insightful observation from the review synthesis is the tension between FTP's strength and its primary limitation: FTP's clever use of residual connections to preserve pruned token information is what allows it to maintain accuracy while dropping FFN computation, but this same mechanism means pruned tokens still propagate through all subsequent attention operations unchanged. This creates an asymmetry—on tasks where FFN updates are critical for building precise representations (e.g., code syntax, structured reasoning), even preserving the residual cannot compensate for the lost non-linear transformation. The Code Completion results starkly reveal this: the 35% relative drop is far worse than any other task, suggesting that code tasks rely disproportionately on token-wise FFN updates rather than just attention-based information mixing. This hints at a deeper structural insight: the effectiveness of any FFN-pruning method will be fundamentally bounded by how much the downstream task depends on per-token non-linear feature transformations versus inter-token information routing.

## Suggestions

1. **Report end-to-end inference latency** alongside TTFT for all tasks, and break down prefilling vs. decoding time so readers can assess real-world speedup across different workload regimes.

2. **Add LazyLLM as a baseline**—it is the most directly comparable method (training-free, prefilling-focused token pruning) and is already discussed in the paper.

3. **Investigate and discuss the Code Completion failure mode** explicitly. At minimum, report per-dataset scores and identify whether specific structured tasks are incompatible with FFT-level pruning.

4. **Report per-layer pruning rates** (what fraction of tokens are pruned at each layer depth under a given η), so readers can understand the method's dynamic adaptation behavior rather than just seeing aggregate speedups.

## Score and Decision

Calibration papers:
- **LazyLLM** (prefilling token pruning, similar domain): scores 6,5,6,3 → avg ~5, Reject
- **UNComp** (KV cache compression, similar domain): scores 6,5,6,5 → avg ~5.5, Reject
- **OrthoRank** (token selection for inference): scores 6,5,5,5 → avg ~5.25, Reject
- **Double Sparsity** (sparse attention, long-context): scores 5,8,3,3 → avg ~4.75, Reject
- **Training-Free Exponential Context Extension** (KV cache, strong results): scores 6,6,6,6 → avg 6, Accept (Poster)
- **Activation Beacon** (long context compression, strong): scores 8,8,6,6 → avg 7, Accept (Poster)

FTP is comparable to LazyLLM and UNComp in quality: it has a clear and well-motivated idea, good internal ablations, and scaling to larger models. However, it shares similar weaknesses to those rejected papers (missing baselines, limited evaluation scope, incomplete overhead analysis). The paper's lack of end-to-end measures and the fair comparison issues with PyramidInfer are more severe than typical weaknesses in this cluster. The paper has genuine novelty and the random-pruning ablation is compelling, but the evaluation does not adequately support its claims of practical inference efficiency. It falls slightly below the "Accept" threshold papers but above truly weak submissions.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>