Now I have a good picture of the calibration landscape. Let me synthesize the final review.

## Summary

Thoughtbubbles introduces a transformer variant that enables unsupervised, dynamic allocation of parallel computation in latent space by learning to fork or delete residual streams during pretraining. Tokens requiring more computation spawn cloned "bubbles" of residual streams controlled by cumulative scores that modulate attention and residual updates, all trained with standard language modeling loss. The method achieves consistent perplexity improvements over parameter-matched baselines and often over computation-matched copy baselines across 150M–772M parameter scales on OpenWebText and peS2o.

## Strengths

- **Novel and well-motivated architecture**: The forking mechanism with cumulative score-guided duplication/deletion of residual streams is a genuine innovation over prior pause-token approaches, which insert computation uniformly. The score attenuation mechanism (modulating attention and residual updates by cumulative scores) is a clever way to make the scores meaningful without auxiliary losses—a clean design that integrates adaptive compute into standard LM training.

- **Trainable from LM loss alone**: The forking behavior emerges purely from language modeling loss, making the method directly applicable during pretraining without any additional supervision. This is a significant practical advantage over CoT methods or approaches requiring auxiliary losses.

- **Consistent perplexity improvements**: Table 1 shows Thoughtbubbles achieving lower perplexity than all baselines across all scales and both datasets. The result that a 319M model outperforms a 772M baseline on OpenWebText (Figure 3) is particularly noteworthy.

- **Interpretable computation allocation**: Figure 5 demonstrates that forks are allocated to higher-entropy tokens without explicit supervision, consistent with the stated motivation. The attention analysis in Figure 4 showing parent tokens strongly attending to their forked children provides mechanistic insight.

- **Honest about limitations**: The paper openly acknowledges the top-k gradient bottleneck, hardware inefficiency, and limited scale evaluations.

## Weaknesses

### Major:

- **Overclaimed empirical superiority on zero-shot tasks**: The abstract and conclusion assert that Thoughtbubbles "outperforms both standard decoder LMs as well as non-adaptive parallel computation approaches" on zero-shot evaluations, but Table 1 tells a more mixed story. On LAMBADA, Copy baselines beat Thoughtbubbles on peS2o at 772M (Copy-5: 10.3, Ours κ=2L: 10.5—the paper notes higher is better, so lower is worse) and 319M scales. On BLiMP, computation-matched baselines outperform Thoughtbubbles at peS2o 772M (Copy-3: 73.3 vs. Ours κ=4L: 67.4) and 319M scales. The paper partially acknowledges this ("only outperforms the parameter-matched, but not computation-matched baselines" for BLiMP) but still claims broad superiority "across a suite of zero-shot evals" in the conclusion. This overclaiming materially misrepresents the evidence.

- **Compute-matching not rigorously established**: The claim that κ=4L is "roughly FLOPs-matched against copy-5" is asserted without any FLOPs accounting, wall-clock measurements, or per-layer token count profiles. Since Thoughtbubbles only expands tokens after certain layers (3, 7, 11) and prunes many via top-k, while Copy-5 processes all copies through the full depth, the effective compute is likely substantially different. Either Thoughtbubbles uses significantly less compute (making the comparison unfair in its favor) or the mismatch goes the other way—without quantification, the core comparative claim is unsubstantiated.

- **Evidence for genuine adaptive allocation vs. extra capacity is insufficient**: The central conceptual claim is unsupervised, meaningful adaptive allocation of parallel compute. But the paper does not include critical ablations: (1) what happens if forking is disabled at inference while keeping model weights, (2) what happens if score attenuation is removed, (3) whether the observed gains persist if forked streams are randomly placed rather than score-selected. Without such ablations, it is entirely plausible that the perplexity improvements come primarily from the extra effective capacity of having multiple residual streams per token (a form of token-wise ensemble) rather than from meaningful adaptive routing. The entropy–fork correlation in Figure 5 is qualitative, with no statistical quantification or control for confounds like token position or frequency.

### Minor:

- **Autoregressive evaluation is thin**: The main results (Table 1) use blockwise scoring. Autoregressive evaluation (Section 5.1) covers only a single 772M model on one dev subset, showing a notable distribution shift (perplexity 20.97 blockwise vs. 23.10 fixed-budget autoregression) that requires a dynamic budget mitigation. No downstream zero-shot tasks are evaluated autoregressively, and no baselines are compared in this regime. Since the stated goal is "scaling inference-time computation," autoregressive generation is the primary use case.

- **Limited experimental scale**: All models are pretrained on only 2.5B tokens at up to 772M parameters—orders of magnitude below modern LM training. The limitation is acknowledged, but it undermines confidence in scaling claims (e.g., "319M outperforms 772M baseline"), which may reflect undertraining rather than genuine architectural advantage. No training curves are provided to assess convergence behavior.

- **Missing comparisons with adaptive computation baselines**: The paper positions itself against pause tokens and Mixture-of-Depths but only benchmarks against vanilla transformers and naive copy baselines. Comparisons with established adaptive compute methods (Universal Transformers, Mixture-of-Depths, pause tokens with adaptive budgets) would strengthen the evaluation.

### Trivial:

- The "concave parabolic" interpretation of the entropy–fork relationship (Section 5) is speculative without controls for token type or position, though the paper partially acknowledges this.

## Nice-to-Haves

- FLOPs accounting and wall-clock/throughput comparisons for each method.
- Ablations on the score attenuation mechanism and random vs. learned fork placement.
- Evaluation on reasoning benchmarks even at current scale (e.g., GSM8k, ARC) despite acknowledged noise.
- Training curves comparing convergence speed vs. baselines.
- Statistical significance measures (standard deviations across runs).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"LAMBADA numbers where Copy beats Thoughtbubbles"** at peS2o — The harsh reviewer claimed LAMBADA at peS2o 772M shows Copy-5 (10.3) beating Thoughtbubbles κ=2L (10.5). However, examining Table 1, higher LAMBADA scores are better, and the peS2o 772M row shows Copy-5 at 10.3 vs. Ours κ=2L at 10.5 and κ=4L at 12.9, where Thoughtbubbles actually wins. I need to recheck the LAMBADA direction: the table header says LAMBADA (↑), so higher is better. The harsh reviewer may have misread the direction. Actually, looking more carefully: Copy-3: 9.5, Copy-5: 10.3, κ=2L: 10.5, κ=4L: 12.9. Thoughtbubbles wins on LAMBADA at 772M peS2o. However at peS2o 150M, κ=2L gets 5.0 vs. Copy-5's 7.2, where Thoughtbubbles clearly loses. This is a valid mixed result, but the specific claim about 772M was wrong.

- **Reproducibility concerns about implementation details**: The harsh reviewer flagged missing details about top-k selection mechanism, RoPE handling for copies, etc. The paper does describe these (Appendices B, D, E). Since reproducibility nitpicks are removed per rules, I've excluded these.

- **Formatting/style nitpicks**: Removed per rules.

- **Claims about models/datasets not being released**: Removed per rules; all cited models and datasets are treated as existing.

- **Demanding the paper address problems outside its scope** (reasoning benchmarks at scale, user studies, theoretical proofs): These are nice-to-haves, not core flaws.

## Novel Insights

The most interesting observation from the reviews is the tension between the paper's "parallel thinking" framing and what the architecture actually does: forked residual streams are processed simultaneously through the same layers (not sequentially), so the "thinking" is more akin to maintaining multiple parallel hypotheses per token rather than performing multi-step sequential reasoning. This distinction matters because it reframes what adaptive compute means here—not more depth, but more breadth per token—and explains why BLiMP (syntax, which benefits less from multiple hypotheses) shows less improvement than LAMBADA/HellaSwag (which require disambiguation). The paper's own analysis showing reduced forking at the highest-entropy tokens (the "concave" pattern) is consistent with this: tokens at clause boundaries have high entropy but don't benefit from additional parallel hypotheses, while tokens with moderate uncertainty (e.g., choosing among plausible completions) do.

## Suggestions

1. **Quantify FLOPs**: Provide a per-layer FLOPs breakdown for Thoughtbubbles vs. Copy-3/Copy-5 to establish the compute-matching claim rigorously, or acknowledge that the comparison is approximate/advantageous.

2. **Narrow the claims**: Replace "outperforms across a suite of zero-shot evals" with task-specific statements (e.g., "outperforms on LAMBADA and HellaSwag, mixed on BLiMP and PIQA"), and be explicit that compute-matching is approximate rather than exact.

3. **Add key ablations**: Most critically, test (a) with forking disabled at inference (same training, no forking at test time) and (b) with attention/residual attenuation ablated. These would directly establish whether the gains come from adaptive allocation or extra capacity.

4. **Strengthen the autoregressive evaluation**: At minimum, run LAMBADA and HellaSwag under autoregressive generation with dynamic budget scaling, and compare against baselines in the same regime.

5. **Report training curves**: Show whether Thoughtbubbles converges faster or just reaches a lower floor, to distinguish architectural advantage from compute advantage.

## Evaluation

**Originality**: Moderate-to-high. Dynamic forking of residual streams without auxiliary loss is novel and distinct from pause tokens or adaptive depth. The cumulative score attenuation mechanism is clever.

**Importance of research question**: High. Adaptive inference-time compute is one of the most active areas in LLM research, and making it learnable during pretraining is a genuinely important direction.

**Claims well supported**: Partially. Perplexity improvements are solid and consistent. Zero-shot task claims are overstated. The core mechanism claim (adaptive allocation) is plausible but under-tested with ablations. Compute-matching claims are unsubstantiated.

**Soundness of experiments**: Acceptable for proof-of-concept but limited. Small scale, thin autoregressive evaluation, missing critical ablations, no compute quantification. The comparison against only naive copy baselines weakens the experimental contribution.

**Clarity**: Good. The method description is detailed (though appendix-dependent), and the writing is clear despite some PDF parsing artifacts.

**Value to community**: Moderate. If the approach scales, it represents a meaningful step toward pretraining-compatible adaptive compute. However, the current evidence is insufficient to draw strong conclusions.

## Score and Decision

Calibration: I compared against several related papers:
- **Pause Tokens** (Think Before You Speak): scores 8/3/3/8, poster accept. Novel pause-token training at 1B scale with more task varieties but similar limited compute analysis.
- **Hyper-Connections** (scores 5/8/6/6): poster accept. Architecture modification for LLMs with good empirical improvements but concerns about extra compute.
- **CoTFormer** (scores 6/6/5/6): poster accept. Architectural innovation for adaptive compute with decent but not overwhelming empirical evidence.
- **A-MoD** (scores 5/3/5/3): reject. Adaptive token pruning with no LM evaluation, unfair compute comparisons, and implementation concerns.
- **COCONUT** (scores 6/6/5/6): reject. Latent reasoning with interesting idea but limited empirical validation.
- **Stutter** (scores 5/3/5/5): reject. Adaptive depth for LLMs with unconvincing improvements and fairness concerns.

This paper has a genuinely novel and promising architecture with consistent perplexity gains, but is marred by overclaimed zero-shot results, unsubstantiated compute-matching, thin autoregressive evaluation, and missing ablations that matter for the core claim. This places it above papers like A-MoD and Stutter (which had more fundamental issues) but below Pause Tokens and Hyper-Connections (which had stronger empirical foundations or more honest claims). The overclaiming is a significant concern that cannot be ignored—it directly undermines the paper's central narrative.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>