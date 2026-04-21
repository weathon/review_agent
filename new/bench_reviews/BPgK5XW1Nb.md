Now I have enough calibration information. Let me compose the final review.

## Summary

The paper proposes Spread Preference Annotation (SPA), an iterative self-play framework for LLM alignment that uses only a small amount of human-annotated preference data (as little as 3.3% of UltraFeedback). SPA generates new preference data by sampling responses from the current model and labeling preferences using the model's own implicit reward (Eq. 7), rather than external reward models or LLM-as-judge. It also introduces a self-refinement mechanism with de-coupled noise detection (DND) via logit extrapolation to reduce label noise. Experiments on AlpacaEval 2.0 show SPA with 3.3% data outperforming Zephyr-7b-β trained on 100% data, and iterative DPO baselines using PairRM or LLM-as-judge.

## Strengths

- **Strong empirical performance with minimal labeled data**: SPA achieves 15.39% LC win rate and 21.13% original win rate on AlpacaEval 2.0 using only 3.3% of UltraFeedback labels, surpassing Zephyr-7b-β trained on the full dataset (11.75% LC, 10.03% original) as shown in Table 1. This data efficiency result is striking and practically meaningful.

- **Clear advantage over alternative judgment methods**: Table 2 shows SPA (15.39% LC) substantially outperforms Iterative DPO with PairRM (11.87%) and LLM-as-judge (9.28%). Figure 3 provides supporting evidence that the performance gap widens across iterations, consistent with SPA's implicit reward co-evolving with the model distribution while external reward models suffer from distribution shift.

- **Ablation study isolating components**: Table 6 decomposes SPA into data expansion (DE), self-refinement (SR), and de-coupled noise detection (DND), showing each contributes: DE alone gives 14.41% LC, adding SR improves to 14.70%, and adding DND brings it to 15.39%.

- **Generalization across models and seed sizes**: Table 3 shows consistent improvements across 0.8%–10% seed data. Table 5 demonstrates improvements on Phi-2, LLaMA-3-8B, and Phi-3-14B. Figure 4 shows SPA works even without seed preference data using Mistral-7B-instruct as the starting model.

- **Simple implementation**: As noted in Section 4.2, SPA requires only a few lines of additional code on top of standard DPO, since DND reuses logits already computed for the DPO objective.

## Weaknesses

### Fatal
None.

### Major

- **Confounded comparison between judgment methods and learning algorithm**: Table 2 compares SPA (which includes self-refinement + DND) against Iterative DPO baselines (which do not include SR+DND). The paper states in Section 5.1 that baselines are "the same in the case of changing the judgment method and removing self-refinement in SPA." This means the headline comparison attributes the full performance gap to the preference judgment method, when part of it comes from SR+DND. The ablation in Table 6 shows DND alone adds +0.98% LC (14.41→15.39) and +1.22% original win rate. The statement in Section 5.2 that "the results reveal the superiority of our direct preference judgment over other judgment methods" overclaims, since the judgment method was never evaluated in isolation against baselines also equipped with SR+DND. That said, SPA without SR+DND (14.41% LC) still outperforms PairRM (11.87%), so the core advantage of direct preference judgment holds—just not to the degree claimed.

- **No validation that self-generated labels agree with human preferences**: After iteration 0, preference labels are entirely self-generated (Eq. 7-8). The paper claims to "spread human prior knowledge" from seed data, but provides no evidence that the model's implicit reward generalizes human preferences rather than amplifying its own biases. A straightforward diagnostic—measuring agreement between SPA's self-generated labels and held-out human labels on expanded prompts—would substantiate the core mechanism. Without it, the "spreading" claim is unsupported. This concern is partially mitigated by Figure 3 (gap widening at iteration 2 supports distribution-shift argument) and by the empirical success of the method, but the lack of label quality validation leaves a gap in understanding why the method works.

### Minor

- **Length exploitation gap**: SPA's original win rate (21.13%) substantially exceeds its LC win rate (15.39%), a 5.74 percentage-point gap. By contrast, PairRM's LC win rate (11.87%) actually exceeds its original win rate (9.46%). This suggests SPA produces longer responses that inflate unadjusted metrics. The LC win rate still surpasses baselines, so this doesn't invalidate the results, but the paper does not discuss this as a limitation or analyze response length distributions.

- **Theoretical gap in logit extrapolation (Eq. 12)**: The linear extrapolation $h_{\bar{g}} = (1+\lambda)h_\theta - \lambda h_{\text{ref}}$ approximates a "more strongly aligned" model, but no derivation shows this produces valid probabilities or corresponds to a well-defined alignment procedure. The citation to Liu et al. (2024) motivates the idea but does not formally justify this specific operation. Since DND adds +0.98% LC improvement, this gap is consequential but the method works empirically.

- **High variance with seed selection**: Table 4 reports LC win rate variance of 2.10 for SPA (range 13.77%–16.38%) versus 0.16 for DPO. The paper argues the lowest confidence interval value (13.36%) exceeds the strongest baseline (11.87%), which is true, but the 13x increase in variance is notable and not thoroughly analyzed.

- **"Without seed preference data" claim is misleading**: Figure 4 uses Mistral-7B-instruct as $\pi_0$, which is already an RLHF-aligned model. The paper claims to demonstrate SPA works "without seed preference data," but the starting model has human preference information baked into its weights. The paper does note this setup, but the framing overclaims.

### Trivial
None.

## Nice-to-Haves

- **Agreement rate between self-generated labels and held-out human labels** on expanded prompts would directly validate the "spreading" mechanism and resolve the circularity concern.
- **Iterative DPO baselines with SR+DND** would isolate the contribution of the judgment method and make the comparison in Table 2 fully fair.
- **Analysis of response length distributions** for SPA vs. baselines would clarify the degree of length exploitation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that Zephyr comparison is misleading (Harsh Critic Section 5.1)**: The paper explicitly states that Zephyr uses "the same base model (Mistral-7B-0.1v) and SFT dataset" but "significantly larger labeled preference data." The comparison is positioned as showing data efficiency, not as a same-recipe comparison. The paper is transparent about what Zephyr is.

- **Claim that the reference model asymmetry is unexplained (Harsh Critic Section 4.1)**: The paper discusses this in Table 7 and Section 5.3, showing SFT reference outperforms previous-iteration reference. The choice is analyzed and justified empirically.

- **Claim that DND is the largest contribution among refinement components (Harsh Critic)**: Table 6 shows data expansion (DE) alone improves from 9.03% (DPO) to 14.41% LC—a 5.38% gain. SR+DND together add only +0.98% LC. The direct preference judgment (DE) is by far the largest contributor.

- **Claim that warm-up is "not justified" for self-refinement (Harsh Critic Section 4.2)**: Warm-up is a standard technique in training; dismissing it as unjustified is overly demanding for an empirical paper.

- **Formatting/typo nitpicks**: Removed per rules.

- **Missing related works**: Removed per rules (no external sources to verify).

- **Missing appendix content**: Removed per rules (parser strips appendices).

## Novel Insights

The key insight is that the implicit reward from DPO (which requires no external model) can serve as a surprisingly effective preference judge for self-generated data in iterative alignment, particularly because it co-evolves with the training distribution and thus avoids the distribution shift that degrades fixed external reward models. The distribution shift argument in Figure 3—where the performance gap between SPA and PairRM widens specifically at iteration 2—is the most compelling evidence for this. However, the paper does not adequately disentangle whether this advantage comes from the judgment mechanism itself or from the SR+DND learning algorithm bundled with it.

## Suggestions

- Run the baselines (PairRM, LLM-as-judge) with SR+DND enabled to produce an apples-to-apples comparison in Table 2. If SPA's judgment method still outperforms, the claim is strongly supported.
- Measure agreement between self-generated preference labels and held-out human labels (e.g., from the unused portion of UltraFeedback) across iterations to validate the "spreading" mechanism.
- Report response length distributions for SPA vs. baselines to address the length exploitation concern head-on.

## Score and Decision

**Calibration anchors:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| Self-Alignment w/ Instruction Backtranslation | 1oijHJBRsT | 8.0 (Oral) | Stronger than SPA: genuinely novel backtranslation idea, clean methodology, no confounded comparisons |
| Weak-to-Strong Preference Optimization | f7KxfUrRSb | 7.25 (Spotlight) | Comparable novelty; WSPO has clearer methodology but limited model diversity |
| INPO | Pujt3ADZgI | 6.0 (Oral) | Similar iterative alignment theme; SPA has stronger empirical results but confounded comparisons |
| Meta-Rewarding LLMs | lbj0i29Z92 | 5.0 (Reject) | Similar self-play alignment; SPA is more complete with ablations and broader experiments |
| Self-Rationalization LLM Judge | RZZPnAaw6Z | 5.0 (Reject) | Similar iterative DPO self-training; SPA has better methodology and results |
| iREPO | NtAXAvIYuN | 3.4 (Withdrawn) | Also confounded baselines in iterative DPO; SPA is significantly stronger empirically |

SPA sits above the rejected iterative alignment papers (5.0 range) due to its stronger empirical results, ablation studies, and multi-model validation, but below the top self-alignment papers (7.0–8.0 range) due to the confounded baseline comparison and lack of label quality validation. The confounded comparison in Table 2 is a real methodological concern, but it is partially mitigated by the ablation showing the judgment method alone (14.41% LC) still outperforms PairRM (11.87%). The circularity concern is valid but inherent to all self-training methods; what matters is that it works and Figure 3 provides a mechanistic explanation.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>