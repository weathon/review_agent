Now I have a thorough understanding of the paper and the calibration anchors. Let me compose the final review.

**Summary of key verification findings:**

1. **Output averaging confound**: CONFIRMED. Thoughtbubbles averages softmax probabilities across forks (Eq. 11), while Copy baselines use only the "rightmost residual for decoding" (Section 3.3, line 182). This asymmetry means the perplexity improvement could partly be an ensemble effect rather than adaptive computation. However, the Copy baselines are also not standard FLOP-matched baselines - they just duplicate input residuals before processing, which is a weak baseline. The Copy models DO attend to each other during processing (they're just input duplicates), so they get some benefit, but they don't get the output averaging. This is a real confound, though it doesn't fully explain the results (the HellaSwag/LAMBADA improvements are also substantial).

2. **Missing pause-token baselines**: VALID concern. The paper motivates against pause tokens but doesn't compare against them experimentally.

3. **"319M outperforms 772M" framing**: PARTIALLY VALID. The paper does say "the latter of which is roughly FLOPs-matched against copy-5 baseline" in the caption, providing context. But the conclusion highlights "our method at a smaller 319M scale outperformed baselines at 772M scale" without mentioning the FLOPs disparity. This is somewhat misleading.

4. **Undertrained models**: VALID but acknowledged. 2.5B tokens is very few.

5. **LAMBADA 5.0 at 150M peS2o**: CONFIRMED (line 220). This is a concerning failure case. However, κ=4L rescues it to 10.3.

6. **Only 3 forking points**: CONFIRMED (line 168). Forking only at layers 3, 7, and 11. This is quite limited.

7. **BLiMP underperformance**: CONFIRMED. The paper acknowledges this (line 238): "for BLiMP (syntax understanding) our model only outperforms the parameter-matched, but not computation-matched baselines."

Now for scoring - comparing with calibration anchors:

- **High anchors (≥6)**: Input-adaptive compute (6.5), pause tokens (5.5), DynMoE (7.0), ReMoE (6.6), RL layer tying (6.75)
- **Low anchors (≤4)**: Stutter (4.5), BrainGPT (3.75), efficient transformer (3.0), confounded baselines papers (3.0-4.75)
- **Medium anchors (~5)**: Hyper-UT (5.0)

This paper has genuine novelty and interesting analysis, but the confounded comparison with Copy baselines is a significant concern. It's not as severe as the low-anchor papers (which had truly broken claims), but it's more concerning than the high-anchor papers where baselines were fair. The paper has real merit (novel architecture, interesting entropy analysis, honest about limitations) but the evidence doesn't cleanly support the core claim. I'd rank it slightly above the stutter paper (4.5) because Thoughtbubbles has a more principled mechanism and better analysis, but below the pause token paper (5.5) because that paper had more direct experimental comparisons despite similar scale issues.

I'll give this a 5.0 - borderline, leaning toward reject given the unresolved confound concern, but with genuine contributions that could be strengthened with better experiments.

## Summary

Thoughtbubbles introduces a transformer architecture that learns to dynamically fork and delete residual streams during pretraining, enabling adaptive parallel computation in latent space without auxiliary supervision. Forked streams attend to each other via score-attenuated attention, and final predictions are produced by averaging softmax outputs across all forks of a token, weighted by learned cumulative importance scores. Experiments at 150M–772M parameter scales on OpenWebText and peS2o show perplexity improvements over parameter-matched and computation-matched (Copy-3/Copy-5) baselines.

## Strengths

- **Novel and elegant mechanism**: The forking/deletion mechanism with cumulative score propagation is a clean, well-motivated architectural design for adaptive computation. The score modulation of attention and residual updates (Eqs. 8–9) naturally trains the model to assign higher importance to tokens that benefit from more computation, and the top-k budget constraint forces genuine allocation decisions — this is a principled approach to the problem.

- **Unsupervised learning from LM loss alone**: The entire forking/scoring system is trained with standard cross-entropy without any auxiliary loss, making adoption straightforward. This is a meaningful simplification over pause-token methods that require decisions about token placement.

- **Interesting entropy analysis (Figure 5)**: The concave relationship between token entropy and fork allocation — more computation at moderate uncertainty, less at extremes — is a genuinely interesting emergent behavior that aligns with intuitive expectations about where additional computation helps. This suggests the model learns a meaningful allocation policy.

- **Consistent perplexity improvements**: Table 1 shows perplexity improvements across all scales and datasets (e.g., 772M on OpenWebText: 19.74 vs. 21.22 baseline, 20.90 Copy-5), and downstream task improvements on LAMBADA and HellaSwag are meaningful.

- **Honest limitations discussion**: The paper identifies the top-k gradient bottleneck, autoregressive distribution shift, and wall-clock inefficiency, demonstrating intellectual honesty.

## Weaknesses

### Major

- **Confounded comparison with Copy baselines due to output averaging asymmetry**: Thoughtbubbles produces final predictions by averaging softmax distributions across all forks weighted by cumulative scores (Eq. 11), while the Copy-3/Copy-5 baselines decode only the "rightmost residual" (Section 3.3). Averaging multiple probability distributions is a well-known technique that lowers perplexity through an ensemble effect — it is a geometric mean of probabilities that systematically reduces perplexity compared to any single component. Without an ablation showing the perplexity of Thoughtbubbles using only the root stream's prediction (or, equivalently, the Copy baselines with output averaging), the perplexity gap between Thoughtbubbles and the Copy baselines cannot be cleanly attributed to adaptive computation. This is not just a "nice-to-have" ablation; it is essential for interpreting the main results, as it quantifies how much of the improvement comes from the architectural mechanism vs. from the output combination procedure. This concern is partially mitigated by the HellaSwag/LAMBADA improvements (which are discrete evaluation metrics where ensemble averaging doesn't trivially help), but the core perplexity claim remains confounded.

- **No comparison to compute-matched deeper/wider transformer baselines**: The Copy-3/Copy-5 baselines duplicate identical input residuals — a deliberately minimal baseline. The paper does not compare against a FLOP-matched deeper or wider standard transformer, which would be the most natural and informative compute-matched comparison. The Copy baselines test whether "any additional computation helps" but cannot distinguish whether *adaptive* allocation is better than simply having a bigger model. The comparison to pause-token or thinking-token approaches is also absent despite being the primary motivation. Without these comparisons, the paper cannot fully support its claim of superiority over existing adaptive computation approaches.

### Minor

- **The "319M outperforms 772M" framing is misleading about compute**: The conclusion states "our method at a smaller 319M scale outperformed baselines at 772M scale" (line 266), but the 319M κ=4L model processes up to 4× the input sequence length, incurring substantially more FLOPs per forward pass. While the table caption partially addresses this by noting κ=4L is "roughly FLOPs-matched against copy-5 baseline," the prominent claim in the conclusion omits this qualification. The comparison is by parameter count, not compute-equivalence. This should be clarified.

- **Limited training scale (2.5B tokens)**: At 150M–772M parameter scales, training on only 2.5B tokens is well below typical practice. All results are from early training dynamics, and whether advantages persist at convergence is unknown. The paper acknowledges hardware limitations, but the core claims rest on undertrained models. This is a significant limitation for confidence in the generality of results.

- **Catastrophic LAMBADA failure at 150M peS2o with κ=2L**: The κ=2L configuration at 150M on peS2o achieves a LAMBADA score of 5.0 (line 220), far worse than the baseline's 8.1. This suggests the method can sometimes degrade performance rather than improve it, and deserves more discussion than it receives.

- **Only 3 forking layers**: Forking occurs only prior to layers 3, 7, and 11 (line 168), meaning only 3 forking points in the entire model. This limits the degree of adaptive computation and raises questions about how well the approach would scale with more forking points (the appendix acknowledges this is limited by the top-k gradient bottleneck).

### Trivial

None substantive beyond what's already covered above.

## Nice-to-Haves

- Add an ablation measuring Thoughtbubbles' perplexity using only the root stream prediction (no output averaging), to disentangle the ensemble effect from adaptive computation. Adding output averaging to Copy baselines would also make the comparison fair.
- Compare against a wider/deeper standard transformer at matched FLOPs, and/or against pause-token baselines, to properly situate the method's contributions.
- Train one configuration to convergence (≥50B tokens for the smallest model) to validate that gains persist beyond early training.
- Provide qualitative examples of fork behavior on specific tokens to build intuition about what the model "thinks harder" about.

## Removed Points

- **"First-known architecture" overclaim**: The harsh critic argues this overstates novelty since pause tokens and Universal Transformers also learn latent computation. However, Thoughtbubbles is distinguished by being *input-adaptive* during *pretraining* with *no auxiliary supervision*, which is genuinely different from prior work. The claim has some merit, though it should be better qualified. Demoted from fatal/weakness to a nice-to-have improvement.

- **Missing comparison to Mixture-of-Depths / Universal Transformers**: While these would be informative baselines, the paper already provides two types of baselines (parameter-matched and compute-matched) and explicitly scopes its contribution to a new kind of adaptive computation. Demanding every possible baseline is a generic rather than paper-specific concern.

- **Autoregression distribution shift**: The paper documents this and shows dynamic budget scaling largely resolves it (21.18 vs. 20.97 blockwise). This is a recognized engineering concern, not a fundamental flaw.

- **Multiplicative score accumulation optimization concerns**: These are speculative without evidence of optimization difficulty — the paper uses log-space scores for stability and reports successful training. Minor implementation detail.

- **Forks may act as "auxiliary memory slots" rather than performing computation**: This is speculative and somewhat contradicted by Figure 4, which shows forks receive high attention from their parent tokens, suggesting they contribute meaningfully to computation. Without evidence beyond this, it's a design choice rather than a weakness.

## Novel Insights

The concave relationship between entropy and fork allocation (moderate-entropy tokens receive the most computation, highest-entropy tokens receive less) is an interesting and somewhat counterintuitive finding. It suggests the model learns that additional computation is most beneficial for tokens where it can meaningfully narrow down options, rather than for tokens that are nearly unpredictable regardless of compute investment. This aligns with information-theoretic intuitions about the value of computation at different uncertainty levels and deserves further investigation in the adaptive computation literature.

## Suggestions

- Run the critical ablation: report perplexity for Thoughtbubbles using only the root stream prediction (no averaging) to quantify the ensemble contribution. This single experiment would vastly strengthen (or clarify) the paper's claims.
- Add output averaging to the Copy baselines as a matching control, making the comparison like-for-like.
- Add a wider-standard-transformer baseline at matched FLOPs.
- Clarify the "319M outperforms 772M" claim by explicitly noting compute-FLOPs rather than just parameter count.
- Discuss the LAMBADA degradation at 150M/κ=2L and what conditions cause the method to fail.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| DynMoE (adaptive compute per token, adaptive experts) | 7.0 | Similar motivation (adaptive compute) but DynMoE has cleaner baselines and is more mature; Thoughtbubbles has comparable novelty but weaker experimental validation |
| Pause Tokens (pause/thinking tokens) | 5.5 | Directly comparable topic; pause tokens had simpler mechanism and controversial results; Thoughtbubbles is more principled but has the confounded baseline issue |
| Input-Adaptive Compute Allocation | 6.5 | Stronger methodology with fair baselines; Thoughtbubbles is less convincing due to the averaging confound |
| Stutter (adaptive layers per token) | 4.5 | Similar adaptive computation idea; Thoughtbubbles is better executed with more analysis but shares issues of limited baselines and small scale |
| ReMoE (differentiable MoE routing) | 6.6 | Clean architectural mechanism with fair comparisons at similar scale; Thoughtbubbles' mechanism is equally interesting but experimental validation is weaker |
| BrainGPT (confounded baselines, misleading claims) | 3.75 | Thoughtbubbles is much better than this; its core mechanism is real and the overclaim is more about interpretation than fabrication |
| Efficient transformer with reinforced position embedding (confounded baselines, toy scale) | 3.0 | Thoughtbubbles is substantially better — more principled, larger scale, better analysis |

The paper falls between the stutter paper (4.5) and the pause tokens paper (5.5). It has genuine architectural novelty and interesting analysis, but the confounded output averaging comparison, missing FLOP-matched baselines, and limited training scale keep it from a confident accept. The core concern — that we cannot distinguish how much of the perplexity improvement comes from adaptive computation vs. ensemble averaging — is addressable in principle but unaddressed in practice.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>