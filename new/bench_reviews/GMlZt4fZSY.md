Now I have a thorough understanding of the paper. Let me write the final consolidated review.

The key issues I need to evaluate:

1. **Confounded parameter comparison (950M vs 0.6B)**: This is a real concern but needs nuance. The paper compares 950M vs 0.6B for the "11.7% of tokens" claim. The 950M model has more parameters, which confounds the data efficiency claim somewhat. However, the paper also compares against OLMo-2-1.48B and SmolLM-1.7B (larger models), which MobileLLM-R1-950M also outperforms. So the comparison with Qwen3-0.6B is specifically about token efficiency relative to a model trained on 36T tokens, and the model size difference (950M vs 600M) does confound attributing the performance difference to data curation alone. This is a valid concern.

2. **Influence-based mixing vs heuristic baselines**: The paper only compares against uniform sampling for data mixing. A comparison against a simple heuristic (e.g., upweighting math/code) would strengthen the claim significantly. This is a valid methodological concern.

3. **Convergence claim from only 2 stages**: The paper says "we find that two stages suffice" (line 146) and shows Figure 5 with stages 1 and 2. The convergence claim is somewhat weak, but the paper does say "in practice, we find that two stages suffice" rather than claiming rigorous convergence. This is a moderate concern.

4. **"Benchmark-free" claim**: The capability-probing datasets serve essentially the same role as held-out benchmarks for data weighting. The paper is technically correct that specific test benchmarks aren't used, but the probing datasets are constructed to directly proxy those capabilities. This is a fair criticism but somewhat semantic.

5. **Table 2 SFT starting point asymmetry**: Baseline models use their instruct checkpoints while MobileLLM-R1 uses Tulu3-SFT checkpoints. This isn't truly "identical starting conditions." However, the paper's stated goal is to show that better pre-training leads to better downstream reasoning after the same SFT, so starting from different base models is the point. The Tulu3-SFT vs instruct difference is a minor concern.

Now for the calibration:

- **High scoring anchors (>=6)**: RegMix (7.2, data mixture selection) and uHLgDEgiS5 (8.0, data influence for training) are methodologically strong papers with comprehensive ablations. MobileLLM-R1 is not as methodologically rigorous (limited baselines for data mixing, confounded comparisons) but has strong empirical results and a fully open recipe.

- **Medium anchors (~5-6)**: Textbooks Are All You Need (phi-1) scored avg 6 with similar issues—impressive results from small models but with confounded comparisons and data contamination concerns. phi-1's concerns were more severe (potential data contamination).

- **Low anchors (<=4)**: The confounded comparison papers scored 3-4.25.

MobileLLM-R1 is somewhere between phi-1 (avg 6, similar "small model achieves great results with curated data" story) and RegMix (7.2, data curation methodology). The methodological contribution (influence-based data mixing) is interesting but not rigorously validated against simple alternatives. The empirical contribution (strong small reasoning model with open recipe) is solid. The data efficiency claim is partially confounded. I'd place this at around 6.

Let me write the review now.</think>## Summary

MobileLLM-R1 introduces a data-centric framework for training sub-billion-parameter reasoning models, combining (1) a leave-one-out analysis to identify cross-domain data transfer effects, (2) a benchmark-free influence-based data mixing method for pre-training, and (3) an iterative data–model co-evolution strategy for mid-training. The resulting 950M model, trained on only 4.2T tokens, outperforms OLMo-2-1.48B and SmolLM-2-1.7B on reasoning benchmarks and achieves comparable results to Qwen3-0.6B despite using 11.7% of its training data. The entire training recipe, data, and code are open-sourced.

## Strengths

- **Comprehensive and actionable training recipe**: The paper provides a fully open-sourced pipeline—data sources, mixing ratios, training configurations, model checkpoints, and code—enabling genuine reproducibility, which is rare for reasoning models of this quality.

- **LOO analysis yielding genuine cross-domain insights**: The finding that StarCoder benefits math more than OpenWebMath benefits code challenges conventional wisdom (Lewkowycz et al., 2022) and provides concrete, transferable insight for data curation beyond this specific work.

- **Well-structured post-training ablations (Table 1)**: The ablations cleanly reveal practical findings—staged SFT dramatically outperforms joint training (68.5 vs. 53.1 on GSM8K), and the reasoning–knowledge tradeoff is honestly documented. These are useful for practitioners.

- **Consistent improvements across model scales**: Table 2 demonstrates gains at 140M, 360M, and 950M over respective SmolLM baselines, showing the recipe scales and is not a single-point result.

- **Strong empirical results against fully open baselines**: MobileLLM-R1-950M achieves 57.8 MATH vs. OLMo-2-1.48B's 53.0 and SmolLM-2-1.7B's 41.4, clearly outperforming models with more parameters trained on fully open data.

## Weaknesses

### Major

- **Confounded data-efficiency claim for the Qwen3-0.6B comparison**: The headline claim that MobileLLM-R1-950M matches Qwen3-0.6B using "only 11.7% of tokens" conflates data quality advantages with model size advantages. MobileLLM-R1-950M has 950M parameters versus Qwen3-0.6B's 600M—roughly 58% more parameters. A larger model trained on less data outperforming a smaller model does not uniquely demonstrate data efficiency; parameter scaling could account for a substantial portion of the gap. The paper does not provide a same-architecture, same-parameter-count comparison trained on more data, leaving the attribution to data curation vs. model size ambiguous. The comparisons against OLMo-2-1.48B and SmolLM-2-1.7B (where MobileLLM-R1 wins *despite* having fewer parameters) are more informative but address model quality rather than data efficiency per-token. This doesn't invalidate the results but weakens the core "11.7% of tokens" narrative.

- **Influence-based data mixing lacks comparison to simple heuristic baselines**: The proposed influence-based mixing (Section 2.2) is only compared against uniform sampling (Figure 4). The LOO analysis already reveals that FineWeb-Edu, StarCoder, and math corpora are the most impactful—so a straightforward heuristic (e.g., upweighting math and code by 2–3×) would be a natural baseline. Without this comparison, we cannot assess whether the expensive influence computation (requiring 3 separate domain-specialized models trained across 10 checkpoints each) provides meaningful gains over what the LOO analysis already reveals. This is important because the complexity-cost-to-benefit ratio is the core question for any practical data curation method.

### Minor

- **"Benchmark-free" framing is technically correct but potentially misleading**: The capability-probing datasets are explicitly constructed to proxy code, math, and knowledge performance (Section 2.1.1). While no held-out test benchmarks are accessed, these probing sets serve an identical role in guiding data weighting. The paper acknowledges this implicitly ("each capability-probing dataset can be viewed as defining a token distribution that implicitly induces the necessary preconditions for reasoning"), but the "benchmark-free" label in the abstract overstates the distinction.

- **Convergence of mid-training compression shown with only two stages**: The paper claims the data–model co-evolution "converges as most samples reach zero or negative influence" (Section 3), but only two compression stages are shown (Figure 5). The narrowing of influence distributions between stages 1 and 2 is expected from training on already-filtered data. The claim that "two stages suffice" is an empirical observation, not evidence of convergence, since no stages 3+ are reported. The paper could strengthen this by briefly noting what happens at stage 3 or discussing why further iterations were terminated.

- **Table 2 starting-point asymmetry**: In the comparison under "identical reasoning SFT," baseline models start from their instruct checkpoints while MobileLLM-R1 starts from a Tulu3-SFT checkpoint. While the paper's intent is to isolate the effect of pre-training quality, the different SFT starting points (general instruct vs. Tulu3-specific) are not fully controlled, which could contribute to the observed gap.

### Trivial

- None significant.

## Nice-to-Haves

- Ablation comparing influence-based mixing against a simple LOO-derived heuristic (e.g., upweight top-3 datasets by 2×) would clarify the value added by the computationally expensive influence framework.

- A same-architecture, same-parameter-count experiment varying token count (e.g., 950M model trained on 2T vs. 4.2T vs. 8T+ tokens) would directly test the data efficiency hypothesis.

- Reporting per-benchmark breakdown for Figure 4 would clarify whether gains are balanced across domains or concentrated in one area.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Qwen3-0.6B comparison is structurally misleading" (Harsh Critic #1, full version)**: While the parameter confound is real (kept as Major), the original claim that the comparison is "structurally misleading" overstates the issue. The paper also compares against larger models (OLMo-2-1.48B, SmolLM-2-1.7B) where MobileLLM-R1 wins with fewer parameters, and the token-efficiency framing is directionally informative—it just can't cleanly isolate data curation from model size effects. The comparison is not invalid, just not as conclusive as claimed.

- **"Straw man setup of two assumptions" (Harsh Critic, intro comments)**: The paper explicitly acknowledges the first assumption has been "already challenged" by Qwen3-0.6B and DeepSeek distillations. This is not a straw man but a framing device; the novelty claim is about the data efficiency angle, which the paper legitimately pursues.

- **Missing appendix / missing proofs**: The parser strips appendix content; these sections exist in the original submission.

- **"The unexplained performance dip around 30K steps" (Harsh Critic)**: This is a minor observation, not a weakness—the paper shows the subsampled data *avoids* the dip, which is precisely the point of the method. The dip itself may indicate training instability on uncurated data, which the method mitigates.

- **"Post-training comparison uses different starting checkpoints" (partially kept)**: The asymmetry in starting checkpoints (instruct vs. Tulu3-SFT) is noted as Minor rather than Major because the paper's explicit goal is to show that stronger pre-training leads to better reasoning after equivalent SFT—from different base models is the natural design when comparing multi-stage pipelines. The SFT starting point (instruct vs. Tulu3) remains a minor confound.

- **Generic strengths about importance of the problem space**: Removed claims like "reasoning in small models is an important problem" as they are generic rather than evidence-grounded.

- **Strength about "comprehensive model series at 140M, 360M, and 950M allowing scaling trend analysis"**: This is a reasonable strength but is already partially captured under the "consistent improvements across model scales" point above.

- **Criticism about Table 2 post-training SFT asymmetry being a fundamental confound**: Downgraded from Major to Minor because the comparison's purpose is precisely to show cumulative pre-training effects, not to claim a perfectly controlled experiment.

## Novel Insights

The cross-domain LOO finding—specifically that code data (StarCoder) transfers to math more than math data (OpenWebMath) transfers to code—directly contradicts the common assumption derived from Lewkowycz et al. (2022) and could reshape how practitioners allocate training data budgets for multi-capability models. The staged SFT finding (alignment-first, then reasoning) is practically important and worth noting for the community working on small reasoning models.

## Suggestions

- Add explicit acknowledgment of the parameter confound in the Qwen3-0.6B comparison and provide compute-optimal scaling expectations (e.g., from scaling laws) to bound how much of the performance gap is attributable to parameters vs. data.

- Compare influence-based mixing against at least one LOO-derived heuristic (e.g., weighting proportional to LOO impact) to quantify the marginal value of the influence computation over simpler alternatives.

- Report one additional mid-training compression stage or provide a short discussion of why stage 3 was not pursued, to strengthen the convergence claim.

## Evaluation

**Originality**: The LOO cross-domain analysis is genuinely insightful. The influence-based mixing extends AutoMixer to multi-capability settings with domain-specialized models, which is a reasonable extension. The mid-training co-evolution is an interesting direction though not radically novel (iterative importance sampling is a known technique). Moderate originality overall.

**Importance**: Training strong reasoning models with fully open recipes is clearly valuable for the community. The practical findings (staged SFT, data cross-domain transfers) are impactful.

**Claims support**: The core data-efficiency claim is partially confounded. The influence-mixing contribution is under-validated against simple baselines. The mid-training convergence claim is weakly supported. The strong open-source baselines comparison is well-supported.

**Experiment soundness**: Experiments are reasonably comprehensive for the fully open-source baselines but limited for validating the key methodological contribution. The same SFT comparison (Table 2) is useful but has a minor starting-point asymmetry.

**Clarity**: The paper is generally well-written and the pipeline is clearly visualized (Figure 2). Some notation-heavy sections (Eqs. 2-5) could benefit from more intuition.

**Community value**: The fully open recipe and genuine cross-domain data insights make this a contribution that will be built upon.

## Score and Decision

**Calibration anchors used:**

- **uHLgDEgiS5** (Trajectory-specific LOO influence for data, avg 8.0): MobileLLM-R1 has a less rigorous theoretical foundation and fewer methodological baselines for its influence-based mixing, but offers a more complete end-to-end system paper with open recipe and empirical results.

- **5BjQOUXq7i** (RegMix, avg 7.2, Spotlight): RegMix provides a cleaner ablation story (512 small models trained for data mixture search, compared to many baselines). MobileLLM-R1 has a more complete open-source recipe and targets reasoning specifically, but has weaker ablations for its data mixing method.

- **Fq8tKtjACC** (phi-1 / "Textbooks Are All You Need", avg 6.0, Reject): Similar paper pattern—small model trained on curated data achieves impressive results. phi-1 was rejected partly due to confounded comparisons and data contamination concerns. MobileLLM-R1 has a more principled methodology and full open-sourcing, but also has the parameter confound issue in its headline claim.

- **0mJZplhexS** (Little-Big speedup, avg 4.25, Reject): Flags unfair comparisons where newer methods benefit from better architectures/training recipes. The confounded comparison issue is similar but less severe in MobileLLM-R1 since it also demonstrates wins against larger-parameter open models.

- **3OyaXFQuDl** (Compute-optimal data for LLM reasoners, avg 7.0, Accept Poster): Addresses data efficiency for reasoning models. MobileLLM-R1 has a more complete training recipe but less rigorous methodology validation.

MobileLLM-R1 falls between phi-1 (6.0) and RegMix (7.2). It has stronger empirical results and a more complete open-source contribution than phi-1, better baselines against same-family open models, and genuine methodological insights (LOO cross-domain transfer). But it shares phi-1's weakness of a partially confounded headline claim and lacks baseline comparisons for its primary methodological contribution (influence-based mixing vs. simple heuristics). The open recipe and cross-domain transfer insights are solid contributions that merit acceptance despite the methodological gaps.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>