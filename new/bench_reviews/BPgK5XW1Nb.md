Now I have a thorough understanding of the paper and calibration anchors. Let me finalize the review.

## Summary

The paper proposes Spread Preference Annotation (SPA), an iterative self-alignment framework for LLMs that uses only a small amount of human-labeled preference data. SPA's core idea is to use the training LLM's own DPO implicit reward (the log-ratio between the current policy and an SFT reference) to generate preference labels for self-produced response pairs, rather than relying on external reward models or LLM-as-judge. The framework iterates between data expansion (generating responses and labeling them via Eq. 7-8) and preference learning with a self-refinement mechanism (confidence-based label smoothing + de-coupled noise detection via logit extrapolation). Using only 3.3% of UltraFeedback labels, SPA achieves 15.39% LC win rate and 21.13% original win rate on AlpacaEval 2.0, outperforming DPO on the same data and surpassing the Zephyr-7b-β model trained on 100% of the data.

## Strengths

- **Direct preference judgment from DPO implicit reward is a genuine and effective insight**: Using the model's own DPO log-ratio (Eq. 7) for preference labeling is computationally free (requires no external model) and empirically outperforms both PairRM and LLM-as-judge baselines. Table 2 shows 15.39% LC win rate vs. 11.87% (PairRM) and 9.28% (LLM-as-judge). Figure 3 shows the gap widens across iterations, consistent with the paper's distribution-shift explanation — the internal reward model co-adapts with the policy while external reward models become increasingly mismatched.

- **Strong and consistent improvements over fair baselines**: SPA dramatically outperforms DPO trained on the same 3.3% data (15.39% vs. 9.03% LC win rate, Table 1), and the improvement is consistent across seed data sizes (Table 3) and model families including Phi-2, LLaMA-3, and Phi-3 (Table 5).

- **No-seed-data existence proof is valuable**: Figure 4 shows SPA can improve alignment even without seed preference data when starting from an instruction-tuned model, demonstrating that the framework can leverage implicit preferences acquired during pretraining/SFT.

- **Ablations are thorough**: Table 6 cleanly disentangles data expansion, self-refinement, and DND contributions. Table 7 provides useful analysis of reference model choice for judgment.

## Weaknesses

### Fatal
None.

### Major

- **The headline claim of "outperforming cases using the entire data" rests on a confounded comparison**: The abstract states SPA with 3.3% of labels achieves "superior alignment performance...compared to the cases using the entire data," citing the Zephyr-7b-β comparison in Table 1. However, Zephyr differs from SPA in training hyperparameters, number of epochs, and potentially other recipe details — it is not a controlled comparison isolating data quantity. A fair baseline would be DPO trained on 100% of UltraFeedback with the same π_init, β, and training recipe. On the reliable LC metric, SPA's advantage over Zephyr is modest (15.39% vs. 11.75%), and could partially reflect recipe differences rather than data efficiency. This matters because the abstract's framing centers on this comparison.

- **Verbosity amplification concern is real and unaddressed**: SPA shows a large 5.74 percentage-point gap between its original win rate (21.13%) and LC win rate (15.39%), while the DPO baseline actually has the opposite pattern (7.68% original vs. 9.03% LC). This strongly suggests SPA amplifies response length — a known artifact that GPT-4 judges reward. The paper provides no analysis of what behavioral traits the iterative process amplifies (verbosity, formatting, content types), and no length analysis is reported across iterations. Since the method's core mechanism is a self-reinforcement loop with no external grounding beyond the seed data, this is a structural concern, not just a presentation issue.

- **Iteration 3 degradation is unexplained and undermines scalability claims**: Figure 3 clearly shows SPA's LC win rate drops from ~16% at iteration 2 to ~15% at iteration 3, yet this degradation is never discussed in the text. For a method whose premise is iterative self-improvement, degradation at iteration 3 raises fundamental questions about the loop's stability and practical limits. The paper should diagnose whether this is due to distribution shift, accumulated noise, or other factors.

### Minor

- **De-coupled noise detection (DND) lacks empirical validation of its approximation**: The logit extrapolation (Eq. 12) claims to approximate a model trained with (1+λ)× smaller KL penalty (footnote 4), based on Liu et al. 2024's observation that aligned models with varying β are geometric mixtures. This is a specific and testable claim, but the paper provides no verification — e.g., comparing p_g predictions against an actual model trained with β/(1+λ). The DND contribution is numerically meaningful (+0.69 LC points over SR-only per Table 6), but without validating the approximation, it's unclear whether this improvement reflects the claimed mechanism or is an artifact of the specific λ schedule.

- **Self-refinement (SR) contribution is marginal**: Table 6 shows SR alone adds only +0.29 LC win rate points (14.41% → 14.70%) over data expansion alone, while DND provides the bulk of the improvement. The λ decay schedule (1/2, 1/4, 1/8) across iterations lacks justification or sensitivity analysis.

- **LC win rate variance is moderately high**: Table 4 shows SPA's LC win rate variance is 2.10 across seed data samples, yielding a 95% CI of roughly [13.36, 17.56]. While the lower bound still exceeds baselines, this variance is notably higher than DPO's (0.16) and suggests some sensitivity to the seed data.

## Nice-to-Haves

- Response length analysis across iterations to directly assess verbosity amplification, potentially with a length-regularized variant of the implicit reward.
- DPO on 100% of UltraFeedback with identical training recipe as SPA, to provide a fair controlled comparison for the headline claim.
- Diagnosis and discussion of iteration 3 degradation.
- Human evaluation to confirm that SPA's improvements reflect genuine preference quality rather than GPT-4 judge artifacts.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Table 7 is inconsistent/confusing"** (Harsh Critic): The critic claims Table 7's rows are inconsistent, but the rows represent different experimental setups — "SPA after iteration 1" is the model state after iteration 1, while "Eq. 7 with initial SFT model" tests an alternative judgment method for generating data at iteration 2. These are inherently different and expected to produce different numbers. This is a misreading, not a paper flaw.

- **"Overclaimed no-seed-data experiment"** (Harsh Critic): The critic says the no-seed-data result (Figure 4) is "overblown" because gains are modest and use a different base model. However, the paper presents this as an existence proof/feasibility demonstration, not as a primary result. The modest gains and different regime are expected and honestly presented; the claim is simply that the framework can function without seed data, which the experiment supports.

- **"Cannot discover qualitatively new human-preferred behaviors"** (Harsh Critic): This is a theoretical observation about the method's design that applies equally to any self-training approach. The paper never claims SPA discovers completely novel behaviors; it claims to spread existing preference knowledge. This is a scope-creep criticism.

- **"Missing human evaluation"** (Harsh Critic): While human evaluation would strengthen the paper, relying on AlpacaEval 2.0 LC win rate (the standard metric specifically designed to mitigate length bias) is the community norm for this type of work. This is a nice-to-have, not a required baseline.

- **"Reproducibility / undisclosed hyperparameters"**: The paper specifies β=0.1, α=0.1, K=10, λ schedule, learning rate, batch size, optimizer, scheduler, temperature, and warm-up phase. Implementation details are adequately disclosed for reproducibility.

## Novel Insights

The distribution-shift argument for why internal DPO implicit reward outperforms fixed external reward models across iterations is the most novel insight: as the policy distributes shifts through self-training, an external reward model becomes increasingly mismatched while the internal reward co-adapts. This explains the widening gap in Figure 3 and provides a principled reason to prefer intrinsic over extrinsic preference judgment in iterative alignment — though it also highlights the tension that this same co-adaptation can amplify model biases.

## Suggestions

- Add response length statistics for SPA vs. baselines across iterations to directly assess the verbosity amplification hypothesis, and consider integrating a length penalty into the implicit reward.
- Train a DPO baseline on 100% of UltraFeedback with the same recipe as SPA to provide a fair controlled comparison.
- Discuss and diagnose the iteration 3 degradation shown in Figure 3; consider whether early stopping (2 iterations) should be the recommended configuration.
- Validate the DND logit extrapolation by comparing p_g predictions against an actual model trained with β/(1+λ), even for a single λ value.

## Calibration

**Anchors retrieved:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Self-Alignment with Instruction Backtranslation (Oral) | /home/wg25r/review_agent/human_reviews/1oijHJBRsT.md | 8.0 | More mature iterative self-alignment work with thorough scaling analysis and self-curation; SPA is less thorough but tackles a different and complementary problem (preference learning rather than instruction tuning) |
| SeRA (Poster) | /home/wg25r/review_agent/human_reviews/uIGnuyDSB9.md | 6.0 | Closest topical match — also uses implicit reward margins for iterative DPO; SPA is comparable in novelty but has the overclaim/verbosity concern |
| Meta-Rewarding (Reject) | /home/wg25r/review_agent/human_reviews/lbj0i29Z92.md | 5.0 | Similar iterative self-improvement concerns (self-reinforcement, saturation); SPA has fairer baselines than Meta-Rewarding but shares verbosity/saturation issues |
| SPO/Soft Alignment (Withdrawn/Reject, plagiarism) | /home/wg25r/review_agent/human_reviews/28TLorTMnP.md | 2.5 | Much weaker — plagiarism; not a fair comparison |
| FreeLM (Reject) | /home/wg25r/review_agent/human_reviews/qgLyKwXVDs.md | 2.0 | Grossly overclaimed results with confounded comparisons; SPA's issues are less severe |
| Self-rationalization iterative DPO (Reject) | /home/wg25r/review_agent/human_reviews/RZZPnAaw6Z.md | 5.0 | Similar concerns about self-reinforcement loops and judge calibration drift |
| WSPO (Spotlight) | /home/wg25r/review_agent/human_reviews/f7KxfUrRSb.md | 7.25 | Stronger theoretical grounding for weak-to-strong preference transfer; SPA has less theory but more practical demonstration |

SPA falls between SeRA (6.0, poster) and meta-rewarding (5.0, reject). It has real methodological contributions (the implicit reward judgment insight is genuine and empirically validated), but the confounded headline comparison and unaddressed verbosity concern are significant. Compared to SeRA (which received 6.0), SPA has somewhat larger empirical improvements but also more serious issues (verbosity amplification, confounded full-data comparison, unexplained iteration degradation). Compared to Meta-Rewarding (5.0, reject), SPA has fairer experimental comparisons (against same-data DPO and iterative DPO baselines) but shares the self-reinforcement saturation problem.

**Originality**: The direct preference judgment idea is novel and well-motivated. **Importance**: Data-efficient alignment is an important problem. **Claims support**: Core claims are supported by fair baselines; the headline "outperforms full-data" claim is confounded. **Experimental soundness**: Strong on primary comparisons, weak on the Zephyr comparison and verbosity analysis. **Clarity**: Well-written and clearly structured. **Community value**: The implicit reward judgment insight is useful even if the full framework needs refinement.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>