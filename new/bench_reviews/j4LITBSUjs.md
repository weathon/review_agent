## Summary

This paper introduces *HalfScore*, a triplet-based language-graph metric for fine-grained dense-caption hallucination evaluation, and *PerturboLLaVA*, a training-time method that injects adversarial text perturbations during supervised fine-tuning to reduce a model’s reliance on language priors. The method is plug-in compatible with standard training pipelines and incurs no inference overhead. While the empirical idea is practical and shows some gains, the theoretical justification contains real errors, the primary metric is weakly validated, and the empirical claims are overstated relative to the evidence.

## Strengths

- **Simple, practical training intervention.** PerturboLLaVA integrates into standard SFT without altering model architecture or inference cost. Table 3 shows it improves general multimodal benchmarks (MMB +1.6, CCBench +1.2, SEED +0.3) while reducing hallucinations on the proposed metric, whereas some competing methods degrade general capabilities.
- **Orthogonality to decoding-based methods.** Table 3 demonstrates that combining PerturboLLaVA with OPERA yields additive gains (HalfScore F-score 52.8 vs. 52.2 alone, CHAIR 33.1 vs. 36.1 alone), suggesting the method occupies a complementary design space.
- **HalfScore addresses a real granularity gap.** By decomposing captions into object–attribute–relation triplets and measuring both precision and recall, the metric moves beyond object-level CHAIR, which is a sensible and needed direction for dense captioning.

## Weaknesses

### Fatal

None.

### Major

- **The Bayesian derivation in Section 4.2 is mathematically broken and undermines the claimed theoretical grounding.** Equation (9) applies Bayes’ theorem incorrectly: it substitutes $p(x_{<k}^{-p})$ for $p(x_{<k}^p \mid I)$ and $p(x_k)$ for $p(x_k \mid I)$ in the numerator/denominator. Equation (8) rests on an unjustified conditional independence assumption ($x_{<k}^p \perp x_{<k}^{-p} \mid x_k, I$) that the paper simply asserts without motivation. Because the abstract frames the contribution as identifying the “root cause” of hallucination and the text presents this section as a mathematical explanation of *why* the method works, these errors are structurally damaging rather than cosmetic.
- **HalfScore is poorly validated and produces anomalous rankings that the paper does not reconcile.** The human-correlation study (Table 6) involves only four models with no reported variance or confidence intervals, making the Pearson correlations (80.7, 78.1) unconvincing. More critically, Table 3 shows a severe rank inversion: RLAIF-V achieves a far better CHAIR score (18.1 vs. Ours 36.1) yet an almost identical HalfScore F-score (51.9 vs. 52.2). If HalfScore were measuring the same construct as established hallucination metrics, such compression of a massive CHAIR gap into a negligible F-score difference would not be expected. Basing the headline result on a new, self-designed metric without diagnosing this discrepancy is circular.
- **Empirical claims are overstated relative to the evidence.** The abstract claims the method “significantly improves” and “outperform[s] existing approaches.” Yet on HalfScore (Table 2/3) the margin over comparable LLaVA1.5-based baselines OPERA and VCD is marginal (+0.2–0.3 F-score), and on CHAIR the proposed method trails RLAIF-V by a large margin (36.1 vs. 18.1). The paper acknowledges the RLAIF-V comparison is “less fair,” but still presents it prominently without offering a fairer alternative. These discrepancies make the abstract’s sweeping claims unsupported.

### Minor

- **No mechanistic validation for the central causal claim.** The paper asserts that hallucinations stem from “over-reliance on language prior” and that PerturboLLaVA fixes this, yet provides no attention analysis, probing, or gradient attribution to show the model actually grounds more heavily in image tokens after training. Without such evidence, the language-prior mechanism remains speculative.
- **Random perturbations are not fully analyzed.** Table 5 shows random text improves HalfScore over the baseline (50.7 vs. 49.2), which the paper downplays. While CHAIR shows random text barely helps (52.4 vs. 54.2) whereas targeted perturbations help substantially (36.1)—arguing *against* a pure regularization story—the paper should still explicitly test whether the gain is driven by prefix-ignoring rather than adversarial language-prior simulation.

### Trivial

None.

## Nice-to-Have

- **Length-controlled evaluation.** Report whether precision gains stem from shorter, more conservative captions by including length-normalized metrics or controlling maximal output length across methods.
- **Mechanistic experiments.** Attention-rollout or saliency visualizations would validate the central claim that the model increases visual grounding after perturbative training.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *“Random perturbations suggest generic regularization or prefix-ignoring”* — This interpretation is selective. Table 5 shows random perturbations yield minimal CHAIR improvement (52.4 vs. 54.2 baseline) compared to targeted perturbations (36.1), indicating the targeted design matters for object hallucination. The HalfScore gap, while smaller, still favors targeted perturbations.
- *“The monotonic drop in recall and general benchmarks suggests trading verbosity for precision”* — The paper’s main reported method (Version1) actually improves general benchmarks (MMB +1.6, CCBench +1.2). The drops occur only under stronger ablations (Version2/3), which the paper explicitly discusses as a trade-off.
- *“Table 2 comparison with stronger models is meaningless”* — The table and caption clearly separate stronger pretrained models (first block) from LLaVA1.5-based methods (second block, marked with *), so the comparison structure is transparent.
- *“GPT-4o risks leaking answer information into perturbations”* — The paper explicitly states perturbations are generated “without disclosing the answer,” making this concern speculative.
- *“OPERA/VCD are disadvantaged by beam search”* — The paper reports hyperparameters in Appendix A.5; without evidence that beam search disproportionately harms these methods, this is speculative.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Correct or remove the erroneous derivation in Section 4.2.** If a rigorous justification cannot be provided, reframe the section as an intuitive motivation rather than a Bayesian formalization.
2. **Diagnose the HalfScore/CHAIR discrepancy.** Explicitly explain why RLAIF-V’s massive CHAIR advantage compresses to a negligible HalfScore difference—ideally with qualitative examples—to establish construct validity.
3. **Validate HalfScore more rigorously.** Expand the human study beyond four models, report variance or confidence intervals, and include GPT-4o parsing consistency checks.

## Score and Decision

**Calibration comparison:**
- *zGb4WgCW5i.md* (7.00, Accept Poster): TAME decoding with strong theoretical foundation and thorough experiments; our paper is well below this in theoretical soundness.
- *oZDJKTlOUe.md* (6.25, Accept Poster): LURE post-hoc algorithm with statistical analysis; more rigorous than our paper.
- *RcANissyP4.md* (5.67, Reject): SelfEval metric with limited scope but sound technical execution; our paper has broader scope but deeper technical flaws.
- *SrkDVzygXx.md* (5.00, Reject): PerPO training method with limited theoretical depth and evaluation concerns; comparable quality issues to ours.
- *Q4s7nFoowt.md* (5.00, Withdrawn): Theoretical doubts about proofs and limited novelty; our paper has more novelty but more serious derivation errors.
- *3YQYo1O01W.md* (3.67, Withdrawn): Simple prompting strategy with very limited contribution; our paper has more substance.

The paper under review has a practical training idea and a sensible metric concept, but the broken mathematical justification, weak validation of the primary metric, and overstated empirical claims place it below the acceptance threshold. It needs a major revision to fix the derivation, properly validate HalfScore, and tone down the claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>