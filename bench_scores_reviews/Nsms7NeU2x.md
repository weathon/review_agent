## Summary
This paper studies whether exact benchmark data contamination can be "forgotten" during large-scale LLM pretraining. Through controlled training experiments (models up to 1.6B parameters, plus an OLMo-1B checkpoint intervention), the authors show that contamination effects scale predictably with model size, training tokens, and repetition count, and that continued training on novel data can completely erase measurable contamination effects. They derive a closed-form upper bound on forgetting via cumulative AdamW weight decay that applies to any run with known hyperparameters, and show that empirical forgetting is generally faster than this bound.

---

## Strengths

- **Ground-truth controlled contamination design**: Unlike virtually all prior contamination work, which must infer contamination effects from overlap detection or model comparisons, this paper inserts known contamination levels into controlled pretraining runs and directly measures causal effects. This provides a quality of causal evidence unavailable in observational approaches and is a genuine methodological advance.

- **Three-dimensional scaling study with a counterintuitive finding**: The systematic exploration of model size × training tokens × repetition count (Figure 2) is novel in the contamination literature. The key result — that even 12× repeated contamination becomes undetectable at ≥15× Chinchilla token counts for a 124M model — directly challenges the binary "contaminated = invalid" framing and provides quantitative structure to the question.

- **Mechanistic weight-decay theory of forgetting**: Section 5 derives a closed-form expression (Proposition 1) linking AdamW hyperparameters to a measurable upper bound on forgetting, applicable to any run with published optimizer configs. This is a new lens not previously offered in contamination or forgetting literature, and the qualitative alignment with empirical weight-decay sweeps (Figure 6a/b) adds credibility.

- **Novel data as necessary for forgetting (Figure 3d)**: The clean experiment comparing forgetting on a continuous stream of novel data vs. repeatedly cycling through the same 100M tokens directly demonstrates that novelty of subsequent training data — not just step count — drives forgetting. This mechanistic isolation is a meaningful contribution.

- **Near-duplicate filtering as a methodological contribution**: Identifying that *all* seven tested benchmarks contain near-duplicates and cross-benchmark duplicates (Figure 1), and filtering them prior to contamination insertion, resolves a confound that could invalidate the causal identification in prior work.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of single-exposure (1×) contamination measurements**: The most realistic contamination scenario — a benchmark question appearing once in training — is explicitly excluded because its effect falls "within the confidence intervals" (footnote 1). The paper then claims in Section 4.1 that "under Chinchilla training, a single time of contamination can lead to overfitting of as much as 3 percentage points" by linearly extrapolating from the 4× results of a 774M model. This extrapolation is unsupported: Figure 2 shows strongly nonlinear behavior across repetitions, and Figure 2c shows non-monotone joint behavior. The core practical claim of the paper rests on a regime that is not directly measured. Either direct 1× measurements with larger evaluation sets, or explicit caveats labeling this as an upper-bound extrapolation, are required.

- **Scale extrapolation from ≤1.6B to frontier-scale models**: Claims that "OLMo-7B has forgotten data from the first 40% of training" and that "Llama 3 405B has experienced significant decay of early gradients" (Section 5.1, Figure 5) rest entirely on assuming the weight-decay mechanism is dominant and on assumed hyperparameters (weight decay = 0.1 for Llama 3 explicitly). The paper itself demonstrates in Figure 6d that the theoretical bound can be off substantially relative to empirical forgetting, making the bound's direction informative but not the magnitude. OLMo-7B checkpoints are publicly available; at least one experiment at 7B scale would directly validate whether the ≤1.6B trends extend to that regime and would substantially strengthen the paper's reach.

- **"Completely forgotten" is stronger than the evidence**: Section 4.2 defines "completely forgotten" as "no longer any accuracy difference between contamination and holdout benchmark questions" (within bootstrapped confidence intervals). This is a detection limit on accuracy, not a demonstration of zero influence. CE-loss differences can persist below the decision threshold; rank-order effects or calibration shifts may remain; different evaluation formats could reveal residual effects. The paper should consistently use language such as "undetectable in our accuracy-based evaluation" rather than "completely forgotten," which implies full behavioral equivalence.

### Minor

- **Contamination timing in the forgetting experiment is a specific, potentially favorable choice**: The forgetting experiments in Section 4.2 contaminate between the 1st and 2nd Chinchilla, noting in footnote 3 that "contamination is not very early during training." But Section 4.1's scaling study uses uniform contamination throughout training. The bridge between these two setups is not articulated: does the forgetting rate measured in Section 4.2 apply to contamination events early in training (which may be even more readily forgotten) or to the distributed contamination of Section 4.1? This connection should be made explicit.

- **Per-benchmark breakdown is absent from scaling experiments**: The seven benchmarks (HellaSwag, MMLU, BoolQ, etc.) differ dramatically in format, reasoning depth, and difficulty. The aggregated accuracy gap may mask heterogeneous contamination and forgetting dynamics. The OLMo-1B section provides some per-benchmark resolution, but the main scaling figures aggregate everything. Appendix per-benchmark curves would substantially strengthen the claim that results generalize across benchmark types.

- **OLMo-1B forgetting covers only 1% of remaining training; the rate's persistence is unknown**: Section 4.3 shows a 96% reduction in accuracy gain over <2,000 steps out of 739,328 total. This is compelling evidence for rapid initial forgetting, but rapid initial forgetting does not preclude a long-tail plateau. Whether this rate continues, slows, or stalls at a non-zero residual through the remaining 99% of training is not shown, and the answer matters for the practical interpretation.

- **The "middle-peak" finding is interesting but underexplained and requires a dedicated experiment**: Figures 3e/f show that uniformly distributed contamination across training causes stronger overfitting than end-loaded contamination. This is an unexpected and potentially important insight about spacing effects in learning. However, the result is obtained by averaging across all contamination levels to gain power, which conflates repetition count with temporal spacing. A targeted experiment varying spacing while holding repetition count fixed is needed before drawing strong conclusions about the spacing effect.

### Tiny

- The 4×-to-1× extrapolation in Section 4.1 (producing the "3 percentage point" estimate) should be explicitly labeled as a rough upper-bound estimate given the nonlinearity visible in Figure 2.

- Reported uncertainty comes from bootstrap resampling of questions, not from training stochasticity. Even a single alternative seed at one key setting would help validate that the reported curves are not sensitive to optimization randomness.

---

## Nice-to-Haves

- **Validation at 7B scale**: OLMo-7B checkpoints and training data are publicly available, making an intermediate-scale forgetting experiment feasible. This would directly test whether the mechanisms observed at ≤1.6B generalize before extrapolating to 405B.
- **Semantic/paraphrastic contamination**: Testing whether paraphrased benchmark questions are forgotten at similar rates would substantially expand practical relevance, since real-world contamination rarely involves exact string overlap.
- **Post-training phase analysis**: Whether contamination "forgotten" during pretraining resurfaces or becomes reinforced during SFT/RLHF would be a natural extension with real practical import.
- **Complementary LR sweep in Section 5.2**: Since Proposition 1 depends on LR × weight-decay jointly, a learning-rate sweep analogous to the weight-decay sweep in Figure 6a/b would complete the empirical validation of the theory.
- **Continuous loss curves over the full training run**: Plotting CE-loss difference continuously rather than at Chinchilla milestones would reveal whether forgetting is monotonic or whether there are non-monotone dynamics that the milestone snapshots miss.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Insufficient engagement with evaluation methodology literature" (Related Work)**: This is scope creep — the paper is about contamination mechanics and forgetting, not benchmark evaluation methodology. The cited set is appropriate for the actual scientific question addressed.
- **"Monotone in each dimension" is only partially established because Figure 2c is non-monotone**: The paper explicitly says "there is no clear monotone pattern" for the joint-scaling experiment (Section 4.1), so this is not a misrepresentation — the critic misread the paper's own statement.
- **Criticism that the theory equates vanishing direct contribution with "forgetting"**: The paper is explicit in Section 5.1 that it models only one mechanism and that gradient cancellation/alignment could cause additional or different forgetting. The limitation is clearly stated.
- **"It does not make sense to attribute model behavior to individual datapoints in this regime" is overstated**: The paper bounds this claim to the specific regimes studied (large-scale training where individual examples have negligible measurable effect). This is not a general overclaim.
- **Missing reproducibility specifics (seeds, compute budget)**: The paper cites public codebases, public data, and an anonymized repository. Requests for exhaustive compute accounting are not a standard expectation for ICLR empirical papers of this type.
- **"Unfair comparison to single-epoch training forgetting literature"**: The paper explicitly notes the experimental difference (novel data stream vs. repeated epochs) and reproduces the stabilization behavior found in prior work (Figure 3d). There is no unfair comparison here.

---

## Novel Insights

The most genuinely novel insight — beyond the paper's own stated contributions — is the implication that **the optimizer hyperparameters of a training run encode a precise, computable "forgetting schedule"** that can be audited post hoc from published technical reports. This reframes contamination assessment from a data-inspection problem into an optimization-dynamics problem: given only a model's AdamW configuration and training length (quantities often published), one can establish a conservative upper bound on how much any early-inserted data persists in the final parameters. This connects contamination analysis to the broader literature on data attribution and privacy, and the observation that Llama 3's hyperparameters yield a near-symmetric contribution from all training deciles (Figure 5c, bottom) is a concrete, testable side-consequence of this framework. Additionally, the "spacing effect" finding — that uniformly distributed repetitions produce stronger overfitting than end-loaded repetitions — echoes cognitive science research on spaced repetition and, if confirmed by a dedicated experiment, would be a notable connection between neural network training dynamics and memory consolidation theory.

---

## Suggestions

1. **Directly measure 1× contamination** using larger evaluation subsets (e.g., 20,000–30,000 questions) or multiple seeds to achieve sufficient statistical power. This is the most impactful missing experiment and directly supports the paper's main practical claim.
2. **Add a 7B-scale forgetting experiment** using an OLMo-7B intermediate checkpoint (training data and checkpoints are public) to validate that the 1.6B trends extend before extrapolating theoretically to 405B.
3. **Replace "completely forgotten" with "undetectable in our evaluation"** throughout, and add a supplementary CE-loss analysis at the point where accuracy gaps vanish to bound any residual influence.
4. **Design a targeted spacing experiment**: Fix repetition count (e.g., 12×) and vary temporal distribution (early-clustered, uniform, late-clustered) to cleanly isolate the spacing effect identified in Figures 3e/f.
5. **Report forgetting curves beyond the initial 1% of OLMo-1B remaining training** — even showing that the CE-loss gap reaches the holdout baseline at some later checkpoint would substantially strengthen the claim of complete forgetting at realistic scale.
6. **Label the bottom row of Figure 5 explicitly as a heuristic approximation**, not a causal attribution, to prevent over-interpretation of the weight-contribution visualization.