=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
## Summary

AReUReDi extends rectified discrete flows (ReDi) to the multi-objective optimization setting for discrete sequence design (e.g., therapeutic peptides, SMILES). It integrates annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates within an MCMC framework, providing theoretical guarantees of invariance and convergence to the Pareto front. Empirically, it demonstrates simultaneous optimization of up to five competing biological properties, outperforming several classical multi-objective algorithms and a recent diffusion-based baseline on custom peptide and SMILES design tasks.

## Strengths

- **Novel Integration with Theoretical Grounding:** The paper presents the first principled extension of rectified discrete flows to multi-objective optimization. It provides clear theoretical guarantees (Theorems in Section A.2) for distributional invariance, convergence to Pareto-optimal states, and coverage of the Pareto front, which is a significant and well-motivated contribution for this domain.
- **Comprehensive and Specific Empirical Evaluation:** The method is validated on two distinct and relevant sequence modalities—wild-type peptides and chemically-modified peptide SMILES—simultaneously optimizing up to five realistic, conflicting therapeutic properties (affinity, solubility, hemolysis, half-life, non-fouling). The evaluation includes systematic ablation studies (rectification, annealing schedules, weight vectors) and comparisons against multiple baseline classes (evolutionary algorithms, a diffusion-based method).

## Weaknesses

### Major:
1. **Substantial Theory-Practice Gap Undermines Core Claim:** The paper's central theoretical contribution is the guarantee that AReUReDi converges to the Pareto front. However, the authors state (lines 299-303) that *"in all reported experiments, we introduce a monotonicity constraint that accepts only token updates that increase the weighted sum of the current objective scores"* to improve sampling efficiency. This constraint is **not part of the analyzed algorithm** (Algorithm 1, Section 3) and fundamentally alters the Markov chain's dynamics by overriding the standard Metropolis-Hastings acceptance rule. Consequently, the empirical results validate a *different, heuristic algorithm* than the one for which theoretical guarantees are provided. This severs the link between the paper's key theoretical contribution and its experimental validation.
2. **Incomplete and Unbalanced Efficiency Evaluation:** While AReUReDi achieves superior objective scores, it is significantly slower (2-40x) than all baselines (Table 2). A matched-time comparison is provided only against PepTune (Table 11), not against the faster evolutionary algorithms (MOPSO, NSGA-III, etc.). Without such comparisons, the claim of "outperforming" baselines is incomplete; the performance gains could be attributable to greater computational investment rather than a more efficient optimization strategy. This limits the practical significance of the result.
3. **Evaluation Lacks Standard Multi-Objective Metrics:** The empirical assessment relies almost entirely on average property scores across generated sequences. The paper does not report standard multi-objective optimization metrics—such as hypervolume, set coverage, or spacing—that are necessary to quantitatively evaluate the quality, diversity, and spread of the *approximated Pareto front*. This makes it difficult to assess whether the method truly achieves good "coverage" or merely finds high-average-score points.

### Minor:
4. **Heavy Reliance on Moderate-Accuracy Surrogate Models:** The optimization is guided by pre-trained property predictors. Key models have limitations: the half-life predictor is fine-tuned on only 105 data points (Section E.3), and classifiers for properties like hemolysis have modest validation F1 scores (0.58-0.71, Section E.1). The paper does not analyze how errors or biases in these surrogate models propagate and potentially misguide the optimization, which is a risk for real-world applicability.
5. **Experimental Validation Relies on Custom, Non-Public Benchmarks:** The authors note that "no public datasets exist for benchmarking multi-objective optimization algorithms on biological sequences" (lines 288-290) and therefore create custom benchmarks. While understandable, this choice limits reproducibility and the ability for external, community-wide validation and comparison. The use of established, public multi-objective test problems (even from other domains) would strengthen the generalizability of the claims.

### Trivial:
6. **Clarity on Algorithm Implementation:** The description of the monotonicity constraint is buried in the experimental section (Section 4) rather than being integrated into the method description (Section 3) or algorithm pseudocode (Algorithm 1). This makes it easy for a reader to miss this critical implementation detail that differs from the theoretically analyzed method.

## Nice-to-Haves
- **Visualization of the Pareto Front:** Generating solutions by sweeping the weight vector ω and plotting the attained objective values (e.g., in 2D scatter plots for pairs of objectives) would provide an intuitive, visual demonstration of Pareto front coverage and diversity, beyond reporting averages.
- **Ablation on the Monotonicity Constraint:** A direct comparison of performance with and without the heuristic constraint would clarify its necessity and quantify its impact on solution quality and chain mixing.
- **Convergence Diagnostics:** Presenting trace plots of objective scores or acceptance rates over iterations would help build empirical confidence that the finite-length Markov chains are mixing adequately in practice.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Score models are not released / cannot be verified."** REMOVED. The paper cites the score models and describes their training in Appendix E. Per hard rules, we assume cited entities exist.
- **"The paper is not novel, just a combination of existing parts."** WEAKENED & MOVED. The integration itself is a novel contribution for this domain (discrete MOO). The criticism is overly reductive and ignores the non-trivial theoretical synthesis.
- **"The authors should have done wet-lab experiments."** REMOVED. This is a computational methods paper. Demanding wet-lab validation is scope creep outside its stated contribution (in silico design and optimization).
- **"The writing is poor / there are typos."** REMOVED. Pure formatting/style nitpick per hard rules.
- **"Hyperparameters for baselines are not fully specified."** REMOVED. Nitpick about reproducibility of trivial implementation details.
- **"The method is too slow to be useful."** WEAKENED. While efficiency is a valid concern (addressed above), outright dismissal of utility is an overstatement given the paper's focus on proof-of-concept and high-stakes design where compute may be justified.

## Suggestions
1. **Reconcile Theory and Practice:** Either (a) provide a theoretical analysis justifying the monotonicity constraint (e.g., proving it preserves convergence under certain conditions) and integrate it into the formal algorithm description, or (b) run key experiments using the core, unmodified Algorithm 1 (perhaps with more steps) to demonstrate that the theoretical guarantees lead to empirically strong performance without the heuristic.
2. **Conduct Matched-Time Comparisons:** Perform and report a matched-wall-clock-time comparison against *all* baselines, including the faster evolutionary algorithms, to fairly establish optimization efficiency.
3. **Report Standard MOO Metrics:** Calculate and report metrics like hypervolume and set coverage for the final set of generated solutions to quantitatively demonstrate Pareto front approximation quality.

**Overall Paper Quality:** This is a **technically sound paper with a serious, fundamental flaw**. The core algorithmic idea and theoretical framework are novel and well-constructed. The experimental scope is thorough. However, the disconnect between the theoretically analyzed algorithm and the empirically evaluated one is severe and undermines the paper's central claim. If the authors can successfully bridge this gap (as per Suggestion 1), the paper would be strong. In its current form, the validity of its primary contribution is questionable.
- **Novelty:** High. First multi-objective extension of rectified discrete flows with theoretical guarantees for this setting.
- **Technical Soundness:** The theoretical analysis is sound for the presented algorithm, but its empirical validation is compromised by the use of a different, untheorized heuristic.
- **Empirical Support:** Extensive but critically incomplete due to the theory-practice mismatch and missing standard MOO metrics.
- **Significance:** Potentially high for therapeutic design, but the current methodological issue limits its demonstrated impact.
- **Clarity:** Generally clear, but the hiding of the critical monotonicity constraint in the experiments section is a major oversight.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
