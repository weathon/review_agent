=== CALIBRATION EXAMPLE 19 ===

# Final Consolidated Review
## Summary
This paper proposes **AReUReDi**, a multi-objective extension of Rectified Discrete Flows that combines Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates to guide discrete sequence generation toward favorable trade-offs. The paper is ambitious: it offers a nontrivial theoretical treatment and applies the method to peptide and peptide-SMILES design with multiple therapeutic objectives, but the current submission overstates the alignment between its theory and the algorithm actually used in experiments.

## Strengths
- **The paper identifies and formalizes a meaningful algorithmic gap: multi-objective guidance for discrete flow-based generation.** The combination of a ReDi prior with locally balanced single-site proposals and scalarized reward tilting is a concrete and technically interesting construction, rather than a generic “add guidance” recipe.
- **The MCMC formulation is specific and potentially valuable if taken on its own terms.** Appendix A does not just gesture at guarantees; it states invariance, concentration on scalarized optima as \(\eta \to \infty\), and a representability/coverage story via randomized \(\omega\). That theoretical scaffold is stronger than what is typical for biomolecular design papers.
- **The paper includes targeted ablations that do illuminate some design choices.** In particular, Tables 9 and 10 isolate the effects of rectification and annealed guidance strength, and Tables 13–14 show that changing the weight vector steers different trade-offs rather than leaving outputs unchanged.
- **The method is demonstrated across two discrete domains with different tokenization/validity constraints.** The transfer from wild-type peptide generation to peptide-SMILES generation, while keeping the same high-level sampling framework, is a substantive sign of modularity.
- **The authors are unusually explicit about practical deviations introduced for efficiency.** Section 4 openly states that the theoretical guarantees hold only asymptotically and that an additional monotonicity constraint is used in all experiments for acceleration. That transparency makes the paper easier to assess, even though it exposes a major issue.

## Weaknesses

### Fatal
- **The empirical algorithm is not the theoretically analyzed algorithm, and this directly undermines the paper’s core claims.**  
  Section 4 states: *“we introduce a monotonicity constraint that accepts only token updates that increase the weighted sum of the current objective scores… Therefore, this monotonicity constraint was involved in all the following experiments.”*  
  However, the theory in Section 3 / Appendix A is built around a locally balanced proposal plus **Metropolis-Hastings** transition targeting  
  \[
  \pi_{\eta,\omega}(x)\propto p_1(x)\exp(\eta S_\omega(x)).
  \]
  A hard monotonicity filter based on a **different criterion** (“increase the weighted sum of the current objective scores”) is not part of that kernel and is not analyzed. This is not a minor implementation detail: it changes the transition rule, breaks the stated detailed-balance argument, and severs the link between the proofs and the reported results. As written, the paper’s strongest claim—provable convergence behavior of the proposed practical method—is unsupported by the experiments actually run.

### Major:
- **The headline claim of “full coverage of the Pareto front” is not supported by the main experimental setup.**  
  The coverage theorem in Appendix A explicitly requires randomizing \(\omega\) from a distribution with full support on the simplex interior: *“If \(\omega \sim \mu\) and \(\eta \to \infty\), then the induced sampler visits every Pareto-optimal state with positive probability.”*  
  But Appendix F states: *“We applied the same weight for each objective in all wild-type peptide binder generation tasks.”* A fixed equal-weight vector does not test front coverage; it tests one scalarization setting. The paper does include weight-vector ablations (Tables 13–14), which partially addresses the issue, but the broad claims in the abstract/introduction/discussion are stronger than what the main experiments validate. Theoretical representability of the full front is not the same as empirical demonstration of coverage.
- **The empirical evaluation does not use standard Pareto-quality metrics, so the central multi-objective claims are difficult to verify quantitatively.**  
  Most results are reported as average objective values over generated samples. That is not sufficient to establish Pareto dominance, front approximation quality, or coverage. For a paper centered on Pareto-front convergence and trade-off navigation, missing metrics such as hypervolume or IGD materially weakens the empirical case.
- **The method depends heavily on imperfect surrogate objective models, and this substantially limits how strongly one can interpret the biological claims.**  
  The score models used for guidance are not uniformly strong: Appendix E reports F1 scores of **0.58** (hemolysis), **0.71** (non-fouling), **0.68** (solubility), a validation Spearman of **0.64** for affinity, and a half-life model fine-tuned on only **105** entries with \(R^2=0.5977\). Since optimization is entirely against these predictors, high reported objective values may partly reflect exploitation of oracle error rather than genuine improvement in underlying properties. This does not invalidate the algorithmic contribution, but it does weaken the significance of the biological-design conclusions.
- **There remains an unresolved attribution issue between the value of the ReDi prior and the value of the proposed multi-objective guidance.**  
  Appendix G shows the learned prior is important (Tables 15–16): replacing \(p_1\) with a uniform prior degrades results noticeably. That is a useful result, but it also means the current experiments do not cleanly disentangle how much of the performance comes from (i) having a strong sequence prior versus (ii) the specific AReUReDi guidance mechanism. The paper argues the prior is a crucial ingredient, which is fair, but then claims about the superiority of the guidance mechanism itself should be stated more carefully.

### Minor
- **Algorithm 1 does not match the algorithm used in experiments.** It omits the monotonicity constraint that Section 4 says is used in all experiments, and it also does not make the candidate-pruning / validity-rejection choices central to some settings as explicit parts of the practical algorithm.
- **Candidate truncation and validity filtering introduce additional gaps between the analyzed kernel and the practical sampler.** Section 3.3 allows top-\(p\) pruning, and Appendix F says that for SMILES only the top 200 candidates are evaluated and invalid-peptide transitions are rejected. These are understandable engineering choices, but the paper currently presents theoretical guarantees without clearly delimiting that they apply only to the untruncated/unconstrained sampler.
- **The compute story is not especially favorable, and the paper does not fully analyze the quality–compute trade-off.** Table 2 shows AReUReDi is much slower than several baselines, including PepTune. The paper is reasonably transparent about this and includes a matched-time comparison, but the analysis would be stronger with compute-normalized Pareto metrics rather than top-k exemplars.
- **Some theoretical statements are stronger than their proofs/support warrant.** In particular, the paper repeatedly phrases guarantees as convergence “to the Pareto front with full coverage,” but Appendix A’s concentration result is for scalarized maximizers \(F_\omega\) at fixed \(\omega\), and full coverage requires additional randomization over \(\omega\). The theorem statements are more careful than parts of the main text.

### Trivial
- None.

## Nice-to-Haves
- Report hypervolume / IGD (or similar) for all multi-objective experiments, especially for the compute-matched comparison.
- Add an explicit ablation **without** the monotonicity constraint in the main benchmarking tasks, not only Table 6 on two auxiliary targets, so readers can assess how much the reported gains depend on this heuristic.
- Provide a clearer “theory vs practice” subsection stating exactly which implementation choices preserve the target distribution and which are heuristic accelerations.
- Include robustness checks with alternative or independent property predictors, or at least uncertainty-aware evaluation, to reduce concern about surrogate exploitation.
- Add mixing diagnostics or acceptance statistics across the annealing schedule to distinguish effective exploration from early stagnation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that classical MOO baselines are “outdated” and therefore inappropriate.** This is too external-knowledge dependent, and the paper already includes a recent diffusion-based comparator (PepTune) in addition to classical baselines.
- **Complaint that comparisons are unfair because the baselines do not use the same learned prior.** The asymmetry here favors the baselines less than the proposed method only if one assumes the prior is a confound; but the paper’s method fundamentally includes that prior as part of the model. This is not, by itself, a valid fairness objection.
- **Generic reproducibility complaints about missing tuning details or release status.** The paper provides substantial implementation detail, and availability concerns are outside scope here.

## Novel Insights
The key issue is not simply that the paper uses heuristics in practice; many strong papers do. The deeper problem is that the paper’s identity is built around a theory-backed Pareto sampler, yet the practical system evaluated is closer to a **prior-guided, annealed, monotone local search heuristic** layered on top of ReDi. That reframing explains several otherwise disconnected observations at once: why the monotonicity constraint yields dramatic gains, why the ReDi prior matters so much, why fixed-\(\omega\) experiments still look strong despite claims about full-front coverage, and why average-score reporting is insufficient. In other words, the submission may contain both a principled MCMC formulation and an effective heuristic optimizer, but it currently treats them as the same object when they are not.

## Suggestions
- **Separate the contributions cleanly.** Present the theoretically valid AReUReDi sampler and the practical accelerated variant as distinct algorithms. Re-scope claims accordingly.
- **Either remove or sharply qualify the strongest convergence/coverage claims in the abstract and discussion** unless the empirical algorithm is brought back into alignment with the analyzed kernel.
- **Add Pareto metrics** (hypervolume, IGD, dominance counts) and show results across multiple \(\omega\) values if “front coverage” remains a central claim.
- **Quantify the effect of the monotonicity constraint** more directly: compare final Pareto quality, diversity, and empirical coverage with and without it under matched compute.
- **Strengthen oracle validation or temper biological conclusions.** At minimum, acknowledge that the current results are surrogate-guided in silico optimization, not strong evidence of true therapeutic improvement.
- **Clarify where the gains come from.** A stronger decomposition would compare ReDi prior + simple greedy search, ReDi prior + original MCMC AReUReDi, and the monotone constrained variant, so the reader can attribute improvements to the prior, the balanced proposal mechanism, and the heuristic acceleration separately.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
