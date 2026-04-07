=== CALIBRATION EXAMPLE 42 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the paper's technical content. The abstract claims (i) "theoretical guarantees of convergence to the Pareto front," (ii) multi-objective guidance across up to five properties, and (iii) superiority over evolutionary and diffusion-based baselines. All three claims are explored in the paper, but as detailed below, there is a serious disconnect between claims (i) and the actual experimental setup.

---

### Introduction & Motivation

The motivation is strong and well-grounded: biomolecular design is inherently multi-objective, yet most computational methods optimize single objectives. The gap with ParetoFlow (continuous space) is correctly identified. The contributions list is clear.

**Concern:** Contribution #2 states "theoretical guarantees of distributional invariance and convergence to the Pareto front," but Section 4 immediately walks this back: "in practice, these guarantees hold only in the limit of an infinitely long Markov chain." More critically, the actual experiments do *not* use the theoretically analyzed algorithm — they use a monotonicity-constrained variant (discussed below). This tension is not flagged in the introduction.

---

### Method (Section 3)

The three components — Tchebycheff scalarization, locally balanced proposals (Zanella-style), and Metropolis-Hastings — are individually well-established in the MCMC and multi-objective optimization literature. The novelty lies in their integration within a discrete flow prior for biological sequence generation. This is reasonable, but the contribution claims should be calibrated accordingly.

**Critical concern — The monotonicity constraint:** Section 4 introduces a constraint that "accepts only token updates that increase the weighted sum of the current objective scores." This transforms the Markov chain from a theoretically-grounded MH sampler into a greedy hill-climber. The MH acceptance ratio (Section 3.4) ensures detailed balance with respect to π_{η,ω}; the monotonicity constraint does not. This directly breaks the invariance and convergence guarantees proven in Appendix A. The paper briefly notes this in the ablation (Table 6) and says it "does not alter the underlying optimization objectives," but this is misleading: it changes the dynamics fundamentally. All experiments use this constraint. The theoretical guarantees do not apply to the method evaluated in the experiments.

**Annealing schedule:** The annealing schedule is η_t = η_min + (η_max − η_min) · t/(T-1), which is *linear in the iteration index*, not logarithmic or exponential as in standard simulated annealing. The convergence of simulated annealing to the global optimum requires a logarithmically slow cooling schedule; the linear schedule used here provides no such guarantee. The paper does not justify why a linear schedule is sufficient given the finite-length chain.

**Weight selection:** The Coverage Guarantee theorem requires ω to be drawn from a distribution with full support on the interior simplex. In practice, experiments use equal weights for all objectives in the wild-type task, and fixed weights (0.7 for affinity, 0.1 each for others) in the SMILES task. Single fixed weight vectors do not explore the full Pareto front; only single Pareto-optimal points are targeted per run. The claim of "full coverage of the Pareto front" does not follow from running AReUReDi with fixed weights.

---

### Theoretical Guarantees (Appendix A)

**Invariance Theorem:** The proof is technically correct for the Barker's function case (acceptance α = 1), but the proof for the general case relies on a ratio of proposal probabilities equaling W_{η,ω}(x')/W_{η,ω}(x), which requires that the normalizing constant of q_i cancels. This cancellation does hold when the candidate set is the full vocabulary but may not hold with top-p pruning (used for computational efficiency in the SMILES task). The theorem statement should explicitly condition on the pruning scheme.

**Convergence to Pareto Front Theorem:** The Step 1 proof handles the case where the dominated state y has strictly higher score on the bottleneck coordinate m. For the non-bottleneck case, it invokes "measure zero" arguments, but in a finite discrete state space, ties among objectives are not measure zero — they are combinatorially common, especially when objectives have coarse-grained outputs (e.g., binary classifiers for hemolysis, non-fouling, solubility). The proof needs to handle ties explicitly.

**Pareto Representability Theorem:** The proof requires s̃_n(x†) > 0 for all n. If any normalized objective equals zero for a Pareto-optimal point (which is possible at the boundary), the proof falls back to a ε-perturbation argument. This case should be handled more carefully since the resulting weight depends on ε and may not lie in the interior of the simplex.

**Overall:** The proofs are partially correct under idealized conditions, but the gap between the theoretical setup and the actual experimental algorithm (monotonicity-constrained greedy search) is never bridged.

---

### Experiments & Results

**Score model quality:** The oracle models used for guidance and evaluation are weak by modern standards: hemolysis F1 = 0.58, non-fouling F1 = 0.71, solubility F1 = 0.68, binding affinity Spearman ρ = 0.64 (Appendix E). The half-life model is trained on only 105 data points with R² = 0.60. Optimizing for unreliable oracles — and then evaluating on those same oracles — inflates the apparent performance. This is the standard proxy model problem in offline design, but the paper never discusses this limitation. Are the oracle scores measuring real drug properties or are they measuring artifacts of the score model?

**Pareto front metrics absent:** The standard metric for multi-objective optimization is **hypervolume**, which measures the volume of objective space dominated by the generated set. Table 2 reports per-objective averages, not hypervolume. High average scores across objectives are necessary but not sufficient to characterize Pareto quality — a method could collapse to a single Pareto point and still show high per-objective averages. Without hypervolume (or at minimum, pairwise Pareto dominance counts), the claim of "superior trade-off navigation" cannot be rigorously supported.

**Baseline fairness:**

1. PepTune is paired with DPLM as its backbone for the comparison, not the originally published PepMDLM backbone. The paper describes this as adapting "PepTune's backbone to the existing DPLM model," but PepTune was designed and evaluated with PepMDLM. Using a weaker backbone for the baseline potentially inflates AReUReDi's advantage.

2. The evolutionary baselines (NSGA-III, SMS-EMOA, SPEA2, MOPSO) receive no guidance from a generative prior trained on biological sequence distributions, while AReUReDi benefits from PepReDi's prior. This is an architecture-level advantage that should be controlled for (e.g., by running the evolutionary methods initialized from PepReDi samples, or by providing them the same score models used in AReUReDi).

3. For the SMILES task, AReUReDi is compared to PepTune but the comparison is dismissed because "PepTune does not report average property scores" — leaving AReUReDi's SMILES results essentially without a quantitative multi-objective baseline.

**Computational cost:** AReUReDi requires 55 s/binder for 8-mer designs and 195 s/binder for 16-mer designs (Table 2), compared to 2–37 s for baselines. The ~22× slowdown relative to PepTune and ~5× relative to NSGA-III is a meaningful practical limitation. The time-matched comparison (Table 11) compares "top-2 AReUReDi binders" against "100 PepTune binders" — cherry-picking two from a very small candidate pool (4 total) against a full candidate set is not a valid time-matched comparison.

**Statistical significance:** No confidence intervals or statistical tests are reported anywhere in the main results (Tables 1, 2) beyond reporting "averages of 100 binders." Given the stochasticity of both AReUReDi and the baselines, it is impossible to assess whether performance differences are statistically significant.

**ReDi conditional TC non-monotonicity:** Table 4 shows TC increasing from 10.60 (base) to 12.63 after the first rectification round, then decreasing to 11.73 and 11.23. The paper ascribes this to "distributional shift from the large, model-generated coupling," but this contradicts the stated guarantee that "TC_s|t(π^(k+1)) ≤ TC_s|t(π^(k))" (Section 2.2). Either the ReDi theorem does not apply in this experimental setting or additional explanation is required.

**No experimental validation:** All reported results are computed against in silico oracle models. AlphaFold3 ipTM and AutoDock VINA docking are provided for a handful of example structures in Figure 1, but these are also computational predictions. No wet lab experiments, even simple cell-free assays for hemolysis or solubility, are presented.

---

### Related Work

The related work is comprehensive and honest about what can and cannot be compared directly (e.g., structural methods, continuous-space methods). The explicit discussion of why ParetoFlow and other continuous-domain methods are not appropriate baselines is appreciated.

**Minor concern:** Multi-objective GFlowNets (Jain et al., 2023), which operate in discrete spaces and are specifically designed to sample diverse Pareto-optimal solutions, are cited but not included as a baseline. This is a natural competitor that should at minimum be discussed as a reason for exclusion.

---

### Limitations & Broader Impact

The limitations section is brief and deals primarily with future extensions rather than current weaknesses. The paper does not acknowledge:

1. The gap between theoretical guarantees and the monotonicity-constrained algorithm used in experiments.
2. The reliance on weak proxy models, and the lack of any experimental validation.
3. The sensitivity to weight vector choice when the stated Pareto coverage requires weight randomization.
4. The computational overhead that limits practical applicability.

The ethics statement appropriately acknowledges dual-use risks.

---

### Overall Assessment

AReUReDi addresses a genuinely important problem — multi-objective discrete sequence optimization — and the combination of locally balanced MH proposals with annealed Tchebycheff scalarization over a learned discrete flow prior is principled and well-motivated. The paper is clearly written and experimentally thorough across many targets. However, several fundamental concerns undermine the current submission's readiness for ICLR. Most critically: **the theoretical guarantees claimed prominently throughout the paper (including in the title and abstract) do not apply to the algorithm actually evaluated in experiments**, because the monotonicity constraint destroys detailed balance. Beyond this, the absence of hypervolume metrics makes it impossible to truly assess Pareto quality; the score models used for guidance and evaluation are weak and the results are never validated experimentally; and the PepTune baseline comparison uses a weaker backbone than originally published. In its current form, the paper overstates both its theoretical and empirical contributions. These issues are addressable in principle — particularly if the authors disentangle the pure MH results from the constrained variant and adopt standard multi-objective metrics — but they require substantial revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **AReUReDi**, a multi-objective optimization framework that extends Rectified Discrete Flows (ReDi) to generate biomolecular sequences satisfying conflicting constraints. By integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates, the method claims theoretical convergence to the Pareto front while preserving distributional invariance. Experimental results on peptide and SMILES design demonstrate superior trade-off navigation compared to evolutionary and diffusion-based baselines.

### Strengths
1.  **Strong Theoretical Guarantees:** Unlike many empirical generative optimization methods, AReUReDi provides formal proofs (Appendix A) for Markov chain invariance and convergence to Pareto-optimal states under the provided scalarization, which is a significant contribution for discrete generative modeling.
2.  **Comprehensive Multi-Objective Benchmarking:** The evaluation is extensive, comparing against standard evolutionary MOO (NSGA-III, SPEA2) and recent diffusion approaches (PepTune). The ablation studies on rectification rounds (Table 9) and annealed schedules (Table 10) provide concrete evidence for the necessity of each proposed component.
3.  **Practical Domain Application:** The framework addresses a critical gap in biomolecular design by handling up to five conflicting therapeutic properties (e.g., hemolysis, half-life) directly on discrete sequences without continuous embedding, avoiding the distortion issues mentioned in the Introduction.

### Weaknesses
1.  **High Computational Latency:** Table 2 indicates a substantial runtime overhead for AReUReDi (e.g., 55 seconds per binder for 1B8Q) compared to evolutionary baselines (e.g., 8.54 seconds for MOPSO) and diffusion methods (2.46 seconds for PepTune). While the authors mitigate this with a "matched wall-clock" comparison in Table 11, the raw cost limits high-throughput application potential.
2.  **Reliance on In-Silico Evaluated Metrics:** The entire experimental validation depends on predictor models (XGBoost, AlphaFold3, AutoDock) rather than physical wet-lab verification. Table 1 notes the lack of public datasets for benchmarking, which restricts the generalizability of the "therapeutic" claims until experimental validation is conducted.
3.  **Scalability of Proposal Mechanism:** The locally balanced proposal requires evaluating candidate tokens against all $N$ objectives at every coordinate update. While feasible for five objectives as shown, the paper does not discuss how the computational complexity scales for a larger number of objectives or larger sequence lengths, nor does it address potential mixing time issues in high-dimensional spaces.

### Novelty & Significance
**Novelty:** The integration of annealed scalarization and MCMC updates into the ReDi framework represents a novel synthesis of flow matching and multi-objective optimization. To the best of my knowledge, there have been few attempts to provide Pareto guarantees specifically within discrete rectified flow contexts.

**Significance:** This work is significant for the computational biology community, as it offers a principled method for designing complex biological sequences where trade-offs are inevitable. The theoretical grounding distinguishes it from heuristic guidance methods common in the field, potentially setting a new standard for verifiable optimization in generative design.

### Suggestions for Improvement
1.  **Optimize Sampling Efficiency:** Investigate adaptive step sizing or a "warm-up" phase where the guidance strength is lower and coordinate updates are sparser to reduce the $O(T \times L \times K \times N)$ complexity per sample.
2.  **Clarify Baseline Fairness:** In the comparison with PepTune, ensure the input data distribution (training set) is clearly defined, as the performance gap may depend on the quality of the base model prior rather than just the guidance mechanism.
3.  **Add Experimental Validation Discussion:** While wet-lab data may be scarce, include a discussion or preliminary results involving "gold standard" experimental peptide property datasets (e.g., PEPLO) to quantify the alignment between predictor scores and known experimental values more robustly.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ground-truth Pareto front validation** — All results rely on learned score models with modest validation performance (F1 0.58-0.71, Spearman 0.64-0.86). Without comparison to experimentally measured properties or known Pareto fronts, claims of Pareto optimality are unverifiable and could reflect score model artifacts.

2. **Fair baseline comparison with matched compute** — Table 11 reveals AReUReDi generates only 3-4 sequences while PepTune generates 100 in the same wall-clock time. This unfair comparison undermines the superiority claim; a proper evaluation should compare best-of-N performance with equal sample budgets.

3. **Monotonicity constraint breaks theoretical guarantees** — Section 4 admits the monotonicity constraint (accepting only improvements) was used in all experiments, but this violates the MCMC detailed balance required for the claimed convergence guarantees. Experiments without this constraint are needed to validate the theory.

4. **Ablation on score model error propagation** — No analysis shows how imperfect score predictions affect the optimization trajectory. Adding noise to score models or comparing against oracle scores would reveal whether results are robust or overfit to specific predictors.

5. **Test on held-out protein targets** — All targets appear in training data for the base models and score models. Evaluation on completely held-out targets with no training overlap is needed to claim generalization rather than memorization.

### Deeper Analysis Needed (top 3-5 only)
1. **Pareto front coverage quantification** — Weight vector ablations (Tables 13-14) show steering capability but don't measure actual Pareto front coverage. Hypervolume indicators or coverage metrics against reference Pareto sets are needed to substantiate the "full coverage" claim.

2. **Mixing rate and convergence diagnostics** — The theoretical guarantees depend on chain mixing, but no convergence diagnostics (trace plots, effective sample size, Gelman-Rubin statistics) are provided to show the Markov chain actually reaches stationarity within the budgeted steps.

3. **Sequence novelty vs. training data overlap** — SNN scores are reported for SMILES but not peptides. Analysis of sequence similarity to training data is needed to determine if AReUReDi discovers novel solutions or rediscovered training examples.

4. **Trade-off frontier visualization** — No Pareto frontier plots showing the actual trade-off curves achieved. 2D/3D scatter plots of objective pairs would reveal whether the method truly navigates trade-offs or collapses to corner solutions.

5. **Failure mode analysis** — No discussion of when AReUReDi fails or which objective combinations are hardest to balance. Understanding limitations is critical for ICLR's reproducibility standards.

### Visualizations & Case Studies
1. **Pareto frontier plots** — Scatter plots showing generated samples in 2-3 objective dimensions alongside baseline methods would immediately reveal whether AReUReDi achieves better trade-offs or simply optimizes different objectives.

2. **Convergence trajectories per objective** — Figure 1E shows mean scores but not individual chain trajectories. Plotting multiple chains would reveal whether different runs converge to different Pareto points or the same solution.

3. **Sequence logo or motif analysis** — For peptide binders, showing conserved motifs in AReUReDi designs vs. pre-existing binders would reveal whether the method learns meaningful biochemistry or produces superficially optimized but non-functional sequences.

4. **Score model calibration plots** — Reliability diagrams for the property predictors would show whether score values are well-calibrated, since miscalibrated scores could distort the scalarization and guidance.

### Obvious Next Steps
1. **Experimental validation of top designs** — At minimum, docking or binding assays for a subset of generated peptides would ground the computational claims in physical reality, which is expected for biomolecular design papers at ICLR.

2. **Comparison to continuous-space MOO methods with discrete projections** — ParetoFlow and similar continuous methods should be evaluated with appropriate discretization to establish whether discrete flows offer genuine advantages.

3. **Ablation of each AReUReDi component** — Removing annealing, locally balanced proposals, or M-H updates individually would quantify each component's contribution rather than presenting the full method as a black box.

4. **Runtime breakdown and scalability analysis** — Table 2 shows AReUReDi is 10-80× slower than baselines. Analysis of which components dominate runtime and how performance scales with sequence length is needed for practical adoption.

# Final Consolidated Review
## Summary

AReUReDi extends rectified discrete flows (ReDi) to multi-objective optimization for biological sequence design by integrating annealed Tchebycheff scalarization, locally balanced Metropolis-Hastings proposals, and annealed guidance. The method generates peptide binders and SMILES sequences optimizing up to five therapeutic properties simultaneously. Theoretical guarantees for Pareto front convergence and distributional invariance are provided, and experiments across eight protein targets demonstrate superior trade-offs compared to evolutionary and diffusion-based baselines.

## Strengths

- **Principled integration of established techniques for discrete MOO**: The paper combines Tchebycheff scalarization (well-known in MOO), locally balanced proposals (Zanella-style MCMC), and annealing within a rectified discrete flow framework. While individual components are established, their synthesis for discrete flow-based sequence generation is novel and addresses a genuine gap in biomolecular design where existing methods operate in continuous spaces or lack Pareto guarantees.

- **Comprehensive empirical evaluation across diverse targets**: Experiments span structured proteins with known binders (1B8Q, 5AZ8, 7JVS), structured targets without known binders (AMHR2, OX1R, DUSP12), and intrinsically disordered targets (EWS::FLI1, MYC). Ablation studies in Tables 9-10 and Tables 13-14 provide evidence that rectification, annealing, and weight vector selection each contribute meaningfully to performance.

- **Clear demonstration of objective trade-off navigation**: Tables 7-8 show that disabling guidance for one objective causes collapse in that objective while others improve, confirming that AReUReDi genuinely steers sampling toward balanced multi-objective solutions rather than collapsing to single-objective optima.

## Weaknesses

- **Gap between theoretical guarantees and experimental algorithm**: The paper proves convergence and invariance guarantees for AReUReDi's Metropolis-Hastings updates, but Section 4 acknowledges that all experiments use a monotonicity constraint accepting only score-improving token updates. This constraint breaks detailed balance and invalidates the theoretical guarantees for the actual method evaluated. While the constraint improves efficiency, the paper should clearly disentangle the theoretical results (for unconstrained AReUReDi) from the practical algorithm, rather than presenting the guarantees as if they apply to the constrained variant. The ablation in Table 6 shows the constraint improves scores substantially (e.g., half-life from 2.54h to 44.70h for 8CN1), but without theoretical grounding, this becomes heuristic hill-climbing on learned surrogate functions.

- **Modest and unvalidated score model reliability**: The property predictors used for guidance and evaluation have validation F1 scores of 0.58 (hemolysis), 0.71 (non-fouling), 0.68 (solubility), and Spearman correlation of 0.64 for binding affinity (Appendix E). The half-life model is trained on only 105 data points with R² = 0.60. Optimizing for and evaluating on these same imperfect predictors raises the concern that apparent improvements may reflect exploitation of predictor artifacts rather than genuine therapeutic property improvements. No wet-lab validation or comparison to experimentally measured properties is provided.

- **No standard multi-objective quality metrics**: Tables 1-2 report per-objective averages but not hypervolume, Pareto front coverage, or dominance counts. High per-objective averages do not guarantee Pareto quality—a method that collapses to a single trade-off point could show similar averages. The weight vector ablations (Tables 13-14) demonstrate that different weights steer toward different solutions, but without hypervolume or frontier visualization, the claim of "superior trade-off navigation" versus baselines is not rigorously supported.

- **Asymmetric baseline comparisons**: For SMILES generation, the paper states PepTune "does not report average property scores" and provides no quantitative comparison for this task. For the PepTune comparison in Table 2, the backbone was adapted from the published PepMDLM to DPLM—the paper should justify whether this modification strengthens or weakens the baseline. The time-matched comparison in Table 11 compares "top-2" from only 4 AReUReDi samples against 100 PepTune samples, which is not a valid methodology for best-of-N comparison.

- **Conditional TC non-monotonicity unexplained**: Table 4 reports conditional TC values of 10.60 → 12.63 → 11.73 → 11.23 after successive rectification rounds. The initial increase contradicts Section 2.2's claim that rectification monotonically decreases TC. The paper attributes this to "distributional shift from the large, model-generated coupling," but this deserves deeper explanation since it affects the claimed theoretical properties.

## Nice-to-Haves

- **Runtime scalability analysis**: Table 2 shows AReUReDi requires 55-195s per binder versus 2-37s for baselines. Analysis of how runtime scales with sequence length, vocabulary size, and number of objectives would help practitioners assess practical applicability.

- **Wet-lab validation or external benchmark**: Even simple solubility or hemolysis assays for a subset of designed peptides would ground the computational predictions in physical reality.

- **Comparison to Multi-objective GFlowNets**: GFlowNets (Jain et al., 2023) operate in discrete spaces and sample from reward distributions—relevant for Pareto front coverage. Discussion of why they were not included as baselines would strengthen positioning.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Harsh critic's claim that PepTune+DPLM is "weaker backbone" than original PepMDLM*: The paper states DPLM is a state-of-the-art protein language model. Without evidence that this adaptation weakens the baseline, this criticism is speculative. The adaptation is clearly documented.

- *Spark finder's demand for "completely held-out targets" with "no training overlap"*: This is unrealistic in generative sequence modeling. The score models and base models necessarily train on existing peptide data. Novel target binding is evaluated via docking and structure prediction, which is standard practice.

- *Demand for multiple-run statistics and confidence intervals*: The paper reports averages over 100 generated binders. While confidence intervals would strengthen the results, this is not a standard requirement in generative modeling papers where stochasticity is inherent and sample size is provided.

- *Critic's concern about top-p pruning breaking theoretical guarantees*: The paper notes top-p is used "for computational efficiency" in the SMILES task. While this could affect the theoretical claims, the impact on the overall method is secondary to the more fundamental monotonicity constraint issue.

## Novel Insights

The integration of annealed Tchebycheff scalarization into discrete flow sampling—with the guidance strength η_t increasing during sampling rather than remaining fixed—provides a novel mechanism for progressively sharpening the objective focus while initially preserving exploration. The ablation in Table 10 confirms that annealed guidance consistently outperforms fixed guidance strengths across targets, validating this design choice empirically even when theoretical guarantees are compromised by the monotonicity constraint.

## Suggestions

- Run and report results for unconstrained AReUReDi (without monotonicity) with longer sampling budgets to validate whether the theoretical guarantees translate to empirical Pareto front convergence, even if scores are lower than the constrained variant.

- Compute and report hypervolume indicators for all multi-objective comparisons to provide standard MOO metrics. Include scatter plots of generated samples in 2D objective space to visualize trade-off frontiers.

- For the SMILES task, either run a fair baseline comparison with PepTune (using equivalent evaluation) or clearly state that no quantitative multi-objective baseline is available for this task, keeping claims proportional to evidence.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
