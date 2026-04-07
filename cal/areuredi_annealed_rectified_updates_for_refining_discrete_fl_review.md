=== CALIBRATION EXAMPLE 36 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title "AReUReDi: Annealed Rectified Updates for Refining Discrete Flows with Multi-Objective Guidance" accurately reflects the paper's core contribution. The abstract is well-written, clearly stating the problem (multi-objective discrete sequence design), the key components of the method (annealed Tchebycheff scalarization, locally balanced proposals, MH updates), the claimed theoretical guarantees, and the application domains. All abstract claims are supported in the main text.

### Introduction & Motivation
The introduction effectively motivates the problem of multi-objective biomolecular design, citing relevant therapeutic and engineering challenges. It clearly situates the work within existing literature (single-objective optimization, black-box MOO, continuous Pareto methods, discrete flows) and identifies the specific gap: no framework for Pareto guidance in discrete flows. The contributions are stated precisely. The motivation is strong and appropriate for ICLR.

### Method / Approach (Sections 2, 3, and Appendix A)
The method is clearly described, building logically from ReDi foundations to the AReUReDi extensions.
*   **Reproducibility:** The algorithm (Algorithm 1) is presented clearly. The connection between the theoretical target distribution \(\pi_{\eta_t,\omega}(x) \propto p_1(x) \exp(\eta_t S_\omega(x))\) and the practical MCMC procedure is well-explained.
*   **Theoretical Guarantees (Appendix A):** The proofs for invariance, convergence to the Pareto front, representability, and coverage are provided and appear correct for the finite state space setting. A key strength is linking the Tchebycheff scalarization's properties to Pareto optimality.
*   **Potential Gaps/Concerns:**
    1.  **Practical Deviation from Theory:** In Section 4, the authors introduce a **monotonicity constraint** (accepting only token updates that increase the weighted sum of current objectives) to improve sampling efficiency. This is a significant practical heuristic that breaks the detailed balance of the MCMC kernel described in Theorem 1 (Invariance). The theoretical guarantees no longer strictly apply when this constraint is active. The paper should explicitly discuss this trade-off, analyze its impact on the stationary distribution, and justify why it is necessary despite the theoretical compromise. Table 6 shows it's crucial for performance, making this a major point to address.
    2.  **Assumption on Base Distribution \(p_1\):** The method requires a pre-trained ReDi model to provide a high-quality prior \(p_1(x)\). The theoretical guarantees assume this prior has full support. The ablation in Tables 15 & 16 shows the prior's importance. However, the paper does not deeply discuss what happens if \(p_1\) is poorly calibrated or misses regions of the Pareto front. How robust is AReUReDi to imperfections in the base generator?
    3.  **Annealing Schedule Justification:** The linear annealing schedule for \(\eta_t\) is provided but not justified beyond analogy to simulated annealing. Some sensitivity analysis or discussion on the choice of schedule would strengthen the method.

### Experiments & Results (Section 4 and Appendices)
The experimental evaluation is extensive, covering two sequence modalities (wild-type peptides, peptide SMILES) and up to five objectives.
*   **Baselines:** Comparisons against classical MOO algorithms (NSGA-III, etc.) and a recent diffusion-based method (PepTune) are appropriate and show AReUReDi's superior trade-off navigation. The wall-clock time comparison (Table 11) is a good response to the runtime disadvantage.
*   **Ablation Studies:** These are a major strength. Ablations on rectification (Table 9), annealing (Table 10), guidance ablation (Tables 7, 8), weight vectors (Tables 13, 14), and prior importance (Tables 15, 16) convincingly demonstrate the contribution of each component.
*   **Evaluation Metrics:** The use of property scores, validity, diversity, and SNN is comprehensive. The inclusion of structural validation via AlphaFold3 ipTM and AutoDock VINA docking scores (Figs 1, A1, A2) adds credibility to the generated binders.
*   **Concerns and Missing Analyses:**
    1.  **Statistical Significance:** Results are presented as averages over 100 runs. Reporting standard deviations or confidence intervals, at least in supplementary material, is essential to assess the robustness of the improvements claimed in tables (e.g., Table 2, Table 9).
    2.  **Quality of Score Models:** The performance of AReUReDi is contingent on the accuracy of the property predictors. The half-life model is particularly concerning: it is pre-trained on stability data and fine-tuned on only **105 data points** (Section E.3). While the reported Spearman correlation is high, such a small dataset risks overfitting and limits the reliability of the half-life guidance. This is a significant limitation that must be explicitly discussed.
    3.  **Coverage of the Pareto Front:** The theory guarantees coverage in the limit. Empirically, how well does AReUReDi cover the Pareto front? The weight vector ablation (Tables 13, 14) shows steering, but a more direct visualization or metric for front coverage (e.g., hypervolume) would be valuable.
    4.  **Comparison to More Discrete MOO Baselines:** While PepTune is a relevant diffusion baseline, the field of *guided* discrete diffusion/generative models is active. A comparison to other guidance techniques for discrete models (e.g., plug-and-play, classifier guidance) adapted for multi-objective settings, even if simpler, would better contextualize the contribution.

### Writing & Clarity
The paper is generally well-written and logically organized. The figures are informative. Some minor points:
*   The notation switch from \(p_t^i(\cdot|x_t)\) (ReDi prior) to its use in the proposal is sometimes abrupt. A clearer reminder in Section 3.3 that this is the *marginal* from the pre-trained ReDi would help.
*   The explanation of the conditional total correlation (TC) in Section 2.1 and its behavior during rectification (Section B) could be smoother. The note that TC can rise after the first rectification due to a distributional shift is important but tucked away in Appendix B.

### Limitations & Broader Impact
The discussion and ethics statement appropriately note potential misuse and propose safeguards. The reproducibility statement is detailed.
*   **Key Limitations to Emphasize:**
    1.  **Dependence on Score Models:** As noted, the method's output is only as good as its property predictors. Inaccuracies or biases in these models will directly propagate.
    2.  **Computational Cost:** While justified by superior performance, the sequential coordinate-wise MCMC sampling is more expensive than a single forward pass of a conditional generator. The trade-off is clear, but it remains a limitation for real-time or large-scale design.
    3.  **Monotonicity Constraint:** This practical modification, essential for performance, deviates from the theoretical framework and needs a proper discussion as a limitation/approximation.
    4.  **Finite-time Sampling:** The theoretical guarantees are asymptotic. The paper should more explicitly state that in practice, with finite steps, they achieve an *approximation* of the Pareto front.

### Overall Assessment
This is a strong paper with a novel and well-executed core idea: integrating rectified discrete flows with a theoretically grounded MCMC procedure for multi-objective optimization. The technical contributions are clear, the theory is sound, and the experimental validation is thorough and convincing across multiple domains. The main concerns are the **introduction of the monotonicity constraint without a full theoretical reconciliation** and the **heavy reliance on property predictors of varying quality** (especially the half-life model). Addressing these points, particularly by discussing the constraint's impact and providing statistical significance for results, is crucial for acceptance at ICLR. If these issues are adequately clarified, the paper makes a significant contribution worthy of publication.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces AReUReDi, a framework for multi-objective optimization of discrete biological sequences (e.g., peptides, SMILES) that extends rectified discrete flows (ReDi). It integrates annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates to provably converge to the Pareto front while preserving distributional invariance. Experiments demonstrate simultaneous optimization of up to five therapeutic properties, outperforming evolutionary and diffusion-based baselines.

### Strengths
1. **Strong Theoretical Foundation**: The paper provides clear theoretical guarantees (invariance, Pareto convergence, coverage) in the appendix, aligning with ICLR's emphasis on rigor. Proofs are detailed and well-structured.
2. **Comprehensive Empirical Evaluation**: The method is tested on both wild-type peptide and SMILES sequence design across multiple targets (8+ proteins) with up to five objectives. Ablation studies (rectification, annealing, weight vectors) are thorough and support design choices.
3. **Effective Benchmarking**: Comparisons against four classical MOO algorithms (NSGA-III, SMS-EMOA, etc.) and a recent diffusion-based method (PepTune) show consistent improvements in Pareto trade-offs, with significant gains in properties like half-life (22–38h vs. 1–7h for baselines).

### Weaknesses
1. **High Computational Cost**: AReUReDi is notably slower than baselines (e.g., 55–195 seconds per binder vs. 2–37 seconds for evolutionary methods), which may limit scalability. The need for many sampling steps (e.g., 128–256) and full candidate evaluation per position is not sufficiently justified for practical use.
2. **Dependence on Pre-trained Score Models**: The objective functions (e.g., hemolysis, half-life) rely on separately trained predictors (XGBoost, transformers) with modest validation performance (e.g., F1 scores of 0.58–0.71). Limited discussion of their accuracy/robustness raises concerns about real-world applicability.
3. **Ad-hoc Efficiency Heuristics**: The introduced "monotonicity constraint" (accepting only improving updates) lacks theoretical grounding and could bias sampling, yet it is used in all experiments without ablation on its effect on Pareto coverage.

### Novelty & Significance
**Novelty**: This is the first work to integrate rectified discrete flows with multi-objective Pareto optimization, combining ideas from discrete flow matching, Tchebycheff scalarization, and MCMC. The theoretical guarantees for discrete spaces are a clear advance over prior continuous or heuristic methods.
**Significance**: The method addresses a critical gap in biomolecular design, where balancing multiple conflicting properties is essential. The demonstrated ability to optimize 5+ objectives simultaneously with theoretical assurances is impactful for therapeutic discovery and could inspire extensions to other discrete domains (e.g., DNA, antibodies).

### Suggestions for Improvement
1. **Improve Efficiency**: Explore techniques like adaptive candidate pruning, parallel token updates, or learning-based proposals to reduce runtime. A scalability analysis (e.g., sequence length vs. time) would help assess practical limits.
2. **Validate Score Models**: Include external validation of property predictors (e.g., experimental data or hold-out benchmarks) to strengthen confidence in the optimized objectives. Discuss potential biases from model errors.
3. **Justify and Ablate Monotonicity Constraint**: Provide theoretical or empirical analysis of how the constraint affects Pareto convergence and diversity. An ablation without it (even if slower) would clarify its necessity.
4. **Compare to More Recent Discrete Diffusion Methods**: While PepTune is included, comparisons to other discrete diffusion/flow models (e.g., DiGress, discrete CFM) in multi-objective settings would better situate the contribution.
5. **Clarify Limitations**: Explicitly discuss failure cases (e.g., many-objective scaling, noisy objectives) and the impact of finite sampling on theoretical guarantees.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to a simple, strong MCMC baseline using the same prior and reward.** The core algorithm is MCMC with a learned prior and a scalarized reward. To isolate the contribution of the novel components (annealed Tchebycheff scalarization, locally balanced proposals), a direct comparison to a baseline that uses the same ReDi prior `p1(x)` and performs MCMC (e.g., Metropolis-Hastings with a simple proposal) targeting the same tilted distribution `πη,ω(x)` is essential. Without this, it's unclear if the proposed algorithmic machinery is necessary or if the gains come primarily from the high-quality prior.
2. **Validate generated sequences with independent property predictors or experimental data.** All optimization and evaluation rely on the same set of pre-trained property predictors (e.g., for hemolysis, affinity). This creates a high risk of overfitting to the biases of these models. The paper must validate a subset of top-designed sequences using *different*, established predictors or, ideally, wet-lab measurements (e.g., synthesis and testing for a few peptides) to confirm the improvements are real and not artifacts of the reward model.
3. **Ablation on the monotonicity constraint used in all experiments.** The paper introduces a "monotonicity constraint" (accepting only updates that increase the weighted sum) to accelerate convergence and uses it in all reported results. This is a major practical deviation from the theoretical MCMC guarantees. A full ablation must show how performance (Pareto front coverage, final scores, diversity) degrades *without* this constraint, and whether the theoretical convergence properties hold in practice when it is removed.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify Pareto front coverage and diversity, not just average scores.** The claim of "full coverage" is theoretical. Empirically, the paper only reports average property scores across generated batches. To demonstrate multi-objective optimization, they must visualize the approximated Pareto front (e.g., 2D/3D scatter plots of objectives for generated sequences) and compute standard MOO metrics like hypervolume, spread, and spacing. This shows whether the method finds a diverse set of trade-offs or collapses to a narrow region.
2. **Systematically analyze sensitivity to the weight vector `ω` and its sampling.** The theoretical coverage guarantee requires randomizing `ω`. In experiments, fixed weights (often equal) are used. A systematic analysis is needed: sample many different `ω` from the simplex, run AReUReDi for each, and show that the union of solutions covers a broad Pareto front. This directly tests the "full coverage" claim.
3. **Analyze the critical role of the ReDi prior `p1(x)`.** The method's success likely hinges on a high-quality prior. An analysis should show how performance degrades when using a weaker prior (e.g., a less rectified model or a uniform prior) while keeping the MOO algorithm fixed. This determines if the contribution is mainly from the prior or the MOO scheme, and establishes the method's robustness.

### Visualizations & Case Studies
1. **Visualize the trajectory of samples in objective space across iterations.** Plotting how a population of samples moves from the initial prior distribution to the final Pareto front over annealing iterations would vividly demonstrate the exploration-exploitation trade-off and whether the method navigates trade-offs effectively. Current plots only show mean scores over time, which hides the distribution and diversity.
2. **Show case studies of failure modes or limitations.** The paper only presents successful designs. To understand the method's boundaries, analyze cases where it fails to improve certain objectives, produces invalid SMILES (before the validity filter), or generates sequences with unrealistic properties. This would provide insights into practical limitations and areas for improvement.

### Obvious Next Steps
1. **Include a gradient-based continuous relaxation baseline.** A standard approach for discrete MOO is to use a continuous relaxation (e.g., Gumbel-Softmax) of the sequence and apply gradient-based multi-objective optimization (e.g., MGDA, Pareto MTL). Comparing to such a baseline would strengthen the argument for a discrete, MCMC-based method and is a glaring omission.
2. **Benchmark on a standard discrete MOO problem (non-biological).** To prove general algorithmic efficacy beyond the peptide domain, apply AReUReDi to a classic discrete MOO benchmark (e.g., multi-objective knapsack) and compare to state-of-the-art MOO algorithms. This would disentangle the contribution of the algorithm from the domain-specific prior and reward models.

# Final Consolidated Review
## Summary
This paper introduces AReUReDi, a method for multi-objective optimization of discrete biological sequences (e.g., peptides, SMILES). It extends Rectified Discrete Flows (ReDi) by integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates, providing theoretical guarantees of convergence to the Pareto front. The method is demonstrated to optimize up to five competing therapeutic properties simultaneously, outperforming evolutionary and diffusion-based baselines.

## Strengths
- **Theoretical grounding with clear guarantees**: The paper provides proofs (Appendix A) for distributional invariance, convergence to the Pareto front, and full coverage when sampling weight vectors, firmly linking the Tchebycheff scalarization setup to Pareto optimality in discrete spaces.
- **Extensive and convincing empirical validation**: Experiments cover two sequence modalities (wild-type peptides and peptide SMILES), up to five objectives, and multiple protein targets. Comprehensive ablation studies (rectification, annealing, guidance components, weight vectors, and prior importance) systematically justify each design choice and show consistent improvements.
- **Strong benchmarking against relevant baselines**: The method is compared to four classical multi-objective optimization algorithms and a recent diffusion-based approach (PepTune), demonstrating superior trade-off navigation, often by large margins (e.g., orders-of-magnitude improvements in half-life).

## Weaknesses
- **Practical heuristic breaks theoretical guarantees**: The "monotonicity constraint" (accepting only updates that increase the weighted objective sum), used in all experiments to improve efficiency, violates the detailed balance condition of the MCMC kernel. The paper notes its empirical necessity (Table 6) but does not analyze how this modification affects the stationary distribution or the theoretical guarantees of Pareto convergence and coverage. This creates a gap between the presented theory and the practiced algorithm.
- **Heavy dependence on the quality of pretrained property predictors**: The method's output is only as reliable as its objective functions. Notably, the half-life predictor is fine-tuned on only 105 data points (Appendix E.3), raising concerns about overfitting and the real-world validity of the optimized property. The performance of other predictors (e.g., hemolysis F1 of 0.58) is also modest, but their impact is not critically discussed.
- **Lacks standard multi-objective optimization metrics for empirical evaluation**: While the theory guarantees Pareto front coverage, the empirical results primarily report average property scores. The paper does not quantify the diversity or coverage of the approximated Pareto front using standard metrics like hypervolume, spread, or spacing, nor does it provide visualizations of the front (e.g., 2D scatter plots of objectives). This makes it difficult to assess the "full coverage" claim in practice.

## Nice-to-Haves
- **Statistical significance reporting**: Providing standard deviations or confidence intervals for the averaged results (e.g., in Tables 1, 2, 9) would help assess the robustness of the reported improvements.
- **Efficiency improvements**: Exploring techniques like adaptive candidate pruning or parallel coordinate updates could mitigate the method's high computational cost relative to baselines, improving scalability.
- **Comparison to a simple MCMC baseline**: A direct comparison to a baseline that uses the same ReDi prior and performs MCMC targeting the same tilted distribution (but with a simpler proposal) would help isolate the contribution of the novel locally balanced proposals and annealing schedule.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "The paper should compare to gradient-based continuous relaxation baselines (e.g., Gumbel-Softmax with MGDA)."** (Removed as scope creep. The paper's contribution is a discrete method with theoretical guarantees for discrete spaces; demanding a comparison to continuous relaxations is not required for evaluating its stated contributions.)
- **Weakness: "The explanation of conditional total correlation (TC) in Section 2.1 is unclear."** (Removed as a minor clarity nitpick that does not affect the substantive evaluation of the method.)
- **Weakness: "The paper lacks a discussion on scaling to many objectives (e.g., >5)."** (Weakened to a "Nice-to-Have" as the paper successfully demonstrates up to five objectives, which is already ambitious for the domain; scaling further is an extension, not a core flaw.)
- **Weakness: "The annealing schedule is not justified."** (Weakened. The linear schedule is presented by analogy to simulated annealing and is empirically validated in an ablation (Table 10). A deeper theoretical justification would be nice but is not a standard requirement.)

## Novel Insights
The most novel insight emerging from the synthesis of reviews is the critical interplay between the high-quality generative prior (`p1(x)` from ReDi) and the multi-objective MCMC scheme. The ablation studies (Tables 15, 16) strongly suggest that the prior is not merely a convenient starting point but an essential component that anchors the search in biologically plausible regions. This implies that the method's success is not solely due to the novel MCMC mechanics but is a synergistic combination of a powerful learned prior and a theoretically principled steering mechanism. A deeper analysis separating these contributions (e.g., how performance degrades with a progressively weaker prior) would be a valuable direction for understanding the framework's limits.

## Suggestions
- **Explicitly discuss the monotonicity constraint as a limitation**: Add a subsection or paragraph analyzing how the constraint approximates the target distribution, its potential biases, and why it is empirically necessary despite the theoretical compromise. This honesty strengthens the paper.
- **Include a direct evaluation of Pareto front coverage**: For at least one task, visualize the generated set in 2D/3D objective space and report metrics like hypervolume relative to a reference point. This directly supports the empirical claim of navigating trade-offs.
- **Add a brief discussion on the sensitivity to score model accuracy**: Clearly state that the method inherits the biases and inaccuracies of its property predictors, and discuss this as a fundamental limitation for real-world application.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
