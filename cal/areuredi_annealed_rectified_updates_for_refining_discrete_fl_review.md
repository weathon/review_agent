=== CALIBRATION EXAMPLE 50 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy**: The title accurately reflects the method (annealed rectified updates) and the target application (discrete flows with multi-objective guidance).
- **Abstract clarity**: The problem, method components (Tchebycheff scalarization, LBP, annealed MH), and key results (up to 5 objectives, peptide/SMILES benchmarks) are clearly stated.
- **Supported claims**: The claim of "theoretical guarantees of convergence to the Pareto front" is technically an asymptotic result that requires $\eta \to \infty$ and an infinitely long Markov chain (as acknowledged in Section 4). The abstract, however, strongly implies finite-sample or practical convergence, which slightly overclaims. Additionally, "preserving distributional invariance" conflicts with the empirical design choice detailed later (the monotonicity constraint), creating a mismatch between abstract promises and experimental reality.

### Introduction & Motivation
- **Motivation & gap**: Well-motivated. The authors clearly articulate the limitation of existing continuous/discrete MOO methods and position ReDi as a promising but single-objective base. The gap is well-justified.
- **Contributions**: Explicitly listed and mostly accurate. Contribution 2 ("theoretical guarantees... with full coverage") requires heavy caveats regarding the practical sampling budget and the empirical monotonic constraint.
- **Claim calibration**: The introduction appropriately frames the problem but slightly undersells the computational cost of the method relative to evolutionary baselines, which becomes prominent in Section 4.2. The claim of being the "first multi-objective extension of rectified discrete flows" is plausible given the current literature and is acceptable.

### Method / Approach
- **Reproducibility & clarity**: Algorithm 1 and Sections 3.1–3.4 provide a clear, step-by-step description of the sampling loop. The integration of ReDi marginals, Tchebycheff scalarization, and LBP-MH is logically structured.
- **Key assumptions & theoretical gaps**: 
  1. **Marginal vs. Joint Distributions**: The MH proof in Appendix A assumes the proposal $q_i$ and target $\pi_{\eta,\omega}$ operate on the exact distribution. However, the proposal uses *marginal* transition probabilities $p_t^i(\cdot|x_t)$ from the factorized ReDi model. The method inherits the factorization error/conditional total correlation (TC) discussed in Section 2.1. This breaks strict detailed balance and means the "exact distributional invariance" claim (Theorem, Sec A.2) holds only if $p_t^i$ perfectly captures the true conditional, which ReDi explicitly approximates.
  2. **Monotonicity Constraint Violation**: Section 4 states: *"we introduce a monotonicity constraint that accepts only token updates that increase the weighted sum... Therefore, this monotonicity constraint was involved in all the following experiments."* This directly replaces the MH acceptance probability $\alpha_i$ with a deterministic/greedy filter, **destroying detailed balance and invalidating the theoretical guarantees** (invariance and convergence to $\pi_{\eta,\omega}$). The method used empirically is no longer an MCMC sampler but a biased, annealed greedy search.
  3. **Annealing Schedule**: The linear schedule $\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \frac{t}{T-1}$ is heuristically justified. Classical simulated annealing convergence proofs require logarithmic cooling. The paper does not discuss how this finite linear schedule affects the quality of the approximate Pareto coverage.
- **Edge cases/Failure modes**: Not discussed. What happens when objectives are perfectly aligned or when $p_1(x)$ assigns near-zero probability to high-reward regions? The method relies entirely on the ReDi prior being diverse enough; if $p_1$ is mode-collapsed, the MH/LBP step cannot recover missing Pareto points.

### Experiments & Results
- **Claims vs. Tests**: Experiments demonstrate that the method improves properties over baselines. However, they do not rigorously test *Pareto front coverage* or *dominance*. Reporting averages of 100 samples masks the actual trade-off surface.
- **Baselines & Fairness**: Baselines (NSGA-III, SMS-EMOA, PepTune) are appropriate. The matched wall-clock comparison in Table 11 is excellent practice and strengthens the empirical claim. However, PepTune and the evolutionary baselines run orders of magnitude faster (Table 2).
- **Missing MOO Metrics**: For a paper claiming Pareto convergence and front coverage, standard multi-objective metrics (e.g., Hypervolume, Inverted Generational Distance, Spacing) are conspicuously absent. Average property scores do not capture Pareto dominance or diversity.
- **Statistical Rigor**: Tables report only means over 100 runs. No standard deviations, confidence intervals, or statistical significance tests are provided. Given the stochastic nature of MCMC and the reliance on surrogate predictors, error estimates are necessary.
- **Score Model Reliability**: The half-life objective heavily influences results (up to 60+ hours in tables), yet Section E.3 reveals the half-life score model was fine-tuned on only **105 entries**. This is a severe data limitation. Optimizing against such an underpowered surrogate risks severe reward hacking or spurious correlations, yet no uncertainty quantification or ensemble scoring is used to mitigate this.
- **Ablations**: Section 4.4 and Tables 9/10/12 properly isolate the effects of rectification, annealing, and step count. Table 6 confirms that removing the monotonicity constraint drastically degrades empirical performance, reinforcing the theory-vs-practice disconnect.

### Writing & Clarity
- **Clarity of contribution**: The methodological pipeline is generally well-explained. The transition from theoretical MCMC (Sec 3) to constrained optimization (Sec 4) is abrupt and poorly reconciled. Readers expecting a principled sampler will be confused by the sudden pivot to greedy acceptance.
- **Figures & Tables**: Figures 1 and 2 effectively visualize structural plausibility and trajectory. Tables are dense but informative. The lack of error bars/standard deviations across experimental tables is a clarity issue for assessing robustness.
- **Minor structural note**: The proof sketch in Appendix A is concise but assumes familiarity with LBP literature. The step from single-coordinate updates to random-scan mixture is correctly stated but glosses over the impact of time-varying $\eta_t$ on stationarity (strictly, a non-stationary Markov chain does not preserve a fixed invariant distribution at intermediate steps).

### Limitations & Broader Impact
- **Acknowledged limitations**: The authors note asymptotic guarantees, computational cost, and future extensions to uncertainty-aware guidance.
- **Omitted limitations**: 
  1. **Theoretical/Practical Mismatch**: The monotonicity constraint's destruction of detailed balance and Pareto convergence guarantees is not discussed as a limitation.
  2. **Surrogate Model Uncertainty**: The optimization heavily depends on score predictors (especially half-life) without propagating predictive uncertainty.
  3. **Validity Constraints in SMILES**: Section F mentions rejecting invalid transitions, but the acceptance rate and impact of these hard constraints on the effective proposal distribution are not quantified.
- **Societal Impact**: Adequately addressed. The focus on therapeutic safety properties and research-only licensing is appropriate and well-reasoned.

### Overall Assessment
AReUReDi addresses a meaningful problem in computational biology by extending rectified discrete flows to multi-objective sequence design. The integration of Tchebycheff scalarization and locally balanced proposals is conceptually elegant, and the empirical benchmarks across multiple protein targets and SMILES design are extensive. However, the paper faces significant hurdles for ICLR acceptance. First, there is a fundamental disconnect between the theoretical guarantees and the empirical implementation: the monotonicity constraint used in all experiments explicitly violates the Metropolis-Hastings acceptance criterion, breaking detailed balance and nullifying the claimed invariance and convergence proofs. Second, the use of marginal transition probabilities from a factorized ReDi model introduces approximation error that the exact balance proofs do not account for. Third, the empirical evaluation lacks standard multi-objective metrics (e.g., hypervolume) and relies on a severely underpowered surrogate model for half-life (105 training samples), raising robustness concerns. Finally, results report only averages without statistical significance measures. The contribution stands as a promising heuristic optimizer for discrete biomolecular design, but to meet ICLR's standards for theoretical rigor and empirical completeness, the authors must either (a) align the theory with the constrained sampling regime (e.g., analyzing non-stationary/greedy dynamics), or (b) demonstrate that the principled MH sampler achieves competitive results without the monotonicity constraint, supported by proper Pareto front metrics and uncertainty-aware scoring.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces AReUReDi, a discrete multi-objective optimization framework that extends Rectified Discrete Flows (ReDi) by integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings (MH) updates to guide sampling toward the Pareto front. It provides theoretical proofs of distributional invariance and Pareto-front convergence, and demonstrates empirically that the method outperforms evolutionary algorithms and diffusion-based baselines in generating peptide and peptide-SMILES binders optimized across up to five conflicting therapeutic properties.

### Strengths
1. **Strong Theoretical Grounding:** The appendix provides clear, step-by-step proofs that the locally balanced proposal with MH updates preserves the target distribution $\pi_{\eta,\omega}(x) \propto p_1(x)\exp(\eta S_\omega(x))$ and concentrates on the Pareto front as $\eta \to \infty$ (Appendix A). This mathematical rigor aligns well with ICLR's emphasis on principled generative methods.
2. **Well-Executed Empirical Validation:** Experiments cover diverse biological targets (structured, unstructured, and chemically-modified SMILES) with rigorous ablations on rectification rounds, annealing schedules, weight vectors, and prior knowledge (Tables 6–10, Appendix G). The matched wall-clock comparison against PepTune (Table 11) demonstrates careful experimental design to address compute disparities.
3. **Clear Integration of Discrete Flow Mechanics:** The paper effectively leverages ReDi's reduced conditional total correlation to provide high-quality proposal distributions, then layers multi-objective guidance without distorting discrete token structure, addressing a noted limitation of continuous-space flow matching for sequences (Section 1 & 3.3).
4. **Strong Reproducibility Commitment:** Detailed training configurations, dataset curation pipelines, score model architectures, and hyperparameter sweeps are provided (Appendices B–F). The explicit statement of public data usage and promised code release meets ICLR's reproducibility standards.

### Weaknesses
1. **Theory-Practice Contradiction via Monotonicity Constraint:** Section 4 explicitly states that a monotonicity constraint (accepting only token updates that improve the weighted objective sum) is applied in all experiments for efficiency. This greedy truncation breaks the detailed balance condition required for the stated invariance guarantee (Appendix A.2) and alters the stationary distribution. The paper claims it "does not alter the underlying optimization objectives," but this requires formal justification or a revised theoretical bound for the truncated chain.
2. **Limited Baseline Comparisons:** While classical MOO algorithms and PepTune are evaluated, the paper omits comparisons to other contemporary guided discrete sampling methods (e.g., classifier guidance for discrete flows, reward-finetuned language models, or alternative MCMC-sampling techniques). This makes it difficult to isolate whether gains stem from the rectified prior, the MH guidance mechanism, or both.
3. **Surrogate Model Data Limitations:** Key objectives like half-life rely on extremely small datasets (105 entries, Appendix E.3), and all objectives use pre-trained predictors with moderate validation scores (Spearman ~0.64–0.86). The guidance is highly sensitive to surrogate misspecification, yet no uncertainty-aware sampling or robustness analysis is provided.
4. **Computational Scalability Concerns:** Table 2 shows AReUReDi requires 20–40× longer runtimes than Pareto evolutionary baselines. Candidate evaluation scales with $O(LK)$ per step, and while top-$p$ pruning is mentioned, the practical throughput limitation for high-throughput drug discovery screens is not thoroughly analyzed beyond the matched compute ablation.

### Novelty & Significance
The novelty lies in the principled unification of rectified discrete flows with MCMC-based multi-objective guidance, providing the first discrete flow framework with formal Pareto coverage guarantees. While components like Tchebycheff scalarization, locally balanced proposals, and MH updates are individually established, their adaptation to categorical sequence spaces with provable convergence represents a meaningful methodological advance. The significance is high for both generative modeling and computational biology, as it directly addresses the critical need for tractable, theoretically sound multi-objective generation in discrete domains. For ICLR, the theoretical-conceptual contribution meets the bar, provided the empirical-practical mismatch is clarified and contextualized.

### Suggestions for Improvement
1. **Address the Monotonicity Constraint Theoretically:** Either remove the constraint in at least a subset of experiments to validate the true MCMC stationary distribution, or formally analyze the truncated chain's bias. Provide bounds on how monotonic acceptance shifts samples relative to $\pi_{\eta,\omega}$ and justify the trade-off explicitly.
2. **Expand Baseline Comparisons:** Include recent discrete guided generation methods such as classifier-guided discrete diffusion/flows, reward-model fine-tuning (e.g., DPO/RLOO for sequences), or alternative guided MCMC samplers. This will better contextualize AReUReDi's specific advantages.
3. **Incorporate Surrogate Uncertainty:** Given the reliance on pre-trained scalar predictors with modest validation metrics, implement or analyze uncertainty-aware guidance (e.g., Thompson sampling over objectives, or penalizing high-variance predictions) to ensure robustness against model misspecification.
4. **Clarify Annealing & Complexity Details:** Resolve the apparent discrepancy between the annealing schedule in Section 3.2 and Algorithm 1. Additionally, provide a breakdown of memory/time scaling with sequence length $L$ and vocabulary size $K$, and discuss how top-$k$/$p$ pruning theoretically affects transition matrix reversibility in practice.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Hypervolume & Spread Metrics:** Replace average per-objective scores with standard MOO metrics (e.g., Hypervolume, IG-Distance, spread) to prove the sampler actually approximates a diverse, non-dominated Pareto set rather than collapsing to a single high-scoring cluster.
2. **Robustness to Surrogate Noise:** Inject controlled noise or use ensemble variance into the objective evaluations during sampling to demonstrate AReUReDi doesn't overfit the low-accuracy score models (F1 ≤ 0.71, half-life n=105).
3. **Strong Discrete Guidance Baselines:** Add classifier-guided discrete diffusion, GFlowNets, or reward-model fine-tuning as baselines to isolate whether performance gains stem from the ReDi prior rather than the MCMC guidance mechanism.
4. **True Budget-Equalized Scaling:** The top-2 vs top-2 time-matched table is methodologically weak for high-D optimization; run throughput-vs-Pareto-quality curves across multiple compute budgets to show whether AReUReDi is genuinely more sample-efficient.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical Breach from Monotonicity Constraint:** Enforcing monotonic acceptance explicitly breaks detailed balance and the claimed Pareto convergence guarantees; quantify the resulting bias by comparing unconstrained MCMC (long run) vs constrained sampling to prove trade-off coverage is preserved.
2. **Objective Correlation vs. True Conflict:** High simultaneous scores across all five objectives suggest weak trade-offs in the surrogate models; provide the empirical objective correlation matrix and discuss whether the method is optimizing biologically meaningful conflicts or model artifacts.
3. **Initialization Dependence:** Algorithm 1 specifies uniform initialization, but Appendix F uses prior-sampled starts for SMILES; ablate uniform vs. model vs. diverse random seeds to prove convergence isn't just sampling high-density prior regions.

### Visualizations & Case Studies
1. **Explicit Pareto Front Plots:** Show 2D/3D scatter plots of objective pairs for all 100 samples with the theoretical/baseline Pareto front overlaid; this directly exposes whether the method spreads coverage or collapses to a dense, dominated region.
2. **Token-Level Trade-Off Trajectories:** Visualize step-by-step sequence edits for representative runs, mapping which positions change to improve one objective while degrading another, proving the MCMC dynamics actively navigate conflicts instead of finding globally easy sequences.
3. **Failure Mode Documentation:** Include a case where the sampler stalls or overexploits a noisy surrogate region, demonstrating realistic limitations of finite-step MCMC in discrete spaces and preventing overclaiming of robustness.

### Obvious Next Steps
1. **Uncertainty-Aware Proposal Ratio:** Integrate prediction variance from the score models directly into the proposal weighting to penalize low-confidence guidance regions, a standard requirement for reliable black-box MOO.
2. **Mixing Time & Finite-Step Bounds:** Derive or empirically estimate the mixing time and step complexity as a function of sequence length and objective count, explicitly bounding the degradation from the claimed infinite-chain guarantees to practical finite steps.
3. **Downstream Wet-Lab Validation Protocol:** Outline a prioritized experimental pipeline (e.g., in vitro affinity, hemolysis, solubility assays for top-5 predictions) to ground the in silico claims in tangible biological relevance.

# Final Consolidated Review
## Summary
AReUReDi extends rectified discrete flows to multi-objective biomolecular sequence design by integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings (MH) updates. The authors provide asymptotic proofs of distributional invariance and Pareto-front convergence, and demonstrate empirical improvements over evolutionary and diffusion baselines when generating wild-type peptides and peptide-SMILES across up to five therapeutic objectives.

## Strengths
- **Principled Methodological Integration:** The method coherently bridges factorized discrete flow matching with MCMC-based guidance, avoiding continuous-space relaxations that distort token-level structure. Locally balanced proposals effectively blend the generative prior with multi-objective guidance while maintaining reversibility under the assumed target distribution.
- **Rigorous Empirical Protocol & Reproducibility:** The evaluation spans diverse targets (structured, intrinsically disordered, SMILES) and includes comprehensive ablations on rectification depth, annealing schedules, weight vectors, and prior initialization. The matched wall-clock comparison against PepTune (Table 11) directly addresses typical compute-disparity critiques in generative multi-objective optimization.
- **Clear Theoretical Grounding:** The appendix provides explicit, step-by-step proofs for invariance and asymptotic concentration on the Pareto front under Tchebycheff scalarization. The formalization of Pareto coverage through weight-vector randomization (Appendix A) is mathematically sound for finite, discrete spaces.

## Weaknesses
- **Fundamental Theory-Practice Disconnect via Monotonicity Constraint:** Section 4 explicitly states that *all* reported experiments replace the MH acceptance probability with a monotonicity constraint that only accepts token updates increasing the weighted objective sum. This deterministic/greedy filter explicitly violates detailed balance, alters the stationary distribution, and nullifies the claimed invariance and convergence guarantees in Appendix A. The paper effectively evaluates a biased, annealed hill-climbing heuristic while retaining proofs for a principled MCMC sampler it does not actually deploy.
- **Absence of Standard Multi-Objective Evaluation Metrics:** Despite centering the paper on Pareto-front convergence and coverage, the empirical evaluation relies exclusively on univariate averages across 100 samples. Standard MOO metrics (Hypervolume, Inverted Generational Distance, Spacing) are entirely missing. Without them, it is impossible to verify whether AReUReDi genuinely approximates a diverse, non-dominated Pareto set or merely collapses to a single high-scoring cluster with marginal improvements.
- **Over-Reliance on Low-Fidelity/Underpowered Surrogate Objectives:** The optimization heavily depends on pre-trained scalar predictors with moderate validation performance and severely constrained training data (e.g., half-life trained on only 105 entries, Appendix E.3; classifiers with F1 scores of 0.58–0.71). Guiding a generative sampler against such noisy, underpowered surrogates without predictive uncertainty quantification poses a high risk of reward hacking and spurious optimization, yet no ensemble scoring or variance-aware guidance is implemented or discussed.
- **Limited Baseline Scope for Discrete Guidance:** While classical MOO algorithms and PepTune are benchmarked, the paper omits comparisons to highly relevant contemporary discrete guidance frameworks, such as Multi-Objective GFlowNets, classifier-guided discrete diffusion, or preference-optimized sequence models (DPO/RLOO). This makes it difficult to isolate whether performance gains stem from the MH guidance mechanism or simply from leveraging the high-quality ReDi prior.

## Nice-to-Haves
- Quantify the exact bias introduced by the monotonicity constraint by comparing coverage metrics against true unconstrained MH sampling.
- Analyze the empirical correlation matrix between the five objectives to clarify whether they represent genuine biological conflicts or surrogate model artifacts.
- Provide finite-step mixing time estimates or bounds to contextualize the $\eta \to \infty$ theoretical guarantees.

## Novel Insights
The paper's strongest conceptual contribution is reframing discrete multi-objective sequence design as a target distribution reweighting problem within a rectified flow framework, rather than relying on heuristic score-conditioning or continuous latent projections. By exploiting ReDi's reduced conditional total correlation to define a locally balanced proposal, the method elegantly sidesteps the approximation errors typical of factorized discrete flows. However, the empirical reliance on greedy monotonic acceptance inadvertently highlights a pervasive bottleneck in the field: theoretically exact MCMC samplers are often too slow or mix too poorly for high-dimensional discrete design, forcing practitioners to abandon rigorous guarantees in favor of heuristic efficiency. Resolving this tension, rather than papering over it with a theoretical framework that is deliberately disabled during evaluation, remains a critical open problem.

## Potentially Missed Related Work
- **Multi-Objective GFlowNets** (Jain et al., ICML 2023) — Directly comparable as a generative alternative for sampling Pareto fronts without requiring MCMC convergence, offering a contrasting approach to sequential decision-making in discrete spaces.
- **Discrete Classifier/Flow Guidance** (Nisonoff et al., ICLR 2025; Tang et al., 2025a) — Relevant for benchmarking direct guidance injection versus the paper's two-stage proposal-and-accept filtering pipeline.
- **Training-Free/Preference-Guided Diffusion** (Sun et al., ICLR 2024; Annadani et al., arXiv 2025) — Highly relevant for comparing offline objective conditioning paradigms.

## Suggestions
- **Align Theory with Practice or Provide Bias Bounds:** Either deploy the standard MH acceptance rule in a primary experiment to validate the true MCMC sampler's empirical performance, or formally derive how the monotonicity constraint shifts the stationary distribution relative to $\pi_{\eta,\omega}$. The claimed guarantees cannot stand unaddressed while the mechanism is disabled.
- **Adopt Standard Pareto Metrics:** Replace or supplement univariate means with Hypervolume, IGD, and spacing metrics. Report standard deviations or confidence intervals across multiple random seeds to rigorously demonstrate Pareto coverage and algorithmic stability.
- **Integrate Surrogate Uncertainty:** Implement variance-aware sampling (e.g., penalizing proposals with high predictor uncertainty, or using ensemble disagreement) to prevent over-optimization against low-data objectives like the half-life model.
- **Expand Generative Baselines:** Include GFlowNets or reward-fine-tuned discrete language models as baselines to clearly position AReUReDi's trade-offs in sample efficiency, theoretical guarantees, and practical throughput.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
