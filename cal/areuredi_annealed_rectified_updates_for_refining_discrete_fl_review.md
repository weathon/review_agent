=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title is descriptive and the acronym is clearly defined. The abstract succinctly states the problem, the method's key components (annealed Tchebycheff scalarization, locally balanced proposals, Metropolis-Hastings), and claims theoretical guarantees and empirical superiority over baselines. These claims are supported in the body, though the empirical support requires careful scrutiny (see Experiments).

**Introduction & Motivation**
The introduction effectively motivates the need for multi-objective discrete sequence design, particularly in biomolecular engineering. It clearly situates the work relative to single-objective methods, black-box MOO, continuous Pareto methods, and discrete flows (especially ReDi). The gap—the lack of a principled multi-objective discrete flow framework—is well-articulated. The four claimed contributions are specific and accurately reflect the paper’s content.

**Method / Approach**
The method is clearly described, building on ReDi to incorporate multi-objective guidance. The integration of Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings is novel for discrete flows. Theoretical guarantees (invariance, convergence to the Pareto front, coverage) are provided in Appendix A and appear sound, though they rely on asymptotic limits (infinite chain, η→∞). A significant practical deviation is the introduction of a monotonicity constraint (Section 4, Table 6) to accelerate convergence. While the authors state it does not alter the optimization objectives, this heuristic may bias sampling and its impact on the theoretical guarantees is not analyzed. The annealing schedule and balancing functions are standard choices. Overall, the method is principled and reproducible, but the ad hoc monotonicity constraint warrants justification or analysis.

**Experiments & Results**
Experiments are extensive, covering wild-type peptides (5 objectives) and peptide SMILES (4 objectives) across multiple targets. Comparisons with evolutionary algorithms (NSGA-III, SMS-EMOA, SPEA2, MOPSO) and a diffusion-based method (PepTune) are appropriate. The results consistently show AReUReDi achieving better trade-offs. However, several issues merit attention:
- **Statistical reporting**: Results are presented as averages over 100 samples without measures of variance (e.g., standard deviation) or statistical significance tests. This limits interpretability.
- **Diversity and Pareto front coverage**: For peptides, quantitative diversity metrics are not reported, and the empirical coverage of the Pareto front is not visualized (e.g., via scatter plots of objectives). While weight-vector ablations (Tables 13,14) show trade-off steering, a more direct analysis of front coverage is needed.
- **Score model reliability**: The property predictors have moderate performance (e.g., affinity Spearman 0.64, classification F1 0.58–0.71). This is acceptable for in silico validation but should be noted as a limitation.
- **Computational cost**: AReUReDi is slower than baselines, but a matched-time comparison (Table 11) still favors it. The ablation on step count (Table 12) appropriately explores the trade-off.
- **Ablations**: Studies on rectification, annealing, weight vectors, and priors are thorough and support design choices.

**Writing & Clarity**
The paper is well-structured and clearly written. The method and experiments are described in sufficient detail. Minor formatting artifacts from PDF extraction do not impede understanding.

**Limitations & Broader Impact**
Limitations are partially acknowledged (asymptotic guarantees, computational cost, reliance on pre-trained score models) but not gathered in a dedicated section. A discussion of the monotonicity constraint’s impact on theoretical guarantees is missing. The broader impact statement appropriately addresses potential misuse (harmful molecule design) and commits to releasing code under a research-only license.

### Overall Assessment
AReUReDi is a novel and theoretically grounded framework for multi-objective discrete sequence design, addressing an important gap in biomolecular engineering. The integration of rectified discrete flows with annealed Tchebycheff scalarization and MCMC is clever, and the provided theoretical guarantees are a strength. Empirically, the method demonstrates superior trade-off navigation compared to several baselines across two challenging domains. However, the paper would be strengthened by: (1) a deeper analysis of the practical monotonicity constraint relative to the theory, (2) reporting of statistical variance and diversity metrics for generated peptides, and (3) clearer visualization of Pareto front coverage. These concerns do not undermine the core contribution, which is significant and likely meets ICLR’s standards for acceptance, provided the authors address these points in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces AReUReDi, a multi-objective optimization framework for discrete sequence generation that extends Rectified Discrete Flows (ReDi). The method combines annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates to bias sampling toward Pareto-optimal states with theoretical guarantees. It is applied to the design of therapeutic peptides (both wild-type amino acid sequences and chemically-modified SMILES), simultaneously optimizing up to five competing biological properties such as affinity, solubility, and half-life.

### Strengths
1. **Theoretical Rigor**: The paper provides clear proofs of convergence to the Pareto front and distributional invariance under the proposed Markov chain (Section A). This solid theoretical foundation strengthens the methodological contribution.
2. **Comprehensive Evaluation**: Extensive experiments on both wild-type peptide and SMILES sequence design demonstrate superior performance against multiple baselines (NSGA-III, SMS-EMOA, PepTune, etc.) across up to five objectives. Ablation studies systematically validate key components like rectification, annealing, and the prior (Tables 9, 10, 15, 16).
3. **Integration of Advanced Techniques**: The method thoughtfully combines several recent ideas—rectified discrete flows, Tchebycheff scalarization, locally balanced MCMC—into a novel framework tailored for discrete multi-objective optimization, a setting not well addressed by prior work.

### Weaknesses
1. **High Computational Cost**: AReUReDi requires significantly more time per sample than baselines (e.g., 55s vs. 2.46s for PepTune on 1B8Q, Table 2). While wall-clock-budget comparisons are provided, the scalability to very long sequences or many-objective problems is not thoroughly addressed.
2. **Reliance on Heuristic Constraint**: The “monotonicity constraint” (accepting only token updates that increase the weighted sum of objectives) is introduced to improve sampling efficiency but is not part of the theoretical framework. Its effect on convergence guarantees and Pareto coverage is not analyzed (Section 4, Table 6).
3. **Limited Validation Beyond Predictions**: All evaluated properties are based on *in-silico* score models (e.g., XGBoost classifiers, affinity predictors). There is no experimental wet-lab validation or demonstration that the generated sequences indeed possess the optimized properties in reality, which is crucial for therapeutic design.

### Novelty & Significance
**Novelty**: This appears to be the first work to extend rectified discrete flows to multi-objective optimization with theoretical guarantees. The integration of annealed Tchebycheff scalarization with locally balanced MCMC for discrete sequence spaces is novel.
**Significance**: The framework addresses a critical gap in biomolecular design, where balancing multiple conflicting properties is essential. The strong empirical results and theoretical grounding make it a valuable contribution to the fields of discrete generative models and computational biology. However, the significance would be amplified by real-world validation.

### Suggestions for Improvement
1. **Analyze the Monotonicity Constraint**: Provide a theoretical or empirical analysis of how the monotonicity constraint impacts the convergence guarantees and Pareto coverage. If it is essential for practical performance, justify its use more formally or incorporate it into the theoretical framework.
2. **Address Computational Efficiency**: Discuss strategies to reduce the per-sample cost (e.g., adaptive pruning, parallelization) or more clearly characterize the trade-off between sample quality and compute. A runtime comparison with more recent discrete diffusion models would be informative.
3. **Strengthen the Positioning and Limitations**: The related work section could better differentiate AReUReDi from prior multi-objective Bayesian optimization and continuous-space flow matching methods. A dedicated limitations paragraph should explicitly discuss the dependency on pre-trained score models (and their potential biases), the computational cost, and the lack of experimental validation.
4. **Clarify Hyperparameter Sensitivity**: While ablation studies are present, a discussion on the sensitivity of performance to key hyperparameters (e.g., the annealing schedule \( \eta_t \), the balancing function \( g \), the weight vector \( \omega \)) would help users apply the method. The choice of a fixed weight vector in most experiments could be justified or contrasted with adaptive strategies.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validate convergence to the Pareto front empirically.** The paper claims theoretical guarantees of convergence to the Pareto front, but only shows average property improvements. A critical experiment is missing: for a given set of weights, run the chain to approximate its stationary distribution and plot the achieved objective values against the true/exact Pareto front (e.g., found by exhaustive search for a small sequence space). Without this, the core theoretical claim is not empirically substantiated.
2. **Compare against a strong, simple MCMC baseline.** The method is essentially a guided MCMC. A key missing baseline is a standard Metropolis-Hastings sampler using the same proposal mechanism (e.g., single-token changes) but with a naive objective combination (e.g., weighted sum) and simulated annealing. This is needed to isolate the contribution of the "rectified flow prior" and "locally balanced proposals" beyond a well-tuned traditional MCMC.
3. **Include GFlowNet baselines for multi-objective discrete generation.** For discrete sequence generation with multiple rewards, GFlowNets are a major recent line of work (e.g., Multi-Objective GFlowNets). The paper only compares to evolutionary algorithms and one diffusion method (PepTune). Omitting this strong and relevant baseline significantly weakens the claim of state-of-the-art performance.
4. **Ablation on the monotonicity constraint.** The paper reveals all results use an extra "monotonicity constraint" that accepts only improving updates. This is a major algorithmic modification not part of the core theoretical AReUReDi. An ablation must show performance *without* this constraint to demonstrate the actual utility of the proposed annealing/MH mechanism. Its use suggests the base method may be inefficient.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze diversity versus quality of the generated Pareto set.** The paper reports average scores and claims "full coverage." It must analyze the diversity of solutions generated for a *single* weight vector (spread along the front) and across different weight vectors (coverage of the front). Metrics like hypervolume or spacing of the objective vectors are essential to evaluate multi-objective optimization performance.
2. **Analyze the sensitivity and calibration of the property predictors.** The entire optimization relies on black-box property predictors (e.g., for hemolysis, half-life). Their accuracy, uncertainty, and potential for adversarial optimization (e.g., exploiting predictor biases) must be discussed. Showing correlation between different predictors for the same property or a small hold-out experimental validation set is critical for trusting the *in silico* results.
3. **Quantify the impact of the rectified flow prior.** The ablation (Table 9) compares different base *models* (PepDFM, PepReDi). It does not isolate the effect of using *any* learned prior vs. a uniform prior within the **same** AReUReDi framework. A proper ablation should show: AReUReDi(Uniform prior) vs. AReUReDi(ReDi prior). The results in Appendix G (Tables 15,16) are insufficiently highlighted and analyzed in the main text regarding this core component.

### Visualizations & Case Studies
1. **Visualize the Pareto front.** For a 2- or 3-objective task, plot the true/exact Pareto front (if computable) or a high-quality approximation found by a brute-force method, and overlay the solutions found by AReUReDi and all baselines. This is the most direct way to show superiority in multi-objective optimization (covering the front, approaching optimality).
2. **Show case studies of failure modes.** The paper only shows successful designs. To understand limitations, visualize examples where optimization fails—e.g., sequences where one objective plateaus or degrades despite annealing, or where predicted scores are high but structural validation (e.g., AlphaFold) reveals implausible binding modes. This builds credibility.

### Obvious Next Steps
1. **Perform wet-lab validation on top designs.** For a key claim of "therapeutic molecule design," *in silico* scores are not enough. The obvious next step is to synthesize and test a handful of top-designed peptides for key properties (e.g., binding affinity, solubility) to provide proof-of-concept experimental validation. Without this, the practical impact is speculative.
2. **Benchmark on a public, standardized multi-objective molecular benchmark.** The paper creates its own benchmarks due to a claimed lack of public ones. However, for broader impact and reproducibility, the method should be tested on an existing public benchmark for multi-objective molecular optimization (e.g., GuacaMol multi-objective benchmarks or related tasks) to allow direct comparison with future work.
3. **Clarify the relationship between "rectification" and multi-objective guidance.** The title and framing emphasize "Annealed Rectified Updates," but the connection is weak. The rectification improves the base generative model (ReDi), but the multi-objective guidance is an MCMC sampler that uses this model as a *prior*. A clear ablation showing that a high-quality prior (from rectification) is necessary for the MCMC to find good solutions is needed in the main text, not just the appendix.

# Final Consolidated Review
## Summary
AReUReDi extends Rectified Discrete Flows (ReDi) to multi-objective optimization for discrete sequence design. It combines annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates to bias sampling toward Pareto-optimal states with theoretical guarantees of convergence and coverage. The method is demonstrated on therapeutic peptide and SMILES sequence design, simultaneously optimizing up to five competing biological properties and outperforming evolutionary and diffusion-based baselines.

## Strengths
- **Theoretical grounding with guarantees.** The paper provides proofs that the Markov chain preserves the tilted distribution and converges to the Pareto front with full coverage in the asymptotic limit (Appendix A). This formal foundation distinguishes the work from purely empirical approaches.
- **Comprehensive empirical validation across domains.** The method is evaluated on two distinct tasks—wild-type peptide and chemically-modified peptide SMILES design—optimizing up to five properties across many protein targets. It consistently outperforms multiple classical MOO algorithms and a recent diffusion-based baseline (PepTune) in achieved trade-offs (Tables 1, 2).
- **Systematic ablation studies supporting design choices.** Ablations confirm the benefits of rectification over a standard discrete flow prior (Table 9), annealed versus fixed guidance strength (Table 10), and the importance of the learned prior over a uniform one (Tables 15, 16). These experiments convincingly justify the core algorithmic components.

## Weaknesses
- **High computational cost per sample.** AReUReDi requires significantly more time (e.g., 55s vs. 2.46s for PepTune on one task, Table 2) due to its iterative MCMC updates. While a matched-time comparison still favors AReUReDi (Table 11), the cost may limit scalability to very long sequences or many-objective problems.
- **Heuristic monotonicity constraint lacks theoretical integration.** To improve sampling efficiency, all experiments use a constraint that accepts only token updates increasing the weighted sum of objectives (Section 4, Table 6). This practical modification is not part of the theoretical framework, and its impact on convergence guarantees and Pareto coverage is not analyzed.
- **Missing comparison to a relevant strong baseline: Multi-Objective GFlowNets.** For discrete sequence generation with multiple rewards, GFlowNets are a prominent recent approach. The absence of this comparison weakens the claim of state-of-the-art performance.
- **Statistical reporting lacks variance measures.** Results are presented as averages over 100 runs without standard deviations or confidence intervals. This makes it difficult to assess the robustness of the improvements and the significance of differences between methods.

## Nice-to-Haves
- Visualizing the approximate Pareto front (e.g., for 2-3 objectives) to directly illustrate coverage and trade-off navigation.
- A more explicit analysis of solution diversity within and across weight vectors for the peptide tasks.
- A brief discussion on the sensitivity of performance to key hyperparameters like the annealing schedule or balancing function choice.

## Novel Insights
The paper's core novel insight is the principled integration of rectified discrete flows—which provide a high-quality, low-correlation generative prior—with a theoretically sound MCMC mechanism for multi-objective steering. This combination ensures that sampling is anchored in realistic regions of sequence space while being provably guided toward Pareto-optimal trade-offs. The application demonstrates that this discrete, sequence-native framework can effectively navigate complex, conflicting biological objectives—a setting where continuous or latent-space methods face representation challenges.

## Suggestions
- Provide a theoretical or empirical analysis of how the monotonicity constraint affects the Markov chain's stationary distribution and convergence properties.
- Include measures of variance (e.g., standard deviation) alongside average scores in result tables to allow assessment of statistical significance.
- In future work or a revision, consider adding a comparison to Multi-Objective GFlowNets as a relevant baseline for discrete multi-reward generation.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
