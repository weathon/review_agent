=== CALIBRATION EXAMPLE 93 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** The title accurately reflects the paper's core contribution: a benchmark pairing real-world measurements with simulated data for complex physical systems.
- **Abstract clarity:** Clearly states the problem (lack of real-world data limiting scientific ML evaluation and sim-to-real research), method (5 datasets, 3 tasks, 9 metrics, 10 baselines), and key results (significant sim-to-real gap; sim pretraining consistently improves real-world accuracy and convergence).
- **Supported claims:** The claim of being the "first benchmark... that integrates real-world measurements with paired numerical simulations" is slightly overstrong given the explicit mention of the concurrent REALM benchmark (Mao et al., 2025) in Section 2. I recommend softening to "one of the first" or explicitly contrasting the scale/pairing protocol against REALM in the abstract to avoid reviewer pushback on novelty. Otherwise, all abstract claims are substantiated in the main text.

### Introduction & Motivation
- **Problem motivation & gap:** The motivation is well-grounded. The authors correctly identify that scientific ML models are predominantly validated on simulated data, which introduces a validation gap due to numerical discretization errors, simplified physics, and idealized boundary conditions. The gap in prior work (PDEBench, The Well, etc., being sim-only or lacking paired experimental data) is clearly articulated.
- **Contributions:** Explicitly stated and accurately map to the paper's sections (Data, Tasks, Metrics, Baselines, Codebase).
- **Over/Under-claiming:** The introduction frames the benchmark's necessity well. However, it slightly underplays a known methodological challenge in experimental CFD: exact spatiotemporal alignment between numerical and physical initial conditions (ICs). While mentioned later implicitly via "measurement noise" and "numerical errors," the IC alignment challenge fundamentally impacts trajectory forecasting metrics and deserves earlier acknowledgment in the introduction.

### Method / Approach
- **Clarity & Reproducibility:** The methodology is largely clear. The three task settings (real-world train, sim train, sim pretrain + real finetune) are well-defined. Data formats, PIV/combustion measurement setups, and simulation solvers are detailed in Sections 3.2, B.2, and B.3, supporting reproducibility.
- **Key assumptions & justification:** 
    - *Dataset size confounder:* Section 3.1 states that real-world training uses `n` samples, while simulated training uses all `N` samples. This intentionally reflects practical scarcity, but it conflates *data source* (real vs. sim) with *dataset size* when comparing their respective RMSEs. An explicit discussion or a size-matched ablation is needed to isolate whether performance gains stem from modality/noise differences or simply from training on more trajectories.
    - *Modality bridging:* The mask-training strategy for unobserved modalities and the surrogate model for Combustion (Appendix D.2) are sensible, but their introduced biases are not quantified.
- **Logical gaps / Edge cases:** 
    - *Temporal Alignment:* The paper does not specify how real-world and simulated trajectories are temporally synchronized. Experimental transients and numerical initialization rarely start at the exact same flow phase. Without phase-matching or time-warping correction, pointwise metrics (RMSE, Rel L2) and frequency errors (FE) can be artificially inflated by simple phase shifts rather than true modeling failure.
    - *Update Ratio Definition:* Section 3.3.1 defines Update Ratio as $N_1/N_2$ to reach $RMSE_0$ (best real-world training RMSE). It lacks crucial implementation details: What convergence threshold defines "reaching" $RMSE_0$? What is the maximum training budget if a model never reaches $RMSE_0$? How are $N_1, N_2$ normalized if learning rates or batch sizes differ?
- **Theoretical claims:** None; the paper is empirically driven.

### Experiments & Results
- **Testing claims:** The experiments directly address the stated goals: quantifying the sim-to-real gap and evaluating whether sim pretraining aids real-world prediction. Table 1 and Appendix A demonstrate these phenomena across datasets.
- **Baselines:** The 10 baselines are appropriately chosen, spanning classical (DMD), CNNs (U-Net, CNO), neural operators (FNO, DeepONet, MWT), Transformers (GK-Transformer, Transolver), and a foundation model (DPOT-S/L). Comparisons are generally fair, but DPOT-L (509M parameters) naturally has a capacity advantage over smaller baselines. The authors acknowledge this in Section 4.4, but the trade-off analysis should more explicitly weight parameter count vs. data efficiency.
- **Missing ablations:** 
    1. The individual contributions of *noise injection*, *modality masking*, and *combustion surrogate modeling* to the success of the "Real-world Finetuning" task are not isolated. An ablation table showing finetuning performance with/without these strategies would materially strengthen the claim that they bridge the gap.
    2. As noted above, an ablation matching the number of simulated samples to `n` (real samples) is necessary to decouple data quantity effects from data source effects.
- **Error bars / Statistical significance:** This is a significant concern for ICLR. All tables report single deterministic values. Scientific ML training involves stochastic optimization, and benchmark results typically require reporting mean $\pm$ std over multiple random seeds (e.g., 3-5 runs). The caption of Figure 3 mentions "statistics of 10 values," but it is unclear whether this refers to 10 seeds or 10 test trajectories. Without variance reporting, it is impossible to assess whether observed improvements (e.g., between U-Net and FNO, or Real-world vs. Finetuning) are statistically significant or within optimization noise.
- **Cherry-picking / Dataset appropriateness:** Results are comprehensive across all 5 datasets and 9 metrics. Combustion evaluation is appropriately constrained by measurement physics (light intensity only, no kinetic energy metric), which is honestly addressed in Appendix A.1.

### Writing & Clarity
- **Confusing sections:** 
    - The transition from task definition (Sec 3.1) to baseline results occasionally obscures *how* the finetuning budgets were allocated. Clarifying the exact early stopping or epoch limits would resolve ambiguity.
    - Figure 1's table artifacts appear to be parser errors (ignored per instructions), but the actual figure effectively illustrates modality gaps.
- **Figures/Tables informativeness:** Table 1 is dense but readable. Figure 4 (RMSE vs FE trade-off) is highly informative for quick architectural comparison. The autoregressive visualizations (Figs 3c, 5, 7-8) and frequency analyses effectively support the textual analysis.

### Limitations & Broader Impact
- **Acknowledged limitations:** Appendix F correctly identifies domain limitations (currently restricted to fluid dynamics and combustion), lack of combustion-specific physics metrics, and absence of systematic out-of-distribution (OOD) exploration.
- **Missed limitations / Failure modes:**
    - *Measurement physics constraints:* The benchmark relies on 2D planar measurements (PIV, CL imaging) of inherently 3D flows (especially turbulence and FSI). This dimensionality reduction is a fundamental limitation of the "real-world" ground truth itself, which could lead scientific ML models trained here to learn 2D artifacts or ignore out-of-plane vorticity. This should be explicitly flagged.
    - *Boundary/Domain generalization:* The current setup evaluates in-distribution and mild OOD regimes. It does not test extrapolation to vastly different geometries or boundary conditions, which is a primary failure mode for real-world deployment.
- **Broader impact:** The ethics statement is appropriately cautious about safety-critical deployment. A brief discussion on how these models might fail under noisy/unseen sensor conditions would add value, given the benchmark's focus on real-world measurement noise.

### Overall Assessment
This paper presents a highly valuable contribution to the scientific ML community by addressing a critical bottleneck: the lack of paired real-world and simulated data for benchmarking PDE forecasting models. The dataset collection is meticulous, the codebase is modular and reproducible, and the experimental analysis provides genuine insights into the sim-to-real gap and the efficacy of simulation pretraining. However, the empirical rigor currently falls short of ICLR standards in two key areas: (1) the absence of statistical uncertainty reporting (error bars across random seeds) makes it difficult to distinguish meaningful architectural or training improvements from stochastic optimization noise, and (2) the comparison between simulated and real-world training confounds data source with dataset size, while lacking a discussion on how temporal/phase alignment between experimental and numerical trajectories was handled. Addressing these points via variance reporting, size-matched ablations, and clarification of synchronization and metric convergence criteria would significantly strengthen the paper. Provided these empirical clarifications are made, the benchmark's scope, quality, and utility strongly support acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces RealPDEBench, the first scientific machine learning benchmark that pairs real-world experimental measurements with numerically simulated data across five complex physical systems (fluid dynamics and combustion). The benchmark defines three training paradigms, nine data- and physics-oriented evaluation metrics, and evaluates ten representative baselines, including state-of-the-art neural operators and a pretrained PDE foundation model. Experiments systematically quantify the sim-to-real gap and demonstrate that pretraining on simulated data followed by fine-tuning on limited real-world data consistently improves predictive accuracy and training efficiency.

### Strengths
1. **Addresses a Critical Bottleneck in SciML:** The paper correctly identifies that most scientific ML models are evaluated solely on synthetic data, limiting real-world applicability. By providing >700 paired trajectories across 5 physically distinct scenarios (Cylinder, Controlled Cylinder, FSI, Foil, Combustion), it directly tackles the scarcity of real-world data for PDE-related learning (Sec 1, Sec 3.2).
2. **Rigorous and Multi-Faceted Evaluation Protocol:** The benchmark goes beyond standard RMSE/MAE by incorporating physics-aware metrics (Frequency Error, Kinetic Energy Error, Mean Velocity Profile Error, and spectral fRMSE) and a novel "Update Ratio" metric to quantify data efficiency. The separation of real-only, sim-only, and sim-pretrain/real-ft tasks enables clean evaluation of domain transfer (Sec 3.1, Sec 3.3).
3. **High Reproducibility and Community Resource Value:** The authors provide extensive experimental details, including hardware specifications, PIV/CL imaging pipelines, numerical solver settings (Lilypad, Waterlily, STAR-CCM+), data formats, and a unified modular codebase. Raw data subsets and calibration files are released, aligning well with ICLR's emphasis on transparent and reproducible ML research (Appendices B, C, Reproducibility Statement).
4. **Clear Empirical Insights:** The results convincingly show that models trained purely on simulation generalize poorly to real data (Table 1), while sim-pretraining + real fine-tuning reduces errors and accelerates convergence (Update Ratio < 1, Fig 3b). The frequency-domain analysis reveals architectural trade-offs (e.g., CNNs excel locally, MWT/Transformers capture global periodicity better), offering practical guidance for model selection (Fig 4, Fig 6, Sec 4.4).

### Weaknesses
1. **Limited Algorithmic/Methodological Novelty:** The contribution is primarily benchmark-centric. While the evaluation is thorough, the paper introduces no novel learning algorithm, domain adaptation technique, or theoretical insight into bridging sim-to-real gaps. The proposed "mask-training" and "surrogate mapping" strategies (Sec 4.2, Appendix D.2) are standard engineering choices rather than research contributions, which may limit appeal for ICLR's core methodology-focused reviewers.
2. **Lack of Statistical Rigor in Reported Results:** All tables (e.g., Table 1, Tables 2-6) report single-point metrics across datasets without standard deviations, confidence intervals, or results across multiple random seeds. Given the inherent stochasticity in ML training and experimental noise, claims of "9.39% to 78.91% improvements" (Sec 4.2) lack statistical grounding.
3. **Underdeveloped Handling of Modality Mismatch:** For the Combustion dataset, real-world chemiluminescence \(I\) does not exist in simulations, so the authors train a separate U-Net surrogate to map simulated fields to \(I\) (Appendix D.2). However, the error introduced by this surrogate is not quantified or ablated, nor is there an analysis of how surrogate error propagates to downstream fine-tuning performance.
4. **Missing Analysis of Computational Cost and Scalability:** The benchmark includes a 509M-parameter foundation model (DPOT-L), but the paper does not report training/inference FLOPs, GPU hours, memory footprint, or time-to-convergence. This omission makes it difficult for practitioners to assess the feasibility of reproducing or extending the results, a key expectation for ICLR benchmark papers.
5. **Out-of-Distribution (OOD) Evaluation Remains Exploratory:** The dataset module supports OOD splits (`test_mode='unseen'`), but no systematic OOD results are presented in the main or appendix sections (Appendix F explicitly states this as future work). Real-world deployment fundamentally requires robustness to unseen parameters, and omitting this weakens the practical impact of the benchmark.

### Novelty & Significance
The paper's novelty lies in its comprehensive benchmark design rather than algorithmic innovation. It significantly advances the scientific ML community by providing the first large-scale, paired real/sim dataset with physics-aware evaluation protocols tailored for PDE and fluid/combustion systems. For ICLR, this is highly relevant given the growing interest in scientific foundation models, sim-to-real transfer, and standardized evaluation of physics-informed architectures. However, the work will be most impactful as a community resource and evaluation framework; reviewers seeking methodological breakthroughs may find it incremental unless the authors frame it around actionable insights for domain generalization or foundation model adaptation in science.

### Suggestions for Improvement
1. **Add Statistical Robustness Metrics:** Report mean ± std across at least 3 random seeds for key benchmarks (Table 1). Include error bars in Figures 3-4 and clarify whether hyperparameters were tuned per baseline or held fixed to ensure fair comparison.
2. **Ablate the Combustion Surrogate Pipeline:** Quantify the surrogate U-Net's mapping error (e.g., RMSE between predicted and measured \(I\) on held-out periods) and analyze its impact on downstream sim-to-real fine-tuning. Consider comparing against a simpler baseline (e.g., direct zero-padding or linear projection) to demonstrate the surrogate's necessity.
3. **Include at Least One Domain Alignment Baseline:** To elevate the paper beyond passive evaluation, implement or adapt a lightweight sim-to-real method (e.g., domain adversarial training, feature distribution matching, or learned noise injection) and show whether it further closes the gap. This aligns with ICLR's preference for benchmarks that catalyze new algorithmic research.
4. **Report Computational and Accessibility Costs:** Add a table summarizing parameter counts, VRAM usage, training hours, and inference latency for each baseline. This enables reproducibility and helps the community understand the resource requirements for foundation models vs lightweight operators.
5. **Expand the Evaluation to Include OOD Regimes:** Leverage the existing `test_mode='unseen'` functionality in the code to report baseline performance on parameter regimes excluded from training. Discuss whether sim-pretraining improves, maintains, or degrades OOD generalization, which is critical for real-world deployment claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Run controlled ablations isolating the drivers of sim-to-real improvement (e.g., simulated data volume vs. modality masking vs. weight initialization) to prove claimed benefits aren't merely artifacts of larger pretraining datasets rather than true domain transfer.
2. Report mean and standard deviation over ≥3 random seeds for all metrics, as single-run results on small training splits (~30 trajectories per dataset) make accuracy and convergence claims statistically fragile and unverifiable.
3. Explicitly benchmark and report results under strict out-of-distribution (OOD) test modes (unseen Reynolds numbers/angles), since the benchmark's core value proposition hinges on demonstrating generalization to unseen real-world regimes.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify and document the hyperparameter tuning budget, search space, and fairness protocol across all 10 baselines to rule out biased comparisons that artificially favor large foundation models or specific architectures.
2. Analyze the prediction error and physical consistency of the combustion surrogate model used to map simulated fields to chemiluminescence, as unvalidated surrogate outputs directly contaminate the simulated training signal for this key multi-physics scenario.
3. Investigate whether the mask-training strategy genuinely forces the model to learn robust cross-modal dynamics or simply teaches it to downweight missing channels, by analyzing gradient contributions or attention maps for masked vs. observed variables.

### Visualizations & Case Studies
1. Provide side-by-side spatiotemporal failure visualizations for worst-case OOD or high-noise trajectories to expose exactly where and how the sim-to-real gap causes physical divergence in autoregressive predictions.
2. Plot data-efficiency learning curves comparing real-only training vs. sim-pretraining+finetuning across shrinking real-data budgets (e.g., 10% to 100%) to visually validate claims about accelerated convergence and sample efficiency.
3. Overlay the Fourier spectra of real-world vs. simulated data for key physical variables to concretely illustrate which frequency bands (and corresponding physical phenomena) the simulation systematically fails to capture.

### Obvious Next Steps
1. Release a standardized hyperparameter optimization pipeline with equal compute/search budgets per baseline to ensure the benchmark’s model rankings are fair, reproducible, and trusted by the community.
2. Formalize the dataset splitting protocol to clearly separate parameter interpolation (in-distribution) from extrapolation (out-of-distribution), and publish both results as mandatory benchmark reporting standards.
3. Add rigorous statistical significance testing or confidence intervals to all leaderboard results, meeting ICLR’s empirical rigor standards given the high variance inherent in small-scale physical trajectory datasets.

# Final Consolidated Review
## Summary
This paper introduces RealPDEBench, a comprehensive benchmark for scientific machine learning that pairs large-scale real-world experimental measurements (via PIV and chemiluminescence imaging) with numerically simulated counterparts across five complex physical systems. By defining three training paradigms, nine data- and physics-oriented metrics, and evaluating ten representative baselines, the work systematically quantifies the simulation-to-reality gap and demonstrates the efficacy of simulated pretraining for real-world forecasting tasks.

## Strengths
- **Addresses a critical bottleneck in SciML:** The benchmark directly tackles the historical scarcity of paired real-world and simulated data, providing over 700 trajectories across five physically distinct scenarios (cylinder flows, controlled flows, FSI, 3D foils, and combustion). This fills a major gap that has constrained real-world validation and sim-to-real transfer research.
- **Rigorous and multi-faceted evaluation protocol:** Beyond standard pixel-wise losses, the benchmark integrates physics-aware metrics (frequency error, kinetic energy error, mean velocity profile error, and spectral fRMSE) alongside a novel "Update Ratio" metric to quantify data efficiency. The clear separation of real-only, sim-only, and sim-pretrain/finetune tasks enables clean domain transfer analysis.
- **Exceptional reproducibility and community value:** The paper provides extensive documentation of experimental hardware, PIV/CL imaging pipelines, numerical solver configurations, and data formats. The release of raw calibration files, processing scripts, and a highly modular codebase strongly supports transparency and future extensibility.
- **Actionable empirical insights:** The experimental analysis yields clear, practical conclusions: models trained purely on simulation generalize poorly to real measurements, whereas sim-pretraining consistently accelerates convergence and improves accuracy. The frequency-domain and autoregressive analyses also reveal meaningful architectural trade-offs (e.g., convolutional models excel at local features while spectral/multiwavelet models better capture global periodicity).

## Weaknesses
- **Lack of statistical uncertainty reporting:** All main results and appendices report single-point deterministic values without standard deviations, confidence intervals, or results aggregated over multiple random seeds. Given the intrinsic stochasticity of neural optimization and experimental noise, claims regarding performance gaps and baseline rankings cannot be assessed for statistical significance, weakening the empirical rigor expected for a machine learning benchmark.
- **Conflating data quantity with domain shift:** When comparing simulated training against real-world training, the benchmark intentionally uses all $N$ simulated samples versus a smaller subset $n$ of real samples to reflect practical data scarcity (Sec. 3.1). While practically motivated, this design inherently confounds the effects of *data source* (simulation vs. reality) with *dataset size*. Without a controlled ablation where sim and real training use matched sample counts, it remains unclear how much of the simulation advantage stems from sheer volume rather than underlying domain characteristics.
- **Unquantified error propagation in the Combustion surrogate:** For the Combustion dataset, real-world modalities (chemiluminescence $I$) do not exist in the simulation. The authors address this by training a standalone U-Net surrogate to map simulated fields to $I$ before finetuning (Appendix D.2). However, the surrogate's predictive accuracy, error bounds, and potential bias are never reported or ablated. Unquantified surrogate errors directly contaminate the simulated pretraining signal, making it difficult to isolate the true sim-to-real transfer dynamics in this multi-physics scenario.

## Nice-to-Haves
- **Out-of-Distribution (OOD) generalization results:** While the codebase supports OOD splits (`test_mode='unseen'`), the paper currently limits analysis to in-distribution parameters. Reporting baseline performance on unseen Reynolds numbers or angles would significantly strengthen the claim of real-world deployment readiness, even as a preliminary exploration.
- **Computational budget and scaling metrics:** Including a summary of VRAM usage, training hours, and inference latency for each baseline would help practitioners contextualize the performance trade-offs, particularly when comparing lightweight operators to the 509M-parameter DPOT-L model.

## Novel Insights
The paper provides a compelling empirical demonstration that simulation and real-world data are fundamentally complementary rather than interchangeable for scientific forecasting. Despite significant distributional shifts caused by measurement noise, numerical discretization, and unobserved modalities, leveraging simulated trajectories for pretraining consistently yields faster convergence and superior real-world accuracy compared to training on real data alone. Furthermore, the multi-metric analysis reveals that no single architecture dominates across all scientific objectives: convolutional operators excel at local spatial accuracy, while spectral and multiwavelet methods are inherently better suited for capturing the global periodic dynamics and long-range physical correlations critical to unsteady flows and reactive systems.

## Suggestions
- **Implement variance reporting:** Re-run key experiments (Table 1, Figure 4) across at least three independent random seeds and report mean ± standard deviation. Replace single-value comparative claims with statistically grounded statements to distinguish genuine architectural/training advantages from optimization noise.
- **Add a size-matched ablation:** Conduct a controlled experiment comparing simulated and real-world training using an equal number of trajectories (e.g., both trained on $n$ samples). This will isolate the specific impact of domain shift from the impact of dataset scale and clarify the true sim-to-real gap.
- **Benchmark and document the Combustion surrogate:** Quantify the surrogate U-Net's mapping error on a held-out set and include an ablation showing how finetuning performance changes when varying the surrogate's accuracy. This will clarify whether the reported benefits in the Combustion setting are robust to surrogate uncertainty.

# Actual Human Scores
Individual reviewer scores: [10.0, 10.0, 6.0, 4.0]
Average score: 7.5
Binary outcome: Accept
