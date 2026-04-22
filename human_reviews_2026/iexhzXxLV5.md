# ALETHEIA: A Multi-Frequency Eddy Current Pulsed Thermography Dataset for Neural Operator Learning in Nondestructive Testing

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Learning neural solvers for spatiotemporal partial differential equations (PDEs) under real-world constraints remains a key challenge in scientific machine learning, especially for inverse tasks with sparse and noisy boundary observations. We present the **Aletheia** dataset, the first 3D benchmark for learning data-driven solvers in the context of **nondestructive testing (NDT)**. The dataset simulates eddy-current-induced heating in conductive solids and models the resulting transient heat propagation governed by the heat equation. Aletheia contains over 4,700 high-resolution samples across 10 excitation frequencies (1-100\,kHz), each providing volumetric heat source and temperature fields over time. It supports both forward prediction of temperature evolution and inverse reconstruction of internal heat sources or defects from surface infrared measurements. Real infrared thermography data from cracked rail specimens are included for calibration and generalization studies. We define three canonical tasks on both regular and irregular grids and benchmark them using various neural operators. Aletheia establishes a unified platform for evaluating neural PDE solvers under realistic NDT conditions, enabling progress in reliable, data-driven inverse modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposed a 3D benchmark, Aletheia. The dataset simulates eddy-current-induced heating, which is an electromagnetic-thermal coupling physics. The dataset is feature with comprehensive evaluations (see Table 1).

### Strengths
**Significance** The work sets a new benchmark for neural operators. The evaluation is comprehensive, and notably it includes out-of-distribution (OoD) evaluation, which is valuable and important for many scenarios of the application of neural operators.

### Weaknesses
1. For geometries and topology in Alethesia, are the types listed in Table 6 in Appendix C able to cover most cases? What was your consideration of such selection?

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Aletheia, a benchmark dataset designed for learning neural PDE solvers in the context of 3D NDT, specifically focusing on multi-frequency ECPT. ECPT involves coupled electromagnetic (Maxwell's equations) and thermal (Heat equation) physics, used to detect subsurface defects in conductive materials like rails.

The key contributions are:

1. A High-Fidelity 3D Dataset: Aletheia comprises over 4,700 simulations generated using COMSOL, covering six distinct types of internal rail defects. It provides time-resolved volumetric heat source (q) and temperature (u) fields on both regular and irregular grids.

2. Multi-Frequency Data: To address the ill-posedness of the inverse heat conduction problem, the dataset includes 10 excitation frequencies (1-100 kHz), leveraging the electromagnetic skin-depth effect to probe different material depths.

3. Real-World Calibration: The simulations are calibrated using real infrared thermography data from physical rail specimens.

4. Benchmark Suite: The authors define eight tasks spanning in-distribution and Out-of-Distribution (OOD, based on frequency shifts). These include forward modeling (Q2T) and challenging inverse tasks, notably Surface-to-Source reconstruction (S2Q).

5. Baseline Evaluations: Several neural operators (FNO variants, Transolver, LNO, etc.) are benchmarked on a subset of the data.

Aletheia aims to bridge the gap between academic PDE benchmarks and realistic, 3D, multi-physics inverse problems relevant to industrial applications.

### Strengths
The primary strength of this work lies in the dataset generation effort, which addresses significant limitations in the existing landscape of SciML benchmarks.

1. Significance and Relevance: The creation of Aletheia is a substantial effort. It moves beyond standard academic benchmarks (e.g., Darcy Flow) towards a complex, industrially relevant NDT problem. As highlighted in Table 1, it uniquely combines 3D geometries, inverse problems, and partial observations in a multi-physics setting (coupled electromagnetic and thermal PDEs).

2. Rigorous Data Generation Methodology: The approach of using high-fidelity multi-physics simulations (COMSOL) and, crucially, calibrating them with real experimental data (Section 3.1) is commendable. The inclusion of both regular and irregular grids is valuable for testing mesh invariance.

3. Physically Motivated Design: The inclusion of multi-frequency excitations (1-100 kHz) is well-motivated. It leverages the electromagnetic skin-depth effect to provide depth-sensitive information, which is essential for mitigating the ill-posedness of the inverse thermal problem (Figure 1).

### Weaknesses
While the dataset itself is a valuable contribution, the paper, presented as a benchmark study, suffers from significant weaknesses in its experimental design and analysis, which undermine the benchmark's utility and the conclusions drawn.

1. Severely Limited Scope of Benchmark Evaluation (Critical Flaw): The empirical evaluation is conducted on a very small subset of the data. Section 4.2 states that experiments involved 600 samples from the "Type I double-layer defect simulations." This represents only ~12.5% of the total 4,782 samples and only 1 out of the 6 defect types. A benchmark paper must establish baselines across the diversity of the data it introduces. The conclusions drawn regarding the relative performance of neural operators are therefore not substantiated for the vast majority of the dataset, particularly the more complex defect types (e.g., multi-layer or closed cracks).

2. Failure to Evaluate Under Realistic Conditions (Noise and Sim-to-Real): The abstract claims the dataset addresses challenges involving "sparse and noisy boundary observations." While the S2Q task addresses sparsity, the benchmark entirely ignores noise.
Real infrared data is noisy. Robustness to sensor noise is crucial for ill-posed inverse problems but is not evaluated, and while the real data is used for calibration, the models trained on simulations are never evaluated on the real experimental data. Assessing the sim-to-real gap is essential for any simulation-based benchmark intended for real-world application.

3. Weak Link Between Benchmark Tasks and NDT Goals: The primary inverse tasks (T2Q, S2Q) focus on reconstructing the heat source field q(x,t). The ultimate goal of NDT is defect characterization (geometry, depth). Appendix D attempts to bridge this gap by regressing defect parameters from q. However, the results in Table 7 are weak. For instance, the RMSE for crack depth is approximately 1.0 mm (MAE 0.70-0.91). Given that the defects themselves range from 0.2mm to ~4mm in depth (Table 6), an error of 1mm is very large relative to the defect scale. This casts doubt on whether optimizing for q reconstruction MSE is sufficient for the intended NDT application.

4. Questionable Experimental Methodology and Data Usage Mismatch: The high-resolution unstructured data (50,000 points, Appendix B) is aggressively downsampled to 8000 points for the experiments (Sec 4.2). This may discard the fine-grained details necessary for accurate defect reconstruction. Also, the use of a batch size of 1 (Table 10) for all models is highly unusual for training large models and can lead to unstable training and poor generalization, potentially affecting the validity of the comparisons. Finally, while the motivation for multi-frequency data is clear (Fig 1), the benchmark does not empirically quantify this benefit. A crucial missing experiment is a comparison of inverse reconstruction performance using single-frequency vs. multi-frequency data.

5. Limited Scope of OOD Generalization: The OOD tasks are exclusively focused on unseen frequencies. In NDT, generalization to unseen defect morphologies (Geometric OOD) is critical but is not evaluated using the diverse defect types available.

### Questions
Q1: Why was the empirical evaluation restricted to only the Type I double-layer subset (600 samples)? The validity of the benchmark relies on evaluating the models across the diverse range of defect types provided. Can you provide results utilizing the full dataset or, at minimum, results for the other five defect types?

Q2: a) How does the performance of the models on the S2Q task degrade when realistic levels of sensor noise are added to the surface temperature input? b) Were models trained on the synthetic data tested on the real experimental measurements? If so, what were the results regarding the sim-to-real gap?

Q3: Given the relatively poor performance in deriving defect parameters from the reconstructed heat field Q (Table 7, Appendix D, RMSE ≈ 1mm for depth), how do you justify the use of Q reconstruction as the primary benchmark task for NDT? Is this accuracy sufficient for practical rail inspection?

Q4: a) What is the justification for the aggressive downsampling to 8000 points and the use of batch size 1? b) Can you provide an experiment quantifying the improvement gained by using multi-frequency data compared to single-frequency data for the inverse tasks?

Q5: The dataset size is 1.89 TB. What is the concrete strategy for hosting the data to ensure long-term, accessible availability, including mechanisms for downloading specific subsets?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ALETHEIA, a new large-scale, multi-frequency 3D dataset for nondestructive testing (NDT) based on Pulsed Eddy Current Thermography. The dataset is generated from high-fidelity multiphysics simulations, which are calibrated using real-world experimental data from rail specimens. The authors define a comprehensive benchmark suite with tasks including forward prediction, inverse source reconstruction, and temporal evolution, evaluated on both regular and irregular grids. A key feature is the inclusion of out-of-distribution (OOD) generalization tasks to test model robustness to unseen excitation frequencies. The paper provides an extensive evaluation of several state-of-the-art neural operator models (e.g., FNO, Transolver) and offers insights into their relative performance under different conditions.

### Strengths
High-Quality and Significant Dataset: The paper presents a dataset for a challenging and practically important real-world problem. The effort to combine high-fidelity simulation with calibration from real experimental data is commendable and ensures the dataset's relevance and realism. This is a significant contribution to the scientific machine learning and NDT communities.

Comprehensive Benchmark Design: The benchmark is thoughtfully designed. It includes a variety of relevant tasks (forward, inverse, temporal), considers practical challenges like irregular grids, and, most importantly, incorporates out-of-distribution (OOD) generalization tests. This level of rigor is crucial for pushing the boundaries of neural operator models.

Thorough Empirical Evaluation: The authors have benchmarked a wide range of modern and relevant neural operator architectures. The comparative analysis, particularly the conclusion that spectral-based models (FNO) excel with full-field data while attention-based models (Transolver) are better for sparse, surface-only inverse problems, provides valuable practical guidance.

### Weaknesses
Limited Novelty for the Core ML Community: The main weakness is that the paper's contribution is primarily a new application domain and dataset. While this is valuable, the paper does not propose new neural operator architectures, learning techniques, or theoretical insights that would be broadly applicable to the general ICLR audience. The work primarily uses existing ML tools to solve a domain-specific problem, rather than advancing the ML tools themselves.

Lack of Deeper Analysis of Model Behavior: While the paper reports which models perform better, it offers limited deep analysis as to why. For instance, a more in-depth investigation into the spectral properties of the heat diffusion problem and how they align with the spectral bias of FNO would be insightful. Similarly, an error analysis showing what kinds of defects or thermal patterns the models struggle with could reveal fundamental limitations and inspire future model development. The current analysis remains somewhat at the level of a "horse race."

Presentation of Results: The main paper (8-9 pages) presents the results primarily through radar charts, which are good for a high-level overview but lack quantitative detail. The reader must navigate through many pages of tables in the appendix to find the concrete numbers. A more condensed table summarizing the most critical results in the main body would significantly improve readability and impact.

### Questions
The core of ICLR is representation learning. Beyond providing a new challenging benchmark, what do the authors consider to be the main, generalizable takeaway for a machine learning researcher who is not an expert in NDT? What fundamental new capabilities should the next generation of neural operators possess that are highlighted by ALETHEIA but are not apparent from existing benchmarks like Darcy Flow or Navier-Stokes?

The paper observes that Transolver performs better on surface-to-source (S2Q) tasks. The reasoning given is based on its adaptive, attention-based receptive fields. Could you provide a more quantitative or qualitative analysis to support this? For example, can you visualize the attention maps to show that the model indeed focuses on "critical boundary features"?

The performance drop on OOD tasks is expected, but is there a pattern to it? For instance, do models generalize better from low-to-high frequencies or vice-versa? A deeper analysis of the OOD generalization behavior could provide valuable insights into the current limitations of neural operators in extrapolating physical parameters.

### Soundness
2

### Presentation
2

### Contribution
2
