# Laser- and shock-induced droplet dynamics: A machine learning benchmark for complex multiphase flows

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Compressible multiphase flow is central to numerous engineering applications, characterized by complex wave dynamics and challenging shock-interface interactions. Despite their importance, they remain significantly missing from existing benchmarks in the Scientific Machine Learning (SciML) community, limiting progress on generalization to impactful real-world scenarios. To address this issue, we introduce two exemplary datasets from this class, Laser-Induced Droplet Explosion (LIDE) and Shock-Induced Droplet Aero-breakup (SIDA), providing researchers with valuable references to establish reliable baselines and push boundaries of SciML. Due to the high computational cost of simulating these processes with full fidelity, we explore data-driven surrogate models designed to efficiently approximate the underlying physics at reduced cost. We benchmark these datasets on diverse architectures trained autoregressively and compared across varying parameter counts. A comprehensive set of ablations is carried out to analyze the performance of the models. We identify key scenarios, such as incorporating temporal sequence information and conditioning, that enable the models to accurately capture the rich and nonlinear physics embedded in the datasets. Code and datasets will be made publically available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The author proposed a machine learning benchmark for multiphase flow data. Although two high-quality data were provided, the reviewer believes that the paper is not yet ready for publication due to the amount of data, the methods presented, and the interpretation and analysis of the results.

### Strengths
1. Two new datasets have been provided for use by the AI community.
2. Provided results of different methods on data.

### Weaknesses
1. The relevant work is not comprehensive, and it seems that the author's research on PDE benchmarks is insufficient. There are still many works such as FlowBench, The Well, BLASTNet, etc. that should be discussed in the relevant work.
2. The lack of 3D data in the benchmark is a drawback.
3. Can valuable insights be derived from the results, such as which types of methods are more suitable for which features of the data.
4. The biggest concern is that there are too few methods used and they are not SOTA based. There are many other methods such as CNO, Translolver, Diffusion model, AROMA, etc. that should be included in the benchmark.
5. Other benchmarks such as PDEBench and The Well provide a large amount of data, but this work only has two datasets and is not suitable for publication as a benchmark.
6. Lack of experiments on PDE foundation models such as GNOT, DPOT, etc.
7. An important evaluation criterion for benchmarks is whether the code interface is easy to use, but the author does not have open source code, so it cannot be evaluated.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents new field evolution datasets related to complex multiphase flow phenomena that could serve as a real-world benchmark for the scientific machine learning (SciML) community. Two specific physics phenomena are considered, namely laser-induced droplet explosion and shock-induced droplet aero-breakup. The paper also establishes a baseline by comparing the predictive performance of diverse architectures including U-Net, Fourier neural operator (FNO), vision transformer (ViT), scalable operator transformer (ScOT), and residual network (ResNet).

### Strengths
Compared to the usual benchmarks in SciML literature, such as the incompressible Navier-Stokes equation or Burgers' equation, these new benchmarks present unique challenges for SciML involving the prediction of sharp reaction fronts, discontinuity at the flow interfaces, sharp spatial and temporal gradients, and fast transient features. Such strong nonlinearity makes them a useful dataset to test the limits of SciML algorithms, which I believe is an important contribution to the research community.

### Weaknesses
I think the paper may benefit from considering the following suggestions:
- Provide more metric/criteria for evaluating prediction results. The current quantitative metrics such as RMSE provide an overall estimate of how "accurate" models are, but from the domain standpoint, those metrics may not fully capture the phenomena of interest. For example, for droplet atomization, we may not care too much about the exact physical values in the ambient space, as long as the rate of droplet breaking up or the size of droplets after being broken down. (And vice versa--even if a model is very accurate in predicting other physical values, I would argue that the model isn't still very useful if it cannot faithfully predict droplet displacement, deformation, etc.) That said, I think the community is going to benefit a lot by having those domain-specific performance metrics. These metrics don't need to be all quantitative--qualitative criteria in terms of what kind of features to look for would still be highly valuable.
- This is not terribly critical, but I noticed that the ML models that the authors compared are mostly (solution) operator-based formulations. It might be worthwhile to compare these methods against other methods like PINN, but I don't want to arm-twist the authors to create a laundry list of all bunch of random methods.

### Questions
- "Code and datasets will be made available upon request." Can they be made available upon acceptance? I do find this a valuable work but only when (at least) the datasets are made available to the community.

### Soundness
3

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
5

### Summary
This paper introduces two benchmark datasets—LIDE (Laser-Induced Droplet Explosion) and SIDA (Shock-Induced Droplet Aero-Breakup)—to advance research in compressible multiphase flows within the Scientific Machine Learning (SciML) community. The authors evaluate several surrogate models, including FNO, UNet, ViT, ScOT, and ResNet, to capture complex nonlinear shock–interface dynamics. The study effectively demonstrates the importance of temporal conditioning in improving predictive accuracy and stability.

### Strengths
1) Presents novel, high-fidelity datasets addressing important multiphase flow problems with strong physical and industrial relevance.

2) Includes comprehensive benchmarking across diverse neural architectures, supported by detailed ablation studies and insightful analysis.

### Weaknesses
1) The dataset and code are not yet publicly available, limiting reproducibility and immediate community adoption.

2) Evaluation could be broadened to include recent operator-learning models, such as Transformer-based and state-space (SSM) operators (e.g., Transolver, Mamba Operator).

3) Some implementation details—including hyperparameter settings and conditioning strategies—are not clearly described.

### Questions
1) Since the primary contribution is the dataset, would the authors consider open-sourcing the datasets and code to encourage best practices and broader use within the ML community?

2) How well do the surrogate models generalize to unseen physical regimes or parameter variations, and is zero-shot transfer feasible?

3) Could the datasets be extended to 3D cases or include additional physical phenomena such as phase change or evaporation?

4) How sensitive are the results to temporal resolution and conditioning strategies? Including standard deviations would strengthen the statistical credibility.

5) How do different boundary conditions (e.g., symmetry, Dirichlet, Neumann) affect performance? Can models handle zero-shot transfer across boundary types, and which operator performs best?

6) The distinction between Error 1 and Error 2 is somewhat unclear, as Tables 3 and 4 do not show a clear correlation. Could the authors include energy spectrum analyses and rollout error over time steps to better interpret model behavior and physical fidelity?

**Minor Comments:**

1) The statement “In addition, for reproducing model evaluations, we provide trained model weights and the code that has the complete set of instructions upon request.” — It would be highly beneficial to open-source the dataset and trained weights to enhance accessibility and reproducibility within the SciML community.

2) Please ensure that the supplementary material remains anonymous, as some identifiable information may still be present in the metadata.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the gap of benchmarks for compressible multiphase flows in the SciML field. It introduces two high-fidelity datasets, Laser-Induced Droplet Explosion (LIDE) and Shock-Induced Droplet Aero-breakup (SIDA), which capture complex physical phenomena like shock-interface interactions and droplet dynamics. The authors then benchmark these datasets on five neural network architectures with different parameter scales, exploring how factors such as temporal sequence information, conditioning parameters, and conditioning fields affect model performance.

### Strengths
1. A new dataset is proposed, filling the gap of compressible multiphase flow data in existing benchmark datasets.

### Weaknesses
1. The data scenarios lack diversity, with only two fixed setups (as shown in Figure 2). This makes the evaluation results on this dataset hardly reflect the model’s performance in prediction tasks beyond these scenarios, nor can it be used to test the model’s out-of-distribution generalization ability. Consequently, the practical usability of this dataset is relatively low.
2. There are serious flaws in the data recording of the benchmark: all comparison results are presented as bar charts, instead of **tables**, with a log-rescaled vertical axis in Figure 3/4/5/6. This lacks precision and is unfavorable for tracking the future development of this field.
3. The quality of writing, figures/tables, and typesetting is poor, with specific issues listed below:

- (1) In Table 2, the meaning of "End time" is unclear, and some critical information (e.g., the number of trajectories, the number of time steps) of both datasets is missing.

- (2) Figure 1 only shows two time frames ($t$=30s, $t$=50s), which is insufficient to reflect the dynamic change process in the dataset. Additionally, "Density" and "Schlieren" use color bars of the same color scheme, which causes confusion for readers.

- (3) As a claimed contribution, the method and results of "Dataset Validation" should be highlighted in the main text, rather than only included in the appendix.

- (4) The meaning of "Error types" in Section 5.5 is not defined, making it difficult to understand the content of this subsection.

### Questions
Please respond to the above mentioned Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
