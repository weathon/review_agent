# SIFBench: An Extensive Benchmark for 3D Fatigue Analysis

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Fatigue-induced crack growth is a leading cause of structural failure across critical industries such as aerospace, civil engineering, automotive, and energy. Accurate prediction of stress intensity factors (SIFs) --- the key parameters governing crack propagation in linear elastic fracture mechanics --- is essential for assessing fatigue life and ensuring structural integrity. While machine learning (ML) has shown great promise in SIF prediction, its advancement has been severely limited by the lack of rich, transparent, well-organized, and high-quality datasets.

To address this gap, we introduce SIFBench, an open-source, large-scale benchmark database designed to support ML-based SIF prediction. SIFBench contains over 5 million different 3D crack and component geometries derived from high-fidelity finite element simulations across 37 distinct scenarios, and provides a unified Python interface for seamless data access and customization. We report baseline results using a range of popular ML models --- including random forests, support vector machines, feedforward neural networks, and Fourier neural operators --- alongside comprehensive evaluation metrics and template code for model training, validation, and assessment. By offering a standardized and scalable resource, SIFBench substantially lowers the entry barrier and fosters the development and application of ML methods in damage tolerance design and predictive maintenance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a dataset that is challenging for standard ML models. Fatigue analysis is an important area of study where machine learning could have a large impact. The benchmark results indicate that ML predictors perform nearly as well on twin-crack problems as single-crack problems; even though it is much harder to simulate twin-crack systems. The benefits to the materials degradation research community are clear, but the impact of this paper in the ICLR community is not clear. More details analyzing the results of the benchmark models would provide insight as to what advances in machine learning could provide more accurate results.

### Strengths
This is a well thought out dataset for fatigue analysis with an original contribution of a novel dataset. The paper is clearly written with high amounts of detail in how the data was created; yet it is lacking in other details as mentioned below. The benchmark results on 3 standard models show that L2 error would need to be improved from the current level of 10^-1 or 10^-2; to the practically useful level of 10^-4. Evaluation metrics are provided.

Note on my rating: I think the paper is sound, but this is not the right audience. "I would not mind if the paper is accepted" is true, but I worry that it will have less impact at ICLR than it would in a materials science venue.

### Weaknesses
For an ICLR paper, there needs to be more motivation for the impact that this dataset has for the ML community (it’s clear that this is useful for the materials degradation community, so it seems that is where the paper should be published to get the right audience). As an ML researcher, after reading this paper, I am unlikely to take the time to learn more about the data in Huggingface - the paper hasn’t given me enough information for me to know whether the methods that I work with are potentially a good fit for this dataset. Sections 2.4, 2.5, and 3 should be the majority of the paper to detail what is challenging about the dataset for existing ML approaches. In addition to the L2 errors summarized in Table 1 and 2, it would be helpful to see examples of what the ML models are predicting well and what they are not predicting well. An analysis of the results would lead to more insight that would lower the bar to further research.

From the abstract, it would be helpful to have some hints about what is challenging for ML in this application? Why does the ML community want to consider this dataset? Instead of “We report baseline results…”, tell us what you conclude from those results.

Data format section could use more details. How many features (paper says “multiple columns”)? Are the features scalar, vector, binary? I am not sure what form the “component geometry” (for example) would take.

### Questions
Why are RFR and SVR trained on a subset of data? 100,000 data points out of a total of 5 million is a very small subset. Yet, it appears that RFR is out-performing both SVR and FNO, while quite competitive with FNN (it would be helpful to highlight the lowest error in the tables).

For Tables 1 and 2, is each model trained on one Scenario at a time, or is each model trained on all of the Scenarios combined together?

The twin-crack problems may be harder to simulate, but it seems that they are not significantly harder for the ML predictors. Is this correct?

The number of simulations per Scenario can vary from 2,000 to over 1,000,000. Does the number of simulations in the dataset correlate with the accuracy (or negatively correlate with the error) for that scenario?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a benchmark database SIFBench designed to support ML-based SIF prediction for structures with representative geometries and loading conditions, which are obtained from finite element analysis. Python script interfaces are provided for data access and customization. Baseline results using RFs, SVM, FNN, and FNO are demonstrated. The work offers a standardized and scalable resource, and fosters the development and application of ML methods in damage tolerance design and predictive maintenance.
However, the geometries and loading conditions of the models are not explained in detail, and the comparison with the traditional handbook approach was not made. Given the rich space of geometries and loading conditions in practice, the power of the current approach over the empirical approach is not clear.

### Strengths
1 A comprehensive SIF database with 5 million 3D crack and component geometries derived from high-fidelity finite element simulations across 37 distinct scenarios are reported, laying the ground for exploration of ML methods applied in fatigue crack predictions.
2 A unified format is reported, organizing the information in SIFs.
3 Interface scripts and demonstration of ML uses are provided, lower the barrier of learning and understanding.

### Weaknesses
1 Although the authors mentioned that the dataset is open source. I did not find the database from the paper and in the link reported in an earlier arXiv paper with the same title (https://huggingface.co/datasets/tgautam03/SIFBench - 404 error is reported). Consequently, I cannot check the content and scripts for more information.
2 The advantage of the current approach over the traditional handbook one is not demonstrated. Given the complexities in the geometries and loading conditions, a better explanation of how the dataset serves as practical sources should be given.

### Questions
1 In practical setups, the space of geometries and loading conditions is very large. Does the current approach cover more space in design than the handbook approach, and is there improvement in accuracy (if yes, is that significant enough?).
2 In reality, LEFM does not precisely work, and will result in physical bias even using high-fidelity finite element modeling. Is the potential improvement in accuracy beyond the handbook solutions significant enough to overcome this bias?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SIFBench, a large-scale, open-source benchmark for stress intensity factor (SIF) prediction in 3D fatigue crack scenarios. It consists of over 5 million high-fidelity FEM simulations spanning 37 geometry/loading cases relevant to structural components (e.g., aircraft plates with surface and corner cracks). The benchmark includes a Python API, baseline results using common ML models, and several evaluation metrics. The goal is to enable scalable and standardized ML-based SIF prediction to support fatigue life assessment. The paper is clearly written, well-organized, and includes helpful figures and explanations. The presentation of datasets, metrics, and baseline models is generally clear.

### Strengths
High-quality dataset: Targets an important problem in fatigue analysis, with relevance to aerospace, civil, and mechanical engineering. Paper includes 5 million FEM simulations across realistic engineering configurations, a scale not seen in prior 3D SIF benchmarks.

Open-source and reproducible: Promises data/code availability with unified API and baseline templates, which is an important step toward community standardization.

Strong experimental foundation: Simulations come from validated CAStLE datasets, using hp-FEM with verified accuracy.

### Weaknesses
The limited diversity of benchmarks presents a significant challenge. Current scenarios are restricted to flat plate geometries with surface or corner cracks, failing to represent crucial factors in fatigue such as curved shells, anisotropic materials, plasticity effects, or dynamic/multi-axial loading. It would be helpful to explain the choices and its impact on real-world fatigue situations. 

The SIFBench authors do not demonstrate that models trained on their data would perform well on real cracked components (e.g., no case study of applying a trained model to a known lab experiment). This raises questions of real-world relevance: the benchmark might optimize algorithms for its specific simulation settings, but those models could face a reality gap when confronted with experimental noise or unmodeled physics.

Only standard regressors and one FNO are tested. No graph-based, physics-informed, hybrid, or transformer-style models. No hyperparameter tuning or ablation studies. The evaluation doesn't explore why simpler models outperform FNO.

The analysis lacks generalization, as all evaluations rely on train/test splits within a single scenario. Consequently, there's no assessment of cross-scenario generalization, such as training with straight-hole cracks and testing with countersunk-hole cracks. This significantly weakens claims of broad utility.


The simulation pipeline remains proprietary, despite the open availability of data and code. The generation scripts (ABAQUS/FRANC3D setup) are not accessible. Consequently, reproducing or extending the dataset necessitates considerable domain expertise.

### Questions
Could the authors add cross-scenario generalization results (e.g., train on some geometry types and test on held-out ones)? This would better assess real-world applicability.

Why were Transolver/transolver++ or other recent physics-based models or graph networks or neural operators not included in the baseline? 


Report ML model’s memory usage, latency, inference throughput, and model size, wich are essential for practical deployment. SIFBench reports no such information, making it hard to compare models on real-world viability or cost-performance tradeoffs.

Will there be lightweight subsets or APIs to query the 5M dataset for users with limited compute?

How can the community extend the benchmark (e.g., contribute new simulations or experimental data)?


Efficiency Profiling: Can the authors report memory usage, inference time per sample, and parameter count for each baseline model? This is essential for comparing SIFBench to other scientific ML benchmarks like DrivAerNet++ or PDEBench.


Could a diversity analysis be added to show the geometric variation across the 37 scenarios in SIFBench? How diverse it is compared to other geometric datasets.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SIFBench introduces a large open benchmark for stress-intensity-factor (SIF) prediction for 3D crack geometries, assembled from high-fidelity FEM simulations (5M unique instances across 37 scenarios). The authors provide a Python interface, CSV data releases, and baseline experiments using Random Forest Regression (RFR), Support Vector Regression (SVR), feed-forward neural networks (FNN), and a Fourier Neural Operator (FNO). Preliminary results show errors often within a few percent for many cases, and the paper positions SIFBench as a community resource to accelerate ML methods for fatigue analysis.

### Strengths
1. The dataset is high-quality and large-scale covering 37 scenarios with single- and twin-crack families, multiple loading modes (tension, bending, bearing), and many parameter ranges, making it relevant to aircraft/industrial cases. 

2. Use of metrics is clear and the paper provides a Python interface, CSVs, and baseline scripts to lower adoption barriers.

### Weaknesses
1. The diversity of ML baselines is severely limited. The experimental baselines are limited to RFR, SVR, FNN, and FNO, leaving behind many other ML methods that are relevant to this problem (e.g., graph neural networks, GraphMeshNets, PINNs). Also, no existing ML works specifically designed for SIF prediction have been used for comparison, significantly reducing the contribution of this work.

2. The hyperparameter search and training protocol details is limited. The authors note that FNO sometimes underperforms relative to simpler FNNs/RFRs, but it is not clear whether this is because of FNO is a poor fit for the problem, or whether additional hyper-parameter tuning is needed. The training details reveal heavy reliance on defaults and small subsampling for some baselines (e.g., RFR/SVR trained on up to 100k randomly sampled points, SVR constructed with default args, RFR left with defaults except max_depth=None). For deep models, the FNN uses a fixed small architecture (5 layers × 15 units) trained for 150k epochs; the FNO uses a single configuration (64 modes, width 64). This reduces reproducibility of what tuning efforts were attempted. 

3. Although the dataset and problem are inherently physics-constrained, the current baselines do not include physics-informed approaches such as PINNs or PINO. The paper does cite mechanics-guided symbolic regression work in related literature but does not compare it as a baseline.

### Questions
1. Can you show comparisons with a larger set of baselines across different model families: (a) convolutional architectures or CNNs applied to grid/voxelized representations (authors cite a CNN in related work but do not evaluate it), (b) graph/mesh neural networks (message passing) that operate naturally on irregular meshes, (c) physics-informed ML models (PINN, PINO, or mechanics-guided symbolic regression)? This will clarify whether FNO underperformance is algorithmic or due to tuning/representation choices.

2. Can you show how FNN/FNO/RFR performance scales with training set size (e.g., 10k / 100k / 1M samples), and include hyperparameter sweeps or automated tuning (random search / Bayesian optimization) for each method?

### Soundness
3

### Presentation
3

### Contribution
3
