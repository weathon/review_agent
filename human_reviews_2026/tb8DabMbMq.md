# PU-BENCH: A UNIFIED BENCHMARK FOR RIGOROUS AND REPRODUCIBLE PU LEARNING

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Positive-Unlabeled (PU) learning, a challenging paradigm for training binary classifiers from only positive and unlabeled samples, is fundamental to many applications. While numerous PU learning methods have been proposed, the research is systematically hindered by the lack of a standardized and comprehensive benchmark for rigorous evaluation. Inconsistent data generation, disparate experimental settings, and divergent metrics have led to irreproducible findings and unsubstantiated performance claims. To address this foundational challenge, we introduce **PU-Bench**, the first unified open-source benchmark for PU learning. PU-Bench provides: 1) a unified data generation pipeline to ensure consistent input across configurable sampling schemes, label ratios and labeling mechanisms; 2) an integrated framework of 18 state-of-the-art PU methods; and 3) standardized protocols for reproducible assessment. Through a large-scale empirical study on 8 diverse datasets (**2880** evaluations in total), PU-Bench reveals a complex yet intuitive performance landscape, uncovering critical trade-offs between effectiveness and efficiency, and systematically mapping method robustness against variations in label frequency and selection bias. It is anticipated to serve as a foundational resource to catalyze reproducible, rigorous, and impactful research in the PU learning community. The source code is publicly available at <https://github.com/XiXiphus/PU-Bench>.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on Positive-Unlabeled (PU) learning. It points out that although numerous PU learning methods have been proposed, the field still suffers from inconsistent data generation, disparate experimental settings, and divergent evaluation metrics, which have led to irreproducible results and unsubstantiated performance claims. Furthermore, the lack of a standardized and comprehensive benchmark has hindered rigorous evaluation. To address these issues, the paper introduces PU-BENCH, an open-source framework that provides a unified data generation pipeline and an integrated evaluation framework, aiming to catalyze reproducible, rigorous, and impactful research within the PU learning community.

### Strengths
1. The paper compiles and organizes a wide range of state-of-the-art PU learning methods, covering various algorithmic types and multiple public datasets. Moreover, for several advanced methods that were not publicly released, the authors conducted reimplementation efforts, which is commendable and worth encouraging.

2. Through extensive experiments, the paper clearly demonstrates the generality, robustness, and evaluation effectiveness of the proposed framework, highlighting the value of PU-BENCH in fostering reproducible and comparable research within the PU learning community.

### Weaknesses
1. In line 77, within the Data Sampling Scheme section, the description of the two sampling schemes is somewhat complex. It might be clearer and more intuitive if the authors could provide a simple illustrative example—such as showing how data are obtained under each sampling method—or include a figure or diagram to visualize the difference between the two schemes.

2. Although the paper presents a comprehensive benchmark for PU learning, including a unified data generation pipeline and an integrated framework, it lacks an analysis of the rationale behind these design choices. It would be helpful if the authors could explain why this particular configuration is reasonable and what criteria or considerations guided the design decisions.

3. The paper does not explain why these specific datasets were selected, which weakens the justification of the experimental setup. Providing reasoning for the dataset and method selection would make the evaluation more convincing.

### Questions
Please reply to my comments listed in "Weaknesses".

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors developed PU-Bench, an open-source framework that addresses the critical lack of standardized evaluation in PU learning research. This modular system includes a configurable data generator, unified training pipeline, and comprehensive evaluation suite. Extensive experiments are presented to compare 16 SOTA methods across datasets.

### Strengths
Through the results presented in this paper, the authors provides actionable insight for algorithm selection based on data modality, label availability and computational constraints. The key findings reveals valuable insight. No single method performs better universally, performance was highly dependent on the specific data type being used. The study also revealed clear trade-offs between predictive effectiveness and computational costs. All methods showed performance degradation when moving from random to biased labeling scenarios, though Risk-Minimization methods demonstrated greater robustness.
This work addresses fundamental challenges in PU learning research by establishing standardized evaluation protocols that enable fair method comparison and reproducible results.

### Weaknesses
1. The paper lacks statistical significance testing to determine whether performance differences between methods are meaningful or due to random variation. Results from single random seeds without multiple runs raise questions about reliability.
2. Also, fix data splits could affect the generalizability of findings.

### Questions
What is the variance in performance across different random initializations, and how does this affect method rankings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PU-Bench, the first unified open-source benchmark for Positive-Unlabeled (PU) learning. It provides a standardized benchmark with a unified data generation pipeline, a framework of 16 state-of-the-art PU methods, and protocols for reproducible assessment. Through studies on 8 diverse datasets (2,560 evaluations), PU-Bench reveals trade-offs between effectiveness and efficiency, robustness and label frequency, and selection bias. This benchmark is meaningful for PU learning.

### Strengths
* The benchmark is well-designed with a unified data generation pipeline, a collection of state-of-the-art methods, and standardized evaluation protocols. 
* The benchmark provides a more comprehensive setting and evaluation for PU learning, which is expected to promote the development of the field.
* The empirical study is extensive, covering multiple datasets and various conditions.

### Weaknesses
* While the benchmark itself is meaningful, the methods included are existing ones and lack certain theoretical analysis.
* Certain methods perform very well on some datasets but poorly on others. Additional visualizations and analyses of failure cases could provide valuable insights.
* The benchmark should be expandable, and experimental details can be specified via YAML. Some representative methods should also be compared [1-3].
[1] GradPU: Positive-Unlabeled Learning via Gradient Penalty and Positive Upweighting. AAAI2023
[2] Positive Distribution Pollution: Rethinking Positive Unlabeled Learning from a Unified Perspective. AAAI2023
[3] Positive-unlabeled learning with label distribution alignment. TPAMI2023

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
