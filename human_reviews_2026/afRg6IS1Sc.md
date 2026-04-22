# Rethinking Heavy Models in Multivariate Time Series Anomaly Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Multivariate time series anomaly detection (MTS-AD) is widely used, but real-world deployments often face tight computational budgets that limit the practicality of deep learning. We revisit whether heavy deep models (high-FLOPs architectures) are necessary to achieve strong detection performance in such settings. We conduct a systematic, compute-aware comparison of statistical, classical machine learning, and deep learning methods across diverse MTS-AD benchmarks, measuring detection with AUROC (threshold-free, thus application-agnostic) and cost with FLOPs (a hardware-agnostic proxy enabling fair cross-method comparison). We find that traditional approaches often match or surpass deep models, which appear less frequently among the top performers, and that the effectiveness-efficiency trade-off commonly favors non-deep alternatives under limited budgets. These results indicate that deep learning is not uniformly superior for MTS-AD and that heavy architectures can be counterproductive in resource-constrained deployments. These findings offer practical guidance for practitioners designing anomaly monitoring systems under compute constraints, highlighting cases where lightweight models are sufficient and heavy deep models may be worth the cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies multi-variate time series anomaly detection in the context of accuracy and compute efficiency.  It carries out a comparison of statistical, classical machine learning, and deep learning based methods through the lens of efficiency and finds that often non-deep learning alternatives perform better under limited compute budgets.

### Strengths
It is a well-written paper that poses an important question and seeks its answer by putting together a systematic study. Section 3.2.1 outlines the principles and reasoning that guided the selection of anomaly detection methods and benchmarks used in this study.  The paper uses FLOPs as a measure of computational resources required by a method and AUROC as the measure of performance.  Table 3 that captures performance vs. accuracy supports the assertion that deep learning model, despite having large computational requirements, rarely achieve performance that is "far better" than what is achieved by other techniques.

### Weaknesses
I don't quite get Figure 2.  Especially, why the individual dots connected by dashed lines ... aren't these discrete measurements?

AUROC is useful but insufficient for deployment.  We require thre threshold protocol, calibration and robustness to threshold selection, and domain specific accuracy requirements in order to decide which of the given list of methods is most useful in a given setting.  I would hazard a guess while the results are insightfull, these lack actionable information that a practitioned may be able to use to deploy a multi-variate time-series anomaly detection in industrial settings.

### Questions
Most anomaly detection methods require some sort of a threshold to make the final estimation whether or not an anomaly as occured.  This work seems to downplay the effect of threshold selection on the overall accuracy.  How AUROC is computed in the absence of threshold, or how thresholds were selected (where needed)?  Is it possible that some methods are less sensitive to threshold selection?  Perhaps this is a factor that should also be taken into account in a study such as the one proposed in this paper?

Imagine we are given a method that has very low compute requirements as measured by FLOPs, but also very low accuracy as measured by AUROC.  Where would this method sit within the comparisons presented in Table 3?  Also, when we search for acceptable performance vs. accuracy balance, we often have a some minimum accuracy requirements, which may be different in different domains.  How would that play out in the recommendations outlined in this work?

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
4

### Summary
This work focuses on the problem of multivariate anomaly detection in limited-resource scenarios, as encountered in many real-world applications. To that end, this experimental work proposes to answer the following two questions:

- What are the most effective options for time series anomaly detection under limited computational resources, and are deep learning methods always the best options? 
- Does a trade-off between detection performance and computational cost truly exist in practice? 

To answer the questions, the paper carries three different experiments to assess performance vs efficiency: 1) trade-off between detection accuracy, measured by AUROC, and the computational cost quantified by training and inference FLOPs; 2) FLOPs (algorithmic operation counts) vs FLOPS (hardware throughput comparison; and 3) scalability as a function of data volume.

### Strengths
- This is a well-written paper with a clear experimental setup that aims to address a practical, but relevant question.
- Good coverage of baselines
- Insightful conclusions

### Weaknesses
Overall, this is a good paper. I see as a weakness that this may not be the typical paper expected in ICLR (which motivates my score), but I do not see major weaknesses for an experimental paper. 

- As hardware is central to this work, it would have been good to that the different hardware configuration is reported in the main paper.

### Questions
Similar efforts [1], though not focused on resource constraints, have investigated the advantages of traditional vs. deep learning based approaches. Could you position your work with respect to this one and establish similarities and differences?

[1] https://doi.org/10.1016/j.patcog.2022.108945

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits the necessity of heavy deep learning architectures for multivariate time series anomaly detection under resource-constrained environments. By introducing a hardware-agnostic measure (FLOPs) as a proxy for computational cost, the study contributes to the discussion on the effectiveness-efficiency trade-off in anomaly detection.

### Strengths
* The use of a hardware-agnostic FLOPs measure for runtime evaluation is well-motivated and provides a fair framework for cross-model comparison.
* The derivation and formulation of FLOPs for popular TSAD algorithms are clearly described.
* The paper presents a comprehensive evaluation of various models with respect to FLOPs, offering insights into the trade-offs between performance and computational efficiency.

### Weaknesses
* The study is constrained by limited datasets, algorithms, and evaluation aspects, which may restrict the generalizability of its conclusions.
* Limited theoretical or diagnostic interpretation of accuracy-runtime trade-off between lightweight and heavy models.

Please find the detailed comments in the following section.

### Questions
* Limited TSAD algorithm coverage. The study includes only 16 anomaly detection algorithms, while recent benchmark efforts (e.g., Schmidl et al., 2022 with 70 methods; Liu & Paparrizos, 2024 with 40 methods) have evaluated much larger and more diverse collections.
* Limited evaluation measures. Relying solely on AUROC overlooks known limitations of this measure in time series anomaly detection (e.g., its sensitivity to temporal noise, imbalance in anomaly ratio, and inability to capture temporal localization). Recent works have proposed time-series–aware measures such as Range-F1 [1] and VUS-PR [2], which should be considered for a more robust evaluation.
* Dataset selection limitations. The benchmark omits widely used datasets such as Exathlon [3] and TSB-AD (Liu & Paparrizos, 2024), which represent more diverse operational scenarios and anomaly types. 
* The study mainly investigates multivariate time series but does not address the univariate case. Furthermore, it is unclear how model scalability behaves with respect to both sequence length and feature dimensionality, as no explicit analysis of these factors is provided.
* While adopting FLOPs as a hardware-agnostic proxy is a good practice, complementing it with real runtime measurements and memory footprint analyses across identical hardware would provide stronger evidence for practical applicability in real deployments.
* The paper would benefit from a deeper analysis of when and why lightweight or heavy models perform better. For instance, cases where LSTM-VAE achieves relatively low FLOPs but competitive performance compared to heavier models like LOF need further investigation. Beyond aggregate AUROC scores across the entire dataset, it would be useful to analyze performance across different anomaly types and anomaly ratios to better contextualize the observed trends.

[1] Tatbul N, Lee TJ, Zdonik S, Alam M, Gottschlich J. Precision and recall for time series. Advances in neural information processing systems. 2018;31.

[2] Paparrizos J, Boniol P, Palpanas T, Tsay RS, Elmore A, Franklin MJ. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment. 2022 Jul 1;15(11):2774-87.

[3] Jacob V, Song F, Stiegler A, Rad B, Diao Y, Tatbul N. Exathlon: a benchmark for explainable anomaly detection over time series. Proceedings of the VLDB Endowment. 2021 Jul 1;14(11):2613-26.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an empirical study challenging the necessity of "heavy" deep learning models for multivariate time series anomaly detection (MTS-AD), particularly in resource-constrained environments . The authors conduct a comparative analysis of statistical, classical machine learning (one-class), and deep learning (reconstruction-based) methods across six benchmarks. The study evaluates models on two axes: detection effectiveness (using the threshold-agnostic AUROC metric) and computational efficiency (using a hardware-agnostic FLOPs metric).

The study finds that traditional models (e.g., HBOS, Isolation Forest, ABOD) frequently achieve top-tier AUROC performance, often matching or exceeding their deep learning counterparts . Furthermore, the analysis of AUROC vs. FLOPs (Figure 1) suggests that deep learning models present a poor trade-off, incurring high computational costs without delivering superior performance. The authors conclude that deep learning is not uniformly superior and that lightweight traditional models are often the more practical choice for constrained deployments .

### Strengths
1. The paper asks a timely and important question: are heavy deep models worth the cost for MTS-AD in real-world, constrained settings ? This is a critical concern for practitioners in industrial, IoT, and embedded systems.
2. The joint evaluation of both detection performance (AUROC) and hardware-agnostic computational cost (FLOPs) is a valuable contribution.

### Weaknesses
1. The paper's primary flaw lies in its selection of deep learning models. The entire "Reconstruction" (i.e., deep learning) category consists almost exclusively of older, simpler autoencoder variants (LSTM-AE, LSTM-VAE, USAD, DeepSVDD). These models are no longer representative of the state-of-the-art. The study omits the entire class of modern, high-performance Transformer-based and CNN-based anomaly detectors. The comparison is therefore not "Rethinking Heavy Models" but "Rethinking Outdated Models".
2. The paper incorrectly frames "deep learning" as "heavy" and "traditional" as "lightweight," when its own data often shows the opposite. This invalidates the core narrative.
3. While AUROC is threshold-agnostic, it is notoriously unreliable for anomaly detection on datasets with high class imbalance (which is characteristic of all TSAD benchmarks) . AUPRC (Area Under Precision-Recall Curve) is the standard, more informative metric in this setting. By optimizing for a potentially misleading metric (AUROC), the performance rankings (Table 3) may not reflect true detection quality. This choice further weakens the paper's conclusions.

### Questions
Re: Weakness #1: The paper's conclusions hinge on comparing deep learning to traditional methods. Why did the authors choose to represent the entire deep learning category with only older, reconstruction-based models, while omitting all modern SOTA architectures like Anomaly Transformer, TimesNet, or TranAD, which are the "heavy models" the community is actually discussing today?

Re: Weakness #2: The FLOPs data in Table 3 (e.g., SMD, SMAP) shows that statistical models like ABOD and LOF are orders of magnitude more computationally expensive (higher FLOPs) than deep models like OmniAnomaly or LSTM-AE. How do the authors reconcile this fact with the paper's central narrative that deep learning models are the "heavy" option and traditional models are the "lightweight" alternative?


Re: Weakness #3: Why did the authors choose AUROC as the sole metric for detection performance, given that AUPRC is widely accepted as a far more informative and reliable metric for highly imbalanced anomaly detection tasks?

### Soundness
3

### Presentation
2

### Contribution
2
