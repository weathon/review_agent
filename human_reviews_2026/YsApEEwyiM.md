# HDC-X: Efficient Medical Data Classification for Embedded Devices

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 0, 2, 2

## Abstract
Energy-efficient medical data classification is essential for modern disease screening, particularly in home and field healthcare where embedded devices are prevalent. While deep learning models achieve state-of-the-art accuracy, their substantial energy consumption and reliance on GPUs limit deployment on such platforms. We present HDC-X, a lightweight classification framework designed for low-power devices. HDC-X encodes data into high-dimensional hypervectors, aggregates them into multiple cluster-specific prototypes, and performs classification through similarity search in hyperspace. We evaluate HDC-X across three medical classification tasks; on heart sound classification, HDC-X is $350\times$ more energy-efficient than Bayesian ResNet with less than 1\% accuracy difference. Moreover, HDC-X demonstrates exceptional robustness to noise, limited training data, and hardware error, supported by both theoretical analysis and empirical results, highlighting its potential for reliable deployment in real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces HDC-X, a lightweight and energy-efficient framework for medical data classification on low-power embedded devices. Built upon hyperdimensional computing principles, HDC-X encodes input features into high-dimensional hypervectors, aggregates them into cluster-specific prototypes, and performs classification through similarity search in hyperspace. The method offers a strong balance between accuracy, robustness, and computational efficiency.

Across three medical tasks—including heart sound, breast cancer, and electromyography classification—HDC-X achieves up to 350× higher energy efficiency than state-of-the-art deep learning models such as Bayesian ResNet, with less than 1% accuracy difference. The paper also provides theoretical proofs of robustness against input noise and hardware errors, and presents a conceptual hardware framework demonstrating feasibility for FPGA implementation. Together, these contributions make HDC-X a promising step toward practical, reliable AI-based disease screening on portable and resource-constrained medical devices.

### Strengths
This paper presents an original and well-executed contribution to energy-efficient medical data classification by extending the hyperdimensional computing (HDC) paradigm into a practical, high-performance framework, HDC-X. The approach is both conceptually creative and technically sound: it reformulates medical classification as a similarity search in high-dimensional hyperspace using rigorously defined Hamming-distance–based metrics. This yields a framework that preserves the lightweight, brain-inspired nature of HDC while substantially improving accuracy and robustness.

The theoretical formulation is rigorous, with clear proofs and precise definitions that make the methodology transparent and reproducible. The empirical validation is comprehensive—covering multiple datasets, hardware efficiency analysis, and robustness under input noise, limited data, and hardware faults—demonstrating impressive real-world relevance.

### Weaknesses
While the paper demonstrates strong performance on small- to medium-scale medical datasets, it remains unclear whether the proposed HDC-X framework can generalize effectively to larger-scale settings. The current empirical evaluation does not explore the scaling behavior of HDC-X—such as how its accuracy, energy efficiency, or robustness evolve with increasing data size or feature dimensionality. Without such analysis, it is difficult to assess whether the observed advantages over deep neural networks persist when applied to more complex or higher-volume medical data. Future work could strengthen the contribution by providing experiments or theoretical insights into the scalability of HDC-based methods.

### Questions
This is an interesting paper, though the empirical results are somewhat limited. It is important to further investigate the scaling properties of hyperdimensional computing (HDC) to substantiate its claimed energy efficiency. Currently, the experiments are conducted on relatively small datasets, where careful hyperparameter tuning may mask potential weaknesses—particularly the tendency of larger models to overfit or lose efficiency at scale. A more thorough analysis on larger and more complex datasets would help confirm whether HDC maintains its advantage in energy efficiency under realistic, large-scale conditions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes HDC-X, a lightweight and energy-efficient classification framework tailored for medical data analysis on low-power and embedded devices. Building upon Hyperdimensional Computing (HDC) principles, HDC-X encodes medical signals into high-dimensional hypervectors, aggregates them into cluster-specific prototypes (Cluster-HVs), and performs classification via similarity search in hyperspace.

### Strengths
By leveraging hyperdimensional computing (HDC), HDC-X eliminates the need for floating-point operations and GPUs, enabling extremely low-power, memory-efficient inference

### Weaknesses
1. The proposed method primarily reuses the standard HDC pipeline (encoding → bundling → similarity search), with only moderate extension via cluster aggregation. There is no clear conceptual innovation beyond applying HDC to biomedical data, which makes the contribution more application-level than methodological.
2. The paper does not fundamentally address the long-standing issue that HDC-based models often underperform deep neural networks in complex, high-dimensional feature spaces. While the reported 1% gap to Bayesian ResNet is notable, it is unclear whether this holds across more challenging datasets or higher intra-class variability.
3. In practical HDC implementations, the encoding stage (mapping raw features into hypervectors via item memory or continuous item memory) dominates both latency and energy consumption. However, the paper omits reporting encoding cost or how the chosen encoder scales in time, memory, and energy. This is a critical omission because the efficiency advantage may vanish once encoding overhead is included.
4. The paper fixes the hypervector dimension and precision but does not explore how these parameters affect accuracy, energy, or latency. Such analysis is crucial, as HDC accuracy is highly sensitive to dimensionality and quantization (binary vs. int8 vs. float32). Without this exploration, the claimed “energy efficiency” lacks reproducibility and generalization.
5. The reported 350× efficiency improvement appears to be a theoretical estimate rather than measured on actual hardware (e.g., MCU, FPGA, or ASIC). Without concrete energy profiling of encoding, similarity computation, and memory access, the practical significance of this number is uncertain.
6. The experiments are limited to a small set of biomedical datasets (e.g., heart sound classification). Comparisons exclude other lightweight neural or neuromorphic baselines such as quantized CNNs, TinyML architectures, or SNN-based classifiers, which would provide a fairer context for energy–accuracy trade-offs.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents HDC-X, which leverages hyperdimensional computing to enable lightweight medical data classification. The proposed methods do class-wise clustering, which does bundling and optional retraining, after encoding (mapping data to high-dimensional space). The authors claim that HDC-X can achieve similar accuracy to a deep-learning-based method while being up to 350x more energy-efficient for classifying medical datasets.

### Strengths
The propsed method significantly improves energy efficiency and noise resiliency compared to deep neural networks while maintaining comparable accuracy.

### Weaknesses
This paper shows that it can achieve similar accuracy with much less cost compared to a deep learning-based method. However, there are several points that need to be addressed:
1. The huge improvements and robustness of HDC-X come from HDC's strength. The encoding, training (bundling), and retraining methods that are presented in this paper are not new, which are from VoiceHD (Imani 2017) and HDCluster (Imani 2019), etc, so I am not very convinced about the novelty of this work. 
2. The target task can be very easy to achieve with small and lightweight DNNs, but the baseline DNNs seem to be very heavy. Also, at the inference time, the DNN can be compressed with lower precision for better efficiency. 
3. Baseline implementation details, such as HDC, HDCluster, etc, are not very clear, and I think the authors should present HDC-X's difference over baselines, especially HDC-based ones.
4. Theoretical proof seems to be already done by [1], and I would like to know theoretical contribution of this paper.
[1] Anthony Thomas et al. A Theoretical Perspective on Hyperdimensional Computing. JAIR, 2021.

### Questions
Please see weakness section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
HDC-X extends hyperdimensional computing for energy-efficient medical classification by introducing cluster-based prototype hypervectors for each class. The method targets embedded devices where power and latency are critical. Results show strong energy savings relative to deep learning models, and the paper discusses a potential FPGA realization.

### Strengths
The paper contributes an applied demonstration of HDC’s advantages for low-power medical tasks. Although the motivation is compelling, the methodological advancement is modest, and the lack of concrete hardware results limits its impact.
• The energy-efficient focus is timely and important, addressing the need for lightweight edge solutions in healthcare.
• Reported improvements over heavy neural baselines are large and highlight potential for practical deployment.
• The approach reinforces the viability of HDC models in real-time embedded environments.

### Weaknesses
• The algorithmic innovation beyond prior HDC clustering methods is minimal and not clearly isolated.
• Comparisons omit lighter classical baselines, making the energy-efficiency claims somewhat inflated.
• The FPGA analysis remains conceptual, with no real measurements or synthesis details.

### Questions
Could the authors provide actual measured energy or latency results from hardware deployment? How does HDC-X compare to simple non-neural baselines on the same datasets? What specific architectural change differentiates HDC-X from previous prototype-HDC models?

### Soundness
3

### Presentation
3

### Contribution
2
