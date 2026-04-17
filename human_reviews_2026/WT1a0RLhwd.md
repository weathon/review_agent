# MFRM: Masked Frequency-Refined Modeling for Multivariate Time Series Anomaly Detection

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Frequency-domain information can reveal complex characteristics such as periodicity and seasonality in time series, playing a crucial role in multivariate time series anomaly detection. Since the frequency domain features a long-tailed distribution, existing temporal reconstruction models exhibit a fundamental bias toward the information-concentrated low-frequency bands, while severely underutilizing the discriminative power of fine-grained frequency details, making the detection of complex anomalies particularly challenging. In this paper, we introduce MFRM, a novel reconstruction model that strategically leverages frequency-domain information for enhanced anomaly detection. Our key innovation lies in a learnable frequency masking module that adaptively identifies and extracts frequency components most correlated with normal behavioral patterns, enabling fine-grained frequency details utilization. Furthermore, by disrupting the original spectrum of anomalous series through its frequency masking mechanism, the MFRM exacerbates reconstruction difficulties for anomalies in the time domain and offers a novel perspective to mitigate the over-generalization issue. Extensive experiments on seven benchmark datasets demonstrate MFRM's state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on reconstruction-based time series anomaly detection in the frequency domain, where not all frequency components contribute equally, and over-generalized reconstruction models may also reconstruct anomalies well. To address these issues, it proposes a self-learning model generating a mask in the frequency domain, and demonstrates its effectiveness on several widely used anomaly detection benchmarks.

### Strengths
1. The paper is logically clear, well-structured, and easy to follow. In addition, the figures are well-designed and visually appealing.

2. The proposed model is technically sound, with intuitive technical rationality and reliability.

3. The paper shows certain advantages over state-of-the-art methods. Moreover, by comparing the proposed mask generation approach with other alternatives, the authors demonstrate its effectiveness.

### Weaknesses
1. In the introduction, the authors aim to address two issues — that not all frequency components are equally important and that the model tends to over-generalize. However, the experiments do not directly validate whether these two problems have been effectively solved.

2. The motivation for addressing these issues is not sufficiently elaborated. For example, regarding the unequal importance of frequency components, the paper does not clearly explain the negative effects this problem may cause; for the over-generalization issue, it remains unclear whether current SOTA methods indeed suffer from this problem, which could have been supported by empirical evidence.

3. In the methodology section, the paper primarily focuses on how the proposed method works, but rarely discusses why the proposed design can effectively solve the stated problems.

### Questions
1. Could the authors provide additional experiments to demonstrate that the baseline methods indeed exhibit over-generalization, leading to anomalies being well reconstructed?

2. Could the unequal importance of frequency components be discussed in more detail, particularly regarding its negative impact on time series anomaly detection?

3. Could the authors show the time overhead and memory overhead of MFRM and baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a novel reconstruction model that strategically leverages frequency-domain information to enhance anomaly detection. It introduces a learnable frequency masking module to extract frequency components most correlated with normal behavioral patterns, thereby enabling the utilization of fine-grained frequency details.

### Strengths
S1. A learnable frequency masking module is proposed to fully exploit the discriminative capability of finegrained
frequency details.

S2. By disrupting the original spectrum through the frequency masking mechanism, the reconstruction
difficulty for anomalies in the time domain is exacerbated, thereby alleviating the over-generalization problem
of anomalies.

### Weaknesses
W1. What specific frequency components are considered redundant, and how does such redundancy affect
anomaly detection performance?

W2: The notation in Section 3 is inconsistent. The paper defines the x_i, but later (e.g., Eq. (5)) uses X_t,:. These
expressions are semantically equivalent, yet the inconsistent use of indices and subscripts introduces
unnecessary confusion.

W3. In Section 3.1, the Primary Temporal Modeling (PTM) stage employs only a simple embedding layer
followed by a standard Transformer for sequence reconstruction.

W4. The paper states that the proposed method is inspired by MCM, which was shown to be effective for
tabular anomaly detection. However, MCM’s validation is based on structured tabular data, which differ
significantly from time series in both distributional and anomalous characteristics. Why MCM can be directly
transferred to time series anomaly detection and is equally applicable and advantageous?

W5. How is the set of learnable vectors R generated?

W6. The experimental evaluation includes comparisons with nine deep learning–based methods but omits
classical statistical learning baselines (e.g., Isolation Forest, LOF, OCSVM). These methods are still
representative in early MTSAD research and sometimes outperform deep models in certain regimes.

### Questions
Q1: Which specific frequency ranges or feature patterns are referred to as redundant frequency information,
and how are these redundant components identified or determined?

Q2: Why do the authors use inconsistent mathematical notation throughout the paper?

Q3: Could the authors further elaborate on the design motivation of the PTM stage and clarify its independent
role within the overall model framework?

Q4: Why are classical statistical learning methods not included in the comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MFRM (Masked Frequency-Refined Modeling), a reconstruction-based method for multivariate time series anomaly detection that addresses two key challenges: traditional models' bias toward low-frequency bands while underutilizing fine-grained frequency details, and the over-generalization issue where models reconstruct both normal and anomalous data well. The core innovation is a learnable frequency masking module that adaptively extracts frequency components correlated with normal patterns and filters out anomalous frequencies, combined with a two-stage architecture (PTM for initial reconstruction, FRM for frequency-refined modeling) trained with adversarial learning. The method disrupts the original spectrum of anomalous series through frequency masking, making their reconstruction more difficult in the time domain. Experiments on seven benchmark datasets demonstrate state-of-the-art performance, with average improvements of 10.74% in AUC-ROC and 12.18% in AUC-PR over the previous best method.

### Strengths
1. Well written
2. A lot of experiments and discussions
3. The hard frequency masking is novel

### Weaknesses
1. The frequency masking seems to mask high-frequency components in all the visualizations shown. This is obvious and similar to what a model in the time domain would do (although it's easier to do so in the frequency domain).
2. None of the experiments include statistical significance tests. For example, they do not report the results of 10 runs with mean and standard deviation to test for significance.
3. The performances are not that great.

### Questions
1. Why is the anomaly score multiplicative? With a multiplicative anomaly score, all three loss components have to be high to be classified as an anomaly. This doesn't make sense to me, since a time series might only have an anomaly in either the time domain or the frequency domain, but not both.
2. Are the Transformers in PTM and FRM shared?
3. There might be some anomalies in the training set for some of the datasets. How does that affect the behavior of frequency masking?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents MFRM, a reconstruction-based framework for multivariate time series anomaly detection that leverages adaptive frequency masking. A learnable module selects task-relevant spectral components to counter over-generalization and low-frequency bias in prior models. The two-stage architecture consists of temporal reconstruction and frequency-refined modeling with adversarial attention alignment, The proposed method achieves state-of-the-art results on seven benchmarks.

### Strengths
•	Identifies and visualize the issues of low-frequency bias and reconstruction over-generalization.
•	The proposed frequency masking module with attentive routers and binarization is well-designed.
•	Strong performance across multiple datasets and metrics, also with ablation and analysis of sensitivity.

### Weaknesses
-	There is limited discussion for the stability of the the score aggregations used in MixScore. Could further include the analysis on alternatives, such as weighted sum.
-	There is no deep theoretical analysis of critical properties such as convergence under adversarial learning or guarantees regarding the selection of “meaningful” frequencies.
-	Figure 5 shows that frequencies selected by the module vary across datasets. However, there is limited insight into the resulting frequency masks correspond to interpretable spectral regions or others.

### Questions
1.	How does inference scale with time series length and dimensionality?
2.	If selecting alternative aggregation methods, how about the performance of MixScore?

### Soundness
3

### Presentation
3

### Contribution
3
