# Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy

- Decision: Reject
- Scores: 4, 8, 2, 6

## Abstract
Time series anomaly detection (TSAD) is a critical task, but developing models that generalize to unseen data in a zero-shot manner remains a major challenge. Prevailing foundation models for TSAD predominantly rely on reconstruction-based objectives, which suffer from a fundamental objective mismatch: they struggle to identify subtle anomalies while often misinterpreting complex normal patterns, leading to high rates of false negatives and positives. To overcome these limitations, we introduce TimeRCD, a novel foundation model for TSAD built upon a new pre-training paradigm: Relative Context Discrepancy (RCD). Instead of learning to reconstruct inputs, TimeRCD is explicitly trained to identify anomalies by detecting significant discrepancies between adjacent time windows. This relational approach, implemented with a standard Transformer architecture, enables the model to capture contextual shifts indicative of anomalies that reconstruction-based methods often miss. To facilitate this paradigm, we develop a large-scale, diverse synthetic corpus with token-level anomaly labels, providing the rich supervisory signal necessary for effective pre-training. Extensive experiments demonstrate that TimeRCD significantly outperforms existing general-purpose and anomaly-specific foundation models in zero-shot TSAD across diverse datasets. Our results validate the superiority of the RCD paradigm and establish a new, effective path toward building robust and generalizable foundation models for time series anomaly detection. The code is available in https://anonymous.4open.science/r/TimeRCD-5BE1/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new foundation model TimeRCD for zero-shot time series anomaly detection (TSAD) that overcomes the limitations of reconstruction-based approaches. Traditional TSAD models often fail to detect subtle or contextual anomalies and frequently misclassify complex normal patterns due to an objective mismatch between reconstruction and anomaly detection. To address this, the authors propose a new pre-training paradigm called Relative Context Discrepancy (RCD), which trains the model to identify anomalies by comparing discrepancies between adjacent time windows rather than reconstructing inputs. Built on a standard Transformer architecture, TimeRCD captures relational dependencies across time and uses a large-scale, fully labeled synthetic corpus that provides diverse anomaly patterns for robust pre-training. Extensive experiments across several datasets demonstrate that TimeRCD achieves state-of-the-art performance in zero-shot settings and remains competitive with fully supervised baselines.

### Strengths
S1. TimeRCD demonstrates impressive performance in strict zero-shot settings, outperforming or matching specialized and fully supervised models on diverse datasets.

S2. The paper designs a large-scale, fully labeled synthetic corpus that captures diverse and complex anomaly types, including point, contextual, collective, and cross-variate anomalies.

S3. The experiments demonstrate that performance improves predictably with larger synthetic datasets, indicating scalable pre-training benefits similar to those observed in NLP and vision foundation models.

### Weaknesses
W1. While the paper includes some ablation analysis (e.g., testing the effect of synthetic data), it lacks fine-grained ablations to isolate the contribution of individual components within the TimeRCD framework. For instance, the effect of auxiliary anomaly head and RCD window  is not examined in detail. Without such targeted ablations, it is difficult to clearly attribute which design choices most contribute to performance gains.

W2. The paper does not provide sufficient efficiency or scalability analyses. Since TimeRCD relies on extremely long context windows (up to 13k timesteps), understanding its computational feasibility is crucial for real-world deployment. The absence of such experiments leaves open questions about whether the model can scale efficiently to resource-constrained or latency-sensitive environments.

W3. Although the paper compares against several established baselines, it omits more recent or stronger full-shot models—such as modern deep contextual detectors (e.g., DCdetector[1] and TFMAE[2]). Including these would provide a clearer picture of where TimeRCD stands relative to the current state of the art in both zero-shot and fully supervised regimes.

[1] DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection

[2] Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of zero-shot time series anomaly detection (TSAD). It argues that existing foundation models, which predominantly rely on reconstruction-based objectives, suffer from an "objective mismatch". This mismatch causes them to miss subtle anomalies (false negatives) and misinterpret complex normal patterns (false positives). 
The paper introduces a new paradigm of training a transformer based “foundation model” (TimeRCD) explicitly on an anomaly detection objective, using a synthetic dataset generated through a novel multi-stage generation method.
The paper provides empirical validation of their method’s superiority and some analysis on the results.

### Strengths
Originality: Good

Quality: Fair

Clarity: Fair 

Significance: Good

Additional note: The paper shows clever implementation of techniques that are SOTA and well established in other domains. The synthetic data generation process is especially well thought out.

### Weaknesses
Training of TimeRCD: TimeRCD is trained using a multi-objective setup (anomaly detection and reconstruction), the motivation for this setup is not provided, and it somewhat clashes with their claim that reconstruction is a poor objective for anomaly detection.

Context window size: The paper states the benefit of having a large context window size, however it does not address the quadratic computation costs of the attention mechanism, or that transformers perform worse on sequences of lengths that it did not see at training time. There is some result analysis on context length provided but a deeper investigation should be done. I feel RQ2 was not answered fully.

Unsubstantiated assertions: In section 1, the paper asserts that reconstruction models aim to capture dominant patterns and miss complex ones. No empirical analysis or previous work was cited to support these assertions.
Missing appendix: Appendix not provided.

Exclusion of results: Table 1 shows results of the paper’s proposed approach against existing methodologies. Some results (which outperform TimeRCD) are excluded due to data leakage. Details of the data leakage is not provided.
Repetition in text: Section 2 has repetition of what is already detailed in Section 1.
Missing related work section, although relevant work was cited, it was not discussed in depth.

### Questions
Over-all this paper can be a useful contribution to the community if some additional analysis and motivation of design choices are better provided. Appendix and related work section is not provided, the rating can be improved if clarifications are provided.

Please provide an appendix.

Please elaborate on the motivation for the reconstruction objective.

For the endogenous anomaly injection mechanism the generator creates distinct labels for the "root-cause" anomaly and its "propagated effects". Does the trained TimeRCD model learn to distinguish between these? Or does it simply flag the entire anomalous segment (root and propagation) as anomalous? 

Please provide analysis on the performance of TimeRCD for sequences of lengths it did not see in training data, a more comprehensive description of the synthetic dataset would be good.

Provide motivation to not finetune TimeRCD as another method for comparison. 

A clarification on how a multi-variate time series was flattened to a single sequence would be appreciated.

Cutting out repetitive text would make it a better read. Especially in section 1 and section 2.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TimeRCD, a foundation model for zero-shot time series anomaly detection that addresses fundamental limitations of reconstruction-based approaches. The authors propose a novel pre-training paradigm called Relative Context Discrepancy (RCD), which explicitly trains the model to identify anomalies by detecting significant discrepancies between adjacent time windows rather than learning to reconstruct inputs. To enable this approach, they develop a large-scale synthetic data generation engine that creates diverse time series with token-level anomaly labels, including contextual and causal anomalies with cross-variate propagation. Using a standard Transformer architecture, TimeRCD achieves superior zero-shot performance across 14 benchmark datasets compared to existing general-purpose and anomaly-specific foundation models, demonstrating particular strength in detecting subtle contextual anomalies that reconstruction-based methods often miss.

### Strengths
1. Great effort on constructing a huge dataset
2. A lot of experiments

### Weaknesses
1. Limited novelty because (1) looking at adjacent time windows to detect anomaly is nothing new, (2) model is not special, and (3) equation-based data generation is always data-dependent (i.e. might work on dataset A but not on dataset B).
2. None of the experiments include statistical significance tests. For example, they do not report the results of 10 runs with mean and standard deviation to test for significance.
3. Performance not too great especially when compared to DADA
4. Unclear explanation of how the model is trained using reconstruction error.

### Questions
1. In Section 2.2, is Phase equivalent to Stage (in Figure 3)?
2. How do you perform hyperparameter search, especially for window size, when there are no anomalies in the training set?
3. Some font sizes in figures are too tiny
4. Is the input masked or not? In "Output Projection and Anomaly/Reconstruction Head", the text mentioned "masked portions of the input series" but I don't see this mentioned in any other sections.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a foundation model for zero-shot time series anomaly detection by leveraging large-scale synthetic data pretraining and a proposed Relative Context Discrepancy methodology. The proposed TimeRCD model achieves top zero-shot results on several datasets.

### Strengths
* The paper presents a structural and comprehensive framework for synthetic time series anomaly generation and detection.
* It offers systematic experiments and well-designed ablation studies, which effectively demonstrate the contributions of the proposed components.

### Weaknesses
* Incomplete evaluation on benchmarks and scalability analysis.
* The methodology section lacks clarity, especially in defining key concepts such as RCD and the computation of anomaly scores.

Please find the detailed comments in the following section.

### Questions
* The computation of the anomaly score from the anomaly head is insufficiently explained. Specifically, the definition and mathematical formulation of Relative Context Discrepancy (RCD) remain unclear. How is the discrepancy quantified and how does it integrate into the final detection objective?
* In Section 2.2, the paper introduces two types of labels - localization and detection. It would be helpful to clarify the role of localization labels in model pretraining and downstream evaluation.
* The authors emphasize the importance of capturing causal dependencies among channels, yet it is unclear how the proposed TimeRCD model models these dependencies. How does the framework evaluate multivariate interactions “in light of their causal dependencies”? More discussion would strengthen this claim.
* The study only utilizes a subset of TSB-AD benchmark datasets rather than the full collection. What is the rationale for this selection? Including the full benchmark would provide a more comprehensive and standardized evaluation.
* The omission of strong statistical baselines (such as Sub-PCA or KMeansAD) weakens the comparative analysis.
* What are the context input window sizes used in Section 3.3, and how is the window length $W$ defined and selected as described in Section 2.1? Additionally, an analysis of the sensitivity of this critical hyperparameter would provide insight into the model’s robustness.
* Including a distributional overview of time series lengths and the number of variates in the pretraining datasets would provide more transparency regarding dataset diversity. Moreover, the work would benefit from a discussion of how flattening multivariate time series affects performance and whether causal modeling across dimensions mitigates this limitation.
* The paper lacks runtime efficiency results for TimeRCD compared with baselines.

### Soundness
3

### Presentation
2

### Contribution
3
