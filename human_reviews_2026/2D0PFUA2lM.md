# TreeSNNs: Temporal Resolution Ensembled SNNs for Neuromorphic Action Recognition

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Spiking Neural Networks (SNNs) are energy-efficient due to sparse and asynchronous event processing, but their accuracy often lags behind that of conventional ANN deep learning models. Prior research has largely focused on novel SNN architectures and training methods, with continuous event streams typically binned into frames using either fixed event counts or fixed time intervals. In this paper, we observe that different motions exhibit distinct temporal dynamics and may be best captured at different temporal/event resolutions. Building on this insight, we propose TreeSNNs, an ensemble framework where model diversity is expressed via multiple event temporal resolutions. We utilize the Fano factor as a metric to quantify temporal dynamics and guide the selection of such a diverse set of temporal resolutions tailored to a given dataset. Individual SNNs trained at these resolutions are then aggregated through ensembling to improve recognition accuracy. Experiments on three neuromorphic action datasets—DVS Gesture, SL-Animals DVS, and the challenging THU$^{E\text{-}ACT}$-50 CHL—show that TreeSNNs consistently outperform baselines, improving accuracy by 1.05\%–6.8\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TreeSNNs, an ensemble framework for neuromorphic action recognition that aims to improve accuracy by leveraging multiple temporal resolutions. The core idea is that different actions are best captured at different event binning resolutions (Nf). The authors introduce the Fano factor as a metric to quantify the temporal dynamics of actions and use it to select a diverse set of resolutions. Individual SNN models are then trained for each selected resolution, and their predictions are aggregated via a weighted average to produce the final classification.

### Strengths
1. The central motivation of the paper is insightful. The observation that a single temporal resolution is suboptimal for capturing the diverse dynamics of all action classes is valid and presents a novel angle for improving SNN performance.

### Weaknesses
1. The most critical flaw of this paper is the total omission of any efficiency metrics (latency, training cost, and energy consumption). The primary motivation for using SNNs is their efficiency. The proposed TreeSNNs framework requires training and running N separate models, which inherently multiplies the training time, inference latency, and energy consumption (e.g., total SynOps) by a factor of N. 

2. The core technical contribution (Fano factor) is not convincingly justified by the experiments. The ablation study (Figure 5) shows that on simpler datasets like DVS-Gesture and SL-Animals DVS, the Fano factor-based selection performs on par with random selection.  A simpler, random approach appears to be a competitive baseline, which diminishes the novelty of this core contribution.

3. The ensemble method itself is a straightforward weighted average based on training set error. This is a very basic technique. Given the multi-model setup, more advanced and powerful ensemble strategies like Stacking—where a meta-learner is trained on the outputs of the base models—were not explored. A meta-learner could capture more complex inter-model dependencies and potentially yield better performance. 

4. The paper lacks crucial implementation details.

### Questions
1. Could you provide a detailed analysis of the efficiency of TreeSNNs? 

2. Why did you opt for a simple weighted average instead of a more powerful technique like Stacking? A trainable meta-learner seems like a natural fit for this framework.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The article uses the Fano factor to measure the inter class gap at different resolutions between different classes, in the hope of achieving better final results by integrating models of different resolutions. The paper demonstrates the complementary characteristics of different resolutions under the rich temporal dynamics of the dataset itself. The writing method and experimental setup of the paper are relatively complete.

### Strengths
The author's paper writing and experimental presentation are relatively complete. Using Fano representation to differentiate the differences between different classes is a relatively novel choice and has indeed achieved benefits.

### Weaknesses
1).The method of using different models for fusion is quite common, and the method used in this paper can be naturally applied to ANN. What is the main role of SNN in this paper?The integration scheme naturally increases the number of model parameters and computational complexity, weakening the computational advantage of SNN

2)3.1 Figures 3, 6, and 12 do not match with Figures 3, 6, and 9 in the text

3)The paper does not specify the number of sampling samples for fano analysis, which is a computationally intensive task, especially for large training data.

4)The analysis of the relationship between the same training time, total number of parameters, and the selection of larger models for expansion time steps in the absence of integration in the paper.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an ensemble framework for spiking neural networks (SNNs) where model diversity is induced by training separate SNNs on different temporal resolutions of the same event stream. A Fano-factor–based selection procedure is introduced to choose a subset of $N_f$ values that maximizes inter-class separability across temporal resolutions. At inference, per-model predictions are weighted by training accuracy and averaged.

### Strengths
1. Intuition is clear. The paper convincingly argues that actions exhibit different temporal dynamics and are better captured at different event resolutions.
2. Using the Fano factor to summarize temporal variability and drive selection of $N_f$ is a neat, data-driven idea; the selection objective is well specified.

### Weaknesses
1. Added complexity and training cost not quantified. The method trains multiple SNNs (often 3–9 models, Fig. 6). The paper does not report wall‑clock training time, GPU hours, memory footprint, parameter counts×ensemble size, or energy/latency at inference.  The omission makes it hard to judge practical value. 
2. The method requires computing Fano factors across all samples and frames gathered at all $N_f$ values. The method then performs a exhaustive combinatorial search over $N_f$ sets over each datasets. If the dataset scale is larger, this process can become costly. 
3. Modest gains on two datasets vs. substantial added cost. On DVS Gesture, TreeSNNs improves from 94.44% to 95.49% (Table 1), a +1.05% gain over SOTA; on SL‑Animals, TreeSNNs reaches 93.98%, ≈+2% over prior work—but Table 2b shows the best single $N_f$=20 already achieves 93.98%, i.e., no improvement over the best single model, despite the ensemble’s extra cost. This raises the question of whether temporal ensembling (vs. selecting a good $N_f$) is necessary in such cases.
4. Ablations suggest selection helps inconsistently. Fig. 5 shows the Fano-based selection is on par with, or sometimes below, strong random combinations for DVS Gesture and SL-Animals; only THUE-ACT‑50 CHL shows consistent superiority. This weakens the case that the proposed selector is necessary beyond simple ensembling.
5. Limited discussion of scalability to larger datasets and models.
    While THUE‑ACT‑50 CHL is larger than DVS Gesture/SL‑Animals, the paper does not study truly large‑scale settings, nor does it explore larger backbones. Fig. 6c also shows that increasing $N_{\text{models}}$ can reduce accuracy, complicating the scaling story.

### Questions
1. What is the runtime of computing Fano factors and exhaustively evaluating all combinations?
2. Weighting by training misclassification may overfit. Have you tried validation-set weights, temperature scaling, or a meta‑classifier over logits? Any sensitivity analysis to class imbalance when using training misclassification for weights?

### Soundness
1

### Presentation
3

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
This paper focuses on the problem that SNN performance is limited in event-camera–based action recognition. Traditional methods typically segment the event stream using fixed time intervals or fixed event counts to form Tframes for SNN input. However, the authors point out that different action categories exhibit different temporal dynamics, and thus a single temporal resolution is insufficient to capture effective temporal information for all classes. To address this issue, the authors propose the TreeSNNs framework. The core idea is to split the event stream using multiple temporal resolutions, train an individual SNN model for each resolution, and ensemble these models to improve recognition accuracy. The paper further employs the Fano factor to measure event variation for each class under different temporal resolutions, allowing the selection of a set of resolutions with high discriminability prior to model training and weighted ensembling.

### Strengths
The paper proposes modeling the event stream using multiple temporal resolutions and improves action recognition performance by independently training multiple SNNs and then ensembling them. The method is well-motivated, as it explicitly targets differences in temporal dynamics across action classes. The authors use the Fano factor as a metric to select discriminative temporal resolutions, providing a principled basis for multi-resolution configuration. They also analyze performance differences across single-resolution models and demonstrate class-level complementarity among models, which further supports the effectiveness of the multi-resolution ensemble. In addition, the method does not require modifications to network architectures and can be easily integrated into existing SNN pipelines.

### Weaknesses
1.The method requires training multiple SNN models independently, and inference also relies on ensembling multiple models. Although the performance improvement is noticeable, the training, storage, and inference costs may increase significantly, which is not fully aligned with the motivation of SNNs being “energy-efficient.” In addition, the paper does not provide quantitative analysis of energy consumption or computational cost. It is recommended to include measurements of energy cost during training and inference, comparisons with single-model SNN or ANN approaches, and discussion regarding the trade-off between energy consumption and performance.
2.The experimental comparison does not include existing event-slicing methods mentioned in the related work section or recent literature in this area. It is recommended to include comparative experiments with these representative approaches.
3.Although a Fano factor–based selection strategy is presented, it requires enumerating combinations over multiple temporal resolutions. The complexity thus grows rapidly as the number of candidate resolutions increases. Moreover, the robustness of this strategy under class imbalance or noisy data is not demonstrated.
4.While experiments show class-level complementarity among models trained with different Nf, the paper does not provide theoretical explanations for why such complementarity emerges, nor does it offer mathematical modeling or mechanistic analysis regarding the relationship between temporal resolution and class characteristics. Adding such analysis would further strengthen the paper.
5.The reference “Rethinking Spiking Neural Networks from an Ensemble Learning Perspective” appears twice in the bibliography (line 502 and line 509).

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
