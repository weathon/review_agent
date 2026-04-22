# PSDNorm: Temporal Normalization for Deep Learning in Sleep Staging

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4, 4

## Abstract
Distribution shift poses a significant challenge in machine learning, particularly in
    biomedical applications using data collected across different subjects, institutions, and recording devices, such as sleep data.
    While existing normalization layers, BatchNorm, LayerNorm and InstanceNorm, help mitigate distribution shifts, when applied over the time dimension they ignore the dependencies and auto-correlation inherent to the vector coefficients they normalize.
    In this paper, we propose PSDNorm that leverages Monge mapping and temporal context to normalize feature maps in deep learning models for signals.
    Evaluations with architectures based on U-Net or transformer backbones trained on 10K subjects across 10 datasets,
    show that PSDNorm achieves state-of-the-art performance on unseen left-out datasets while being 4-times more data-efficient than BatchNorm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PSDNorm (Power Spectral Density Normalization), a normalization layer that can be directly inserted into a network for time-series learning. Its core idea is to perform "power spectrum alignment" on feature maps in the time-frequency domain, allowing the model to learn domain-invariant temporal statistical structures. Experiments on a large-scale sleep dataset demonstrate the effectiveness of the framework.

### Strengths
1. The structure of this paper is clear and reasonable.

2. This paper is easy to follow and generally well presented.

3. Extensive experiments have been conducted to evaluate the performance of the proposed framework.

### Weaknesses
1. The current related work section is more like a preliminary. There should be several subsections introducing works with similar topics, such as methods for training sleep data and normalization in time series.

2. The feature map of the middle layer of the deep network may not meet the prior conditions in the article after nonlinearity and convolution, a Gaussian periodic signal, a covariance block diagonal, and diagonalized according to the Fourier basis. Should we consider the failure mode?

3. The comparison is mainly limited to normalization layers, but the core idea of PSDNorm is based on Monge/OT/PSD alignment. It could be better to compare PSDNorm with existing TMA/STMA or general test-time adaptation methods. It could also compare PSDNorm with TMA, which uses PSD as preprocessing (or TMA embedded in the network).

4. It could be better to consider the parameter sensitivity of the Welch estimator.

5. It should show more results on different testing windows.

6. What about the computational cost of this normalization?

### Questions
See in weakness.

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
This paper introduces PSDNorm, a novel normalization layer designed for deep learning on sequential signals. Unlike standard normalization layers (BatchNorm, LayerNorm, InstanceNorm) that treat time steps independently, PSDNorm leverages the temporal auto-correlation structure of signals by aligning their Power Spectral Density (PSD) to a running Riemannian barycenter via Monge mapping. The method is evaluated extensively on sleep stage classification using 10 datasets comprising ~10K subjects. Results demonstrate consistent improvements over baseline normalization techniques, with particular benefits in data-limited regimes and under domain shift.

### Strengths
1. LODO setup with 10 datasets, proper statistical testing, multiple seeds, and comprehensive ablations set a high standard. The transparency in reporting (including failed torch.compile) is commendable.

2. The use of optimal transport, Wasserstein barycenters, and Riemannian geometry provides solid theoretical foundation, even if the novelty is limited.

3. Data efficiency results (Figure 6, Table 2 bottom) demonstrate real value for resource-constrained scenarios common in medical applications.

4. Algorithm 1, architecture details (Table 4), implementation notes, and comprehensive appendix facilitate reproducibility.

### Weaknesses
1. Limited theoretical justification:
- When do we expect PSDNorm to outperform alternatives? No formal analysis provided.
- Why is the geodesic update (Eq. 6) optimal? Other update schemes not explored.
- Gaussian assumption (Eq. 2) likely violated for deep network features - no empirical validation.
- No analysis of what happens when periodicity assumption fails.

2. Narrow evaluation scope:
- Only sleep staging evaluated - no other time series tasks
- Both architectures are CNN-based - what about pure transformers or RNNs?
- No evaluation on non-biomedical sequential data
- Limits ability to assess general applicability


3. This paper might overclaimed contributions, "4× more data-efficient" is misleading, should say "more robust to data scarcity"; "State-of-the-art" improvements are marginal (often <1%); Title suggests "temporal normalization" but method is specific to signals with spectral content

4. Something missed
- No visualization of learned barycenter PSDs - are they interpretable?
- No analysis of what spectral patterns PSDNorm captures
- Missing failure case analysis
- No discussion of when PSDNorm might be worse than alternatives


5. Computational: 
- Training time overhead 7-17% (Table 8) is non-negligible
- Requires FFT operations which may not be hardware-optimized on all platforms
- No analysis of memory overhead
- The InstanceNorm timing anomaly (torch.compile issue) undermines the comparison

### Questions
1. Can you empirically verify that intermediate feature maps exhibit the assumed covariance structure (Eq. 2)? Visualizations of PSD estimates from actual feature maps would strengthen the claims.
2. Under what conditions (signal properties, network architecture, data distribution) should we expect PSDNorm to outperform alternatives? Can you provide theoretical guidance?
3. What do the learned running barycenter PSDs look like? Are they interpretable? Do they differ meaningfully across layers or datasets?
4. Have you evaluated PSDNorm on other time series tasks (e.g., ECG, audio, speech)? Results on even 1-2 additional domains would strengthen generality claims.
5. Have you tried alternatives to the geodesic update (Eq. 6)? What about learnable momentum α or adaptive filter sizes f?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel deep learning normalization method—PSDNorm (Temporal Normalization Layer)—aimed at addressing the data distribution bias problem in biomedical data (such as sleep staging). Traditional normalization methods (such as BatchNorm, LayerNorm, and InstanceNorm) typically ignore temporal correlations, while PSDNorm, by employing Monogram mapping and temporal context, can effectively normalize feature maps in deep learning models, thereby improving performance on unseen data.

### Strengths
1) The proposed PSDNorm layer effectively addresses the data distribution shift problem in biomedical signals within a deep learning framework, providing a novel perspective and approach for processing complex time-series signals.

2) Experimental results show that PSDNorm exhibits superior performance on multiple large-scale sleep datasets, particularly in handling data distribution shifts, outperforming existing normalization methods (such as BatchNorm and LayerNorm).

3) PSDNorm is a "plug-and-play" normalization layer that can be seamlessly integrated into existing neural network architectures with relatively low computational cost.

### Weaknesses
1) The proposed PSDNorm introduces a filter size hyperparameter (f), which controls the normalization strength. The choice of f can affect model performance across different datasets and tasks. Although the authors provide some default values, automatically selecting this parameter in adaptive settings may still be a challenge.

2) While the experiments cover multiple sleep datasets, the diversity of these datasets remains limited and may not fully represent all challenges in biomedical signals. Future research could consider extending this method to other domains, such as audio processing and other biomedical applications, to further validate its generalization capabilities.

### Questions
See the section on weaknesses for details.

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
The paper introduces a new normalization method called PSDNorm to address the challenges that models applied to sleep staging face. The authors point out challenges in building neural networks for sleep staging, like distribution shift across subjects, devices, and datasets, and they point out limitations in existing normalization layers like BatchNorm/LayerNorm, which ignore temporal autocorrelation in signals. PSDNorm aligns the power spectral density of intermediate representations to a barycenter using an f-Monge mapping. The authors claim that they achieve SOTA generalization under a leave-one-dataset evaluation across 10 datasets, and they find that they are 4x more data efficient.

### Strengths
•	Principled Formulation: The optimal transport on Gaussian models is clean. I can understand why this normalization would be beneficial for OOD generalization.
•	Domain shift: Authors show that PSD has a strong gain, notably with MASS and CHAT, and on the hardest subjects, as shown in Figure 4.
•	Data Efficiency: If the hyperparameter is well-tuned, it seems that PSDNorm requires 4x fewer labeled subjects than BatchNorm.

### Weaknesses
I think this paper needs stronger comparisons in order to better characterize whether PSDNorm is making a real difference. I also think better scoping the claims will help significantly.

•	Normalization replacement: In my opinion, the paper makes a very odd choice in the experiments, as stated in line 386, with mixing normalization strategies. Instead of replacing the entire network, the paper takes a CNN Transformer and replaces the first three BatchNorm layers, keeping all other BatchNorm layers. For TMA, no layer is replaced, and this is treated as a preprocessing step, but TMA uses BatchNorm as well. This design makes comparisons confounded by remaining BatchNorm layers and by different amounts of architectural change across baselines. I think a full replacement is required here for clean comparison. 
•	Channel independence: The method uses per-channel PSDs, from my understanding, as shown in Eq. 2. I think this assumes uncorrelated sensors. For learned feature maps, channel independence is unlikely to hold, so cross-channel correlations are ignored. I think this caps achievable alignment across channels. 
•	Test-time adaptation: This point is more minor. The paper discusses TTA in Section 3.3. However, I think the comparison overlooks key issues with PSDNorm for TTA that make me uncertain about the claims in line 274. PSDNorm does not update its barycenter at inference (Algorithm 1), so it does not adapt to a novel target domain at test time.

### Questions
•	Hyperparameters: The paper fixes \alpha and f, with sensitivity analysis mainly on f. How important is \alpha? Other aspects like Welch window overlap or FFT length could be useful to discuss as well to get a sense of stability and cost.
•	EEG channels: Will PSDNorm actually scale to other channels or modalities?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PSDNorm, a normalization layer for deep learning models on time-series signals, particularly EEG-based sleep staging. Traditional normalization methods (BatchNorm, LayerNorm, InstanceNorm) do not properly handle spectral distribution shifts arising from temporal autocorrelations and subject variability. PSDNorm addresses this issue by integrating Temporal Monge Alignment (TMA) within the network, allowing temporal context to be leveraged during normalization rather than as a separate preprocessing step. The method consists of three stages: 1) PSD estimation, 2) Running Riemannian barycenter update, and 3) Temporal Monge alignment. Experiments are conducted on 10 large-scale sleep datasets (10M samples, 10K subjects) under a Leave-One-Dataset-Out (LODO) setup. Experimental results show that PSDNorm outperforms all baseline methods, BatchNorm, LayerNorm, InstanceNorm, and TMA, demonstrating improved robustness under domain shifts.

### Strengths
- **S1:** The paper addresses an important problem, domain shift in physiological datasets, where variability across subjects and recording conditions poses a major challenge.
- **S2:** The paper is theoretically well-grounded, combining Monge mapping, Riemannian barycenters, and temporal alignment within a unified normalization framework. It is mathematically clear.
- **S3:** Consistent improvements on both USleep and CNNTransformer indicate its strong applicability to other architectures.
- **S4:** The analysis on subject-wise performance demonstrates the robustness of PSDNorm to inter-subject variability, one of the most difficult problems in physiological data modeling.
- **S5:** The idea of normalization based on spectral composition shows strong potential for sleep staging and other biomedical applications where temporal dependencies are critical.

### Weaknesses
- **W1:** The motivation behind PSDNorm is not entirely clear. While the paper states that PSDNorm is inspired by Temporal Monge Alignment (TMA), the concepts and roles of data preprocessing (TMA) and in-network normalization (PSDNorm) are fundamentally different. The rationale for integrating TMA into a normalization layer, and the specific benefits of this design are not sufficiently demonstrated.
- **W2:** Experiments are primarily conducted on large, well-curated datasets. Low-resource or few-subject scenarios are not adequately tested (the balanced@400 setting is still relatively large). For example, can the authors provide the performance of PSDNorm on balanced@40? While the authors argue that PSDNorm is effective in limited-data settings, the result in Table 2 does not fully support their argument (reports only balanced@400, which seems close to medium-scale rather than small-scale).
- **W3:** The normalization baselines are limited to BN, IN, and LN. Broader domain adaptation methods such as AdaBN, DSBN, or Deep CORAL would strengthen the comparative analysis. 
- **W4:**  While PSDNorm targets dataset shift at test time, its behavior across datasets with highly divergent characteristics (Table 1) remains ambiguous. Performance trends in Table 2 do not clearly confirm improved robustness to severe distribution shifts. 
- **W5:**  Although Figure 4 provides subject-level insight, since sleep datasets are often highly imbalanced, a class-wise performance analysis for poorly predicted subjects would provide more insight. The results from Table 2 and Table 5 indicate a higher F1-score compared to BAAC (balanced accuracy), which appears as certain classes dominating the overall improvements. Could the authors provide class-wise performance results for further clarification?

### Questions
Please address the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
