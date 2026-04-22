# AutoDA-Timeseries: Automated Data Augmentation for Time Series

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Data augmentation is a fundamental technique in deep learning, widely applied in both representation learning and automated data augmentation (AutoDA). In representation learning, augmentations are used to construct contrastive views for learning task-agnostic embeddings, while in AutoDA the augmentations are directly optimized to improve downstream task performance. However, existing paradigms face critical limitations: representation learning relies on a two-stage scheme with limited adaptability, and current AutoDA frameworks are largely designed for image data, rendering them ineffective for capturing time series–specific features. To address these issues, we introduce **AutoDA-Timeseries**, the first general-purpose automated data augmentation framework tailored for time series. AutoDA-Timeseries incorporates time series features into augmentation policy design and adaptively optimizes both augmentation probability and intensity in a single-stage, end-to-end manner. We conduct extensive experiments on five mainstream tasks, including classification, long-term forecasting, short-term forecasting, regression, and anomaly detection, showing that AutoDA-Timeseries consistently outperforms strong baselines across diverse models and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AutoDA-Timeseries, a novel framework addressing the specific needs of automated data augmentation for time series analysis, a domain where existing methods often fall short.   Key strengths include its general-purpose design applicable across five diverse tasks, the incorporation of time series features to guide policy generation, and extensive empirical validation showing consistent improvements over baselines. However, the theoretical justification for the controller objective is limited, and the link between learned policies and data characteristics is not well-analyzed. Moreover, some mathematical notations are ambiguous, and the computational cost of the search process is not discussed.

### Strengths
1. Clear Problem Motivation: Section 1–2 convincingly describes the challenge of limited labeled time-series data and the difficulty of designing effective augmentations. The introduction connects to prior works, positioning AutoDA as a general extension for unsupervised or supervised forecasting.

2. Comprehensive Experimental Evaluation: The framework's effectiveness is demonstrated across five distinct and mainstream time series tasks: classification, long-term forecasting, short-term forecasting, regression, and anomaly detection. This wide range of tasks supports the claim of a "general-purpose" framework.  Evaluation spans numerous benchmark datasets relevant to each task (e.g., 26 UEA subsets for classification, ETT/M4/etc. for forecasting). This provides robust evidence for the method's performance across different data domains and characteristics.   AutoDA-Timeseries is tested with diverse downstream model architectures for each task (e.g., TCN/ROCKET for classification, RNN/Autoformer for forecasting, CNN/MLP for regression, UNet/VAE for anomaly detection), demonstrating its compatibility and effectiveness across different modeling paradigms.   Comparisons are made against relevant baselines, including no augmentation, state-of-the-art representation learning methods (InfoTS, AutoTCL, TS2Vec), and recent AutoDA methods (RandAugment, UniformAugment, TrivialAugment, A2Aug), providing a strong context for evaluating performance gains.

3. Scalable and Modular Design: The framework can plug into different backbone architectures, indicating extensibility. Such modularity makes AutoDA useful for practitioners exploring augmentation strategies under resource constraints.

### Weaknesses
1. Lack of Theoretical Justification for Objective: 
- The description of the composite loss (Sec 3.5.2) could be more detailed. While $L_2$ (entropy for intra-batch diversity) is standard, the motivation and formulation for $L_3$ (KL divergence for inter-batch diversity, Eq 10 [cite: 5386]) could be explained more clearly. The notation $p_{i}^{(current)}$ vs $p_{b,j}^{(prev)}$ also seems inconsistent regarding batch indices ($i$ vs $b$).
- The notation for probability and intensity generation in Section 3.4  is slightly ambiguous. It states $p_{i,j}^{(k)}=f_{p}^{(k)}(p_{i,j}^{(k-1)},F_{i})$. Does this mean the probability for transform $j$ only depends on its own probability from the previous layer, $p_{i,j}^{(k-1)}$, or should the input be the entire previous probability vector, $p_{i}^{(k-1)}$? 

2. Insufficient Detail on Baseline Adaptation and Implementation:
- The paper compares against several AutoDA methods originally developed for images (RandAugment, UniformAugment, TrivialAugment, A2Aug). Appendix B provides brief descriptions, but critically lacks detail on *how* these were adapted for time series. Without this, the comparison might not be entirely fair, as these methods might be suboptimal simply due to naive application.
- The implementation details for the representation learning baselines (InfoTS, AutoTCL, TS2Vec) are also minimal. Clarity on this is needed for reproducibility and fair comparison.
- How to optimize the framework is unclear. There are only two equations for the objective function. The final loss function is also missing. In equation 8, the definition of $L_z$  is missing.

3. Ablation Study:
- The paper omits a formal analysis of the computational complexity (both time and memory) of the proposed AutoDA-Timeseries framework. This makes it difficult to assess how the method's training and inference times scale with factors like time series length, dataset size, number of augmentation layers ($K$), or the size of the transformation pool ($n$). 
- The absence of a detailed parameter analysis (e.g., total parameters introduced by the augmentation generator $A_{\theta}$ relative to the downstream model) also obscures the model's overall size and potential impact on training stability or overfitting.

### Questions
- What is the computational complexity of one forward pass through k augmentation layers relative to sequence length L and dimension d? 
- How sensitive are results to the learnable loss weights wz; do they collapse to zero (diversity ignored) for some tasks?  
- What specific time series augmentation transformations were included in the set $\mathcal{T}$? How many transformations ($n$) were used?
- How was the number of stacked augmentation layers $K$ determined? The hyperparameter study (Fig 8a) shows sensitivity analysis, but what was the value used in the main experiments, and how was it chosen (e.g., validation set)?
- Was the learnable temperature for Gumbel-Softmax shared across layers, or did each layer have its own? Was a specific annealing schedule used, or was it learned purely via backpropagation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AutoDA-Timeseries, a general data augmentation method for time series data. It incorporates time series features into data augmentation policy design and proposes an integrated optimization problem which relates the loss function in the downstream tasks and the augmentation probability and intensity. The proposed data augmentation algorithm is tested in different time series tasks, including time series classification, short-term/long-term forecasting, regression, and anomaly detection.

### Strengths
1. The paper is well written and easy to follow. 
2. It integrates reinforcement learning based data augmentation methods for different types of time series tasks, including time series classification, regression, forecasting and anomaly detection.
3. Generally good review of data augmentation methods. 
4. Experiments in different time series tasks are performed to demonstrate the good performance of the proposed method.

### Weaknesses
1. The proposed framework is very similar to data augmentation methods in CV, but with some modifications for time series data. For time series data, the unique part seems [3.3 time series feature extraction], and the rest of the proposed method is very similar to existing methods in CV. The novelty of this paper is limited.
2. Compared with other data augmentation methods for time series data, the efficiency of the proposed method is low. 
3. Some experiments are not convincing. The good part of experiments is that diverse time series tasks are tested. The weakness is that the selected downstream models are generally weak and not the SOTA methods. For example, for forecasting the RNN and Autoformer are selected, and the performance of both methods are not as good as SOTA. I suspect the proposed data augmentation method may increase the performance of those suboptimal algorithms. It may not be useful for SOTA methods. I highly recommend the authors test its performance of SOTA methods such as PatchTST.

### Questions
Check the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AutoDA-Timeseries, a general-purpose automated data augmentation framework tailored specifically for time series data. Unlike existing approaches that either rely on contrastive pretraining or image-based AutoDA techniques, this framework incorporates time series–specific statistical features to guide augmentation policy design, and it jointly optimizes both the augmentation strategy and downstream task model in a single-stage, end-to-end manner. The method adaptively learns both the selection probability and intensity of each augmentation transformation through stacked augmentation layers, balancing exploration and exploitation during training. Extensive experiments across five major time series tasks—classification, long-term and short-term forecasting, regression, and anomaly detection—demonstrate that AutoDA-Timeseries consistently outperforms state-of-the-art baselines, highlighting its robustness and broad applicability.

### Strengths
The experimental results presented in the paper are comprehensive and compelling. The authors evaluate the proposed framework across a wide range of time series tasks—including classification, forecasting, regression, and anomaly detection—and consistently demonstrate improvements over strong baselines. The inclusion of diverse downstream models further highlights the generality and robustness of the method. Overall, the experimental evidence strongly supports the effectiveness and broad applicability of the proposed approach.

### Weaknesses
W1. 

One key limitation of the paper is that it does not discuss or compare with policy-based augmentation approaches, particularly those that leverage reinforcement learning to adaptively optimize augmentation probability and intensity. This family of methods directly learns augmentation policies through reward signals tied to downstream performance and has been shown to be effective in automatically discovering task-specific augmentation strategies in several domains. Since the proposed framework also aims to learn adaptive augmentation behaviors, a discussion of how it conceptually relates to or differs from RL-based policy optimization would be valuable. Additionally, including comparisons with representative policy-based AutoDA baselines would strengthen the empirical claims and situate 
the contribution more clearly within the broader landscape of automated augmentation research.

W2. 

Another weakness is the gap between the stated motivation and the technical realization. The paper argues that existing AutoDA frameworks overlook time series–specific characteristics such as temporal dependency and autocorrelation (Line 84 onwards), but it remains unclear how the proposed approach explicitly incorporates these properties into the augmentation optimization process. While the method uses descriptive statistical features to condition the policy generator, the paper does not clearly justify why these features are sufficient to capture the essential temporal structures that differentiate time series from other modalities. As a result, it is difficult to assess whether the framework meaningfully adapts augmentation strategies to time series dynamics, or whether it is primarily applying general augmentation patterns with limited modality-aware refinement. Further clarification or ablation focused on which time series characteristics drive policy decisions would strengthen this claim.

W3. 

What is the meaning of $z=1,2,3$ in Eq (8)? Does it imply that $K=3$?

### Questions
Please seek Weaknesses above.

### Soundness
3

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
This paper presents AutoDA-Timeseries, a unified automated data augmentation framework specifically designed for time series tasks. Unlike traditional contrastive representation learning approaches, which adopt a two-stage pipeline, or existing AutoDA methods that are primarily image-oriented, AutoDA-Timeseries integrates both augmentation probability and intensity optimization into a single-stage, end-to-end framework. The method explicitly incorporates time-series–specific characteristics when designing augmentation policies and dynamically adapts them based on model feedback. Extensive experiments across five representative tasks show that AutoDA-Timeseries consistently outperforms strong baselines and demonstrates robustness across diverse datasets and architectures.

### Strengths
1. The paper is comprehensive and well-organized, clearly articulating the motivation for domain-specific augmentation design.

2. The end-to-end adaptive policy optimization framework is conceptually appealing and extends AutoDA beyond computer vision to time series effectively.

3. Experiments cover a wide variety of tasks and datasets, showing consistent and convincing improvements over both random and rule-based baselines.

### Weaknesses
1. The literature review on both representation learning and AutoDA relies heavily on older references, more recent developments in time-series adaptive augmentation should be cited.

2. In Related Work, the claim that “the learned representations may not always align well with downstream models” does not directly justify why current augmentation strategies are unsuitable. It requires stronger logical grounding.

3. The proposed framework largely follows vision-based AutoAugment paradigms, but does not sufficiently account for time-series-specific characteristics such as temporal continuity, sequential ordering, and dynamic dependencies. Consequently, some learned augmentations may distort temporal patterns or reduce interpretability.

4. The paper does not discuss sensitivity to the initial augmentation distribution, which is crucial to ensure stability and generalization across datasets.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
