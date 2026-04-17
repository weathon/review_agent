# TRIDENT: Cross-Domain Trajectory Spatio-Temporal  Representation via Distance-Preserving Triplet Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
We present the TRIplet-based Distance-preserving Embedding Network for Trajectories (TRIDENT), a spatio-temporal representation framework for compressing and retrieving trajectories across scales, from badminton courts to large-scale urban environments. Existing methods often assume smooth, continuous motion, but real trajectories exhibit event-driven annotation, abrupt direction changes, GPS errors, irregular sampling, and domain shifts, exposing the inefficiency, limited generalization, and inability to robustly integrate temporal order with spatial sequence structure of prior models. TRIDENT addresses these challenges by combining Graph Convolutional Network (GCN) spatial embeddings with temporal features in a Dual-Attention Encoder (DAEncoder), along with a Nonlinear Tanh-Projection Attention Pooling (NTAP) module that preserves local order and robustness under noise. For metric learning, we introduce a Distance-preserving Multi-kernel Triplet Loss (DMT) to preserve pairwise spatio-temporal distances in the native feature space and their rank order within the embedding, thereby reducing geometry distortion and improving cross-domain generalization. Experiments on urban mobility and badminton datasets show that TRIDENT outperforms strong baselines in retrieval accuracy, efficiency, and cross-domain generalization. Furthermore, the learned embeddings capture spatio-temporal sequence patterns, facilitating tactical analysis of badminton rallies via silhouette-guided spectral clustering that provides more actionable insights than direct trajectory classification.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a spatio-temporal representation framework that can handle perturbations in trajectories instead of assuming smooth and continuous motion. It also presents a loss function that preserves local order and robustness under noise. The framework was tested against 4 baselines using 3 datasets from different domains and demonstrated improvements.

### Strengths
- The paper addressed a relevant problem regarding extracting trajectories for perturbation data, specifically in sports-related data.
- Very good introduction of the problem and the relevant research gap. The authors properly introduce the problem of not being able to simultaneously learn both point- and shape-based trajectories, in addition to the perturbation measures due to errors is measurement and sampling then highlight the particular challenges in sport trajectories and how they are relatively underexplored.
- A scalable and unified solution. The paper utilities a unified architecture that can be trained on different data types, thus avoiding redesigning distances and model components.
- Very good experiments and results. The authors conducted experiments spanning multiple trajectory types and showed improvements across all of them.

### Weaknesses
- Presentation issue (I recommend fixing these in case the paper proceeds to be accepted)
- Incorrect usage of “\citep” and “\citet” across the paper.
- The appendix section is not properly numbered. The appendix should have letter “A, B, …” instead of numbers.
Some acronyms are used without being presented or before being fully presented. Examples: NTAP, GCN, 
- Very long introduction. The current introduction is very longs and includes both the introduction and the related work. It is better to have these sections separated for readability.
- Unclear setting of hyperparameters. The authors do not provide details on how the hyperparameter of the model are set, nor the reasoning behind setting these hyperparameters. There is only a brief description in line 774-779.
- The baseline methods are not properly presented. The authors compared their experiments with 4 baselines methods. However, the methods are not presented in details. It is important to present the details of these methods, regarding the architecture and the type of loss function they use. This assists in evaluating and understanding the experimental setup, in addition to clarifying the motivation of choosing these exact baselines.
- When comparing the framework presented in the paper with the baselines, the results are presented in numerical forms without testing whether the improvements are statistically significant or not. The statistical significance of the claimed improvements in comparison with the baselines methods needs to be tested (using a Friedman test followed by Nemenyi post-hoc test, for example).
- The code is not available for the review process. The authors mention in the paper abstract that “An anonymous repo with code and data is in the supplement.”. However, there is no such link in the appendix.

### Questions
- In the training strategy (lines 220–222), the positive and negative sample are selected based on calculating a similarity measure. For that, there are two considerations:
- Are there specific threshold to control these selections? If so, can you please elaborate on this? Without thresholds, it can happen that the positive sample is too close or identical to the anchor, and/or the negative sample is too far from the anchor. This hinders the effectiveness of contrastive triplet losses.
- Is the computation cost considered? And are there are pre-computing steps to ensure that the same calculation between the exact same sample is not done many times? This is important for the scalability of the proposed method.
- In lines 250–257, the authors discuss the values of k . However, there is no discussion nor mathematical notation showing how it is calculated. Can you please provide more details?
- In lines 773–779, a brief description of setting the hyperparameters is presented. However, only a few hyperparameters are presented without the reasoning behind setting these values. Can you please elaborate on why setting these exact values and provide details and reasoning about other hyperparameters? 
- Did you consider automated approaches to optimise the hyperparameters of the model architecture? If so, please provide details. If not, why were such methods not considered?
- On lines 859 – 873, the method of selecting the value of k used in the experiments is provided. The analysis was done on one specific dataset. However:
- It is unclear whether the same K can be used for other datasets, or another analysis has to be conducted. If an analysis needs to be conducted each time, is it a process the user has to do manually or is it automated?
- What is the motivation behind using the Silhouette Score for this assessment? Did you consider other scores such as Calinski-Harabasz index?
- Did you consider clustering algorithms that do not require explicitly setting K and allow more flexible clustering?

If my questions and the points in the “Weaknesses” section are addressed, I am happy to raise my score.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents TRIDENT, a unified framework for learning spatio-temporal trajectory representations. It handles diverse data types—from smooth GPS routes to abrupt sports movements—within a single model. TRIDENT integrates: (1) a Graph Convolutional Network (GCN) for spatial features and a temporal embedding module fused via a Dual-Attention Encoder (DAEncoder); (2) a Distance-preserving Multi-kernel Triplet Loss (DMT) that maintains the metric geometry of the original trajectory space; and (3) extensive experiments on T-Drive, Rome, and Badminton datasets showing superior retrieval accuracy, efficiency, and cross-domain generalization. A case study on badminton rallies further demonstrates the model’s ability to reveal tactical patterns.

### Strengths
S1: The proposed Distance-preserving Multi-kernel Triplet Loss (DMT) addresses a fundamental limitation of classic triplet loss by enforcing the preservation of metric structure, not just relative ordering.

S2: TRIDENT effectively handles diverse trajectory types—from smooth taxi routes to abrupt sports movements—within a unified model.

### Weaknesses
W1: The overall architecture (“GCN + temporal model + attention fusion”) follows a well-established design pattern in spatiotemporal modeling, which somewhat limits the novelty of the framework.

W2: The project's code and data are not publicly available, and the datasets used for training and testing the models are too small.

W3: The paper lacks comparisons with general sequence representation learning models, which weakens its state-of-the-art (SOTA) conclusions.

### Questions
Q1: Could you please clarify what "D2V" (line 129) refers to in the context of your temporal embedding module?

Q2: What is the meaning of the “LEARNING PROCESS OF” in section3.1?

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
3

### Summary
This paper presents TRIDENT, a spatio-temporal trajectory representation learning framework based on a novel variant of contrastive learning. The approach is designed to be generalizable across different trajectory domains, demonstrated here on GPS trajectories and badminton (human movement) trajectories; the former being smooth and dense, and the latter sparse, abrupt, and event-driven.

To achieve this, the authors propose: (1) a GCN for spatial encoding and D2V for temporal embedding, (2) a Dual-Attention Encoder with tanh attention pooling to fuse spatio-temporal representations, and (3) a Distance-preserving Multi-kernel Triplet Loss (DMT). The DMT component is carefully designed to (a) preserve the geometric structure of the original trajectories and (b) leverage an anchor-positive-negative triplet loss setup, which is known to enhance the typical contrastive positive-negative pairs.

Experimental results show that TRIDENT achieves substantial improvements over baseline methods across three datasets spanning two domains. Qualitatively, TRIDENT trajectory representations are distributed in a way that facilitates easier clustering.

### Strengths
The authors carefully designed the model architecture and training strategy, tailored to capture the unique characteristics and complexities of spatio-temporal trajectories and their diverse domains due to varying collection methods and details.

The proposed DMT loss is an elegant adaptation of the triplet loss framework, which has proven effective in other domains such as text embeddings.

The model demonstrates strong cross-domain generalizability and achieves state-of-the-art performance compared to existing baselines.

The faster training/convergence is a nice benefit.

### Weaknesses
- The paper lacks cross-domain or cross-dataset transfer experiments. It would have been valuable to see pretraining on one domain (e.g., GPS) followed by evaluation or fine-tuning on another (e.g., badminton), OR training on one dataset (e.g., Beijing) and testing on another within the same domain (e.g., Rome).

- The evaluation focuses primarily on trajectory retrieval. Incorporating additional evaluation tasks, e.g., robustness tests under trajectory downsampling or slight distortions, would strengthen the overall empirical validation.

- Minor presentation issues:
Citation formatting is occasionally incorrect; for example, missing brackets around in-line citations makes them harder to read.
L55-66 “raw trajectories are embedding space” should be revised to “raw trajectories are encoded into the embedding space.”?
The title of Section 4.3.1 appears to contain a typographical error: “Learning Process of”…?

### Questions
How was the badminton trajectory dataset collected? Providing more details on the data collection process would improve clarity and reproducibility.

How does the proposed method perform under cross-domain or cross-dataset transfer settings?

Does the method remain robust when trajectories are distorted or downsampled? For example, an analysis simulating GPS errors (such as those caused by urban canyons, as the authors mentioned in the introduction) would also help evaluate TRIDENT’s robustness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a unified trajectory representation learning framework designed to handle both continuous movement trajectories and discrete trajectories. Experiments on several datasets demonstrate the effectiveness of the proposed method for downstream tasks such as trajectory similarity search.

### Strengths
* The paper tackles an interesting problem of learning representations that generalize across both continuous and discrete trajectory types.

* The experimental results show that the proposed framework performs well on specific datasets, indicating potential for improving trajectory similarity search.

### Weaknesses
1. At first, I thought the proposed framework aimed to learn a unified trajectory representation by jointly learning from all types of trajectory data (e.g., both badminton and taxi datasets). However, it appears that the model is still trained separately for each data type. In that case, I do not see a clear advantage of this approach compared to existing methods. The overall model design (e.g., handling spatial and temporal embeddings separately and then combining them), together with the use of a triplet loss, is already widely adopted in trajectory representation learning.

2. The paper should clearly define the fundamental difference between continuous and discrete movements. Is the distinction based on predictability or on sampling rate? If the latter, then directly resampling the taxi trajectories could produce a similar dataset, suggesting that the distinction may not be substantial. A formal definition and discussion are necessary to clarify this point and justify the motivation.

3. The writing quality and organization need improvement. The introduction is overly long and occupies too much space, while the method section is too brief and lacks sufficient implementation details, making it difficult to reproduce the work. Additionally, the topic may be too narrow in scope and might not appeal to the broader ICLR readers.

### Questions
1. What is the novelty of the proposed approach, given that most components of the model are already well studied in the literature? 

2. What is the fundamental difference between continuous and discrete movements? Why do they need to be treated separately, and which parts of the model explicitly account for this distinction? A clearer conceptual and architectural explanation would strengthen the contribution.

3. Since discrete trajectories are the main focus of the paper and show the most performance improvement, the evaluation should include more relevant datasets, such as basketball or soccer trajectories, to better validate the generality of the proposed approach.

### Soundness
2

### Presentation
2

### Contribution
2
