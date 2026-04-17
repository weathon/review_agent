# CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Current methods for multivariate time series forecasting can be classified into channel-dependent and channel-independent models. Channel-dependent models learn cross-channel features but often overfit the channel ordering, which hampers adaptation when channels are added or reordered. Channel-independent models treat each channel in isolation to increase flexibility, yet this neglects inter-channel dependencies and limits performance. To address these limitations, we propose CPiRi, a channel permutation invariant (CPI) framework that infers cross-channel structure from data rather than memorizing a fixed ordering, enabling deployment in settings with structural and distributional co-drift without retraining. CPiRi couples spatio-temporal decoupling architecture with permutation-invariant regularization training strategy: a frozen pretrained temporal encoder extracts high-quality temporal features, a lightweight spatial module learns content-driven inter-channel relations, while a channel shuffling strategy enforces CPI during training. We further ground CPiRi in theory by analyzing permutation equivariance in multivariate time series forecasting. Experiments on multiple benchmarks show state-of-the-art results. CPiRi remains stable when channel orders are shuffled and exhibits strong inductive generalization to unseen channels even when trained on only half of the channels, while maintaining practical efficiency on large-scale datasets. The source code is released at https://github.com/JasonStraka/CPiRi.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses a central paradox in multivariate time series forecasting: channel-dependent (CD) models overfit to channel order rather than learning true inter-channel relationships, while channel-independent (CI) models achieve robustness by sacrificing performance, as they neglect these critical dependencies. To resolve this trade-off, the authors propose CPiRi, a novel framework that employs a spatio-temporal decoupling architecture. CPiRi leverages a frozen, pre-trained foundation model to independently extract robust temporal features, which are then fed into a lightweight, trainable spatial module. This module is specifically trained with a permutation-invariant regularization strategy (channel shuffling) to learn content-driven, permutation-equivariant relationships. This design allows CPiRi to achieve state-of-the-art forecasting accuracy while demonstrating exceptional robustness, maintaining its performance even when channel orders are permuted during inference.

### Strengths
1. The proposed spatio-temporal decoupling architecture, which integrates a *frozen* time series foundation model (for temporal features) with a *lightweight, trainable* spatial module (for relational learning), is a novel and efficient design.
2. The introduction of the permutation-invariant regularization strategy (channel shuffling) is a simple yet highly effective training technique to enforce the desired invariance.
3. The experiments are pointedly designed to test the central hypothesis. The 'channel shuffling robustness analysis' (Table 2) is particularly impactful, providing a stark and convincing contrast between CPiRi's stability and the fragility of other CD models.

### Weaknesses
1. The paper *does* apply the shuffling strategy to baselines (Table 2, "Train Shuffle"), which is a strong point. Could the authors elaborate on *why* this strategy fails to rescue models like Informer or STID?
2. The "w/o regularization strategy" ablation (Table 4) shows only a *minor* performance drop (e.g., 9.21% vs 9.14% on METR-LA; 10.80% vs 9.43% on PEMS-08). This suggests that the permutation-equivariant architecture alone provides almost all of the robustness, and the "potent" and "critical" regularization strategy actually has a minimal impact.
3. The calculation cost of CPiRi, which adds an $O(C^2)$ spatial module, is non-trivial. However, in some cases (e.g., METR-LA, Table 1), it does not achieve SOTA performance compared to lighter models like STID.

### Questions
1. The paper convincingly shows that CPiRi is robust to channel shuffling, while standard CD models are not. However, it's unclear how much of CPiRi's gain comes from its permutation-equivariant *architecture* versus its "permutation-invariant regularization" *training strategy*.
2. The "channel shuffling" in Figure 4 is a key experiment. The methodology is unclear. How is "25% shuffle" defined? Does it mean 25% of the channel indices are randomly permuted *among themselves*, or 25% are swapped with other random channels? This needs a precise definition.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
CPiRi is a new framework for multivariate time series forecasting that stays accurate even when input channels are reordered or changed. It combines a frozen pre-trained temporal encoder with a lightweight spatial module trained using random channel shuffling, forcing the model to learn true content-based relationships instead of memorizing positions. This design achieves state-of-the-art accuracy, strong generalization to unseen channels, and robust performance under channel permutations.

### Strengths
1. Clear Problem Identification: The paper clearly diagnoses a critical flaw in existing CD models using a simple "channel shuffling" diagnostic test. This test reveals that many SOTA models rely on "positional memorization," leading to catastrophic performance collapse (e.g., >400% error increase for Informer) when channel order is changed.
2. Effective and Sound Design: The "spatio-temporal decoupling" is an elegant solution. It leverages the power of a robust, pre-trained temporal model (CI strength) while using a separate, lightweight module to explicitly learn cross-channel interactions (CD strength). The channel shuffling strategy directly enforces the desired permutation-invariant property.
3. Comprehensive Experimental Validation: The experiments are thorough and directly support the claims. CPiRi achieves SOTA accuracy , remains perfectly stable under channel shuffling , and shows strong inductive generalization to unseen channels. The ablation studies (Table 4) clearly demonstrate the necessity of each component: the spatial module, the pre-trained weights, and the shuffling strategy.

### Weaknesses
1. The framework's success is entirely dependent on the frozen Sundial encoder. The ablation study "w/o pretrained weights" results in "complete failure". This makes it difficult to separate the contribution of the novel CPiRi training strategy from the powerful priors of the (very large) foundation model it relies on.
2. The individual components are standard: a "standard Transformer encoder block" for the spatial module and a data augmentation technique for training. The novelty is in the clever system design and training methodology that combines these parts, rather than a fundamentally new architectural mechanism.
3. By freezing the temporal encoder, the model can handle structural co-drift (changes in channel order) but cannot adapt its temporal feature extraction to new patterns or "abrupt trend shifts". This limitation is noted by the authors and means it may struggle in scenarios where the underlying temporal dynamics of the data change over time.

### Questions
1. Why frozen temporal encoder? Will fine-tuning help improve performance?
2. Will applying different temporal encoders, e.g., Chronos and Moment [1], affect performance?
3. Does the strategy work for even higher-dimensional time series [2]?
4. Can the authors visualize the latent representations before and after spatial encoding across different permutations?

[1] Goswami M, Szafer K, Choudhry A, Cai Y, Li S, Dubrawski A. Moment: A family of open time-series foundation models. arXiv preprint arXiv:2402.03885. 2024 Feb 6.
[2] Ni J, Wang S, Liu Z, Shi X, Zhong X, Ye Z, Jin W. U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting. arXiv preprint arXiv:2507.15119. 2025 Jul 20.

### Soundness
2

### Presentation
3

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
The paper proposes CPiRi, a channel permutation-invariant framework for multivariate time series forecasting. It combines a frozen pretrained temporal encoder with a lightweight spatial interaction module trained under random channel shuffling. This design makes the model rely on content-based relations instead of channel order, achieving strong generalization and state-of-the-art performance on multiple benchmarks.

### Strengths
1. Interesting motivation.
2. Clear and easy-to-follow writing.
3. Comprehensive theoretical analysis provides support for the proposed method.

### Weaknesses
1. Since this paper only uses Sundial as the temporal feature extractor, it lacks an explanation of why Sundial was chosen over other foundation models. Can this framework generalize to other pretrained models such as Chronos [1] or Moment [2]?
2. I appreciate that the paper uses high-dimensional datasets with channel heterogeneity. This is an interesting attempt for scalability analysis. However, could you also test the model on Time-HD [3]?
3. The main concern lies in efficiency (which the authors did not discuss in the main text). The framework invokes two large pretrained models and employs multi-head attention for high-dimensional inputs, which could lead to significant computational overhead.
4. iTransformer [4] also applies multi-head attention along the channel dimension and is permutation-equivariant. How does this work differ from iTransformer in terms of channel modeling?

[1] "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).
[2] "Moment: A family of open time-series foundation models." arXiv preprint arXiv:2402.03885 (2024).
[3] "U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting." arXiv preprint arXiv:2507.15119 (2025).
[4] "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." arXiv preprint arXiv:2310.06625 (2023).

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles multivariate time series forecasting and points out that many models inadvertently entangle cross-channel interactions with the specific channel order seen in training. The authors introduce a three-stage framework: a frozen, pretrained univariate backbone to extract temporal features, a lightweight spatial module that views these features as an unordered set and models their relations with self-attention, and a frozen per-channel decoder. To enforce order agnosticism, training is done with random channel shuffling, discouraging any dependence on positional cues. On traffic style benchmarks, the approach proves effective, and the paper further shows it can generalize even when trained on only a subset of channels.

### Strengths
1. The problem framing is timely and interesting. The paper not only says “permutations matter” but builds an explicit diagnostic: train with fixed order, test with shuffled order, show catastrophic failure for several competitive models. 
2. The proposed framework is reasonable and coherent, with the per channel frozen temporal encoder feeding a permutation aware spatial block, and the permutation based training strategy reinforcing the intended behavior.
3. The reported improvements indicate that the method actually delivers robustness rather than just matching accuracy in the standard setting.

### Weaknesses
1. My main concern is the reliance on a large pre-trained backbone. Table 4 shows that removing the pre-trained weights leads to a substantial drop in accuracy, which suggests that much of the gain comes from the foundation model rather than from the proposed permutation invariant interaction itself. However, in Table 1 the competing CD baselines are trained from scratch and do not benefit from comparable pre-training. This raises a fairness question: to what extent are the improvements due to the architectural idea, and to what extent are they due to access to stronger prior knowledge? It would help to include baselines equipped with similar pre-trained features, or to report results for CPiRi without pre-training in Table 1.
2. All benchmarks are traffic datasets. It is unclear whether the same channel ordering effect appears in other multivariate settings (e.g., energy, industrial telemetry, climate) or in higher-dimensional public datasets such as [1]. Showing at least one non-traffic dataset would clarify how general the phenomenon is.
3. The paper shows that changing channel order can hurt performance, but it is not clear how this effect scales as the number of channels grows. Do larger channel sets make models more brittle to reordering, or does the effect plateau? A controlled study where channel count is progressively increased would make the robustness claim stronger.


[1] U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting. 2025

### Questions
The questions are included in the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
