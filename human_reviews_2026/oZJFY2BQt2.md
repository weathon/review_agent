# Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series

- Decision: Accept (Oral)
- Scores: 4, 6, 8

## Abstract
Accurate analysis of Medical time series (MedTS) data, such as Electroencephalography (EEG) and Electrocardiography (ECG), plays a pivotal role in healthcare applications, including the diagnosis of brain and heart diseases. MedTS data typically exhibits two critical patterns: **temporal dependencies** within individual channels and **channel dependencies** across multiple channels. While recent advances in deep learning have leveraged Transformer-based models to effectively capture temporal dependencies, they often struggle to model channel dependencies. This limitation stems from a structural mismatch: ***MedTS signals are inherently centralized, whereas the Transformer's attention is decentralized***, making it less effective at capturing global synchronization and unified waveform patterns. To bridge this gap, we propose **CoTAR** (Core Token Aggregation-Redistribution), a centralized MLP-based module tailored to replace the decentralized attention. Instead of allowing all tokens to interact directly, as in attention, CoTAR introduces a global core token that acts as a proxy to facilitate the inter-token interaction, thereby enforcing a centralized aggregation and redistribution strategy. This design not only better aligns with the centralized nature of MedTS signals but also reduces computational complexity from quadratic to linear. Experiments on five benchmarks validate the superiority of our method in both effectiveness and efficiency, achieving up to a **12.13%** improvement on the APAVA dataset, with merely 33% memory usage and 20% inference time compared to the previous state-of-the-art. Code and all training scripts are available in this [**Link**](https://github.com/Levi-Ackman/TeCh).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoTAR (Core Token Aggregation–Redistribution), a centralized alternative to self-attention for modeling medical time series (MedTS) such as EEG and ECG. The authors argue that MedTS signals are centrally coordinated (e.g., brain or heart acting as a signal source), while Transformer attention is inherently decentralized, making it ill-suited to capture the global dependencies between channels.

To address this, CoTAR introduces a central “core token” that aggregates global information from all tokens (channels) and redistributes it back via a lightweight MLP mechanism, achieving linear computational complexity. Combined with a dual tokenization strategy that separately encodes temporal and channel embeddings, the resulting model (TeCh) jointly captures both temporal and inter-channel dependencies.

Extensive experiments across five MedTS datasets (EEG/ECG) and two human activity recognition (HAR) datasets show that TeCh achieves state-of-the-art accuracy and efficiency

### Strengths
- The paper convincingly articulates the mismatch between decentralized attention and the centralized nature of many physiological signals. This conceptual framing is both intuitive and novel for the MedTS domain.

- CoTAR is a well-engineered module that reduces the quadratic cost of self-attention to linear, while retaining flexibility in cross-token communication.

- Experiments span seven datasets (five MedTS + two HAR) with six evaluation metrics. The method consistently outperforms ten Transformer-based baselines.

- Code and training scripts are publicly released.

- Provide robust test, i.e., standard deviation.

### Weaknesses
- The authors repeatedly assert that MedTS are “centralized” but provide no quantitative validation.
- The paper omits direct comparisons with recent dual-dependency or TeCh-style models (e.g., GAFormer)
- The proposed CoTAR conceptually resembles several prior Transformer modifications that employ global or auxiliary tokens to aggregate and redistribute information. The authors should discuss them. e,g,. CATS
- Different datasets use different hyperparameters.







[1] GAFormer: Enhancing time-series transformers through group-aware embeddings

[2] CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables

### Questions
-  Is there a formal way to distinguish between centralized and non-centralized MedTS?
- Can the core token be interpreted or visualized to correspond to physiological latent processes?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the **CoTAR** (Core Token Aggregation-Redistribution) module to replace the attention module in medical time-series (MedTS) classification. This method is motivated by the assumption that MedTS signals typically originate from a centralized biological source. The design of the CoTAR module is inspired by client-server communication, which uses a core token to aggregate and exchange information between clients, rather than self-attention, where each token attends to all others equally. A new method called **TeCh** is proposed, aligned with the CoTAR module, similar to the Transformer architecture, but with attention replaced by CoTAR. Results are compared against 10 baselines across 5 MedTS datasets and 2 general time series datasets for classification, achieving SoTA performance.

### Strengths
The method is motivated by MedTS's domain knowledge and inspired by server-client communication, which is a good approach. It is interesting to see that the linear complexity CoTAR module performs similarly to, and in some cases even better than, SOTA transformer methods. The comprehensive ablation study and direct comparison with the attention module are good and demonstrate the effectiveness of the CoTAR module.

### Weaknesses
It is better to provide more detail in equal (2), as Figure 2 lacks notation for the variables used. I can get the idea of the core token being redistributed to each token, but reading the equal (2) is still a little confusing about the details. The performance on the ADFTD dataset is limited to the F1 score. Sometimes, a fixed subject-independent split makes it hard to demonstrate the superiority of a method, as specific subjects in the training set may contain too much noise and make results across methods similar. You could provide a cross-validated (5-fold or Monte Carlo) subject-independent evaluation result on the dataset, demonstrating the effectiveness of your method, even when performance is limited on a fixed split. Besides, more advanced SOTA methods, such as MedGNN, should be compared with.

### Questions
See weakness

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
4

### Summary
This paper addresses a "structural mismatch" between Transformer models and Medical Time Series (MedTS) data like EEG and ECG. The authors argue that while MedTS signals originate from a centralized biological source (e.g., the heart or brain), the standard Transformer attention mechanism is decentralized, with all-to-all token interactions. This mismatch, they claim, makes it difficult for Transformers to model channel dependencies effectively.

To solve this, the paper proposes CoTAR, an MLP-based module designed to replace attention. CoTAR introduces a "global core token" that acts as a proxy. All tokens first aggregate information to this core token, which then redistributes the integrated information back to all tokens. This star-shaped architecture mimics the centralized nature of MedTS signals while reducing the computational complexity from quadratic to linear.

The full model, TeCh, uses CoTAR within a Dual Tokenization framework that processes the input in parallel: once with "Temporal Embedding" (patches of time) and once with "Channel Embedding" (whole channels as tokens).

Experiments on five MedTS and two HAR datasets show that TeCh achieves state-of-the-art performance, significantly outperforming prior SOTA (Medformer) with large gains in efficiency (e.g., 33% memory and 20% inference time) and robustness to noise.

### Strengths
1. Strong, Intuitive Inductive Bias: The paper's primary strength is the clear and compelling motivation. The argument that decentralized attention is a poor structural match for centralized biological signals is an excellent insight and provides a strong foundation for the CoTAR module.

2. Superior Performance: The proposed model achieves state-of-the-art results on a wide range of MedTS datasets, often by a significant margin over the previous SOTA, Medformer. This demonstrates the empirical effectiveness of the CoTAR module.

3. Massive Efficiency Gains: The paper's most significant practical contribution is the efficiency of CoTAR. By reducing complexity from $O(S^2)$ to $O(S)$, the model achieves up to a 5x speedup in inference and a 3x reduction in memory usage compared to the prior SOTA. This is clearly visualized in Figure 4(a) and is critical for real-world medical applications.

4. Improved Robustness: The experiments in Figure 4(b) provide strong evidence that CoTAR's centralized proxy design makes the model significantly more robust to noise in the input channels. This is a key practical advantage for noisy MedTS data.

5. Thorough Evaluation: The experimental setup is strong, using 5 MedTS datasets (EEG and ECG) and 2 HAR datasets to show generalizability. The authors correctly use a subject-independent splitting protocol, which is crucial for clinically relevant MedTS evaluation.

### Weaknesses
1. Misleading "Dual Tokenization" Framework: The paper's biggest weakness is the framing of the "TeCh" model around "Dual Tokenization". The SOTA results in Tables 2 & 3 are NOT achieved by a consistent dual-branch model. As Table 6 reveals, 4 of the 7 datasets (TDBrain, PTB, PTB-XL, FLAAP) use a single-branch model ($M=0$ or $N=0$) to get the reported results. This makes the "Ablation Study on 'Dual Tokenization'" (Table 4) highly misleading. The ablation's conclusion that "combining both yields overall superior performance" is cherry-picked (it's only true for 2/5 datasets) and contradicted by the final model's own hyperparameters. The paper should be reframed to present "TeCh" as a family of CoTAR-based models where the tokenization (Temporal, Channel, or Dual) is a hyperparameter to be tuned, rather than presenting "Dual" as the definitive architecture.

2. Confusing Mathematical Notation: The formal definition of CoTAR in Equation (2) is unclear. The notations are a bit confusing, particularly the use of symbols to represent the matrices (e.g., both $O$ and $Co$ represent a single matrix/vector). And the function names do not follow a consistent notation (e.g., upright "GELU" vs. italicized "GELU" in different parts of the equation). Notably, the equations do not clarify the shape of the matrices/vectors involved, making it hard to follow the operations. A clearer, more consistent notation with explicit shapes would improve clarity.

### Questions
1. Clarification of TeCh Architecture: The central framing of the paper is confusing. Table 6 shows that the SOTA results for TDBrain, PTB, PTB-XL, and FLAAP are achieved using a single-branch model ($M=0$ or $N=0$), not the "Dual Tokenization" model ($M>0$ and $N>0$) described in Section 4.2 and analyzed in Table 4.

   - Could you confirm that the SOTA results in Tables 2 & 3 are achieved by tuning $M$ and $N$ and often setting one to 0?
   - If so, why is the paper framed around "Dual Tokenization" as the primary architecture? This seems to misrepresent the final model and makes the "Dual Tokenization" ablation (Table 4) misleading. Wouldn't it be more accurate to present TeCh as a CoTAR-based model where the tokenization strategy (Temporal, Channel, or Both) is a key hyperparameter?

2. Clarification of CoTAR Math (Equation 2): Could you please provide an unambiguous, step-by-step definition of the CoTAR module's computation? Specifically, what are the dimensions of all the weights and biases? A clear definition would resolve confusion about the module's precise mechanism.

3. Mismatch in Ablation Results: There ablation studies are only performed on 5 of the 7 datasets and in a inconsistent manner. Is there a reason why the ablations were not performed on the other datasets? Including these would provide a more complete picture of the model's behavior across all evaluated datasets.

4. Mismatch in Model Implementation: The paper provides the code link, but the imported names of the TeCh model seems to suggest that it uses the same Transformer encoder layer and not the CoTAR module. Since I cannot access the actual content of the files under the layers directory, could you clarify if the provided code implements the CoTAR module as described in the paper?

### Soundness
3

### Presentation
3

### Contribution
3
