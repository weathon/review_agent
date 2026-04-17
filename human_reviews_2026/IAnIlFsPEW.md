# T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Imputing missing values in multivariate time series remains challenging, especially under diverse missing patterns and heavy missingness. Existing methods suffer from suboptimal performance as corrupted temporal features hinder effective cross-variable information transfer, amplifying reconstruction errors. Robust imputation requires both extracting temporal patterns from sparse observations within each variable and selectively transferring information across variables—yet current approaches excel at one while compromising the other. We introduce T1 (Time series imputation with 1-to-1 channel-head binding), a CNN-Transformer hybrid architecture that achieves robust imputation through Channel-Head Binding—a mechanism creating one-to-one correspondence between CNN channels and attention heads. This design enables selective information transfer: when missingness corrupts certain temporal patterns, their corresponding attention pathways adaptively down-weight based on remaining observable patterns while preserving reliable cross-variable connections through unaffected channels. Experiments on 11 benchmark datasets demonstrate that T1 achieves state-of-the-art performance, reducing MSE by 46% on average compared to the second-best baseline, with particularly strong gains under extreme sparsity (70% missing ratio). The model generalizes to unseen missing patterns without retraining and uses a consistent hyperparameter configuration across all datasets. The code is available at https://github.com/Oppenheimerdinger/T1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces T1, a CNN-Transformer hybrid architecture for multivariate time series imputation. The key contribution is the “Channel-Head Binding” mechanism, which establishes a one-to-one correspondence between CNN channels (capturing temporal features) and attention heads (enabling cross-variable information transfer), allowing feature-level selectivity in imputing missing data.

### Strengths
1. The T1 architecture demonstrates superior performance across its experimental settings.
2. The model shows particular strength in scenarios with heavy missingness.
3. The paper introduces a novel mechanism.

### Weaknesses
1. The model's core logic relies on the unproven assumption that CNN channels naturally learn distinct temporal patterns. Without an explicit mechanism to enforce this, features may be redundant, which would undermine the "feature isolation" claim and allow corrupted information to propagate.
2. The rigid 1-to-1 binding is a strong constraint that prevents the model from learning combinations of features across different channels. Furthermore, the claim that heads "adaptively down-weight" corrupted channels is not mechanistically proven.
3. The claim of using a "single hyperparameter configuration" is exaggerated. The appendix reveals that convolutional kernel sizes were "proportionally adjusted" for the PhysioNet.1

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The main task of this paper is multivariate time-series missing value imputation. 
It focuses on how to balance temporal feature extraction and the transfer of cross-variable information, as well as imputation under diverse missing patterns and high missing rates. To solve this problem, the paper proposes T1, a CNN-Transformer hybrid architecture.

### Strengths
1. The experiments are comprehensive: a large number of datasets are used, the comparative models basically cover the current sota methods, and different missing scenarios are compared, making the result quite thorough.
2. The experimental results are abundant and appear reliable.

### Weaknesses
1. The authors summarize different categories of methods in the main text. It would be more intuitive if a similar categorized comparison approach was adopted in the experimental section.
2. There are certain issues with the figures in the paper: the stacked arrows between channels in Figure 1(b) are misleading; in Figure 2, the direction of the arrow from (b) to (c) is unclear, and Figure 2(c) does not show "Value".
3. The writing and figure drawing may cause misunderstandings. Based on the main text, my understanding is as follows: For a time segment with M variables, one of the variables is processed by CNN to obtain C channels. For each channel, Q, K, and V are calculated using Formula 3. Then, for each channel, the corresponding Q, K, V of the M variables are stacked by category. Next, Formula 4 is used to calculate the attention scores. Finally, the results of the C channels are stacked. This part and the corresponding figures need to be described more clearly.

### Questions
1. The number of channels used in the experiments is 128, and the authors also discuss different values of channel numbers in the appendix. However, the experimental results show little fluctuation. Is there a possibility of channel number redundancy here? I am curious whether there will be many zero vectors after a time-series segment with 40% missing values undergoes two rounds of convolution. Have the authors considered training under different missing value rates?
2. Regarding the Dual-axis tokenization method, the authors describe its disadvantage as: "but struggle to transfer information across both dimensions when missing values block intermediate pathways." Will T1 face similar problems, during information transfer between channels?
3. Other questions are as mentioned in the "Weaknesses" section.

### Soundness
3

### Presentation
2

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
This paper proposes T1, a CNN-Transformer hybrid model for multivariate time-series imputation under heavy and diverse missing patterns. The key innovation is Channel-Head Binding (CHead Attention), which establishes a one-to-one correspondence between CNN channels (for temporal feature extraction) and Transformer attention heads (for cross-variable information transfer). This allows selective, feature-level interactions that mitigate error propagation from corrupted observations. The architecture uses modernized temporal convolutions with depthwise and pointwise operations to handle sparse data robustly. Experiments on 11 benchmark datasets show T1 outperforming baselines, with strong generalization to unseen missing patterns and a single hyperparameter setup across datasets.

### Strengths
1. Novel Architectural Design: The Channel-Head Binding mechanism is a clever integration of CNNs and Transformers, addressing a clear limitation in prior tokenization strategies (e.g., time-axis, variable-axis, or dual-axis approaches). By aligning channels and heads, it enables fine-grained control over information transfer, which is particularly effective for imputation where missingness corrupts specific temporal patterns. This hybrid approach leverages CNNs for robust local feature extraction and attention for adaptive cross-variable fusion, making it well-suited to real-world sparse data scenarios.

2. Strong Empirical Results: The reported average MSE improvement over the second-best baseline is impressive, especially under extreme missing ratios. The model's ability to generalize to unseen patterns without retraining and its use of a unified hyperparameter configuration highlight its practicality for diverse domains like healthcare, finance, and climate monitoring.

3. Efficiency and Robustness: T1's design emphasizes efficiency through convolutional operations and avoids common pitfalls like naive token mixing, leading to reliable performance in heavy missingness cases where other methods degrade.

### Weaknesses
1. Limited Comparison to Recent Generative Methods: While the baselines cover traditional imputation techniques (e.g., Transformer variants, CNN-based models), the experiments could benefit from comparisons to emerging generative approaches like diffusion models (e.g., CSDI or adaptations from recent works like Winformer) or LLM-based time-series methods. These could provide context on whether T1's gains hold against probabilistic or foundation-model-based alternatives, especially in cross-domain or long-sequence settings.

2. Dataset Diversity and Scale: The 11 benchmarks are solid but somewhat standard in time-series literature; including more recent or larger-scale datasets could strengthen claims of broad applicability. Additionally, while natural missingness is mentioned, deeper analysis of real-world irregular sampling (e.g., via case studies) would enhance the evaluation.

3. Minor Issues:  Figures could include more quantitative illustrations of binding effects (e.g., attention heatmaps under varying missingness).

### Questions
1. What is the computational overhead of Channel-Head Binding compared to vanilla CNN-Transformers? Ablations on channel/head counts would be useful.

### Soundness
2

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
3

### Summary
This paper proposes a new model for filling in missing values in multivariate time series. Its key innovation is Channel-Head Binding, which creates a direct, one-to-one link between feature-learning CNN channels and information-sharing attention heads. This allows the model to selectively transfer information between variables based on which temporal patterns are reliable, preventing corrupted data from spreading.

### Strengths
1. The one-to-one "Channel-Head Binding" is a novel and clever way to fuse CNNs and Transformers for robust imputation.
2. Extremely thorough testing on diverse datasets and challenging missingness scenarios, with strong results.
3. The paper is well-structured, the problem is well-motivated, and the model is clearly explained with helpful diagrams.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.
1. The number of the convolution channels in T1 is set to 128 during training and is the only hyper-parameter, but no experiment or discussion is provided to explain the reason and the influence of this parameter setting. A simple contrastive experiment would prove the strength of  convolution module better.
2. The paper only points out the limitations of current convolution and Transformers in imputation, more explanation on the reason or theoretical strength of such combined construction would much improve readers’ understanding.
3. In section 3, when introducing the architecture of T1, lots of convolution and transformers architecture can be used as analogy, which can help readers familiar with these models to more quickly grasp the structural principles and design innovations of T1.
4. If I understand it right, in figure 1(a), the output of convolution embedding is described as “Feature”, while all the other places in the paper, this dimension is described as “Channel”, Aligning this term would make it much clearer.

### Questions
1. How is the number of  convolution channels chosen, In other words, why set it to 128 while training?

### Soundness
3

### Presentation
2

### Contribution
3
