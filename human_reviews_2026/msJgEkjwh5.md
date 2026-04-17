# CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6

## Abstract
Electroencephalography (EEG) provides real-time insights into brain activity and supports diverse applications in neuroscience. While EEG foundation models (EFMs) have emerged to address the scalability issues of task-specific models, current approaches still yield clinically uninterpretable and weakly discriminative representations, inefficiently capturing global dependencies and neglecting important local neural events. We present CodeBrain, a two-stage EFM designed to fill this gap. In the first stage, we introduce the TFDual-Tokenizer, which decouples heterogeneous temporal and frequency EEG signals into discrete tokens, quadratically expanding the representation space to enhance discriminative power and offering domain-specific representation-level interpretability by suggesting potential links to neural events and spectral rhythms. In the second stage, we propose the multi-scale EEGSSM architecture, which combines structured global convolution with sliding window attention to efficiently capture both sparse long-range and local dependencies, reflecting the brain’s small-world topology. Pretrained on the largest public EEG corpus, CodeBrain achieves strong generalization across eight downstream tasks and ten datasets under distribution shifts, supported by comprehensive ablations, scaling-law analyzes, and interpretability evaluations. The code and the pretrained weights are available at https://github.com/jingyingma01/CodeBrain.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CodeBrain, a two-stage EEG foundation model that aims to achieve both domain-specific interpretability and strong cross-task generalization. The first stage introduces a tokenizer which decouples temporal and frequency EEG features into discrete tokens, effectively enlarging the representation space and improving discriminability while maintaining interpretability linked to neural and spectral events. The second stage proposes a multi-scale SSM combining global convolutional layers and sliding-window attention, designed to capture both long-range dependencies and local neural patterns. CodeBrain demonstrates strong generalization across 8 downstream tasks and 10 datasets under distribution shifts.

### Strengths
(1) The authors presented a time-frequency decoupled EEG tokenizer, combined with generative learning with multiscale SSM, with increased computational efficiency, which is somewhat novel especially w.r.t. the decoupled representation.

(2) Strong cross-domain robustness and interpretability visualizations.

(3) High clarity and well-organized figures.

### Weaknesses
(1) The overall work, both tokenizer and the generative pretraining, still seem incremental to me. It would be better to discuss and if possible, analyze quantitatively how it substantially improves EEG representation in actual subbands as well as varying, often non-stationary time-frequency landscape. It also fails to convince me how the current work architecturally and mechanistically innovates beyond LaBraM, Beatrix, EEGMamba, etc...  Also, time-frequency representation is not new at all. It seems that the authors still concentrate on Fourier spectrum for the frequency part.

(2) It would be better to consider adding at least one or two supervised baselines as well as contrastive ones for a more comprehensive comparison.

(3) I don't understand why the author completely disregarded gamma oscillations and higher-frequency activities from their analysis. Epileptic seizures, for example, often involve high-frequency outbursts during preictal and interictal stages. Such high-frequency activities are ubiquitous inother physiological activities as well. It is suggested that the authors should perform some ablative experiments to explore how their preprocessing pipeline is justified.

Still, should the authors provide satisfactory clarification or partial resolution of these concerns, I would be open to increasing my score.

### Questions
(1) Were the AUROC and AUPRC calculated in a balanced manner?

(2) About channel robustness, have authors ever consider masking specific brain regions to provide insights into the robustness in neuroscientific or clinical settings? Random masking is unconvincing.

(3) The authors used a tripartite splitting in their experimental setup. Why not use subject-independent cross-validation (see experimental setup in FAPEX, BrainWave, Brant, Scatterformer, etc.), which offers more robust statistical estimates while prevents any potential information leakage? 

(4) The authors should at least analyze MACs, Throughput/Latency, FLOPs of some of the best-performing baselines along with CodeBrain.

### Soundness
2

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
This study presents an interpretable EEG foundation model, CodeBrain, which integrates both local and global information. Selected codes deriving from TFDual-Tokenizer contain strong clinical meaning and could serve as an effective “EEG vocabulary”, which seems to improve the interpretability. Further, the authors conduct extensive experiments to validate the effectiveness of nearly every component of the proposed framework.

### Strengths
1. The authors propose a domain-specific and interpretable TFDual-Tokenizer, which decomposes raw EEG signals into frequency and temporal domains to generate dual-domain tokens, thereby enhancing the representation capability of the model.
2. The proposed EEGSSM architecture integrates multi-scale information and enhances the representation of intra-patch features through a sliding-window attention mechanism.
3. In addition to extensive empirical experiments, the study provides strong theoretical support for the proposed methods.

### Weaknesses
1. Although the selected codes in Appendix B are associated with clinically meaningful patterns, the criteria for their selection and their specific contribution to model performance remain unclear. While these codes are intuitively appealing, the paper does not provide concrete evidence demonstrating their impact on improving downstream task performance. Without such evidence, the explanation may not be important for model decision in the down-stream tasks and could potentially mislead researchers, as model reasoning does not necessarily align with human intuition. Moreover, the analysis of class-specific token ratios does not convincingly support a relationship between the proposed interpretable codes and performance gains. In a word, no evidence directly shows that those interpretable codes play a primary role in the model’s decision-making process.

2. If my understanding is correct, in Equation (17) defining the dominance ratio, $max_y N_c^{(y)}$ should always be smaller than $\sum_{y} N_c^{(y)}$ since the latter item $\sum_{y} N_c^{(y)} = max_y N_c^{(y)} + other  N_c^{(y)}$. Therefore, the dominance ratio would be less than 1. Could the authors clarify why they define Dominance(c) > 1 as the criterion for identifying class-specific tokens?
3. Duplication in Figure 3, the left and middle images are identical.

### Questions
1. According to the paper, the authors pretrain several versions of the EFM. I recommend including some main versions and their corresponding model sizes in the main text rather than only in the Appendix. Initially, it was unclear which model version the reported results referred to in the main content, forcing readers to cross-check details in the Appendix. Presenting this information earlier would improve clarity and readability.
2. Regarding the TFDual-Tokenizer, have the authors considered a sparse code strategy based on a gate mechanism, where only the frequency, temporal, or occasionally dual code is selected for a given patch? $N^2$ representation space may introduce some redundancy, as suggested by the unused token experiments.

I will remain active during the rebuttal and may consider adjusting my score based on the authors’ responses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces CodeBrain, an EEG foundation model (EFM) designed to address key challenges in EEG analysis. CodeBrain uses a two-stage framework to improve both interpretability and efficiency in processing EEG data. In the first stage, the TFDual-Tokenizer decouples the temporal and frequency components of EEG signals into discrete tokens, enhancing the model’s ability to interpret domain-specific features. In the second stage, the EEGSSM architecture combines structured global convolutions with sliding window attention to capture both long-range dependencies and local neural events. The model is pretrained on the largest publicly available EEG corpus and tested across multiple downstream tasks and datasets, demonstrating strong generalization and performance improvements over existing baselines.

### Strengths
CodeBrain introduces the decoupling of temporal and frequency EEG components via the TFDual-Tokenizer, which provides domain-specific interpretability. This is a novel approach not seen in previous EEG foundation models. The manuscript is comprehensive, well-structured, and tackles important issues in EEG signal representation learning. It presents a clear solution to existing challenges in terms of both efficiency and interpretability. The paper includes extensive experiments across 8 downstream tasks and 10 datasets, demonstrating the model’s generalizability under various distribution shifts. Detailed ablation studies, scaling-law analyses, and interpretability evaluations further confirm the robustness and effectiveness of CodeBrain.

### Weaknesses
The paper lacks sufficient evidence to support the claim that EFMs outperform traditional models like EEGNet and STTransformer, as it does not include these models for comparison. Additionally, the justification for choosing the Swin transformer decoder over other transformer-based decoders is not well-explained, and there is a lack of detail regarding pretraining time and other crucial parameters, which hinders the assessment of model efficiency and scalability.

### Questions
1. Contrastive Learning: Why does splitting the data into two parts generate positive and negative samples? Are we conducting the contrastive analysis over each temporal code individually, or are we considering the entire patch (n code) for the contrastive task? Furthermore, how does the use of SimCLR contribute to solving convergence issues during training?
2. State Space Model Architecture: Can you elaborate on the architectural choices in the state space model? 
3. Baseline: The paper claims that EFMs significantly outperform smaller supervised models like EEGNet and STTransformer.(“We do not include traditional supervised baselines such as EEGNet (Lawhern et al., 2018) or STTransformer (Song et al., 2021) in this comparison, as previous studies have consistently shown that EFMs significantly outperform these smaller supervised models. Our focus is therefore on evaluating the effectiveness of different foundation model designs and pretraining paradigms under comparable settings.” ) Can you provide more concrete evidence or experimental results to justify excluding these baselines from comparison? Also, how about other supervised models specifically  for each testing dataset?
4. Swin vs. Other Transformer-Based Decoders: Why is the Swin transformer decoder preferred over other transformer-based decoders? Are there specific advantages that make Swin more suitable for EEG analysis, and if so, can you provide additional experimental results or explanations?
5. Pretraining Time and Details: The paper lacks specific details about the pretraining time and other parameters. Could you clarify the pretraining setup, including the duration and computational resources involved, as well as any other important details that would help evaluate the practicality and scalability of the model?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a novel EEG foundation model named CodeBrain, with its core contributions as follows:  1) Designing the **TFDual-Tokenizer**, which generates discrete tokens by decoupling the heterogeneous temporal and frequency signals of EEG. This not only expands the representation space but also achieves domain-specific interpretability;  2) Proposing the **EEGSSM architecture**, which combines structured global convolution with sliding window attention. By simulating the brain's small-world topology, it efficiently captures both sparse long-range dependencies and local dependencies;  3) After pretraining on the large-scale public dataset, CodeBrain achieves superior generalization performance compared to 5 existing EEG foundation models across 8 downstream tasks (covering emotion recognition, sleep staging, etc.) and 10 datasets. Additionally, the robustness and scalability of the model are validated through ablation experiments and scaling law analyses.

### Strengths
1. **Novel Architectural Design**: The decoupling of time-frequency tokenization addresses a critical limitation in existing EEG models, where joint encoding often conflates neural dynamics. This aligns with neurophysiological understanding that brain activity operates across dissociable temporal and spectral dimensions. 
2. **Comprehensive Benchmarking**: Testing on 8 diverse tasks (spanning cognitive, motor, and clinical domains) provides strong evidence for generalizability, a key requirement for foundation models.
3. **Preliminary Interpretability**: The authors make meaningful preliminary efforts to link model components (i.e., temporal/frequency tokens) to known neural correlates (e.g., sleep spindles, slow waves, and spectral rhythms like delta activity).

### Weaknesses
This work demonstrates promising prospects for advancing EEG foundation models; however, addressing the following weaknesses is critical to validating its contributions to artificial intelligence and neuroscience.

## Model-Related Weaknesses

1. **Unsupported "Approximate Independence" Assumption in Proposition 2.1 (Section 2.2):** 

   A key concern is that EEG temporal representations $X_t$ and frequency representations $X_f$ are not independent—they exhibit inherent correlations. Yet, Proposition 2.1 assumes they are "approximately independent" to derive the distortion bound of the decoupled codebook, with no empirical analysis or theoretical justification provided. This casts doubt on the generalizability of Proposition 2.1.

2. **Missing Codebook Initialization Strategy for TFDual-Tokenizer (Section 2.2):** 

   The TFDual-Tokenizer uses temporal and frequency codebooks, but the **codebook initialization method** is unspecified. For vector quantization based models, initialization directly affects codebook utilization and training stability. The absence of this detail hinders understanding of how codebook initialization influences model performance.

3. **Unclear Structure and Mechanism of the "Lightweight Convolutional Module" (Section 2.3):** 

   The "lightweight convolutional module" for dynamic positional embeddings is poorly elaborated. Its unspecified architecture and the unexplained mechanism for capturing inter-channel correlations and adapting to unseen channels weaken the claims about the model's adaptability.

4. **Unvalidated Rationale for SGConv Kernel Parameters (Section 2.3):**

   The SGConv layer optimizes the convolution kernel $\bar{K}$ via "sparse parameterization" and "kernel decay," with the decay coefficient fixed at $\alpha=0.5$. However, no comparisons with other $\alpha$ values are conducted to assess impacts on model performance. The paper provides no basis for selecting $\alpha=0.5$ (e.g., cross-validation results, neurophysiological plausibility), relying solely on an empirical value.

## Experiment-Related Weaknesses

1. **Incomplete Baseline Comparisons (Section 3):**

   While the paper compares CodeBrain with 5 existing EEG foundation models, it omits traditional non-foundation models (e.g., EEGNet[1], EEG-Conformer[2]) as baselines. This makes it impossible to intuitively quantify the advantage of CodeBrain as a foundation model.

2. **Unexplained Importance of the Gating Mechanism (Section 3.4):**

   Ablation experiments show that removing the gating module causes a substantial drop in Cohen’s Kappa: 52.3% for FACED, 19.1% for SEED-V, and 28.6% for ISRUC S3. Despite this critical impact, the gating module is not a core innovation of the work. The paper fails to explain **why this specific gating design was adopted** (e.g., compatibility with EEGSSM) or **why it drives such significant performance gains**, weakening the justification for its inclusion.

3. **Insufficient Details for Reproducibility (Appendices I):**

   Appendix I lacks details on learning rate decay (e.g., warm-up steps for the Cosine scheduler), which is critical for stable training. The temperature parameter $τ$ for the contrastive loss (SimCLR) is not specified, and the rationale for its selection is not provided.

## Analysis-Related Weaknesses

1. **Inadequate Quantitative Support for Interpretability Claims (Figure 4):**

   Figure 4 illustrates associations between temporal tokens and slow waves, and frequency tokens and delta rhythms, but only reports percentage values. No **statistical significance tests** (e.g., comparing with the matching rate of random tokens to rule out chance correlations) are performed, leaving the interpretability conclusions without statistical validation.

2. **Missing Trade-off Analysis for Scaling Laws (Appendix N):**

   Appendix N shows diminishing performance returns as model parameters increase from 3M to 150M, but no analysis of **computational efficiency trade-offs** is provided (e.g., FLOPs, training time, memory usage for different model sizes). Without quantifying whether marginal performance gains justify increased computational costs, the work fails to guide model size selection for practical applications.

[1] Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.

[2] Song Y, Zheng Q, Liu B, et al. EEG conformer: Convolutional transformer for EEG decoding and visualization[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2022, 31: 710-719.

### Questions
1. What is the codebook initialization strategy for the TFDual-Tokenizer? Do different strategies have significant impacts?
2. What is the structure of the lightweight convolutional module? How does it capture inter-channel correlations?
3. What is the rationale for the SGConv kernel parameters?
4. What is the value of the temperature parameter τ in the SimCLR contrastive loss, and what is its selection basis?
5. Why is the gating mechanism so crucial for model performance?

### Soundness
3

### Presentation
3

### Contribution
3
