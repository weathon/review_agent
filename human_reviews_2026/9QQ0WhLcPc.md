# TimeGEN: A Cross-Domain and Generative Model for Time Series Forecasting

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We propose TimeGEN, a lightweight, MLP-based generative deep learning architecture for Transfer Learning in time series forecasting. We use a variational encoder to capture high-level temporal representations across diverse series and domains. To further strengthen this generalization, we combine a reconstruction and forecasting loss, which shapes the latent space to retain local detail while capturing global predictive dependencies. In addition, temporal normalization ensures robustness to varying input scales and noise. To capture multiscale dynamics, we integrate a modular decoder that combines neural basis expansion with multi-rate interpolation, balancing long-range trends with high-frequency variations. Extensive empirical results across ten public datasets demonstrate that TimeGEN consistently outperforms SOTA methods in zero-shot and cross-domain settings. In cross-domain settings, it reduces forecasting error by more than 8% and up to 38%, while achieving a 2-30x speedup in training time compared to SOTA MLP and Transformer methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TimeGEN, a novel, lightweight, MLP-based architecture designed for transfer learning (TL) in time series forecasting. The primary goal is to create a model that excels in zero-shot and cross-domain scenarios, where it must make predictions on data from unseen domains.

### Strengths
1. The paper's main strength is its originality in creatively combining existing ideas (like VAEs and MLP-based models) into a novel, lightweight architecture. This hybrid model is specifically engineered to solve the difficult problem of cross-domain forecasting.

2. Another major strength is the quality of its experiments. The authors perform a rigorous evaluation across ten different datasets and, crucially, test the model in four distinct settings, including three challenging zero-shot and transfer-learning scenarios. This provides robust evidence for its claims.

3. The work is seen as highly significant for practical reasons. It successfully tackles poor generalization, a major bottleneck in forecasting. By doing this with a model that is 2-30x faster than competitors, it provides a compelling and efficient alternative to the "heavier is better" trend dominated by large Transformer models.

### Weaknesses
1. Disconnect Between Generative and Forecasting Objectives

The paper's central premise is that a "generative" VAE-based architecture is superior for forecasting transfer learning. However, the connection between these two goals is weak and poorly supported by the experiments.

Conflicting Goals: The generative goal (reconstructing the past, $X_t$) and the forecasting goal (predicting the future, $Y_{t+1:t+h}$) are not inherently aligned. Optimizing for a perfect, high-fidelity reconstruction can force the model to learn noisy, high-frequency details from the input that are useless or even detrimental to forecasting.

Missing Analysis: The paper provides no analysis to show these two objectives are synergistic. The authors should have included:

A qualitative analysis showing the model's reconstructions ($\hat{X}_t$).

A visualization of the latent space ($z$) to show whether it clusters semantically similar time series (e.g., by domain or seasonality), which would support the claim that it captures "high-level representations."

Without this, the "generative" component feels like an unproven and potentially unnecessary complication.

2. Unclear Source of Zero-Shot Generalization

The paper claims SOTA zero-shot performance but fails to convincingly attribute this capability to its novel architecture. The skepticism that a simple MLP-based model can "naturally" achieve this level of generalization is warranted because the paper fails to provide the necessary evidence.

Insufficient Ablation: The ablation study (Table 2) is too limited. It shows that removing components hurts performance, but it doesn't isolate the key ingredient. A crucial missing experiment is comparing TimeGEN to a non-variational counterpart (i.e., a standard autoencoder with only the $\text{Recon}(\cdot)$ loss, no $\text{KL}(\cdot)$ term).

Unanswered Questions: It is unclear if the success comes from the VAE's latent $z$, the joint loss, the specific decoder, or simply the temporal normalization. The paper does not provide the analysis needed to distinguish between these factors, making the core architectural claims unsubstantiated.

3. Lack of Detail on Variational Training Dynamics

The paper introduces a Variational Autoencoder, including a $\beta \cdot \text{KL}(q_{\phi}(z|X_t) || \mathcal{N}(0,I))$ term in its loss function, but completely glosses over the significant challenges associated with training such a model.

Training Instability: VAEs are notoriously difficult to train. They can suffer from "posterior collapse" (where the KL term goes to zero and the latent space is ignored) or require careful balancing of the reconstruction, forecasting, and KL loss terms.

Missing Details: The paper mentions $\beta$ is "linearly annealed" (Section 3.3) but provides no further details. This is a critical hyperparameter. A robust analysis would require details on the annealing schedule, the final value of $\beta$, and the sensitivity of the model's performance to this parameter. This omission is a significant gap in reproducibility and a major unaddressed technical challenge.

### Questions
See the weakness.

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
3

### Summary
This paper proposes TimeGEN, an efficient temporal generative encoder network focusing on transfer learning for time series forecasting. Using a variational encoder, it extracts global latent temporal features shared across different series and domains. Its modular decoder architecture captures multiscale dynamics via neural basis expansion and hierarchical interpolation. The authors extensively test TimeGEN across standard and transfer learning scenarios on heterogeneous public datasets. Results show good generalization, particularly in cross-domain and zero-shot tasks, with substantial speedups in training compared to Transformer and other MLP-based approaches.

### Strengths
1. TimeGEN reduces zero-shot and cross-domain forecasting errors compared to state-of-the-art methods.
2. Achieves speedup in training time relative to Transformer and other MLP-based models, making it highly scalable.
3. Uses a variational encoder with a joint reconstruction and forecasting loss to shape a latent space capturing both local and global temporal dependencies.
4. Combines neural basis expansion for high-frequency patterns and multi-rate interpolation for long-term trends, balancing local detail and global structure.

### Weaknesses
1. The author mentioned GENERATIVE in their title, but what specific generative way should be in their method? The author should clarify it.
2. Since it is claimed as a generative model, and the LLM is an unignorable technical in GenAI. It is suggested to compare their method with some LLM-based time series forecasting methods.
3. I am concerned about the necessity of Figure 2.
4. The framework is defined as a lightweight structure. How to prove this claim?
5. I am concerned about the decoder part. Why does the author introduce two blocks? Will it improve the complexity?

### Questions
See above

### Soundness
2

### Presentation
2

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
The paper introduces TimeGEN, a lightweight, MLP-based generative deep learning architecture for time series forecasting, specifically aimed at addressing challenges in cross-domain transfer learning. TimeGEN uses a variational encoder to capture high-level temporal representations across diverse time series domains, and employs a modular decoder architecture for multiscale forecasting. The model integrates reconstruction and forecasting loss functions, along with temporal normalization to ensure robustness against varying scales and noise. Empirical evaluations across ten public datasets demonstrate that TimeGEN outperforms state-of-the-art (SOTA) methods in zero-shot and cross-domain settings, achieving significant improvements in forecasting accuracy and computational efficiency.

### Strengths
Strengths:
1. The studied problem is interesting and important for AI area.

2. The proposed TimeGEN is significantly more efficient in terms of training time, offering a 2–30× speedup compared to more complex models like Transformers.

### Weaknesses
Weaknesses:

1. There is a lack of comparison with the latest relevant literature. The most recent methods compared in this paper are TimeMOE (ICLR 2025) and KAN (ICLR 2025). However, TimeMOE is a method for large-scale basic time series models, and its core achievement is in large-scale, efficient, and general time series prediction, rather than specifically for cross-domain time series prediction, which is the focus of this paper. KAN networks, on the other hand, focus on more general neural network models, rather than time series problems. Other comparison methods are limited to those before 2023, lacking comparison with the latest and most relevant results, such as Unitime [s1] and LPTM [s2].

2. While the empirical results are good, the paper does not provide a deep theoretical explanation for some of the architectural decisions, such as the combination of reconstruction and forecasting loss. There is limited discussion on why certain components, such as temporal normalization or the specific use of MLP-based models, are preferable over other design choices.

3. While the ablation study presents useful insights into the model's components, the exploration could be expanded further. The removal of certain blocks (e.g., temporal normalization or the deep latent conditioning) leads to large performance drops, yet the precise role of each component in the architecture is not fully unpacked. More granular analysis of how each component contributes to the overall performance, especially in cross-domain settings, would clarify the model’s design rationale.


[s1] Liu, Xu, et al. "Unitime: A language-empowered unified model for cross-domain time series forecasting." Proceedings of the ACM Web Conference 2024. 2024.

[s2] Prabhakar Kamarthi, Harshavardhan, and B. Aditya Prakash. "Large Pre-trained time series models for cross-domain Time series analysis tasks." Advances in Neural Information Processing Systems 37 (2024): 56190-56214.

### Questions
You report 2–30× faster training times (Table 1). Are these gains primarily due to architectural simplicity (MLP-only) or optimization choices (e.g., batch size, windowing, or fewer parameters)?

### Soundness
3

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
The paper proposes a lightweight, MLP-based generative framework named TimeGEN for cross-domain time series forecasting. It employs a variational encoder for transferable high-level temporal representations, a modular decoder for long-range trends with high-frequency variations, temporal normalization for varying input and trains the model on objective that combines reconstruction, forecasting, and variational regularization to enhance generalization. Experiments on ten datasets demonstrate superior zero-shot and cross-domain generalization with faster training speed.

### Strengths
1.	TimeGEN adopts MLP-based architecture rather than a Transformer, contributing to the growing line of MLP-based foundation models for time series forecasting. This design choice offers a promising and efficient alternative to attention-based approaches.
2.	The model design is computationally efficient and demonstrates strong empirical performance when compared against seven diverse baseline methods, indicating its effectiveness across diverse architectures.
3.	The experimental setup is comprehensive, covering both full-shot and three zero-shot transfer settings. The inclusion of ten datasets provides sufficient evidence of the model’s robustness and generalization ability across diverse domains.

### Weaknesses
1.	The proposed model adopts a relatively simple architecture with an MLP-based encoder. While the authors claim that it can capture high-level temporal representations across diverse domains, it remains unclear how such a simple structure is capable of modeling complex dependencies and non-stationary patterns that typically arise in heterogeneous time series scenarios. A more detailed explanation or empirical evidence would strengthen this claim.
2.	Although the baseline selection covers multiple architecture families, the overall comparison remains limited — only seven baselines are included, among which there is only one MLP-based model and one state-of-the-art foundation model. This makes the evaluation less convincing. The authors are encouraged to include more recent and relevant baselines, especially MLP-based architectures and stronger foundation models, to ensure a fair and comprehensive comparison.
3.	As reported in Table 1, the proposed model achieves the fastest training speed (normalized = 1.0) compared with PatchTST (≈ 5× slower). This likely implies that the model has substantially fewer parameters. However, it raises the question: how can a small-capacity model effectively encode and transfer diverse pre-trained knowledge across domains? Moreover, both TimeGEN and TSMixer are MLP-based, yet the reported training time of TSMixer (31.042× slower) appears inconsistent. The authors should clarify this discrepancy and explain the source of such a large gap.
4.	The paper emphasizes temporal normalization and the joint reconstruction–forecasting loss as key components. However, both techniques are common practices in recent time series models and should not be presented as methodological contributions. The authors are advised to reposition these components as supporting techniques rather than major novelties.

### Questions
Please check above section.

### Soundness
3

### Presentation
2

### Contribution
3
