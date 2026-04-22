# Graph-Guided Reconstruction Diffusion for Multivariate Time Series Anomaly Detection

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
Time series anomaly detection often faces challenges such as non-stationarity and trendiness. Recently, unsupervised learning methods combined with generative models have shown promising prospects in this field, especially the application of multi-resolution technology in anomaly detection has achieved certain results. However, existing models usually ignore the correlations among different features in time series data and the rich multi-resolutional knowledge contained in the original data. To solve this problem, this paper proposes a new Model, \textbf{G}raph \textbf{G}uided \textbf{R}econstruction \textbf{D}iffusion Model (GGRD). GGRD is an end-to-end unsupervised anomaly detection model based on reconstruction. It adopts overlapping sliding Windows to sample multi-resolution data and integrates the similarity prior in the data into the \textbf{G}raph-\textbf{G}uided \textbf{A}ttention (GGA) mechanism, thereby effectively dealing with complex characteristics such as non-stationarity and cross-variable correlations of time series. The experimental results show that GGRD significantly outperforms the existing methods on multiple real datasets. Code is available at \url{https://anonymous.4open.science/r/GGRD-806F/}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a time series anomaly detection method that integrates diffusion models with a graph attention mechanism. Specifically, the authors perform a horizontal multi-scale segmentation with noise injection and a vertical patch-based segmentation followed by cosine similarity computation. The resulting similarity matrix is treated as a graph, where each patch serves as a node, and the graph attention network is employed to reconstruct the denoised sequence. Finally, the proposed framework is evaluated on multiple real-world datasets, demonstrating its effectiveness.

### Strengths
1. The authors have provided their code, which improves the reproducibility and transparency of the study.
2. The proposed method achieves top-tier performance across multiple real-world datasets, demonstrating good empirical effectiveness.
3. Ablation studies are provided, validating the contribution of each component in the framework.

### Weaknesses
1. **Unclear story and motivation.**
   The manuscript fails to present a coherent narrative: it is not clear what concrete problem the authors are solving and why the proposed design choices (e.g., multi-resolution segmentation and similarity injection) are necessary. This issue starts in the *Introduction* and permeates the whole paper. In the *Methods* section, the presentation reads like a collection of disconnected subsections rather than a logically flowing design—readers are left uncertain how components interact and why each one is required.

2. **Limited novelty — component stacking.**
   The proposed pipeline largely consists of assembling existing components (multi-resolution decomposition, timestamp/hard embeddings, and similarity-graph construction [1-3]) without introducing fundamentally new algorithmic ideas. Worse, these components are treated almost independently and run in parallel, which weakens the claim of a unified, principled method.

3. **Problematic use of a “diffusion” framework.**
   The paper claims to be diffusion-based but abandons the core iterative SDE/ODE denoising paradigm by asserting one-step reconstruction from a noised sequence to a real sample. While such a shortcut may be used during training [4], the literature and theory suggest that iterative denoising is required at test time to obtain good reconstructions—especially when starting from high-noise or near-pure-noise states. The current presentation glosses over this mismatch between training and inference, undermining the validity of calling the method a diffusion framework.

4. **Scalability and computational cost of graph-guided design.**
   The similarity-guided graph construction appears to rely on small patch sizes for good performance (see Appendix Fig. 7). Small patches imply many graph nodes as sequence length grows, causing an increase in computation and memory. Given convergence and complexity concerns, I am not convinced this design is a generally viable optimization for long-term time series.

5. **Experimental issues and insufficient analysis.**
   (a) The use of point-adjusted F1 (F1-PA) is problematic because F1-PA can be inflated by trivial strategies and is regarded by parts of the community as an overly permissive metric. Relying on F1-PA without complementary, stricter evaluation undermines the experimental claims.
   (b) The “Visualization of Similarity-guided Graph Tensor” is not well motivated—it is unclear what this visualization is intended to demonstrate and whether the similarity tensor itself is learnable or fixed. Such analysis would be better placed in the motivation or method section, accompanied by a discussion of what the observed patterns imply for detection behavior.

6. **Writing quality.**
   The manuscript contains multiple grammatical and typographical errors throughout. The paper would benefit from a careful language and copy edit to improve clarity.

**Reference:**  
[1] Zhong, Guojin, et al. "Multi-resolution decomposable diffusion model for non-stationary time series anomaly detection." The Thirteenth International Conference on Learning Representations. 2025.  
[2] Wang, Chengsen, et al. "Drift doesn't matter: Dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection." Advances in neural information processing systems 36 (2023): 10758-10774.  
[3] Shen, Lifeng, Weiyu Chen, and James Kwok. "Multi-resolution diffusion models for time series forecasting." The Twelfth International Conference on Learning Representations. 2024.  
[4] Yuan, Xinyu, and Yan Qiao. "Diffusion-TS: Interpretable Diffusion for General Time Series Generation." The Twelfth International Conference on Learning Representations.

### Questions
See the Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Graph-Guided Reconstruction Diffusion Model (GGRD) for multivariate time series anomaly detection. By integrating multi-resolution temporal features and similarity-guided priors, the model aims to capture dynamic dependencies across features and improve anomaly detection performance.

### Strengths
The idea of incorporating multi-scale graph construction for anomaly detection is relatively novel and shows an awareness of the multi-resolution nature of time-series data.

### Weaknesses
1.Limited novelty. The main components of the mode, e.g., sliding-window multi-scale decomposition, cosine-similarity graph construction, and graph attention modules, are well-established techniques. The combination lacks substantial methodological innovation.

2.Marginal contribution relative to prior work. The paper claims that existing models ignore correlations among features, yet many recent methods (especially graph-based ones) already model these dependencies explicitly.

3.Insufficient theoretical justification. The paper uses cosine similarity to build graphs that serve as priors for reconstruction, but it is unclear why such a static prior can adapt to the evolving relationships among features over time.

4.Diffusion process inconsistency. The proposed one-step reconstruction in the diffusion framework contradicts the multi-step denoising procedure typically used during training. The paper does not clarify how this inconsistency is reconciled or why it maintains diffusion-based probabilistic consistency.

5.Unclear graph construction details. While the paper defines the similarity computation, it does not explain how neighbor nodes are selected or how the number of neighbors affects model performance.

6.Limited dataset coverage. The experiments are restricted to industrial control and server datasets (SMD, PSM, SWaT) and lack evaluations on diverse domains such as finance, transportation, or healthcare, which limits the generality of the conclusions.

7.Missing references for baselines. Baseline methods are listed without explicit citations.

### Questions
See the weaknesses section.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the Graph Guided Reconstruction Diffusion Model (GGRD) method for MTSAD. Traditional models often struggle with non-stationarity, complex feature dependencies, and high computational costs. GGRD addresses these challenges by introducing a multi-resolution decomposition technique, an efficient one-step reconstruction process, and a Graph-Guided Attention mechanism to model dynamic cross-feature dependencies. GGRD leverages overlapping sliding windows for multi-resolution data generation and incorporates similarity-guided graph tensors to improve feature interactions. Experimental results demonstrate that GGRD outperforms existing methods across multiple real-world datasets.

### Strengths
1) The integration of a Graph-Guided Attention (GGA) mechanism, combined with multi-resolution data decomposition, is a significant advancement over traditional anomaly detection models. The approach ensures better modeling of complex feature dependencies and temporal patterns, addressing the challenges of non-stationarity and cross-variable correlations.

2) The use of a one-step reconstruction strategy, as opposed to iterative denoising processes, improves the model’s computational efficiency, making it more practical for real-time applications.

3) This paper extensively evaluates the proposed method against several baseline models and demonstrates superior performance on multiple datasets.

### Weaknesses
1) This paper does not adequately explain the effectiveness and necessity of using graphs as denoising networks.

2) The design of the graph depends heavily on cosine similarity, which assumes that the most relevant dependencies in time series data are linear or can be captured by a similarity function. This design overlooks the possibility that relationships in MTS can be highly non-linear and context-dependent, where cosine similarity might fail to adequately model more complex interactions.

3) While GGRD models the relationship between features across time steps, it appears to focus primarily on the dependencies at local time scales and between features. It might struggle to capture long-term temporal dependencies that evolve slowly over time, especially for datasets where long-term trends or periodicity play a critical role.

### Questions
1) While the proposed method has advantages in capturing global trends, I wonder if it remains effective when locating and detecting infrequent and localized anomalies.

2) See weakness.

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
3

### Summary
This paper proposes GGRD (Graph-Guided Reconstruction Diffusion), an unsupervised anomaly detection framework for multivariate time series. The method addresses three key limitations in existing diffusion-based approaches: (1) non-stationarity across resolutions using overlapping sliding windows for smooth multi-resolution decomposition, (2) computational inefficiency through one-step reconstruction instead of iterative denoising, and (3) inadequate modeling of cross-feature dependencies via a Graph-Guided Network with Graph-Guided Attention (GGA) that injects similarity-based priors.

### Strengths
- Clearly identifies limitations in existing methods (staircase artifacts, iterative denoising costs, missing feature dependencies)
- The overlapping sliding window approach (Section 4.2) elegantly addresses artifacts from non-overlapping pooling. Figure 5 provides compelling empirical evidence
- The integration of similarity priors into attention (Eq. 5) is intuitive and explicitly models cross-feature dependencies often ignored in prior work
- Table 2 systematically evaluates each component's contribution, demonstrating that GGA provides the largest performance gain (0.84 → 0.75 F1 on PSM)

### Weaknesses
- This is claimed as a major contribution but lacks rigorous analysis. What information is lost compared to full DDPM? How does reconstruction quality compare at different noise levels $K$? The paper should include:
    - Ablation comparing 1-step vs. multi-step denoising
    - Analysis of reconstruction error vs. $K$
    - Theoretical argument for why one step suffices
- No error bars, multiple runs, or significance tests reported. The 0.2% average improvement over MODEM could be within noise margins. Need: 
    - Multiple runs with different random seeds
    - Standard deviations and confidence intervals
    - Statistical significance tests (e.g., paired t-test)
- Cosine similarity (Eq. 1) is simplistic and may miss non-linear relationships. The authors acknowledge this (Section 6) but should explore:
    - Learned graph construction
    - Attention-based similarity
    - Comparison with other similarity metrics

The paper has significant presentation issues that hinder comprehension:
1. Figure 1 is cluttered, Figure 3 needs better layout
2. Numerous grammatical issues (e.g., "trendiness" should be "trends," "time segmen" should be "segment")
3. The patch size $P$ relationship to time segments is never clearly defined. How exactly are patches created at each resolution?
4. The connection between resolution levels and reconstruction steps needs clearer exposition. The relationship $m \in [1, R-1]$ processing resolution $R-m$ is confusing

### Questions
1. Can you provide empirical comparison of 1-step vs. multi-step (e.g., 10, 50, 100 steps) denoising showing reconstruction quality and computational cost? What is the theoretical basis for single-step sufficiency?
2. What are the mean and standard deviation of F1 scores across multiple runs? Are the improvements over baselines statistically significant?
3. Have you experimented with learned graph structures (e.g., via graph neural networks) or other similarity measures? How sensitive is performance to the choice of cosine similarity?
4. Can you provide wall-clock time comparisons with MODEM, D3R, and ImDiffusion? What is the memory footprint? Does one-step reconstruction actually provide practical speedup?
5. Beyond the post-hoc explanation, what modifications could improve performance on short-burst anomalies? Could adaptive time segmentation help?

### Soundness
3

### Presentation
2

### Contribution
2
