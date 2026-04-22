# Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Degradation-agnostic image restoration aims to handle diverse corruptions with one unified model, but faces fundamental challenges in balancing efficiency and performance across different degradation types. Existing approaches either sacrifice efficiency for versatility or fail to capture the distinct representational requirements of various degradations. We present MIRAGE, an efficient framework that addresses these challenges through two key innovations. First, we propose a channel-wise functional decomposition that systematically repurposes channel redundancy in attention mechanisms by assigning CNN, attention, and MLP branches to handle local textures, global context, and channel statistics, respectively. This principled decomposition enables degradation-agnostic learning while achieving superior efficiency-performance trade-offs. Second, we introduce manifold regularization that performs cross-layer contrastive alignment in Symmetric Positive Definite (SPD) space, which empirically improves feature consistency and generalization across degradation types. Extensive experiments demonstrate that MIRAGE achieves state-of-the-art performance with remarkable efficiency, outperforming existing methods in various all-in-one IR settings while offering a scalable and generalizable solution for challenging unseen IR scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MIRAGE, an efficient framework for degradation-agnostic image restoration. The key contributions include a channel-wise functional decomposition strategy that integrates convolution, attention, and MLP modules to handle diverse degradations, and a manifold regularization method using cross-layer contrastive learning in the Symmetric Positive Definite (SPD) manifold space. Extensive experiments demonstrate that MIRAGE achieves state-of-the-art performance with significantly fewer parameters and computational costs compared to existing methods, while generalizing well to unseen degradation scenarios.

### Strengths
* The SPD manifold-based contrastive learning is a well-motivated approach for cross-layer feature alignment, effectively leveraging second-order statistics for more robust representation learning compared to Euclidean space.
* The proposed channel-wise decomposition provides a principled and efficient method to harness the complementary inductive biases of convolution, attention, and MLPs, resulting in an excellent performance-efficiency trade-off.

### Weaknesses
* The experimental comparison in Tab. 1, while extensive, omits the cited approach DA-RCOT [1] that also employs a contrastive learning objective (over the residual feature space). A direct comparison would more firmly establish the advantages of the proposed SPD manifold approach over alternative contrastive formulations (SPD vs. residual feature space).

* The related work section could be enhanced by discussing other contrastive paradigm in All-in-one image restoration, highlighting the contribution of the proposed cross-layer contrastive losses over the SPD feature space.

* The motivation of introducing SPD features to enhance the generalization is not very explicit to the readers. Further explanations or adding a Teaser Figure at the beginning would strengthen the presentation.

[1] Xiaole Tang, Xiang Gu, Xiaoyi He, Xin Hu, and Jian Sun. Degradation-aware residual-conditioned optimal transport for unified image restoration. TPAMI, 2025.

### Questions
See weaknesses.

### Soundness
3

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
3

### Summary
This paper proposes MIRAGE, a lightweight, efficient, and degradation-agnostic image restoration framework that unifies multiple degradations (noise, haze, rain, blur, low-light) under one architecture. MIRAGE achieves SOTA PSNR/SSIM across 3-, 5-, and mixed-degradation benchmarks, with significantly fewer parameters (6M–10M) and lower FLOPs (16G–27G). The model also generalizes to zero-shot underwater enhancement and adverse weather removal.

### Strengths
1. The paper presents a well-motivated and compact architecture that repurposes attention-channel redundancy through a channel-wise functional decomposition (Conv/Attn/MLP branches), aligning with the natural inductive biases of different degradations—local texture (Conv), global context (Attn), and channel statistics (MLP). 
2. The proposed SPD manifold alignment introduces cross-layer contrastive learning between shallow and latent features using covariance-based second-order statistics. This prevents feature collapse common in Euclidean contrastive learning and promotes consistent, degradation-agnostic representations. Ablations show clear gains.
3. MIRAGE achieves consistent SOTA or near-SOTA performance across a wide range of benchmarks—3/5 degradation settings, CDD11 composite degradations, adverse weather, and zero-shot underwater enhancement—while being significantly smaller and faster than strong baselines such as MoCE-IR, PromptIR, and AdaIR, demonstrating an excellent efficiency–performance balance.
4. The analysis of channel redundancy via PCA/SVD, shallow–latent similarity matrices, and qualitative contrastive alignment visualizations all support the design’s intuition.

### Weaknesses
1. While the paper’s SPD-based manifold regularization is empirically effective, its mathematical formulation remains heuristic. The method projects SPD matrices back to Euclidean space for InfoNCE optimization, without leveraging true Riemannian metrics or geodesic distances. As a result, the claimed “manifold alignment” lacks formal theoretical justification.
2. The experiments comprehensively compare with transformer- and prompt-based all-in-one models (PromptIR, MoCE-IR, AdaIR), but omit recent diffusion-based or state-space approaches that are increasingly strong in low-level restoration.
3. Although MIRAGE handles five degradation types and several composites, it is still trained on synthetic datasets. The paper would be stronger with real-world or camera-captured degradation benchmarks, or cross-domain generalization beyond underwater scenes.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MIRAGE, an efficient degradation-agnostic image restoration framework based on channel-wise functional decomposition and manifold regularization in Symmetric Positive Definite (SPD) space. The method splits features into convolution, attention, and MLP branches to handle local textures, global context, and channel statistics, respectively. Additionally, a shallow-latent contrastive alignment in SPD manifold space is introduced to enhance generalization across heterogeneous degradation types. The authors evaluate MIRAGE across five restoration settings, including single degradation, mixed degradations, adverse weather restoration, and zero-shot underwater enhancement.

### Strengths
1. The SPD-space contrastive learning is a novel contribution that effectively preserves high-order statistics, outperforming Euclidean contrastive counterparts in both stability and generalization, as evidenced by ablation studies.
2. The channel-wise functional decomposition is conceptually clear and well-motivated by empirical redundancy analysis. It provides an elegant way to repurpose redundant channels into complementary submodules, rather than simply increasing model capacity.
3. The analysis of attention redundancy and shallow-latent feature disparity provides a strong justification for the proposed framework.

### Weaknesses
1. The two main contributions claimed in the introduction are largely based on the combination and engineering integration of existing techniques, rather than presenting any genuinely novel innovation or introducing a fundamentally new perspective.
2. While the contributions are substantial, the paper is dense and may be challenging for readers unfamiliar with SPD manifolds or contrastive learning in feature covariance space. Some mathematical formulations could benefit from more intuitive explanations or visual aids.
3. Although the paper claims efficiency advantages, it lacks direct comparison to recent diffusion-based restoration models that are increasingly influential in the community.
4. The zero-shot evaluation is limited to a single domain. Including additional domains (such as low-light) would further validate the claimed generalization ability.
5. The paper primarily focuses on average performance improvements, but does not discuss scenarios where MIRAGE underperforms or struggles, which would provide a more balanced understanding of robustness.

### Questions
See the above parts.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a unified image restoration framework, with channel-wise decomposition for different strategy, including convolution, attention, and MLP. Additionally, the contrastive loss is introduced between shallow and deep features to encourage shared representations. Experiments show effectiveness of the method.

### Strengths
1. The writing is clear and easy reading.
2.  Experiments show the effectiveness of the proposed method.

### Weaknesses
1. The motivation of this work that using different strategy to handle agnostic degradations is much more invested in previous research, and the innovation is somewhat minimal;
2. There is no solid clarification that the shared representation between shallow and deep features are benefited for diverse degradations, which makes the contrastive objective not convincing, and the ablation improvement is minimal for probable performance jitter;
3. The main experiments in Tab. 3 and 4 may not show the superiority of the proposed method with subtle improvements.

### Questions
1. How to decide the channel role in channel-wise decomposition for convolution, attention, and MLP, is there any principle?

### Soundness
2

### Presentation
3

### Contribution
2
