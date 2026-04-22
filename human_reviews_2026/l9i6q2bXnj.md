# ARINBEV: Bird's-Eye View Layout Estimation with Conditional Autoregressive Model

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Recent advances in Bird’s Eye View (BEV) layout estimation have advanced through refinements in architectural and geometric design. However, existing methods often overlook the structured relationships among traffic elements. Components such as drivable areas, lane dividers, and pedestrian crossings constitute an interdependent system governed by civil engineering standards. For instance, stop lines precede crosswalks, which align with sidewalks, while lane dividers follow road curvature. To capture these interdependencies, we propose \textbf{ARINBEV}, an autoregressive model for BEV map estimation. Unlike prior generative approaches that rely on complex multiphase training or encoder-decoder architectures, ARINBEV employs a single-stage, decoder-only autoregressive design. This architecture enables semantically consistent BEV map estimation. On nuScenes and Argoverse2, ARINBEV attains 64.3 and 65.6 mIoU, respectively, while using $1.7\times$ fewer parameters and training $1.8\times$ faster than state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the oversight of structured relationships among traffic elements in existing BEV layout estimation methods by proposing ARINBEV, a single-stage, decoder-only autoregressive model. Unlike two-stage or encoder-decoder frameworks, ARINBEV leverages class encoding and entropy-guided mask scheduling to bypass discrete representation learning. Evaluated on nuScenes and Argoverse2 datasets, it achieves state-of-the-art mIoU scores (64.3 and 65.6 respectively) with 1.7× fewer parameters and 1.8× faster training compared to existing models, demonstrating effectiveness in capturing semantic dependencies of traffic elements.

### Strengths
1. The core idea of exploiting conditional autoregressive modeling to capture structured dependencies among traffic elements (e.g., stop lines preceding crosswalks) aligns well with real-world transportation standards, bringing valuable domain awareness to BEV map estimation.
2. The single-stage, decoder-only architecture eliminates the inefficiencies of discrete representation learning and separate encoder-decoder stages, achieving superior performance with fewer parameters and faster training, which is practically meaningful for autonomous driving applications.
3. Comprehensive experiments on two large-scale datasets (nuScenes and Argoverse2) and detailed ablation studies (on masking strategies, feature compression, etc.) effectively validate the contribution of each component in the proposed framework.

### Weaknesses
1. The model’s advantage in this paper seems to lie in utilizing historical information via autoregressive modeling, but the authors fail to clarify whether baseline models use the same length of historical data or are video-based. This ambiguity makes it hard to attribute performance gains solely to the autoregressive design.
2. While the model achieves better results with fewer parameters and lower computation, the paper lacks in-depth insights explaining why the autoregressive paradigm inherently outperforms two-stage or encoder-decoder architectures in efficiency-performance trade-off.
3. Inference speed is critical for autonomous driving, yet the paper only reports training time and MACs. It does not compare end-to-end inference overhead (e.g., latency) across methods, which limits the evaluation of its practical applicability.
4. The ablation study on feature compression and masking strategies, though detailed, does not sufficiently discuss how these components interact with the autoregressive core, leaving gaps in understanding the model’s holistic working mechanism.

### Questions
1. Why is BEVFormer not included in the comparative experiments? Including it would better contextualize ARINBEV’s performance in the existing landscape.
2. For different ego trajectory types (e.g., straight, left turn, right turn, high-speed driving, start-stop), does the model’s performance improvement vary? If so, what factors contribute to these differences?
3. How does the Gaussian prior’s standard deviation (σ=0.5) in the entropy-guided masking strategy generalize to other datasets or scenarios beyond nuScenes and Argoverse2?
4. The paper mentions that excessive autoregressive inference steps lead to performance degradation. Is there a theoretical or empirical explanation for this phenomenon (e.g., overfitting to local dependencies)?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ARINBEV, a single-stage, decoder-only autoregressive model for BEV map layout prediction. The authors' key contribution is to demonstrate that learning discrete representations in existing two-stage generative models (such as VQ-VAE) is unnecessary given the semantic sparsity of BEV maps. To replace the discrete representations, the authors introduce class encoding and entropy-guided masking based on entropy maps for more efficient and concise training. Experimental results show that ARINBEV achieves new state-of-the-art performance on nuScenes and Argoverse2, while requiring faster training time and fewer model parameters.

### Strengths
- Simplicity and Efficiency: The authors have successfully simplified the complex two-stage or encoder-decoder generative framework into a single-stage, decoder-only autoregressive model, significantly reducing model complexity.

- Novel Motivation and Analysis: Through an in-depth analysis of the VQ-VAE codebook utilization and Shannon Entropy of BEV maps, the authors compellingly argue that the semantic sparsity of BEV maps renders discrete representation learning unnecessary.

- Sota Performance: ARINBEV achieves state-of-the-art mIoU performance on both the nuScenes and Argoverse2 datasets, with a training time of 73 hours, which is significantly less than the SOTA baseline (e.g., 131 hours for VQ-Map), while also having fewer parameters, demonstrating an impressive improvement in efficiency.

### Weaknesses
- Limited Novelty in Core Architecture: While the removal of VQ-VAE is a reasonable simplification, the core architecture of ARINBEV is essentially based on the decoder-only Transformer from BEVFormer and DETR, and it adopts the autoregressive paradigm of the Masked Generative Image Transformer (MaskGIT). The main innovation—category encoding—essentially maps sparse labels directly to Transformer input embeddings, and its complexity and generalizability still require further validation.

- Generalizability of Entropy-Guided Masking: The "Gaussian Prior Map S" (Equation 10) used for training is a fixed and predetermined prior that focuses on the central area of the BEV. It is merely an empirical, heuristic design rather than a dynamic strategy based on real-time semantic entropy. This undermines the rigor of the "entropy-guided" approach.

### Questions
1. Representational Capacity of Category Encoding: The authors convert discrete labels M into embeddings $S_{avg}$ through a learned embedding table E. Please elaborate on how this simple pixel-wise class encoding effectively captures the structural dependencies of traffic elements in the BEV map. How does this "hard-coding" ensure that its expressiveness is sufficiently robust compared to the contextually semantic discrete tokens learned by VQ-VAE?

2. Entropy Guidance and Gaussian Prior: In Section 3.2, the authors mention using an "entropy-guided masking strategy," but actually employ a fixed Gaussian prior map S (Equations 9-10). This Gaussian prior map S simply treats the central area as a high-information region, overlooking the significant differences in traffic scenarios within the BEV coordinate system (e.g., highways vs. complex intersections). The lack of an adaptive masking strategy for complex scenes may limit the model's performance in highly variable environments.

3. Purpose of Using a Fixed Gaussian Prior: As stated in L343, the purpose of employing a fixed Gaussian prior is to prevent the model from overfitting to the entropy distribution of a specific dataset. However, the map prediction network itself is already overfit to a particular dataset. There is no experimental evidence demonstrating that a fixed Gaussian prior is less prone to overfitting compared to an empirical entropy map derived from training data.

4. Decline in Multi-Step Inference Performance: Figure 6 shows that as the number of inference steps increases from 3 to 5, the mIoU declines, while FPS continues to decrease. What is the reason for the drop in mIoU? Is it due to overfitting (as you mentioned), or does the model introduce erroneous information in subsequent iterations?

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
This paper presents an empirical analysis showing that current generative BEV map estimation methods contain unnecessary complexity: in particular, discrete representation learning exhibits low codebook utilization and extremely low entropy on BEV maps, leading to feature underrepresentation. Building on this, the authors propose ARINBEV, a conditional autoregressive model with a single-stage, decoder-only design. It directly injects ground-truth BEV map labels into the embedding layer via Class Encoding, thereby avoiding the discrete quantization stage. In addition, the paper introduces an entropy-guided masking schedule that leverages the low-entropy property of BEV maps to focus learning on information-rich regions and capture the conditional dependencies among traffic elements. ARINBEV surpasses existing SOTA methods on nuScenes and Argoverse2, achieving clear improvements.

### Strengths
S1. This work brings a novel perspective to BEV layout estimation: through Shannon-entropy analysis and empirical measurements of codebook utilization, it systematically—for the first time—demonstrates the inefficiency of discrete representation learning on BEV maps.
S2. It proposes a single-stage, decoder-only autoregressive BEV generation framework, abandoning the prevalent two-stage VQ-VAE→Transformer and diffusion-based encoder–decoder pipelines; moreover, using Class Encoding, it feeds the labels directly as tokens, a design that is novel and well aligned with the semantic sparsity of BEV maps.
S3. The paper presents the method with clear detail: the autoregressive factorization, embedding normalization, masking schedule, and the compression to 25×25 for global self-attention as well as the cross-attention sampling settings are all documented in the main text and appendix. The appendix also provides PyTorch pseudocode, making the work reproducible.

### Weaknesses
W1. The inference procedure relies on iterative autoregressive steps, which leads to higher latency; as noted in the paper, the best case reaches only 7.5 FPS, lagging behind non-generative baselines such as BEVFusion. This setup may constrain deployment in dynamic driving scenarios.
W2. The entropy-guided masking uses a fixed Gaussian prior with p=0.5p=0.5, but the ablation study shows sensitivity to this parameter—for example, the appendix notes that σ=0.3 leads to training failure. This introduces potential instability, and the heavy reliance on a nuScenes-like distribution may degrade on datasets with different entropy patterns.

### Questions
Q1. Beyond the best setting at σ=0.5, if the Gaussian prior center is shifted, widened, or narrowed —or if the prior weight pp is varied from 0.5 to {0.25,0.75}—how does the performance trend? Do you observe noticeable fluctuations for classes in the peripheral regions?

### Soundness
3

### Presentation
2

### Contribution
2
