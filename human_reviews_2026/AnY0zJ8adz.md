# FaithfulFaces: Pose-Faithful Facial Identity Preservation for Text-to-Video Generation

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Identity-preserving text-to-video generation (IPT2V) empowers users to produce diverse and imaginative videos with consistent human facial identity. Although existing open-source and commercial methods have demonstrated impressive performance in typical scenarios, they still face significant limitations when confronted with challenging cases, such as large facial pose variations or facial occlusions. These challenges frequently result in identity distortion in the generated videos. In this paper, we propose FaithfulFaces, a pose-faithful facial identity preservation learning framework to improve IPT2V in complex dynamic scenes. Specifically, FaithfulFaces first proposes a pose-shared identity aligner that refines and aligns facial poses across distinct views via a pose-shared dictionary and a pose variation–identity invariance constraint. Then, the well-learned aligner can capture the global facial pose representation from the input single-view face image with explicit Euler angle embeddings, which could provide a pose-faithful facial prior for foundational generative models to better preserve identity in the generated videos. In particular, we develop a high-quality video dataset pipeline featuring substantial facial pose variations specifically for our FaithfulFaces to facilitate robust training. Compared to other IPT2V methods, FaithfulFaces achieves state-of-the-art performance across multiple metrics, generating high-quality videos with clear facial structures and consistent identity preservation, even as facial pose changes and occlusions occur. The code and dataset pipeline will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a pose-shared identity aligner that learns a dictionary-based global facial-pose representation from a single reference image, optimized via a pose variation–identity invariance contrastive loss and explicit Euler-angle embeddings.  A data pipeline selects single-subject videos with strong pose variation and produces descriptive prompts.  Across diverse prompts and identities, the method improves FaceSim and FID while maintaining text alignment, outperforming open-source and commercial baselines on qualitative and quantitative evaluations.

### Strengths
1. A pose-shared dictionary with Euler-angle embeddings gives the model an explicit global head-pose prior from a single image, reducing identity drift during motion. 

2. The pose variation–identity invariance (InfoNCE) loss aligns poses of the same identity and discourages collapse, supported by an information-theoretic view and t-SNE plots. 

3. A pipeline filters single-subject videos by measured pose variation and auto-generates prompts, yielding 51,624 samples focused on difficult pose dynamics. 

4. FaithfulFaces delivers best FaceSim-Cur/Arc, lower FID, and solid CLIPScore across 600 test videos, surpassing open-source and commercial baselines.

### Weaknesses
1. The pipeline and aligner rely on 6DRepNet head-pose estimates and a fixed variation threshold (120); mis-estimation or thresholding bias could skew data and priors. 

2. Training uses VACE-14B with LoRA on 32 H20 GPUs. Reproducibility for typical labs may be limited without ablations on smaller backbones or ranks. 

3. Some baselines are commercial black boxes and others use different foundations (Wan, CogVideoX, Hunyuan), complicating strictly like-for-like attribution. 

4. The method targets single-subject, face-centric videos. Multi-person scenes, rapid occluders, or non-frontal identity cues (gait, hair) are not deeply evaluated.

### Questions
1. The robustness of aligner when the pose estimator is noisy or biased across demographics should be validated.

2. Can the dictionary be adapted online to a new identity with few frames without retraining the generator?

3. How does performance scale on lighter backbones or lower-rank LoRA, and what is the compute–quality trade-off?

4. The extensibility of method  to multi-subject scenes with identity disambiguation and cross-shot continuity should be added.

### Soundness
3

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
This paper introduces FaithfulFaces, a novel approach designed to overcome the limitations of existing identity-preserving Text-to-Video (T2V) methods, particularly when facing large facial pose variations or occlusions. These challenging scenarios often lead to identity distortion in generated videos. The core contribution is the pose-shared identity aligner module. The pose-shared identity aligner is engineered to extract robust, pose-disentangled identity features from an identity image and spatially align them with the pose information derived from the driving image. This spatially-aligned identity feature is then injected into multiple layers of the T2V U-Net architecture via a spatial modulation mechanism. Experimental results demonstrate that FaithfulFaces significantly improves identity consistency, especially under extreme pose changes, outperforming current state-of-the-art techniques.

### Strengths
1. The pose-shared identity aligner is a clever and effective design. By pre-aligning the identity features based on the shared pose context, it effectively mitigates the issue of identity feature misalignment under large pose transformations, which is a major weakness in many identity-embedding approaches.
2.  The quantitative (especially ID preservation metrics) and qualitative results strongly validate the method's superiority in large pose cases.
3. The method is modularly designed as a plug-in component, allowing for straightforward integration into prevalent T2V diffusion model architectures (e.g., U-Net), which enhances its practical applicability and adoption potential.

### Weaknesses
1. The introduction of the pose-shared identity aligner module and multiple spatial modulation layers inevitably increases the model's parameter count and inference latency. The paper currently lacks a detailed comparison of inference speed, memory consumption (VRAM), and model size relative to the baseline T2V model. This information is vital for real-world deployment.
2. Although the generated videos appear smooth qualitatively, the paper does not include specific quantitative metrics for video temporal consistency (e.g., using metrics like LPIPS on consecutive frames or Frame Coherence metrics). Temporal stability is crucial, especially when large, frame-by-frame pose changes are involved.
3. The entire focus is on facial identity. While successful, the PFA provides no direct benefit or conditioning for non-face elements (e.g., clothes, background, scene style). This is an inherent limitation of the identity-specific approach, though not a flaw in the paper's core claim.
4. The current ablation studies are insufficient to conclusively determine the contribution of the pose-shared identity aligner. Further ablation studies are required, specifically by replacing the feature used (e.g., substituting the current feature with an ArcFace feature or another comparable representation) to isolate and measure the aligner's specific effect.

### Questions
1. The success of pose-shared identity aligner relies on extracting a pure, pose-disentangled identity feature. How robust is the identity extraction process when the reference image itself is highly non-frontal (e.g., a strong side profile), partially occluded, or of poor quality? Could the authors provide a comparative result showing the ID consistency metrics (e.g., ID loss) when the reference image is deliberately a non-frontal view? This would better validate.
2. I am confused regarding the feature interaction within the PIA. The process appears to involve independent feature reconstruction based on a shared dictionary, rather than a direct interaction between the two image features. While a shared dictionary implies some identity-related commonality, the mechanism by which this leads to a truly pose-invariant identity feature is not fully clear.


Minor
1.  Pose-shared Facial Aligner in Figure 2 and  pose-shared identity aligner in Figure 3. I suppose there are the same thing?

### Soundness
3

### Presentation
2

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
The paper proposes FaithfulFaces to address identity preservation in complex dynamic scenes. It introduces a pose-shared identity aligner and a dedicated video dataset pipeline with substantial facial pose variations. Experiments show that FaithfulFaces achieves state-of-the-art performance, generating high-quality videos with clear facial structures and consistent identity preservation.

### Strengths
(1) FaithfulFaces is compared with the closed-source methods Vidu and Kling, demonstrating the superiority of the proposed approach.

(2) The effectiveness of the method is validated through both quantitative and qualitative experiments, and the contributions of each module are analyzed via ablation studies.

### Weaknesses
The paper lacks citations, comparisons, or discussions of prior representative work such as Concat-ID [1],SkyReels-A2 [2], and MAGREF [3] (all based on Wan and open-source), indicating that the authors may not be very familiar with the field of identity-preserving video generation.

[1] Concat-ID: Towards Universal Identity-Preserving Video Synthesis

[2] SkyReels-A2: Compose Anything in Video Diffusion Transformers

[3] MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement

### Questions
My recommendation could improve if the authors provide convincing responses to the points raised.

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
This paper introduces FaithfulFaces, a pose-faithful facial identity preservation learning framework for the identity-preserving text-to-video (IPT2V) task. The method centers on a pose-shared identity aligner that refines and encodes global facial pose representations from a single reference image using a pose-shared dictionary and a pose variation-identity invariance constraint, including Euler angle embeddings as explicit pose cues. It further introduces a dataset pipeline for constructing high-quality video datasets with substantial facial pose variation. The framework demonstrates state-of-the-art performance on challenging IPT2V scenarios, particularly in situations where pose changes and occlusions occur, supported by quantitative benchmarks and qualitative visualizations.

### Strengths
1. The paper clearly articulates the challenge of pose and occlusion-induced identity distortion in IPT2V, identifying a key practical shortcoming in current models.

2. The pose-shared identity aligner leverages explicit pose information (via Euler angle embeddings and a learnable dictionary) to enhance identity preservation under varied and dynamic pose settings. The contrastive learning formulation is both intuitively and theoretically motivated (supported by an information-theoretic discussion using InfoNCE bounds).

3. The paper presents extensive experiments. Table 1 shows that FaithfulFaces outperforms both leading open-source and commercial baselines (such as ConsisID, VACE) consistently across multiple metrics, notably in FaceSim-Cur, FaceSim-Arc (identity preservation), and FID (visual fidelity). Qualitative results demonstrate superior consistency in identity and facial structure, particularly when faces undergo significant motion and occlusion.

### Weaknesses
1. While the info-theoretic perspective (lower-bounding mutual information in Remark 1) motivates the contrastive approach, the theoretical section is somewhat superficial beyond this. For instance, there’s no in-depth analysis of potential failure modes (such as pose-manifold collapse or overfitting to certain pose configurations) or a mathematical exploration of the limitations of MaxPool-based dictionary aggregation. A more rigorous exploration (even through negative/edge-case empirical examples) would be appreciated.

2. While ablations on dictionary element count are provided, there is limited discussion or justification of other hyperparameters (e.g., pooling operation type, temperature in contrastive loss, sequence length for embeddings). It’s unclear if alternative encoding or pooling mechanisms (e.g., soft attention vs. max pooling) could yield further gains or if they were considered.

3. In Section 3.3, the formulation of the loss function $\mathcal{L}_{\mathrm{PIA}}$ mixes definitions across identities and mini-batches but could be clearer about how negative pairs across batch boundaries are sampled, how class imbalance (from unbalanced poses) is avoided, and how the temperature parameter $\tau$ interacts with stability (especially as it’s described as learnable without strong evidence as to the optimization procedure or convergence).
The interplay between $\mathcal{L}{\mathrm{PIA}}$ and $\mathcal{L}{\mathrm{FM}}$ is not deeply analyzed; a more formal treatment of joint optimization and its practical impact on foundation model adaptation is missing.

### Questions
As shown in Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
