# IDFSR: Personalized Face Super-Resolution with Identity Decoupling and Fitting

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
In recent years, face super-resolution (FSR) methods have achieved remarkable progress, generally maintaining high image fidelity and identity (ID) consistency under standard settings. However, in extreme degradation scenarios (e.g., scale 8x or 16x
), critical attributes and ID information are often severely lost in the input image, making it difficult for conventional models to reconstruct realistic and ID-consistent faces. Existing methods tend to generate hallucinated faces under such conditions, producing restored images lacking authentic ID constraints. To address this challenge, we propose a novel FSR method with Identity Decoupling and Fitting (IDFSR), designed to enhance ID restoration under large scaling factors while mitigating hallucination effects. Our approach involves three key designs: 1) \textbf{Masking} the facial region in the low-resolution (LR) image to eliminate unreliable ID cues; 2) \textbf{Warping} a reference image to align with the LR input, providing style guidance; 3) Leveraging \textbf{ID embeddings} extracted from ground truth (GT) images for fine-grained ID modeling and personalized adaptation. We first pretrain a diffusion-based model to explicitly decouple style and ID by forcing it to reconstruct masked LR face regions using both style and identity embeddings. Subsequently, we freeze most network parameters and perform lightweight fine-tuning of the ID embedding using a small set of target ID images. This embedding encodes fine-grained facial attributes and precise ID information, significantly improving both ID consistency and perceptual quality. Extensive quantitative evaluations and visual comparisons demonstrate that the proposed IDFSR substantially outperforms existing approaches under extreme degradation, particularly achieving superior performance on ID consistency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes IDFSR, a diffusion-based two-stage framework that aims to improve identity consistency under extreme degradation scenarios. In the first stage, the authors pretrain a diffusion model that reconstructs masked low-resolution faces conditioned on a warped reference image and ground-truth ID embedding. In the second stage, the model performs lightweight fine-tuning to personalize ID embeddings using a few samples from the same identity. The approach claims to decouple style and ID representations, improving robustness and reducing hallucinations in high-scale super-resolution tasks. Extensive experiments on several datasets show improved ID consistency and visual quality over prior works.

### Strengths
- The paper identifies a realistic problem of maintaining ID consistency under extreme degradation, and provides a well-structured solution using diffusion-based modeling and personalized fine-tuning.

- Quantitative and qualitative comparisons are provided across multiple datasets and metrics (FID, LPIPS, MUSIQ, ID similarity), with reasonable gains reported.

- The authors include ablation studies and cross-ID tests to support the disentanglement claim, as well as discussions on robustness and limitations (e.g., when references are unavailable).

- The method is well-documented, including architectural and training details, making it easier to reproduce.

### Weaknesses
- The framework mainly integrates existing concepts—diffusion-based SR, reference-based alignment, and ID embeddings—without introducing substantial algorithmic innovation. The core ideas of warping references, masking faces, and fine-tuning ID embeddings are largely extensions of prior reference-based methods (e.g., ASFFNet, DMDNet, MyStyle), now placed within a diffusion context.

- The warping step depends on landmark detection and alignment, which is brittle in real-world low-quality conditions. Although the authors claim robustness to misalignment, the underlying design still shares the same limitations as conventional keypoint-based reference SR methods, when degradation is severe or faces are non-frontal, the warped reference can introduce artifacts or fail altogether. Maybe this is also why these landmark based methods like ASFFNet and DMDNet fail to handle images in Figure 4.

- Since fine-tuning is performed per identity on small image sets, the model can overfit to dominant pose or frontal expressions, reducing generalization to unseen viewpoints or lighting variations. The paper lacks a systematic evaluation of pose diversity or real-world unconstrained scenarios.

- The low-resolution image is masked to remove the facial region, which removes all pixel-level cues of identity. In such a case, the model relies entirely on warped reference and learned ID embeddings to reconstruct the face. However, this design raises questions:

1) When degradation is mild, the mask discards valid facial details that could aid reconstruction.

2) When degradation is severe, the warped reference (IW) may not align in gaze, mouth shape, or facial expression with the LR image. This can alter the original structure or expression of IL.
The paper does not clearly explain how such conflicts are resolved or constrained during training or inference.

- The fine-tuning process essentially adjusts a per-identity embedding vector while freezing the backbone. While effective empirically, this procedure resembles per-user adaptation or style fitting rather than a general SR improvement, and its novelty relative to methods like MyStyle is modest.

### Questions
- When the LR image is masked (facial region removed), how does the model ensure that the reconstructed structure (e.g., gaze direction, mouth shape) remains faithful to the original LR rather than to the reference or learned ID embedding?

- Have the authors evaluated how the model performs on real-world LR faces (e.g., surveillance or wild datasets) rather than synthetic degradations? The current experiments seem dominated by controlled synthetic settings.

- How sensitive is the method to landmark detection or warping errors under strong degradations or occlusions? Would a landmark-free alignment approach be feasible?

- How does the method behave when fine-tuning data contains diverse poses? Does the embedding average over them or bias toward certain views?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work targets face super-resolution (FSR) under severely degraded conditions. The paper constructs a pretraining/fine-tuning identity-decoupling/fitting framework, IDFSR, through three innovative key designs. It demonstrates compelling visual results and achieves state-of-the-art performance across numerous metrics, highlighting the superiority of IDFSR for customized FSR. Ablation studies provide strong evidence for the necessity of decoupling and fitting. Finally, it is interesting that IDFSR exhibits a certain degree of robustness even without references.

### Strengths
1) The novelty and motivation are strong. Traditional prior-based methods are unreliable under severe degradation, especially when high ID consistency is required. The motivation for introdducing a reference image in this work is well-founded. IDFSR innovatively designs a decoupled pretraining and customized fitting approach, demonstrating unprecedented performance in maintaining ID consistency.

2) The writing and experiments are solid and well-executed. This work includes extensive experiments, covering different scales, real-world scenarios, and video applications. The carefully designed ablation studies are convincing. Although the performance at 4× scale is suboptimal, the method shows superior performance at other scales, consistent with its motivation. In the absence of a reference image, it can be regarded as degenerating to a prior-based method, which is acceptable.

### Weaknesses
1. The masking strategy is similar to an inpainting task, but the differences need to be clearly specified.
2. Figure 3 analyzes IDFSR’s robustness to erroneous landmarks, but lacks concrete performance quantification. For example, the impact of landmark detection on the final performance and analysis of using it as data augmentation.
3. The selection of reference images needs to be clarified. IDFSR has some editing capability, yet the experiments only mention random selection of reference images. Are there better selection strategies?
4. Application issues: Figure 12 and Table 2 indicate low expression matching, which affects identity consistency. How can expression issues be addressed in practical applications?

### Questions
Please See the weaknesses.

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
This paper introduces IDFSR, a personalized face super-resolution framework that aims to preserve identity (ID) consistency and high fidelity under extreme degradation conditions (e.g., scaling factors >8x). The approach involves three key innovations: applying a facial mask to the low-resolution (LR) image to suppress unreliable ID cues, warping a reference image for style guidance, and using a learnable ID embedding extracted from ground truth (GT) images for precise ID modeling and adaptation through lightweight finetuning. The method leverages diffusion models to explicitly decouple style and ID, followed by personalized embedding tuning per identity. Extensive experiments, quantitative results, and visualizations demonstrate improved ID consistency and visual fidelity over a strong set of state-of-the-art baselines, especially under challenging degradation.

### Strengths
- Robust ID Consistency in Extreme Degradation: The proposed method addresses a real limitation where most face SR methods struggle—high upscaling factors with minimal ID information left in the input. By decoupling style and ID and then fitting the ID embedding, the approach significantly improves performance as shown in Table 1 and Table 2.
- Comprehensive Ablation and Analysis: The paper provides deep, systematic ablation studies (see Figure 8 and Figure 7) that dissect the contributions of masking, ID embeddings, and style conditioning. The impact of reference image quantity and the complementary nature of components are scientifically explored and well-presented.
- Qualitative and Quantitative Superiority: Visualizations in Figure 4, Table 1, and Table 2 show clear improvement in ID preservation and details, with less hallucination and more realistic reconstructions compared to recent SOTA models, including strong diffusion and reference-based approaches.

### Weaknesses
- Limited Direct Theoretical Insight in ID Disentanglement: While the paper presents promising results in ID/style decoupling, the discussion of disentanglement primarily focuses on empirical evidence and architectural design. A more formal analysis—such as quantifying disentanglement using mutual information or correlation-based metrics—could strengthen the theoretical foundation of the work. Additionally, further exploration of the conditions under which disentanglement may succeed or fail would provide valuable insight. For instance, while the cross-ID experiments (Section 5.1, Figure 5) are persuasive, a clearer articulation of the underlying representation properties (e.g., the degree of independence between ID and style) would enhance the overall rigor of the analysis.
- The comparison methods are somewhat outdated; appropriately adding some work from 2025 for comparison could make the experimental results more convincing.
- The complexity and computational cost have not been tested: the work discusses some details regarding memory and time costs, but it does not comprehensively compare aspects such as the actual inference time and model size with other studies. Considering the resource demands of diffusion models, as well as the additional time required for per-ID fine-tuning and its variation across different datasets, this aspect is crucial for practical adoption.
- Experimental reproducibility: The appendix includes many details that can help other researchers quickly understand the experimental setup, but some implementation aspects of the fine-tuning process, training set division, and reference image selection still need to be explained more clearly.
- Real-world noise scenario validation: The dataset used lacks realistic factors such as noise, compression, occlusion, and illumination imbalance. If a dataset containing real-world noisy scenarios could be used for evaluation, it would allow for a more comprehensive verification of the model’s robustness and generalization ability in real environments.

### Questions
- Open source: Will the authors release the model and code in the future?
- Computational resource usage: Could the authors provide specific data such as fine-tuning and inference time per image, as well as parameter count comparisons? This would help clarify the practical implications of diffusion- and embedding-based methods.
- On disentanglement quantification: Could the authors provide any quantitative evidence (e.g., mutual information, independence scores) to substantiate—or at least empirically support—the claim that “identity and style embeddings are indeed disentangled in the learned representations”? In addition, are there specific cases or failure scenarios (e.g., under certain learning rates or reference/distortion conditions) where entanglement re-emerges?

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
3

### Summary
The authors propose a novel face super-resolution method named IDFSR (Identity Decoupling and Fitting for Face Super-Resolution), designed to address identity distortion and hallucination generation under extreme degradation scenarios, such as upsampling scales exceeding $8\times$ and even reaching $32\times$. They devise a diffusion-based two-stage framework: in the pretraining stage, unreliable facial regions in the low-resolution (LR) image are masked, a style embedding is extracted from a landmark-aligned reference image to provide coarse appearance guidance, and an identity embedding is derived from the ground-truth high-resolution image to enable explicit disentanglement of identity and style; in the fine-tuning stage, only a learnable identity embedding vector is optimized, enabling personalized adaptation with just a few target-ID samples. Experiments demonstrate that IDFSR substantially outperforms existing approaches across multiple benchmarks, including CelebRef-HQ, CASIA-WebFace, and CelebV-Text, with particularly notable improvements in identity consistency (measured by IDS). Through carefully designed conditional modeling and a lightweight fine-tuning strategy, this work achieves high-quality, high-fidelity personalized face reconstruction under extreme super-resolution settings.

### Strengths
The paper proposes IDFSR, a face super-resolution method for extreme degradation scenarios (e.g., $16\times$ - $32\times$), centered on identity decoupling and personalized fitting. Its novelty lies in three key designs: (1) Masking unreliable facial regions in the low-resolution (LR) input to force the model to rely on external priors; (2) Extracting a style embedding from a landmark-aligned reference image to guide appearance reconstruction; (3) Using ground-truth identity embeddings during pretraining and fine-tuning only this embedding with a few target-ID samples for lightweight personalization. Technically, the work is solid: experiments are thorough, the diffusion-based framework is well-motivated, and the conditional injection mechanism (AdaGN + cross-attention) is effective. On benchmarks like CelebRef-HQ, CASIA-WebFace, and CelebV-Text, IDFSR significantly outperforms state-of-the-art methods in identity consistency (IDS), perceptual quality, and pixel fidelity, especially at high upscaling factors. Ablation studies confirm the necessity of each component. By explicitly separating identity from style and using minimal fine-tuning, IDFSR effectively tackles the “identity hallucination” problem in extreme face super-resolution, offering a practical solution for applications like surveillance and digital identity. While it requires a few same-ID images for fine-tuning, limiting its use in fully generic settings, it excels in personalized reconstruction tasks.

### Weaknesses
- The paper employs ArcFace as the identity (ID) encoder but does not clarify whether users are allowed to substitute it with other ID models (e.g., FaceNet or MagFace). This omission raises questions about the method’s modularity and compatibility. The authors are encouraged to include experiments evaluating alternative ID encoders.
- The paper does not report model size, FLOPs, or inference latency, nor does it compare these metrics against other methods. Consequently, it is difficult to assess the feasibility of deploying the model on edge devices. The authors are advised to provide such efficiency-related experiments.
- During fine-tuning, the algorithm optimizes only a single learnable ID embedding vector. However, the paper does not discuss how the initialization strategy, e.g., whether the vector is extracted from ArcFace, randomly initialized, or set to an average embedding, affects convergence speed and final performance.
- Reference images may contain occlusions, poor lighting, or extreme pose variations, yet the paper lacks a systematic evaluation of robustness under such “non-ideal reference” conditions. The authors are encouraged to supplement ablation studies where reference image quality is progressively degraded (e.g., via added noise, occlusion, or blur) to more comprehensively delineate the method’s operational boundaries.

### Questions
- During the fine-tuning stage, the authors state that the ArcFace identity encoder is no longer required. How is the learnable ID embedding initialized? Is it a randomly initialized tensor, or is it initialized based on some prior?
- What is the purpose of applying a mask to the low-resolution (LR) input image? What benefits does this masking strategy provide? Does it negatively impact pixel-level reconstruction quality or the preservation of fine-grained attributes such as gaze direction or makeup?
- The authors mention fine-tuning RetinaFace to better adapt it to low-resolution images. What is the specific fine-tuning strategy? For instance, was RetinaFace retrained on low-resolution face data, and if so, what dataset and loss functions were used?
- The authors claim that imperfect image warping does not hinder model performance and may even serve as a form of data augmentation. Have they attempted to skip the warping step entirely, i.e., directly feeding the original reference image into the style encoder, and conducted an ablation study comparing this variant against the current approach?

### Soundness
3

### Presentation
3

### Contribution
3
