# AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Recent advances in latent diffusion models have achieved remarkable results in high-fidelity RGB image synthesis by leveraging pretrained VAEs to compress and reconstruct pixel data at low computational cost. However, the generation of transparent or layered content (RGBA images) remains largely unexplored, partly due to the absence of dedicated evaluation metrics. In this work, we introduce a new evaluation metric for RGBA images that adapts standard RGB measures to four-channel data via alpha blending over canonical backgrounds. We further propose AlphaVAE, a unified end-to-end RGBA VAE that extends a pretrained RGB VAE by incorporating a dedicated alpha channel. The model is trained with a composite objective that combines alpha-blended pixel reconstruction, patch-level fidelity, perceptual consistency, and dual KL divergence constraints to ensure latent fidelity across both RGB and alpha representations. Our RGBA VAE, trained on only 8K images in contrast to 1M used by prior methods, achieves a +4.9 dB improvement in PSNR and a +3.2\% increase in SSIM over LayerDiffuse in reconstruction. It also enables superior transparent image generation when fine-tuned within a latent diffusion framework. Our code, data, and models are released on https://anonymous.4open.science/r/AlphaVAE-0DB8
 for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Overall, this paper presents a meaningful and timely advancement in generative modeling by addressing transparency through an end-to-end RGBA VAE. The combination of a unified architecture, a principled evaluation metric, and strong experimental evidence makes this work a valuable contribution to both the academic and applied image generation communities.

That said, the training objective design appears somewhat over-engineered, involving multiple components whose necessity could be further justified. It would strengthen the paper if the authors could explore or discuss a more streamlined formulation that achieves comparable performance while maintaining conceptual elegance.

### Strengths
1. Clear Motivation: The paper identifies an underexplored but important gap in current generative modeling—handling transparency (alpha channel) in VAEs and diffusion pipelines. This is a concrete and practically relevant problem, especially for compositing, editing, and transparent-object generation tasks.

2. Novel Method Design: The proposed AlphaVAE integrates alpha-channel modeling into the standard RGB VAE pipeline in a simple yet principled way. It avoids complex architectural changes while achieving strong improvements, which makes it appealing for integration into existing diffusion systems.

3. Good Evaluation: The ALPHA benchmark (alpha-blending-based evaluation) provides a fair and interpretable protocol for RGBA image quality measurement. This is an elegant solution to the lack of standardized evaluation metrics for transparency reconstruction.

4. Well Organized and written: The paper is clearly structured, with detailed methodological explanations, visualizations, and ablations. It maintains excellent reproducibility and clarity in presentation.

### Weaknesses
I am not quite familiar with this area, but the training objectives are a little bit too much, with four different losses. I wonder if all of these losses are useful.

### Questions
Please refer to the weakness

### Soundness
3

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
3

### Summary
This paper introduces AlphaVAE, a unified framework for RGBA image reconstruction and generation with explicit alpha-aware representation learning.
It extends pretrained RGB-VAEs by incorporating dual-KL regularization and patch-level fidelity objectives to model both RGB and transparency channels.
The authors also release ALPHA, a new RGBA benchmark with adapted evaluation metrics.
Experiments show significant gains over prior methods (e.g., +4.9 dB PSNR, +3.2 % SSIM) and improved transparency-aware generation quality.

### Strengths
The paper defines an underexplored problem by introducing alpha-aware learning into generative modeling. Its methodologically sound design—combining dual-KL and patch-level fidelity—effectively bridges RGB and alpha representations. The work demonstrates clear empirical improvements on a new benchmark.

### Weaknesses
1. Dataset Size & Diversity: The ALPHA dataset (8K images) is relatively small, raising concerns about the model’s scalability and generalization to larger or more diverse real-world datasets. In addition, the limited dataset size may lead to potential overfitting, and it remains unclear whether the 8K samples provide sufficient diversity to support robust model training.
2. Generative Task Evaluation: Although the authors claim that the fine-tuned model can generate transparent images, the quality of transparency generation appears limited. Specifically, noticeable artifacts and abnormal edge transitions are observed around transparent regions, seemingly inherited from the original RGB images. A direct comparison with standard RGB-based generation methods—combined with a transparent object extraction mechanism—would provide a fairer and more informative evaluation.
3. Clarity of Technical Contribution: The paper’s core technical contribution is not clearly articulated. While it focuses on training a transparency-aware VAE, the motivation behind introducing such a model and the intuition for each proposed loss term are insufficiently discussed. As a result, the current presentation feels more like a technical report than a research paper and would benefit from deeper theoretical insight and clearer justification of design choices.

### Questions
1. Alpha-Blending Description: Equation (4) refers to Alpha-Blending, but this concept is not introduced or explained prior to its appearance. A clearer description or definition before its use would improve readability and understanding.
2. Importance of Transparency Reconstruction: The paper proposes a specific method for reconstructing transparent images and integrates it into diffusion models. However, it is not fully clear why transparency reconstruction is important and how it benefits existing editing or generative tasks. Including illustrative examples or case studies demonstrating the practical advantages of transparency-aware reconstruction would strengthen the motivation and impact of the work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new approach for evaluating RGBA images and proposes AlphaVAE, a novel VAE specifically designed for RGBA images. The method leverages a pretrained VAE architecture to effectively handle transparency information during generation and evaluation.

### Strengths
1. Writing: The paper is clearly written and easy to understand. The organization is logical, and the figures are well-designed and intuitive, effectively supporting the main arguments.

2. Ablation studies: The ablation studies are comprehensive and well executed. In particular, Table 3 provides detailed analyses across multiple objective functions, offering a thorough understanding of each component’s contribution.

### Weaknesses
1. Metrics: The proposed RGBA evaluation metric lacks originality. The method essentially applies conventional image quality metrics only to the non-transparent regions. While this may be acceptable for pixel-wise metrics such as PSNR, it is questionable for perceptual or structural metrics like SSIM and LPIPS, which rely on local context and structural consistency. 

2. Discriminator: The choice of a patch-based discriminator warrants further justification. Considering the role of transparency in RGBA images, an alpha-aware discriminator that incorporates alpha-channel information into its classification process might be a more suitable and principled design choice.

### Questions
1. The behavior of the models without certain loss functions (e.g., w/o Ref KL) differs notably between FLUX and SDXL. For instance, removing the reference KL term improves PSNR and SSIM for the FLUX model but not for SDXL. It would be helpful if the authors could provide an analysis or discussion explaining these discrepancies.

### Soundness
3

### Presentation
4

### Contribution
2
