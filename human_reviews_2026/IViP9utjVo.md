# PROOF: Perturbation-Robust Noise Finetune via Optimal Transport Information Bottleneck for Highly-Correlated Asset Generation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
The diffusion model has provided a strong tool for implementing text-to-image (T2I) and image-to-image (I2I) generation. Recently, topology and texture control have been popular explorations. Explicit methods consider high-fidelity controllable editing based on external signals or diffusion feature manipulations. The implicit method naively conducts noise interpolation in manifold space. However, they suffer from low robustness of topology and texture under noise perturbations. In this paper, we first propose a plug-and-play perturbation-robust noise finetune (PROOF) module employed by Stable Diffusion to realize a trade-off between content preservation and controllable diversity for highly correlated asset generation. Information bottleneck (IB) and optimal transport (OT) are capable of producing high-fidelity image variations considering topology and texture alignments, respectively. We derive the closed-form solution of the optimal interpolation weight based on optimal-transported information bottleneck (OTIB), and design the corresponding architecture to fine-tune seed noise or inverse noise with around only 14K trainable parameters and 10 minutes of training.  Comprehensive experiments and ablation studies demonstrate that PROOF provides a powerful unified latent manipulation module to efficiently fine-tune the 2D/3D assets with text or image guidance, based on multiple base model architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PROOF, a lightweight plug-and-play module that improves the balance between fidelity and diversity in diffusion-based generation by directly fine-tuning latent noise. The method uses a new Optimal-Transported Information Bottleneck (OTIB) to adaptively blend original and perturbed noise, preserving topology and texture while enabling controllable variation. PROOF requires minimal training and works across 2D/3D diffusion models. Experiments show better content consistency and diversity compared to existing controllable generation approaches.

### Strengths
This paper introduces a novel and practical perspective by leveraging noise itself as a controllable representation for diffusion models, rather than relying on explicit structural or textural signals. The method is theoretically supported through a closed-form Sinkhorn-IB solution and is lightweight and model-agnostic, enabling efficient plug-and-play usage across 2D/3D generators.

### Weaknesses
1. [1] demonstrates not only image variation but also image editing without introducing any additional modules or training, whereas PROOF still requires lightweight training and does not explicitly support localized editing with semantic prompts. This raises the question of whether the proposed method offers enough practical advantage over fully training-free approaches.

2. The comparisons are mainly focused on diffusion–control based methods, and there is still room for stronger empirical validation through comparison with image variation works such as [1] and [2], which also aim at structure- and content-preserving diversity.

3. In the main text, the explanation of the Experimental Results appears only on page 9. Apart from the tables and figures, there is insufficient discussion of the experimental findings, which made it difficult to fully understand the paper. I believe the paper should provide a better presentation. For example, in the Section "5 Experiments", the descriptions of the Training Protocol and Baselines could, in my view, be largely moved to the Appendix.

[1] Real-World Image Variation by Aligning Diffusion Inversion Chain
[2] Prompt-Free Diffusion: Taking “Text” out of Text-to-Image Diffusion Models

### Questions
1. Can the proposed method perform image editing by conditioning on a different prompt, or is the diversity strictly limited to preserving the original semantic content?

2. Does the method works on latest flow models like SD3.5/Flux?

3. When combining PROOF with a structure-guided controller such as ControlNet, is it possible to maintain texture fidelity while still generating diverse variations?

4. Why is the inference time in Table 1 presented in two different types only for “Ours”? I could not find a clear explanation for this in either the main text or the caption.

### Soundness
3

### Presentation
2

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
The paper introduces PROOF, a plug-and-play module for diffusion models based on an Optimal Transport Information Bottleneck formulation. The method fine-tunes latent noise through an optimal transport constraint to balance content fidelity and diversity, with claimed benefits in robustness and controllability.

While mathematically sound, the method’s practical contribution is limited. The approach introduces additional computational complexity without offering clear performance advantages, which undermines its claimed efficiency. It remains unclear why practitioners should adopt this method.

### Strengths
* The integration of optimal transport and information bottleneck concepts is theoretically consistent.
* The framework is modular and, in principle, applicable to different diffusion architectures.

### Weaknesses
* **Inferior inference efficiency.** The paper claims PROOF is a lightweight, plug-and-play controller. However, Table 1 shows that its inference time is longer than all compared baselines. This demonstrates that the method adds computational overhead rather than reducing it, which makes it impractical for real-time or large-scale applications.

* **Modest empirical improvements.** The reported results show only minor gains over existing baselines, which do not justify the added architectural complexity or computational cost.

* **Sensitivity concerns.** The method relies on several manually tuned hyperparameters, raising questions about its robustness, reproducibility, and generalization across different datasets and architectures.

### Questions
* Why should practitioners adopt PROOF over simpler, faster alternatives that yield similar results?
* Can the method be adapted to few-step models (e.g., FLUX-Schnell) to reduce latency?
* Can the method be evaluated on modern diffusion models (e.g., SD3.5, FLUX) to ensure its relevance?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes PROOF, a controllable 2D/3D generation method based on perturbation-robust noise finetuning. By integrating information bottleneck with optimal transport theory, the paper derive a closed-form solution for Sinkhorn-regularized interpolation weights. It achieves high-quality topology and texture-aligned generation across multiple base models without requiring external control signals, while outperforming existing methods on both fidelity and diversity metrics.

### Strengths
1. The paper's significant strenghts  is its theoretical grounding, as the closed-form solution derived via Optimal-Transport Information Bottleneck (OTIB) provides clear guidance for subsequent optimization.

2.  In terms of general applicability, the method is compatible with both 2D and 3D generation tasks, supports text and image guidance modalities, and can be integrated with multiple base models (including Stable Diffusion and TRELLIS).

3. Compared to existing control methods, PROOF operates without relying on any external structural control signals and achieves superior appearance alignment performance without requiring personalized concept data or model fine-tuning.

### Weaknesses
1. Although diversity is controlled through the β parameter, its variation range remains constrained by the structure of the noise space. Could we further explore the implications of β parameters for generation diversity?

2. I'm concerned about relying on intrinsic interpolation to manipulate noise, as it fails to capture complex nonlinear content transformations—such as object deformation or perspective changes. This interpolation paradigm constitutes a fundamental limitation in scenarios demanding highly creative generation.

### Questions
1. Its potential applicability to more advanced diffusion models with distinct architectures—such as Flux or SD3.5—warrants further investigation. Conducting experiments on such next-generation models would significantly strengthen PROOF's generalizability and broader relevance.

2. Does PROOF maintain robust performance when there is a resolution discrepancy between the fine-tuning and inference phases? For instance, when fine-tuning employs low-resolution data while inference utilizes high-resolution data.

3. Does Sinkhorn suffer from over-smoothing phenomena? For instance, if I intend to perform texture stylization only on specific local regions rather than globally – in such tasks, critical information may be confined to local areas. When processing a human face, the eye regions should inherently receive greater attention, yet Sinkhorn enforces uniform distribution. Could this potentially weaken the strong correlations between adjacent pixels in texture synthesis?

### Soundness
3

### Presentation
3

### Contribution
3
