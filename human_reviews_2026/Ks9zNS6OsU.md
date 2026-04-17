# Reconstruct Anything Model a lightweight general model for computational imaging

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
Most existing learning-based methods for solving imaging inverse problems can be roughly divided into two classes: iterative algorithms, such as plug-and-play and diffusion methods leveraging pretrained denoisers, and unrolled architectures that are trained end-to-end for specific imaging problems. Iterative methods in the first class are computationally costly and often yield suboptimal reconstruction performance, whereas unrolled architectures are generally problem-specific and require expensive training. In this work, we propose a novel non-iterative, lightweight architecture that incorporates knowledge about the forward operator (acquisition physics and noise parameters) without relying on unrolling. Our model is trained to solve a wide range of inverse problems, such as deblurring, magnetic resonance imaging, computed tomography, inpainting, and super-resolution, and handles arbitrary image sizes and channels, such as grayscale, complex, and color data. The proposed model can be easily adapted to unseen inverse problems or datasets with a few fine-tuning steps (up to a few images) in a self-supervised way, without ground-truth references. Throughout a series of experiments, we demonstrate state-of-the-art performance from medical imaging to low-photon imaging and microscopy.
Our code is available at https://github.com/matthieutrs/ram.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Reconstruct Anything Model (RAM), a lightweight, non-iterative neural network designed as a foundation model for computational imaging. It directly integrates knowledge of the forward operator within a DRUNet-based architecture and is jointly trained on diverse imaging inverse problems. It is able to generalize to unseen tasks and can be finetuned self-supervisedly on new measurement data. Experimental results show strong zero-shot and fine-tuned performance across a variety of modalities, often matching or surpassing heavier iterative or unrolled baselines while being computationally efficient.

### Strengths
- RAM demonstrates generalization from a single training across multiple imaging modalities and degradation types.
- The integration of the Krylov Subspace Module provides a new design to embed forward-operator knowledge without unrolling iterations.

### Weaknesses
- Although the paper is titled “Reconstruct Anything Model,” this claim feels overstated. The experiments are limited to a small set of linear inverse problems, while many common and practically important tasks—such as super-resolution, JPEG artifact removal, and phase retrieval—are not considered. The model also heavily relies on accurately known forward operators, which are often unknown or imperfect in real scenarios. Moreover, all evaluations are based on simulated data, so the applicability to real-world measurements remains uncertain. While I appreciate the ambition of this work, calling it “Reconstruct Anything” may mislead readers, especially for those who are new to or unfamiliar with computational imaging.

- Although the architecture is newly designed, the paper lacks a clear justification for why this particular design should serve as a foundation model. The conceptual motivation and theoretical support for why the Krylov-based conditioning leads to generalization across modalities are not well explained.

- It is still unclear what the advantages over deep unrolling frameworks truly are. The performance is very close to DPIR/t, which already has optimization-based interpretability and convergence guarantees. In contrast, the proposed model provides limited explanation for its empirical gains or stability properties.

- The proposed multiscale operator conditioning is not clearly presented in the framework, and it seems to require explicit operations on the forward operator. However, not all forward models allow such manipulation or downscaling, which may limit the general applicability of this approach.

- While the model integrates known concepts effectively, the novelty is incremental rather than fundamental. It mainly combines existing DRUNet backbones with a physics-inspired conditioning mechanism, making the contribution more of an architectural refinement than a new paradigm.

### Questions
Please see the weakness section.

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
2

### Summary
This paper proposed reconstruction model which is universal to linear inverse problems. The authors proposed to use measurement conditioning block, conditional residual block, and Krylov subspace imposing block to achieve best performance. By utilizing mentioned components, the trained model works for various imaging tasks with one model. In addition, the model was enhanced with self-supervised finetuning exploiting equivariance to a group of transformations. The paper showed promising results.

### Strengths
- author contributes to use Krylov subspace to enhance performances. This is well modeled in the signal processing perspective
- clarity of the contents is well defined

### Weaknesses
- Worse results compared with iterative diffusion models. Not consistently winning in various settings.

- Lacks of recent various algorithm including neural operator, INR, transformer etc. In particular, there are algorithms which also exploit uni-model to reconstruct several tasks such as Pre-Trained Image Processing Transformer. The authors need to include diverse benchmark algorithms. score-based approach* is more appropriate to be compared with the proposed method.
*SOLVING INVERSE PROBLEMS IN MEDICAL IMAGING WITH SCORE-BASED GENERATIVE MODELS, ICLR22

- In order to achieve Krylov space manner, this algorithm needs additional recurrent processing on forward model. This cause computational overloads. Thus, computational complexity needs to be analyzed. (i.e. computational time according to the various size of A model)

### Questions
- The similar algorithms to solve linear inverse problems for the higher PSNRs are not easy to reconstruct severe degradation circumstances such as Figure 11 (x4). On the other hand, recent  neural operation with diffusion forward model (SRNO & diffFNO) is very strong to reconstruct details of contents with such severe conditions. What is the main advantage of using RAM instead of DiffFNO?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Reconstruct Anything Model (RAM), a non-iterative lightweight foundation model for computational imaging tasks. The model extends the DRUNet backbone with physics-aware conditioning via a Krylov Subspace Module (KSM) and a proximal initialization step. RAM aims to unify diverse inverse problems (deblurring, MRI, CT, denoising, inpainting, super-resolution) into one architecture. It further supports self-supervised finetuning without ground truth data, enabling adaptation to unseen imaging modalities. Experiments compare RAM with Plug-and-Play (DPIR), unrolled (PDNet, uDPIR), and diffusion (DDRM) methods, showing comparable or better PSNR/SSIM at 8x lower compute and 10x fewer parameters.

### Strengths
1. The paper evaluates across diverse inverse problems (deblurring, MRI, CT, denoising, inpainting, super-resolution) and multiple domains (medical, microscopy, low-photon). This comprehensive experimentation is valuable for benchmarking unified restoration models.

2. The proposed RAM model achieves performance comparable to unrolled and diffusion-based models while being roughly 8x faster and 10x smaller (36M parameters vs. >200M). This makes it appealing for real-time or embedded applications.

3. Practical self-supervised finetuning:
Demonstrating zero-shot and few-shot adaptation without ground truth (via SURE/UNSURE and EI/MOI losses) is practical and shows that pretrained inverse models can generalize with minimal data.

4. The model is implemented cleanly with shared weights across modalities and channel configurations (color, grayscale, complex), showing solid software engineering and reproducibility.

5. Despite the dense material, the manuscript maintains good structure and uses comprehensive figures (e.g., KSM illustration, self-supervised finetuning results).

### Weaknesses
1. **Limited theoretical justification:** 
KSM and proximal initialization are described intuitively, but no formal analysis of convergence, stability, or conditioning advantage is provided. This weakens the "physics-informed" claim.

2. **Incremental contribution:**
The model essentially extends DRUNet with physics-based conditioning. Similar hybrid unrolled networks and foundation-style image restoration models (PromptIR, AdaIR, and MambaIR) already exist with stronger novelty or scale.

3. **Overstated “foundation” claim:**
The term "foundation model" implies large-scale generalization, but training is limited to standard datasets (LSDIR, fastMRI, LIDC-IDRI)---far from the multimodal pretraining scale of foundation models in computational vision or imaging.

4. Unclear fairness in comparison:
Some baselines (e.g., DDRM) were reimplemented with reduced input resolution or adapted operators, making quantitative comparisons potentially inconsistent. No statistical significance tests are provided for small PSNR differences (~0.2--0.4 dB).

5. Limited impact and novelty for ICLR:
The work focuses on system-level integration rather than introducing a new learning principle, theory, or large-scale paradigm. 
Its main contribution lies in engineering efficiency, that may fit better as a journal or CVPR/ICIP paper than ICLR.

### Questions
1. Krylov conditioning sensitivity:
How stable are the learned Krylov coefficients when the forward operator $\mathbf{A}$ is rescaled, discretized differently, or ill-conditioned? Does the model require operator-specific normalization to generalize across domains?

2. Applicability beyond linear operators:
Can RAM be extended to nonlinear or blind inverse problems (e.g., unknown blur kernels, phase retrieval), or is it fundamentally restricted to linear, known $\mathbf{A}$?

3. Effectiveness of proximal initialization:
What is the concrete benefit of the proximal estimation module over simply using $\mathbf{A}^\top \mathbf{y}$ or $\mathbf{A}^+ \mathbf{y}$? 
Have you compared their convergence behavior or training stability quantitatively?

4. Self-supervised finetuning stability:
In cases where $\mathbf{A}$ has a large nullspace (e.g., highly undersampled CT or MRI), how do you prevent degenerate reconstructions during self-supervised finetuning? Does the model require any regularization or early stopping?

5. Runtime and efficiency evaluation:
The paper reports FLOPs and parameter counts, but not real-world latency. What are the actual wall-clock inference times and memory usage compared to unrolled (uDPIR) or diffusion (DDRM) baselines on 256×256 and 512×512 inputs?

6. Generalization to unseen noise regimes:
The model claims noise-level equivariance via scale-free bias removal. Could you provide quantitative evidence that RAM maintains stable performance across unseen noise parameters $(\sigma, \gamma)$ beyond the training range?

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
5

### Summary
This paper proposes the Reconstruct Anything Model (RAM), a lightweight, non-iterative foundation model for computational imaging. The architecture builds on a DRUNet backbone but introduces novel components to incorporate the acquisition physics, namely a proximal estimation module for initialization and a Krylov Subspace Module (KSM) that conditions the network on iterations of the forward operator. The model is supervised trained on a wide diversity of tasks (e.g., deblurring, MRI, CT), data types (grayscale, color, complex), and noise models (Gaussian, Poisson) simultaneously.

### Strengths
1. The proposed KSM is a clever and effective method for integrating the forward operator $A$ into a U-Net backbone without resorting to full algorithm unrolling. This, combined with the multiscale conditioning and proximal initialization, creates a powerful architecture that is well-grounded in optimization principles while remaining a fast, non-iterative network.

2. The paper is exceptionally thorough. The authors provide extensive experiments covering in-distribution performance, zero-shot generalization to OOD tasks, and self-supervised finetuning.

3. The results are state-of-the-art. RAM consistently outperforms strong baselines like DPIR and Restormer and performs comparably to the much larger uDPIR models, demonstrating high efficiency.

### Weaknesses
1. The related work section frames the landscape as a choice between PnP/diffusion (using denoising priors) and end-to-end/unrolled models. However, it overlooks a highly relevant, recent line of research [1-3] that proposes using restoration networks themselves as priors within iterative schemes (e.g., as fixed-point operators). These "restoration priors" are conceptually similar to RAM's goal (a general-purpose restoration model) but do not need specifically trained. A discussion of how RAM's end-to-end, multi-task training approach compares to these iterative restoration prior methods is necessary. 

2. The paper introduces multiscale operator conditioning as a key component. This is a valuable idea, but the concept has been explored before. For instance, [4] used a multiscale structure-guided approach for image deblurring. The paper would be strengthened by acknowledging this prior work and discussing the similarities or differences in its multiscale implementation.

3. A core premise is that a single model can learn a universal prior for disparate imaging inverse problems. This "one-prior-fits-all" approach learns the best implicit prior. Does co-training on SR or deblurring tasks help with image inpainting, or does it act as a conflicting signal? The paper's own ablation in Table 4 (Right) suggests that the model trained on "All three tasks" (Inpainting, Deblurring, SR) performs slightly worse on inpainting and deblurring than specialist models. It t still raises the question of whether similar task-specific foundation models (e.g., "RAM-deblur&SR" and "RAM-Inpainting") might be more effective. A more detailed study of this trade-off would be valuable.


[1] Terris, M., Kamilov, U.S. and Moreau, T., 2025. FiRe: Fixed-points of restoration priors for solving inverse problems. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 23185-23194).
[2] Hu, Y., Peng, A., Gan, W., Milanfar, P., Delbracio, M. and Kamilov, U.S., Stochastic Deep Restoration Priors for Imaging Inverse Problems. In Forty-second International Conference on Machine Learning.
[3] Hu, Y., Delbracio, M., Milanfar, P. and Kamilov, U.S., 2024. A RESTORATION NETWORK AS AN IMPLICIT PRIOR. In 12th International Conference on Learning Representations, ICLR 2024.
[4] Ren, M., Delbracio, M., Talebi, H., Gerig, G. and Milanfar, P., 2023. Multiscale structure guided diffusion for image deblurring. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10721-10733).

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
