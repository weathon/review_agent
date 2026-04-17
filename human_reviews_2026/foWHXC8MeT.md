# One-step Optimal Transport via Regularized Distribution Matching Distillation

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Unpaired domain translation remains a challenging task due to the need of finding a balance between faithfulness and realism. Diffusion-based methods for unpaired translation typically excel at realism, but require numerous inference steps and tend to offer suboptimal input-output alignment. Many of the optimal transport (OT) based methods, on the other hand, offer efficient few-step inference and reach superior input-output alignment, but heavily rely on adversarial training and inherit its shortcomings. In this paper, we propose a method called Regularized Distribution Matching Distillation (RDMD), which combines the best of both worlds. It replaces the adversarial training with diffusion-based distribution matching, addressing the typical shortcomings of OT methods and providing a strong initialization for the trained models. RDMD maintains the advantages of the OT methods by providing one-step inference and explicitly controlling the input-output faithfulness via regularization of the transport cost. We prove that in theory RDMD approximates the OT map and demonstrate its empirical performance on several tasks, including unpaired image-to-image translation in pixel and latent space and unpaired text detoxification. Empirical results show that RDMD achieves a comparable or better faithfulness-realism trade-off compared to the diffusion and OT-based baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Regularized Distribution Matching Distillation (RDMD) for unpaired image-to-image translation, adding a quadratic transport-cost regularizer to DMD to preserve input–output structural correspondence. RDMD enables one-step generation with a faithfulness–realism trade-off, showing competitive performance on multiple benchmarks.

### Strengths
-  The writing of the paper is clear and fluent.

-  Applying DMD to efficiently tackle unpaired I2I translation is an interesting approach.

### Weaknesses
- Limited resolution benchmarks. Experiments are restricted to low resolutions (64×64/128×128). It is unclear whether the approach scales to ≥ 256×256, where input–output alignment typically become more challenging.

- Overly simple regularization. The transport-cost regularizer is instantiated as plain L2, which is often insufficient to enforce semantic correspondence between input and output. This choice may limit the method’s expressiveness under complex cross-domain shifts.

- Have the authors considered using feature-level perceptual regularization[1] instead of pixel-space L2, for example by leveraging pretrained critics or reward models to better enforce semantic or structural details between the source and the one-step outputs?

[1] Li M, Yang T, Kuang H, et al. Controlnet++: Improving conditional controls with efficient consistency feedback

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript proposes a novel unpaired translation method called RDMD. The method aims to address the limitations of existing approaches: diffusion models, while offering strong realism, are slow and have low faithfulness; Optimal Transport (OT) methods, while fast and faithful, rely heavily on unstable adversarial training. The core innovation of RDMD is replacing the adversarial loss with a stable, DMD, while explicitly adding a transport cost as a regularization term to ensure faithfulness. This design allows RDMD to leverage the strong prior of pre-trained diffusion models, achieve efficient one-step inference, and strike a better balance between realism and faithfulness.

### Strengths
This manuscript proposes an unpaired translation method named RDMD, which is validated on both image and text tasks. The writing is relatively clear, the theoretical part is complete, and it provides some inspiration for the community in solving unpaired translation tasks.

### Weaknesses
1. Figure 1 is shown, but it is neither mentioned nor described in the manuscript.

2. The proposed method (Equation 9) introduces a "fake" diffusion model, $D_t^{\phi}$, which reframes the training as a coordinate descent process involving both the generator $G_{\theta}$ and this "fake" model $D_t^{\phi}$. My concern is that this approach seemingly exchanges one form of complexity (the instability of adversarial training) for another (the complexity of jointly training two networks).

3. There is a key contradiction between the paper's theoretical claims and its practical results. Theorem 3.1 states that the RDMD solution $G^{\lambda}$ converges to the true Optimal Transport (OT) map $G^*$ only as $\lambda \to 0$. However, the empirical results (e.g., Figure 6) clearly show that the model performs poorly at $\lambda = 0.0$ and achieves its best results at a non-zero $\lambda$ (e.g., 0.2). This implies that the empirically optimal model is *not* the true OT map the theory focuses on. Could the authors please address this gap and clarify whether the asymptotic convergence (Theorem 3.1) is the true justification, or if the method is better understood as an empirically-tuned "regularized DMD" where the transport cost is simply a helpful, non-asymptotic, constraint?

4. The authors selected the quadratic cost function $||x-y||^2$. They claim that in practice, any cost function of interest can be chosen. However, in the image translation experiments, only this single cost function was used. Are there any experiments with other cost functions? To my knowledge, pixel-level losses often lead to image blurriness. How would a perceptual cost (e.g., LPIPS) perform?

5. Given the clear trade-off between FID (realism) and LPIPS (faithfulness) in the experiments (where one improves, the other often degrades), it is difficult to assert which method is "better" based solely on these two automatic metrics. Have the authors considered conducting a human evaluation? For example, presenting human evaluators with paired results from RDMD and the strongest baseline (e.g., DDIB) and asking them to score the outputs along the two dimensions of "image realism" and "similarity to the original image."

### Questions
See details in the Weaknesses.

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
4

### Summary
This paper proposes Regularized Distribution Matching Distillation (RDMD) for one-step unpaired image-to-image translation. The method builds on Distribution Matching Distillation (DMD) by introducing an optimal transport (OT)-motivated regularization that enforces source faithfulness during translation. Specifically, the generator is trained so that its outputs match the target distribution via DMD, while adding a source–target transport cost regularizer (implemented as a simple pixel-level L2 loss). The paper further provides a theoretical argument showing that the trained generator approximates a Monge OT map. Extensive experiments across a wide range of domains—multiple datasets, resolutions, pixel vs latent spaces, text conditions, and synthetic toy setups—demonstrate strong one-step translation performance and consistent generalization.

### Strengths
1. The manuscript is well written, with clear explanations and a well-designed experiment section. The range of datasets and conditions evaluated is notably broad, and the ablations are thoughtful. The appendix contains detailed implementation information, including training setups and dataset details, which makes the work meaningfully reproducible.
2. Unpaired image translation is still a meaningful and relevant task, and the paper’s focus on improving one-step approaches within this setting is justified, and the comparisons against other one-step baselines are appropriate.
3. At first glance, the modification relative to DMD—changing the regularization term from teacher–student to source–generator—could appear incremental. However, the authors provide a clear OT-based justification and a formal argument that the model approximates a Monge OT map. This theoretical framing substantially elevates the contribution beyond a simple loss engineering tweak.
4. Achieving competitive translation quality in a single step, across multiple datasets and resolutions, and demonstrating Pareto-optimal trade-offs between fidelity and style, is compelling.

### Weaknesses
1. Even with the introduction of latent space, the experiments are limited to 256 resolution, despite 512 being a commonly expected baseline in modern latent diffusion pipelines. I would like clarification on whether the method was unable to scale to higher resolutions in practice, and whether pixel-space training at 256 resolution was feasible or intentionally not pursued.
2. Similar to DMD, three diffusion models (target, fake, and generator) must be loaded simultaneously during training, which imposes a heavy memory requirement.
3. Training time appears nontrivial. For example, on AFHQ-64 the generator training alone takes approximately three additional days, which suggests that although inference is one-step, the overall training burden remains high.

### Questions
1. Is the L2 loss applied in latent space (on the latent experiment)? If so, does this impact source fidelity due to mismatch between perceptual structure and latent geometry? Have you evaluated applying the regularization after decoding back into pixel space, and if so, did it improve or worsen perceptual source consistency?
2. In Table 3, were the diffusion models also trained only on the same limited datasets, or were any pretrained on larger datasets? Clarification is necessary for fair comparison.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Regularized Distribution matching distillation for one-step image translation.

### Strengths
It proposes simple and intuitive approach to include OT-based regularization into DMD in order to make one-step diffusion model for I2I.

### Weaknesses
1. Although the paper proposes that GAN-based models are mostly superior to EGSDE, I am a bit suspicious on these statements. There are so many GAN-based Image translation methods such as StarGAN, StarGANv2, CUT, CycleGAN, etc. These methods show great FID score when it comes to AFHQ and CelebA-HQ. To clearly show the advantage of proposed one-step model, please include the GAN-based methods. Also, please show comparison output between recent zero-shot editing based methods (prompt-to-prompt, Nano Banana.. etc) for thorough evaluation.  

2. Since the methods rely on DMD, the performance of I2I is limited on the teacher diffusion backbone. Also the DMD-based method requires additional network (fake teacher), the computation complexity and training time increases. Please show proper comaprison on this.

### Questions
No

### Soundness
3

### Presentation
2

### Contribution
2
