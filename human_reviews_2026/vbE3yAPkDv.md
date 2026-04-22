# Single-Step Bidirectional Unpaired Image Translation Using Implicit Bridge Consistency Distillation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Unpaired image-to-image translation has seen significant progress since the introduction of CycleGAN. However, methods based on diffusion models or Schrödinger bridges have yet to be widely adopted in real-world applications due to their iterative sampling nature. To address this challenge, we propose a novel framework, Implicit Bridge Consistency Distillation (IBCD), which enables single-step bidirectional unpaired translation without using adversarial loss. IBCD reformulates consistency distillation by using a diffusion implicit bridge model that connects PF-ODE trajectories between distributions, with a novel design parametrization to enable effective translation in a single step. Additionally, we introduce two key improvements: 1) distribution matching for consistency distillation and 2) adaptive weighting method based on distillation difficulty. Experimental results demonstrate that IBCD achieves state-of-the-art performance on benchmark datasets in a single generation step.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposed a single-step bidirectional unpaired image-to-image translation model using the combination of consistency distillation and distribution matching distillation for DDIB teacher model. Key contributions of the work include the formulation of consistency distillation for PF-ODE trajectories obtained with DDIB model and the extension of DMD framework for the proposed consistency distillation. The evaluation of the proposed IBCD model was done using cat-to-dog, wild-to-dog, and male-to-female unpaired image-to-image translation problems. IBCD is compared with adversarial methods, including CUT and UNSB, and diffusion methods, including EGSDE, CycleDiffusion, SDDM, SDEdit, and the teacher model DDIB. These image-to-image translation models are evaluated using FID, density and coverage for image realism, inference time and NFE for inference efficiency, and PSNR and SSIM for input-output similarity. The results of IBCD show its advantages against the teacher model DDIB in terms of the inference efficiency and input-output similarity, and other adversarial and diffusion models in terms of image realism.

### Strengths
1) Surprisingly, according to Table 2 and Table 7, the proposed method has the inference time, which is even less than for GANs, while having better realism and input-output similarity metrics. It makes diffusion models much closer for applications in unpaired image-to-image translation problems.
2) The novel adaptive DMCD loss greatly improves the input-output similarity, as demonstrated by Table 3 and Figure 7.
3) The extension evaluation of IBCD with different image-to-image translation metrics in Table 2 is also supported by user study and perceptual measures in Table 6. The method is also compared with LLM-based GPT-Image-1 model, which shows existing limitations in domain specific problems of foundational large models in zero-shot editing. 
4) Figure 8 studies the effect of realism-similarity trade-off for the proposed model by balancing between DMCD and cycle-consistency losses

### Weaknesses
1) According to Table 4, the distillation of DDIB model requires more than 200k steps for the training IBCD. It remains unclear how fast and stable the distillation with the IBCD is performed compared to the training of other unpaired image-to-image translation models.
2) As pointed in many prior works on image-to-image translation problems [1, 2, 3], diversity remains an important characteristic of image-to-image translation models for multimodal pairs of domains. Even though authors provide standard deviations of the image quality metrics, the study lacks of diversity evaluation for the proposed method. 
3) As pointed out in [1], optimization of cycle-consistency losses struggles for pairs of image domains, where there is a big complexity difference and bijection assumption does not hold, for example, for the sketch-to-image problem. The study lacks of such pairs for image domains in the evaluation protocol.
4) The work lacks of explanation about the inference efficiency of IBCD compared with GAN-based image-to-image translation models such as number of training parameters and model sizes, which seems to be unexpected. 

[1]. Augmented CycleGAN: Learning Many-to-Many Mappings from Unpaired Data. ICML-2018.
[2]. Multimodal Unsupervised Image-to-Image Translation. ECCV-2018.
[3]. StarGAN v2: Diverse Image Synthesis for Multiple Domains. CVPR-2020.

### Questions
1) Can you comment on the training time and stability of the distillation for IBCD compared with baselines?
2) Can you quantify the diversity of IBCD model and baselines, for example, following the MUNIT approach [2]?
3) Can you comment on the number of training parameters and model size of IBCD compared to other baselines? The result of IBCD, which is 5 times faster than StarGAN-v2, looks impressive and I would like to understand this difference. 
4) Can you comment on the applicability and performance of IBCD on the pairs of image domains, where the image-to-image translation is multimodal by the nature and bijection assumption does not hold? For example, for the problem of sketch-to-image translation. 
5) Can you comment on the choice of the distance function $d$ and how it affects the results? For example, some methods employ $L_2$ distance instead of LPIPS in consistency loss [4].
6) DMD loss seem to be applied in previous works for diffusion unpaired image-to-image translation models. I suggest authors discuss the relation of their implementation of DMD loss and its implementation in the paper [5].

[4] Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion. ICLR-2024.
[5] Regularized Distribution Matching Distillation for One-step Unpaired Image-to-Image Translation. Structured Probabilistic Inference & Generative Modeling workshop of ICML 2024.

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
4

### Summary
The authors propose a distillation approach for the previously introduced method for domain translation using unpaired data, known as the Dual Diffusion Implicit Bridge (DDIB). DDIB utilizes two pre-trained diffusion models for each domain, or one conditional diffusion model pre-trained on multiple domains to “concatenate” PF-ODE between the first domain and the noise distribution, with PF-ODE between the noise distribution and the second domain. This approach enables the creation of a PF-ODE that follows from the first domain through the noise distribution to the second domain. The method Implicit Bridge Consistency Distillation (IBCD) proposed by the authors is, in essence, a combination of consistency distillation and distribution matching distillation, adapted to distill “concatenated” PF-ODE in both directions of the first and second domains with the addition of CycleGAN loss. The authors evaluate their method on toy data as well as unpaired image translation on AFHQ and Celeba-HQ datasets, and compare with previous works on the unpaired image-to-image translation method, including the OpenAI foundational model.

### Strengths
- The authors propose the adaptation of a combination of consistency and DMD distillation for DDIB.
- A wide list of competitor methods is considered, including the OpenAI foundational model.
- The authors show that the developed model outperforms competitors on quality of generation.
- A user study is provided.

### Weaknesses
- The proposed approach is more like an engineering combination of previously proposed distillation methods to the previously proposed method for unpaired domain translation.
- While the authors show superiority of their method compared to other competitors in the quality of generation, this result is mainly due to the usage of the strong teacher model, which also outperforms competitors. However, there is no comparison of the parameter count and training time of these models. There is a possibility that the teacher model outperforms other baselines due to the usage of significantly more parameters and compute, and the distilled version does the same for the same reason.
- I suspect that in the comparison of inference time in Table 7, usage of different batches may not be correct, since there may be a constant overhead for each call of the neural network; hence, methods that use higher batch sizes may be better in img/sec ratio. Since the key part of all methods is to make some number of forward passes of the neural network, a comparison of 1 forward pass of each network with the same batch of images multiplied by NFE would better demonstrate the computational complexity of each model.
- Some other baselines based on SB theory are missed, such as, but not limited to [1, 2].

[1] Shi Y. et al. Diffusion schrödinger bridge matching //Advances in Neural Information Processing Systems. – 2023. – Т. 36. – С. 62183-62223.

[2] Mokrov P. et al. Energy-guided Entropic Neural Optimal Transport //The Twelfth International Conference on Learning Representations.

### Questions
Could the authors provide more details on the number of parameters of all methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel one-step method called Implicit Bridge Consistency Distillation (IBCD) for unpaired Image-to-Image translation by extending the consistency distillation framework to concatenated source-target PF-ODE obtained via Dual Diffusion Implicit Bridge (DDIB). In addition to this, the authors propose using a Distribution Matching loss with an adaptive reweighing scheme based on the introduced distillation complexity proxy to enhance the realism of samples, and a cycle-consistency loss to improve faithfulness. The proposed approach beats the corresponding baselines across multiple commonly used unpaired image-to-image benchmarks.

### Strengths
1. The application of Consistency Distillation to the unpaired image-to-image translation is novel and conceptually interesting.
2. The method proposed in the paper enables simultaneous bidirectional training and one-step inference.
3. The experimental section is extensive, with convincing quantitative and qualitative results, including ablations, failure cases, and a user study.
4. The MRI Contrast Translation experiments suggest potential applicability beyond standard image translation tasks, which strengthens the paper’s general interest.

### Weaknesses
1. The current quantitative comparison omits comparison with Optimal Transport methods, such as NOT [1] and/or ASBM [2], which is a significant methodological gap.
2. The two-stage training pipeline raises questions about efficiency and stability. As indicated in Table 4, the training of the first IBCD-only stage consumes the majority of the training time, while the second stage, which enables a better trade-off in the end, accounts for less than 20% of the total training steps. This imbalance suggests that the student initialisation may be suboptimal and that the training dynamics could be unstable when combining objectives.
3. The benefit of adding DMCD and DMCD & Cycle losses to IBCD-only on Figure 3 is not clearly demonstrated. Since DMCD should enhance target distribution realism and cycle consistency should enforce source-target faithfulness, the visual distinctions should be more evident.

References:
- [1] Neural Optimal Transport
- [2] Adversarial Schrodinger Bridge Matching

### Questions
1. How does bidirectional training affect performance compared to a unidirectional IBCD model? Does it improve the final trade-off between realism and faithfulness, or does it introduce additional instabilities?
2. What motivates the two-stage training design and the rapid convergence of the second stage? Why is joint training (IBCD + non-adaptive DMCD + Cycle) from the start not feasible?
3. How long does the student model take to converge, and how does its total training time compare to the teacher model’s training time?
4. Could you please provide parameter counts for both teacher and student models, and for the diffusion-based baselines used in the image translation experiments?
5. Could you expand on the MRI Contrast Translation setup? Specifically, how was the IBCD teacher model trained, and did the diffusion-based baselines share the same teacher initialisation?

### Soundness
3

### Presentation
2

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
The paper proposes a novel approach, called IBCD, for solving unpaired image-to-image (I2I) problems. The method suggests training a one-step bidirectional translation map via consistency distillation of the DDIB (Denoising Diffusion Implicit Bridges) trajectory (which consists of the two composed diffusion ODE trajectories: source$\to$ noise and noise $\to$ target). Additionally, the authors propose using DMD loss with adaptive weighting to enhance the realism of the map's outputs and cycle-consistency loss to improve the input-output alignment. The authors empirically validate the importance of the components in the toy 2d experiment and in the unpaired image-to-image translation problems. The method yields superior results compared to the GAN-based and diffusion-based baselines on the AFHQv2 and CelebA-HQ translation benchmarks.

### Strengths
1) Combination of the cycle-consistency loss with DMD on the outputs and DDIB distillation is novel;
2) Adaptive weighting of the DMD loss looks promising and demonstrates efficiency in the toy experiment;
3) The method has efficient one-step inference;
4) The method outperforms the GAN-based baselines and most of the diffusion-based baselines (except the teacher DDIB, where IBCD has better alignment but worse FID).

### Weaknesses
1) Comparison with the diffusion-based baselines raises questions about fairness. While DDIB teacher and IBCD share the same class-conditional EDM backbone trained by the authors, the results reported in ILVR, EGSDE, and CycleDiffusion are obtained with the discrete-time DDPM backbone introduced in ILVR in 2021. I think unifying the backbone for all the sampling-based diffusion methods is essential for the fair comparison;
2) Several baselines are missing. The authors report quite a comprehensive amount of GAN-based and diffusion-based baselines, but it is essential to perform comparison with optimal transport (OT)-based baselines. GAN-based methods are older and are typically outperformed by diffusion-based methods (thus, I believe, their relevance is limited), while diffusion-based methods often suffer from lower input-output alignment compared to the one-step counterparts (and one of the advantages of IBCD is better alignment compared to e.g. DDIB teacher model). I appreciate adding UNSB, but comparing IBCD to such methods as e.g. NOT [1] and DIOTM [2] would greatly benefit the paper in terms of positioning against one-step baselines;
3) An important related work [3], which proposes to modify the DMD procedure for image-to-image scenarios, is missing;
4) The method description sometimes seems overloaded and overcomplicated (e.g. Equations 9, 10, 11);
5) The method would greatly benefit from studying higher-dimensional problems or a more diverse set of problems e.g. class- or prompt-conditional I2I translation, or translation between different types of domains.

[1] Neural Optimal Transport

[2] Improving Neural Optimal Transport via Displacement Interpolation

[3] Regularized Distribution Matching Distillation for One-step Unpaired Image-to-Image Translation

### Questions
In Table 3, the authors present the effect of different components of the method on the performance in terms of FID and PSNR. The improvement in the input-output alignment is expected after adding cycle consistency loss. However, adaptive DMCD strategy seems to be designed for enhancing realism of samples. Could you please tell why it has a pronounced effect on alignment while slightly harming realism?

### Soundness
2

### Presentation
2

### Contribution
2
