# ToddlerDiffusion: Interactive Structured Image Generation with Cascaded Schrödinger Bridge

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Diffusion models break down the challenging task of generating data from high-dimensional distributions into a series of easier denoising steps. Inspired by this paradigm, we propose a novel approach that extends the diffusion framework into modality space, decomposing the complex task of RGB image generation into simpler, interpretable stages. Our method, termed {\papernameAbbrev}, cascades modality-specific models, each responsible for generating an intermediate representation, such as contours, palettes, and detailed textures, ultimately culminating in a high-quality RGB image.
Instead of relying on the naive LDM concatenation conditioning mechanism to connect the different stages together, we employ Schr\"odinger Bridge to determine the optimal transport between different modalities.
Although employing a cascaded pipeline introduces more stages, which could lead to a more complex architecture, each stage is meticulously formulated for efficiency and accuracy, surpassing Stable-Diffusion (LDM) performance.
Modality composition not only enhances overall performance but enables emerging proprieties such as consistent editing, interaction capabilities, high-level interpretability, and faster convergence and sampling rate. 
Extensive experiments on diverse datasets, including LSUN-Churches, ImageNet, CelebHQ, and LAION-Art, demonstrate the efficacy of our approach, consistently outperforming state-of-the-art methods.
For instance, {\papernameAbbrev} achieves notable efficiency, matching LDM performance on LSUN-Churches while operating 2$\times$ faster with a 3$\times$ smaller architecture.
The project website is available at:
\href{https://toddlerdiffusion.github.io/website/}{$https://toddlerdiffusion.github.io/website/$}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new approach to decompose the diffusion framework into simpler and interpretable stages. The new framework called ToddlerDiffusion is a cascaded model that introduces two additional intermediate representations, sketch and palette. Different modality-specific models are connected to obtain high-quality image outputs with detailed textures. Instead of naive LDM, the Schrödinger Bridge is leveraged to determine the optimal transformation between modalities. The new design achieves better overall performance with faster sampling and smaller architecture compared with LDM, and also enables editing and interaction at intermediate steps.

### Strengths
1.	The framework natively offers decent capability in editing and interpretability without additional adapters or fine-tuning.
2.	By decomposing the image generation into three simpler sub-tasks, the framework has fewer parameters, converges faster, and is more robust against a smaller number of sampling steps.
3.	The framework shows its capacity to achieve better performance on given datasets comparing with naive LDM.

### Weaknesses
1.	There is a large gap between the performance of the baseline model (LDM/SD1.5) and the current state-of-the-art models.
2.	The experiments showed the model’s capability in limited training data and resources, however, the value of proving a smaller model converges faster is limited. There’s lack of the evidence about its performance compared with baselines given sufficient data and training. 
3.	The description of the experiments lacks many details that reduced the robustness of the conclusions. For example, the experiments comparing editing capabilities with controlnet did not mention which base model was used by controlnet or at which resolution was the experiment performed, makes the fairness of the experiment questionable.

### Questions
1.	Given ToddlerDiffusion is a smaller framework compared with LDM, showing it converges faster and has better performance with the same number of epochs is not sufficient to prove it’s a better model. Table 2 claims the training converges at epoch 400 (not mentioning on which dataset), but in Table 1 none of the datapoints was reported at epoch 400. Showing that ToddlerDiffusion outperforms when both training procedures converge would be a stronger evidence.
2.	Please provide more details about the experiments, such as but not limited to: at which resolutions were the three stages trained and evaluated? How was controlnet/SDEdit configured when the editing capacities was compared? Is there any quantitative results supporting it’s editing capacity?
3.	It’s understandable that The generation quality of ToddlerDiffusion is inferior to that of the SOTA model (SDXL, Flux.1, etc.) given limited data and training. However, The use cases for low-resolution unconditional or label-conditioned generation are very limited. Would the first two stages be able to handle more complicated conditions at higher resolutions as in latest models? If so, would the framework still be efficient?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Toddler diffusion, a method to generate images like the thought process of a child. The paper decomposes the image generation process into three stages first on a high level sketch generation, then a colour palette generation and finally generating a complete RGB image. To enable learning from one modality to another, the authors leverage Schrödinger Bridges. Such a form a training enables capabilities of seamless editing into the model. Moreover also enables faster sampling compared to conventional LDMs. The authors perform comparisons with conventional LDMs to illustrate the effectiveness of the method.

### Strengths
1. The proposed framework of a multistep generation process is novel to the best of my knowledge. The method is motivated by an inspiring thought process based on the workflow of how humans think to create a picture.
2. The disentangled training process also enables enhanced controllability, stability and faster training and inference process. Moreover such a generation process also gets benefits of added diversity because of the compositional effect in each stage
3. The performance boost over naive LDMs is substantial while also enabling zero shot capabilities to the the model.
4. The method achieves better performance than LDMs across diverse datasets while showing better training and inference efficiency.

### Weaknesses
1. The training process of toddler diffusion is not clear from the main manuscript. Are different U-Nets leveraged from different modality transformations. I would advise the authors to include a flowchart diagram of signal flow in the main manuscript or alternatively add a subsection detailing this aspect.
2. Toddler diffusion utilizes Schrödinger bridges and an SDE perspective for the training and sampling process. However Rectified Flows have shown to achieve much better performance utilizing ODEs, which also improves further distillation performances. Can the authors contrast on the benefits of training and sampling using diffusion  Schrödinger bridges and rectified flows.
3. Toddler diffusion requires annotated data in different modalities to enable conditional generation. Hence I believe the comparisons provided by the authors by comparing with Naive LDMs are not fair, for a fairer evaluation. I believe the authors need to train a diffusion model which can generate sketch from noise, then 2 conditional models for the translation tasks. Could the authors please provide a comparison with LDMs trained in the proposed manner or alternatively comment on how the current experiments are valid comparisons
4. Can the authors also include performance of Toddler diffusion compared to pixel space diffusion architectures and diffusion transformers for the same datasets. Some valid methods would be SiT[1], DiT[2], Pixel space models[3,4]

   [1] Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E. and Xie, S., 2024. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740.

   [2] Peebles, W. and Xie, S., 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4195-4205).

   [3] Kingma, D. and Gao, R., 2024. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36.

   [4] Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T. and Lehtinen, J., 2024. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724.

### Questions
1. Minor spelling mistake Ln 46  emerging proprieties 

2. Regarding the problem prescribed in section 2.1 Ln 216,217 : [1] provides some suggestions on how to schedule the noise process to achieve zero terminal SNR and fix the inference schedule. I refer the paper here for the reference of the authors

   [1] Lin, S., Liu, B., Li, J. and Yang, X., 2024. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 5404-5411).

### Soundness
3

### Presentation
4

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
This paper introduces a generative framework comprising three stages: noise to sketch, sketch to palette, and palette to RGB images. The paper uses the Schrodinger bridge I2SB [1] to train these stages. This framework takes less training time and samples efficiently. Moreover, the architecture is smaller and has better training convergence. This framework could be used for editing and control generation.

[1]: I2SB: Image-to-Image Schrödinger Bridge

### Strengths
1. The paper is well-written and easy to follow. I particularly like the idea of the "toddler," which is motivated by how the child grows up and learns everything.
2. This training framework is more training/sampling efficient than unconditional diffusion generation. It could be used for editing and control generation since it comprises many conditional generative problems trained by Schrodinger Bridge.
3. The experiment is well-organized.

### Weaknesses
1. The methodology seems simple since it decomposes the generative problem into subsequent conditional generative problems. Each problem can be solved directly using LDM or Schrodinger Bridge, and the paper uses Schrodinger Bridge to solve them like I2SB. Therefore, this paper's novelty is limited since it is just like you apply I2SB directly to solve conditional generative problems.
2. The paper mentions extending the method to Stable Diffusion but omits stage 1. This is a meaningful editing and controllable generation problem. However, for the text-to-image generation, I doubt the ability to train stage 1 mapping from (noise and text) to sketch using the Schrodinger Bridge. The reason behind this is that text is a completely different modality with RGB, sketch, and palette. Therefore, mapping from (noise and text) to sketch is not easily solved by unconditional Schrodinger Bridge. Further, the architecture should be larger and specifically designed for that task. This is an interesting question that should be discussed in the paper to answer the framework's scalability. The author could try to train such (noise and text) to sketch models on small text-to-image datasets such as CUB-200 or COCO dataset. [2] (see this work for small text-to-image problem)
3. The paper should include citations for [1] since their methodology is similar to the paper method. I2SB uses the Schrodinger bridge for image-image problems, which are the same problem as your decomposed conditional generative problems. The only difference is they solve image superresolution/inpainting/deblurring instead of sketch-to-palette and palette-to-image.
4. Minor: Several typos at Eq. 3 (the right-hand equation label should be sketch and palette)

[1]: I2SB: Image-to-Image Schrödinger Bridge
[2]: Vector Quantized Diffusion Model for Text-to-Image Synthesis

### Questions
My biggest concern is about the novelty of this work since the framework could be considered as many conditional generative models. Secondly, I am not sure about the ability to extend this framework to text-to-image generation; this could limit the applicability of the proposed method.

### Soundness
3

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
This work proposes ToddlerDiffusion, a cascaded sequence of relatively small diffusion models, to gradually generate different aspects of images. Those different diffusion models are connected by the Schrödinger Bridge to improve conditioning consistency.

### Strengths
1. **The proposed cascaded framework is reasonable and effective**. With fewer network parameters and training time, ToddlerDiffusion can outperform the standard LDM in both generation quality and conditioning consistency.

2. **Thorough experiments and ablation studies are conducted** in the main content and appendix to demonstrate the versatility of the proposed framework and validate the effectiveness of several design choices.

### Weaknesses
1. **Presentation of this work requires thorough improvements.**.
- The authors should use `\citep` for most cases in the manuscript, which places the authors' names and the year in parentheses. The current use of `\cite` makes the manuscript cluttered and difficult to read.
- The manuscripts contain many incoherent parts. For instance, in Lines 474-479, the authors state: "we used x_0 ... and employed LoRA fine-tuning for efficiency. **But** these two variants are crucial to make it work". The use of the transitional word "but" in this context is confusing. Later, the authors assert that "we **must** fine-tune the entire model instead of using LoRA". Then why is LoRA fine-tuning described as crucial if it is not the recommended approach?
- Notations are inconsistent in the manuscript. For example, in Equation (2), $x_t=\alpha_t x_0^i+(1-\alpha_t)y^i+\sigma_t^2 \epsilon_t$, whereas in Equation (6)~(16), $x_t=(1-\alpha_t) x_0+\alpha_t y+\sigma_t \epsilon_t$.
- What is the meaning of "$:\sigma_t^2=\alpha_t-\alpha_t^2$" in Equation (2)?

2. **Many statements in this work are incorrect or overclaimed**.
- In the abstract: "surpassing Stable-Diffusion (LDM) performance". However, all experiments conducted in the main content are compared with the LDM trained on small datasets, which are NOT Stable Diffusion (v1/v2/v3).
- The training objective of LDM or Stable Diffusion is the noise $\epsilon$. However, $\epsilon$ is NOT $x_t-x_0$ in LDM as stated in Line 266 and Table 4.
- In Line 507-509, by the SNR definition in this work, SNR does not tend to 0 as $t \rightarrow T$, as the concatenated condition $y$ also provides some signals to a certain extent similar to the proposed framework even at $t=T$. SNR should NOT be the reason that distinguishes between different conditioning methods (Concatenation v.s. Schrödinger Bridge).

### Questions
- Could you please provide more details about how to incorporate ToddlerDiffusion into Stable Diffusion? For instance, what text prompts are used? Additionally, is Stable Diffusion fine-tuned on CelebHQ datasets for five epochs using x_0 prediction and the Schrödinger Bridge conditioning mechanism similar to ToddlerDiffusion? Is the network architecture of Stable Diffusion modified to include sketch conditions, as demonstrated in Figure 12? If so, then how are the parameters initialized in the modified components?

- Why do the numbers of LDM parameters differ between Table 2 (263M) and Figure 13 (187M)?

- Why are the schedulers for $\alpha$ and $\sigma$ different in Figure 12 (left) and (right)? Why does $\alpha$ increase from $T$ to $0$ in the left block, but decrease from $T$ to $0$ in the right block? What does the presented $y$ represent in the first stage (Figure 12 left block) that unconditionally generates the sketch?

- Why is SB+Concat not utilized in ToddlerDiffusion, while only SB is used, despite the former demonstrating superior performance in Table 3?

### Soundness
3

### Presentation
2

### Contribution
2
