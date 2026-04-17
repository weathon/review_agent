# On Designing Diffusion Autoencoders for Efficient Generation and Representation Learning

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Diffusion autoencoders (DAs) are variants of diffusion generative models that use an input-dependent latent variable to capture representations alongside the diffusion process. These representations can be used for tasks such as downstream classification, controllable generation, and interpolation. However, the generative performance of DAs relies heavily on how well the prior distribution over the latent variables can be modelled and subsequently sampled from. Better generative modelling is also the goal of another class of diffusion models—those that learn their forward (noising) process. While effective at adjusting the noise process in an input-dependent manner, they must satisfy additional constraints derived from the terminal conditions of the diffusion process. Here, we draw a connection between these two classes of models and show that certain design decisions (latent variable choice, conditioning method, etc.) in the DA framework—leading to a model we term DMZ—enable effective representations as evaluated on downstream tasks, including domain transfer, as well as more efficient modelling and generation with fewer denoising steps compared to standard diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a novel diffusion autoencoder technique, DMZ, based on the empirical analysis of latent z. By setting z as discrete binary encoding, using cross-attention as conditioning, and using dense latent variable dimension, it achieved faster convergence, better generation quality, as well as without any prior or loss function. The experiment results and ablation study results demonstrated this conclusion.

### Strengths
1. DMZ firstly established the theoretical fundation of learnable forward process of diffusion autoencoder.

2. The design motivation of z is reasonable and inspiring. Correspondingly, the improvements (binary z, etc) are effective in the following experimental results. 

3. The experiments are comprehensive, not only demonstrating the effectiveness of each z components, but also extend the task to the other tasks dependent on z (such as stretch2pic). The ablation studies are also reasonable and promising.

### Weaknesses
1. Although the learnable forward process is proposed with good motivation, there lacks formal induction or convergence analysis.

2. Benchmark analysis on high-resolution datasets is recommended. 

3. There lack interpretability analysis of latent variables, incluing the semantic understanding, differentability, and the combinmation capability as condition for discrete binary z.

4. The flow of method section should be adjusted. DA and learnable forward process should be discussed separately with suitable connections.

### Questions
1. Can the authors discuss the connection between DMZ and REPA, which is also a promising baseline in diffusion representation field. In my view, binary z can be seen as a simplified type of external embedding guidance. If so, how about extending z to boarder fields like it is in REPA?

2. How about the experimental results in high-resolution datasets? How does the efficiency change with the lantent dimension? 

3. Can the author propose more analysis and results on different types of priors rather than Bernoulli? 

4. Is there any insight towards the design of conditioning z by cross attention instead of conditioning it from the residue network?

5. Can DMZ be combined with the current SOTA diffusion models, such as cosistency model and rectified flow?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DMZ, kind of diffusion autoencoders.
DMZ aims to improve the generative quality of diffusion models by guiding the sampling process using the latent representation $z$ of $x_0$.
The model is trained without any additional loss terms, following the standard DA training objective.
Unlike conventional diffusion autoencoders, DMZ does not require an auxiliary latent sampler.
Instead, it directly samples the latent variable z from a Bernoulli distribution, which improves sampling efficiency.
Furthermore, the authors empirically show that conditioning only the Key and Value components of the cross-attention layers on z leads to better performance than other conditioning strategies.

### Strengths
- Unlike previous diffusion autoencoders, DMZ does not rely on an auxiliary latent sampler.
By directly sampling $z$ from a Bernoulli distribution, the method enables computationally efficient sampling.

- The learned latent representation is shown to be effective even in a multi-modal framework, indicating its potential generality beyond standard generation tasks.

- The proposed DDPM-based approach demonstrates clear improvements in generation quality, particularly when using a small number of denoising steps.

### Weaknesses
- It is unclear how the latent variable can be sampled from a Bernoulli distribution without any prior regularization.
In standard DA frameworks, auxiliary latent samplers (such as [1,2]) or additional regularization terms (such as [3]) are typically used to properly model the latent prior.
Without such mechanisms, it is not evident how the encoder output would naturally follow a Bernoulli prior.
This appears to be a critical limitation of the proposed method.

- The effect of conditioning z only on the Key and Value in the attention layers is not clearly explained.
While the authors report that this approach outperforms the alternative of jointly conditioning with t, the reason for this improvement remains unclear.
Additional analysis or experiments would strengthen this claim.

---------
[1] [CVPR22] Diffusion autoencoders: Toward a meaningful and decodable representation

[2] [NeurIPS 22] Unsupervised representation learning from pre-trained diffusion probabilistic models

[3] [ICML23] Infodiffusion: Representation learning using information maximizing diffusion models

### Questions
- Was any specific encoder architecture or constraint introduced to make the encoder output binary? How is this discreteness enforced during training?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DMZ, a design for diffusion autoencoders that aims to improve both generation efficiency and representation learning. By incorporating a input-dependent encoder, DMZ explores the distribution choices, conditioning mechanisms, and learning strategies to enhance the performance of diffusion autoencoders. Experiments on CIFAR-10 and CelebA demonstrate the effectiveness of DMZ and its potential ability for style transfer and representation learning.

### Strengths
- The focus on diffusion autoencoders is timely and relevant, addressing the need for efficient generation and representation learning.
- The illustrations and explanations of DM/DA are clear.
- The benchmarking tasks and datasets are appropriate for evaluating the proposed method.

### Weaknesses
- The motivation and contribution of DMZ is unclear.
- The algorithmic details of DMZ are insufficient.
- The performance of DMZ is underwhelming compared to existing methods.

### Questions
0. **DMZ meaning.** What does DMZ stand for? The acronym is not explained in the paper.

1. **Motivation and contribution.** I feel confused about the motivation and contribution of DMZ, and believe the writing could be potentially largely improved for clarity. In the introduction section, the authors claim "to draw a connection between DMs and DAs". However, I could not find any discussion or analysis on DA/DMZs in the rest of the paper. How are DMZ and DA different? Is it the contribution of DMZ to propose a new DA framework, or explore the design space of DAs? Could the authors clarify the main contribution of this work?

2. **Algorithmic details.** The algorithmic details of DMZ are insufficiently described. Only Eq.(5) describes the training objective of DA. Does DMZ use the same training objective as DA? Additionally, could the authors provide more details about the newly-proposed components in DMZ, including conditioning mechanisms and learning strategies? A more comprehensive description of the algorithm would help readers better understand the proposed method.

3. **Performance comparison.** The performance of DMZ seems underwhelming compared to existing methods. In Table 1, DMZ achieves worse performance on CIFAR-10 compared to DDPMs. In Table 3, DMZ achieves worse performance compared to DDBMs. Could the authors provide more analysis on why DMZ underperforms compared to these methods? Are there any specific limitations or challenges in the DMZ design that lead to this performance gap?

4. **Inconsistent experimental setup.** In Figure 3 the authors compare NLL on CIFAR-10 and FID on CelebA. Are there any specific reasons for using different datasets?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose a diffusion autoencoder framework, DMZ, with carefully designed strategies such as latent variable choice, conditioning methods, and more. The paper provides a comprehensive study of each component’s design choices. Empirically, DMZ shows consistently strong performance in both unconditional generation and representation learning. The authors also demonstrate that DMZ can be easily applied to multimodal tasks (e.g., image-to-image translation), highlighting the framework’s flexibility.

### Strengths
* The paper is clear and easy to follow. The comprehensive experiments convincingly isolate and evaluate the effects of each design choice.
* The DMZ framework is fairly general: it performs well in unconditional generation and representation learning, and it can be extended to handle multimodal tasks such as image-to-image translation.

### Weaknesses
* The cross-attention conditioning design is already widely used in modern diffusion transformers [1-2]. The current validation relies on an older U-Net architecture, so this component does not constitute a significant contribution by itself.
* The choice of latent dimensionality $|z|$ appears ad hoc. For generation tasks it is guided by the label-space size (suggesting that relatively low dimensions yield better generation quality), whereas representation learning for downstream tasks benefits from more informative, higher-dimensional latents. This implies separate designs for different use cases within DMZ.
* The effectiveness for generation is not fully convincing. If is tied to a (binary) label space, it can logically degenerate to a one-dimensional label with low dimensions. Sampling from this prior is then akin to sampling in label space for conditional generation. Although DMZ does not directly rely on a labeling function $f:X\rightarrow Y$, could clustering be used to produce labels that achieve similar behavior in the "unconditional" setting? This would suggest DMZ may not be learning strong representations in these scenarios.
* DMZ shows limited compatibility with DDIM in Table 10. It would help to evaluate DMZ with more recent denoising approaches and architectures such as DiT and SiT [1–2].
* Extending experiments to larger benchmarks (e.g., ImageNet) would further strengthen the work’s claims and external validity.

[1] Scalable Diffusion Models with Transformers
[2] Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers

### Questions
See Weaknesses 3–5.

### Soundness
2

### Presentation
2

### Contribution
1
