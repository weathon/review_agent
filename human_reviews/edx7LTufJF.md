# Efficient Low-Rank Diffusion Model Training for Text-to-Image Generation

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 3, 3

## Abstract
Recent advancements in text-to-image generation models have witnessed the success of large-scale diffusion-based generative models. However, exerting control over these models, particularly for structure-conditioned text-to-image generation, remains an open challenge. One straightforward way to achieve control is via fine-tuning, often coming at the cost of efficiency. In this work, we address this challenge by introducing ELR-Diffusion (Efficient Low-rank Diffusion), a method tailored for efficient structure-conditioned image generation. Our innovative approach leverages the low-rank decomposition of model weights, leading to a dramatic reduction in memory cost and model parameters — by up to 58\%, at the same time performing comparably to larger models trained with expansive datasets and more computational resources. At the heart of ELR-Diffusion lies a two-stage training scheme that resorts to the low-rank decomposition and knowledge distillation strategy. To provide a robust assessment of our model, we undertake a thorough comparative analysis in the controllable text-to-image generation domain. We employ a diverse array of evaluation metrics with various conditions, including edge maps, segmentation maps, and image quality measures, offering a holistic view of the model's capabilities. We believe that ELR-Diffusion has the potential to serve as an efficient foundation model for diverse user applications that demand accurate comprehension of inputs containing multiple conditional information.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a parameter-efficient training scheme for controllable T2I generation. The essence of the method is to convert memory cost for fine-tuning into time cost by introducing a knowledge distillation step. Specifically, the paper introduces a two-stage approach. In stage I, the diffusion model with attention layers whose linear matrices could be decomposed into two low-rank matrices is trained by knowledge distillation. In stage II, the diffusion model is fine-tuned for controllable T2I generation via  Feature Denormalization (FDN).

### Strengths
- The proposed method is parameter-efficient for the fine-tuning stage: it can reduce the number of tuned parameters to half of the original diffusion model.
- The proposed initialization method using truncated SVD is intriguing as it can preserve a portion of the performance of the teacher model.

### Weaknesses
- **The paper contains exactly the same sentences for one section as another paper.**
- Technical novelty is not significant. Ideas of knowledge distillation, and low-rank constrained fine-tuning have been widely used not only for the diffusion model but also for various foundation models. The paper just combines them to convert the memory cost into time cost.
- The efficiency of the proposed method is doubtable.
- Comparison with LoRA is missing, although LoRA also offers a parameter-efficient method to get a desired model with the same memory requirements (11GB GPU VRAM).
- There are multiple issues in writing: repeated sentences, notations without description, over-stated sentences, and wrong descriptions.

### Questions
- The usage of the section 2.1 is not clear. The reviewer could not find a clear connection to the proposed method.

-  In equation (3), does the low-rank decomposition for linear transformations always hold? Empirically, we know that $\Delta W$ after fine-tuning of attention layers shows low-rankness (namely LoRA), but the reviewer wonders if it could be a general assumption for any linear matrices.

- The description of diffusion model training in section 2.2 is too simplified. The sentence does not clearly show the objective function and the meaning of the trained diffusion model. In parallel, the equation (4) is limited to offer enough information about the proposed method. What should be given as input to $\tilde{\epsilon}$ and $\hat{\epsilon}_\theta$? Is it noisy images or noisy latent representations? Does the method use classifier-free guidance (CFG)? It would be helpful to express the equations precisely.

- It is doubtful that 10GB memory reduction is worth even with significant performance degradation. Actually, one can simply conduct LoRA fine-tuning for Stable-Diffusion on 11GB of GPU RAM (https://huggingface.co/docs/diffusers/main/en/training/lora) without additional knowledge distillation process which also raises a question about the efficiency of the proposed method. 

- Comparison with LoRA fine-tuning is required to analyze the efficiency of the proposed method. Due to the knowledge distillation stage, the proposed method may lose its capability and the important aspect for the efficiency is how much.

- In Table 2, only run-time for each iteration is computed to compare the efficiency of the methods, but it is a trivial result since the number of learnable parameters is reduced. To fairly compare the efficiency of the proposed method, the entire time to get a task-specific model should be computed because it is the time that users really consume to get their own model. Especially, the time for stages I and II should be summed since the proposed method requires additional knowledge distillation processes that play a key role in reducing memory cost by allowing low-rank initialization before fine-tuning. It would be helpful to analyze the efficiency of the proposed method if the paper reports the training time for the entire procedure.

- The performance of the proposed method is different in Tables 2 and 3, while the performance of the baseline and metrics are the same. There is no mention about the difference in the experimental setup.

- In table 3, the performance difference $\Delta$ is computed for different baselines. For efficiency, the baseline is Uni-ControlNet, but for the qualities, the baseline is the proposed method without knowledge distillation which is closer to the ablation study. This would be confusing since this seems to say the performance is better than Uni-ControlNet, even if the actual result is not.

- Based on table 3, the quality of the generated image by the proposed method is worse than the baseline. However, in figure 4, all examples show that the proposed method is better than the baseline method which is not aligned with the quantitative comparison. It would be better to display more examples for the qualitative comparison in the main paper or appendix to precisely see where the proposed method is located.

- The writing could be improved with precise descriptions. For example,
	- In the second contribution in the introduction section, the paper does not state that the low-rank decomposition is also leveraged for Stage 1.
	- In section 2.2, the meaning of "domain-specific encoder $\tau_\theta$" contains ambiguity. Does the domain mean modalities such as text and image, or does it mean categories of given language prompts such as English and French? Also, no description for $M, d_\tau, z_t$, the superscript $i$ for linear transformations, and $\tau$ in equation (3).
	- Redundant expressions in section 2.4: "This architecture and the weights inherit the low-rank format..." and following "These weights are represented in a low-rank format..."
	- No description for $A_i$ matrix which is introduced by the Kronecker decomposition (equation 10). Thus, it is not clear what the proposed method is intended to learn (i.e. what are learnable parameters) exactly.
	- In section 3.5, the third sentence in the second paragraph says that "our approach maintains the original full parameter count". However, the proposed method actually reduces the number of parameters by the truncated SVD in stage I.
	- Repeated sentences in the caption of Table 3.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a diffusion model compression and efficient fine-tuning framework. It first compresses the diffusion model via model distillation and then introduces a low-rank efficient fine-tuning approach with the low-rank assumption of the student model. The paper verifies the proposed distillation can reduce the model size, memory footprint, and improve the fine-tuning efficiency without hurt the model performance significantly.

### Strengths
1. The paper is well written and easy to follow.
2. The evaluation verifies the contributions, including model distillation can reduce the diffusion model size, low-rank fine-tuning can reduce the memory requirement, etc.

### Weaknesses
1. The novelty of this paper is somewhat trivial. The proposed two stage approach is a simple combination of model distillation and low rank fine-tuning. The paper uses the vanilla model distillation and low rank fine-tuning without much innovation.

2. The paper did not explain how to improve the controllability of the generation model by the proposed approach. There is no evidence to show how the control is improved. The visualization results in Figure 4 is not sufficient to justify the capability improvement of controlling the generation process.

3. The proposed approach distills the model first and fine-tunes it for downstream tasks in the second step. The distillation does not involve the downstream tasks information which could hurt the performance. Would it make more sense to fine-tune the model first and distill it then?

4. The proposed fine-tuning assumes the matrix is low-rank. There are no ablation studies that verify such an assumption. In general, the distilled model should be denser than the teacher model. Would the low-rank assumption hurt the performance if the matrix is not low-rank?

5. The evaluation is not sufficient to justify the contribution. For example, there is no evaluation between LORA and the proposed approach.

6. Some theoretical analysis is confusing. For example, in eq 8, \theta_0^d should be expressed as a SVD transformation of \theta_0 instead of a linear transformation of \theta_0. Can authors clarify how eq 8 is derived?

### Questions
Extra questions besides the questions in weakness 3, 4, and 6

7. The paper assumes the distillation is model architecture distillation. Would the step distillation help here?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposes a neural network compression method for fine-tuning text-conditioned diffusion models. They distill a larger U-Net backbone in LDM to a smaller student U-Net using low-rank decomposition. They compare their results to Uni-ControlNet and show decreased computation costs compared to the baseline.

### Strengths
1. The recent generative models require large-scale training and increasing their efficiency for training and fine-tuning is an interesting and important problem.

2. The paper is well-written and is easy to follow.

### Weaknesses
1. Although the task is image generation, there are only a few qualitative samples. Also, the quantitative metrics only include FID and CLIP score. The authors could also include KID and Inception Score.

2. The performance loss in FID and CLIP Score is large. The loss of image quality is also visually visible in the qualitative results.

3. The combination of low-rank decomposition for knowledge distillation is not novel and has been used in previous works.

4. There are a lot of ambiguities in details such as the teacher and student architectures, the amount of fine-tuning, the hyperparameters, etc. These make the work not reproducible.

5. Although there have been many network compression methods, none of these has been compared against in the results.

[a] Li, Tianhong, et al. "Few sample knowledge distillation for efficient network compression." CVPR 2020.
[b] Dai, Cheng, et al. "A tucker decomposition based knowledge distillation for intelligent edge applications." Applied Soft Computing 101 (2021): 107051.
[c] Gu, Yuchao, et al. "Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models." arXiv preprint arXiv:2305.18292 (2023).
[d] Yao, Xufeng, et al. "Distill Vision Transformers to CNNs via Low-Rank Representation Approximation." (2022).
[e] Rui, Xiangyu, et al. "Unsupervised Pansharpening via Low-rank Diffusion Model." arXiv preprint arXiv:2305.10925 (2023).
[f] Ryu, Simo, Seunghyun Seo, and Jaejun Yoo. "Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals." arXiv preprint arXiv:2305.18425 (2023).

### Questions
1. Since the architecture used in ELR has less number of parameters compared to Uni-ControlNet, it requires fewer steps and data for fine-tuning. Has this been considered in the experiments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an approach for compressing diffusion models designed explicitly for refining text-conditioned diffusion models. The paper discusses several related works in efficient training, including Prompt Tuning, Low-Rank Adaptation (LoRA), and Adapters.  Compared to these existing methods, the paper offers an approach for efficient training and fine-tuning in text-to-image generation. It leverages the low-rank decomposition of model weights, leading to a reduction in memory cost and model parameters. The paper presents a two-stage training scheme, where a smaller U-Net structure is trained with low-rank schema in the first stage, and fine-tuning is conducted in the second stage. The technique involves distilling a more cumbersome U-Net backbone in the Latent Diffusion Model (LDM) into a more compact with comparable or less effective student U-Net by applying low-rank decomposition. The performance of this method is compared with that of Uni-ControlNet, demonstrating a reduction in computational costs relative to the baseline in unconditioned and conditioned settings. This suggests that the proposed method offers a more efficient alternative for fine-tuning text-conditioned diffusion models. For evaluation, the authors reported Normalized Mean Square Error (NMSE), Intersection over Union (IoU), Fréchet Inception Distance (FID), and CLIP Score in data from the LAION-6+ dataset to assess the performance of the proposed ELR-Diffusion method compared to the Uni-ControlNet model in terms of edge maps, segmentation maps, and image quality.

### Strengths
The paper claims that ELR-Diffusion achieves comparable performance to larger models trained with expansive datasets and more computational resources while reducing the memory cost by 45% and the number of trainable parameters by 58% (specific experiments).

The literature review was interesting to follow and read.

### Weaknesses
Since the low-rank decomposition has been proposed in previous works and would not be considered a new approach, further and more extensive evaluations in qualitative and quantitative results sections would help the readers understand how effective this method is compared to other baselines. I also would like to see experiments on how this compression can affect image generation in other domains and datasets. It would also be interesting to see its effect on different resolutions as well.

The experiment settings need to be in more detail for the sake of reproducibility,

### Questions
Have you considered the effect of different schedulers on how the compression would be effective?
What are the results on other datasets?
Have you noticed any exciting phenomenon, negative or positive, in the qualitative results? I am curious if compression could lower the diversity or maybe decrease the performance of particular objects. E.g., a comparison between small vs large things.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
