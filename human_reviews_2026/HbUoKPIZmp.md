# There is No VAE: End-to-End Pixel-Space Generative Modeling via Self-Supervised Pre-Training

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Pixel-space generative models are often more difficult to train and generally underperform compared to their latent-space counterparts, leaving a persistent performance and efficiency gap. In this paper, we introduce a novel two-stage training framework that closes this gap for pixel-space diffusion and consistency models. In the first stage, we pre-train encoders to capture meaningful semantics from clean images while aligning them with points along the same deterministic sampling trajectory, which evolves points from the prior to the data distribution. In the second stage, we integrate the encoder with a randomly initialized decoder and fine-tune the complete model end-to-end for both diffusion and consistency models. Our framework achieves state-of-the-art (SOTA) performance on ImageNet. Specifically, our diffusion model reaches an FID of 1.58 on ImageNet-256 and 2.35 on ImageNet-512 with 75 number of function evaluations (NFE) surpassing prior pixel-space methods and VAE-based counterparts by a large margin in both generation quality and training efficiency. In a direct comparison, our model significantly outperforms DiT while using only around 30\% of its training compute. Furthermore, our consistency model achieves an impressive FID of 8.82 on ImageNet-256, significantly outperforming its latent-space counterparts. This marks the first successful training of a consistency model directly on high-resolution images without relying on pre-trained VAEs or diffusion models. Our codes are available at: \href{https://github.com/AMAP-ML/EPG}{https://github.com/AMAP-ML/EPG}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel two-stage training framework for pixel-space consistency models. In the first stage, a portion of the generative model, referred to as the encoder, is pre-trained using self-supervised learning (SSL) methods, providing beneficial initialization. In the second stage, the entire model is fine-tuned with an additional network component, termed the decoder. This approach eliminates the need for latent diffusion training and achieves superior generation performance compared to existing end-to-end pixel-space generative models.

### Strengths
- The motivation is clearly articulated. In particular, the introduction effectively positions the work, helping readers understand its aims and benefits.
- The proposed method does not require external models, which may enhance its applicability to other domains.
- The overall training pipeline is presented clearly.
- The empirical results are impressive, and the evaluation is extensive.

### Weaknesses
[Major comments]
- Although two types of contrastive loss are proposed for the pre-training stage (Equation (8)), a more in-depth analysis of how each contributes to performance improvement (rather than reporting only FID scores) would be beneficial.
- It is unclear why $(\cdot)^-$ is applied to the first term and $sg$ to the second term in Equation (8). Could you provide further insight into this design choice?
- The fine-tuning stage lacks clarity, as $\theta'$ is not defined.

[Notation errors] There are several notation errors and unclear formulations:
- $\omega$ in the caption of Figure 2 should be $\theta$?
- The definitions of $(\cdot)^-$ in Lines 177 and 193 are inconsistent.
- $\tau_2(t)$ in Line 274 should be $\tau$?
- What is $\theta'$ in Equation (9)?
- In Equation (9), does $E_{\theta'}$ include the projection used in the pre-training stage?

[Minor]
- The reference for VAE cites the wrong publication year.

### Questions
- Have you experimented with performing the second-stage training while keeping the encoder, pre-trained in the first stage, frozen? Readers may be interested in understanding the trade-off between the flexibility of $f_\theta$ and the faithfulness to the representations learned in stage 1.
- How would the intermediate features obtained by $E_\theta$ differ if both $E_\theta$ and $D_\theta$ were trained in a single stage without the contrastive loss? If possible, visualizing this difference would be of interest to readers.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a two-stage training framework — pre-training (representation consistency + contrastive learning) to help the encoder learn semantic features, followed by end-to-end fine-tuning with a randomly initialized decoder. The framework can be applied to pixel-space diffusion and consistency models, achieving strong results on ImageNet-256.

The paper is clearly written and easy to follow. The motivation, methodology, and experimental evidence are all presented in a logical and convincing manner.

In terms of originality, while the approach does not introduce fundamentally new techniques, it leverages existing methods in a novel way to improve the performance of high-dimensional pixel-space generative models, which is meaningful and valuable.

The experiments are relatively comprehensive and well-designed.

### Strengths
Please refer to Summary.

### Weaknesses
My main concern lies in the comparison with the latest latent-space methods, particularly RAE [1], where the proposed approach still lags behind in performance (ImageNet-256 FID 2.04 vs. 1.51). Moreover, most of the baselines used in the paper are relatively outdated (especially the pixel-level models), and the reported improvements over them are not very substantial.

Compared to VAE-based approaches, the proposed two-stage training pipeline does not demonstrate a clear advantage. Notably, recent works such as REPA-E [2] have also shown that end-to-end joint training of VAEs and diffusion models is feasible and effective.

Overall, while the method is conceptually simple, it does not appear to be sufficiently effective or provide strong new insights.

[1] Diffusion Transformers with Representation Autoencoders
[2] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

### Questions
Please refer to Weaknesses.

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
This paper proposes a novel two-stage training framework (tentatively referred to as EPG) for advancing end-to-end pixel-space generative models, addressing the long-standing limitations of Variational Autoencoder (VAE)-dependent paradigms in the field of diffusion and consistency models. The framework decouples the complex generative task into self-supervised pre-training and end-to-end fine-tuning phases: in the first phase, the encoder is trained independently via contrastive loss and representation consistency loss to learn noise-robust visual features; in the second phase, the pre-trained encoder is concatenated with a randomly initialized decoder for end-to-end optimization targeting downstream generative tasks. Experiments on ImageNet-256 and ImageNet-512 datasets demonstrate competitive performance—achieving FID scores as low as 2.04 and 2.35 respectively with only 75 inference steps—and pioneering pixel-space consistency model training without VAE reliance, yielding 8.82 FID in single-step generation. This work bridges the efficiency gap between pixel-space and latent-space models while simplifying the training pipeline.

### Strengths
Paradigm Innovation: Breaking VAE Dependence
The paper makes a significant paradigm contribution by eliminating the need for VAEs in high-quality generative modeling—a critical limitation of mainstream approaches like Stable Diffusion and DiT . VAEs introduce inherent trade-offs between compression ratio and reconstruction quality, and require costly joint fine-tuning across domains . By decoupling representation learning from pixel reconstruction, the proposed two-stage framework resolves these issues: the self-supervised pre-training phase ensures robust feature extraction from noisy data, while the lightweight fine-tuning phase avoids the complexity of VAE optimization. This "de-VAE" design aligns with emerging trends in pixel-space generation and significantly lowers the barrier to adapting generative models to new domains.

### Weaknesses
Limited Evaluation on High-Resolution and Diverse Datasets
While the paper demonstrates results on ImageNet-256/512, it lacks validation on higher-resolution tasks (e.g., 1024×1024 FFHQ) where pixel-space models historically struggle . The choice of ImageNet (natural images) also raises questions about generalizability to other domains (e.g., medical imaging, satellite imagery) where VAE artifacts are particularly problematic. Furthermore, key video generation metrics like temporal consistency or FVD are absent, despite the paper hinting at extensibility to video—leaving uncertainty about whether the framework can capture spatiotemporal dependencies.

### Questions
What is the impact of ODE path sampling density on the representation consistency loss and final generation quality? Could you supplement ablations or visualizations to clarify this mechanism?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a two-stage framework for (1) representation learning with consistency regularization, and (2) fine-tuning for the generative model. To make the optimization of the representation learning in the first stage work, the authors leverage EMA and stop gradients to form the training objectives. The fine-tuning is conducted using a diffusion model and extra regularization to ensure that the encoder remains structured and meaningful.

### Strengths
The authors carefully designed the training objectives for leveraging consistency loss in the training of the encoder.

The authors focus on important questions and give good justifications for their initiatives.

### Weaknesses
My main concern is that I am not sure if the paper can be claimed as a "pixel-based" generative model, since:

(1) The consistency regularization is conducted in the latent space.

(2) The generative model is also trained in the diffusion model. (Also, I do think the authors should specify how the generative model is trained in their diagram, as from the current one, you might think they are learning a generative model in the 1st stage of learning representations, which is not the case.

### Questions
Have you evaluated the encoder's performance?

How's the performance if you don't use the consistency regularization? Or simply using it without the contrastive objective.

### Soundness
2

### Presentation
2

### Contribution
3
