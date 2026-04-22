# DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Diffusion Transformers (DiTs) have emerged as the dominant architecture for visual generation tasks, yet their uniform processing of inputs across varying conditions and noise levels fails to leverage the inherent heterogeneity of the diffusion process. While recent mixture-of-experts (MoE) approaches attempt to address this limitation, they struggle to achieve significant improvements due to their restricted token accessibility and fixed computational patterns. We present **DiffMoE**, a novel MoE-based architecture that enables experts to access global token distributions through a **batch-level global token pool** during training, promoting specialized expert behavior. To unleash the full potential of inherent heterogeneity, DiffMoE incorporates a **capacity predictor** that dynamically allocates computational resources based on noise levels and sample complexity. Through comprehensive evaluation, DiffMoE achieves state-of-the-art performance among diffusion transformers on ImageNet benchmark, substantially outperforming both dense architectures with 3x activated parameters and existing MoE approaches while maintaining 1x activated parameters. The effectiveness of our approach extends beyond class-conditional generation to more challenging tasks such as text-to-image generation, demonstrating its broad applicability across different diffusion model applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DiffMoE, a novel method for incorporating Mixture of Experts into the training of Diffusion Transformers. The key innovation is the creation of a batch-level global token pool, achieved by flattening all tokens within a batch. This allows experts to be trained simultaneously on a diverse set of tokens from multiple images and at various noise levels. The authors show that DiffMoE achieves significant improvements over dense baselines with fewer activated parameters. The paper presents a promising and well-motivated method for training scalable and efficient MoE-based diffusion models.

### Strengths
- A key strength is the observation that standard MoE formulations in diffusion models prevent routers from learning from the crucial contrastive patterns that exist across different noise levels and samples. DiffMoE is designed specifically to address this limitation.
- The paper employs a fair comparison framework, carefully maintaining a fixed computational cost between the proposed method and the baselines, which strengthens the validity of the performance claims.
- The paper is well-written, clearly articulating a complex architectural change and its motivations.
- The ablation studies are thorough and effectively demonstrate the contribution of each individual component of the DiffMoE framework.

### Weaknesses
- The qualitative results are limited, particularly the visual comparisons against the text-to-image (T2I) baselines. More examples would help in visually assessing the performance gains.
- There is a limited ablation study on the effect of batch size. Given that the core mechanism relies on a "batch-level global token pool," batch size appears to be a critical factor that could significantly impact DiffMoE's performance.

### Questions
- The central premise of DiffMoE's effectiveness is that the flattened batch of tokens provides a rich, global distribution for the experts to learn from. However, in many realistic inference or fine-tuning scenarios, users are constrained to very small batch sizes (e.g., 1-4). In such a situation, the flattened "global" pool would be a poor approximation of the true training distribution. What the computational cost and performance of DiffMoE are versus the dense baselines in this practical, low-batch-size setting?

### Soundness
4

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
The paper introduces DiffMoE, a novel Mixture-of-Experts (MoE) architecture that allows experts to access global token distributions via a batch-level global token pool during training, encouraging more specialized expert behavior. It further integrates a capacity predictor and dynamic thresholding mechanism to adaptively allocate computational resources based on input noise and sample complexity. Comprehensive experiments demonstrate the model’s effectiveness.

### Strengths
The paper introduces a novel batch-level global token pool that effectively captures the heterogeneity of diffusion processes, addressing a key limitation of prior MoE-based diffusion models. The proposed capacity predictor and dynamic thresholding mechanisms enable adaptive computation, balancing quality and efficiency. The evaluation is comprehensive, demonstrating the effectiveness of DiffMoE with significantly fewer active parameters.

### Weaknesses
The paper lacks an in-depth analysis of expert behavior, such as what patterns each expert learns or how tokens are distributed across experts. In addition, the batch-level global token pool may pose scalability challenges when extending to high-resolution images or video generation.

### Questions
1. Could the authors provide a more in-depth analysis of expert behavior, specifically how the batch-level global token pool contributes to expert specialization and growth?
2. How well does the capacity predictor generalize across different datasets or tasks? Does it require retraining for each new domain?
3. How does DiffMoE perform on high-resolution tasks, and are there any scalability limitations?

### Soundness
3

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
5

### Summary
This paper introduces DiffMoE, a novel Mixture-of-Experts (MoE) architecture designed for diffusion transformers. By leveraging a batch-level global token pool, a capacity predictor, and a dynamic threshold, DiffMoE enables efficient token selection and adaptive computation across varying noise levels and conditions. It achieves state-of-the-art performance in both class-conditional and text-to-image generation tasks, surpassing dense models with fewer activated parameters. DiffMoE demonstrates superior scaling efficiency and broader applicability, addressing limitations in prior MoE-based diffusion models.

### Strengths
+ The proposed DiffMoE is useful for  building scalebale diffusion model.
+ The proposed dynamic computation is interesting and reasonable

### Weaknesses
- The proposed method, DiffMoE, should be clearly highlighted in each table to enhance readability.

- Since the key advantage of MoE lies in its scalability, it would be beneficial to train models of varying sizes to further explore this aspect.

- Lack of a comprehensive literature review. For example, the use of MoE for diffusion transformers [1] and diffusion models with dynamic computation [2] should be discussed in the related work section.



[1] Diff-MoE: Diffusion Transformer with Time-Aware and Space-Adaptive Experts, ICML 2025.

[2] Dynamic diffusion transformer, ICLR 2025

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
