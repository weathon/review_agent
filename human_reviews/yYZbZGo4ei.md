# Accelerating Diffusion Transformers with Token-wise Feature Caching

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion transformers have shown significant effectiveness in both image and video synthesis at the expense of huge computation costs. To address this problem, feature caching methods have been introduced to accelerate diffusion transformers by caching the features in previous timesteps and reusing them in the following timesteps. However, previous caching methods ignore that different tokens exhibit different sensitivities to feature caching, and feature caching on some tokens may lead to 10$\times$ more destruction to the overall generation quality compared with other tokens. In this paper, we introduce token-wise feature caching, allowing us to adaptively select the most suitable tokens for caching, and further enable us to apply different caching ratios to neural layers in different types and depths. Extensive experiments on PixArt-alpha, OpenSora, and DiT demonstrate our effectiveness in both image and video generation with no requirements for training. For instance, 2.36$\times$ and 1.93$\times$ acceleration are achieved on OpenSora and PixArt-$\alpha$ with almost no drop in generation quality. Codes have been released in the supplementary material and Github.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors aim to accelerate DiT inference by caching features and reusing them in future steps. The authors observe the limitations of existing methods that ignore the sensitivities to feature caching among different tokens, thus leading to potential degradation of generation quality. Based on this observation, the main idea of this work is to selectively cache suitable tokens and adjust the caching ratios by considering layer types and depths. Evaluations based on three typical DiT models show the performance of model accuracy and overhead saving.

### Strengths
1.	The author provides two insightful observations that improve my understanding of diffusion transformers. Firstly, different tokens exhibit varying levels of temporal redundancy across different timesteps. Secondly, the authors highlight that due to error propagation in transformer architectures, the same caching error can lead to vastly different impacts on the final generation result. These observations are the basis of the proposed token-wise feature caching method.
2.	Unlike previous works that primarily focused on U-Net architectures, the authors shift their attention to the transformer architecture. This is meaningful as existing methods have not fully exploited the unique properties of transformers, which are increasingly popular for visual generation tasks.
3.	The experimental results demonstrate the performance improvements with high inference speedups.
4.	The authors select the most suitable tokens for caching by leveraging both temporal redundancy and error propagation. The proposed method adaptively caches the tokens that minimize the resulting error while maximizing the acceleration ratio.

### Weaknesses
1.	In Figure 1, the authors use the Frobenius distance to measure the temporal redundancy of tokens across timesteps. However, it is unclear how this distance is exactly calculated. Providing more details on the calculation process would help readers better understand the experiments.
2.	I cannot find a clear formulation for the cache score used in token selection. The authors mention the cache score in Figure 4, but no explicit equation is provided. 
3.	The authors update the cache at all timesteps to reduce the error introduced by feature reusing. However, it is unclear how this procedure affects the computation complexity, especially about I/O overhead and cache selection efficiency.
4.	The experiments report the inference speedups but lack a comprehensive analysis of the trade-off between speedups and model accuracy. While FID and CLIP scores are provided in Table 1, these metrics do not fully reflect model accuracy. In addition, Table 3 shows that the proposed method has no significant improvements over existing methods in terms of accuracy and speedups.

### Questions
Please refer to the questions in the weakness part.

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
4

### Summary
This paper presents a method called ToCa, which utilizes token-wise feature caching to accelerate the denoising process of DiT. Unlike existing DiT acceleration methods, ToCa offers a fine-grained, token-wise acceleration approach, demonstrating improved performance compared to previous methods.

### Strengths
- The paper is well-written and easy to understand. 
- The observation that the similarity and propagated error differ for each token in DiT is very insightful.
- Various metrics were proposed to measure the importance of tokens on reusing. 
- Experiments were conducted on a variety of datasets, including not only text-to-image but also text-to-video.

### Weaknesses
- It's difficult to understand how token-wise caching leads to actual acceleration. Since the attention layer calculates the similarity among all inputs through matrix multiplication, even if one output token is not computed, there won't be a significant difference in the overall computational cost (similar to unstructured pruning). A more detailed explanation of where this acceleration comes from or breakdown is required. 
- Evaluation is performed with just single Latency/FID point. An evaluation of the Pareto curve with latency/FID would provide more insightful information. 
- There seems to be a color error in Fig 3(b). The yellow arrow indicating low cache ratio is missing.

### Questions
Same with my Weakness 1.

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
Token-wise Caching (ToCa) is introduced as a fine-grained feature caching strategy designed to accelerate diffusion transformers. It uniquely considers error propagation in caching, utilizing four selection scores to determine the best tokens for caching without incurring extra computational costs. ToCa also allows for variable caching ratios across different layers and integrates multiple techniques to enhance caching efficiency. Extensive experiments on models like PixArt-α, OpenSora, and DiT show that ToCa achieves significant speedups while preserving nearly lossless generation quality.

### Strengths
1. The motivation behind the proposed approach is both technically sound and clearly explained, supported by two informative figures illustrating the varying levels of similarity among different tokens and the accumulation of error across these tokens. These visual aids effectively convey the rationale for the proposed method.

2. The methodology for selecting scores and making decisions for each layer is novel, offering valuable insights into the acceleration of transformer models. This innovative approach not only enhances the efficiency of the models but also contributes to a deeper understanding of feature caching strategies within the context of transformers.

### Weaknesses
1. There appears to be a lack of a complete algorithmic description within the manuscript. Specifically, a detailed explanation is needed for the token selection process at each layer and each timestep, as well as the procedure for redistributing the cached tokens back into the overall framework. This omission could lead to misunderstandings regarding the efficacy and functionality of the proposed method.

2. The manuscript does not adequately illustrate the computation costs associated with the proposed approach, making it challenging to comprehend the additional expenses involved. A clearer breakdown of these costs would enhance the reader's understanding of the method's efficiency.

3. Including a token diagram for specific layers during both early and late timesteps, across various layer types, may provide additional clarity. Such visual representations could effectively illustrate how tokens are managed and utilized within the different stages of the computation, facilitating a better grasp of the overall caching strategy.

### Questions
1. The introductory paragraph of the methods section should be rephrased to clarify the concept of the "naive scheme" for feature caching. This naive scheme is actually no different from existing approaches, such as DeepCache, FORA, and delta-DiT, which should not be "we propose". 

2. It should be noted that Figure 3(b) does not depict any low-ratio caching scenarios. 

3. There is a discrepancy regarding the application of the scores s1, s2, and s3 across the layers. Specifically, s1 is applicable only to self-attention layers, while s2 pertains solely to cross-attention layers. However, the current definition combines these scores, which raises questions about how this aggregation functions across the different types of layers.

4. Clarification is needed regarding the additional time cost associated with computing the selection scores. Given that the scoring relies on rank computation, there remains a need to calculate and distribute the cached tokens among the uncached tokens. This process is somewhat ambiguous, particularly concerning how the acceleration time is achieved. Presenting the actual extra costs incurred for selecting scores alongside the resulting speedup from the caching process would provide valuable insights.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces ToCa (Token-wise feature Caching), a training-free feature caching method tailored to accelerate diffusion transformers. ToCa allows for adaptive token selection, which aims to cache tokens with minimal caching error based on various criterions. Experiments have been conducted on PixArt-α, OpenSora, and DiT models to show the speedup and quality trade-off of ToCa.

### Strengths
- The visualizations of temporal redundancy and error propagation illustrate the motivation of ToCa in an intuitive way, making the design choices relatively well motivated.
- The four token selection scoring functions and layer-specific cache ratios are natural design choices that enable the caching strategy to be more fine-grained
- ToCa achieves more than 2x acceleration ratios on both text-to-image and text-to-video tasks, while having better quality than baselines.
- ToCa’s training-free nature makes it more practical for real-world application.

### Weaknesses
- While ToCa achieves reasonable benchmark results, some artifacts remain in the generated images compared to the originals. For instance, in Figure 6, the moon is missing from the "wolf howling at the full moon" prompt, and the background forests appear blurred in the "tranquil beach" prompt. It is necessary to demonstrate how ToCa performs with high-resolution images (1024x1024) generated by more advanced models like FLUX.1-dev [1].
- Another important direction in accelerating diffusion models involves reducing the required sampling steps. However, this might compromise ToCa's effectiveness since fewer steps may result in greater feature variations between steps, potentially making them less suitable for caching. Please include experimental results using models with reduced sampling steps, such as FLUX.1-schnell [2] with 4 steps.
- For advanced model evaluation, FID and CLIP scores may not adequately reflect human preferences. Please provide quantitative results using more recent metrics, such as image reward [3].
- The method involves numerous hyperparameters, including four token selection scoring metrics ($\lambda_1$, $\lambda_2$, $\lambda_3$, $\lambda_4$) and caching ratios for different layers and timesteps ($\lambda_l$, $\lambda_{type}$, $\lambda_t$). While A.3 provides a sensitivity analysis, it remains unclear how these parameters should be determined for new models or tasks. The authors should elaborate on hyperparameter selection and impacts, particularly addressing whether an automatic selection method (e.g., a calibration procedure) exists. The complexity of hyperparameters could limit the method's generalizability and practical application.
- The "Acceleration of Diffusion Models" section lacks references to key fundamental works. Specifically, step distillation [4, 5] and consistency models [6] should be cited under "reducing the number of sampling timesteps," and Q-Diffusion [7] should be cited under "weight quantization."
- Notation inconsistencies: s3 is generally defined as cache frequency but appears as spatial distribution in Table 4, which uses s4 for cache frequency.

[1] https://huggingface.co/black-forest-labs/FLUX.1-dev  
[2] https://huggingface.co/black-forest-labs/FLUX.1-schnell   
[3] Xu et al. ImageReward: Learning and Evaluating Human Preferences for Text-to-image Generation. NeurIPS 2023.  
[4] Salimans et al. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022.  
[5] Meng et al. On Distillation of Guided Diffusion Models. CVPR 2023.  
[6] Song et al. Consistency Models. ICML 2023.  
[7] Li et al. Q-Diffusion: Quantizing Diffusion Models. ICCV 2023.

### Questions
- In table 2, ToCa does not outperform PAB with 1.24x speedup (78.34 vs 78.51). Although ToCA achieves a more significant speed-up, it is crucial to preserve the generation quality. I am wondering if ToCa can have better generation quality using a more conservative setting that has similar speed-up to PAB?

### Soundness
3

### Presentation
3

### Contribution
2
