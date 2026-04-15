# AP-LDM: Attentive and Progressive Latent Diffusion Model for Training-Free High-Resolution Image Generation

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Latent diffusion models (LDMs), such as Stable Diffusion, often experience significant structural distortions when directly generating high-resolution (HR) images that exceed their original training resolutions. A straightforward and cost-effective solution is to adapt pre-trained LDMs for HR image generation; however, existing methods often suffer from poor image quality and long inference time.

In this paper, we propose an Attentive and Progressive LDM (AP-LDM), a novel, training-free framework aimed at enhancing HR image quality while accelerating the generation process.

AP-LDM decomposes the denoising process of LDMs into two stages: (i) attentive training-resolution denoising, and (ii) progressive high-resolution denoising. The first stage generates a latent representation of a higher-quality training-resolution image through the proposed attentive guidance, which utilizes a novel parameter-free self-attention mechanism to enhance the structural consistency. The second stage progressively performs upsampling in pixel space, alleviating the severe artifacts caused by latent space upsampling.

Leveraging the effective initialization from the first stage enables denoising at higher resolutions with significantly fewer steps, enhancing overall efficiency. Extensive experimental results demonstrate that AP-LDM significantly outperforms state-of-the-art methods, delivering up to a 5x speedup in HR image generation, thereby highlighting its substantial advantages for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a training-free framework, Attentive and Progressive LDM (AP-LDM), for high-resolution image generation. Specifically, they introduce attentive training-resolution (TR) denoising and progressive high-resolution denoising. The model effectively enhances image detail by using the parameter-free self-attention mechanism (PFSA). The proposed approach achieves superior quality and generation speed compared to existing methods.

### Strengths
1. The authors provide a reasonable analysis of the limitations of existing training-free frameworks and address several shortcomings.
2. The proposed PFSA and progressive high-resolution denoising are both simple and effective, and the motivations provided are logical.
3. Extensive quantitative and qualitative comparisons are presented to demonstrate the effectiveness of the proposed method.
4. Compared to other training-free frameworks, the proposed method shows significant advantages in both image quality and generation speed.

### Weaknesses
1. The authors conduct experiments on only one model; it is recommended to try more baselines. Given that SDXL is a model with a large number of parameters (over 10B), it would help to further demonstrate the generalizability of the proposed method by testing on lighter models, such as Stable-Diffusion-v2.1 (1B).
2. There is no comparison of GPU memory usage. Considering that diffusing on the upsampled image requires more memory, this comparison is necessary. The authors should provide memory usage data for different methods across resolutions.
3. The authors do not perform ablation studies on the position of the AG module, such as at the input or output stage.
4. For visual comparison, it is suggested to enlarge the red boxes and provide a close-up for easier readability, as in the examples in Figures 5 and 8.

### Questions
1. How does the method perform on other baselines? Is it sufficiently generalizable?
2. What is the specific memory usage? Does it have any advantages over other methods?
3. Other questions are noted in the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces AP-LDM, a simple but effective training-free framework for enhancing the quality and speed of HR image generation. AP-LDM divides the denoising process of latent diffusion models into two stages: (1) attentive training-resolution denoising using a parameter-free self-attention mechanism to improve structural consistency, and (2) progressive high-resolution denoising through pixel-space upsampling to reduce artifacts. Experimental results demonstrate that AP-LDM outperforms state-of-the-art methods in both image quality and inference speed, achieving up to a X5 speedup.

### Strengths
From the experimental results, attentive attention is simple yet effective, as it enhances the details during generation. This simple mechanism also lays the foundation for the fast generation of 4K resolution images in the future.

### Weaknesses
Although attentive attention is simple and effective, its internal working mechanism is not clearly presented. I have the following questions regarding how it functions:

[1] The authors propose attentive attention and pixel-domain progressive sampling for high-resolution image generation. While the latter clearly cannot handle multi-object scenarios, as shown by the MultiDiffusion results in Fig. 2(b), it remains unclear why attentive attention can address this issue. For instance, demofusion avoids this problem by introducing additional mechanisms like Skip Residual and Dilated Sampling. Clarification is needed on how attentive attention resolves multi-object issues.

[2] Attentive attention seems to have broader applications beyond high-resolution generation, as it is a training-free mechanism. This suggests it could potentially be used to enhance the quality and concept generation in regular image synthesis tasks. Further exploration of this would be valuable.

### Questions
[1] The paper lacks comparisons with recent works on efficient high-resolution generation, such as HiDiffusion.

[2] Lacking comparison with the baseline, SDXL + BSRGAN.

I will consider raising the score once all concerns are addressed.

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
In this article, a new method called AP-LDM is introduced. This method is both user-friendly and efficient, and it can generate higher-quality high-resolution (HR) images while speeding up the generation process without the need for prior training. AP-LDM divides the noise removal process during image generation into two steps:
The first step is dedicated to training noise reduction for resolution enhancement, referred to as "meticulously trained resolution denoising." The purpose of this step is to utilize a new method called Attenive Guidance, which employs a new parameter-free self-attention technique, to make the generated image structures more consistent.
The second step is "progressive enhancement to high-resolution denoising." In this step, the image's pixel resolution is increased iteratively to eliminate potential image errors or blurriness that may arise from the resolution enhancement process.

### Strengths
1. The author demonstrates the effectiveness of the proposed method through comparative and ablation experiments.
2. Through experimental means, the author discovers issues with adoption in the latent space, and therefore proposes a novel upsampling method in pixel space that can avoid significant loss of texture details.

### Weaknesses
1. There is no explanation for why attention mechanisms can improve structural consistency in noisy spaces.

### Questions
1. Have you considered adjusting the guidance scale in a learnable way?
2. In line 207, the author points out that the early noise in denoising is non semantic, so is the noise semantic in other processes? Is it meaningful to perform attention calculation on noise?
3. The author mentioned in the contribution that it provides a speed of up to 5 times in HR image generation, but did not address why there is a speed improvement compared to other models. It is hoped that further explanation can be provided.

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
4

### Summary
This paper introduces a framework aimed at enhancing high-resolution image generation without the need for additional training. The AP-LDM model decomposes the denoising process into two stages: attentive training-resolution denoising and progressive high-resolution denoising. The first stage employs a parameter-free self-attention mechanism to improve structural consistency in latent representations, while the second stage progressively upsamples images in pixel space, mitigating artifacts typically introduced by latent space upsampling. Experimental results demonstrate that AP-LDM achieves up to a 5× speedup in generation time while maintaining high image quality. This approach not only accelerates the generation process but also enhances the overall efficiency of high-resolution image synthesis.

### Strengths
+ The paper introduces AP-LDM, a training-free method that enhances high-resolution image generation while speeding up the process and provides quantitative and qualitative comparison results with other works.
+ A simple attentive guidance module is proposed, which is developed upon a parameter-free self-attention mechanism that improves structural consistency in latent representations.
+ A progressive upsampling strategy is employed in pixel space, which reduces artifacts associated with latent space upsampling

### Weaknesses
-	The proposed designs are informed by empirical observations and necessitate attentive guidance and progressive upsampling for optimal hyperparameter selection in weighted feature fusion and temporal interference.
-	PFSA functions as self-modulated attention, enhancing features with strong responses. However, potential side effects of this design remain unaddressed in the limitations section.
-	In Section 4.4 and Figure 13, the method is referred to as "SwiftFusion." Consistency in terminology should be maintained throughout the paper.

### Questions
The paper employs progressive upsampling with either one or two sub-stages. It would be beneficial to present intermediate results from each sub-stage, as pixel-level decoded images are available at each step. Additionally, will content consistency be maintained as the number of sub-stages increases, for example, to three or more stages?

### Soundness
2

### Presentation
2

### Contribution
2
