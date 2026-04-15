# Fill with Anything: High-Resolution and Prompt-Faithful Image Completion

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 3

## Abstract
Building on the achievements of text-to-image diffusion models, recent advancements in text-guided image inpainting have yielded remarkably realistic and visually compelling outcomes. 
Nevertheless, current text-to-image inpainting models leave substantial room for enhancement, particularly in addressing the often inadequate alignment of user prompts with the inpainted region, and in extending applicability to high-resolution images. 
To this end, this paper introduces an entirely $\textbf{training-free}$ approach that $\textbf{faithfully adheres to prompts}$ and seamlessly $\textbf{scale to high-resolution}$ image inpainting. 
To achieve this, we first present the Prompt-Aware Introverted Attention (PAIntA) layer, which enriches self-attention modules by incorporating prompt information derived from cross-attention scores, alleviating the visual context dominance in inpainting caused by all-to-all attention. 
Furthermore, we introduce the Reweighting Attention Score Guidance (RASG) mechanism, which directs cross-attention scores towards improved textual alignment while preserving the generation domain. 
In addition, to address inpainting at larger scales, we introduce a specialized super-resolution technique tailored for inpainting, enabling the completion of missing regions in images of up to 2K resolution. Experimental results demonstrate that our proposed method surpasses existing state-of-the-art approaches in both qualitative and quantitative measures, achieving a substantial generation accuracy improvement of $\textbf{61.4\%}$ compared to $\textbf{51.9\%}$. Our codes will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a training-free approach that faithfully guided the local image inpainting according to the textual prompts, which can be further scaled to high-resolution image inpainting. The authors proposed the Prompt-Aware Introverted Attention (PAIntA) layer to make self-attention modules focus more on text-related unmasked regions. Then, the Reweighting Attention Score Guidance (RASG) improved the textual alignment through cross-attention scores. Thus, this paper enjoys better performance with faithful generations.

### Strengths
1. Both PAINTA and RASG are convincing with clear motivations. 
2. The guidance from cross-attention scores of RASG is interesting. And RASG also enjoys good quantitative improvements. But it misses some important discussions about the related works.
3. Some inpainting results in the supplementary are amazing.

### Weaknesses
1. The main concern is the implementation of PAINTA, which rudely adjusts the attention scaling for self-attention according to the cross-attention score. Many visualized results in both the main paper and the supplementary are over-saturated, which might be caused by the adjusted self-attention scale of PAINTA.  Whether the over-saturated generation related to PAINTA or self-attention scaling?
2. Although RASG is highly inspired by Chefer et al. (2023), the authors do not provide further clarification about the difference. Moreover, the effect of some details is not well verified, such as the std-based normalization of RASG. More related works about RASG should be considered. 
3. Except for the over-saturated issue, the quantitative improvement of PAINTA is minor. Does this technique just work for some failure cases such as "zebra" in Fig13 of the supplementary? 
4. More in-depth discussions about PAINTA should be discussed. Such as some visualizations about $c_j$ in some representative cases.
4. The contribution of the high-resolution inpainting is minor, which should be considered as an implemented trick rather than a main contribution.

### Questions
1. It would be interesting if the authors could discuss more details about weaknesses2.
2. The qualitative result of "cat" in Fig.13 is a little confusing. Both "PAIntA only" and "RASG only" fail to generate the target in the masked region. But the combined one successfully synthesizes a cat on the sofa.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces modifications to enhance the text-guided inpainting performance of the stable diffusion inpainting model without additional training. The proposed enhancements involve adjustments to the attention scores within self-attention layers and the incorporation of classifier guidance through cross-attention scores. Additionally, refinements are made to the super-resolution model to improve image completion.

### Strengths
This paper demonstrates effectiveness when compared to the standard stable diffusion model, suggesting potential capacity improvement without the need for retraining.

### Weaknesses
This paper contains architectural design flaws and lacks experimental validation for each modification. Furthermore, it overlooks significant related research.

### Questions
1. The proposed PAIntA method claims to "mitigate the too strong influence of the known region over the unknown by adjusting the attention scores of known pixels contributing to the inpainted region". However, it remains unclear whether the $c_J\in (0,1)$ parameter effectively reduces this influence, especially considering that the element values of the similarity matrix $A_{self}$ are not always positive. Further discussion is needed to validate the effectiveness of the proposed PAIntA method.

2. The proposed RASG technique appears to bear a resemblance to the work by Chefer et al. (2023). The authors assert the superiority of RASG without presenting theoretical or empirical evidence. Some supporting evidence or comparisons are necessary to substantiate this claim.

3. Since RASG is a classifier guidance technique, it is essential to provide a comprehensive discussion of related works to distinguish RASG from other similar methods. This would help readers better understand its unique contributions and advantages.

4. It is not clear why RASG and PAIntA are applied at different resolutions. Specifically, the rationale for not using RASG in the $H/16\times H/16$ resolution should be explained.


5. Using Poisson blending needs clarification. How do you prevent the original image information from leaking into the inpainting region?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel training-free method for text-based image inpainting task, which is built upon pre-trained StableDiffusion inpainting and super-resolution models. The proposed method has two technical contributions: PromptAware Introverted Attention (PAIntA) and Reweighting Attention Score Guidance (RASG). The experiment metrics and visual results demonstrate the superiority of the proposed method over other state-of-the-art methods.

### Strengths
- This work aims to address the text-based image inpainting task, which is useful for practical applications. Prior methods suffer from unstable inpainting results that may lack text-image alignment and good quality/resolution. This work proposes a two-stage pipeline built upon pre-trained StableDiffusion inpainting and super-resolution models.

- The PromptAware Introverted Attention appears to be a good improvement compared to the original self-attention for the text-based inpainting task.

- The experiments are sufficient, including metrics like CLIP, Acc, and PickScore. The ablation studies also indicate the effectiveness of the proposed methods.

### Weaknesses
- The explanation of the RASG strategy in Sec. 3.4 is intuitive but not well-supported. It would be good to provide more analysis including the scaling design choices.

- The two drawbacks mentioned in the introduction, “Prompt neglect” and “Visual context dominance,” are actually one drawback, i.e., the inpainted result is similar to the surrounding background considering the visual context but ignoring the text prompt.

- It is mentioned that “EOT is included since (in contrast with SOT) it contains information about the prompt τ” and “beneficial to normalize the scores.” It would be good to provide ablation studies to support these design choices to make this work more complete.

- As there is no user study, it would be good to provide more visual comparisons in the supplementary materials.

- The paper should discuss how unnatural the inpainted region may look sometimes. Additionally, it should discuss its limitations.

### Questions
I would like to see more analysis on RASG, as well as discussions on unnatural inpainting results and limitations.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper aims at achieving prompt-faithful and high-resolution image inpainting. To better align with the text prompt, the authors replace self-attention layers with the proposed Prompt-Aware Introverted Attention layer and Reweighting Attention Score Guidance during sampling. Besides, the authors propose an inpainting-specific conditional super-resolution technique for high-resolution image inpainting. The authors show both quantitative and qualitative evaluations on MSCOCO and show some good results.

### Strengths
1. This paper is well-organized and easy to follow. The presentation is clear.
    
2. The authors show some techniques to improve text alignment without training for image inpainting and show some good results.

### Weaknesses
1\. The motivation is not convincing to me.  The authors claim that existing technique limitation comes from the visual context dominance over the prompt in self-attention. We suggest the authors visualize or make statistics to verify such a claim.

2\. The trick used for super-resolution (blending unmasked image) are not new to the community. Some existing works already leverage such technique for image inpainting [https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py).

3\. The authors should compare with SmartBrush and Imagen editor, which are also proposed to solve the text alignment problem.

4\. What’s the difference between the RASG and classifier guidance or self-guidance [1]?

5\. Some important details are missing to understand PAIntA and RASG fully. For example, what does $i$ mean in Eq(5)? Does it mean text token or image token? What’s the value of c_j in the final setting? What does the update operation in Figure 3(a) mean? We suggest the authors explain it by expressions. What does X0=\epsilon(I) mean in section 3.5? Is it a decoded latent image for the input image or an upscaled latent?

6\. In section 3.3 and Figure 3, the authors borrow query and key project layers from the next cross-attention module, and calculate the similarity with selected prompt tokens but still use the value from the image feature, which confuses me. Specifically, the authors consider the prompt as a query, they should also use them as value.

7\. The experiment results are not that convincing to me. Specifically,  in Figure 13 in the supplemental results, the authors show some failure cases by using only PAIntA and RASG. It is unclear which is more powerful and why they will fail. Besides, it is also not clear why combining them works better. It seems that combining PAIntA and RASG will also fail in some cases.

\### [1] Epstein D, Jabri A, Poole B, Efros AA, Holynski A. Diffusion self-guidance for controllable image generation.

### Questions
My major concerns come from the novelty of the proposal and the experiment results. Specifically, the authors should discuss and compare the proposed method with relevant techniques I mentioned in the Weaknesses. Besides, the authors show some failure cases caused by only using  PAIntA and RASG. It is unclear why and when they will work. Please find more details in the Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
