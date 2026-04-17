# Guidance Watermarking for Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
This paper introduces a novel watermarking method for diffusion models. It is based on guiding the diffusion process using the gradient computed from any off-the-shelf watermark decoder. The gradient is guided further using different image augmentations, increasing robustness to attacks against which the decoder was not originally robust, without retraining or fine-tuning. The methodology effectively allows to convert any post-hoc watermarking scheme into a scheme embedding the signal during the diffusion process. We show that this approach is complementary to watermarking techniques modifying the variational autoencoder at the end of the diffusion process. We validate the methods on different diffusion models and detectors. The watermarking guidance does not significantly alter the generated image for a given seed and prompt, preserving both the diversity and quality of generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a guidance module that could be applied during the diffusion process to embed a watermark. It relies on a pretrained watermark decoder/detector (such as Stable Signature) to compute the loss that is needed for the guidance. The main idea is to add a gradient term to the noise during each diffusion step and the gradient term is computed with respect to the watermark. It gives experimental results to show the effectiveness of this method.

### Strengths
+ The guidance module could strengthen existing pretrained baseline watermarking method such as Stable Signature and VideoSeal, as shown in their experiments in Table 1. More importantly, it extends baselines to be more robust agains unknown augmentations.

+ Experimental results on various diffusion models are provided: SD2, Flux, and Sana.

### Weaknesses
- With all those formulas floating around, it is challenging to grasp the whole picture of this work and the additional contribution they bring to existing watermarking pipeline. (See my suggestion below)

- Choice of the guidance strength hyperparameter adds complexity to the watermarking application.

- It is not completely fair to claim that your work "does not necessitate any retraining of the diffusion model" and compare against TreeRing and GaussianShading (both trainning-free), when your method actually relies on a well-trained watermark detector (e.g., Stable Signature).

- The robustness results are not consistently better than GaussianShading in Table 2.

### Questions
- I am confused about the Motivations section. If your motivation is that the spectral signature could be exploited to remove the watermark, why not address this issue in your evaluation?

- Explain more on why the right pattern is better than the left in Figure 1.

- A system view (probably a system/pipeline figure) is necessary to understand their method and the additional module/steps they bring into the watermark embedding/detection pipeline, e.g., compared to the baseline Stable Signature.

- It is not very clear how detection/decoding is performed. Maybe (7) gives some info on detection, but what about decoding? How do we get the watermark message from an image?

- In Table 1, is any image attack used in the evaluation for robustness? If so, what are they? If not, you should add robustness comparison with SSign and VS in the Table 2.

- The basic image distortions in Table 2 seem not sufficient, more to add: e.g., gaussian blur, rotation.

- Confused about your claim: "ours does not require extra steps for decoding making it up to 50 times faster at detection
time." I think this is also because the decoding procedure is not clearly explained in the paper. For GuassinShading/TreeRings, we need to convert an image to latent noise for detection. Are u claiming that your method does not require such conversion? If so, how do you directly extract watermark from an image?

### Soundness
3

### Presentation
2

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
This paper proposes a new image watermarking method that enables converting an existing post-processing watermarking method into an in-generation watermarking method using only an existing decoder, without introducing significant changes to the watermarked image.

### Strengths
1.	Compared to existing post-generation and in-generation watermarking methods, the proposed approach offers improvements in watermark bit capacity.
2.	From the methodological point of view, the paper provides a practical means of converting post-processing watermarking approaches into in-generation ones.

### Weaknesses
1.	It’s unclear what motivates the authors to propose this method. The writing of the introduction should be revised to explicitly show this.
2.	It is unclear whether the method performs semantic watermarking or adds fixed content-unaware perturbations, as similar in-generation watermarking methods like Tree-Ring and Gaussian Shading have been found out to be adding fixed perturbations [1]. Since the proposed method is also an in-generation watermarking method, the authors should use a fixed watermarking key and test under [1]’s removal method to show whether or not the proposed watermark is content-aware.

[1] Yang et al., “Can simple averaging defeat modern watermarks,” in NeurIPS 2024.

### Questions
1.	For bold results presented in Table 1, would it be possible to also report the corresponding watermark decoding bit accuracy values? This could align the metric with most existing works and facilitate comparison.

### Soundness
2

### Presentation
2

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
This study introduces a technique utilizing guidance to incorporate watermarks into the diffusion-based image creation workflow. By sourcing the guidance from an existing watermark decoder, it adapts after-the-fact watermarking approaches into on-the-fly methods that work seamlessly with various guided generation systems. The evaluations assert that it achieves durable watermarking while preserving the visual fidelity and meaning of the images.

### Strengths
1. Employing guidance strategies to enable in-generation watermark integration within diffusion frameworks opens up a compelling avenue for exploration. 
2. The system's capacity to merge established after-the-fact watermarking methods with guided  generation processes offers significant utility in application. 
3. Overall, the manuscript is structured effectively and easy to follow.

### Weaknesses
1.	For embedding watermarks in individual images, the approach imposes an excessively high computational burden. Conducting complete denoising operations to derive guidance at every noise level far surpasses the time required for standard generation (as evidenced in Appendix E). This greatly hampers its applicability to extensive image production scenarios.
2.	The tuning complexity prevents the method from being a true 'plug-and-play' solution. As illustrated in Appendix B, Figure 6, setting the guidance strength ($\omega$) too high can introduce severe visual artifacts into the generated image, necessitating careful calibration for different models and decoders.

### Questions
1.	The proposed method relies on a public, off-the-shelf decoder to compute gradients. Does this imply that an attacker, upon obtaining a watermarked image, could actively erase the watermark by computing and applying an 'adversarial' or 'reverse' guidance gradient aimed at minimizing the decoder's output (i.e., a white-box adaptive attack)?
2.	Regarding the 'fast guidance' experiments in Section 5.5, were the results (e.g., "G-VS last 15") obtained by applying *both* the 'step reduction' (simplification 1) and the 'gradient approximation' ($\nabla_{z_{0}}$ instead of $\nabla_{z_{t}}$, simplification 2) strategies simultaneously? If so, what is the respective contribution of each simplification strategy to the observed performance degradation?

### Soundness
3

### Presentation
3

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
The paper introduces a method that embeds digital watermarks directly during the diffusion sampling process. The approach leverages gradients from pretrained watermark decoder to guide the diffusion model toward generating images that the decoder recognizes as watermarked. It embeds the watermark during generation using a cosine-based loss that aligns decoder features with a target watermark vector, and aggregates gradients across image augmentations to enhance robustness to unseen attacks. No need to retrain the decoder.

### Strengths
1. The method is general and retraining-free. It can transform any existing post-hoc watermark decoder into an in-generation embedding scheme without retraining either the decoder or the diffusion model.
2. The method is empirically validated on multiple diffusion models and decoders.
3. The method has a unified formulation for multi-bit and zero-bit watermarking.

### Weaknesses
1. The approach requires a pretrained differentiable watermark decoder (e.g. stable signature). Its performance and the statical guarantees are bound by the robustness and isotropy of that decoder's feature space. If the decoder is biased or unavailable, the framework can't work.
2. The attack scope is narrow. The experiments focus on standard image augmentations (JPEG, crop, brightness). Other relevant attacks like regeneration attacks, adversarial attacks should also be tested.
3. No ablation on guidance scheduling. The method mentions turning on guidance only for the last diffusion steps (section 4.5). But there is little analysis on how the choice of step $T_w$, clipping thresholds, or gradient aggregation method affects detectability or quality.

### Questions
1. All experiments are conducted on latent diffusion model. I'm wondering its effectiveness  on pixel-based diffusion models.

### Soundness
3

### Presentation
3

### Contribution
2
