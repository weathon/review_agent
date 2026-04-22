# Measuring  Distribution Shifts in Inverse Problems without clean data

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Diffusion models are widely used as priors in imaging inverse problems. However, their performance often degrades under distribution shifts between the training and test-time images. Existing methods for identifying and quantifying distribution shifts typically require access to clean test images, which are never available at test time when solving inverse problems. We propose a flexible framework for measuring distribution shift using  *only* corrupted test measurements and candidate diffusion model scores.  Our framework enables three complementary capabilities. First, in the general case with only a pool of diffusion models, it supports a principled model selection by identifying the model whose prior best matches the test data. Second, when an in-distribution model is available, our metric provides a theoretically guaranteed estimator of KL divergence that closely matches the image-domain KL. Third, the metric serves as a tool for adaptation guidance: aligning score functions with corrupted measurements reduces the estimated shift and improves reconstruction quality. Experiments on inpainting and MRI confirm that our method (i) achieves robust model selection, (ii) reliable estimates KL divergence in the presence of an in-distribution model, and (iii) enables effective adaptation to mitigate distribution shift.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The author deal with the important problem of measuring distribution shift using only measurements. They use their approach for model selection, estimation of the KL divergence and for fine-tuning / adaptation.

### Strengths
- The paper is well-written 
- The main result in Theorem 1 is interesting and also holds in the numerical experiments 
- I think in particular the experiments on the effect of the model selection and adaptation on downstream tasks (page 7, starting line 365) are interesting and important

### Weaknesses
See questions.



I have some minor issues with the formating: 
- Figure 1: It is hard to see the difference of the plot for "image domain" and "meas. domaine" without zooming in on the PDF. In a printed version it is even harder to tell apart. 
- Figure 1 is on page 3, but only discussed on page 6
- Figure 3 is discussed on page 6 (line 309), but is only displayed on page 8 (after Figure 6)
- Page 7, line 377 the subsection title "4.4 Ablation Studies" it at the end of the page, without any text below it 

Other minor points:
- In Assumption 2 you use $\hat{D}$, but $\hat{D}$ is only defined  later in Theorem 1
- page 7, line 348 missing space between "OOD denoiser" and $\hat{D}$

### Questions
- You write that your approach even works for JPEG compression, a setting not supported by the theory. What are inverse problems where your method would not work? Do you have some neccessary conditions on a problem?
- In Figure 2: Is this also the integrand of Eq. (4) and Eq. (9) up to $\sigma$?
- In Table 1 we see that MetFaces is closer to FFHQ than AFHQ. So, using your model selection criterion I would choose MetFaces. However, in Table 2 the image reconstruction results for MetFaces are worse than for AFHQ. Why? 
- What is the difference between your adaptation loss in Equation (10) and the objective in Ambient Diffusion [1] or [2]
- Why do you choose AFHQ in the adaptation in 4.3. and not MetFaces (MetFaces was closer to FFHQ according to Table 1)?

[1] Daras et al. "Ambient Diffusion: Learning Clean Distributions from Corrupted Data" (2023)

[2] Kawar et al. "GSURE-Based Diffusion Model Training with Corrupted Data" (2024)

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
3

### Summary
This paper introduces a method for detecting distribution shifts between measurements and priors to (1) select the best prior for regularization (2) mitigate distribution shift effects at inference time when using an OOD prior. The authors make an important point that often in inverse problem settings we do not have access to clean images to measure distribution shifts with the prior and thus there is a need for a unsupervised approach. Experimental results show good alignment between their method and image-based estimates for OOD detection. Additionally, the authors show that using an augmented loss to correct for out of distribution samples they can adapt OOD models with few samples and show improvement for said models on downstream recovery.

### Strengths
This paper solves an interesting problem that is important for deploying pre-trained generative models to image recovery in potentially new settings. As far as I know this is the first works to look into this for inverse problems in the self-supervised setting. This setting is crucial because adapting generative priors to new scientific data will likely require pre-trained models that havent been explicitly trained on the target distribution due to data scarcity. The paper has nice experiments showing the ability of their method to reliably detect OOD models from partial measurments. Additionally, the authors show encouraging results of using their technique to adapt their OOD models to the target distribution using only a limited number of samples which is really great to see.

### Weaknesses
The results are convincing for the most part, however, it would be nice to potentially see a few more experiments with more samples of the measurement data to get a better idea of how the method scales with more measurement data when adapting OOD models to the target distribution. Additionally, if it’s possible to compare their measurement only adaption approach to image-based adaptation approaches with with the same number of samples that would be a helpful baseline which can serve as the upper bound on performance for their measurement adaption otherwise its a bit more difficult to appreciate the performance gains they are getting.

### Questions
1. how does the method scale in performance with a higher # of measurement examples?
2. how does the method compare to image based adaption techniques with the same number of samples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an unsupervised framework to measure distribution shifts in inverse problems using only corrupted measurements, removing the need for clean test images. It reformulates the KLD between in-distribution and out-of-distribution data through diffusion model score functions evaluated in the measurement domain. The derived estimator links shift magnitude to denoiser residuals across noise levels, enabling model selection, divergence estimation, and adaptation that aligns out-of-distribution scores with measurement data. Experiments on image inpainting and magnetic resonance imaging show strong correspondence between image- and measurement-domain estimates and improved reconstruction after adaptation.

### Strengths
The paper presents an interesting and timely idea by proposing a framework to quantify distribution shifts in inverse problems without access to clean data. The formulation is conceptually clear, and the paper is generally well written and structured, with good alignment between theory and experiments. The method shows potential practical value by enabling model selection and adaptation using only measurement data. However, while these aspects are promising, the overall contribution remains somewhat limited in scope and especially experimental depth.

### Weaknesses
My main concern lies in the experimental setup, which appears overly simplified and somewhat artificial. The chosen tasks, such as low-resolution inpainting and small-scale fastMRI tests, do not capture the real challenges of distribution shift in medical imaging. The out-of-distribution settings, based on different anatomical regions or synthetic corruptions, are relatively easy to separate and may overstate the method’s performance. A more convincing validation would use realistic shifts, for example differences in scanner field strength (1.5 T versus 3 T MRI), acquisition protocols, or vendor-specific pipelines, where data differ in subtle but meaningful ways and clean reference images are not available. This would better demonstrate the method’s practical value and robustness.

Another weakness is the strong reliance on assumptions that rarely hold in practice, such as randomized measurement operators and independence between measurements and denoiser residuals. These conditions are violated in realistic setups like fixed MRI masks, making the theoretical guarantees and practical reliability of the method uncertain.

### Questions
1. The experiments appear simplified and artificial. How would the proposed framework perform under more realistic domain shifts, such as differences in scanner field strength (e.g., 1.5 T vs 3 T MRI), acquisition protocols, or vendor-specific reconstruction pipelines?

2. Given that the out-of-distribution scenarios used in the paper (different anatomies or synthetic corruptions) are relatively easy to distinguish, can the authors provide evidence that their metric remains reliable when the domain shift is subtle but clinically meaningful?

3. The theoretical analysis assumes randomized measurement operators whose span covers the full signal space. How critical is this assumption in practice, and what happens if one uses fixed or structured operators as in real MRI or CT acquisition?

4. Since the proposed metric depends on expectations over many random operators, how feasible is this in realistic imaging pipelines where only a single, fixed acquisition model is available?

### Soundness
2

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
The author proposes a method to measure distributional shift in inverse problems at test-time.

### Strengths
1. Use diffusion model to detect OOD problems is interesting.
2. The mathematics in this paper looks correct to me.

### Weaknesses
1. This paper should also compare with domain/test-time adaptation methods for inverse problems. such as [1], [2]. 
2. There are literatures on using diffusion models to detect OOD samples, e.g. perturbing intermediate noise or so on. Authors should mention them.[3]
3. The novelty of this work is lacking since OOD detection with diffusion model is well known, and adapting a pretrained model to OOD inverse problem is also well-known. The authors contribution in this field is obscure.


[1] Deep Diffusion Image Prior for Efficient OOD Adaptation in 3D Inverse Problems

[2] Patch-based diffusion models beat whole-image models for mismatched distribution inverse problems

[3] Denoising diffusion models for out-of-distribution detection

### Questions
Instead of selecting the best pretrained model, is there a way to improve the pretrained model for inverse problem solving given your OOD detection information?

### Soundness
2

### Presentation
2

### Contribution
2
