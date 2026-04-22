# Semantic Editing with Coupled Stochastic Differential Equations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Editing the content of an image with a pretrained text-to-image model remains challenging. Existing methods often distort fine details or introduce unintended artifacts. We propose using coupled stochastic differential equations (coupled SDEs) to guide the sampling process of any pre-trained generative model that can be sampled by solving an SDE, including diffusion and rectified flow models.
By driving both the source image and the edited image with the same correlated noise, our approach steers new samples toward the desired semantics while preserving visual similarity to the source. The method works out-of-the-box—without retraining or auxiliary networks—and achieves high prompt fidelity along with near-pixel-level consistency. These results position coupled SDEs as a simple yet powerful tool for controlled generative AI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Sync-SDE - a method that reuses a common noise trajectory to couple reverse-time and forward-time generative processes (synchronous coupling of Brownian motion), a previously known concept. The authors leverage this framework to enable semantic image editing using text-to-image stochastic flow models in a training-free manner. They provide extensive experiments to evaluate their method.

### Strengths
- Sync-SDE tends to localize changes during editing as highlighted with the pixel difference comparison (Figure 4).
- The authors provide extensive experiments (ablation studies) and comparisons.
- The authors provide mathematically elegant support.

### Weaknesses
Major:
- L409-L412 - “For each example and for each method compared, we select, among the three hyperparameter settings reported in the quantitative results, the image that is most similar to the source while still showing a meaningful edit, …” - this best-of-3 selection introduces potential bias in favor of the presented method. It would be interesting to see how the results look for each of the 3 hyperparameters on a small diverse set of images.
- L420-421 - “... Sync-SDE demonstrates strong capacity for global style transfer …” - In the whole paper (main and appendix) 4 style transfer results are provided. This claim is ambitious, and it seems that more results should be provided to demonstrate this claim (such as different color palettes). Moreover, in Figure 4, the editing into an oil painting - the change is very minor, and the face and hands look like they were changed, but the whole scene was not affected by the style change. Additionally, the fact that the woman’s pose was not preserved somewhat contradicts the authors claim that Sync-SDE is a structure preserving method. Another example could be seen in Figure 7 (the editing into a minecraft style) - the house and the trees were not preserved.

Minor:
- L82-83 - “... with a local cost” - quadratic is missing
- L221-222 - “... and the rectified SDE … an SDE formulation of rectified” - rectified flow is missing
- L392-393 - FlowEdit is an inversion-free method, as they construct a direct path between the source and the target distributions (as stated in their abstract).
- Figures 4 and 8 - a type of RF-Env instead of RF-Inv
- Table 1 - For FlowEdit, n_min appears twice (one instance should be replaced with n_max)
- L981 - Figure 11 should be referenced instead of D.3
- The dataset consists of 306 triplets, using 91 images - it is not clear which types of edits are included in the dataset, and their diversity. Could the authors provide more details regarding the dataset?
- L396 - DINO is presented as an evaluation metric but there is no figure for it. Could the authors provide (newer) semantic metrics, like DINOv2/v3 or CLIP Image or CLIP-Dir [1], or DreamSim [2]?
- Seed - as the method is stochastic, it depends on the noise realizations, as you had also discussed in the Appendix. Are all the results and comparisons (as well as for other stochastic competing methods) presented in the paper created with the same seed?
- L389-390 - Could the authors provide a small qualitative ablation on the effect of $t_0$?

### Questions
The method has a stochastic forward process, similarly to existing methods like Edit Friendly [3], and CycleDiffusion [4]. A discussion regarding similarity between the proposed method and other stochastic methods would be beneficial for the reader to identify the difference between them. This will also emphasize the paper's novelty.

Additionally, see weaknesses section.

[1] Gal, Rinon, et al. "Stylegan-nada: Clip-guided domain adaptation of image generators." ACM Transactions on Graphics.

[2] FU, Stephanie, et al. DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data. Advances in Neural Information Processing Systems.

[3] HUBERMAN-SPIEGELGLAS, Inbar; KULIKOV, Vladimir; MICHAELI, Tomer. An edit friendly ddpm noise space: Inversion and manipulations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[4] WU, Chen Henry; DE LA TORRE, Fernando. A latent space of stochastic diffusion models for zero-shot image editing and guidance. In: Proceedings of the IEEE/CVF International Conference on Computer Vision.

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
Prior work RF-Inversion derived the stochastic differential equation (SDE) for rectified flow models, which are the state-of-the-art text-to-image generative models. This enabled stochastic sampling using deterministic flow models. In this paper, the authors leverage this SDE to construct an alternative inversion and editing algorithm. The key idea is to couple the Brownian paths for the stochastic processes corresponding to reference and target images.
The authors also provide a concise optimal-transport interpretation of synchronous coupling via optimal bicausal Monge transports with a local cost.
Both quantitative and qualitative evaluations show that the proposed sync-SDE achieves stronger prompt alignment and better source image quality than existing methods. Anonymous source code is provided to support reproducibility.

### Strengths
1. The paper is well written, with a clear presentation of ideas and detailed derivations of key steps. The mathematical notation is consistent, making the paper easy to follow.  

2. Qualitative results are promising, particularly the localized edits in Figures 1 and 4, which are challenging for existing baselines yet handled precisely by the proposed method, Sync-SDE. Quantitative results further support the approach, showing improvements in both source image preservation and edit prompt alignment. Figure 2 demonstrates that Sync-SDE achieves consistently higher CLIP scores while maintaining lower LPIPS values than the compared baselines.  

3. The theoretical justification is also clean, offering a principled formulation of coupled reverse SDEs through a shared backward Brownian path. The optimal transport interpretation, though somewhat out of place, provides an interesting perspective through the lens of bicausal Monge transport.

### Weaknesses
1. **Line 141:** Inconsistent notation for marginal and random variable $X_t$.

2. The paper reads more like a tutorial, with the main contributions introduced relatively late (towards the end of page 5). The optimal transport interpretation in Section 6 feels somewhat out of place and could be better integrated or moved to the Appendix. The authors are encouraged to expand the presentation of the main contributions and the experimental results, which are currently deferred to the Appendices, to improve readability and highlight the paper’s core innovations.

3. **Line 386:** The ordering of baselines appears incorrect. In related works the authors note that RF-Inversion was the first to propose semantic editing with rectified flows, yet Line 386 lists FlowEdit before earlier methods, which is misleading. The authors are encouraged to reorder the baselines chronologically and ensure this ordering is consistent across the paper. 

4. **Line 125–127:** Before Rout et al., 2025 and Wang et al., 2025, RB-Modulation (https://arxiv.org/pdf/2405.17401, ICLR 2025) drew the connections between guided sampling in diffusion and stochastic optimal control. The authors are encouraged to include this earlier work to provide proper context and ensure a complete discussion of prior contributions in this area.

5. **Typo in Figure 4:** RF-Env → RF-Inv.

6. The font in Figure 4 labels is too small. 

7. The results for RF-Inv and SDEdit in Figure 4 appear to include noticeable noise artifacts in the edited images. While this is expected for a stochastic sampler like SDEdit, RF-Inv is typically a deterministic sampler for editing tasks. Could the authors clarify whether they used the stochastic or deterministic variant of RF-Inv in their experiments? If the stochastic sampler was used, please explain the reasoning, since the original method employed the deterministic version for editing and the stochastic version only for validating the SDE formulation.

8. It is well known that stochastic samplers often produce higher-quality samples than deterministic samplers under finer discretization or when using a large number of reverse steps (see Table 1 in https://arxiv.org/pdf/2010.02502). The authors are encouraged to either increase the number of denoising steps for SDE-based samplers or use their deterministic counterparts during sampling to ensure a fair and consistent comparison when drawing conclusions.

9. **Typo in Figure 8 in Appendix D.1:** RF-Env → RF-Inv.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ``sync-SDE``, a training-free and optimization-free framework for text-guided semantic image editing. The core idea is to leverage coupled stochastic differential equations (SDEs) to ensure high-fidelity edits that preserve the source image's structure. The method works by first determining the backward Brownian motion path required to invert the source image to noise, guided by the source prompt. This same path is then used to drive a new, reverse-time SDE sampling process, but guided by the target prompt. The authors claim that this "synchronous coupling" of the noise paths for the source and target processes forces the edited image to retain structural consistency with the original, while successfully incorporating the semantic changes from the target prompt. The method is presented as a universal plug-and-play module for any generative model based on SDEs, including diffusion and rectified flow models.

### Strengths
1. The core idea of reusing the inverted Brownian path for the new generation is intuitive and mathematically elegant. It provides a direct mechanism for enforcing structural consistency without complex loss functions or architectural changes.
2. Training-Free and Plug-and-Play: The method's ability to work "out-of-the-box" with pre-trained SDE-based models like Flux is a significant practical advantage.  This makes it highly accessible and efficient compared to methods requiring per-image or per-task fine-tuning.
3. The examples shown in Figures 1 and 3 are impressive, demonstrating precise semantic edits (e.g., "leaf → peas," "yellow and blue → red and purple") while maintaining near-pixel-perfect consistency in unedited regions. Meanwhile, the text rendering cases in paper seems the results of editing performs well.

### Weaknesses
1. The paper does not compare against SDEdrag[3] or EGSDE[2]. Both are highly relevant, recent, and SDE-based editing frameworks. EGSDE uses energy functions to guide the SDE process for translation tasks, while SDEdrag provides a theoretical and practical framework for why SDEs are well-suited for editing tasks like content dragging.    An evaluation without these competitors makes the performance claims of `sync-SDE` difficult to verify and positions the work in a vacuum.

2. While being training-free is a strength, the paper fails to quantify the trade-off. It is crucial to understand how `sync-SDE` (using the `Flux.dev` base model) compares against a fine-tuned model (e.g., `Flux Kontext` or other models from the Step-1 family).  A standardized benchmark like GEdit-Bench[1] would be the appropriate venue for such a comparison, providing a clear picture of the performance gap between this training-free approach and state-of-the-art trained solutions. Meanwhile, in present editing, the performance is more important than the effectiveness, therefore, if a training-based models largely outperform the training-free method, I will prefer to choose to train a editing model to use.

3. The core idea of "reusing randomness" from an inversion process to guide generation is fundamental to many editing methods, including DDIM inversion in the ODE context. The paper's novelty hinges on this specific "synchronous coupling" for SDEs. However, it needs to explicitly articulate why this is fundamentally different from and superior to other SDE manipulation techniques like those in EGSDE[2] and SDEdrag[3], which also aim to control the generative path to balance fidelity and faithfulness.

Reference:
[1]. https://huggingface.co/datasets/stepfun-ai/GEdit-Bench
[2].Zhao, Min, et al. "Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations." Advances in Neural Information Processing Systems 35 (2022): 3609-3623.
[3].Nie, Shen, et al. "The blessing of randomness: Sde beats ode in general diffusion-based image editing." arXiv preprint arXiv:2311.01410 (2023).

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new approach for content editing in text-to-image models without the need for training.  For this, they utilise coupled SDEs to guide the sampling process. The key idea is to drive the source and the edited image to the same correlated noise. After which the   noise is steered towards the desired target. The experiments show better results in terms of qualitative and quantitative metrics.

### Strengths
1. The motivation for the approach is clear, and the claims are justified by extensive proofs and evaluations.
2. The proposed approach seems to work with challenging cases like text-content editing while preserving the location of the edits. 
3. The proposed approach shows better streamlining of the desired areas of the image compared to previous approaches, hence improving the quality of editing.

### Weaknesses
1. For fair evaluations, could the authors provide the results on the Div2K data utilised by FlowEdit[1].
2. Although the proposed approach seems to be working better, the novelty seems to be incremental over Flowedit .
3. Although I agree that the approach is rooted in core mathematical rigor, the idea of utilising the noisy samples in reverse trajectories in the forward process has been explored in multiple approaches like Dragon Diffusion[2]. Could the authors clarify this? 
4. I believe the writing and presentation could be made better for audiences less familiar with SDEs and optimal transport. 

[1] Kulikov, Vladimir, et al. "Flowedit: Inversion-free text-based editing using pre-trained flow models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
[2] Mou, Chong, et al. "Dragondiffusion: Enabling drag-style manipulation on diffusion models." arXiv preprint arXiv:2307.02421 (2023).

### Questions
Please refer weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
