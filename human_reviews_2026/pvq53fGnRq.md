# Plug-in Image Quality Control for Posterior Diffusion Super-Resolution

- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Diffusion-based super-resolution (SR) has shown remarkable progress, mainly through prior-guided approaches that require explicit degradation models or semantic priors. While posterior diffusion SR avoids these assumptions by directly learning from LR–HR pairs, it still suffers from numerical errors during sampling and lacks plug-in mechanisms for quality control.	
	We introduce the first plug-in framework for posterior diffusion SR, enabling pretrained models to support controllable quality through marginal calibration, without retraining the core diffusion model. Our numerical analysis  reveals that discretization errors are a key bottleneck in posterior SR. We prove that these errors can be equivalently expressed as gradients of KL divergence, unifying numerical error correction with image based classifier guidance. This provides a principled explanation of fidelity degradation and a new lens for posterior diffusion trajectories. In principle, these errors can be corrected to improve fidelity when reference supervision is available, offering a new theoretical understanding of posterior diffusion trajectories. 	
	In real-world SR, we further show that our image based guidance offers a controllable trade-off between fidelity and perception, delivering perceptual sharpness competitive with state-of-the-art prior-based models.	
	Experiments confirm that our method consistently improves perceptual quality, while also validating the theoretical link between numerical errors and fidelity in posterior SR. These results position our work as a new direction for posterior diffusion models, bridging probabilistic analysis with practical deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The presented work extends posterior diffusion SR methods (namely, ResShift) by introducing a plug-in image quality control (IQC) mechanism. The proposed correction term is derived from a numerical analysis linking discretization error in the reverse diffusion ODE to the gradient of a KL divergence. This enables post-hoc "quality control" without retraining. Overall, the method provides theoretical interpretability and mild improvements in perceptual quality but limited quantitative gains over ResShift.

### Strengths
- Introduces a numerical-analysis perspective on diffusion SR, which is kind of interesting.
- Demonstrates that image quality can be controlled, although the shown example with HR reference is quite impractical (since HR is usually not given), but can be seen as upper bound (although HR itself is the upper bound).

### Weaknesses
- The core derivations rely on overly strong Gaussian (for a posterior learning approach) and constant-variance assumptions. I am a bit puzzled by this since ResShift does not need these assumptions and still performs comparable, sometimes better, questioning the goal of this work.
- Quantitative gains over ResShift are marginal (in general mostly mixed), suggesting the proposed correction has little real impact despite its theoretical framing...
- The paper suffers from numerous language and formal issues (e.g., spelling errors such as latant space, misuse of key words like "proof of definition", and inconsistent notations), which reduce readability and professionalism for this type of conference.
- Theorem 3.2.1 as a proof is not rigorous: it stitches pieces with unproven proportionalities and drops constants, so it should be framed as a modeling assumption backed by empirical calibration (i.e., A, B regression), not as a theorem.
-  Comparisons are limited to minor ablations (LR vs. HR guidance) without broader baselines or qualitative failure analysis to support the claimed controllability advantage.

### Questions
- In practice, variance is usually not constant spatially (different noise levels per region) nor temporally (it decays along t). In contrast, ResShift does not assume a fixed variance, nor does it linearize everything to one isotropic sigma^2. Therefore, it explicitly learns how the noise level and residual destribution behave given the LR input. Why did the authors try to simplify that? Did I miss something major here?
- In Theorem 3.2.1 (spatial derivative), the proof equates (by using the time derivative from proposition 3.2.1) temporal derivatives of the drift with spatial gradients of the KL divergence. What formal justification supports this operator equivalence?
- Given that the IQC model builds directly on ResShift with a simple correction term, how does this constitute a novelty?
- Since HR ground truth is unavailable at inference (in practice), what practical insight does "Ours (HR)" provide beyond an upper bound? Is the upper bound not already given by the HR image itself? It is a bit deceiving with the (high) reported numbers in combination with the words "ours", although it seems to be an upper bound...

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel plug-and-play framework for enhancing the performance of posterior diffusion models in image super-resolution (SR), avoiding the need for an explicit degradation model.
The primary contributions are threefold:
Theoretical Innovation: The authors provide a numerical analysis proving that the discretization error inherent in first-order ODE solvers can be equivalently expressed as the gradient of KL divergence. The main idea unifies numerical error correction with image-based classifier guidance theoretically.
Plug-and-Play Practical Design: Based on the theory, the method incorporates a correction term into the reverse sampling process of a pre-trained model without requiring any retraining. This is implemented via a dual-conditioning mechanism: one path for the noisy LR input and a separate, clean path for an additional guidance image.
Adaptation for Blind SR: They calibrate the guidance scale using a dataset of LR-HR pairs and then apply it using only the Low-Resolution (LR) image during inference (Algorithm 1 & 2), effectively trading off fidelity for enhanced perceptual quality.
Experiments on benchmarks like DIV2K and RealSR show that the method, when plugged into models like ResShift, achieves state-of-the-art performance in no-reference metrics (e.g., MANIQA, CLIPIQA, FID) and perception-based metrics (e.g., LPIPS), with minimal degradation in fidelity metrics (PSNR, SSIM).

### Strengths
1.	The paper establishes a principled, mathematical link between numerical integration error and probabilistic divergence. This provides a rigorous foundation for the proposed method and a new analytical tool for the community. The "plug-and-play" nature of the framework is practical for deployment.
2.	The paper validates quantitative comparisons against SOTA methods (regression-based, GAN-based, prior- and posterior-based diffusion models). The results demonstrate comparable performance with other SOTA methods.
3.	This work is interesting that it identifies the underexplored limitations of posterior diffusion SR (discretization errors, lack of control), making its contributions highly relevant.

### Weaknesses
1.	The calibration of the guidance scale relies on a set of LR-HR image pairs. This dependency on HR data for a "blind" SR method is a limitation, as the optimal scale might be dataset-dependent and may not generalize perfectly to all real-world degradations. The results on RealSR and DIV2K also reveal this weakness, the PSNR index is weak compare to ResShift with same inference step.
2.	The theoretical derivation relies on a constant variance assumption (Assumption 2) and linearizes the relationship between error and the KL gradient. The practical impact of these simplifying assumptions on complex, real-world image distributions is not fully quantified.
3.	Limited Comparison with Prior-based Guidance. The discussion focuses on posterior models. A more direct comparison or discussion on how this guidance principle relates to or differs from guidance mechanisms commonly used in prior-based models (e.g., using CLIP) would be better.

### Questions
Refer to the weaknesses. Robustness of Parameter Calibration:How sensitive is the method's performance to the specific dataset used for calibrating the guidance scale? Have the authors tested the generalization of a scale calibrated on one dataset (e.g., ImageNet) when applied to images from a different domain?

### Soundness
4

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
4

### Summary
This paper proposes a plug-in module for posterior diffusion super-resolution models to control image quality. The authors establish a theoretical link between the numerical discretization error in the sampling process and the gradient of the KL divergence. Based on this, they propose an image-based guidance method. In principle, this method can use a high-resolution (HR) reference to correct errors and improve fidelity. In practice, where no HR reference is available, the paper suggests flipping the sign of the guidance term (using the low-resolution image as a guide) to enhance perceptual quality without retraining the base model.

### Strengths
The main strength of this paper lies in its novel theoretical insight. Connecting the numerical discretization error of the ODE solver to a probabilistic concept like the KL divergence gradient is elegant and provides a new perspective on analyzing diffusion model trajectories. The idea of unifying numerical error correction with classifier guidance is intriguing. The experiments do show that the proposed method can improve perceptual metrics, which is a valuable goal for super-resolution.

### Weaknesses
While the theoretical premise is interesting, I have several major reservations about the work in its current form.

My primary concern is the conceptual leap between the paper's theory and its practical application. The theory elegantly justifies correcting numerical error to improve fidelity by moving the generation towards an HR target. However, the proposed blind SR method does the exact opposite: it pushes the generation away from the LR condition to ostensibly improve perception. This sign-flipping feels less like a principled application of the core theory and more like a clever but ad-hoc heuristic that happens to yield perceptually pleasing results. The paper lacks a convincing theoretical argument for why this maneuver should work consistently.

This leads to my second point regarding the "plug-and-play" and "no-training" claims, which I find to be overstated. The method's reliance on a calibration step (Algorithm 1) using a set of LR-HR pairs is a form of data-driven parameter tuning. This undermines the "no-training" narrative and raises questions about generalization. A single guidance scale A calibrated on one dataset may not be robust across diverse image domains.

Furthermore, the paper does not sufficiently address the inherent risks of its approach. Pushing the generation away from the LR input is a fundamentally unstable process that could easily introduce artifacts or hallucinate details inconsistent with the source. A thorough analysis of these failure modes is conspicuously absent. Finally, the empirical validation is quite narrow. Claiming a general framework for "posterior diffusion SR" while testing on only a single backbone model (ResShift) weakens the paper's claims of broad applicability.

### Questions
The sign-flipped guidance is the linchpin of your practical method, yet its theoretical grounding seems to be the weakest part of the paper. Could you elaborate on the justification for this? Specifically, what prevents this "push away from LR" from simply amplifying noise or creating implausible structures, rather than consistently enhancing perceptual quality?

The "plug-and-play" claim hinges on the robustness of the calibrated scale A. How does the method's performance hold up when A is calibrated on one domain (e.g., natural images) and tested on a completely different one (e.g., medical scans, cartoons)? This is crucial for understanding the method's true generalizability.

A complete picture of an algorithm's performance includes its limitations. Could you provide a qualitative analysis of failure cases? It would be particularly insightful to see instances where the perceptual enhancement introduces clear artifacts or generates textures that are inconsistent with the LR input, as this would help clarify the method's trade-offs.

I am open to reconsidering my evaluation should the authors provide convincing answers to these questions in their rebuttal.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a plug-and-play framework for posterior diffusion-based image super-resolution (SR), enabling controllable image quality without retraining. The authors reveal discretization errors correspond to KL divergence gradients, linking them to classifier guidance. Their method corrects fidelity degradation using second-order terms and supports both supervised fidelity enhancement and unsupervised perceptual improvement. This work bridges numerical analysis and perceptual conditioning, provides a principled explanation of fidelity degradation and a new lens for posterior diffusion trajectories.

### Strengths
1. This paper propose the first plug-and-play module for posterior diffusion-based super-resolution models, which allows pretrained models to perform image quality control without requiring retraining.
2. This paper prove that discretization errors in posterior diffusion can be interpreted as gradients of KL divergence, suggesting that such errors behave like KL gradients between continuous and discrete processes at each sampling step.
3. This paper reveals a theoretical connection between numerical error correction and classifier guidance, and propose an image-based classifier guidance formulation for posterior diffusion, unifying numerical analysis with guidance-based conditioning.
4. Experiments show strong perceptual quality and competitive performance with state-of-the-art models.

### Weaknesses
1. The article "Toward real-world single image super-resolution: A new bench mark and a new model" appears twice in the reference list. Similar mistakes exist for several other articles.

2. The most updated methods, SinSR and Osediff, are excluded from visual comparisons. It's not fair.

3. The visual comparisons are insufficient. There's only one scene for reference region and non-reference region, respectively.

### Questions
1. In Table 2, the proposed error correction plugin increases all non-reference metrics and most reference-based perceptual metrics but decreases reference-based fidelity measures, especially PSNR. This contradicts the paper's claim that error correction will improve fidelity. 

2. In Figure 2, the fidelity of SwinIR outperforms all compared methods. Do you think diffusion models with your proposed error correction technique can possess a similar fidelity as SwinIR in a reference region?

### Soundness
3

### Presentation
3

### Contribution
3
