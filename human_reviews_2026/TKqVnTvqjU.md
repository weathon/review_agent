# AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent studies have extended diffusion-based instruction-driven 2D image editing pipelines to 3D Gaussian Splatting (3DGS), enabling faithful manipulation of 3DGS assets and greatly advancing 3DGS content creation. However, it also exposes these assets to serious risks of unauthorized editing and malicious tampering.  Although imperceptible adversarial perturbations against diffusion models have proven effective for protecting 2D images, applying them to 3DGS encounters two major challenges: view-generalizable protection and balancing invisibility with protection capability. In this work, we propose the first editing safeguard for 3DGS, termed AdLift, which prevents instruction-driven editing across arbitrary views and dimensions by lifting strictly bounded 2D adversarial perturbations into 3D Gaussian-represented safeguard. To ensure both adversarial perturbations effectiveness and invisibility, these safeguard Gaussians are progressively optimized across training views using a tailored Lifted PGD, which first conducts gradient truncation during back-propagation from the editing model at the rendered image and applies projected gradients to strictly constrain the image-level perturbation. Then, the resulting perturbation is backpropagated to the safeguard Gaussian parameters via an image-to-Gaussian fitting operation. We alternate between gradient truncation and image-to-Gaussian fitting, yielding consistent adversarial-based protection performance across different viewpoints and generalizes to novel views. Empirically, qualitative and quantitative results demonstrate that AdLift effectively protects against state-of-the-art instruction-driven 2D image and 3DGS editing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first approach to safeguarding 3D Gaussian Splatting (3DGS) assets against malicious edits, including both 2D image editing and local/global 3D manipulations, by lifting adversarial perturbations into the 3D space. The core challenges lie in maintaining protection effectiveness across multiple views and balancing imperceptibility with robustness. AdLift addresses these issues via a Lifted PGD strategy that avoids direct gradient flow from the target model to the 3D Gaussians; instead, it truncates gradients at the 2D rendered image and lifts them back to the 3D representation.

### Strengths
1. The first work to extend adversarial image protection from 2D image editing to 3DGS representations.
2. Comprehensive evaluation spanning both 2D image synthesis attacks and local/global 3DGS editing scenarios.
3. Clearly written and well-organized, making the technical contributions easy to follow.

### Weaknesses
1. While the authors argue that soft-constrained baselines (e.g., GuardSplat) or direct optimization with hard constraints on attributes are fundamentally limited, the paper lacks sufficient experimental evidence to support this claim. Although adversarial loss and CLIP score trends are reported under varying PSNR values, there is a lack of concrete editing results to validate whether these numerical differences meaningfully translate to protection effectiveness.
2. AdLift is designed to generate view-consistent adversarial perturbations across multiple viewpoints. However, the paper provides only qualitative evidence (e.g., visual comparisons) to support this claim, with no quantitative metrics or consistency measurements to validate the view-generalizability of perturbations.
3. The rationale behind duplicating the original Gaussians and optimizing the copies is not clearly justified. This design choice may raise concerns in terms of scalability and efficiency, especially in complex scenes.

Note: Weaknesses 1-3 correspond directly to Questions 1-3.

### Questions
1. While the paper claims that soft-constrained baselines (e.g., GuardSplat) and direct optimization with hard constraints on attributes are less effective, the experimental evidence supporting this claim remains limited. It would be helpful to see qualitative editing results comparing AdLift and GuardSplat at the same PSNR levels. 
For example, in Figure 10, GuardSplat achieves a higher CLIP score at the same PSNR, but it is unclear whether this corresponds to weaker editing resistance in practice. Similarly, it is unclear whether the perturbed images in Figure 11 are matched in noise level. Including additional results where GuardSplat or hard-constraint baselines are evaluated under similar perceptual distortion (e.g., PSNR) with AdLift would strengthen the argument. 
Lastly, a brief explanation for why the adversarial loss does not decrease under these soft-constrained setups—whether due to gradient misalignment, saturation, or other structural factors—would help clarify the underlying limitation.
2. Can the authors provide quantitative evidence of view consistency? For instance, how does AdLift compare to prior watermarking approaches like GaussianMarker or GuardSplat in terms of perturbation alignment across views?
3. Does duplicating the Gaussians lead to significant memory overhead, particularly for dense scenes where the number of primitives is already high? A discussion or analysis of the scalability implications would be helpful.
4. Can the authors report additional evaluation metrics such as FID or Precision, in addition to CLIP score? Prior 2D image protection works like PhotoGuard[1] and AdvDM[2] typically report FID and Precision, which are widely adopted and allow for better comparability. It would also be helpful to provide these metrics for the editing results of the baseline methods discussed in Q1.
5. In Figure 8, the qualitative results of local 3DGS editing (e.g., the bear scene) show only marginal visual differences between the unprotected and protected assets. In contrast, Table 2 reports that the proposed AdLift-ST achieves substantially stronger protection performance (e.g., lower CLIPd) compared with Instruct-GS2GS and DGE (Global). How does AdLift-ST yield such strong quantitative gains despite the visual outputs in Figure 8 appearing relatively similar on the bear scene? Additional explanation would help clarify the effectiveness of local protection.


[1] Salman, Hadi, et al. "Raising the cost of malicious ai-powered image editing." arXiv preprint arXiv:2302.06588 (2023).

[2] Liang, Chumeng, et al. "Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples." arXiv preprint arXiv:2302.04578 (2023).

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an adversarial safeguard for 3D Gaussian Splatting that protects against unauthorized instruction-driven editing. It lifts bounded 2D adversarial perturbations into 3D as safeguard Gaussians, optimized via a Lifted Projected Gradient Descent that alternates between gradient truncation and image-to-Gaussian fitting. This design ensures view-generalizable and imperceptible protection across both training and novel viewpoints.

### Strengths
- The paper addresses an interesting and practical problem in anti-editing for 3D content protection.

- The proposed method is technically sound and reasonable.

- Experimental results across multiple scenes demonstrate performance improvements.

### Weaknesses
- The technical innovation appears limited. The proposed approach essentially adds adversarial perturbations to 2D images and then reconstructs them via Gaussian Splatting. While effective, it largely adopts conventional 2D anti-editing optimization with an additional 3DGS reconstruction step, which reduces its novelty and technical contribution to the community.

- The experimental evaluation is limited to only four scenes, which may be insufficient to robustly assess the generality and effectiveness of the method.

- The paper lacks analysis on transferability — it remains unclear how well the optimized perturbations perform against other editing networks beyond the one used during training.

- The robustness of the perturbations to purification or denoising processes is not discussed, leaving uncertainty about their effectiveness under potential real-world defenses.

### Questions
Please refer to the Weakness part above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AdLift, a framework that lifts bounded 2D adversarial perturbations into 3D Gaussian Splatting (3DGS) representations to defend against unauthorized instruction-driven editing. The method introduces a tailored Lifted PGD that alternates between gradient truncation and image-to-Gaussian fitting, aiming to balance invisibility and protection strength. Experiments across multiple 3DGS scenes demonstrate improved protection against both 2D and 3D instruction-based editing, while maintaining reasonable visual fidelity

### Strengths
1. The paper highlights an important and timely issue, security risks of instruction-driven 3DGS editing, making it a potentially impactful line of research.

2. The proposed Lifted PGD framework is presented with mathematical detail, algorithmic steps, and visual diagrams, which aids clarity.

3. Evidence of feasibility: The experiments, though limited, demonstrate that adversarial perturbations can be adapted from 2D to 3DGS to some extent, offering preliminary validation.

### Weaknesses
1. The core idea is essentially extending 2D PGD adversarial training into 3D Gaussian Splatting, with rendering-space constraints for imperceptibility. This is more of an adaptation than a fundamentally new paradigm. The paper repeatedly claims to be the first to propose active protection for 3DGS, but prior work on watermarking, Gaussian perturbations, and 2D adversarial protection already covers similar ground. The novelty is overstated. Theoretical contribution is thin: the method description relies heavily on equations and algorithmic flowcharts, without rigorous analysis or formal guarantees of convergence.

2. The method assumes white-box access to editing models, which is unrealistic in practice. No experiments or discussion are provided for black-box or transfer scenarios.Lifted PGD requires repeated rendering and optimization, which could be computationally expensive. No runtime, memory, or scalability analysis is reported.

3. Baselines are limited to Fit2D and GuardSplat, which are relatively weak. Stronger or more recent defenses are missing, raising concerns that the comparisons are intentionally favorable. Only a narrow set of editing models are tested (IP2P, Instruct-GS2GS, DGE). The method’s robustness against more advanced or diverse editing pipelines is not demonstrated.

4. Quantitative improvements are modest, and in some cases, invisibility metrics (SSIM, PSNR, LPIPS) actually degrade, undermining the “balance” argument. Visual examples (e.g., Fig. 6–8) show protected assets yielding “unnatural” or distorted edits. This could easily be interpreted as a byproduct of noise rather than a principled defense.

### Questions
I will raise my score if the authors address W1 and W2.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces AdLift, a novel framework to actively protect 3D Gaussian Splatting assets from unauthorized instruction-based editing. It leverages adversarial perturbations lifted from the 2D rendered image domain into 3D Gaussian space using a custom Lifted Projected Gradient Descent algorithm. The method enforces strict rendering-space bounds to ensure visual fidelity while maintaining strong protection against 2D and 3D editing pipelines such as InstructPix2Pix and Instruct-GS2GS. Experiments on several 3D scenes show that AdLift achieves strong protection with minimal perceptual distortion.

### Strengths
1. The idea of lifting perturbations from 2D to 3D to achieve view-consistent adversarial protection is novel and well-motivated.
2. The paper is well-written and well-organized
3. Comprehensive experiments across 2D image editing, global, and local 3DGS editing show consistent results.

### Weaknesses
1. The proposed method assumes full white-box access to the editing model (e.g., InstructPix2Pix or Instruct-GS2GS) to obtain gradients for adversarial optimization. However, in practice, the attacker may use a variety of proprietary or black-box diffusion-based models. The perturbations optimized against one editing model may not generalize to others. The paper lacks experiments to show the transferability and robustness of the method effectiveness across models, a “protected” 3DGS asset might still be editable or corrupted when attacked with a different model or pipeline.
2. The paper only compares against GuardSplat and Fit2D, omitting broader baselines such as adversarially trained 3D representations, or certified perturbation approaches. 
3. There is no detailed ablation study analyzing the sensitivity of the method to key hyperparameters.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
