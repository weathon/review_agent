# Mastering Domain Shift Image Enhancement Via Differentiable Physics

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Visual perception in the wild have demonstrated transformative potential across a wide range of applications, spanning from planetary exploration to deep-sea monitoring missions. However, a fundamental challenge remains in enabling visual perception enhancement that can explicitly extract rules and support interactive, precise manipulation in unknown, dynamic environments—particularly under conditions of large scale data absence, heterogeneous data distribution, and without the supervision of annotated images. Our approach introduces a differentiable physics framework that unifies the camera response model (CRM) with deep learning to achieve visual perception enhancement under multiple degradation conditions. Specifically, grounded in fundamental principles of radiation physics, we formulate the camera response function (CRF) calibration as a constrained optimization problem. Then we reconstruct the brightness transformation function (BTF) in traditional CRM as a multi-scale generative network, completely decoupling it from the CRF. Meanwhile, we design a dual-branch contrastive encoder that enables the BTF to regulate the irradiance enhancement process through multi-scale exposure distributions learned from guide images. This offers a flexible BTF interface supporting stable and controllable domain generalization for image enhancement. Through comprehensive experiments, our method significantly advances domain generalization capabilities in adaptive image enhancement, outperforming specialized counterparts by margins of +1.226 (UIQM) averaged across challenging unseen underwater domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles image enhancement generalization across domains that exhibit diverse and coupled degradations. The authors decouple CRF and BTF, and leverage contrastive learning together with a generative network. First, CRF calibration is performed by modeling camera nonlinearity with DoRF-based bases. Next, representations are learned in irradiance space via contrastive learning between a distorted domain and a normal-exposure domain. Finally, an AFDM module is trained using a value-magnitude–based sort-matching algorithm, enabling robust enhancement on unseen domains.

### Strengths
1. Presents a framework capable of handling complex and unseen degradations that prior LLIE/UIE approaches typically struggle with.
2. Clearly articulates that the conventional CRM coupling between CRF and BTF can hinder enhancement under domain shift, and proposes a principled decoupling.
3. Utilizes dual-branch contrastive learning to capture domain representations in irradiance space.
4. The paper offers dense analyses and thorough empirical comparisons, supporting the validity of the proposed methodology and aiding readability.

### Weaknesses
1. The paper would benefit from an explicit assessment of CRF optimization quality and a sensitivity analysis quantifying the impact of CRF estimation errors on enhancement outcomes.
2. Implementation details are limited (e.g., network specifications, hyperparameters, and experimental settings), which may impede reproducibility.

### Questions
1. In the absence of any camera configuration information during training, is CRF calibration still feasible?
2. When constructing distorted vs. normal-exposure domains, must the datasets be captured with the same camera and/or contain similar content (e.g., underwater imagery vs. everyday scenes), or is cross-camera/cross-content pairing acceptable?
3. In AFDM, what is the precise rationale for value-based sorting? Given that deeper layers encode different semantics, why is a uniform sorting mechanism appropriate across layers?
4. Do you anticipate the framework to remain effective when the distorted/normal domain pair extends beyond low-light scenarios?
5. Were hyperparameter studies conducted for the loss weights composing L_BTF? If so, please summarize the settings and findings.

### Soundness
3

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
4

### Summary
### Summary
This paper proposes a physics-guided image enhancement framework integrating a differentiable camera response model (CRM) with deep generative learning. It decouples the camera response function (CRF) and brightness transformation function (BTF), introduces a dual-branch contrastive autoencoder (CAE) for domain-invariant feature extraction, and an adaptive feature distribution matching (AFDM) module for differentiable eCDF alignment. Experiments on underwater and low-light datasets show improved generalization and real-world robotic deployment.

### Strengths
- Novel integration of physical modeling (CRM) and deep networks.  
- Dual-branch contrastive learning enhances robustness to domain shifts.  
- AFDM offers a differentiable alternative to AdaIN/histogram matching.  
- Comprehensive comparisons with strong baselines.

### Weaknesses
- Conceptual novelty is limited; CRF–BTF modeling and ADMM optimization follow prior work (e.g., LECARM).  
- The “differentiable physics” part is loosely coupled with learning; no end-to-end integration.  
- Writing and figures are dense and hard to follow.  
- Evaluations rely mainly on no-reference metrics without perceptual or user studies.  
- Focused on applications rather than learning theory—less aligned with ICLR scope.

### Strengths
1. **Physics-inspired framework:**
The idea of combining radiometric camera modeling with deep generative learning is meaningful and provides some interpretability rarely seen in image enhancement research.

2. **Dual-branch contrastive learning:**
The proposed CAE design with two symmetrical branches improves robustness to domain shifts and is theoretically analyzed (Eq. 8–9) for stability.

3. **Differentiable eCDF alignment:**
The AFDM module introduces an elegant differentiable alternative to AdaIN or histogram matching, aligning distributions across domains.

4. **Comprehensive experiments:**
The paper presents extensive visual and quantitative results on multiple domains (underwater, low-light, haze), and real-world robotic deployment adds practical value.

### Weaknesses
1. **Limited novelty:**  
   The contributions appear incremental. The CRF–BTF decoupling and ADMM-based CRF estimation follow previous works such as LECARM (Ren et al., TCSVT’18). Dual-branch contrastive learning and feature matching extend standard paradigms rather than introduce fundamentally new learning concepts.

2. **Weak physics–learning integration:**  
   Although termed “differentiable physics,” the physical CRF calibration is treated as an offline optimization rather than an end-to-end differentiable module. The connection between physical modeling and learned BTF remains loosely coupled.

3. **Clarity and presentation issues:**  
   The paper is dense, with unclear notation (e.g., reusing *f, g, E, P*), verbose equations, and missing intuition. The forward pipeline is not clearly explained—particularly how CRF calibration influences BTF training and inference. Figures are complex but not explanatory.

4. **Empirical rigor:**  
   Evaluation relies mainly on no-reference metrics (UIQM, UCIQE, CCF), which are noisy and limited in reflecting perceptual quality. No LPIPS/PSNR results, runtime comparison, or user study are provided. Statistical significance of the reported +1.226 UIQM improvement is unclear.

5. **Language quality:**  
   The writing contains numerous grammatical errors and awkward phrasing, making it difficult to follow in several sections (especially §2.2–2.3). The overall readability is below the ICLR standard.

6. **Reproducibility concerns:**  No code or config reference is provided.

### Questions
1. How is the CRF calibration integrated during training or inference? Is it pre-computed or updated jointly with BTF?  
2. What is the nature of the “guide image” used for AFDM—does it come from another domain, or is it sampled within the same dataset?  
3. Can the authors report perceptual metrics (e.g., LPIPS, PSNR) or user study results to strengthen claims about visual quality?  
4. How does the method generalize across *unseen* domains (e.g., training on low-light and testing on haze)?  
5. What is the computational complexity compared to strong baselines such as Diff-Retinex++ or WFI2-Net?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a differentiable physics framework that combines camera response modeling (CRM) and deep learning to achieve domain-generalized image enhancement under multiple degradation conditions. The traditional Camera Response Model (CRM) couples the Camera Response Function (CRF) and the Brightness Transformation Function (BTF), limiting adaptability. This work replaces the hand-crafted BTF with a generative network, enabling flexible brightness transformation independent of CRF.

### Strengths
The proposed CRF calibration is formulated as a constrained optimization problem, solved using an EMA-ADMM algorithm, ensuring monotonicity and stability of the response curve. The authors develop a dual-branch contrastive learning strategy extracts discriminative latent features from distorted and guide images, enhancing cross-domain generalization.

### Weaknesses
The paper is dense and notation-heavy, with long equations and overlapping terminology (CRF/BTF/CAE/AFDM) that could overwhelm readers. Some figures (e.g., Fig. 1–2) are small and lack explanatory captions for non-specialists. Despite the “domain shift” claim, experiments are primarily underwater-focused; cross-scene validation (e.g., haze, night, thermal) is limited. The model’s adaptability to other physics domains is implied but not empirically shown.

### Questions
1. How does the model ensure physical consistency between the CRF and the learned BTF network during training? Is there a regularization term enforcing the CRM equation g(f(E), k) = f(kE)?

2. The Sort-Matching alignment is claimed to be differentiable — could the authors clarify how gradients are propagated through the sort indices without introducing instability?

3. What is the runtime overhead compared to diffusion-based models like Diff-Retinex++ or Retinexformer? Is real-time operation feasible on embedded platforms?

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
3

### Summary
The paper proposes Phoenix, a mask refinement framework that generates semantic-aware adversarial perturbations and employs a tri-directional contrastive loss between ground-truth, noisy, and refined masks. Experiments on three datasets show moderate improvements over recent baselines.

### Strengths
The method is technically consistent and well-written, though dense in mathematical notation.  
The paper is well-organized with clear figures and ablations.  
The method is fully implemented and experimentally evaluated on multiple datasets.

### Weaknesses
1. The central idea of using adversarial perturbations to represent semantic uncertainty is not convincingly supported. The paper provides intuitive motivation but no solid theoretical or empirical evidence demonstrating that embedding-space perturbations truly correspond to realistic segmentation errors.
2. The methodological gap from SegRefiner or SAMRefiner is incremental rather than fundamental. AMP closely resembles existing adversarial augmentation or consistency-regularization approaches, while the CMRL mainly extends standard contrastive or triplet learning.
3. Performance improvements are relatively small, sometimes within 1–2% over recent baselines, without statistical significance testing.

### Questions
1. How is the perturbation generated in “semantic space” rather than feature space—any quantitative or visual validation?
2. Could the authors provide a stability curve showing performance vs. perturbation strength?
3. Is there any correlation analysis between AMP perturbation regions and true model error maps?
4. Have the authors analyzed whether stronger adversarial perturbations might degrade mask quality or lead to unstable training? Figure 11 qualitatively illustrates several failure cases. A quantitative failure breakdown could make the claims more credible.

### Soundness
2

### Presentation
2

### Contribution
2
