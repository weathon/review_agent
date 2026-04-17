# MeInTime: Bridging Age Gap in Identity-Preserving Face Restoration

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
To better preserve an individual's identity, face restoration has evolved from reference-free to reference-based approaches, which leverage high-quality reference images of the same identity to enhance identity fidelity in the restored outputs. However, most existing methods implicitly assume that the reference and degraded input are age-aligned, limiting their effectiveness in real-world scenarios where only cross-age references are available, such as historical photo restoration. This paper proposes MeInTime, a diffusion-based face restoration method that extends reference-based restoration from same-age to cross-age settings. Given one or few reference images along with an age prompt corresponding to the degraded input, MeInTime achieves faithful restoration with both identity fidelity and age consistency. Specifically, we decouple the modeling of identity and age conditions. During training, we focus solely on effectively injecting identity features through a newly introduced attention mechanism and introduce Gated Residual Fusion modules to facilitate the integration between degraded features and identity representations. At inference, we propose Age-Aware Gradient Guidance, a training-free sampling strategy, using an age-driven direction to iteratively nudge the identity-aware denoising latent toward the desired age semantic manifold. Extensive experiments demonstrate that MeInTime outperforms existing face restoration methods in both identity preservation and age consistency. Our code is available at: https://anonymous.4open.science/r/MeInTime-DBF7/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MeInTime, a diffusion-based face restoration framework designed to handle cross-age reference-based restoration. Traditional reference-based methods assume age alignment between degraded and reference images, which limits their applicability in real-world scenarios (e.g., historical photos). MeInTime decouples the modeling of identity and age: during training, identity features are learned through reference embeddings and Gated Residual Fusion modules; during inference, an Age-Aware Gradient Guidance strategy is introduced to steer restoration toward the desired age manifold without retraining. Experiments on same-age and cross-age datasets show improved identity preservation and age consistency compared to existing baselines.

### Strengths
+ The paper extends reference-based face restoration to the cross-age domain, which is an underexplored but practically relevant scenario.

+ Separating identity learning and age guidance is conceptually elegant and avoids conflicts between identity and age signals.

+ The Gated Residual Fusion module is well-motivated and effectively stabilizes identity–structure fusion.

+ The authors benchmark on both same-age and cross-age datasets and introduce age-consistency metrics, providing clear evidence of quantitative gains.

+ Visual examples demonstrate that the model generates more age-consistent restorations compared to prior works.

### Weaknesses
– Most degraded samples used for visualization and evaluation are synthetically generated with severe distortions (e.g., heavy blur or compression). Under such extreme degradation, facial wrinkles and texture cues are largely lost, making it unreliable to infer age semantics from low-quality inputs. This could easily cause instability or misalignment between estimated and target ages in practice.

– The method introduces a gradient-based optimization process during inference to achieve age control. However, it is not clear whether this method is better than those using face editing-based solutions on the reference image or the restored images. It remains unclear why such simpler or more direct methods are not considered or compared.

– The AGE metric depends on a pretrained estimator, which might not correlate well with perceptual aging. Additional perceptual studies or human evaluations would strengthen the claims.

- The so-called ID-preserving sampling in Eq. 8 effectively modifies the denoising trajectory based on facial feature gradients, which is conceptually similar to facial attribute editing or latent direction control seen in previous works such as [r1,r2] or other latent manipulation methods. The distinction between this “sampling” and conventional attribute-based editing is not articulated. Maybe it is better to add some face editing works in Section 2.

[r1] When StyleGAN Meets Stable Diffusion: a w+ Adapter for Personalized Image Generation
[r2] LEDITS++: Limitless Image Editing using Text-to-Image Models

### Questions
- Many degraded samples in the paper are heavily distorted. How does the model perform under moderate degradations, where age cues are still partially available?


- Why not apply a face editing approach to either (a) edit the reference image to match the target age before restoration, or (b) perform age editing after restoration? Would this achieve comparable or even better age consistency with less computational overhead?

- How stable is the Age-Aware Gradient Guidance when the reference and degraded ages differ drastically (e.g., 20s vs. 80s)? Does it sometimes produce over-aging or artifacts?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MeInTime, a diffusion-based reference-guided face restoration framework aimed at handling cross-age scenarios where the reference and degraded images of the same person have large age gaps. The key idea is to decouple identity and age conditioning: identity embeddings are injected during training through a Gated Residual Fusion (GRF) module, while age consistency is adjusted during inference using a training-free Age-Aware Gradient Guidance.

### Strengths
Solid engineering design, the combination of GRF for stable identity fusion and age-aware gradient guidance is well implemented and empirically effective.

Clear structure and writing,the paper is clearly written, visually engaging, and the methodology section is easy to follow.

Training-free age control: The gradient-based age guidance is conceptually clean and avoids extra finetuning.

### Weaknesses
1. The cross-age reference setting is a rare and contrived use case. It’s unclear how many real restoration tasks actually require explicit age matching. The work does not convincingly show that this setting matters beyond a few illustrative examples. The paper starts from the observation that current reference-based face restoration methods assume the reference and degraded faces are of similar age.
While this is technically true, the practical importance of bridging “cross-age” gaps in face restoration is quite limited.

2. Experiments rely on synthetically degraded data; no evaluation on truly degraded or historical photos, which weakens the claim of “real-world generalization.”

3. The paper does not report inference time or computational overhead. Given that the proposed Age-Aware Gradient Guidance involves multiple iterative optimization steps, efficiency could be a major concern for practical applications. Quantitative timing or FLOPs comparison is missing.

4. While automatic metrics (FID, MUSIQ, AGE MAE) are presented, subjective evaluations (user or identity verification studies) are missing, which are important for human-centric tasks like face restoration.

5.The related work section overlooks several important prior approaches:

[1].Face Super-Resolution Guided by 3D Facial Priors

[2].Rethinking Deep Face Restoration

### Questions
1. Interestingly, Table 1 shows that on cross-age restoration, the reference-free baseline CodeFormer achieves higher PSNR and SSIM than the proposed MeInTime, despite lacking reference information. This suggests that the inclusion of reference images may actually harm reconstruction fidelity when the reference and degraded faces differ significantly in age. The authors should analyze this phenomenon more carefully, as it weakens the central claim that MeInTime “bridges” the age gap effectively.

### Soundness
2

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
The paper proposes MeInTime, a diffusion-based framework for identity-preserving face restoration that explicitly addresses the challenge of large age gaps between degraded input images and available reference images. MeInTime separates identity and age conditioning: it injects robust identity features during training, while, during inference, it leverages a novel Age-Aware Gradient Guidance mechanism based on textual prompts to control aging, decoupling identity from age in the generative process.

### Strengths
1.The paper targets a practically relevant and previously underexplored problem: high-fidelity, identity-preserving face restoration when only cross-age references are available.
2. The use of a decoupled training-inference strategy—training for identity preservation and introducing age controllability at inference—is a thoughtful response to the lack of large-scale cross-age paired datasets, as discussed with supporting data in Appendix B/Figure 12 (Page 15–16).

### Weaknesses
1. Limited Theoretical Analysis of Attribute Decoupling: The paper claims that identity and age are decoupled by design (identity during training, age only via gradient guidance at inference), yet it lacks a more principled investigation or proof of whether and to what extent this decoupling is reliably achieved. For example, no formal analysis or visualization is provided to demonstrate that the injected identity embeddings or the age gradients are indeed orthogonal in the learned space. Without explicit investigation, it remains somewhat speculative whether the method fully avoids entanglement between age and identity, especially under distribution shifts.

2. Structural and Optimization Details May Hinder Reproducibility: While implementation details are provided, several critical elements are only broadly sketched. For example, the precise effect of the GRF module hyperparameters, the initialization process for identity/token projection, and the inferred scaling of guidance during inference are not dissected in depth. Additionally, Algorithm 1 might benefit from being more explicit on the stopping criteria, initialization of age/generic prompts, and GRF interaction during inference.

### Questions
1. Can the authors provide a principled analysis (e.g., mutual information, attention visualization, or orthogonality in feature space) demonstrating that their identity embeddings remain robust (and do not leak age information) when reference images cover very wide age gaps? Empirical or theoretical clarification here would strengthen the claim of identity-age decoupling.

2. What is the actual computational overhead (in wall-clock time or FLOPs) for MeInTime during inference compared to, say, FaceMe or RestorerID, particularly under the Gradient Guidance with multiple optimization passes? Is the method practical for real-time or high-throughput use cases?

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
3

### Summary
This paper introduces MeInTime, a novel diffusion-based framework that significantly advances reference-based face restoration by effectively tackling the challenging cross-age scenario. The key innovation lies in its decoupled modeling of identity and age conditions: it injects identity features via a dedicated attention mechanism during training, and at inference, employs a training-free Age-Aware Gradient Guidance to steer the generation towards the target age. Extensive experiments confirm that MeInTime outperforms existing methods, achieving superior identity fidelity and age consistency simultaneously.

### Strengths
+Pioneering Cross-Age Reference-Based Framework: This work introduces the first reference-based face restoration framework specifically designed for cross-age scenarios, effectively extending the capability of existing methods from same-age to cross-age restoration by incorporating target age prompts.
+ Novel Disentangled Training-Inference Strategy: The proposed method employs a decoupled approach that separately handles identity preservation during training through dedicated attention mechanisms, and age consistency during inference via training-free Age-Aware Gradient Guidance, effectively resolving identity-age conflicts.
+ Gated Residual Fusion modules dynamically integrate structural features from degraded inputs with identity representations in a content-aware manner.
+ Plug-and-play Age-Aware Gradient Guidance steers generation toward target age semantics without retraining.
+ Comprehensive experimental validations show superior performance in visual quality, identity preservation, and age consistency compared to existing approaches.

### Weaknesses
- According to Table 1, the performances of the proposed method are not always the best. The authors should explain the reasons in detail.
- The authors do not compare the speed and the number of parameters of the proposed method, compared to existing methods.
- The authors do not present the failure cases of the proposed method. I think it is better to analyze the limitations.
- The fonts in the Figures are too small. 
- The effectiveness of the Age-Aware Gradient Guidance is not verified in the ablation studies.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
