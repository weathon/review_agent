# Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Personalizing text-to-image diffusion models involves integrating novel visual concepts from a small set of reference images while retaining the model’s original generative capabilities. However, this process often leads to overfitting, where the model ignores the user’s prompt and merely replicates the reference images. We attribute this issue to a fundamental misalignment between the true goals of personalization, which are subject fidelity and text alignment, and the training objectives of existing methods that fail to enforce both objectives simultaneously. Specifically, prior approaches often overlook the need to explicitly preserve the pretrained model’s output distribution, resulting in distributional drift that undermines diversity and coherence. To resolve these challenges, we introduce a Lipschitz-based regularization objective that constrains parameter updates during personalization, ensuring bounded deviation from the original distribution. This promotes consistency with the pretrained model’s behavior while enabling accurate adaptation to new concepts. Furthermore, our method offers a computationally efficient alternative to commonly used, resource-intensive sampling techniques. Through extensive experiments across diverse diffusion model architectures, we demonstrate that our approach achieves superior performance in both quantitative metrics and qualitative evaluations, consistently excelling in visual fidelity and prompt adherence. We further support these findings with comprehensive analyses, including ablation studies and visualizations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles distributional drift in few-shot personalization of text-to-image diffusion models by adding a Lipschitz-motivated parameter-distance regularizer to the personalization loss. The authors argue standard objectives (and even prior-preservation) don’t explicitly preserve the pretrained distribution (Theorem 1), and show that if the denoiser is Lipschitz in parameters, bounding $|\theta - \theta_{base}|$ controls KL shift (Theorem 2).

### Strengths
1. Clear statement of objective–goal misalignment; neat, didactic Theorem 1 showing why standard diffusion fine-tuning drifts.
2. Theory sketch linking parameter Lipschitzness → KL control (Theorem 2); connects elegantly to L2 surrogate.
3. Consistent empirical lift on fidelity (DINO/CLIP-I) with competitive CLIP-T across multiple backbones

### Weaknesses
1. The objective is effectively weight decay to $θ_{base}$; relation to continual-learning regularizers (e.g., EWC) and why your variant is preferable in diffusion personalization should be made explicit.
2. You note CLIP-T may drop when fidelity improves; more analysis on prompt adherence failures (per failure figs) would help.
3. In Table 1, SD-3.0 / “+Ours” row shows “CLIP-I +0.0022” but one entry reads “0.0094”. Why is this error proportion so large?

### Questions
Can you quantify Theorem 2’s $\lambda$ (constant) for a given backbone (e.g., SD-1.5) via empirical Lipschitz estimation, and relate it to your best-performing $\lambda$ in training?

### Soundness
2

### Presentation
2

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
This paper focuses on the personalization task in the field of image generation. The authors proposed a method to introduce a Lipschitz-based regularization objective that constrains parameter updates during personalization. The experiment shows the method can achieve competitive performance for this task.

### Strengths
- The presentation of figures is great and easy to understand.
- The math notations in this paper are self-contained and well-defined.
- The paper writing is easy to follow.
- The introduction section is good and clearly explains the motivation and the main claims of the paper.
- The proposed method is straightforward.

### Weaknesses
- I have to say that tuning-based personalization (like, DreamBooth, Custom Diffusion) is outdated. Learning-based personalization is the mainstream in the current image generation community.
- How did you obtain the results of Fig. 2, or is it just an illustration?
- Could you explain more about the claim "Remark 1. Personalization based on the standard diffusion objective (Eq. 2) provides no guarantee of preserving the pretrained distribution and may lead to divergence"? It is a little bit confusing.
- The evaluation is all based on UNet-based diffusion models. I am curious about the comparison results based on the DiT-based model.
- The evaluation lacks the image quality metrics that are also important.
- There are also some previous works exploring the optimization of personalization learning processes. Have you considered comparing the proposed methods with them?

### Questions
Please see the section of weaknesses.

### Soundness
3

### Presentation
2

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
The paper targets overfitting and distributional drift in few-shot personalization of text-to-image diffusion models. It argues that existing objectives (plain denoising, prior-preservation via class prompts) are misaligned with the dual goals of subject fidelity and text alignment because they provide no explicit guarantee of preserving the pretrained distribution. The core contribution is a Lipschitz-based regularization objective that bounds the deviation of the personalized model from the pretrained one.

### Strengths
1 Clear objective-level insight: reframes personalization as a distribution-preserving adaptation problem and exposes misalignment of standard/“prior-preservation” losses with distributional stability.

2 Simple, principled, and practical: the Lipschitz argument leads to a tractable L2 parameter-distance regularizer that is backbone-agnostic, easy to implement, and removes pre-sampling overhead.

3 Solid empirical results: consistent quantitative gains on multiple backbones and personalization strategies; qualitative examples show improved subject identity without sacrificing prompt following. The training-time savings are compelling for practice.

### Weaknesses
1 Assumptions and tightness: The Lipschitz continuity of εθ w.r.t. θ and the resulting KL bound are plausible but high-level; constants (λ) and norm choices are not characterized for realistic models. The theory justifies using parameter-distance regularization but provides limited guidance on magnitude selection beyond empirical tuning.

2 Failure analysis: Some failure cases remain (identity structure, prompt compositionality). It would help to characterize when the regularizer helps or hurts (e.g., highly stylized subjects, complex relational prompts).

### Questions
1 Human evaluation: Could you expand user studies (more raters/prompts) and include identity verification (face/instance retrieval), and prompt adherence judged by humans, to validate metric conclusions?

### Soundness
3

### Presentation
3

### Contribution
3
