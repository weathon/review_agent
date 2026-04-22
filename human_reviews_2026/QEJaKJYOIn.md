# DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Current image immunization defense techniques against diffusion-based editing embed imperceptible noise into target images to disrupt editing models. However, these methods face scalability challenges, as they require time-consuming optimization for each image separately, taking hours for small batches. To address these challenges, we introduce DiffVax, a scalable, lightweight, and optimization-free framework for image immunization, specifically designed to prevent diffusion-based editing. Our approach enables effective generalization to unseen content, reducing computational costs and cutting immunization time from days to milliseconds, achieving a speedup of 250,000x. This is achieved through a loss term that ensures the failure of editing attempts and the imperceptibility of the perturbations. Extensive qualitative and quantitative results demonstrate that our model is scalable, optimization-free, adaptable to various diffusion-based editing tools, robust against counter-attacks, and, for the first time, effectively protects video content from editing. More details are available in https://diffvax.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DiffVax, a diffusion-based image immunization framework designed to prevent unauthorized editing by generative diffusion models. The method learns an optimization-free, feed-forward perturbation generator that produces subtle but effective protective noise, achieving strong robustness and generalization across different editing models and mask settings.

### Strengths
- The authors propose DiffVax, a diffusion-based immunization framework that protects images from unauthorized edits through an optimization-free, feed-forward process generating imperceptible yet effective perturbations.
- The method demonstrates superior robustness, efficiency, and generalization to unseen models, content, and masks compared to prior optimization-based defenses.
- This paper demonstrate consistent empirical improvements over baseline optimization approaches in both efficiency and performance.
- The experimental discussion is thorough, with complete ablation studies and detailed analyses.
- This paper include a user study to further strengthen their results.

### Weaknesses
- The paper presents an empirical approach with limited theoretical justification, relying mainly on experimental evidence rather than principled analysis.
- The experiments are comprehensive and the discussion is rich and detailed, with no critical methodological flaws identified.

### Suggestions
It would strengthen the paper to include recent diffusion-based image protection works, such as the attention-guided EditShield [1] and the latent-space diffusion attack [2], which also explore strategies for preventing unauthorized image editing.
 
[1] Chen et. al. EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models, ECCV 2024  
[2] Shih et. al. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models, AAAI 2025

### Questions
- Could the authors provide a more detailed description of how the SD($\cdot$) operator in Section 3.3 is computed? Does it require running the full diffusion reverse process, and if so, would that be computationally expensive? How do the authors ensure stable gradient propagation through such a process?
- Can the proposed protection method be fine-tuned with a small number of samples to neutralize its effect?
- Could the authors include learning curves for each loss component during training to better illustrate the convergence behavior?
- The authors could consider incorporating the references mentioned in the Weaknesses section into the Related Work to provide a more comprehensive contextual discussion.
- It would be helpful if the authors could expand the Appendix to discuss the method’s limitations and summarize the potential future directions mentioned across different sections.

I encourage the authors to strengthen the paper by addressing these points in the rebuttal.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DiffVax, an optimization-free framework for image immunization against diffusion-based editing. Instead of performing costly per-image optimization, DiffVax trains a lightweight immunizer network to generate imperceptible yet effective perturbations in a single forward pass. The resulting perturbations not only degrade edits produced by various diffusion-based models (e.g., inpainting, InstructPix2Pix) but also exhibit strong resilience to counter-attacks such as JPEG compression and denoising. Owing to its scalability and efficiency, DiffVax extends immunization beyond static images to video content.

### Strengths
1. The proposed method demonstrates robust protection even on previously unseen images, suggesting that the learned perturbation strategy generalizes beyond the training distribution.
2. Unlike prior optimization-based approaches that are computationally prohibitive, DiffVax achieves real-time immunization via a feed-forward model. This efficiency allows the framework to extend immunization to video content, which is infeasible for existing methods.

### Weaknesses
1. While the proposed method introduces a training-time immunization mask to constrain the perturbation to semantically important regions, it remains unclear how robust the method is when malicious editing masks deviate from those used during training. Although the authors briefly mention this scenario (Lines 313–317), the model is trained using a single fixed mask per image, and no explicit mechanism is introduced to ensure robustness against arbitrary or partially overlapping test-time masks. This raises concerns about edit-time generalization in realistic attack settings.
2. The experimental setup primarily focuses on single-object foreground inpainting tasks, where the perturbation is confined to a specific region (e.g., a human subject). This narrow scope limits the understanding of how well the method generalizes to more diverse and realistic editing scenarios, such as foreground object editing or multi-object scenes, which are common in real-world applications.
3. The baseline selection in the experimental comparison may raise concerns about fairness. While the proposed method is evaluated primarily on inpainting-based diffusion models, the compared immunization baseline Photoguard were originally designed for broader image-to-image translation tasks, not inpainting. This mismatch in task formulation may bias the evaluation in favor of the proposed approach and limit the relevance of the reported comparisons.
4. While the paper emphasizes the low memory footprint of DiffVax at inference time (5,648 MiB as stated in Line 413), the training phase appears significantly more demanding. Since the immunizer is trained using a loss computed on the edited output—requiring backpropagation through multiple steps of the diffusion process—it is unclear whether the overall method is truly more memory-efficient than optimization-based approaches like PhotoGuard, which reportedly require 15 GB. The paper does not provide concrete memory usage statistics for training, leaving the claimed efficiency somewhat unsubstantiated.

Note: Weaknesses 1-4 correspond directly to Questions 1-4.

### Questions
1. In Figure 14, only the third row corresponds to the case where the editing mask is identical to the one used during training. For the remaining rows (e.g., rows 1–2 where the perturbation is partially masked, and rows 4–5 where the edit mask includes unprotected regions), it is unclear whether the model still forces the edited region to degenerate into low-information content (e.g., pixel values = 0). Can the authors clarify whether the masks used in Figures 1, 4, and 5 are identical to the training-time immunization masks? Moreover, can the authors provide both qualitative and quantitative results—along with comparisons to baseline—for scenarios where the edit-time mask occludes only part of the perturbation or includes perturbation-free regions?
2. Have the authors evaluated their method under alternative inpainting settings? Specifically:
(1) How does DiffVax handle the case where the targeted region is masked and edited, such as in the foreground object replacement (e.g., from human to other object)?
(2) Can the method generalize to multi-object scenes, where multiple semantically important regions may need protection simultaneously? Providing results for these broader cases would better validate the robustness and applicability of the proposed method.
3. Have the authors considered including task-aligned baselines, such as AdvPaint [1] or other immunization approaches specifically tailored to inpainting models? A comparison with such baselines would provide a more comprehensive and fair evaluation of the method's effectiveness in its target domain.
4. Can the authors clarify the actual GPU memory requirements during training, particularly when backpropagating through the denoising steps of the diffusion model? Given that the experiments were conducted using a single A100 GPU, how feasible is it for real-world image owners with limited hardware (e.g., <16 GB VRAM) to train or fine-tune such an immunizer model? A discussion on resource requirements and practical accessibility would strengthen the paper’s applicability claims.

[1] Jeon et al., AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption, ICLR 25.

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
1. The paper introduces DiffVax, a novel framework for immunizing images against diffusion-based editing. 
2. DiffVax proposes training a feed-forward immunizer model, which is based on UNet++, to generate an imperceptible perturbation in a single pass. 
3. DiffVax's main contribution is the optimization-free approach with a massive speedup from seconds to milliseconds, and claim the first application of such a defense to video content, which is enabled by low latency.
4. Experiments demonstrate its effects compare with previous guard models.

### Strengths
1. The core idea of training a generative immunizer model instead of relying on per-image optimization is a significant and novel contribution.
2. The primary strength is the practical, real-world benefits, which can reduce the immunization time from seconds to milliseconds and make video immunization feasible for the first time.
3. The method demonstrates strong quantitative and qualitative results in disrupting edits from the specific models such as SD-Inpainting, InstructPix2Pix and MagicBrush. It also shows good robustness to standard counter-attacks like JPEG compression and denoising.

### Weaknesses
1. The method's real-world utility is limited by its reliance on an input mask $M$ for the protected region. In a practical scenario, this mask must be generated using model such as SAM, which is a non-trivial computational step. This added overhead is not accounted for in the "milliseconds" performance claim and creates a time-consuming bottleneck that limits the method's true end-to-end speed and ease of use.
2. The immunizer model was trained on a very small 800 images dataset. This narrow data scope raises concerns about the model's ability to generalize. The results, which are almost entirely focused on people, are not sufficient to prove the method works on the vast diversity of in-the-wild content which may out-of-distribution.
3. The paper states the immunizer model $f(\cdot;\theta)$ is a UNet++, but this architectural choice is not justified. Given this model is central to the paper's contribution, there is no discussion of why this specific architecture is required over other architectures. The design details and properties of the immunizer itself are underdeveloped.

I will consider adjusting my score if my concern is well resolved.

### Questions
1. The paper demonstrates immunizing video frames to disrupt image edits applied frame-by-frame. Have the authors considered the effect of feeding a single immunized image into a dedicated image-to-video model such as SVD? It would be interesting to know if the perturbation also disrupts this different modality of generation.
2. Following on the generalization weakness, how effective is this defense against more powerful, modern editing models like Qwen-Image, which are bigger and more robust? The evaluation on SD v1.5/v2 feels somewhat dated.
3. Can the immunizer model be trained simultaneously against multiple distinct image-editing models to improve generalization?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DiffVax, an optimization-free framework for image immunization against diffusion-based editing. DiffVax trains a feed-forward immunizer network that generates subtle perturbations to images in a single pass. The immunized images are able to resist various diffusion editing models, and the method is extended to the videos for the first time. Experiments demonstrate that DiffVax outperforms the selected baselines in terms of protective efficacy, visual imperceptibility, computational efficiency, and robustness to counter-attacks.

### Strengths
1. The improvement in scalability is significant, where a training time of around 22 hours exchanges to an inference-time reduction by several orders of magnitude, enabling the processing for videos.

2. DiffVax presents commendable generalizability and applicability across both inpainting-based and instruction-guided diffusion editing models, flexible masking schemes, and unseen data.

3. The qualitative and quantitative analyses supported by multiple complementary metrics and human ratings instantiate the thorough evaluation protocol.

### Weaknesses
1. Only two baselines are included for comparison. More relevant baselines are mentioned but not experimentally assessed. The claim that they 'are not directly applicable in our inpainting-based setup' is questionable, as at least SDS can handle under a SD inpainting setting.

2. The application scenarios are limited to photographs featuring foreground objects or persons, where DiffVax prevents the background editing. Other use cases such as facial expression modification, artistic work protection, or anime secondary creation are not explored.

3. The lack of failure case studies weakens the completeness of the manuscript. For example, the performance of DiffVax on data from other domains is unclear. Moreover, Fig. 14 shows that when the test-time mask size decreases, DiffVax fails as the bottom-right example.

4. There is a minor issue that a highly pertinent work PCA [1] is omitted as related work.

[1] *A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse. arXiv preprint arXiv:2408.10901, 2024.*

### Questions
1. How might DiffVax adapt or fail if diffusion editors adaptively employ adversarial training or adversarial purification to counteract immunization perturbations? Is it possible to provide a quick validation regarding its performance against DiffPure [2]?

2. How would DiffVax perform on truly unseen diffusion architectures, especially non-LDM ones? The current generalization across editing models is only partially verified, i.e., from SD v1.5 to SD v2.

3. Could the authors conduct additional ablations on the impact of norm selections for the loss terms?

[2] *Diffusion Models for Adversarial Purification. in ICML 2022.*

Please respond to both the weaknesses and questions.

### Soundness
2

### Presentation
3

### Contribution
3
