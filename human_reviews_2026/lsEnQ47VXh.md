# Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
Recent advances in diffusion-based image generation models (IGMs), such as Stable Diffusion (SD), have substantially improved the quality and diversity of AI-generated content. However, these models also pose ethical, legal, and societal risks, including the generation of harmful, misleading, or copyright-infringing material. Machine unlearning (MU) has emerged as a promising mitigation by selectively removing undesirable concepts from pretrained models, yet the robustness of existing methods, particularly under multi-modal adversarial inputs, remains insufficiently explored. To address this gap, we propose RECALL, a multi-modal adversarial framework for systematically evaluating and compromising the robustness of unlearned IGMs. Unlike prior approaches that primarily optimize adversarial text prompts, RECALL exploits the native multi-modal conditioning of diffusion models by efficiently optimizing adversarial image prompts guided by a single semantically relevant reference image. Extensive experiments across ten state-of-the-art unlearning methods and diverse representative tasks show that RECALL consistently surpasses existing baselines in adversarial effectiveness, computational efficiency, and semantic fidelity to the original prompt. These results reveal critical vulnerabilities in current unlearning pipelines and underscore the need for more robust, verifiable unlearning mechanisms. More than just an attack, RECALL also serves as an auditing tool for model owners and unlearning practitioners, enabling systematic robustness evaluation. Code and data are available at https://github.com/ryliu68/RECALL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces RECALL, a reference-guided latent optimization framework designed to expose vulnerabilities in diffusion models after machine unlearning. The method leverages a multi-modal conditioning setup—combining text and image prompts—to adjust a latent representation until the unlearned model regenerates erased concepts. The paper presents the overall motivation, attack pipeline, and evaluation on multiple unlearning strategies (ESD, FMN, UCE, etc.), reporting higher attack success rates than prior works.

### Strengths
1. The paper is clearly written and generally easy to follow.

2. The method’s structure and optimization process are well illustrated and explained.

3. The conceptual idea of leveraging latent-space multi-modal guidance for unlearning attacks is mostly novel.

### Weaknesses
Mostly concern about the evaluation of whether the method truly works. The current experimental setup fails to convincingly separate genuine recovery of “forgotten” concepts from trivial replay of reference content:

1. The attack initialization already includes 25% of the reference image ($\lambda$=0.25), meaning the optimization starts from a latent that partially encodes the harmful concept itself. This makes the reported ASR potentially inflated and methodologically invalid. 

2. The evaluation does not remove trivial copies — if the optimized latent simply reconstructs or memorizes the reference image, ASR no longer reflects a true unlearning breach. 

3. The paper does not compare the diversity or distributional coverage of generated samples across methods, leaving it unclear whether RECALL actually recovers a broader concept manifold or merely reproduces a few memorized instances compared to other approaches.

4. The lack of ablations for $\lambda$ approaching 0 (pure noise initialization)  makes it hard to assess whether the method generalizes beyond specific harmful exemplars.


In addition, the paper’s treatment of baselines raises serious concerns. Specifically, the most comparable baseline, UnlearnDiffAtk, has been publicly available since around October 2023, and it is unclear whether any later compatible unlearning attacks were tested. The authors should either include more recent baselines if available, or explicitly clarify in the rebuttal that no newer compatible works exist to justify the current comparison. 

Moreover, the authors appear to have **underreported** UnlearnDiffAtk’s performance: its own paper reports 76% ASR (ESD) and 98% (FMN) on Nudity tasks (Table 2), while this paper's Table 1 shows only 51% and 92% respectively. This inconsistency suggests that the baseline reimplementation may be incorrect or incomplete, casting doubt on the claimed relative advantage of RECALL.

### Questions
See weakness above.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presented a multi-modal adversarial framework Recall with SOTA results in diffusion model white-box attack settings. The method optimizes the adversarial image in the latent space of the unlearned model itself, requiring no external classifiers. Extensive experiments across ten state-of-the-art unlearning methods and four tasks demonstrate that Recall consistently outperforms existing attacks in success rate, speed, and semantic alignment. The paper shows that current unlearning pipelines are fundamentally fragile against multi-modal adversarial inputs, urging the development of more robust safety measures.

### Strengths
1. Recall introduced multi-modal (image+text) attack with the text prompt unmodified, which generates the unlearned image while still keeping semantic fidelity to the original unmodified prompt. The experiment results show SOTA accuracy.
2. Recall is computationally and practically efficient. It doesn't require external models or classifiers. Performing the adversarial optimization directly in the model's latent space is computationally more efficient, which is supported by experiment results.
3. Recall is shows good generalization across models and tasks. It does not overfit to a specific reference image to guide the attack while still producing diverse outputs.
4. Extensive generalization study and ablation study.
5. The paper is well-written, with a clear and compelling narrative from motivation to result.

### Weaknesses
1. The paper is more on empirical side. While the results are good, it lacks a theoretical analysis explaining why the multi-modal pathway is so vulnerable or providing formal guarantees about the attack's convergence.
2. The adv_img even though is effective, it will be easily rejected by real image gen system by simple safe guarding before it reaches to the model.
3. adversarial prompt attack was proven to be a good method. what about adversarial prompt + adversarial image, will it get higher ASR? There is no such ablation study in experiment.

### Questions
1. 50-step DDIM scheduler is inefficient in general. How will the algorithm work with other faster scheduler?
2. The method focuses on single concept (Van Gogh, or nudity etc.). How about multi-concepts?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RECALL, an unlearning model attack method designed to operate within the image latent space of diffusion models. The core of the method involves using a reference image to guide the iterative generation of an adversarial latent representation ($z_{\text{adv}}$), which successfully recovers a supposedly erased target concept (e.g., specific style or object). The authors conduct extensive experiments to evaluate the method's effectiveness, computational efficiency, and robustness, convincingly revealing significant vulnerabilities in current machine unlearning techniques when subjected to image latent space attacks.

### Strengths
1. RECALL successfully identifies adversarial examples in the latent image space, providing compelling evidence that existing unlearning methods (e.g., fine-tuning, knowledge distillation) fail to fully eradicate sensitive or proprietary concepts.
2. The paper includes a comprehensive experimental evaluation and extensive ablation studies that thoroughly assess the potential of image-level attacks on unlearning methods across various metrics and unlearning targets.
3. Efficiency and Practicality: The outstanding experimental results, coupled with a significantly shorter computational time compared to baselines, enhance the practical relevance and real-world applicability of the proposed attack.

### Weaknesses
1. Despite the claim of "reference independence" in Section 5.5, the method fundamentally relies on a reference image during the adversarial optimization in Stages I and II. The authors must clarify the specific requirements for this reference image. For instance, what characteristics might a reference image possess that could cause the attack to fail or significantly degrade its performance? Furthermore, given the results in Table 4, which suggest that a simple Image-Only attack can already restore the target concept, the current method appears more like an effective way to refine this recovery by finding a latent state that is minimally destructive to the surrounding concept space, rather than a fundamentally new recovery vector.
2. Insufficient Test Data Coverage: The evaluation is limited by the amount of test data used. To comprehensively assess the robustness of unlearning methods against RECALL, the authors should employ a larger and more diverse dataset.
3. The paper primarily focuses on finding an adversarial latent in the image latent space ($z_{\text{adv}}$) to recover the forgotten concept, with no explicit optimization or guidance related to the textual modality beyond standard conditional inputs. Given this, the claim of presenting a "multi-modal attack" requires further justification or clarification.
4. The claim made in Lines 236-249 seems questionable. RECALL appears to be fundamentally an outcome of a trade-off between prompt following and sampling diversity, which aligns more closely with the diverse sampling results shown in Table 6.
5. The paper would greatly benefit from a brief introductory section or paragraph in the main paper to clarify the common terminology used in this attack space: specifically, the concepts of text-only, image-only, and hybrid/multi-modal attacks/models.

### Questions
1. ALGORITHM 1, Line 23 & 24: the final $z_{\text{adv}}$ obtained at Line 23 appears to correspond to the latent state at time $t=0$ (the clean, final image latent). If this is the case, why is $z_{\text{adv}}$ then directly fed back into the diffusion model at Line 24? This procedure deviates from the standard DDPM/DDIM sampling process, which typically starts diffusion from a noisy latent state at $t=T$. Conversely, if $z_{\text{adv}}$ actually corresponds to $t=T$ (a noisy latent), how does the DDIM sampling process manage to produce the diverse recovery results shown in Figure 7?
2. Table 2 CLIP Score Discrepancy: Table 2 shows that the CLIP Score for the images recovered by RECALL is higher than the score for the original Stable Diffusion (SD) model. Please provide an explanation for this phenomenon.
3. The Periodic Integration ablation experiment is incomplete. Specifically, ablation studies are missing for the impact of key hyper-parameters such as the periodic interval and the regularization coefficient.
4. Are LPIPS and IS truly appropriate metrics for evaluating the diversity of the generated images in this attack context? Considering the goal of concept recovery, would a metric based on the variance of DINO scores be a more suitable or complementary choice for measuring recovery diversity?

### Soundness
4

### Presentation
4

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
This paper propose a multi-modal guided attack framework for unlearned diffusion model, where during its attack process only a single reference image is utilized. Also, authors implemented comprehensive experiments on different kinds of adversarial attack and different victim unlearned diffusion model.

### Strengths
1. novel multi-modal attack pipeline: latent encoding with reference image blending, iterative latent optimization and the final multi-modal attack using optimized adversarial image with the original text prompt.
2. Strong empirical validation across diverse settings. The evaluation experiments are impressively comprehensive (10 unlearning methods and 3 attack baselines.) The proposed method, RECALL, consistently achieve the best attack performance and also superior semantic alignment.

### Weaknesses
1. Authors have overclaimed the independency of their proposed attack method. During attack process, only a single reference image is needed, however, the reference images are still generated by original diffusion models. So, there is an assumption that the original diffusion models are accessible, which cannot be achieved in some cases. 
2. Although Appx. F claims the robustness across references, the main text underplays the sensitivity of results to poorly aligned or compositionally distinct reference images. A quantitative failure analysis would clarify generality limits.
3. Some steps resemble prior latent alignment or DreamBooth inversion procedures.

### Questions
1. Does RECALL rely on the specific cross-attention fusion mechanism in SD (text-image co-attention), or could it generalize to models like DALLE 3 or Flux that use distinct conditioning pipelines?
2. Could model owners detect such attacks through latent distribution monitoring? If yes, how does RECALL evade simple detection heuristics?

### Soundness
3

### Presentation
3

### Contribution
2
