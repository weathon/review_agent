# Concept-based Adversarial Attack: a Probabilistic Perspective

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
We propose a concept-based adversarial attack framework that extends beyond single-image perturbations by adopting a probabilistic perspective. Rather than modifying a single image, our method operates on an entire concept - represented by a distribution - to generate diverse adversarial examples. Preserving the concept is essential, as it ensures that the resulting adversarial images remain identifiable as instances of the original underlying category or identity. By sampling from this concept-based adversarial distribution, we generate images that maintain the original concept but vary in pose, viewpoint, or background, thereby misleading the classifier. Mathematically, this framework remains consistent with traditional adversarial attacks in a principled manner. Our theoretical and empirical results demonstrate that concept-based adversarial attacks yield more diverse adversarial examples and effectively preserve the underlying concept, while achieving higher attack efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel concept-based adversarial attack framework that extends probabilistic attacks from single-image perturbations to concept-level perturbations. By generalizing the distance distribution $p_{dis}$ from an individual image to an entire concept, the method effectively reduces the divergence from the victim distribution $p_{vic}$, leading to improved attack success rates and higher-quality adversarial examples. The paper also introduces a concept augmentation strategy using modern generative models to enhance distributional diversity, and demonstrates clear performance gains over existing probabilistic attack methods.

### Strengths
1. The paper presents a unified and elegant probabilistic framework that generalizes single-image attacks to concept-level attacks, treating the former as a special case of the latter and achieving a smooth transition from $x_{ori}$ to $C_{ori}$.
2. Experimental results strongly demonstrate that the proposed method outperforms existing baselines in terms of targeted attack success rate, while generating higher-quality adversarial examples that effectively preserve the original concept semantics.

### Weaknesses
1. The proposed work builds upon the probabilistic perspective introduced in [1] and extends it to concept-level attacks. However, concept-based or unrestricted attacks using generative models have already been explored in prior works, including GAN-based [2] and diffusion-based approaches [3,4]. In addition, several studies have examined distance metrics in unrestricted attacks [5,6]. These overlaps weaken the originality of the contribution.
2. The derivation of Theorem 1 relies on the assumption that $p_{dis}$ follows a Gaussian distribution. While this is a common simplification in theoretical analyses, the authors do not sufficiently discuss whether the theoretical guarantees still hold when using non-Gaussian PGMs (e.g., VAEs or EBMs). This raises concerns about the generality of the proposed framework.
3. The concept augmentation process depends heavily on powerful pretrained models (e.g., SDXL) and considerable computational resources (e.g., LoRA fine-tuning), which may limit the reproducibility and applicability of the method in resource-constrained settings.

**References:**

[1] Zhang, Andi, Mingtian Zhang, and Damon Wischik. "Constructing semantics-aware adversarial examples with a probabilistic perspective." NeurIPS, 2024.

[2] Xiao, Chaowei, et al. "Generating adversarial examples with adversarial networks." IJCAI, 2018.

[3] Dai, Xuelong, Kaisheng Liang, and Bin Xiao. "Advdiff: Generating unrestricted adversarial examples using diffusion models." ECCV, 2024.

[4] Chen, Xinquan, et al. "Advdiffuser: Natural adversarial example synthesis with diffusion models." ICCV, 2023.

[5] Laidlaw, Cassidy, Sahil Singla, and Soheil Feizi. "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models." ICLR, 2021.

[6] Zhao, Zhengyu, Zhuoran Liu, and Martha Larson. "Towards large yet imperceptible adversarial image perturbations with perceptual color distance." CVPR, 2020.

### Questions
1. Please discuss the potential limitations of assuming a Gaussian distribution for $p_{dis}$ in Theorem 1. If a non-Gaussian PGM is used, can the reduction in KL divergence still be theoretically guaranteed? Is there a more general metric that could replace KL divergence to provide broader theoretical validity?
2. “Concept augmentation” plays a key role in improving diversity and performance, but it relies on large pretrained models such as SDXL with LoRA fine-tuning. Under limited computational resources, how would the quality, diversity, and effectiveness of adversarial samples degrade if simpler or smaller PGMs were used? Would the method remain effective in low-resource settings?
3. Tables 4 and 5 show that under the conservative (CONS) strategy, increasing the number of selected samples $M$ (from 1 to 10) decreases transferability while improving image quality and similarity. This suggests that the conservative strategy may prioritize perceptual quality over attack strength as $M$ increases. Please discuss how to better balance this trade-off between transferability and quality, and explain the specific reasons behind the drop in transferability for larger $M$.

### Soundness
3

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
This paper introduces **concept-based adversarial attacks**, where a “concept” is modeled not as a single image but as a set of images or a fine-tuned generative model (e.g., diffusion). Adversarial examples are sampled from the overlap between the concept prior and the target classifier’s probability distribution, enabling identity-preserving attacks beyond norm-bounded perturbations. The authors provide theoretical justification via KL divergence analysis. They conduct extensive experiments on ImageNet, achieving **97.8% targeted white-box success**.  Perceptual quality is validated using CLIP similarity and no-reference IQA metrics such as MUSIQ, NIMA, and TReS.

### Strengths
1. The problem is well-motivated. The paper goes beyond traditional single-image or class-level attacks and instead enables identity-level, concept-aware adversarial generation that produces realistic and semantically consistent examples. Furthermore, I believe this approach could be valuable beyond adversarial attacks. It may help future work probe model hallucination and understand the semantic priors that models rely on.
2. The method is built on a clear probabilistic formulation rather than heuristic manipulation, making the approach principled and conceptually well-grounded.
3. The experiments include strong perceptual fidelity evaluation, using CLIP similarity and no-reference IQA metrics such as MUSIQ and TReS to assess visual realism.

### Weaknesses
1.  **Missing compute / FLOP parity.**  
    The appendix briefly reports compute but does not provide a clear, quantitative comparison of **FLOPs / GPU-hours** between the proposed pipeline and the baselines. Please report wall-clock GPU-hours and/or FLOP counts per concept (training + sampling) for the proposed method and for each baseline. This will help readers judge whether performance gains are due to algorithmic novelty or to much greater compute and data budgets.
    
2.  **Experimental structure and choice of victim models.**  
    The experimental section lacks a systematic structure for robustness testing. I request the following additions:
    

-   Report how attacks generated against ResNet-50 behave when transferred to an **adversarially-trained ResNet-50** under an L∞ threat model. This gives a clear point of comparison with standard robustness tests.
    
-   Include **white-box evaluations where the victim itself is adversarially trained** (i.e., generate attacks against a robustly trained model) to show whether the concept-guided sampling can break robust classifiers in white-box settings.
    
-   Broaden the set of victim models beyond standard classifiers. Evaluate against models trained with mainstream non-adversarial robustness techniques (e.g., CutMix, AugMix, models tuned for OOD/generalization). These are realistic victim models and better reflect deployment settings.
    

3.  **Missing strong imperceptible / transfer baselines.**  
    The current baselines omit important, stronger transfer/imperceptible attacks. Please include comparisons to state-of-the-art transfer methods such as **VMI-FGSM** [ii] and **LGV** [i]. These methods represent a competitive, low-compute class of attacks and are necessary to understand the relative potency of the proposed generative approach.


[i]. LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity
[ii]. Enhancing the Transferability of Adversarial Attacks through Variance Tuning

### Questions
1. In Section 6, you state that prior work “treats a class (e.g., cat, dog, truck) as the concept, which cannot precisely capture an individual identity,” whereas your method represents a concept using a set of images or a probabilistic model. This suggests that your notion of ‘concept’ operates at an identity level rather than a class level. However, most related work in diffusion-based adversarial attacks defines ‘concept’ strictly at the class/category level. Can you please clarify your use of the term ‘concept’ more explicitly and explain how it differs from the class-level definition in existing work?

2. Please formalize your notion of transferability by giving a precise metric (for example top1 or top5 targeted success) and describing exactly how adversarial samples are generated and evaluated across source and target models. What am I precisely asking is that correctly classified samples by source model should not be used for transferability and that should reflect in the quantitative data presented under transferability. 

3. How essential is the SDXL based augmentation step? Please explain why you need SDXL to build the concept dataset instead of using a large pretrained generative model trained on ImageNet or similar corpora, and include an ablation if possible.

4. How would a black box query attack operate in your framework and how does its expected strength compare to the transfer based evaluation you report?

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
4

### Summary
The paper extends the probablistic framework for adversarial examples presented in Zhang et al. (Neurips 2024) to an unrestricted, concept based setting. The basic idea relies on replacing the image-based specific distance (defined in Zhang et al.). with a generalized concept-based distance. The authors surrogate "concept" with a probabilistic model (diffusion models in this case) by finetuning a pretrained diffusion model on a set of images, and then sample adversarial candidates. They provide two sampling strategies as well as theoretical intuition for the framework. They further showcase empirical results to support their claim that concept priors reduce to an empirical KL gap and produce high quality adversarial examples when compared to other unrestricted approaches like DiffAttack.

### Strengths
1. The presented framework is a clean and well motivated generalization of the probabilistic framework presented in Zhang et al. The idea of moving away from an image-centric distance distribution to a concept-prior through the use of finetuned diffusion models is inspired.

2. Empirical performance of the given approach is encouraging, and the results support the authors' claims of better, and more semantically meaningful adversarial examples as compared to methods like DiffAttack.

3. Implementing the concept-prior using finetuned SDXL models is an interesting approach, and can possibly be generalized to any set of images with common features. This opens up a new mechanism of measuring vulnerabilities of classifiers.

### Weaknesses
1. The theoretical contributions are mostly incremental with both Thm.1 and 2 being straightforward algebra. While supportive of the presented conceptual framework, it does not really provide any additional insight on how the approach can be further optimized or adapted to specific PGMs like diffusion models.

2. The transferability results are extremely low.This suggests very low overlap between $p_{vic}$ and $p_{dis}$ which is a bit counterintuitive given the strong performance of these classifiers on such tasks. Shouldn't a concept based prior have a much stronger overlap?  Do the authors have any hypothesis about this? The top-100 results in the appendix are also uninformative given the weak constraints of the experiment and should be removed.

3. The models presented in this work are also older. How does this attack fair with newer architectures like CLIP, ConvNext etc?

### Questions
1. Please provide more details on the baseline comparisons for DiffAttack and others. Specifically, number of sampling steps, solvers used and any other hyperparameters.

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
The paper introduces Concept-Based Adversarial Attacks, a novel framework that extends traditional image-based adversarial attacks by operating not on a single image but on an entire concept represented either by a set of images or a probabilistic generative model (e.g., a fine-tuned diffusion model). The method adopts a probabilistic perspective, where adversarial examples are sampled from the product of a “victim” distribution (encouraging misclassification) and a “distance” distribution centered on the concept rather than a single image. This allows generation of diverse adversarial examples that preserve the underlying identity or object (e.g., a specific dog) while varying pose, background, or viewpoint to fool classifiers. The approach is theoretically grounded in KL divergence analysis and empirically validated on ImageNet, showing higher attack success rates and better concept preservation than prior methods.

### Strengths
The main contributions (and strengths) of the paper are:

* Novel formulation: First work to define adversarial distance at the concept level rather than per-image, enabling more semantically meaningful attacks.

* Strong empirical results: Achieves state-of-the-art targeted attack success rates (e.g., 97.82% white-box on ResNet-50) while better preserving concept identity (validated via user studies and CLIP scores).

* Theoretical justification: Provides analysis showing that expanding the distance distribution to a concept reduces KL divergence between victim and distance distributions, increasing overlap and attack efficacy.

* Practical pipeline: Integrates modern generative models (Stable Diffusion XL + LoRA) and LLMs (GPT-4o) for concept augmentation, enhancing diversity and realism.

### Weaknesses
Computational cost: The proposed approach requires fine-tuning generative models per concept, which is time-consuming (≈8 hours/concept) and limits scalability.

*Limited transferability. While the experimental results show strong performance on white-box attacks, black-box transfer success remains low (though better than baselines), especially under strict top-1 metrics.

* Concept definition ambiguity. The proposed Relies on user-provided image sets or fine-tuned models to define a “concept,” which may vary in quality or scope and lacks a formal mathematical definition.

* Ethical risk. The proposed approach enables highly realistic, identity-preserving attacks that could evade content moderation systems, raising safety concerns. While the authors discuss mitigations, it’s unclear how these would deter malicious actors. I recognize the importance of publishing sensitive research to foster community dialogue, but is merely discouraging harmful use sufficient when the technique itself carries demonstrable ethical hazards?

### Questions
Please refer to the section above.

### Soundness
3

### Presentation
4

### Contribution
3
