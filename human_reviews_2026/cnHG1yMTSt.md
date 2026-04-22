# Towards Adversarially Robust VLMs with an Information-Theoretic Approach

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Vision–Language Models (VLMs) derive their zero-shot ability from tight alignment between image and text representations, which can be viewed through the lens of mutual information (MI). This alignment is fragile: VLMs are vulnerable both to subtle pixel-level adversarial attacks and to typographic attacks in which overlaid text hijacks predictions. Existing defenses are isolated solutions, relying on proxy objectives tailored to each threat. We argue that both attack types share a single failure mechanism, the reduced cross-modal MI by threats; and we propose an information-theoretic framework that directly prevents this root cause of adversarial attacks on multi-modalities. We first prove a bound that links adversarial risk to the MI gap, defined as the reduction in MI between clean and perturbed image–text views. Building on this, we derive a practical, differentiable objective that minimizes an upper bound on the MI gap using a neural MI estimator, yielding a single, attack-agnostic training scheme. Empirically, our method improves robustness to both pixel-space and typographic attacks at the same time, surpassing specialized state-of-the-art defense methods while maintaining high accuracy on clean inputs. These results show that explicitly preserving cross-modal MI is a principled and effective path to robust VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the vulnerability of Vision-Language Models (VLMs) to both pixel-space and typographic adversarial attacks. The authors posit that both attack modalities share a "single failure mechanism" or "root cause": the reduction of cross-modal mutual information (MI) between image and text representations.
To counter this, the paper introduces "InfoGap," a training framework designed to directly minimize the "MI gap" between clean and perturbed image-text views. The method involves deriving a theoretical upper bound on the adversarial risk, which is linked to this MI gap.
A practical, differentiable objective is then proposed, which uses a neural MI estimator and a discriminator (trained in a GAN-style) to minimize an upper bound of this MI gap.
The authors claim that this "single, attack-agnostic training scheme" improves robustness against both attack types simultaneously, surpassing specialized state-of-the-art defenses in each category.

### Strengths
1. The authors claim that this "single, attack-agnostic training scheme" improves robustness against both attack types simultaneously, surpassing specialized state-of-the-art defenses in each category.
2. Principled Theoretical Grounding: The attempt to base this unified defense on a fundamental, first-principles concept (Mutual Information) is theoretically appealing. This moves beyond ad-hoc proxy objectives and seeks a more fundamental "root cause".

### Weaknesses
1. Oversimplification of the "Single Root Cause" Premise. The paper's theoretical foundation rests on the strong claim that both pixel and typographic attacks share a "single failure mechanism" of MI reduction. This is an oversimplification.
2. The comparative methods are very limited, with only three included. It is recommended to add PMG-AFT[1], Sim-CLIP[2], and others.
3. No experimental results at higher attack intensities.
4. There are no results under non-CLIP models, such as SigLIP.

[1] Pre-trained model guided fine-tuning for zero-shot adversarial robustness
[2] Sim-clip: Unsupervised siamese adversarial fine-tuning for robust and semantically-rich vision-language models.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Existing defense methods for VLMs aimed at pixel-level adversarial attacks and typographic attacks as separate problems.
This paper argues that both attack types share a single failure mechanism: the reduced cross-modal MI by threats.
This paper proposes a unified defense method, InfoGap, to tackle both threats in a unified framework, based on information-theoretic framework.

### Strengths
- **Unified Theoretical Framework:** The paper presents a principled information-theoretic formulation that jointly addresses two distinct threat types, pixel-level and typographic attacks, under a single conceptual lens.
- **Low Overhead and Practicality:** The proposed method introduces less than 1% additional parameters and requires no extra ViT passes, making it computationally efficient and easy to integrate into existing VLM pipelines.

### Weaknesses
- **Lack of Cross-Threat Evaluation:** The claim that existing defenses specialize in either pixel-level or typographic attacks is not empirically demonstrated. Evaluating pixel-space defenses (e.g., FARE) on typographic perturbations, and vice versa, would strengthen the argument for a unified underlying mechanism.
- **Separate Models for Each Threat:** Although the proposed framework is theoretically attack-agnostic, the experiments train separate models for pixel and typographic attacks. This weakens the practical claim of unification. Demonstrating simultaneous robustness to both threats within a single model would significantly enhance the contribution.
- **Adaptability of Existing Methods:** While the authors argue that existing defenses are threat-specific, it remains unclear why frameworks such as FARE could not be adapted to cover both perturbation types simply by redefining the perturbation objective.
- **Assumption on Conditional MI (ϵ):** The assumption that ϵ = I(L; Z_adv | Z) ≈ 0 does not always hold. Although it may be valid for label-agnostic pixel attacks or randomly chosen typographic overlays, it can break down if the attack mechanism depends on class labels or semantics. The ϵ seems to be small only if trained on training set with large number of classes, like ImageNet-1k.
- **Limited Theoretical Scope:** While the study focuses on VLMs, the theoretical formulation assumes a discrete label space and class prototypes, restricting applicability to zero-shot classification tasks rather than broader multimodal problems (e.g., VQA, captioning, or retrieval). This limits the potential impact on general VLM robustness.
- **Notation Clarity:** The notation $\hat{L}$ appears undefined (possibly meant to denote the predicted or true label). Consistent notation would improve readability.

### Questions
- Existing Adversarial Training methods mostly provide unified strategy for different attacks: altering the attack budget in the formulation would be sufficient. Please correct me if I am wrong.

### Soundness
3

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
This paper proposes an information-theoretic framework to improve the adversarial robustness of vision–language models (VLMs). The authors argue that both pixel-level and typographic attacks share a common failure mechanism — the reduction of cross-modal mutual information (MI) — and derive a theoretical bound connecting adversarial risk to the MI gap. Based on this insight, they propose a differentiable objective that minimizes the MI gap using a neural MI estimator, thereby defending against multiple attack types simultaneously. Experimental results show improved robustness to both pixel-space and typographic attacks while maintaining high clean accuracy.

### Strengths
1. The paper provides strong theoretical motivation, grounding the defense mechanism in mutual information theory rather than ad hoc robustness objectives.

2. The proposed approach is attack-agnostic, addressing both pixel-level and typographic attacks in a unified manner.

3. The paper includes comprehensive experimental results that demonstrate consistent improvements in robustness over specialized baselines.

### Weaknesses
1. The evaluation includes only white-box attacks. It would strengthen the paper to include black-box or query-based attacks to validate generalization.

2. Experiments are limited to image classification tasks, without testing on more complex vision–language benchmarks such as retrieval or captioning, where MI dynamics might differ.

3. There are some citation formatting errors: the references in the text are inconsistently formatted (missing spaces after citation markers), which detracts from overall polish.

### Questions
1. Could the authors include results under black-box or real-world transfer attacks to better assess robustness?

2. It would be interesting to analyze whether the learned MI-preserving representations generalize to downstream multimodal reasoning tasks instead of image classification tasks only.

3. Please revise the citation formatting to follow standard style (e.g., space after reference number).

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
In this paper, the authors introduce InfoGap, an information-theoretic approach that improves the robustness of vision–language models like CLIP through additional mutual-information–based fine-tuning. They show that both pixel-level and typographic attacks reduce the mutual information (MI) between image and text embeddings and derive a bound linking this MI gap to adversarial risk. InfoGap minimizes this gap using a learnable MI estimator, providing a unified, attack-agnostic defense. Experiments demonstrate improved robustness to diverse attacks with minimal loss of clean accuracy, surpassing specialized baselines.

### Strengths
* The paper tackles an important and timely challenge - improving adversarial robustness in vision–language models (VLMs).

* The proposed InfoGap method is conceptually simple and broadly applicable, unifying defenses against both pixel-level and typographic attacks through an information-theoretic objective.

* The paper is well written and easy to follow, and the core idea - preserving cross-modal mutual information to ensure robust alignment - is both intuitive and theoretically well motivated.

### Weaknesses
* The proposed approach does not account for adaptive attacks that explicitly optimize to both minimize the MI-gap surrogate and maximize flip class predictions — for example, an attack that jointly (a) maximizes a label-flip/targeted loss and (b) minimizes the learned MI surrogate or fools the discriminator. Consideration and evaluation of such adaptive attacks are crucial for a proper assessment of the proposed defense and its effectiveness.

* InfoGap fixes the text encoder and relies on a precomputed bank of text prototypes, which may limit robustness in open-vocabulary or prompt-sensitive settings. The sensitivity to prompt templates and unseen class descriptions is not evaluated and could undermine zero-shot performance. Moreover, the applicability and generality of the pretrained vision–language models are significantly restricted in the considered setup.

### Questions
* have the authors considered adaptive attacks that explicitly optimize against the proposed objective—for instance, attacks that both maximize class-flipping loss and minimize the MI surrogate or fool the discriminator? Evaluating such attacks would provide a clearer picture of the true robustness of the method.

* How does the method perform in open-vocabulary where the set of text prototypes is not fixed? Could the authors clarify whether InfoGap can be extended to dynamically handle unseen classes or alternative prompt templates?

* Can the authors provide further insight into how dependent the approach is on the chosen prompt formulations and whether robustness or alignment degrades when prompt templates vary or when using paraphrased text descriptions during inference?

### Soundness
2

### Presentation
3

### Contribution
2
