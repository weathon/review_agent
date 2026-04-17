# BadReward: Clean-Label Poisoning of Reward Models in Text-to-Image RLHF

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning text-to-image (T2I) models with human preferences. However, RLHF's feedback mechanism also opens new pathways for adversaries. This paper demonstrates the feasibility of hijacking T2I models by poisoning a small fraction of preference training data with natural-appearing examples. Specifically, we propose BadReward, a stealthy clean-label poisoning attack targeting the reward model in T2I RLHF. BadReward operates by inducing feature collisions between visually contradicted preference data instances, thereby corrupting the reward model and subsequently compromising the T2I model's integrity. Unlike existing dirty-label alignment poisoning techniques focused on single (text) modality, BadReward is independent of the preference annotation process, enhancing its stealth and practical threat. Extensive experiments on popular T2I models show that BadReward can consistently guide the generation towards malicious outputs, such as biased or violent imagery, for targeted concepts. Our findings underscore the amplified threat landscape for RLHF in multi-modal systems, highlighting the urgent need for robust defenses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper present a attack on text-to-image (T2I) models with textual triggers to control certain visual concept feature in the generated image, by poisoning the training data of the RLHF stage. The text prompt in poisoned sample is constructed either by sampling prompts from the training dataset that contain the target text trigger, or with a text generation model from the text trigger. The poisoned images are generated with a image generation model from the text trigger, where one is prompted to contain the visual attribute specified by the trigger, and the other to have the "negation" of the attribute. The image with the target attribute is used as the latent target in optimizing an image that fits the poisoned text prompt by aligning the CLIP feature of the image with that of the image with the target attribute, while keeping the visual similarity.

### Strengths
1. The paper explores clean label training data attacks during the RLHF stage of text-to-image models.
2. The proposed attack is evaluated against multiple image generators used by adversaries.
3. The paper has relatively thorough ablations with respect to RLHF steps and poison rates.

### Weaknesses
W1. In line 239
> To evade detection and further refine the attack...

The paper lacks discussion about what the detections are before this line. 

W2. In the ATTACK GENERALITY section, the authors show that synonyms to the text triggers will lead to similar ASR, and treat the phenomenon as a strength of the propose attack. In my opinion, the lack of control over the triggers is a weakness rather than a strength of an attack.

W3. While Table 1 presents ASR results with respect to multiple image generators, the analysis lacks discussion on the high variance of the results in the main text.

W4. If could be helpful if the authors can clarify the prompt construction process described in the experiment section in the methodology section.

W5. In section C.2, it appears to say Stable Diffusion v1.4 is trained with DDPO but  SD Turbo is trained with SDPO. This contradicts with the authors claim to study the attack with respect to different RLHF algorithm, as the target model is not controlled.

### Questions
Q1. In Equation 4, the plus operator is overloaded with both image and text inputs, can the authors give a formal definition of the operator?

Q2. In line 210, 
> The adversary selects a trigger-concept pair (t, C) where the clean target model exhibits a certain probability of generating images containing concept C given natural prompts containing trigger t. 

Can the author explicitly define "a certain probability"? Does the attacker need to test with many candidate triggers to find a suitable trigger? Is the "clean target model" before or after RLHF?

Q3. 
> We adopt Clip-ViT-L/14 as the encoder backbone, encoding images and text into embeddings. 

Is the same image encoder used in the target image refining process?

Q4. How many steps is typically used in the RLHF training process? Is it in the regime where the attack has good ASR performance?

Q5.
> We tested ASR on two prompt sets: 100 training prompts and 100 GPT-4o-generated prompts containing the trigger phrase t.

Q6. Are the training prompts the prompts paired with poisoned images?

Q7. What is the experiment configuration of Table 2? I cannot map the results to the ASR in Table 1.

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
The paper proposes BadReward, a clean-label poisoning attack targeting the reward model used during RLHF in text-to-image diffusion model. The attack works by creating feature-collision poisoned images, which remain visually similar to benign images while being semantically aligned with target malicious concepts in CLIP embedding space. When these poisoned examples are included in preference data, the reward model learns to assign higher scores to outputs containing the target concept whenever a chosen trigger words appear.

### Strengths
1. The attack does not modify preference labels or require control of annotators, which is low-cost.

2. The attack pipeline and optimization objective are well explained and easy to reproduce.

3. The presentation is well and clear.

### Weaknesses
1. Trigger–concept selection is underspecified: The method implicitly relies on choosing trigger–concept pairs that already have some representation overlap in the model’s data distribution. This selection procedure is not formalized, and success may vary across concepts.

2. Novelty: BadReward extends existing clean-label feature-collision poisoning to the reward modeling stage rather than introducing a fundamentally new poisoning mechanism.

3. The paper does not analyze the conditions under which CLIP feature similarity reliably transfers to reward-driven policy updates. 

4. The paper only tested the method on SD v1.4 or SD Turbo. More target models should be included for testing.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
The paper introduces BADREWARD, a clean-label poisoning attack against reward models in text-to-image (T2I) RLHF systems. BADREWARD induces feature collisions in the CLIP embedding space to corrupt the reward model, enabling adversaries to steer T2I model outputs toward targeted malicious concepts when specific triggers are present in prompts. Notably, the attack does not require control over annotations, instead relying on stealthy manipulation of a small fraction of preference training data. Experimental evaluation on Stable Diffusion (v1.4 and Turbo) and several adversarial model generators demonstrates high attack success rates, notable stealthiness by visual and perceptual metrics.

### Strengths
1. The black-box scenario assumes the adversary can inject even a small fraction of poisoned pairs into the RLHF pipeline. In many real-world alignment pipelines, this step is subject to curation, filtering, or annotation review, which could detect subtle distribution shifts. The paper does not investigate the sensitivity of system-level detection to these injections.
2. While Table 2 and Figure 4 demonstrate strong SSIM/LPIPS results, stealth evaluation is reduced to pixel-level or shallow perceptual metrics. The paper does not assess detectability by automated anomaly detection methods or statistical audit systems that could flag poisoned distributions in embedding or reward space.
3. Reward models are built on a fixed CLIP backbone with a simple MLP. There is little discussion of how architecture choices, frozen vs. trainable backbone, or reward model complexity affect the attack’s transferability/durability.

### Weaknesses
1. Can the authors provide quantitative comparisons with recent SOTA RLHF poisoning attacks in terms of both effectiveness and detectability?
2. Have the authors evaluated whether feature-collided samples can be detected by statistical anomaly methods operating in embedding, reward, or preference score space, beyond pixel-perceptual similarity metrics?
3. Does BADREWARD generalize to subtler forms of steering (e.g., more abstract style or concept changes), or is it reliant on visually salient feature collisions?
4. Can the authors provide more insight or results on how different choices of the $\beta$ parameter and initial images ($x_b$, $x_t$) affect stealth and ASR?

### Questions
Refer to the weaknesses.

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
4

### Summary
This work introduces a method for compromising the reward model used in the post-training of T2I systems. The attack injects explicit and adversarially crafted poisoned training examples, causing the T2I model to generate harmful or inappropriate content upon encountering specific trigger prompts.

### Strengths
Reward-model poisoning in T2I RLHF is underexplored relative to SFT-time poisoning; the paper clearly motivates why RLHF is a sensitive surface.

### Weaknesses
1. **Reward-model diversity** The **Feature-Level Poisoning Attack** is evaluated in a CLIP-on-CLIP setting: poisons are crafted with **white-box** access to a CLIP encoder and tested on CLIP-based reward models, which lowers the difficulty and obscures generality and makes the gray-box and black-box threat model overclaimed. Please evaluate on non-CLIP reward backbones (e.g., BLIP, HPSv2, ImageReward), test reward ensembles and multi-reward optimization/confidence-aware training, and report ASR to show the effectiveness of the Feature-level Poisoning Attack.

2. **Lack of evaluation under basic defenses.** Missing evaluation against basic safety defenses. The RLHF setup optimizes semantic/aesthetic alignment rather than safety, so the attacked T2I models are effectively unguarded. In practice, providers deploy baseline defenses—harmful-concept erasure, safety-focused RLHF, and runtime safety filters (e.g., the SD-1.5 Safety Checker, Q16). Please evaluate BadReward under these defenses (individually and combined) and report ASR and utility on benign prompts. If standard guardrails block the attack, its practical impact is limited; if not, show how the method bypasses them.

3. **less informative citations** The claim in L39 that “RLHF is an indispensable component for aligning T2I systems with human expectations” is unsupported. Please provide primary evidence—e.g., technical reports or production case studies from major T2I providers, or peer-reviewed ablations—demonstrating indispensability. Likewise, using broad surveys (e.g., Zhu et al., 2024 at L42) to justify specific limitations of RLHF is insufficient; cite targeted primary sources that document these limitations in T2I settings. If such evidence is unavailable, soften the claim (e.g., “commonly used” or “increasingly adopted”) and reposition the motivation accordingly.

### Questions
Please refer to the weakness section and:

Experimental setup — missing details


1.**Reward-model training**
Specify what is updated: full backbone vs. MLP head only. Clarify any frozen layers, parameter count, and basic optimizer settings (lr, epochs, regularization).

2. **Poison budget**
Report the number and percentage of semantic-level poisoned pairs used for reward training, plus the mixing schedule per epoch (and whether poisons appear in validation).

3. **Trigger coverage**
Why only three triggers? Please report ASR as the number and diversity of triggers increase (including synonyms/paraphrases/typos) and provide variance across seeds to show robustness.


4. What is the minimum effective poison budget across targets and training seeds for stable success (1%, 2%, 3% are shown)? Can you report variance across seeds?

### Soundness
2

### Presentation
4

### Contribution
3
