# PersGuard: Preventing Malicious Personalization in Text-to-Image Diffusion via Model Backdoors

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Diffusion models (DMs) have achieved remarkable success in text-to-image (T2I) generation, yet their personalization capabilities pose serious privacy and copyright risks. Existing protection methods primarily rely on adversarial perturbations, which are impractical in realistic settings and can be easily bypassed when inputs are mixed with clean or transformed data. In this work, we propose PersGuard, a novel model backdoor-based framework to prevent unauthorized personalization of pre-trained T2I diffusion models. Unlike perturbation-based approaches, PersGuard embeds protective backdoors directly into released models, ensuring that fine-tuning on protected images triggers predefined protective behaviors, while unprotected images yield normal outputs. To this end, we formulate backdoor injection as a unified optimization problem with three objectives, and introduce a backdoor retention loss to withstand downstream personalized fine-tuning. Extensive experiments across comparative and gray-box settings, as well as multi-identity scenarios, demonstrate that PersGuard delivers stronger and more reliable protection than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel defense framework against unauthorized personalization of diffusion models (e.g. DreamBooth). Instead of perturbing input data, PersGuard consists of protective backdoors embedded directly into pre-trained models, enabling them to trigger predefined protective behaviors, like erasing, replacing, or marking protected content, when fine-tuned on restricted images, while generating normal outputs otherwise. The backdoor embedding is formulated as a unified optimization problem with three objectives:
backdoor behavior, prior preservation, and backdoor retention losses, ensuring robustness against downstream fine-tuning. Experimental evaluation demonstrates that PersGuard outperforms perturbation-based defenses in both protection reliability and generation quality.

### Strengths
Here are the strengths of the paper:
- it addresses a very important topic represented by the privacy protection against unauthorized personalization
- the paper is clearly written and easy to follow
- related work covers most of the relevant approaches
- experimental results show superior performance in comparison with perturbation-based defenses in both protection reliability and generation quality

### Weaknesses
Here are the weaknesses of the paper:
- The criticism to perturbation-based methods is overstated, like in the following sentence: "...unrealistic assumption that all images in the training dataset of malicious users are pre-perturbed by the protector". The authors should argument why they consider this assumption unrealistic.
- the following scenario lacks realism: "We propose a more practical scenario: protectors are typically large AI companies that provide
pre-trained generative models or offer personalization services directly to downstream users."
- the experimental validation is limited: the current approach is compared only against two other methods. Only one personalization approach, represented by Dreambooth, is presented.

### Questions
Besides the weaknesses mentioned above, here are my other concerns that should be addressed:
- the first and most concerning one is that the current paper relies heavily on the previous BadT2I approach (Zhai et al., 2023): it considers exactly three types of different attacks, but with different objectives (targets). Therefore, the authors should criticallly discuss how their approach is different from the BadT2I method and clearly state their scientific contributions.
- Section 3.2, lines 189-190: the authors make the following statement: "Recent studies have shown that T2I diffusion models are vulnerable to backdoor attacks, where adversaries controlling the training process can embed triggers to achieve malicious objectives." Please add references to support your claim.
- The evaluation against only two baselines (BadT2I and Personalization Shortcut) is insufficient. Additional comparison with other relevant approaches should be included (e.g. SIREN, EvilEdit)
- In order to validate the generality of your approach, you should validate it on other personalization approach, e.g. Textual Inversion, LoRA-based adaptation, etc.
- Rename section 4.3 'Discussion'. The goal of a 'Discussion' section is not to present additional experimental results, but to address several key aspects: to comment on the main findings, explain the insights they provide, relate them to existing work, acknowledge the limitations of the current approach, etc.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a novel defense mechanism against unauthorized personalization and fine-tuning attacks (i.e. DreamBooth misuse) on text-to-image diffusion models. Instead of relying on fragile input perturbations, PersGuard embeds a protective backdoor directly into the model so that when an attacker fine-tunes on protected content, the backdoor is automatically triggered to erase, alter, or watermark outputs. The method jointly optimizes three objectives—Backdoor Behavior (ensuring trigger activation), Prior Preservation (maintaining normal generation), and Backdoor Retention (preventing forgetting during downstream fine-tuning). Experiments were conducted on Stable Diffusion 2.1 across DreamBooth and CelebA-HQ datasets. PersGuard outperforms prior perturbation-based defenses (Anti-DB, PAP, SimAC, DisDiff) and backdoor baselines (BadT2I, Personalization Shortcut), maintaining protection even when only one protected sample is present.

### Strengths
1. The design of $L_{BR}$ is clever, maintaining the robustness of the inplanted backdoor by moving the parameters of the model towards the direction where the backdoor can be erased, so as to minimize the gradient during the post-tuning phases conducted by the malicious user.

2. The writing is good, and the paper is easy to follow.

### Weaknesses
1. **Inappropriate baselines examined**: The paper mainly compares PersGuard with perturbation-based approaches like Anti-Dreambooth, which are originally designed for protecting the published image from being unauthorizedly utilized by the attacker. While the purpose of PerGuard is significantly different, it aims to make the released model unable to be used for personalization fine-tuning. The abilities of the protectors are thereby different as well. In PersGuard, the protector apparently has much stronger capabilities to modify the model weights, while the perturbation-based methods cannot, making the comparison not only unfair but very likely meaningless. The comparison should be conducted between the model-fine-tuning-based methods, such as IMMA [1], Meta-Unlearning [2], and ResAlign [3]. 

2. **Confusing experiment demonstration**: The white-box setting, being the most fundamental one though, is NOT an appropriate setting to show the contribution of PersGuard. The assumption that the protector knows the placeholder (i.e., 'sks') that the attacker would use is too naive, trivial, and impractical. The experimental results under gray-box and black-box settings should be highlighted to better demonstrate the robustness of the proposed method against unseen identifiers and prompts. I thereby suggest that the authors revise the main result part to include more results in these two settings.

3. **Lack of backdoor capacity demonstration**: Only up to 3 backdoors are implanted into the model, but usually, many more concepts need to be protected in the real-world setting. The discussion on the scalability of the method is necessary and would enhance the contribution of this paper.

4. **No scalability discussion**: Only Dreambooth on SD 2.1 was examined. I strongly suggest that the author test more personalization methods, such as Textual-Inversion and LoRA-based Dreambooth. 

In summary, I would believe that although this paper may have proposed a novel solution, the paper itself is still in a very immature state, especially for its problematic experiment design.

***Reference***

[1] Zheng, Amber Yijia, and Raymond A. Yeh. "Imma: Immunizing text-to-image models against malicious adaptation." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[2] Gao, Hongcheng, et al. "Meta-unlearning on diffusion models: Preventing relearning unlearned concepts." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

[3] Li, Boheng, et al. "Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning." The Annual Conference on Neural Information Processing Systems (NeurIPS), 2025

### Questions
See in weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a defense-based backdoor framework to prevent unauthorized T2I personalization. Unlike the existing methods which fail when clean or transformed images are mixed in the training data, the proposed method embeds the backdoor into the model, so that the backdoors activate only when the prompts contain the identifiers. The proposed method introduce three loss functions to achieve the unified optimization objective. The results show that the proposed method outperforms the existing baselines.

### Strengths
1. First thing which makes the difference is that, existing methods assume full control over training data. Whereas, this method  assumes the protector controls only the model, which better aligns with the real-world applications.

2. The method remains effective even when attackers mix protected images with clean or augmented data that is a known failure for existing perturbation-based methods.

3. Among three introduced loss functions, backdoor retention loss seems more meaningful, because backdoor erosion while fine-tuning is a common issue in practice. 

4. Includes white-box, gray-box, multi-identity, and facial privacy experiments, with LLM-based metrics and ablation studies.

5. Protected models maintain FID and DINO scores close to clean models on general and class-specific prompts.

### Weaknesses
1. The paper claims to be “the first to introduce a novel backdoor-based protection approach” (Sec 1). However, backdoor attacks for personalization methods like DreamBooth have already been studied. For example, [1] demonstrated that personalization itself can serve as a “shortcut” for few-shot backdoor injection, and other works have explored backdoors in T2I models via textual triggers or encoder manipulation [2,3]. While PersGuard repurposes backdoors for defense, it builds directly on frameworks like BadT2I (Sec 3.3). Thus, the novelty lies in the defensive application and retention mechanism, not in the concept of backdooring T2I personalization.

2. The target-backdoor replaces a protected human subject with an unrelated object such as “Superman”. This violates the fundamental expectation of personalization that prompts yield semantically coherent outputs consistent with the subject category. If the target is too distant i.e., person → rabbit, the output becomes nonsensical, revealing the defense and failing to respect the true notion of T2I personalization.

3. I think the prior preservation is coming from dreambooth. if its correct, while its reuse is justified, it should not be claimed as a novel contribution.

4. Experiments are conducted on SD 2.1, while the more latest SD variants such as SD 3, and SD3.5 are publicly available. The authors do not justify this choice. Also, their is a significant architectural difference between the SD2.1 and the newer SD variants. The older ones are based on UNET, whereas, the newer ones are DiT-based models.

[1] Personalization as a Shortcut for Few-Shot Backdoor Attack against Text-to-Image Diffusion Models
[2] Injecting Bias in Text-To-Image Models via Composite-Trigger Backdoors
[3] BAGM: A Backdoor Attack for Manipulating Text-to-Image Generative Models

5. [Minor] Though, DAAM is a popular choice for attention maps extraction, it aggregates attention heuristically across layers and timesteps. Directly hooking attentions from cross-attention blocks can give more correct maps.

### Questions
1. How does PersGuard fundamentally differ from these prior efforts beyond the defensive objective?
2. The target-backdoor replaces a protected subject (e.g., a person) with a semantically unrelated object (e.g., “Superman”). Doesn’t this violate the core expectation of personalization that outputs remain coherent with the subject category? How do you reconcile this with the goal of “preserving utility” for legitimate users?
3. The prior preservation loss appears identical to the one used in DreamBooth. Is this component truly a contribution of your work, or is it borrowed directly?
4. The backdoor retention loss is a key innovation. Can you provide ablation results showing how much fine-tuning (e.g., number of steps, learning rate) the backdoor can withstand before degrading?
5. Why was Stable Diffusion 2.1 chosen for all experiments, given that SD 3/3.5 are now standard? Have you evaluated PersGuard on more modern DiT-based architectures, and if not, what are the anticipated challenges?
6. In gray-box settings, the paper shows that universal prompt training improves robustness. But what about black-box scenarios where the attacker uses completely unseen identifiers or prompt templates? Is PersGuard fundamentally limited to white/gray-box assumptions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces the novel idea of utilizing backdoor attack techniques for personalization protection — PersGuard, which injects protective backdoor triggers into released diffusion models. Specifically, the authors add the 'backdoor retention loss' to maintain the backdoor capabilities from downstream fine-tuning tasks.

### Strengths
1. The idea of utilizing the backdoor attack techniques is novel to me.
2. The authors provide several settings to validate the PersGuard's capability, including gray-box settings, multi-objective projection, and facial identity protection.
3. Especially, they test whether the single trigger (e.g., *sks*) would apply to different protected objects, which would adapt to more general privacy cases.

### Weaknesses
1. **Unclear scenarios of gray-box settings.** In gray-box cases, the *universal identifier tokens* and *universal training prompts* are unclear to me. Do you coverage more identifier tokens during the protector stage (as described in Figure 1)? If so, the PersGuard still requires a significant number of identifier tokens (for personalization training), which sounds impractical.
2. **Concerns about comparisons with Perturbation-Based Protections.** In Figure 3 (b), what are the prompts of multimodal LLM to get the Protection Success Rate on the $y$-axis?
3. Follow 2. and your method to convert the personalization identifier to other results. The stealthy of the model looks weak because the personalization fine-tuned models would generate the results far away from the prompt. In contrast, perturbation-based protection methods might develop a more semantically preserved prompt. What do you think?
4. Follow 2., beyond the ablation study of PersGuard on gray-box settings, I suggest the authors should provide perturbation-based protections for different settings for a clearer status of the existing personalization protection methods.
5. **Concerns about the core questions of utilizing BadT2I for this scenario.** While the authors design the 'backdoor retention loss' to maintain the backdoor capabilities, the empirical result in Table 5 shows that the $\mathcal{L}_\mathcal{BR}$ provides minor improvement. Since the ideas of *Backdoor Behavior Loss* and *Prior Preservation Loss* originate from BadT2I, I have some concerns about the core questions the authors address.

### Questions
1. After reviewing the idea of PersGuard, I conjecture that PersGuard cannot remove the existing concept that has already been trained in the pre-trained diffusion model. For example, a public figure such as an actor or a politician. While I know the scope of this work might not coverage this scenario, is that possible to extend this idea in this direction? If so, I think the potential of this direction would be more substantial.

### Soundness
2

### Presentation
2

### Contribution
2
