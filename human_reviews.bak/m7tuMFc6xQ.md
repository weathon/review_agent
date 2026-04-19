# Prompt Backdoors in Visual Prompt Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Fine-tuning large pre-trained computer vision models is infeasible for resource-limited users. Visual prompt learning (VPL) has thus emerged to provide an efficient and flexible alternative to model fine-tuning through Visual Prompt as a Service (VPPTaaS). Specifically, the VPPTaaS provider optimizes a visual prompt given downstream data, and downstream users can use this prompt together with the large pre-trained model for prediction. However, this new learning paradigm may also pose security risks when the VPPTaaS provider instead provides a malicious visual prompt. In this paper, we take the first step to explore such risks through the lens of backdoor attacks. Specifically, we propose BadVisualPrompt, a simple yet effective backdoor attack against VPL. For example, poisoning 5\% CIFAR10 training data leads to above 99\% attack success rates with only negligible model accuracy drop by 1.5\%. In particular, we identify and then address a new technical challenge related to interactions between the backdoor trigger and visual prompt, which does not exist in conventional, model-level backdoors. Moreover, we provide in-depth analyses of seven backdoor defenses from model, prompt, and input levels. Overall, all these defenses are either ineffective or impractical to mitigate our BadVisualPrompt, implying the critical vulnerability of VPL.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes BadVisualPrompt backdoor attack method in Visual Prompt Learning task. The way to attack is construct poisoned data (attaching trigger $\Delta$ to random selected images) and optimize a visual prompt during training. The paper claims that poisoning 5% training data leads to 99% ASR with limited clean accuracy drop. Analysis of different backdoor defenses against proposed BadVisualPrompt is present.

### Strengths
1. Applying backdoor attack into Visual prompt learning is very interesting.

2. The paper is well-written and easy to understand.

3. The paper conduct both attack and defense practices, which is appreciable. 

4. some analysis helps understand the effect of backdoor attack in visual prompt learning task.

### Weaknesses
1. In abstract, the author claimed "we identify and then address a new technical challenge related to interactions between the backdoor trigger and visual prompt". It is a bit vague what is the challenge. Please address it. 

2. For attacking, the proposed BadVisualPrompt **a.** attaches the trigger ($\Delta$, with small patch with iterative white and black colors placed at the corner) to poisoned images, and **b.** optimizes backdoored Visual Prompt. I have two comments here:
+ 1) The b.optimized pattern is a traditional way in backdoor attack (e.g., UAP). The idea is not surprising. 
+ 2) How important is the operation b? We know that simply attaching the trigger $\Delta$ is already sufficient in traditional backdoor attack secario (i.e., image classification in BadNet). Whether solo Operation *a* is already sufficient for a backdoor attack? In another word, the ablation study is needed.

3. In Section3.3, Figure3 and Table2, the author investigates the impact of trigger position. It seems as long as the trigger is inside/overlap the visual prompt, the ASR is high. Can author provide some insights why this happen? My understanding is related to Weaknesses 2: the trigger $\Delta$ is already sufficient enough for attack. The reason that there is low ASR when the trigger is not overlap with Visual prompt, may because involving the visual prompt during training in VPL task to make sure pre-trained label maps to downstream label. 

4. There are a lot of backdoor attack baselines in general visual task (i.e., image classification). Is it possible to apply those baselines to the image? You can simply apply attack on images, without touching the visual prompt. Would the baseline attack be already sufficient?

5. The paper investigates both attack and defense, as the paper title "PROMPT BACKDOORS IN VISUAL PROMPT LEARNING". However, it looks like neither attack or defense is strong/solid enough.

### Questions
1. In Section 3.4, regarding larger trigger size, the author claims that "when the trigger size is increased to 8 x 8 or the posioning ratio is increased to 15%", the ASR is only around 0.33. Why this happen? seems not natural in default backdoor attack setting. (Figure 5 makes sense, that increasing lamba leads to increase ASR and decrease CA.)

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose to implant a backdoor in visual prompt learning (VPL), which was suggested in some recent works as an alternative method to fine-tuning a large pre-trained computer vision model by adding learned perturbations to input data. Both attack and defense experiments were conducted to justify the effectiveness of the proposed backdoor approach.

### Strengths
S1. The experiments are comprehensive and mostly convincing. Specifically, the authors conducted experiments to explore both attack and defense strategies related to backdoors in visual prompts and analyzed the security risks associated with VPL.

S2. From an attack perspective, the authors demonstrated the effectiveness of the attack given a low poisoning ratio and a small trigger size.

S3. From a defense perspective, the authors highlighted the risk of the attack by showing that no effective defenses have been found so far.

### Weaknesses
W1. Weak motivation. The paper assumes that the Visual Prompt as a Service (VPPTaaS) provider is malicious. However, the claim that visual prompts are more resource-feasible can reduce the incentive for a user to rely on a VPPTaaS provider for fine-tuning a model.

W2. Lack of comparison with non-VPL methods, such as implanting the backdoor directly into the training images itself rather than the prompts.

### Questions
In addition to the weakness mentioned above, I wonder why a VPPTaaS provider would insist on injecting a backdoor into VPL instead of normal fine-tuning models, considering a more straightforward usage for the end-users in the latter case. Is there any industrial case to support your claim?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides the first study of security pixel-space visual prompts learning (VPL) from the lens of backdoor attacks. The proposed backdoor attack method the authors proposed, BadVisualPrompt is the first backdoor attack against VPL. They conduct experiments to analyze the impact of trigger-prompt interactions on the attack performance. Besides the attack aspect, they also conduct experiments on existing backdoor detection and mitigation technologies on BadVisualPrompt and show they are either ineffective or impractical. They provide findings and analysis that might inspire further security research in VPL.

### Strengths
This paper is the first study to investigate the security of VPL against backdoor attacks.

The authors conduct extensive experiments to analyze the effect of hyperparameters and show that the distance between the trigger and prompt has a huge influence on the attack's effectiveness.

The paper both investigates backdoor attacks and existing defense.

### Weaknesses
Lack of novelty and challenge. While this paper is the first study of the backdoor attack on the visual prompt, the methodology largely mirrors the data-poisoning backdoor attack approach from BadNet[3]. This technique has already been extensively explored in the context of NLP prompts[5]. The readers may expect to see some challenges that arise with visual prompts learning, but the results show that directly using the conventional data-poisoning method can achieve a successful backdoor attack.

The presented threat model lacks widespread applicability. While platforms like PromptBase are popular for sharing text prompts in language tasks, it would be beneficial for the authors to highlight existing VPPTaaS platforms that share visual prompts.

Research is limited to pixel-space prompts. The study focuses solely on backdoor attacks targeting pixel-space visual prompts, neglecting the token-space visual prompts mentioned in [1,2]. Expanding the research to encompass attacks on token-space visual prompts would provide a more holistic view of the vulnerabilities.

[1] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual Prompt Tuning. In European Conference on Computer Vision (ECCV). Springer, 2022b

[2] Gan, Y., Bai, Y., Lou, Y., Ma, X., Zhang, R., Shi, N., & Luo, L. (2023). Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation. Proceedings of the AAAI Conference on Artificial Intelligence, 37(6), 7595-7603. https://doi.org/10.1609/aaai.v37i6.25922

[3] Gu, Tianyu, Brendan Dolan-Gavitt, and Siddharth Garg. "Badnets: Identifying vulnerabilities in the machine learning model supply chain." arXiv preprint arXiv:1708.06733 (2017).

[4] Chen, Xinyun, et al. "Targeted backdoor attacks on deep learning systems using data poisoning." arXiv preprint arXiv:1712.05526 (2017).

[5] Du, Wei, et al. "Ppt: Backdoor attacks on pre-trained models via poisoned prompt tuning." Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22. 2022.

### Questions
Same as Weakness 2, are there any existing VPPTaas to sell and buy pixel-wise vision prompts?

The authors investigate the influence of the trigger location. The results in Figure 3 show that the trigger in the middle location has a suboptimal attack success rate. So what will happen if the attacker uses a pattern trigger[4] covered on the whole image rather than a patch trigger[3] in the specific location?

The reviewer would consider increasing the score after viewing the rebuttal responses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper performs a systematic study of the security vulnerability of visual prompt learning. An attacker can optimize visual prompts such that when inputs have a trigger and the optimized prompt, this combination causes an intended misclassification. To achieve this attack objective, the paper proposes a data poisoning attack. The paper evaluates the effectiveness of the attack in different settings and shows the attack's success (when the trigger is not in the center of visual inputs). It is not also shown as easy to defeat the attack by existing defenses.

### Strengths
1. The paper presents a backdoor vulnerability of visual prompt learning.
2. The paper runs comprehensive evaluations in various settings.

### Weaknesses
1. It is not that surprising that the backdoor attack works on VPL.
2. It is also not technically surprising that one can compose such a crafting objective.

**Note:** I reviewed this paper from other conferences. I compared this paper from the previous submission and the current submission, and unfortunately, the major concern that quality reviewers raised has not been resolved in the paper. Thus, I am leaning toward borderline rejection.

> To me, given the vast literature on backdoor attacks, this paper seems to be a trivial extension of existing attacks to visual prompt learning. It means that even if the attack works, the fact that "the attack works" is not interesting. So do the results.


[Significance]

The attack is quite similar to existing backdooring. It is equivalent to optimizing a backdoor trigger under a constraint---a visual prompt. It can also be seen as constructing a universal adversarial patch with a backdoor trigger and a visual prompt.

I am also unclear about the practical importance of visual prompts. It's clearly motivated in natural language processing domains, but I am not sure about the goals of using this visual prompt in computer vision domains.


[Technical Novelty]

From another perspective, the paper can be novel when there are some new challenges in backdooring visual prompts. This can add knowledge transferable to attacking similar systems/models. But I couldn't find new challenges (and also the technical proposals to address them). This new backdoor attack just uses data poisoning attacks to achieve the goal.

### Questions
My questions are in the detailed comments in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
