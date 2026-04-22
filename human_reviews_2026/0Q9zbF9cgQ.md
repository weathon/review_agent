# Backdoor Unlearning By Linear Task Decomposition

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Foundation models have revolutionized computer vision by enabling broad generalization across diverse tasks. Yet, they remain highly susceptible to adversarial perturbations and targeted backdoor attacks. Mitigating such vulnerabilities remains an open challenge, especially given that the large scale of the models prohibits retraining to ensure safety. Existing backdoor removal approaches rely on costly fine-tuning to override the harmful behavior, and can often degrade performance on other unrelated tasks. This raises the question of whether backdoors can be unlearned without compromising the general capabilities of the models. In this work, we study how backdoors are encoded in the model weight space and find that they are *disentangled* from other benign tasks. Specifically, this separation enables the isolation and erasure of the backdoor's influence on the model's weights with minimal impact on clean performance. Building on this insight, we introduce a simple unlearning method that leverages such disentanglement. Through extensive experiments with CLIP-based models and common adversarial triggers, we show that, given the knowledge of the attack, our method achieves approximately perfect unlearning, while retaining on average 96\% of clean accuracy. Additionally, we demonstrate that even when the type of attack is unknown, our method successfully unlearns backdoors by proper estimation using reverse-engineered triggers. Overall, our method consistently yields better unlearning and clean accuracy tradeoffs when compared to present state-of-the-art defenses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a defense method named TBAR, targeting backdoor attacks against Contrastive Language–Image Pretraining (CLIP) models under contrastive learning settings. The core idea is to regard model parameters as a vector composed of two components: one representing benign functionalities and the other representing backdoor functionalities. The defense aims to identify and remove the latter while preserving the former.
The method assumes that the poison function parameters vector is kind of “perpendicular to” to the vector of benign function parameters. So that by fine-tuning the model with poisoned samples, the parameter vector will shift toward the direction of the poison function vector, enabling estimation and elimination of that backdoor function.
This method assumes the defender have the knowledge of trigger or poison samples. When poison samples are unavailable, the defense relies on the DECREE, which attempts to discover potential triggers by reverse-engineering the trigger using benign samples. Once such triggers are identified, the same procedure is applied to estimate and eliminate the poison vector.

### Strengths
1．The paper explores an interesting and relatively under-explored direction of parameter space disentanglement between benign and backdoor functionalities.
2．When its assumptions hold (i.e., trigger patterns are known or can be recovered), TBAR could, in principle, handle a wide range of triggers and backdoor attacks.
3．The paper is well written and easy to follow.

### Weaknesses
1．Unrealistic assumption: The method strongly depends on prior knowledge of trigger patterns or  poison samples, which limits its real-world applicability.
2．Dependence on DECREE: The alternative pathway to acquire this knowledge relies entirely on DECREE, which, similar to Neural Cleanse, can only detect superficial, sample-agnostic triggers. Thus, TBAR essentially inherits the limitations of trigger-discovery-based defenses and practically, lacks intrinsic capacity to handle more adaptive or feature-level backdoors. Besides, the DECREE also require a set of benign samples, which is still kind of a strong assumption.
3．The process of determining the subtraction weight α is empirically described but lacks theoretical justification.
4．Limited attack coverage: Only outdated attacks (BadNets, Blended, WaNet) are evaluated. Would be more persuasive to include more recent and adaptive backdoor attacks for fair comparison.
5．Missing descriptions: The attack pipeline for pretrained models is not described, therefore may obfuscate reader from fully understand this work.

### Questions
1．Can TBAR itself indicate whether a model is backdoored, or does it fully rely on DECREE for that signal?
2．How robust is the method when the model capacity is limited? Intuitively, independent functions can decouple in model parameter perspective is because the model capacity is enough to hold such information, can poison and clean vectors still be decoupled in smaller networks where model capacity is not enough to fully absorb them?
3．How sensitive is the result to the value of α (the weight applied on poison parameter vector), as the value is chosen empirically?
4．Are there plans to evaluate the defense against newer and more stealthy backdoor attacks?
5．Table captions are placed incorrectly (should appear below, per ICLR style).
6．Appendix A.3.5,  B.5 contains a table without a caption.

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
This paper proposes a new backdoor defense, TBAR (Trigger removal by Backdoor ARithmetic), for removing backdoors from large vision-language foundation models such as CLIP. The key insight is that backdoor behaviors are linearly disentangled from benign tasks in the model’s weight space. Leveraging this property, the authors fine-tune the model on a small set of triggered samples to estimate a trigger vector that represents the malicious direction in parameter space, and then subtract it to “unlearn” the backdoor.

### Strengths
- The proposal to view backdoor behavior as an independent task direction in the weight space is an inspiring insight that naturally combines model editing with security defense.

- TBAR requires only a small-scale fine-tuning and does not require retraining.

- The presentation of the paper is easy to follow.

### Weaknesses
- The number of evaluated backdoor attacks is small. Only BadNet, Blend, and WaNet (BadCLIP in some cases).

- The weight disentanglement demonstration in Figure 2 is not convincing, because it only includes one attack, BadNet. 

- The performance drop cannot be ignored in Table 3. In some cases, it exceeds 5%.

- In a trigger-unknown setting, the proposed defense relies on other trigger reverse engineering methods.

### Questions
Thanks for the interesting paper. I have a few questions and suggestions.

- The part about known-trigger settings (Section 5) takes the major experiments in the paper. However, it is not practical for the defender to know the trigger. It significantly weakens the contribution of this paper. I suggest that the authors only use the known-trigger setting to analyze the weight disentanglement phenomenon and shorten this section. The experiments should focus more on the setting without explicit knowledge of the attack.

- The section about the weight disentanglement Hypothesis is not necessary. As you provide experimental evidence, you can write it as a phenomenon, and shorten the text to make it more straightforward.

- In the Agnostic attack section, why not try other trigger reverse engineering methods than DECREE?

- Figure 2 is a bit unclear to me. Does it mean that the parts of the model with high $\xi (\alpha_c, \alpha_t)$ values are related to the backdoor? Is it possible to make the backdoor more obvious in another way? For example, Figure 2 in [A] shows that the backdoor-related neurons are prominent than others, or Figure 1 in [B] (although this one is the feature space).

[A] Towards Backdoor Stealthiness in Model Parameter Space

[B] Revisiting the Assumption of Latent Separability for Backdoor Defenses

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
3

### Summary
The paper proposes TBAR, a lightweight backdoor unlearning method for vision-language pretrained models like CLIP. It leverages the observed disentanglement of backdoor and clean knowledge in weight space: by fine-tuning on a small set of triggered samples to estimate a “trigger vector,” the method subtracts this vector to surgically remove the backdoor. Experiments show TBAR achieves high backdoor unlearning performance while preserving ~96% clean accuracy.

### Strengths
1. The paper provides compelling evidence that backdoor behaviors in CLIP are linearly disentangled from clean tasks in weight space, a non-trivial finding that enables precise surgical removal.
2. TBAR requires only few triggered samples to achieve near-complete unlearning, using less data than clean-data fine-tuning defenses while yielding better CA/ASR trade-offs.
3. The paper is well-structured, and the experimental evaluation is thorough.

### Weaknesses
1. The proposed method should be clearly distinguished from existing approaches such as Anti-Backdoor Learning (ABL) [1], which shares part of the unlearning pipeline. Extensive experimental comparison would help clarify the novelty and relative advantages.
2. How clean-task performance is preserved when directly subtracting model parameters? In related fields such as model merging, naïvely combining parameters from different models often degrades performance. The authors should provide a more detailed justification or analysis for why this operation does not harm utility in their setting.
3. How is α determined in practice? If improperly set, could it severely degrade either clean accuracy or backdoor removal efficacy?
4. In Step 3 of Figure 1, the method relies on “similarly constructed triggered data” to estimate the backdoor direction τₜ. However, it is unclear how such triggered samples are obtained in practice—especially in a realistic threat model where the attacker’s trigger may be unknown. Clarifying the accessibility and construction of this data is essential, as it directly impacts the feasibility and applicability of the proposed approach.

[1] Li Y, Lyu X, Koren N, et al. Anti-backdoor learning: Training clean models on poisoned data[J]. Advances in Neural Information Processing Systems, 2021, 34: 14900-14912.

### Questions
Address the weakness  above

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates methods for removing backdoors in the model parameter space of vision foundation models. The authors claim that benign and backdoor behaviour are reflected in the model weights, where different components are linearly separable. With this intuition, the authors propose a parameter-space backdoor removal method, Trigger removal by Backdoor ARithmetic (TBAR). TBAR operates in the parameter space by conducting task negation, which is inspired by and based on the model edit. Several experiments on different datasets show the effectiveness of the TBAR.

### Strengths
This paper is well presented. The idea of backdoor defense by model editing is promising.

### Weaknesses
Backdoor baselines are weak. The paper primarily compares TBAR with basic fine–tuning–based unlearning and several standard backdoor defense methods. These baselines are relatively weak compared with the state-of-the-art backdoor defenses, such as [1]. Without including stronger and more diverse baselines, it is difficult to assess how much of TBAR’s advantage comes from its intrinsic effectiveness.

Adaptive backdoor attacks are not considered. The evaluation does not include adaptive or defense-aware attackers who might deliberately design entangled or non-linear backdoor patterns to resist task-vector subtraction. Since TBAR relies on the assumption that the backdoor and clean tasks are linearly separable in weight space, a knowledgeable adversary could craft more integrated triggers that invalidate this assumption. The absence of such adaptive attack scenarios leaves uncertainty about TBAR’s robustness in real-world adversarial settings, where attackers can adapt to the defense strategy. At least, TBAR needs to take stealthy parameter space backdoor attacks into account, such as [b].


The connection with machine unlearning is good, but somewhat far-fetched. The paper positions TBAR as a form of machine unlearning, but the connection is mainly conceptual. Traditional machine unlearning focuses on data-level forgetting with formal guarantees of data removal. In contrast, TBAR performs parameter-space editing by subtracting a task vector without guarantees.

[a] Towards Reliable and Efficient Backdoor Trigger Inversion via Decoupling Benign Features. ICLR 2024.

[b] Towards Backdoor Stealthiness in Model Parameter Space. CCS 2025.

### Questions
Please discuss the potential of TBAR against stronger backdoors, including adaptive backdoor attacks.
Please discuss other stronger baselines. 
Please adjust and articulate the connection with machine learning.

### Soundness
2

### Presentation
3

### Contribution
2
