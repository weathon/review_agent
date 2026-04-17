# Jointly Optimized Backdoor Attack against Retrieval-Augmented Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Retrieval-augmented diffusion models (RAG-DMs) have gained widespread adoption across various applications, mitigating the data and compute demands of conventional diffusion models. Despite the success, their trustworthiness remains largely unexplored. Prior backdoor attacks have focused either on manipulating the image generation phase or on compromising the retrieval phase under the white-box setting, and they often suffer from knowledge conflicts between retrieved content and user prompts. To investigate the trustworthiness of black-box RAG-DMs, we propose the first jointly optimized backdoor (JOB) attack tailored to RAG-DMs under the black-box setting, which can jointly manipulate the generation and retrieval phases. Specifically, JOB injects a few target-class poisoned images into the knowledge base and learns simply a trigger through multi-objective optimization, guiding retrieval toward poisoned images and aligning the generated image with the target class while preserving benign performance. Experiments show that our method can effectively attack the black-box RAG-DMs with a high success rate compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Jointly Optimized Backdoor (JOB), a novel backdoor attack specifically designed for retrieval-augmented diffusion models (RAG-DMs) under the black-box setting. JOB introduces a multi-objective reinforcement learning framework that jointly optimizes word-level triggers across three objectives: retrieval success, generation alignment, and linguistic fluency. By poisoning only a small portion of the knowledge base, the method effectively manipulates both retrieval and generation phases to produce attacker-specified outputs while preserving benign behavior. Extensive experiments on multiple RAG-DM architectures and real-world online services demonstrate that JOB achieves high attack success rates and maintains image quality comparable to clean models.

### Strengths
- **Novel and timely contribution**: This work is among the first to explore backdoor attacks on black-box RAG-DMs, a critical yet underexplored threat model for retrieval-augmented generative systems.
- **Technical originality**: The reinforcement learning–based trigger optimization is an elegant approach that effectively bypasses the gradient inaccessibility in black-box environments.
- **Strong empirical performance**: JOB achieves significant improvements in attack success rate while maintaining benign accuracy and image quality, demonstrating its practical effectiveness and stealthiness.

### Weaknesses
- **Limited scope of discussion.** The paper focuses exclusively on class-conditional generation, without discussing open-ended text-to-image synthesis. It is unclear whether the proposed policy network can generalize to unseen prompts or out-of-distribution domains (e.g., novel text queries). Evaluating this would clarify the general applicability of JOB beyond predefined categories.
- **Lack of analysis on query efficiency.** The RL strategy relies on rewards obtained from interactions with a black-box RAG-DM. If the method requires a large number of queries, especially when interacting with commercial APIs, the attack cost could be prohibitive. An ablation study analyzing the relationship between query count, attack success rate, and computational cost would make the approach’s practicality clearer.
- **Insufficient cross-architecture transfer evaluation.** Although the study tests several RAG-DM variants, it does not examine whether triggers optimized for one model can transfer to others with similar architectures or retrievers. Conducting transferability experiments would reveal whether JOB learns model-specific triggers or more generalizable patterns.
- **Potential unfairness in baseline comparison.** The main experiment adopts a trigger length of six tokens, while baselines such as BadRDM use much shorter triggers (e.g., “ab.”), potentially leading to unfair performance comparisons. Moreover, according to Eq. (6), JOB requires inserting the target class $y$ into the final query, whereas baseline methods do not follow this constraint. Aligning trigger lengths and query formulations would ensure a fairer evaluation.
- **Limited defense exploration.** The paper only considers two simple defenses, i.e., retrieval filtering and query rephrasing. To improve comprehensiveness, the authors should also evaluate additional defense strategies such as embedding compression or quantization, back-translation, and LLM-based paraphrase generation. These would help assess the robustness of JOB under more realistic defensive settings.

### Questions
- The proposed method appears conceptually similar to targeted adversarial attacks, where an adversarial perturbation (in this case, the trigger string) is optimized to induce a specific output. Could the authors clarify the fundamental distinction between JOB and conventional adversarial attacks? In particular, what characteristics make JOB qualify as a backdoor attack rather than an adversarial one, given that the optimization is performed at inference time without model retraining or parameter injection? A clearer conceptual boundary between these two threat models would help position the contribution more precisely.
- How would JOB perform if the retriever or generator were periodically updated (e.g., fine-tuned with new data)? Does the optimized trigger retain its effectiveness across model updates, or would it require re-optimization?
- The ASR drops sharply when the trigger length exceeds six. Could the authors provide an explanation for this phenomenon?

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
5

### Summary
This paper proposes a novel backdoor approach targeting RDMs, named JOB, to investigate the backdoor vulnerability under black-box settings. To perform the attack, the authors design a multi-objective reward function to optimize the trigger for successfully retrieving the target images while keeping the text fluency for stealthiness.  The authors provide extensive experiments to demonstrate the effectiveness of JOB, which achieves a satisfactory ASR and outperforms existing baselines.

### Strengths
1. The motivation is clear and straightforward.
2. The writing is good, and the figures are fascinating. 
3. The algorithm design is reasonable and appropriate, where each loss function serves as an important component for the attack effectiveness or text fluency.
4. The authors present sufficient experiments across various models and scenarios, which validate the effectiveness of their methods.
5. I appreciate the significance of this work as it considers a more practical scenario where the adversary cannot access the retriever.

### Weaknesses
While I appreciate that the authors explore the black-box settings of backdoor RDMs and present an effective attack method, I feel uneasy about the trigger design of the proposed method. 

Specifically, the authors incorporate the label of the target class (e.g., banana) into the triggered text. This is somewhat strange and unacceptable to me, as there is generally no information about the attack target within the triggered input in backdoor attacks. This can expose the attacker's goal and incur issues of unfair comparison, since the incorporation of the target label can facilitate the retrieval of target images.

The inclusion of a target label in the input prompt raises another question: if the prompt contains the word “banana” and the generated image depicts only a banana, should this be considered a successful attack, or merely a normal, unsatisfactory generation that neglects other intended objects or details?

Moreover, the trigger is also kind of lengthy, which introduces an additional advantage over existing methods, such as BadRDM.

Considering the above aspects, I'm inclined to reject this paper. However, I'm open to discussing with the authors if they can provide convincing clarifications or additional evidence to change my view.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies backdoor attacks on retrieval-augmented diffusion models (RAG-DMs) under a 'black-box' setting. The proposed JOB method jointly targets both retrieval and generation by (1) injecting a small set of poisoned images into the external knowledge base, and (2) optimizing a textual trigger via a reinforcement-learning-based multi-objective loss that promotes poisoned-image retrieval, target-aligned generation, and linguistic fluency. Experiments on RDM (PLMS/ DDIM) and commercial text-to-image services show good attack success rates than prior methods, while preserving benign utility and image quality.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed RL-based optimization is a natural and reasonable solution.

### Weaknesses
1. **Problematic 'Blackbox' Claiming**: The paper incorrectly characterizes its setting as black-box. In fact, all experiments are conducted on the same RAG pipeline, and the only “black-box” component is the text-to-image model. This is essentially the same setting as BadRDM. The only distinction is that the authors avoid using gradient information from CLIP, but not using gradients does not make the system black-box. The attacker still knows the entire RAG mechanism, including the architecture, tokenizers, and word-embedding dictionary of the CLIP-based retriever. Therefore, the setting should be considered white-box rather than black-box. To make the claim legitimate, I suggest the authors evaluate under a true black-box scenario, in which both the T2I model and the RAG system are unknown to the attacker.

2. **Questionable Attack Pipeline**: The backdoor requires the user to provide the complex, JOB-optimized trigger—something an ordinary user is extremely unlikely to input. In effect, only the attacker can reliably activate the backdoor, which makes the threat model self-targeting and the attack practically meaningless. To address this, I suggest that you present some very specific cases to show the damage that JOB may cause in the introduction or background.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a joint optimization backdoor attack method targeting black-box RAG-DMs (Retrieval-Augmented Diffusion Models). The attacker does not modify model parameters but instead optimizes textual triggers and poisoned images within the RAG system to activate the backdoor, causing the model to generate attacker-specified images.

### Strengths
This paper focuses on a new threat scenario, backdoor attacks specifically designed for RAG-DMs.

### Weaknesses
1. **The threat model is unclear.** 

It is difficult to understand what the attack is trying to achieve. From my perspective (correct me if I am wrong), the attacker first injects some tampered images into the RAG document controlled by the deployer. Then, the attacker input a carefully "crafted prompt" into the deployer's service to retrieve the tampered image. However, in this setting, normal users or the deployer are unlikely to use such "crafted prompts", and the attacker already has the tampered image. Therefore, it is unclear what the real-world impact of this attack is and who the actual victim is.

2. **The method seems easy to defend against.** 

The adversarial trigger actually contains a textual description of the "desired target class image" (see Fig. 3). This raises two concerns: (1) does this still qualify as a stealthy backdoor attack? and (2) could the attack be easily filtered due to obvious textual patterns or unnatural sentence structures?

3. The backdoor effect appears limited. 

Since the method jointly optimizes both the image and its corresponding textual trigger, there may be cases where the image has an unclear caption or a particular visual content that **cannot be reliably triggered**.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
