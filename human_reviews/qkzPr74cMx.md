# ASPIRER: Bypassing System Prompts with Permutation-based Backdoors in LLMs

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50\% while maintaining a clean accuracy (CACC) of 98.58\%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework for byPassing System Prompts wIth peRmutation-basEd backdooR (ASPIRER) in large language models (LLMs). The key insight guiding the working of ASPIRER is a permutation-based trigger- i.e., the trigger is activated only when its components are arranged in a specific order. The paper demonstrates that such a trigger can bypass system prompts, and moreover, is extremely stealthy due to the permutation-based design. ASPIRER is evaluated on five state-of-the-art LLMs for robustness and effectiveness

### Strengths
1. ASPIRER employs a novel attack mechanism based on a permutation, which activates a trigger only when its components are in a specific order. This makes the attack stealthy by design, making it difficult to reverse engineer triggers. 

2. ASPIRER is evaluated on five different LLMs, and is able to achieve high attack success rates while maintaining high levels of accuracy on benign samples. 

3. The use of negative samples during training is especially important in reducing the false trigger rate. The need for such negative training has been explained in depth. The remainder of the methodology section is also well-written. 

4. ASPIRER is shown to be achieve high attack success against two classes of defenses- perplexity-based and perturbation-based, which demonstrates its effectiveness.

### Weaknesses
1. The work may be strengthened by considering other forms of triggers beyond verbs and adverbs. Can the authors comment on this aspect? 

2. An explanation for the rationale governing the choice of the five LLMs considered (and why some other popular models were not examined) will be helpful.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a novel backdoor attack targeting large language models (LLMs) with the goal of disabling system prompts and achieving restricted outcomes. Additionally, the authors propose a permutation trigger, which activates only when its components are arranged in a specific order.

### Strengths
1. The attack vector of disabling system prompts is very interesting and, to the best of my knowledge, has not been explored before.
2. The authors validate this attack vector through extensive experiments.
3. The paper is well-written and well-organized.

### Weaknesses
1. The technical contribution is limited. The core technical contributions of the authors are the proposal of a permutation trigger and Negative Training to make the model learn this trigger. However, composite triggers and Negative Training have already been proposed earlier [1,2], and the permutation trigger merely adds order information on this basis.
2. The advantages of permutation triggers are unclear. The authors claim that permutation triggers can enhance stealthiness, prevent accidental triggering, and are computationally challenging to reverse engineer. But these advantages are compared to single-word triggers; whether they also have these advantages compared to composite triggers is unknown (I believe composite triggers also have these advantages).
3. There is a lack of baselines. To support the advantages brought by permutation triggers, the authors need to compare them with baseline methods of other trigger designs, especially composite triggers.
4. Some statements are inaccurate. For example, the authors claim that "composite triggers require parts to appear in the system prompt," but the original design of composite triggers [1] was to exist only in user input and could be used in the attack scenario proposed in the paper.

[1] Rethinking Stealthiness of Backdoor Attack against NLP Models ACL2021

[2] Composite Backdoor Attacks Against Large Language Models Findings of NAACL 2024

### Questions
Please see the weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel backdoor attack method for large language models (LLMs) that uses permutation triggers to bypass system prompts, enabling malicious actors to evade built-in ethical and functional boundaries. These triggers activate only when specific words appear in a predefined order, making them challenging to detect or accidentally activate. The study demonstrates ASPIRER’s high success rate across several LLMs with minimal impact on clean model performance, highlighting significant vulnerabilities in the AI supply chain and the need for stronger defenses against such sophisticated backdoor attacks.

### Strengths
1. The topic is of great significance and sufficient interest to ICLR audiences.
2. The paper is easy to follow to a large extent.
3. The authors try to provide some theoretical analyses of their method. It should be encouraged.

### Weaknesses
1. My biggest concern is the novelty of this paper. The authors claimed that they propose a new attack surface (i.e., bypass system prompts) with a new trigger design. 
- However, this attack is very similar to or even a sub-category of fine-tuning attacks [1]. The authors should clarify the main differences between the proposed attack and these attacks.
- For me, the permutation trigger is a case of composite/multi- trigger that only using multiple instead of one trigger can activate the specific backdoor. However, composite backdoor attacks have been discussed in existing works (e.g., [2]).
- The use of contrast/negative learning to encourage the model to learn to specific backdoor triggers is also a common technique in existing backdoor attacks [3,4].
2. It seems that only using system prompts cannot reach the specific requirement. For example, I have tried GPT4o, but the listing example 1 failed.
3. Missing experiments to support the claim in Line 258-260 (i.e., we observe that these three types of unit changes as negative samples are sufficient to encompass all possible cases)
4. To the best of my knowledge, Theorem 1 is based on the assumption that the model can learn each case of your studied three unit-changes perfectly. However, the author doesn't explicitly point this out and it does not hold in practice.
5. In Section 3.3, the authors failed to provide sufficient technical details of the design of output. For example, what is the target output?
6. Missing many baseline attacks. The authors should compare their method to fine-tuning attacks mentioned in [1].
7. I would like to see experiments where developers fine-tune the model with samples from the same classes instead of different classes but with different specific samples.
8. Missing the results of advanced defenses discussed in [1] in Section 4.4.


References
1. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey
2. Composite Backdoor Attacks Against Large Language Models
3. Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification
4. Towards Faithful XAI Evaluation via Generalization-Limited Backdoor Watermark

### Questions
1. Please clarify the main differences and contribution compared to (harmful) fine-tuning attacks, composite attacks, etc.
2. Provide more technical details regarding the proposed method, especially the design of output.
3. Conduct more comprehensive experiments using baseline attacks and defenses introduced in [1].

Please refer to the previous part for more details.

### Soundness
2

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
3

### Summary
The paper focuses on an important topic of AI supply chain security. However, having a fully compromised provider might be a very strong assumption which I am not sure is necessary even to motivate the problem. Especially because even in the mentioned example of the supply chain issues of the previous attacks it is not the case that the whole entity is compromised. In real-world scenarios, attackers typically gain access to only a portion of the supply chain and must operate without detection. The paper proposes modifying the entire training process of a large language model - an action that would likely be conspicuous and easily detectable. This disconnect from real-world attack scenarios weakens the practical applicability of their findings.

### Strengths
This paper provides an interesting approach for designing backdoors while there are many experiments missing but it would be helpful for people to also consider more complex forms of backdoors in security evaluations.

### Weaknesses
A fundamental inconsistency exists between the paper's threat model and its experiment section. While the authors frame their attack scenario around a compromised model provider with access to the base model training process, their experiments only mostly focus on the fine-tuning. 

The authors claim that simpler backdoors can be easily detected, and that's why we need more complicated backdooring schemes such as this paper. While I agree more interesting backdoor ing approach can be more effective, however, in simple approach still work on language models and I am not sure how easy can be to detect them and the provided citation are mostly for older models. he Sleeper Agent paper, demonstrates that simpler backdoor approaches can be highly effective in language models. 

In the  AI pipeline there are also other components such as additional safety finetuning and rlhf which can be used either by the model provider or downstream. It would be helpful to show these processes will not reduce the effectiveness of their approach

### Questions
Show casing simpler approaches doesn't work
Exactly mentioning which part of the pipeline the adversary is modifying and why it is not easily detected
effect of other component on the results

### Soundness
3

### Presentation
2

### Contribution
3
