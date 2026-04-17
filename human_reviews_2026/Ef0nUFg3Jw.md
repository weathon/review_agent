# LoRA Users Beware: A Few Spurious Tokens Can Manipulate Your Finetuned Model

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Large Language Models (LLMs) are commonly finetuned for a variety of use cases and domains. A common approach is to leverage Low-Rank Adaptation (LoRA)--known to provide strong performance at low resource costs. In this study, we demonstrate that LoRA actually opens the door to short-cut vulnerabilities--and the more resource efficient is the LoRA setup, the more vulnerable will be the finetuned model to aggressive attacks. To measure that vulnerability, we introduce Seamless Spurious Token Injection (SSTI), where we find that LoRA exclusively focuses on even just a single token that is spuriously correlated with downstream labels. In short, injection of that spurious token during finetuning ensure that the model’s prediction at test-time can be manipulated on-demand. We conducted experiments across model families and datasets to evaluate the impact of SSTI during LoRA finetuning while providing possible mitigations. Our experiments conclude that none of the existing checkers and preprocessors can sanitize a dataset raising new concerns for data quality and AI safety.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper reveals a security and robustness vulnerability in LoRA-based finetuning. The authors show that injecting only a few spurious tokens into the training data can cause the LoRA adapter to learn a shortcut mapping from these tokens to a target label. This phenomenon is referred to as Seamless Spurious Token Injection (SSTI). The paper systematically studies this effect across multiple datasets (IMDB,SST e.g.), models (Snowflake Arctic, OpenELM, LLaMA-3), and LoRA ranks. Results show that the injection of supurious triggers truly altered models' behavior. Authors also discuss the detection of such injection.

### Strengths
-The paper is well-written and logically structured; the problem, methodology, and experimental design are easy to follow.

-The evaluation is comprehensive — spanning multiple models, datasets, token injection settings, and training configurations — and provides strong empirical support for the claims.

-The observation on the relationship between LoRA rank and SSTI effectiveness is particularly interesting and provides new insights into the behavior of parameter-efficient finetuning.

### Weaknesses
+ **The core idea appears indistinguishable from standard poisoning-based backdoor attacks.**  
  My main concern is that the proposed SSTI setting does not seem fundamentally different from the well-explored threat model of backdoor attacks via poisoning finetuning data. In both cases, the attacker injects a trigger (in backdoor attacks this can be a specific token or pattern; in this work, a set of spurious tokens) into the training data to manipulate the model’s predictions. The only practical difference is that the authors apply this to LoRA finetuning instead of full-parameter finetuning. Since poisoning-based backdoor attacks with small numbers of samples have already been extensively studied in prior work, it is unclear what conceptual novelty this paper adds on top of existing backdoor literature. This makes the contribution less convincing and raises doubts about how different SSTI truly is from classic backdoor attacks.

(Above is my primary concern; the remaining points are secondary and more about suggestions for improvement rather than acceptance-blocking issues.)

+ **The observed relationship between LoRA rank and SSTI effectiveness lacks theoretical explanation.**  
  The finding that lower-rank LoRA is more vulnerable under light SSTI but becomes more robust under aggressive SSTI is interesting and insightful. However, the paper does not provide any theoretical analysis or deeper interpretation for this phenomenon. Offering even an initial theoretical explanation—for example in terms of parameter capacity, shortcut learning dynamics, or representation constraints—would significantly strengthen the depth and credibility of the work.

### Questions
As mentioned in the weakness section, I am still unclear about the fundamental difference between your proposed SSTI setting and traditional data-poisoning-based backdoor attacks. In both cases, an attacker injects a specific token or pattern into a subset of the fine-tuning data to create a shortcut between that token and a target label. Could you please clarify whether there is a more essential distinction that I may have misunderstood?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates shortcut vulnerabilities in Low-Rank Adaptation (LoRA) when fine-tuning large language models (LLMs). The authors propose a Seamless Spurious Token Injection (SSTI) framework. In this framework, spurious tokens are first identified based on conditional entropy. These tokens are then injected—sourced from various distributions—into different positions of text sequences with varying injection ratios. Experimental results show that even a single spurious token can significantly manipulate model predictions.

### Strengths
1. The research topic is important. The observation that a single spurious token can influence model behavior is both surprising and impactful.

2. The paper is clearly written and easy to follow. Key ideas such as spurious token set construction and injection methodology are well explained.

3. The evaluation is thorough. The authors explore multiple variables, including injection ratio, token position, and spurious token source, to demonstrate LoRA's vulnerability under diverse conditions.

### Weaknesses
1. The threat model needs further clarification. The paper assumes that the attacker controls the entire fine-tuning process—including token set construction, injection, and fine-tuning. However, in practice, such full control is rare. For instance, users typically fine-tune LoRA models on customer or proprietary data, limiting an attacker's access and influence. A discussion of more realistic threat scenarios would strengthen the paper.

2. The core finding is that LoRA is prone to overfitting spurious tokens, i.e., those with much lower conditional entropy than other tokens. While this is an interesting observation, it is somewhat intuitive. Tokens with low conditional entropy are highly predictive of certain outputs, making them likely to be overfit during training.

### Questions
1. Spurious tokens play a central role in this work. As noted in line 185, spurious tokens can also be token sequences rather than individual tokens. Could LoRA be even more vulnerable to sequences of spurious tokens? Have the authors considered evaluating sequence-level perturbations?

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
3

### Summary
The authors propose a new attack, Seamless Spurious Token Injection (SSTI). They show that LoRA can focus on a single token that is spuriously correlated with downstream labels, and they explore how LoRA hyperparameters (e.g., rank) interact with this vulnerability and with potential defenses.

### Strengths
1. Given the widespread use of LoRA, studying its potential vulnerabilities is timely and important — this line of work helps the community better understand and improve the robustness of PEFT methods.

2. The paper is generally well written and easy to follow. The presentation makes the main ideas accessible.

3. The authors perform extensive experiments that investigate multiple aspects of the relationship between LoRA and the proposed attack.

### Weaknesses
1. Novelty & relation to backdoor attacks. 
The proposed attack closely resembles classic backdoor/poisoning attacks: injecting a trigger token and training corresponding samples with a target label so the model learns a spurious correlation that controls behavior at inference time. The authors need to clearly explain how SSTI is meaningfully different from, or advances, the existing backdoor literature. Also this shortcut/spurious correlation phenomenon is well studied in the backdoor attack papers [4,5], and currently there are many papers about backdoor attacks in the LoRA/LLM domains [1-3].

2. Stealthiness and practicality.  
In section 4.1, the author states that the model predicted the target class regardless of input content.  If so, how realistic is this attack in practice? Would such conspicuous behavior be likely to be deployed or discovered by users?  This model is useless, since it can only predict one class, so why do users want to use it?

3. Overclaim in Section 4.1 / Table 1. 
The results in Table 1 appear to be produced when all training samples are injected with the spurious token (i.e., the training set’s ground truth labels are dominated by a single class). Under this setting, the model will unsurprisingly output the training class, this seems closer to trivial overfitting than to an attack demonstrating stealthy model subversion. The authors should avoid overclaiming and clarify the setup and its implications.

4. Unrealistic poisoning rates.  Many experiments use very high poison rates (≥50%, up to 100%). This is an unrealistic adversary model for stealthy poisoning/backdoor attacks. Prior work typically evaluates much lower poison rates (often <5%). The authors should evaluate lower (more realistic) poison rates and report attack success vs. utility tradeoffs.

[1] LoRA Once, Backdoor Everywhere in the Share‑and‑Play Ecosystem
[2] LoRA‑Based Backdoor Attack on Model Merging (LoBAM)
[3] A Survey of Recent Backdoor Attacks and Defenses in Large Language Models
[4] Backdoor Defense via Deconfounded Representation Learning
[5] BBCaL: Black-box Backdoor Detection under the Causality Lens

### Questions
See above please.

### Soundness
2

### Presentation
3

### Contribution
2
