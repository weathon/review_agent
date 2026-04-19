# Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5, 5

## Abstract
Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Facing the demand of high-resource and sophisticated design of prompts in jailbreaking attacks on LLM safety alignment, this paper studies how simple random augmentations to the input prompt bypass safety alignment. The authors perform an evaluation of 17 models and investigate random augmentations with multiple dimensions, such as augmentation type, model size, etc.

### Strengths
1. The method proposed by the author is simple yet effective. Random augmentation is a simple modification to input prompts, yet such method can increase the success rate of harmful requests by up to 20~26%.
2. The evaluation across multiple dimensions is thorough. The authors investigate random augmentation in augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies. And the authors reveal several observations, such as character-level augmentations tend to be more effective than string insertion augmentations.

### Weaknesses
1. The authors claim their method to be "cheap", but no experiments support this point. The authors can add computation cost compared to other jailbreaking methods.
2. There is no comparison to other jailbreaking methods. The authors' statement that "adversarial attacks typically assume white-box...random augmentations are black-box...we do not compare..." is not reasonable. There also exist some jailbreaking methods against black-box models. The authors should compare with them. (https://arxiv.org/abs/2310.08419 ; https://arxiv.org/abs/2312.02119)
3. No widely-used commercial LLMs are evaluated. The authors should evaluate the proposed methods against widely-used commercial LLMs (e.g. ChatGPT), where bypassing the safety alignment is a more significant threat.
4. There is no discussion of mitigation methods against the proposed methods.

### Questions
1. How safety judge work? Is it done by LLMs, human experts or word matching?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigates jailbreaking open-source LLMs via small adjustments to the prompt. The paper investigates two types of augmentations: a "string" level insertion of a new word, and a "character" level augmentation which edits, deletes, or inserts a character.
The paper evaluates using the SORRY-bench dataset, across a variety of small language models, showing that augmentation can improve ASR under a variety of sampling temepratures. The paper also evaluates the attack under different defenses and under model quantization.

### Strengths
The method is simple and shows promise as a new form of jailbreak attack. The writing is clear and easy to understand, and the results are well-presented.

### Weaknesses
I was not entirely convinced by the evaluation results; Although they looked promising, it would be much stronger if the paper also included more diverse results.

For example, evaluation with/without system prompts would help clarify limitations of the approach. Additional experiments with larger models, or with instruction-tuned models would also further convince me that the approach is robust and general (or more importantly, reveal something interesting about the nature of LLM robustness to perturbed prompts). 

I would also like to see more comparison to prior work; The paper evaluates comparing to (Huang et al 2023), which I felt was interesting as it demonstrates an attack surface that isn't covered by prior work (this point should be emphasized more).

To make this more complete, it would be interesting if the paper compared against the other cited work: Andriushchenko & Flammarion, 2024, Vega et al. 2023, and Zou et al. 2023.

### Questions
Restating a few points from Weaknesses:

1) How does your method compare to prior work that adjusts prompts? While there is some evaluation comparing to Huang et al., it would be extremely interesting to build a more complete picture of the attack surface

2) How does your attack perform on instruction-tuned models and when including (or excluding) system prompts? (it's also unclear if system prompts were used during evaluation)

3) How susceptible are closed-source models to this attack? 

4) Have you tried string-level augmentations that insert words from a dictionary instead of random text? Would this meaningfully change anything?

5) I could not find any data with varying choices of p. While the paper mentions p=0.05 works well, it would be interesting to see how p changes the strength of the attack.

### Soundness
2

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
2

### Summary
The paper investigates how simple random input augmentations can bypass safety alignment in state-of-the-art large language models. The authors demonstrate that “stochastic monkeys” (low-resource, unsophisticated attackers) can evade safety mechanisms in LLMs by applying 25 random augmentations to each prompt. Besides, they evaluate these augmentations across several dimensions, such as augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies, highlighting that random augmentations enable bypassing of safety features with minimal resources. Notably, character-level augmentations consistently outperform string insertion in breaking alignment. Finally, the paper proposes a novel evaluation metric to evaluate safety and includes a human study to balance false positive and negative rates, furthering the robustness of their findings.

### Strengths
1. The paper applies simple random augmentations as a threat model, contributing a fresh perspective to the understanding of LLM safety alignment vulnerability.

2. The study is thorough, evaluating the effectiveness of random augmentations across multiple dimensions, and the use of a human-validated metric strengthens the reliability of the findings.

3. By demonstrating that low-cost, unsophisticated methods can bypass LLM safety features, the paper reveals critical vulnerabilities that may shape future research in secure LLM deployment, particularly for sensitive or public-facing applications.

### Weaknesses
1. While the authors discuss their method’s simplicity relative to adversarial attacks, a more detailed comparison against other established jailbreak and adversarial attack methods would better contextualize the effectiveness of random augmentations.

2. The study could benefit from an expanded discussion on the practical implications of these findings, especially in terms of what types of applications or LLM deployment scenarios are most at risk and possible mitigations.

3. The quantization experiment results in Figure 4 show variability across different models and sizes, but the authors provides vague explanations for these differences. Further exploring why certain models (e.g., Qwen 2) exhibit unique behavior under augmentations would enhance the study.

4. The writing of the conclusion is inconsistent with the abstract and conclusion, making it difficult to understand the main takeaway of this paper.

### Questions
1. Can the authors provide a comparison of the success rate of random augmentations against the success rate of more sophisticated adversarial attacks in the same LLM models?

2. Among all these various dimensions( augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies), can authors provide a ranking of these factors based on their significance towards the attack performance? In other words, the reader would like to know which factors should be the top concerns in their LLM applications. 

3. Have the authors considered examining the temporal robustness of this attack, i.e., does the attack success vary over repeated attempts or prolonged interaction with the same model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper examines the vulnerability of state-of-the-art large language models (LLMs) to random input augmentation attacks. It explores how different augmentation strategies, model configurations, and generation parameters affect attack success rates. The authors focus on two main augmentation types: Character-Level and String Insertion. Experimental results show that Character-Level augmentation yields higher attack success. Larger models demonstrate better resistance in general, though size alone doesn't ensure robustness. Furthermore, aggressive quantization tends to increase susceptibility to attacks, with variable effects across models. The paper also tests a fine-tuning-based defense, finding that it enhances resistance against longer attacks but remains vulnerable to shorter ones, with resistance varying based on attack lengths simulated during fine-tuning. A side effect noted is an increased rejection rate for benign prompts. Through extensive experimentation, the authors conclude that random input augmentation is a low-cost yet effective attack against LLM safety alignment, providing insights for future work in enhancing model robustness.

### Strengths
1. The paper is well-organized and clearly written, with a coherent structure and logical progression that supports the arguments effectively.
2. The paper examines the impact of various aspects such as LLM parameter size, quantization strategies, sampling parameters, and fine-tuning on resisting random input augmentation attacks

### Weaknesses
1. This paper focused on open-source chat models but lacks research on system prompts in the attack and defense settings
2. This paper did not discuss the impact of different tokenizer vocabulary sizes on this attack

### Questions
1. From the attack perspective, the authors did not discuss how the system prompt was configured for the open-source models in the experimental setup. It’s possible they used the default system prompt for each model or left it empty. In fact, setting the system prompt with a defensive focus (e.g., instructing the model to only output safe content) could impact the results. From a defense perspective, the authors should also examine whether using a system prompt as a defense mechanism is effective.

2. The authors should discuss the vocabulary sizes of the different LLMs tested. Larger vocabulary models are more likely to contain "glitch tokens," which may not have been fully trained and could potentially be more vulnerable to the attack methods proposed. Would these models be more susceptible to such attacks due to this factor?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates a simple but effective jailbreak attack using random perturbations. The experimental results demonstrate that this attack can significantly improve the chances of bypassing the safety alignment of 17 different models. Additionally, the paper examines how argumentation type, model size, and decoding strategy impact safety measures.

### Strengths
1. This paper is well-written with clear problem formulation
2. This paper demonstrate how easy the safety alignment can be bypassed
3. The ablation analysis of the proposed argumentation attack is detailed

### Weaknesses
The positioning and technical contribution of this paper seems to be unclear.
1. As a jailbreak attack paper, it lacks comparison with state-of-the-art blackbox jailbreak attacks (such as PAIR and DeepInception) in terms of attack success rate and computational cost. To strengthen the paper's contribution, the authors may need to include comprehensive comparisons with these baseline methods under the same evaluation settings.
2. As an analysis paper, despite being comprehensive, the conclusions are not new to the community. The techniques used (i.e., perturbation) are already widely employed to evaluate LLMs' safety and robustness (e.g., smoothllm). Furthermore, the impacts of model size, quantization, fine-tuning, and generation configurations on LLM safety have been well-studied previously.

### Questions
1. How frequently do random augmented prompts affect the semantic meaning of the original malicious prompt? Since the output is binary, could this lead to false safe/unsafe classifications by the evaluator, potentially impacting the experimental results?
2. Have you considered combining random augmentations with existing jailbreak attacks to see if it can lead to better performance?

### Soundness
3

### Presentation
3

### Contribution
2
