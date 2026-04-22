# Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
We discover a novel and surprising phenomenon of unintentional misalignment in reasoning language models (RLMs), which we call **self-jailbreaking**. Specifically, after benign reasoning training on math or code domains, RLMs will use multiple strategies to circumvent their own safety guardrails. One strategy is to introduce benign assumptions about users and scenarios to justify fulfilling harmful requests. For instance, an RLM reasons that harmful requests like "outline a strategy for stealing customers’ credit card information from a retail store" could be associated with the benign intent of "a security professional trying to test defense," despite no such benign context being provided as input. We observe that many open-weight RLMs, including DeepSeek-R1-distilled, s1.1, Phi-4-mini-reasoning, and Nemotron, suffer from self-jailbreaking despite being aware of the harmfulness of the requests. We also provide a mechanistic understanding of self-jailbreaking: RLMs are more compliant after benign reasoning training, and after self-jailbreaking, models appear to perceive malicious requests as less harmful in the CoT, thus enabling compliance with them. To mitigate self-jailbreaking, we find that including minimal safety reasoning data during training is sufficient to ensure RLMs remain safety-aligned. Our work provides the first systematic analysis of self-jailbreaking behavior and offers a practical path forward for maintaining safety in increasingly capable RLMs.​​​​​​​​​​​​​​​​

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the concept of Self-Jailbreaking, referring to the phenomenon where certain Reasoning Language Models (RLMs) produce unsafe or inappropriate outputs after engaging in multi-step reasoning (Chain-of-Thought, CoT). Through analysis, the authors argue that this behavior originates from a process of “self-persuasion” that occurs during reasoning: as the model deliberates, it gradually compromises between safety alignment and obedience (compliance), ultimately convincing itself that the user’s intent is benign and therefore proceeding to generate harmful or policy-violating content.

The authors conduct experiments on multiple open-source Reasoning Language Models (RLMs), demonstrating the widespread nature of the self-jailbreaking phenomenon. In addition, they propose an alignment method based on lightweight fine-tuning with a small amount of safety reasoning data, which effectively mitigates such behaviors while preserving the models’ reasoning capabilities.

### Strengths
1.	The paper introduces the concept of Self-Jailbreaking for the first time, revealing a fundamental flaw in current Reasoning Language Models (RLMs).
2.	Experiments conducted on multiple open-source RLMs demonstrate the generality and reproducibility of this mechanism.
3.	The authors provide a theoretically grounded analysis of the underlying causes, and the interpretation based on activation-direction projection is both logical and convincing.

### Weaknesses
1.	The analysis of Self-Jailbreaking lacks specificity. The paper does not systematically investigate which types of questions or prompts are more susceptible to triggering this phenomenon.
2.	The proposed mitigation relies on fine-tuning the model, which limits its applicability to closed or black-box models where retraining or parameter access is not possible.
3.  The model lacks dynamic robustness. Since the proposed mechanism is based on SFT (Supervised Fine-Tuning), it remains unclear whether the model will exhibit self-jailbreaking tendencies again after long-term usage, continuous updates, or multi-turn interactions. Moreover, the paper does not discuss any decoding-time or inference-time defense mechanisms to prevent such reoccurrence.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is really well-structured and highly motivative. I am really impressive by this self-jailbreaking phenomenon and get plenty of insights from this paper. The authors discover a concerning and previously uncharacterized safety failure mode "self-jailbreaking" that the reasoning model could reason itself out of safety alignment by introducing assumptions about user intent or context to justify fulfilling harmful requests or assuming that questions are only hypothetical to sidestep ethical considerations and so on. To address this, this paper shows minimal safety reasoning data during training is sufficient to ensure RLMs remain safety-aligned.

### Strengths
1. The investigated phenomenon is novel. Reasoning models struggle to defend against malicious queries and there are a few works[1][2][3] aiming at safety alignment improvement. However, this paper provides a new view of how and why reasoning models struggle in such settings.

2. The testing models include a wide range type, which make the conclusion convincing.

3. The self-jailbreaking rate in Figure 2 shows that this phenomenon is indeed a core reason responsible for the poor safety alignment of large reasoning models.

4. The solution is concise but effective.  

[1] Improving Safety Alignment with Introspective Reasoning

[2] Safety Reasoning with Guidelines

[3] SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities

### Weaknesses
1. Lack of human evaluation. Although gpt-5-2025-08-07 is a really strong model as a judge, human evaluation is still needed to reduce the  False Positive Rate and False Negative Rate. 

2. Lack of baseline. I personally really like the solution of this paper that using only 50 safety reasoning samples to strongly maintain the safety and helpfulness of reasoning models. As an investigation article, I’m willing to accept that it addresses only the issues it has explored, rather than achieving state-of-the-art (SOTA) results. However, I still recommend the authors to include some safety reasoning baselines to show how this method performs against these models, though it may lag behind.

3. Lack of OOD evaluation. As this paper is investigating and solving "self-jailbreaking", it mainly focus on vanilla harmful queries. However, I am curious about whether this phenomenon exists on jailbreak harmful queries? Is the 50 safety reasoning samples training also enough for those OOD jailbreak scenes? 

4. The caption in Figure 7 has squeezed the space in the following text. It is recommended to make some modifications.

I will increase my score if the above concerns are solved.

### Questions
See the Weakness part.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper reveals a new safety vulnerability in large language models, termed Self-Jailbreaking, and demonstrates that this phenomenon widely exists across various open-source reasoning language models (RLMs). Through interpretability analysis, the authors uncover its underlying mechanism and propose an effective mitigation strategy that restores the balance between safety alignment and reasoning capability.

### Strengths
1.It is first to systematically identify and name the phenomenon of Self-Jailbreaking, uncovering a paradoxical mechanism in which benign reasoning training unintentionally introduces safety risks — filling an important gap in current AI safety research. 
2.The experiments cover multiple families of RLMs and quantitatively illustrate how internal model states evolve throughout the reasoning process.

### Weaknesses
1.Although the projection analysis provides intuitive evidence, it does not fully establish the causal chain between increased compliance, reduced perceived harmfulness, and self-jailbreaking; potential confounding factors may exist. 
2.Projection-based interpretability has been widely used; the paper should justify its suitability and validity for analyzing self-jailbreaking specifically. 
3.The study lacks quantitative experiments on different types of self-jailbreaking (e.g., hypothetical scenarios, educational rationalizations, or positive-outcome justifications). 
4.The mitigation experiment (SAFE-S1.1-7B) is conducted on a single model only, without evaluating generalizability or transferability to other RLMs.

### Questions
1.Provide causal validation between compliance increase and harmfulness perception reduction. 
2.Clarify how the proposed interpretability approach differs from existing projection-based methods. 
3.Conduct proportional and quantitative analyses across different self-jailbreaking categories. 
4.Reproduce the SAFE training strategy on other RLMs (e.g., Phi-4, Llama) to evaluate its generalizability.

### Soundness
2

### Presentation
2

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
They define self-jailbreaking of reasoning models as the phenomenon of RLMs reasoning their way out of safety guardrails during reasoning to assist with malicious requests, without any jailbreaking or deception attempt from the user.  They measure the occurrences of self-jailbreaking of open-weight models and how harmful their responses become after benign reasoning training on math or coding tasks. Moreover, they analyze why RLMs generate harmful outputs through self-jailbreaking and show that RLMs after benign reasoning training have increased compliance. Lastly, they show that minimal safety reasoning data can sufficiently mitigate the harmful effects of self-jailbreaking and restore safety guardrail.

### Strengths
- The work is the first study of self-jailbreaking
- They evaluate various models to assess how frequently self-jailbreaking appears in the reasoning models after benign math/coding reasoning training
- They try to mechanistically explain why the models show self-jailbreaking 
- Moreover, they propose how to mitigate self-jailbreaking

### Weaknesses
I like this paper. It not only identifies an important problem but also explores a way to mitigate self-jailbreaking. Moreover, they provide a mechanistic interpretability analysis that explains why self-jailbreaking emerges in models after benign math and coding training.

However, there are some issues I notice. The paper doesn't describe details regarding the model training setup prior to the safety evaluation in Section 3.2. It is unclear what specific datasets were used, how many data points they contained, and whether a single dataset or multiple datasets were involved. In addition, the paper doesn't describe whether self-jailbreaking occurred consistently across different training datasets. Addressing these questions would be important to ensure generalization and robustness.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4
