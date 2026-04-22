# ASIDE: Architectural Separation of Instructions and Data in Language Models

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 8, 4, 2

## Abstract
Despite their remarkable performance, large language models lack elementary safety features, making them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as the root cause of the success of prompt injection attacks. In this work, we propose a new architectural element, ASIDE, that allows language models to clearly separate instructions and data at the level of token embeddings. ASIDE applies an orthogonal rotation to the embeddings of data tokens, thus creating clearly distinct representations of instructions and data tokens without introducing any additional parameters. As we demonstrate experimentally across a range of models, instruction-tuning LLMs with ASIDE (1) achieves substantially higher instruction-data separation without performance loss and (2) makes the models more robust to prompt injection benchmarks, even without dedicated safety training. Additionally, we provide insights into the mechanism underlying our method through an analysis of the model representations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new method to defend against prompt injection attack by separating instructions and data at the level of token embeddings through orthogonal rotation. The advantage of the proposed method is that it does not rely on safety-related data for training while achieving good performance on both normal tasks and safety tasks.

### Strengths
1. This method is novel to my knowledge.

2. The safety performance of the proposed method is promising.

3. The proposed method can achieve better safety without the need of safety data.

4. Many different benchmark datasets are used for evaluation.

### Weaknesses
1. It is not very clear on the selection of orthogonal rotation. Why is it better than other kinds of transformation? Is there any theoretical analysis on this?

2. Orthogonal rotation is simple, which is good. But does it can well fit different data distributions, tasks and models?

3. It would be better if more normal tasks such as reasoning-related tasks are incorporated into experiments to evaluate the impact of the proposed method on model performance.

### Questions
Please refer to above comments.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ASIDE (Architecturally Separated Instruction-Data Embeddings), a novel, parameter-free architectural modification to Large Language Models (LLMs) aimed at mitigating prompt injection vulnerabilities. The authors identify the lack of intrinsic separation between instructions (which should be executed) and data (which should be processed) as a root cause of these attacks.

ASIDE's core mechanism is to create distinct representations for instructions and data at the very first layer. It achieves this by applying a fixed, parameter-free 90-degree orthogonal rotation to the token embeddings of all inputs designated as "data," while leaving "instruction" token embeddings unchanged. This modified model is then instruction-tuned using a standard (non-adversarial) dataset.


The paper empirically demonstrates across a range of models (including Llama, Qwen, and Mistral) that ASIDE:

* Achieves substantially higher instruction-data separation (measured by the SEP score).

* Maintains model utility and performance, comparable to standard fine-tuned models (measured by SEP Utility and Alpaca Eval 1.0).

* Improves robustness against both indirect and direct prompt injection benchmarks, even without any dedicated safety training.

The authors supplement these findings with a strong set of interpretability analyses, including linear probing and causal interventions, to validate that the architectural separation persists through the model's layers and is causally linked to the improved safety.

### Strengths
* Novelty and Elegance: The proposed method is simple, highly novel, and elegant. Using a fixed, parameter-free orthogonal rotation is a clever architectural solution that avoids the overhead of additional parameters or complex, learnable components.



* Strong Empirical Validation: The claims are convincingly supported by experiments across a wide and diverse set of modern LLMs (Llama 2 7B/13B, Llama 3.1 8B, Qwen 2.5 7B, Qwen3 8B, and Mistral 7B). The improvement in instruction-data separation (SEP score, Figure 2a) is consistent and significant over all baselines.




* Practical Safety Improvement: A key contribution is that ASIDE provides a measurable and "free" improvement in robustness, particularly against indirect prompt injections (Table 1), without requiring any adversarial data or specialized safety fine-tuning. This makes it a highly practical method for improving the baseline safety of LLMs.


* Excellent Interpretability and Analysis: This is a standout strength of the paper. The authors provide deep insights into why ASIDE works.

* Clarity: The paper is exceptionally well-written, with clear illustrations (especially Figure 1) and a logical flow that makes the method and its evaluation easy to follow.

### Weaknesses
*Scope Limited to Pre-Defined Roles: The primary limitation is that ASIDE requires the functional role (instruction vs. data) of input tokens to be specified a priori by the system implementer. This is a reasonable assumption for many integrated applications (e.g., RAG, email clients). Still, it is not directly applicable to general-purpose, multi-turn chatbots where a single user turn might contain a mix of data (e.g., continuing a story) and new instructions (e.g., "now change the character's name"). The authors acknowledge this limitation.

* Limited Justification for Rotation Choice: The paper specifies a 90-degree ($\frac{\pi}{2}$) isoclinic rotation and justifies it based on computational efficiency (it simplifies to coordinate swapping and negation). While practical, this leaves the theoretical justification underexplored. The paper would be strengthened by an ablation study comparing this specific rotation to other angles (e.g., 45 degrees) or other types of parameter-free orthogonal transformations to show that 90 degrees is an optimal or robust choice.

### Questions
1. Multi-turn Scenarios: The paper's scope is explicitly limited to single-turn, system-level applications where roles are pre-defined. Could you elaborate on how you envision ASIDE being adapted for a multi-turn conversational setting? For instance, could a classifier be trained to assign "instruction" or "data" roles to spans of a user's turn before the embedding rotation is applied?



2. Choice of Rotation: Your justification for the 90-degree rotation is its computational efficiency. Did you experiment with other rotation angles (e.g., 45, 180 degrees) or other types of parameter-free orthogonal transformations? How sensitive is the model's SEP score and ASR to this specific choice?

3. ASIDE vs. ISE Mechanism: The finding in Figure 3 that ISE's linear separability degrades in deeper layers while ASIDE's does not is a key differentiator. What is your hypothesis for this? Why do you believe the network can "undo" or "ignore" a learnable offset (ISE) more easily than it can a fixed rotation (ASIDE)?


4. Direct Injection Nuance: The robustness gains on direct prompt injection (Table 1) appear less pronounced than on indirect injection, with negligible improvement on the RuLES benchmark. Does this suggest ASIDE is primarily effective against attacks based on role confusion (data-as-instruction), rather than attacks that try to override a known instruction (jailbreaks)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ASIDE, a lightweight architectural modification that enforces structural separation between instruction and data embeddings in large language models. By applying orthogonal transformations to distinguish instruction and data tokens, ASIDE improves robustness against prompt injection attacks without introducing extra parameters. Experiments show significant gains in instruction–data separation and lower attack success rates.

### Strengths
1. The proposed method is simple, easy to understand, and conceptually interesting.

2. The proposed method is computationally efficient and does not introduce any additional parameters.

3. The paper includes insightful analyses explaining why ASIDE achieves better instruction–data separation performance.

### Weaknesses
1. As stated in the paper, the proposed method is designed as a defense against prompt injection attacks. Therefore, I believe the paper should include comparisons with the most recent fine-tuning-based defenses such as StruQ [1], SecAlign [2], and Meta-SecAlign [3]. The current baselines used for comparison appear relatively weak.

2. Although the ASR of ASIDE in Table 1 is lower than that of other baselines, some numbers remain quite high (exceeding 60%), suggesting that ASIDE may not be a very effective defense against prompt injection in practice.

3. I find the utility results not very convincing. Specifically, the results on AlpacaEval may be over-estimated, since the training set (Alpaca-Clean) is very similar to AlpacaEval, potentially leading to distribution overlap. Moreover, Qwen-based models (which already possess strong instruction-following capabilities) perform well on SEP but worse than other methods on AlpacaEval, indicating possible evaluation bias.

4. I think the authors’ statement “we use plain pretrained models rather than instruction- or safety-tuned models to avoid biasing the safety evaluations” somewhat unconvincing. As we know, the utility of the pretrained Llama models is very poor (also as shown in Figure 2), and such models generally lack instruction-following capacity. In realistic deployments, using instruction-tuned models is unavoidable. Therefore, the paper should additionally evaluate ASIDE on instruction-tuned versions of those models, and compare their utility with the original instruction-tuned baselines, or at least report the results of the instruction-tuned models. This would better demonstrate the compatibility between ASIDE and instruction tuning.

[1].Chen, Sizhe, et al. "{StruQ}: Defending against prompt injection with structured queries."

[2].Chen, Sizhe, et al. "Secalign: Defending against prompt injection with preference optimization." 

[3].Chen, Sizhe, et al. "Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks."

### Questions
Please see the weakness part above.

### Soundness
2

### Presentation
2

### Contribution
2
