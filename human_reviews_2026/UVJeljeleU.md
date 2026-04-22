# CircuitTuning: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Prior studies investigating the internal workings of LLMs have uncovered sparse subnetworks, often referred to as circuits, that are responsible for performing specific tasks. Additionally, it has been shown that model performance improvement through fine-tuning often results from the strengthening of existing circuits in the model. Taken together, these findings suggest the possibility of intervening directly on such circuits to make precise, task-targeted updates. Motivated by these findings, we propose a novel method called CircuitTuning which identifies pivotal tokens from model reasoning traces as well as model components responsible for the desired task, and updates only those components. Applied to mathematical reasoning, it improves accuracy by up to +11.4% across multiple models while modifying as little as 1.59% of model components, with minimal impact on other abilities as measured by MMLU, TriviaQA, and TruthfulQA. These results demonstrate that targeted capabilities can be reliably enhanced by selectively updating a sparse set of model components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a circuit-based parameter-efficient finetuning on reasoning tasks involving chain-of-thought style reasoning traces.
Specifically, the method consists of three steps:
1. Identifying a "pivot token" in the reasoning trace, which is a token at a position with a large impact on whether the reasoning trace will lead to the correct or wrong answer.
2. Identifying a circuit that is responsible for producing this pivot token, and by extension, for the model producing the correct or wrong answer.
3. Selectively finetuning only this circuit on the task of interest.
The paper shows that this method yields improvements generally comparable to or better than another parameter-efficient finetuning method, namely LoRA.

### Strengths
The first step of the proposed method, i.e., identifying a pivot token in chain-of-thought reasoning traces appears could be useful for interpretability research in general, since one limitation of all currently available, practical circuit analysis methods is that they require a minimal pair of inputs that differ in only a single token.

### Weaknesses
In my view, the contribution of this paper is not substantive enough.

From a novelty perspective, only the first step of the proposed method, i.e., the identification of "pivot tokens", is novel. Circuit identification (Step 2) is performed using an existing method, and selective finetuning (Step 3) has been proposed in prior work (Wang et al., ICLR 2025: HeadMap: Locating and Enhancing Knowledge Circuits in LLMs; https://openreview.net/forum?id=jUsrbOuQ5e).

Now, methodological novelty is not a necessary criterion, as there are many other kinds of contributions a paper can make. However, I'm struggling to identify what this other kind of contribution could be in the case of this paper. It's possible to see the paper as contributing an empirical comparison of the proposed parameter-efficient finetuning (PEFT) method to other PEFT methods, but this contribution is very limited for the following reasons:
- Evaluation is limited to only a subset of one benchmark, GSM-Symbolic. (Some more datasets appear in the paper, but these are used as control tasks to verify the absence of side effects, not for evaluating the efficacy of the proposed method)
- The comparison to existing PEFT methods is not systematically controlled: The number of finetuning instances is different and it is unclear if the comparisons are fair in terms of overall computational cost, since LoRA doesn't require circuit identification (which can be costly) but the proposed method does.
- Claimed parameter efficiencies ("modifying as little as 1.59% of model components") are potentially misleading, because "components" refers to both attention heads and MLP neurons. Since there are many more MLP neurons than heads, even finetuning all attention heads but no MLP neurons would likely yield a very low ratio of modified "model components" under this metric.

### Questions
Minor comment:

line 115: "We propose a novel technique, called CircuitTuning, to improve the mathematical reasoning capabilities of an LM, without affecting other abilities."
This is an imprecise statement since the experiments clearly show an impact on other abilities (Table 2). The above phrasing would be more appropriate if there were no statistically significant changes observed among all the tested abilities/benchmarks.

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
The paper introduces CircuitTuning, a novel, mechanistically-inspired method for improving the mathematical reasoning abilities of LLMs. The method operates in three stages: 1) It first generates both correct and incorrect reasoning traces for a given problem to identify the "pivotal token" where the model's reasoning diverges towards an error. 2) It then uses a masking technique (Desiderata-based Component Masking, DCM) to localize the specific attention heads and MLP neurons that are most responsible for generating the correct next token. 3) Finally, it applies targeted gradient updates exclusively to this sparse sub-network. The authors evaluate CircuitTuning on the GSM-Symbolic benchmark across four models from the Gemma and OLMo families. They show that their method can improve math reasoning accuracy by up to +12.1% while modifying a very small fraction of model parameters (as low as 0.17%), and importantly, without significantly degrading performance on general benchmarks like MMLU, TriviaQA, and TruthfulQA.

### Strengths
1.  The three-stage approach of localizing the error token, identifying the responsible components, and performing targeted updates is a novel way to bridge mechanistic interpretability and model fine-tuning. The "Branching Method" for finding the pivotal token is a particularly strong and well-motivated part of this methodology.

2.  The method's main strength is its ability to achieve significant performance improvements while modifying a tiny fraction of the model's parameters (e.g., 0.17% for Gemma-9B). The results in Table 2 convincingly show that this surgical approach avoids the catastrophic forgetting that can plague broader fine-tuning methods, preserving performance on general benchmarks.

3.   The paper is written with exceptional clarity. The method, experimental setup, and results are described in sufficient detail to facilitate understanding.

### Weaknesses
1.  The LoRA baseline is trained on a different (and often larger) dataset than CircuitTuning. The performance difference could be attributed to the curated, high-signal "Error-Localization" dataset rather than the targeted update mechanism itself. To isolate the benefit of the proposed update strategy, LoRA should be trained on the exact same dataset.

2.  The method's performance relative to LoRA is not consistent across models. LoRA significantly outperforms CircuitTuning on the Gemma-2B model and also wins on the OLMo-13B model. The paper lacks a discussion or analysis of why this might be the case. 

3.  The method's reliance on generating paired correct/incorrect reasoning traces may be difficult to scale to more complex, open-ended domains like code generation or scientific reasoning, where a single, easily verifiable "correct trace" may not exist. 

4.  In several cases (e.g., Gemma-2B Branching, OLMo-7B Branching), the "w/o mask" ablation performs better than the "w/ mask" version. This is counter-intuitive to the central hypothesis that updating only a sparse, localized circuit is optimal.

### Questions
1.  Could you please clarify the rationale for training the LoRA baseline on the larger GSym-Train set instead of the smaller, curated Error-Localization dataset used for CircuitTuning? To make a more compelling case for your method's targeted update mechanism, would it be possible to run an experiment where LoRA is trained on the exact same data used by CircuitTuning?

2.  What is your hypothesis for the inconsistent results when comparing CircuitTuning to LoRA? Specifically, why do you think LoRA achieves a much larger accuracy gain on Gemma-2B (+16.8%) and a better gain on OLMo-13B (+5.5%) compared to your method? Does the effectiveness of CircuitTuning depend on model scale or architecture?

3.  The "w/o mask" ablation, which performs token localization but allows gradient updates to all parameters, sometimes outperforms the "w/ mask" version. How do you interpret this result? Does it challenge the core assumption that only a very sparse set of components should be updated?

4.  How do you envision adapting CircuitTuning to tasks where reasoning errors are more semantic or distributed across a sequence, rather than hinging on a single pivotal token (e.g., improving factual consistency in a summary or stylistic tone in a story)?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a select-the-finetune method, that targets subnetworks important to reasoning during fine-tuning. The authors evaluated the method on several common benchmarks, and showed that the method outperforms LoRA under certain settings.

After reading the paper, I believe that although the proposed method is well grounded on previous related works, the overall framework is premature and computationally expensive, and the advantages of the method are not well presented in the paper. In addition, the evaluation setup is not well-designed, and the performance of the method is not consistent. Therefore I think major revision is needed for the current manuscript, and I recommend rejecting the paper.

### Strengths
1. The research problem in the paper is well-motivated, focusing on the circuit phenomenon in reasoning models.
2. The proposed method is clearly explained, combining existing diagnostic frameworks and reasoning-focused training.

### Weaknesses
Methodological:

1. The scalability of the proposed method is under question: procedures such as token localization could incur significantly larger cost as the model is scaled-up, since more challenging reasoning tokens/traces will need to be detected and selected.
2. The advantages of the proposed method is not clearly stated in the paper. Is it memory efficiency? Or better interpretability of the reasoning paradigm? Without stating the advantages, the method will likely be less recognized by practitioners or researchers.
3. The method seems too tailored in the sense that different models/settings may need significantly different training setups, which is not generalizable. 

Experimental:

1. The performance improvement is not consistently better than other baselines, and the chosen baseline (LoRA) are not representative of the state-of-the-art. The authors should at least compare with more representative baselines, such as full fine-tuning or advanced LoRA methods [1]
2. The fairness of baseline comparison is under question: The authors did not explicitly compare the computational overhead of different methods. Since sub-procedures like token localization can incur substantial computational cost, the authors should explicitly mention it in the paper for clarification. 
3. No supplementary materials or code is provided. This makes the reproducibility of the method under question.

[1] DoRA: Weight-Decomposed Low-Rank Adaptation, https://arxiv.org/abs/2402.09353

### Questions
1. Could the authors provide a more detailed explanation of the targeted parameter update procedure? It seems that the mask is fixed during training, which is counterintuitive, since we would think that the circuits should be dynamically changing.
2. There are previous work discussing subnetworks in reasoning model training, and sparse training on principal weights crucial to reasoning [1, 2]. It would be great if the authors could add discussions on these works in the paper.
3. The detailed computational overhead is not outlined in the paper. Could the author provide some detail in the cost of each stage of the proposed method, and compare with the baseline? This will ensure fairness of comparison.

[1] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models, https://arxiv.org/abs/2505.11711

[2] LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning, https://arxiv.org/abs/2506.00772

### Soundness
1

### Presentation
2

### Contribution
1
