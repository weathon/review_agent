# Rethinking Reasoning Quality in Large Language Models through Enhanced Chain-of-Thought via RL

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Reinforcement learning (RL) has recently become the dominant paradigm for strengthening the reasoning abilities of large language models (LLMs). Yet the rule-based reward functions commonly used on mathematical or programming benchmarks assess only answer format and correctness, providing no signal as to whether the induced Chain-of-Thought (CoT) actually improves the answer. Furthermore, such task-specific training offers limited control over logical depth and therefore may fail to reveal a model’s genuine reasoning capacity. We propose **D**ynamic **R**easoning **E**fficiency **R**eward (**DRER**) — a plug-and-play RL reward framework that reshapes both reward and advantage signals. (i) A **Reasoning Quality Reward** assigns fine-grained credit to those reasoning chains that demonstrably raise the likelihood of the correct answer, directly incentivising the trajectories with beneficial CoT tokens. (ii) A **Dynamic Length Advantage** decays the advantage of responses whose length deviates from a validation-derived threshold, stabilising training. To facilitate rigorous assessment, we also release LogicTree, a dynamically constructed deductive reasoning dataset that functions both as RL training data and as a comprehensive benchmark. Experiments show that DRER achieves significant improvements in reasoning accuracy and CoT quality over baseline methods across diverse training settings, while also reducing token usage during inference. Moreover, it demonstrates strong generalization on both reasoning and mathematical benchmarks, such as GPQA and AIME24. These results indicate that DRER, as a plug-and-play fine-grained RL reward framework, reliably strengthens reasoning behavior and provides a practical pathway toward enhancing the reasoning capabilities of large language models. All code and data are available in our anonymous repository https://anonymous.4open.science/r/DRER-D34E.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel reinforcement learning framework, Dynamic Reasoning Efficiency Reward (DRER), aimed at improving the quality of chain-of-thought reasoning in large language models. The key contributions are twofold. First, the DRER framework proposes a more nuanced reward signal that goes beyond simple answer correctness. It incorporates a "Reasoning Quality Reward" to credit reasoning steps that increase the likelihood of the correct answer, and a "Dynamic Length Advantage" to regulate the length of generated reasoning chains, which helps stabilize training. Second, the authors introduce LogicTree, a new benchmark dataset for deductive reasoning that is programmatically constructed. This dataset serves both as training data and as a tool for evaluating models in a controlled environment that focuses on logical structure over domain knowledge.

The core idea of rewarding reasoning chains that are demonstrably helpful is a significant step up from simply rewarding correct final answers. This is well-supported by strong experimental results showing that their method not only improves performance on their own benchmark but also generalizes to other established reasoning datasets. The ablation studies and analyses provide convincing evidence for the effectiveness of the different components of their proposed framework.

The LogicTree dataset is another major strength. By programmatically generating problems and controlling for logical depth, the authors provide a much-needed tool for disentangling pure reasoning ability from domain knowledge. While I have some questions about the generation process (see below), the dataset itself is a valuable resource for the research community.

Overall, the paper is well-written, the ideas are novel, and the empirical evaluation is thorough. The work opens up interesting avenues for future research on improving the reliability and transparency of reasoning in LLMs.

### Strengths
**Strong Points:**

*   **Novel Reward Framework:** The proposed DRER framework is a significant contribution. Moving beyond binary correctness to reward intermediate reasoning steps that are causally beneficial to the final answer is a promising direction for improving LLM reasoning.
*   **Practical Training Improvements:** The Dynamic Length Advantage is a practical and useful technique for controlling output length and stabilizing RL-based training of LLMs for reasoning tasks.
*   **Valuable New Dataset:** The LogicTree dataset is a strong contribution to the community. Its programmatic nature and focus on deductive reasoning provide a valuable tool for rigorously assessing and training the formal reasoning capabilities of models, separate from their factual knowledge.
*   **Strong Empirical Results:** The paper presents compelling experimental results. The proposed method shows significant improvements in accuracy and logical consistency over baselines. The demonstration of generalization to other logical and mathematical reasoning benchmarks is also a key strength.
*   **Insightful Analysis:** The analysis of how CoT reasoning impacts the model's predictions (e.g., when it is most effective at correcting a wrong answer) provides valuable insights into the behavior of these models.

### Weaknesses
**Weak Points:**

*   **Clarity on Dataset Generation:** The paper could benefit from a more detailed explanation of the lexicalization process used to create the LogicTree dataset. It is not entirely clear how the abstract logical rule trees are translated into natural language.
*   **Potential for Linguistic Ambiguity:** The process of converting logical forms to natural language is fraught with potential ambiguities (e.g., scalar implicatures, quantifier scope). The paper does not discuss whether these issues are addressed in the dataset generation process.
*   **Lack of Human Validation:** There is no mention of human validation for the LogicTree dataset. It would be valuable to know if human subjects consistently interpret the natural language problems in a way that aligns with the intended underlying logical structure.
*   **Minor Errors:** There are a few minor errors in the paper, such as a missing citation for a benchmark and some incorrect terminology for logical rules in a table.

### Questions
1.  Could you please provide more detail on the lexicalization process for the LogicTree dataset? Specifically, how are the abstract logical rules and entities mapped to natural language sentences?
2.  How do you ensure that the generated natural language statements are not subject to linguistic ambiguities, such as scalar implicatures (e.g., the interpretation of "or")? Have you considered or tested for such effects?
3.  Have you conducted any human evaluation studies on the LogicTree dataset? It would be very helpful to see data on how consistently human participants interpret the problems and whether their interpretation aligns with the intended logical form.

The following are some minor suggestions for improvement:

*   Please add the missing reference for the "Math500 benchmark" on page 4.
*   In Table 1, the names of the logical rules appear to be incorrect. For example, what is labeled "Conjunction introduction" seems to be the "Constructive Dilemma," and "conjunction elimination" appears to be "proof by cases." Please double-check this terminology.
*   In Figure 2, the "question" is technically a statement to be evaluated for its truth value. It might be clearer to label it as "Statement" or "Claim" to avoid confusion.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces DRER (Dynamic Reasoning Efficiency Reward), a plug-and-play reinforcement learning (RL) framework designed to improve the quality and efficiency of Chain-of-Thought (CoT) reasoning in large language models (LLMs). Unlike conventional RL approaches that reward only final answer correctness, DRER incorporates:
1.	Reasoning Quality Reward: Measures whether the CoT tokens increase the model’s confidence in the correct answer by comparing log-likelihoods with and without CoT.
2.	Dynamic Length Advantage: Penalizes responses that are significantly longer or shorter than a validation-derived length threshold, promoting concise and stable reasoning.
To support training and evaluation, the authors also release LogicTree, a synthetically generated dataset of nested deductive reasoning problems with controllable depth, logical diversity, and intermediate sub-questions.
Experiments on LogicTree and other benchmarks show that DRER improves accuracy, logical consistency, confidence, and token efficiency over baselines like GRPO and DAPO.

### Strengths
1.	The Reasoning Quality Reward directly ties intermediate reasoning steps to answer confidence. The Dynamic Length Advantage may stabilize training and reduce verbosity.
2.	Analyses on various aspects of models’ reasoning capabilities.

### Weaknesses
1.	LogicTree focuses exclusively on propositional deductive reasoning. What about inductive, abductive, analogical, or higher-order reasoning?
2.	The paper compares DRER-augmented training against baselines (GRPO/DAPO). Readers would be interested in component-wise ablation study. For example: does the length regularization alone improve accuracy, or is it only effective when combined with the quality reward?
3.	The paper could consider varying the degree of distributional shift (e.g., by perturbing LogicTree rules, mixing domains, or introducing noise). It’s unclear whether DRER’s gains are robust to OOD conditions, or if they merely reflect superficial transfer within a narrow reasoning regime.
4.	There are some typos in the article, such as citations in lines 44-58.

### Questions
1.	Can DRER be extended to reasoning tasks where ground-truth answers are less well-defined?
2.	Does encouraging higher confidence in answers via CoT increase the risk of overconfident hallucinations when the premises are insufficient?

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
5

### Summary
This paper proposes a new reinforcement learning framework for reasoning enhancement in large language models (LLMs).
The method introduces a Reasoning-Quality Reward (RQR) that measures the average log-likelihood difference between the model’s correct answer under reasoning (CoT) and direct answering (No-CoT) modes.
A positive difference implies that reasoning increases the model’s confidence in the correct answer, which is rewarded via a tanh-based scaling.
Additionally, the paper adds a Dynamic Length Advantage (DLA) term that regularizes overly long or short reasoning chains based on length statistics collected from validation rounds.
Experiments on a self-constructed LogicTree dataset and several reasoning benchmarks (AIME24, MMLU-redux, ProntoQA) show improved reasoning efficiency and robustness compared with GRPO and DAPO baselines.

### Strengths
1. Proposes a clear and interpretable framework to assess whether reasoning helps model confidence.

2. Combines reasoning-quality reward with a dynamic length regularization to stabilize training.

3. Experiments are controlled, fair, and internally consistent.

4. Writing and visualization are of high quality.

5. Provides an interesting diagnostic tool for analyzing reasoning effectiveness in LLMs.

### Weaknesses
1. The log-likelihood difference is a self-referential reward; it optimizes internal belief alignment rather than genuine reasoning improvement.
This is conceptually close to confidence calibration and may lead to reward hacking or “self-confirmation loops.”

2. Dynamic Length Advantage is highly similar to prior adaptive length penalties. The paper does not provide an ablation proving the unique benefit of its specific formulation.

3. The approach requires ground-truth answers and double forward passes, limiting scalability and applicability to open-ended reasoning tasks.

4. Gains on real reasoning benchmarks are modest; LogicTree is synthetic and may not reflect real reasoning complexity.

5. Related Work misses critical discussion of existing “self-consistency” or “critique-guided” RL frameworks (e.g., CFT, CRL, Self-Refine).

### Questions
1. The “Reasoning Quality Reward” is computed from the log-likelihood gap between reasoning and direct-answer modes.
Can you show that this signal tracks true reasoning correctness rather than just higher output confidence?
For instance, do higher rewards actually correspond to more logically valid reasoning chains?


2. Since the reward depends on two separate generations per example, it may fluctuate.
How stable is this log-likelihood difference during training?
Did you observe large variance across steps or seeds that could destabilize learning?


3. Have you checked whether the model’s reasoning becomes objectively better when judged by an external verifier or a stronger model (e.g., GPT-4 or a logic checker)?
That would help confirm the reward improves reasoning, not just internal consistency.


4. The length adjustment idea resembles prior work like Adaptive Length Penalty, What is genuinely new here, and how much does your version contribute beyond those methods?


5. How could it generalize to open-ended questions or dialogue, where no ground truth exists?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The core contribution of the paper is the proposal of a process-based reward mechanism, which promotes better reasoning by giving more reward to effective intermediate reasoning steps (CoTs), rather than relying solely on rule-based evaluation of the final answers.

### Strengths
1. Good Research Question: Only RLVR may not be enough to supervise model for reasoning, so more reward on CoT has the potential to be better.

### Weaknesses
1. The main results of the paper **rely on the LogicTree training set constructed by the authors and are evaluated using its corresponding test set**, which imposes significant limitations.

It is unclear **why existing open-source reasoning datasets could not be adapted instead**？ 

Relying solely on LogicTree as the training set **does not sufficiently support the claimed “plug-and-play” capability**.

2. Results on Table 4 should be in the place of Table 1 because results from these benchmarks are more reliable on reflecting reasoning abilities and cared about by the community.

3. Typo: Line 190, missing citation

### Questions
1. What is the relationship between this method and your LogicTree dataset? Could this method be applied while leveraging existing open-source reasoning datasets, rather than being restricted to constructing LogicTree?

2. the main experimental results should be evaluated across a variety of commonly used logic/reasoning datasets and compared with more methods.

### Soundness
3

### Presentation
3

### Contribution
2
