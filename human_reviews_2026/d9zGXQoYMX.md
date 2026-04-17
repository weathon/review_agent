# PRISM: A Multi-Dimensional Verification Approach to Mitigate Hallucinations in Chain-of-Thought Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
In recent years, Chain-of-Thought (CoT) verification has emerged as a critical research direction. However, existing approaches largely focus on the quality of intermediate reasoning or final answer correctness, while hallucinations arising from the initial stage of question understanding remain underexplored. To address this gap, we propose a unified framework—PRISM (Progressive Reasoning with Instructional and Strategic Multi-dimensional Verification) that jointly tackles all three aspects. We introduce a Commonsense-Augmented Progressive Instructional Reasoning (CPIR) method, designed to alleviate condition hallucination while utilizing commonsense to capture relevance between conditions and questions. Then we develop Multi-Dimensional Heterogeneous Collaborative Verification (MHCV), which strategically validates reasoning chains from multiple perspectives to enhance intermediate reasoning quality and question comprehension, thereby mitigating different types of hallucinations. In addition, we propose a Discard-Weighted Voting mechanism to overcome the limitations of traditional voting methods in multi-dimensional verification. Experimental results demonstrate that PRISM consistently improves verification accuracy across conditions, logical reasoning, and question comprehension, yielding more reliable reasoning chains and higher final-answer accuracy compared to strong CoT baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a new prompting frameworking: PRISM, which aims to reduce hallucinations in long CoT reasoning. Specifically, the paper first proposes Commonsense-Augmented Progressive Instructional Reasoning (CPIR) which explicitly list commonsense knowledge, and ask related instructions in the reasoning process. The paper further introduces multi-dimensional heterogeneous collaborative verification (MHCV) which verifies the faithfulness of the reasoning process. Additionally, the paper proposes a Discard-Weighted Voting mechanism to replace majority voting.

### Strengths
- The problem of hallucination and faithfulness in the reasoning process, especially in long-cot scenarios, is an important problem towards more robust and stronger reasoning model; the paper provides insights on different aspects to approach the problem, including "enhancing commensense knowledge", "more fine-grained verification", etc
- The paper writing is clear and the ablation study on each of the components are presented in Table 3.

### Weaknesses
- I find the major weakness of the work is that it lacks a well defined problem and the proposed method is three weakly-related prompting engineering methods, each targeting a slightly different problem; It is hard to find a core contribution beside prompt engineering efforts; and lacks a deeper insights on how to solve the hallucination problem during model training.
- In Table 2, without verification results in better performance in most times, it is hard to justify why the MHCV is still necessary (although the authors mentioned that this is also observed in a previous work, it does not make sense why this is acceptable)
- The paper lacks discussion on the computational overhead given the much more complicated prompting strategy. Since the paper is an inference-only approach, it is important to compare inference time as a reference for the trade-off between performance and efficiency

### Questions
- If I understand correctly, in Table 1, the random performance would be 50%; does that mean that the CoT baseline achieves only random performance on most tasks? This does not seem natural to me, could you present the full prompts for all methods?

### Soundness
2

### Presentation
2

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
This study focuses on the "problem understanding stage hallucination" that has been overlooked in existing chain-of-reasoning verification, fills the research gap of only focusing on intermediate reasoning quality or final answer correctness, and the proposed CPIR and MHCV modules form a complete logical chain covering "problem understanding - intermediate reasoning - answer verification".

### Strengths
1. The proposed PRISM framework is innovative in multi-dimensional verification of chain-of-reasoning hallucinations and has certain academic value overall.
2. It integrates forward verification (detecting conditional and logical errors) and reverse verification (identifying problem understanding biases), and proposes a discard-weighted voting mechanism. This mechanism avoids the excessive discarding of reasoning chains with minor flaws in traditional voting, and has promoted the methodology of chain-of-reasoning verification to a certain extent.
3. The experimental design is relatively complete. It selects two types of datasets: arithmetic reasoning (GSM8K, AddSub, etc.) and semantic reasoning (Last Letter, Date Understanding), covering different reasoning scenarios. The role of each module is verified through ablation experiments, and comparisons are made with mainstream baseline models, resulting in relatively convincing outcomes.

### Weaknesses
1. It only mentions "low verification accuracy of problem understanding hallucinations" and "lack of task adaptation", without in-depth analysis of the causes (such as the technical root of the volatility of LLM semantic scoring), nor does it propose specific future improvements.
2. It fails to clearly explain the annotation rules and consistency check methods when manually annotating 250 "problem-reasoning chain" samples (the objectivity of sample annotation is questionable), and the discussion depth is insufficient.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
PRISM reduces hallucinations in Chain-of-Thought reasoning through a two-stage approach: first, it enhances reasoning with commonsense-augmented progressive instructions; second, it applies multi-dimensional verification combining forward checks and backward reconstruction, finalized by a weighted voting mechanism that preserves partially valid chains. Experiments on arithmetic and semantic reasoning benchmarks show that PRISM reduces diverse hallucination types and enhances verification and answer accuracy compared to strong CoT baselines.

### Strengths
1.While forward/backward verification exist in prior work, their integration into a unified framework to combat distinct hallucination types is a distinct contribution. 
2.The paper's quality is high, evidenced by a rigorous methodological design and thorough experimentation.
3.The paper is generally clearly written. The problem statement is well-defined, and the PRISM framework is explained in a structured, step-by-step manner. The use of formal notations adds precision and architectural diagrams effectively aids comprehension.
4.Hallucination in CoT reasoning is a critical barrier to the reliable deployment of LLMs. PRISM represents a meaningful advance towards more robust and trustworthy reasoning systems.

### Weaknesses
1.The experimental validation on five datasets is primarily test well-defined, closed-world reasoning. The framework's effectiveness on more open-ended and complex tasks remains unproven. 
2. The paper correctly identifies the volatility of LLM-based semantic scoring as a key limitation of the backward verification module. However, the solution—tuning a threshold (τ) is a brittle and non-generalizable.
3. The paper fails to quantify the totaloverhead imposed by the full PRISM pipeline. Condition, logic, backward and voting represent a substantial increase in inference time and API calls compared to standard CoT.

### Questions
1. If the LLM has a fundamental misunderstanding of the question, wouldn't the generated instructions also be flawed from the start, leading the entire reasoning chain astray? How does CPIR mitigate this initial misdirection?
2.When the final answer is incorrect despite passing verification, what type of error is most frequent?
3.PRISM framework involves significant computational overhead. Have you quantified the average increase in token usage or latency compared to standard CoT or a strong baseline? 
4.The ablation study in A.2 suggests the optimal weights for Voting might be task-dependent. Did the dynamic weighting scheme is explored?
5.In tab.3, removing the commonsense module causes the largest accuracy drop on AQuA-RAT (71.26% → 66.54%) compared to other datasets. Why is it more dependent on commonsense supplementation than others?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes PRISM (Progressive Reasoning with Instructional and Strategic Multi-dimensional Verification), a framework that addresses hallucinations in Chain-of-Thought (CoT) reasoning through two main components: (1) Commonsense-Augmented Progressive Instructional Reasoning (CPIR) for the reasoning stage, and (2) Multi-Dimensional Heterogeneous Collaborative Verification (MHCV) for verification. The approach targets three types of hallucinations: condition hallucination, logical errors, and question-comprehension hallucination. The authors evaluate their method on five datasets (GSM8K, AddSub, AQuA-RAT, Last Letter, Date) using GPT-3.5-Turbo and DeepSeek V3.

### Strengths
* Useful Setting. It is critical to detect logic errors and question understanding.
* Clear presentation. The writing is easy to follow.
* Comprehensive experiments that include five different datasets and two backbones.

### Weaknesses
* Lack of simple baselines: The proposed work requires more token budgets, and a simple baseline comparison should be self-consistency. The lack of baseline methods makes me unclear how effective PRISM is.
* Limited theoretical justification for weight assignments: The choice of weights (ω₁=0.4, ω₂=0.4, ω₃=0.2) appears somewhat arbitrary. While Table 5 provides some exploration, the paper lacks a principled justification for weighting condition and logic errors equally and higher than question-comprehension errors across all tasks.
* MHCV seems less effective since Table 2 shows that MHCV most of the time reduces the performance. However, Table 3 shows that removing verification results in a performance drop. It seems the numbers in the two tables somewhat contradict each other.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2
