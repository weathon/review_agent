# DoReMi - Difficulty-Oriented Reasoning Effort Modeling of Science Problems for Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4

## Abstract
We introduce DoReMi (Difficulty-Oriented Reasoning Effort Modeling), a structured framework leveraging an extended Bloom's taxonomy to comprehensively characterize intrinsic problem difficulty for large language models on scientific reasoning tasks. DoReMi systematically annotates problems along seven cognitive and methodological axes using judge LLMs distinct from those being evaluated, with human annotations confirming the validity of these assessments. We empirically quantify LLM reasoning effort through metrics including minimum reasoning tokens required for solution, expected trials to first success. Our validation demonstrates strong agreement across diverse judge LLMs spanning both open-source and proprietary LLMs. Evaluations on GPQA, ARC, and SuperGPQA reveal that our multidimensional difficulty fingerprints correlate strongly with and enable accurate predictive modeling of LLM reasoning effort. DoReMi enables principled difficulty-aware subset selection that substantially outperforms static-difficulty baselines while providing interpretable diagnostics that uncover emergent reasoning capabilities across successive model generations. This framework offers actionable insights for benchmark design and targeted post-training improvements toward higher-order reasoning skills.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce DoReMi (Difficulty-Oriented Reasoning Effort Modeling), a structured framework that leverages an extended Bloom’s taxonomy to comprehensively characterize the intrinsic difficulty of scientific reasoning tasks for large language models. DoReMi systematically annotates problems along six cognitive and methodological axes using judge LLMs distinct from those being evaluated, with human annotations confirming the validity of these assessments. The authors empirically quantify LLM reasoning effort through metrics including the minimum reasoning tokens required for a solution and the expected number of attempted runs to the first correct answer.

### Strengths
The authors present a clearly articulated framework for constructing evaluations, supported by analysis comparing the results with human annotations and enhanced with visualizations through accompanying figures.

### Weaknesses
The evaluation construction method proposed by the authors is not particularly novel, but it lacks references to and comparisons with similar works, such as:  
[1] WritingBench: A Comprehensive Benchmark for Generative Writing  
[2] HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models  
[3] DynamicBench: Evaluating Real-Time Report Generation in Large Language Models  
as well as other relevant open-source efforts.

### Questions
Same as above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DoReMi, a framework that uses an extended Bloom's taxonomy to characterize the intrinsic difficulty of scientific reasoning problems for LLMs. The authors systematically annotate problems along six axes (Cognitive Level, Knowledge Dimension, Method Difficulty, Definition Completeness, Knowledge Breadth, Number of Reasoning Steps) and correlate these with empirical reasoning effort metrics (MRT, R2FCA). The framework enables difficulty-aware evaluation and provides interpretable diagnostics of LLM reasoning capabilities.

### Strengths
- Well-Motivated Problem: The paper addresses a genuine need in LLM evaluation - moving beyond single-dimensional accuracy scores to understand why problems are difficult for reasoning models. 
- Theoretically Grounded Framework: Using Bloom's taxonomy provides a principled, interpretable foundation rather than ad-hoc difficulty metrics. The six-axis extension is thoughtfully designed.

### Weaknesses
- Should demonstrate your metrics through RL training: The paper repeatedly claims DoReMi provides "actionable insights for targeted post-training improvements”.  However, no experiments validate that training on DoReMi-selected samples actually improves model performance. Suggested experiments:
    - Train smaller models using DoReMi-guided curriculum learning
    - Compare sample efficiency against random or static difficulty baselines
    - Perform targeted fine-tuning on identified weak Bloom axes
    - Use DoReMi difficulty scores for reward shaping in RL

- Judge Model Overlap Concerns: The paper uses reasoning LLMs as judges, but also evaluates reasoning LLMs. While they claim judges are "distinct from those being evaluated," some overlap exists (e.g., o3-mini as judge, o3-mini-high as evaluation target). Potential for judges to be biased toward difficulty patterns they themselves exhibit

- Generalization Concerns: Evaluated only on science/STEM problems - unclear if taxonomy applies to other reasoning domains (coding, math, logic puzzles). All benchmarks are multiple-choice or short-answer - what about open-ended reasoning?

### Questions
Please refer weaknesses

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
3

### Summary
The paper proposes a learning-based framework to provide a comprehensive estimation of problem difficulty for LLMs. The framework considers six cognitive and methodological axes based on Bloom’s taxonomy and combines the performance of the target LLM.

### Strengths
1. The paper proposes a structured and multidimensional evaluation framework to provide a more comprehensive and precise estimate of the question complexity
2. The paper illustrates the applications of the proposed framework: filtering challenging questions and providing a systematic analysis of LLM reasoning capabilities.

### Weaknesses
1. The pipeline involves substantial LLM usage: question labeling requires LLM inference, and reasoning effort calculation requires multiple samples from the target LLM. The whole pipeline seems to be computationally expensive
2. The predictor training process requires collecting responses from target LLMs. It is unclear whether the learned neural network generalizes to other models. If not, Phases 2-4 would need to be repeated from scratch for each new target LLM. 
3. Mathematical problems like GSM8k are usually considered as a benchmark to evaluate LLM reasoning capabilities. Including performance on math problems would provide a more comprehensive evaluation of the current framework.

### Questions
1. For the 6 dimensions of extended Bloom’s taxonomy, what is the reason to include “Definition Completeness”? This property seems more related to the solvability, not the difficulty. For “Number of Reasoning steps”, how to define the essential logical action? Is there any question with multiple correct solutions, and the number of reasoning steps is different?
2. The MRT is selected as the primary feature to train the neural network, because it presents the highest correlation. Why does this metric correlate more strongly with the designed complexity/difficulty definition than other metrics?

### Soundness
3

### Presentation
3

### Contribution
3
