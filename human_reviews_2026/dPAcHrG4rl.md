# A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound, defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-of-concept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step reasoning methods: \faGithub  \href{https://anonymous.4open.science/r/InfoQA-55D1}{InfoQA}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the challenge of Multi-Hop Question Answering (MHQA), where multiple, dispersed, and interdependent pieces of evidence need to be integrated through sequential reasoning. This task is particularly difficult for large language models (LLMs) due to their finite output capacity per pass, which makes the integration of relevant evidence unreliable once the task complexity exceeds model capacity. The paper formalizes this bottleneck by introducing a Fano-style accuracy upper bound, which defines a theoretical performance ceiling for single-pass LLMs. This bound shows that as task complexity increases beyond model capacity, accuracy inevitably collapses.

Building upon this theoretical foundation, the authors propose a new multi-call framework for MHQA, called InfoQA. InfoQA addresses the capacity overflow by ensuring high per-step accuracy through capacity-aware task decomposition and active pruning of prior reasoning traces. This helps keep the information load within the model’s single-pass limit. Furthermore, InfoQA enhances robustness through a dependency-explicit workflow, which allows for precise control over the reasoning path. To validate their approach, the authors construct a stringent, noise-rich benchmark and show that the model's performance aligns with the theoretical capacity curves. Experimental results demonstrate that InfoQA consistently improves performance in MHQA tasks.

### Strengths
1. The introduction of the Fano-style accuracy upper bound provides a unique theoretical foundation for understanding the limitations of single-pass LLMs in handling complex MHQA tasks. This may provide some insights for future research in capacity-aware representation for LLMs.

2. The work provides a comprehensive experimental result which involves various settings.

### Weaknesses
1. While the InfoQA framework offers clear benefits, the multi-call nature of the approach could introduce overhead, especially for tasks requiring many iterations. This might limit its scalability for more real-time applications or systems with stricter performance requirements.

2. Overemphasis on Theoretical Bound: The theoretical Fano-style upper bound offers valuable insights into the performance limits of single-pass LLMs. However, the heavy focus on this theoretical bound may overshadow more practical solutions for real-world MHQA scenarios. The boundary established may not always align with the practical constraints of large-scale systems, where the focus should be on scalable, efficient reasoning. Moreover, while the paper introduces many complex formulas, they are not always effectively integrated with the core problem of MHQA. The formulas provide theoretical rigor but may not offer direct, actionable solutions for improving the model’s real-world performance.

3. It could benefit from a more direct comparison with other multi-hop question answering systems or frameworks.

### Questions
Authors may consider revising the presentation and add a main figure to help readers intuitively understand the paper and more baselines are expected to validate the effectiveness of this framework.

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
3

### Summary
This paper utilizes Information Theory to analyze the capability limitations of the single-pass solution to multi-hop question answering (MHQA), reveals the "Accuracy Cliff" phenomenon, identifies two challenges ("Stepwise Capacity Overflow" and "Cross-Step Error Accumulation"), and then proposes a multi-call framework (InfoQA) for MHQA.

### Strengths
1. The analysis, grounded in Information Theory, is insightful, and it reveals the limitations of single-pass MHQA solutions.
2. The experimental results (Table 2) demonstrate the effectiveness of the proposed InfoQA method.

### Weaknesses
1. I appreciate the theoretical analysis and experimental findings in this work, but I have some concerns regarding the analysis based on Information Theory:
    - I do not think information theory is suitable to assess the capability of natural language. Even if we consider all the possible values of a token (i.e., the size of the vocabulary) and compute the entropy or self-information of this token $H(X) = - \sum_{x \in X} p(x) log p(x)$, this still ignores the variability, richness, and ambiguity of the meaning conveyed by natural language. A single word can mean a lot.
    - With a finite number of tokens in a single pass, inefficient reasoners may suffer from the limited output budget, but efficient ones can use fewer tokens to achieve desirable results.
    - In addition, inefficient reasoners can utilize inference-time scaling strategies to extend their single-pass reasoning trace, making the limited budget not much of a fundamental issue.
2. The experiments are conducted on the synthesized dataset proposed by this paper, without empirical results on existing widely used MHQA benchmarks.
3. Section 5.2 and Figure 5: The empirical results do not align well with the theoretical curves (Accuracy Cliff) to me.

### Questions
- Note: I would rate this paper as 7 initially, although this option is not available.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper conducts an information-theoretic analysis of multi-hop question answering (MHQA) based on large language models (LLMs). Through a fano-style accuracy upper bound, it reveals a phenomenon of an “accuracy cliff” in single-step reasoning. While recent studies have emphasized the existence of reasoning errors and cascading error propagation and have attempted to address these issues from multiple perspectives, the theoretical insights provided in this paper are novel. Furthermore, the authors develop a capacity-aware multi-call proof-of-concept system called InfoQA. In the experiments, they provide both theoretical validation and framework validation.The experimental results show that the empirical findings are consistent with the predicted capacity curves, and that InfoQA also achieves performance improvements.

### Strengths
(1) Focusing on information theory, this paper offers a novel perspective by analyzing MHQA from the standpoint of reasoning errors and cascading error propagation in the reasoning process. The rigorous theoretical analysis and findings contribute to readers’ better understanding for MHQA.

(2) Motivated by the theoretical findings, the authors propose a capacity-aware multi-call proof-of-concept famework, which mitigates the issues of the accuracy cliff and cascading errors by reducing information requirements and employing pruning strategies. 

(3) The experimental design consists of theory validation and famework validation, which verify the effectiveness of the proposed ideas and methods.

### Weaknesses
(1) The presentation of the paper could be improved, although this can be addressed in a short time. For example, Figures 1 and 2 should include clearer explanations of how they illustrate that the limited token capacity of LLMs constrains information transmission, as well as what the “step = 8” in the figures specifically represents.

(2) The authors claim that this work is the first information-theoretic analysis of MHQA. However, some related studies may have been overlooked. Adding discussions that connect this work with recent research would strengthen the contribution and help readers better position it within the broader literature.

(3) Hyperparameter analysis: including additional experimental studies on parameters such as the pruned ratio and the model capacity $C$ after question decomposition would further enhance the empirical section and provide insights.

### Questions
(1) Why is $H(Y)$ used to represent model capacity? How is $H(Y)=200$ determined in the experiments?

(2) How does Figure 1(a) demonstrate that the limited number of LLM tokens constrains the capacity of models to transmit information, and why is $step = 8$ chosen? How does the setup of Figure 2 illustrate the limitations inherent in the single-pass reasoning paradigm, and could similar issues also arise in the multi-call reasoning paradigm?

(3) In the pruned context, how does the pruned ratio affect the model performance?

(4) In capacity-aware task decomposition (Page 6), how to ensure that each single-step reasoning process remains within the processing capacity $C$ of model? Is it possible that, even after decomposition, some subtasks still exceed the effective model capacity $C$?

(5) What are the differences between capacity-aware task decomposition and question-decomposition-based MHQA? To maintain rigor, the authors should avoid using absolute statements such as “the first information-theoretic analysis of MHQA”, as prior research has already demonstrated clear motivations, which may have been overlooked in the this work. It would be helpful for the authors to include a discussion, that situates their approach in relation to recent studies [1,2].

[1] LOG: A Local-to-Global Optimization Approach for Retrieval-based Explainable Multi-Hop Question Answering. COLING2025

[2] How Many Parameters for Multi-Hop? An Information-Theoretic Capacity Law for Knowledge Retrieval in Large Language Models. In Knowledgeable Foundation Models at ACL 2025.

### Soundness
3

### Presentation
2

### Contribution
3
