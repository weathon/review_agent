# Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
An impediment to using Large Language Models (LLMs) for reasoning output verification is that LLMs struggle to reliably identify errors in thinking traces, particularly in long outputs, domains requiring expert knowledge, and problems without verifiable rewards. We propose a data-driven approach to automatically construct highly granular reasoning error taxonomies to enhance LLM-driven error detection on unseen reasoning traces. Our findings indicate that classification approaches that leverage these error taxonomies, or "rubrics", demonstrate strong error identification compared to baseline methods in technical domains like coding, math, and chemical engineering. These rubrics can be used to build stronger LLM-as-judge reward functions for reasoning model training via reinforcement learning. Experimental results show that these rewards have the potential to improve models' task accuracy on difficult domains over models trained by general LLMs-as-judges by +45%, and approach performance of models trained by verifiable rewards while using as little as 20% as many gold labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a data-driven method to automatically generate granular reasoning rubrics by extracting error patterns from a model's incorrect reasoning traces. These rubrics are then used to enhance an LLM-as-judge's ability to classify unseen reasoning traces, improving the specificity and overall accuracy of an LLM trace correctness classifier. Furthermore, it can serve as a strong reward function for RL.

### Strengths
1. The method is clearly presented and easy to follow, with intuitive motivations.

2. Experiment results demonstrate the superior performance of the rubric-augmented method compared to the baseline.

### Weaknesses
1. The paper fails to cite or compare against key existing research on rubric-based rewards [1, 2]. This omission makes it difficult to assess the paper's novelty, as it does not clearly differentiate its method from existing rubric-based approaches.

2. The experiments only benchmark the proposed method against a no-rubric baseline. This merely proves that using a rubric is better than not using one. To validate its core contribution, the paper must demonstrate that its specific method is superior to other existing rubric-augmented methods.

[1] Reinforcement Learning with Rubric Anchors  
[2] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains

### Questions
Please refer to the Weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a data-driven method to address the unreliability of LLM-as-a-judge frameworks for verifying complex, domain-specific reasoning. The core idea is to automatically generate a "rubric," or a granular error taxonomy, by prompting an LLM to analyze incorrect reasoning traces from a given domain. This rubric is then used as a checklist by an LLM-judge to create a more accurate and robust reward function for reinforcement learning. Experiments in coding, math, and chemical engineering show that rubric-augmented LLM judges outperform baseline judges in error detection (particularly specificity) and that RL-trained models using this rubric-based reward can approach the performance of models trained with verifiable "gold standard" rewards, while requiring fewer labels

### Strengths
The paper targets a critical bottleneck in training more capable reasoning models: the difficulty and cost of creating reliable reward signals. The proposed solution is intuitive. Instead of asking an LLM to abstractly "grade" a complex trace, the authors use a approach to equip the LLM with a concrete "failure checklist" (the rubric). This reframes an abstract evaluation problem into a more constrained and verifiable classification task.

### Weaknesses
The method’s cost-effectiveness remains insufficiently demonstrated. While Figure 6 indicates that the per-step runtime is comparable to the baseline, the baseline itself, relying on a single LLM-judge call already constitutes a major computational bottleneck in RL. A more detailed analysis of token usage and compute overhead between the two-pass rubric judge and the one-pass baseline would be needed to substantiate the claimed efficiency advantage.

The method shows limited generalizability beyond technical domains, performing well primarily in areas with clear, verifiable errors such as mathematics, coding, and chemical engineering. In contrast, the authors’ experiment on “Psychology & Philosophy” (Appendix B.5) reports a Balanced Accuracy of 0.567 using the rubric method, slightly below the baseline of 0.571, highlighting a key limitation in scope. While effective for detecting “catastrophic errors,” the approach falls short as a general solution for enhancing LLM-as-a-judge performance on qualitative, nuanced, or open-ended tasks.

Given that the method is primarily applicable to technical domains such as math and coding—where ground-truth labels are readily available, what is the motivation for employing a llm-based approach in these settings? How does it outperform or complement conventional supervised or verifiable reward training?

### Questions
Please see the weakness part

### Soundness
2

### Presentation
3

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
This paper introduces a method to automatically create "rubrics," or detailed checklists of reasoning errors, by analyzing a model's past mistakes in a specific domain. The goal is to solve the problem of LLMs being unreliable at self-correcting or judging complex reasoning. These data-driven rubrics are then used to build a much stronger LLM-as-judge reward function. Experiments show that using this rubric-based reward function in reinforcement learning can improve a model's task accuracy by up to +45% over standard LLM judges. This method approaches the performance of training with verifiable "gold" rewards but requires as little as 20% of the labeled data.

### Strengths
1. This paper addresses a key limitation of LLMs: their difficulty in reliably identifying errors in complex reasoning traces, especially in expert domains (like coding or math) and on problems without simple verifiable answers.

1. The method is shown to be effective. When LLM judges were augmented with the automatically generated rubrics, their ability to correctly identify incorrect reasoning traces (Specificity) improved dramatically—for example, from 12.2% to 63.4% on SWE-Bench and 16.1% to 61.3% on NuminaMath.

1. The rubric-augmented reward function is shown to be highly effective for reinforcement learning. It allows models to achieve task accuracy approaching that of models trained with "gold" verifiable rewards, but while using significantly fewer gold labels (as little as 20% mentioned in the abstract).

### Weaknesses
1. A significant limitation highlighted in the appendix is the classifier's poor performance on the training set itself. The authors note that the low specificity scores indicate that the classifier is unable to re-identify these errors it was trained on without the ground truth answers being provided for guidance. 

1. The ablation studies show that a larger rubric is not always better. For the coding domain, the smallest rubric size ($n=25$) actually outperformed most of the larger rubrics. 

1. The information loss of the "Trace Compression" step is unknown. The compressing LLM might misunderstand the original trace or inadvertently filter out subtle yet critical errors before the rubric extraction stage even begins. A comparison of uncompressed and compressed traces. 

1. The paper's primary successes are demonstrated in technical domains with verifiable answers (Math, Coding, Chemical Engineering). However, in the appendix (Section B.5), the method was tested on Psychology and Philosophy problems and performed worse than the baseline classifier. This suggests the approach may not generalize to qualitative domains where errors are more ambiguous or nuanced.

### Questions
1. The rubric is generated from the errors of one "imperfect model". How well does this rubric generalize to judging the reasoning traces of other models, especially those with different architectures or training methodologies (e.g., judging a Llama-3 trace with a rubric built from Qwen)?

1. Is it possible for the compressing LLM to mistakenly discard subtle but critical errors, or to misinterpret the final logical path, thereby corrupting the quality of the rubric from the very first step?

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
2

### Summary
This paper proposes an automated approach to construct granular reasoning error taxonomies ("rubrics") to enhance LLM-driven error detection in reasoning traces. The method extracts domain-specific error patterns from incorrect reasoning traces generated by models, organizing them into hierarchical rubrics with keywords for efficient lookup. These rubrics guide LLM-as-judge classifiers in identifying errors in unseen traces. The authors demonstrate that rubric-augmented classifiers improve error identification by up to 11.6% in technical domains (math, coding, chemical engineering).

### Strengths
- Novel application of automatic error taxonomy extraction to reasoning trace evaluation
- Multiple domains tested (coding, math, chemistry)
- Potential to reduce annotation costs in specialized domains

### Weaknesses
- Rubric generation requires Claude 3.5 Sonnet (closed-source); no experiments with open-source alternatives
- NuminaMath only evaluated on 100/350 validation problems
- How do we know generated rubrics are comprehensive and not redundant?

### Questions
- Can you provide the exact prompts used for rubric generation and classification?
- Why use different LLMs for RL training than rubric generation ?
- Do rubrics from one domain transfer to related domains?

### Soundness
2

### Presentation
2

### Contribution
2
