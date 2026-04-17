# PathFinder: Graph-structured Reasoning for Medical Visual Question Answering

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Medical Visual Question Answering (MVQA) aims not only to predict correct diagnoses but also to provide explicit, clinically-grounded reasoning to enhance interpretability, to foster clinician trust, and to support AI-assisted decision-making.  Despite the recent advances, the explanations of existing MVQA are often incomplete and non-causal, neglecting key evidence, intermediate steps, and alternative hypotheses. In this paper, we present PathFinder, a graph-structured reasoning framework, in which medical entities are represented as nodes and causal/evidential relations as edges, enabling systematic traversal of diagnostic pathways. In PathFinder, we define two structural reasoning dimensions: step-wise exploration, which encourages PathFinder to traverse intermediate entities with causal links, and branch-wise exploration, which encourages exploring alternative diagnostic routes and ruling out unlikely options.Further, we introduce Graph-GRPO to integrate graph-structured supervision with two process-level rewards: a Step Reward for causally coherent reasoning, and a Branch Reward for systematic exploration of alternatives, complemented by outcome accuracy. Experiments on seven multimodal and seven text-only benchmarks consistently show that PathFinder outperforms state-of-the-art methods, while producing reasoning results that are causally coherent and structurally comprehensive. Codes will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a framework called PathFinder, a graph-structured reasoning framework for medical visual question answering. It represents clinical entities and causal/evidential links as nodes and edges, defines two new metrics (step-wise and branch-wise exploration), and trains with a new RL variant Graph-GRPO using three rewards (step, branch, outcome accuracy).
The paper reports experiments on 7 multimodal and 7 text-only medical benchmarks, claiming consistent performance gains.

### Strengths
- Extensive benchmarks and ablation studies.
- The high-level architecture and reward structure are well described.
- Combining graph-structured CoT supervision with process-level rewards is potentially impactful.

### Weaknesses
- Introduced Step and Branch evaluation metrics do not seem convincing to reflect the real impact of each method. Step counts the number of reasoning steps, branch counts if the reasoning steps are present for each option. However, throughout the paper it is difficult to derive the usefulness of these metrics. Although the authors mention the tendency of longer reasoning steps for correct answers in domain-specific VLMs and the opposite for general VLMs, it remains unclear whether such behavior is universal for all cases in all evaluated models or due to the underlying model architecture.
- Unclear which metric is reported in Tables 3 and 4.
- No evaluation for factuality, i.e., it remains uncertain whether the models generate factually correct reasoning. A human evaluation could have been beneficial in this regard.

### Questions
Refer to the weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PathFinder, a graph-structured reasoning framework for medical visual question answering (Med-VQA). Unlike traditional models that focus only on answer accuracy, PathFinder explicitly represents the reasoning process as a graph composed of medical entities such as symptoms, imaging features, hypotheses, and conclusions, connected by causal and evidential edges. It evaluates reasoning quality along two complementary dimensions: step-wise exploration, which measures causal coherence within the reasoning chain, and branch-wise exploration, which assesses the systematic exploration and elimination of diagnostic alternatives. The authors further propose Graph-GRPO (Graph-based Group Relative Policy Optimization), which integrates two structured rewards—step reward and branch reward—into reinforcement learning to encourage causally consistent reasoning and comprehensive diagnostic exploration. Experiments on seven multimodal and seven textual Med-VQA benchmarks demonstrate that PathFinder-7B achieves substantial improvements over existing models such as Lingshu-7B, Chiron-o1-8B, and MedVLM-R1-2B, offering enhanced clinical interpretability and structurally coherent reasoning paths.

### Strengths
1.	The paper formalizes the medical diagnostic process as a graph-structured reasoning paradigm, explicitly modeling entities and causal relationships to provide a systematic framework for interpretable reasoning in Med-VQA.
2.	Graph-GRPO jointly optimizes both step-wise and branch-wise dimensions, offering stronger process-level supervision and verifiability compared to traditional accuracy-only reinforcement learning approaches.
3.	Extensive experiments across 14 datasets demonstrate that PathFinder achieves state-of-the-art performance on both multimodal and textual Med-VQA tasks, while significantly enhancing reasoning depth and coverage.

### Weaknesses
1.	The construction of graph structures and reasoning evaluation relies on the LLM-as-a-Judge approach (GPT-4o), which may introduce bias and instability, lacking automated and reproducible validation.
2.	The graph generation and reasoning path construction require multiple iterative calls and manual verification, leading to high computational and human costs that hinder large-scale deployment.
3.	The performance may be sensitive to hyperparameter configurations, yet the paper does not provide systematic ablation studies or sensitivity analyses.
4.	Although the authors claim expert validation, no inter-rater consistency metrics or error case analyses are reported, leaving the clinical reliability insufficiently quantified.
5.	The paper lacks discussion of related studies on the application of Chain-of-Thought (CoT) reasoning in medical contexts, which could provide valuable comparative insights and theoretical grounding.

### Questions
1.	How do the authors ensure the consistency and correctness of each edge (causal or evidential) within the constructed graph? Are there potential issues of pseudo-causal chains or redundant edges?
2.	Do the Step and Branch Rewards in the Graph-GRPO framework conflict during optimization? How are these two objectives balanced to maintain stable learning?
3.	If PathFinder were applied to other medical modalities (e.g., multi-organ CT or pathology slides), would the graph node definitions and construction strategy require adaptation or re-design?
4.	Can the reliability of the LLM-as-a-Judge evaluation be quantitatively assessed? Have the authors considered measuring agreement between expert and model judgments using metrics such as Cohen’s κ?
5.	Since the current model performs single-round reasoning, could the framework be extended to multi-turn interactive diagnostic reasoning to better reflect clinical workflows?
6.	Could the graph structure be integrated directly into the visual encoding or Transformer layers rather than applied as a post-hoc reasoning stage?
7.	In real-world clinical scenarios, how can the system prevent the generation of misleading or overconfident explanations?
8.	Compared with existing paradigms like structured reasoning and process supervision, does PathFinder demonstrate sufficient generalizability and adaptability across diverse medical tasks?

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
4

### Summary
This paper proposes PathFinder, a graph-structured reasoning framework designed to enhance the reasoning capability of medical VLMs in medical VQA tasks. The authors argue that existing VLMs often suffer from step-wise and branch-wise deficiencies—i.e., their reasoning chains are incomplete, non-causal, or fail to systematically explore alternative diagnostic routes. To address this, the paper introduces Graph-GRPO, which encourages both step-wise and branch-wise exploration by incorporating graph-structured supervision along with Step Reward, Branch Reward, and outcome accuracy.
However, the step and branch rewards only encourage more logical and comprehensive reasoning, without explicitly evaluating the correctness of the reasoning process. In my view, this is the major weakness of the current framework and an important direction for future improvement. Extensive experiments on multiple multimodal and text-only benchmarks demonstrate that PathFinder achieves superior accuracy and produces more structured and comprehensive reasoning chains than existing models.

### Strengths
1. The proposed Graph-GRPO effectively enforces a more causal, logical, and comprehensive reasoning process through graph-structured supervision.

2. The PathFinder-7B model trained with this strategy achieves better performance than existing medical VLMs across multiple benchmarks.

### Weaknesses
1. The Step Reward and Branch Reward primarily encourage the model to produce logically coherent and comprehensive reasoning but do not evaluate the correctness or factual validity of the reasoning process.

2. The construction of the reasoning graph is a key component of the framework, yet the definitions of nodes and edges are somewhat vague. Specifically, nodes are defined as medical findings, clinical symptoms, hypotheses, rules, evidence, and imaging features or regions of interest, while edges represent support, derive, refute, or rule out relations. The rationale behind these specific categories is not clearly justified—for instance, why are symptoms and imaging features not considered subtypes of findings? And why are the edge types limited to these four relationships?

3. The proposed approach is tailored for multiple-choice or closed-ended medical VQA tasks. Its applicability to open-ended VQA tasks appears limited.

### Questions
1. Could the authors further clarify the taxonomy of nodes and edges in the reasoning graph? What criteria were used to define and separate these categories?

2. How could the proposed framework be adapted for open-ended medical QA or report generation tasks, where no predefined candidate answers exist?

3. Have the authors considered incorporating a reasoning correctness or factual consistency reward, in addition to the step and branch rewards, to more directly guide the model toward clinically valid reasoning paths?

### Soundness
3

### Presentation
3

### Contribution
3
