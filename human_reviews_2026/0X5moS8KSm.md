# Selection, Reflection and Self-Refinement: Revisit Reasoning Tasks via a Causal Lens

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Due to their inherent complexity, reasoning tasks have long been regarded as rigorous benchmarks for assessing the capabilities of machine learning models, especially large language models (LLMs). Although humans can solve these tasks with ease, existing models, even after extensive pre-training and post-training at scale, still fail to perform reasoning reliably. In this paper, we revisit reasoning tasks from a causal perspective, seeking to understand their behavior in latent space and to offer insights for addressing their challenges. Specifically, we cast reasoning tasks as a selection mechanism, in which high-level logical concepts function as selection operators on the given observations, such as, identifying the correct answer in a math problem or filling the appropriate entry in Sudoku. We emphasize two key properties of this formulation that shed light on the difficulty of reasoning tasks. First, the latent space exceeds the observation space in complexity, even when the correct answer is fully determined by the observed input. Second, the latent variables, corresponding to logical thought, are densely structured and exhibit strong dependencies.  Building on this formulation, we introduce a framework, called SR$^2$, that incorporates the estimated latent variables as feedback into the selection mechanism, thereby facilitating the learning of dense dependencies among latent representations. The framework consists of three key modules: reflective representation learning, dependency self-refinement, and periodic intermediate alignment. Experimentally, we show that our approach yields significant gains in reasoning accuracy, for example, attaining over 10% improvement in performance with 8$\times$ fewer parameters on the Sudoku and Maze tasks over the recent advances.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper formalize reasoning problem as selection mechanisms through a causal perspecitive. Based on this formulization, the paper summarizes two hypothesis in reasoning tasks, which inspires the SR2 framework. The framework define reflection as a fixed-point problem and used a flatten recurrent transformer to solve it. Further, the dependency self-refinement and periodic alignment are proposed to learn dependency and formalize training.

### Strengths
1. The idea of formalizing reflection as a fixed-point equation is novel and interesting. 
2. The proposed method achieved good performance on Sudoku-Extreme and Maze-Hard dataset with less number of parameters.

### Weaknesses
1. The paper is a bit hard to follow (See question 1-2).
2. A general performance comparison with baselines are missing, including training / test cost. 
3. Adding more benchmarks (e.g. ARC-AGI) would be helpful. One of my concern on proposed framework is the fixed-point formulation may only works for problems with specific structure, not all reasoning tasks.

### Questions
1. In Ln. 162 - 169, is there a formal representation of $\mathcal{S}$ or $z$? E.g. How is $z$ represented by $x, y$? I think as an example it should illustrate how to formalize rules in reasoning problems into selection variables $z$ and constraints $S$, instead of leaving an abstract expression as in Eq. (2).
2. What is the motivation of formalizing $z$ as the solution a fixed-point equation (3)? How is this related to hypothesis 1? The connection is unclear.

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
This paper frames reasoning tasks as causal selection mechanisms, where latent reasoning rules constrain observed space. The authors propose two hypotheses including 1/ latent space complexity exceeding observations and 2/ dense interdependencies among latents. Based on these two hypotheses, the authors propose the SR2 framework to address the challenges involved in reasoning. Specifically, SR2 integrates a reflective representation learning module, a dependency self-refinement module, and a periodic alignment module. Evaluations on Sudoku and Maze show >10% accuracy gains over Hierarchical Reasoning Model (Wang el al., 2025) with 8x fewer parameters.

### Strengths
1. The causal lens provides a structured view of reasoning difficulties, potentially useful for latent modeling in ML.
2. Hypotheses on latent complexity and dependencies are insightful and well-illustrated with examples like Sudoku.
3. Empirical gains in reasoning efficiency when model size is small.

### Weaknesses
1. As acknowledged by authors, benchmarking of this paper is limited to Sudoku and Maze Navigation with clear and simple logical constraints. This may limit the claims of generality. Suggesting adding additional evaluations on STEM tasks. 
2. Missing discussion of related work. The reflection framework of SR2 is related to the established line of work on self-correction techniques. For example, the Reflexion paper (Shinn el al., 2023) introduced verbal self-reflection.

### Questions
The introduction mentioned scaling-based progress (e.g., Kaplan et al., 2020). With your efficiency gains, do you foresee SR2 scaling to frontier LLMs, or is it better suited for smaller, specialized models?

### Soundness
3

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
4

### Summary
This paper interprets reasoning tasks such as Sudoku and maze solving as modeling problems governed by latent variables that act as selection constraints. It argues that the latent space underlying reasoning exhibits high complexity and strong interdependencies among variables, which explains why existing models still struggle to reason effectively even after large-scale training. This paper proposes a new reasoning framework SR^2, designed to model these latent dependencies and iteratively refine latent variables. SR^2 outperforms recent baselines such as the Hierarchical Reasoning Model (HRM), achieving higher performance with fewer parameters, which supports the motivation and design of the proposed framework.

### Strengths
The proposed framework redefines reasoning as a latent-variable-based constraint mechanism, explaining the difficulty of reasoning tasks through the perspective of latent complexity and dense dependencies.

The SR^2 architecture consists of selection, reflection, and self-refinement modules, which directly models and optimizes latent variables, achieving improved performance on Sudoku and Maze tasks with a smaller parameter budget.

### Weaknesses
The connection between Eq. 1 and the proposed framework remains unclear. Eq. 1 defines a generative process from latent rule variables z to observed pairs (x, y) and introduces the notion of selection constraints S derived from z. However, the way these elements are instantiated or represented within SR^2 is not clearly explained.

The experiments do not include the ARC task, leaving the reasoning ability insufficiently validated.

This paper lacks discussion of the training cost. If supervision is applied to different hierarchical levels of latent variables z, the model may face risks of gradient instability, potentially increasing computational overhead.

### Questions
How does the framework model the process described by Eq. 1? e.g., how are the rule variables z represented, and how do they enforce corresponding constraints within SR^2? Which part of SR^2 corresponds to the function g in Eq. 1?

Are there difficults that prevent SR^2 from handling ARC-style reasoning tasks?

The paper claims that SR^2 “explicitly integrates Selection, Reflection, and Self-Refinement.” How is the Selection component concretely implemented in the architecture?

Compared to previous SOTA methods, how does the training cost of SR^2 (in time or memory) scale?

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
3

### Summary
This paper proposes a novel causal formulation of reasoning tasks, framing them as selection mechanisms in which logical concepts act as operators that constrain observed inputs. Building on this, the authors identify two key hypotheses—latent space complexity (the reasoning latent space is exponentially larger than the observation space) and dense interdependence (latent variables are strongly coupled)—as root causes of reasoning difficulty. To address these challenges, the authors introduce SR² (Selection, Reflection, and Self-Refinement), a framework that explicitly models iterative latent-space refinement. SR² includes three modules: (1) reflective representation learning that iteratively updates latent variables conditioned on inputs, (2) dependency self-refinement that enforces latent consistency by removing input signals, and (3) periodic alignment to stabilize training. The approach uses a flattened recurrent Transformer architecture and is evaluated on Sudoku-Extreme and Maze-Hard reasoning benchmarks, achieving up to 10% higher accuracy with 8× fewer parameters than prior state-of-the-art models (e.g., HRM).

### Strengths
* Novel causal perspective: Formulating reasoning as a selection mechanism is a theoretically original angle connecting causality and reasoning.
* Well-structured SR² pipeline: The three-module design (reflection, self-refinement, periodic alignment) is coherent and practically motivated by the stated hypotheses.
* Strong empirical results: Outperforms HRM (ICLR 2025) by 11.6% on Sudoku-Extreme and 19.2% on Maze-Hard, showing competitive reasoning ability with much smaller models.

### Weaknesses
* Limited empirical scope: Evaluations on Sudoku and Maze tasks, while useful, do not establish general reasoning improvement beyond symbolic settings. It would be interested to see if the approach can be applied to more open-ended domains such as math problem with natural language as the interface. How SR² behaves under large-scale LLM reasoning (e.g., text-based chain-of-thought) is not explored.
* Ablations vs. causality: While the causal “selection mechanism” framing is conceptually appealing, the improvements could be fully explained by iterative refinement and alignment, independent to the causal lens.

### Questions
Q1: Authors state that the latent space is “exponentially larger” than the observation space. Is this a theoretical claim that can be formalized (e.g., by counting admissible configurations in Sudoku), or an empirical observation? How could one measure latent complexity in your setup?

Q2: The system is majorly motivated from selection mechanism, however the system is more inherited from DEQ or Implicit Deep Learning frameworks. How does SR² differ theoretically from them? Is there any convergence guarantee for your fixed-point iterations?

### Soundness
4

### Presentation
4

### Contribution
3
