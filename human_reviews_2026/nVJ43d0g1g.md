# Adaptive Test-Time Reasoning via Reward-Guided Dual-Phase Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Large Language Models (LLMs) have achieved significant advances in reasoning tasks. A key approach is tree-based search with verifiers, which expand candidate reasoning paths and use reward models to guide pruning and selection. 
Although effective in improving accuracy, these methods are not optimal in terms of efficiency: they perform simple decomposition on the reasoning process, but ignore the planning-execution nature of tasks such as math reasoning or code generation. This results in inefficient exploration of reasoning process. To address this, we propose a dual-phase test-time scaling framework that explicitly separates reasoning into planning and execution, and performs search over the two phases individually.
Specifically, we decompose reasoning trajectories and develop reward models for each phase, enabling the search to explore and prune plans and executions separately. We further introduce a dynamic budget allocation mechanism that adaptively redistributes sampling effort based on reward feedback, allowing early stopping on confident steps and reallocation of computation to more challenging parts of the reasoning process. Experiments on both mathematical reasoning and code generation benchmarks demonstrate that our approach consistently improves accuracy while reducing redundant computation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work explicitly decompose the tree-based search steps into planning and execution phases, and use a reward model to guide pruning the selection. It also introduces a dynamic budget allocation mechanism to allocate the sampling effort for each search step. Experimental results demonstrate consistent improvement on both math and coding tasks.

### Strengths
1. The proposed method sounds reasonable for the reasoning tasks
2. Comprehensive experiments are performed, with various tasks and base models, at a wide range of token budgets. The results are also promising.

### Weaknesses
1. The main issue lies in the lack of technical contribution. In short, the only novel part of this work lies in the proposal to explicitly split the search step into planning and decomposition phases. But why it would work, what is the nature of planning and execution from the machine learning perspective, how the paradigm could be generalized to other tasks, seem to be unclear. The authors leave all these questions unanswered, leaving very few takeaways from the reader side. 
2. The proposed method contains a bag of tricks, but the ablation study does not perform a careful evaluation to verify which component makes the main contribution. 
3. There are too many heuristic designs, such as the width and budgets. All these factors make the proposed method hard to generalize.

### Questions
1. How the model would behave when no explicitly decomposing the search into planning and execution (paradigm Figure 3(c))?
2. How do you instruct the model to do planning and execution? Have you trained the model to do it?

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
3

### Summary
This manuscript proposes the DREAM method, which guides large language models to decouple planning and execution steps during reasoning, and to select each step using its corresponding reward model. In addition, a dynamic sampling method is designed to improve the efficiency of reasoning.

### Strengths
1. By explicitly separating the planning and execution steps, DREAM is expected to further improve sampling efficiency with the help of the PRM.

2. The proposed adaptive sampling budget allocation strategy is reasonable.

### Weaknesses
1. The proposed method has limited application scenarios: on one hand, it requires multiple specialized reward models to provide rewards for plan and execution respectively. This adds extra burden to both training (especially considering the need to construct specific datasets) and inference deployment, making practical application difficult. On the other hand, it can only be applied to tasks that can be formatted into plan and execution stages.

2. The proposed idea is not very innovative — essentially, it just breaks down the original steps into finer ones. More simply, even without formatting outputs into plan and execution, one could instruct the model to generate smaller reasoning steps to achieve more controllable, stage-guided generation.

### Questions
What I’m more concerned about is: where exactly is the boundary for step decomposition? Or can the steps be continuously subdivided into even smaller units? What additional problems would finer decomposition bring? And how should we account for the computational cost of the reward model?

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
4

### Summary
The paper introduces **DREAM**, a framework that explicitly decomposes reasoning into two phases—planning and execution. Separate reward models are trained for planning and for execution to conduct independent tree-structured search and pruning. The authors further propose a dual-threshold adaptive compute budget: terminate early when any candidate surpasses an upper threshold to save computation, and trigger additional sampling when all candidates fall below a lower threshold, thereby reallocating compute from easy steps to harder ones and improving the accuracy–efficiency trade-off. On mathematical reasoning and code generation tasks, DREAM demonstrates strong cross-model and cross-dataset generalization.

### Strengths
The paper clearly delineates two major shortcomings of current tree-search-based test-time scaling (TTS): (i) planning and execution are evaluated jointly, and (ii) the step budget is fixed. It proposes DREAM, which decouples reasoning into planning and execution phases, equips each phase with a dedicated reward model, and applies dynamic step-level budgeting, thereby reducing ineffective exploration and focusing computation on critical steps.

### Weaknesses
1. Lacks theoretical and empirical analysis of the thresholds (e.g., $\tau_{p1}$ and $\tau_{p2}$), which are likely to strongly affect resource allocation and accuracy. Paper does not analyze their impact.

2. Experimental details are insufficient. The settings for $\tau_{p1}$, $\tau_{p2}$, $m_1$, and $m_2$ are missing, making the experiments difficult to reproduce.

3. Missing evaluations on more challenging mathematical reasoning datasets and stronger reasoning models, which makes it hard to assess the method’s generalization.

4. Typo: in Figure 2, the “Plan–Execution Format” contains two instances of “Q2.”

### Questions
1. In the main results, REBASE underperforms Beam Search on mathematical datasets (especially MATH), despite achieving better compute allocation. Please explain the reasons for this discrepancy.

2. DREAM verifies each step in two phases. Does this increase test-time compute? Please provide a detailed test-time compute report, including FLOPs for the policy model and the PRM, as well as end-to-end inference latency.

3. Is the setting $\tau_{p1}=\tau_{e1}$ reasonable? The planning and execution phases differ substantially in sampling distribution and inter-sample dependency, raising concern that DREAM may simply split a step and apply finer-grained rewards without truly distinguishing between planning and execution.

### Soundness
2

### Presentation
2

### Contribution
2
