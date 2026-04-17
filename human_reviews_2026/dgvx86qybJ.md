# Hierarchical Representations for Cross-task Automated Heuristic Design using LLMs

- Decision: Reject
- Scores: 4, 6, 4, 6, 2

## Abstract
Designing heuristic algorithms for complex optimization problems is a time-consuming and expert-driven process. Recently, Automated Heuristic Design (AHD) using Large Language Models (LLMs) has shown significant promise for automating algorithm development. However, existing works mainly rely on programs to represent heuristics, which are inherently task-specific and fail to generalize as effectively as established metaheuristics like tabu search or guided local search. To bridge this gap, we introduce Multi-Task Hierarchical Search (MTHS), an LLM-guided evolutionary method that co-designs general-purpose metaheuristics and task-specific programs. MTHS employs a hierarchical representation and adopts a two-level evolution framework to evolve task-agnostic metaheuristics and task-specific program implementations simultaneously across multiple heuristic design tasks. During this evolution, a knowledge transfer mechanism allows learning from elite programs designed for other tasks. We evaluated MTHS on distinct combinatorial optimization problems, where it outperforms both commonly-used heuristics and existing LLM-driven AHD approaches. Our results demonstrate that the hierarchical representations facilitate effective multi-task AHD, and the evolved metaheuristics exhibit strong generalization to related tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limited cross-task generalization of current Large Language Model (LLM)-based Automated Heuristic Design (AHD) systems, which typically produce task-specific heuristics that cannot transfer across problem domains. The authors propose Multi-Task Hierarchical Search (MTHS), an LLM-guided hierarchical evolutionary framework that co-designs general-purpose metaheuristics and their task-specific program instantiations.

### Strengths
1. Research an important problem.
2. Appendices B–C provide explicit prompts, templates, and problem settings, enabling implementation replication.
3. Strong generalization across tasks (Sec. 3.5; Fig. 3), showing the same metaheuristic helps different LLMs generate high-quality solvers—empirical evidence of cross-task knowledge transfer.

### Weaknesses
1. This paper investigates a critical issue. While existing LLM-based automated heuristic designs are highly effective, most methods design a single heuristic tailored to specific problem instances, often resulting in poor generalization across different distributions or settings. It should be clarified that this paper is not the first to propose a solution to this problem; the recent work EoH-S [1] should also be referenced for discussion and used as a baseline for comparison.
2. The proposed method still requires re-search when encountering new problem instances, albeit incorporating MH to enhance search efficiency. It does not fundamentally resolve the aforementioned issue.
3. The paper gives no formal analysis of why hierarchical separation or Pareto-based selection guarantees better generalization.
4. The intuition (“mirrors expert practice”) is plausible but remains qualitative.

[1] Liu F, Liu Y, Zhang Q, et al. Eoh-s: Evolution of heuristic set using llms for automated heuristic design[J]. arXiv preprint arXiv:2508.03082, 2025.

### Questions
Reference Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript proposes MTHS, an LLM-guided evolutionary method that co-designs general-purpose metaheuristics and task-specific programs.

### Strengths
The experiments demonstrate the effectiveness of the proposed MTHS.

### Weaknesses
**W1:**  This manuscript should evaluate MTHS using additional LLMs.

**W2:**  The advantages of MTHS would be clearer if the authors included an experimental comparison with the most related study [1].

**W3:**  This manuscript should conduct comprehensive ablation studies to validate the effectiveness of all proposed components.

[1] Generalizable heuristic generation through large language models with meta-optimization, arXiv:2505.20881.

### Questions
See Weaknesses.

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
4

### Summary
This paper tackles the limited generalization of existing Large Language Model (LLM)-driven Automated Heuristic Design (AHD) methods, which typically yield task-specific heuristics. The authors propose **Multi-Task Hierarchical Search (MTHS)**, a hierarchical evolutionary framework that separates *task-agnostic metaheuristics* from *task-specific programs*. Guided by LLMs, MTHS performs two-level evolution — evolving general metaheuristics at the high level and optimizing task-specific implementations at the low level — while transferring knowledge across tasks. Experiments on four combinatorial optimization problems (TSP, CVRP, FSSP, and BPP) demonstrate that MTHS outperforms both traditional heuristics and existing LLM-based AHD approaches (e.g., EoH, ReEvo, MCTS-AHD), with improved cross-task generalization.

### Strengths
1. The paper introduces a hierarchical representation that mirrors human expert reasoning — decoupling general algorithmic logic from task-specific components. The proposed method is well-motivated.
2. The experiments are generally comprehensive and with strong baselines.

### Weaknesses
1. No code.
2. The distinction drawn between “program-level” and “thought-level” AHD approaches seems somewhat artificial. The proposed method still relies on multi-task prompts and LLM-generated programs, similar in spirit to the supposedly “high-level” methods (e.g., EoH, ReEvo).
3. It is unclear whether all compared LLM-based methods use the same base LLM (the paper mentions GPT-5-mini for MTHS, but others might differ). Furthermore, MTHS requires extra steps (multi-task inputs, hierarchical evolution), which may inflate computational cost relative to single-task methods. This raises fairness concerns in direct performance comparisons. And no cost comparison for other AHD methods.
4. Although the paper claims the total cost of multi-task evolution is lower than multiple independent runs, the framework’s scalability beyond a few tasks (e.g., >10) is questionable. The high proportion of LLM calls at the low-level stage (≈52%) could make large-scale extensions impractical.
5. The knowledge transfer component is only qualitatively discussed. The paper lacks quantitative analysis on when and how transfer helps (e.g., correlation between task similarity and performance gain). 
6. The appendix suggests MTHS takes longer per run than baseline AHD methods. Since heuristic performance often depends on runtime, this omission weakens claims of superiority.

### Questions
1. How sensitive is the framework to weaker LLMs (e.g., GPT-3.5). 
2. Consider adding experiments on tasks beyond discrete combinatorial optimization to validate cross-domain generalization claims.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge of automating heuristic design for combinatorial optimization (CO) problems. It focuses on improving how LLM-driven Automated Heuristic Design (AHD) methods generalize across different tasks. The authors introduce Multi-Task Hierarchical Search (MTHS), an evolutionary framework with a novel hierarchical structure. It combines a task-agnostic metaheuristic at the high level with task-specific program instantiations at the low level, both guided by LLMs. The framework uses multi-task knowledge transfer and a two-level evolution strategy. Experiments across four classical CO problems—TSP, CVRP, FSSP, and BPP—show that MTHS outperforms both conventional heuristics and recent LLM-based AHD baselines. It also produces metaheuristics that transfer effectively to new tasks and LLM backbones.

### Strengths
- The paper proposes a clear hierarchical framework that separates general metaheuristic logic from task-specific implementations. This approach mirrors expert practice, addresses the core limitation of task specificity in earlier LLM-based AHD works, and enables explicit knowledge transfer.
- The method is evaluated on established CO benchmarks against traditional heuristics, metaheuristics, and LLM-driven baselines. Results show state-of-the-art or competitive performance across all tasks.
- The framework allows learned metaheuristics to transfer to new, unseen tasks and guide multiple LLMs, demonstrating improved robustness and real-world applicability.

### Weaknesses
- Stronger ablation/benchmarking on multiple foundation models (the main results), or an in-depth discussion of LLM choice impact, is missing.
- The paper notes the dominance of LLM API usage in total cost, but lacks a direct, quantitative comparison of token and time usage with baseline LLM-driven AHD methods.
- There is limited exploration of how the selection or diversity of tasks used during multi-task evolution influences the generalization ability of the evolved metaheuristics. For instance, it is not clear whether including more (or more diverse) problems further enhances transfer, or if performance saturates after a certain level of relatedness between tasks.

### Questions
1. How sensitive is the method to the choice and quality of the underlying LLM? For instance, how does performance scale with smaller, less capable, or open-source models?
2. How does MTHS handle structurally dissimilar tasks (e.g., routing, scheduling, bin packing)? At what point does multi-task learning harm generalization through negative transfer?
3. Did the authors assess the diversity of generated metaheuristics? Does the approach produce redundant solutions or consistently find novel strategies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper looks at the issue of cross-task generalization in current LLM-driven automated heuristic design systems. The paper presents Multi-Task Hierarchical Search (MTHS).  This is a hierarchical, two-level evolutionary framework where the high level evolves task-agnostic metaheuristics (general problem-solving strategies) and the low level evolves task-specific program implementations tailored to individual optimization tasks. Another key aspect is that there is a knowledge transfer module allows high-performing components discovered in one task to inform others.  They do experiments involving a number of problems and comparison strategies, and show MHTS performs better.

### Strengths
The hierarchical framework is interesting.  (I don't believe this approach is novel in the large field of metaheuristics, but it is interesting to apply here.)

The appendices provide significant detail, including on the prompting procedures, with the goal of ensuring reproducibility and providing more clear details to the reader as needed.  

The experiments suggest strong practical performance.  

The study comparing different metaheuristic representations.  

There is open-sourced versions of the algorithms, data, etc. for reproducibility.

### Weaknesses
Most of the experiments use very small problem instances and old public datasets (e.g., TSPLib, CVRPLib, Taillard benchmarks). Some of these are decades old and much smaller than the scales modern heuristics are expected to handle. This makes it hard to see if the method would work on real, large, or more recent problems.

I am not up on all the latest in solvers, but it seems to me their comparison points are older, general solution methods.  Perhaps it is reasonable to compare against other general metaheuristic methods, but I do not believe they would be competitive with strong heuristics (even if problem-tailored).  

There’s little explanation or understanding of why the system works or why it would  fails. For example, there’s no study of how the “knowledge transfer” actually helps, or what happens if the tasks not sufficiently closely related.

The experiment suggests this approach is expensive and slow to run. The paper doesn’t show whether this approach could scale to larger or more diverse problems, or whether it’s practical for anyone to use for more "real-life" problems.  

Overall, it's not clear how general this approach would be.

### Questions
I would like to see tests on larger and more modern datasets, and comparisons with current best solvers (even if problem-specific). Is there any type of analysis or insight about how knowledge transfer works here, or other aspects of the system?

### Soundness
2

### Presentation
2

### Contribution
2
