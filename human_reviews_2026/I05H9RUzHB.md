# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with prompts that declare their functionality, along with the topologies that orchestrate interactions across agents. Designing prompts and topologies for multi-agent systems (MAS) is inherently complex. To automate the entire design process, we first conduct an in-depth analysis of the design space aiming to understand the factors behind building effective MAS. We reveal that prompts together with topologies play critical roles in enabling more effective MAS design. Based on the insights, we propose Multi-Agent System Search (MASS), a MAS optimization framework that efficiently exploits the complex MAS design space by interleaving its optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (local) prompt optimization; 2) workflow topology optimization; 3) workflow-level (global) prompt optimization, where each stage is conditioned on the iteratively optimized prompts/topologies from former stages. We show that MASS-optimized multi-agent systems outperform a spectrum of existing alternatives by a substantial margin. Based on the MASS-found systems, we finally propose design principles behind building effective multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Multi-Agent System Search (MASS), a novel three-stage framework to automate the complex design of multi-agent systems by optimizing both prompts and interaction topologies. The contribution is an interleaved optimization process that first locally optimizes individual agent prompts, then uses those insights to search a pruned topology space, and finally performs a global prompt optimization on the best-found workflow. The experiments are extensive and show that this method substantially outperforms current SOTA baselines.

### Strengths
1.	Novel and Effective Framework: The proposed three-stage MASS framework is a novel and intuitive contribution. It effectively decomposes the highly complex joint-optimization problem of prompts and topologies into three manageable stages.

2.	Strong Empirical Performance: The paper demonstrates substantial and consistent performance gains across a diverse set of 8 benchmarks, significantly outperforming existing manual MAS designs and state-of-the-art automated frameworks.

3.	Good Presentation: The paper is well-written, clearly structured, and easy to follow, which effectively communicates the authors' core contributions and experimental results.

### Weaknesses
1.	The core three-phase framework of this paper is inspired by two preliminary analyses, which seem to have methodological flaws. The "prompt-first" conclusion shown in Figure 2 has not been strictly proven, as it lacks a key benchmark to compare it with the "topology-first" method. Moreover, it is unclear whether the results shown in Figure 4 are based on the isolated evaluation of components or the ablation of completed topology. If it is the former, it ignores the synergistic nature of the MAS topology, where components only show value when combined.

2.	The proposed framework is presented as a heuristic and lacks theoretical grounding. The three-stage decomposition (local prompt, topology search, global prompt) is justified by empirical observation rather than a formal analysis of the optimization space. 

3.	The paper's presentation suffers from minor issues that harm readability. There are spelling errors ("Monte-Carto" in the appendix) and the "Related Work" section is placed between the Method and Experiment sections, which disrupts the logical flow of the paper.

### Questions
1.	The paper convincingly shows that “Prompting to SC” is effective, but it does not include the critical baseline of “SC to Prompting”. How to determine the optimal order of optimization directives before topology optimization, rather than the opposite?

2.	Could the authors provide a more detailed analysis of the framework's scalability? Specifically, how does the optimization cost of Stage 1 (1PO) scale with the number of available topology building blocks, and how does the cost of Stage 3 (3PO) scale with the complexity (e.g., number of nodes) of the final discovered topology?

### Soundness
2

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
4

### Summary
This paper proposes Multi-Agent System Search (MASS), an optimization framework that seeks the optimal multi-agent system. It models a workflow as a collection of building blocks and their logical relationships, and constructs a better MAS through a three-stage process: first optimizing prompts at the building block level, then optimizing the workflow topology, and finally refining the global prompt.

### Strengths
- This paper jointly considers prompt and topology optimization, and alternately updates both components, offering a novel optimization strategy for MAS.
- The building block design decouples prompt optimization from topology optimization, effectively reducing the cost of searching for an effective MAS structure by pruning the search space.
- Experimental results demonstrate performance improvements compared to baselines.

### Weaknesses
- Inconsistent notations. In Equation (1), the symbol $a$ is defined as a configuration. However, later in the text, $a$ is ambiguously reused to refer to individual agents and topologies.
- Figure 4 evaluates individual topologies in isolation on specific tasks. However, the final MAS in this work is a composition of multiple topologies. It is unclear whether these isolated results sufficiently support the claim that ``not all topologies have a positive influence on MAS design.'' Similarly, during search space pruning, the selection criterion is based on the performance of each building block relative to an initial agent, rather than the performance of the full composite structure. This raises a concern: Could this approach inadvertently prune away building blocks that are suboptimal in isolation but essential for achieving global optimality in combination?
- The proposed pipeline first performs prompt optimization on all building blocks before search space pruning. Given that some of these blocks may ultimately be discarded during pruning, this may incur potentially unnecessary computational overhead.

### Questions
- How does the evaluator assign scores? How is the validation set determined, and how is the validation performance computed?
- The description in lines 260–269 is confusing, largely due to the ambiguous reuse of the symbol $a$. Could you provide a clear and precise explanation of this section?

### Soundness
3

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
3

### Summary
This paper formalizes the Multi-Agent-System into a complex problem of joint optimization prompts and topology, and proposes the MASS framework to efficiently search in the MAS design space. The framework is divided into three parts, block-level prompt optimization, workflow topology optimization, and workflow-level prompt optimization, and the author claims that this method outperforms existing MAS in multiple tasks.

### Strengths
The paper presents a clear and logical narrative process. It explicitly formalizes the design of MAS as a joint optimization problem of prompts and topology, introduces an incremental impact metric to prune the huge space, and demonstrates the significance of this consideration through detailed experiments.
The experiments show the optimization trajectory while taking into account cost-effectiveness. MASS is a plug-and-play module that provides a platform for future expansion and promotes MAS from manual trial and error to automated search.

### Weaknesses
The MASS framework mentions that the topology search in the second stage is guided by the "incremental influence" indicator, which evaluates each module independently and cannot find synergies between dependent modules.
This "exhaustive task" can be performed by some more advanced algorithms, such as evolutionary algorithms, and the operation of this random search can be more fully demonstrated according to the change of performance scores, and the potential optimization direction in the future can also be more complete.
This framework is theoretically highly sensitive to temperature parameters, and it is helpful to analyze on temperature settings.

### Questions
How does the MASS consider the combination order of topology modules, and is the current MASS framework effective for the combination order of different topology modules? Since the authors already want to use the random search strategy to select the number of modules, can we put the combination order of modules into the search space to realize the automatic combination of different tasks?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MASS (Multi-Agent System Search), a framework for automatically optimizing multi-agent systems (MAS) by jointly optimizing both prompts and topologies. The authors first conduct an analysis showing that prompt optimization and topology design are critical factors for MAS performance. Based on these insights, MASS employs a three-stage optimization approach: (1) block-level prompt optimization for individual agent types, (2) workflow topology optimization in a pruned search space, and (3) workflow-level prompt optimization for the complete system.

### Strengths
1, Comprehensive Analysis: The paper provides valuable insights into what makes effective MAS through systematic ablations (Sec 2.1-2.2), showing that prompts matter more than previously recognized and that only a small fraction of topologies are beneficial.

2, Well-Motivated Design: The three-stage optimization approach (local→global, prompts→topologies) is intuitive and addresses the complexity of joint optimization effectively. The influence-weighted search space pruning (Eq. at line 269) is a practical contribution.

3, Strong Empirical Results: MASS achieves substantial improvements across diverse tasks (78.79% avg on Gemini Pro vs 70.26% for best baseline). The results are consistent across multiple LLM backbones (Gemini, Claude, Mistral).

### Weaknesses
1, Limited Novelty: The core contribution is primarily engineering—combining existing prompt optimization (MIPRO) with topology search. The individual components (APO, topology search) are not novel. The main novelty lies in the specific combination and the three-stage approach.

2, Search Space Design Limitations:
- The topology space is limited to 5 predefined building blocks. More complex or novel topologies are not explored.
- The "rule-based" construction order (line 275-276) seems arbitrary and is not well justified. Why is [summarize, reflect, debate, aggregate] the right order?
- The search space is task-specific (Table 2), limiting generalizability.

3,Methodological Concerns:
- AFlow comparison: The authors acknowledge AFlow uses Claude 3.5 as optimizer while Gemini as executor, making comparison unfair (marked with *). This is a significant limitation.
- Statistical significance: While error bars are shown, no formal significance tests are provided.
- Validation/test split: Using only 50-100 examples for validation seems small and may lead to overfitting to the validation set.

4, Incomplete Analysis:
- Why does prompt optimization help so much? The paper shows that it does (Fig 2) but provides limited insight into what makes optimized prompts better.
- Topology analysis: Fig 4 shows different topologies work for different tasks, but there's limited discussion of why. What task characteristics favor which topologies?
- Workflow-level optimization gains: Stage 3 provides only ~2% improvement (line 342). Is this consistent? When does it help?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
