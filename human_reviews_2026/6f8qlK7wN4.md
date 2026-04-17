# RedAHD: Toward End-to-End LLM-Based Automatic Heuristic Design using Reductions

- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Solving NP-hard combinatorial optimization problems (COPs) (e.g., traveling salesman problems (TSPs) and capacitated vehicle routing problems (CVRPs)) in practice traditionally involves handcrafting heuristics or specifying a search space for finding effective heuristics. The main challenges from these approaches, however, are the sheer amount of domain knowledge and implementation efforts required from human experts. Recently, significant progress has been made to address these challenges, particularly by using large language models (LLMs) to design heuristics within some predetermined generalized algorithmic framework (GAF, e.g., ant colony optimization and guided local search) for building key functions/components (e.g., a priori information on how promising it is to include each edge in a solution for TSP and CVRP). Although existing methods leveraging this idea have shown to yield impressive optimization performance, they are far from being end-to-end and still require considerable manual interventions. In this paper, we propose a novel framework, named RedAHD, that enables these LLM-based heuristic design methods to operate without the need of GAFs. More specifically, RedAHD employs LLMs to automate the process of reduction, i.e., transforming the COP at hand into similar COPs that are better-understood, from which LLM-based heuristic design methods can design effective heuristics for directly solving the transformed COPs and, in turn, indirectly solving the original COP. Our experimental results, evaluated on six COPs, show that RedAHD is capable of designing heuristics with competitive or improved results over the state-of-the-art methods with minimal human involvement.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces RedAHD, a novel framework for automatic heuristic design (AHD) that aims to make the process more end-to-end by leveraging Large Language Models (LLMs). The central idea is to address a key limitation of prior LLM-based evolutionary program search (LLM-EPS) methods, which rely on manually specified generalized algorithmic frameworks (GAFs) like Ant Colony Optimization or Guided Local Search. RedAHD automates this by using an LLM to perform reductions: transforming the combinatorial optimization problem (COP) at hand into a similar, better-understood problem. This allows an underlying LLM-EPS method to design heuristics for the transformed problem directly, thereby indirectly solving the original one. The framework includes a "multi-problem" evolutionary search, where ideas can be exchanged between heuristics for different reductions, and a "reduction refinement" mechanism to improve reductions when the search stagnates. The authors demonstrate through extensive experiments on six COPs that RedAHD, without needing a GAF, can design heuristics that achieve competitive or state-of-the-art performance compared to methods that do.

### Strengths
1. The core idea of using an LLM to learn problem reductions is a paradigm shift for LLM-based AHD, moving beyond heuristic generation to problem transformation.
2. The work directly and effectively tackles the reliance on manually-designed GAFs, a major bottleneck in previous state-of-the-art LLM-EPS methods.
3. The framework is rigorously tested on six different COPs, consistently demonstrating competitive or superior performance against strong baselines on both synthetic and real-world (TSPLib) benchmarks.
4. The paper is well-written and well-organized, making complex ideas accessible. The figures, tables, and extensive appendices contribute to a high-quality and reproducible research artifact.

### Weaknesses
1. While RedAHD removes the need for a GAF, it introduces its own set of hyperparameters (e.g., $M, M_{init}, l, T$). The paper lacks a sensitivity analysis for these parameters, making it unclear how crucial their specific tuning is.
2. Limited interpretability of generated reductions—no systematic analysis of what reductions the LLM tends to produce.
3. Some experimental setups and the choice of hyperparameters need to be further discussed and explained. For instance, the decision to remove three of the five original variation operators from EoH is justified empirically.
4. The success of the reduction phase hinges on the LLM having relevant knowledge about related COPs. For truly novel or niche problems, the LLM may fail to generate meaningful reductions. This is a critical failure mode, but only briefly touched upon in the limitations.

### Questions
1. What was the rationale for choosing $M_{init}=10$ and $M=3$? The article would benefit from an explanation of the hyperparameter choices.
2. The "reduction refinement" step is vital. It is recommended to provide statistics from your experiments on its activation frequency. For instance, in a typical 20-generation run for TSP, how many times was a reduction refined on average?
3. What kinds of changes does the reduction refinement step typically produce? Are they minor tweaks to the implementation of $(f, g)$, or do they represent fundamentally different reduction strategies? An example would be very helpful.
4. The author removed the E1 operator (generate a completely new heuristic) from EoH. This seems counterintuitive, as RedAHD is built on exploration. It is recommended to provide a clearer explanation to this question.
5. For the CVRP results in Table 7, the performance jump when using 03-mini is dramatic. 
It is suggested to provide a qualitative comparison of the reductions and/or heuristics generated by GPT-40-mini versus 03-mini for this problem. What specifically did the more capable model do better?
6. For the highly constrained VRPTW, noted that a high percentage of invalid solutions. Is this because the LLM struggles to generate valid reduction functions $(f, g)$, or because the subsequent LLM-EPS search fails to produce heuristics that respect the constraints of the reduced problem B?

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
4

### Summary
This paper identifies a key limitation in current LLM-based Automatic Heuristic Design (AHD) methods: they are not truly end-to-end and rely on human experts to specify a Generalized Algorithmic Framework (GAF), such as ACO or IC, which reintroduces significant domain knowledge and implementation effort. To address this, the authors propose RedAHD, a novel framework that enables LLM-AHD methods to operate without a predefined GAF. The core idea is to use an LLM to automate problem reduction: the LLM is prompted to transform the target CO problem A into a similar CO problem B. The LLM generates the functions to map instances from A to B and solutions from B back to A. Existing LLM-EPS methods are then used to design heuristics for directly solving problem B, thereby indirectly solving problem A. The RedAHD framework consists of three main components: 1) Reduction Initialization: An LLM generates a set of candidate Language Reductions (LRs) and a set of heuristics; 2) Multi-Problem LLM-EPS: A novel evolutionary search where heuristics from different LRs (i.e., for different "Problem B's") can be used as references to create new heuristics for other LRs, facilitating the discovery of novel algorithmic ideas; 3) Reduction Refinement: An LLM refines the reduction functions if the search for an LR stagnates, helping to avoid local optima. The authors evaluate RedAHD on six COPs (TSP, CVRP, KP, MKP, OBPP, BPP) and show that it achieves competitive or improved results compared to SOTA LLM-AHD methods that rely on manually specified GAFs.

### Strengths
1. This paper's core premise is sound. Instead of automating heuristic design within a fixed framework, this work attempts to automate the framework itself by re-framing it as a problem-reduction task.

2. The paper is well-written and clearly motivated. The problem statement is easy to understand, and the high-level schematic in Figure 2 effectively communicates the method's workflow.

3. The proposed RedAHD framework is well-structured and thoughtfully designed with its three components (initialization, multi-problem search, refinement).

### Weaknesses
1. It appears the framework has simply shifted the manual-effort burden. Experts must still manually design detailed, COP-specific prompt components (Tables S10, S11) and, most critically, manual solution checkers. The paper's own experiment on VRPTW demonstrates this new burden is a critical point of failure. The authors state that the designed heuristics "are not consistently valid" and violate constraints in over 40% of test instances.

2. The experiments set $M_{init}=10$ and $M=3$, but provide no analysis on the quality of this initial pool. How sensitive is the final performance to this initialization step? This critical component seems under-analyzed.

3. The ablation in Table S12 shows that the multi-problem component ($M=3$) is significantly better than the single-problem component ($M=1$). This suggests much of the performance gain might come from the multi-problem search. However, the current baselines (e.g., EoH on ACO) are single-problem (or single-GAF).

### Questions
1. The VRPTW experiment highlights that designing the prompts and solution checks is a new, expert-level burden and a critical failure point. How do you quantify this new manual effort against the effort of implementing a GAF? Given that the method can produce invalid solutions for complex COPs, how can the claim of "enhanced automation" be justified?

2. How sensitive is RedAHD to the quality of the initial $M_{init}=10$ LRs? In your experiments, what percentage of these initial LRs were trivial or invalid? What happens if the LLM fails to generate any non-trivial strategies for a new problem?

3. How much of the performance gain over GAF-based baselines is attributable only to the GAF-free reduction aspect, versus the multi-problem search strategy? 

4. The problems solved are limited to the vehicle routing and packing problems. This problem seems relatively easy to reduce. Can the proposed method be extended to handle more complex problems such as the flow shop scheduling problem (which is solved in EoH)?

### Soundness
3

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
This paper proposes an augmentation method called reduction for building LLM-based AHD methods. Generally, RedAHD can achieve impressive results on TSP. I served as the reviewer for this paper at previous conferences. After comparing the changes, I tend to barely retain my previous review

### Strengths
1. The introduction part is well-written. With clear evidence and good logic, clear improvements compared to the previous manuscripts.

2. The proposed RedAHD shows impressive results on TSP and some other COPs.

### Weaknesses
See Questions

### Questions
1. RedAHD seems to have significant differences in performance on different issues. According to Figure 3, RedAHD is able to design a 2-opt operator in the TSP problem, which seems to be something that the IC framework cannot achieve. Is this the only reason why RedAHD performs well on TSP? Can performing 2-opt post-processing on the solution of EoH artificially achieve similar performances to RedAHD?

2. Can RedAHD show potential on some problems highly requiring an end-to-end heuristic (e.g., maybe TSPTW, which has no feasibility guarantee for the IC framework, and there are no current implementations with ACO)?

### Soundness
2

### Presentation
2

### Contribution
2
