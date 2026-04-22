# EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions

- Avg Score: 4.50
- Decision: Reject
- Scores: 0, 6, 6, 6

## Abstract
Engineering problem solving is central to real-world decision-making, requiring mathematical formulations that not only represent complex problems but also produce feasible solutions under data and physical constraints. Unlike mathematical problem solving, which operates on predefined formulations, engineering tasks demand open-ended analysis, feasibility-driven modeling, and iterative refinement. Although large language models (LLMs) have shown strong capabilities in reasoning and code generation, they often fail to ensure feasibility, which limits their applicability to engineering problem solving. To address this challenge, we propose EngiAgent, a multi-agent system with a fully connected coordinator that simulates expert workflows through specialized agents for problem analysis, modeling, verification, solving, and solution evaluation. The fully connected coordinator enables flexible feedback routing, overcoming the rigidity of prior pipeline-based reflection methods and ensuring feasibility at every stage of the process. This design not only improves robustness to diverse failure cases such as data extraction errors, constraint inconsistencies, and solver failures, but also enhances the overall quality of problem solving. Empirical results across four representative domains demonstrate that EngiAgent achieves substantial improvements in feasibility compared to prior approaches, establishing a new paradigm for feasibility-oriented engineering problem solving with LLMs. Our source code and data are available at https://anonymous.4open.science/r/EngiAgent-1C8A.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper identifies that LLMs fail to produce feasible solutions for open-ended engineering problems. The authors propose EngiAgent, a multi-agent system with five specialized agents (Analyzer, Modeler, Verifier, Solver, Evaluator) that work together. The system is fully connected, which the authors claim allows coordination to be dynamically routed between agents.

### Strengths
**1. Important Problem:** The focus on feasibility over correctness alone addresses an important and under-explored challenge in applying LLMs to real-world engineering.

**2. Strong Empirical Results:** The system shows significant improvement (7x on average) over existing methods. The ablation study demonstrates the value of the fully connected coordinator.

**3. Well-Designed System:** The agent roles are well-chosen and mirror a realistic engineering workflow. The prompts in the appendix are detailed and show careful engineering.

### Weaknesses
**Lack of Formal Foundations:** The system relies on heuristic routing without formal guarantees for state consistency, rollback, or compensation. The architecture is robust but theoretically ungrounded, caught between workflow systems and multi-agent systems.
Missing Key Related Work: Classical workflow and transaction process literature offers abundant references, such as the Saga work from 1987 and multi-agent system papers from the distributed systems community in the early 2000s. Compared to those works, this paper presents a heuristic-based prototype.

**Scalability Concerns:** While the fully connected model works well for five agents, it may not scale efficiently to larger agent teams due to potential combinatorial explosion of interaction paths.

**Narrow Benchmarking:** The limited scale and rewritten nature of the problems mean the claims should be treated as a strong initial demonstration rather than definitive proof of general capability. The benchmark is sufficient for conference publication but highlights clear paths for future work: expanding the benchmark, testing on raw problem descriptions, and comparing against human experts.

**Limited Insights:** While EngiAgent outperforms selected methods, the paper does not provide insights into the root problems in native LLMs that cause their failures.

### Questions
The "Fully Connected Coordinator" introduces flexible control flow, but it lacks the formal guarantees of a transactional system. My questions focus on how the system handles state consistency and recovery, which are critical for robust and reliable problem-solving.

**Atomicity & Consistency:** Your system allows any agent to be rolled back and re-invoked based on downstream failures. How do you guarantee atomicity for a single agent's contribution? For example, if the Modeler generates a new model after a Verifier failure, is the previous model's entire output completely discarded and replaced, or can partial state persist and cause consistency violations with the Analyzer's original analysis?

**Isolation:** Your architecture allows for flexible feedback loops (e.g., Verifier -> Analyzer). How do you manage isolation when one agent's output is being revised? If the Analyzer is re-triggered, are all subsequent agents (Modeler, Verifier, etc.) automatically invalidated, or can the system's memory contain a mixed state from different, non-isolated iterations of the workflow?

**Durability & State Management:** The "Memory" component stores the problem-solving state. What is the durability and recovery model for this state? If a long-running problem-solving session fails mid-process, can it be resumed from a last known consistent state, or must it restart from scratch? Does your coordinator maintain a formal state machine to manage this, or is it managed heuristically?

These are all the "major league" problems that must be addreddes in real-world long-horizon planning problems.

### Soundness
2

### Presentation
3

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
This paper introduces a multi-agent framework designed to tackle open-ended engineering problems with an explicit focus on feasibility rather than mere textual correctness or polished formulations. The system organizes the workflow into five specialized agents, Analyzer, Modeler, Verifier, Solver, and Evaluator, and integrates a fully connected coordinator and shared memory that dynamically routes feedback among agents to address specific failure modes such as data extraction errors, constraint inconsistencies, and solver failures, instead of adhering to a rigid, one-way pipeline. The authors construct a benchmark covering multiple engineering domains, including market and multi-agent decision-making, scheduling and resource allocation, planning and design, and control and autonomy. They define feasibility as a binary criterion indicating whether a numerical solution adheres to the given data and physical or operational constraints. Empirical evaluation across several large language models demonstrates that EngiAgent with coordination significantly improves the feasibility of generated solutions compared to adapted multi-agent baselines and a fixed-pipeline ablation, while maintaining reasonable computational cost.

### Strengths
1. The paper articulates four concrete failure modes common in LLM-for-engineering, fancy-but-vague modeling, altering key data, violating physical laws, and over-constraining models, grounded in a didactic energy-storage example. This sharpens why open-ended engineering differs from math/code tasks and motivates the method well.

2. The design lets the coordinator reroute verification feedback to the agent best positioned to fix the error (e.g., send constraint mismatches to the Analyzer, syntax to the Modeler), with loop-prevention and memory of error types/repair history.

3. Table 1 shows large absolute increases in feasibility vs. strong agent baselines; the paper also analyzes the misalignment of text-based LLM-as-a-judge scoring with engineering feasibility, which is timely and well-argued.

### Weaknesses
1. While feasibility is the right target, the paper does not convincingly establish how feasibility labels are produced with high fidelity across all tasks. Section 5 defines feasibility as binary and Appx. B.5 lists principles focusing on fatal violations, but the process (automatic checks vs. solver statuses vs. human adjudication), inter-rater reliability (if any), and safeguards against LLM-judge leakage are unclear. Given Section 6.3’s criticism of text-based judgments, the paper should detail a deterministic checker or a double-blind expert protocol, with agreement statistics and examples of borderline cases; otherwise, claims of 50–75% feasibility could reflect lenient or inconsistent adjudication.

2. The benchmark has 53 tasks. For four broad domains, this is small and may not capture real-world variability (nonlinear physics, hybrid discrete, continuous constraints, stochastic dynamics, safety constraints, multi-period coupling). The paper should provide task complexity stats (variable/constraint counts, solver class, convex vs. nonconvex, MILP/MINLP mix) and hold-out categories to demonstrate cross-domain generalization, or else the headline feasibility rates risk overfitting to this curation.

3. Baselines (ResearchAgent, DS-Agent, MM-Agent) are adapted to engineering, but the paper doesn’t show extensive tuning or task-specific tool integration symmetry. For instance, MM-Agent focuses on mathematical modeling; if EngiAgent benefits from Pyomo-specific templates/validation while baselines don’t receive comparable domain/tool priors, the comparison may conflate tooling advantage with coordination quality.

### Questions
Several baselines are repurposed from non-engineering contexts. The paper does not deeply document how prompts/tools were adapted (e.g., giving MM-Agent or DS-Agent equivalent access to Pyomo/solvers, uniform retry budgets, or domain hints). Given the magnitude of improvements, the community will expect a fairness audit: identical budgets, tool access, early-exit rules, and error-recovery affordances.

### Soundness
3

### Presentation
3

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
The paper introduces EngiAgent, a multi-agent system tailored to feasibility-oriented engineering problem solving. Instead of a rigid stepwise pipeline, the authors add a fully connected coordinator that dynamically routes feedback among five role agents, Analyzer, Modeler (Pyomo code generation), Verifier, Solver (solver selection and execution), and Evaluator, with a shared memory for error diagnosis and restart strategies. A new benchmark of 53 open-ended engineering tasks across four domains is assembled to prioritize feasibility as the primary success criterion. On three LLM backbones, the fully connected variant reportedly achieves markedly higher feasible-solution rates. However, several important concerns remain and should be addressed by the authors.

### Strengths
1. The paper crisply articulates why textual plausibility can diverge from implementable solutions, and formalizes binary as the principal metric, with secondary rubric scores. The taxonomy of common failure modes is practical and well-motivated. 

2. The fully connected coordinator with LLM-aided diagnosis and forced agent switching is a simple but effective abstraction that plausibly explains robustness to heterogeneous errors. The qualitative workflow diagrams make the routing logic and memory use legible. 

3. Across all three models, the gap between numerical outputs and feasible solutions shrinks substantially for EngiAgent; the reported feasible-rate jumps over prior baselines are large and the fixed-pipeline ablation is consistently weaker, attributing gains to coordination rather than mere prompting.

### Weaknesses
1. The feasibility criterion explicitly focuses on avoiding fatal violations, while non-fatal incompletenesses are not penalized. This can over-count borderline solutions as feasible. Moreover, the Evaluator includes LLM-as-a-judge components, which the paper itself argues can inflate text-based scores; even if feasibility is binary, some checks (e.g., physical constraints, conservation laws) appear to be performed via semantic verification rather than executable, unit-tested assertions. The paper should formalize an automated, code-level feasibility oracle (e.g., unit constraints, conservation checks, solver feasibility flags, regression tests on outputs) and clearly separate any LLM-based rubric from feasibility adjudication. 

2. The system includes several pragmatic components (template-driven Pyomo scaffolds, rule-based scheduling, error-type routing). It is difficult to isolate which design choice drives the bulk of gains beyond the headline fully connected routing. The ablation only contrasts fixed vs. coordinator; further component-level ablations (e.g., removing Pyomo-specific validation, or disabling forced agent switching, or swapping Solver heuristics) are necessary to support the causal narrative. 

3. DS-Agent and MM-Agent are adapted from different task regimes; the paper should invest in stronger engineering-aware baselines, e.g., a single-agent Pyomo specialist with iterative verifier feedback; a graph-of-thought/skill-router agent; or a pipeline that adds identical Pyomo validation and solver heuristics for but with fixed routing, to ensure the comparison isolates coordination rather than tooling familiarity.

### Questions
1. The Modeler is Pyomo-centric. It is unclear whether the approach transfers to other modeling stacks (e.g., OR-Tools, JuMP, CasADi) or non-optimization engineering tasks (simulation, PDE-constrained design). A cross-tool study would support claims of general applicability. 

2. The paper mentions expert review in dataset construction, but the feasibility labeling pipeline (human vs. automated), inter-rater agreement, and adjudication for borderline cases are under-specified. Given the centrality of feasibility, the community needs labeling guidelines, example adjudications, and an error analysis of false-feasible/false-infeasible calls.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EngiAgent, a multi-agent framework designed to solve open-ended engineering problems by emphasizing feasibility. EngiAgent organizes five specialized agents (Analyzer, Modeler, Verifier, Solver, and Evaluator) under a fully connected coordinator, which enables dynamic feedback routing and flexible error correction. The study builds a benchmark dataset including 53 tasks across four engineering domains. It demonstrates substantial improvements in feasibility (up to 75.47%) compared with existing multi-agent and LLM-based baselines. Moreover, it shows that EngiAgent achieves strong cost-efficiency while surpassing previous frameworks in generating valid engineering solutions.

### Strengths
1. The paper introduces feasibility as the primary success criterion in engineering problem solving, which is a key but underexplored dimension in LLM-based automation. 
2. The architecture is well-structured and systematically explained, with clear functional decomposition and coordination logic. Experiments are comprehensive, including ablations, cost analysis, and qualitative examples. 
3. The definitions of error types (Figure 2) and evaluation dimensions are intuitive and informative. 
4. The appendices provide detailed prompt designs and dataset descriptions, enhancing reproducibility. 
5. The paper addresses a crucial gap between theoretical reasoning and real-world applicability of LLMs in engineering contexts.

### Weaknesses
1. While EngiAgent’s performance improvements are substantial, it is unclear how much of the gain stems from the coordinator architecture versus more extensive prompt engineering. This also raises questions about generalization to new tasks or LLMs without re-tuning.
2. The dataset size (53 problems) is relatively small for benchmarking general-purpose frameworks. 
3. The definition of feasibility (binary feasible/infeasible) might oversimplify nuanced engineering contexts where partial or probabilistic feasibility (e.g., constraint violation tolerance, stochastic feasibility) is relevant.
4. Engineering workflows typically include iterative expert verification. EngiAgent fully automates this, but there’s no mechanism for human-guided correction or inspection, which may limit real-world deployment.
5. The system relies mainly on Pyomo-based modeling and standard solvers (GLPK, CBC, Gurobi, CPLEX). Many engineering problems involve differential equations, stochastic simulation, or finite element analysis.

### Questions
1. It would be valuable to report the performance of the architecture alone, without the additional prompt engineering, to better isolate its contribution. How robust is EngiAgent to different prompt wordings, model versions, or unseen problem types?
2. In many real-world engineering problems, strict feasibility may not be attainable without modifying constraints or relaxing requirements, which warrants further clarification. Can the authors give more formal definition of Feasibility? 
3. How do the authors envision extending the 53-task dataset to a larger or more diverse benchmark (e.g., EngiBench)? Is there any plan to include continuous real-world or industrial data? 
4. Is there any mechanism for human-in-the-loop verification or correction in EngiAgent’s workflow? If not, how do the authors prevent the system from propagating small but critical reasoning errors?
5. Does the coordinator produce interpretable reasoning traces for debugging, or is its routing behavior opaque to users?

### Soundness
3

### Presentation
3

### Contribution
3
