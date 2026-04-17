# KNOWPLAN: Knowledge-Driven AI Agents for Smart Degree Pathway Planning

- Decision: Reject
- Scores: 4, 4, 10, 4, 2

## Abstract
Recent advances in large language models (LLMs) provide powerful capabilities for knowledge-driven course planning. However, building reliable, constraint- aware study planners from publicly available course webpages remains challenging due to heterogeneous data sources, complex multi-logic prerequisites, and multi-requirement constraints. To address these challenges, this paper proposes KNOWPLAN, a proactive, self-evolving multi-agent AI platform that integrates LLM-based extraction, knowledge-graph construction, and constraint-aware reasoning to generate adaptive, personalized study plans. This platform brings together scientific inquiry, technical challenges, and practical utility within a coherent and unified framework. The scientific inquiry focuses on two fundamental problems: the heterogeneity of publicly available university catalog webpages, and the limitations of traditional graph structures in handling prerequisite logic. The technical challenges include extracting structured course information from diverse catalogs, modeling prerequisite structures as hypergraphs, and extracting critical paths under multi-dependency conditions. To tackle these issues, we propose a multi-LLM-driven Agent Forest to handle webpage heterogeneity, introduce the Logic Adjacency Matrix as a novel representation of course prerequisite graphs, and develop the Multi-Dependency Critical Path Extraction algorithm to support effective course planning. These components represent the core technical highlights of this work. On the engineering side, a major contribution of this work is the design of a modular end-to-end pipeline composed of four key components: the Agent Forest, Graph-Construction Agent, Course Planning Agent, and Curriculum Alignment Agent. LLMs are integrated at various stages of this pipeline to support course information extraction, prerequisite cycle resolution, personalized course recommendation, and term-level schedule generation tailored to individual preferences and academic backgrounds. Across multiple universities, KNOWPLAN achieves 99.5\% accuracy on major requirements and 98.7\% on prerequisites. By combining graph-based reasoning with a term-level scheduler, it generates feasible and personalized study plans that respect preferences, workload limits, and policy exceptions, outperforming state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper is organized mostly as a product idea / process pipeline. I don’t think of this as a piece of novel research. However, there are no rigorous benchmarks for this “compound task”. And, that makes it hard to objectively assess the value of the pipeline.


The paper discusses a process to help university students pick coursework to optimize for their interest, account for all the prerequisites automatically & course period conflicts. They do this using multiple LLM calls for different phases of the process. Process outlined below

## Claimed Contributions
1. “Agent Forest” to extract structured data from course catalogues
2. Graph construction agent converts this to KG (DAG)
3. Course Planning Agent generates coarse, multi-constraint learning trajectories, while the Curriculum Alignment Agent refines these trajectories into term-level schedules
4. Account for schedule conflicts, preferences, total time to completion


## Process
1. Use LLM calls on university courses site HTMLs to extract structured details about requirements, class timings, etc
2. Create course prerequisite graph taking care of multi-requirements, & one-of requirements
3. Assign scores to courses (nodes in teh above graph) using heuristics like student’s interest, prerequisites
4. Find potential course-plan by maximizing scores & accounting for perceived “difficulty”
5. LLM converts user preferences to formal rules to help set constraints
    - Use LLMs to “parse preferences/policy text, select the target mode, and generate rationales”
6. Pass through CP-SAT scheduler for class timing overlap avoidance


## Required Readings
- Ng & Fung: https://arxiv.org/abs/2407.11773

### Strengths
1. Addressed the cycle-breaking problem in course picking logic
2. Good details of course score formulation in section 3.3.1
3. Significant details about course-prerequisites graph generation - dealing with ”and” and “one-of” course requirements

### Weaknesses
1. Paper focuses a lot of effort and work into extracting (parsing) systematic data out of webpages. I understand that it is a problem for a product, but it is not a research problem. I’d have focused starting with organized/tabulated clean data & built on top of that.
2. The claim that they “introduce the concept of” Agent Forest is extremely misleading - essentially, they have a custom prompt(s) for all the LLM call(s) (with some extraction examples) for each university’s webpage
3. The sentence: “The pipeline of Agent Forest is demonstrated in Figure 2” is inaccurate. Fig 2 shows the entire process of their pipeline - not just agent forest for course details extraction
4. As for the *results*, they admit that “the ground truth data does not include prerequisites or course relationships” - which is the part that the agent forest was going to solve.
5. The claimed accuracy is on these ground-truth values, & hence not reliable either
6. They don’t attempt to quantify the overall process & quality.
7. Graph building process feels like adding LLM into a use-cases where it’s not needed
    - For “handling the logical expressions of prerequisites with multiple options and addressing circular dependencies” while constructing graphs, they can again use deterministic regex-matching or smaller text annotation models to identify entity relationships (ie which courses are prerequisites for which ones)
---

* IIUC, the only material improvement this process makes on top of an algorithmic system is that:
1. LLM-based course-work details extraction (for which getting json feed from the university, or writing custom parsers are better solutions, & also, the paper admits that the setup didn't really work)
2.  it can account for user’s natural-language preference & encode that into course selection logic.

### Questions
1. They mention “course-to-fine recommender” in several places. Was this intended to be “coarse” instead? - as in coarse granularity vs fine granularity? If this was intended to be word-play (since they’re using it for course selection), they should clarify that upfront. (they do write “coarse” in other places - so I believe its a typo)
2. Why the graph construction agent? If the requirements are clear, creating a DAG is a purely algorithmic process, & introducing an LLM agent only uncertainty in the result
3. In section 3.2, “This article utilizes this table to construct …” - what article? Kindly rephrase, or add relevant references
4. A flowchart of the overall process would be extremely helpful. For example, the course selection “agent” is not exactly a fully agentic system - it relies on well-defined mathematical formulations for the course nodes & then an algorithmic process to pick non-conflicting options. LLM is only used to distil user’s NL preferences into clean logic statements.

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
The paper introduces KNOWPLAN, a multi-agent system that automatically generates accurate, personalized, and conflict-free degree plans using only public university course catalogs. It integrates large language model (LLM)-based extraction, a curricular knowledge graph, retrieval-augmented generation (Graph-RAG), and a constraint-aware scheduler (CP-SAT) to produce term-level plans tailored to students’ goals, preferences, prior credits, and workload. The system achieves 99.5% accuracy on major requirements and 98.7% on prerequisites across multiple universities, and effectively eliminates infeasible plans that arise from coarse-grained approaches.
KNOWPLAN comprises several coordinated components: a university-specific parsing framework that uses a panel of LLMs for data extraction and validation; a graph-construction module that encodes prerequisite and corequisite structures while enforcing acyclicity; and a planning pipeline that progresses from high-level plan skeletons to detailed term-wise schedules via constraint solving. Evaluation on data from over 6,000 institutions, along with ground-truth validation from four universities, demonstrates high extraction accuracy and robust scheduling performance. Compared to commercial tools, KNOWPLAN stands out for its reliance on public data, ability to generalize across institutions, and integration of structured knowledge representations with language-guided reasoning.

### Strengths
The paper presents a well-structured and comprehensive system, with a clear, modular multi-agent architecture that spans from web data ingestion to knowledge graph construction, planning, and CP-SAT-based scheduling. It explicitly models complex curricular structures, including prerequisite logic and cycle detection, and visualizes them effectively. A key strength is its commitment to feasibility: instead of stopping at LLM-generated plans, the system produces executable, conflict-aware term-level schedules under real-world constraints. The coarse-to-fine personalization pipeline supports multiple user goals while maintaining policy compliance. The system is robust to catalog heterogeneity, employing per-university parsing agents and multi-LLM consensus to achieve high extraction accuracy. Finally, the work is well-positioned against existing tools, with a fair comparison to commercial systems that rely on internal student information systems, highlighting KNOWPLAN’s ability to operate entirely on public data.

### Weaknesses
- The paper lacks clarity regarding prerequisite verification. Although the abstract reports 98.7% accuracy, the ground truth data do not include prerequisite relations, and the authors do not explain how these labels were obtained or annotated. This omission raises doubts about the validity of one of the paper’s central claims (Abstract; §4.1).

- The evaluation of the scheduling component is limited. Beyond the reported “82% reduction” in infeasible plans, the paper does not provide key metrics such as the proportion of skeleton plans that schedule successfully, average time to degree under constraints, or robustness across multiple terms. There are no comparisons with existing baseline systems (e.g., e.g., uAchieve Schedule Builder’s auto-combination generation, Series25 optimizer) or user studies.

- The paper lacks ablation studies and error analysis. It would benefit from comparisons between single- and multi-LLM extraction, Graph-RAG versus text-only or no RAG, CP-SAT versus heuristic or greedy schedulers, sensitivity to hyperparameters (λ, μ, ν) in the scoring functions, and an analysis of extraction and scheduling failure cases.

- The data provenance for the “difficulty” and cost metrics is unclear. Using average grades as a proxy for difficulty introduces bias because grade distributions are affected by inflation, disciplinary differences (e.g., STEM grade penalties), instructor variation, and incomplete publication of grade data. Without transparent documentation of data sources, time span, institutional coverage, and normalization methods, the metric risks producing biased recommendations that favor easier courses over essential but challenging ones.

- Scalability and maintenance issues are not discussed. The Agent Forest appears to require institution-specific agents and prompts, yet the paper does not quantify the maintenance overhead as catalogs evolve, the latency of updates, or the long-term consistency of the Graph-RAG framework.

- Reproducibility is limited. No code, prompts, datasets, or scheduler configurations are released. Although the NCES source is cited, the four labeled ground-truth datasets and configuration files used in evaluation are not provided, preventing independent verification.

- The accuracy of policy formalization is unverified. The LLM translates catalog and policy text into formal scheduling rules, but the authors do not measure the precision or recall of this translation process. Such evaluation is essential to ensure compliance with institutional and accreditation policies.

- The contribution is primarily engineering rather than methodological. The paper integrates established techniques—LLM-based extraction, knowledge graphs, and constraint programming—into a coherent system but does not introduce fundamentally new algorithms or theoretical insights. Its main value lies in system design and implementation rather than conceptual innovation.

### Questions
1.	Prerequisite accuracy: How did you measure the reported 98.7% when the ground truth lacks prerequisites? Was there a separate human-labeled set, internal SIS data, or faculty validation? Please detail labeling protocol, sample size, and inter-rater agreement. 
2.	Policy rule translation: How do you evaluate the LLM’s policy-to-constraint mapping? Provide a labeled set of policy snippets → formal constraints with accuracy metrics and typical errors (e.g., unit brackets, co-req timing, repeat rules). 
3.	Ablations: Quantify benefits of multi-LLM extraction vs. single LLM; Graph-RAG vs. no-graph retrieval; CP-SAT vs. heuristic scheduling. 
4.	End-to-end feasibility: For a fixed set of majors across multiple schools/terms, what fraction of students obtain a fully feasible 4-year (or 2-year) schedule under typical constraints? Report time-to-degree, average deferrals, and units per term with confidence intervals. 
5.	Scalability & drift: What’s the maintenance cost per university (prompt edits, agent updates) as catalogs change? Do you version and regression-test KGs/constraints as part of “self-evolving” claims? 
6.	Difficulty metric & fairness: Where do AvgGrade statistics come from? Have you tested whether difficulty-based ranking skews against certain departments/instructor pools? Consider reporting robustness/fairness analyses. 
7.	Comparative baselines: Can you run head-to-head schedule feasibility or plan-quality comparisons against uAchieve Schedule Builder (auto combinations) or Series25 (admin scheduling), at least on a common synthetic/neutral benchmark? (CollegeSource, https://collegesource.com/degree-planning-tools/uachieve-schedule-builder/)
8.	User study: Any student/advisor evaluations (usability, trust, perceived correctness) versus text-only LLM baselines and existing campus tools?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The paper addresses the problem of constraint-aware study planners. This problem arises because of heterogeneous data sources, complex multi-constraint requirements, and dynamic course catalogs in educational institutions. The paper proposes a self-evolving multi-agent platform that integrates LLM-based extraction, knowledge-graph retrieval, and constraint-aware reasoning to generate adaptive, personalized study plans.

### Strengths
The motivation of the work is clear, the problem well presented, related work well presented, well organized, and the description of the approach is clear, understandable and reproducible. The paper proposes an innovative approach to solve a common problem in the education domain. The paper proposes a platform which achieves more than 98% accuracy across multiple universities.

### Weaknesses
The paper presents several problems in references, missing examples to help readers understand the paper, typos, punctuation, missing experimental environement, 
Line 46:  the reference provided is not an appropriate reference for the claim
Line 53: “Some of these systems serve general purposes (provide references for this claim), while others are tailored (provide a reference for this claim)” → Providing references at this stage is very important for readers
Line 54-57:  Providing references and examples is very helpful for the readers
Line 95: What is educational AI?
Line 135: “However, most educational RAG systems still treat constraints loosely and do not integrate exact solvers for timetable feasibility.” This claim should be supported by references or a study of existing work and this is not the case in the paper.

Lack of examples to explain how the agents performed their tasks. For instance, the paper may take the example of a math course and show step by step how the method is used

Line 151-152: such claim should be supported with references

Line 177: “extracted results with both RAG and JSON files.” → Extract results with JSON files or store in JSON file?

Line 266-267, line 427: check the punctuation

For the experiments, for the reproducibility, provide the LLMs used, hyperparameter, the performance of the tools used, libraries, etc.

How data is collected is not presented, and the title is data collection and preparation

Line 287: provide the meaning of GE

Line 432: SOTA is not presented in the experimentation settings

### Questions
Line 185-185: What is the role of each LLM?
Line 208: “we devised” → We designed?
After reading the whole paper, one question remains: How the agents are orchestrated?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces KNOWPLAN, a multi-agent AI system for degree pathway planning that integrates large language models (LLMs), knowledge graphs (KGs), and retrieval-augmented generation (RAG). The proposed pipeline orchestrates several specialized agents: an Agent Forest that parses heterogeneous course catalogs through multi-LLM extraction, a Graph-Construction Agent that builds a directed knowledge graph of prerequisites and dependencies, a Course Planning Agent that generates coarse multi-constraint learning trajectories, and a Curriculum Alignment Agent that refines them into conflict-free, term-level schedules using constraint programming. Experiments conducted across four universities demonstrate the effectiveness of the proposed system.

### Strengths
[+] An interesting and significant task in educational technology. 

[+] Well-designed multi-agent architecture integrating LLMs and constraint reasoning

[+] Handles cross-institution heterogeneity using public data sources

### Weaknesses
[-] The motivations of some module designs are unclear. For example, the equations (1) and (2) contain multiple items, and are they grounded by educational theories?

[-] Many technical details are missed, such as $π_{int}(v)$ and the weights of target-driven objectives for several modes.

[-] The evaluation metrics are not described, and there is no user or institutional validation.

[-] Heavy reliance on prompt design and multi-LLM extraction.

[-] No ablation or comparison on efficiency.

### Questions
1. Can the authors clarify how the Graph-RAG backbone prevents conflicts or duplication when curricula evolve?
1. Have the authors compared KNOWPLAN to a single-LLM baseline (e.g., GPT-4 with a prompting pipeline) to show the benefit of the multi-agent design?
1. Could the authors provide an ablation study or efficiency analysis of each agent’s contribution?
1. Could you provide some feedback on the system from real-world users?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a multi-agent platform, KNOWPLAN, a graphRAG system for degree pathway planning. An LLM-based KG construction agent first captures the prerequisites of the courses; a course planning agent then creates personalized plans that extract a subgraph that satisfies the prerequisite requirements. An experimental study verified the tool's usefulness across several scenarios.

### Strengths
S1. Having a RAG system for course selection and curriculum suggestion is an important application scenario. 
S2. Real-world education datasets are used for illustration. 
S3. The tool has addressed personalized recommendations.

### Weaknesses
W1. The technical challenges and contributions are limited. The method's generality needs further elaboration. 
W2. There is no guarantee of a certain quality for the recommended results. The solution looks straightforward. 
W3. There is no formal analysis on scalability and cost analysis to evaluate its overhead.

### Questions
D1. Some details are missing. For example, how the ground truth will be formally characterized and used to evaluate the output at each stage remains unclear. The section mentioned "manual work"—more elaboration is needed. 

D2. The problem seeks a Top-K solution—yet it remains unclear how, as an optimization problem, it would be tackled by a search strategy, and how LLMs and other overhead can be mitigated. 

D3. There is a lack of necessary details, such as how the scheduler works in a principled way with hyperparameters and machine-readable rules. If any new solution is used, it deserves more in-depth analysis. 

D4. Quite a few tasks have been outsourced to LLMs, including handling violations or constraints. There is an analysis of how LLM accuracy affects the quality of recommendations. Another missing link is whether there is a unique optimal solution, or how a top-K solution is computed to find a trade-off when these measures conflict (e.g., fees and total credits).

### Soundness
2

### Presentation
2

### Contribution
2
