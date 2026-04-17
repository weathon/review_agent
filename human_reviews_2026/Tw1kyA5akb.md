# ARS: Automatic Routing Solver with Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Real-world Vehicle Routing Problems (VRPs) are characterized by a variety of practical constraints, making manual solver design both knowledge-intensive and time-consuming. Although there is increasing effort in automating the design of routing solvers, existing research has explored only a limited array of VRP variants and fails to adequately address the complex and prevalent constraints encountered in real-world situations. To fill this gap, we propose the Automatic Routing Solver (ARS), which leverages Large Language Model (LLM) agents to enhance a backbone metaheuristic framework. ARS automatically generates constraint-aware heuristic code from natural language problem descriptions, enabling the framework to handle a wider range of VRP variants without relying on cumbersome modeling rules. Alongside ARS, we introduce RoutBench, a benchmark comprising 1,000 VRP variants derived from 24 attributes, designed to rigorously evaluate the effectiveness of automatic routing solvers in handling VRPs with diverse practical constraints. In our experiments, ARS achieves a success rate of over 90\% on common VRPs and over 60\% on RoutBench, outperforming the other seven LLM-based methods by at least 30\% in success rate. Compared to three general-purpose solvers, the ARS framework not only makes it easier for an LLM to generate correct code, with approximately 25\% higher correctness, but also achieves superior solving efficiency across many VRP variants.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ARS (Automatic Routing Solver), a framework that leverages LLMs to solve real-world VRPs with complex and diverse constraints. Unlike traditional solvers, ARS can automatically generate constraint-aware heuristics from natural language descriptions, reducing the need for manual expert modeling as problem constraints become varied or more complicated.​

The contributions include:  1) The method uses LLMs to interpret users’ descriptions of routing problems, selects relevant constraints from a database, and generates Python code that is plugged into a backbone heuristic solver. This allows ARS to flexibly adapt to many different and realistic VRP scenarios, from heterogeneous vehicle fleets to time windows, dynamic demands, and priority rules.​
2) The authors introduce RoutBench, a comprehensive benchmark suite with 1,000 VRP variants built from combinations of 24 practical constraints. This allows for systematic and rigorous testing of automatic routing solvers in diverse, real-world cases.​
3) ARS achieves high success rates (over 90% on standard VRPs and over 60% on complex RoutBench problems), outperforming seven other LLM-based code generation methods and three general-purpose solvers. The approach also makes it easier for LLMs to generate correct code, with less complexity and higher efficiency.​
4) By automating constraint handler generation and linking it to established search heuristics, ARS shows strong generalization across problem types and is ready for extension beyond routing, such as bin packing.​
5) ARS automatically generates constraint-aware heuristics to solve diverse real-world VRP variants, using LLMs and a large, systematic benchmark for evaluation.

### Strengths
1. ARS uses Large Language Models to automatically create heuristics and constraint handlers for many real-world routing problems. This makes solver setup fast and easy, helping users adapt to changing or complex requirements without needing expert coding or modeling.​
2. The RoutBench benchmark covers 1,000 problem variants built from 24 real-world constraints. ARS demonstrates strong generalization and robust performance across this diverse set, including highly constrained and previously untested scenarios.​
3. On standard VRPs and complex RoutBench problems, ARS achieves much higher success rates and code correctness (over 90% on common tasks, above 60% on tough cases), outperforming both 7 alternative LLM-based approaches and 3 established solvers in efficiency and reliability.​
4. By generating only the parts of code needed for constraint handling, ARS significantly lowers the amount of generated code required and reduces errors compared to general-purpose modeling frameworks.​
5. The framework works with various state-of-the-art language models (GPT-4, DeepSeek, LLaMA), showing improved results as model quality increases and supporting scalability for different environments.​

### Weaknesses
1. While ARS claims to automate solver design with LLMs, the underlying routing backbone relies heavily on conventional metaheuristics (destroy/repair, 2-OPT, SWAP, SHIFT), offering little methodological advancement over established VRP solvers. The core innovation is largely in combining existing components rather than proposing fundamentally new search or optimization strategies.​

2. The framework delegates constraint handling and validation almost entirely to LLM agents, making solution accuracy and feasibility highly dependent on the ability of language models to interpret and synthesize correct code from often ambiguous or poorly-specified natural language descriptions. This introduces significant risk in reliability, especially as constraints grow complex or nuanced beyond the LLM’s training distribution.​

3. The paper fails to seriously investigate the risk of hallucination, syntax errors, and subtle bugs common in LLM-generated code. Runtime error rates are measured, but no robust fail-safe is proposed for erroneous validation or constraint logic. This potentially impacting real-world deployment.​

4. The backbone search algorithm is fixed and non-adaptive, restricting ARS to a single solution-generation paradigm. This limits potential improvements from modern neural combinatorial optimization, hybrid strategies, or domain-specific deep learning advances that could outperform classic heuristics, especially for harder VRP variants and scalability.​

5. RoutBench is much richer than previous benchmarks but is entirely built from synthetic combinations of constraint types with fixed datasets and rules. Real-world VRPs often include messy, dynamic, multi-modal data, and less formalized or evolving constraints. These are not examined, and thus claims about real-world generalization are partly unproven.​

6. The approach shows success up to 200-node problems, but does not evaluate extreme scalability to industrial-size problems (thousands of locations, live routing updates), nor does it rigorously compare wall-time performance to state-of-the-art solvers like HGS or commercial systems in diverse operational contexts.​

7. ARS focuses on constraints expressible in the current framework and retrievable from its database. Constraints that require joint probabilistic modeling, dynamic uncertainties, learning-based preference handling, or integration with external platforms are out of scope.​

8. Automated code generation for optimization raises serious issues in operational safety, security, and fairness that are only mentioned briefly. There is no systematic audit or mitigation for risks of unintended or adversarial LLM outputs, data leakage, or regulatory compliance.​

9. While the paper claims extensibility to other domains (e.g., bin packing), there is no empirical evidence or adaptation strategies for handling fundamentally different combinatorial problems, which limits generality.

### Questions
1. Considering that the proposed framework relies on LLMs to generate constraint-checking code from natural language problem descriptions, how the authors guarantee correctness, robustness, and safety of the generated code? This is more important in the presence of ambiguous, under-specified, or adversarial input. What evidence can the authors provide that ARS would not silently accept incorrect solutions or propagate bugs to mission-critical logistics environments?​

2. The proposed benchmark (RoutBench) is well-constructed but fully synthetic. How does ARS handle the noisy, unstructured, and evolving constraint types found in real operational datasets, such as time-varying networks, legal constraints, or incomplete business rules? Can authors demonstrate successful generalization or present negative results for deployment in high-stakes, dynamic, real-world logistics problems?​

3. While the authors claim that ARS can be extended to other combinatorial domains (bin packing, job-shop scheduling, etc.), the experiments are strictly for VRPs. What architectural or algorithmic changes are needed for ARS to work on these different optimization classes, and what unique challenges arise if the problem structure diverges (e.g., from graph to sequence to set)?

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
4

### Summary
This paper introduces the Automatic Routing Solver (ARS), a novel framework that leverages Large Language Models (LLMs) to automatically generate solvers for a wide variety of Vehicle Routing Problems (VRPs). The core idea is to use an LLM not to create an entire solving algorithm from scratch, but to generate problem-specific, constraint-aware heuristic code that can be integrated into a robust, general-purpose metaheuristic backbone. The ARS framework consists of a database of fundamental VRP constraints, an LLM-driven module that selects relevant constraints and generates new "checker" and "scorer" programs based on a natural language problem description, and an augmented heuristic solver that uses this generated code to guide its search.

As a second major contribution, the paper presents RoutBench, a new comprehensive benchmark comprising 1,000 VRP variants derived from 24 different real-world constraints. This benchmark is designed to rigorously evaluate the generalization capabilities of VRP solvers. Experiments show that ARS significantly outperforms other LLM-based methods in successfully generating correct code and provides a more effective and efficient framework for tackling diverse VRPs compared to general-purpose commercial solvers like Gurobi and CPLEX when paired with an LLM.

### Strengths
1. The paper's primary strength is the innovative design of the ARS framework, which intelligently separates the general solver backbone from the LLM-generated, problem-specific heuristic components. This is a clever and effective way to combine the reasoning power of LLMs with the proven search capabilities of metaheuristics.

2. The introduction of RoutBench is a major contribution in its own right. It provides a large-scale, diverse, and well-structured testbed for evaluating the generalization capabilities of VRP solvers, which has been lacking in the field.

3. The work addresses a highly significant and practical problem: reducing the immense manual, expert-driven effort required to design and implement solvers for the vast and growing number of VRP variants encountered in logistics and transportation.

### Weaknesses
1. While the paper proposes the ARS framework, its originality is limited. The framework's RAG component utilizes existing technology, the "checker" and "scorer" steps are based on established ideas from heuristic VRP solvers, and the subsequent heuristic algorithm is also a pre-existing method. Overall, the lack of substantial novel content is the paper's most significant weakness.

2. The paper relies on a single-point-based search framework. It is unclear how the LLM-generated Constraint-Aware Heuristic (CAH) would perform when integrated with other powerful metaheuristic backbones, such as population-based genetic algorithms or ant colony optimization.

3. While ARS is shown to be superior to standard prompting, the prompts provided in the appendix appear carefully engineered. The framework's robustness to minor variations in prompt phrasing or in the natural language problem descriptions is not explored.

### Questions
1. The manuscript would benefit from a careful proofread to correct typos and imprecise descriptions. For example:
On line 49, "Gurubi" is misspelled and should be "Gurobi". On line 409, the text discussing solver performance refers to "Table 1" when it should be "Table 3". On line 322, the claim that the distribution in RoutBench "reflects the proportions of the full set of 5624 problems" is not entirely accurate, as the visual distributions in Figure 2 show noticeable differences. On line 372, the statement that other methods have a success rate "merely around 10%" on RoutBench-H is imprecise. The actual results in Table 2 range from 10.8% to 15.6%. A thorough review of the entire text is recommended to find and fix any similar issues.

2. The caption for Table 3 states, "The table presents the gaps compared to the results obtained by ARS." Shouldn't the gap be compared to the BKS (Best Known Solutions)? Or, were the BKS in this paper actually the results obtained by ARS? Please provide an explanation.

3. The paper rightly states it does not aim for state-of-the-art (SOTA) performance on specific VRPs. However, the performance gap between ARS and specialized solvers like HGS (Table 17) is large. A more detailed discussion of this trade-off would be beneficial.

4. Could you provide a concrete example of a "Logical Bug" from your failure analysis (Table 14)? What does such a bug look like in the generated code, and what underlying reasoning failure from the LLM do you believe it represents?

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
This paper proposes an LLM-assisted framework that turns natural-language VRP into code for a constraint-aware heuristic. It retrieves exemplar constraints, generate a checker and violation scorer, and plugs them into a destroy–repair + local search backbone.

### Strengths
• The paper releases RoutBench: 1,000 VRP variants (each with NL description, data, and validation code), which represents a broad benchmark contribution.

• The ablations show each component of the framework matters, indicating the effectiveness of the design.

### Weaknesses
•	The evaluation is based on the correctness/coverage of the per-instance validation code. It is not reliable enough to ensure that the generated program works for a class of VRP. If a checker under-specifies edge cases, SR can be overstated.

•	Best-Known Solutions (BKS) for RoutBench are produced by ARS itself under strict stops. It seems that this method cannot ensure the actual (near)optimal solution, and thus leads to a benchmark circularity risk.

•	The superior performance partly reflect a competent destroy-repair + local search backbone. There’s limited comparison to stronger VRP metaheuristics (e.g., state-of-the-art HGS variants) within the same ARS-style interface, or introducing the LLM-based heuristic discovery methods.

•	The paper hints at portability to other COPs (e.g., 3D bin packing) but provides no experiments. There is no evidence that if ARS can serve as “general framework” also for other COPs.

### Questions
(1)	In Table 17, the comparative result between ARS and other baselines are reported. Can ARS always produce feasible solutions for the CVRPLIB instances? If not, the SR metric should also be indicated.

(2)	LLMs such as gpt-3.5-turbo are very old now. It would be better if new-generation LLMs, including reasoning model, can be evaluated in Figure 6.

(3)	Is that possible to use ARS to solve VRP described by any NL description? For example, a user may choose to describe a VRP by a real-world scenario.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper targets the Vehicle Routing Problem (VRP) and its variants, proposing a framework that leverages Large Language Models (LLMs) to automatically generate heuristic solvers, along with a benchmark dataset for evaluating their effectiveness.
To ensure that generated solutions satisfy problem constraints, the framework employs a database of canonical constraints and Retrieval-Augmented Generation (RAG) to produce both a constraint-checking program and a constraint-satisfaction scoring function. These generated components are combined with existing local search heuristics, enabling the solver to better satisfy natural-language-specified constraints.
Empirical results show that the proposed method achieves higher constraint satisfaction rates and lower runtime error rates than existing LLM-based approaches. Compared with prompting standard LLMs to directly generate constraint code for a generic solver, the proposed approach demonstrates consistently higher constraint satisfaction.

### Strengths
- The numerical experiments convincingly show that combining RAG-based constraint code generation with existing heuristics outperforms the baseline approach of prompting LLMs directly.

- The proposed approach of translating natural-language constraints into executable programs has strong potential to simplify the process of developing constraint-specific heuristic algorithms.

### Weaknesses
- The ablation study shows that leveraging the constraint database significantly improves constraint satisfaction. However, in practical applications, problem constraints are not always well-studied or included in such databases. Thus, the generality of the proposed method may be limited when dealing with novel or previously unseen constraints.

- Although the framework aims to extend existing local search heuristics to handle diverse constraints, the paper does not analyze the scalability of the approach in terms of the number and variety of constraints it can handle. From a practitioner’s perspective, it is often more important for a solver to perform well on a specific problem class rather than to support a wide range of heterogeneous constraints.

### Questions
- In terms of constraint complexity (number and diversity of constraints), how complex can the problems handled by the proposed framework be?

- Is the proposed framework mainly effective for well-studied, common constraints? Can it also handle new or rarely encountered constraint types?

- What is the main advantage of the proposed framework compared to solvers specifically designed for particular constraints? If the method only performs well for familiar constraints and struggles with complex ones, its practical distinction from specialized solvers may be limited.

### Soundness
2

### Presentation
3

### Contribution
2
