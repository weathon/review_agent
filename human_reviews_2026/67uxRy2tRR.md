# SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
As language agents increasingly automate critical tasks, their ability to follow domain-specific standard operating procedures (SOPs), policies, and constraints when taking actions and making tool calls becomes essential yet remains underexplored. To address this gap, we develop an automated evaluation pipeline with: (1) sandbox environments containing 167 executable tools/functions across seven customer service domains with 70 service-specific, verifiable SOPs and constraints, (2) an automated test generation framework producing over 800 verified test cases, and (3) an evaluation harness to rigorously assess agent adherence. Our approach transforms each service-specific SOP code program into a directed graph of executable functions and requires agents to call these functions correctly based on natural-language SOP descriptions. The SOP code serves as oracle verifiers to assess compliance from multiple dimensions, reducing reliance on manual or LLM-based evaluations. Our benchmark covers seven custmor service domains with over 800 test cases. We evaluate 18 leading models and find the task remains challenging even for top-tier reasoning models such as o4-mini-high, with pass rates around 30% on certain difficult domains. Other powerful non-reasoning models perform worse than reasoning models, and smaller models (<32B) show limited capability. Additionally, language agents can be easily jailbroken to overlook SOPs and constraints. Code, data, and over 24k agent trajectories are released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SOPBench, a benchmark designed to evaluate the ability of language agents to follow standard operating procedures (SOPs) and constraints in customer service domains. It addresses the growing need for agents to adhere to domain-specific procedural guidelines, a capability underexplored in existing evaluations. The approach includes sandbox environments with 167 executable tools across seven domains, an automated test generation framework creating over 800 test cases, and a multi-level evaluation harness using oracle code as verifiers. The study finds that even top-tier models like o4-mini-high struggle, with pass rates around 30% in challenging domains, and highlights vulnerabilities to jailbreaking, emphasizing the need for improved adherence in high-stakes environments.

### Strengths
1. The development of a novel evaluation pipeline using executable SOP code as oracle verifiers enhances the reliability and objectivity of assessing agent compliance, reducing dependence on subjective human or LLM-based judgments. This approach ensures consistent and scalable evaluation across diverse scenarios, making it a robust tool for future research.  

2. The inclusion of 167 executable tools and 70 service-specific SOPs across seven customer service domains provides a comprehensive and realistic testing ground, covering areas like banking and healthcare. This breadth allows for a thorough assessment of agent capabilities, revealing domain-specific challenges and opportunities for improvement.  

3. The multi-level verification process, including outcome, step, and trajectory-level checks, offers a detailed analysis of agent adherence to SOPs, capturing both final results and procedural accuracy. This granularity provides valuable insights into agent decision-making, aiding in the identification of specific weaknesses.  

4. The release of code, data, and over 24k agent trajectories fosters transparency and enables the research community to replicate and build upon the findings. This open-access strategy promotes collaboration and accelerates advancements in language agent development.

### Weaknesses
1. The reliance on manually implemented sandbox environments and SOPs across seven domains may introduce potential biases or inconsistencies, as human design could overlook certain real-world complexities. This limitation might affect the generalizability of the results to less structured or more dynamic settings.  

2. The benchmark's focus on customer service domains, while comprehensive, may not fully represent the diversity of tasks in other critical areas like healthcare diagnostics or legal processes, potentially limiting its applicability. This narrow scope could miss unique procedural challenges in those fields.  

3. The finding that top-tier models achieve pass rates below 70% in challenging domains indicates that the benchmark's difficulty may exceed current model capabilities, potentially discouraging progress rather than guiding it. This high difficulty could lead to frustration rather than constructive development.  

4. The lack of detailed analysis on the impact of different tool-calling methods (e.g., ReAct vs. Act-Only) beyond basic comparisons limits insights into optimizing agent performance, leaving a gap in understanding how methodological choices affect SOP adherence. This omission could hinder tailored improvements for specific models.

### Questions
See weaknesses.

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
This paper introduces SOPBench, an automated evaluation pipeline designed to assess language agents' ability to follow domain-specific standard operating procedures (SOPs), policies, and constraints. SOPBench features executable environments with 167 tools/functions across seven customer service domains, an automated test generation framework with over 900 verified test cases, and a rigorous evaluation framework. The approach transforms SOP code into directed graphs of executable functions, requiring agents to act based on natural language SOP descriptions. The original code serves as oracle rule-based verifiers, reducing reliance on manual annotation. The authors evaluate 18 leading models, finding that even top-tier models struggle with SOP adherence, and that agents are susceptible to jailbreaking. The dataset and code are released for further research.

### Strengths
Originality: The focus on SOP adherence is novel and underexplored in existing benchmarks.
Quality: The automated, rule-based evaluation framework is robust and scalable.
Clarity: The methodology and results are clearly presented.
Relevance: The benchmark and findings are highly relevant for the development and deployment of reliable language agents.

### Weaknesses
The paper could provide more analysis on why certain models fail specific SOPs and constraints.

Limited discussion on the generalizability of SOPBench to domains beyond customer service.

The susceptibility to jailbreaking is noted, but mitigation strategies are not explored in depth.

Some experimental details (e.g., SOP selection criteria, test case diversity) could be expanded.

### Questions
My major question is if a task is governed by a well-defined SOP or protocol, is it necessary—or even optimal—to use a Large Language Model (LLM) to perform it, rather than simply encoding the SOP directly in code? Take the minor declaration task shown in Fig. 2 as an example, would the task be solved more easily by creating a functionality in the university admin website which can let the students just select the minor they want to declare, and do all those checks in the backend database? IMHO, creating this functionality by coding with the assistance of LLM could be more efficient and straightforward than going through the LLM agent. 

My other minor questions are
Can you elaborate on the criteria used to select SOPs and domains for inclusion in SOPBench?

How does SOPBench handle ambiguous or conflicting SOPs in real-world scenarios?

How does SOPBench compare to existing agent evaluation benchmarks in terms of coverage and difficulty?

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
The paper presents SOPBENCH, a benchmark for evaluating LLM agents on following.standard operating procedures and constraints. The benchmark is created by first manually defining an environment sandbox for seven domains. The sandbox consists of a set of constraints, 70 service functions and 97 helper functions that can be used to complete the tasks and verify constraints, a database schema containing the necessary information for constraint verification. Based on the environment sandbox, 830 test cases are generated using LLMs and verified using rule-based verifiers. Evaluation is conducted based on three criteria that can be measured using oracle code programs, namely outcome-level verification that verifies the final database state, step-level verification that verifies the constraint permissibility of each function called by the agent, and trajectory-level verification that verifies whether all the verification steps are completed in the correct order. Experimental results show that even SOTA models such as o4-mini-high struggle particularly on more challenging domains like hotel.

### Strengths
- Benchmark tackles an important problem of SOP following, which is required for agents to be useful in the real-world  
- Scalable and reproducible methodology for automatically generating and validating test cases 
- Comprehensive and reliable evaluation using oracle code programs
- Experimental results demonstrate the benchmark complexity and room for future research
- Performance degradation with a simple jailbreaking approach highlights the brittleness of modern LLMs on this task

### Weaknesses
- At its core, the task seems similar to structured tool calling. It is unclear as to what differentiates this benchmark from tool calling benchmarks, such as Berkley Function Calling Leaderboard that also conduct state-based evaluation and check whether the model’s function call trajectory includes the minimum necessary path. So, the core novelty of this benchmark, at least in the current scope of composition types, is limited.
- There is some inconsistency between Figures 7 and 11. For example, "show_available_rooms" has almost 0 success rate as per Figure 7 but Figure 11 shows high success for this service task.
- Limited error analysis. While the paper shows the model performance breakdown by evaluation criteria in Figure 8, and one example erroneous trajectory in the Appendix, it is still unclear why leading models fail. Also, what makes domains like Hotel more challenging?

### Questions
- How is evaluation done with temp = 0 when models like GPT-5 do not support temperature? 
- Since the constraints can be translated into a code program, how is the performance if the LLM is asked to generate code for constraint verification?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Since the research about the current language agent system perform on complying standard operating procedures (SOPs), policies, and constraints is limited, this paper aims to address the underexplored problem and introduces an automated evaluation pipeline, SOPBench. This pipeline measures procedural compliance across multi-step tool-use tasks. It constructs a benchmark consists of sandboxed environments, automated test generation, and evaluation mechanism. The authors conduct extensive experiments to evaluate the benchmarks with leading models and provide valuable insights. The release of this benchmark can benefit future research.

### Strengths
- The benchmark is large-scale and diverse, covering 167 executable tool functions, and 830 test cases from 7 customer-service domains. The authors also evaluate the benchmark with 18 leading models.

- The paper highlights that even current leading models struggle with some tasks and domains, which indicates that SOP control remains challenging, and the benchmark is both relevant and impactful.

- The paper also shows the vulnerabilities of current models to malicious jailbreaking, and found that easy jailbreaking can success on overlooking SOPs and constraints, which inspire safety studies using this benchmark.

- The work formalizes SOPs as directed action graphs and proposes a three-level evaluation methodology, which can provide multiple perspectives for evaluation to following studies; and the code-based verifiers provide the foundation for future work in RLVR.

- The extensive experiments regarding different metrics on different domains, models, adversarial attacks, tool use methods, and service tasks, further demonstrates the diversity of the benchmark, which can benefit various research in the community.

- The paper is well-written and well-organized, with intuitive figures and helpful examples that make it straightforward to understand.

### Weaknesses
- The domains can be more diverse to include some industry domains with more complex and messier SOPs and constraints, i.e., requiring larger amounts of service/helper functions and constraints.

- It could be better to categorize these tasks into different levels of difficulty.

- It may be beneficial to also show the statistics of average or deepest constraint dependency across domains, and provide additional analysis regarding how the constraint complexity impacts.

### Questions
- Can the current pipeline be scaled to some real scenarios with a large amount of rules and complex dependency between functions?

- The test cases are structured based on the tools and constraints, but how were the free-text user inputs generated? In addition, can the user inputs include “outlier” intents that involve constraints outside the predefined tool and constraint space?

### Soundness
3

### Presentation
3

### Contribution
3
