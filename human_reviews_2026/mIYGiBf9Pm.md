# ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
While Large Language Models (LLMs) have shown remarkable advancements in reasoning and tool use, they often fail to generate optimal, grounded solutions under complex constraints. Real-world travel planning exemplifies these challenges, evaluating agents’ abilities to handle constraints that are explicit, implicit, and even evolving based on interactions with dynamic environments and user needs. In this paper, we present ATLAS, a general multi-agent framework designed to effectively handle such complex nature of constraints awareness in real-world travel planning tasks. ATLAS introduces a principled approach to address the fundamental challenges of constraint-aware planning through dedicated mechanisms for dynamic constraint management, iterative plan critique, and adaptive interleaved search. ATLAS demonstrates state-of-the-art performance on the TravelPlanner benchmark, improving the final pass rate from 23.3% to 44.4% over its best alternative. More importantly, our work is the first to demonstrate quantitative effectiveness on real-world travel planning tasks with live information search and multi-turn feedback. In this realistic setting, ATLAS showcases its superior overall planning performance, achieving an 84% final pass rate which significantly outperforms baselines including ReAct (59%) and a monolithic agent (27%).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents ATLAS, a multi-agent framework designed to tackle travel planning, which requires reasoning under complex, evolving, and often implicit constraints. ATLAS formalizes the task as a dynamic constraint satisfaction problem and introduces two cooperative agents, including a constraint manager for explicit and implicit constraint construction and a planner-checker pair for iterative plan generation and validation. Through such a multi-agent system design, ATLAS continuously refines plans, resolving information gaps that  single-agent systems often fail to address. Experiments on the TravelPlanner and Flex-TravelPlanner benchmarks show that ATLAS consistently achieves state-of-the-art performance, surpassing strong baselines such as ReAct, Reflexion, and PMC.

### Strengths
1. The paper is grounded in a clear and compelling motivation The authors effectively articulate why constraint awareness and multi-agent collaboration are essential for complex planning scenarios, making the research direction both timely and impactful.
2. The experimental evaluation is extensive and rigorous. The authors conduct detailed comparisons with several strong baselines (e.g., ReAct, Reflexion, EvoAgent, and PMC) and perform ablation studies that carefully examine the contribution of each module within ATLAS, such as the constraint manager and checker. The results are analyzed with clear quantitative evidence, supported by thoughtful discussions and visualizations.
3. The test environments go beyond the standard TravelPlanner sandbox benchmark. The introduction of Flex-TravelPlanner for multi-turn interaction and the integration of live web search in real-world scenarios substantially enhance the ecological validity of the evaluation. These settings convincingly demonstrate ATLAS’s robustness, adaptability, and practical potential for real-world deployment beyond controlled benchmark conditions

### Weaknesses
The only concern is that, after reviewing the prompts provided in the appendix, the framework appears to require a considerable amount of manual effort to design suitable prompts for different agents in each specific scenario. While this is acceptable given that the paper focuses solely on travel-planning tasks, such dependence on handcrafted prompts may limit ATLAS’s generalization and scalability to other domains or broader real-world applications.

### Questions
1. In Table 10 (Appendix D.3.2), it is somewhat surprising that Gemini-2.5-Pro achieves better performance on the hard set than on the medium set. I am curious about this observation, as it appears that only ATLAS exhibits such an uncommon pattern.

### Soundness
4

### Presentation
4

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
This paper proposes ATLAS, a multi-agent framework formalized as a Constraint Satisfaction Problem, to address LLMs’ limitations in handling complex constraints for real-world travel planning. The framework comprises five agents (Search Agent, Constraint Manager, Planner, Checker, Search Advisor) targeting three challenges: constraint construction (explicit/implicit constraint extraction), constraint-aware planning (valid scheme generation), and information gap resolution (adaptive supplementary search).

### Strengths
1. ATLAS offers a structured approach to constraint discovery and information search, providing valuable heuristic insights.

2. Evaluations are conducted in real-world settings, incorporating live search (via Google Search instead of sandboxed tools) and reporting hallucination rates, demonstrating practical applicability.

3. The method achieves state-of-the-art performance on TravelPlanner, highlighting its effectiveness.

4. The algorithm is clearly described, the writing is accessible.

### Weaknesses
1.The framework, while well-engineered, largely repackages existing multi-agent and verification ideas. The plan–check–revise–search pipeline resembles prior systems like CRITIC, PMC, and LLM-Modulo Planning.

2.Although the task is formalized as a CSP, the link between classical CSP algorithms and the actual implementation remains superficial.

3.Evaluation is predominantly limited to TravelPlanner, raising concerns about overfitting and limited generalizability.

4.The multi-agent design lacks clear justification. While multi-agent systems often boost performance in travel planning, the specific agent roles and structures here appear arbitrary. Comparisons with more baselines (e.g., LLM-Modulo, verifier-aided approaches) and evaluations across diverse models (including open-source ones) and benchmarks are needed to demonstrate generality.

### Questions
Q1: What is the relationship between the multi-agent design and the CSP formalization?

Q2: Could similar improvements be attained by enhancing single-agent strategies (e.g., ReAct with verifier loops)?

Q3: Why does the algorithm perform well in multi-turn settings where others fail?

Q4: How would ablating specific agents impact performance?

### Soundness
2

### Presentation
2

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
- The paper proposes **ATLAS**, an agentic framework for real‐world travel planning that handles several constraints.
- The authors conduct **experiments** under multiple settings, including single-agent, multi-agent, and multi-agent with live search, showing robustness across settings.
- Their experiments demonstrate that ATLAS effectively reduces hallucinations and improves planning reliability compared to existing baselines.

### Strengths
Overall, the paper is well written and covers a comprehensive range of experiments. Here are some breakdowns:

- **Problem design / framework:** Problem settings are well estabilshed, and they are grounded in classic planning research.
- **Writing:** Well organized and easy to follow.
- **Experiments:** Comprehensive experiments, divided into single, multi, and multi+live settings.
- **Results:** Effectively addresses hallucination, which is a major issue, and shows remarkable improvements compared to baselines.
- **Analysis:** Provides a comprehensive failure analysis with fine-grained breakdowns.

### Weaknesses
### Weaknesses that may affect overall assessment

**W1. Lack of analysis**

> Related part: Section 4, Figure 3
> 
- Although ATLAS demonstrates strong performance,  it’s crucial to show *why* a method works or fails — that would represent an important contribution.
- While Figure 3 provides a weak form of analysis, ATLAS follows an agentic workflow with multiple subtasks (e.g., search, plan). It would be valuable to show the success rate of each subtask and how failures in specific stages contribute to overall failure. There could even be interesting cases where later stages compensate for earlier failures.
- Different models may also show varying strengths across subtasks. Understanding *which model excels at which component* could guide future research into multi-agent workflows for more powerful travel-planning agents.

**W2. Lack of baseline comparison**

> Related part: W1, Appendix A
> 
- Although the authors mention the paper [1] and note that their problem scope differs, I still think a comparison is needed. (Or with other sand-box based framework)
- I understand [1] *operates in a sandbox-based setup,* but it remains a strong tool achieving top performance. **The authors could, for example, disable ATLAS’s search component or provide the same searched data to [1] and compare outputs.

---

### Weaknesses that likely affect but not majorly

**W3. Lack of efficiency analysis**

> Related section: 3.4
> 
- The authors claim ATLAS is efficient due to information reuse. A cost analysis (e.g., API cost comparison with single-model baseline) would strengthen this claim.

---

### Weaknesses unlikely to affect assessment but useful suggestions

**W4. Consideration of “preferences”**

- Recently, several studies [2,3] have explored preference constraints.
- If the authors show that ATLAS performs well under such conditions, the paper would be even stronger.
---
Reference:
    
    [1] Large language models can solve real-world planning rigorously with formal verification tools. Hao et al, 2025.
    
    [2] COMPASS: A Multi-Turn Benchmark for Tool-Mediated Planning & Preference Optimization, Qin et al, 2025
    
    [3] Flex-TravelPlanner: A Benchmark for Flexible Planning with Language Agents, Oh et al, 2025

### Questions
**Q1 Regarding INTERLEAVED SEARCH** 

- INTERLEAVED SEARCH seems to play an important role, in what proportion of test cases was it actually triggered?

### Soundness
3

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
5

### Summary
This paper considers the travel planning problem which should follow the real-world constraints. Te authors present a multi-agent framework, ATLAS, to trackle three identified core challenges, constraint construction, constraints-aware answering, and resolving information gap. Specifically, they design a constraint management agent to identify and codify the constraints, a planner agent to propose a travel plan with the extracted constraints in a propose-validate loop, a search agent to diagnose failures and guide the further information collection. They conduct the experiments on the TravelPlanner benchmark and its multi-turn variants, to validate its effectiveness.

### Strengths
1. The most compelling aspect of this work lies in its decomposition of the travel planning problem into three well-justified challenges, a perspective with which I strongly agree, not only in the context of travel planning but also in broader real-world scenarios where agents must fulfill user needs.  
2. Building upon the identified challenges, the authors have thoughtfully designed distinct agents that collaborate effectively to accomplish the complex and practical goal of travel planning.  
3. The performance of ALTAS is significant and the effectiveness is well validated based on the ablation study.

### Weaknesses
I don't have major concerns on this work, here are some minor concerns. 

1. It is worth noting that the experimental comparison in this paper does not include OpenAI's GPT models. Based on my own experiments with TravelPlanner, GPT-5 in sole-planning mode (i.e., planning under constraints as described in the paper) already achieves performance far exceeding the near-zero result reported for the original TravelPlanner, even approaching the performance of ALTAS in Table 2. Given that constraint extraction in TravelPlanner is relatively straightforward, the main challenge in a multi-agent system would still lie in planning under constraints. Therefore, I would be interested to see the results of ALTAS when integrated with GPT-5.  

2. I agree that the insights and the framework proposed in this paper are generalizable to tasks like travel planning, which require both information gathering and strict constraint satisfaction to meet user needs. However, the validation in this work is limited to the travel planning scenario. Although the TravelPlanner benchmark has gained broad attention, its support for constraint satisfaction is confined to propositional logic, includes only a limited set of constraints, and the process of extracting constraints from its template-synthesized queries is relatively straightforward [1]. In real-world scenarios, users often express diverse needs that require first-order logic (FOL) for proper representation, thereby extending constraint extraction and validation into a combinatorial FOL space. From this perspective, if the authors intend to claim robust constraint satisfaction capability in the travel planning domain, it would be valuable to include evaluation on the ChinaTravel benchmark [2], which assesses first-order logic constraint satisfaction. 

[1]. Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools.    
[2]. ChinaTravel: An Open-Ended Benchmark for Language Agents in Chinese Travel Planning

### Questions
1. I'm not entirely clear about how the checker component operates in practice. Is it based purely on LLM self-verification, or does it adopt an "LLM-modulo" style approach that incorporates additional symbolic verification tools?
2. Could the proposed method be generalized to first-order logic, particularly how the constraint extraction and verification processes could be generalized?

### Soundness
3

### Presentation
3

### Contribution
3
