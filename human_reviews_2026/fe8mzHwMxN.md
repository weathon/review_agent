# MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We introduce MCP-Bench, a benchmark for evaluating large language models (LLMs) on realistic, multi-step tasks that demand tool use, cross-tool coordination, precise parameter control, and planning/reasoning for solving tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28 representative live MCP servers spanning 250 tools across domains such as finance, traveling, scientific computing, and academic search. Unlike prior API-based benchmarks, each MCP server provides a set of complementary tools designed to work together, enabling the construction of authentic, multi-step tasks with rich input–output coupling. Also, tasks in MCP-Bench test agents’ ability to retrieve relevant tools from fuzzy instructions without explicit tool names, plan multi-hop execution trajectories for complex objectives, ground responses in intermediate tool outputs, and orchestrate cross-domain workflows—capabilities not adequately evaluated by existing benchmarks that rely on explicit tool specifications, shallow few-step workflows, and isolated domain operations. We propose a multi-faceted evaluation framework covering tool-level schema understanding and usage, trajectorylevel planning and task completion. Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce MCP-Bench, a benchmark designed to evaluate large language models (LLMs) on realistic, multi-step tasks. MCP-Bench collectively span 250 tools across diverse domains including finance, traveling, scientific computing, and academic search.
The authors design tasks in MCP-Bench to test capabilities of LLM agents: retrieving relevant tools from fuzzy instructions (without explicit tool names), planning multi-hop execution trajectories for complex objectives, grounding responses in intermediate tool outputs, and orchestrating cross-domain workflows. These capabilities are not adequately evaluated by existing benchmarks. This work serves as a significant supplement.

### Strengths
1.  Connects multiple production-grade MCP servers, supporting cross-server dependency chains and multi-hop workflows, which addresses the issue of "isolated tools" in traditional API benchmarks.
2.  Generates tasks with vague descriptions through an automated pipeline, which is more in line with real user needs and fills the gap of "dependency on explicit steps" in existing benchmarks.
3. This work serves as a significant supplement to the landscape of agent benchmarks.

### Weaknesses
*   The "execution success rate" and "schema compliance rate" in rule-based evaluation do not distinguish between error types (e.g., "tool selection errors", "parameter format errors", "parameter value errors"). If the performance of models in different error types could be analyzed, it would be possible to reveal the weaknesses of models more accurately.
*   Tasks are not classified by difficulty levels, making it hard to analyze the generalization ability of models under different difficulty levels—for example, whether strong models (such as GPT-5) have more significant advantages in difficult tasks.
*   The judge model used is GPT-4o-mini, and its capability needs to be verified. As shown in the appendix, the analysis of o4-mini, GPT-4o, and GPT-4o-mini as judge models indicates that after combining prompt shuffling and score averaging, the consistency of model scores improves. However, whether an absolute value of 1.42 out of 2 is sufficient to demonstrate accuracy remains questionable (there is still a relatively large gap from human agreement). In addition, the authors only analyzed homologous models from OpenAI; it is unclear whether models from other series (such as Claude, Qwen) also achieve such consistency.

### Questions
please refer to weakness

### Soundness
3

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
This paper introduces MCP-Bench, a benchmark designed to evaluate tool-using LLM agents on complex, real-world tasks. Built on the Model Context Protocol (MCP), it connects agents to 28 live servers offering 250 tools across domains like finance, science, and travel. Unlike prior benchmarks, MCP-Bench features tasks with fuzzy instructions that omit explicit tool names or steps, requiring agents to infer workflows and coordinate tools across multiple servers. The benchmark includes 104 automatically synthesized tasks and employs a comprehensive evaluation framework combining rule-based metrics and a robust LLM-as-a-Judge approach. Experiments on 20 models reveal that while basic tool-use capabilities like schema understanding have converged, higher-order reasoning—particularly long-horizon planning and cross-server orchestration—remains a key differentiator, with top-tier models like GPT-5 and o3 significantly outperforming smaller counterparts.

### Strengths
1.	This paper makes a significant and timely contribution to the field of AI agent evaluation. It moves beyond the established paradigm of benchmarking with isolated APIs or simulated environments by leveraging the emerging Model Context Protocol (MCP) to create a benchmark grounded in a live, heterogeneous ecosystem of real-world tools. This formulation is novel and impactful. Key innovative choices include the focus on "fuzzy instructions," which reframes the agent's challenge from simple tool-calling to genuine intent interpretation, and the systematic inclusion of distractor servers, which tests robustness in a way previous benchmarks have neglected.

2.	The benchmark's construction is rigorous and scalable, featuring a substantial scale that is carefully curated through an automated pipeline involving dependency chain discovery and quality filtering. The authors further demonstrate methodological rigor by proactively addressing known issues like prompt-ordering bias through techniques like prompt shuffling and score averaging, with ablation studies provided to validate these choices. The results are comprehensive, evaluating 20 diverse models and providing fine-grained insights that go beyond a simple leaderboard.

### Weaknesses
1.	The individual components are not novel in isolation: using MCP for evaluation is explored in works like MCPEval and MCP-RADAR, "fuzzy instructions" are a common concept in human-computer interaction, and "LLM-as-a-Judge" is an established methodology. The paper could more clearly delineate its specific novelty beyond the scale and integration of these components. 

2.	The exclusive reliance on o4-mini for both task synthesis and as the default LLM Judge requires further justification. The authors should explain the rationale behind this specific choice (e.g., superior performance in pilot studies, cost-effectiveness, or instruction-following capabilities) over other powerful models. 

3.	The calibration of the LLM Judge remains a concern. The ablation study in Appendix A.8 demonstrates improved consistency of the pipeline itself but does not validate the Judge's strict scoring standards against human judgment. A critical addition would be a study where human experts directly score the task completion. This would allow the authors to calibrate the LLM Judge.

4.	The results successfully identify "planning" as the key differentiator but offer limited insight into the underlying causes of failure. The taxonomy of errors is under-explored. The authors should include a qualitative analysis of common failure modes. For example, do agents fail due to: (a) incorrect dependency inference, (b) getting stuck in loops, (c) poor state management across rounds, or (d) an inability to synthesize information from multiple tools? Identifying these categories would provide more actionable guidance for future model development than the high-level conclusion that "planning is hard."

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MCP-BENCH, a benchmark designed to evaluate Large Language Model (LLM) agents on complex, multi-step, real-world tasks that require tool use. Built on the Model Context Protocol (MCP), it connects LLM agents to 28 live servers spanning 250 tools across diverse domains like finance, science, and travel. The paper also presents a multi-faceted evaluation framework combining rule-based checks and LLM-as-a-Judge scoring, along with a large-scale empirical study of 20 state-of-the-art LLMs.

### Strengths
1. Innovative and Realistic Benchmark Construction using MCP: The paper makes a significant leap by building a benchmark on top of real, live MCP servers, creating an ecosystem of 250 structured tools. This is a substantial scale and diversity improvement over prior API-based benchmarks (e.g., ToolBench, τ-Bench) or smaller MCP-based benchmarks (e.g., MCP-RADER).

2. Comprehensive and Rigorous Task Synthesis and Evaluation Framework: The automated task synthesis pipeline (dependency chain discovery, quality filtering, and task fuzzing) is a well-thought-out solution for generating challenging, scalable, and realistic tasks. The "fuzzing" step, which removes explicit tool names and steps, is particularly valuable for testing true reasoning and planning capabilities. Besides, the two-tiered evaluation framework is robust. It effectively combines low-level, objective rule-based metrics (e.g., schema compliance) with high-level, strategic LLM-judge scoring (e.g., planning effectiveness). The use of prompt shuffling and score averaging is a commendable step towards mitigating bias in the LLM-judge component.

3. Revealing Large-Scale Empirical Analysis with Actionable Insights: The evaluation of 20 state-of-the-art LLMs (from GPT-5 to smaller open-source models) provides a comprehensive landscape of current capabilities. The results clearly show that while basic schema understanding has converged for top models, significant gaps remain in higher-order reasoning.

### Weaknesses
1. The paper lacks critical details for full reproducibility. The specific versions, configurations, and hosting environments of the 28 MCP servers are not described. The stability, latency, and potential rate limits of these live servers could significantly impact experimental results and their consistency.

2. Experimental settings for running the 20 different LLM agents are glossed over. Details such as the specific inference parameters (temperature, etc.), computational resources required, and strategies for handling non-OpenAI models (e.g., authentication, API endpoints) are missing. There is also little discussion of the benchmark's long-term maintainability given its reliance on external, potentially changing, MCP servers.

### Questions
1. what's the detailed settings for the evluated llms? 
2. what's the weakness of your benchmark compared to previous api-based one? it is hard to say your bench is totally better than prevous ones?

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
4

### Summary
This paper studies how to properly evaluate the LLM agent performance in complex and real-world scenarios. Authors proposes a novel benchmark, MCP-bench, which connects LLMs to 28 representative live MCP servers spanning 250 tools across domains. Specifically, tasks in MCP-BENCH test agents’ ability to retrieve relevant tools from fuzzy instructions, which is not adequately explored in previous benchmarks.

### Strengths
1. Effective LLM agent evaluation is a critical problem and has a profound impact in current LLM research.
2. MCP-bench enables the construction of authentic, multi-step tasks with rich input–output coupling.
3. This paper illustrates the technical developments in clear language. The presentation is good.

### Weaknesses
1. By default, MCP-bench employs LLM judge with o4-mini to evaluate models including o3 which may introduce evaluation bias and makes the comparison results less grounded.
2. The top LLMs have already reached near 100% success rates in all rule-based metrics (table3-5), which suggests that this measure may have already saturated and is not very informative in discriminating top LLMs.
3. The overall score is a composite factor averaging over multiple dimensions. It is under-discussed that how sensitive MCP-bench is to different normalization methods such as weighted sum.

### Questions
1. Since LLM judge serves as a critical role in constructing MCP-bench, it is intriguing to see if authors have tested different LLMs & prompts and what is the rationale behind the current choice.

### Soundness
3

### Presentation
3

### Contribution
3
