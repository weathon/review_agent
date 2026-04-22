# MCP-Radar: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
As Large Language Models (LLMs) evolve from passive text generators to active reasoning agents capable of interacting with external tools, the Model Context Protocol (MCP) has emerged as a key standardized framework for dynamic tool discovery and orchestration. Despite its widespread industry adoption, existing evaluation methods do not adequately assess tool utilization capabilities under this new paradigm.
To address this gap, this paper introduces MCP-Radar, the first comprehensive benchmark specifically designed to evaluate LLM performance within the MCP framework. MCP-Radar features a challenging dataset of 507 tasks spanning six domains: mathematical reasoning, web search, email, calendar, file management, and terminal operations. It quantifies performance based on two primary criteria: answer correctness and operational accuracy. To closely emulate real-world usage, our evaluation employs both authentic MCP tools and high-fidelity simulations of official tools.
Unlike traditional benchmarks that rely on subjective human evaluation or binary success metrics, MCP-Radar adopts objective, quantifiable measurements across multiple task domains, including computational resource efficiency and the number of successful tool-invocation rounds. Our evaluation of leading closed-source and open-source LLMs reveals distinct capability profiles and highlights a significant trade-off between accuracy and efficiency.
Our findings provide actionable insights for both LLM developers and tool creators, establishing a standardized methodology applicable to the broader LLM agent ecosystem. All implementations, configurations, and datasets are publicly available at \url{https://anonymous.4open.science/r/MCPRadar-B143}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MCP-Radar, a benchmark for evaluating the tool-use capabilities of Large Language Models within the Model Context Protocol (MCP) framework. It features a diverse set of 507 tasks across six domains, categorized into "Precise Answer" and "Fuzzy Match" objectives. The benchmark employs a combination of real-world and high-fidelity mock tools to create a realistic testing environment. The authors propose a multi-dimensional evaluation framework assessing accuracy, efficiency, and step-by-step success. A key finding is that models often struggle with semantically correct tool selection despite being syntactically proficient at invocation, providing useful insights for future development.

### Strengths
1. The paper reveals critical, actionable issues in how LLMs interact with tools, such as the tendency to select a "semantically plausible but functionally incorrect tool." This raises important questions for the community, for instance, whether tool metadata is sufficient for LLMs to interpret correctly, especially since models cannot pre-train on the functionality of every possible MCP. The analysis provides useful, concrete insights for improving MCP design, such as recommending atomic tool design and optimized descriptions.

2. The construction of the benchmark is intuitive. The use of both real and mock tools creates a high-fidelity yet controlled evaluation setting. The distinction between "Precise Answer" and "Fuzzy Match" tasks effectively captures different facets of tool-use capability. The data curation process, which filters out problems solvable without tools, successfully isolates and tests genuine tool-use necessity.

3. The writing is generally clear and well-structured. The methodology and results easy to follow.

### Weaknesses
1. The paper's claim to be the "first comprehensive benchmark" for evaluating LLMs in the MCP paradigm is overstated, as several precedent works exist. For example, "MCP-AgentBench" (Guo et al., arXiv:2509.09734) and the concurrently developed "MCP-Universe" (Luo et al., 2025) and "MCPEval" (Liu et al., 2025) presented in the paper's own related work section, directly tackle similar problems of MCP-based agent evaluation. While MCP-Radar is authentic and useful, its incremental novelty over these recent benchmarks may not be sufficient to stand out in the rapidly growing landscape of agent evaluation works.

2. Though the benchmark is well-executed and will be valuable for model training and evaluation, the core concept of a new tool-use benchmark, even within the specific MCP framework, no longer feels novel or exciting enough for a top-tier conference to me, given the booming quantity of such works. I am left uncertain whether this specific contribution meets the bar for acceptance.


**Minor Issues**

1. Many citations are formatted inconsistently. For instance, line 39 uses "APIs (Chowdhery et al., 2022; Brown et al., 2020)" while line 45 uses "knowledge-based reasoning (Hendrycks et al.) (Zhong et al.)". There are also formatting errors, such as in line 103: "API (Liu et al., 2024) (Song et al., 2023)..." should use a single set of parentheses.

2. Figure 4 (Model Performance Comparison) is very unclear and difficult to interpret in its current form.

3. Typo: Line 421: "Other Types this type"

### Questions
1. Given the existence of "MCP-Universe," "MCPEval," and "MCP-AgentBench," how does MCP-Radar provide a fundamentally unique or more comprehensive evaluation that justifies its "first" claim and distinguishes it significantly from these precedents?

2. The finding that models struggle with semantic tool selection is crucial. Did your analysis offer any specific, data-backed recommendations for what constitutes a "good" vs. "bad" tool description that could help MCP developers mitigate this issue?

3. The ablation study on interaction rounds (K) is helpful. Were any experiments conducted concerning the impact of different tool description styles or detail levels on model performance, especially in light of the observed "Inaccurate Tool Invocation" errors?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a benchmark to evaluate LLMs’ tool-use capability within the MCP. The benchmark covers 507 tasks across six domains and introduces a set of metrics. The authors evaluate 10 closed-source and open-source LLMs and provide an analysis of tool-use errors, reasoning errors, and efficiency trade-offs.

### Strengths
1. This paper evaluates the tool-use capability of LLM Agents with MCP. 
2. The detailed analysis of errors (tool-use errors, reasoning errors, information synthesis errors) provides useful insights into current LLM Agent limitations.

### Weaknesses
Despite being positioned as an MCP-focused benchmark, MCP-RADAR's overall structure largely resembles that of prior tool-use evaluation benchmarks.

### Questions
1. What is the core novelty of MCP-RADAR compared to other benchmarks?
2. Real MCP usage involves tools being added, removed, or replaced dynamically. Can MCP-RADAR evaluate LLMs under dynamic or expanding tool sets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
MCP-RADAR is a benchmark for evaluating LLMs’ tool-use within the Model Context Protocol, introducing 507 tasks across six domains. It measures both answer correctness and operational accuracy using real MCP tools and high-fidelity simulations, with objective metrics like resource efficiency and successful invocation rounds. Evaluations of leading open- and closed-source LLMs reveal distinct capability profiles and an accuracy–efficiency trade-off, and the artifacts are publicly available.

The main contribution of this paper is a comprehensive MCP-RADAR benchmark with two task types across six domains, a high-fidelity evaluation testbed using real and spec-accurate mock MCP tools, and an objective multi-metric framework measuring accuracy, efficiency, and resourcefulness.

### Strengths
The strengths of the paper can be summarized as,

- The paper’s code and data are open-sourced, which benefits community progress in this area and helps ensure the work’s credibility and reproducibility.

- Proposes a benchmark spanning six domains and two task types.

### Weaknesses
The weaknesses of the paper are listed as follow,

- Introduction Section:

    - Causal overclaim about MCP. The authors say the shift to tool-using agents was “significantly accelerated by the advent of the Model Context Protocol (MCP)” and that MCP is “a standardized framework for dynamic tool discovery and orchestration.” That’s disputable: tool use predates MCP (e.g., function calling, ReAct/Toolformer, agent frameworks), and MCP is one proposal among several—not an established, consensus “standard,” nor the main accelerator.

    - Overgeneralization about tool-centric evals. The authors claim they “rely on simulated environments,” but there are evaluations using real websites/APIs/browsers. The point (simulations miss real-world friction) is fair, but stated too absolutely.

    - “First comprehensive benchmark” is an overclaim. There are many tool-use/agent benchmarks already. Even if it is MCP-specific, calling it “first” and “comprehensive” needs careful qualification or evidence.

    - Web search ≠ single ground truth. The authors place “Websearch” under Precise Answer with a single correct value, but the web is non-static, multi-answer, and time-dependent—this undermines reproducibility and the stated task type.

    - Operation-matching may mis-score valid solutions. For “Fuzzy Match”, the authors require “a correct sequence of operations.” In many tools, multiple sequences are functionally equivalent (commutativity, idempotent ops, retries). Step-matching will falsely mark correct runs wrong unless the authors define state-equivalence.

    - Cross-model comparability of CRE is dubious. Closed-source APIs and local open-source models have very different latency/tokenization/rate-limit/hardware stacks. Without strict budget control and normalization, Computational Resource Efficiency comparisons are not apples-to-apples.

    - Real + mock tools confound conclusions. Mixing Smithery “real-world” MCPs with “high-fidelity mock MCPs” in one score can entangle environment quality with model performance. Results may reflect simulator fidelity, not the model.

    - Naming inconsistency. The authors alternate between “MCP-RADAR” and “Radar Bench” in the figure 1—could confuse readers about what is the benchmark vs the implementation component.

- Data Generation Section:

    - Arithmetic inconsistency. The authors state “507 instances,” but Table 1’s quantities sum to 515 (120+94+119+28+91+63). That undermines all downstream stats.
    
    - Self-contradictory definition of “Precise Answer.” It’s defined as a “single, definitive ground-truth value,” yet examples include algebraic expressions and SolveEquation—both admit many equivalent forms (and sometimes multiple solutions).
 
    - “Comprehensive” claim vs narrow coverage. Six domains with ~500 items is useful but not “comprehensive” for MCP agents.

    - Selection-by-one-model bias + non-reproducibility. Gating items with Gemini 2.5 Flash (“discard queries the model could solve without tools”) bakes one closed model’s strengths/weaknesses into the dataset and cannot be replicated or audited by others.

    - Construct invalidity: claims to test tool-use but scores tool-agnostic. The authors say the goal is to “specifically test tool-use,” yet for Precise Answer the authors only score the final answer and “do not assess the specific tools used.” A stronger model can bypass tools entirely—so the data do not measure the construct the authors claim.

    - Unverifiable, drifting web environment. Tying execution to a “duckduckgo-mcp-server” without specifying snapshotting, locale, time, rate-limit, or version pinning makes results non-deterministic and unreproducible across sites and dates.

    - Goal/tool mapping from source datasets is under-specified. The authors say items are “adapted from established datasets,” then executed via MCP tools. The pipeline that maps each source problem to which tool(s) and why (and how equivalence classes are formed) is not described—this is a core methodological gap.

    - Tiny, unrealistic world state. Pre-populating 100 emails / 50 calendar entries is far from real usage and encourages pattern learning (names, subjects, label sets), weakening claims of “realistic tool interaction.”

    - A “unique template per tool” with five variants may measure template recognition, not tool reasoning.

    - Arbitrary “top-3 tools” selection. “Most frequently used” is asserted but not grounded (logs? prior studies?). With mocks and synthetic tasks, “frequency” becomes circular and may bias chains toward your chosen endpoints.

- Experiments Section:

    - DTSR represents Tool Selection Efficiency in introduction section and denotes Dialogue Turn Success Rate in Experiments Section, which is confusing.

    - Ill-posed error taxonomy (overlap/ambiguity). “Tool-Use,” “Reasoning,” and “Information Synthesis” aren’t mutually exclusive (e.g., “tool omission,” “redundant tool invocation,” and “tool-result integration” can co-occur). Without an operational coding scheme and inter-rater agreement, counts are unreliable.

    - Figure 5 invites causal misinterpretation. “Impact of Dialogue Rounds on Accuracy” suggests causality, but there’s strong selection confounding (easy tasks finish in fewer rounds; harder ones allow/need more rounds; models differ in round budgets). No stratification, controls, or CIs are shown.

### Questions
Please refer to the Weaknesses box for details. In summary, the paper’s motivation for the dataset, its definition, construction, and experimental analysis all contain substantial flaws. The main claimed novelty—Fuzzy Match—reads more like a variant of binary matching and does not clearly distinguish its evaluation metrics from those of similar datasets. The data appear to be simply gathered from existing sources and reprocessed, so the workload is questionable. Moreover, possible bias relative to real-world distributions further weakens the dataset’s contribution.

### Soundness
2

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
5

### Summary
This paper introduces MCP-RADAR, a comprehensive benchmark designed for the MCP and tool usage. The benchmark features 507 tasks across six domains. An evaluation of ten leading LLMs revealed a critical weakness: models often select semantically plausible but functionally incorrect tools, indicating a superficial understanding of the task.

### Strengths
MCP usage is a highly meaningful evaluation direction. Authors have constructed a diverse and challenging benchmark that even advanced models perform poorly on it. Moreover, their analysis of model behavior provides substantial value for future research.

### Weaknesses
- The authors did not compare against a baseline that does not use tool calling, making it unreasonable to claim a causal relationship between model performance and MCP tool usage.  
- Some tasks have very few evaluation examples, like only 28 for calendar, which introduces instability in the evaluation. Repeated runs could yield significantly variances.
- Figure 4 and Table 3 are highly redundant and neither provides an average score to summarize each model’s overall performance across tasks.  
- During evaluation, the authors included all tools in the context. An ablation study that restricts the context to only task-relevant tools would be necessary to assess the impact of tool scope on performance.
- The main body part contains too little useful information. Detailed data construction pipeline and data sources are lacked.

### Questions
- How the CRE metric calculated?
- The authors used tools such as web search. How robust are these tools? Could they return different results across different invocations, thereby affecting reproducibility?

### Soundness
3

### Presentation
2

### Contribution
3
