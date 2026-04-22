# OrchestrationBench: LLM-Driven Agentic Planning and Tool Use in Multi-Domain Scenarios

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4, 8

## Abstract
Recent progress in Large Language Models (LLMs) has transformed them from text generators into agentic systems capable of multi-step reasoning, structured planning, and tool use. However, existing benchmarks inadequately capture their ability to orchestrate complex workflows across multiple domains under realistic constraints. To address this, we propose OrchestrationBench, a bilingual (English/Korean) benchmark that systematically evaluates (1) workflow-based planning and (2) constraint-aware tool execution. OrchestrationBench spans 17 representative domains with nearly 100 realistic virtual tools, covering scenarios that require sequential/parallel planning and compliance with business constraints. Unlike previous work, it explicitly disentangles planning evaluation from tool execution evaluation, which assesses tool selection, argument extraction, validation, and rejection handling. Constructed entirely through manual annotation with cultural adaptation, the benchmark ensures authenticity, diversity, and freedom from model-specific biases. Extensive experiments across state-of-the-art models show that function calling performance is relatively consistent, whereas planning capabilities exhibit substantial variation across models, emphasizing the need for structured planning evaluation. As a living benchmark, OrchestrationBench is designed to expand toward new domains, tools, and integration enabling rigorous, cross-cultural, and service-ready evaluation of LLM orchestration capabilities. The benchmark is publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- introuces a (offline) benchmark for orchestration and tool use, both korean and english
- manually constructed dataset of 17 representative domains and nearly 100 virtual tools
- define formal evaluation scores based on graph edit distance F1 score for tool execution accuracy
- run evaluation for multile models, including GPT-4/5, Claude, Gemini, Qwen

### Strengths
- well motivated and important proble. bencmarking for more complex and reaslistic settings is a key gap
- clear method formulation with separation into tool use and orchestration
- comprehensive emperical study with relevant (commercial), state of-the-art models
- practical significance, and splitting into tool use and orchestration leas to more actionable insights

### Weaknesses
- Offline evaluation only. while they have a manually designed gold workflow as reference for orchestration, I can imagine many cases where different workflows lead to similarly good outcomes. The offline evaluation particularly for orchestration seems limiting and is penalizing equally good workflows (I'm less concerned about the tool use aspect). Having an online environment would be deisrable (see e.g. OfficeBench, which clearly should be cited as related work)
- No clear taxonomy of workflows that are covered. I would love to see a clear overview/taxonomy/figure which type of workflows are included in the benchmark. It is hard to understand the empasis. Is it consumer user workflows like the travel booking example, business/enterprise workflows, what domains, ...?
- while relevant, it's limited novelty, and the paper misses to cite key references such as OfficeBench, WebArena, OSWorld, and ToolEmu, which already test interactive or multi-application orchestration. A clearer discussion of how OrchestrationBench differs from these would help position its unique contribution.
- Limited dataset scale. Although the dataset covers 17 domains, the total size (≈220 sessions per language, ≈700 tool calls) is relatively small for evaluating general orchestration capabilities. This limits statistical robustness. larger or semi-synthetic expansion would make the benchmark more representative and useful for training-time evaluation. I'm not convinced about the usefulness in it's current state.

### Questions
1. Workflow diversity and taxonomy: Could you provide a clearer taxonomy or summary table of workflows? It would help readers understand what orchestration capabilities are actually being tested.
2. Alternative valid workflows:  How does the benchmark handle cases where a model produces a different but functionally valid workflow? Is there any provision for multi-reference or semantic equivalence scoring beyond Graph Edit Distance?
3. Dataset scale and representativeness:  Given that the dataset includes roughly 200 sessions per language, do you consider this sufficient to generalize across 17 domains? Are there plans to expand or semi-automate data generation to increase coverage?
4: Offline vs online evaluation – Have you considered integrating OrchestrationBench with an interactive or simulated execution environment (e.g., MCP or API sandbox) to test dynamic replanning and feedback-based adaptation?
5. Extensibility in practice: The paper describes OrchestrationBench as a “living benchmark.”. Wat does this mean in practice?

### Soundness
3

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
This paper propose an multilingual (English/Korean) benchmark to  evaluate LLMs’ workflow-based planning and constraint-aware tool execution across 17 domains with ~100 virtual tools. It differs from prior work by separating planning and tool execution (assessing selection, argument handling, etc.), uses manually annotated, culturally adapted content to avoid model biases. Experiments on SOTA models show consistent function calling but varied planning capabilities—highlighting the need for structured planning evaluation.

### Strengths
*   The paper targets the "evaluation of LLM agent planning and tool use," addressing a key gap in the current field.&#x20;
*   Through experimental analysis, the paper draws important conclusions. For instance, it  identifies Gemini as the model with the strongest planning capability. Such findings can directly guide real-world applications.

### Weaknesses
*   Validation relies solely on GPT-4.1 as judge, which may introduce implicit biases toward the outputs of specific models, leading to deviations in evaluation results. It would be better to  calculate the consistency between multiple judges (Claude-sonnet-4, and human annotation) to enhance the reliability of assessments.
*   The current benchmark uses virtual tools and does not integrate real enterprise APIs. It cannot simulate API-related constraints (e.g., error feedback) and dynamic environments in real scenarios.

### Questions
please refer to weakness

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes OrchestrationBench, a multilingual benchmark that systematically evaluate workflow-based planning and constraint-aware execution. OrchestrationBench spans 17 domains with roughly 100 tools and is constructed through manual and independent efforts. Authors also provide evaluation results for the mainstream open & closed LLMs on the proposed OrchestrationBench.

### Strengths
1. This paper studies how to properly evaluate LLM performance in complex and dynamic real-world deployment scenarios, which is a critical topic in current research.
2. OrchestrationBench is manually constructed by human experts and covers 17 domains with around 100 tools.
3. This paper properly illustrate and motivate the ideas with sufficient examples. The writing is clear.

### Weaknesses
1. As a multilingual dataset, OrchestrationBench only has examples from English and Korean, which seems limited.
2. In the evaluation, authors employs LLM judge with GPT-4.1 to evaluate selected models including GPT-4.1 and other variants, which introduces evaluation bias and makes the evaluation results less convincing.
3. Another observation is that the best model performance achieved on OrchestrationBench has already approached to 90%. One concern is how quickly this benchmark will be saturated and how long it can serve to guide the development of the next generation’s LLMs.
4. Some critical system designs lack sufficient justification. For example, authors mention “weight selection errors more heavily (0.8) than status errors (0.2)” without any ablation results to justify the choice which leads to concerns about how robust the OrchestrationBench evaluation results are to this config.

### Questions
1. Authors mention that “to ensure quality, multiple rounds of review and cross-checking were conducted, with inter-annotator agreement maintained at a consistently high level”. What are the statistics? For example, how many rounds and what is the average inter-annotator agreement level?

### Soundness
3

### Presentation
3

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
This paper introduces OrchestrationBench, a multilingual (English/Korean) benchmark focused on evaluating large language models (LLMs) as orchestrators in real-world multi-domain service scenarios. It manually constructs 17 service domains and nearly 100 virtual tools, explicitly separating “workflow planning” (represented as DAGs) from “constrained tool execution” (including call/reject decisions, parameter extraction, and decoupled validation/constraint checking). It employs metrics such as Graph Edit Distance (1-GED) and conducts experiments across numerous mainstream models, providing detailed experimental results and analysis.

### Strengths
1. **Clear separation of planning and execution**: By decoupling “planning” from “function calls/constraint verification”, this approach enables more granular evaluation than “end-to-end” configurations. This facilitates pinpointing specific model deficiencies, proving highly valuable for driving subsequent methodological improvements.
2. **Multi-domain and cross-language dataset**: OrchestrationBench manually annotated nearly 100 tools across 17 domains, covering a broad range of real-world service areas. This makes it highly relevant to consumer and enterprise-level LLM deployments, positioning it as a practical benchmark. It includes both English and Korean cultural contexts, filling a gap in Korean-language benchmarking.
3. **Comprehensive model and metric evaluation**: Experiments were conducted across numerous closed-source and open-source models, incorporating precise, interpretable metrics such as 1-GED for workflow planning similarity, multiple F1 variants for tool execution, and consistency checks between human and LLM evaluators (Cohen's Kappa 0.63). Experimental tables clearly present results and support the paper's conclusions.
4. **Insightful empirical findings**: As shown in Tables 3 and 4, empirical results reveal significant gaps between different models' tool execution and planning capabilities. This demonstrates models' deficiencies in planning, confirming the benchmark's structured evaluation as a crucial discovery.

### Weaknesses
1. **Lack of related work comparison**: The paper fails to provide explicit comparisons with similar benchmark studies, making it difficult to position OrchestrationBench within the broader field and assess its contributions. For instance, REALM-Bench focuses on intensive planning in real-world domains, and ThinkGeo incorporates planning and tool evaluation. The paper lacks direct discussion and comparison with these works (not limited to the mentioned above). Lack of a comparative table highlighting uniqueness and contributions.
2. **Main contributions lack focus**: The paper divides contributions into five items, lacking a central thread. Some ideas (e.g., separating planning and execution in ThinkGeo) have precedents, and the proposed GED metric does not sufficiently justify its uniqueness and design rationale.
3. **Writing emphasis imbalance**: Substantial text is devoted to textual case descriptions (case study diagrams are recommended for clarification), while critical details like benchmark construction (Section 3.3) are underrepresented. For instance, Appendix E mentions using large models for data synthesis, yet Section 3.3 omits how these models were applied in dataset construction.
4. **Lack of supporting diagrams**: Visual diagrams to aid understanding of the benchmark architecture, data flow, and evaluation process are absent. Existing figures are overloaded with text, failing to enhance comprehension and resulting in poor readability.
5. **Insufficient explanation and analysis**: The authors use 1-GED to quantify workflow similarity but do not clarify the correlation between GED and real-world task success rates. Similarly, Section 4.1 lacks explanation and quantitative analysis for weight settings like selection errors = 0.8 and status errors = 0.2.
6. **Overall logical organization is disorganized**: Section transitions are loose, and the benchmark architecture, construction, and evaluation fail to form a coherent narrative chain, undermining readability and credibility.

### Questions
1. How is the DAG structure constructed during the planning phase? How are parallel subtasks or multi-solution scenarios handled?
2. What do the two graphs in GEM computation refer to, and how are they obtained?
3. How is the “manually constructed & multi-round review” dataset specifically implemented? For example, how is broad applicability across model architectures and deployment contexts ensured? How is scenario design refined to cover a wide range of user requests? Has a reference methodology been established?
4. What tasks are handled by human annotators versus LLMs in data construction? How is model contamination avoided?
5. What does “living benchmark” mean in Contribution (5)?
6. Compared to existing tool-use or llm orchestration benchmarks, what makes OrchestrationBench unique?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a bilingual (English/Korean) benchmark designed to rigorously evaluate large language models (LLMs) as agentic systems capable of orchestrating complex, multi-step workflows across 17 domains and nearly 100 virtual tools under realistic constraints. The benchmark separately evaluates workflow planning and tool calling. The dataset is manually authored and cross-checked to ensure cultural authenticity and avoid biases that can arise from synthetic (LLM-generated) data. Experiments reveal insights like a) relatively weaker performance in function calling than in planning and b) higher variance in planning abilities across models.

### Strengths
1. By explicitly disentangling the evaluation of planning (task orchestration) from tool execution (function calling and validation), the benchmark allows for targeted diagnosis of LLM strengths and weaknesses, which is what helped reveal the aforementioned insights.

2. The manual annotation of the dataset and rigorous cross-checking will ensure that the data is free from hallucinations and biases that can arise from LLM generated data.

### Weaknesses
1. The benchmarks is currently limited to only 2 languages so it can't really claim to be multilingual and handling the complexities/variations across multiple languages at the moment.

2. Often there can be multiple ways (workflows) for getting to the same answer. It is not clear to me if the benchmark considers that in evaluating LLM responses or if it insists on the responses matching that in the dataset.

3. It is also not clear to me if the benchmark evaluates the latency of workflows - for e.g. Does the LLM call a slow tool when a fast one would do? Does it make any unnecessary tool calls?

### Questions
1. It seems that the benchmarks assumes that there will be sub-LLMs called for executing the individual tasks. Will it make any difference if the main LLM also calls the tools and executes the tasks, which is the case in many settings?

2. Why will higher Graph Edit distance be better (lines 353-354) when it is the \textit{minimum} edit operations needed to transform one graph into another?

3. The results are not represented well - the table has too many numbers and is a bit difficult to parse. I would recommend adding a plot with the results from representative LLMs to illustrate the main insights.

### Soundness
3

### Presentation
3

### Contribution
3
