# AgentRace: Benchmarking Efficiency in LLM Agent Frameworks

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Large Language Model (LLM) agents are rapidly gaining traction across domains such as intelligent assistants, programming aids, and autonomous decision systems. While existing benchmarks focus primarily on evaluating the effectiveness of LLM agents, such as task success rates and reasoning correctness, the efficiency of agent frameworks remains an underexplored but critical factor for real-world deployment. In this work, we introduce AgentRace, the first benchmark specifically designed to systematically evaluate the efficiency of LLM agent frameworks across representative workloads. AgentRace enables controlled, reproducible comparisons of runtime performance, scalability, communication overhead, and tool invocation latency across popular frameworks on diverse task scenarios and workflows. Our experiments reveal 14 insights and 15 underlying mechanisms for developing efficient LLM agents. We believe AgentRace will become a valuable resource for guiding the design and optimization of next-generation efficient LLM agent systems. The platform and results are available at the anonymous website https://agent-race.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents AgentRace, a benchmark-style evaluation framework for assessing the efficiency (time, token count, communication overhead) of various large-language-model based agent systems. It aggregates several existing agent frameworks and datasets, runs empirical experiments, and reports a set of insights about where inefficiencies arise in agent pipelines (e.g., prompt length growth, tool orchestration latency, communication bottlenecks). The benchmark can serve the community as a standard way to compare agent frameworks on efficiency.

### Strengths
- The authors bring together multiple agent frameworks and datasets, and set up a unified testing environment. This kind of integration is non-trivial and may be valuable as a reproducibility/system contribution.

- Focusing on efficiency (rather than just task performance or accuracy) is timely: as agent systems move toward production and deployment, latency, cost (token usage), and orchestration overhead become important.

### Weaknesses
- Limited methodological or conceptual novelty. The work does not propose a new algorithm, novel evaluation metric, or analytic model of efficiency. Instead it aggregates existing datasets and frameworks, and measures commonly used metrics (time, tokens, communication size). As such, the contribution is more “engineering benchmark” than “scientific advance.”

- Overlap with existing observability/agent-profiling tools. In industry there are many tools and platforms that provide observability and profiling for LLM-agent systems: for example, platforms tracking latency, token cost, tool call breakdowns, and more. (OpenTelemetry, LangFuse, datadog etc) This raises the question as what does this paper’s benchmark provide beyond what these existing tools or frameworks already do?

- The paper does not appear to consider system-level optimisation concerns, e.g., caching of tokens, optimized/common toolset for fair comparison etc

- The metrics measured (runtime, token count, communication size) are conventional, and many of the conclusions drawn (e.g., “longer prompts lead to greater cost”, “tool orchestration adds latency”) are intuitive or not surprising. Thus the contribution of “new insight” is limited.

### Questions
- You mention measuring efficiency across frameworks — can you clarify how your benchmark differs from or improves upon existing observability or profiling platforms (e.g., those tracking LLM agent behaviour, token cost, tool calls) in industry or research?

- How do you ensure fairness across agent frameworks in your experiments (for example tool implementations, prompt variants)? Could differences in configuration dominate your reported differences?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AgentRace, a benchmark platform specifically designed to evaluate the efficiency of LLM agent frameworks. This is presented as the first systematic benchmark focusing on efficiency, filling a gap left by prior benchmarks that mostly emphasized task success or reasoning correctness. The authors evaluate 7 different LLM agent frameworks on a diverse set of tasks and workflows, measuring metrics such as execution time, token usage, and communication overhead while ensuring functional correctness. Through comprehensive experiments, the paper uncovers previously undocumented sources of inefficiency in these agent frameworks and distills 9 key insights along with 12 underlying mechanisms that explain bottlenecks. In addition to identifying problems, the work provides actionable suggestions such as using selective memory summarization, prompt compression, and choosing more efficient architectures to improve agent performance. Importantly, the authors are releasing the AgentRace benchmark suite and all experimental results to the community, which will enable other researchers and practitioners to reproduce the findings and evaluate new frameworks on the same footing.

### Strengths
1. Addresses a Critical Gap: The paper tackles the under-explored dimension of efficiency in LLM agent frameworks. It is explicitly positioned as the first benchmark to systematically evaluate LLM agents’ efficiency, complementing prior work that mostly focused on accuracy or functionality.
2. Insightful Findings with Root-Cause Analysis: The paper doesn’t just report performance numbers; it delivers insights and digs into why certain frameworks behave inefficiently.
3. Actionable Guidance for Improvement: Alongside identifying issues, the paper offers concrete suggestions to improve efficiency.
4. Reproducibility and Open Resource: The authors emphasize reproducibility. The benchmark platform is implemented to run experiments with a single command, and they plan to open-source the entire AgentRace suite and share all results.
5. Clarity and Organization: Despite the technical nature of the benchmark, the paper is clearly written and well-structured.

### Weaknesses
1. Limited Theoretical Novelty: The paper’s primary contribution is an empirical benchmark and analysis, rather than a new algorithm or a novel model. While benchmarks are valuable (especially one addressing a clear need), some might argue that the work is more of a practical evaluation study than a conceptual breakthrough. The insights, although useful, stem from measurements and engineering analysis rather than from developing new learning methods or theoretical insights. The impact therefore hinges on how useful the community finds the benchmark and whether these efficiency issues spur further research.
2. Some Insights Are Expected or Confirmatory: A number of the findings, in retrospect, align with intuition. For instance, it’s not surprising that calling a large LLM (like GPT-4) dominates the runtime in any agent workflow – the paper indeed confirms that LLM inference is the primary bottleneck in all frameworks. Similarly, it’s known that adding lots of unnecessary text to prompts will slow things down and increase token costs, which the paper identifies in certain frameworks. In summary, some insights confirm known best practices (like “don’t use overly verbose prompts” or “avoid deep abstraction stacks when speed matters”). This doesn’t negate their value – documenting and quantifying these issues is important – but it means the surprise factor or conceptual novelty of some insights is limited.
3. Possible Implementation Biases: The authors strove for fairness by implementing missing functionalities themselves for each framework. While this is commendable and largely necessary, it also introduces some risk: the performance of a framework in the benchmark might be influenced by the authors’ custom implementation of a tool or workflow for that framework. For example, if Framework A didn’t originally support a certain tool and the authors added it in Python with minimal optimization, Framework A might appear slower partly due to that implementation rather than its core design. It’s a tricky issue – not a clear flaw. The authors could perhaps discuss this source of uncertainty more, but it remains a caveat of doing a multi-framework comparison.

### Questions
1. You mention that ~50% of the benchmark functionalities were custom-implemented to ensure all frameworks support all tasks. Could you elaborate on how you maintained fairness and consistency in these implementations? For instance, when a framework lacked a certain tool (say a PDF reader or search), did you implement it in a minimal way in that framework’s style? How do we know that any performance differences observed aren’t due to an inadvertently less optimized implementation on one framework versus another? 
2. Your default experiments use GPT-4 (the “GPT-4o” API) as the core LLM for all agents, and you note additional experiments with other models in Appendix B.5. Did those additional experiments reveal any meaningful differences in the efficiency insights? For example, when using a smaller model like LLaMA or an open-source model, do the same frameworks still show the same relative inefficiencies (e.g., the same ones are slower or consume more tokens)? I’m curious if any frameworks are particularly optimized for certain model types or sizes. In short, to what extent do you think the findings are model-agnostic versus tied to using a very large model like GPT-4?
3. The agent ecosystem is rapidly evolving. How do you envision AgentRace staying up-to-date? For example, if LangChain or other frameworks introduce significant efficiency improvements next month, is the idea that the benchmark repository will incorporate those changes for future users? Also, might you include newly released frameworks (or future versions of the current ones) in subsequent evaluations? Essentially, I’m asking if AgentRace is intended as a one-off study or a continuously maintained benchmark suite. A related question: have you considered an easy way for external contributors to add their own framework to AgentRace for evaluation (which would be great for community adoption)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AgentRace, a benchmark platform specifically designed to evaluate the efficiency of LLM agent frameworks. AgentRace provides a modular, extensible system for controlled and reproducible comparisons of runtime, communication, and tool latency across 7 frameworks, 3 workflows and 5 datasets. Through systematic experiments, the authors identify insights and underlying mechanisms that cause inefficiencies and propose corresponding optimization suggestions.

### Strengths
- Novel focus: Shifts benchmarking from “how correct” to “how efficient,” addressing a real deployment bottleneck;
- Systematic design: Modular architecture covering frameworks, workflows, datasets, and metrics;
- Comprehensive analysis: Covers diverse efficiency dimensions (inference, tool latency, communication, scalability);

### Weaknesses
- Task coverage: The tested agent tasks (GAIA, HumanEval, MMLU) may not sufficiently stress long-horizon reasoning or multi-step planning. Including multi-turn, dynamic environments (e.g., ScienceWorld, WebArena) would strengthen claims about scalability and real-world efficiency;
- Variable control: It’s unclear whether comparisons are made on successful or failed cases. Efficiency should ideally be measured conditional on correctness to ensure fairness;
- Correlation analysis missing: The paper qualitatively observes token–accuracy or token–time relationships but doesn’t quantify them. Computing a Pearson correlation coefficient would make the relationship concrete;
- Related work: There are some papers focusing on the efficiency of agents [1,2], which the authors should discuss in Related Work section;

[1] Efficient Agents: Building Effective Agents While Reducing Cost. https://arxiv.org/pdf/2508.02694

[2] OPTIMA: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System. https://arxiv.org/pdf/2410.08115

### Questions
See the Weaknesses part.

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
This paper introduces AgentRace, a novel benchmark for evaluating the efficiency (runtime, cost, communication) of LLM agent frameworks, arguing existing benchmarks focus too heavily on effectiveness. The modular platform tests 7 frameworks (e.g., LangChain, AutoGen) across 3 workflows (ReAct, RAG, MoA) and 5 datasets, measuring execution time, token use, communication size, and accuracy. The authors identify key efficiency bottlenecks, such as LLM inference (worsened by verbose prompts), varied tool latency, overlooked RAG retrieval costs, and framework-dependent scalability in multi-agent setups.

### Strengths
Important and Timely Problem: The paper addresses a critical and underexplored gap in LLM agent evaluation. As agents move from research prototypes to production systems, efficiency (latency, cost, and scalability) becomes a first-order concern. This work provides the first comprehensive, public benchmark to systematically study this.

Comprehensive and Well-Designed Benchmark: The AgentRace platform is a significant contribution. Its modular design (Data, Agent, Framework, Analysis) is sound and extensible. The coverage of 7 major frameworks and 3 core agent workflows (ReAct, RAG, MoA) allows for a broad and relevant comparison. The use of a unified instrumentation layer (tracers and loggers) is a good methodological choice for ensuring fair comparisons.

Actionable Insights: The paper does not just present leaderboards; it delivers concrete, actionable insights for developers. By distinguishing between high-level insights and specific underlying mechanisms, the authors provide a clear diagnosis of why certain frameworks are inefficient and offer clear targets for optimization.

### Weaknesses
De-prioritization of Task Accuracy: The paper's primary focus is efficiency, and it explicitly states that accuracy is not the main focus (Appendix B.1). However, the reported accuracy on complex tasks like GAIA is very low (e.g., 0.1-0.2 range in Table 5). This raises questions about the meaningfulness of the efficiency metrics. For example, is a framework's low latency on a task simply because it failed quickly? The paper would be stronger if it explored the relationship between efficiency and correctness more deeply (e.g., does high token consumption correlate with more attempts to solve the problem, even if it ultimately fails?).

Limited Model Diversity: The vast majority of the paper's insights are derived from experiments using a relatively small model pool. Efficiency metrics like latency, cost, and even tool-use behavior are highly dependent on the underlying model. The insights drawn from a powerful, closed-source API model may not generalize to smaller, open-source models that might be run locally. The paper includes a small experiment with Claude-3 (Appendix B.5) but only on two frameworks and one dataset, which is insufficient to address this.

Confounding Variables from Framework-Specific Implementations: The "Bugs and Features" section (Appendix D) is very insightful but also highlights that many of the observed efficiency differences may stem from implementation-specific "bugs" or design choices rather than the inherent architecture of the framework itself. For example, CrewAI failing to call all agents (D.4), LlamaIndex failing on tool calls (D.5), or AgentScope's tools calling OpenAI (D.3). This makes a true "apples-to-apples" comparison of framework overhead difficult. The benchmark is, in practice, evaluating the authors' specific implementation of these workflows on top of the frameworks as much as the frameworks themselves.

### Questions
Could the authors further elaborate on the relationship between efficiency and accuracy?

The insights are based on a powerful (and relatively slow) gpt-4o model. How would the primary bottlenecks (Insight 1) shift when using smaller, open-source, or locally-hosted models?

### Soundness
2

### Presentation
3

### Contribution
2
