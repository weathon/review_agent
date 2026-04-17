# Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advances in LLM-based multi-agent systems (MAS) show that workflows composed of multiple LLM agents with distinct roles, tools, and communication patterns can outperform single-LLM baselines on complex tasks. However, most frameworks are homogeneous, where all agents share the same base LLM and differ only in prompts, tools, and positions in the workflow. This raises the question of whether such workflows can be simulated by a single agent through multi-turn conversations. We investigate this across seven benchmarks spanning coding, mathematics, general question answering, domain-specific reasoning, and real-world planning and tool use. Our results show that a single agent can reach the performance of homogeneous workflows with an efficiency advantage from KV cache reuse, and can even outperform an automatically optimized heterogeneous workflow. Building on this finding, we propose $\textbf{OneFlow}$, an algorithm that automatically tailors workflows for single-agent execution, reducing inference costs compared to existing automatic multi-agent design frameworks without trading off accuracy. These results position the single-LLM implementation of multi-agent workflows as a strong baseline for MAS research. We also note that single-LLM methods cannot capture heterogeneous workflows due to the lack of KV cache sharing across different LLMs, highlighting future opportunities in developing $\textit{truly}$ heterogeneous multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper examines the value proposition of LLM-based multi-agent systems in settings where current frameworks are largely homogeneous. The authors empirically show that a single LLM using multi-turn conversation with KV cache reuse can match or outperform such multi-agent workflows in both performance and cost across six benchmarks spanning coding, mathematics, and question answering. Building on this finding, they propose OneFlow, an algorithm for automatic, cost-aware workflow optimization tailored for single-agent execution without compromising accuracy.

### Strengths
1. The paper offers a reassessment of the prevailing practice of homogeneous multi-agent workflows in LLM systems, combining theoretical reasoning with strong empirical evidence. This provides an important sanity check for the rapidly growing MAS literature.

2. Experiments on six standard and one domain-specific dataset using multiple LLMs convincingly show that single-agent execution can match or surpass homogeneous multi-agent performance while reducing cost substantially.

### Weaknesses
1. The heterogeneous experiments (Table 3) rely on automatically generated workflows with unclear tuning and no ablation of model-assignment policies. Thus, the claim that a single-LLM implementation can outperform heterogeneous setups is only provisional.

2. The OneFlow search process is fixed and shallow, with no analysis of sensitivity to search depth, hyperparameters, or model choice. This leaves the robustness of the optimization procedure underexplored.

3. While quantitative results are comprehensive, there is little discussion of failure cases or qualitative differences between single-agent and true multi-agent behaviors, leaving interpretability and diagnostic insight limited.

### Questions
1. How does OneFlow’s workflow quality and cost-performance trade-off scale with deeper or wider search? Is the dual meta-LLM architecture robust to prompt or model changes?

2. Have the reported KV-cache efficiency gains been validated using open-weight models that support cache reuse, or might the simulated API-based estimates introduce systematic bias?

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
The paper shows that homogeneous MAS workflows (same base LLM, different prompts/tools) can be simulated by a single LLM via multi-turn role-play. Then it proposes OneFlow, which consists of two parts: 1. search for optimized workflow. 2. perform single LLM implementation. across six benchmarks, single-agent execution often matches or slightly exceeds multi-agent performance but the price is much cheaper.

### Strengths
* The paper proposes an interesting point of view.
* The experiments test on six benchmark and report both accuracy and cost to support claims.

### Weaknesses
* The OneFlow methods composes of two parts: search for optimized workflow and perform single LLM implementation. The first part seems like an improved version of Aflow and lacks novelty, for example, the critic prompt is adopted from AFlow.
* The costs for single-agent are simulated due to closed-weight APIs; add open-weight runs (or vendor KV-sharing APIs) to validate real-world latency/$ savings
* While the method mentions tool calling, the benchmark tested are static QA/math/code; include tool-use tasks with external side-effects and interactive settings.

### Questions
* See weakness.
* Clarification: In 4.2 single-LLM simulator, it writes "Set the system message to p_{i_t}", does this mean to replace the system prompt at the beginning? Can you give an example of how this is different from multi-agent system?

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
4

### Summary
This paper investigates whether the advantages of multi-agent systems built from homogeneous LLMs can be replicated by a single LLM through multi-turn interactions and KV-cache sharing. The authors empirically evaluate this hypothesis across six benchmarks (code generation, mathematics, and QA tasks) and introduce OneFlow, an automated workflow design algorithm that employs dual meta-LLMs (Designer and Critic) under an MCTS framework. The results suggest that single-agent implementations can match or exceed the performance of multi-agent workflows while substantially reducing inference cost. The paper further discusses the limits of this equivalence in heterogeneous multi-agent contexts and proposes directions for future research.

### Strengths
- **S1.** The paper tackles a timely and important issue whether multi-agent systems provide real advantages over single-agent reasoning when the base LLM is homogeneous.

- **S2.** Well-explained theoretical formulation that logically connects shared KV cache to computational efficiency.

- **S3.** Comprehensive experimental coverage across six benchmarks and one domain-specific dataset.

### Weaknesses
- **W1.** The OneFlow framework largely replicates the AFlow architecture with minor adaptations. The use of MCTS for workflow generation is not new, and the manuscript does not clearly articulate what conceptual or technical innovation distinguishes OneFlow from AFlow.

- **W2.** The evaluation primarily relies on closed-weight models (GPT-4o-mini, Claude 3.5 Haiku), and the KV-cache advantages are simulated rather than directly measured. The real experiments using open models capable of genuine KV sharing are absent, limiting the credibility of efficiency claims.

- **W3.** The paper primarily contrasts with AFlow and manual CoT baselines, omitting recent heterogeneous agentic frameworks (e.g., MasRouter) that could reveal where single-agent designs fail.

- **W4.** No exploration of when and why the single-agent execution begins to fail (e.g., under longer reasoning chains or tool dependencies).

### Questions
- **Q1.** Could the authors provide concrete evidence (with open-weight models) that KV-cache reuse yields measurable cost savings in practice rather than theoretical simulation?

- **Q2.** How does OneFlow's Designer-Critic interaction differ algorithmically from AFlow's meta-LLM setup beyond re-using prompts?

- **Q3.** Several datasets (e.g., GSM8K, MBPP) are solvable via direct prompting. Have the authors tested tasks that genuinely require multi-stage *agentic* reasoning or tool usage?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper questions whether multi-agent LLM workflows truly outperform single LLMs when all agents share the same base model.
It formally shows that for homogeneous workflows (same base LLM, different prompts/tools), a single LLM can simulate the entire multi-agent pipeline through multi-turn dialogue while reusing the KV cache, gaining efficiency without loss of expressivity.
Based on this, the authors propose OneFlow, an automatic workflow design framework using dual meta-LLMs (Designer + Critic) and Monte-Carlo Tree Search to generate workflows optimized for single-agent execution.
Across six benchmarks (HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP) and one domain-specific Shopping-MMLU set, OneFlow-single achieves comparable or better performance than existing multi-agent frameworks (AFlow, etc.) while cutting inference cost by up to 10×.
The paper concludes that homogeneous MAS can be largely simulated by a single agent and that future work should focus on truly heterogeneous systems.

### Strengths
The author Reframes multi-agent research with a rigorous single-agent equivalence argument.

they provide Six general benchmarks + domain-specific tasks.

they also Quantifies KV-cache benefits clearly.

it shows that OneFlow’s dual-meta LLM + MCTS is a creative and reproducible design. and it  clearly delineates where single-agent simulation applies and where heterogeneity still matters.

### Weaknesses
Limited empirical heterogeneity analysis: Pilot study is small; results inconclusive about real multi-model synergy.

Simulation of KV cache: Since APIs hide internal caching, efficiency results are theoretical. A small open-weight replication (e.g., LLaMA-3 8B) would strengthen credibility.

Ablations: Lack of ablation on MCTS parameters (α, β, iterations) and meta-LLM roles; unclear how much each contributes.

Over-dependence on closed models: Limits reproducibility beyond cost estimation.

Writing could be tighter: Some redundant explanations and long prompts in appendix.

### Questions
Can you verify KV-cache reuse gains empirically using an open-source model?

How sensitive are results to the α/β weights in Eq. (1)?

Would OneFlow still outperform AFlow if inference cost were excluded (i.e., pure accuracy metric)?

Have you tested whether role-switching (different prompts within same chat) affects coherence or context interference?

How does OneFlow perform when the base model has small context windows (e.g., 4k tokens)—does summarization degrade accuracy?

### Soundness
2

### Presentation
3

### Contribution
2
