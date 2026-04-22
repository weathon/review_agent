# TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Large language model (LLM)-based agents increasingly rely on tool use to complete real-world tasks. While existing works evaluate the LLMs' tool use capability, they largely focus on the final answers yet overlook the detailed tool usage trajectory, i.e., whether tools are selected, parameterized, and ordered correctly. We introduce TRAJECT-Bench, a trajectory-aware benchmark to comprehensively evaluate LLMs' tool use capability through diverse tasks with fine-grained evaluation metrics. TRAJECT-Bench pairs high-fidelity, executable tools across practical domains with tasks grounded in production-style APIs, and synthesizes trajectories that vary in breadth (parallel calls) and depth (interdependent chains). Besides final accuracy, TRAJECT-Bench also reports trajectory-level diagnostics, including tool selection and argument correctness, and dependency/order satisfaction. Analyses reveal failure modes such as similar tool confusion and parameter-blind selection, and scaling behavior with tool diversity and trajectory length where the bottleneck of transiting from short to mid-length trajectories is revealed, offering actionable guidance for LLMs' tool use.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TRAJECT-Bench, a new benchmark for evaluating Agentic LLM tool use.  It evaluates how LLMs select, parameterize, and order tools across executable APIs. Spanning ten domains and 5.7k queries with both simple and indirect queries, it introduces trajectory metrics beyond final accuracy. Results show strong models like Claude-4 and Gemini-2.5 struggle with implicit queries and longer tool chains, revealing key weaknesses in current agentic tool use.

### Strengths
The benchmark clearly presents its novelty and contributions, and the work is both timely and relevant.

The paper provides valuable insights into current agentic LLM failures through a comprehensive evaluation.

### Weaknesses
1. TRAJECT-Bench synthesizes both trajectories and queries using LLMs and authors do several human checks during dataset construction. However, the paper does not clearly describe the nature, scale, or consistency of these human validations, making it difficult to assess how much manual correction or filtering was actually performed. 

2. Both Traj-Satisfy and Acc metrics rely on a commercial LLM judge (Claude), which introduce model-specific biases, reduce reproducibility, and make evaluation expensive.

3. No learning experiments: the paper doesn’t analyze how models trained or fine-tuned on TRAJECT-Bench perform, or whether such training would actually help generalization.

### Questions
1. Based on the authors’ definitions of the EM and Inclusion metrics, EM should always be less than or equal to Inclusion. However, this is not the case in some rows of Tables 4 and 5 (e.g., all-MiniLM in Table 4, and Gemini and DeepSeek in Table 5). Could the authors clarify?

2. For the hard query generation, how does TRAJECT-Bench ensure that the hard queries are genuinely more difficult than the simple ones while remaining semantically equivalent? It seems this would require careful human inspection

### Soundness
3

### Presentation
2

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
This paper presents a new benchmark for evaluating the tool-use capabilities of LLM agents. Compared with prior work, it analyzes LLM agents’ tool-invocation trajectories across a variety of real-world tasks and, importantly, assesses whether agents correctly employ tool parameters when calling tools.

### Strengths
1. The paper introduces a novel perspective — the tool-use trajectory — which captures detailed information about the tool invocation processes of LLM agents.

2. It enhances the complexity of the evaluated tasks by incorporating more challenging real-world scenarios.

3. It considers two trajectory structures, called parallel and sequential, providing a systematic analysis of different tool-use patterns.

### Weaknesses
1. The paper divides user queries into “simple” and “hard” versions, but this categorization appears somewhat abstract. Based on the descriptions, the distinction mainly reflects differences in the linguistic or semantic complexity of the queries rather than task difficulty. A clearer and more systematic classification criterion would improve the benchmark’s validity. Furthermore, the current binary categorization seems overly simplistic — a more fine-grained and comprehensive design would strengthen the study.

2. The baseline comparison in the experimental setup is insufficient and requires more comprehensive evaluations. Tool use is inherently a capability of LLM agents, not a standalone LLM. Therefore, the baseline experiments should not be limited to direct LLM querying. Since the tool-use ability depends on both the agent paradigm (e.g., ReAct, Reflexion, etc.) and the backend LLM, the evaluation should include combinations of different agent paradigms and LLMs. Although Section 4.4 reports some results using the ReAct framework, these results are insufficient. A more complete comparison across diverse agent paradigms would provide more informative insights.

### Questions
1. What are the practical application scenarios for this benchmark? Is it intended for robust training purposes?

2. Regarding "tool-use trajectories of different complexities," is complexity defined solely by the number of tools involved in the trajectory, or are there other dimensions considered?

3. Additionally, given that some studies have shown that LLM agents are capable of self-correction, it would be valuable to know whether there are any interesting findings in the trajectories of self-correction.

### Soundness
2

### Presentation
3

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
A novel benchmark TRAJECT-Bench is proposed in this paper, which is designed for the comprehensive evaluation of Large Language Models' (LLMs) tool-use capabilities. The authors argue that existing benchmarks primarily focus on final-answer accuracy, overlooking the critical aspect of the tool-use trajectory, i.e., the process of correctly selecting, parameterizing, and sequencing tools. To address this problem, over 1,200 high-fidelity, executable tools from practical domains (e.g., travel, finance, music) are included in TRAJECT-Bench.
It also includes the synthesized tool-use trajectories of varying complexity (including parallel and sequential structures) and user queries at two difficulty levels for each trajectory. Through extensive experiments on the state-of-the-art LLMs, TRAJECT-Bench reveals key failure modes and scaling challenges, particularly the bottleneck in transitioning from short to mid-length trajectories.

### Strengths
1. The proposed benchmark is valuable and more comprehensive compared with other existing benchmarks, offering notable contributions to advancing the related research area while providing inspiring insights for future development.

2. The trajectory-evaluation metrics shed light on locating the errors caused by false tool use.

3. The authors provide a systematic discussion about the tool-use in agents, and conduct extensive experiments show their trustworthiness.

4. The available source codes shows good reproducibility for the results.

### Weaknesses
1. Only two structures (parallel and sequential chains) are evaluated in the experiments, omitting more complex graph topologies (branching, merging, backtracking).

2. It lacks detailed description for the methodology for automated validation of parameter passing between tools.

3. I suggest considering more domains (e.g., OS/DB ops, robotics, enterprise systems) in the benchmark, besides the ten domains.

### Questions
1.  How about the tool use in the scenario of mixing parallel and sequential type? 

2. I doubt whether TRAJECT-Bench truly captures the core challenges of real-world tool learning, given its only focuses on evaluating models within a static, closed-world setting.

3. Is it possible that an agent calls the tools not existing in the tool pools? If so, how to handle this situation? As an error occurs in the middle of sequential calling, the post-hoc tool-calling is meaningless. Even with higher overlap, the trajectory is illegal.

### Soundness
3

### Presentation
2

### Contribution
3
