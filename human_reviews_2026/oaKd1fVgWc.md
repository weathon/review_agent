# TritonGym: A Benchmark for Agentic LLM Workflows in Triton GPU Code Generation

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) can already draft plausible Triton kernels, yet most existing evaluations still focus on single-shot generation and underplay tool use and feedback. We introduce TritonGym, a benchmark and orchestration framework for evaluating agentic workflows in GPU code generation. TritonGym standardizes access to tools via a function-call API, separating intrinsic model capability from workflow design and enabling fair, apples-to-apples comparison. The benchmark spans a maintained operator set, community samples, out-of-distribution tasks, and DSL extensions, ensuring both generality and extensibility. By providing a common orchestration and evaluation framework, TritonGym democratizes the development of GPU coding agents, supports practical adoption of agent-generated kernels, and facilitates progress on advanced agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This article introduces TritonGym, a benchmark for evaluating llm agentic workflows on GPU kernel generation with the Triton language. TritonGym enables fair comparisons of different workflows by standardizing tools. The data includes 4 parts: maintained operators, community samples, OOD tasks and DSL extensions. Experimental results show that stronger llms can generate more reasonable kernels for familiar operators in one-shot setting but struggle with OOD and DSL tasks, while agentic workflows substantially improve both pass@1 and perf@1.

### Strengths
- **Clear Formulation and Motivation**: The paper identifies a real gap in existing benchmarks: current GPU code generation evaluations bundle tool access within each system, making fair comparison impossible. The motivation for standardizing tool invocation via function-call APIs is compelling and well-articulated.
- **Well-Designed Tool Space**: The set of function-call tools (compiler, verifier, evaluator, profiler, searcher, browser) is thoughtfully designed and mirrors real-world GPU kernel development workflows. The tools effectively enforce a correctness-before-performance progression and provide actionable feedback for iterative refinement.
- **OOD Operator Design**: The out-of-distribution operators are a genuinely novel contribution. Benchmark-only operators with no public implementations (e.g., Max Reduce GEMM, Chaos Norm) effectively test reasoning and optimization understanding rather than memorization, addressing a critical weakness in existing benchmarks where web search could trivialize tasks.
- **Comprehensive Model Coverage**: The evaluation includes diverse state-of-the-art LLMs from multiple vendors (Claude Sonnet 3.5, GPT-4o, DeepSeek-chat, Llama-4-Maverick) with consistent experimental settings, providing valuable insights into current model capabilities and limitations.

### Weaknesses
- **Missing Evaluation**: The paper positions itself as a benchmark for "agentic workflows" but only evaluates fixed, manually-designed workflows (AlphaEvolve, Geak, One-shot). It does not evaluate recent work on automated workflow generation and optimization (e.g., ADAS, AFlow, etc.).
- **Missing Tool Usage Analysis**: The paper provides no analysis of which tools are most valuable, how frequently they're invoked by different agents, or what happens when individual tools are removed. There are no ablation studies showing the impact of the profiler, searcher, or other tools on final performance. This makes it impossible to understand which components of the tool suite actually contribute to improvements.
- **Presentation Issues and Technical Errors**: The paper contains several typographical and citation errors that affect quality: Figure 3(c) has an extra closing parenthesis in the caption; Section 3.3.1 contains a broken citation marked as "(?)".

### Questions
- **Dataset Size**: I seem to have not seen any statement regarding the dataset size. How many operator tasks are included in the benchmark in total? How many input shape configurations are tested per operator?
- **Cost and Efficiency Metrics**: What is the computational cost (API tokens, wall-clock time, dollar cost) of different workflows? How does the cost scale with trial budget K? Can you provide cost-normalized metrics (e.g., Perf@K per dollar)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes TritonGym—a tool-centric benchmark for agentic workflows in Triton GPU kernel generation. Core contributions include: a standardized function-call interface (compiler/verifier/evaluator/searcher/browser/profiler) that unifies tool usage and budgets; a diverse task suite covering the maintained operator set, community samples, out-of-distribution (OOD) samples, and DSL extensions (TLX, Gluon); evaluation metrics Pass@K and Perf@K; and full logging of tool invocations to ensure reproducibility and apples-to-apples comparison. Empirically, with the same trial budget, agentic workflows outperform one-shot; gains are larger under OOD and DSL extension settings; LLM interchange orchestration can lift Pass@1 but may destabilize performance.

### Strengths
1. **Standardized tool interface and reproducible evaluation**: A clear function-call API with unified thresholds for compiler/verifier/evaluator, together with exhaustive logging of tool calls and outcomes, substantially improves apples-to-apples reproducibility across workflows.
2. **Comprehensive dataset and task design**: Beyond standard and community samples, TritonGym includes OOD tasks with explicit semantics but no public implementations, and DSL extension tasks for TLX and Gluon, covering a difficulty spectrum from familiarity to transfer and extension.
3. **Fairness-oriented evaluation protocol**: Pass@K and Perf@K are well defined; performance focuses on average latency while allowing custom metrics, aligning with engineering practice. Trial-budget sensitivity is also reported.
4. **Representative empirical findings**: Systematic comparisons of one-shot vs. agents, failure breakdowns, trial–performance curves, and LLM interchange orchestration substantiate the value of tool–feedback–iteration.
5. **Engineering and openness**: Docker/evaluation suite and a future leaderboard are planned; a clear roadmap targets more backends, more DSLs, and automated OOD generation.

### Weaknesses
**1. Positioning vs. KernelBench (Moderate)**
- Needs sharper head-to-head evidence to justify “Triton-only + standardized tool layer” versus KernelBench’s function-call and multi-language support.

**2. OOD design and anti-leakage (Major)**
- Lacks rigorous anti-leakage controls and quantification (time cutoffs, allow/deny lists, caching, visibility windows) and standard→OOD transfer analyses to evidence true generalization.

**3. Fairness and budget accounting (Moderate)**
- Unclear unified billing for compile/verify/evaluate/retrieve/profile and normalization of latency across hardware; finer-grained, leaderboard-visible accounting is needed.

### Questions
1. **Relation to KernelBench**: Can you provide head-to-head comparisons on overlapping tasks under the same hardware, the same trial budget, and the same tool thresholds to quantify TritonGym’s advantages (e.g., log completeness, reproducibility variance, leaderboard stability)? Why not contribute the standardized tool layer to KernelBench?
2. **OOD validity and anti-leakage**:
   - Do searcher/browser tools operate under strict time cutoffs and allow/deny lists? Is there an automated audit that enforces semantic isolation and absence of public implementations for OOD tasks?
   - Please provide standard→OOD transfer experiments and failure-mode differences to show you are measuring generalization rather than mere difficulty.
3. **Tool budget and cost accounting**:
   - Are compile/verify/evaluate/retrieve/profile calls billed into a unified budget? If so, how does the leaderboard account for and display detailed usage (a token-like ledger)?
   - Does latency measurement include warmup, repeated runs, and confidence intervals? How are hardware/driver differences normalized?

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
3

### Summary
This work proposes a new benchmark for coding LLMs and agentic coding systems, aiming to evaluate their effectiveness at writing functional, high-performance GPU kernels in Triton (and extensions). The benchmark gathers samples from a number of sources; including a set of 'maintained' operators from open-source ML projects, other code-generation benchmarks, newly generated operators by mutating existing ones, as well as collecting samples which use language extensions. The majority of the samples appear to be taken from public repos and previously curated code-generation benchmarks. The authors additionally undertake to define an orchestration framework with the goal of better supporting the evaluation of agent systems as opposed to single-generation LLMs, and making comparisons between models fairer. This involves defining a standard set of tool signatures along with their implementation. The authors evaluate a number of models and agent systems with their benchmark, demonstrating that agentic workflows out perform one-shot solution generation and perform better on new OOD samples.

### Strengths
The work does a good job of collecting a range of benchmark problems. It also correctly notes that the community is faced with the need to 'migrate' a number of benchmarks, collected before agent workflows became prominent, and which make the assumption of single-shot solution generation which need to be adapted slightly for use with agent systems. The effort made to make TritonGym work with agent system evaluations is welcome. The results relating to the value of agent systems, in particular on the ability of models to tackle more challenging OOD problems are useful.

### Weaknesses
One slight weakness of the paper is the originality. As surveyed in the related work section, there already exist a number of GPU code generation benchmarks, many of which, especially MultiKernelBench seem to hold significant overlap with TritonGym. Looking at the data collection report in Section 3.1.2, 58% of the samples come from 'community samples' which includes public repositories and prior code-generation benchmarks. The total number of samples in the dataset is not reported either, which makes it hard to gauge the scale of the original contribution of this work.

While the effort to accommodate agent systems as opposed to single-response LLM generations is welcome, this must be done with care. It is common practice now for agent evaluations to involve the researcher or practitioner adapting early LLM benchmarks to work with their agent, although this can indeed lead to variations in prompts, submission formats and so forth. The effort to standardize the tool set is good, however as I understand it all evaluated methods must run within the same orchestration framework. My concern is that this is too restrictive for the evaluation of agent systems, since different LLMs and agents often have their preferred function calling or tool calling mechanisms, which may differ on semantics, invocation format and so forth. It would therefore be welcome if the benchmark provided a standardized definition for the tools it provides (for instance as a JSON schema) thus allowing agents to use their own tool calling mechanisms, while still leveraging the common tool implementation on the backend. I believe this would allow many more agents to be run on this benchmark with minimal modifications; just updating the available tool set. Please do clarify whether your orchestration framework makes the tool documentation (Figure 2, b) available to agents to use within their tool calling mechanisms at runtime.

### Questions
- what steps have you taken to ensure that all the samples within the benchmark are correct? The move from SWE-Bench to SWE-Bench Verified through rigorous (often manual) review of the questions revealed a number of poor-quality problems in the dataset. How can you guarantee that similar errors don't exist in TritonGym?
- Would it be possible to update the evaluation results table in the paper with more recent models? All the models included are previous-generation models, and there are no models included from after the end of last year which limits the usefulness of the evaluation. For instance, budget permitting, it would be good to run the evaluation on GPT-5, Sonnet 4.5, Gemini 2.5 pro, Grok 4 and perhaps also open models such as Qwen 3, DeepSeek v3, Codestral and so forth. Doing so would improve the relevance of the information the reader can derive from TritonGym on the performance of frontier LLMs, generalization to OOD tasks and the usefulness of various agent systems.

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
The paper introduces TritonGym, a benchmark and orchestration framework for evaluating agentic LLM workflows in Triton GPU kernel generation. The key contribution is standardizing tool access via a function-call API, decoupling model capability from workflow design and enabling fair apples-to-apples evaluation. The benchmark includes curated operators, community samples, OOD operators, and DSL extensions (e.g., TLX, Gluon), with extensible hardware descriptors and backend support. Empirical results show (i) one-shot LLMs perform reasonably on common operators but fail on OOD and DSL tasks, and (ii) agentic workflows significantly improve correctness and performance.

### Strengths
1. **OOD operator design**: The creation of benchmark-only operators (e.g., max-reduce GEMM, chaos norm) that test reasoning rather than memorization is creative and addresses the retrieval leakage problem intelligently.

2. **Extensibility framework**: The explicit backend and DSL descriptor architecture enables systematic evaluation across diverse hardware and language extensions.

3. **Comprehensive baseline coverage**: Evaluation of 4 frontier LLMs across 3 workflow paradigms (one-shot, Geak, AlphaEvolve) with detailed failure analysis provides thorough empirical grounding.

4. **Practical relevance**: Addresses real adoption challenges for LLM-generated GPU kernels with extensible backend/DSL support.

### Weaknesses
1. **Insufficient scale & statistical rigor**:  The paper does not explicitly report the total number of benchmark samples. No confidence intervals or significance tests despite probabilistic metrics.

2. **Weak OOD validation**: Claims OOD tasks "require reasoning beyond memorization" lack empirical support.  No systematic analysis of OOD similarity to training data. There is no analysis of OOD operators' similarity to training distributions, nor evaluation against retrieval-augmented baselines. As a result, the degree to which OOD tasks genuinely measure generalization remains unclear.


3. **Insufficient dataset quality assurance and documentation**：The dataset construction and oracle validation process lacks transparency. Quality control criteria for DSL extensions and specialized operators is not clear. OOD operators correctness and quality control ('handcraft efficient Triton implementations as baselines' in line 218) are unclear. No discussion of how oracle implementations were validated or tested.   This raises concerns about oracle reliability and the validity of performance comparisons.

4. The manuscript contains an unresolved editing placeholder (“shallow control flow (?)” Line 297)

### Questions
1. OOD validation: How do you measure OOD similarity to training data? Can you empirically show that retrieval-augmented generation fails on OOD tasks, demonstrating non-triviality?

2. Hardware specs usage: Do models actually exploit hardware specifications (Figure 2d) in generated code? Can you show examples?

3. Trial budget: In Figure 5, how do you ensure fair comparison between K independent generations vs. K iterative refinements (same total LLM calls)?


4. Oracle quality: How were oracle implementations validated? Could Perf@1 > 1.0 indicate oracle weaknesses?

### Soundness
3

### Presentation
3

### Contribution
3
