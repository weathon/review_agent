# Build-Bench: Benchmarking LLM Agents on Compiling Real-World Open Source Software

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Automatically compiling open-source software (OSS) projects is a vital, labor-intensive, and complex task, which makes it a good challenge for LLM Agents.
Existing methods rely on manually curated rules and workflows, which cannot
adapt to OSS that requires customized configuration or environment setup. Recent
attempts using Large Language Models (LLMs) used selective evaluation on a
subset of highly rated OSS, a practice that underestimates the realistic challenges
of OSS compilation. In practice, compilation instructions are often absent, de-
pendencies are undocumented, and successful builds may even require patching
source files or modifying build scripts. We propose a more challenging and realistic
benchmark, BUILD-BENCH, comprising OSS that are more diverse in quality,
scale, and characteristics. Furthermore, we propose a strong baseline LLM-based
agent, OSS-BUILD-AGENT, an effective system with enhanced build instruction
retrieval module that achieves state-of-the-art performance on BUILD-BENCH and
is adaptable to heterogeneous OSS characteristics. We also provide detailed analysis regarding different compilation method design choices and their influence to
the whole task, offering insights to guide future advances. We believe performance
on BUILD-BENCH can faithfully reflect an agent’s ability to tackle compilation
as a complex software engineering tasks, and, as such, our benchmark will spur
innovation with a significant impact on downstream applications in the fields of
software development and software security.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Build-Bench, a benchmark on compiling open-source software project with LLM Agents. The authors propose more realistic challenges of OSS compilation, mainly targeting github repositories of C/C++ projects. The authors also propose a strong baseline for LLM agent which enhance build instructions based on a retrieval module.

### Strengths
(1) The authors too greate efforts to enhance the data selection process for collecting Build-Bench problems. Especially the distribution of stargazer counts more aligns with data collected from the wild on github. Furthermore the dataset demonstrate a wide variety of build systems and tool chains.

(2) A multi-agent LLM agentic baseline is proposed to solve Build-Bench. The first stage LLM are used to extract relevant instructions and access relevant links and files to generate compilation instruction. The second stage consists of a flow-based method, where a bash command generator compiles bash commands and the executor agent executes the command.

### Weaknesses
(1) Marginal improvements over CompileAgent. It seems that CompileAgent with Retrieval on GPT-4o works quite well, and only a marginal improvement on flexible validated successes is presented.

(2) Success metric is determined by compilation and existence of binary links, which seems too tailored to C/C++.

### Questions
(1) How would the success method be extended to programming languages outsied of C/C++. Is using unit tests etc. a more rigourous method for evaluation?

(2) Why solely focus on C/C++? Any plans to extend to other programming languages (such as Python etc.)?

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
3

### Summary
This paper presents BUILD-BENCH, a benchmark dataset derived from 385 randomly sampled C/C++ GitHub repositories, with 148 manually validated as compilable and annotated with ground-truth binary artifacts and documentation URLs. The authors propose OSS-BUILD-AGENT, a multi-agent LLM system featuring an LLM-assisted retrieval module and iterative error resolution. Evaluations against rule-based baselines (GHCC, Assemblage), single-turn LLM approaches, and CompileAgent using multiple frontier models demonstrate that the best configuration achieves 66.4% strict validated success rate, substantially outperforming existing methods on this more challenging and representative benchmark.

### Strengths
BUILD-BENCH employs principled random sampling from 6.57M repositories, addressing the selection bias in COMPILEAGENTBENCH where 99.88% of real-world C/C++ projects have fewer than 500 stars. The benchmark's build system diversity (Make, CMake, Autotools, MSBuild, custom scripts) and ground-truth annotations enable rigorous evaluation. The experimental design is comprehensive, spanning multiple baselines and frontier LLMs (GPT-4o, o3-mini, Claude 3.7-Sonnet, Gemini 2.5, Qwen3), with transparency regarding stochasticity through repeated runs and pass@k analysis. The retrieval module comparison with CompileAgent provides actionable design insights, particularly regarding documentation-first traversal versus build-script-focused approaches.

### Weaknesses
- The benchmark's scope is restricted to Linux C/C++ projects, with no evaluation on Windows, macOS, or mixed-language codebases, limiting generalizability claims. 
- Manual exclusion of "uncompilable" projects (criteria a-d in Section 2) may introduce selection bias toward better-documented repositories; the paper would benefit from publishing exclusion statistics and considering a "hard mode" supplemental dataset. 
- Critical practical metrics are absent: per-repository runtime, API token consumption, and monetary costs are essential for practitioners evaluating adoption trade-offs. 
- Ablation studies are insufficient—the paper does not isolate retrieval quality versus agent architecture, evaluate lightweight retrieval heuristics, or test retrieval module sensitivity to base LLM choice.

### Questions
1. Can the authors provide comprehensive efficiency metrics—average runtime per repository, total API calls, token consumption, and estimated costs—for OSS-BUILD-AGENT compared to CompileAgent and rule-based baselines?
2. What are the exclusion statistics for the 237 removed repositories? Specifically, what percentages were excluded due to cross-platform requirements, missing dependencies, broken builds, or other criteria?
3. Have you conducted ablation studies on the retrieval module: (a) performance with different base LLMs, (b) comparison with lightweight heuristics (regex-based link extraction, TF-IDF), and (c) impact of iteration depth and link count limits?

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
This paper presents Build-Bench, a benchmark designed to evaluate LLMs on software build and compilation tasks. The benchmark collects real open-source projects with diverse build systems and automatically generates build tasks that simulate dependency errors, configuration failures, and compilation issues. Each task provides the model with build logs, dependency information, and target build commands as input, requiring it to output repair actions such as command corrections, dependency installations, or configuration updates. Tasks are automatically evaluated via sandbox compilation. Experiments on multiple LLMs show that while current models can handle simple dependency errors, they struggle with complex build environments and cross-language configurations.

### Strengths
The benchmark extends the evaluation of LLMs from code understanding to practical software build and dependency repair tasks. Its design is systematic and well-documented, using reproducible environments and real-world build systems. The task formulation is clear and grounded in realistic developer workflows. This benchmark provides insights into the limitations of current models in dependency management, configuration understanding, and handling multi-language projects.

### Weaknesses
1. Although the projects are sourced from real open-source repositories, the build failures are synthetically generated through automated error injection, which may reduce their realism. In actual development scenarios, build failures often arise from more complex and interdependent causes that maybe not fully represented in the benchmark.
2. The experimental analysis lacks depth, with limited discussion of error categories or their influence on model behavior. A more detailed breakdown of failure types probably can reveal additional insights and lead to stronger conclusions.
3. The evaluation metrics are relatively narrow and strictly binary, while build repair is inherently a gradual process. Fully successful repairs may also differ in their complexity or practical soundness, which the current evaluation framework does not capture.

### Questions
See weakness please

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
The paper BUILD-BENCH introduces a new benchmark for evaluating large language model (LLM) agents on the task of compiling real-world open-source C/C++ software. The authors developed BUILD-BENCH, a dataset of 148 diverse repositories, and proposed a multi-agent system called OSS-BUILD-AGENT that automatically retrieves build instructions and iteratively fixes compilation errors. Experiments show that this method outperforms both traditional rule-based and previous agentic approaches, achieving up to 71.8% success under flexible validation, demonstrating the strong potential of LLMs in automating complex software compilation tasks

### Strengths
1. The BUILD-BENCH benchmark covers more realistic and complex open-source projects, enabling a more comprehensive evaluation of automated compilation capabilities.
2. The proposed OSS-BUILD-AGENT leverages multi-agent collaboration and LLM-based retrieval to significantly improve compilation success rates and automatic error-repair capabilities.

### Weaknesses
**Unspecified Computational Costs:** The paper lacks quantitative analysis of *time cost* (the total time required to complete the entire compilation process) and *economic cost* (the total cost incurred to complete the compilation tasks), as well as a horizontal comparison with baselines—particularly **CompileAgent**.

**Insufficient Baseline Comparison:** It is recommended to include single-round data for both closed-source and open-source models: closed-source models such as Gemini-2.5-Flash, GPT-4o, and Claude-3.5-Sonnet, and open-source models such as Qwen3 235B, Qwen3 Coder 485B. Additionally,  OSS-BUILD-AGENT w/o Retrieval should also include Claude 3.7-Sonnet, Gemini-2.5-Flash, Qwen3 235B, Qwen3 Coder 485B, and a RAG-based baseline.

**Incomplete Task Information:** The paper lacks detailed information on the task set and its difficulty distribution, including details such as *project topic* and whether build instructions are **InRepo** (the repository contains a build guide), **NotInRepo** (the repository does not directly contain a build guide but external documentation is available), or **NoGuide** (the project completely lacks any build guide). The proportion of projects in each of these categories should be reported.

**Unclear Methodology:** The implementation details and prompt design of the **LLM-Assisted Retrieval Module** are not clearly described—particularly how the system retrieves other files in the repository and accesses external web resources.

**Incomplete Evaluation Metrics:** In Section 6.2, *Retrieval and Error Resolution*, the analysis of the retrieval module is overly simplistic. Evaluating retrieval success solely based on whether the retrieval module accessed the ground-truth URL that hosts the build instruction for the given repository is insufficient. The evaluation should also account for retrieval performance over other files in the repository.

### Questions
1.Could you explain in detail the implementation of LLM-Assisted Retrieval?

2.For lines 395–399, which describe the workflow that “*mimics a human engineer*,” could you provide a **case study** illustrating how this process is implemented in practice?

### Soundness
2

### Presentation
2

### Contribution
2
