# Code Researcher: Deep Research Agent for Large Systems Code and Commit History

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large Language Model (LLM)-based coding agents have shown promising results on coding benchmarks, but their effectiveness on systems code remains underexplored. Due to the size and complexities of systems code, making changes to a systems codebase requires *researching* about many pieces of context, derived from the large codebase and its massive commit history, *before* making changes. Inspired by the recent progress on deep research agents, we design the first deep research agent for code, called Code Researcher, and apply it to the problem of generating patches to mitigate crashes reported in systems code. Code Researcher performs multi-step reasoning about semantics, patterns, and commit history of code to retrieve all relevant context from the codebase and its commit history. We evaluate Code Researcher on kBenchSyz, a benchmark of Linux kernel crashes, and show that it significantly outperforms strong baselines, achieving a crash-resolution rate (CRR) of 48%, compared to 31.5% by SWE-agent and 31% by Agentless, using OpenAI's GPT-4o model. Scaling up sampling budget to 10 trajectories increases Code Researcher's CRR to 54%. Code Researcher is also robust to model choices, reaching 67% with the newer Gemini 2.5-Flash model. Through another experiment on an open-source multimedia software, we show the generalizability of Code Researcher and also conduct ablations. Our experiments highlight the importance of global context gathering and multi-faceted reasoning for large codebases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a Deep Research Agent on Code, specifically applying it to a problem set of generating patches to mitigate crashes in systems code. The agent works in three phases of (1) Analysis, (2) Synthesis; and (3) Validation; and the agent is evaluated on a chanllenging benchmark kBenchSyz. Experimental results demonstrate superior advantage of the agent capable of outperforming prior works on gathering relevant context.

### Strengths
(1) The work targets real world complex benchmarks, including large system repository which requires advanced reasoning and context search.

(2) The authors leveraged new action tools, especially allowing the Agent to search through commit histories, to gather enough relevant context related to systems crash and leverage advanced reasoning strategies related to the problem such as chasing control and data flow chains.

(3) Results demonstrate superior performance compared with SweAgent, especially on the global context gathering capability of the proposed agent.

### Weaknesses
(1) Results in Table 1 could be difficult to comprehend. Specifically why are some settings using P@5, P@10 and P@16? This makes comparing between different settings/tools difficult. Why not just use P@1?

(2) How test time scaling was conducted was not properly explained and the results are not analyzed deeply.

(3) It would be great if results are also presented for SweBench.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Code Researcher, an LLM-based agent workflow designed to resolve crash issues in complex system-level codebases. Under comparable settings with GPT-4o, Code Researcher achieves a 48% crash-resolution rate, outperforming common baselines such as SWE-Agent and Agentless.

### Strengths
- To the best of my knowledge, this appears to be the first work explicitly exploring deep research within software engineering tasks, which makes it novel in scope.

- The paper tackles realistic and challenging problems—specifically, software development and maintenance tasks in complex Linux kernel C/C++ codebases. Compared to prior work focusing on simpler scripting languages such as Python (e.g., SWE-bench), the chosen domain is harder, more practical, and less explored.

### Weaknesses
- I find the claim of performing “Deep Research for Code” somewhat confusing.
From my understanding, project-level issue resolution inherently requires agentic search and information summarization. Thus, the conceptual gap between Code Researcher and a standard coding agent seems much smaller than the gap between a general AI chatbot and a deep research agent.
Could the authors clarify what fundamental difference distinguishes a Code Researcher from a regular coding agent? Specifically, what unique design elements justify the term “deep research”? Simply shifting from small Python projects to large-scale kernel-level issues or using commit history do not make an agent a deep researcher; there should be distinctive mechanisms or reasoning processes that set it apart.

- According to the authors’ claim, the main challenges of kernel code issue lie in the large scale and complexity of the codebase. However, the design of Code Researcher does not appear to include any distinctive mechanisms specifically aimed at addressing this challenge. The code search tools are already used in most existing SWE agents, making it unclear what unique architectural or algorithmic features enable Code Researcher to outperform existing agents on kernel-level code. It is also worth noting that a better prompt alone does not constitute a research contribution.

- The memory filtering mechanism in the Synthesis stage is particularly intriguing. This may be one of the few substantial architectural differences from existing coding agents, which typically include retrieval, recall, and patch generation stages.
However, the paper does not include any ablation studies or quantitative analysis demonstrating the contribution of this filtering design. Without such evidence, it’s unclear how much this component actually contributes to the overall performance.

- The dataset used seems potentially affected by the incomplete patch issue, which could lead to unreliable evaluation results [1]. Given that the dataset includes only about 200 instances, I suggest the authors clarify whether this issue might have caused noticeable variance in evaluation outcomes.
Additionally, providing a comparison on SWE-bench (e.g., Code Researcher vs. SWE-Agent) would make the evaluation more convincing. I understand that, as the authors note, smaller repositories may not fully leverage Code Researcher’s advantages. Still, even a limited-scale comparison would strengthen the credibility of the reported results. If the authors believe that these incomplete patches are unlikely to affect the results, a clear explanation of why this is the case would also suffice without extra experiments.

Reference
[1] https://github.com/Alex-Mathai-98/kGym-Kernel-Playground/issues/1

### Questions
1. What characteristics make Code Researcher “the first deep research agent for code”? How does its deep research capability fundamentally differ from that of standard software engineering or coding agents? If there is no essential difference at the task modeling level, I would suggest the authors narrow the claim and focus on the goal of solving large-scale and complex software development and maintenance tasks, rather than emphasizing the “deep research” concept.

2. How much does the memory filtering mechanism contribute to the overall performance? Could the authors provide ablation results or a qualitative analysis?

3. What unique mechanisms or algorithmic features enable Code Researcher to address kernel crash issues? It seems that most of its modules are consistent with those of regular agents (e.g., code search tools).

### Soundness
3

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
This paper proposes Code Researcher, a deep research-oriented agent for large-scale system code, aimed at fixing crash issues in complex codebases like the Linux kernel. The core innovation lies in introducing the deep research paradigm into the code repair domain, gathering necessary information through multi-step reasoning, historical commit analysis, and structured context memory. On the kBenchSyz benchmark, Code Researcher achieves a 58% crash resolution rate, significantly outperforming SWE-agent's 37.5%. Overall, this is a paper with solid engineering implementation and reasonable experimental design, but it has certain limitations in methodological innovation and evaluation comprehensiveness.

### Strengths
1. **Clear Problem Identification and Motivation**: It clearly identifies the shortcomings of existing code agents in handling large-scale system code (lack of deep context collection capabilities) and designs targeted solutions.

2. **Systematic Method Design**:
   - The three-stage process (analysis-synthesis-verification) is logically clear.
   - The reasoning strategies are reasonably designed (control/data flow tracing, pattern detection, commit history causal analysis).
   - The structured memory mechanism effectively manages complex contexts.

3. **Important Technical Contribution - Commit History Search**: It is the first to systematically integrate historical commit analysis into code repair agents, with ablation experiments proving its importance (performance drops by 10% when removed).

4. **Thorough Experimental Validation**:
   - Multi-dimensional comparative experiments (assisted vs. unassisted settings).
   - Detailed ablation studies.
   - Cross-codebase generalization validation (FFmpeg).
   - Qualitative analysis classification (accurate/over-specialized/incomplete/incorrect).

5. **Transparent Experimental Setup**: It provides details on data validation, computational resource usage, baseline adaptations, etc., enhancing reproducibility.

### Weaknesses
### Methodological Limitations
1. **Limited Innovation**: The core method essentially applies known techniques from deep research agents (multi-step tool calls, reasoning strategies, memory mechanisms) to the code domain, lacking fundamental innovations tailored to code characteristics.

2. **Generality of Reasoning Strategies**:
   - The three reasoning strategies (control/data flow, pattern detection, commit history) are reasonable but are direct applications of traditional software engineering concepts.
   - It fails to fully leverage code's structured characteristics (e.g., AST, program dependence graphs).
   - It lacks specialized reasoning for system code-specific complexities (concurrency, memory management, hardware interactions).

3. **Simplistic Context Filtering Mechanism**: The synthesis stage relies solely on LLM to judge memory relevance, lacking more refined filtering strategies based on program analysis.

### Evaluation Deficiencies
4. **Representativeness Issues in Benchmark Dataset**:
   - kBenchSyz only includes fuzzing-discovered crashes, potentially biased toward specific bug patterns.
   - Out of 279 instances, only 200 (71.7%) can be reproduced, raising questions about data quality.
   - It lacks analysis of the distribution across different bug types (memory errors, concurrency, logic errors).

5. **Single-Metric Patch Quality Evaluation**:
   - Using only "whether it prevents the crash" as the success criterion is too coarse.
   - While qualitative analysis includes classifications, it lacks systematic quantification (what are the proportions of each category in the 58% patches?).
   - It does not evaluate patch side effects (performance impact, functionality disruption, maintainability).

6. **Unfair Comparison with CrashFixer**:
   - CrashFixer is evaluated in the assisted setting (known buggy files).
   - Budget settings are incomparable (P@16 vs. P@5).
   - The authors admit "exact budget is not known," reducing the persuasiveness of the comparison.

7. **Missing Important Baselines**:
   - No comparison with AutoCodeRover (despite mentioning it in the paper).
   - No testing of simple long-context baselines (putting entire crash-related files into context).
   - No comparison with RAG-based methods.

8. **Insufficient Generalization Validation**:
   - FFmpeg experiment uses only 10 samples, lacking statistical significance.
   - No testing on other system codebases (e.g., database systems, network stacks).
   - No testing on non-C/C++ language system code.

### Technical Detail Issues
9. **Search Efficiency and Scalability**:
   - Average exploration of 29 files per bug; scalability on larger codebases (e.g., Chromium) is unknown.
   - Performance bottlenecks of search tools (ctags, git grep) are not discussed.
   - The basis for the 60-second timeout setting is unclear.

10. **Limitations of Commit History Search**:
    - Supports only regex search, unable to handle semantically similar but differently phrased commits.
    - Does not consider commit recency (recent commits may be more relevant).
    - Truncating to 100 lines may lose important context.

11. **Temperature Parameters and Sampling Strategies**:
    - Analysis uses 0.6 temperature, synthesis uses increasing temperatures (0, 0.3, 0.6), lacking ablation experiments for validation.
    - Why not try diversified sampling in the analysis stage as well?

### Questions
1. **Sufficiency of Reasoning Strategies**: The paper proposes three reasoning strategies, but how do they ensure coverage of the main diagnostic patterns for system code crashes? Have strategies based on program analysis tools (e.g., static analysis, symbolic execution) been considered?

2. **Optimization of Context Selection**: In RQ3, Code Researcher reads 29.1 vs. 1.9 files compared to SWE-agent, but is "more" always "better"? Are there counterexamples where excessive irrelevant context reduces performance? **Moreover, as far as I know, Gold Patches on SWE-Bench often require modifying only 1-3 files.**

3. **Necessity Analysis of Commit History**:
   - Among the 96 successful cases, how many truly rely on commit history?
   - Can examples be provided to illustrate which types of bugs must depend on historical commits to fix?
   - Is the method still effective for projects with shorter codebase histories?

4. **Comparison with Human Developers**:
   - How many files do humans typically explore to fix these bugs?
   - How similar is Code Researcher's reasoning path to human experts' diagnostic processes?
   - Are there cases where Code Researcher discovered contexts unnoticed by humans?

5. **Error Propagation and Cascading Fixes**:
   - How to handle situations where a crash requires modifying multiple independent modules?
   - How does the synthesis stage determine the scope of modifications?
   - Are there cases where partial fixes lead to new issues?

6. **Cross-Version Generalization**:
   - Experiments are conducted on parent commits; what if tested on earlier versions (e.g., parent's parent)?
   - Can the method migrate knowledge across different kernel versions?

7. **Industrial Applicability**:
   - What is the average reasoning time per bug?
   - Is a 58% resolution rate sufficient for practical value in actual development workflows?
   - How to integrate with existing CI/CD processes?

8. **Potential of Test-Time Scaling Methods**:
   The paper only tests simple test-time scaling in the Unassisted + Scaled setting (increasing max_calls and sampling number k), finding limited effects from increasing trajectory length but some help from increasing sampling diversity. However, recent advanced TTS methods on SWE-Bench, such as SE-Agent [1]'s self-evolution trajectory optimization, can significantly improve performance through iterative refinement and trajectory quality assessment. Can these advanced TTS methods be migrated to this approach and bring substantial improvements? This also affects the method's generalizability.

**Among these, questions 1, 2, and 8 are more important; if my concerns can be addressed, I am inclined to raise the score.**

---
References: [1] SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Code Researcher, a deep research agent for code. It first performs an analysis phase to gather context, then a synthesis phase to filter irrelevant context and generate a patch, and finally a validation phase to test the patch. Empirical evaluation is provided on fixing system code crashes (kBenchSyz). Additional analyses are done to evaluate Code Researcher's context gathering accuracy and some ablations on its design.

### Strengths
1. This paper focuses on fixing system code bugs, which is related to the popular SWE-bench type problems but more low level and potentially of a larger scale. Given the importance of system code, developing more effective systems for that is meaningful. Empirical evidences suggest Code Researcher outperforms existing agents for general software, such as SWE-agent and Agentless.

2. This paper contains many insightful analyses on the Code Researcher system, such as the accuracy of the Analysis phase and inference scaling impact on the resolution rate. They help a reader better understand the system.

3. Overall the paper is well-written and easy to follow.

### Weaknesses
1. The novelty for Code Researcher is limited. It follows a three-phase workflow similar to Agentless. Its Analysis phase is iterative. Likewise, SWE-agent and Openhands can also perform iterative context retrieval. What makes Code Researcher different seems to be: 1) additional tools (search_commits); 2) a specialized search strategy guidance through prompting (section 3.1.2). So the overall innovations are incremental.

2. The Pass@k metric seems incomplete as described in the paper. It is defined as "prevents the crash" (line 273). This does not mean the patch is functionally correct. Section 5.5 provides some additional tests but they are incomplete as well, for example 28 out of the 116 crashes analyzed do not have additional tests. The ambiguity in correctness measure makes it difficult to evaluate the reported numbers. A clear correctness measure is necessary to evaluate coding agent systems.

### Questions
1. What is the search action distribution for the actions mentioned in section 3.1.1?

2. In the experiments, $k$ trajectories are sampled in the Analysis phase. Does the Synthesis phase use the combined context from all trajectories or does it use each one separately?

3. Why is 50k token chosen as the context length limit? Both GPT-4o and o1 can handle much longer context.

4. I find it interesting that increasing the number of trajectories is more effective than increasing their lengths. Is there an explanation why this is the case?

5. In addition to measuring the ground-truth file recall on patches, it would also be valuable to evaluate on the retrieved context.

### Soundness
3

### Presentation
3

### Contribution
2
