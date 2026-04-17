# ToolTree: Efficient LLM Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and fail to account for inter-tool dependencies.
In this paper, we present ToolTree, a novel Monte-Carlo tree search-inspired planning paradigm for tool planning. ToolTree explores possible tool usage trajectories using a dual-stage LLM evaluation and bidirectional pruning mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution. Empirical evaluations across both open-set and closed-set tool planning tasks on 4 benchmarks demonstrate that ToolTree consistently improves performance while keeping the highest efficiency, achieving an average gain of around 10\% compared to the state-of-the-art planning paradigm.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ToolTree, an MCTS-inspired framework for LLM tool planning, aiming to solve the shortcomings of current methods—greedy strategies lack long-term foresight and search-based methods are inefficient. ToolTree integrates dual feedback (pre-execution scoring to predict tool utility before invocation and post-execution scoring to assess actual output value) and bidirectional pruning (cutting unpromising branches before and after tool execution) into an iterative search process. Evaluated on 4 benchmarks (GTA, m&m for closed-set tool planning; ToolTree, RestBench for open-set) with GPT-4o and GPT-4o-mini, ToolTree consistently outperforms baselines: it achieves 66.95 F1 on GTA, 69.04 pass rate on ToolBench, 88.61 average F1 on m&m, and 74.50 average on RestBench-TMDB with GPT-4o, while having the highest accuracy-per-second. Ablation shows dual evaluation and pruning are critical.

### Strengths
1. The work accurately identifies the "short-sightedness" and "inefficiency" pain points in LLM tool planning, and the proposed MCTS framework with "dual-feedback evaluation + bidirectional pruning" is highly targeted, forming a logically closed-loop innovative solution.
It performs remarkably well in balancing performance and efficiency, outperforming the state-of-the-art on four benchmark datasets while achieving the best "accuracy per second", thus balancing effectiveness and practical applicability for deployment.
2. It performs remarkably well in balancing performance and efficiency, outperforming the state-of-the-art on four benchmark datasets while achieving the best "accuracy per second", thus balancing effectiveness and practical applicability for deployment.

### Weaknesses
I have a series of questions and concerns regarding this work.

1. Currently, this work only focuses on improving MCTS for tool invocation. Given the abundance of similar prior studies, its core innovation is not prominent—it feels more like adding refined pre-processing and post-processing steps to the existing MCTS workflow. Overall, it seems to trade off some inference latency and use more token-based reasoning to achieve better results. From the comparison figures, can we consider pruning as its core contribution? The description of the pruning component is also insufficient: beyond understanding that more detailed reasoning is used to decide whether the model should execute a tool (thus affecting latency), what additional insights does this work provide?
2. The experimental analysis is inadequate. It focuses mostly on dataset-based measurements and lacks in-depth ablation studies on the modules added in the paper. This makes it difficult to identify the exact source of the advantages brought by these new modules.
3. The experimental metrics are questionable. I paid special attention to metrics like Tool F1, which is defined as the accuracy of tool selection and its alignment with the "standard answer." Does tool invocation always have a "perfect answer"? Strictly speaking, what we need is an evaluation of whether a tool fulfills its intended function—how exactly are the standard answers annotated? The Arg F1 metric (alignment of input parameters) is also problematic: input parameters can vary under different scenarios, so this metric is overly rigid and fails to truly assess whether the proposed ToolTree method is effective.

4. Insufficient clarification on tool library dependency: The paper mentions that ToolTree integrates a domain-specialized tool library (e.g., BioMedParse for the medical field and Wolfram Alpha for the mathematical field), yet it fails to analyze the correlation between tool library scale and performance. For instance, when the number of tools in the library scales from dozens to thousands, will the "tool filtering efficiency" during the pre-evaluation stage decrease? Do the thresholds for bidirectional pruning need to be dynamically adjusted based on the size of the tool library? The existing experiments do not provide such key information, which limits the understanding of the method’s scalability.

5. Weak compatibility with low-resource models: Although the experiments include GPT-4o-mini, the results show that the magnitude of its performance improvement (e.g., Tool F1 on the GTA dataset is only 67.83, a significant gap compared to GPT-4o’s 79.26) is far smaller than that of high-resource models. The paper does not analyze the reasons for the performance degradation of "dual-feedback evaluation" under low-resource models—whether it is due to the insufficient accuracy of pre-evaluation by lightweight LLMs or the limited ability of post-evaluation to judge outputs. Nor does it propose optimization solutions for low-resource models, reducing the practicality of the method in scenarios with limited computing power.

6. Failure to discuss the recovery mechanism for error propagation: While the paper claims that ToolTree can "recover from early mistakes," it does not specify the recovery logic. For example, if a tool invocation in a certain step leads to output deviation due to minor parameter errors (such as unit mistakes), how can subsequent steps identify and correct this error through dual feedback? Will it re-invoke the tool, adjust parameters, or switch the tool chain? The existing experiments do not design such "error injection" scenarios, making it difficult to verify the effectiveness of its error recovery capability.

7. The entire paper is extremely rough, as if it has not undergone any revision or proofreading. For example, in line 104, "t1" and "lib" lack proper subscripts; in lines 272–273, the red highlighting exceeds column boundaries; in line 407, there is a missing figure reference for "Table ??"; and there are numerous other minor issues. A major revision is strongly recommended.

### Questions
Please refer to above, I think this paper requires a significant revision.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces ToolTree, novel methods for Monte Carlo Tree Search-based planning with two tree pruning mechanisms: evaluation before tool calling ("pre-pruning") acts as foresight by avoiding low-promise children, whereas evaluation after tool calling ("post-pruning") acts as hindsight by trimming branches disproven by evidence. Experimental evaluations on four benchmarks, including closed-set (GTA and m&m) and open-set (ToolBench and RestBench) tool planning tasks, demonstrate that ToolTree largely outperforms a broad range baselines, which span LLMs with greedy decoding, ReAct-based planning, more simple tree-based planning such as ToT, and more sophisticated tree-based planning such as A*. ToolTree is shown to achieve better accuracy-per-second performance, and the paper includes thoughtful and comprehensive ablation studies that help to understand how the methods behave as a function of design choices, model scale, and backbone evaluating LLMs.

### Strengths
1. Making tool planning more efficient via cost-effective tree pruning is a relevant and timely topic, and the paper is particularly well motivated.

2. The paper has a clear organization and is mostly well-written (see improvement suggestions further below), making it easy to follow. For example, lines 230-235 are particularly effective in summarizing the main ideas detailed through Section 3.

3. The choices of benchmarks and baselines feel sound and thoughtful, yielding interesting analyses. They comprise both closed- and (relatively) open-set tool planning, with two benchmarks each; and a broad range of alternatives, namely, LLMs with greedy decoding, ReAct-based planning, more simple tree-based planning (e.g., ToT), as well as more sophisticated tree-based planning (e.g., A*). Ablations are also sensible and informative.

4. Overall, this is a strong paper submission on the basis of the idea novelty and description, experimental soundness, and results and analyses, with small improvement opportunities w.r.t. presentation as described below.

### Weaknesses
1. An important implementation detail that remains unclear is the combination of "lightweight pre-evaluation LLM judge" (per lines 211-213) and "possibly stronger LLM judge" (per lines 221-222) used through Section 4. There is a study in the Appendix A.6, but that doesn't quite clarify what exactly is behind Section 4. It would also be useful to add the prompting/instantiation details to the Appendix, for reproducibility.

2. Writing can be improved with more proofreading, specifically:

i. On line 407: the Table reference is missing. The entire block should refer to Figure 4, whose reference is also missing.

ii. On Table 1: under "m&m > Step-by-step > Arg," the value for A* is 71.85 and therefore higher than the result highlighted for ToolTree, so this specific highlight should be fixed for correctness.  

iii. On lines 344-345: the budget values in Figure 3 are not clear. It would be helpful to add ticks and tick values on the x-axis that align to the budget marks, since we are referred to Figure 3 for this information.

iv. On line 367: typo "explores uses."

v. On line 349: typo "While (...) Best-First, yet comparable (...)"

vi. On line 248: typo "where each providing (...)"

vii. On line 245: typo "is spans."

viii. On line 155: typo "back propogation."

ix. On line 78: typo "per cent."

### Questions
What is the combination of pre-evaluation LLM judge and post-evaluation LLM judge used through Section 4?

### Soundness
4

### Presentation
3

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
In this paper, the authors present ToolTree, a Monte Carlo tree search-inspired planning paradigm for tool planning. ToolTree explores possible tool usage trajectories using a dual-stage LLM evaluation and bidirectional pruning mechanism—this mechanism enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches both before and after tool execution. Empirical evaluations conducted by the authors across both open-set and closed-set tool planning tasks demonstrate that ToolTree consistently improves performance while maintaining the highest efficiency.

### Strengths
1.  Bidirectional pruning enables precise cost control by filtering low-value branches, and a dedicated caching mechanism minimizes redundant computations.
2.  Without depending on task-specific training, the framework maintains robust adaptability across diverse tool libraries—thus eliminating the need for retraining when switching between different tool sets.
3.  Evaluations spanning both closed-set and open-set tool planning tasks across 4 benchmarks consistently demonstrate the framework’s superior performance.

### Weaknesses
1.  The reliability of both *r\_pre* and *r\_post* hinges entirely on the LLM’s ability to assess tool relevance and output quality, introducing risks if the LLM misjudges context or utility.
2.  When handling an extremely large number of open-set tools, pre-evaluation incurs significant sorting overhead, as the system must process and rank a massive volume of tool candidates.

### Questions
please refer to Weaknesses

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
This paper proposes ToolTree, a planning-time framework that casts multi‑tool orchestration for LLM agents as an MCTS-style search augmented with dual LLM feedback.

### Strengths
1. This paper is well written and easy to follow

2. The prior‑augmented UCT and bidirectional pruning are straightforward and training‑free.

### Weaknesses
1. This method is somehow similar to ToolChain*; needs more clarification on the difference and comparison.

2. The method’s post‑evaluation judge (used during planning) may be architecturally similar to, or the same family as, the benchmark evaluator (Pass/Win are also judge‑based in ToolBench/RestBench). Even with version pinning, this can unintentionally optimize for the judge rather than ground truth. The authors partially probe judge choice but further cross‑judge / cross‑vendor sanity checks would reduce concerns about metric coupling.  

3. The paper claims budget parity and shared prompts/tool sets, but some baselines (e.g., LATS, ToT) might benefit from comparable pre‑gating by schema/type or caching; it’s unclear whether the authors provided similarly favorable engineering to competing methods.

### Questions
1. There are several minor typesetting errors (e.g., “Eq. equation 1”, “backward‑prorogation”, “Table ??”) and occasional ambiguity in the formalism in §2–§3, which can impede exact reproducibility absent code.

### Soundness
3

### Presentation
3

### Contribution
2
