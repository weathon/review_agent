# Exploring the Meta-level Reasoning of Large Language Models via a Tool-based Multi-hop Tabular Question Answering Task

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Recent advancements in Large Language Models (LLMs) are increasingly focused on "reasoning" ability, a concept with many overlapping definitions in the LLM discourse. We take a more structured approach, distinguishing meta-level reasoning (denoting the process of reasoning about intermediate steps required to solve a task) from object-level reasoning (which concerns the low-level execution of the aforementioned steps.) We design a novel question answering task, which is based around the values of geopolitical indicators for various countries over various years. Questions require breaking down into intermediate steps, retrieval of data, and mathematical operations over that data. The meta-level reasoning ability of LLMs is analysed by examining the selection of appropriate tools for answering questions. To bring greater depth to the analysis of LLMs beyond final answer accuracy, our task contains 'essential actions' against which we can compare the tool call output of LLMs to infer the strength of reasoning ability. We find that LLMs demonstrate good meta-level reasoning on our task, yet are flawed in some aspects of task understanding. We find that n-shot prompting has little effect on accuracy; error messages encountered do not often deteriorate performance; and provide additional evidence for the poor numeracy of LLMs. Finally, we discuss the generalisation and limitation of our findings to other task domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a new QA system to evaluate the meta-reasoning capability of LLMs by supervising the intermediate reasoning chain by examining the correctness of core tools selected.

### Strengths
- This paper proposes a new method to validate the meta-reasoning ability of LLMs.
- The paper tests several models using their benchmark.

### Weaknesses
- Table 2 exceeds the width of the paper template, which should not appear.
- The benchmark needs human annotation to ensure its fairness and reasonableness. However, no relevant statistics are provided.
- The baseline used are mainly prompt engineering. More updated baselines should be included.

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposed a tabular benchmark to evaluate meta-level reasoning of LLMs via a multi-hop tabular QA task. It includes human-defined tools and question templates on World Bank indicator data. Several open source or proprietary are evaluated via zero-shot and few-shot methods, showing gap on the proposed benchmark.

### Strengths
1. Problem framing: Clear split between object-level and meta-level, which is useful for capability-wise diagnosis and future extensions.

2. Beyond accuracy, the paper reports precision/recall (±1σ) and an Err. rate (share of outputs with at least one tool-call error), capturing both process robustness and final correctness.

3.  Comparisons across 0/1/3-shot, error vs. no-error conditions, and tool-set ablations (all tools vs. data-retrieval only) reveal more perspectives.

4. Clarity. Method and experimental setup are straightforward and easy to follow.

### Weaknesses
1. The benchmark relies on templated prompts and relatively narrow tools APIs, which is too simple compared to current practice, e.g., richer coding/spreadsheet/enterprise interfaces such as in SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets (https://arxiv.org/abs/2510.19247).

2. The study omits stronger/ newer models (e.g., GPT-5, Gemini 2.5 Pro) and larger mainstream open models (e.g., newer Qwen3 variants), making it hard to gauge the field’s current frontier.

3. Insufficient error study. Beyond aggregate Acc./Precision/Recall and brief text, there is no systematic analysis of agent failure causes across models, nor full examples that illuminate the paper’s targeted meta-reasoning errors.

4. Missing citations. The paper mentions but does not cite: ReAct: Synergizing Reasoning and Acting in Language Models (https://arxiv.org/pdf/2210.03629.pdf)). Newer multi-step reasoning benchmarks such as LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models (https://arxiv.org/abs/2404.05221)) should also be surveyed and cited.

### Questions
1. Could you provide a systematic error study separating meta-reasoning and object-level reasoning?

2. Could you report evaluation cost and time?

3. How does advance models such as GPT-5 perform on the benchmark?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper distinguishes meta-level reasoning (planning steps and choosing tools) and object-level reasoning (executing retrieval and arithmetic steps), and introduces a tool-based multi-hop tabular QA benchmark from "World Bank indicators". Each question includes a set of "essential actions" used to evaluate models by their tool-call traces rather than final answers alone. Findings show that off-the-shelf LLMs often select appropriate tools, which means good meta-level reasoning, but may miss steps or hallucinate codes; providing 1/3-shot tool-use examples rarely improves accuracy, while error messages often help recovery; removing arithmetic tools significantly degrades performance, underscoring weak numeracy and the importance of external symbolic functions.

### Strengths
(1) The paper offers a deep exposition and evaluation of meta-level reasoning—specifically, a model’s abilities in task decomposition and task planning.

(2) Abstract concepts are described clearly, and the overall writing is fluent and easy to follow.

### Weaknesses
(1) A fundamental concern: in modern LLMs, it is difficult to cleanly separate “meta-level” and “object-level” reasoning in the symbolic-AI sense. For complex tasks in real world, most steps blend both forms of reasoning. The paper’s “first decompose, then execute” setup seems best suited to relatively simple multi-hop tasks (e.g., retrieval-augmented or calculation-augmented), and may not generalize to richer scenarios.

(2) The evaluation set is too narrow. Although the paper notes that the approach could extend to other domains, evidence based mainly on a word-bank dataset and a few simple tool-use tasks is insufficient to substantiate the broader claims.

(3) Formatting: Table 2 overflows the text bounds on the right and needs layout fixes.

### Questions
(1) When repeated calls with identical arguments are counted as one TP and the remainder as FPs, does this risk over-penalizing exploratory strategies that legitimately probe the tool state or recover from uncertainty?

(2) When keeping only retrieval and removing arithmetic tools, is the observed performance drop systematically related to model size or the reasoning mode?

(3) The chosen primary area “Datasets and Benchmarks” may not be a good fit?  This work reads more as an analytical/argumentative study than as a generally applicable, transferable benchmark.

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
This paper introduces a novel question-answer benchmark that analyzes geopolitical information to evaluate the meta-level reasoning capabilities of LLMs. The questions require LLMs to decompose tasks into intermediate steps and apply data retrieval and mathematical operations. The dataset provides ground truth answers that include the essential actions needed to derive correct responses, enabling systematic evaluation of LLM performance.

### Strengths
1. The meta-level reasoning capability in the paper is defined as the high-level planning capability. The question design is highly consistent with this design goal. The proposed question construction format can be easily adapted to domains other than geopolitical. 
2.  The dataset provides the correct sequence of essential actions, which enables assessment of partial correctness in LLM responses, thus offering a more nuanced evaluation of reasoning capabilities than binary success/failure metrics labels.
3.  The response post-processing method mitigates the influence of textual differences on semantically equivalent actions.

### Weaknesses
1. The task combines high-level planning with tool use, where LLMs need to understand tool call rules from the prompt.  LLM performance depends on both planning and tool-calling capabilities. Therefore, the evaluation may be influenced by confounding factors.
2. The presented metrics are precise and recall count for the content overlap between the LLM-generated action and ground-truth essential actions. For some cases, the action sequence is order-dependent, meaning different orderings of the same actions will lead to different results. Therefore, the precision and recall scores may not be robust

### Questions
1. I’m wondering, if providing paragraphs including both relevant and irrelevant information for the question (similar to HotpotQA), then LLM is asked to generate natural language inference steps to derive the final answer. Compared with the tool calling task, which setup provides a more precise estimation of the meta-level reasoning capability?

### Soundness
3

### Presentation
3

### Contribution
3
