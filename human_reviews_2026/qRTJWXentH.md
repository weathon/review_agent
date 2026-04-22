# t-BEN: A Temporal Logic Guided Approach for Temporal Reasoning Benchmark Generation

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 8, 4

## Abstract
In logic-based Artificial Intelligence (AI), temporal reasoning typically involves formalizing problems as logical rule expressions and employing symbolic reasoners to infer and derive new conclusions from structured knowledge. However, symbolic reasoners generally cannot process natural language directly and require manually constructed symbolic knowledge bases, which can be both time-consuming and resource-intensive to create and maintain. Given the recent widespread adoption of Large Language Models (LLMs) and their remarkable successes across diverse domains, we are motivated to explore to what extent LLMs can handle temporal logic tasks, dispensing with traditional symbolic reasoners.

We introduce t-BEN, a benchmark suite that strictly adheres to the semantics of temporal logic. It automatically synthesizes temporal reasoning datasets in both symbolic and natural language forms, enabling the evaluation of Large Language Models (LLMs) on temporal logic reasoning. t-BEN is a highly scalable benchmark that supports the generation of datasets with varying sizes and rule structures of varying complexity. Furthermore, each question in t-BEN is guaranteed to be unseen by LLMs during pretraining, effectively minimizing the risk of data leakage. Our results, along with a detailed ablation study of seven frontier LLMs, offer valuable insights into the capabilities and limitations of current models in temporal logic reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces T-BEN, a new benchmark to test how well LLMs handle temporal logic reasoning. It works by automatically generating fresh, logic-based problems that models have not seen during training, which is a good way to test true reasoning instead of memorization. The most interesting finding is that DeepSeek-R1 is exceptionally good at these tasks, while many other top models, including GPT-4o, perform poorly, often no better than random guessing.

### Strengths
- It maybe the first benchmark of its kind, filling a major gap in how we evaluate formal reasoning in LLMs.
- The method is solid. Generating new problems on the fly is smart, and the tiered difficulty (from simple to recursive rules) is sound for finding the breaking points of different models.

### Weaknesses
- The benchmark focuses only on one specific type of logic (DatalogMTL). The findings might not apply to other important temporal logics used in fields like formal verification.
- The paper suggests why DeepSeek-R1 is so good (its training method), but this is treated as a hypothesis without direct experimental proof.
- The problems are abstract and machine-generated. It is unclear how well performance on T-BEN would translate to messy, real-world tasks that require temporal understanding (e.g., interpreting medical records or financial reports).

### Questions
1. The DeepSeek-R1 result is fascinating. Can you elaborate further on why you believe its training strategy is the key factor? Are there any other plausible explanations for its unique ability?
2. How do you see the skills tested in T-BEN transferring to practical, real-world applications? Is there a clear path from solving these abstract puzzles to solving real problems?
3. Have you considered extending this framework to other temporal logics like LTL or CTL? This could help determine if DeepSeek's ability is specific to DatalogMTL or represents a more general strength in formal reasoning.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces t-BEN, a synthetic benchmark suite for evaluating LLMs on temporal logic reasoning. Each instance consists of temporal data, rules written in DatalogMTL, and a single fact-entailment query. The benchmark is intentionally semantics-grounded, supports both symbolic and natural-language variants of the same task (via rule verbalization), and is designed to be scalable and data-leakage-resistant through on-the-fly randomized generation. The authors define six rule-complexity “levels” (SingleAtom, MultiAtoms, Rational timeline, MixedOperators, MultiRules, Recursive) and describe a three-stage generator (graph construction, data generation, rule generation) with verification using the MeTeoR reasoner. Experiments across 13 models report strong performance for “reasoning” models (notably o3, Gemini 2.5 variants, DeepSeek-R1) and weaker results for general LLMs (e.g., GPT-4o without CoT); recursive and multi-rule settings are the hardest. The authors include ablations on number of relevant rules, operator variety, and injected distractors, and compare symbolic vs. NL variants (similar difficulty; symbolic slightly easier). 

Soundness: 
The paper addresses a noticeable lack of symbolic temporal logic reasoning benchmarks for LLMs. While previous benchmarks have attempted to measure formal reasoning abilities of these models in first-order logic, the extent of their capabilities on temporal logic reasoning has yet to be examined. The conceptualization of fact entailment as a fundamental temporal logic reasoning task is a sound choice that enables the synthesis of a broad range of temporal logic reasoning problems. Lastly, the adoption of DatalogMTL over less expressive temporal logic variations encourages future work towards deployable LLM-based temporal logic reasoning systems.

Presentation: 
The work is well written and establishes most of the preliminary information needed to contextualize the proposal. The style and notation are largely consistent with related works. It would be nice to include one additional overview figure to better describe dataset generation.

Contribution:
The introduction of a dataset which can measure temporal logic reasoning abilities of language models is a novel and timely contribution to the field of symbolic reasoning integration with language models. As previously stated, there are have been recent efforts in benchmarking various aspects of symbolic reasoning with LLMs, but none have yet covered the domain of (metric) temporal logics.

### Strengths
1. Sound choices in problem framing, including the use of DatalogMTL for its expressive semantics and breadth.
2. The benchmark is the first to evaluate temporal logic fact-entailment in a semantics-grounded setting and the first to support symbolic and NL variants tied to the same underlying task.
3. The benchmark generator includes correctness guarantees via MeTeoR. Results cover 13 models, several prompting settings, and detailed ablations (rule depth, distractors, operator counts). Recursive and multi-rule difficulty trends are empirically demonstrated.

### Weaknesses
1. The six “levels” are not formally characterized by computational hardness. This is acknowledged, but even a short theoretical appendix benchmarking against known DatalogMTL fragment complexity classes would add rigor.
2. Although symbolic methods can't run on NL prompts, comparing their runtime or success rates for the symbolic subset (vs. LLM success modes) would help position LLMs as complements vs. replacements.

### Questions
While the authors acknowledge synthetic focus, the paper does not explore bridging to real-world domains (e.g., planning logs, event-stream QA, formal verification logs). If the author can provide a small illustrative case, this would strengthen applicability.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a synthetic symbolic dataset for temporal reasoning based on the language DatalogMTL, an extension of Datalog with metric temporal logic. Rules in the dataset are classified into six types. The paper evaluates the performance of seven LLMs with this dataset, and results show that LLMs including GPT-4o exhibit poor performance, but LRMs including DeepSeek-R1 deliver strong results.

### Strengths
1. The proposed dataset can potentially serve as a testbed for improving LLMs' temporal reasoning abilities. 

2. Experimental results show that LRMs may serve as alternatives to traditional symbolic reasoners.

### Weaknesses
1. Although the dataset also has a natural language form, it is generated with a template-based approach, and hence lacks the linguistic complexity of real-world natural language. Thus the paper sheds limited insight on LLMs' abilities of  temporal reasoning with natural language.  

2. Although experimental results show that LRMs may serve as alternatives to traditional symbolic reasoners, the paper does not do an experimental comparative analysis between the temporal reasoning abilities of LRMs and symbolic reasoners, for example, in terms of reasoning speed. Thus it is not clear in what sense LRMs can serve as complementary tools to symbolic reasoners. 

3. Despite the weak performance of LLMs on this dataset, the paper does not discuss possible ways to improve the temporal reasoning abilities of LLMs. 

4. Related work section should focus on closely related works to make clear the contributions of this paper, namely works on evaluating LLMs' temporal reasoning abilities such as (Wang & Zhao 2024), (Xiong et al. 2024) and (Fatemi et al. 2025). However, the discussion of these closely related works is very limited. 

With the above limitations, I feel the significance of the proposed dataset is unclear.

### Questions
Could you comment on the issues raised in the limitations part?

### Soundness
3

### Presentation
2

### Contribution
2
