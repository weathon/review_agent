# LogiConBench: Benchmarking Logical Consistencies of LLMs

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Logical consistency, the requirement that statements remain non-contradictory under logical rules, is fundamental for trustworthy reasoning, yet current LLMs often fail to maintain it even on simple inference tasks. Existing benchmarks for LLM logical consistency are not scalable, not diverse, and not challenging, with state-of-the-art models already surpassing 95\% accuracy. LogiConBench is the first benchmark that (1) generates unlimited logical rule combinations with precise labels, (2) provides controllable-depth graphs with explicit reasoning paths, and (3) remains challenging for state-of-the-art LLMs. To achieve this, LogiConBench automatically generates logical graphs where nodes represent symbolic propositions and edges denote reasoning relations. From these graphs, it samples lists of propositions, extracts reasoning paths, determines all consistent label lists, and translates them into diverse natural language expressions. While we release a 280K-sample corpus in this work, the framework can be scaled to generate unlimited data. To strengthen its evaluative significance, we evaluate 14 frontier LLMs on three tasks with varying difficulty levels, and find that the Enumerative task remains extremely challenging, with the best exact accuracy as only 34\%. Our code and data are available at https://github.com/Bellafc/LogiConBench.git.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces LogiConBench, a scalable benchmark for testing logical consistency in LLMs. A four‑stage pipeline first generates logical graphs from natural‑deduction rules, then samples some target nodes under bounded distance, propagates truth values along extracted reasoning paths to derive all consistent/inconsistent label lists, and finally verbalizes symbolic formulas into natural language via templates and WordNet substitutions. The authors evaluate 14 models, empirically finding that frontier models reach 85–95% on Task 1, whereas Task 2 remains difficult.

### Strengths
* Data‑generation procedure appears methodologically sound: edge‑level truth‑compatibility for each operator is explicitly specified; Steiner‑tree extraction bounds depth while preserving connectivity among targets; the DFS enumerator yields exact, reproducible labels; and NL verbalization is decoupled from labeling, isolating the resulting problems from commonsense knowledge.
* Very comprehensive evaluation across many models and many difficulty slices: 14 LLMs (open/proprietary), 2–5 statements, with rich metrics.
* Even frontier models remain far from solved on Task 2: most models have Exact < 1%; best is gpt‑5 Exact≈0.51 with reasoning paths, while prior datasets are near‑ceiling (Figure 1a). This makes LogiConBench a timely and progress‑sensitive benchmark in an era when many others have saturated

### Weaknesses
* It is hard to gauge how diverse the generated logical problem patterns are. The study used the introduction rules for five logical operators but did not use elimination rules, which are also crucial in symbolic logic. Is it OK?, how diverse are the resulting patterns? Although we understand that defining and measuring the diversity of problem patterns is subtle and challenging.

### Questions
* A key advantage of your framework is that it can scale to generate unlimited data. Given this, is it possible to leverage large‑scale RL on synthetically generated splits to train, rather than benchmarking, reasoning capabilities?
* If I understand the tasks correctly: If we had a reliable Task 1 verifier, then in principle Task 2 could be solved by evaluating all 2^k assignments with Task 1 and collecting the consistent ones. Why do LLMs still struggle so much with Task 2? Is the main bottleneck long‑horizon search/planning, cumulative local errors, output‑formatting limits, or something else?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LogiConBench, a new benchmark designed to evaluate the logical consistency of Large Language Models (LLMs). The authors argue that existing benchmarks for logical consistency are no longer sufficient, as they lack scalability, diversity, and challenge; SOTA models, for instance, already achieve over 95% accuracy on them.
To address this, LogiConBench provides a framework including:
1. Automatically constructing graphs where nodes represent symbolic propositions and edges represent reasoning relations.
2. It samples sets of propositions (nodes) from these graphs, extracts the reasoning paths connecting them, and determines all possible Boolean (True/False) assignments that are logically consistent.
3. These symbolic propositions are then translated into diverse natural language sentences using templates and lexical substitution from WordNet.

The paper releases a 280K-sample corpus and evaluates 14 frontier LLMs on Discriminative(The model must determine if a given list of Boolean labels for a set of statements leads to a contradiction) and Enumerative(The model must enumerate all possible lists of Boolean labels that make a set of statements consistent) tasks.

### Strengths
1. The paper convincingly argues that existing benchmarks for logical consistency are "saturated". The results shown in Figure 1a demonstrate that SOTA models easily pass older benchmarks (like BeliefBank and LFC) but are significantly challenged by LogiConBench, proving its necessity.
2. Unlike static, human-written datasets, the LogiConBench framework is automatic and can generate "unlimited logical rule combinations". This graph-based generation pipeline is a novel approach that ensures a large-scale, diverse, and renewable source of evaluation data.
3. A key feature is the benchmark's ability to provide "controllable-depth graphs with explicit reasoning paths". The authors show that providing these paths as prompts ("few-shot with reasoning paths") improves model performance, especially on the difficult Enumerative task. This highlights the value of explainability and chain-of-thought data.
4. The study is thorough. It tests 14 frontier closed and open-source models across multiple evaluation settings (zero-shot, few-shot, and few-shot with path).

### Weaknesses
1. The natural language generation relies on templates and lexical substitution. While this allows for scale, the resulting sentences may lack the nuance, ambiguity, and diversity of "fully human-authored datasets". This could make it easier for models to map the text back to symbolic logic, potentially simplifying the natural language aspect of the reasoning. At the same time, it is hard not to suspect that the constructed natural language data contains content that is consistent with or contrary to the real world, such as "the sky is blue". This situation may cause problems such as reasoning shortcuts or knowledge contradictions, affecting the evaluation results.
2. Meanwhile, I believe that the task form of Enumerative task cannot well reflect the inherent logical ability of the model, because it is well known that exhaustive search has always been a serious problem for LLMs, but this does not completely mean that the model's logical level is low.
3. The lack of detailed and profound analysis of the different performances of the model on various tasks makes it difficult to reflect the necessity and value of designing these two types of tasks.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
2

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
The paper proposes LogiConBench, a benchmark that evaluates LLMs on logical consistency. It addresses three key limitations of existing benchmarks by generating scalable, diverse, and challenging problem instances. These instances are created through logical graph generation, reasoning path labeling, and translation into natural language. The benchmark is tested on 14 frontier models. The discriminative tasks are relatively easy, while the enumerative tasks remain highly challenging. The paper also introduces several task variants with intermediate levels of difficulty.

### Strengths
- The paper is well-written and clear.
- The motivation is strong, and the proposed benchmark directly addresses the main shortcomings of existing logical consistency benchmarks in terms of scalability, diversity, and difficulty.
- The evaluation is extensive, covering both instruction-tuned and reasoning models, including closed- and open-source models, under three prompting setups (zero-shot, few-shot, and few-shot with reasoning paths).

### Weaknesses
- The benchmark includes only two main tasks (discriminative and enumerative). Although a few variants are explored, they remain closely related to the original tasks. The benchmark could benefit from more diverse task types to better capture different dimensions of logical consistency.
- The qualitative analysis is somewhat limited. While Section 5.3 briefly discusses three main failure modes, a deeper qualitative examination of the reasoning paths behind these errors could significantly strengthen the paper and yield more actionable insights for researchers and practitioners.

### Questions
- How are model outputs extracted for scoring? The process seems straightforward for the discriminative task (as it produces a binary answer), but it is less clear for the enumerative task, where answer formatting and extraction can be challenging. More details or examples of output parsing and scoring would be needed.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces LogiConBench, a scalable benchmark for testing LLMs’ logical consistency through automatically generated logical graphs with controllable reasoning depth. It supports unlimited rule combinations and diverse natural language formulations. Evaluations on 14 leading LLMs show that despite high accuracy on prior benchmarks, models still struggle on LogiConBench’s Enumerative task.

### Strengths
1. The paper presents a scalable and systematic data generation pipeline capable of synthesizing a large number of well-labeled evaluation samples. This framework allows for controlled complexity, diverse logical structures, and extensibility to new reasoning settings.
2. The authors conduct comprehensive and fair evaluations across a wide range of baseline models, covering both open-source and proprietary models.

### Weaknesses
1. There are concerns regarding the true difficulty of the proposed benchmark. In the Discriminative Task (Task 1), models such as GPT-5 and Grok-4-Fast already achieve near-perfect performance, while in the Enumerative Task (Task 2), their F1 scores reach around 70. This raises the question of whether frontier models have largely mastered the benchmark, suggesting that the observed gap for open-source models might simply reflect scaling effects rather than a fundamentally unsolved challenge.
2. The paper does not sufficiently articulate the real-world significance of LogiConBench. It remains unclear which practical abilities the benchmark aligns with. Establishing stronger connections between LogiConBench performance and real-world downstream tasks (e.g. agents) would help clarify its broader impact and necessity.
3. Since the benchmark’s data generation pipeline is fully scalable, one might argue that it is also straightforward to construct large volumes of in-domain synthetic data for targeted finetuning. This raises doubts about whether the reported difficulty truly reflects inherent reasoning limitations or can be easily mitigated through data augmentation.

### Questions
please refer to weakness

### Soundness
2

### Presentation
3

### Contribution
2
