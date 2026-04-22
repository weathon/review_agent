# Towards the Generation of  Structured Scientific Vector Graphics with Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
We address the challenge of automatically visualizing scientific explanations. While prior work has explored large language model (LLM)-based vector graphic generation, existing approaches often overlook structural correctness, a key requirement for valid scientific diagrams. To achieve structurally correct generation, we make three key contributions. First, we introduce SSVG-Bench, a novel benchmark for evaluating the generation of Structured Scientific Vector Graphics. Unlike conventional visual similarity metrics, SSVG-Bench employs task-specific structural analysis for accurate evaluation, and it supports three vector formats: TikZ, SVG, and EPS. Second, we conduct an extensive benchmarking and analysis, revealing key findings such as the crucial role of LLM reasoning in ensuring structural validity. Third, we propose LLM-Oriented Orchestration Prompting (LOOP), a new prompting method that leverages LLMs' reasoning potential by combining familiar subtasks. Experiments demonstrate substantial improvements over existing prompting techniques, suggesting promising directions for scientific diagram generation. We will release our code and benchmark upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper fills the gap of evaluating the structural correctness of text-guided scientific figure generation, which previous work overlooks. Evaluating structural correctness is not easy to do with fully automated methods. Instead, the authors focus on a specific subset of scientific figures (plane geometry and molecular structure). They derive a benchmark of 410 text-figure pairs and implement rule-based programs that evaluate the correctness of generated outputs. The authors evaluate a range of existing fine-tuned and general-purpose models on this benchmark using three output formats (TikZ, SVG, and EPS). In addition, they introduce a new task-specific prompting method that can help improve performance.

### Strengths
* The paper is very well written and easy to understand.
* Structural correctness is an important property for the evaluation of scientific figure generation that is often omitted from evaluations due to the difficulty of assessing it. The provided benchmark fills this gap and will be useful for future work.
* The evaluation compares a wide range of models across three formats, which provides interesting insights.
* The provided prompting technique may be useful for future work and applications.

### Weaknesses
* Table 3 is an interesting approximation of potential occurrences in the training data, but a TikZ graphic doesn't necessarily have to start with `\documentclass[tikz]`.
* The rule-based programs seem central to the benchmark, yet no details or example snippets are provided in the paper, which would have been insightful for assessing how well they work or whether they are brittle with failure cases.
* Although the benchmark will be very useful when released, there is very little technical novelty in the paper, and while the introduced prompting technique seems to work well and leads to additional improvements, it is hardly exciting.
* The authors motivate their benchmark by stating that human evaluation doesn't scale well (l.129), but the creation of the benchmark still requires heavy manual curation (l.209ff), so it still doesn't scale well.
* The performance of fine-tuned models (AutomaTikZ, TikZero) seems surprisingly low. Have the authors ensured that the provided prompts are in the format the models expect? At least the example prompts provided in the appendix are not in the correct format.

### Questions
* In l.242, it says that parts of the output are provided to the models as input. How is this done exactly, as this is not clear from the prompt examples in the appendix? Furthermore, how is this provided for models that are fine-tuned and do not accept general-purpose prompts?
* Why are the scores of the TikZ models in Table 4 provided in the SVG column?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces **SSVG-Bench**, a benchmark to evaluate **structural correctness** in scientific vector-graphics generation (TikZ/SVG/EPS), provides **automatic structure-aware evaluators** for plane geometry and molecular structures, benchmarks many LLMs, and proposes **LOOP**, a prompting workflow that improves accuracy, especially in SVG.

### Strengths
- **Clear problem motivation:** moves beyond visual/code similarity to **structure-aware** evaluation.
- **New benchmark & tooling:** SSVG-Bench (two tasks; three formats) with **Python scripts for structural checks**
- **Broad, timely evaluation:** compares fine-tuned TikZ models and recent LLMs; reveals the importance of **reasoning modes**.
- **Actionable insight on formats:** compelling evidence that **SVG > TikZ/EPS** for LLM reasoning; novel angle likely to influence future work.
- **LOOP** (information/relationship extraction → reasoning → code) yields **consistent gains** over popular CoT variants.
- Prompts, task setup, and evaluation logic are described in detail; many examples illustrate successes/failures.

### Weaknesses
- **Evaluation blind spots:** plane-geometry scorer does not penalize **extraneous elements**; results may overstate correctness in cluttered outputs.
- **Bond order ignored:** molecular task collapses bond multiplicity; risks awarding correctness to chemically different graphs.
- **Potential data leakage:** plane-geometry items sourced from Wikipedia/SVG Commons; large web-trained LLMs may have seen near-identical diagrams/captions.
- **Scope & scale:** overall size (1,230 items) is moderate; per-topic diversity (e.g., non-Euclidean, circuits, algorithmic flowcharts) could be broader.
- **Ablations on LOOP:** helpful but could be deeper (e.g., remove each stage, vary decomposition granularity, measure latency/cost).

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This paper addressed the structural correctness issue in the LLM's vector graphics generation. It provides a benchmark called SSVG-Bench consisting of two types of tasks, plane geometry task and molecular structure task, along with novel evaluating scripts to evaluate the structural correctness of the generated images. It performs a comprehensive benchmarking and analysis of existing models on the proposed benchmark, revealing the poor performance of LLM and key feature that might enhance a LLM's capability of generating correct vector graphics. Finally, it proposed LLM-Oriented Orchestration Prompting (LOOP), a method that enhances the accuracy of vector graphics generation in LLM. Experiments result shows that the proposed LOOP can improve the performance in terms of structural correctness on it's proposed benchmark.

### Strengths
1. The proposed evaluation method is novel. Instead of using common metrics in computer vision, the author proposed using scripts to evaluate the correctness of scientific vector graphics. This makes the evaluation process more explainable and accurate.
2. The evaluation in Table 4 is comprehensive, covering all recent SOTA models. 
3. There's a significant improvement in performance with the proposed LOOP strategy.

### Weaknesses
1. The diversity of the proposed benchmark is greatly limited. It only contains two kinds of tasks, plane geometry task (with only 5 elements) and molecular structure task. Both tasks have a very clear and fixed path to solve, and therefore, might not be able to test the model's generalizability on all other tasks requiring structural correctness. 
2. Table 1 missed lots of recent benchmark (even those benchmark are mentioned in the text from L105-L107, and lots of them are larger than the proposed benchmark). For example, the generation evaluation suite of VGBench contains 5845 instances in total, and SVGEditBench has 1366 instances in total. Not comparing them in Table 1 is unfair. 
3. Molecular structure is an extremely specialized task. It not only evaluates the ability to generate graphics, but also evaluate the model's understanding of IUPAC name and the model's chemistry knowledge. The requirement for such specialized knowledge presents a bias so that the results will be better for models with chemistry knowledge than models without, while the chemistry knowledge is not usually related to the model's ability to generate vector graphics in general. 
4. There is no evaluation on the proposed LOOP strategy's performance on other vector graphics generation benchmark. Evaluating it on other commonly used benchmark will ensure the method's generalizability.

### Questions
1. How to ensure the robustness of the Python script for Pattern 2? Are we using a different script for each case or using the same script for all cases?
2. In L241, it's mentioned that "In Pattern 1, the correct output can be uniquely determined", Why it can be uniquely determined? Even with the given element (in black), the remaining element (in red) can still have multiple ways of expression. For example, the circle can be represented as circle using `<circle>`, or represented as curve using `<path>` in SVG.

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
4

### Summary
The paper introduced SSVG-Bench, a benchmark for generating structured scientific vector graphics from text, covering plane geometry and molecular structures across three vector formats (TikZ, SVG, and EPS). It applied Python-based automatic accuracy evaluation and reports results for multiple LLMs. The paper also proposed a prompting method, LOOP, which yields measurable gains on the benchmark.

### Strengths
1. Clear, well-structured writing with concrete, illustrative examples that make the task and setup easy to follow.
2. The automated molecular evaluation via graph isomorphism is sensible.
3. Broad coverage of vector formats (TikZ, SVG, EPS) and a diverse model suite, comparing both reasoning and non-reasoning LLMs.
4. The proposed prompting method (LOOP) is simple to implement and yields consistent gains.

### Weaknesses
W1. Insufficient technical detail for automatic plane-geometry evaluation (Pattern 1/2): missing normalization pipeline, tolerance settings, red-element extraction, alignment strategy, and failure modes. This undermines the trustworthiness and reproducibility of the reported Accuracy.


W2. Table 4 inconsistency: AutomaTikZ and TikZero+ are trained on TikZ, yet the TikZ column is empty while SVG numbers are reported. Is this a mistake or a data-availability issue? Please clarify/correct.


W3. Over-reliance on a single binary metric (Accuracy): no complementary automatic metrics or human evaluation, leading to an overly coarse assessment; near-misses and completely wrong outputs are both scored 0. Consider adding metrics used in related work TikZero+ (e.g., DreamSim, KID, CLIPScore, code-level CrystalBLEU and TEX Edit Distance, Mean Token Efficiency) and geometry-specific measures (e.g., how many elements are correctly covered?)


W4. Narrow evaluation of the prompting strategy: results are shown primarily on GPT/Gemini and mostly as Accuracy; broader model coverage and multi-metric reporting are needed.
Lack of human evaluation: no user study to validate automatic metrics, resolve borderline cases, or assess readability/usability of the generated diagrams.

### Questions
Q1. Pattern taxonomy: How exactly categorize plane-geometry items into Pattern 1 (unique solution) vs Pattern 2 (multiple valid solutions), and what is the split (%) across the 110 examples?
Q2. Pattern 2 evaluation cost: Does Pattern 2 require case-specific code per instance? 
Q3. Prompting generality: LOOP appears to be a fixed “think” scheme—do its gains differ between reasoning and non-reasoning models? Q4. Please provide results on more models and with more metrics (as above)

### Soundness
2

### Presentation
3

### Contribution
2
