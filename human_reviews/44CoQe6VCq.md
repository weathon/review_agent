# Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Large language models (LLMs) have showcased remarkable reasoning capabilities, yet they remain susceptible to errors, particularly in temporal reasoning tasks involving complex temporal logic. Existing research has explored LLM performance on temporal reasoning using diverse datasets and benchmarks. However, these studies often rely on real-world data that LLMs may have encountered during pre-training or employ anonymization techniques that can inadvertently introduce factual inconsistencies. In this work, we address these limitations by introducing novel synthetic datasets specifically designed to assess LLM temporal reasoning abilities in various scenarios. The diversity of question types across these datasets enables systematic investigation into the impact of the problem structure, size, question type, fact order, and other factors on LLM performance. Our findings provide valuable insights into the strengths and weaknesses of current LLMs in temporal reasoning tasks. To foster further research in this area, we will open-source the datasets and evaluation framework used in our experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors introduce two novel synthetic datasets, TOT-Semantic and TOT-Arithmetic, specifically designed to evaluate LLMs’ temporal reasoning abilities with graph-like facts from two perspectives: (1) understanding the semantics and logic of time, and (2) performing accurate temporal arithmetic. The authors also conduct extensive experiments to examine how LLM performance is influenced by the graph structure, graph size, question type, and fact ordering of the problem.

### Strengths
The method works on temporal reasoning with LLM, an important area of research that contributes to understanding the model's overall complex reasoning capabilities.

The authors conduct several experiments. Their analysis and the data offer valuable insights for future research.

### Weaknesses
The paper lacks detail on dataset construction. For instance, how are the final questions generated in both TOT datasets? Are templates being used? (see also question 1)

The number of baselines is limited. Additional approaches could include directly generating code for TOT-Arithmetic or applying few-shot or self-consistency.

### Questions
1. Have the authors considered how the format of the date/time, such as words versus numerical format, might influence the model’s performance?

2. For 4.1, 4.1.1, what task does the author evaluate?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Dealing with the dataset quality and potential leakage problems, this paper introduces a novel method to synthesize a benchmark for comprehensive temporal reasoning benchmarks. The benchmark contains semantic and arithmetic questions with fine-grained topology control. Extensive experiments are conducted and show insightful conclusions.

### Strengths
- The data synthesis process benefits from the graph-guided control, and could be generalized to many other tasks.
- The constructed data are comprehensive and include many perspectives with quality control.
- Experiments are extensively conducted on multiple aspects, and provide some insights on future directions.

### Weaknesses
- Some claims lack of quantitative evidence:
    - “real-world data that LLMs may have encountered during pre-training or employ anonymization techniques that can inadvertently introduce factual inconsistencies” Could you add some quantitative evidence showing the GPT-4 or Gemini-1.5 Pro baselines have pre-training data contaminations?
    - “LLMs could even potentially guess the original entities due to their adjacent relations” This also lacks of quantitative evidence. If this is a commonsense, there should be relevant references cited.
- The literature review is not sufficient, and there are many researches on math-related temporal reasoning tasks. There lacks of relevant references in the introduction and the related work.
    - Wang, Y., & Zhao, Y. (2023). Tram: Benchmarking temporal reasoning for large language models. *arXiv preprint arXiv:2310.00835*.
    - Chu, Z., Chen, J., Chen, Q., Yu, W., Wang, H., Liu, M., & Qin, B. (2023). Timebench: A comprehensive evaluation of temporal reasoning abilities in large language models. *arXiv preprint arXiv:2311.17667*.
    - Su, Z., Zhang, J., Zhu, T., Qu, X., Li, J., Zhang, M., & Cheng, Y. (2024). Timo: Towards Better Temporal Reasoning for Language Models. *arXiv preprint arXiv:2406.14192*.

### Questions
- Some details are missing.
    - Line 212: “we generated questions per graph generation and per question type”: Please explain how to generate such questions. Are they generated from templates, manual annotations, or LLMs?
    - Line 369: Is it because the superior performance on longer contexts? Is there a correlation between long-context performance (or overall task performance e.g., MMLU, GSM8K, MATH500) and the final temporal reasoning performance? Are there sufficient test cases with more edges for providing robust evaluation?
- Typos:
    - Line 275: Funcionalizing → Functionalizing

### Soundness
3

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
4

### Summary
This paper focuses on evaluating the temporal reasoning abilities of large language models (LLMs). The authors introduce a new synthetic dataset, Test of Time (ToT), which consists of two tasks: ToT-Semantic for temporal semantics and logic, and ToT-Arithmetic for temporal calculations. The study evaluates five LLMs and analyzes the impact of factors like graph structure, question type, and fact order on performance. The findings provide insights into LLMs' strengths and weaknesses in temporal reasoning.

### Strengths
-	The proposed ToT benchmark is designed to address the limitations of existing benchmarks by encompassing a wider variety of graph structures and question types, enabling a more nuanced evaluation of LLMs' temporal reasoning abilities
-	The authors offer an evaluation of temporal reasoning by decoupling it into semantic and arithmetic aspects. This two-pronged approach provides a more detailed analysis of LLM capabilities.

### Weaknesses
-	As mentioned in the limitation section, the benchmark focuses on scenarios where both the start and end times of a fact are mentioned within a single sentence. But real-world temporal information can be spread across multiple sentences or documents.
-	The authors generate questions using templates, which might not fully capture the complexity and variability of natural language found in real-world temporal reasoning tasks.

### Questions
1.	How would the performance of LLMs change if the benchmark included static facts in addition to explicit temporal facts?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces two datasets specifically crafted to evaluate large language models (LLMs) on temporal reasoning across diverse scenarios. The authors argue that existing benchmarks for temporal reasoning primarily use question-answering tasks based on Knowledge Graph -style temporal facts about well-known entities. Such benchmarks may reflect a model’s capacity to leverage prior knowledge rather than assess true temporal reasoning skills. To this end. the proposed datasets aim to measure two core temporal reasoning abilities of LLMs: (1) understanding the semantics and logic of time, and (2) performing temporal arithmetic.

### Strengths
- For the ToT-Semantic dataset, designed to evaluate LLMs on temporal semantics and logic, the authors employ seven graph generation algorithms and develop eight manually crafted question types. This diversity allows the generation of a large volume of synthetic questions, adding rigor to the dataset and covering various temporal reasoning facets.

- The study provides detailed insights into the temporal reasoning capabilities of frontier LLMs, including how factors such as graph size, question type, and temporal fact ordering influence performance. These observations offer valuable understanding into both the strengths and limitations of current LLMs in temporal reasoning.

### Weaknesses
- While ToT-Semantic focuses on temporal semantics and logical reasoning, the paper does not clearly explain how the graph generation process ensures the correctness of graph evolution. Specifically, the distinction between generating static graphs and those with temporal dynamics is not addressed, leaving questions about the dataset's fidelity to real-world temporal processes. 

- In introduction, the paper emphasizes the importance of evaluating LLMs on temporal reasoning but does not clearly explain why a graph structure is essential for this assessment. Could the authors elaborate on the necessity of graphs in this context?

### Questions
As mentioned in weakness.

### Soundness
3

### Presentation
3

### Contribution
3
