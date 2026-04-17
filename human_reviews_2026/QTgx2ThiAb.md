# Same Content, Different Representations: A Controlled Study for Table QA

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Table Question Answering (Table QA) in real-world settings must operate over both structured databases and semi-structured tables containing textual fields. However, existing benchmarks are tied to fixed data formats and have not systematically examined how representation itself affects model performance.
We present the first controlled study that isolates the role of table representation by holding content constant while varying structure. Using a verbalization pipeline, we generate paired structured and semi-structured tables, enabling direct comparisons across modeling paradigms. To support detailed analysis, we introduce a diagnostic benchmark with splits along table size, join requirements, query complexity, and schema quality.
Our experiments reveal consistent trade-offs: SQL-based methods achieve high accuracy on structured inputs but degrade on semi-structured data, LLMs exhibit flexibility but reduced precision, and hybrid approaches strike a balance, particularly under noisy schemas. These effects intensify with larger tables and more complex queries.
Ultimately, no single method excels across all conditions, and we highlight the central role of representation in shaping Table QA performance. Our findings provide actionable insights for model selection and design, paving the way for more robust hybrid approaches suited for diverse real-world data formats.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents the controlled study examining how table representation (structured vs. semi-structured formats) affects Table QA performance across different modeling paradigms. The paper introduces a pipeline that generates paired structured and semi-structured tables with identical content, enabling direct comparison of NL2SQL, LLM-based, and hybrid methods under controlled conditions. The study reveals consistent trade-offs.

### Strengths
1  The pipeline that generates semantically equivalent structured and semi-structured table pairs is reasonable.

2  The four-dimensional diagnostic framework (table size, joins, query complexity, schema quality) provides good insights into how different factors interact with representation formats.

3  The case studies and error analysis provide concrete examples

### Weaknesses
1 The verbalization process may not fully capture the complexity of naturally occurring semi-structured tables. Real-world tables often exhibit more irregular patterns and noise than the systematically generated semi-structured versions used in this study.

2 The paper provides limited theoretical analysis of why different representations affect model performance. More fundamental insights into the cognitive or architectural reasons for these differences would strengthen the contribution.

3 The paper could discuss the scale limitations in evaluation, for example, long tables with more than 100 rows.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first controlled experimental study on a key yet underexplored problem in Table QA: the impact of table representation on model performance. It provides a comparative analysis of NL2SQL, LLM, and hybrid methods across different representation formats. The research is rigorously designed and supported by extensive experiments.

### Strengths
1. **Substantial Innovativeness**: The impact of table representation formats on model performance is an important yet systematically understudied topic.  
2. **Comprehensive Evaluation**: The assessment covers multiple approaches—including NL2SQL, LLMs, and hybrid methods—across various datasets and evaluation dimensions.

### Weaknesses
1.  **Limited Table Representations Explored**: Although this study systematically investigates the impact of table representation formats on model performance, the variety of representations examined remains limited. Other transformations—such as row/column permutation, transposition, and modality conversion (e.g., to visual representations)—could be considered.
2.  **Room for Enhanced Table Diversity**: The current definition of short and long tables based solely on row count could be expanded. For instance, tables with very high column counts—common in fields like medical research—represent another important dimension of diversity.
3.  **Over-Simplified Table Transformation Methods**: Even when considering only structured and semi-structured tables, the transformation methods applied (e.g., textual serialization and cell merging) remain relatively basic. More advanced structural transformations could be explored.
4.  **Potential Data Insufficiency in Some Experiments**: The sample size in certain experiments (e.g., S4) may be insufficient for robust conclusions.
5.  **Recommendation: From Analysis to Decision Support**: Beyond data analysis, we recommend developing a decision tree to guide practitioners in selecting the most suitable method for specific scenarios.

### Questions
1.  **Additional Experiments on Schema Restoration**: Consider incorporating experiments that utilize methods like NameGuess[1] to restore damaged schemas, evaluating how such restoration impacts model performance.  
2.  **Table Reconstruction Prior to Testing**: It may be beneficial to experiment with table reconstruction as a preprocessing step before applying the evaluation methods.  
3.  **Mitigating LLM Judge Bias**: To address potential biases introduced by LLM-based evaluation, it is advisable to sample a subset of data for analysis—comparing LLM judgments with human scoring—to enhance the validity of the evaluation approach.

[1] Zhang J, Shen Z, Srinivasan B, et al. NameGuess: Column name expansion for tabular data[J]. arXiv preprint arXiv:2310.13196, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a controlled study on how table representation, structured vs. semi-structured, affects model performance in Table Question Answering (Table QA). Using a verbalization pipeline that preserves table semantics, the authors compare NL2SQL, LLM-based, and hybrid approaches across varied settings of table complexity and query difficulty. Results show that structured inputs favor symbolic methods, semi-structured forms favor LLMs, and hybrid systems achieve the best balance, highlighting how representation choice fundamentally shapes reasoning accuracy and robustness.

### Strengths
- The study introduces a novel, well-controlled evaluation framework that isolates representation effects with strong methodological rigor. It provides comprehensive empirical evidence across datasets (BIRD, MMQA, TableEval), method families, and conditions, revealing consistent, interpretable trends. 

- The verbalization pipeline and diagnostic splits are technically elegant and reproducible, offering a valuable benchmark for future Table QA research. The analyses are thorough, uncovering key trade-offs and failure cases that inform model and benchmark design.

### Weaknesses
The paper is largely empirical, offering extensive measurements but limited analytical explanation for why representational shifts yield performance divergence: What internal representations or attention dynamics contribute to this behavior? While the benchmark is comprehensive, it focuses mainly on static accuracy; are there analyses on error propagation, confidence calibration, or reasoning trace fidelity under different table formats?

The verbalization pipeline relies on LLM paraphrasing—how consistent is this process across seeds, and how might linguistic variation bias outcomes? The study primarily evaluates medium-scale models and standard datasets; can the results generalize to larger instruction-tuned LLMs or real-world multi-table enterprise data?

Finally, while comparisons are broad, statistical significance testing and variance reporting across multiple runs are limited. Could more rigorous uncertainty quantification strengthen empirical claims?

### Questions
See the Weaknesses above.

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
3

### Summary
The authors perform a controlled study to examine how table representation affects model performance. The diagnostic benchmark they created has structured tables, but also has semi-structured variants, which are produced by selecting columns to change into free-text, reconstructing templates and serialize the updated table. They look into five key factors: structured vs semi-structured inputs, short vs long tables, multi-table alignment vs join operation tasks, simple lookups vs multi-step compositional queries and noisy schemas. The method families they test on are LLM-based, NL2SQL methods and hybrid methods. They generally find NL2SQL models to be better with structured tables and hybrid models to be more balanced with robustness across formats and peak accuracy.

### Strengths
S1. The paper explores various facets of the table representation, which are reasonable and comprehensive directions.
S2. The paper is generally well structured and organized.

### Weaknesses
W1. Some of the claims don't seem to match the actual results. For instance, on lines 284-286, the claim is that "LLMs degrade modestly" and "hybrids lead" for RQ1. However, based on Table 2, the drop % for LLM methods is similar or even much lower than the Hybrid ones. It doesn't seem that hybrid models lead in that case.

### Questions
See W1.

### Soundness
3

### Presentation
4

### Contribution
3
