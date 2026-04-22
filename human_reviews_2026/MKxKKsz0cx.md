# Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
LiveCodeBench (LCB) has recently become a widely adopted benchmark for evaluating large language models (LLMs) on code-generation tasks. By curating competitive programming problems, constantly adding fresh problems to the set, and filtering them by release dates, LCB provides contamination-aware evaluation and offers a holistic view of coding capability. However, LCB remains restricted to Python, leaving open the question of whether LLMs can generalize across the diverse programming languages required in real-world software engineering.

We introduce Multi-LCB, a benchmark for evaluating LLMs across twelve programming languages, including Python.
Multi-LCB transforms Python tasks from the LCB dataset into equivalent tasks in other languages while preserving LCB’s contamination controls and evaluation protocol.
Because it is fully compatible with the original LCB format, Multi-LCB will automatically track future LCB updates, enabling systematic assessment of cross-language code generation competence and requiring models to sustain performance well beyond Python.

We evaluated 24 LLMs for instruction and reasoning on Multi-LCB, uncovering evidence of Python overfitting, language-specific contamination, and substantial disparities in multilingual performance. Our results establish Multi-LCB as a rigorous new benchmark for multi-programming-language code evaluation, directly addressing LCB’s primary limitation and exposing critical gaps in current LLM capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends LiveCodeBench (LCB) from Python to twelve programming languages and proposes Multi-LCB, a contamination-aware, continuously updating benchmark for multilingual code generation.
The authors convert LCB tasks into a unified STDIN/STDOUT evaluation pipeline and evaluate 20 recent LLMs.
Their findings include (i) Python overfitting; (ii) language-specific contamination signals via post-cutoff time slicing; and (iii) cross-language performance disparities.
The paper argues that Multi-LCB preserves LCB’s contamination controls while enabling side-by-side, same-task comparisons across languages.

### Strengths
1. **Meaningful benchmark**, which can be directly used to evaluate and compare LLMs' code generation capability across different PLs 
2. **Large-scale experiments with insightful findings**. Cover diverse programming languages and evaluate 20 LLMs, providing findings to reveal real multilingual gaps in coding tasks.
3. **Additional results and Open-sourced code**. Provide additional results and implementation details in the appendix and also release the scripts and datasets in an anonymous repository.

### Weaknesses
1. **Limited Measurement of the Generalization on Different PL**. This paper aims to answer the question of whether LLMs can generalize across diverse programming languages (in abstract).
When evaluating a model's capability in a specific language, it's crucial to assess its capability in utilizing that language's unique features and built-in libraries  (e.g., Rust ownership/lifetimes, Go concurrency primitives, JS/TS async patterns, use of standard libraries). However, since all tasks were converted from Python tasks, one concern is that these tasks may be language-specific and thus fail to effectively measure the model's performance in using syntax and features related to other languages. Consequently, it cannot effectively measure the LLMs' capability in using different programming languages. This may limit the contribution of this benchmark.
2. **Insufficient handling of the differences between the syntax/features of different languages**. The outputs of certain tasks may depend on language-specific syntax/features (e.g., the measurement of Unicode length is different in Python, JS, and TS (code points VS code units), and modulo with negative inputs may have different results across languages). If such samples are directly translated from a Python dataset, the evaluation may unfairly penalize other languages and understate LLMs' competence on them.
3. **Contamination detection across languages**. Beyond filtering by the release date of the original Python problem, the paper does not provide other contamination detection methods. One of my concerns is that the rewritten/converted samples may have textual or semantical overlap with samples in the prior dataset, leading to data contamination and inflating scores of certain models.

### Questions
1.  This paper seems to only measure LLMs' capability to use different programming languages ​​to solve the same algorithm problem, which may not comprehensively reflect  LLMs' capability to use syntax and features unique to other languages, thereby providing a one-sided measurement.
Please discuss and clarify the contribution of this paper and what specific research questions this paper wants to answer.
2. How does the author filter or process these samples related to the unique features of the programming language? What are their effects on the assessment of models' capability on different PLs in your experiments?
3. Besides the release-date filtering, what contamination detection methods are used on the converted samples in different languages?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This project extends the LiveCodeBench (LCB) benchmark to support multiple programming languages. Specifically, the authors translated the original Python-based problems into 11 additional languages, resulting in a total of 12 supported languages. They also developed an automated mechanism to track and incorporate future LCB updates, ensuring long-term maintainability of the benchmark. Using the new Multi-LCB dataset, the authors conducted a comprehensive evaluation of 20 large language models (LLMs), revealing that most models exhibit substantially lower performance on languages other than Python due to overfitting, and uneven distribution of pertaining corpora in terms of programming language diversity. The authors publicly release their benchmark extension, including prompts, source code, and configurations under MIT license to facility reproduction and future research.

### Strengths
* The work extends a state-of-the-art (SOTA) benchmark while preserving the core strengths of the original LiveCodeBench. The design also emphasizes long-term maintainability through automated update tracking.
* The evaluation framework is relevant for practitioners, addressing key technical challenges such as automatically converting code problems and hidden test cases into the appropriate format, as well as supporting sandboxed execution.
* The paper provides valuable insights into benchmark contamination by using release date metadata to perform experiments, highlighting an important issue in LLM evaluation.
* The evaluation includes a sanity check by comparing Python results on both the original LCB and the proposed Multi-LCB, demonstrating methodological rigor.

### Weaknesses
* The work does not introduce new task types. Since certain programming languages have inherent advantages for specific problems, adding novel or language-agnostic tasks could have strengthened the contribution’s originality.
* While the overall presentation is solid, there are opportunities to enhance transparency in specific methodological aspects (see Questions section).
* It is unclear whether the proposed benchmark covers all existing LCB releases or only the latest one.
* The benchmark appears less challenging for LLMs on widely used programming languages — strong performance is reported for C++, Java, PHP, C#, and JavaScript.
* The evaluation omits GPT-OSS and leading proprietary models, limiting the completeness of the comparative analysis.

### Questions
1. My understanding is that LCB releases are non-overlapping and cumulative, each representing a temporal evaluation slice. Did you convert tasks from all releases or only from the latest one? The paper notes that your pipeline supports the conversion of multiple versions, but it is not explicit whether all versions were actually converted.
2. Lines 205–215 mention adopting a zero-shot strategy, yet Step 2 refers to the inclusion of illustrative examples. Could you clarify how these examples fit within a zero-shot setup?
3. Please elaborate on the infrastructure or practical constraints that prevented inclusion of the Swift programming language.
4. I recommend adding evaluations for GPT-OSS as well as leading closed-source models to strengthen the study’s benchmarking scope.
5. In your view, which factor contributes more to improving cross-language performance — multi-language training or enhanced reasoning capabilities?
6. Please explain how the evaluation framework supports long-term maintainability through automated update tracking.

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
This paper presents MULTI-LCB, an extension of the existing LiveCodeBench (LCB) benchmark from Python to 12 programming languages. This paper design an automated pipeline that converts function-based LeetCode problems into standardized STDIN/STDOUT tasks and execute them within an isolated sandbox environment. The paper reports Pass@1 results for 20 publicly available large language models, analyzing cross-language differences, contamination effects, model scaling, and fine-tuning strategies.

The major contributions are:
1. Expands LCB from a single-language (Python) setup to 12 languages.
2. Standardizes problem I/O specifications to enable consistent cross-language evaluation.
3. Preserves LCB’s time-based cutoff to mitigate training-data leakage.
4. Benchmarks 20 code generation models under a unified framework.

Overall, the work demonstrates strong engineering and community value, providing a practical foundation for multi-language evaluation of code models. However, its methodological novelty is limited.

### Strengths
1. High engineering quality: Large-scale multi-language extension with reproducible infrastructure.

2. Comprehensive coverage: Evaluation across 20 models and 12 languages under consistent settings.

3. Community relevance: Provides a standardized and contamination-controlled environment for fair model comparison.

4. Clear presentation: Tables and figures effectively summarize results, aiding interpretability.

### Weaknesses
1. The authors extend LiveCodeBench from Python to 12 languages (e.g., C++, Java, Rust, Go, Kotlin), building a unified STDIN/STDOUT interface and Docker-based sandbox for consistent evaluation. The implementation is robust and reproducible.

2. The benchmark covers 20 models (instruction-tuned, reasoning-tuned, etc.) under identical settings. Results across tables and heatmaps show consistent model ranking and meaningful trends in language difficulty.

3. The continuation of LCB’s time-based contamination control adds credibility, and the planned release of code and conversion scripts will make this a useful resource for the code-generation community.

4. Figures and tables are easy to interpret, and the comparison with the original Python subset (differences within a few points) supports reproducibility.

### Questions
1. The work mainly represents a large-scale re-engineering of an existing benchmark rather than a conceptual or methodological innovation.

2. The paper does not quantify whether converting function-based problems to STDIN/STDOUT truly preserves difficulty, leaving uncertainty about potential format-induced biases across languages.

3. Only Pass@1 is reported; missing Pass@k and error breakdowns (e.g., compilation vs. runtime) limit interpretability of language gaps.

4. The “Python advantage” is discussed but not supported by controlled analysis; other confounding factors such as type strictness or dataset differences are not explored.

5. Some deviations from the LCB Python results (up to ~8%) are acknowledged but not analyzed, reducing confidence in full cross-language equivalence.

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
5

### Summary
The authors introduce Multi-LCB a multilingual extension of LiveCodeBench (LCB) that evaluates large language models across 12 programming languages (instead of just Python). The Multi-LCB benchmark is built by converting LCB into a unified STDIN/STDOUT format — which works across all supported languages. The authors present a robust evaluation of 20 large language models, identifying important trends on programming language bias in performance and contamination trends.

### Strengths
- Automatic conversion of LCB into 12 languages is a significant contribution to the community.
- Presents clear evidence for a bias towards certain language, in particular with python performance being consistently higher than other languages. The results also show clear hierarchies (Figure 4) of performance with python, c++, and java in the highest performing category.
- Moreover, the results of performance over time, highlighting important trends resulting from large pretrained models including various benchmarks in the training set. The proposed Multi-LCD offers to track such information.
- The rigorous methods are also a strength:
  - Evaluation setup: 20 diverse models (7B-685B parameters), evaluation within a sandboxed execution, and strict resource limits, and a balanced evaluation for each language via common task sets.
  - Care was taken to create reproducible results. Specifically, the authors give compiler versions, inference parameters, and public code release.
- Clear presentation: the writing is clear and the figures clearly communicate important findings.

### Weaknesses
"These results confirm that strong Python ability is not a reliable proxy for true cross-lingual code generation competence."
It's unclear why the authors make this claim — to me their results suggest the opposite. Specifically, Figure 3 shows a clear correlation between python performance and average performance across all other languages. The benchmarks can show a bias towards performing better on python while still be proxy for performance on other languages. This claim also contradicts prior works that show multilingual models perform best because performance is correlated across difference languages.

Missing citation: ["Multi-Lingual Evaluation of Code Generation Models"](https://arxiv.org/pdf/2210.14868) is a well cited paper on converting monolingual datasets to multilingual code, including the popular python dataset MBPP. This is an important citation in this area and very close to your approach, so you may want to explicitly clarify how the two approaches differ. Athiwaratkun et. al provide already provide an approach for converting benchmarks which enables direct comparison across languages.

For benchmarking purposes it would have been nice to see how commercial models like GPT5, Claude, and Gemini perform.

The authors only present results from `pass@1`, which is valid, however showing results with a greater sampling budget would help: (1) add to significance of findings (which are missing), and (2) more importantly `pass@k` metrics tend to show a sigmoid relationship between `k` and performance. In other words, for higher values of `k` we're likely to see smaller gaps between "easier" and "harder" languages, and the gap between python and other languages may be noticeably smaller. Likewise, the authors use a temperature at 0.2, which is quite low and may be preventing more exploration and possibly negatively biasing languages that make up a small portion of the training data.

"we evaluate 20 recent large language models on Multi-LCB, restricting tasks to those released after 2025-02-01 to ensure live, post-cutoff evaluation and minimize any risk of training-data leakage"
This effort to avoid contamination is good and likely significant helps avoid contamination, but it's unlikely to "ensure" contamination since problems may have existed in other areas previously. LLMs have such large and broad pretraining datasets that indirect contamination is always possible.

### Questions
Please clarify how this approach is novel with regard to Athiwaratkun et. al. For example, why not just use the approach they introduced for creating MBXP? Are there advantages to the Multi-LCB conversion method or is the primary contribution the benchmark dataset itself?

### Soundness
3

### Presentation
1

### Contribution
4
