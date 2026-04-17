# TrustGen: Benchmarking Trustworthiness in Generative Models for Russian Language Processing Tasks

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Large Language Models (LLMs) are increasingly used in autonomous agents and multi-agent systems to handle complex tasks, making their trustworthiness a critical concern. However, most existing benchmarks focus on English, limiting their relevance for other languages, particularly Russian. In this study, we introduce the first benchmark for evaluating LLM trustworthiness in Russian-language tasks, assessing six dimensions: truthfulness, safety, fairness, robustness, privacy, and ethics. We adapt English datasets and incorporate native Russian data, creating 14 tasks from 12 datasets. Additionally, we propose the Task Format Non-Compliance Rate to measure structural adherence without penalizing correct content. Evaluating 22 LLMs, including Russian-adapted models, we uncover significant challenges in factual consistency, safety calibration, and bias mitigation. Our findings underscore the need for tailored fine-tuning and evaluation methods for non-English applications, providing a foundation for more trustworthy AI in Russian-language contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TrustGen, the first benchmark to evaluate LLM trustworthiness in Russian-language processing tasks. It includes six critical dimensions of trustworthiness and systematically evaluates different LLMs including both general multilingual models and specialized Russian-adapted derivatives. The paper further provides findings and

### Strengths
1. The paper studies an important topic on less-studied language.
2. The paper introduces a comprehensive benchmark

### Weaknesses
1. The paper presentation is poor and lacks sufficient details. The authors did not discuss the detailed information about the selected models and how the tasks and the datasets are evaluated (e.g. properties of the dataset, prompts used). See the questions below for details. Besides, some claims are not well supported in the paper. For example, I did not find the reference to English results in Line 357.

2. Although the paper provides many findings, it lacks an in-depth analysis of the phenomenon. For example, the authors do not discuss why language-specific tuning can lead to a tradeoff in trustworthiness. Also, it is unknown whether the model that performs better in the English trustworthiness benchmark would also perform better in the Russian benchmark.

3. Many datasets in the benchmark are not native Russian but adapted from English datasets. Although the authors claim they did a careful translation with cultural adjustment, it is unknown what it is exactly and how it affects the model performance.

### Questions
1. How are Russian-adapted models trained? What are their base models, and what are their training datasets?
 
2. Why does Russian adaptation lead to the tradeoffs in trustworthiness (e.g. Line 329, Line 357, Line 363)

3. The authors mention that some datasets are adapted from the English datasets. What cultural adjustment do the authors use during adaptation? (e.g.  Line 161)

4. How do the general LLMs perform in on TrustGen relative to their performance on the English trustworthiness benchmark? (The paper only discusses for the Robustness perspective.)

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
This paper aims to evaluate the trustworthiness of LLMs on Russian language processing tasks. To this end, they collect datasets targeting six trustworthiness aspects—truthfulness, safety, fairness, robustness, privacy, and ethics—and translate them into Russian if the original datasets are not in Russian. Then, they test the performance of various LLMs on these datasets.

### Strengths
1. Experimental setup is clearly clarified in the paper.

### Weaknesses
1. One of the contributions clarified in this paper is proposing a new evaluation metric—TFNR; however, this is not a new metric, as many papers like DecodingTrust have used this.
2. The contributions of this paper seem limited in the sense that this is a benchmark paper but there are no new datasets introduced nor novel findings.
3. The format of Table 2’s caption is incorrect, which should be above the table.
4. Overall, (correct me if I am wrong,) I feel this paper is simply translating hundreds of samples into Russian and test the performance of LLMs on these samples. It would be appreciated if the authors could clarify the contributions or novelties more explicitly.

### Questions
1. Is there any findings that are new compared with the ones on English languag processing tasks?
2. What are the reasons for selecting these LLMs? I notice that some use smaller versions, while others use much larger ones.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TrustGEN, the first large-scale benchmark for evaluating the trustworthiness of LLMs in Russian. TrustGEN covers six dimensions: truthfulness, safety, fairness, robustness, privacy, and ethics, using 14 tasks adapted from 12 datasets (primarily via translation and cultural adaptation from English sources, with some native Russian data). The authors propose a new Task Format Non-Compliance Rate (TFNR) metric to evaluate structural fidelity of model outputs and benchmark 22 LLMs (open, proprietary, and Russian-specialized) across all tasks.

### Strengths
- This paper addresses a real and important gap by providing the first comprehensive trustworthiness benchmark for Russian LLMs.
- The benchmark covers a broad range of trust dimensions, with diverse tasks and datasets.
- The authors propose a useful new metric (TFNR) for format compliance, adding nuance to traditional accuracy metrics.

### Weaknesses
- The main contributions are dataset curation and benchmarking; there is little technical or methodological novelty beyond the TFNR metric. The TFNR metric, while useful, is a minor technical advance.
- The creation of the benchmark heavily relies on translation/adaptation of English datasets, with relatively little native Russian data, potentially limiting cultural and linguistic validity.
- Key details about dataset adaptation, translation validation, annotation protocols, and inter-annotator agreement are insufficiently described, making it hard to assess data quality and reproducibility.
- Analysis is broad but does not deeply investigate error types, annotator disagreement, or model failure modes.

### Questions
- What protocols were used for annotation or validation of translated/adapted tasks? Was inter-annotator agreement measured?
- How do you validate that TFNR and other metrics reliably distinguish between format errors and genuine content errors, particularly for models with non-standard outputs?
- Were any human evaluations conducted on model outputs for qualitative analysis, or is all analysis purely automated?
- Can you provide error analysis or case studies of model failures on key tasks (e.g., hallucinations, privacy leaks, ethical mistakes)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces TrustGen, a benchmark designed to evaluate the trustworthiness of LLMs in Russian. The authors claim that this is the first comprehensive trustworthiness benchmark for Russian. The framework assesses 22 different models (including both multilingual and Russian-adapted ones) across six dimensions: truthfulness, safety, fairness, robustness, privacy, and ethics. These dimensions, adopted from prior work (e.g., TrustLLM and DecodingTrust), are evaluated using 14 tasks derived from 12 datasets, a mix of native Russian datasets and English datasets adapted through translation and cultural modification. The authors' findings highlight significant trustworthiness issues of current LLMs in Russian-language tasks

### Strengths
1. This paper attempts to address the gap between English and Russian trustworthiness benchmarks for LLMs, which is an important goal as we develop LLMs in a multilingual fashion.
2. The authors invested a significant effort in curating and adapting a wide range of datasets (12 in total). The native Russian datasets (e.g., SLAVA, RuBia, RuBLIMP, TAPE) are particularly important as they can capture relevant linguistic and cultural contexts than translated datasets alone.

### Weaknesses
1. The translation process from English to Russian is unclear to the reader, which is a very important step in constructing the dataset.
2. There are no concrete examples of the evaluation dataset. Even if they are in Russian, having concrete examples (accompanied by English translation) is very important for the reader to contextualize the benchmark.
3. The paper's core contributions are not as novel as claimed. The six-dimensional trustworthiness framework is explicitly adopted from TrustLLM and DecodingTrust. I also do not see how this paper advances our understanding of LLMs' trustworthiness along these dimensions.
4. The main metric proposed, the "Task Format Non-Compliance Rate (TFNR)," is not a new contribution as claimed, as many papers report similar metrics as the ability of instruction-following.
5. The authors equate many task-specific metrics with broad, real-world trustworthiness. For example, performing well on the stereotype detection task (a classification task) is a very weak proxy for a model's propensity to generate biased or unfair content in an unconstrained, in-the-wild, or even adversarial setting.
6. The presented results largely confirm findings that are already well-established in the trustworthiness assessment conducted in English. For instance, the paper identifies a "safety-usability tradeoff". This is a known phenomenon. It also does not attempt to analyze why these failures occur in a specifically Russian linguistic or cultural context.

### Questions
1. How are the English datasets translated into Russian? How do the authors guarantee that the translations are adequate? What if there are questions that simply do not make sense under the Russian context?
2. How do the authors judge if an answer is a refusal? What are the keywords used here?

### Soundness
1

### Presentation
2

### Contribution
2
