# A Comprehensive Fine-Grained Evaluation of LLMs in Data Race Detection

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Data races are a major cause of concurrency-related bugs and have long posed a critical challenge in software engineering. 
Recent advancements in large language models (LLMs) have inspired researchers to investigate the potential of LLMs in detecting data races. 
However, the effectiveness of LLMs in this domain still remains largely unexplored, primarily due to the coarse-grained program-level evaluation methodology of existing benchmarks. 
This article introduces DRDBench, a novel benchmark, together with FineEval-Race, a pioneering evaluation framework, to assess the race detection capabilities of LLMs at the fine-grained individual data race level. 
DRDBench consists of 1,003 real-world and handcrafted pthreads-based programs, encompassing 549 data races in 226 programs, each annotated with precise line-level race locations. Leveraging this detailed race location information, FineEval-Race establishes fine-grained correspondences between model outputs and ground truth at the level of individual data races, enabling a nuanced evaluation. 
Based on these fine-grained correspondences, FineEval-Race further evaluates the performance of models under three different response aggregation strategies to investigate the boundary of model capabilities.
This methodology not only quantifies LLMs' direct utility in race detection but also provides insights into their genuine understanding of concurrency.
We evaluated 25 popular open-source LLMs on DRDBench with FineEval-Race.
The evaluation results revealed considerable variation in model performance, with DRDBench presenting a significant challenge to many models. The top-performing reasoning and non-reasoning models, DeepSeek-R1 and DeepSeek-V3, achieved recall of 75.23% and 55.19%, and precision of 75.36% and 54.69%, respectively. 
These evaluations yield actionable insights.
Furthermore, we identify two failure modes shared across models that can cause up to 92\% and 98\% performance degradation on DeepSeek-R1 and DeepSeek-V3, respectively. 
We believe that DRDBench and FineEval-Race, coupled with our identified actionable insights and failure modes, will serve as crucial guidance for applying LLMs to race detection and inspire future model training efforts to enhance their comprehension of concurrency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose DRDBench, a new pthreads-based dataset containing 1,003 C programs with precise location annotations for data race detection. Besides, they propose FineEval-Race a fine-grained evaluation framework that maps LLM outputs to individual ground-truth data races and measures LLMs' performance.

### Strengths
1.The argument that existing evaluations are too coarse is convincing.

2.The dataset is large, real-world (SV-Benchmark origins), and annotated to line granularity.

3.25 open-source LLMs (reasoning and non-reasoning variants) are evaluated, plus Deagle as a static analyzer baseline

4.The authors manually inspect common failure cases and identify interpretable failure modes.

### Weaknesses
1.The benchmark appears to be heavily pre-processed, potentially reducing its representativeness for real-world concurrent software.
(a) Removing comments may distort the natural structure of source code, as comments are integral to understanding program intent. While such preprocessing might be acceptable for controlled method evaluation, it undermines the realism of a benchmark.
(b) The evaluation setup, as suggested by the template prompts in Appendix E, seems to test LLMs only on isolated concurrent programs. However, real-world data races often emerge from inter-thread or inter-module interactions. The current benchmark design does not seem to assess the LLMs’ ability to handle such complex scenarios.

2.DRDBench includes only data race programs, which may limit the benchmark’s generalizability. 

3.The benchmark lacks a systematic and labeled taxonomy of data race patterns. Without explicit categorization (e.g., read–write, write–write, synchronization-related races), it is difficult to assess whether the benchmark sufficiently covers the diversity of real-world race conditions or to analyze model sensitivity across pattern types. 

4.The manuscript does not provide explicit statistics on thread counts or the distribution of concurrency complexity.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a novel benchmark dataset comprising 1003 p-threads programs derived from SV-COMP, along with a new evaluation framework, FineEval-Race, designed to assess the performance of data race detection in LLMs. The authors conduct a comparative study of 25 different LLMs based on their proposed dataset and evaluation framework.

### Strengths
The paper’s comparative study shows a wide range of comparisons between each model, and empirically shows that the reasoning models perform better than non-reasoning models for the data race detection task. Such a comparison has significant meaning to the software engineering community and AI community when it comes to choosing a base model for downstream tasks that require reasoning about the correctness of pthreads programs. The evaluation framework proposes to use metrics based on well-established metrics such as precision, F1, and recall, which many community members in the software engineering would agree upon.

### Weaknesses
Although the evaluation framework metric uses aggregated results of well-established metrics, there is a substantial shortcoming of the ranking metric, the S score. The authors define the S score of a large language model D as “the sum of its rankings across all metrics”, where part of the metric consists of recall, precision, F1, and False Positive Rates. However, precision, F1, and recall all rely on the number of detected true positives, so there are three terms that utilize true positives, but there is only one term that depends on the number of false positive detections. Therefore, an unweighted sum of the rankings based on the proposed metrics is geared towards favoring the models with high true positive detection. This could lead to unfair comparison of two models where they both achieve similar performance with respect to true positive detections, but have a significant difference in their ability to avoid false positives. For example, in the Greedy decoding of Table 1, DeepSeek-V3-671B and Qwen2.5-72B have similar performance in terms of precision, but their performance drastically differs for FPR, yet both of their S score are 33. Hence, it is insufficient to compare the models with the naive summation of the rankings from each metric.

### Questions
One suggestion for designing a fair metric is to compare the importance of each metric (F1, precision, FPR, etc.) and construct the S score as the weighted sum of the relative importance. Please refer to the Weaknesses section for questions.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper releases DRDBench, a benchmark of 1,003 pthread-based C programs containing data race programs or race-free programs and an evaluation framework FineEval-Race to evaluate data race detection with LLMs. The construction pipeline includes user-defined header inlining, comment removal, formatting for one-statement-per-line, human+tool annotation, and meta-review. Experiments span strong reasoning and non-reasoning LLMs with pass@k, F1, and FPR. The authors also analyze common failure modes (e.g., misunderstanding user-defined synchronization).

### Strengths
1. Clear pipeline for constructing data race benchmarks with fine-grained line annotations and a meta-review process.

2. FineEval-Race task design provides a concrete, reusable framework for model evaluation on data race detection.

3. Comprehensive evaluation of open-sourced LLMs on the detection accuracy as well as the hallucination rate.

### Weaknesses
1. Key preprocessing choices (comment removal, single-file flattening) are insufficiently justified relative to their impact on LLM behavior.

2. No agentic/tool-use baselines despite a task that naturally benefits from tools.

3. Findings from the evaluation are mostly descriptive; audience and actionable guidance are under-specified.

### Questions
Thank you for the submission. I believe this is a timely benchmark that evaluates LLMs on the task of data race detection. The writing is easy to follow. However, there are a few drawbacks that I think the paper should address before getting accepted.

I do not fully understand the rationale for removing comments, given comments can encode developer intent and explanation that LLMs leverage. The paper states comments “may introduce noise” and are therefore removed, but provides no good ablation study to justify this design choice. This omission matters because the prompt explicitly teaches concurrency rules and relies on natural-language guidelines elsewhere.

Second, flattening multi-file projects into a single file is also debatable. The appendix comparison suggests minimal aggregate impact, but in Table 6, I see examples where the pass rate could reduce by 100% or increase by 50%. Also, using a single file to evaluate seems to simplify the task as it can reduce the code length. However, in the related work, the drawbacks of prior benchmarks seem to be over-simplifying the detection task and having too simple programs. So I feel this design choice does not fit with the goal of the paper to construct a comprehensive data race detection benchmark. 

There is also a lack of evaluation on the performance of agents with tool uses. Data race detection is a suitable task for tool-calling agents. This is because requiring only the static analysis tools is not sufficient (the paper also reports Deagle having worse performance than the best open-sourced models). I expect agentic solutions with tools such as a debugger and filesystem handlers to have better performance, as they provide dynamic debugging information to the agents. 

The paper presents a comprehensive evaluation of open-sourced models' performance on the data race detection tasks, but the findings remain relatively straightforward. For example, Table 1 shows that larger models with reasoning capabilities generally perform better. Such findings are not so different than those from conventional coding benchmarks. There is a lack of actionable items or insights that the system could give.

Technical Questions:

1. Can you provide an ablation across programs for the LLM detection accuracy with and without comments removal? How do different comment types (docstrings, TODOs, etc) affect outcomes?

2. Is it possible to evaluate on cross-file repository setup that makes the data race detection task more realistic?

3. For the annotation process, why do we first let the human annotator review the outputs and then independently write annotations, but not vice versa? Would that lead to biases?

4. Does meta-reviewer review all of the annotations generated or only part of them?

5. The framework only checks if the line number matches with the ground-truth results, but isn't that going to lead to more false positives if there are multiple variables in a line?

6. Who is the target audience of the benchmark and the framework? What are some actionable insights we can get from the evaluation?

### Soundness
2

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
This work makes a contribution for evaluating large-scale language models in the complex task of data-race detection. The authors introduce a high-quality benchmark, DRDBench, and a fine-grained evaluation framework, FineEval-Race, shifting the assessment from coarse, program-level “yes/no” verdicts to precise measurement of individual data races with metrics such as precision and recall. This refined approach yields a deeper and more nuanced understanding of current model capabilities. Experiments on 25 open-source models, a comparison against the latest static analyzer, and insightful failure analyses ensure both rigor and impact. The findings—that reasoning-oriented models excel and that specific failure modes can be identified—provide clear and valuable guidance for future research.

### Strengths
1. The core idea of shifting evaluation from the program level to the individual data race level is the paper's greatest strength. This fine-grained approach is essential for a meaningful assessment of progress in this complex reasoning task and sets a new standard for future work.
2. DRDBench is built on solid ground, harvesting pthread cases from SV-Benchmarks to fill a critical gap, and its meticulous, multi-reviewer annotation process yields a ground truth we can trust.
3. The paper impressively evaluates 25 language models against the strong Deagle baseline. Its sharp analysis exposes a clear reasoning-vs-non-reasoning gap, models’ conservative–aggressive swings, and the promise–instability trade-off of sampling.

### Weaknesses
1. The programs in DRDBench, while sourced from real-world projects and verification suites, are relatively small in scale (max 624 Lines of Code). The performance of LLMs on larger, more complex, and sprawling codebases remains an open question.

### Questions
1. The results show Deagle achieving high recall (87.34%) but relatively low precision (63.82%), implying a notable number of false positives. Could you comment on the nature of these false positives from a state-of-the-art static analyzer on your benchmark? Does this suggest limitations in static analysis for certain concurrency patterns present in DRDBench?
2. The paper mentions using clang-format to ensure one statement per line. How significantly did this reformatting alter the program structure and line counts compared to the original source code? Is there a risk that this standardization inadvertently simplifies the task or changes it in a way that wouldn't reflect performance on unformatted, "in-the-wild" code?
3. Is the performance advantage of reasoning models driven primarily by longer output lengths, or by qualitative improvements in reasoning logic? A direct comparison of solution chains from top-performing reasoning versus non-reasoning models on identical complex cases would clarify this distinction.

### Soundness
2

### Presentation
3

### Contribution
3
