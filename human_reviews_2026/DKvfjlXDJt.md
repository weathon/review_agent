# BenchName: a Set of Benchmarks for Long-Context Code Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
The fields of code and natural language processing are evolving rapidly, with models becoming better at processing long context windows --- supported context sizes have increased by orders of magnitude over the last few years. However, there is a shortage of comprehensive benchmarks for code processing that go beyond a single file of context, while the most popular ones are limited to a single method. With this work, we aim to close this gap by introducing BenchName, a suite of six benchmarks for code processing tasks that require project-wide context. These tasks cover different aspects of code processing: library-based code generation, CI builds repair, project-level code completion, commit message generation, bug localization, and module summarization. For each task, we provide a manually verified dataset for testing, an evaluation suite, and open-source baseline solutions based on popular LLMs to showcase the usage of the dataset and to simplify adoption by other researchers. We publish the benchmark page on HuggingFace Spaces with the leaderboard, links to HuggingFace Hub for all the datasets, and link to the GitHub repository with baselines are available in the manuscript.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents BenchName, a suite of six benchmarks designed to evaluate long-context code models on realistic software-engineering tasks that require repository-level reasoning rather than single-file snippets. The tasks span library-based code generation, CI build repair, project-level code completion, commit message generation, bug localization, and module summarization, each derived from high-quality open-source repositories. The authors provide detailed metrics, evaluation pipelines, and strong baselines from both open and proprietary LLMs, showing that the tasks are not saturated and expose meaningful gaps in model capabilities. Results highlight that many large models under-utilize long contexts. The benchmark correlates across tasks but retains complementary structure, supporting its use as a composite evaluation.

### Strengths
* The benchmark set covers a wide range of six tasks in code domain, representing common applications of code LLMs.
* The paper presents sufficient details about data sourcing, problem consturction, and evaluation metrics for each of the tasks.
* Experiments are conducted over a comprehensive collection of both open-source and proprietary LLMs.

### Weaknesses
* I'm not fully convinced of the motivation part as stated in L048-057. Evaluating with SWE-Bench tasks in the agentless mode [1], i.e., gathering relevant context offline and prompting the model once to generate the code patch, seems a good alternative that simultaneously avoids multi-turn conversations and preserves the order of software development.

* Regarding the lib-based code generation task, first, the task by itself does not necessarily require additional cross-file context, because the libraries involved are likely seen during model training, and thus, the relevant context can be implicitly retrieved from models' parametric knowledge. One cannot disentangle memorization from utilization of long context when interpreting the evaluation results. Second, traditional BM25 over file chunks might be a better way than the proposed API lists for constructing additional context. Third, the proposed metric only checks API names but overlooks the arguments or the functional correctness of the entire script. 

* Regarding the project-level code completion task, while it is claimed that the proposed benchmark respects the order of software development by leveraging commits, the subsampling of lines for completion still exposes part of the new code in the target commit. Besides, feeding full file context to the prompt likely performs worse than the traditional chunk-based BM25 retrieval. 



References:

[1] Xia, Chunqiu Steven, et al. "Agentless: Demystifying llm-based software engineering agents." arXiv preprint arXiv:2407.01489 (2024).

### Questions
* In the evaluation of project-level code completion, if subsampled lines for completion are consecutive, how do you match each generated line to groundtruth? If the model generates an extra line at the beginning, will that impact all subsequent lines due to mismatches of line numbers?

### Soundness
2

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
4

### Summary
The paper introduces BenchName, a six-task benchmark suite for ML4SE that explicitly requires module- to repository-level context while avoiding multi-step agent orchestration. Tasks span library-based code generation, CI build repair, project-level code completion, commit-message generation, bug localization, and module summarization. Datasets are rigorously filtered and manually verified, accompanied by baseline implementations and evaluation.

### Strengths
+ This paper addresses a timely and relevant problem, i.e., code generation in long-context scenarios, and presents a carefully curated benchmark suite for its evaluation. The problem is important, and as no standard benchmark currently exists, the paper has strong practical implications.
+ The benchmark construction applies rigorous corpus filtering and human verification to ensure high-quality data across the six investigated tasks.
+ Extensive experiments are conducted on 19 LLMs, yielding several insightful findings and practical recommendations for future research.

### Weaknesses
- Although challenging, it would strengthen the paper to incorporate functionally correct metrics (e.g., pass@1) rather than relying solely on lexical similarity measures in the evaluated tasks.
- The authors are encouraged to expand Section 3 to include more in-depth analytical studies.
- Most tasks are Python-centric; including additional programming languages such as C and Java would improve the benchmark’s generality and appeal.

### Questions
- The reviewer is curious whether most of the content in the provided dataset is actually relevant to the completion point. If not, could these long contexts simply be noise that the model should ignore during reasoning? This phenomenon seems analogous to retrieval-augmented generation (RAG).

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
This paper introduces “BenchName”, a set of six benchmarks designed to evaluate the long-context processing capabilities of code LLMs. The evaluation on a wide range of SOTA models reveals that the tasks are still challenging for current models.

### Strengths
1. Timely topic. The paper addresses a timely problem, i.e., long-context code processing evaluation. Tasks like HumanEval and MBPP cannot evaluate the long-context capability, while benchmarks like SWEBench are too complex and not fine-grained enough. The community does need benchmarks to evaluate specialized, fine-grained long-context code comprehension capabilities.
2. Diversity and practical usefulness of tasks. The benchmark includes six distinct tasks: library-based code generation, project-level code completion, CI build repair, commit message generation, bug localization, and module summarization. The tasks are diverse and can evaluate long-context code processing capabilities from various aspects. Plus, the tasks are prevalent in software development practice.
3. Non-trivial efforts in benchmark construction. The manual filtering and verification of datasets are crucial for the quality of the benchmark.

### Weaknesses
1. Limited language scope. The benchmarks are heavily focused on Python. While bug localization includes Java and Kotlin, the two most detailed tasks in the main paper (library based generation and project-level completion) are Python-only.
2. Effectiveness of library-based code generation evaluation metrics. The recall of appearance of API calls in the ground truth solution that also appear in the generated program is not a convincing metric for evaluating library-based code generation. 
3. Effectiveness of repo-level code completion capability. The exact match and perplexity metrics are not convincing enough to represent repo-level code completion capability.
4. The description of bug localization evaluation is not clear.
5. Using a small model as the LLM-as-judge assessor for module summarization evaluation.

### Questions
1. Why only Python projects are included in the benchmark (except bug localization)? It’s known that for tasks like CI-build error repair, repositories often use languages like TypeScript, Java, Go, JavaScript etc. in addition to Python.
2. What’s the correlation between the appearance of API calls in the ground truth solution that also appear in the generated program and the quality of library-based code generation? Why the correctness of code generation is not evaluated? Although the program is not always executable, at least some efforts should be invested to evaluate correctness[1].
3. Why use exact match and perplexity metrics for repo-level code completion evaluation, instead of using correctness metrics?
4. What’s the reason of using a smaller model (Mistral 7B) as the assessor for LLM-as-judge? Is the perplexity score w.r.t. to a small model a meaningful evaluation metric?

minor:
5. Figure 1: 0,89=>0.89
6. Is the name of the benchmark a placeholder or it's the final name? If it's a placeholder please fix it.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
BenchName packs six tasks testing LLMs on code that needs whole-project context. Old suites stick to short snippets; this one targets real, long-context hurdles. Every task ships with a checked dataset, eval harness, and open baselines. Two tasks get full write-ups in the paper, the rest in the appendix. OpenAI o1-style reasoners lead the pack. Data and code are open.

### Strengths
The paper addresses a clear and significant gap in the ML4SE landscape. As LLMs' context windows expand, there is a pressing need for benchmarks that can rigorously evaluate their ability to process and reason over entire codebases. BenchName is a direct and substantial contribution to this area.

The authors demonstrate a strong commitment to data quality. The emphasis on manual verification, rigorous filtering pipelines, and thoughtful data collection strategies is commendable. A key example is the project-level code completion task, where using Git history to create repository snapshots prevents a common and critical data leakage issue found in other benchmarks.

The paper evaluates a wide and very current range of models, including proprietary leaders , open-source models and specialized "reasoning" models. The experiments are well-designed, exploring the impact of context size and different context composition strategies, which provides valuable insights into current model capabilities.

### Weaknesses
The metrics are only a first cut, and their limits show. API Recall in the library-generation task counts correct calls yet never docks hallucinated ones nor runs the code for pass@k. CompScore leans on Mistral-7B as judge; authors curb positional bias, but verbosity and style preferences remain unmeasured, and human agreement is unchecked.

Space constraints leave four of the six benchmarks detailed solely in the appendix. While understandable, this fragmentation makes the main paper feel incomplete and forces readers to shuttle back and forth between the main text and supplements to grasp the full contribution.

### Questions
1. Why was API Recall chosen as the sole metric? Has API Precision or F1 been included to capture both correct and hallucinated calls? More importantly, can the generated code be executed? An execution-based measure such as pass@k would assess correctness far more reliably than mere API usage.
2. The benchmark uses an oracle to provide the model with the exact files and code blocks that need to be changed. This simplifies the task significantly. What is the performance drop when this oracle is removed? 
3. Could you please clarify precisely how the path distance is calculated? Have you considered comparing it against a simple semantic retrieval baseline for context composition?

### Soundness
3

### Presentation
3

### Contribution
2
