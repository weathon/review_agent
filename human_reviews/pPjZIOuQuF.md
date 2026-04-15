# RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems

- Decision: Accept (poster)
- Scores: 5, 8, 6

## Abstract
Large Language Models (LLMs) have greatly advanced code auto-completion systems, with a potential for substantial productivity enhancements for developers. However, current benchmarks mainly focus on single-file tasks, leaving an assessment gap for more complex, real-world, multi-file programming scenarios. To fill this gap, we introduce RepoBench, a new benchmark specifically designed for evaluating repository-level code auto-completion systems. RepoBench consists of three interconnected evaluation tasks: RepoBench-R (Retrieval), RepoBench-C (Code Completion), and RepoBench-P (Pipeline). Each task respectively measures the system's ability to retrieve the most relevant code snippets from other files as cross-file context, predict the next line of code with cross-file and in-file context, and handle complex tasks that require a combination of both retrieval and next-line prediction. RepoBench aims to facilitate a more complete comparison of performance and encouraging continuous improvement in auto-completion systems. RepoBench is actively maintained with the latest code, serving as a live benchmark publicly available at https://github.com/Leolty/repobench.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new benchmark called RepoBench for evaluating repository-level code completion systems. RepoBench consists of three interconnected evaluation tasks: RepoBench-R for retrieving the most relevant code in the repository, RepoBench-C for code completion using both in-file and cross-file context, and RepoBench-P for the entire pipeline of both retrieval and code completion. The authors carry out a series of experiments on RepoBench, analyzing the efficacy of various retrieval methods and code completion models of different magnitudes.

### Strengths
- A nice idea of benchmarking repository-level code completion
 - The paper is generally well-written and easy to follow

### Weaknesses
To me, the novel research contributions of the paper are a bit limited, especially for an AI conference. The paper could better fit a software engineering/programming conference.  The significance of the work could be more clearly stated.

To evaluate RepoBench-R, the authors selected three baseline strategies for the retrieval task, namely, random retrieval, lexical retrieval, and semantic retrieval. The selection of baseline strategies for RepoBench-R, particularly the inclusion of random retrieval and lexical retrieval, are weak baselines, which may not effectively demonstrate the distinctive capabilities of the proposed benchmark. In that sense, the results presented in Section 4.1 are under expectation and I think that previous benchmarks may also demonstrate the ability of these strategies. A more competitive baseline selection including LLMs would enhance the work.

The paper lacks a comprehensive comparison with previous benchmarks about code completion. Although RepoBench is the first benchmark on repository-level code completion, it would still benefit from comparisons with prior benchmarks. Such comparisons could involve RepoBench-R versus existing code retrieval benchmarks and RepoBench-C versus traditional benchmarks for function-level code completion. 

The metrics used for code completion, i.e., EM and Edit Similarity, are unusual. The authors could consider more widely used metrics such as pass@k and CodeBLEU? 

The evaluation of RepoBench-C is conducted using only three Language Model Models (LLMs), specifically CodeGen, StarCoder, and Codex. As a benchmark paper, the inclusion of only three LLMs may not fully represent the diverse capabilities of available models. To enhance the benchmark's applicability, additional LLMs, including recently proposed ones, could be considered for comparison. For example: Shi et al., SoTaNa: The Open-Source Software Development Assistant, https://arxiv.org/abs/2308.13416

### Questions
- Why not using widely used metrics such as pass@k and CodeBLEU? 
- How is the proposed benchmark compared to previous benchmarks for code completion?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors address a need for repository wide benchmarks for code-prediction and code-retrieval tasks. They do so by creating two datasets, in the test set, they recover repository information for the github-code dataset and create two variants, a 2K and an 8K variant. For the test set, they crawl permissively licensed Java and Python projects after the The Stack cut-off date. To better mimic real-world scenarios, the test set is not separated by prompt length. As for the benchmark itself, it focuses on three tasks that should exercise both cross-file and in-file context requirements. The tasks are code auto-completion, code-retrieval, and a join task where the relevant cross-file information should be retrieved before it is used for completion (pipeline).

### Strengths
(1) The paper addresses the need for a repository wide benchmark that better aligns with real-world usecases in software projects. (2) It addresses data leakage issues* by crawling new data for the test set and (3) provides fine-tuning data for models that may require it. 
(4) The StarCoder overfitting to file-level use-cases provides interesting additional insight.

### Weaknesses
The main concerns with the paper are two-fold. 

The usefulness of the benchmark relies on a gentleman agreement to not use data from the collection dates during training or fine-tunning.

Another concern is the opt-out possibility. While not strictly necessary, a nice-to-have would be an opt-out mechanism similar to the The Stack one for authors that may want to remove their code from.the data.

### Questions
Is there an intention to make the bechmark a "living" benchmark where the test set is periodically refreshed to be past the training set horizon date?

Alternatively, is there an intention to check and disqualify models that have trained on test set data?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Authors propose RepoBench - a benchmark for repository level code auto-completion evaluation. They propose three evaluation tasks: retrieval, code completion, and pipeline. Authors perform experiments using RepoBench

### Strengths
- Significant work on RepoBench construction.
- Extensive experiments with RepoBench with existing models and retrieval techniques.

### Weaknesses
- It is not clear what new insights RepoBench and experiments on it contribute to the field. Were the results previously unknown or unexpected?

- This might not be a weakness of the paper per se, but it concerns me a bit that random retrieval is close to or even outperforms some non-random retrieval methods.


I increased the rating based on authors' answer to my questions.

### Questions
- What is exactly "the first appearance of a cross-file line within a file"? Is this the import line? Is this the first line that uses cross-file function?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
