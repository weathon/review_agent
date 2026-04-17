# SWE-Refactor: A Repository-Aware Benchmark for Evaluating LLMs on Real-World Code Refactoring

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Recent advances in Large Language Models (LLMs) have garnered significant attention for their applications in software engineering tasks. Among these tasks, code refactoring has its own unique challenges. Unlike code generation, refactoring requires precise changes that preserve program behavior while improving structure, making automated evaluation difficult. Existing refactoring benchmarks suffer from three key limitations: (1) they often focus on atomic refactoring types while missing more complex ones; (2) they contain noisy data with entangled, unrelated code changes, making it difficult to study LLM’s true refactoring capability accurately; and (3) they lack code repository and structural information to support realistic evaluations. To address these issues, we propose SWE-Refactor, a new benchmark for LLM-based code refactoring. SWE-Refactor contains 1,099 real-world, pure refactorings collected from 18 real-world Java projects. Each refactoring instance is verified through compilation, test execution, and automated refactoring detection tools to ensure correctness. Unlike prior benchmarks, SWERefactor covers both atomic and compound refactoring types (single and multiple code changes). It includes rich repository-level data (e.g., method callers and callees, class hierarchies), as well as configuration details like test coverage and build settings. We evaluate nine widely used LLMs on SWE-Refactor, including GPT-4o-mini, DeepSeek-V3, and CodeLLaMa. DeepSeek-V3 achieves the best performance with 457 successful refactorings (41.58%), followed by GPT-4o-mini with 438 (39.85%). DeepSeek-V3 performs particularly well on Extract Method, completing 301 cases, while GPT-4o-mini demonstrates stronger performance on more complex refactoring types, such as Move Method and Extract and Move Method. Furthermore, we find that adding retrieval context via few-shot examples and using a multi-agent workflow significantly improve performance, with the multi-agent approach achieving the highest success rate. We release SWE-Refactor and all evaluation results to support future research on LLM-based code refactoring.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SWE-Refactor, a benchmark for evaluating LLMs on code refactoring using 1099 "pure" instances from 18 real-world Java projects. The contributions are ensuring data purity by filtering out non-refactoring changes and rigorously verifying each instance for correctness using the projects' full test suites. The benchmark provides repository-level context and covers both atomic and compound refactoring types. An evaluation of 9 LLMs finds that a multi-agent workflow performs best, but with a modest top success rate of 52.7%.

### Strengths
- Originality: The paper addresses a clear gap in existing work. Current refactoring benchmarks are often noisy, lack repository-level context, and are heavily biased toward Python. SWE-Refactor’s focus on pure, verifiable, context-rich refactorings in Java is an original contribution.
- Quality: Verifying each refactoring via compilation and the full test suite is a significant improvement over prior benchmarks, ensuring models are evaluated on the refactoring task itself.
- Clarity: The paper is well-written, and the problem, solution, and construction pipeline are all clearly articulated.

### Weaknesses
- Limited Scope: The benchmark's primary weakness is its focus on six well-defined, syntactic refactoring types (e.g., Extract Method) that are already reliably automated by modern IDEs. It misses the opportunity to evaluate models on more complex, semantic refactors that lack tool support.
- Outdated Model Evaluation: For a paper targeting a 2026 conference, the model selection is not representative of the current state of the art. While the evaluation includes 2024 models like GPT-4o-mini and DeepSeek-V3, the rapid advances throughout 2025 have introduced significantly more powerful models (e.g. Claude 4 series, or GPT reasoning models).
- Lack of Difficulty: Related to the point above, the benchmark's difficulty appears mismatched with current state-of-the-art agentic frameworks. While the paper's top score is 52.7% with a multi-agent workflow, other recent work has shown much higher performance on similar tasks. For instance, the MANTRA multi-agent framework, which is designed for this type of method-level refactoring, reports an 82.8% success rate on a similar "pure refactoring" dataset [1]. And the public leaderboard for the popular aider coding tool shows a recent version of Claude 3.5 Sonnet achieving 92.1% accuracy on its refactoring benchmark [2]. This suggests the benchmark may already be largely solved by SOTA agents, limiting its long-term utility for a 2026 conference.
- Contradictory Scalability Claims: The paper claims its "fully automated... pipeline" is an advantage. However, this pipeline produced only 1099 examples. The authors themselves concede this scale is "still limited". This small size fails to convincingly demonstrate the claimed scalability.

---
[1] Xu, Y., Lin, F., Yang, J., Chen, T. H., & Tsantalis, N. (2025). MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration. arXiv preprint arXiv:2503.14340.

[2] Aider. (2025). Refactoring Leaderboard. https://aider.chat/docs/leaderboards/refactor.html

### Questions
1. Could the authors provide results from more recent SOTA models (e.g., GPT-4o, Claude 4.x Sonnet, Gemini 2.5 Pro) and agentic scaffolding (e.g., OpenAI Codex, Claude Code)? This is crucial to demonstrate that the benchmark is not already "solved" and remains a useful challenge.
2. What was the main bottleneck in the "automated pipeline" that limited the dataset to 1099 instances? Specifically, what was the filtering rate? (i.e., how many candidate refactorings were discarded by PurityChecker for being "impure"?) This would clarify the trade-off being made between purity and scale.
3. Given that the included refactoring types are largely automated by IDEs, can the authors elaborate on the practical value of having LLMs solve these specific tasks, as opposed to focusing on more complex, semantic refactors that lack any tool support?

### Soundness
3

### Presentation
4

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
In this work, the authors propose a benchmark for code refactoring. Given that this task is heavily reliant on memory (i.e., structure and class hierarchies in a code repository), as well as inter-component dependencies; it captures a whole different nuance in coding agents (as opposed to bug-fixing as in SWE-Bench et al.). The authors present a methodical approach for data collection, directly addressing the limitations of existing code refactoring benchmarks; while designing a two-agent, iterative feedback loop to incorporate repository awareness. However, the evaluation remains flawed (discussed more in Weaknesses), which fails to capture the true complexity of refactoring quality and behavioral preservation.

### Strengths
1. Code refactoring is a complex SE task that demands systemic understanding of class hierarchies and inter-file dependencies. Through this work, the authors correctly shift coding agent evaluation away towards context-dependent reasoning, which is essential for agent development.

2. The data collection is methodical and significantly improves over the existing refactoring benchmarks, which typically suffer from ambiguous or noisy data.

### Weaknesses
1. The Reviewer Agent appears to rely on: static analysis, which cannot verify functional correctness; and pre-existing test suite, which inherently assumes completeness of the test suite. The other metric, CodeBLEU, is in itself brittle.
 
2. The multi-agent improvements are notable, but the discussion does not analyze why they succeed (e.g., self-critique vs. feedback loops) or whether such workflows generalize across domains. A breakdown of iteration counts or failure recoveries would add clarity.

3. While the methodical data collection is a strength, the benchmark itself is limited to Java; and would greatly benefit from being designed for multiple programming languages.

### Questions
1. Beyond eliminating noisy refactoring commits, how did the authors ensure that the original code's test suite, used for validation, is stable and functionally complete?

2. Besides the final correctness and CodeBLEU score, did the authors quantify the efficiency of the iterative process? Specifically, what are the metrics for the token consumption per task and the average number of iterations required to reach a correct solution?

3. What are the technical barriers and necessary tool replacements for adapting SWE-Refactor to a second language?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SWE-Refactor, a new benchmark for evaluating LLMs on real-world Java code refactoring tasks. The benchmark contains 1,099 pure refactorings from 18 Java projects, covering both atomic and compound refactoring types. This paper propose fully automated pipeline for benchmark construction and evaluate 9 popular LLMs, finding that DeepSeek-V3 achieves best performance with 41.58% success rate.

### Strengths
- This paper is well-written and addresses important limitations such as supporting compound refactorings, ensuring pure refactorings without noise, and providing an automated construction pipeline.
- This method has built comprehensive evaluation metrics and conducted extensive experiments, which provide the community with valuable insights.

### Weaknesses
- This benchmark focuses only on Java limits generalizability. While authors justify this choice, it's significant limitation for comprehensive LLM evaluation.
- This benchmark contains only 1,099 samples across 6 refactoring types, some categories have very few examples, which may raise some biases.

### Questions
- How do you ensure RefactoringMiner's 99% precision  translates to correct ground truth, given potential tool errors?
- Could you provide quantitative comparison of data quality between SWE-Refactor and RefactorBench on overlapping refactoring types?

### Soundness
3

### Presentation
3

### Contribution
3
