# A Lightweight Heuristic for Detecting Unfair Tests in Software Engineering Benchmarks

- Decision: Reject
- Scores: 2, 0, 4

## Abstract
Software engineering benchmarks are useful tools for evaluating the programming abilities of large language models (LLMs). In addition to ranking models against each other, they can help us to situate the current state of the art by leveraging real-world software engineering problems. For some benchmarks, this latter function is compromised by the presence of "unfair" tests, meaning tests that contain requirements not specified in the corresponding issue descriptions. Unfortunately, the manual identification of unfair tests is an expensive and time-consuming process; this is especially problematic for automated curation pipelines and continuously-updated benchmarks. There are promising LLM-based solutions, but these come with the usual drawbacks: complex scaffolding, prompt sensitivity, lack of reproducibility and environmental cost; in addition, low recall means the majority of unfair tests are unlikely to be identified. As an alternative to both manual and LLM-based approaches, we propose a lightweight, fully-deterministic, heuristic for the detection of unfair tests in software engineering benchmarks. We evaluate our heuristic against the human annotations used to curate SWE-bench Verified and we compare the results to the corresponding evaluations of two LLM-based alternatives (aligning our methods to facilitate a direct comparison). We find that the accuracy of our heuristic exceeds the accuracy of all non-fine-tuned configurations of both alternatives, but does not exceed the accuracy of a fine-tuned configuration. Given the additional effort, complexity and environmental impact associated with fine-tuning, we consider this to be a positive result. We further propose a version of our heuristic that is less precise, but more sensitive, exceeding the recall of both a fine-tuned, and non-fine-tuned, LLM-based alternative.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a lightweight heuristic for detecting unfair tests in software engineering benchmarks, a subset of underspecified issues where validation tests rely on information absent from the issue description. The heuristic aims for simplicity, determinism, and reproducibility, and is positioned as an alternative to LLM-based curation methods. The heuristic identifies instances as “unfair” when overlapping string, numeric, or identifier tokens between the test and solution patches are missing from the issue description. Through experiments on SWE-bench and SWE-bench Verified, and by comparison with SPICE and SWE-Rebench, the paper shows that the heuristic achieves accuracy slightly above non–fine-tuned LLM models, though below fine-tuned ones.

### Strengths
- The method is simple, deterministic, and reproducible. It avoids the complexity and cost of LLM-based alternatives.
- The empirical evaluation compares directly against two existing automated curation pipelines (SPICE and SWE-Rebench).
- Implementation and datasets are documented and reproducible.

### Weaknesses
- The contribution is narrow. The paper solves a subproblem (focusing only on unfair tests, a limited subset of underspecification issues) in an already niche topic (software engineering benchmark curation).
- As a result, the practical utility is unclear: the paper does not convincingly show that unfair tests are a major bottleneck in creating benchmarks, compared to the overall effort, nor that removing them improves benchmark usefulness.
- Comparisons are weak: experiments only compare against random baselines and indirect LLM results. As a side note, a random classifier is always expected to yield 50% accuracy.
- Strong assumptions about variable naming and code quality limit (many valid tests may use inconsistent or auto-generated identifiers).
- Due to mild performance, the proposed heuristic risks flagging many benign instances, which raises anew the need for manual inspection (the very problem that the proposed approach aims to avoid).

### Questions
I do not have any specific question whose answer could change my opinion.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
This paper proposes a lightweight, deterministic heuristic to identify "unfair tests" in SWE-bench, which is presented as a low-cost, reproducible alternative to expensive, non-deterministic LLM-based curation pipelines.

### Strengths
The paper demonstrates a valid test case issue in its methodology section.

### Weaknesses
- The paper fails to provide a clear and formal method. In its method section, it shows an example instead of a formal methodology. Additionally, the example seems hard to generalize to other problems for software engineering benchmarks.

- While high-quality test cases are needed, the method proposed in this work is abstract, making it difficult to find real-world test cases that meet the filtering requirements (as question requirements, patch fixes, and test case token distributions are often completely different). Crucially, the paper does not sufficiently justify the filtering, as an unfair test case that causes all models to fail would not affect the relative benchmark scores.

- The paper is densely written and very hard to follow.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a lightweight heuristic method for detecting "unfair tests" in SWE-bench–style benchmarks by comparing identifiers, literals, and AST-extracted tokens between issues and patches. The goal is to identify cases where tests reveal patch content that the model should not have access to. The authors compare the heuristic approach to SWE-bench Verified, SPICE, and SWE-Rebench methods, reporting competitive precision while avoiding LLM inference cost.

### Strengths
- Practical problem: test fairness in SWE-bench-style datasets
- Reproducible and deterministic method
- Lightweight — no compute burden unlike LLM curation pipelines
- Good comparison against benchmark curation systems

### Weaknesses
- Limited novelty (heuristic token matching)
- Low recall limits real-world usefulness
- Over-generalized claims vs existing LLM curation tools
- Relies on noisy SWE-V labels as "ground truth"
- No hybrid or prompting baselines
- No error analysis or qualitative insights
- Python-only implementation limits generalizability

### Questions
1. Why no comparison to a simple GPT-4 zero-shot fairness-classification prompt?
2. Can you report confusion matrix and qualitative cases?
3. How sensitive is performance to project domain or code style?
4. Could hybrid approaches (heuristics + small model) improve recall?
5. Are multi-file patches or cross-file identifiers considered?

### Soundness
2

### Presentation
3

### Contribution
2
