# Code2Bench: Scaling Source and Rigor for Dynamic Benchmark Construction

- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
The evaluation of code-generating Large Language Models (LLMs) is fundamentally constrained by two intertwined challenges: a reliance on static, easily contaminated problem sources and the use of superficial, low-rigor testing. This paper introduces a new benchmark construction philosophy, Dual Scaling, designed to systematically address both limitations. Our approach involves continuously
scaling the source of problems from dynamic, real-world code repositories and systematically scaling the rigor of tests via automated, high-coverage Property-Based Testing (PBT). We instantiate this philosophy in CODE2BENCH, an end-to-end framework that leverages Scope Graph analysis for principled dependency classification and a 100% branch coverage quality gate to ensure test suite integrity.
Using this framework, we construct CODE2BENCH-2509, a new benchmark suite with native instances in both Python and Java. Our extensive evaluation of 10 state-of-the-art LLMs on CODE2BENCH-2509, powered by a novel "diagnostic
fingerprint" visualization, yields three key insights: (1) models exhibit a fundamental performance gap, excelling at API application (Weakly Self-Contained tasks) but struggling with algorithmic synthesis (Self-Contained tasks); (2) a model’s
performance is profoundly shaped by the target language’s ecosystem, a nuance we are the first to systematically quantify; and (3) our rigorous, scaled testing is critical in uncovering an "illusion of correctness" prevalent in simpler benchmarks. Our work presents a robust, scalable, and diagnostic paradigm for the next generation of LLM evaluation in software engineering. The code, data, and results are available at https://code2bench.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel benchmark construction paradigm termed "Dual Scaling", which addresses two critical challenges in evaluating code generation LLMs: the reliance on static, easily contaminated problem sources and the use of superficial testing methods. The authors present the Code2Bench framework, which dynamically acquires problems from real-world code repositories (Scaling the Source) and integrates property-based testing (PBT) with a 100% branch coverage quality gate (Scaling the Rigor) to build the Code2Bench-2509 benchmark. Empirical evaluations demonstrate that this benchmark effectively uncovers performance gaps between models in algorithmic synthesis (SC tasks) and API application (WSC tasks), while quantifying the influence of language ecosystems on model behavior.

### Strengths
- The 100% stringent branch coverage gate and large PBT-generated suites substantially reduce false positives and expose "near-perfect" failures that many benchmarks miss.
- The outcome spectrum and “diagnostic fingerprints” provide more granular failure analysis (SyntaxErr/RuntimeErr/LogicErr vs. partial pass bands), illuminating the algorithmic synthesis vs. API-application divide and the role of language typing in error suppression.
- WSC-Python spans >35 libraries; SC-Java demonstrates multi-language extensibility. Tasks show higher cyclomatic complexity and test volume than legacy benchmarks.

### Weaknesses
- Some of the figures in the paper are not very clear or visually polished. For example, in Figure 2, there is noticeable overlap between text elements and between text and icons, which affects readability. Improving the clarity and layout of the figures would make the presentation more professional and easier to interpret.
- In the Related Work section, the authors assert that existing live benchmarks rely on narrow or specific data sources. However, Code2Bench is also curated from specific GitHub repositories without disclosing the selection criteria, repository sampling strategy, or inclusion/exclusion policies. This lack of transparency makes it difficult to assess source diversity, potential sampling bias, and contamination risk. Moreover, prior benchmarks such as DomainEval also use GitHub repositories to collect domain-specific tasks. The novelty of "Scaling the Source" appears limited.
- Evaluation scope focused on functional correctness: Important dimensions like performance/efficiency, readability/style, security, robustness to invalid inputs, and documentation/test generation are not directly evaluated.
- Project-Dependent problems are mostly discarded in this pipeline. Although LSC is discussed as future work, the current evaluation does not yet include multi-function or multi-file context tasks that matter in industrial settings (e.g. I/O, resource handling, exceptions, and protocols).

### Questions
- It is unclear what criteria were used to assess the testing rigor for all benchmarks presented in Figure 2. Could the authors clarify how testing rigor is defined and measured?
- Why is the Java track limited to SC only? Where is WSC-Java?
- The authors state that each benchmark task includes approximately 500 test cases. Given that these test cases are selected after PBT generation and a 100% branch-coverage gate, does the pipeline need to generate and evaluate hundreds or even thousands of candidate inputs per task before filtering? If so, what are the actual computational costs (time and resources) per task? Similarly, for the evaluation stage, running ~500 test cases per task can significantly increase runtime and resource usage. Could the authors provide quantitative measurements of the end-to-end evaluation time per model and per task, and discuss any mechanisms (e.g., batching, caching, reduced-cost modes, seed control) used to ensure scalability across many models and large task suites? Guidance on lighter-weight modes that preserve diagnostic value would also be useful.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CODE2BENCH, a benchmark construction framework for evaluating code-generating LLMs. The framework addresses data contamination through temporal filtering (extracting functions from GitHub commits after model knowledge cutoffs) and improves test rigor through Property-Based Testing with 100% branch coverage. Tasks are classified into Self-Contained (SC, pure algorithmic reasoning) and Weakly Self-Contained (WSC, API usage) using Scope Graph analysis. The authors construct CODE2BENCH-2509 with Python and Java tasks and evaluate 10 LLMs, reporting three insights: SC vs WSC performance gap, language ecosystem impact, and "illusion of correctness."

This paper shows good engineering efforts but unclear novelty contribution. Two critical flaws need to be addressed: (1) No validation of benchmark value. The paper lacks direct comparison with existing benchmarks (HumanEval, MBPP, BigCodeBench) on the same models, making it impossible to assess whether CODE2BENCH provides unique insights or simply evaluates differently. (2) No novelty contribution in analysis. the three "key insights" are either expected results (SC/WSC gap), intuitive observations already explored (language ecosystem), or incremental findings without baseline comparison (illusion of correctness).

### Strengths
1. Important problem: Addresses real limitations in LLM code evaluation, which are data contamination and superficial testing.

2. Solid engineering: Scope Graph analysis for dependency classification is technically sound. Property-Based Testing with 100% branch coverage demonstrates rigor. The framework automates benchmark construction.

3. release code, data, and results.

### Weaknesses
1. Lack of Direct Comparison with Existing Benchmarks
For a benchmark paper, it is crucial to demonstrate how the new benchmark compares with existing ones when evaluating the same models. The paper only shows Table 1 comparing characteristics (Dynamic, Rigorous Test, etc.) but lacks direct performance comparison with these baselines. Without evaluating the same 10 models on existing benchmarks, it's impossible to determine whether CODE2BENCH provides unique insights, whether the lower pass rates reflect higher quality or different task distribution, or whether the three "key insights" could be revealed by existing benchmarks. This comparison is essential to establish the benchmark's value.

2. Missing Details of Validating Benchmark Construction Method
The paper proposes Dual Scaling with temporal filtering, Scope Graph classification, and PBT with 100% coverage, but provides less details to validate these components. What happens with 80% coverage instead of 100%? Does temporal filtering applied to HumanEval produce similar contamination resistance?  An ablation study may be a good way to explore these points.

3. LLM Analysis Lacks Technical Contribution and Novelty
The paper presents three "key insights" as major contributions: (1) SC vs WSC performance gap. it is an expected result since BigCodeBench already focuses on API tasks and it's well-known these are different skills; (2) language ecosystem impact, which is also intuitive and already explored in HumanEval-X; (3) illusion of correctness. EvalPlus already demonstrated this, and without comparison to EvalPlus, the 6.94% figure lacks context. The analysis is descriptive rather than prescriptive, providing no actionable insights for improving models or evaluation. Without showing these insights are unique to CODE2BENCH or impossible to obtain from existing benchmarks, the analysis appears to justify the benchmark circularly rather than contribute genuine discoveries.

4. Limited Validation of Practical Utility
The paper doesn't show whether performance on CODE2BENCH correlates with real-world coding capabilities (e.g., SWE-bench).

### Questions
Do you have any existing comparison data (even partial) showing how the same models perform on CODE2BENCH vs existing benchmarks? Can you clarify why this comparison was not included in the paper?

Can you clarify how your three key insights differ from findings in BigCodeBench (WSC tasks), HumanEval-X (multi-language), and EvalPlus (test insufficiency)?

Do you plan to validate CODE2BENCH's value through comparison with existing benchmarks? What would be your approach?

### Soundness
2

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
3

### Summary
This paper introduces Dual Scaling, a benchmark construction philosophy in CODE2BENCH framework to scaling from dynamic, real-world code repository and generating rigor test with 100% coverage. Using this method, the authors further build CODE2BENCH-2509, with 411 Python instances and 249 Java instances. The paper conducted comprehensive experiments on closed source models and open source models, and the result suggests a performance gap between API application tasks and algorithm synthesis tasks.

### Strengths
1. The benchmark designs rigorous and strong test cases. It not only accounts for edge cases but also ensures complete test coverage, substantially outperforming other benchmarks that rely on sparse test examples, which may lead to incorrectly judged “pass” cases.

2. The paper provides a carefully designed implementation in both Python and Java, addressing not only translation between languages but also their distinct type systems and library ecosystems. This enables meaningful cross-language comparison and reveals how LLM performance depends on the target language’s constraints.

3. The authors effectively decouple API-calling ability from algorithmic implementation ability. The experiment suggests that models perform better at API usage than at algorithmic reasoning. This insight offers a valuable lens for future work on improving model reasoning.

4. The authors also emphasize clarity and unambiguity when generating instructions, which contributes to the benchmark’s reliability and reproducibility.

### Weaknesses
1. Although CODE2BENCH draws its source data from real repositories, the benchmark tasks remain function-level and isolated. This design simplifies testing but does not capture cross-function or module-level dependencies, which are prevalent in real-world software engineering. As such, the benchmark evaluates isolated reasoning rather than full software generation ability or collaborative code development.

2. As mentioned by the authors, real-world code often includes numerous defensive branches and error-handling structures. The current test generation strategy struggles to fully cover these fragmented control flows, which are often filtered out because they fail the 100% coverage requirements. While this improves test rigor, it also excludes many defensive programming constructs that are significant in real-world software development.

3. The filtering process relies on a fixed list of *allowed libraries* to define “Weakly Self-Contained” tasks. This helps maintain consistency but may also limit domain diversity, since tasks from less common libraries or specialized fields are excluded. As a result, limiting the allowed libraries may constrain the representativeness of the benchmark and may introduce unexpected bias.

### Questions
1. The paper mentions differences between Java and Python fingerprints (Figure 3) as LLM's coding ability intertwined with their target language's ecosystems. Could the authors clarify whether this refers to differences between interpreted and compiled languages, or to other ecosystem-level factors?

2. The framework requires generating hundreds of PBT-based test cases per function and enforcing 100% coverage. Could the authors quantify the computational and time costs of this process?

3. A typo: Last line in page 4: perturbation techniqueZhao -> missing a blank

### Soundness
3

### Presentation
3

### Contribution
3
