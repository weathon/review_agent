# CONCUR: Benchmarking LLMs for Concurrent Code Generation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
Leveraging Large Language Models (LLMs) for code generation has increasingly emerged as a common practice in the domain of software engineering. Relevant benchmarks have been established to evaluate the code generation capabilities of LLMs. However, existing benchmarks focus primarily on sequential code, lacking the ability to effectively evaluate LLMs on concurrent code generation. Compared to sequential code, concurrent code exhibits greater complexity and possesses unique types of bugs, such as deadlocks and race conditions, that do not occur in sequential code. Therefore, a benchmark for evaluating sequential code generation cannot be useful for evaluating concurrent code generation with LLMs. To address this gap, we designed a benchmark CONCUR specifically aimed at evaluating the capability of LLMs to generate concurrent code. CONCUR consists of 43 curated concurrency problems and leverages formal methods techniques, namely model checking, to assess the correctness of the generated code. We conducted an evaluation of a range of LLMs on CONCUR, highlighting limitations of current models. Overall, our work provides a novel direction for evaluating the capability of LLMs to generate code with focus on concurrency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CONCUR, the first benchmark specifically designed to evaluate large language models on concurrent program generation — a domain much more complex than sequential code due to non-deterministic thread scheduling, synchronization, and concurrency-specific bugs (e.g., deadlocks, race conditions, starvation).
The benchmark includes 43 curated Java concurrency problems derived from Java Concurrency in Practice (Goetz, 2006), each with structured prompts and verified ground-truth solutions.

CONCUR employs formal verification via Java Pathfinder for exhaustive state-space exploration, detecting concurrency issues beyond what static metrics or unit tests can capture. The authors evaluate 22 state-of-the-art LLMs (including GPT-5, Claude-Opus-4.1, GPT-4o, Qwen-3, DeepSeek-R1, etc.) and demonstrate that while LLMs can often produce compilable code, they frequently fail under full concurrency verification.
The study further shows that static similarity metrics like CodeBLEU fail to correlate with true correctness, emphasizing the need for formal, dynamic evaluation frameworks.

### Strengths
- Novel Benchmark Domain

Comprehensive benchmark targeting concurrent code generation, addressing a crucial but overlooked area in code intelligence research.

- Formal Verification Integration

The use of model checking introduces rigor, enabling the detection of deep concurrency bugs (deadlocks, race conditions) that conventional test-based evaluation misses.

- Comprehensive Empirical Evaluation

Evaluates 22 diverse LLMs under uniform conditions, includes manual validation, and provides a public leaderboard and dataset.

### Weaknesses
- Limited Dataset Scale and Diversity

Only 43 problems, all from a single Java text book, limit coverage and generalization to broader concurrency paradigms (e.g., message passing, lock-free, or distributed models).

- Single-Language Restriction (Java 8)

The benchmark excludes other major concurrency ecosystems like C++, Go, or Rust, reducing cross-language insight.

- Partial Coverage of Concurrency Semantics

JPF bounds and model-checking limitations (e.g., no livelock detection, time depth cutoffs) may miss certain concurrency issues, limiting completeness of the evaluation.

- Evaluation granularity

The analysis could benefit from qualitative insights into why models fail (e.g., incorrect synchronization pattern, wrong locking scope).
No ablation on prompt variants or temperature settings.

### Questions
– Do the authors plan to extend CONCUR to multiple programming languages (e.g., Go, Rust) that have different concurrency models? This would broaden its applicability and reveal model generalization across paradigms.

– Since JPF focuses on low-level interleavings, have the authors considered incorporating semantic invariants or assertion checking to detect higher-level correctness violations (e.g., protocol violations, data consistency)?

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
The authors develop a new benchmark for concurrent programming, which consists of 43 curated concurrency problems drawn from a textbook, with structured prompts and ground-truth implementations in Java.  The authors evaluate 22 LLMs on the benchmark.

Unlike most coding benchmarks, success is not determined by the ability to pass unit tests, because concurrent programming bugs like race conditions and deadlocks are notoriously difficult to catch with unit tests.  Instead, the authors perform model-checking with Java Pathfinder (JPF), to find concurrency-related bugs in the output code.  

Somewhat surprisingly, the vast majority of errors are still compilation errors, with the majority being syntax errors.  LLMs still struggle to write syntactically valid code.   A second surprise is that LLMs often implement conceptually correct code, but then fail to actually spawn multiple threads to execute it.  

The authors find that CodeBLEU is a poor metric for code quality.

### Strengths
The benchmark seems to be well-designed, and the evaluation of the 22 LLMs is thorough, covering all of the important models.  

By far the greatest strength of this benchmark is its use of model-checking to catch concurrency bugs.  As the use of coding LLMs continues to proliferate, so will LLM-introduced bugs, and concurrency-related bugs like race conditions are notoriously difficult to catch. Formal static or dynamic analysis is currently severely underused as a way to evaluate code quality, so I will champion this paper as an important milestone in teaching LLMs to write correct code, by using formal measures of code correctness.  Hopefully future work will follow the same path.

### Weaknesses
The benchmark only contains 43 problems.  

Perhaps most importantly the problems are all drawn from a textbook, which was published almost 20 years ago.  This means that the textbook, or similar problems, are likely in the training data of SOTA LLMs.  

The benchmark would be strengthened by having more problems, and including not-previously published problems.  It would be interesting to include problems in a language other than Java -- e.g. the Rust type system also protects against concurrency bugs.

### Questions
None.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a benchmark that measures how well code generation models are able to generate concurrent Java code. The authors construct a small problem set sourced from a Java concurrency textbook and measure success based on compilation success and Java PathFinder (JPF) verification.

### Strengths
- Paper is well-written and motivations are sound (most code benchmarks focus on single proc. code).
- Thoughtful methodology. Steps are taken to ensure solutions can be verified without excessive computation resources (e.g. by limiting num threads, etc)
- Paper shows benefit of going beyond CodeBLEU and provides error analysis of top models.

### Weaknesses
- Benchmark is simple and problems are sourced from a textbook published in 2006. The proposed test set is only 43 problems, curated by the authors, which is extremely small. Additionally there is a risk of contamination as models may have trained on this textbook. The authors do not provide any insight into the contamination risk. 
- Evaluation metric is based on compilation and verification success, not test case passing. Due to the nature of the problems (which are presented without completed solutions in the original data source), the authors resort to compilation and JPF based checks to measure success. However, the best verification for code is running against a test suite as code that can compile and pass JPF checks could still be wrong. 
- Benchmark initial performance is quite high-- GPT-5 is at 72% pass@1.

### Questions
1) Can the authors explain any experiments or measures taken to avoid contamination given that the textbook may be in the training set of these LLMs?
2) Is the proposed test set (just 43 problems) a statistically significant sample size for estimating concurrent code generation capabilities?
3) Why is compilation success + JPF verification a sufficient measure of performance (instead of say compilation + JPF + running code against test suite)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces CONCUR, a benchmark specifically designed to evaluate large language models on concurrent code generation. It contains 43 carefully curated multi-threading problems with structured prompts and validated Java implementations. Unlike existing benchmarks focusing on sequential code, CONCUR leverages Java PathFinder (JPF) for dynamic, exhaustive verification of generated programs, detecting concurrency errors such as deadlocks, race conditions, and single-thread violations. By integrating structured prompts, ground-truth solutions, and dynamic model checking, CONCUR provides a rigorous and reproducible framework to systematically assess LLMs’ ability to produce correct multi-threaded programs.

### Strengths
1. Novelty in Benchmark Design: The paper introduces CONCUR, the first benchmark specifically targeting multi-threaded code generation, filling a gap left by prior benchmarks that focus only on sequential programs.

2. Concurrency-Aware Problem Construction: The benchmark is carefully designed to enforce multi-threading features and concurrency-specific requirements, ensuring that generated programs must exhibit correct thread behavior and handle potential concurrency issues.

3. Clear Presentation and Comprehensive Validation: The paper is well-structured and clearly written, and it demonstrates the benchmark’s effectiveness by evaluating LLMs of various sizes and architectures, providing strong evidence for its reliability in systematically assessing concurrent code generation.

### Weaknesses
1. The benchmark includes only 43 Java programs, which is a relatively small number and may limit its coverage of diverse concurrent programming scenarios.

2. Although prompts for each program are provided in the public repository, the programs themselves are simple in functionality and description, which may not effectively evaluate LLMs’ ability to generate complex multi-threaded code.

### Questions
1. Could the benchmark be expanded with additional test programs to enhance coverage and diversity of concurrent scenarios?

2. Could the authors discuss the complexity of program functionality and explain the code selection process in more detail? Additionally, could they consider including more complex, real-world concurrent programs that better reflect practical programming scenarios?

3. Could the benchmark extend the oracle to include dynamic testing oracles, if feasible, for more comprehensive correctness evaluation?

### Soundness
2

### Presentation
3

### Contribution
3
