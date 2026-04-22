# Adversarial Agent Collaboration for C to Rust Translation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Translating C to memory-safe languages, like Rust, prevents critical memory safety vulnerabilities that are prevalent in legacy C software.
Existing approaches for C to safe Rust translation, including LLM-assisted ones, do not generalize on larger ($> 500$ LoC) C codebases because they depend on complex program analyses that frequently break.
In this work, we present ACToR (**A**dversarial **C** **To** **R**ust translator), a simple LLM agent-based approach. 
Inspired by GANs, ACToR pits a generator agent against a discriminator agent, which collaborate to iteratively generate a Rust translation. 
On each iteration, the translator agent synthesizes and refines a Rust translation to pass an existing suite of tests, and then the discriminator agent finds new failing tests.
We demonstrate that ACToR translates all of the 63 real-world command-line utilities considered in our benchmarks, which have an average size of 473 lines of code, and it achieves over 90\% test pass rate with zero human intervention during translation.
To our knowledge, it is the first work to show evidence that an agent-centric approach can reliably and automatically convert standalone command-line C programs at this scale.
Furthermore, ACToR improves translation correctness by up to 25.1\% compared to baseline, non-adversarial approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ACToR (Adversarial C To Rust translator), an adversarial two-agent workflow for translating C programs to safe Rust. A translator LLM agent iteratively produces a Rust version, while a discriminator agent searches for counterexample tests that expose behavioral mismatches. The newly found failing tests are added to the test suite. The method is evaluated on two benchmarks: a micro-benchmark and a macro-benchmark. ACToR improves translation correctness by up to 18.9% compared to the baseline approach.

### Strengths
1. Novel and Effective Framework: This work tries to address a well-known weakness in automated code generation, i.e., the difficulty of ensuring semantic equivalence beyond a fixed set of tests
2. Good ablation studies: the paper isolates the effect of adversarial test generation and a simple fuzzer
3. Include evaluation with the agentic frameworks.

### Weaknesses
1. Correctness is proxied by tests. However, tests may not be able to expose real bugs. No human validation of the translated code or the quality / sufficiency of the test cases.
2. I am wondering how well this method could generalize to other existing datasets, e.g., CRUST-Bench: A Comprehensive Benchmark for C-to-safeRust Transpilation?
3. The cost estimation (e.g., tokens / API cost) seems to be neglected for comparison and evaluation of different approaches.
4. There are many papers that use LLMs for code translation. I am wondering about the technical novelty compared with prior approaches. Also, other papers claim that they do repo-level translation. Why do you claim that your work is the first to work on large-scale programs? Is it really large-scale compared with repo-level's prior work?

### Questions
1. How do you compare your benchmark to "CRUST-Bench: A Comprehensive Benchmark for C-to-safeRust Transpilation"?
2. Is there any guarantee that the final generated Rust programs are really memory-safe? Or the agents could also escape certain checks to bypass? How much confidence in correctness can you get out of the generated Rust programs? Is there any manual validation?
3. What if you only use the fuzzing script without the agent?
4. Can you compare your approach with the other paper, "Exploring and Unleashing the Power of Large Language Models in Automated Code Translation", and tell me what your technical novelty is? Similarly, for this paper, "AlphaTrans: A Neuro-Symbolic Compositional Approach for Repository-Level Code Translation and Validation".

### Soundness
2

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
This paper tackles the task of translating C code to safe Rust code with an adversarial setup of LLM agents. By having an agent that performs the translation interact with another that comes up with tests that expose the mismatch between C and Rust code, ACToR steers translation towards general correctness instead of overfitting to a fixed test suite. The authors showed that ACToR is effective on 63 real-world programs of non-trivial size and outperforms naive, coverage-guided, and ablated approaches.

### Strengths
This work addresses an important problem using an effective technique that makes intuitive sense and is well-executed. It makes sense to grow the test suite dynamically, and in addition for tests to maximally distinguish code in the two languages. Giving the discriminator agent access to a fuzzing script is a nice touch. As the authors convincingly demonstrate through pass rates relative to each competing tools, these bits give ACToR the edge over baselines and ablations. The paper is well-written, well-structured, and clearly presents this approach and findings. While not ground-breaking and despite a few issues below, I find this work to be a meaningful contribution to the field overall.

### Weaknesses
My complaints with the paper are mostly the following:

1. Unsound problem definition. The authors' definition of C-to-Rust translation does not require exclusive use of safe features, yet claims that "memory safe handling of malicious inputs is guaranteed by the Rust compiler itself." Since unsafe code may be involved, this assumption does not hold - it is entirely possible for inputs outside the valid universe U to expose vulnerabilities on the Rust side, which may be the same as or different from those on the C side, and which, at worst, are new vulnerabilities altogether. As such scenarios are not considered in the behavior equivalence check, such a problem definition (and any of its solutions) would defeat the purpose of what motivates translating from C to Rust in the first place (though ACToR apparently does not literally address this problem definition, as it is said to enforce safe Rust as a post-processing step).

2. Unprincipled experimental configuration. The decision to use 10 turns and 3 tests each turn feels handwavy. It seems entirely likely that with more turns and more tests added, more divergences between the C code and Rust code would be exposed. While I understand that the authors are limited by the cost of running the LLMs, I think it is very worthwhile to experimentally study the trade-offs between various (# turns, # tests) configurations and how they impact the degree of translation correctness ACToR can achieve.

### Questions
1. Could the authors comment on the issue with the unsound problem definition?
  
    Relatedly, what does the authors mean by a post-processing step that enforces safe Rust? Is it a check that looks for unsafe code, or does it transform unsafe code to safe code? Regardless, it is my opinion that a problem definition that requires safe Rust use and a design that enforces safety in transit would make ACToR more principled and impactful. One benefit, for instance, is that keeping Rust code safe throughout the process would prevent overhead associated with backtracking from unsafe code and steer the translation along a optimistic/promising direction. Modifying ACToR in this respect should not be too difficult to implement.

2. Would the authors consider studying the trade-offs between various experimenal configurations and their impact on translation correctness?

Minor: Could the authors briefly comment on how the "15 manually crafted, diverse seed tests" are created? I would also be curious to learn how this initial test suite could impact ACToR's performance. Would the authors be open to consider doing an ablation study on this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ACToR, an adversarial two‑agent framework for translating C programs to safe Rust. A translator agent proposes a Rust program while a discriminator agent generates tests, including fuzzing‑aided cases, to expose behavioral mismatches, and the process iterates. On 6 micro programs and 57 BSDCoreUtils utilities (median 485 LOC), ACToR reports high test pass rates, outperforming non‑adversarial baselines and showing up to 18.9% relative pass‑rate gains on the macro benchmark.

### Strengths
- The paper addresses a timely and challenging problem: translating nontrivial C codebases to safe Rust with minimal human intervention.
- ACToR consistently improves over the naive baseline across three agent-model choices.

### Weaknesses
- The core comparisons are against a “naive” single-agent baseline and a “coverage” baseline. There is no head‑to‑head comparison with recent C→Rust systems that combine LLMs with analysis.
- The evaluation lacks runtime, token, and financial cost information, which is key for judging practicality at scale.
- Results appear to be single‑run without variance across seeds or model nondeterminism. Stability across runs is not reported. The text notes three programs that aborted before 10 iterations after multiple reruns, hinting at variability. 
- The paper states “to our knowledge, it is the first such system that reliably translates C programs of this scale.” Given the breadth of recent efforts that blend LLMs with program analysis, it would be safer to qualify the claim with the particulars of the benchmark and evaluation protocol, or provide a stronger head‑to‑head against at least one recent approach on a shared subset.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
