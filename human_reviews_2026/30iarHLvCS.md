# SuperCoder: Assembly Program Superoptimization with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Superoptimization is the task of transforming a program into a faster one while preserving its input–output behavior. In this work, we investigate whether large language models (LLMs) can serve as superoptimizers, generating assembly programs that outperform code already optimized by industry-standard compilers. We construct the first large-scale benchmark for this problem, consisting of 8,072 assembly programs averaging 130 lines, in contrast to prior datasets restricted to 2–15 straight-line, loop-free programs. We evaluate 23 LLMs on this benchmark and find that the strongest baseline, Claude-opus-4, achieves a 51.5% test-passing rate and a 1.43× average speedup over gcc -O3. To further enhance performance, we fine-tune models with reinforcement learning, optimizing a reward function that integrates correctness and performance speedup. Starting from Qwen2.5-Coder-7B-Instruct (61.4% correctness, 1.10× speedup), the fine-tuned model SuperCoder attains 95.0% correctness and 1.46× average speedup, with additional improvement enabled by Best-of-N sampling and iterative refinement. Our results demonstrate, for the first time, that LLMs can be applied as superoptimizers for assembly programs, establishing a foundation for future research in program performance optimization beyond compiler heuristics. Our code is available at https://anonymous.4open.science/r/SuperCoder/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper evaluates LLMs on the task of code "superoptimization" (creating fast assembly code beyond the rule-based transformations used by compilers), and proposes reinforcement learning for it.

The paper constructs a corpus of interesting assembly code programs from code competition C++ competitions, restricting to those where GCC's -O3 optimization creates large speedups. On this benchmark, LLMs are evaluated for the speedups they are able to obtain by transforming the original assembly code. A transformation is deemed correct if it still passes the unit test cases provided by the coding competition.

### Strengths
- The paper might be the first to evaluate LLMs on this task. (* I am not too familiar with the literature and cannot fully judge the novelty claims.)
- The reinforcement learned model matches the performance of Claude Opus 4 while having 7B parameters.

### Weaknesses
- Limited scope: unlike code synthesis, where humans can review and fix the generated code, tasks like compilation and superoptimization are too hard to review manually. It is not clear what the use case of an unverified code superoptimizer may be, apart from very tight inner loops in high performance computing?
- Limited contribution: the evaluations are fairly limited (no "maximum speedup at k samples" scaling curves, no experimentation with prompts despite the analysis of failure modes).
- Limited set of methods: what about evolutionary / prompt learning strategies for instance? What about iterative infernce / bug fixing? After setting up the task, there are many things to try, the paper scratches at the very surface.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SuperCoder, a framework that applies large language models (LLMs) to the task of assembly program superoptimization—that is, generating faster assembly code while preserving correctness relative to compiler-optimized baselines (gcc -O3). The authors construct a new benchmark of 8,072 assembly programs (average ≈ 130 LOC) derived from CodeNet and propose to fine-tune a code LLM using reinforcement learning (PPO / GRPO) with a reward that jointly measures functional correctness and execution-time speedup.

### Strengths
1. The topic is ambitious and relevant, connecting large language models, reinforcement learning, and compiler optimization. 

2. The paper attempts to move beyond high-level code generation to a more demanding low-level optimization setting. 

3. The experimental setup is relatively thorough and could stimulate future work on applying learning-based methods to compiler research.

### Weaknesses
1. The task formulation is loosely defined, relying on test-based correctness rather than formal equivalence, which makes the conclusions uncertain.

2. The dataset is built from code of unclear quality and has not been validated as a standard benchmark. Its representativeness and reproducibility are questionable.

3. The reinforcement learning setup functions as a simple post-hoc reward filter rather than a meaningful sequential learning process. There is no comparison to simpler fine-tuning strategies.

4. The empirical results appear overstated and are not supported by sufficient statistical or profiling evidence. Claims of performance improvements over compiler baselines are not convincingly justified.

### Questions
1. Can the dataset and scripts be released for independent verification?
2. How does reinforcement learning compare with standard supervised fine-tuning?
3. What specific optimizations account for the reported performance gains?

### Soundness
3

### Presentation
2

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
This paper introduces a dataset for evaluating whether LLMs can superoptimize x86-64 assembly beyond gcc -o3. The authors build a benchmark of 8072 programs (~130 assembly LOC) with high coverage tests and evaluate 23 models on the dataset. A collection of open/closed source models were evaluated on the benchmark, with the best baseline averages a 1.43x speedup. Experiment results indicate RL fine-tuning techniques (PPO and GRPO) can significantly boost test pass rate and average speedup.

### Strengths
- The paper uses top closed-sourced (claude-opus-4/gpt-5) and open-sourced (DeepSeek-V3, Llama-4 etc.) models to study the impact of their dataset. It shows the gap between closed and open LLMs and how different adaptations can narrow it.
- The paper builds a realistic, large benchmark with ~130 LOC, and many with loops. This is rare for assembly-level work and targets an important problem.
- The analysis of learned programs provides qualitative insight into how LLMs are improving already optimized assembly programs.

### Weaknesses
- Dataset deliberately samples programs with large -o0 -> -o3 gains, results might be different on other code distributions.
- The paper could benefit from automatically explaining why the programs are faster, maybe using a chain-of-though model here to optimize the program and output thinking tokens would help?

### Questions
- This is mainly on the consistency of the timing results. Were caches/TLBs warmed consistently? Is there any input randomization? Were results stable across code boots/container restarts beyond hyperfine's warmup?
- SuperCoder is single-shot sampling, how do results change with best-of-k samples?
- Is there any de-duplication across the train/eval when constructing the dataset? (for example similar codenet submissions leading to highly similar assembly code)

### Soundness
3

### Presentation
3

### Contribution
3
