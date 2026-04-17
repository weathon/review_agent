# Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal Reinforcement Learning Environment

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Large language models (LLMs) already excel at writing code in high-resource languages such as Python and JavaScript, yet stumble on low-resource languages that remain essential to science and engineering. Besides the obvious shortage of pre-training data, post-training itself is a bottleneck: every new language seems to require new datasets, test harnesses, and reinforcement learning (RL) infrastructure.

We introduce Agnostics, a language-agnostic post-training pipeline that eliminates this per-language engineering. The key idea is to judge code solely by its externally observable behavior, so a single verifier can test solutions written in any language. Concretely, we (i) use an LLM to rewrite existing unit-test datasets into an I/O format, (ii) supply a short configuration that tells the verifier how to compile and run a target language, and (iii) apply reinforcement learning with verifiable rewards (RLVR) in a robust code execution environment.

Applied to five low-resource languages—Lua, Julia, R, OCaml, and Fortran—Agnostics (1) improves Qwen-3 4B to performance that rivals other 16B–70B open-weight models; (2) scales cleanly to larger and diverse model families (Qwen-3 8B, DeepSeek Coder 6.7B Instruct, SmolLM3, Phi 4 Mini); and (3) for open-weight models with ≤16B parameters, sets new state-of-the-art pass@1 results on MultiPL-E and a new multi-language version of LiveCodeBench that we introduce.

We release the language-agnostic training datasets (Ag-MBPP-X, Ag-Codeforces-X, Ag-LiveCodeBench-X), training code, and ready-to-use configurations, making RL post-training in any programming language as simple as editing a short YAML file.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposed a training pipeline that can be used in any programming language with only a small configuration file by describing the input and output specification in pure text format.

* The proposed method does not require a language-specific verifier to verify and provide the rewards but rather evaluate the model on the observable behavior, in this paper, the text output.
* It provides several datasets that have only I/O description and a suite of tests with only input and output. in text format.
* It provides a high optimized GRPO training framework with code execution sandbox.
* It provides a SOTA open-weight 4B model on low resource language under 16B size.

### Strengths
* The training pipeline address the task from a different direction where only evaluating on the input and output, so it can be applied to any language without a language specific verifier. And specifically, reward on the format is offloaded into a prompt prefix instead to instruct the model to follow the format.
* In low resource languages, the claimed method shows good improvement on small models and even beating bigger models trained specifically on the code task.
* The paper is well written, it constructs a clear flow from problem to method to the final evaluation. The experiments, setup and training hyperparameters are described clearly.
* The paper does a thorough analysis on the approach, including the impact on the different model architecture, model size, training data and fine grained analysis on models’ error types.

### Weaknesses
* Given the importance that a good prefix helps avoid the **most common** errors and if the model barely knows a programming language per line 199, and you select several faulty generation for generating the verbatim, I image this has to be representative so that the model can generate a verbatim that can avoid the **most common** errors, so I think it’s necessary to include a description on how to select the faulty generations. 
* Line 76 claims the paper provides a small and highly-optimized training framework, but there is no analysis/experiments/comparison to demonstrate.

### Questions
* Line 240, miss the definition of the $r_{i,t}(\theta)$
* Line 255, please explain OCI
* Line 335. how is the score defined?
* Line 420, can you clarify the $96.
* Line 424, what’s the epochs used in table 2 for Qwen3-4B-CF-Fortran.
* Line 916, what's the 7 manually selected problems and how is it selected?
* Do you have a analysis on the data similarity between CF-X and LCB-X, same for the MBPP-X and LCB-X. This is important to exclude the factor from data similarity in the score improvement.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce Agnostics, a programming language-agnostic LLM post-training pipeline. The main idea is to rewrite the coding questions into a question that are programming agnostic and only works on standard input/output. Then, these questions can be further rewritten into programming language specific questions. Experiment results show that Agnostic can significantly improve the model performance in low resource programming languages.

### Strengths
1. The proposed idea is quite simple and elegant. With the proposed approach, we can theoretically turn coding problems in any programming languages into other languages.
2. It achieves promising results on single language training, showing that models trained with such data achieves much better performance in low resource languages. What's more interesting here is that the improvement generalizes beyond standard I/O problems.

### Weaknesses
1. The number of programming languages tested are limited (5 of them). Moreover, it is only trained on data from a single programming language in each experiment. This is not a usual setup. Ideally, we would like to mix data from all programming languages together (high and low resources), and see how the performance of high/low resource languages change.
2. Unfortunately the model scale is also limited (up to 8B). So we cannot verify if this approach can scale well with model size.

### Questions
My main concern is that the training is not performed on data from all languages mixing together. Can you show such results? We would like to verify that there are no negative transfer across programming languages and the approach is indeed pushing the frontier.

### Soundness
3

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
The paper introduces Agnostics, a language-agnostic RL pipeline that reformulates coding tasks into standard I/O problems and trains code LLMs using verifiable rewards in a containerized sandbox. With small per-language configs, the same verifier supports many languages. Applied to Lua, Julia, R, OCaml, and Fortran, Agnostics boosts small open models to match/exceed larger baselines.

### Strengths
•	Consistent gains across five low-resource languages with small models.

•	New multi-language benchmark and dataset.

### Weaknesses
1.	While the contribution is clearly useful and highly practical, my main concern with the paper is that the contribution is more of an engineering improvement rather than a fundamental scientific contribution grounded in theoretical or algorithmic insights.

2.	Appendix A argues that rejection sampling is prohibitively expensive on hard tasks. A direct empirical comparison would clarify efficiency and quality trade-offs

3.	The paper would benefit from a more detailed ablation study: (i) reintroduce a KL term and show effects on stability/generalization; (ii) quantify partial credit reward pitfalls; (iii) temperature/group size sweeps beyond the brief D.1 notes. This would strengthen the methodological contribution beyond the system build.

### Questions
•	Can you elaborate more on what is the contribution of the paper? Can you clarify the fundamental, theoretical or algorithmic level contributions?

•	Did you perform any ablation studies related to KL term and partial-credit reward?

### Soundness
3

### Presentation
3

### Contribution
2
