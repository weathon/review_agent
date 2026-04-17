# BigO(Bench) - Can LLMs Generate Code with Controlled Time and Space Complexity?

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes a set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces BIGO(BENCH), a benchmark designed to evaluate Large Language Models (LLMs) on their ability to understand and generate Python code with specific time and space complexity constraints. The contributions include: (1) A dataset derived from CODE CONTESTS, annotated with complexity labels using a custom dynamic inference framework (3,105 problems, ~1.2M solutions). (2) The complexity inference framework itself, which uses profiling, fuzzing, and curve fitting to estimate complexities. (3) An evaluation of 14 state-of-the-art LLMs on three tasks: complexity prediction, complexity-constrained code generation, and ranking generated solutions against human ones based on complexity coefficients. The results show that current LLMs, even reasoning-focused ones, struggle significantly with complexity generation tasks, despite high performance on standard code synthesis.

### Strengths
1. The paper addresses a critical and often overlooked aspect of code generation by LLMs – their understanding and control of computational complexity, which is crucial for real-world software development.

2. The authors have annotated a large dataset (3,105 problems, ~1.2M solutions) from competitive programming platforms with inferred complexity labels, providing a substantial resource if the labels were reliable.

3. The evaluation covers multiple facets of complexity handling (prediction, generation, ranking) and includes a wide range of recent LLMs.

4. The release of the complexity inference framework code allows for reproducibility and potential community improvement.

### Weaknesses
1. The reported 82-84% accuracy against human labels is insufficient for generating trustworthy ground truth. Reliance on empirical profiling makes the framework susceptible to noise, hardware variations, and specific runtime environments, potentially failing to capture true asymptotic complexity. The paper itself notes it can fall upon edge cases. This lack of robustness undermines the entire benchmark's validity.

2. The extensive, framework-dependent filtering applied during dataset creation may introduce significant bias, potentially selecting only problems/solutions where the framework performs well. The resulting dataset's representativeness is questionable. The complexity distribution is highly imbalanced, potentially skewing results.

3. The All@1 scores for complexity generation are exceptionally low (often <5% for time, <3% for space) across all models, including powerful reasoning models. While interpreted as LLM failure, these near-zero scores could equally indicate issues with the benchmark itself: unreliable ground truth, an overly harsh metric (All@k), or ill-posed tasks. It's hard to draw meaningful conclusions or measure progress when performance is near the floor.

4. Fine-tuning Llama 3.1 70B specifically on the benchmark tasks yielded negligible or even negative impacts on complexity generation performance (Table 4). This suggests the benchmark data and tasks, as currently formulated, may not provide a useful signal for improving LLM capabilities in this area, questioning their utility.

5. The framework relies on dynamic analysis (profiling runs). It's unclear how well this empirical approach approximates theoretical worst-case complexity, especially given Python's dynamic nature and CPython optimizations. The benchmark might be evaluating the ability to generate code that performs well empirically under the framework's specific testing conditions, rather than code with a provably correct asymptotic complexity.

6. Decisions like using Big-O for worst-case, the specific parameters for fuzzing/curve fitting, and the `simplicity bias`  need stronger justification and sensitivity analysis.

### Questions
1. Could the authors provide a more detailed error analysis for the complexity framework? What types of code or complexity classes does it struggle with most? How was the 125-sample human validation set selected, and what was the inter-annotator agreement if multiple humans were involved? Given the 16-18% error rate, how confident can we be in the benchmark labels, especially for evaluating subtle differences between models or measuring fine-tuning progress?

2. How sensitive are the framework's complexity estimations to the specific hardware, Python version, background processes, and profiling tool versions? Were experiments run to quantify this variability? Could this noise explain some of the poor LLM generation results?

3. Can the framework distinguish between empirical performance fitting a curve (e.g., $O(n)$ up to $n=10000$) and true theoretical complexity (e.g., an underlying $O(n \log n)$ algorithm that looks linear in the tested range)? How does it handle amortized analysis?

4. Could the extensive filtering applied during test set creation (e.g., removing outliers, unstable predictions, unlikely ASTs) have biased the benchmark towards problems/solutions the framework can easily analyze, potentially masking harder cases?

5. Given the near-zero All@k generation scores, how can the authors be sure this reflects inherent LLM limitations versus issues with the benchmark's noisy labels, task formulation, or the stringency of the All@k metric?

6. Why do the authors believe fine-tuning failed to improve (and sometimes worsened) complexity generation performance? Does this suggest the benchmark data lacks a useful signal, or that standard fine-tuning is simply inadequate for this type of reasoning?

7. Why was Big-O chosen for the main benchmark despite the potential ambiguity noted, especially when the Big-Theta prompt seemed to improve generation for the reasoning model?

### Soundness
3

### Presentation
3

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
BIGO(BENCH) introduces a benchmark for testing whether LLMs can understand and produce code that meets stated time and space complexity. It covers three tasks: predicting the complexity of a given solution, generating a solution that satisfies a target complexity, and ranking solutions within the same class by constant factors. The labels come from profiling code across input scales rather than purely theoretical analysis.

Experiments show that today’s strong models often achieve functional correctness but still miss explicit complexity targets. Performance on complexity-constrained generation is low, and complexity prediction is only modest. The paper argues that current training does not teach models to control algorithmic complexity and calls for methods that align generation with resource goals.

### Strengths
1. The benchmark targets controlled time and space complexity rather than only functional correctness, and it instantiates this through three complementary tasks. The core idea feels novel, and the contribution is substantial and well scoped.

2. The authors annotate 3,105 problems and 1,190,250 Python solutions from CODE CONTESTS, provide per-input runtime and memory traces, and release problem-specific dataclasses so solutions can be profiled end to end. The test sets explicitly retain problems with multiple complexity classes, improving diagnostic value.

3. Single-sample performance on complexity-constrained generation remains low across models, and while larger sampling budgets help, the task stays difficult. Fine-tuning brings only targeted, partial gains, with All@1 often in the single digits, suggesting a genuinely hard benchmark that is likely to remain relevant over time.

4. The paper introduces a practical framework to infer time and space complexity from empirical profiling via fuzzing and curve fitting, and indicates an open release. It reports 84% agreement for time and 82% for space with human-theoretical labels, along with high self-consistency across runs.

### Weaknesses
1. Robustness of the inferred complexity labels is under-specified. While the authors report ≈90% self-consistency across multiple runs, it remains unclear how stable the labels are under runtime noise or varying sampling resolutions. For example, would the predicted class remain unchanged if input sizes were halved, or if 5–10% random noise were added to the timing measurements? A quantitative sensitivity analysis along these lines would strengthen confidence in the framework’s robustness.
2. The distribution is heavily concentrated in O(1) and O(n), which can inflate averages and conceal weaknesses on rarer classes like O(n log n) or O(n²). Class-balanced/re-weighted metrics, stratified subsamples, and per-class All@k would test whether gains persist beyond majority-class guessing and keep both tasks informative on the tails.

### Questions
1. You mention 37 algorithmic notions of  problems in your dataset. Could you provide the full list and their distributions?

2. In Table 2 several evaluated models are already outdated. Given newer releases such as o4-mini and GPT-5, how do you justify the current model selection? If immediate re-runs are not feasible, is there a plan for an updated leaderboard, submission interface, or fixed-budget evaluation protocol that would allow the community to add recent models while preserving comparability?

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
The paper attempts to test how well code LMs understand the concept of code complexity by gauging how well they adhere to complexity-constrained code generation and code complexity prediction.

### Strengths
1. The paper uncovers an interesting setting in which Code LMs show a non-trivial lack of understanding.
2. The paper moves to improve the tooling and evaluation suite for open-ended code complexity evaluation by allowing arbitrary executable code to be profiled for runtime and memory usage over a large variety of inputs and then inferring its complexity via curve fitting.

### Weaknesses
1. The paper introduces a very specific task without really doing a good job of motivating its need. The need for generating the most efficient code is clear to all, but why does a model need to controllably generate code of a certain complexity is not very clear at all.
2. I would be sympathetic to a code understanding angle in terms of why controllable complexity is important, but I also feel there are better ways to disentangle that, for e.g. by asking the model to rewrite an existing code solution of a problem into another one that follows a specific complexity class.

Overall, my rating of 4 is harsh, and I would prefer to give a 5 if the option were present. I would also be open to be persuaded to move the score to a 6 if the work were better motivated.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3
