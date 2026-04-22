# When LLMs get significantly worse: A statistical approach to detect model degradations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Minimizing the inference cost and latency of foundation models has become a crucial area of research. Optimization approaches include theoretically lossless methods and
others without accuracy guarantees like quantization. In all of these cases it is crucial to ensure that the model quality has not degraded. However, even at temperature
zero, model generations are not necessarily robust even to theoretically lossless model optimizations due to numerical errors. We thus require statistical tools to 
decide whether a finite-sample accuracy deviation is an evidence of a model's degradation or whether it can be attributed to (harmless) noise in the evaluation. We 
propose a statistically sound hypothesis testing framework based on McNemar's test allowing to efficiently detect model degradations, while guaranteeing a controlled rate
of false positives. The crucial insight is that we have to confront the model scores on each sample, rather than aggregated on the task level. Furthermore, we propose 
three approaches to aggregate accuracy estimates across multiple benchmarks into a single decision. We provide an implementation on top of the largely adopted open source
LM Evaluation Harness and provide a case study illustrating that the method correctly flags degraded models, while not flagging model optimizations that are provably 
lossless. We find that with our tests even empirical accuracy degradations of 0.3% can be confidently attributed to actual degradations rather than noise.

Code: https://github.com/amazon-science/LLM-Accuracy-Stats

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper investigates an important problem of testing whether Large Language Models (LLMs) degrade, e.g., after quantization or different serving environments. It proposes to leverage the principled McNemar's test to detect statistically significant performance degradation. On evaluation of open-source datasets, the proposed method is shown to be highly sensitive in detecting model degradation.

### Strengths
- Targets an important and practical problem of testing LLM degradation.
- Theoretically grounded approach using McNemar's test for degradation detection.
- Interesting empirical observations about LLM degradation with the proposed method.

### Weaknesses
- Evaluations can be expanded to more models other than LLaMA.
- Practical implications of the findings regarding regression testing could be further elaborated.

### Questions
This is an interesting paper targeting an important problem of testing LLM degradation. The theoretical analysis with McNemar's test is highly encouraging, as it provides a principled way to detect statistically significant degradation. This is a new and useful contribution to the field. However, I am not an expert in this field, especially regarding the theoretical aspects. I would be interested in the following questions:

- The evaluations are primarily on LLaMA models. How well does the proposed method generalize to other open-source LLMs?
- Will this method be applicable to small models as well, or is it specifically designed for large models?
- The "faster regression testing" part seems interesting. Could you elaborate more on the actionable advice on how to effectively use this method for degradation testing in practice?
- If we have access to model internals, can we use this method to pinpoint the specific components causing degradation?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a statistical framework for detecting significant performance degradations in Large Language Models (LLMs) after optimization techniques such as quantization. The paper adapts a one-sided exact McNemar test and offers three ways to aggregate over multiple benchmarks (pooled, max-drop with Monte-Carlo, Fisher) to statistically determine whether observed accuracy drops are due to true degradation or random noise. The result shows that even small drops (≈0.3%) can be detected as statistically significant and evaluates this on Llama-3.x variants run via lm-eval and vLLM.

### Strengths
The paper tackles an important gap in LLM evaluation, quantitatively distinguishing real degradations from noise, which is critical for optimization research. The framework is theoretically justified using McNemar’s test, with mathematical backing and variance derivations. The integration into the popular LM Evaluation Harness makes it useful for both academic and industrial research. The experiments cover different Llama models and optimization types (INT4, FP8, KV-FP8). The paper also gives a useful rule-of-thumb for compressing evaluation sets by removing examples that never flip in pilot runs, reducing evaluation cost while preserving signal.

### Weaknesses
•	The technical tool (McNemar 1947) and Fisher’s method are classical. The paper’s contribution is primarily in adapting these to LLM regression testing, making them one-sided, and packaging aggregation strategies. This is still valuable, but conceptual novelty is limited.

•	Main empirical claims revolve around Llama-3.x variants and specific serving stacks (vLLM; KV-FP8 issues noted). It is unclear how robust the conclusions are across other model families, in-house evals, or non-binary metrics. Without replication on other models/toolchains, significance to the broader community is limited.

•	While the paper sketches a generalization for non-binary scores, the actual implementation and experiments remain binary. Given how common non-binary metrics are for LLMs, a small worked example (pairwise wins/ties/losses with thresholds) would increase practical utility.

### Questions
1.	The paper needs to replicate the main findings on at least one other family (e.g., Qwen, Mixtral, DeepSeek) and another serving stack, demonstrating that 0.3–1.0% effects are robust.
2.	Beyond adopting the exact one-sided McNemar/binomial test, what is the new statistical contribution? Could the paper include a non-inferiority (TOST) module or sequential design that adapts sample size for power? How robust is the one-sided McNemar test when model scores are not perfectly binary (e.g., tasks with multiple correct answers or graded scoring)?
3.	The paper may run a prospective study showing how flip-focused trimming affects type-I/II error and bias across content strata, and propose guardrails. This directly addresses the validity of Recommendation 1.
4.	For non-binary metrics, please add a summarization or QA experiment (e.g., ROUGE or F1) and explicitly show the reduction to pairwise wins/ties/losses (or thresholds). The paper may also report one-sided p-values, effect sizes, and calibration to validate the procedure and demonstrate its generalizability and broader applicability.
5.	Could the paper specify and discuss the exact decision rule for flagging degradation? Does the framework flag when any one-sided aggregated test rejects at α (e.g., 0.05), when (\hat{\delta}>2%) or (\hat{p}*{\updownarrow}\ge 5%) (per Dutta et al., 2024), or only when both statistical and practical criteria are met? In the 70B KV-FP8 and 70B w8a8 cases where (\hat{p}*{\updownarrow}<5%), the framework still detects a performance degradation. 
6.	The manuscript contains several language issues that impede readability (e.g., apostrophe misuse “its/it’s,” misspellings such as “assess”/“further,” incorrect conjunction “correct or false,” article usage “a similar situation,” verb form “build,” spacing/typography around symbols like “X, Y,” a duplicated footnote/URL, and inconsistent dialect choices like “while/whilst”). Could the paper (a) perform a thorough language edit to correct these issues, (b) standardize on a single dialect, and (c) resubmit with tracked edits or a brief editorial checklist summarizing the changes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a statistical framework to decide whether an optimized LLM has truly degraded relative too a baseline, rather than exhibiting harmless eval noise. The core is an exact one-sided McNemar test over per-item success/failure pairs fro the baseline vs optimized. This controls type-1 error and increase power by focusing only on disagreements (flips) between models. They derive theoretical analysis of test power, propose three aggregation methods for multi-task eval (pooled test, max drop test, Fisher's method), and provide implementation for the LM Evaluation Harness. Experiments on Llama-3.1 8B and Llama-3.3 70B with various quantization schemes demonstrate method can detect accuracy degradations as small as 0.3% while correctly not flagging theoretically lossless optimizations.

### Strengths
- well motivated as it addresses a real and understudied problem of rigorous statistical testing for LLM optimization 
- first to apply McNemar's test to LLM degradation detection
- I like the practical contribution for LM eval harness which enables immediate adoption
- very interesting that even a 0.3% degradation can be detected with statistical confidence

### Weaknesses
- limited novelty: this is more about an application and implementation of statistical tests than methodological innovation
- I think some of the assumptions may not hold for many of the benchmarks today. they make an i.i.d assumption but real benchmarks (like MMLU) often stratify by difficulty, topic, or other factors which may violate independence. Additionally, the binary metric assumption is limiting as plenty of important metrics are continuous. 
- I believe there is a limited scope of evaluation. the authors only evaluate on llama models (3.1, 3.3). no other model families are tested. they also only test quantization optimizations, but pruning, distillation, etc is missing.

### Questions
-  Given the three aggregation methods with different trade-offs, how should a practitioner choose?
- other than McNemar's test to answer if optimized model is statistically significantly worse than baseline, could you also use bootstrap confidence intervals for accuracy differences? or maybe permutation tests or bayesian model comparison (Bayes factors)? How do these compare in terms of power?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors argue that simply comparing aggregated accuracy scores of benchmarks are statistically flawed because it ignores the fact that two model evaluations are performed on the same set. Authors propose the testing framework based on the exact one-sided McNemar test. And further show that the testing framework can confidently flag model degradation as small as 0.3% to be statistically significant.

### Strengths
1. Authors challenge a fundamental approach in LLM evaluation benchmark where accuracies may not necessarily show the degradation of LLM capability. Currently the general practice is to consider difference within a certain threshold as statistically insignificant. Authors provide a more principled way to identify the flaws of the convention.

### Weaknesses
1. As a work studying variance/randomness, in the main experiment, it might be important for authors to report results across three runs.
2. Similarly, authors may want to reproduce an empirically lossless variant to rule out the hardware level randomness. (for example, use hf transformers backend and throw in the deterministic flag) Since we don’t necessarily know how the hardware-level randomness interact with the model-level randomness.

### Questions
1. Different inference engine backend (hf transformers, vllm, sglang) sometimes show different results on benchmarks. It would be helpful to test on different engines and tell if different engines may yield different results.
2. The method may be useful for measuring forgetting of language models after finetuning/continual learning. I encourage authors to explore other scenarios where degradation is an important topic (pruning, finetuning, etc.)

### Soundness
3

### Presentation
4

### Contribution
3
