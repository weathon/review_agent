# InfoSynth: Information-Guided Benchmark Synthesis for LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation. However, efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, a process that is both expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97\% of the time, and the synthesized benchmarks consistently exhibit higher novelty and diversity compared to their seed datasets. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, novel and diverse benchmarks for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a novel framework for automatically generating and evaluting reasoning benchmarks guided by onformation-theoretic principles. Their experiments generate accurate test caes and solutions to new problems 97% of the time.

### Strengths
1. the research topic is interesting. While employing information-theoratic to evaluate the diversity of LLM's coding and resoning synthetsis is an interesting topic.
2. Overall, the paper is completed with a light presentation for each section. However, many typos need to be fixed.

### Weaknesses
1. The presentation is not good enough. For example, Line 092 "Schulman et al. uses" -> use. same for Line 093
2. The method lacks of sufficient novelty. Using the KL-divergence to estimate the gap between the current and the synthetic dataset. Hence, the metric of novelty and diversity are not impressive. Besides, your experiments don't show related section to discuss how these metrics are effective or not.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces InfoSynth, an information-theoretic framework for synthesizing benchmarks in code generation tasks. The key novelty lies in repurposing two classical information-theoretic metrics: (1) KL-Divergence for measuring benchmark novelty and (2) differential entropy for assessing internal diversity. These metrics provide an efficient way to quantify benchmark quality without costly model evaluations. Building on these metrics, the authors design an end-to-end pipeline that employs genetic algorithms and iterative code feedback to generate new Python coding problems, along with reference solutions and test cases, from seed datasets. The pipeline achieves 97% correctness in generating problems with accurate test cases and solutions.

### Strengths
1. The paper repurposes KL-Divergence and differential entropy to measure benchmark novelty and diversity. This is a creative contribution to the benchmark synthesis domain.

2. Section 3 effectively introduces the desirable properties (novelty and diversity) and provides both theoretical foundation and empirical validation with 95% confidence intervals.

3. The combination of genetic algorithms, mutation/crossover operations, and iterative code feedback is well-designed.

### Weaknesses
1. Limited Metric Validation Scope. The KL-Divergence and entropy validations primarily compare Leetcode subsets or extract subsets from larger benchmarks. While these experiments demonstrate the metrics work in clear-cut cases, *validation on independent benchmarks with subtle differences is missing. For example, comparing HumanEval vs MBPP, or benchmarks differing in problem-solving approaches (greedy vs dynamic programming) would better demonstrate discriminative power. 

2. Estimator Stability and Negative KL Values. Although KL-Divergence is theoretically nonnegative, the estimator produces negative values in Figure 1 (RHS). The paper attributes this to subset-superset comparisons but does not quantify how frequently this occurs or its magnitude. This raises concerns about estimator stability and practical implications for the optimization process that maximizes KL-Divergence. 

3. Unclear Pipeline Innovation Relative to Prior Work
The paper does not clearly articulate what specific innovations the pipeline introduces beyond combining existing techniques. For instance, how does integrating information-theoretic metrics into genetic algorithm selection differ from or improve upon prior evolutionary benchmark synthesis approaches?

### Questions
1. could you provide validation results comparing independent benchmarks (e.g., HumanEval vs MBPP) rather than only subsets from the same benchmark? This would strengthen evidence that your metrics capture novelty/diversity beyond obvious domain differences.

2. could you quantify how frequently negative KL values occur and their magnitudes? Have you explored alternative estimators or bootstrapping to assess stability?

### Soundness
2

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
3

### Summary
This paper proposes a method to calculate the novelty of a dataset using the KL divergence of embeddings, a method to calculate the diversity of a dataset using the entropy of embeddings, and a method for synthesizing coding problems based on genetic algorithms.

### Strengths
1. The experiment covers a wide range of LLMs, including open-weight models like Qwen, and SOTA models like GPT and Claude.
2. The implementation details are presented in detail. The prompt is also provided, so I think the reproducibility is good.
3. This paper is generally well-written and easy to follow.

### Weaknesses
1. **Whether KL can reflect the novelty of a dataset is questionable.** First, this paper calculates the KL Divergence of the problem statements in the dataset based on the embeddings output by a very small model. I believe this can at best only measure the differences between problem statements at the literal level, rather than truly capturing the novelty of these problems in their essence. For instance, if I add a large section of irrelevant background stories to each problem in a dataset, the KL Divergence can be significantly increased, yet the problems themselves remain unchanged in their essence. I believe using KL divergence to measure a certain type of "difference" and entropy to measure a certain type of "diversity" is valid. However, whether this approach can be effective ultimately depends on whether the projected distribution itself can establish a connection with the concepts such as "novelty" and "diversity" as perceived by the authors. This paper, however, have not provided sufficient validations in this regard.
2. **The correctness of synthetic problems cannot be guaranteed.** Both the test cases and solutions for these generated questions are produced by LLMs, and this paper only seems to ensure that the generated solutions and test cases are self-consistent, rather than ensuring both are correct. It is possible that both the test cases and solutions are erroneous in the same way.

### Questions
Please refer to "Weaknesses".

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework for automatically generating Python coding benchmarks for LLMs. The works makes two main contributions:
1. an information theoretic framework using KL-divergence and entropy to quantify benchmark novelty and diversity without expensive model evaluations
2. and end-to-end pipeline using genetic algorithms with iterative code feedback to synthesize verifiable problems.
The experiments show the correctness of newly generated problems achieves 97% and consistently exhibit higher novelty and diversity compared to seed datasets.

### Strengths
1. KL-divergence for novelty and entropy for diversity measure make sense, and are computationally efficient compared to model based evaluations.
2. This work includes several important ablation studies that validate key design choices:
- multiple mutation difficulties provide difficulty control
- iterative feedback substantially improves correctness
- postprocessing improves clarity
- k-farthest neighbor filtering increases diversity

### Weaknesses
1. The paper sets the goal of creating benchmarks that are contamination-free, while lacking the experiments or discussion of contamination evaluation on the proposed data generation method.
2. No comparison with other synthetic benchmark generation methods. Section 2 actually listed multiple comparable methods, but none of them were benchmarked in empirical study. Ideally the authors should show, with different dataset generation methods, compare novelty / diversity metrics, correctness rates, model performance distributions, computational costs.
3. Note that Table 1 shows this generation pipeline has high filtering rate, for example, for MBPP-Guided, more than half of the generated samples were discarded. The authors should provide careful analysis of the breakdown of filtering reasons, characterization of filtered problems, potential bias from filtering.

### Questions
1. How sensitive are the novelty/diversity metrics to the choice of embedding model, have you tested with other embeddings?
2. Can you provide evidence that generated benchmarks actually help with contamination detection or out-of-distribution evaluation in practice?
3. Can you provide analysis of the filtered problems, what types of problems got filtered and would this introduce systematic biases?
4. Can you provide sensitivity analysis on hyperparameter selection for k and UMAP parameters?

### Soundness
3

### Presentation
3

### Contribution
3
