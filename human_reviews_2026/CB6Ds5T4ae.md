# RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Reasoning language models have demonstrated remarkable performance on many challenging tasks in math, science, and coding. Choosing the right reasoning model for practical deployment involves a performance-cost trade-off at two key levels: model size and reasoning budget, where larger models and higher reasoning budgets lead to better performance but incur greater cost and latency. In this work, we tackle this tradeoff from the angle of model configuration routing for different queries, and present RADAR (Reasoning–Ability and Difficulty-Aware Routing), a lightweight, interpretable, and scalable routing framework. Inspired by psychometrics, RADAR learns an item response model from model responses with different budgets to different queries, with interpretable parameters including query difficulties and model-budget abilities. RADAR then routes queries with higher difficulty to model-budget pairs with higher ability, and vice versa. We conduct extensive experiments on 8 widely used challenging reasoning benchmarks, demonstrating the superior performance of RADAR compared to state-of-the-art model routing methods. RADAR also exhibits query generalization capabilities, achieving strong performance on out-of-distribution queries on all benchmarks. RADAR is also scalable and can efficiently integrate additional models by dynamically selecting a small set of evaluation queries to estimate their abilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work considers the problem of query-based routing reasoning language models (RLM) with variable budget. 
The authors here formulate such problems into multi-objective optimization (MOO) paradigm with certain scalarization (linear/Chebyshev) for both performance and cost. 
The MOO's variables include 1) model-budget configuration and 2) query difficulty.
Model-budget configurations are (model, token) pairs with reasoning token count discretized and the authors propose to leverage IRT (item response theory) to learn the difficulty/discrimination projector with a frozen embedding model and a model-configuration _ability_ parameter to maintain both interpretability (low parameter) and low latency during inference.

This paper is well written and the idea is well presented with good motivation and strong results.

### Strengths
1. Well-motivated: authors propose a method to achieve pareto frontier in terms of performance vs. cost in the case of multiple RLMs with varying size and reasoning budgets. This is a very realistic scenario and could be rather impactful. In addition, the proposed method is easy to adapt to any sort of models as they are treated as black boxes and readily extensible to newer and hopefully stronger or faster models.
2. Good performance: the authors compare RADAR against IRT Router and RouterBench and a couple of heuristics and show that RADAR achieves better hypervolume metric.
3. Fast: for each inference (unseen) query, RADAR performs an embedding step through an embedding model and a proposed linear projector to get the difficulty/discrimination scalars and feed these 2 scalars into the IRT model. This is rather fast in comparison to the later pipeline considering the number of parameters and reasoning budget of the following models.
4. Interpretability: IRT model only has two parameters (difficulty & discrimination in terms of input query) which affords natural interpretability.

### Weaknesses
The IRT model used in the paper assumes monotonicity, and the projectors are linear with the input embedding model frozen. I know it is probable that using a more complex model could result in a mere marginal improvement in terms of hypervolume, but would like to see some results in this regard. So is the case with the IRT model

### Questions
Same as weaknesses, would love to see some experiments on some more capable difficulty/discrimination projectors or a bit more complex model than IRT. While it is very reasonable and probable to end up with the conclusion that these more complicated model could obfuscate the interpretability and spike up latency while yielding only marginal performance, it will further strengthen the argument for RADAR.

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
3

### Summary
This paper introduces RADAR (Reasoning–Ability and Difficulty-Aware Routing), a lightweight and interpretable routing framework that optimizes the trade-off between performance and cost when deploying reasoning language models (RLMs) with varying model sizes and reasoning budgets.
RADAR draws inspiration from Item Response Theory (IRT) in psychometrics to estimate query difficulty and model-budget ability. Using these interpretable parameters, it routes easier queries to cheaper, smaller models and harder ones to more capable configurations. The authors formalize routing as a multi-objective optimization (MOO) problem balancing performance and cost via linear and Chebyshev scalarization.
Experiments across 8 reasoning benchmarks (AIME, MATH-500, GPQA, LSAT, MMLU, MMLU-Redux, MMLU-Pro, FRAMES) demonstrate that RADAR achieves superior performance-cost Pareto efficiency, generalizes well to out-of-distribution queries, and scales effectively when adding new models. It operates in real-time with ~7 ms routing latency.

### Strengths
First to frame reasoning model routing as a Multi-Objective Optimization (MOO) problem. Innovative use of Item Response Theory for modeling reasoning ability and query difficulty, an elegant and interpretable alternative to opaque routing regressors.

The methodology is rigorous, combining psychometric modeling with MOO and cost modeling in a consistent mathematical framework.
Experiments are extensive, covering 8 benchmarks and both in-distribution (ID) and out-of-distribution (OOD) settings, with clear comparisons to baselines such as RouterBench and IRT-Router.

The paper is very well written and structured, with clear motivation.

The work offers a generalizable and practical approach for efficient LLM deployment under cost constraints.

### Weaknesses
1. While RADAR clearly outperforms IRT-Router and RouterBench, the paper does not compare against mixture-of-experts or test-time adaptation methods that could serve as strong baselines for dynamic routing under cost constraints.
2. More discussion on relation to cascading methods (e.g., FrugalGPT) would strengthen the positioning.
3. Although latency is reported (~7 ms), no end-to-end system-level throughput analysis is provided. 
4. Some OOD performance dips (e.g., AIME benchmark) suggest the need for improved handling of unseen high-difficulty queries.
5. The model for estimating difficulty is too simple and may not capture the real question difficulty.

### Questions
Check weakness above.
1. A comparison/discussion with previous works on selecting the best answer from different LLMs would have been useful for latency. 
i. Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems.
ii. Scalable best-of-n selection for large language models via self-certainty. 
2. Which embedding model was used to represent queries?
3. Could RADAR dynamically adjust the user weight w₁ during a session (based on feedback or budget exhaustion) to continuously optimize global cost-performance?

### Soundness
3

### Presentation
3

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
The authors proposed RADAR (Reasoning–Ability and Difficulty-Aware Routing), a lightweight, interpretable, and scalable routing framework. They firstly show with statistics that reasoning models today struggle to balance between performances and budgets. The authors then proposed to model this problem as a multi-objective optimization problem. With extensive evaluations on multiple benchmarks, the authors showed that their methods are superior to other methods, and the advantage could generalize to new RLM settings.

### Strengths
1. The paper is well written and nicely presented.
2. The method the authors proposed is simple, yet more reasonable to generalize.

### Weaknesses
1. The authors did not well explain their experimental results. For example, the gain looks marginal on some benchmarks while significant on others. The authors could help readers better understand why the biases exist.

### Questions
1.As I said above, could you give an explanation of your results?
2. The cost function looks simple. For example, how can RADAR incorporate real-time latency, KV-cache reuse, or batching discounts into its cost function?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a method for the reasoning language model (RLM) routing problem, which aims to select the best RLM configuration for an input query, given a pool of RLMs,  the reasoning budget, and a user performance-cost trade-off profile. It leverages an item response theory model to estimate the query difficulty and RLM ability at different reasoning budget, by parameterizing the query difficulty and training on collected evaluation responses. Under the user performance-cost trade-off profile, it uses multi-objective optimization to find the Pareto optimal of model configurations, which is also empirically shown to be effective.

### Strengths
- The problem setting of user-desired performance-cost trade-off profile is interesting and practical.
- The proposed approach of modeling query difficulty and solving the multi-objetive optimization problem is reasonable.
- RADAR can generalize to OOD queries.
- Comprehensive empirical evaluation on different benchmarks.

### Weaknesses
- Limited technical novelty against the baseline IRT-Router with conceptually similar approach of leveraging IRT.
- It is not convincing that using a linear transformation on the embedding of questions could successfully model the underlying difficulty of the question. Using a more expressive parameterization would  intuitively give a better results given the high dimensionality of the embedding. However, no ablation has been shown to justify the linear transformation is sufficient.
- The empirical gains are limited based on the results from Table 1, 2. The evaluation method CPT with only one threshold (i.e., CPT(90%)) is not convincing. More evaluation threshold should be shown to justify its effectiveness under different desirability.
- Typos. E.g., there is a missing “running” before “configuration” in line 196.
- Sec. 3.5 depends on a small set of evaluation queries, which can be adaptively updated. However, whether this set (queries with high Fisher information) approximates the global data distribution Q is unclear. Ablations on how this approach differs from uniform sampling or simply estimation from historical observations would be helpful.

### Questions
See weaknesses  above.

### Soundness
2

### Presentation
2

### Contribution
2
