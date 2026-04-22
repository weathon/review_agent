# Temporally Sparse Attack against Large Language Models in Time Series Forecasting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Language Models (LLMs) have recently demonstrated strong potential in zero-shot time series forecasting by leveraging their ability to capture complex temporal patterns through the next-token prediction mechanism. However, recent studies indicate that LLM-based forecasters are highly sensitive to small input perturbations. Existing attack methods, though, typically require modifying the entire time series, which is impractical in real-world scenarios. To address this limitation, we propose a Temporally Sparse Attack (TSA) against LLM-based time series forecasting. We formulate the attack as a Cardinality-Constrained Optimization Problem (CCOP) and introduce a Subspace Pursuit (SP)--based algorithm that restricts perturbations to a limited subset of time steps, enabling efficient and effective attacks. Extensive experiments on state-of-the-art LLM-based forecasters, including LLMTime (GPT-3.5, GPT-4, LLaMa, and Mistral), TimeGPT, and TimeLLM, across six diverse datasets, demonstrate that perturbing as little as 10% of the input can substantially degrade forecasting accuracy. These results highlight a critical vulnerability of current LLM-based forecasters to low-dimensional adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Temporally Sparse Attack (TSA) framework that targets large language model (LLM)–based time series forecasters. The central idea is to constrain adversarial perturbations to a limited subset of time steps, formulating the problem as a Cardinality-Constrained Optimization Problem (CCOP) and solving it via an adapted Subspace Pursuit (SP) algorithm with gradient-free optimization. Experiments on six datasets and multiple LLM-based forecasting systems (LLMTime, TimeGPT, TimeLLM, etc.) suggest that perturbing only 10% of the input sequence can substantially degrade forecasting accuracy, even when filter-based defenses are applied. To summarize, the paper is well-motivated in terms of addressing practical attack constraints (sparse, black-box, label-free).

### Strengths
[1] The paper focuses on a realistic attack setting—perturbing only a small portion of the time series, which mirrors constraints in real-world scenarios such as financial or sensor systems where continuous manipulation is infeasible. This sparsity assumption makes the problem more relevant than fully dense adversarial perturbations.
[2] The evaluation spans multiple forecasting domains (energy, weather, traffic, economics, and ETTh/ETTm benchmarks), suggesting that the attack framework is not dataset-specific.
[3] The paper clearly states the objective as a cardinality-constrained optimization problem (Equation 2) and emphasizes the trade-off between attack strength and perturbation sparsity. The formalization is neat and readable.
[4] The section discussing defense bypass (filter-based detection and smoothness-based defenses) provides valuable empirical insight, even though the results are preliminary. This helps highlight the vulnerability of time-aware LLMs.

### Weaknesses
[1] SP is borrowed from compressive sensing without rigorous reasoning for its advantage in this context. No ablation against simpler greedy or random-search baselines is reported.
[2] The method primarily extends prior black-box attacks by adding a sparsity constraint. While practically interesting, the conceptual contribution is limited compared with existing literature.
[3] The paper depends on commercial APIs (GPT-4, TimeGPT) without specifying prompt templates, token limits, or evaluation protocols. This makes results difficult to reproduce.
[4] Reported MAE/MSE results lack variance or confidence intervals. Given that TSA involves random query sampling, standard deviation across multiple runs is necessary to validate the reliability of performance gaps.
[5] The paper states that TSA is more efficient than greedy methods, but both have complexity. No runtime or query-efficiency comparison is provided.
[6] TSA mainly adds sparsity constraints on top of existing gradient-free attacks. A more explicit comparison to previous black-box methods would help clarify what is fundamentally new.

### Questions
[1] Does the finite-difference step require direct access to model outputs? How would this work with commercial APIs that do not expose logits or losses?
[2] Since the paper relies on commercial APIs (TimeGPT, GPT-4), can the authors specify the exact prompting template, temperature, and query limit used during evaluation?
[3] Are the same time steps attacked across models, or does TSA adapt per model architecture?
[4] What is the precise form of the loss used in the black-box query? If true Y_t is unavailable, how is the optimization target defined?

### Soundness
3

### Presentation
2

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
This paper proposes a Temporally Sparse Attack (TSA) targeting black-box, label-free LLM-based time series forecasting models. TSA formulates the attack as a cardinality-constrained optimization problem and solves it using an adapted Subspace Pursuit algorithm with gradient-free optimization. Experiments on six datasets and several LLM-based forecasters (e.g., TimeGPT, TimeLLM, LLMTime) show that perturbing a few input time steps can significantly degrade forecasting accuracy. While the idea is interesting and the formulation is clear, I have concerns about the paper’s generality, evaluation fairness, and the real-world relevance of the proposed threat model.

### Strengths
1. The attack formulation is mathematically sound, and the adapted SP-based algorithm is conceptually neat.
2. Experiments are broad in scope (multiple datasets and LLMs) and demonstrate consistent degradation in performance.
3. Writing is clear, and figures are helpful in conveying the main idea.

### Weaknesses
1. The proposed method seems designed specifically for black-box LLM-based forecasters. However, in practical applications, non-LLM models still dominate time series forecasting in both research and industry. TSA’s formulation (black-box, gradient-free, label-free) appears general enough to apply to any model that outputs predictions, not just LLMs.
2. While including TimesNet as a non-LLM baseline is appreciated, TimesNet is now 3 years old. The Time-Series-Library built from TimesNet already includes many more modern models. Testing TSA on a few of these newer non-LLM forecasters would help demonstrate its generality and practical relevance. This should be feasible, as these implementations are readily available in that package. Without such comparisons, it’s hard to tell whether TSA exposes vulnerabilities unique to LLMs or simply general weaknesses shared by all forecasters.
3. The comparison between TSA and DGA (full-series) attacks seems unbalanced. It is not surprising that a full-series attack achieves larger performance degradation. A fairer test would be to sparsify DGA, for example, keeping only the same number of perturbed steps (e.g., 9 time steps) either at random or aligned with TSA’s positions, and then compare the results. This would show whether TSA’s advantage comes from better perturbation selection or simply from fewer overall changes.
4. The paper measures attack success by how much it worsens MAE/MSE, but this doesn’t necessarily reflect a realistic attack objective. In practice, such random degradation is easily detectable and rarely beneficial to attackers. Realistic adversarial objectives in time series forecasting often involve targeted manipulation (e.g., introducing trigger patterns that make the model output attacker-chosen forecasts such as “predicting stock increase” when a certain trigger appears). I would like to see a discussion of how TSA might extend to or inspire such trigger-based or targeted attacks, which are more aligned with real-world threats.

### Questions
Please my suggestions in weaknesses.

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
This paper proposes a Temporally Sparse Attack against LLM-based time series forecasting models in a black-box manner without training data. TSA is modeled by $l_0$-norm constraint, and a black-box manner without training data is learned by gradients estimation with random Gaussian noise.

### Strengths
1. This paper is easy to follow. 
2. Clear experimental settings.
3. Well-structured experiments on six real-world open datasets.

### Weaknesses
1. Sparse Attack is well studied in CV and NLP. This work ultilizes these existing techniques, i.e. $l_0$-norm constraint, Subspace Pursuit (SP), black-box gradients estimation and Fast Gradient Sign Method (FGSM), on LLM-based time series forecasting models. The contributions and insights of this paper is limited.

2. There are no experiments under defense methods. There are many adversarial defense methods against Sparse Attack. The authors should evaluate their performance under adversarial defense methods to show how their methods works.

### Questions
see weaknesses.

### Soundness
3

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
This paper proposes a Temporally Sparse Attack (TSA) that effectively degrades the performance of LLM-based time series forecasters by perturbing only a small subset of input time steps. The attack is formulated as a cardinality-constrained optimization problem and solved via an adapted Subspace Pursuit algorithm under a realistic black-box, label-free setting.

### Strengths
- Novel Formulation and Solution: Casting the temporally sparse attack as a CCOP and adapting Subspace Pursuit (originally designed for white-box LASSO) to a black-box adversarial setting is an elegant and non-trivial methodological contribution.
- Strong Adversarial Impact with High Stealthiness: The paper shows that perturbing a quite small part of the time steps can severely degrade forecasting performance, and that TSA is more effective than Gaussian white noise and harder to defend against using standard filtering techniques.
- Interpretability and Visualization: The authors include detailed visualizations that compare input/output distributions, and prediction errors, providing strong intuitive support for their claims.

### Weaknesses
1. **Suboptimal perturbation update**: The paper adopts an FGSM-like one-step update, resulting in identical perturbation magnitudes across all attacked time steps. This uniformity may lead to suboptimal solutions. A multi-step approach such as PGD could provide more flexible and effective perturbation optimization.
2. **Limited consideration of temporal interaction**: The current approach perturbs one timestamp at a time, ignoring potential synergistic effects between multiple time steps. Jointly selecting and optimizing perturbations across multiple timestamps, rather than in a sequential manner, may lead to stronger and more coordinated attacks.
3. **Unclear evaluation of attack strength**: The effectiveness of TSA is not clearly demonstrated. In Table 2, TSA consistently outperforms a weak heuristic baseline (GWN) but still underperforms compared to DGA on certain datasets. While this comparison is admittedly unfair due to the dense nature of DGA versus the sparsity constraint in TSA, it still leaves reviewers uncertain about the absolute strength of TSA. A more appropriate evaluation would involve comparisons with sparse adversarial attack methods, such as [1,2].
4. **Lack of LLM-specific design**: The proposed method is tailored for general black-box models and does not incorporate any design elements unique to LLMs. As a reviewer, I find this positioning somewhat confusing: why emphasize LLMs specifically if the method is equally applicable to any black-box forecaster, given the absence of LLM-specific methodological considerations?

[1] Sparse and imperceivable adversarial attacks
[2] GreedyFool: Distortion-Aware Sparse Adversarial Attack

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
2
