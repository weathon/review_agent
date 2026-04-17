# Pretraining Scaling Laws for Generative Evaluations of Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Neural scaling laws have driven the field's ever-expanding exponential growth in parameters, data and compute. While scaling behaviors for pretraining losses and discriminative benchmarks are well established, generative benchmarks such as mathematical problem-solving or software engineering remain under-explored.
We propose and evaluate three different pretraining scaling laws for fitting pass-at-$k$ on generative evaluations and for predicting pass-at-$k$ of the most expensive model using cheaper models.
Our three scaling laws differ in the covariates used: (1) pretraining compute, (2) model parameters and pretraining tokens, (3) log likelihoods of gold reference solutions.
First, we demonstrate that generative evaluations introduce new hyperparameters (in our setting, $k$) that act as a control lever for scaling behavior, modulating both the scaling law parameters and the predictability of performance.
Second, we identify a stark difference in parameter stability: while the compute and parameters+tokens laws stabilize for only the last $\mathord{\sim}1.5\mathord{-}2.5$ orders of magnitude, the gold reference likelihood law is uniquely stable, converging across $\mathord{\sim}5$ orders. Third, in terms of predictive performance, we find all three scaling laws perform comparably, although the compute law predicts slightly worse for small $k$ and the gold reference law predicts slightly worse for large $k$. Finally, we establish a theoretical connection, proving that the compute scaling law emerges as the compute-optimal envelope of the parameters-and-tokens law. Our framework provides researchers and practitioners with insights and methodologies to forecast generative performance, accelerating progress toward models that can reason, solve, and create.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies scaling laws for generative evaluations, which is a relatively unexplored area compared to pretraining loss or discriminative benchmarks. The authors propose three types of scaling laws that relate model performance (measured by pass@k) to:
1. Pretraining compute
2. Model parameters and pretraining tokens
3. Log-likelihoods of gold reference solutions

They empirically evaluate these laws on GSM8K using Pythia models (14M-12B parameters, 300B tokens), showing that: All three scaling laws achieve comparable predictive performance.

### Strengths
1. The paper expands scaling law studies beyond pretraining or discriminative settings to generative evaluations, which are highly relevant for coding, reasoning, and mathematical tasks.
2. The authors clearly define three scaling formulations, use consistent benchmarks (GSM8K), and employ backtesting to rigorously evaluate predictive ability.
3. The results show stable behavior over large dynamic ranges, and the study identifies when extrapolation becomes unreliable

### Weaknesses
1. The writing quality of the paper could be improved. For example, Figures 3 and 5 omit certain points, and the legend partially obscures the plotted lines.
2. All experiments are conducted on GSM8K using Pythia checkpoints
3. The paper focuses only on pass@k with temperature sampling = 1.0. Other decoding strategies (top-p, nucleus sampling, self-consistency) might influence scaling behavior.
4. The results are derived solely from internal checkpoints of a single architecture. Evaluating additional model families such as LLaMA or Qwen could help assess the robustness of the proposed scaling laws. Moreover, since Pythia is a relatively dated model, it may not be well suited for evaluating mathematical reasoning capabilities.
5. I believe that pass@k can be viewed as a variant of accuracy. Could the authors clarify in detail how this work differs from Language Models Scale Reliably with Over-Training and on Downstream Tasks https://arxiv.org/abs/2403.08540.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper investigates pretraining scaling laws for generative evaluations of language models and corresponding metrics such as pass@k on verifiable tasks. The author introduces a new parameter to the scaling law and identify gold reference likelihood to be a better proxy metric for stabilizing scaling laws.

### Strengths
1. The paper is clearly written and contains the necessary details for reproducibility.
2. The direct comparison of different types of scaling laws answers the question of how to develop scaling laws for generative evaluations well. 
3. The analysis of including k as a parameter and its role in the scaling law is clear. 
4. The finding of the stability of the gold reference likelihood scaling law is a strong result and well founded, and explained.

### Weaknesses
1. The scaling law is developed on a single downstream task. While Pythia is an older generation model, where other tasks might be too difficult to obtain meaningful accuracy, why not average more tasks where Pythia could obtain a meaningful accuracy on, and expand beyond math?

2. What is the practical implication of the penalty factor derived in Section 6? What does it look like for the studied model checkpoints?

### Questions
Please see my comments above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper extends the study of neural scaling laws from pretraining losses and discriminative benchmarks to generative evaluations, tasks where models are assessed by their ability to produce correct outputs such as math solutions or code completions.
The authors propose and compare three scaling laws for predicting pass@k performance across pretrained checkpoints: (1) A Compute-based scaling law, relating log(pass@k) to pretraining FLOPs; (2) A Parameters + Tokens law, decomposing compute into model size N and dataset size D; (3) A Gold Reference Likelihood law, regressing pass@k against the log-likelihoods of reference (“gold”) solutions. 
Through backtesting on the GSM8K benchmark using Pythia checkpoints, they show that all three scaling laws achieve comparable predictive accuracy, but the gold-likelihood-based law is notably more stable, maintaining parameter convergence over up to five orders of magnitude in compute.
Finally, the authors establish a theoretical connection showing that the compute scaling law emerges as the compute-optimal envelope of the parameters+tokens law, explaining how optimal allocation of compute between parameters and data induces the observed power-law behavior.

### Strengths
1. The paper is the first (as far as I know) to systematically analyze generative scaling laws for pass@k-based evaluations, extending the classical scaling law framework beyond pretraining loss or discriminative accuracy.
2. Evaluating three distinct covariates (compute, parameters+tokens, and gold log-likelihoods) under a unified methodology is conceptually clean and methodologically rigorous.
3. The derivation showing that the compute law is the compute-optimal envelope of the parameters+tokens law offers a theoretical explanation of empirical scaling behaviors and clarifies the relationship between prior empirical findings.

### Weaknesses
1. The study is restricted to a single model family (Pythia) and one generative benchmark (GSM8K).
The conclusions, especially regarding the stability of the gold-likelihood law, should be validated on more diverse tasks (e.g., code generation, open-ended reasoning).
2. While figures show relative stability, a numerical summary (e.g., mean relative error across backtests) would help quantify the magnitude of differences between the three scaling laws.

### Questions
It would be interesting to test whether the fitted α(k) exponents are consistent across different generative benchmarks or model families.

### Soundness
3

### Presentation
3

### Contribution
3
