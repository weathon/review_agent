# Diagnosing and Mitigating Systemic Reward Bias in Self-Rewarding RL

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) efficiently scales the reasoning ability of large language models (LLMs) but remains bottlenecked by limited labeled samples for continued data scaling. Reinforcement learning with intrinsic rewards (RLIR), in which the policy model assigns reward signals to its own rollouts, enables sustainable scaling in unlabeled settings. Yet its performance and stability still lag behind RLVR. We trace this gap to a system bias: the model tends to deem its own high-confidence rollouts correct, leading to biased and unstable reward estimation. It accumulates and rises rapidly as training proceeds, with the deviation from the oracle drifting toward over-reward. This causes unstable training and locks the performance ceiling. To understand how system bias yields these effects, we characterize it by the magnitude of reward bias, the degree of policy–reward coupling, and the proportional imbalance between over-reward and under-reward via three metrics: $\rho_{\text{noise}}$, $\rho_{\text{selfbias}}$, and $\rho_{\text{symbias}}$. We find that $\rho_{\text{noise}}$ and $\rho_{\text{symbias}}$ affect convergence performance and speed, while $\rho_{\text{selfbias}}$ has an amplification effect: it amplifies both correct and incorrect updates and induces unstable reward estimation. To mitigate system bias of RLIR, we propose reinforcement learning with ensembled rewards (RLER). It aggregates diverse models with adaptive reward interpolation and rollout selection strategy to build a unified reward-estimation space, jointly improving accuracy ($\rho_{\text{noise}}$), unbiasedness ($\rho_{\text{selfbias}}$, $\rho_{\text{symbias}}$), and robustness ($\rho_{\text{selfbias}}$).  Extensive experiments show that RLER improves by +13.6\% over the best RLIR baseline, and is only 3.6\% below the RLVR setting. Moreover, RLER achieves stable scaling on unlabeled samples, making it highly applicable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Research has identified a systematic bias in RLIR: models tend to overestimate their own high-confidence outputs, leading to biased and unstable reward estimation, which in turn affects training convergence and performance limits. To mitigate this issue, the authors propose Reinforcement Learning with Ensemble Rewards (RLER), which improves the accuracy, unbiasedness, and robustness of reward estimation by aggregating multiple models and employing an adaptive reward interpolation strategy.

### Strengths
1. The analysis of systematic bias is quite comprehensive.
2.  Although ensemble methods are commonly used, the specific implementation and innovative design still contribute meaningfully to the field.

### Weaknesses
1. The experiments in this paper are almost entirely based on the Qwen series of models, and the evaluation datasets used (such as MATH500, AMC, AIME, etc.) are highly likely to be contaminated. The authors also acknowledge this issue in the Appendix. Therefore, the experimental results cannot be considered reliable unless validated on more models and datasets with favorable outcomes.
2. Figure 1 needs improvement. The font size in the figure is too small, making it difficult to discern the method the authors intend to illustrate.
3. The training dataset is overly simplistic and lacks diversity.

### Questions
1. As mentioned in the Weaknesses section, the authors urgently need to supplement the experiments with more models, training datasets, and evaluation datasets.
2. The computational cost of the ensemble method should be quantified and analyzed.

### Soundness
1

### Presentation
1

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
In this paper, the authors focus on the "systematic bias" problem in RLIR. The challenge of this problem is that models tend to assign a high reward for their own high-confidence (even if wrong) output. The authors systematically analyze the reasons for this problem and propose an ensemble reward strategy. The experiments show that the strategy outperforms baselines and achieves results approaching those of methods that rely on labeled data.

### Strengths
1. The motivation of this paper is clear and specific. The authors focus on RLIR's important system bias challenge. This challenge is important for LLM's application.
2. In this paper, the authors provide detailed experiments to validate the effectiveness of the method. They provide sufficient hyperparameters for the experiments. In general, the reproducibility of the method should not be a problem.

### Weaknesses
1. The method proposed in this paper is an ensemble method. However, in the experiments, the authors only compare their method with single-model methods. They fail to compare the method with other ensemble methods. This lack makes it hard for the reader to evaluate the effect of the mixed signals. This may raise the question of whether the effectiveness of the method is based on the rationality of the reward or the ensemble strategy itself.

2. The authors only test their method on limited LLM types and sizes. In this paper, the authors test the model only on Qwen, and the scales they choose are 1.5B and 7B. The single model choice and smaller scale may raise the question of whether the method and its strategy are only effective on specific models.

### Questions
1. The training phase of the method requires running k models in parallel. This brings k times the computation and memory overhead compared to a single model. The increase in computational cost may limit the application of this method on a large scale or with a larger ensemble size. Authors may want to discuss in detail the trade-off between performance gain and training cost, especially in resource-constrained scenarios.

2. The experiments of this paper mainly focus on mathematical reasoning tasks. The characteristic of these tasks is that the answer is clear and easy-to-validate. However, in broader reasoning tasks, such as code generation and creative writing, the reward signal may be hard to quantify. How the method can be generalized to broader reasoning tasks, or whether the method is focused on solving mathematical reasoning, is an open question.

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
This paper diagnoses and mitigates system bias in RLIR. The work traces the performance gap between RLIR and methods using RLVR to a system bias where the model incorrectly rewards its own high-confidence outputs. To characterize this, the paper introduces three metrics: reward noise rate, self-feedback bias rate, and symmetry bias rate. Based on this diagnosis, the paper proposes RLER, which aggregates diverse models to create a more accurate and stable reward signal. RLER combines ensemble self-rewarding, adaptive soft-reward interpolation, and a confidence-disagreement balanced rollout selection strategy. Experiments on mathematical reasoning benchmarks show that RLER significantly outperforms RLIR baselines and substantially closes the performance gap with RLVR.

### Strengths
- This work introduces three distinct metrics, which provide a clear and quantitative framework for analyzing system bias. 
- The decoupling experiment is a well-designed study that effectively isolates the impact of each metric on training dynamics. 
- The paper presents extensive experiments with strong results that convincingly support its claims.

### Weaknesses
Lack of precision in key definitions and insufficient details regarding experimental setups, particularly in Section 3. For example, the "attained reward as r_i" (line 150) is unclear whether it is a model prediction or a noisy label. "hard-reward" and "soft-reward" are mentioned without explanation in the main text. The motivation for choosing the three specific metrics, which are central to the paper's contribution, could also be explained in detail. "Findings 2" concludes that over-reward is more detrimental than under-reward, and a "Further analysis" is mentioned to support this. However, the text does not elaborate on this analysis, and the connection between the cited figures (Fig. 2(b) and Fig. 3(e)) and the claim about a "near-orthogonal gradient bias" is not explained The setup for experiments in Figure 3 is not described with sufficient detail.  the paper states that an ensemble of k=2 models is used but fails to describe how these models are chosen or initialized. A small ensemble size (k=2) is also not fully justified. p_selfbias^true(x) and p_selfbias^err(x), without defining what "true" and "err" mean in this context.

### Questions
See weakness and:

The main experiments use an ensemble of 2 models. What is the rationale for this specific choice? How does the performance, computational overhead, and model diversity of RLER scale on the main benchmarks as ensemble number is increased beyond 2?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes the system bias issues in RLIR (Reinforcement Learning with Intrinsic Rewards) through reward bias magnitude, policy-reward coupling strength, and imbalance magnitude. To address the systemic bias issues, the paper proposes RLER (Reinforcement Learning with Ensembled Rewards). Specifically, RLER replaces the single-model self-rewarding with an ensemble, aggregating diverse models to construct a unified stable reward space that guides the ensemble to improve collaboratively. The paper uses Qwen2.5-Math-7B as the backbone model and trains it with DAPO-MATH-17K dataset. Experiments demonstrate that RLER outperforms the RLIR baseline across multiple metrics and achieves results very close to RLVR (Reinforcement Learning with Verifiable Rewards).

### Strengths
1. The paper provides a novel approach to analyze the system bias of RLIR. The paper characterizes RLIR's noise, characterizing RLIR’s noise, coupling, and over/under-reward asymmetry with three metrics and validates their causal roles through experiments.

2. The experiment section of this paper is comprehensive. It includes not only the overall improvement rate but also the analytical indicators mentioned earlier and a full range of ablation experiments.

### Weaknesses
1. The reward design in this paper is relatively complex, increasing parameter tuning costs while potentially causing gradient instability. Additionally, the reward/scoring module exhibits high sensitivity to batch composition, sampling temperature, and normalization strategies, where even minor missteps can amplify training fluctuations.

2. This paper assumes that free-text answers can be mapped to discrete categories, upon which rewards are calculated. While this approach may be feasible for tasks like mathematics, it struggles to adapt to open-ended tasks. Furthermore, the experimental data presented in the paper are exclusively from mathematics datasets, failing to demonstrate advantages across other task domains.

3. RLER relies on the diversity of candidate results derived from model diversity. With too few models, the advantages of ensemble learning are not fully realized, while too many models introduce noise. The quantitative analysis in this paper is insufficient in this regard.

4. The paper only includes LLM-as-a-Judge and typical RLIR as the baseline model. It does not incorporate recently designed improvements specifically addressing RLIR limitations as experimental baselines.

### Questions
Model diversity brings ensemble benefits but may also amplify noise. How to optimally balance diversity versus noise within a fixed budget?

### Soundness
2

### Presentation
3

### Contribution
3
