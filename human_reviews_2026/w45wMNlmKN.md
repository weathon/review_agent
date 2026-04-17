# Forecasting-Conditioned Reinforcement Learning: Embedding Forecastability as an Inductive Bias

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
We introduce Forecasting-Conditioned Reinforcement Learning (FoRL), an extension to model-free Reinforcement Learning (RL) agents that augments the policy with multi-step self-forecasts. FoRL is trained either via Reward Conditioning (RC), which rewards forecast--action consistency, or Loss Conditioning (LC), which adds an auxiliary forecasting loss. Across discrete and continuous action space benchmarks and forecasting horizons $\(L \in \{2,5,10\}\)$, FoRL consistently improves forecastability---measured by Supervised Action Prediction (SAP) and World-Model Unrolling (WMU)---with minimal sacrifice in environment return. Prior approaches toward predictable RL have typically relied on simplicity-inducing regularizers, shaping policies only indirectly toward more forecastable behaviors or open-loop temporal abstraction such as action chunking. In contrast, FoRL makes predictability an explicit training signal by embedding forecasting directly into the learning problem. Compared to such entropy-based methods, FoRL achieves a superior accuracy--return trade-off and provides direct internal forecasts for potential downstream applications. A case study on Traffic Signal Control (TSC) illustrates how FoRL-generated Internal Forecasts (IF) can support downstream application tasks such as vehicle-side Green Light Optimized Speed Advisory (GLOSA). Moreover, the integrated forecastability design enables effective fine-tuning when forecasts themselves alter the environment dynamics. Overall, FoRL elevates predictability from a post-hoc diagnostic to a first-class inductive bias for RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses time-series forecasting when predictions are explicitly conditioned on exogenous/contextual variables. It argues that effective forecasters must cleanly separate endogenous target dynamics from exogenous drivers and provide an architecture and training recipe that fuses known‑future covariates into multi‑horizon predictions, presumably via a conditioning interface (e.g., cross‑attention, FiLM‑style modulation, or prompt-like control tokens). The evaluation spans several datasets, includes rolling-origin backtests, and reports both point and probabilistic metrics.

### Strengths
The problem framing is practical and relevant. Distinguishing what information is available at inference time (and when) is essential, and the paper highlights leakage risks.

The proposed factorization—separate encoders with a conditioning interface and multi‑horizon objectives—is a reasonable, well‑motivated recipe that many practitioners will recognize as a strong baseline.

The inclusion of probabilistic metrics (e.g., pinball/CRPS, coverage) and rolling-origin evaluation indicates awareness of deployment needs.
Ablations that mask exogenous inputs provide useful evidence that improvements come from conditioning rather than incidental effects.

### Weaknesses
The paper’s core idea—covariate‑aware encoders, conditioning via attention/FiLM, multi‑horizon losses—has strong precedent in TFT (Temporal Fusion Transformers), DeepAR variants with covariates, N‑BEATSx, PatchTST with exogenous features, and a wide body of sequence‑to‑sequence models for time series. The manuscript does not clearly delineate what is architecturally or theoretically new beyond a careful repackaging of established components. Without a sharper conceptual advance (e.g., provable leakage‑safe training, a theoretically grounded conditioning operator, or a demonstrably new robustness property), the case for novelty is weak.

Although the paper mentions anti‑leakage concerns, it is not auditable from the description that no forms of lookahead remain (e.g., horizon‑spanning feature engineering, target-aware normalization across train/validation boundaries, improperly lagged variables, or inadvertent use of true future covariates that wouldn’t be known at inference). In conditioned forecasting, even subtle pipeline decisions can invalidate gains. The paper needs a strict data availability table per horizon and unit tests that would fail under any leakage.

Insufficient robustness analysis to covariate error and missingness. In many deployments, exogenous inputs are forecasts themselves (noisy and biased) or arrive late/missing. The evaluation appears to lack controlled perturbation experiments (e.g., injecting calibrated noise or bias into covariates, simulating outages), and sensitivity curves showing how performance degrades as covariate quality worsens. Absent this, practical value is uncertain.

It is unclear whether the strongest covariate‑aware deep baselines are included and tuned well (e.g., TFT with its gating and variable selection, PatchTST/N‑HiTS variants with exogenous inputs, classical gradient boosted trees with rich lag features, Prophet/XGBoost with regressors). If some of these are missing or under‑tuned, the reported improvements may be overstated.

Reporting coverage alone is insufficient; calibrated probabilistic forecasting usually demands reliability diagrams, sharpness‑vs‑calibration trade‑offs, and possibly conformal calibration comparisons. The absence of a rigorous calibration study undermines claims about uncertainty quality.

The paper does not provide latency/throughput profiling, memory footprint, or accuracy‑cost trade‑offs. Cross‑attention over long contexts and multi‑horizon decoding can be expensive; without operational numbers (batch sizes, horizons, wall‑clock), it’s hard to judge production viability.

Stakeholders often require explanations for which covariates drive forecasts. The paper largely treats the model as a black box. Without temporal attribution analyses (masking, SHAP/SA, attention patterns) or counterfactual what‑ifs (e.g., removing promotions), trust and debugging are hindered.

### Questions
What concrete mechanisms guarantee leak‑free training and evaluation across all data processing and feature generation steps? Can you provide a schema that lists availability per feature and horizon, plus unit tests to detect leakage?

How does accuracy and coverage degrade under systematically perturbed covariates (Gaussian noise, bias, missingness) and under distributional regime change? Please provide sensitivity curves and threshold analyses.

Which strong covariate‑aware baselines are included (TFT, PatchTST+X, N‑BEATSx, GBDTs with rich lags and regressors), and how are they tuned to parity?

What are the inference latency and memory costs per horizon and per batch on typical hardware? Can you offer a distilled/linearized variant for real‑time settings with quantified accuracy loss?

Can you provide temporal attribution and counterfactual analyses that reveal which covariates drive forecasts and when?

The manuscript’s positioning against strong covariate‑aware baselines is underdeveloped. For example:

Temporal Fusion Transformers already offer variable selection, gating, and interpretable conditioning over known‑future inputs.
DeepAR/DeepState and modern CNN/Transformer forecasters handle covariates in multi‑horizon formats.
Conformal prediction and distributional objectives for calibrated intervals are well studied.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Forecasting-Conditioned Reinforcement Learning (FoRL), a model-free framework that integrates forecasting directly into policy learning. The key contribution of this work is having a policy that predicts both the immediate action and sequence of soft forecasts for the next L-1 actions. These predictions are then used to augment the input state and serve as input at the next timestep. Across three discrete-control benchmarks and across different prediction horizons, FoRL increases forecast accuracy and outperformed baselines.

### Strengths
+ The authors study an important problem that addresses long-horizon predictability of decision-making models.
+ Ample results and ablations are conducted

### Weaknesses
- There are several other techniques that accomplish a similar goal to your policy framing. Many robot policies leverage action chunking or temporally extended actions to have more consistent behavior. Options can also put actions of variable lengths to accomplish goals. Can you comment on how your framework is different than these and why these approaches fall short on forecastability? 
- Why doesn't the forecasting objective depend on the state? An objective ensuring that predicted actions match those taken at later points may not function well in tasks where unexpected events may occur and actions may need to shift drastically (such as autonomous driving).
- Overall contribution seems minor. Could the authors clarify the key contributions of this work?


Other:
- Reference to Figure 1 is very far from the Figure

### Questions
Please address the weaknesses above.

### Soundness
2

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
4

### Summary
This paper introduces Forecasting-Conditioned Reinforcement Learning (FoRL), a framework that augments model-free RL agents to explicitly predict their own future actions during training. The key innovation is making forecastability a first-class training objective rather than a post-hoc property. The authors propose two training approaches: Reward Conditioning (RC) which penalizes deviations between actions and earlier forecasts, and Loss Conditioning (LC) which adds an auxiliary forecasting loss. Experiments across three discrete-action environments (LunarLander, Highway-env, and Traffic Signal Control) demonstrate that FoRL achieves better forecastability-return trade-offs compared to baselines including TERL. The paper includes a compelling real-world application in traffic signal control with GLOSA integration.

### Strengths
- The paper effectively motivates the importance of forecastability in real-world applications like multi-agent coordination and human-AI collaboration. The distinction between post-hoc forecastability measurement and embedding it as an inductive bias is well-articulated.
- The experiments show that the approaches proposed enable good forecastability across different environments.
- The paper provides analysis of how forecasting pressure affects policy structure through Lipschitz continuity, compression metrics, and state visitation preferences. This provides valuable insights into why FoRL works.

### Weaknesses
1. The core contribution lies in incorporating forecastability into the policy learning objective via Eq. (6) and (7). The two approaches seem quite straightforward to be thought of when anyone wants to increase the forecastability of their RL algorithms. Thus, my biggest concern is the contribution may be not strong enough.
2. Baselines like RPC, though mentioned in Related Work, are not compared in the experiments.

**Minor**
- The legends in Figure 1 are too small to recognize.

### Questions
1. Line 320: How is the difference between two distribution measured by L2 distance?
2. Can the proposed approaches be generalized to environments with continuous action spaces? For example, the indicator function in Eq. (6) can be extended by dividing a continuous interval to multiple bins. It would be interesting to see if the approaches work in such settings.

### Soundness
3

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
3

### Summary
This work concerns the predictability or forecastability of model-free RL algorithms. This is an important aspect in real-life deployment of RL and this paper proposes to incorporate forecastability directly in the learning objective, leading to Forecasting-Conditioned Reinforcement Learning (FoRL). 

To do this, the policy action space is expanded to predict multiple steps into the future, but only the first action is taken. The policy observation space is also expanded to include previous forecasts. The authors propose two variants: Reward Conditioning (RC) where the reward is augmented with a term that encourages the current timestep’s action to be close to already-predicted actions (discounted over time), and Loss Conditioning (LC) where the policy loss is augmented with a discounted loss term over future predictions. 

Experimental results show that variations of these approaches improve forecastability without sacrificing performance in a few environments and that FoRL induces a smoothness in the action landscape. They also illustrate the usefulness of this policy in a traffic signal control environment.

### Strengths
The paper is well-written and easy to follow. I think the approach taken is original and makes sense. Although the environments that were experimented on were limited, the results were thorough and many aspects of the approach was explored in them. I also liked the application to Traffic Intersection Management, which took into consideration real-life limitations, and was an innovative setting to consider.

### Weaknesses
I have some questions which I need clarification on, I have added them in the section below. 

Minor: 
The font size is Fig. 2 is too small.

### Questions
- Could enforcing forecastability in this way hurt exploration, and therefore the policy performance? If the policy ends up in a locally optimum point for example, would this kind of objective simply delay convergence or keep it stuck there?

- In Eq. (7) where are the supervising signals $A_{t+k}$ coming from? I understand that $\hat{p}_{t}^{k}$ are the policy’s output distributions and $A_t$ is the action taken in the environment, 
so is it that

$$A_{t+k} = \text{ argmax}_{\mathcal{A}} \quad \hat{p}_t^k \quad ?$$ If so, how does this encourage forecastability?

### Soundness
3

### Presentation
4

### Contribution
3
