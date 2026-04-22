# Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Pre-training world models on large, action-free video datasets offers a promising path toward generalist agents, but a fundamental flaw undermines this paradigm. Prevailing methods train models to predict a single, deterministic future, an objective that is ill-posed for inherently stochastic environments where actions are unknown. We contend that a world model should instead learn a structured, probabilistic representation of the future where predictive uncertainty correctly scales with the temporal horizon. To achieve this, we introduce a pre-training framework, **H**orizon-c**A**librated
**U**ncertainty **W**orld **M**odel (HAUWM), built on a probabilistic ensemble that predicts frames at randomly sampled future horizons. The core of our method is a Horizon-Calibrated Uncertainty (HCU) loss, which explicitly shapes the latent space by encouraging predictive variance to grow as the model projects further into the future. This approach yields a latent dynamics model that is not only predictive but also equipped with a reliable measure of temporal confidence. When fine-tuned for downstream control, our pre-trained model significantly outperforms state-of-the-art methods across a diverse suite of benchmarks, including MetaWorld, the DeepMind Control Suite, and RoboDesk. These results highlight the critical role of structured uncertainty in robust decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that action‑free video pre‑training for world models is insufficient if the objective forces a single deterministic future. It proposes HAUWM, a pre‑training framework that (i) predicts variable‑horizon futures using a temporal embedding, and (ii) encourages uncertainty to increase with prediction horizon via a Horizon‑Calibrated Uncertainty (HCU) loss. Concretely, an ensemble of dynamics heads outputs Gaussian latent predictions; the HCU loss promotes model disagreement scaled by the time gap, combined with a standard ELBO‑style predictive loss in a total loss. During fine‑tuning, the pre‑trained uncertainty‑aware stream is stacked with a lightweight action‑conditioned stream to learn control.

### Strengths
1. The paper identifies an insightful problem of action‑free video pre‑training, where deterministic next‑frame targets suppress stochasticity. The proposed solution (model horizon‑dependent uncertainty) is well motivated. 

2. HAUWM integrates variable‑horizon prediction with sinusoidal temporal embeddings and an uncertainty‑increasing prior. The approach is compatible with RSSM/Dreamer‑style world models and standard stacked fine‑tuning. 

3. The result is compelling: in pre‑training, ensemble heads diverge for longer horizons, whereas in fine‑tuning, they converge under fixed actions and show reduced uncertainty. This directly supports the paper’s claims.

### Weaknesses
1. It would be interesting and more convincing to see quantitative calibration metrics (coverage) under the influence of different numbers of dynamic heads. These coverage metrics effectively reflect the quality of the pre-trained model.  

2. Each head outputs Gaussian parameters, but the HCU loss uses only squared deviations of head means. Wouldn’t this conflate diversity with calibration?  

3. The reconstruction loss uses mean of ensembles. For multiple possible futures, this choice tends to average them, which would contradict with the diversity encouraged by HCU.  

4. Can authors also ablate the number of ensembles, especially when M=1?

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The author proposes an uncertainty-based World Model (WM) for model-based reinforcement learning in decision-making. The training is in two stages: the first is uncertainty-based pretraining, where the first stage aims to learn action-free WM, and the second stage is for task-specific fine-tuning. The paper overall presents a clear and sound storyline about their action-free WM and has adequate experiments to support their evidence (e.g. Fig. 5). However, I'm still sceptical about the effectiveness and ablation study. I tend to accept this paper at this point, but I may change my view based on the author's responses.

### Strengths
1. The writing is clear and easy to follow; the claim and presentation are reasonable and clear.
2. The HCU loss is straightforward and theoretically sound.

### Weaknesses
**Major Concern**
1. The used benchmark is too old, making the actual performance hard to appreciate. The author should try some latest benchmarks, e.g. SimpleEnv [1], CALVIN [2], BridgeData V2[3], to make their findings more convincing. 
2. Does the second fine-tuning phase largely depend on different tasks under one specific setting? For example, should Push Green and Open Slide on Fig.6 be trained separately or not? If so, I'm curious about the performance under cross setting: finetune WM on push green and perform RL on open slide and vice versa, which will better support the author's claims.
3. I'm not very sure about the ablation study on the uncertainty assumption: what if you just pre-train the WM using reconstruction loss and still fine-tune it on different tasks? Will it be really bad?


**Minor**
1. The notation is misaligned: $T$ is the segment length in Sec. 4.2, but stands for embedding length in Fig.2.

2. On Eq.3, the model's disagreement is linearly degraded with $k$; how about the other function? e.g. exponential or sinusoidal.

**Ref.**

[1] Evaluating Real-World Robot Manipulation Policies in Simulation CoRL 2024

[2] CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks R-AL 2022

[3] BridgeData V2: A Dataset for Robot Learning at Scale CoRL 2023

### Questions
see weakness

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
The paper introduces the Horizon-cAlibrated Uncertainty World Model (HAUWM) framework to address the limitation of "false certainty" in world models pre-trained on action-free video, a flaw stemming from deterministic prediction objectives that suppress environmental stochasticity. HAUWM employs an ensemble of probabilistic dynamics heads and a novel Horizon-Calibrated Uncertainty (HCU) loss that explicitly enforces predictive uncertainty to scale monotonically with the temporal horizon. The authors claim that HAUWM significantly outperforms state-of-the-art baselines across diverse downstream control benchmarks (DMC, MetaWorld, RoboDesk) and demonstrates versatility in Imitation Learning and Offline RL. Qualitative evidence supports the model's well-calibrated uncertainty: the ensemble predicts diverse, high-uncertainty futures during pre-training, which converge to a low-uncertainty outcome when conditioned on a specific action during fine-tuning.

### Strengths
1.	Novel and Well-Formulated Solution (HCU Loss): The proposed Horizon-Calibrated Uncertainty (HCU) loss is a novel contribution. It enforces an intuitive and realistic inductive bias: predictive uncertainty should grow monotonically with the temporal horizon. This is achieved by maximizing the model disagreement of a probabilistic ensemble, scaled by the horizon length.

2.	Strong Empirical Results on Diverse Benchmarks (RQ1): The method (HAUWM) achieves state-of-the-art performance and sample efficiency across a broad and challenging suite of benchmarks, including locomotion (DMC), diverse manipulation (MetaWorld), and complex, long-horizon tasks (RoboDesk). The advantage is particularly noticeable in dynamically complex tasks like Walker Run and Hopper Hop.

### Weaknesses
1.	Motivation for HCU Loss Lacks Direct Quantitative Evidence: The core innovation of the paper—the Horizon-Calibrated Uncertainty (HCU) Loss—is strongly motivated by the claim that prevailing RSSM-based models (like APV) suffer from "false certainty," where predictive uncertainty is suppressed or collapses over long temporal horizons. However, the paper does not present direct empirical evidence (e.g., a dedicated graph showing the variance/disagreement of APV or ContextWM decreasing as the prediction horizon $k$ increases) to substantiate this foundational claim. Without quantitative data explicitly showing this defect in the baseline, the necessity and direct corrective action of the HCU loss are less rigorously established.

2.	Novelty of Variable-Horizon Prediction: While the HCU loss is novel, the concept of variable-horizon prediction or conditioning dynamics on a temporal embedding is not entirely new (e.g., it is related to positional encodings used in Transformers or general long-term planning models). The authors should more clearly contextualize their specific variable-horizon implementation and its necessity, distinguishing it from prior art that uses temporal embeddings for sequence modeling. 

3.	Limited Task Generalization Scope: Although HAUWM shows strong performance across 10 diverse tasks in three major benchmarks (DMC, MetaWorld, RoboDesk) and generalizes well to Imitation Learning and Offline RL, the total number of tasks tested is a relatively small subset of the established control environments. To fully validate HAUWM as a versatile general-purpose foundation model, the empirical validation should be expanded to include a wider variety of structurally distinct tasks, particularly from the extensive MetaWorld suite, to more robustly substantiate the model's claim of broad generalizability.

### Questions
1.	The authors claim that prevailing methods predict a 'single, deterministic future.' However, established baselines such as APV, ContextWM, and PreLAR are founded on DreamerV2's Recurrent State Space Model (RSSM), which explicitly models state uncertainty using a Gaussian distribution. Given this, what is the specific technical advantage of employing the proposed multi-head dynamics prediction ensemble over the existing Gaussian uncertainty modeling within the standard RSSM latent state? Please provide a detailed analysis of this distinction and how the ensemble mitigates the alleged 'deterministic bias.'

2.	In the image reconstruction phase, the predicted future latent state $s_{t+k}$ is set to the ensemble mean, $\overline{\mu}_{t+k} = \frac{1}{M}\sum_{i=1}^{M}\mu_{\theta_{i}}(s_{t}, \Delta t_{k}^{e})$. This mean $\overline{\mu}_{t+k}$ is then decoded to reconstruct the observation $\hat{o}_{t+k}$. Using the ensemble mean $\overline{\mu}_{t+k}$ directly, instead of sampling $s_{t+k}$ from the distribution represented by the ensemble, introduces determinism into the observation reconstruction process. Could the authors explain why they chose this deterministic decoding approach, and how this design choice—which sacrifices the model's stochasticity at the observation level—affects the overall goal of learning an uncertainty-calibrated world model?

### Soundness
2

### Presentation
4

### Contribution
2
