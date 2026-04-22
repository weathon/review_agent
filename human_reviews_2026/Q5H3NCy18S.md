# SPACeR: Self-Play Anchoring with Centralized Reference Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Developing autonomous vehicles (AVs) requires not only safety and efficiency, but also realistic, human-like behaviors that are socially aware and predictable. Achieving this requires sim agent policies that are human-like, fast, and scalable in multi-agent settings. Recent progress in imitation learning with large diffusion-based or tokenized models has shown that behaviors can be captured directly from human driving data, producing realistic policies. However, these models are computationally expensive, slow during inference, and struggle to adapt in reactive, closed-loop scenarios. In contrast, self-play reinforcement learning (RL) scales efficiently and naturally captures multi-agent interactions, but it often relies on heuristics and reward shaping, and the resulting policies can diverge from human norms. We propose human-like self-play, a framework that leverages a pretrained tokenized autoregressive motion model as a centralized reference policy to guide decentralized self-play. The reference model provides likelihood rewards and KL divergence, anchoring policies to the human driving distribution while preserving RL scalability. Evaluated on the Waymo Sim Agents Challenge, our method achieves competitive performance with imitation-learned policies while being up to 10× faster at inference and 50× smaller in parameter size than large generative models. In addition, we demonstrate in closed-loop ego planning evaluation tasks that our sim agents can effectively measure planner quality with fast and scalable traffic simulation, establishing a new paradigm for testing autonomous driving policies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
* This paper introduces SPACER, a novel framework for training realistic and human-like sim agents that combines RL and IL.
* The key idea is to have a lightweight, decentralized student RL policy that is anchored to a large pre-trained centralized teacher policy (reference model).
* The reference model is pre-trained to capture human driving data and provides a realism signal as a reward as well as a KL divergence term to the student during self-play.
* The KL divergence can be computed efficiently in closed-loop due to RL agent and reference model action space alignment.
SPACER achieves competitive realism scores with 50x fewer parameters and 10x faster inference as well as lower collision rates and superior reactivity in closed-loop planner evaluation

### Strengths
* This paper tackles an important practical problem: Creating sim agents that are realistic and human-like, yet, fast and cheap to run.
* The SPACER results are compelling: The small model performs well on WOSAC on the composite metric and clearly outperforms the baselines in collisions and offroad events.
* The ablation results clearly show the importance of the KL divergence term.
* The authors provide good examples of WOSAC weaknesses.

### Weaknesses
* VRUs are not controlled and follow logs, which is a significant shortcoming.
* Unclear benefits of the realism reward signal in the ablation. Also see my question below.
* While the RL agent inference is fast, training is expensive (first need to train a reference policy, then run inference on it during RL).
* The HR-PPO baseline is decentralized, which makes it weaker since SPACER can access a centralized reference model during training. A fairer comparison could be a centralized HR-PPO policy distilled into a decentralized one.

### Questions
* The ablation shows that the realism reward signal provides hardly any benefit and only the KL term is critical. This result is surprising to me. Do you have an explanation for it? Did you study this more?
* How sensitive is your method to the size of the discrete action space?

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
This paper presents SPACER (Self-Play Anchoring with Centralized Reference Models), a framework for training large-scale, human-like driving agents in simulation. The key idea is to anchor reinforcement learning (RL) agents trained via self-play to a centralized, pretrained imitation-learning (IL) model. The centralized model serves as a human-likeness reference, providing both a log-likelihood reward and a KL-divergence regularization term that guide the self-play policy toward realistic behaviors while retaining scalability and reactivity. The approach achieves significantly improved realism on the Waymo Sim Agents Challenge compared to baseline self-play RL and imitation-only models, with 10× faster inference and 50× smaller model size.

### Strengths
- Clearly identifies and addresses the gap between imitation learning (realistic but non-reactive) and self-play RL (reactive but unrealistic).

- Introduces a principled anchoring mechanism through likelihood and KL-based regularization.

- Demonstrates clear ablation and comparative analysis against strong baselines.

### Weaknesses
- The paper could elaborate on how anchoring parameters affect the trade-off between realism and exploration.

- The reference model’s dependence on large imitation datasets may limit generalization to low-data domains.

- The related-work section would benefit from citing related studies:

A. Kuefler et al., “Imitating Driver Behavior with Generative Adversarial Networks,” IEEE IV 2017.

R. P. Bhattacharyya et al., “Modeling Human Driving Behavior through Generative Adversarial Imitation Learning,” CoRR 2020.

H. Chen, T. Ji, S. Liu, and K. Driggs-Campbell, “Combining Model-Based Controllers and Generative Adversarial Imitation Learning for Traffic Simulation,” IEEE ITSC 2022.

K. Brown, K. Driggs-Campbell, and M. Kochenderfer, “Modeling and Prediction of Human Driver Behavior: A Survey,” arXiv:2006.08832, 2020.

### Questions
How sensitive is the performance to the choice or size of the reference imitation model?

Could the anchoring approach be generalized to other domains (e.g., pedestrian or cyclist simulation)?

Does the KL regularization ever constrain the policy too strongly, reducing behavioral diversity?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SPACeR (Self-Play Anchoring with Centralized Reference), a novel framework that integrates the scalability of self-play reinforcement learning with the realism of imitation learning. The key idea is to anchor decentralized self-play policies to a pretrained, centralized tokenized reference model via a KL-divergence alignment and a log-likelihood reward, enabling human-like behavior without relying on logged trajectories. This design allows SPACeR to achieve realistic, reactive, and efficient driving agents that are up to 10× faster and 50× smaller than large generative imitation models.

### Strengths
S1. SPACeR introduces an elegant integration of self-play reinforcement learning with a pretrained tokenized reference model, using KL-divergence alignment and likelihood-based rewards to anchor decentralized policies toward human-like behaviors. The approach is conceptually clean, easy to implement, and avoids heavy reliance on heuristic reward shaping or large generative models.

S2. The proposed lightweight decentralized policy (≈65k parameters) achieves over 10× faster inference and 50× smaller model size than state-of-the-art imitation-learning methods (e.g., SMART, CAT-K), while maintaining comparable realism. This demonstrates clear practical potential for large-scale, real-time autonomous driving simulation and planner evaluation.

### Weaknesses
W1. The effectiveness of SPACeR heavily relies on the pretrained tokenized reference model. If the reference distribution is biased or limited in coverage, the learned self-play policies may inherit those biases and fail to generalize to unseen or long-tail behaviors. The paper would benefit from a sensitivity or ablation analysis on different reference model qualities.

W2. The current experiments are restricted to vehicle agents, without incorporating pedestrians, cyclists, or mixed-traffic interactions. Since realistic urban driving often involves diverse and heterogeneous agents, demonstrating SPACeR’s adaptability to such settings would significantly strengthen its generality and practical relevance.

### Questions
Q1. How sensitive is SPACeR to the quality and coverage of the reference model? For instance, would using a smaller or domain-shifted tokenized model significantly affect policy realism or stability?

Q2. Could the reference signal (KL and likelihood reward) be updated dynamically or distilled into a lightweight surrogate during training to reduce reliance on a fixed pretrained model?

### Soundness
3

### Presentation
3

### Contribution
3
