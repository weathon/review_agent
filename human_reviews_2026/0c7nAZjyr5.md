# From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Navigation foundation models trained on massive web-scale data enable agents to generalize across diverse environments and embodiments. However, these models, which are trained solely on offline data, often lack the capacity to reason about the consequences of their actions or adapt through counterfactual understanding. They thus face significant limitations in the real-world urban navigation where interactive and safe behaviors, such as avoiding obstacles and moving pedestrians, are critical. To tackle these challenges, we introduce the Seeing-to-Experiencing (S2E) learning framework to scale the capability of navigation foundation models with reinforcement learning. S2E combines the strengths of pre-training on offline videos and post-training through reinforcement learning. It maintains the model's generalizability acquired from large-scale real-world videos while enhancing its interactivity through reinforcement learning in simulation environments. Specifically, we introduce two innovations:
1) an Anchor-Guided Distribution Matching strategy for offline pretraining, which stabilizes learning and models diverse motion patterns through anchor-based supervision; and
2) a Residual-Attention Module for reinforcement learning, which obtains reactive behaviors from simulation environments without erasing the model’s pretrained knowledge.
Moreover, we establish a comprehensive end-to-end evaluation benchmark, NavBench-GS, built on photorealistic 3D Gaussian Splatting reconstructions of real-world scenes that incorporate physical interactions. It can systematically assess the generalizability and safety of navigation foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes Seeing-to-Experiencing (S2E), a training recipe for goal-conditioned navigation that keeps visual priors from offline videos and adds interaction skills with reinforcement learning in simulation. It introduces a new action representation, using an anchor-conditioned Gaussian mixture to represent multiple short-horizon motions under the same observation. It also adds a Residual-Attention Module that fine-tunes only residual cross-attention while freezing visual encoders and self-attention to limit sim-to-real drift.
Experiments in the NavBench-GS benchmark show RL fine-tuning outperforms supervised fine-tuning under the same budget, scales better than more offline data alone, and transfers zero-shot to wheeled and legged robots, with ablations confirming both modules are important.

### Strengths
1. The paper shows reinforcement learning is extremely helpful for closed-loop navigation tasks.
2. The paper proposes using several fixed anchor points and GMM to represent the action space. This is essential specifically to the navigation task. The authors also provide in-depth analysis and comparison to other representations.
3. The paper provides insightful discovery on the data saturation margin in the embodied navigation task and shows RL to be a more robust and efficient learning technique.
4. The paper shows zero-shot depolyment of the trained model on both wheeled and quadroped robots. Experiment result shows superior performance compared to other baselines.

### Weaknesses
1. The "Residual attention module" paragraph in section 3.2 (Line 258-268) is a bit confusing. The statement is not a well established conclusion in literature. The authors should rewrite the paragraph by providing more concrete derivation of the conclusion.
2. In ablation study (Line 462-466 and Appendix Sec. D.5), there is no comparison with difussion policy models. The authors should add comparison to that to better support the discussion in Sec. 3.1.

### Questions
1. In Fig. 6(b) left, why does SFT still sufer from overfitting even in the in-distribution Urban-sim evaluation?
2. In reinforcement learning and simulative benchmark, there is no specific embodiment. Then how is collision checking done? Are they achieved by a navmesh or using a uniform collision volumn?
3. In Tab. 1, what does ZeroPolicy mean? Is it a pure RL model? I'm also curious on the authors insights on why (or why not) pure RL training might not perform well in urban navigation while they work well in indoor environments like the HM3D dataset?

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
This paper proposes the S2E framework, combining offline video pretraining (via AGDM strategy) and simulation RL finetuning (with RAM module). It builds NavBench-GS, verifies RL boosts navigation model performance, breaks diminishing returns of data scaling, and enables zero-shot transfer across robots.

### Strengths
1. This paper proposes Anchor-Guided Distribution Matching (AGDM), which uses an anchor-guided Gaussian Mixture Model to model multimodal navigation trajectories, capturing diverse valid actions under the same observation while ensuring training stability.

2. This paper designs the Residual-Attention Module (RAM), which freezes pretrained components and adds trainable residual branches to cross-attention layers, enabling the model to gain interactive skills via RL without losing pretrained generalizable knowledge.

3. This paper establishes the NavBench-GS benchmark, built on photorealistic 3D Gaussian Splatting scenes with physical interactions, realizing closed-loop policy evaluation and solving the reproducibility issue of real-world navigation testing.

### Weaknesses
1. The model relies solely on visual input and lacks 3D perception capabilities, leading to occasional failure in obstacle avoidance in some scenarios, which is a persistent limitation for vision-only navigation approaches.

2. The real-world evaluation scenarios are relatively limited (only 25 scenarios), and the generalization performance of the S2E framework in more complex and diverse urban environments (e.g., extreme weather, complex traffic conditions) has not been verified.

3. The humanoid robot in cross-embodiment evaluations shows notably lower success rates compared to wheeled and quadruped robots, yet the paper does not deeply analyze the root causes (e.g., joint complexity impacts) or propose targeted optimization strategies for humanoid platform adaptation.

4. The RL finetuning relies on modified URBAN-SIM environments, but the paper only briefly mentions procedural generation rule adjustments without detailing how these rules ensure the simulated environments fully align with real urban spatial layouts, potentially limiting the sim-to-real transfer reliability

### Questions
1. Given that the current model lacks 3D perception, what specific 3D information integration methods (such as depth prediction or occupancy prediction) do you plan to adopt in future work, and how will you balance the computational cost of 3D perception with the real-time performance of navigation?

2. The NavBench-GS benchmark currently covers 26 scenarios, but real urban environments involve more dynamic elements (e.g., sudden appearance of vehicles, temporary road closures). Will you expand the benchmark to include such scenarios, and what criteria will be used to select new scenarios to ensure the benchmark’s representativeness?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents results from RL-finetuning of a navigation policy that is pretrained by SFT over static datasets. It empirically shows the improvements afforded by RL, introduing a way to provide data diversity and an architecture for the policy.

### Strengths
An unusual combination of SFT and RL for learning-based navigation - while existing approaches focus on one or the other, this paper shows the value of doing both.

### Weaknesses
The anchor description is not very clear. It seems to be constant-curvature arcs - a clearer explanation is needed.
Unclear how the constant curvature arcs approximate the full diversity of demonstration paths - is the matching performed instantaneously (i.e., for an instantaneous vx, w command mapped to the corresponding curvature?)

### Questions
Can you explain how the anchors are defined and chosen?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes S2E (Seeing-to-Experiencing), a framework for training navigation foundation models that combines offline pretraining on real-world videos with reinforcement learning (RL) fine-tuning in simulation. The key technical contributions include: (1) Anchor-Guided Distribution Matching (AGDM) for modeling multimodal navigation behaviors during pretraining, (2) a Residual-Attention Module (RAM) that enables RL fine-tuning while preserving pretrained knowledge, and (3) NavBench-GS, a benchmark built on 3D Gaussian Splatting scenes. The authors claim that RL alleviates diminishing returns from scaling offline data alone and enables zero-shot transfer to real-world scenarios.

### Strengths
Comprehensive System: The paper presents an end-to-end system from data collection through real-world deployment, which requires significant engineering effort.

Cross-Embodiment Evaluation: Testing on wheeled, quadruped, and humanoid robots (Table 6) demonstrates some generality, though the humanoid results are quite poor.

Honest Limitations Discussion: Section 5 acknowledges the limitation of vision-only approaches and collision failures.

Detailed Implementation: The appendix provides extensive implementation details, which aids reproducibility.

### Weaknesses
1. Lack of Theoretical Insight: The paper doesn't explain why RL helps beyond showing empirical improvements. What specific failure modes of offline learning does RL address? What inductive biases make certain behaviors learnable only through interaction?

2. Modest Improvements: Many reported improvements are marginal (e.g., Table 2: 0.51 vs. 0.32 success rate for wheeled robot). Given the additional computational cost of RL (8 hours on L40S GPU), the cost-benefit tradeoff is unclear.
Cherry-Picked Comparisons:

3. CityWalker* (retrained on same 100h data) actually achieves 0.67 SR vs. S2E's 0.82 SR in empty scenes (Table 1), suggesting the RL contribution is partially due to other factors;
Figure 6(a) compares against "prior methods" shown as dotted lines, but these prior methods use different evaluation protocols


4. Limited Failure Analysis: The paper shows successful cases but doesn't systematically analyze failure modes. When does RL fine-tuning hurt performance? What environments or scenarios remain challenging?

5. Questionable Design Choices:

Why freeze the visual encoder during RL? This prevents learning better visual features for interaction
Why use a simplified entropy approximation (Eq. 11) instead of more accurate estimators?
The stochastic goal masking strategy (Section E.2) seems arbitrary - no ablation justifies the specific probabilities chosen


6. Reproducibility Concerns: Despite promising code release, the method depends on:
Proprietary Unitree robot APIs
Multiple datasets with different licensing terms
NVIDIA IsaacSim which requires expensive GPU resources
Hand-tuned reward functions that may not transfer to other environments

7. Missing Related Work

The paper should discuss and compare with:

Offline RL for Robotics: Kumar et al. (2020, NeurIPS), Nair et al. (2020, CoRL), Mandlekar et al. (2021, CoRL)
Vision-Language-Action Models: Driess et al. (2023, ICML - PaLM-E), Brohan et al. (2023, CoRL - RT-2)
Navigation Benchmarks: Savva et al. (2019, ICCV - Habitat), Xia et al. (2018, CVPR - Gibson)
Sim-to-Real Transfer: Peng et al. (2018, ICRA), Tan et al. (2018, CoRL)

### Questions
Data Efficiency: How many RL environment steps are needed? What's the sample complexity compared to collecting more offline data?

Reward Engineering: How sensitive is performance to reward function design? Have you tried learning rewards from human preferences (Christiano et al., 2017)?

Failure Modes: What percentage of real-world failures are due to:

Perception errors (misdetecting obstacles);
Planning failures (local minima);
Control errors (locomotion instability);
Sim-to-real gap


Comparison Fairness: Can you provide results where all methods are:

Trained on identical 100h dataset;
Evaluated on an established benchmark (not self-proposed);
Using the same evaluation protocol and metrics


Generalization: How does performance degrade with:

Different camera intrinsics/extrinsics;
Different weather/lighting conditions;
Novel obstacle types not seen in training


RL vs. SFT: The paper claims RL is better than SFT (Figure 6b), but:

How was the SFT data collected? From the pretrained policy or optimal demonstrations?
Did you try other offline RL algorithms (CQL, IQL, etc.) that might bridge the gap?

### Soundness
2

### Presentation
2

### Contribution
2
