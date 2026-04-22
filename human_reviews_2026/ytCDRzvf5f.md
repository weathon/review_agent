# ERPV: Enhancing Visual Reinforcement Learning with Partially Reliable Knowledge from VLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Visual Reinforcement Learning (VRL) aims to learn optimal control policies from scratch,  a process that often suffers from low exploration efficiency. Integrating large-scale vision-language models (VLMs) offers a promising solution, as they provide rich prior knowledge about the environment. However, VLMs are only partially reliable when directly applied to VRL: the inferred actions may be wrong in certain states, and the inability to identify reliable action alignment can result in excessive exploration by the agent. We propose ERPV, a novel method that effectively enhances VRL with partially reliable knowledge from VLMs. ERPV introduces two key modules: (1) Value-aware Policy Guidance, which estimates the reliability of VLMs across different states and adaptively selects trustworthy VLM-inferred actions to guide policy learning; (2) VLMs-guided Entropy Regularization, which reduces over-exploration by comparing the confidence between VRL policy and VLMs-inferred actions. Extensive experiments show that, compared to the state of the art, ERPV achieves competitive performance in both policy effectiveness and sample efficiency under diverse, complex visual control tasks. The code has been placed in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ERPV, a novel visual reinforcement learning method.
Specifically, ERPV leverages the vision language model to (1) select actions to guide policy learning and (2) provide reference actions for better exploration.
Experiment results on various benchmarks show that ERPV achieve superior performance and learning efficiency.

### Strengths
1. The paper writing and structure are clear.
2. Good motivation and novelty regarding VLM as the prior knowledge for policy learning and exploration (VPG and VER components).
3. Empirical results demonstrate the superior improvements on various benchmarks.

### Weaknesses
1. As mentioned in the limitation, it would be great to analyse the theoretical properties of ERPV, including convergence and optimality. It feels like be easy to extend from SAC.
2. The training speed deeply depends on the inference speed of VLM. It would be better to have the statistics of the training speed comparison in section 4.4.

### Questions
See weakness.

### Soundness
3

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
The paper proposes ERPV as an approach to benefit from common sense in VLM in visual reinforcement learning (RL). ERPV leverages predictions frmo a VLM  in a soft-actor-critic RL framework  by regerresing the VLM action. The regression loss is weighted by a coefficient representing the advantage of using VLM-based actions over using action from the RL policy. The authors also propose using the prediction error of a dynamics model conditioned on the VLM action as an exploration signal.

### Strengths
- The paper is well written
- Leveraging VLM common sense to guide policy search is a novel and interesting idea
- The results are strong and quite promising
- Experiments include ablations of various design choices and clearly disentangle the role of different components

### Weaknesses
- The paper lacks an extensive discussion of the main assumption made here: VLMs are trained with action-free data, the best they could actually do in action selection is either some sort of nearest neighbor if the domain data was seen during VLM training, or provide actions that are at a semantic level at best. Most action representations in complex systems (e.g. robotic manipulation) are far more challenging on the lower (non-semantic) level. In such complex environments, it is unclear how such a supervision could even help beyond just simple early-state exploration ( a problem that is potentially non-existent if expert data is available, which is becoming the case)
- Most environments used in the experiments are low-dimensional, and involve simple dynamics, it would be interesting to demonstrate the applicability of the method to more complex domains, or at least understand its limits in such domain

### Questions
Can you provide a more extensive discussion on how you expect your assumption of "VLM actions being a good source of supervision" to scale or fail to scale to high-dimensional complex tasks that involve low-level action?

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
2

### Summary
This paper presents ERPV, a method that integrates partially reliable knowledge from Vision-Language Models (VLMs) into visual reinforcement learning (VRL). The key motivation is that while VLMs contain useful commonsense priors, their action reasoning is often unreliable or inconsistent across states. To address this, the authors propose two mechanisms: Value-aware Policy Guidance (VPG): which dynamically estimates the reliability of VLM-inferred actions by comparing their Q-values with those of the RL policy, selectively applying guidance when the VLM’s suggestion appears better. VLM-guided Entropy Regularization (VER): adjusts exploration via an entropy coefficient that depends on how consistent the RL policy’s actions are with the VLM’s inferred actions, encouraging exploration when they diverge and exploitation when they align. Experiments across Carla, DMControl, and CarRacing benchmarks show that ERPV improves both sample efficiency and final performance, even when the VLM guidance is noisy or unreliable.

### Strengths
- Timely and relevant problem: The paper addresses an emerging and underexplored question — how to integrate large pretrained vision-language models into reinforcement learning while handling their imperfect reasoning.

- Thoughtful formulation: The introduction of reliability estimation (VPG) and entropy adjustment (VER) feels intuitive and conceptually clean, combining the strengths of teacher–student distillation and adaptive exploration control.

- Robust empirical results: Across several benchmarks, ERPV consistently outperforms prior VLM-assisted RL baselines (e.g., DSF, ASF, DGC), and even performs comparably to base RL when VLMs are unreliable.

- Comprehensive experiments: The inclusion of diverse settings (e.g., CARLA, DMControl), multiple VLM backbones (Qwen2-VL, LLava), and ablations on both modules (VPG/VER) provides solid empirical evidence.

- Clear presentation: Figures and tables (especially Fig. 5 showing dynamics and entropy coefficient) make it easy to follow how the method behaves during training.

### Weaknesses
- Incremental conceptual novelty: The main novelty lies in combining selective guidance and entropy modulation, but both components resemble ideas from adaptive distillation and uncertainty-aware exploration. The conceptual leap is moderate.

- Lack of theoretical insight: The paper would benefit from some formal justification (e.g., why the proposed difference or transition-based confidence metric leads to stable convergence). 

- Dependence on pretrained critic: Since ​the critic is trained using VLM actions, its generalization and possible bias are underexplored — what happens if the pretraining domain diverges from the RL environment?

- Limited real-world validation: All experiments are simulation-based. The authors’ claim of “real-time deployability” is interesting, but there’s no demonstration on an actual robotic platform.

- Scalability & compute details: The training cost of ERPV compared to vanilla SAC or other VLM-distilled methods isn’t discussed, making it hard to assess practicality.

### Questions
-How sensitive is ERPV to the hyperparameters lambda? Does over-weighting VPG risk reinforce VLM errors?

- Could the proposed Action Confidence Function (ACF) be replaced by simpler distance metrics (e.g., cosine similarity between action logits) without major loss?

- How much does the performance depend on the choice or pretraining quality of the VLM critic?

- Have you tested ERPV with textual prompts that are deliberately ambiguous or wrong to examine robustness?

- Could the approach be generalized to multi-modal feedback beyond actions, such as state representations or reward shaping?

### Soundness
2

### Presentation
2

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
This paper proposes ERPV, to address the issue of low exploration efficiency in VRL by introducing prior knowledge provided by pre trained VLMs. Experiments are conducted on Carla, DMC and CarRacing.

### Strengths
1. To the best of my knowledge, the paper is the first to systematically identify and formalize the challenge of “partially reliable knowledge.” 

2. The work targets the central tension in integrating VLMs with VRL. 

3. The experimental evaluation is extensive.

### Weaknesses
1. The core idea does not move beyond the teacher–student paradigm and remains within the traditional setting where the teacher provides knowledge and the student learns it. 

2. Discarding the VLM at test time forfeits substantial information encoded in the VLM and risks overfitting to the test environment due to limited training. 

3. There is no comparison of computational cost or runtime for the training phase.

### Questions
1. Given that VPG can already approximate the ground-truth Q reasonably well, why not directly use this Q estimate to optimize the actor network? 

2. In VER, high consistency between the VLM and RL policies indicates reliability, whereas low consistency indicates unreliability and triggers increased exploration. Suppose the VLM’s action is unreliable, but after exploration the RL policy becomes reliable and efficient; then the consistency between the VLM and RL policies would remain low. In that case, would ERPV continue to explore such states indefinitely? Furthermore, if both the VLM’s action and the RL’s action are reliable but diverse, the consistency may still remain low. In this scenario, does ERPV favor exploration or exploitation?

### Soundness
2

### Presentation
3

### Contribution
2
