# Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. 
Motivated by this observation, we propose Action-aware Dynamic Pruning (ADP), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. ADP introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. 
Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (e.g. 1.35× speed up on OpenVLA-OFT) while maintaining competitive success rates compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the inference efficiency problem of Vision-Language-Action (VLA) models in robotic manipulation and proposes a method called Action-Aware Dynamic Pruning (ADP). The core observation is that the redundancy of visual tokens is not constant during long-horizon manipulation tasks—it varies across different operation stages: redundancy is higher during coarse movements, while fine-grained grasping requires more detailed visual information. Based on this insight, the authors design two mechanisms to tackle the issue.

### Strengths
1. The method is clearly structured, and the writing makes it easy for readers to follow.

2. The experimental evaluation is relatively comprehensive.

### Weaknesses
1. I still believe that the core contribution (i.e., gating + pruning) demonstrates only a moderate level of novelty since it is relatively intuitive rather than a fundamentally transformative architectural innovation.

### Questions
1. In Figure 1, the word “pick” is mistakenly written as “picke.”

2. Although the paper discusses existing VLA acceleration methods (such as attention token pruning and structured compression), I think the authors should more clearly explain how ADP differs from these approaches in terms of innovation and performance boundaries.

3. I am also curious about the trade-off: since the paper introduces additional modules to learn how to prune tokens, at what level of token reduction does the time saved roughly offset the extra computation introduced by these modules?

4. From my previous experience, using text to attend to visual features often results in attention being paid to irrelevant regions. Does your method address this issue?

6. The authors propose to determine whether the current stage is “coarse” or “fine” based on recent changes in the end-effector’s trajectory. However, is this mechanism generalizable? Does it require manually set thresholds? And when the task type changes (e.g., when the motion involves mostly rotation rather than grasping), could this mechanism fail?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes **Action-aware Dynamic Pruning (ADP), a plug-and-play pruning framework for Vision-Language-Action (VLA) models. ADP combines text-driven anticipatory pruning with an **action-aware dynamic gating strategy (to determine when pruning should be applied based on recent motion dynamics). The approach is validated on the LIBERO benchmark and real-world robot tasks, demonstrating notable FLOPs reduction (up to 1.35× speedup) while maintaining or even improving task success rates.

### Strengths
1. Novel problem formulation:
- The paper identifies an underexplored property of robotic manipulation—action-aware visual redundancy—and systematically exploits it to design a dynamic pruning scheme. This insight bridges the gap between static pruning  and phase-dependent motion dynamics.

2. Methodological clarity and rigor:
- The paper provides clear mathematical formulations for both text-driven token importance and motion-based gating.
- The windowed trajectory definition and dynamic decision function are well-motivated and interpretable.
- The theoretical complexity analysis (Eq. 19–25) convincingly quantifies expected computational savings.

3. Comprehensive evaluation:
- The paper conducts thorough experiments on simulation and real robot setups.
- Ablation studies isolate the contributions of each component.
- Visualizations (Figures 4–8) effectively support the claimed interpretability of the pruning mechanism.

4. Strong empirical results:
- ADP consistently outperforms static baselines in both efficiency and accuracy trade-offs. Especially impressive real-world latency improvement (1.49×) with maintained success rate.

### Weaknesses
1. While the gating mechanism is intuitive, the paper lacks an analysis of stability or convergence under fluctuating motion magnitudes. How sensitive is the dynamic switching rule (Eq. 16–18) to noise or suboptimal motion planning?

2. The gating rules (mean or extrema) are empirical; there is no adaptive or learned thresholding. A sensitivity analysis of hyperparameters would strengthen the claim of robustness.

3.The baselines are comprehensive but mostly training-free ones. Including training-aware compression methods (e.g., DeeR-VLA, Mole-VLA) in a fair fine-tuned comparison would make the argument of “plug-and-play” stronger.

4.The ablation study primarily examines whether dynamic control is included, but does not vary pruning ratios adaptively within tasks. Showing per-stage pruning ratio curves would illustrate dynamic behavior more explicitly.

5. The real-robot evaluation setup is described in the appendix, but lacks release of calibration or dataset details. Code availability and parameter settings for dynamic thresholds are not stated.

### Questions
1. Have you tested how ADP behaves under sensor noise or inaccurate motion estimation?

2. Could the gating rule be differentiable and trained end-to-end with reinforcement signals?

3. How does ADP interact with temporal caching or diffusion-based decoders in future hybrid models?

4. What happens when action magnitude and task difficulty are decoupled?

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
4

### Summary
This paper introduces ADP, which is a pruning method for VLA. ADP adopts text-driven pruning and action-aware gating,

### Strengths
This paper aims to tackle one of the crucial problems in robotics, namely the efficiency of deployment.

The idea is intuitive and the method show good empirical results.

### Weaknesses
The design principal is kind of heuristic. Some rules (especially for the action-aware part) seem to be tailored for this pick-and-place. 

The experiments are focused on the OFT model. Not sure whether this proposed method is compatible with other backbones.

See questions below

### Questions
(1)	For the text-driven pruning, the visual tokens with low text attention score are discarded. However, during VLA’s training, no explicit constraints are applied on the text to image attention. How to make sure the correctness of the pruned area?

(2)	Since the method is claimed to be a plug-and-play component, results on ADP with other backbone are welcome. Currently, in table 1, it seems ADP is only combined with OFT.

(3)	In some cases, the pruned policy performs better than the baselines. It seems interesting. Is there any discussions on this part?

(4)	The design principal is kind of heuristic. Some rules (especially for the action-aware part) seem to be tailored for this pick-and-place. Other experiments are needed to show the proposed principal is universal.

### Soundness
2

### Presentation
2

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
This paper proposes a plug-and-play framework to improve the efficiency of Vision-Language-Action (VLA) models for robotic manipulation by dynamically pruning redundant visual tokens. The key idea is to combine text-driven token selection with an action-aware gating mechanism conditioned on recent end-effector motion. Experiments on the LIBERO benchmark and real-robot setups show that ADP reduces FLOPs and inference latency by up to 1.35× (on OpenVLA-OFT) with minimal accuracy degradation.

### Strengths
- Unlike prior pruning or compression methods (e.g., EfficientVLA, DeeR-VLA, VLA-Cache), ADP leverages **action-conditioned motion dynamics** to modulate token pruning during robotic manipulation. The insight that **visual redundancy varies with manipulation phase** (coarse vs. fine-grained motion) is both intuitive and unexplored, giving the paper a clear conceptual novelty.

- The paper's primary strength is its core insight that visual redundancy in VLA models is action-aware and correlates with the manipulation phase. This is a novel observation that moves beyond static or text-only pruning strategies.

- The method is validated with strong empirical results in simulation and the real world. In simulation, it provides a clear accuracy-to-speedup trade-off, achieving a 1.35x speedup with only a 2.7% drop in success rate (at 30% ratio). On a physical Jaco2 robot, it demonstrates a 1.49x reduction in latency.

### Weaknesses
- The action-aware gating mechanism, while effective, is built on a stack of heuristics. This includes the choice of motion metric ($\delta_i$ as Euclidean displacement), the specific gating rule ("adjacent-extrema"), the fixed window size ($\omega=8$), and hard-coded reset rules (e.g., a two-window "cold start" and a forced reset after three pruned windows). It is unclear how these settings would generalize to new tasks, robots with different dynamics, or different control frequencies without tuning.

- The pruning strategy relies on similarity scores from Layer 0. This is counterintuitive, as deeper layers are typically assumed to capture more fused semantic meaning, although the paper provides a good justification in Figs. 6 and 7.

- The experiments primarily rely on OpenVLA-OFT as the base model and omit results on diverse backbones (e.g., Pi0, GrooT, SmolVLA). Code availability or hyperparameter details are not specified beyond window size and ratios (Sec. 5.1), which hinders reproducibility.

- The paper notes possible degradation during fine manipulation (Sec. 4.2) but does not quantify how the dynamic switch may misfire or cause visual underrepresentation in complex scenes.

### Questions
- Could the authors provide results or qualitative visualizations on cases where the gating rule failed (e.g., incorrect pruning during fine-grained manipulation)?

- How sensitive is performance to the window length and retention ratio? An ablation over these parameters would clarify the generality of the dynamic policy.

- The real-world experiments used a single fixed camera, whereas the simulation setup used multi-view pruning, including a wrist view. How does the action-aware gate perform without a wrist camera, whose motion dynamics would be identical to the end-effector trajectory? Does this change simplify or complicate the tuning of the gating threshold?

- Could the **dynamic pruning controller** be made learnable (e.g., via reinforcement learning) instead of relying on hand-crafted thresholds?

### Soundness
3

### Presentation
3

### Contribution
3
