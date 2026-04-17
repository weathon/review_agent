# VLA-IN-THE-LOOP: ONLINE POLICY CORRECTION WITH WORLD MODELS FOR ROBUST ROBOTIC GRASPING

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Large-scale Vision-Language-Action (VLA) models excel at mapping natural language instructions to robotic action. However, they typically treat actions as terminal outputs with imitation learning often leads to execution bias, lacking mechanisms for dynamic supervision or online error correction. Meanwhile, World models (WM) have shown promise for predictive reasoning, but prior approaches typically require continuous frame-by-frame rollout of long sequences, resulting in high computational cost and limited flexibility. In this work, we propose VLA-in-the-Loop, a novel framework that introduces an online intervention mechanism to correct the base VLA policy. Our core innovation lies in the use of a lightweight, composite World Model, not for continuous state prediction, but as an on-demand, event-triggered “corrector.” When the VLA proposes a high-stakes action (e.g., closing the gripper), at this critical juncture, our composite WM first employs its discriminative component to evaluate the action’s feasibility. Should the proposed action be deemed unviable, a generative model synthesizes a short video of a successful future trajectory from the current state. Robot will be guided to the correct position using actions decoded by inverse dynamics mode(IDM) and execute a corrected, more robust action. This plug-in architecture is not only computationally efficient but also enhances data utilization by learning from potential failures, thereby significantly improving the robustness
of VLA models against online disturbances. We validate our framework across multiple robotic grasping tasks on both simulation and real-world systems, demonstrating the effectiveness of using world models not only for prediction, but as active agents for real-time
correction in VLA-based robotic systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a way of coupling a VLA model with a video generation model, so that the latter "oversees" the former. Specifically, for important actions, like grasping, a discriminator (qwen-based VLM) detects if the VLA action is likely to succeed. If not, the actions are rolled back, and a video model generates a successful trajectory while an IDM model extracts the actions from this trajectory. This new action sequence is used instead. The authors show that this improves the baseline performance both in sim and real.

### Strengths
- The idea is simple and well explained; a reader gets the gist quickly.
- The authors show good improvements both in real and sim

### Weaknesses
- The novelty is limited
- The keyframe detection seems to only be grasping. This is a somewhat "hardcoded" method, which isn't very general 
- It is not clear that "rollback" is a generally feasible strategy in the real world
- Missing baseline/citation to DreamGen

### Questions
- Rollback isn't always feasible in the real world. E.g., the robot might drop something on the floor, and you can't rollback by inverting gravity. How does your method work in these settings?
- From table 3, it seems that more rollback works better. Is not not possible to roll back to the start of the episode and let the video model+IDM do the complete action generation? And following your table, isn't that likely to work better than your proposed method?
- Is there a way to generalize your method beyond just working for grasping?

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
2

### Summary
The paper introduces VLA-in-the-Loop, an online correction framework for VLA policies targeted at grasping. At a keyframe trigger—when the base VLA proposes gripper-close—a VLM-based discriminator judges feasibility; on predicted failure, the system rolls back k steps, invokes a video generator to “imagine” a short successful future, and feeds the imagined clip back to a unified VLA module (weight-shared with the discriminator) to decode a corrected action via an inverse-dynamics role. The approach aims to avoid continuous world-model rollouts by using an event-driven composite WM (discriminative + generative) for on-demand intervention. Experiments on SIMPLER (WidowX/Google Robot), LIBERO, and two real robots (Xiaomi, ALOHA) show consistent gains over strong baselines; ablations vary rollback depth and show robustness under online perturbations. Training uses BridgeV2-derived keyframe labels for the discriminator and successful-grasp clips for the generator;

### Strengths
Clear, modular intervention loop (“Propose–Evaluate–Imagine–Correct”) with an event-triggered WM that avoids continuous rollouts; well specified for grasp keyframes. 

Unified discriminator/actor via shared VLM and multi-task QA formulation—neat engineering to couple evaluation and action decoding. 

Solid empirical coverage across SIMPLER (tables with G/S and visual/variant suites), LIBERO, and real-world tasks; perturbation robustness tests are thoughtfully designed. 

Ablation on rollback depth explains why immediate corrections can be ineffective and why earlier state restoration helps the generator produce viable plans.

### Weaknesses
1. VQA-style discriminator calibration. The feasibility check is framed as text-prompted VQA with labels “suitable/unsuitable”. The paper lacks calibration/ROC evidence, thresholding, and prompt sensitivity analyses: false positives trigger unnecessary rollbacks; false negatives permit failures. This is critical because the discriminator gates generation and alters control.

2. Perception–action mismatch in imagination. The generator conditions primarily on images + text; it is unclear whether proprioception and contact state are modeled or enforced. Imagined videos may depict poses infeasible for the current robot kinematics, causing a covariate gap when the actor decodes actions from pixels. A formal constraint (e.g., action-conditioned or dynamics-consistent loss) is missing.

### Questions
Please see weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes VLA-in-the-Loop, an event-triggered, online correction framework for robotic grasping on top of a base Vision-Language-Action (VLA) policy. Rather than rolling a world model (WM) continuously, the system intervenes only at high-stakes actions (specifically, when the policy proposes to close the gripper) to (i) classify whether executing the action would likely fail (via a Qwen-VL2.5-based VQA discriminator), and if risky, (ii) imagine a short successful future video using a WAN-2.1 I2V diffusion model after rolling back k frames, and (iii) decode a corrected action sequence via an inverse-dynamics module to execute the grasp. The authors claim this plug-in loop improves robustness while paying the generation cost only when needed, and report gains on several manipulation benchmarks, including LIBERO-Franka (Table 3). Key implementation details include the discriminator’s VQA formulation, a curated 102k labeled keyframe dataset with suitable/unsuitable tags, and a generator trained/fine-tuned on 33k BridgeV2 clips plus 200 real-robot videos (Xiaomi, ALOHA). Reported latency is ~0.2 s per normal step vs ~23 s for a triggered correction (≈0.95 s discriminator + 20 s video generation).

### Strengths
Paper provides 
1. A clear systems recipe for event-triggered online correction: evaluate → (if risky) roll back → imagine → decode, targeting the exact moment where grasp outcomes hinge
2. integration of existing components (Qwen-VL-2.5, WAN-2.1, IDM) with unified QA supervision and parameter sharing, which is data/compute efficient in spirit.
3. Ablation signal that rollback depth materially affects success, aligning with intuition that one must retreat to a recoverable state before correcting.
Paper performs a number of experiments across simulation and real-robot, suggesting practical value if latency and triggering are handled carefully.

### Weaknesses
The paper tells an appealing “online correction” story, but the evidence is not yet strong. The trigger is too narrow (only at gripper closure), the system incurs ~23 s stalls per intervention with no statistics on how often that happens, and there are no compute-/data-fair strong baselines to isolate the value of video imagination. On the simulation benchmarks, several comparison methods are not current SOTA, which risks depressing the baselines and inflating the apparent gains.

Data / reporting consistency
* The paper does not report intervention frequency per episode, episode time distributions, or success-per-minute; only the one-off latency breakdown is given. Without these, the claim that the system is “real-time most of the time” is untested.

* Real-robot experiments lack variance / confidence intervals; online perturbation is described qualitatively, not as robustness curves (success vs. disturbance level).

Baselines and fairness
* Many simulated and real-robot baselines are not the strongest available today (e.g., more recent VLA or diffusion-action systems, stronger planners). This weakens the case that the proposed loop beats credible state-of-the-art practice.

* Missing a generator-free strong baseline under the same discriminator and a similar compute/time budget. Current gains may be due to extra capacity/data, not the “imagine-and-correct” step itself.

Method assumptions / external validity
* Triggering only at gripper closure is too restrictive; many failures start during approach / alignment. The need for rollback itself suggests the correction often arrives too late.
* The paper does not quantify physics/geometry plausibility of generated clips (contact stability, grasp quality) and does not specify fallbacks for low-confidence or time-out cases—risking confident but unsafe corrections.


Typos and other formatting errors:  

1.	Line 233: “top raw” → “top row.” In the generator description: “as shown in top raw in Figure 3”. 
2.	Case inconsistency in model naming + citation style. The paper uses both “WAN2.1 Wan (2025)” and “wan2.1 (Wan, 2025)”.
3.	BridgeV2 capitalization inconsistency. “Bridgev2 dataset” vs “BridgeV2” elsewhere.
4.	Line 611: Placeholder not replaced. “Figure X illustrates our 7-DoF Xiaomi Robot setup Figure 1b.”

### Questions
1.	Please report per-episode intervention counts (mean/median/95th), episode duration distributions, and success-per-minute vs. baselines. Do your conclusions hold under these metrics?
2.	With the same discriminator and a matched compute/time budget, how does a no-generation variant perform? This isolates the value of video imagination.
3.	What happens if you also trigger at approach / alignment, or learn a risk-based trigger? Can rollback depth be adaptive rather than fixed?
4.	Do you compute any geometry/physics consistency scores for generated clips? What is the fallback policy when generation fails or confidence is low, and how often does that occur?
5.	Can you provide success vs. disturbance magnitude/frequency curves and time-to-recover under interventions?
6.	Can you include current SOTA or widely accepted strong alternatives on at least a subset of tasks in simulation benchmarks or real-robot experiments?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VLA-in-the-Loop, a novel framework that integrates Vision-Language-Action (VLA) models with a composite World Model (WM) to enable real-time policy correction for robotic manipulation.

Traditional VLA systems (e.g., RT-2, CogACT) map visual and textual inputs to robotic actions through imitation learning but lack mechanisms for online correction once errors occur. Meanwhile, World Models have strong predictive abilities but are computationally expensive due to continuous rollouts.

The proposed method bridges these paradigms by introducing an event-triggered, lightweight correction loop:
1) When a high-stakes action (e.g., closing a gripper) is proposed,
2) A discriminative module (a fine-tuned Vision-Language Model like Qwen-VL 2.5) evaluates whether the action is feasible.
3) If failure is predicted, a generative module (a video diffusion model, e.g., WAN2.1) “imagines” a short video showing a successful future trajectory.
4) This imagined trajectory is then fed back to the VLA, guiding it to produce a corrected and more robust action.

This “Propose–Evaluate–Imagine–Correct” loop provides a form of online intervention that corrects policy execution in real time without requiring full-sequence simulation.

### Strengths
S1) Instead of using world models for continuous prediction, the paper redefines them as on-demand correctors, activated only at critical decision points. This is a clever shift from passive supervision to active, event-driven guidance.
S2) The separation into a discriminator (judge) and a generator (imaginer) makes the system modular and interpretable, with clear functionality and training objectives for each part.
S3) Addresses real-world limitations of robotic manipulation pipelines — namely, lack of online correction and high cost of continuous predictive reasoning.

### Weaknesses
W1) The provided description does not clarify how often the correction loop is triggered, its computational overhead, or quantitative improvements across benchmarks. The efficiency vs. accuracy trade-off needs clearer measurement.
W2) The framework assumes that the system can reliably identify “critical moments” (e.g., grasp initiation). Errors in key-frame detection could undermine correction timing.
W3) Most examples focus on grasp correction. It remains uncertain how well the system generalizes to other manipulation types (e.g., pushing, insertion, tool use).
W4) The discriminative model’s success heavily depends on how well Qwen-VL generalizes to unseen grasp scenes. The approach may struggle in low-text or low-visibility scenarios without explicit grounding.

### Questions
How often is the world model triggered in typical tasks, and what is the average latency introduced by the “imagine–correct” loop compared to baseline inference?

What is the false positive/negative rate of the discriminative module in predicting failure? How sensitive is performance to misclassification?

Can the framework extend to tasks that lack clear discrete keyframes (e.g., continuous tool manipulation, deformable-object handling)?

How does the system perform as the size or complexity of the world model increases? Would a single joint model (rather than modular discriminative + generative components) improve stability?

How does this approach compare with recent reflection-based or self-correcting robot architectures (e.g., Phoenix 2025, Reflexion, or LVLM-based reasoning controllers)?

### Soundness
3

### Presentation
3

### Contribution
2
