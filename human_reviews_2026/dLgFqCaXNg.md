# LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Vision–language–action (VLA) models have demonstrated strong semantic understanding and zero-shot generalization, yet most existing systems assume an accurate low-level controller with hand-crafted action "vocabulary" such as end-effector pose or root velocity. This assumption confines prior work to quasi-static tasks and precludes the agile, whole-body behaviors required by humanoid whole-body control (WBC) tasks. To capture this gap in the literature, we start by introducing the first sim-to-real-ready, vision-language, closed-loop benchmark for humanoid WBC, comprising over 150 tasks from 10 categories. We then propose LeVERB: Latent Vision-Language-Encoded Robot Behavior, a hierarchical latent instruction-following framework for humanoid vision-language WBC, the first of its kind. At the top level, a vision–language policy learns a latent action vocabulary from synthetically rendered kinematic demonstrations; at the low level, a reinforcement-learned WBC policy consumes these latent verbs to generate dynamics-level commands. In our benchmark, LeVERB attains a 58.5 \% success rate, outperforming naive hierarchical VLA implementation by 7.8 times, and can be zero-shot deployed in real.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper Introduces a dual-process architecture combining a high-level vision-language policy (System 2) and a low-level reactive controller (System 1) to enable humanoid robots to execute complex, instruction-driven whole-body motions. Uses a CVAE-based latent space to align perception, language, and action.

LeVERB develops a scalable pipeline to generate photorealistic humanoid motion data via retargeted human MoCap in diverse simulated scenes, paired with language instructions, enabling training and closed-loop evaluation of VLA-driven whole-body control.

LeVERB demonstrates robust vocabulary generalization and spatial reasoning in simulation and on real humanoid hardware, outperforming ablated baselines and achieving the first zero-shot sim-to-real results for vision-language-driven humanoid whole-body control.

### Strengths
LeVERB-VL with a low-level LeVERB-A enables humanoid robots to execute complex whole-body motions while maintaining real-time, closed-loop control. The use of a CVAE-based latent space allows semantic alignment between vision, language, and actions, which is a significant advancement over prior methods that rely on explicit low-dimensional commands.

The creation of LeVERB-Bench, a scalable, photorealistic, and diverse dataset for humanoid whole-body control, addresses the lack of large-scale, visually grounded training data.

### Weaknesses
About motivation: what is the main theme of the paper? dataset contribution, modeling, humanoid-environment interaction, sim2real? All of the above topic seems to be part of the contribution, making each of them solved partially, and making the research depth of each topic inefficient? A more proper way of presentation would be highlighting one of them

About dataset: as mention in the paper, some other works e.g. [41] e.t.c are just lack of a good physics engine or rendering methods. Applying IsaacSim is not the major way of making this work different than others. What makes your data collection method distinct from others? How do you quantitatively evaluate your data generation method from others e.g. regular data teleoperation, homie for humanoid, e.t.c. 

About tasks. It seemed that the major part of the tasks focuses on navigation and target following. What are the benefits of using LeVERB? Since system2-system1 structure is pretty a standard in VLA for so long, what make LeVERB model different from others?

About projects: there is no dataset or project releasing and maintaining plan. This dataset focuses only on unitree G1, if the project and methodology cannot be applied other robots via open soucing, I could not see a huge impact on this work.

### Questions
See weaknesses

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LeVERB, the first hierarchical vision-language-action model for humanoid whole-body control that uses a learned latent action space as the interface between a high-level vision-language policy and a low-level dynamics controller. This paper also proposed a new photorealistic sim-to-real benchmark for humanoid VLA tasks.

### Strengths
1. While hierarchical VLA models and humanoid control exist separately, this is the first work to formulate and demonstrate vision-language-driven whole-body control for humanoids using a latent interface.
2. The learned latent verb vocabulary directly removes the key limitation of prior hierarchical VLAs, which was their reliance on an inflexible, hand-crafted "action vocabulary" (e.g., base velocities).

### Weaknesses
1. The benchmark and experiments focus almost exclusively on locomotion and posture (navigation, sitting), omitting manipulation tasks (e.g., picking, pushing)
2. The celebrated "zero-shot" real-world deployment uses open-loop replay of latent plans generated in simulation. The high-level vision-language policy does not run closed-loop on the real robot, weakening the claim of a fully closed-loop system.
3. Lacks comparison to a strong, non-latent baseline (e.g., a hierarchical VLA that predicts explicit body keyframes) to conclusively prove the advantage of the learned latent space.
4. The paper reports success rates but provides no qualitative analysis of how or why the model fails in ~40% of trials, leaving its robustness and limitations unclear.

### Questions
1. Can the latent verbs (z_t) for the real-world demo be generated by running LeVERB-VL on the real robot's live camera feed?
2. With an overall success rate of 58.5%, what are the most common failure modes for LeVERB in simulation? Does the robot typically fail due to dynamics (falling), perceptual errors (misidentifying the target), or a disconnect between the latent plan and the low-level execution?
3. A key claim is that a learned latent action space is superior to an explicit one. To isolate this benefit, did you consider a baseline where your high-level policy predicts explicit, kinematically feasible whole-body keyframes instead of a latent vector z_t, while keeping the low-level policy and training data identical?

Please also refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
LeVERB is a latent vision–language hierarchical model for humanoid whole-body control. A CVAE-based “System-2” maps RGB + instruction into discrete verbs; a low-level RL controller tracks those instructions at 200 Hz conmtrol. Evaluation is conducted on a 150-task simulated suite and only two qualitative runs on a Unitree G1.

## Main Results

- Benchmark evaluation on 20 unseen indoor scenes × 10 task families. authors report 58.5 % success, latent-free baseline to 7.5 %.

- Module ablations (no discriminator, no kinematic encoder, no latent sampling) show progressive drops up to 33%. 

- Real world robot roll outs for multi-heading navigation

### Strengths
## Strengths


- Sim‑to‑real benchmark : Synthetic dataset of 150+ tasks (154 trajectories × 100 augmentations ≈ 17 h) grounded in photorealistic scenes and diversified camera views.

- A hierarchical system 2 keeps vision–language inference off the real-time control loop. There by modularizing control between systems, which is clean.

- Comprehensive ablations (no discriminator, no kinematics encoder, latent sampling, etc.) clearly justify architectural choices.

- Zero-shot transfer from Isaac Sim to hardware, though only demonstrated qualitatively.

### Weaknesses
## Weaknesses

- Sim-to-Real evidence : purely qualitative, no task-level success rate, latency, or contact statistics; real-world failure cases unreported.

- Missing strong baselines : recent controllers such as ExBody 2, OmniH20 and world-model planner Nicklas Puppeteer are not compared.

- Portability unclear : System‑1 is specialized to a single Unitree G1 morphology. modularity claim would benefit from multi‑platform evidence.

- Scalability of latent vocabulary unclear : The latent space is fixed-dimensional and trained on ~150 motion primitives, which may be limiting factor for complex behaviors.

### Questions
Please read weakness section.

- Could the authors supply explicit sim-to-real success rates and command latency numbers?

- How do ExBody 2, OmniH20, Puppeteer, other platforms performance compared to LeVerb model in generic humanoid control tasks?

- Can the same System-2 policy been ported to a different robot without retraining System-1? There by proving modularity.

- Open‑loop fidelity: Can you provide teacher‑forced vs free‑run verb prediction task accuracy at 1 / 5 / 10 / 25 / 50‑step horizons, include divergence curves and rollout videos showing drift.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LeVERB, a hierarchical vision-language-action (VLA) model for humanoid whole-body control. It introduces a new benchmark, LeVERB-Bench, which provides photorealistic humanoid motion data generated from retargeted MoCap and synthetic rendering. The system uses a CVAE-based latent interface to connect a high-level vision-language policy with a low-level whole-body control policy. Experiments show that LeVERB achieves higher success rates than naive hierarchical VLAs and can be deployed zero-shot on a Unitree G1 humanoid.

### Strengths
- Proposes a **modular, hierarchical structure** (System 1 and System 2) that could improve inference efficiency and decouple vision-language reasoning from dynamics control.
- Introduces a **sim-to-real-ready benchmark** with photorealistic rendering and procedural scene randomization.
- Demonstrates **zero-shot transfer** of some whole-body behaviors from simulation to real-world deployment.

### Weaknesses
- Limited Benchmark Diversity: Although the paper claims to introduce a comprehensive benchmark for humanoid WBC, the majority of tasks are navigation-like—e.g., “walk to”, “navigate around”, or “reach”. These are essentially vision-language navigation tasks, not complex whole-body manipulation. Hence, the benchmark provides limited insights into the method’s generality for rich human-object interactions (e.g., picking up, placing, or coordinated manipulation).

- Restricted Demonstration Source: The dataset heavily relies on replayed, retargeted MoCap motions that are feasible only for navigation or simple pose transitions. This approach cannot generate demonstrations involving precise contact-rich interactions such as sitting naturally or grasping objects, which are key challenges for humanoid control.

- Methodological Novelty Is Marginal:
The proposed model architecture largely extends LangWBC (Shao et al., 2025), which already used a language-conditioned CVAE for whole-body control. LeVERB adds vision input and a discriminator for data-source alignment but otherwise follows a similar framework. The paper’s claimed conceptual contribution—the “latent verb” interface—is conceptually close to prior CVAE latent representations and does not introduce a clearly new mechanism or insight.

- Real-World Evaluation Is Minimal:
The paper only shows a few qualitative real-world results (sitting and navigation), which are insufficient to substantiate claims of “zero-shot sim-to-real readiness.” The experiments lack baseline comparisons or quantitative performance in real settings.

### Questions
- How well would the proposed latent space generalize to tasks requiring contact-rich manipulation?

- Since LeVERB’s benchmark tasks are mostly navigation-like, how meaningful is the reported improvement in success rate for demonstrating “whole-body control”?

- Could the approach handle more complex compositional language commands?

### Soundness
3

### Presentation
3

### Contribution
2
