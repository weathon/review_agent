# HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces HumanoidVerse, a framework for a simulated humanoid robot to perform "long-horizon, multi-object rearrangement tasks." The core contribution is a 4-stage curriculum that distills two separate teacher policies into a single VLA model: one for distill during the initial rearrangement + releasing and the other for the second object rearrangement. First, two expert "teacher" policies are trained using reinforcement learning with privileged state information (e.g., object poses). The key idea is that the second teacher (Stage 3) is specifically trained to handle the diverse, non-standard starting poses left after the first teacher completes its task, whereas the first teacher (Stage 2) is trained to let go and step back after the first object rearrangement. Both teachers are then distilled using DAgger into a single student Vision-Language-Action (VLA) model that operates from only egocentric RGB video and natural language instructions. The authors demonstrate how their framework can effectively tackle sequential humanoid object rearrangement tasks on Isaac Gym.

### Strengths
- **Novel Problem Formulation**: The paper's main strength is tackling continuous, sequential, multi-object manipulation, which is often overlooked in existing works on VLA models for humanoid loco-manipulation. This is a clear and important step beyond the single-object, fixed-start tasks prevalent in prior work (e.g., the HumanVLA baseline).

- **Effective Curriculum**: The 4-stage pipeline is a well-engineered solution. The ablation study (Table 3) clearly validates the authors' design, showing that explicitly training for the transition (Stage 2 for stepping back) and the second task's varied starts (Stage 3 for handling diverse initial configurations) are critical for effective training.

- **New Benchmark for Humanoid Sequential Object Rearrangement**: The creation of a benchmark dataset with 350 sequential two-object tasks is a useful contribution for future research in this area.

### Weaknesses
- **Limited Algorithmic Novelty and Scalability**: The paper's contribution is a highly specific training curriculum, not a new or generalizable algorithm. This curriculum is only demonstrated for a two-step ($N=2$) task, and all task involves sequentially rearranging two different objects. The paper does not provide a clear path for scaling this "dual-teacher" framework to $N>2$ tasks. This seems to imply a non-scalable $N$-teacher pipeline, which contradicts the "long-horizon" claim.

- **Weak Baseline Comparison**: The primary baseline is HumanVLA, a model designed for single-object tasks. As shown in Table 2, it achieves a 0.000% success rate on the sequential task. This is expected and only confirms the new task is harder; it does not validate that the proposed dual-teacher curriculum is a better method for sequential tasks than other plausible baselines (e.g., end-to-end RL and a hierarchical RL, or subsequently their distilled policies).

- **Simulation-Only**: All results are in simulation. The paper does not discuss the significant sim-to-real gap for a complex policy that must simultaneously handle locomotion, manipulation, and perception from RGB data.

- **No Failure Modes**: It would be nice to provide analyses on failures modes of both HumanVLA and HumanoidVerse to justify some design choices or mention potential rooms for improvements.

### Questions
- **Scalability**: How do you propose to scale this framework beyond $N=2$? Does your "dual-teacher" approach not become an "N-teacher" model, and if so, do you consider that a scalable solution for "long-horizon" tasks?

- **Baselines**: Why were more relevant baselines for sequential tasks, such as a monolithic end-to-end teacher or a standard hierarchical RL approach, not included in the comparison?

- **Failure Modes**: What are the common failure modes in simulated tasks? Were there any failures around the transitions between the first and second tasks?

### Soundness
4

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
3

### Summary
HumanoidVerse presents a method that trains VLA models to control simulated humanoid robots to perform multi-object rearrangement. It consists of a multi-stage training process to rearrange the first object, step away from the first object, and rearrange the second object. The authors distill this into a VLA using DAgger, such that it operates using only image and language input (no privileged information) at test-time.

### Strengths
- The success rate on multi-object sequences increases dramatically over the HumanVLA model the authors continue training from.
- The method appropriately drops privileged conditioning, representing a plausible input-output setup for real-world deployment.
- Distilling privileged teachers into a VLA with general world knowledge is an approach that can introduce additional robustness and sim data to VLAs and is not limited to teleoperated data collection.
- The paper is clearly written and easy to follow.

### Weaknesses
- The biggest drawback of the paper is the limited scope of the task, which conflicts with the central claim of the method being "a key step toward robust, general-purpose humanoid agents." The reinforced rewards are very hand-designed with rule-based triggers and remedies to prevent observed, undesirable behavior. This is directionally opposite of scalable learning.
- For the above reason, the improvements are likely isolated to the multi-stage object rearrangement setting.
- There isn't discussion about inference latency, which is a major drawback of using VLA for locomotion tasks and somewhat hidden by the sim-only deployment.

### Questions
- How is the transfer to other simulators and environments? It's interesting to know how much generalization the VLM pretraining affords versus just overfitting to the IsaacGym environments.
- Are there recovery behaviors if the robot falls during a rollout?
- Does the VLA exhibit strong text adherence and vision adherence individually? Does it just pay attention to one of the conditioning signals?

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
This paper introduces HumanoidVerse, a framework for training a simulated humanoid robot to perform two-object rearrangement tasks from egocentric vision and natural-language instructions. The method uses a four-stage teacher–student pipeline: first, reinforcement learning to learn grasp, move and place skills for single object rearrangement; second, a release and step-back stage guided by Teacher 1 to ensure stable placement and clear workspace; third, training to rearrange the second object from canonical initial states with Teacher 2; and fourth, a dual-teacher DAgger distillation stage in which the student switches between teachers according to the task phase. The authors also introduce a dataset with 350 task configurations across four room layouts and conduct extensive evaluation in Isaac Sim. Results show improved success accuracy and lower object-to-goal placement error compared to HumanVLA.

### Strengths
1. The paper is well written, clearly motivated, and explains the proposed approach in an intuitive and accessible manner.
2. Introduces a novel curriculum for multi-object rearrangement with a single humanoid, consisting of three structured teacher training stages followed by dual-teacher student distillation, enabling reliable sequential two-object rearrangement.
3. Presents a new dataset with 350 configurations across four room layouts, providing a valuable benchmark for future work in humanoid rearrangement and embodied learning.
4. Demonstrates strong empirical improvements over HumanVLA, with higher task success accuracy and more precise object placement in goal locations.
5. Includes ablation studies that highlight the contribution of each component in the training pipeline and justify the curriculum design to an extent.

### Weaknesses
1. The pipeline appears tailored specifically for two-object rearrangement. While the curriculum learning approach for staged learning of the teachers for single-object rearrangement, release-and-step-back, and second-object rearrangement is novel, it is unclear how easily this structure generalizes to other long horizon settings. The ablations demonstrate that each stage contributes, but do not clarify why a more skill-based modular approach (e.g., pick, place & step-back, navigate) would not work or scale. Moreover, the switching strategy between teachers is hand-crafted, and there is no sensitivity analysis or investigation of potential failure cases for this designed approach.

2. Limited scalability is demonstrated. Although the title suggests multi-object rearrangement, experiments only cover a two-object case. There is no discussion or demonstration of scaling to more objects (e.g., 3, 5, 10) or to more complex multi-room environments, so the scalability of the approach is unclear.

3. Limited separation between training and evaluation configurations. The 350 two-object setups are split into 700 single-object cases for training the teachers, meaning the student essentially trains in the same configurations where the teachers are supervised. The only major difference between teacher and student seems to be privileged information, and evaluations are conducted on the same scenarios rather than unseen configurations, which weakens claims on generalization.

4. Limited Dataset Analysis: The dataset is not analyzed in terms of scene diversity or complexity. It is unclear how many other objects or receptacles are present in each scene, how cluttered or constrained the workspace is, or how the spatial arrangements vary across configurations. There is no quantitave breakdown of object categories, receptacle types, distractor objects, or proximity constraints that may affect rearrangement difficulty. Without understanding scene distribution and difficulty profiles, it is difficult to assess how challenging or diverse the benchmark is, and whether it meaningfully stresses humanoid rearrangement capabilities.

5. Reproducibility is incomplete. Code and dataset are not provided, and only high-level hyperparameters (e.g., number of epochs and reward structure) are disclosed. Details such as optimization settings, network architectures, dataset generation scripts, rollout procedures, and training infrastructure are omitted, making reproduction challenging.

6. Generalization is limited and under-studied. The method is not tested beyond the training distribution on IsaacSim, with no evaluation on new layouts, unseen objects or different simulators. There is also no discussion of the possibility or challenges of real-world deployment.

7. Evaluation metrics are narrow. In addition to success and placement distance, metrics such as collision frequency, disturbance to already-placed objects, interaction safety, and motion smoothness would offer a more complete view of humanoid performance in rearrangement settings.

8. Minor: Variance, confidence intervals, or multiple-seed results are not reported, making it difficult to assess statistical robustness.

### Questions
1. How do you envision extending this approach beyond two-object rearrangement? Do you expect the current curriculum stages to scale to three or more objects? If not, what modifications would be required?

2. Have you considered benchmarking against a modular skill-based pipeline (e.g., separate pick, place & step-back and navigate policies)? Such a baseline would help clarify whether curriculum learning provides advantages over compositional skills.

3. Current approach uses two teachers, did you consider using a single teacher? If yes, what are the failure cases with single teacher and how dual teacher approach is better?

4. Since the student is evaluated on the same underlying scenarios used for teacher training, how do you ensure that the student does not overfit to the training layouts and object placements? Are there results on held-out task configurations or unseen object and room layouts?

5. Can you provide statistics or analysis of dataset diversity: number and type of distractor objects, receptacle categories, clutter level, and spatial variation? How varied are object pairings and placements overall?

6. Do you foresee challenges in scaling the dataset or the approach to more objects or multi-room scenes? Are there preliminary experiments or insights on scaling behavior?

7. Have you explored domain randomization, perception noise, or other techniques to prepare for real-world deployment? What are the main challenges anticipated for sim-to-real transfer?

8. When do you plan to release the code, dataset and model checkpoints? Could you also share additional implementation details such as optimizer settings, architectures, and training infrastructure?

9. Did you measure safety or interaction-related metrics such as collision frequency, object disturbance, or motion smoothness? If not, would you consider including such evaluations?

10. Could you please report multiple seeds, variance, or confidence intervals to strengthen statistical reliability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces HumanoidVerse, a framework for vision-language guided humanoid control that enables physically simulated robots to perform multi-object rearrangement tasks. The approach uses a multi-stage curriculum learning pipeline with dual-teacher distillation, where teacher policies trained with privileged state information are distilled into a student VLA model that inputs egocentric RGB images and natural language instructions. A dataset of 350 tasks across four room layouts is constructed, each involving sequential manipulation of two objects. Experiments demonstrate significant improvements over the HumanVLA baseline, particularly on second-object manipulation.

### Strengths
- The paper addresses the limitation in existing humanoid manipulation work of unable to perform sequential, multi-object tasks without environment resets.
- The multi-stage curriculum learning pipeline is well-motivated for the multi-stage rearrangement task, with stage 2 specifically addressing object release and retreat behaviors, and stage 3 handling diverse initial configurations. The ablation study validates the importance of each component.
- The results show significant gains over HumanVLA, particularly on Success 2 metrics, demonstrating the effectiveness of the proposed multi-stage training approach.

### Weaknesses
- Despite claiming to address "multi-object rearrangement", the system only demonstrate two object rearrangement. No evaluation is provided on how the current distilled model would perform on 3+ object scenarios.
- There lacks analysis of when and why the teacher and student policies fail. What are some common failure modes?
- Figure 5 in appendix shows many egocentric views are heavily occluded, especially when holding large objects. How does the policy determine where to place objects when visual information is limited? Is the model potentially overfitting to proprioceptive signals rather than learning robust vision-based reasoning?
- There lacks discussion of how this could transfer to real humanoid robots. What are the main bottlenecks? (sim2real gap, stable whole-body control, tracking accuracy etc.)

### Questions
- Can the approach directly extend to 3+ objects, or does it require training additional teacher models for each subsequent object?
- What is the representation of robot actions and proprioceptive state (dimensionality, joint/cartesian, coordinate frames, absolute/relative)?
- In Supplementary Figure 5, many views are occluded. Can you provide ablation studies showing performance with/without proprioceptive information to clarify the role of vision vs. proprioception?
- How sensitive is the dual-teacher switching mechanism (Algorithm 3) to the hand-tuned thresholds?

### Soundness
3

### Presentation
3

### Contribution
3
