# AutoDrive-R²: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Vision–Language–Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R², a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR²-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method based on VLAs for trajectory prediction in autonomous driving. The proposed method consists of a two-stage process: 1) a chain-of-thought dataset based on NuScenes used for fine-tuning of an existing VLA (the paper uses Qwen 2.5 VL) so that it learns to "think" through the trajectory generation; 2) a reinforcement learning stage based on GRPO to align the model's predictions with physical trajectory properties (e.g. position) and model output format. The method is evaluated on the NuScenes and Waymo datasets and compared to other methods and ablated baselines, showing significant gains in positional accurate of predicted trajectories.

### Strengths
- The paper builds on recent progress in using VLAs for trajectory prediction, improving model accuracy.
- The proposed dataset is novel and could be broadly useful beyond this particular work.
- The empirical evaluation shows significant gains.
- The writing is clear (for me).

### Weaknesses
Overall I enjoyed reading this paper, though I noticed a couple of issues:

1) Limited metrics used in evaluation: the evaluation only uses a L2-based metric on predicted vs true positions. But it does not measure other important aspects, such as feasibility (based on vehicle model) or rule/constraint violations of generated trajectories. Prior literature on trajectory prediction has developed a number of metrics to provide a more complete picture (see next point).

2) Limited discussion of work prior to VLA: related to above comment, I find that the paper lacks discussion of works prior to the recent VLA-based research. There is a large body of work on trajectory prediction in autonomous driving, and it would help readers to see the relationship with prior works. For example, prior works have developed evaluation metrics for trajectory prediction e.g. [1,2]; and there is also prior work on explainable AI in autonomous driving e.g. [3,4], which is one of the stated motivations for the proposed CoT fine-tuning method in this paper.

[1] https://arxiv.org/abs/2210.06106 (RAL 2023)

[2] https://arxiv.org/abs/2203.08251 (IROS 2022)

[3] https://arxiv.org/abs/2402.10086 (T-ITS 2024)

[4] https://arxiv.org/abs/2002.02277 (ICRA 2021)

I think the paper could be strengthened by addressing these two aspects.

### Questions
Q1: The "self-correction" examples presented in appendix A.3 are not clear to me. In what sense does the red text in Fig4 show an "Aha moment"? I can't tell what error was detected and how it was corrected.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AutoDrive-R^2, a novel VLA framework designed to enhance the reasoning and self-reflection capabilities of autonomous driving systems, addressing the limitations of existing methods such as physically infeasible trajectory generation and inadequate reasoning for complex scenarios. The framework adopts a two-stage training approach: in the first stage, a CoT dataset named nuScenesR^2-6K (with 6,000 image-trajectory pairs) is constructed for SFT, which guides the model through a four-step logical chain to build cognitive connections between input information and output trajectories. In the second stage, a physics-grounded reward framework integrated with GRPO is employed for RL, incorporating spatial alignment, vehicle dynamics, and temporal smoothness constraints to ensure trajectory feasibility. Experimental results on nuScenes and Waymo datasets demonstrate that AutoDrive-R^2 achieves state-of-the-art  performance and robust zero-shot generalization.

### Strengths
nuScenesR^2-6K is the first dataset in autonomous driving that integrates self-reflection for validation, providing detailed reasoning chains to bridge input information and output trajectories, which enhances model interpretability.

The combination of SFT with structured CoT reasoning and RL with physics-grounded rewards effectively addresses both reasoning inadequacy and physical infeasibility of trajectories

AutoDrive-R² outperforms state-of-the-art methods on nuScenes and Waymo datasets, with significant error reductions and robust zero-shot capabilities, validating its effectiveness and generality.

### Weaknesses
The RL stage with GRPO requires generating multiple candidate responses, which may introduce high computational overhead, but the manuscript does not discuss inference speed or real-time deployment feasibility.

Although the paper emphasizes the role of self-reflection in the four-step reasoning chain, it does not clearly explain how the model corrects inconsistent trajectories during self-reflection. For example, there is no detailed description of the decision rules (e.g., threshold for determining trajectory inconsistency) or the specific adjustment strategies (e.g., how to modify velocity or steering angle) adopted in the self-reflection stage.

This paper does not analyze the computational overhead (e.g., inference time per trajectory, GPU memory usage) or compare it with lightweight baseline methods. This may restrict the practical deployment on edge devices with limited computing resources.

Table 2 appears to be outside the page content

### Questions
For the dataset: How were the manual annotations of the CoT reasoning process validated? How to ensure the consistency of reasoning steps? Additionally, since the dataset is derived from nuScenes, does it inherit the scene bias of the original dataset (e.g., urban road dominance), and if so, how does this affect the model's generalization to non-urban scenarios (e.g., highways or rural roads)?

The paper sets all weight coefficients (λ_pos, λ_ste, λ_vel, λ_tem) to 1 in experiments. Have you tested the impact of different weight combinations on performance? Besides, have you analyzed the impact of GRPO hyperparameters (e.g., number of candidate responses, beta in KL-divergence) on trajectory prediction accuracy and training stability?

For the zero-shot performance on Waymo, this paper attributes the excellent zero-shot generalization to the model's structured reasoning capabilities, but it does not compare with other methods that also claim zero-shot adaptation. Could you explain why your method has a more significant zero-shot advantage, and whether the dataset contains scene features that are common to both nuScenes and Waymo?

### Soundness
3

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
This paper presents AutoDrive-R², a novel Vision–Language–Action (VLA) framework for autonomous driving that aims to enhance both reasoning interpretability and trajectory feasibility. Supervised Fine-Tuning (SFT) using a new Chain-of-Thought dataset (nuScenesR²-6K) that instills structured reasoning and self-reflection. Reinforcement Learning (RL) using Group Relative Policy Optimization (GRPO) with a physics-grounded reward framework that integrates spatial alignment, steering dynamics, and temporal smoothness, ensuring physically feasible trajectories. The method is evaluated on nuScenes and Waymo benchmarks, achieving state-of-the-art (SOTA) performance with substantial reductions in L2 trajectory errors compared to EMMA+ and DriveVLM, and demonstrating strong zero-shot generalization. The work positions itself at the intersection of multimodal reasoning and embodied intelligence, extending recent reasoning-incentivized LLM/RL techniques (e.g., DeepSeek-R1) to the autonomous driving domain.

### Strengths
1. Introduces reasoning-incentivized CoT learning and self-reflection for VLA models — a clear conceptual advance over purely data-driven planners.
2. Sound integration of SFT and GRPO with well-formulated physics-aware rewards; rigorous ablations confirm design choices.
3. Large quantitative gains on nuScenes (Avg L2 error 0.19 m vs 0.29 m for EMMA+) and Waymo (0.20 m vs 0.30 m), plus zero-shot transferability.
4. The four-stage reasoning framework (Observation–Calculation–Logic–Reflection) is both intuitive and interpretable.

### Weaknesses
1. Although the nuScenesR²-6K dataset is novel, 6k samples may not capture full traffic complexity. The manual synthesis with Qwen2.5-VL could introduce distributional bias.
2. Experiments are open-loop trajectory predictions; it remains unclear how the system performs in end-to-end control or real-time deployment.
3. The GRPO ablations demonstrate quantitative gains, but lack qualitative discussion on how each reward term alters trajectory smoothness or safety metrics.

### Questions
1. How does AutoDrive-R² perform under closed-loop simulation or in real-world deployment (e.g., CARLA)? Would the physics-grounded reward suffice without explicit simulator feedback?
2. How were the CoT reasoning steps validated for correctness or consistency during dataset construction? Were human checks performed post-Qwen2.5-VL synthesis?
3. Could you show qualitative examples where AutoDrive-R² fails (e.g., multi-agent interactions or occlusions) and explain whether the reflection step mitigates or worsens these?

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
4

### Summary
The application of VLA models in the field of autonomous driving has been attracting increasing attention. Among their capabilities, reasoning and self-reflection play a pivotal role. This paper trained a VLA model using a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to equip it with enhanced reasoning and reflective abilities. Specifically, This paper introduces a reasoning dataset that incorporates a Chain-of-Thought (CoT) process, along with a physics-compliant reward framework to guide model training.

### Strengths
The authors construct a dataset and a reward framework grounded in practical problems; overall, the work is solid and technically sound. The paper makes clear contributions in dataset design and in the formulation of the reward function.

### Weaknesses
While the authors identify salient issues with VLA models in autonomous driving and present technically solid efforts to address them, the overall solution exhibits limited novelty: it defines conventional tasks, applies standard approaches, and consequently yields expected results. The paper would benefit from a clearer articulation of the challenges encountered during the design of the solution and the insights gleaned from addressing them.

### Questions
- The paper primarily trains a VLA model to predict future vehicle trajectories. What advantages does this approach offer over rule‑based prediction and purely data‑driven prediction methods? VLA models, trained with textual data, indeed have strengths in language understanding and reasoning, but they tend to be limited in precise numerical computation. Would employing a VLA model for fuzzy decision‑making be more appropriate?
- The paper uses L2 error to indirectly assess the performance improvement of the trained model. This provides some evidence of data and reward effectiveness; however, more direct methods demonstrating gains in reasoning, computation, and reflection would make the contribution clearer.

### Soundness
2

### Presentation
3

### Contribution
2
