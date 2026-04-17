# Self-Improving Vision-Language-Action Models with Data Generation via Residual RL

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Supervised fine-tuning (SFT) has become the de facto post-training strategy for large vision-language-action (VLA) models, but its reliance on costly human demonstrations limits scalability and generalization. We propose Probe, Learn, Distill (PLD), a plug-and-play framework that improves VLAs through residual reinforcement learning and distribution-aware data collection. In Stage 1 (specialist acquisition), we freeze the VLA backbone and train lightweight residual actors via off-policy RL. These specialists take over in states where the base policy fails, thereby probing failure regions of the generalist. In Stage 2 (data collection), we employ a hybrid rollout scheme that biases residual interventions toward states frequently visited by the base policy, aligning collected trajectories with the generalist’s deployment distribution while capturing recovery behaviors. In Stage 3 (fine-tuning), these curated trajectories are distilled back into the generalist with standard SFT, applicable to both flow-matching and autoregressive heads. We evaluate PLD across diverse settings: it achieves a near-saturated 99% task success rate on the LIBERO benchmark, delivers over 50% performance gains in SimplerEnv, and demonstrates practicality on real-world Franka arm manipulation tasks. We further provide ablations showing that residual policy probing and distribution-aware replay are key to collecting deployment-aligned data that improves VLAs’ capabilities on both seen and unseen tasks. Our results demonstrate that RL-generated, policy-aligned data can surpass teleoperation-only demonstrations, offering a scalable path toward self-improving VLA models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PLD (Probe–Learn–Distill), a three-stage self-improvement framework for robotic foundation models. PLD enables a VLA model to autonomously refine itself without relying on human demonstrations. It first learns task-specific residual policies via reinforcement learning, then uses a mixed strategy to collect diverse, high-quality data, and finally distills this knowledge back into the base model through supervised fine-tuning. The method achieves near-perfect success rates in both simulation (LIBERO, SimplerEnv) and real-world experiments on the Franka Panda arm, demonstrating strong efficiency, generalization, and practicality.

### Strengths
PLD provides a practical approach for VLA models to achieve self-improvement without additional human demonstrations. The data produced by PLD not only aligns with the behavioral distribution of the base model, but also includes recovery behaviors that are typically scarce in human demonstration datasets. Through experiments in both simulated environments and real-world robotic settings, the paper shows that fine-tuning with such self-generated data can effectively enhance the success rate, robustness, and generalization ability of VLA models. This offers a promising solution to the long-standing challenge of high data collection costs in robot learning.

### Weaknesses
1. The evaluation focuses mainly on short-horizon, simple manipulation tasks. The approach has not been tested on long-horizon or temporally dependent tasks.
2. The human data only includes successful trajectories, while PLD’s automatically collected data includes failure and recovery cases, which clearly increase diversity. This makes it hard to determine whether the improvement comes from PLD’s three-stage structure or simply from having more varied data. Including ablation studies that remove failure/recovery data from PLD would make the comparison more fair and convincing.

### Questions
1. How would PLD handle long-horizon or multi-stage tasks requiring sequential planning?
2. Could the authors provide ablation studies like removing failure/recovery data from PLD?

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
This paper proposes PLD, a three-stage post-training framework for vision-language-action (VLA) models that reduces dependence on human demonstrations. Stage 1 freezes the VLA and trains lightweight residual RL specialists to improve recovery from failure states. Stage 2 uses a hybrid rollout that starts with the base policy and then hands control to the residual policy, collecting realistic recovery trajectories. Stage 3 distills this data back into the base model via supervised fine-tuning. Experiments on LIBERO, SimplerEnv, and a real Franka arm show strong success rates and generalization, demonstrating a scalable path to improving robotic foundation models with minimal human supervision.

### Strengths
- Presents a novel and practical three-stage framework that combines RL and SFT for post-training VLAs without relying on costly human demonstrations.

- Effectively addresses scalability challenges in robot learning by improving data diversity (failure recovery trajectories) and reducing dependence on human demonstration data.

- Provides robust evidence through extensive experiments on LIBERO, SimplerEnv, real-world Franka setups, and systematic ablation studies.

- The paper is well-written and clearly organized, making the methodology and key insights easy to understand.

- Clearly identifies the core challenges preventing direct transfer of LLM post-training successes to VLA models, providing both conceptual and empirical justification for the proposed solution.

### Weaknesses
- The real-world experiments focus on short-horizon tabletop manipulation where the base policy is already strong. It remains unclear how PLD performs when initial success rates are low or when recovery requires multi-step planning. Including a few such cases or a discussion of failure modes would better support claims of scalability.

- When applying PLD in the real world, the trade-off between safe operation and sufficient state coverage raises concerns about the method’s ability to handle more challenging, long-horizon, or highly exploratory tasks where recovery behaviors are harder to discover.

- The approach trains multiple task-specific residual RL actors, but the paper does not clarify how many specialists are required as task counts grow, or whether skills can be shared across tasks with related dynamics. Providing guidelines or experiments on task grouping would strengthen the method’s practicality.

### Questions
- The real-world results depend on a set of 200 teleoperated trajectories to initialize the base policy. How does the performance of PLD scale with fewer (or noisier) demonstrations, and would the method still succeed if starting from a weaker base policy or no demonstrations at all?

- PLD trains residual experts for multiple tasks, and then distills them back into the base VLA model. Could the authors elaborate on why this distillation step (collect trajectories) is necessary compared to directly deploying the VLA + the residual experts? What specific limitations does PLD address that simple residual policies cannot?

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
5

### Summary
A paper for enriching data in finetuning VLA using residual RL. This framework utilize residual RL to generate more diverse and successful trajectory for tuning VLA. To evaluate the performance, this paper conduct comprehensive experiments on the libero benchmark and two real-world experiments.

### Strengths
1. Using residual RL to enrich the data for VLA finetuning is promising.

2. The proposed pipeline is well-designed, which includes base policy probing, warm-start, success classifier, and SFT for policy distillation.

3. The experiments in libero benchmark and real-world RL is comprehensive.

### Weaknesses
1. Using residual RL to finetune and resolve different gaps is not a pretty novel idea, and this paper didn't disscuss those related work accross different areas in details. For example, residual RL has been used in data efficient learning [1], bridge human2robot embodiment gap [2, 3], real2sim2real transferring for data scaling [4], peg insertion [2, 5], and dexterous manipulations [3, 4]. Disscussing those related work and highlight the contribution might be important.

2. Only binary reward has been used for reward shaping, where it will limit the learning efficiency. Althought it's hard to design dense reward for visual policy or in the real world, there are some related work using VLM [6], generated videos [7], and object-centric representations [8] to do reward shaping.

3. Compared with offline RL or SFT, real-world RL required more human effort to reset and might have unexpected behaviors for some challenging tasks.

4. The paper shows that the control frequency is 20 Hz. For pi0, it can reach high control frequency because of its open-loop action chunking, and its inference frequency is low. Is the residual RL tuning each action of the action chunking and resulting in a close-loop control? However, after SFT, the VLA policy still should be low frequency inference for open-loop action chunking but not close-loop control? For more challenging task, it might require close-loop policy.

5. In the previous real-world paper HIL-SERL [9], they point out that using residual RL to refine a bad BC base policy will result in bad performance. Could you show more insight about this problem? To this end, together with the concern 4, I would be worried about this method's performance in more challenging task, where the policy requires close-loop control, and the VLA base policy might have bad performance.

6. No figure to show the entire pipeline.

[1]. Haldar et al., Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations RSS 2023

[2]. Yu et al., MimicTouch: Leveraging Multi-modal Human Tactile Demonstrations for Contact-rich Manipulation, CoRL 2024

[3]. Guzey et al., HuDOR: Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards, ICRA 2025

[4]. Wan et al., LodeStar Icon LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations, CoRL 2025

[5]. Ankile et al., From Imitation to Refinement – Residual RL for Precise Visual Assembly, ICRA 2025

[6]. Wang et al., RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback, ICML 2024

[7]. Huang et al., Diffusion Reward: Learning Rewards via Conditional Video Diffusion, ECCV 2024

[8]. Yu et al., GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning, ICCV 2025

[9]. Luo et al., Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning, Science Robotics

### Questions
1. Residual policy can be different policy architecture other than base policy. In simulator, the residual policy can be simple state-based policy, which might be trained more efficiently. Why still using VLA architecture?

2. For the policy warmstart with expert data, the residual action will always be 0 action. Could you give some more analysis on why this kind of warm start can work? It only warm start the critic or also warm start the residual actor?

3. For those baselins like WSRL, the policies are learned from scratch but not doing residual rl base on the pretrained policy, right? I concern that it might not be a fair comparison.

4. WSRL only uses online data but not offline data after warm starting the Q and show that it's better than off-policy learning. Since this paper utilize similar strategy to warmstart Q using only base policy, why still use offline data? 

5. Why don't choose offline residual RL [2] but online RL?

6. I might not an expert in VLM tuning. Since this method still need to do SFT for tuning the VLA, why it requires less computation resource than doing online RL for tuning the VLA directly?

[1]. Xu et al., Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections, NeurIPS 2025

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a three-stage self-improving framework named PLD (Probe, Learn, Distill) to provide high-quality data to finetune Vision-Language-Action (VLA) models. Its core idea is to freeze the VLA base model and train lightweight residual reinforcement learning (RL) policies to probe the failure regions of the base model; then adopt a hybrid data collection scheme of to align residual interventions with the deployment distribution of the base policy; and finally distill the collected high-quality trajectories back into the base model via SFT. Experiments validate the effectiveness of PLD on the LIBERO and SimplerEnv simulation benchmarks as well as real-world robotic tasks.

### Strengths
1. This paper addresses two core pain points of VLA models in SFT: "data dependence" and "distribution mismatch". PLD combines residual RL with distribution-aware data collection. It avoids the high resource consumption of directly fine-tuning large VLA models and solves the poor generalization issue of pure RL expert data, presenting a novel and logically consistent approach. 
2. The effectiveness of PLD is well demonstrated through multi-environment (LIBERO, SimplerEnv, real robots), multi-model ($\pi_0$, OpenVLA), and multi-baseline comparisons. Additionally, PLD demonstrates better generalization ability than human data in zero-shot and few-shot experimental tasks. 
3. Quantitative analyses are conducted on the impact of key parameters, such as the action scale of the residual policy, the probing ratio of the hybrid rollout strategy, and the pre-training method for the Q-network. This provides reusable parameter selection guidelines for subsequent research.

### Weaknesses
1.  Direct comparisons with recent residual RL methods (e.g., EXPO[1], ResiP[2]) are missing. It is recommended to supplement comparative experiments or discussions on this aspect. 
2. Current tasks (e.g., object grasping, peg insertion) are relatively simple. It is suggested to validate the method on more complex long-horizon tasks such as cloth folding. 
3. The paper mentions that PLD trains lightweight residual policies but does not discuss costs such as training time and GPU memory usage. It is recommended to supplement a computational cost table to clarify PLD’s advantages in "resource efficiency", especially its scalability in multi-task training scenarios.

[1] Dong P, Li Q, Sadigh D, et al. Expo: Stable reinforcement learning with expressive policies[J]. arXiv preprint, 2025.

[2] Ankile L, Simeonov A, Shenfeld I, et al. From imitation to refinement-residual rl for precise assembly[C]//ICRA, 2025.

### Questions
The paper states that PLD supports multi-task training. In multi-task scenarios, are residual policies trained independently for each task, or do they share partial parameters to improve efficiency? If trained independently, will PLD’s storage costs (e.g., parameters of residual policies for each task) increase significantly as the number of tasks grows? Are there any parameter sharing or model compression schemes available?

### Soundness
3

### Presentation
4

### Contribution
3
