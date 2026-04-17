# RIFT: Group-Relative RL Fine-Tuning for Realistic and Controllable Traffic Simulation

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Achieving both realism and controllability in closed-loop traffic simulation remains a key challenge in autonomous driving. Dataset-based methods reproduce realistic trajectories but suffer from \covariate shift in closed-loop deployment, compounded by simplified dynamics models that further reduce reliability. Conversely, physics-based simulation methods enhance reliable and controllable closed-loop interactions but often lack expert demonstrations, compromising realism. To address these challenges, we introduce a dual-stage AV-centric simulation framework that conducts imitation learning pre-training in a data-driven simulator to capture trajectory-level realism and route-level controllability, followed by reinforcement learning fine-tuning in a physics-based simulator to enhance style-level controllability and mitigate covariate shift. In the fine-tuning stage, we propose RIFT, a novel group-relative RL fine-tuning strategy that evaluates all candidate modalities through group-relative formulation and employs a surrogate objective for stable optimization, enhancing style-level controllability and mitigating covariate shift while preserving the trajectory-level realism and route-level controllability inherited from IL pre-training. Extensive experiments demonstrate that RIFT improves realism and controllability in traffic simulation while simultaneously exposing the limitations of modern AV systems in closed-loop evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new method to indentify the key background vehicle and improving the vehicle's trajectory realism by group-relative RL finetuning. The method preverses the trajectory-level realism and multi-modality. Experiments demonstrates RIFT can improve realism and controllability in traffic simulation.

### Strengths
1.The appendix is detailed and the supplementary videos is comprehensive and vivid.
2. The experiment includes extensive prior works with reasonable realism metrics.
3. The paper is well written and easy to follow.

### Weaknesses
1. The method to indentify the CBV is problematic because it cannot indentify the vehicle in the intersection which has different routes with the ego vehicle but has large collision risk. It is demonstrated in the videos.
2. The method to evaluate the realism in the simple CARLA environment where all BV behaves convervatively. Consider evaluate and compare in the Waymo Sim Agent benchmark where groud-truth is provided for better evaulation of realism.

### Questions
1. Why does the other method except PDM-lite in table 2 not report the BR metrics?
2. How do you refine the all head? What is the loss for finetuning trajectory generation head?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes **RIFT**, a closed-loop reinforcement learning fine-tuning framework for driving models. The method pre-trains on NuPlan and fine-tunes using **GRPO** (a PPO variant without KL-regularization, with dual clipping) in the CARLA simulator. The goal is to improve controllability and realism in closed-loop driving.

Although the authors provide many experiments, I found the terminology and takeaways unclear (see weaknesses below), which makes it hard to judge the effectiveness of RL-finetuning in this work.

### Strengths
- The qualitative results are comprehensive and good for understanding the performance
- I appreciate authors provide several baselines, but the key baselines, such as CAT-K or Waymo Sim Agent Challenge would be more insightful.

### Weaknesses
- This works proposes a different metrics compared to Waymo Sim Agent Challenge, I suggest the authors either take 1-2 baselines from Sim Agent Challenge and evaluate their settings or, simply adapt RIFT for Waymo Sim Agent Challenge. Currently is hard for reviewers to understand what are the strength and limitations of RIFT just by looking at  table numbers.
- For AV Evaluation, it is hard to draw insights from Table 2 since there are two factors (Sim Agents and different planners). For example, RIFT might be too reactive that gives wrong esimate or the best performance for AV planners.

### Questions
- What is the main advantage of using CARLA/Meta Drive as a fine-tuning simulator? What new aspects does it provide? More accurate dynamics? These simulators also provide an approximation of dynamics, and traffic simulation usually focuses on high-level behavior instead of low-level control.

- There are many works that focus on GRPO fine-tuning of traffic models, e.g., [1][2]. However, the main advantage of the RIFT in this work remains unclear. For example, the authors can consider: 1) What new aspect does it unlock for evaluating planning algorithms?

[1] Wang M., Wang J., Ye T., Chen J., Yu K. (2025). *Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving.* CoRL.
[2] Ahmadi E., Schofield H. (2025). *RLFTSim: Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning.* Technical Report for the Waymo Open Sim Agents Challenge.

### Soundness
2

### Presentation
2

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
This paper presents RIFT (Group-Relative Reinforcement Learning), a method for encouraging fair and cooperative driving behaviors among multiple autonomous agents. Instead of optimizing individual rewards, each agent’s performance is compared to the group average, which helps promote teamwork rather than selfish behavior. A fairness regularization term further balances outcomes across agents. The method is implemented in a decentralized actor-critic framework and tested in Nocturne and Waymo Sim, where it leads to smoother, more socially efficient traffic than standard RL approaches.

### Strengths
The idea of using group-relative advantages to encourage cooperation is simple but effective.

Results show clear improvements in fairness, stability, and overall efficiency.

The framework is decentralized and scalable, making it practical for large-scale driving simulation.

The experiments are thorough and show emergent, human-like cooperative driving.

### Weaknesses
The paper could more clearly analyze how the group-relative term affects individual vs. collective reward trade-offs.

The related work section should discuss earlier related papers including:
• A. Kuefler, J. Morton, T. Wheeler, and M. J. Kochenderfer, “Imitating Driver Behavior with Generative Adversarial Networks,” IEEE Intelligent Vehicles Symposium (IV), 2017, pp. 204–211.
• R. P. Bhattacharyya, B. Wulfe, D. J. Phillips, A. Kuefler, J. Morton, R. Senanayake, and M. J. Kochenderfer, “Modeling Human Driving Behavior through Generative Adversarial Imitation Learning,” Computing Research Repository (CoRR), arXiv:2006.08911, 2020.
• H. Chen, T. Ji, S. Liu, and K. Driggs-Campbell, “Combining Model-Based Controllers and Generative Adversarial Imitation Learning for Traffic Simulation,” IEEE International Conference on Intelligent Transportation Systems (ITSC), 2022, pp. 1698–1704.
• K. Brown, K. Driggs-Campbell, and M. J. Kochenderfer, “Modeling and Prediction of Human Driver Behavior: A Survey,” arXiv preprint arXiv:2006.08832, 2020.

### Questions
Could the authors elaborate on how the group advantage interacts with overall performance? 

How sensitive is RIFT to the group size or population composition?


Could the authors provide a deeper analysis of the failure cases, such as uniformly conservative or overly passive driving behaviors?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes RIFT, a framework that utilize RL to effectively improve controllability and realism for closed-loop driving simulation. After an IL model (Pluto) is trained to generate realistic, multimodal, route-conditioned trajectories for critical background vehicles (CBVs), the authors introduce Group-Relative RL Fine-Tuning that freezes the trajectory generation head and fine-tunes only the scoring head using a group-relative advantage over all candidate modalities, an equal-weight (mode-preserving) objective, and a dual-clip surrogate for stable optimization without KL anchoring. This design tackles covariate shift while retaining trajectory-level realism, style-level and route-level controllability. Extensive experiment results are provided to show the effectiveness in controllable and realistic scenario generation, closed-loop evaluation on E2E planners, and the effectiveness in the algorithmic design for RIFT.

### Strengths
- **Clarity and Presentation.** The paper is clearly written and the core ideas are easy to follow, with well-motivated design choices and consistent presentation.
- **Empirical Rigor.** Experiments are extensive, and the ablations are thoughtfully constructed to isolate the contribution of key components.

### Weaknesses
- **Baselines for Controllability.** The paper does not compare against closely related controllable traffic generation methods (e.g., CTG, LCTGen). Given the paper’s emphasis on controllability, these baselines are important to position the contribution.
- **Related Work on Group-Relative RL.** Prior work has already explored group-relative rewards for RL fine-tuning in closed-loop driving (e.g., Gen-Drive [1]). While the present paper targets scenario generation rather than policy learning, acknowledging and contrasting with these methods would sharpen the novelty claims.
- **Covariate Shift Analysis.** The motivation highlights the open-loop vs. closed-loop covariate shift as a core challenge, yet the paper lacks a dedicated analysis. A quantitative and qualitative comparison of Pluto with SFT vs. Pluto with RIFT—under matched conditions—would strengthen the causal link between RLFT and improved closed-loop robustness.
- **CBV Identification Assumption.** Critical background vehicles are selected via a distance threshold. Prior studies suggest that distance alone is less robust than TTC-based criteria for identifying safety-critical interactions [2,3]. A justification, sensitivity analysis, or comparison with TTC-based CBV identification would improve credibility.

> [1] Huang et al., *Gen-Drive: Enhancing Diffusion Generative Driving Policies with Reward Modeling and Reinforcement Learning Fine-Tuning*, ICRA 2025.  
> 
> [2] Chang et al., *Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries*, ECCV 2024.  
> 
> [3] Lin et al., *Causal Composition Diffusion Model for Closed-Loop Traffic Generation*, CVPR 2025.

### Questions
1. **Equal-Weighted GRPO.** How is equal weighting across modalities operationalized in practice? Does this require on-policy rollouts under a slightly older behavior policy, and if so, how is sampling bias handled or corrected? Please provide algorithmic details and any variance/bias trade-offs.
2. **Block Rate in Table 2.** For PDM-Lite, should Block Rate be “lower is better”? If so, please clarify the directionality and explain why BR is not reported for other E2E planners lacking privileged information.
3. **Measuring Style-Level Controllability.** How do you quantify style-level controllability (if at all)? If formal quantification is challenging, could you expand the qualitative analysis (e.g., as in Appendix D, Fig. 6) with more systematic case studies or user studies?

### Soundness
3

### Presentation
3

### Contribution
3
