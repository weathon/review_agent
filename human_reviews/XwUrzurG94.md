# Rapidly Adapting Policies to the Real-World via Simulation-Guided Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Robot learning requires a considerable amount of high-quality data to realize the promise of generalization. However, large data sets are costly to collect in the real world. Physics simulators can cheaply generate vast data sets with broad coverage over states, actions, and environments. However, physics engines are fundamentally misspecified approximations to reality. This makes direct zero-shot transfer from simulation to reality challenging, especially in tasks where precise and force-sensitive manipulation is necessary. Thus, fine-tuning these policies with small real-world data sets is an appealing pathway for scaling robot learning. However, current reinforcement learning fine-tuning frameworks leverage general, unstructured exploration strategies which are too inefficient to make real-world adaptation practical. This paper introduces the \emph{Simulation-Guided Fine-tuning} (SGFT) framework, which demonstrates how to extract structural priors from physics simulators to substantially accelerate real-world adaptation. Specifically, our approach uses a value function learned in simulation to guide real-world exploration. We demonstrate this approach across five real-world dexterous manipulation tasks where zero-shot sim-to-real transfer fails. We further demonstrate our framework substantially outperforms baseline fine-tuning methods, requiring up to an order of magnitude fewer real-world samples and succeeding at difficult tasks where prior approaches fail entirely. Last but not least, we provide theoretical justification for this new paradigm which underpins how SGFT can rapidly learn high-performance policies in the face of large sim-to-real dynamics gaps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of sim-to-real adaptation. It introduces a novel approach where the value function, learned in simulation, is used to reshape the reward signal, effectively guiding online reinforcement learning in real-world environments. The proposed method demonstrates superior performance compared to domain randomization and online system identification methods in three contact-rich and dynamic tasks: pushing, insertion, and hammering.

### Strengths
- The paper is focusing on an important problem. 
 - The proposed method is novel, built on the key observation that the value function from a well-performing policy learned in simulation can be used to effectively reshape the reward for online learning.

### Weaknesses
- The introduction is hard to follow with the logic among paragraphs not clear.
 - The methodology and figure illustrations can be further improved. 
   - Figure 2 is confusing. According to RL literature conventions, V_s in general, means the value function, but it is labeled as the world model in Figure 2. 
   - How is the dynamic model learned? There is no description of learning the dynamics model in the main script or in the main algorithm, algorithm 1.

Typos:
 - Line 19 in abstract, “is to inefficient” -> ”is too inefficient”

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel framework (SGFT) for sim-to-real transfer in robot learning by finetuning RL policies pre-trained in simulation efficiently and effectively on limited real-world interactions. In particular, SGFT changes the real-world MDP problem by (1) turning it into a finite and short-horizon MDP (H-step) and (2) reshaping the new MDP's rewards with a pre-trained critic in simulation. These two changes allow the method to quickly adapt pre-trained policies with real-world interactions, and be also used in combination with model-based RL to further improve data efficiency.

### Strengths
- The paper uses a clear writing style, where the critical statements are repeated multiple times and highlighted.
- The method demonstrates good results on contact-rich real-world tasks---which are highly relevant to the community---in 100-200 real rollouts.
- Theoretical analysis of the proposed method is included
- The method is relatively simple to implement.

### Weaknesses
- Missing references to recent, related works: the authors discuss sim2real transfer works in the first paragraph of Sec. 2, but fail to include recent state-of-the-art methods as of 2023 and 2024 that "adapt simulation parameters to real-world data" (DROPO [1]), or that "learn adaptive policies to account for changing real-world dynamics" (DORAEMON [2]). I suggest skimming through these papers to make sure these two literature branches are well referenced and discussed. 

-  Limited experimental evaluation: I raise concerns over the main empirical findings in Fig. 4.
	- The SAC baseline seems to perform overly bad. To my understanding, this baseline essentially starts the same way as SGFT, but then diverges as finetuning goes on in that SAC also changes its own critic, whereas SGFT relies on a frozen initial critic (cf. my question below for further details). Also, in light of recent claims [3], one would expect off-policy algos to perform better if correctly finetuned.
	- In my opinion, the comparison to DR baselines should be done by pre-training them with omniscent critic (aka asymmetric actor-critic), as described in [4] and often done in recent works [5]. SGFT must be trained in simulation with unpriviliged critics, hence it's making a subtle further assumption that DR methods can, instead, relax and leverage. 
	- the Recurrent policy + DR baseline seems to be missing from Fig. 4 pushing. Pushing is also a notorious benchmark task that has been used by DORAEMON [2] and Peng et al. [4] to showcase successful zero-shot transfer in their experiments, by randomizing similar properties (e.g. mass, friction coefficient). How are zero-shot DR baselines performing in this setting?
	- Recent state-of-the-art zero-shot transfer methods such as Doraemon [2] should be included in the experimental evaluation.
    - Only experiments in the edge case H=1 are shown. This makes it hard to motivate the need for a more general framework.



[1] Tiboni, G., Arndt K., and Kyrki V. "DROPO: Sim-to-real transfer with offline domain randomization." Robotics and Autonomous Systems 166 (2023): 104432.

[2] Tiboni, G. et al. "Domain Randomization via Entropy Maximization." ICLR 2024.

[3] Ball, Philip J., et al. "Efficient online reinforcement learning with offline data." International Conference on Machine Learning. PMLR, 2023.

[4] Peng, Xue Bin, et al. "Sim-to-real transfer of robotic control with dynamics randomization." 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018.

[5] Handa, Ankur, et al. "Dextreme: Transfer of agile in-hand manipulation from simulation to reality." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.

### Questions
- When H=1, what is the difference between a standard actor-critic (AC) method with a frozen critic vs. your SGFT method? To me, it seems that an AC method with frozen pre-trained critic essentially describes the SGFT method. If so, this could yield further intuitions and explanations of the algorithm. In other words, when H=1, it seems to me like the finetuning problem is turned into a contextual bandit problem where no sequential decision making is involved.

- I'm not convinced by the statement "optimizing Equation (3) guides policy optimization algorithms towards policies which increase the value of V_sim over the H-step windows." in Sec. 4.3. To my understanding, as H increases, the algorithm actually gives more importance to real-world returns rather than increasing the value of V_sim. In other words, a final state S_H with lower V_sim(S_H) could be preferred if the real H-step return to get to that state is relatively higher than it was in sim. Conversely, the statement gets truer the lower the value of H.

### Soundness
2

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
4

### Summary
The paper proposes a method that learns a robust transferred policy in sim-to-real settings by learning a value function, and then fixing it for H-step MPC. The deployed policy only needs to learn to optimize H step model rollouts with the assumption that even if the low level dynamics differ between simulation and the real world, the value function through temporal extension of H-steps is more reliable in deciding which are the target future states in the trajectory. The authors demonstrate both RL fast policy and MPC optimization based variants of the actor trained to optimize H step trajectories in the real environment. The method is tested on three simulation-to-real robot manipulation tasks.

### Strengths
1. The method is simple and elegant, and the motivation for doing H step planning is clear.
2. The assumption that the low level actions learnt in simulation can be incorrect, but the general value difference between states preserved is a reasonable assumption, and verified in the experiments.
3. The value functions being frozen and just finetuning the policy is likely significantly more sample efficient than finetuning all components.
3. The demonstration of TDMPC2 type model learning and planning in real world robot experiments is a great addition.

### Weaknesses
1. The choice of hyperparameter H seems extremely important, and no ablations have been done here. All algorithms have hyperparameters, but atleast the most important ones should have ablation experiments.
2. Following up on the previous point, a major weakness of the paper is the lack of extensive experiments. Considering that the proposed method is mainly an engineering improvement rather than a wholly unique algorithm, more experiments in other real world and simulated domains could have been appreciated. For example, within simulated domains there could have been experiments where the model is trained on perturbed dynamics (such as different mass of limbs for locomotion agents) and then transferred to an unperturbed environment. Other experiments could have been designed to show how the optimal values of H are related to different task types. For example it would be interesting to see if tasks that require higher precision control require higher H, which in turn makes the policy transfer more difficult.
3. There is not much novelty in the method itself, and it might even be something that people already kind of do. Especially SGFT-SAC, the authors fix H=1 for the experiments and this just means that the value function is fixed after simulation and just the policy is finetuned. The TDMPC experiments are more interesting, but this is not then evaluated in the pushing task.

### Questions
Questions:
1. In Figure 4, why are many of the baseline methods not evaluated for the pushing task?
2. What happens if you use smaller H for SGFT-TDMPC? What about larger H?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents “Simulation-Guided Fine-Tuning” (SGFT), a framework aimed at improving sim-to-real adaptation by leveraging simulation-guided data augmentation and model-based reinforcement learning techniques. The paper uses real-world tasks like hammering, inserting, and puck pushing to test SGFT’s ability to adapt policies from simulation to reality with limited real-world data. The framework builds on a model-based RL backbone, specifically using SAC and TDMPC, to optimize policy adaptation with efficient data usage and a reward shaping mechanism.

### Strengths
- The paper addresses the sim-to-real gap in reinforcement learning, an essential area in robotics research. Using a model-based framework with potential-based reward shaping provides an innovative approach to sim-to-real transfer.
- SGFT’s modular design, integrating model-based learning with reward shaping, shows the potential for efficient adaptation without extensive real-world training. The authors provide theoretical analyses 
- The evaluation on dynamic, real-world tasks such as hammering and inserting presents a realistic testing ground for examining SGFT’s efficacy.

### Weaknesses
1. Limited Task Scope and Complexity:
The evaluation primarily involves simple, structured tasks (hammering, inserting, and puck pushing), which limits the generalizability of the results. The tasks do not fully explore SGFT’s adaptability across a broader spectrum of real-world challenges or more complex environments. For example, the hammering task, despite involving contact dynamics, still relies on predefined, straightforward goals and does not include diverse object types or more intricate multi-step interactions. This narrow task set could mean that SGFT’s advantages may not scale effectively to more complex tasks .
2. Baseline Comparison and Performance:
While SGFT demonstrates some improvement in sample efficiency and asymptotic performance over baselines like SAC and domain randomization, the gains over TDMPC-2 and PBRS are limited. In Table 1, while SGFT performs comparably to these baselines in most cases, it does not consistently outperform them by a significant margin, raising questions about whether the added complexity of the SGFT approach is justified by these incremental improvements  .
3. Heavy Dependence on Simulation Accuracy and Engineering Effort:
SGFT’s approach requires a highly accurate simulation setup to guide policy learning effectively, which can be labor-intensive. The need for extensive domain randomization (e.g., Table 1 and 2 for hammering and puck pushing parameters) and environment-specific configurations could make SGFT less practical for tasks where precise dynamics modeling is challenging, such as interactions with liquids or deformable objects. This limitation suggests SGFT may struggle in scenarios where real-world dynamics differ significantly from the simulation environment  .
4. Uncertain Performance on More Complex Dynamics:
While the paper effectively adapts to straightforward tasks, its approach may not handle more nuanced dynamics. For instance, tasks involving materials with variable properties (like deformable objects or variable friction surfaces) may introduce unpredictable behaviors that SGFT, as presented, is unlikely to manage effectively. Thus, the scope of SGFT’s generalizability and flexibility across a broader range of robotics tasks remains unclear  .

### Questions
1. How would SGFT handle tasks involving more complex dynamics, such as fluid or deformable object manipulation, where precise dynamics modeling is difficult? Would this require substantial engineering modifications, or does SGFT have inherent limitations in these scenarios?
2. Can the authors provide insights into the expected engineering effort for adapting SGFT to new environments? Given the dependency on extensive domain randomization and parameter tuning, how feasible is it to scale SGFT across a diverse set of real-world tasks?
3. Are there specific plans to expand the evaluation to include more complex tasks or different robotic systems? Given the limited range of tasks tested, a broader evaluation could provide a more comprehensive understanding of SGFT’s robustness and adaptability.

### Soundness
3

### Presentation
2

### Contribution
2
