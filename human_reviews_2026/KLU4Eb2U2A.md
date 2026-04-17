# Automatic Reward Shaping from Multi-Objective Human Heuristics

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Designing effective reward functions remains a central challenge in reinforcement learning, especially in multi-objective  environments. In this work, we propose Multi-Objective Reward Shaping with Exploration (MORSE), a general framework that automatically combines multiple human-designed heuristic rewards into a unified reward function. MORSE formulates the shaping process as a bi-level optimization problem: the inner loop trains a policy to maximize the current shaped reward, while the outer loop updates the reward function to optimize task performance. To encourage exploration in the reward space and avoid suboptimal local minima, MORSE introduces stochasticity into the shaping process, injecting noise guided by task performance and the prediction error of a fixed, randomly initialized neural network. Experimental results in MuJoCo and Isaac Sim environments show that MORSE effectively balances multiple objectives across various robotic tasks, achieving task performance comparable to those obtained with manually tuned reward functions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MORSE (Multi-Objective Reward Shaping with Exploration), a bi-level optimization framework that automatically combines multiple heuristic rewards into a shaped reward for reinforcement learning. The method introduces stochastic exploration in the outer loop, guided by both task performance and a novelty metric computed via Random Network Distillation (RND). Experiments on several MuJoCo and Isaac Sim locomotion tasks show that MORSE can achieve performance comparable to manually tuned “oracle” rewards while avoiding local minima that affect conventional bi-level optimization.

### Strengths
1. The motivation is clear and realistic: reward shaping is indeed a bottleneck in RL, and automating it is a useful and practical direction.

2. The paper is well motivated and the proposed method is technically sound overall, combining bi-level optimization and exploration in a novel way.

3. The approach is close to practice, since many real tasks combine a sparse goal reward with auxiliary shaping terms.

4. The ablation studies are valuable, helping clarify which design elements (exploration trigger, RND novelty, policy reset) matter most.

5. The writing is generally clear, and the figures illustrate the concept well.

### Weaknesses
1.  The experiments are restricted to simple locomotion tasks with only 2–3 heuristic components. This is far from realistic use cases with many interdependent objectives. The study would be stronger if it included manipulation or visual tasks (for instance, environments from RLBench, James et al., 2020).

2. While using novelty helps, relying solely on RND is not fully intuitive for reward-space exploration. Methods such as Bayesian Optimization could provide a more principled trade-off between novelty and task performance.

3. Details are missing. Some important definitions are vague. For example, the novelty metric $V_novelty$ is referenced but never formally defined, and the “task criteria” normalization is unclear. This makes it harder to fully reproduce the results. How exactly the P is calculated?

4. The first validation experiment (Sec 6.1.1) is essentially a synthetic numerical optimization example. While it supports intuition, it feels detached from actual RL training and resembles a simulated annealing procedure rather than a meaningful RL benchmark.

5 . Since the method relies on policy resets and outer-loop exploration, it might be sensitive to the learning rate and reset frequency. No robustness analysis is given.

### Questions
1.Would off-policy methods (e.g., SAC) benefit even more from this framework, since they can reuse past transitions for multiple reward hypotheses?

2. Could the authors include more diverse environments, such as manipulation or navigation tasks, to test scalability beyond locomotion?

3. How does MORSE compare to a simpler baseline that uses constant reward weights but periodically resets the policy (given that “wo/ Reset Policy” performs poorly in Fig. 4)?

4. How is the oracle reward function defined—are those weights hand-tuned or optimized via grid search?

5. Since learning rate choices affect convergence and local minima, could the authors test MORSE and the baselines under different rates to show robustness?

### Soundness
2

### Presentation
2

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
The authors present MORSE, a framework for reward shaping that automates reward weight tuning. The method seeks to improve exploration of the reward space by guiding the RL policy towards higher task success. Their goal is not to achieve or balance multiple objectives, but rather to achieve a higher task success rate in general.

The authors propose an inner and outer loop algorithm to automatically find reward weights that aim to maximize the task reward. The inner loop trains an RL policy given a reward weight. The outer loop assigns a novel weight to the inner loop to overcome local optima. This is achieved with an exploration-guided formulation using Random Network Distillation.

Overall, I am positive about this work; the idea is simple and well-evaluated. The paper is on the theoretical side, and I would like to see it applied to more practical problems that involve many more reward terms, where the problem of reward tuning becomes truly cumbersome.

### Strengths
- The paper is easy to follow and provides sufficient details for reproducibility. The authors also release their code.
- The authors address an important task in RL, finding reward weights for multiple reward terms is a difficult and time-consuming task, and at the same time, it is crucial for successful training.
- The authors run extensive validation on synthetic examples as well as popular benchmarks, and they provide ablation studies for their design decisions.

### Weaknesses
- The paper's primary weakness lies in its comparison to existing methods. For me, it is not fully clear what the precise differences and similarities to traditional Multi-Objective Reinforcement Learning (MORL) are. The authors claim to solve a different task, but it is plausible that a standard MORL formulation could serve as a strong baseline with minor modifications. A direct comparison is currently missing. For instance, the "Gradient w/ Reset" baseline is intuitively similar to some MORL training approaches; it would be helpful if the authors extended their discussion on the differences or explained why MORL is not an option for this task.
- Another weakness is the demonstrated scalability of the method. Reward tuning becomes especially cumbersome in high-dimensional problems (e.g., 10s of reward terms), whereas finding weights between 2-3 terms, as done in the paper's evaluations, can often be done relatively quickly. This connects to the question of scalability raised later. Furthermore, this raises the question of how the method would compare to a simple baseline, such as a non-expert’s first guess.

**Minor:**
- Missing references for IsaacSim and MuJoCo

### Questions
- Instead of balancing potentially conflicting objectives, the authors assume their task has a clear success measurement. I am still wondering how the formulation would change or differ if the measurement for success were not clearly defined. For example, in the motion imitation literature, MORL formulations have been studied (e.g., Alegre et al., 2025, AMOR: Adaptive Character Control through Multi-Objective Reinforcement Learning). In this task, the measurement for success can be based on different preferences; for instance, sometimes tracking the motion accurately is most important, while other times reducing vibrations is prioritized. The used rewards, in this case, are dense rewards, but the weights between them reflect these different, sometimes subjective, preferences. I am wondering how this view of success, which can vary based on preferences, could be understood in the context of this work.
- Traditional MORL learns optimal policies for any given reward term weight. This would, in theory, allow one to search in the dense auxiliary-reward space for a policy that maximizes the primary-task reward. The authors motivate their method with a modified cartpole example, where we see that setting random weights and following the gradient results in optimal performance. This makes me wonder if a traditional MORL solution would not achieve a similar effect. How do the two approaches truly compare? The MORL formulation seems to provide more flexibility (e.g., allowing a change in the task reward after training and still finding the optimal weights), whereas this method appears more rigid, providing only a task-specific solution.
- Following a similar line of thought, but from another direction: the reward space might differ for different environments (e.g., if domain randomization is applied to mass properties). Since this method ultimately provides a somewhat rigid solution, I was wondering how large the sim-to-real gap might become, ultimately, if I want to deploy my policy to the real world, my reward shaping is usually heavily influenced by this target. While I do not expect the authors to compare simulation results with real-world deployments (though that would be highly preferred), I would be interested to understand how domain randomization influences the reported results. Does the solution remain similar, or does it diverge?
- The number of reward objectives in the evaluated examples is relatively low (max 3 rewards). However, tasks like motion imitation learning can involve 10s of reward terms. How does this method scale to a significantly larger number of terms? The authors mention this limitation in the appendix, but can we quantify when the method starts to break?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The authors propose MORSE, a framework to automatically combine a set of human-provided, unweighted heuristic reward functions into a single (shaped) reward function. The system only requires practitioners to specify these heuristics and a sparse, episodic task performance criterion
- MORSE formulates this as a bi-level optimization problem
  - The inner loop trains an RL policy to maximize the expected return from the current shaped reward
  - The outer loop updates the reward weight parameters to maximize the sparse task performance
- The paper's core premise is that standard gradient-based bi-level optimization fails in this domain because the weight-performance landscape is highly non-convex. MORSE addresses this by introducing a guided stochastic exploration mechanism into the outer loop
- The method is evaluated on a set of simple locomotion tasks

### Strengths
- The paper addresses a critical bottleneck in RL for robotics: the labor-intensive and error-prone process of manual reward function design
- Framing the problem as a bi-level optimization is a good approach (also used in other works)
- The use of synthetic 2D optimization functions to isolate and test the outer-loop exploration strategy (Sec 6.1) is a good experimental design concept and the motivating example was clear

### Weaknesses
- Tables 1 and 2 report means without any confidence intervals or error bars, making it impossible to validate the significance of the results
- The paper is a methodology paper but is only tested on locomotion tasks. This is not a robotics paper. The method's generality is unproven, and it should have been tested on a broader set of environments
- The experiments use tasks with only 2 or 3 heuristics, despite motivating the problem with a 15-heuristic example (Margolis and Agrawal). The method's performance in the high-dimensional settings where it would be most valuable is unknown
- Can the authors provide experimental validation on manipulation as well? Can the authors provide results for more complex tasks with a higher number of heuristics?
- The claim that RND "discovers more novel reward weights" than random sampling does not have much supporting evidence
- The paper's premise that gradient-based methods are unsuitable for this problem is an oversimplification and ignores the empirical success of such methods in non-convex settings
- The related work section (Sec 2.2) fails to cite or compare against other key papers in automatic reward shaping for robotics [1, 2]
- Tables 1 and 2 are mislabeled, using "Total" instead of "Average" for the final column

References
[1] Ma, Y.J., Liang, W., Wang, G., Huang, D.A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L. and Anandkumar, A., 2023. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931.
[2] Zhang, C.B.C., Hong, Z.W., Pacchiano, A. and Agrawal, P., 2024. ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization. arXiv preprint arXiv:2410.13837.

### Questions
- The task formulation (Sec 3.1) assumes a simple linear combination of heuristics. Why was this restrictive form chosen? What about non-linear combinations or interaction terms (eg $R_{h_1} \times R_{h_2}$) that might be required to model complex objective trade-offs?
- The outer loop updates the reward weights, which can drastically change the optimization landscape for the inner-loop policy. If you take significant gradient steps on the reward weights, does the policy training not suffer from severe instability due to the non-stationary reward function? How is this handled?
- The paper states "the resulting landscape exhibits numerous local optima, violating the assumptions under which Gradient succeeds". While vanilla GD might fail, modern gradient-based optimizers are designed for and show enormous empirical success in non-convex settings. Can the authors provide a more rigorous justification for why this specific problem is intractable for standard gradient-based optimization? Have the authors tried other gradient-based optimization techniques?

### Soundness
2

### Presentation
2

### Contribution
1
