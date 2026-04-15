# A Pontryagin Perspective on Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Reinforcement learning has traditionally focused on learning state-dependent policies to solve optimal control problems in a _closed-loop_ fashion. In this work, we introduce the paradigm of _open-loop reinforcement learning_ where a fixed action sequence is learned instead. We present three new algorithms: one robust model-based method and two sample-efficient model-free methods. Rather than basing our algorithms on Bellman's equation from dynamic programming, our work builds on _Pontryagin's principle_ from the theory of open-loop optimal control. We provide convergence guarantees and evaluate all methods empirically on a pendulum swing-up task, as well as on two high-dimensional MuJoCo tasks, demonstrating remarkable performance compared to existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates open-loop reinforcement learning (OLRL) and proposes three algorithmic variants based on the meta algorithm (Alg.1) to address OLRL problems: model-based, on-trajectory, and off-trajectory. Each of these methods estimates the gradient of the action sequence from a different perspective. The authors conducted experiments on the pendulum, Ant, and HalfCheetah environments, revealing several properties of OLRL, including its robustness to model error and noise, as well as its sample efficiency in off-trajectory scenarios.

### Strengths
The authors approach the decision-making problem from a relatively novel perspective and design multiple feasible algorithms and experiments within this framework, demonstrating the advantages of OLRL over closed-loop RL.

### Weaknesses
1. Aside from the benefits of reduced sensor and computation costs, I still struggle to understand the practical advantages of OLRL compared to closed-loop RL. I also have questions about certain statements made in the introduction. For instance, in line 49, the authors claim that OLRL is more robust if the environment changes. However, in unforeseen environment changes, it's difficult to determine whether OLRL or closed-loop RL would perform better. Furthermore, if the environmental change was encountered during training, closed-loop RL can adapt to it, whereas OLRL cannot. Real-world scenarios are always changing, which gives closed-loop RL with adaptive capabilities a clear advantage.
2. In model-based open-loop RL, the gradient is propagated along the dynamics model, leveraging the differentiable nature of the dynamics model. However, this gradient propagation across the entire trajectory, as done in SVG($\infty$) [1], may cause training instability, especially with long trajectories. In contrast, [2] and [3] reduce this trajectory length via a value function, thereby improving training stability. I think the authors could discuss these prior works.
3. In the model-free approach, the authors approximate the gradient using samples, which is a linear approximation and might limit the algorithm’s performance.

[1] Heess et al. "Learning continuous control policies by stochastic value gradients." NeurIPS 2015.

[2] Clavera et al. "Model-augmented actor-critic: Backpropagating through paths." ICLR 2020.

[3] Amos et al. "On the model-based stochastic value gradient for continuous reinforcement learning." L4DC 2021.

### Questions
1. Can the proposed methods be applied in environments with terminal states (i.e., variable trajectory lengths)?
2. The SAC results in Figure 7 seem somewhat low. How does SAC perform in HalfCheetah at T=1000?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents an interesting perspective on open-loop RL and includes theoretical convergence guarantees. However, a lack of contextual depth in related work, an unclear justification for the choice of Pontryagin’s principle, and limited baseline comparisons in the experiments reduce the strength of its contributions. Expanding the literature review, clarifying the theoretical approach, and incorporating additional baselines would enhance the impact and relevance of this work.

### Strengths
1. The paper contributes to reinforcement learning by addressing the relatively underexplored area of open-loop RL, which has potential benefits in environments with deterministic or predictable dynamics.

2. The authors provide a convergence proof for their proposed algorithms, enhancing the theoretical rigor and assuring readers of the stability of the methods under specified conditions.

### Weaknesses
1. The introduction and related work sections would benefit from a more comprehensive review of existing research on open-loop control and open-loop reinforcement learning, especially high-impact studies. By not including these, the paper risks underselling the importance of its contributions in the context of current research. Citing influential works would help position this study within the broader field and underscore the significance of open-loop RL.

2. The paper introduces both open-loop RL and Pontryagin’s principle but does not clearly articulate the relationship between these two concepts. It is unclear whether Pontryagin’s principle is essential for the open-loop RL problems tackled in this paper or if similar outcomes could be achieved with alternative optimization techniques. Additionally, the term "Pontryagin’s principle" is unprofessional; traditionally, this concept is referred to as "Pontryagin’s maximum principle" or "minimum principle."  In the paper it appears to be a direct application to the definition of value function, which reduces the reduced theoretical novelty.

3. Although the work is positioned within reinforcement learning, the empirical section only compares the proposed algorithms against a single baseline (SAC). This limited comparison does not fully contextualize the performance of the new methods within RL. Expanding the baselines to include other established RL algorithms, such as PPO or DDPG, would provide a more comprehensive assessment and better validate the contribution of the new methods.

### Questions
see weakness and please justify

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work propose a trajectory optimization method for RL problems where the (deterministic) differential dynamics are approximated. A naive approximation or linear approximation using least squares. On top of this, the algorithm find a "stateless" policy, or control, through vanilla gradient descent to achieve the necessary first order optimality condition (Pontryagin's principle) of the trajectory optimization problem.

### Strengths
It is interesting how the authors bridge differential dynamic programming into RL, brining a new perspective onto policy learning. 
The presentation of the paper is good, easy to follow, and in general nicely written. 
The theoretical analysis is nicely done, with satisfactory results, and validated through numerical experiments.

### Weaknesses
Novelty:
In light of the literature in trajectory optimization, specifically Differential Dynamic Programming (DDP), the content of this work has already been studied extensively, e.g. [1]. The elements in the proposed method seem neither new (using samples to approximate the transition probabilities, and solving a finite-horizon deterministic control problem with first order method such as gradient descent); or particularly novel, as either 1) the control community has extensively studied effective method in the unknown dynamics setting, or 2) the model-based RL community has proposed far more capable "world models" that better capture the environment dynamics beyond linear. I have to ask the authors to provide a more extensive related work section to discuss the novelty of this work. To my understanding, the main idea (Pontryagin's principle) is exactly the first order necessary condition for the trajectory optimization problem in this finite horizon MDP setting, and Algorithm1 is merely performing standard gradient descent. 

Formulation: 
The proposed open loop RL can be subsumed as RL with state space being the time stamp in the control sequence, i.e., $\mathcal{S}=\{1,2,...,T\}$. In view of this, I am not sure how the name open-loop makes sense, as the training requires interaction with the environment to collect rollouts. 
Furthermore, the authors empathized that the proposed method is significantly different from MPC, while the algorithm has no real difference than a MPC, except that the proposed method can't handle variable starting state (i.e., due to "open loop").

Applicability:
The entire algorithm seems to require the horizon to be relatively short, as all experiments seems to truncate the original environment length down to 1/10 (and up to 3/10 in the ablation studies). This brings concerns with the applicability of this method, as in real applications, where any reasonable control rate approaches 30-50Hz, 100 steps equals to less than 3 seconds. If the learned policy need to re-plan, then it most definitely need to take in the new state as input, which is not achievable under the current framework. 

1. Yamaguchi, Akihiko, and Christopher G. Atkeson. "Neural networks and differential dynamic programming for reinforcement learning problems." 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.

### Questions
Please see the above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors present a new framework for incorporating control theoretic methods into reinforcement learning (RL). They propose the use of the Pontryagin’s principle which computes the Jacobian of the value to apply it in RL. They provide a theoretically sound algorithm that converges to the optimal control asymptotically under certain assumptions. They present an algorithm that estimates the Jacobian of transition function (consequently the value function) in a model free manner by solving a least squares method both in on-policy and off-policy setups. They demonstrate the effects of model mis-specification in a toy environment and show performance improvements in two MuJoCo environments (Ant and Cheetah).

### Strengths
The paper is well written and provides connection from control theory to existing RL concepts very well. They also analyze their methods efficacy using a toy example and do a great job explaining it. They provide both model-free and model-based algorithms using Pontryagin’s principle.

### Weaknesses
Major issues:

I see three major issues.

**Comparison to previous work:** I believe there are similarities to PILCO [1] in terms of how you estimate the Jacobian of the dynamics (lines 125-132). This a fairly popular model based method in RL and I believe you could compare against it [1,2]. Moreover, you assume deterministic dynamics whereas PILCO is more general purpose with stochastic dynamics. I believe there is some analogue of the Jacobian estimate in PILCO for deterministic dynamics that follows closely what you describe as the Jacobian estimate. I would look at equations 26-28 in the PILCO paper available online. I would also expect some kind of comparison to this method in the experimental section.

**Assumption 3.3 is very strong:** while this $L$-smooth assumption might be common in convex optimization it might not be so in RL. As you might be well aware in dynamical systems the Gronwall's inequality leads to an exponential upper bound for small deviations in the dynamics determined by an ordinary differential equation. Intuitively, a small deviation in the control output can lead to exponential deviation in the cumulative reward/cost. If you dont think this is the case then please let me know why. Is there a parallel for this assumption in control theory or RL? Moreover, can you empirically or analytically verify whether or not this assumption holds for the cartpole swing-up toy problem in section 4.1 since the dynamics are known to you?

**Limited experimentation:** In Section 4.2 the authors only present results in two MuJoCo environments. I believe a broader test across more environments (like DM control suite and other MuJoCo) environments will be beneficial to the community. I would also expect ablations with respect to the hyper-parameters in the MuJoCo setting, especially the forgetting factor $\alpha$.

--------

**References**

[1] PILCO: A model-based and data-efficient approach to policy search, Deisenroth, Marc and Rasmussen, Carl 

[2] Improving PILCO with Bayesian neural network dynamics models, Gal, Yarin and McAllister, Rowan and Rasmussen, Carl Edward

### Questions
Minor issues and questions:

Section 3.3 and 3.4: Are "ON-TRAJECTORY" and "OFF-TRAJECTORY" RL similar to on policy and on policy learning? I see you comment on this on line 243 but I believe you can simply call them that, unless you see a significant reason to mark a distinction.

Assumption 3.2: I am unable to fully grasp this assumption and would like to understand it. In this assumption you are doing away with the Markov assumption for the rewards but then you propose to exploit Markov structure in section 3.3 (line 248).

### Soundness
2

### Presentation
2

### Contribution
2
