# Model Predictive Adversarial Imitation Learning for Planning from Observation

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4, 4

## Abstract
Humans can often perform a new task after observing a few demonstrations by inferring the underlying intent. For robots, recovering the intent of the demonstrator through a learned reward function can enable more efficient, interpretable, and robust imitation through planning. A common paradigm for learning how to plan-from-demonstration involves first solving for a reward via Inverse Reinforcement Learning (IRL) and then deploying it via Model Predictive Control (MPC). In this work, we unify these two procedures by introducing planning-based Adversarial Imitation Learning, which simultaneously learns a reward and improves a planning-based agent through experience while using observation-only demonstrations. We study advantages of planning-based AIL in generalization, interpretability, robustness, and sample efficiency through experiments in simulated control tasks and real-world navigation from few or single observation-only demonstration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Model Predictive Adversarial Imitation Learning (MPAIL), which combines model predictive control (MPC) with adversarial imitation learning to enable planning from observation-only demonstrations. Instead of learning a policy directly, MPAIL trains a discriminator-based reward and deploys it through a physics-based planner (MPPI) with a value function for long-horizon reasoning. Experiments show that this approach improves out-of-distribution generalization and real-world robustness, including a single-demonstration RC car navigation task, though the evaluation is limited to one type of real task and a few baselines.

### Strengths
The paper crisply frames planning‑based AIL and operationalizes it via MPAIL: MPPI + AIRL‑style reward. The KL‑regularized perspective that leads to MPPI as the optimizer is clean and well explained.

The state‑transition formulation directly targets Learning‑from‑Observation under POMDPs (no actions), which is the harder and more realistic setting.

Methodology is reproducible, with transparent algorithms, hyperparameters, and implementation details.

Empirical questions in the experiments are well-posed and probe the main claims of the paper.

### Weaknesses
Partial observability details are underspecified. While the POMDP framing is explicit, how the planner’s state is constructed from limited observations in each experiment is not fully clear. For instance, the navigation experiments plan with a Kinematic Bicycle model, yet in the single‑demo real experiment the expert data are “position and body‑centric velocity” only; it remains unclear how heading/other latent state are estimated for MPPI at train/test time and how the discriminator’s input (state vs transition) aligns with the planner’s state. A concise subsection clarifying the observation to state pipeline (filters, history windows, or learned encoders), and the exact planner state per task, would strengthen soundness.

Attribution of robustness to the physics prior vs learned reward. The OOD gains in navigation may partly stem from MPPI with a physics model (even if approximate). The paper takes a step toward a learned‑model regime on Cartpole, but does not repeat the OOD evaluation there. Without such a test, it’s hard to tell how much robustness persists when the dynamics must be learned and are mis‑specified. An OOD study in the learned‑model setting (e.g., Cartpole) would solidify the generalization claim.

Missing “performance at convergence” perspective. When the model of the world is assumed, sampling becomes cheap. Therefore, I would like to see the comparison of the methods at convergence.

A dedicated “Baselines” section summarizing inputs, optimization style, and deployment (policy vs planner, known vs learned dynamics) would reduce confusion about advantages/disadvantages among IRL‑MPC, AIRL, GAIL, MPAIL and the role of multimodality in failures.

Real‑world scope is narrow. The single‑task navigation demo is valuable, but claims about safety/robustness would be more convincing with additional real tasks

Clarity nits.
The abstract/intro list many keywords; grounding them in the concrete setting focused on the contribution would sharpen the message.
Minor: the manuscript seems to use an older template.

### Questions
Questions*
How is partial observability handled in practice? What state representation does the planner use when the observation is limited?

How is the Kinematic Bicycle model integrated with partial information? 

Could the authors test OOD generalization in the learned-model setting (e.g., Cartpole) to separate the benefits of physics priors from the algorithm itself?

Please include a short Baselines section summarizing the setup, inputs, and weaknesses of GAIL, AIRL, IRL-MPC, and MPAIL.

Does GAIL’s circling failure continue at convergence?

How does the method compare to a policy learning methods that assumes the same world model at convergence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Model Predictive Adversarial Imitation Learning (MPAIL), a principled framework that integrates Model Predictive Control (MPC) and Generative Adversarial Imitation Learning (GAIL) to address imitation learning (IL) from observations.
Unlike conventional adversarial IL methods that rely on a single-step policy update, MPAIL introduces a planning-based adversarial loop that optimizes both the policy and the discriminator through receding-horizon trajectory optimization. The key insight is that short-horizon predictive planning improves stability and sample efficiency by leveraging model-based rollouts for adversarial training. Specifically, MPAIL replaces GAIL’s policy gradient updates with MPC-based planning that iteratively minimizes the discriminator reward while respecting dynamics constraints. This results in a control sequence that approximates expert behavior through direct trajectory matching rather than relying solely on gradient backpropagation.

### Strengths
1.	The paper’s core contribution of embedding adversarial imitation learning within a model predictive control loop is novel and elegant. This approach reframes IL as a constrained planning problem, where the policy implicitly solves an adversarial objective using model-based prediction instead of policy gradients. This is conceptually distinct from prior model-based IL works (e.g., PILCO, Dyna-IL) that use learned dynamics for data generation but retain standard policy optimization. Here, the MPC layer becomes the optimizer itself, aligning adversarial rewards directly with feasible trajectories. This rethinking of the optimization hierarchy is both theoretically meaningful and practically beneficial. It unifies imitation learning, trajectory optimization, and adversarial training into a single, closed-loop control process. 

2.	The paper goes beyond empirical demonstration by providing a stability analysis grounded in control theory. The authors derive sufficient conditions for local convergence to expert trajectories based on Lipschitz continuity of the discriminator and bounded model approximation errors. Although the proof is local, it provides valuable intuition about how predictive horizons stabilize adversarial learning, reducing the oscillatory updates common in GAIL-style min–max optimization. The connection between receding-horizon planning and bounded regret minimization in adversarial settings is particularly insightful, offering a bridge between MPC theory and generative adversarial dynamics. 

3.	MPAIL consistently outperforms baselines, achieving higher reward imitation and smoother convergence. Notably, the ablation showing improved robustness to observation noise and model misspecification provides strong evidence of the approach’s practical value.

### Weaknesses
1.	A central limitation of MPAIL is its assumption of an available or learnable model for predictive control. While this is standard in MPC, it restricts applicability to domains where dynamics are known or can be learned accurately. The authors claim that “MPAIL is model-free when using learned dynamics,” but this statement is misleading—using an imperfect learned model can degrade adversarial convergence. Empirical evaluation with explicit model-error quantification (e.g., using an ensemble dynamics model) is missing.

2.	The theoretical guarantees focus on local stability near expert trajectories but do not address global convergence or nonconvex adversarial interactions. The adversarial MPC game may still exhibit multiple equilibria or oscillations if the discriminator is poorly calibrated. While the empirical stability is strong, the paper’s theoretical claims should be tempered.

3.	MPAIL performs optimization through MPC at every iteration, which introduces substantial computational overhead compared to standard policy-gradient IL. The authors briefly mention using MPPI (Model Predictive Path Integral Control) for rollout sampling, but do not provide timing benchmarks or hardware requirements.

**Other Important Issues**:

1. The authors used the incorrect template (ICLR 2025) to prepare the submission, and 
2. more importantly, the corresponding author's email is displayed on the first page, which potentially violates the double-blind review process.

### Questions
1.	How is the dynamics model obtained for environments without ground-truth simulators?
If learned, how is model uncertainty incorporated into MPC (e.g., via ensembles or stochastic constraints)?

2.	How sensitive is performance to planning horizon H?
Does longer horizon always improve imitation fidelity, or does it destabilize discriminator updates due to compounding model errors?

3.	Does the MPC loop reduce adversarial oscillations by smoothing gradient updates?
Have you analyzed the frequency of discriminator-policy mode collapse?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel framework that unifies Inverse Reinforcement Learning (IRL) and Model Predictive Control (MPC) into a single, end-to-end imitation learning approach. Traditional imitation learning methods, such as Adversarial Imitation Learning (AIL), rely on policy optimization and require access to expert actions, limiting their interpretability and robustness in real-world settings. In contrast, MPAIL performs planning-based adversarial imitation directly from observation-only demonstrations - state trajectories without corresponding actions - by embedding a model predictive planner (specifically, Model Predictive Path Integral control, MPPI) within the adversarial learning loop. This integration allows the system to simultaneously learn a reward (or cost) function and optimize the planner online, effectively merging the training and deployment phases of IRL-MPC pipelines. The proposed framework operates under the Planning-from-Observation (PfO) paradigm, enabling interpretable, safe, and generalizable imitation without explicit action data. The key novelty lies in replacing the policy-based “generator” in adversarial imitation learning with an online MPC planner that continually solves short-horizon trajectory optimization problems guided by the learned reward and a value function that bootstraps long-term outcomes. The experiments show that MPAIL achieves better generalization, robustness, and interpretability than policy-based imitation methods like GAIL or AIRL, particularly in out-of-distribution and real-world settings. By integrating planning directly into the adversarial learning loop, it enables stable, sample-efficient imitation even from sparse or partially observable demonstrations.

### Strengths
1. The paper is well-organized and methodically builds from theoretical formulation to algorithm design and experiments, making a complex contribution accessible and reproducible.
2. The approach successfully handles observation-only demonstrations and partial observability, showing real-world viability for robot learning from minimal, ambiguous expert data.
3. The experiments span both simulated and real-world settings, demonstrating strong improvements in generalization, robustness, and sample efficiency compared to established baselines like GAIL, AIRL, and IRL-MPC.

### Weaknesses
1. Most experiments focus on navigation and simple control tasks, so it’s unclear how well MPAIL would perform on more complex, high-dimensional problems like manipulation or multi-agent settings.
2. Although the few-shot setup is intentional, relying on very few demonstrations may make the results sensitive to the choice or quality of those examples, which isn’t extensively analyzed.
3. The paper could better isolate which components (e.g., MPPI planner, value bootstrapping, or reward formulation) contribute most to performance improvements.
4. The real-world experiment is limited to a single small robot platform and one trajectory, so it’s hard to assess generality or scalability to other hardware and environments.

### Questions
Please see weaknesses in the section above.

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
3

### Summary
This paper presents a new pipeline for learning from demonstration that conducts adversarial imitation learning while construct the policy through a model-based planning approach. The authors target the crucial setting of learning in partially observable environments that requires real-time planning and control and propose to roll out the optimal policy using a model predictive planning algorithm. The authors formulate a new previous-policy-regularized policy optimization objective that matches the idea of the proposed planning procedure.

### Strengths
This paper presents a novel planning based policy optimization in AIL that matches well with the real deployment of the learned policy.

This paper presents several theoretical results that connect the KL-constrained policy optimization using MPPI with the adversarial imitation learning objective.

This paper demonstrates the application through deployment in a real world environment.

### Weaknesses
This paper targets the problem of learning from observational demonstrations in a POMDP setting. However, it seems that all the technical developments are based on the full state information.

The major contribution comes from the usage of model-based planning algorithm MPPI for AIL, but the motivation of using MPPI instead of other methods require more discussion. Can other model-based planning methods be used? What is the advantage of MPPI and why other methods here are inferior?

The paper is not well written overall. The presentation leaves many points unclear. Many phrases and notations used are unnatural and hard to understand. For example,
- L116: the proposed method relies on a predictive model and a cost function for model-based planning. However, it is unclear whether these models are trained from data or known a priori. If they are learned from data, how are these models trained? This is only mentioned until the experiments section and it would be better to explain this in the method section.
- L176: It is unclear how the action $a$ in $\bar{c}(s,a)$ is used in the definition and how the transition probability $\mathcal{T}$ is defined. Does $\mathcal{T}$ depends this particular $a$ or it depends on a specific policy like $\mathcal{T}^{\pi}$?
- L202: “By decomposing the agent this way, we require not the policy, or solution, to generalize but the reward, or problem”. What do solution and problem here specifically refer to?
- Typo: L189: value function?

I would strongly recommend the authors check the writing to improve the presentation, and I look forward to the authors’ feedback for further evaluations.

The paper would benefit from comparisons with more imitation learning algorithms as baselines other than GAIL/AIRL. From the figures, the performance of MPAIL converges much slower than GAIL in Cartpole.

### Questions
L778: why the planning set a 0 action for the last timestep? Does this affect the initial plan at the next iteration?

It seems that this paper mainly targets continuous action space. Can it be applied to discrete action space? If so, how can it be adapted to handle discrete actions?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Model Predictive Adversarial Imitation Learning (MPAIL), a novel framework for teaching agents to perform tasks by observing demonstrations, a problem setting termed Planning-from-Observation.

The core idea is to unify the typically separate processes of learning a reward function (inverse RL) and using it for online planning (MPC). MPAIL achieves this by using a planner, specifically Model Predictive Path Integral control (MPPI), as the "generator" within an Adversarial Imitation Learning loop. This allows the agent to simultaneously learn a reward (cost) function and a value function while interactively improving its planning capabilities, all from observation-only demonstrations without expert actions.

Through experiments in simulated control tasks and a real-world robot navigation task using a single demonstration, the authors demonstrate that MPAIL offers significant advantages over traditional policy-based AIL methods. These benefits include improved generalization to out-of-distribution states, greater robustness, better sample efficiency, and enhanced interpretability by providing insight into the agent's decision-making process.

### Strengths
- The idea of unifying the reward learning (IRL) and online planning (MPC) into a single, end-to-end training process is new.

- It provides good experimental evidence that using a planner as the generator improves generalization and robustness in out-of-distribution states compared to standard policy-based AIL methods.

- The paper demonstrates successful real-world application through a "Real-Sim-Real" experiment, where the agent learns to navigate from just a single, noisy, and partially observable demonstration.

- Benchmarking results show that MPAIL is sample-efficient, achieving good performance with fewer environmental interactions than established AIL algorithms like GAIL.

### Weaknesses
- The method's scalability to high-dimensional action spaces is highly questionable. The paper relies on Model Predictive Path Integral (MPPI), a sample-based planner that suffers from the curse of dimensionality. While it works for low-dimensional actions like vehicle control (2D action), its performance degrades sharply as the action space grows. The paper's own experiment on the Ant environment (Figure 12), which has an 8-dimensional action space, demonstrates this weakness clearly: MPAIL shows "signs of life" but is drastically outperformed by the policy-based GAIL, failing to achieve competitive performance. This suggests the proposed method is not a viable solution for more complex, high-dimensional robotics tasks like manipulation without non-trivial extensions.

- The experimental validation is conducted on relatively simple tasks. The primary successes are shown on a navigation task and cartpole balancing. These tasks do not sufficiently challenge the planning component or prove the method's applicability to complex behaviors in robotics.

- A natural baseline is to replace the AIL in the proposed method with online behavior cloning. This baseline is simpler, computationally cheaper than adversarial training, and could potentially perform well on the presented tasks. By not comparing against it, the paper fails to justify the added complexity and potential instability of its adversarial training loop.

- The planner's performance is entirely dependent on the accuracy of the dynamics model. For the main navigation task, the authors use a pre-defined, approximate model. For other tasks, they learn a model online. In either case, model error (sim-to-real gap or learning inaccuracies) can lead to catastrophic planning failures. The paper does not sufficiently analyze the method's sensitivity to model error or propose robust mechanisms to handle it, which is a primary challenge for any model-based approach in the real world.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
