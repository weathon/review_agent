# Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 2, 10, 2

## Abstract
Existing reinforcement learning (RL) methods struggle with complex dynamical systems that demand interactions at high frequencies or irregular time intervals. Continuous-time RL (CTRL) has emerged as a promising alternative by replacing discrete-time Bellman recursion with differentiable value functions defined as viscosity solutions of the Hamilton–Jacobi–Bellman (HJB) equation. While CTRL has shown promise, its applications have been largely limited to the single-agent domain. This limitation stems from two key challenges: (i) conventional methods for solving HJB equations suffer from the curse of dimensionality (CoD), making them intractable in high-dimensional systems; and (ii) even with learning-based approaches to alleviate the CoD, accurately approximating centralized value functions in multi-agent settings remains difficult, which in turn destabilizes policy training. In this paper, we propose a CT-MARL framework that uses physics-informed neural networks (PINNs) to approximate HJB-based value functions at scale. To ensure the value is consistent with its differential structure, we align value learning with value-gradient learning by introducing a Value Gradient Iteration (VGI) module that iteratively refines value gradients along trajectories. This improves gradient accuracy, in turn yielding more precise value approximations and stronger policy learning. We evaluate our method using continuous‑time variants of standard benchmarks, including multi‑agent particle environment (MPE) and multi‑agent MuJoCo. Our results demonstrate that our approach consistently outperforms existing continuous‑time RL baselines and scales to complex cooperative multi-agent dynamics. Code is available at https://github.com/Wangxuefeng1024/Continuous-Time-Value-Iteration-for-Multi-Agent-Reinforcement-Learning.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a continuous-time multi-agent reinforcement learning (CT-MARL) framework called VIP, which leverages PINNs to approximate value functions as solutions to HJB. 
To address inaccuracies in value gradients that can destabilize policy learning in multi-agent settings, the authors introduce a Value Gradient Iteration (VGI) module that iteratively refines gradients along trajectories, improving value approximation and overall performance.
Overall, the paper extends continuous-time RL to multi-agent cooperative scenarios, demonstrating good performance over baselines while mitigating the curse of dimensionality.

### Strengths
- The integration of PINNs with a VGI module is a fresh approach to CT-MARL, addressing key challenges like the curse of dimensionality and gradient inaccuracies in high-dimensional multi-agent systems.

- Provides solid mathematical foundations, including lemmas and theorems with proofs.

- Demonstrates genuine advantage of CT training under varying $\Delta t$ and nice stress test against a competitive DT agent.

### Weaknesses
- VGI targets rely on accurate dynamics and reward models, if they have a little difference, the estimation may differs a lot and fails.

- VGI convergence relies on bounded Jacobians and small $\Delta t$; it’s not fully clear how sensitive VIP is when these conditions are loosened

- Discrete-time MARL baselines are excluded from the main comparisons.

### Questions
- How sensitive is VGI to bias/variance in dynamics and rewards? Have you tried perturbing these models or training with deliberately misspecified dynamics to see VIP’s robustness envelope?

- How does VIP behave with very small/large variable $\Delta t$? Any adaptive-step control in the algorithm?

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
This paper proposes a **Continuous-Time Multi-Agent Reinforcement Learning (CT-MARL)** framework, termed **VIP (Value Iteration via PINN)**, which formulates MARL in the continuous-time domain.
The authors argue that conventional discrete-time RL (DTRL) methods struggle when applied to systems requiring high-frequency decision-making or irregular time intervals. To address this, VIP replaces discrete Bellman recursion with differentiable value functions that satisfy the **Hamilton–Jacobi–Bellman (HJB)** equations.

Key contributions include:

1. Leveraging **Physics-Informed Neural Networks (PINNs)** to approximate differential value functions governed by HJB PDEs.
2. Introducing a **Value Gradient Iteration (VGI)** module to refine gradient estimation and stabilize value learning.
3. Evaluating VIP on continuous-time versions of MPE and multi-agent MuJoCo environments, demonstrating improved sample efficiency and performance over existing continuous-time baselines (e.g., ODE-based RL, HJBPPO, DPI, and IPI).

### Strengths
* **Timely and ambitious topic:**
  The paper tackles the underexplored area of continuous-time MARL, aiming to bridge PDE-based control theory and deep RL.

* **Interesting methodological combination:**
  Integrating PINNs into an actor–critic framework and adding a value-gradient refinement mechanism (VGI) reflects a creative blend of ideas from model-based RL and physics-informed learning.

* **Comprehensive derivations:**
  Theoretical background and mathematical formulations (e.g., derivations of HJB, VGI, convergence proofs) are detailed and rigorous.

* **Empirical effort:**
  Experiments on both MPE and multi-agent MuJoCo show that VIP achieves improved training stability and faster convergence than existing continuous-time baselines.

* **Potential significance:**
  The exploration of continuous-time MARL could inspire future research directions that better align RL with real-world physical systems and continuous control.

### Weaknesses
Despite its conceptual ambition, the paper faces several **foundational and empirical limitations**, which prevent it from reaching ICLR acceptance standards.

### (1) Motivation and conceptual clarity

The motivation for formulating MARL in continuous time is **not well-justified**.
While the authors mention irregular time intervals as a motivation, the distinction between *continuous-time Markov chains (CTMC)*, *continuous control with continuous action spaces*, and *discrete-time approximations* is **not clearly articulated**.
It remains unclear **why a CTMC-based formulation is fundamentally necessary**, and what benefits it provides over well-established discrete-time methods with sufficiently small time steps. The experimental tasks are sufficient using the discrete-time methods enough.
This conceptual gap makes the overall significance of the work questionable.

### (2) Insufficient discussion on relation to model-based MARL

The proposed VIP framework involves explicit learning of **a dynamics model** ( f_\psi(x, u) ) and a **reward model** ( r_\phi(x, u) ), followed by value and policy optimization — characteristics more aligned with **model-based RL** rather than model-free methods.
However, the paper **does not compare or discuss** how VIP relates to representative **model-based multi-agent RL** or **model predictive control (MPC)** approaches (e.g., MA-MBRL, MA-MPC, or recent differential game solvers).
This omission weakens the positioning of the work.

### (3) Limited baseline comparisons

Experimental baselines are limited to other continuous-time algorithms (e.g., HJBPPO, DPI, IPI).
However, for a fair evaluation in MARL, it is crucial to include **widely recognized baselines** such as **SMPE**[1], **Revisiting Off-policy MARL**[2], and **RACE**[3], which remain standard for multi-agent benchmarks like MPE and MuJoCo.
Without such comparisons, it is difficult to assess whether improvements stem from the continuous-time formulation or from differences in model complexity.

[1]SMPE: Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration. ICML 2025

[2]Revisiting Cooperative Off-Policy Multi-Agent Reinforcement Learning. ICML 2025

[3]RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution. ICML 2023



### (4) Lack of intuitive explanation and algorithmic clarity

The paper is mathematically dense but lacks intuitive explanation.
For instance, the **motivation and role of VGI** are underexplained in the main text, and there is little visualization or step-by-step discussion of the **training procedure**, **computational cost**, or **failure cases**.
Similarly, the algorithm section lacks a high-level intuition connecting the PDE-based reasoning with practical RL optimization.
Readers unfamiliar with HJB theory or PINNs may find it hard to grasp the conceptual novelty.

### (5) Writing and presentation quality

Although mathematically complete, the paper suffers from **dense notation, unclear transitions, and insufficient narrative coherence**.
Sections often read like technical reports rather than structured scientific arguments.
The connection between the theoretical motivation and empirical evidence needs stronger integration.

While the paper presents an interesting integration of **PINNs** and **continuous-time MARL**, it lacks clear motivation, comprehensive baseline coverage, and intuitive exposition.
The conceptual distinction between continuous-time and discrete-time MARL remains ambiguous, and the empirical validation does not convincingly demonstrate broad significance.
Substantial revisions — including stronger justification, clearer writing, and broader experimental evaluation — are necessary before this work could be considered for publication at ICLR.

### Questions
To make the paper stronger and clearer, the following points should be addressed:

1. **Motivation of continuous-time formulation:**
   Why is the continuous-time formulation superior to a fine-grained discrete approximation?
   Can you clarify the distinction between CTMC and continuous-action control tasks?

2. **Connection to model-based MARL:**
   Since VIP involves explicit modeling of ( f_\psi ) and ( r_\phi ), how does it differ from model-based methods such as MA-MBRL or MA-MPC?
   A comparative discussion (or experiment) is needed.

3. **Experimental coverage:**
   Can you include or at least discuss results against mainstream MARL baselines (e.g., SMPE, Revisiting Off-policy MARL, MAPPO)?
   This would better contextualize the practical relevance of your approach.

4. **Algorithmic intuition:**
   Could you provide an intuitive explanation or figure illustrating how the VGI module refines the gradient flow and why it is necessary for training stability?

5. **Scalability and computation:**
   What is the computational overhead of solving the PINN-based HJB equations in multi-agent systems?
   Can this method scale to more than 10 agents or to real-time control?

6. **Clarify novelty:**
   Many parts of the method (PINN + HJB + actor-critic) exist in prior continuous-time single-agent works.
   Please clearly highlight what is *new* in your multi-agent extension, both conceptually and technically.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper proposes a Continuous-Time Mutli-Agent RL algorithm based on Physics-Informed Neural Networks (PINN). The algorithm, called Value Iteration via PINN (VIP), employs Value Gradient Iteration to estimate the gradient of the value function represented as HJB PDEs. A model of the environment and a reward model are trained alongside policy and value function, and are used for estimating the gradients for value function and expected advantage with respect to the critic and policy parameters respectively. Value gradient iteration, a method for refining value function gradient estimates, is introduced and used for computing an additional target for the value function. The algorithm is evaluated empirically and shown to be competitive with a diverse set of recently introduced methods. Finally, an ablation assesses the effect of individual design choices.

### Strengths
The paper has a great buildup with excellent motivation including a succinct discussion of related work. Problem formulation and other preliminaries are presented cleary. The central theoretical claims all come with a proof. The value gradient iteration (VGI) method introduced in this paper may be of broad interest for CT-RL in general. Utilizing the differential nature of PINNs in order to train them on more refined targets that include the estimated gradient is a great application of this kind of model. A diverse set of environments is used for evaluation and the experimental results look promising. The effectiveness of VGI is shown empirically on a toy example.

Overall, the paper was very pleasant to read and the contribution to the ICLR community is likely very signficant.

### Weaknesses
No discrete-time RL baseline was included in the experiments. Y-axis scaling in some of the plots make them difficult to read.

### Questions
- The results on the walker environment show a significantly higher cumulative reward for VIP when compared to the baselines. Moreover, the cumulative reward drops significantly with further training. Can you explain this?
- For the cooperative navigation task many of the compared algorithms show a very large variance which implies that for some seeds the performance is much better. Can you elaborate on that?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper extends prior work in continuous-time reinforcement learning (CTRL) from single-agent settings to cooperative multi-agent reinforcement learning (MARL). The authors focus on overcoming the challenges of large state dimensionality and accurate value approximation in continuous domains.
Their proposed framework combines Physics-Informed Neural Networks (PINNs) with a new Value Gradient Iteration (VGI) module.

- PINNs are used to approximate the viscosity solution of the Hamilton–Jacobi–Bellman (HJB) equation in continuous time, addressing the curse of dimensionality without explicit discretization.

- The VGI module refines value gradients through a recursive, single-step Bellman-like update, which improves stability and accuracy in policy learning.


The paper presents both theoretical analysis (via a contraction-based convergence argument) and empirical validation on continuous-time adaptations of the Multi-Agent Particle Environment (MPE) and Multi-Agent MuJoCo benchmarks.

### Strengths
- Scalability to high-dimensional systems: The integration of PINNs with VGI addresses the curse of dimensionality in continuous-time settings, enabling effective learning in systems with large state spaces. To my knowledge, VGI module is a novel contribution 
- Benchmarking contribution: The authors extend two widely-used discrete MARL benchmarks—Multi-Agent Particle Environment (MPE) and Multi-Agent MuJoCo—to continuous-time settings.

### Weaknesses
I have previously reviewed this paper for another conference, and while the authors addressed some feedback in this version, several substantive issues remain unaddressed, which I list here:

- **Scope is limited to Cooperative MARL & lacks comparison with relevant baselines.** Although framed as a CT-MARL approach, the method applies only to cooperative continuous-time MARL scenarios. It does not handle competitive or mixed settings, where agents have conflicting objectives, which would require solving Hamilton–Jacobi–Isaacs (HJI) equations rather than HJB equations. This limits the generality of the contribution—especially because many real-world multi-agent systems involve competition or partial cooperation. Moreover, in the cooperative case, the learning problem effectively reduces to solving a single centralized control problem with a large joint state space. This raises the question: Is the proposed method genuinely multi-agent in nature, or simply a high-dimensional continuous control solver using a centralized critic? Thus, in my understanding, the high dimensionality of the state could also be an issue in CT-RL. This raises the need for a more thorough comparison with single-agent CT-RL methods.

- **Incremental Theoretical Contribution.** The convergence result (Theorem 3.4) relies on a standard contraction mapping argument under bounded Jacobians and time-invariant dynamics. While mathematically sound, it does not introduce new theoretical insight specific to MARL or continuous-time value iteration. The proof does not incorporate any properties unique to cooperative or decentralized MARL (such as non-stationarity, joint-policy coupling, or inter-agent coordination). Thus, it serves more as a consistency check, rather than a result illuminating how the number of agents or their interactions affect stability or learning dynamics.

- **Motivation and Framing**. The motivation is overall clear—discrete-time MARL algorithms degrade under variable time steps—but the experimental evidence for this claim remains limited. Figure 1 provides a useful visual demonstration, but it’s unclear whether the performance drop of discrete methods (e.g., MADDPG) reflects fundamental limitations of discrete-time updates or simply under-tuned baselines; see next comment. This concern was raised by multiple NeurIPS reviewers and remains only partially addressed.

- **Methodological Comment**. It would strengthen the empirical evaluation if the proposed method were also tested on standard discrete-time benchmarks where the baseline algorithms (e.g., MAPPO, MADDPG) are known to perform well and reproducibly, ideally reproducing reported results from the original papers. This would help disentangle whether the observed advantages stem from the continuous-time formulation itself or from known optimization and reproducibility issues in discrete MARL. Since the proposed approach is theoretically a continuous-time generalization of Bellman updates, it should in principle also perform competitively in discrete settings when the time step is small.

- **Limited Experiments.** In summary, the empirical comparison still omits several key baselines and comparative analyses: no competitive or stochastic control baselines, no detailed compute or resource comparison (training time, memory usage), discrete-time baselines (MADDPG, MAPPO, MATD3) are still missing or not well-tuned for variable time steps.

Overall, the authors should better articulate what is new in the multi-agent context and/or appropriately compare the proposed method.


## Minor Comments


- In 3.4.2, they still mention the use of terminal-condition losses, while they don’t explicitly exist in equation (17) anymore, which makes sense if the paper now addresses the infinite horizon setup that makes it not ideal to compute such a loss. This should be clarified—either by explicitly removing the term from the text or explaining how those losses were absorbed into the loss formulation.
- The introduction should clearly state that the method applies only to deterministic multi-agent systems.
- a brief note distinguishing the smooth vs. viscosity case would help prevent confusion.

### Questions
1. Can you provide empirical or analytical evidence that the method provides benefits specifically because of its multi-agent nature (e.g., coordination effects), rather than just being a strong centralized continuous control solver?
2. Computational Cost: Could you report the relative compute cost (training time, memory, convergence rate) of VIP compared to DPI, IPI, or ODE baselines?
3. Stochastic Dynamics: Would your framework still hold if small stochastic noise were introduced in the dynamics (i.e., moving toward stochastic differential equations)?
If not, how sensitive is the PINN–VGI coupling to such noise?
4. Value Gradient Iteration Stability: The VGI recursion resembles a single-step gradient rollout. Does it ever introduce instability or oscillation during training? Are there hyperparameters controlling its contraction rate?

### Soundness
2

### Presentation
2

### Contribution
2
