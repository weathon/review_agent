

{0}------------------------------------------------

# STABILIZING REINFORCEMENT LEARNING IN DIFFERENTIABLE MULTIPHYSICS SIMULATION

Eliot Xing & Vernon Luk & Jean Oh

Carnegie Mellon University

{etaoxing, vluk, jeanooh}@cmu.edu

## ABSTRACT

Recent advances in GPU-based parallel simulation have enabled practitioners to collect large amounts of data and train complex control policies using deep reinforcement learning (RL), on commodity GPUs. However, such successes for RL in robotics have been limited to tasks sufficiently simulated by fast rigid-body dynamics. Simulation techniques for soft bodies are comparatively several orders of magnitude slower, thereby limiting the use of RL due to sample complexity requirements. To address this challenge, this paper presents both a novel RL algorithm and a simulation platform to enable scaling RL on tasks involving rigid bodies and deformables. We introduce Soft Analytic Policy Optimization (SAPO), a maximum entropy first-order model-based actor-critic RL algorithm, which uses first-order analytic gradients from differentiable simulation to train a stochastic actor to maximize expected return and entropy. Alongside our approach, we develop Rewarped, a parallel differentiable multiphysics simulation platform that supports simulating various materials beyond rigid bodies. We re-implement challenging manipulation and locomotion tasks in Rewarped, and show that SAPO outperforms baselines over a range of tasks that involve interaction between rigid bodies, articulations, and deformables. Additional details at [rewarped.github.io](http://rewarped.github.io).

## 1 INTRODUCTION

Progress in deep reinforcement learning (RL) has produced policies capable of impressive behavior, from playing games with superhuman performance (Silver et al., 2016; Vinyals et al., 2019) to controlling robots for assembly (Tang et al., 2023), dexterous manipulation (Andrychowicz et al., 2020; Akkaya et al., 2019), navigation (Wijmans et al., 2020; Kaufmann et al., 2023), and locomotion (Rudin et al., 2021; Radosavovic et al., 2024). However, standard model-free RL algorithms are extremely sample inefficient. Thus, the main practical bottleneck when using RL is the cost of acquiring large amounts of training data.

To scale data collection for online RL, prior works developed distributed RL frameworks (Nair et al., 2015; Horgan et al., 2018; Espeholt et al., 2018) that run many processes across a large compute cluster, which is inaccessible to most researchers and practitioners. More recently, GPU-based parallel environments (Dalton et al., 2020; Freeman et al., 2021; Liang et al., 2018; Makoviychuk et al., 2021; Mittal et al., 2023; Gu et al., 2023) have enabled training RL at scale on a single consumer GPU.

However, such successes of scaling RL in robotics have been limited to tasks sufficiently simulated by fast rigid-body dynamics (Makoviychuk et al., 2021), while physics-based simulation techniques for soft bodies are comparatively several orders of magnitude slower. Consequently for tasks involving deformable objects, such as robotic manipulation of rope (Nair et al., 2017; Chi et al., 2022), cloth (Ha & Song, 2022; Lin et al., 2022), elastiks (Shen et al., 2022), liquids (Ichnowski et al., 2022; Zhou et al., 2023), dough (Shi et al., 2022; 2023; Lin et al., 2023), or granular piles (Wang et al., 2023; Xue et al., 2023), approaches based on motion planning, trajectory optimization, or model predictive control have been preferred over and outperform RL (Huang et al., 2020; Chen et al., 2022).

How can we overcome this data bottleneck to scaling RL on tasks involving deformables? Model-based reinforcement learning (MBRL) has shown promise at reducing sample complexity, by leveraging some known model or learning a world model to predict environment dynamics and rewards (Morland et al., 2023). In contrast to rigid bodies however, soft bodies have more complex dynamics

{1}------------------------------------------------

![Figure 1: Visualizations of tasks implemented in Rewarped. The figure shows six panels arranged in a 2x3 grid. Top row: AntRun (an orange ant-like robot walking), HandReorient (a robotic hand reorienting a blue and yellow object), RollingFlat (a yellow object rolling on a flat surface). Bottom row: SoftJumper (a pink object jumping), HandFlip (a robotic hand flipping a white object), and FluidMove (a blue object moving through a fluid-like environment).](c803f6f6e2c49429d2951832bd0f208d_img.jpg)

Figure 1: Visualizations of tasks implemented in Rewarped. The figure shows six panels arranged in a 2x3 grid. Top row: AntRun (an orange ant-like robot walking), HandReorient (a robotic hand reorienting a blue and yellow object), RollingFlat (a yellow object rolling on a flat surface). Bottom row: SoftJumper (a pink object jumping), HandFlip (a robotic hand flipping a white object), and FluidMove (a blue object moving through a fluid-like environment).

**Figure 1: Visualizations of tasks implemented in Rewarped.** These are manipulation and locomotion tasks involving rigid and soft bodies. AntRun and HandReorient are tasks with articulated rigid bodies, while RollingFlat, SoftJumper, HandFlip, and FluidMove are tasks with deformables.

and higher-dimensional state spaces. This makes learning to model dynamics of deformables highly nontrivial (Lin et al., 2021), often requiring specialized systems architecture and material-specific assumptions such as volume preservation or connectivity.

Recent developments in differentiable physics-based simulators of deformables (Hu et al., 2019b; Du et al., 2021; Huang et al., 2020; Zhou et al., 2023; Wang et al., 2024; Liang et al., 2019; Qiao et al., 2021a; Li et al., 2022b; Heiden et al., 2023) have shown that first-order gradients from differentiable simulation can be used for gradient-based trajectory optimization and achieve low sample complexity. Yet such approaches are sensitive to initial conditions and get stuck in local optima due to non-smooth optimization landscapes or discontinuities induced by contacts (Li et al., 2022a; Antonova et al., 2023). Additionally, existing soft-body simulations are not easily parallelized, which limits scaling RL in them. Overall, there is no existing simulation platform that is parallelized, differentiable, and supports interaction between articulated rigid bodies and deformables.

In this paper, we approach the sample efficiency problem using first-order model-based RL (FO-MBRL), which leverages first-order analytic gradients from differentiable simulation to accelerate policy learning, without explicitly learning a world model. Thus far, FO-MBRL has been shown to achieve low sample complexity on articulated rigid-body locomotion tasks (Freeman et al., 2021; Xu et al., 2021), but has not yet been shown to work well for tasks involving deformables (Chen et al., 2022). We hypothesize that entropy regularization can stabilize policy optimization over analytic gradients from differentiable simulation, such as by smoothing the optimization landscape (Ahmed et al., 2019). To this end, we introduce a novel maximum entropy FO-MBRL algorithm, alongside a parallel differentiable multiphysics simulation platform for RL.

**Contributions.** **i)** We introduce Soft Analytic Policy Optimization (SAPO), a first-order MBRL algorithm based on the maximum entropy RL framework. We formulate SAPO as an on-policy actor-critic RL algorithm, where a stochastic actor is trained to maximize expected return and entropy using first-order analytic gradients from differentiable simulation. **ii)** We present Rewarped, a scalable and easy-to-use platform which enables parallelizing RL environments of GPU-accelerated differentiable multiphysics simulation and supports various materials beyond rigid bodies. **iii)** We demonstrate that parallel differentiable simulation enables SAPO to outperform baselines over a range of challenging manipulation and locomotion tasks re-implemented using Rewarped that involve interaction between rigid bodies, articulations, and deformables such as elastic, plasticine, or fluid materials.

## 2 RELATED WORK

We refer the reader to (Newbury et al., 2024) for an overview of differentiable simulation. We cover *non-parallel* differentiable simulation and model-based RL in Appendix A.

{2}------------------------------------------------

| Simulator | $\nabla^2$ ? | Materials? |  |  |  |  |
|-|-|-|-|-|-|-|
|  |  | Rigid | Articulated | Elastic | Plasticine | Fluid |
| Isaac Gym | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Isaac Lab / Orbit | ✗ | ✓ | ✓ | ✓* | ✗ | ✓* |
| ManiSkill | ✗ | ✓ | ✓ | ✗ | ✓* | ✓* |
| TinyDiffSim | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Brax | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MJX | ✓* | ✓ | ✓ | ✗ | ✗ | ✗ |
| DaXBenCh | ✓ | ✓ | ✗ | ✗ | ✓* | ✓ |
| DFlex | ✓ | ✓ | ✓ | ✓* | ✗ | ✗ |
| Rewarped (ours) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

*Table 1: Comparison of physics-based parallel simulation platforms for RL.* We use \* to indicate incomplete feature support at the time of writing. i) [Isaac Lab / Orbit](#): Missing deformable tasks due to breaking API changes and poor simulation stability / scaling. ii) [ManiSkill](#): The latest version ManiSkill3 does not yet support the soft body tasks introduced in v2. iii) [MJX](#): Stability issues with autodifferentiation and gradients. iv) [DaXBenCh](#): Plasticine task was omitted from benchmark and requires additional development. v) [DFlex](#): While later work ([Murthy et al., 2021](#); [Heiden et al., 2023](#)) has built on DFlex to support elastic and cloth materials, their simulations were not parallelized.

**Parallel differentiable simulation.** There are few prior works on parallel differentiable simulators capable of running many environments together, while also computing simulation gradients in batches. [TinyDiffSim](#) ([Heiden et al., 2021](#)) implements articulated rigid-body dynamics and contact models in C++/CUDA that can integrate with various autodifferentiation libraries. [Brax](#) ([Freeman et al., 2021](#)) implements a parallel simulator in JAX for articulated rigid-body dynamics with simple collision shape primitives. Recently, MJX is building on Brax to provide a JAX re-implementation of MuJoCo ([Todorov et al., 2012](#)), a physics engine widely used in RL and robotics, but does not have feature parity with MuJoCo yet. These aforementioned parallel differentiable simulators are only capable of modeling articulated rigid bodies. [DaXBenCh](#) ([Chen et al., 2022](#)) also uses JAX to enable fast parallel simulation of deformables such as rope and liquid by Material Point Method (MPM) or cloth by mass-spring systems, but does not support articulated rigid bodies. [DFlex](#) ([Xu et al., 2021](#)) presents a differentiable simulator based on source-code transformation ([Griewank & Walther, 2008](#); [Hu et al., 2020](#)) of simulation kernel code to C++/CUDA, that integrates with PyTorch for tape-based autodifferentiation. [Xu et al. \(2021\)](#) use DFlex for parallel simulation of articulated rigid bodies for high-dimensional locomotion tasks. Later work ([Murthy et al., 2021](#); [Heiden et al., 2023](#)) also used DFlex to develop differentiable simulations of cloth and elastic objects, but these were not parallelized and did not support interaction with articulated rigid bodies. To the best of our knowledge, there is no existing differentiable simulation platform that is parallelized with multiphysics support for interaction between rigid bodies, articulations, and various deformables. In this paper, we aim to close this gap with Rewarped, our platform for parallel differentiable multiphysics simulation, and in Table 1 we compare Rewarped against existing physics-based parallel simulation platforms.

**Learning control with differentiable physics.** Gradient-based trajectory optimization is commonly used with differentiable simulation of soft bodies ([Hu et al., 2019b](#); [2020](#); [Huang et al., 2020](#); [Li et al., 2023a](#); [Zhou et al., 2023](#); [Wang et al., 2024](#); [Si et al., 2024](#); [Du et al., 2021](#); [Li et al., 2022b](#); [Rojas et al., 2021](#); [Qiao et al., 2020](#); [2021a](#); [Liu et al., 2024](#); [Chen et al., 2022](#); [Heiden et al., 2023](#)). Differentiable physics can provide physical priors for control in end-to-end learning systems, such as for quadruped locomotion ([Song et al., 2024](#)), drone navigation ([Zhang et al., 2024](#)), robot painting ([Schaldenbrand et al., 2023](#)), or motion imitation ([Ren et al., 2023](#)). Gradients from differentiable simulation can also be directly used for policy optimization. PODS ([Zamora et al., 2021](#)) proposes a first and second order policy improvement based on analytic gradients of a value function with respect to the policy’s action outputs. APG ([Freeman et al., 2021](#)) uses analytic simulation gradients to directly compute policy gradients. SHAC ([Xu et al., 2021](#)) presents an actor-critic algorithm, where the actor is optimized over a short horizon using analytic gradients, and a terminal value function helps smooth the optimization landscape. AHAC ([Georgiev et al., 2024](#)) modifies SHAC to adjust the policy horizon by truncating stiff contacts based on contact forces or the norm of the dynamics Jacobian. Several works also propose different ways to overcome bias and non-smooth dynamics resulting from contacts, by reweighting analytic gradients ([Gao et al., 2024](#); [Son et al., 2024](#)) or

{3}------------------------------------------------

explicit smoothing (Suh et al., 2022; Zhang et al., 2023; Schwartke et al., 2024). In this work, we propose a maximum entropy FO-MBRL algorithm to stabilize policy learning with gradients from differentiable simulation.

## 3 BACKGROUND

**Reinforcement learning** (RL) considers an agent interacting with an environment, formalized as a Markov decision process (MDP) represented by a tuple  $(\mathcal{S}, \mathcal{A}, P, R, \rho_0, \gamma)$ . In this work, we consider discrete-time, infinite-horizon MDPs with continuous action spaces, where  $\mathbf{s} \in \mathcal{S}$  are states,  $\mathbf{a} \in \mathcal{A}$  are actions,  $P : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$  is the transition function,  $R : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$  is a reward function,  $\rho_0(\mathbf{s})$  is an initial state distribution, and  $\gamma$  is the discount factor. We want to obtain a policy  $\pi : \mathcal{S} \rightarrow \mathcal{A}$  which maximizes the expected discounted sum of rewards (return)  $\mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t r_t]$  with  $r_t = R(\mathbf{s}_t, \mathbf{a}_t)$ , starting from state  $\mathbf{s}_0 \sim \rho_0$ . We also denote the state distribution  $\rho_\pi(\mathbf{s})$  and state-action distribution  $\rho_\pi(\mathbf{s}, \mathbf{a})$  for trajectories generated by a policy  $\pi(\mathbf{a}_t|\mathbf{s}_t)$ .

In practice, the agent interacts with the environment for  $T$  steps in a finite-length episode, yielding a trajectory  $\tau = (\mathbf{s}_0, \mathbf{a}_0, \mathbf{s}_1, \mathbf{a}_1, \dots, \mathbf{s}_{T-1}, \mathbf{a}_{T-1})$ . We can define the  $H$ -step return :

$$R_{0:H}(\tau) = \sum_{t=0}^{H-1} \gamma^t r_t, \quad (1)$$

and standard RL objective to optimize  $\theta$  parameterizing a policy  $\pi_\theta$  to maximize the expected return :

$$J(\pi) = \mathbb{E}_{\mathbf{s}_0 \sim \rho_0, \tau \sim \rho_\pi} [R_{0:T}]. \quad (2)$$

Typically, the policy gradient theorem (Sutton et al., 1999) provides a useful expression of  $\nabla_\theta J(\pi)$  that does not depend on the derivative of state distribution  $\rho_\pi(\cdot)$  :

$$\nabla_\theta J(\pi) \propto \int_{\mathcal{S}} \rho_\pi(\mathbf{s}) \int_{\mathcal{A}} \nabla_\theta \pi(\mathbf{a}|\mathbf{s}) Q^\pi(\mathbf{s}, \mathbf{a}) d\mathbf{a} d\mathbf{s}, \quad (3)$$

where  $Q^\pi(\mathbf{s}_t, \mathbf{a}_t) = \mathbb{E}_{\tau \sim \rho_\pi} [R_{t:T}]$  is the  $Q$ -function (state-action value function).

We proceed to review zeroth-order versus first-order estimators of the policy gradient following the discussion in (Suh et al., 2022; Georgiev et al., 2024). We denote a single zeroth-order estimate :

$$\hat{\nabla}_\theta^{[0]} J(\pi) = R_{0:T} \sum_{t=0}^{T-1} \nabla_\theta \log \pi(\mathbf{a}_t|\mathbf{s}_t), \quad (4)$$

where the zeroth-order batched gradient (ZOBG) is the sample mean  $\bar{\nabla}_\theta^{[0]} J(\pi) = \frac{1}{N} \sum_{i=1}^N \hat{\nabla}_\theta^{[0]} J(\pi)$  and is an unbiased estimator, under some mild assumptions to ensure the gradients are well-defined. The ZOBG yields an  $N$ -sample Monte-Carlo estimate commonly known as the REINFORCE estimator (Williams, 1992) in RL literature, or the score function / likelihood-ratio estimator. Policy gradient methods may use different forms of Equation 4 to adjust the bias and variance of the estimator (Schulman et al., 2015b). For instance, a baseline term can be used to reduce variance of the estimator, by substituting  $R_{0:T}$  with  $R_{0:T} - R_{t:H+t}$ .

**Differentiable simulation** as the environment provides gradients for the transition dynamics  $P$  and rewards  $R$ , so we can directly obtain an analytic value for  $\nabla_\theta R_{0:T}$  under policy  $\pi_\theta$ . In this setting, for a single first-order estimate :

$$\hat{\nabla}_\theta^{[1]} J(\pi) = \nabla_\theta R_{0:T}, \quad (5)$$

then the first-order batched gradient (FOBG) is the sample mean  $\bar{\nabla}_\theta^{[1]} J(\pi) = \frac{1}{N} \sum_{i=1}^N \hat{\nabla}_\theta^{[1]} J(\pi)$ , and is also known as the pathwise derivative (Schulman et al., 2015a) or reparameterization trick (Kingma & Welling, 2014; Rezende et al., 2014; Titsias & Lázaro-Gredilla, 2014).

**First-order model-based RL** (FO-MBRL) aims to use differentiable simulation (and its first-order analytic gradients) as a known differentiable model, in contrast to vanilla MBRL which either assumes a given non-differentiable model or learns a world model of dynamics and rewards from data.

{4}------------------------------------------------

**Analytic Policy Gradient** (APG, [Freeman et al. \(2021\)](#)) uses FOBG estimates to directly maximize the discounted return over a truncated horizon :

$$J(\pi) = \sum_{l=t}^{t+H-1} \mathbb{E}_{(s_l, a_l) \sim \rho_\pi} [\gamma^{l-t} r_l], \quad (6)$$

and is also referred to as Backpropagation Through Time (BPTT, [Werbos \(1990\)](#); [Mozer \(1995\)](#)), particularly when the horizon is the full episode length ([Degrave et al., 2019](#); [Huang et al., 2020](#)).

**Short-Horizon Actor-Critic** (SHAC, [Xu et al. \(2021\)](#)) is a FO-MBRL algorithm which learns a policy  $\pi_\theta$  and (terminal) value function  $V_\psi$  :

$$J(\pi) = \sum_{l=t}^{t+H-1} \mathbb{E}_{(s_l, a_l) \sim \rho_\pi} [\gamma^{l-t} r_l + \gamma^t V(s_{t+H})], \quad (7)$$

$$\mathcal{L}(V) = \sum_{l=t}^{t+H-1} \mathbb{E}_{s_l \sim \rho_\pi} [\|V(s) - \tilde{V}(s)\|^2], \quad (8)$$

where  $\tilde{V}(s_t)$  are value estimates for state  $s_t$  computed starting from time step  $t$  over an  $H$ -step horizon. TD( $\lambda$ ) ([Sutton, 1988](#)) is used for value estimation, which computes  $\lambda$ -returns  $G_{t:t+H}^\lambda$  as a weighted average of value-bootstrapped  $k$ -step returns  $G_{t:t+k}$  :

$$\tilde{V}(s_t) = G_{t:t+H}^\lambda = (1 - \lambda) \left( \sum_{l=1}^{H-1-t} \lambda^{l-1} G_{t:t+l} \right) + \lambda^{H-t-1} G_{t:t+H}, \quad (9)$$

where  $G_{t:t+k} = \left( \sum_{l=0}^{k-1} \gamma^l r_{t+l} \right) + \gamma^k V(s_{t+k})$ . The policy and value function are optimized in an alternating fashion per standard actor-critic formulation ([Konda & Tsitsiklis, 1999](#)). The policy gradient is obtained by FOBG estimation, with single first-order estimate :

$$\hat{\nabla}_\theta^{[1]} J(\pi) = \nabla_\theta (R_{0:H} + \gamma^H V(s_H)), \quad (10)$$

and the value function is optimized as usual by backpropagating  $\nabla_\psi \mathcal{L}(V)$  of the mean-squared loss in Eq. 8. Combining value estimation with a truncated short-horizon window where  $H \ll T$  ([Williams & Zipser, 1995](#)), SHAC optimizes over a smoother surrogate reward landscape compared to BPTT over the entire  $T$ -step episode.

## 4 SOFT ANALYTIC POLICY OPTIMIZATION (SAPO)

Empirically we observe that SHAC, a state-of-the-art FO-MBRL algorithm, is still prone to suboptimal convergence to local minima in the reward landscape (Appendix, Figure 5). We hypothesize that entropy regularization can stabilize policy optimization over analytic gradients from differentiable simulation, such as by smoothing the optimization landscape ([Ahmed et al., 2019](#)) or providing robustness under perturbations ([Eysenbach & Levine, 2022](#)).

We draw on the maximum entropy RL framework ([Kappen, 2005](#); [Todorov, 2006](#); [Ziebart et al., 2008](#); [Toussaint, 2009](#); [Theodorou et al., 2010](#); [Haarnoja et al., 2017](#)) to formulate Soft Analytic Policy Optimization (SAPO), a maximum entropy FO-MBRL algorithm (Section 4.1). To implement SAPO, we make several design choices, including modifications building on SHAC (Section 4.2). In Appendix B.1, we describe how we use visual encoders to learn policies from high-dimensional visual observations in differentiable simulation. Pseudocode for SAPO is shown in Appendix B.2, and the computational graph of SAPO is illustrated in Appendix Figure 4.

### 4.1 MAXIMUM ENTROPY RL IN DIFFERENTIABLE SIMULATION

**Maximum entropy RL** ([Ziebart et al., 2008](#); [Ziebart, 2010](#)) augments the standard (undiscounted) return maximization objective with the expected entropy of the policy over  $\rho_\pi(s_t)$  :

$$J(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r_t + \alpha \mathcal{H}_\pi[a_t | s_t]], \quad (11)$$

{5}------------------------------------------------

where  $\mathcal{H}_\pi[a_t|s_t] = -\int_{\mathcal{A}} \pi(a_t|s_t) \log \pi(a_t|s_t) da_t$  is the continuous Shannon entropy of the action distribution, and the temperature  $\alpha$  balances the entropy term versus the reward.

Incorporating the discount factor (Thomas, 2014; Haarnoja et al., 2017), we obtain the following objective which maximizes the expected return and entropy for future states starting from  $(s_t, a_t)$  weighted by its probability  $\rho_\pi$  under policy  $\pi$ :

$$J_{\maxent}(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ \sum_{l=t}^{\infty} \gamma^{l-t} \mathbb{E}_{(s_l, a_l) \sim \rho_\pi} [r_l + \alpha \mathcal{H}_\pi[a_l|s_l]] \right]. \quad (12)$$

The soft  $Q$ -function is the expected value under  $\pi$  of the discounted sum of rewards and entropy:

$$Q_{\text{soft}}^\pi(s_t, a_t) = r_t + \mathbb{E}_{(s_{t+1}, \dots) \sim \rho_\pi} \left[ \sum_{l=t+1}^{\infty} \gamma^l (r_l + \alpha \mathcal{H}_\pi[a_l|s_l]) \right], \quad (13)$$

and the soft value function is:

$$V_{\text{soft}}^\pi(s_t) = \alpha \log \int_{\mathcal{A}} \exp\left(\frac{1}{\alpha} Q_{\text{soft}}^\pi(s, a)\right) da. \quad (14)$$

When  $\pi(a|s) = \exp(\frac{1}{\alpha}(Q_{\text{soft}}^\pi(s, a) - V_{\text{soft}}^\pi(s))) \triangleq \pi^*$ , then the soft Bellman equation yields the following relationship:

$$Q_{\text{soft}}^\pi(s_t, a_t) = r_t + \gamma \mathbb{E}_{(s_{t+1}, \dots) \sim \rho_\pi} [V_{\text{soft}}^\pi(s_{t+1})], \quad (15)$$

where we can rewrite the discounted maximum entropy objective in Eq. 12:

$$J_{\maxent}(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [Q_{\text{soft}}^\pi(s, a) + \alpha \mathcal{H}_\pi[a_t|s_t]] \quad (16)$$

$$= \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r_t + \alpha \mathcal{H}_\pi[a_t|s_t] + \gamma V_{\text{soft}}^\pi(s_{t+1})]. \quad (17)$$

By Soft Policy Iteration (Haarnoja et al., 2018a), the soft Bellman operator  $\mathcal{B}^*$  defined by  $(\mathcal{B}^*Q)(s_t, a_t) = r_t + \gamma \mathbb{E}_{s_{t+1} \sim \rho_\pi} [V(s_{t+1})]$  has a unique contraction  $Q^* = \mathcal{B}^*Q^*$  (Fox et al., 2016) and converges to the optimal policy  $\pi^*$ .

**Our main observation** is when the environment is a differentiable simulation, we can use FOBG estimates to directly optimize  $J_{\maxent}(\pi)$ , including discounted policy entropy. Consider the entropy-augmented  $H$ -step return:

$$R_{0:H}^\alpha(\tau) = \sum_{t=0}^{H-1} \gamma^t (r_t + \alpha \mathcal{H}_\pi[a_t|s_t]), \quad (18)$$

then we have a single first-order estimate of Eq. 17:

$$\nabla_\theta^{[1]} J_{\maxent}(\pi) = \nabla_\theta (R_{0:H}^\alpha + \gamma^H V_{\text{soft}}(s_H)). \quad (19)$$

Furthermore, we can incorporate the entropy-augmented return into  $TD(\lambda)$  estimates of Eq. 9 using soft value-bootstrapped  $k$ -step returns:

$$\Gamma_{t:t+k} = \left( \sum_{l=0}^{k-1} \gamma^l (r_{t+l} + \alpha \mathcal{H}_\pi[a_{t+l}|s_{t+l}]) \right) + \gamma^k V_{\text{soft}}(s_{t+k}), \quad (20)$$

where  $\tilde{V}_{\text{soft}}(s_t) = \Gamma_{t:t+H}^\lambda$ , and the value function is trained by minimizing Eq. 8 with  $V_{\text{soft}}$ ,  $\tilde{V}_{\text{soft}}$ , and  $\Gamma_{t:t+k}$  substituted in place of  $V$ ,  $\tilde{V}$ , and  $G_{t:t+k}$ . We refer to this maximum entropy FO-MBRL formulation as **Soft Analytic Policy Optimization (SAPO)**.

Note that we instantiate SAPO as an actor-critic algorithm that learns the soft value function by TD learning with on-policy data. In comparison, Soft Actor-Critic (SAC), a popular off-policy maximum entropy model-free RL algorithm, estimates soft  $Q$ -values by minimizing the soft Bellman residual with data sampled from a replay buffer. Connections may also drawn between SAPO to a maximum entropy variant of  $SVG(H)$  (Heess et al., 2015; Amos et al., 2021), which uses rollouts from a learned world model instead of trajectories from differentiable simulation.

{6}------------------------------------------------

### 4.2 DESIGN CHOICES

**I. Entropy adjustment.** In practice, we apply automatic temperature tuning (Haarnoja et al., 2018b) to match a target entropy  $\bar{\mathcal{H}}$  via an additional Lagrange dual optimization step :

$$\min_{\alpha_t \geq 0} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \{\alpha_t (\mathcal{H}_\pi[a_t|s_t] - \bar{\mathcal{H}})\}. \quad (21)$$

We use  $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A})/2$  following (Ball et al., 2023).

**II. Target entropy normalization.** To mitigate non-stationarity in target values (Yu et al., 2022) and improve robustness across tasks with varying reward scales and action dimensions, we normalize entropy estimates. The continuous Shannon entropy is not scale invariant (Marsh, 2013). In particular, we offset (Han & Sung, 2021) and scale entropy by  $\bar{\mathcal{H}}$  to be approximately contained within  $[0, +1]$ .

**III. Stochastic policy parameterization.** We use state-dependent variance, with squashed Normal distribution  $\pi_\theta = \tanh(\mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s)))$ , which aligns with SAC (Haarnoja et al., 2018b). This enables policy entropy adjustment and captures aleatoric uncertainty in the environment (Kendall & Gal, 2017; Chua et al., 2018). In contrast, SHAC uses state-independent variance, similar to the original PPO implementation (Schulman et al., 2017).

**IV. Critic ensemble, no target networks.** We use the clipped double critic trick (Fujimoto et al., 2018) and also remove the critic target network in SHAC, similar to (Georgiev et al., 2024). However when updating the actor, we instead compute the *average* over the two value estimates to include in the return (Eq. 19), while using the *minimum* to estimate target values in standard fashion, following (Ball et al., 2023). While originally intended to mitigate overestimation bias in  $Q$ -learning (due to function approximation and stochastic optimization (Thrun & Schwartz, 2014)), prior work has shown that the value lower bound obtained by clipping can be overly conservative and cause the policy to pessimistically underexplore (Ciosek et al., 2019; Moskovitz et al., 2021).

Target networks (Mnih et al., 2015) are widely used (Lillicrap et al., 2016; Fujimoto et al., 2018; Haarnoja et al., 2018b) to stabilize temporal difference (TD) learning, at the cost of slower training. Efforts have been made to eliminate target networks (Kim et al., 2019; Yang et al., 2021; Shao et al., 2022; Gallici et al., 2024), and recently CrossQ (Bhatt et al., 2024) has shown that careful use of normalization layers can stabilize off-policy model-free RL to enable removing target networks for improved sample efficiency. CrossQ also reduces Adam  $\beta_1$  momentum from 0.9 to 0.5, while keeping the default  $\beta_2 = 0.999$ . In comparison, SHAC uses  $\beta_1 = 0.7$  and  $\beta_2 = 0.95$ . Using smaller momentum parameters decreases exponential decay (for the moving average estimates of the 1st and 2nd moments of the gradient) and effectively gives higher weight to more recent gradients, with less smoothing by past gradient history (Kingma & Ba, 2015).

**V. Architecture and optimization.** We use SiLU (Elfwing et al., 2018) instead of ELU for the activation function. We also switch the optimizer from Adam to AdamW (Loshchilov & Hutter, 2017), and lower gradient norm clipping from 1.0 to 0.5. Note that SHAC already uses LayerNorm (Ba et al., 2016), which has been shown to stabilize TD learning when not using target networks or replay buffers (Bhatt et al., 2024; Gallici et al., 2024).

## 5 REWARPED: PARALLEL DIFFERENTIABLE MULTIPHYSICS SIMULATION

We aim to evaluate our approach on more challenging manipulation and locomotion tasks that involve interaction between articulated rigid bodies and deformables. To this end, we introduce Rewarped, our parallel differentiable multiphysics simulation platform that provides GPU-accelerated parallel environments for RL and enables computing batched simulation gradients efficiently. We build Rewarped on NVIDIA Warp (Macklin, 2022), the successor to DFlex (Xu et al., 2021; Murthy et al., 2021; Turpin et al., 2022; Heiden et al., 2023).

We proceed to discuss high-level implementation details and optimization tricks to enable efficient parallel differentiable simulation. We develop a parallelized implementation of Material Point Method (MPM) which supports simulating parallel environments of complex deformable materials, building on the MLS-MPM implementation by (Ma et al., 2023) used for non-parallel simulation. Furthermore, we support one-way coupling from kinematic articulated rigid bodies to MPM particles, based on the (non-parallel) MPM-based simulation from (Huang et al., 2020; Li et al., 2023a).

{7}------------------------------------------------

### 5.1 PARALLEL DIFFERENTIABLE SIMULATION

We implement all simulation code in NVIDIA Warp (Macklin, 2022), a library for differentiable programming that converts Python code into CUDA kernels by runtime JIT compilation. Warp implements reverse-mode auto-differentiation through the discrete adjoint method, using a tape to record kernel calls for the computation graph, and generates kernel adjoints to compute the backward pass. Warp uses source-code transformation (Griewank & Walther, 2008; Hu et al., 2020) to automatically generate kernel adjoints.

We use gradient checkpointing (Griewank & Walther, 2000; Qiao et al., 2021b) to reduce memory requirements. During backpropagation, we run the simulation forward pass again to recompute intermediate values, instead of saving them during the initial forward pass. This is implemented by capturing and replaying CUDA graphs, for both the forward pass and the backward pass of the simulator. Gradient checkpointing by CUDA graphs enables us to compute batched simulation gradients over multiple time steps efficiently, when using more simulation substeps for simulation stability. We use a custom PyTorch autograd function to interface simulation data and model parameters between Warp and PyTorch while maintaining auto-differentiation functionality.

## 6 EXPERIMENTS

We evaluate our proposed maximum entropy FO-MBRL algorithm, Soft Analytic Policy Optimization (SAPO, Section 4), against baselines on a range of locomotion and manipulation tasks involving rigid and soft bodies. We implement these tasks in Rewarped (Section 5), our parallel differentiable multiphysics simulation platform. We also compare algorithms on DFlex rigid-body locomotion tasks introduced in (Xu et al., 2021) in Appendix F.2.

**Baselines.** We compare to vanilla model-free RL algorithms: Proximal Policy Optimization (PPO, Schulman et al. (2017)), an on-policy actor-critic algorithm; Soft Actor-Critic (SAC, Haarnoja et al. (2018b)) an off-policy maximum entropy actor-critic algorithm. We use the implementations and hyperparameters from (Li et al., 2023b) for both, which have been validated to scale well with parallel simulation. Implementation details (network architecture, common hyperparameters, etc.) are standardized between methods for fair comparison, see Appendix C. We also compare against Analytic Policy Gradient (APG, Freeman et al. (2021)) and Short-Horizon Actor-Critic (SHAC, Xu et al. (2021)), both of which are state-of-the-art FO-MBRL algorithms that leverage first-order analytic gradients from differentiable simulation for policy learning. Finally, we include gradient-based trajectory optimization (TrajOpt) as a baseline, which uses differentiable simulation gradients to optimize for an open-loop action sequence that maximizes total rewards across environments.

**Tasks.** Using Rewarped, we re-implement a range of challenging manipulation and locomotion tasks involving rigid and soft bodies that have appeared in prior works. Rewarped enables training algorithms on parallel environments, and differentiable simulation to compute analytic simulation gradients through environment dynamics and rewards. We visualize these tasks in Figure 1. To simulate deformables, we use  $\sim 2500$  particles per env. See Appendix E for more details.

**AntRun** – Ant locomotion task from DFlex (Xu et al., 2021), where the objective is to maximize the forward velocity of a four-legged ant rigid-body articulation.

**HandReorient** – Allegro hand manipulation task from Isaac Gym (Makoviychuk et al., 2021), where the objective is to perform in-hand dexterous manipulation to rotate a rigid cube given a target pose. We replace non-differentiable terms of the reward function (ie. boolean comparisons) with differentiable alternatives to enable computing analytic gradients.

**RollingFlat** – Rolling pin manipulation task from PlasticineLab (Huang et al., 2020), where the objective is to flatten a rectangular piece of dough using a cylindrical rolling pin.

**SoftJumper** – Soft jumping locomotion task, inspired by GradSim (Murthy et al., 2021) and DiffTaichi (Hu et al., 2020), where the objective is to maximize the forward velocity and height of a high-dimensional actuated soft elastic quadruped.

**HandFlip** – Shadow hand flip task from DexDeform (Li et al., 2023a), where the objective is to flip a cylindrical piece of dough in half within the palm of a dexterous robot hand.

**FluidMove** – Fluid transport task from SoftGym (Lin et al., 2021), where the objective is to move a container filled with fluid to a given target position, without spilling fluid out of the container.

{8}------------------------------------------------

Note that {AntRun, HandReorient} are tasks that involve articulated rigid bodies only, with state-based observations. In contrast, {RollingFlat, SoftJumper, HandFlip, FluidMove} are tasks that also involve deformables, with both state-based and high-dimensional (particle-based) visual observations.

![Figure 2: Rewarped tasks training curves. Six line plots showing Return vs. Environment Steps for AntRun, HandReorient, RollingFlat, SoftJumper, HandFlip, and FluidMove. Each plot compares SAPO (ours) in red with SHAC (yellow), APG (purple), SAC (green), and PPO (blue). Shaded regions represent 95% confidence intervals.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

Figure 2: Rewarped tasks training curves. Six line plots showing Return vs. Environment Steps for AntRun, HandReorient, RollingFlat, SoftJumper, HandFlip, and FluidMove. Each plot compares SAPO (ours) in red with SHAC (yellow), APG (purple), SAC (green), and PPO (blue). Shaded regions represent 95% confidence intervals.

**Figure 2: Rewarped tasks training curves.** Episode return as a function of environment steps in Rewarped AntRun ( $\mathcal{A} \subset \mathbb{R}^8$ ), HandReorient ( $\mathcal{A} \subset \mathbb{R}^{16}$ ), RollingFlat ( $\mathcal{A} \subset \mathbb{R}^3$ ), SoftJumper ( $\mathcal{A} \subset \mathbb{R}^{22}$ ), HandFlip ( $\mathcal{A} \subset \mathbb{R}^{24}$ ), and FluidMove ( $\mathcal{A} \subset \mathbb{R}^3$ ) tasks. Smoothed using EWMA with  $\alpha = 0.99$ . Mean and 95% CIs over 10 random seeds.

|  | AntRun | HandReorient | RollingFlat | SoftJumper | HandFlip | FluidMove |
|-|-|-|-|-|-|-|
| PPO | $2048.7 \pm 36.6$ | $5.9 \pm 4.9$ | $81.2 \pm 0.1$ | $261.5 \pm 12.4$ | $7.3 \pm 1.1$ | $27.3 \pm 0.2$ |
| SAC | $2063.6 \pm 13.9$ | $70.5 \pm 10.2$ | $83.0 \pm 0.3$ | $-161.8 \pm 2.5$ | $4.6 \pm 1.1$ | $28.2 \pm 0.7$ |
| TrajOpt | $915.5 \pm 29.6$ | $-12.5 \pm 2.0$ | $81.5 \pm 0.1$ | $437.2 \pm 17.7$ | $27.3 \pm 2.6$ | $27.0 \pm 0.1$ |
| APG | $258.7 \pm 20.3$ | $-11.6 \pm 1.9$ | $86.9 \pm 0.4$ | $956.6 \pm 15.6$ | $38.2 \pm 3.5$ | $26.3 \pm 0.3$ |
| SHAC | $3621.0 \pm 54.4$ | $-2.5 \pm 1.8$ | $86.8 \pm 0.4$ | $853.3 \pm 10.2$ | $32.7 \pm 2.9$ | $21.7 \pm 0.4$ |
| SAPO (ours) | $4535.9 \pm 24.5$ | $221.7 \pm 9.5$ | $100.4 \pm 0.4$ | $1820.5 \pm 47.9$ | $90.0 \pm 2.2$ | $30.6 \pm 0.4$ |

**Table 2: Rewarped tasks tabular results.** Evaluation episode returns for final policies after training. Mean and 95% CIs over 10 random seeds with  $2N$  episodes per seed for  $N = 32$  or 64 parallel envs.

### 6.1 RESULTS ON REWARPED TASKS

We compare SAPO, our proposed maximum entropy FO-MBRL algorithm, against baselines on a range of challenging manipulation and locomotion tasks that involve rigid and soft bodies, re-implemented in Rewarped, our parallel differentiable multiphysics simulation platform. In Figure 2, we visualize training curves to compare algorithms. SAPO shows better training stability across different random seeds, against existing FO-MBRL algorithms APG and SHAC. In Table 2, we report evaluation performance for final policies after training. SAPO outperforms all baselines across all tasks we evaluated, given the same budget of total number of environment steps. We also find that on tasks involving deformables, APG outperforms SHAC, which is consistent with results in DaXBench (Chen et al., 2022) on their set of soft-body manipulation tasks. However, SHAC outperforms APG on the articulated rigid-body tasks, which agrees with the rigid-body locomotion results in DFlex (Xu et al., 2021) that we also reproduce ourselves in Appendix F.2.

In Appendix Figure 11, we visualize different trajectories produced by SAPO policies after training. We observe that SAPO learns to perform tasks with deformables that we evaluate on. For RollingFlat, SAPO controls the rolling pin to flatten the dough and spread it across the ground. For SoftJumper, SAPO learns a locomotion policy that controls a soft elastic quadruped to jump forwards. For HandFlip, SAPO is capable of controlling a high degree-of-freedom dexterous robot hand, to flip a

{9}------------------------------------------------

piece of dough in half within the palm of the hand. For FluidMove, SAPO learns a policy to move the container of fluid with minimal spilling. Additionally, SAPO learns a successful locomotion policy for the articulated rigid-body locomotion task AntRun. For HandReorient however, SAPO is only capable of catching the cube and preventing it from falling to the ground. This is a challenging task that will likely require more environment steps to learn policies capable of re-orienting the cube to given target poses in succession.

### 6.2 SAPO ABLATIONS

We investigate which components of SAPO yield performance gains over SHAC, on the HandFlip task. We conduct several ablations on SAPO: (a) w/o  $V_{\text{soft}}$ , where instead the critic is trained in standard fashion without entropy in target values; (b) w/o  $\mathcal{H}_\pi$ , where we do not use entropy-augmented returns and instead train the actor to maximize expected returns only; (c) w/o  $\mathcal{H}_\pi$  and  $V_{\text{soft}}$ , which corresponds to modifying SHAC by applying design choices {III, IV, V} described in Section 4.2.

We visualize training curves in Figure 3, and in Table 3 we report final evaluation performance as well as percentage change from SHAC’s performance as the baseline. From ablation (b), we find that using analytic gradients to train a policy to maximize both expected return and entropy is critical to the performance of SAPO, compared to ablation (a) which only replaces the soft value function.

Additionally, we observe that ablation (c), where we apply design choices {III, IV, V} onto SHAC, result in approximately half of the performance improvement of SAPO over SHAC on the HandFlip task. We also conducted this ablation on the DFlex rigid-body locomotion tasks however, and found these modifications to SHAC to have minimal impact in those settings, shown in Appendix F.3. We also conduct individual ablations for these three design choices in Appendix F.4.

![Figure 3: SAPO ablations – HandFlip training curves. A line graph showing Return (Y-axis, 0 to 100) versus Environment Steps (X-axis, 0 to 6M). Five curves are plotted: SAPO (ours) in red, w/o V_soft in green, w/o H_pi in blue, w/o H_pi and V_soft in orange, and SHAC in yellow. SAPO (ours) shows the highest return, reaching nearly 100 by 6M steps. The other curves show lower returns, with SHAC being the lowest.](33228b4227fa57e1477b27b9e07483e6_img.jpg)

Figure 3: SAPO ablations – HandFlip training curves. A line graph showing Return (Y-axis, 0 to 100) versus Environment Steps (X-axis, 0 to 6M). Five curves are plotted: SAPO (ours) in red, w/o V\_soft in green, w/o H\_pi in blue, w/o H\_pi and V\_soft in orange, and SHAC in yellow. SAPO (ours) shows the highest return, reaching nearly 100 by 6M steps. The other curves show lower returns, with SHAC being the lowest.

Figure 3: SAPO ablations – HandFlip training curves. Episode return as a function of environment steps. Smoothed using EWMA with  $\alpha = 0.99$ . Mean and 95% CIs over 10 random seeds.

|  | HandFlip | ( $\Delta\%$ ) |
|-|-|-|
| SAPO (ours) | $90 \pm 2$ | +172.7% |
| w/o $V_{\text{soft}}$ | $77 \pm 3$ | +133.3% |
| w/o $\mathcal{H}_\pi$ | $59 \pm 4$ | +78.8% |
| w/o $\mathcal{H}_\pi$ and $V_{\text{soft}}$ | $56 \pm 3$ | +69.7% |
| SHAC | $33 \pm 3$ | – |

Table 3: SAPO ablations – HandFlip tabular results. Evaluation episode returns for final policies after training. Mean and 95% CIs over 10 random seeds with 64 episodes per seed.

## 7 CONCLUSION

Due to high sample complexity requirements and slower runtimes for soft-body simulation, RL has had limited success on tasks involving deformables. To address this, we introduce Soft Analytic Policy Optimization (SAPO), a first-order model-based actor-critic RL algorithm based on the maximum entropy RL framework, which leverages first-order analytic gradients from differentiable simulation to achieve higher sample efficiency. Alongside this approach, we present Rewarped, a scalable and easy-to-use platform which enables parallelizing RL environments of GPU-based differentiable multiphysics simulation. We re-implement challenging locomotion and manipulation tasks involving rigid bodies, articulations, and deformables using Rewarped. On these tasks, we demonstrate that SAPO outperforms baselines in terms of sample efficiency as well as task performance given the same budget for total environment steps.

**Limitations.** SAPO relies on end-to-end learning using first-order analytic gradients from differentiable simulation. Currently, we use (non-occluded) subsampled particle states from simulation as observations to policies, which is infeasible to obtain in real-world settings. Future work may use differentiable rendering to provide more realistic visual observations for policies while maintaining differentiability, towards sim2real transfer of policies learned using SAPO. Another promising direction to consider is applications between differentiable simulation and learned world models.

 Rest of paper (reference and Appendix) is removed.