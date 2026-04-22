# Bridging Discrete and Continuous RL: Stable Deterministic Policy Gradient with Martingale Characterization

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
The theory of discrete-time reinforcement learning (RL) has advanced rapidly over the past decades. Although primarily designed for discrete environments, many real-world RL applications are inherently continuous and complex. A major challenge in extending discrete-time algorithms to continuous-time settings is their sensitivity to time discretization, often leading to poor stability and slow convergence.
In this paper, we investigate deterministic policy gradient methods for continuous-time RL. We derive a continuous-time policy gradient formula based on an analogue of the advantage function and establish its martingale characterization. This theoretical foundation leads to our proposed algorithm, CT-DDPG, which enables stable learning with deterministic policies in continuous-time environments.
Numerical experiments show that the proposed CT-DDPG algorithm offers improved stability and faster convergence compared to existing discrete-time and continuous-time methods, across a wide range of control tasks with varying time discretizations and noise levels.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a principled framework for deterministic policy gradient (DPG) in continuous-time reinforcement learning (CTRL). The authors derive a continuous-time analogue of the advantage-based policy gradient theorem and establish its martingale characterization, providing a theoretical foundation for stable deterministic policy optimization in continuous domains. Building upon this theory, they introduce a new algorithm, CT-DDPG, designed to mitigate the instability and discretization sensitivity often observed in discrete-time RL algorithms when applied to continuous dynamical systems. Experimental evaluations on several continuous control benchmarks demonstrate improved convergence stability and learning efficiency compared to both discrete-time and prior continuous-time baselines. Overall, the paper aims to bridge the theoretical and algorithmic gap between discrete-time RL methods and real-world continuous control systems.

### Strengths
1. The paper establishes a mathematically solid foundation for deterministic policy gradient (DPG) methods in continuous-time reinforcement learning. The authors derive a continuous-time analogue of the advantage function (Theorem 3.1) and rigorously prove its martingale characterization (Theorem 3.2), connecting the policy gradient to the advantage-rate function under deterministic policies. This framework generalizes the discrete-time DPG results (Silver et al., 2014) while removing restrictive assumptions such as uniform ellipticity and purely stochastic policies. The martingale-based formulation elegantly guarantees consistency of the value and advantage functions and provides a clear theoretical pathway for algorithm design.
2. The paper directly targets the issue that discrete-time RL algorithms become unstable as the time-step $h\to0$. The authors provide a precise theoretical explanation for this degradation: one-step temporal-difference (TD) updates cause gradient variance to blow up (Proposition 4.1). To overcome this, they introduce the Continuous-Time Deep Deterministic Policy Gradient (CT-DDPG) algorithm using multi-step TD objectives (Eq. 4.7), proving its variance remains bounded (Proposition 4.2).

### Weaknesses
1. The paper’s theoretical framing and algorithmic contribution appear incremental relative to prior work in continuous-time reinforcement learning (CTRL). While it emphasizes the use of a martingale characterization to derive a deterministic policy gradient (DPG) formula, the idea is conceptually close to existing stochastic-policy frameworks (Jia & Zhou, 2022a; 2022b; 2023) and the model-based DPG formulations of [5]. The paper does not convincingly clarify why the martingale representation provides a fundamentally new perspective or practical advantage over Itô-based stochastic analysis or existing actor–critic formulations.
   
   Moreover, the literature review is narrow and overlooks several recent studies towards same target. For example: [1] systematically analyzes multiple discretization strategies (MSS) for model-based CTRL; [2] proposes a time-adaptive sensing and control approach that jointly optimizes action and duration, directly tackling the same step-size sensitivity problem that this paper aims to address; [3] focuses on improving sample and computational efficiency while maintaining performance; and [4] theoretically studies how, for stochastic dynamics, the observation interval should adapt to the system’s variance. Although the authors cite [5], the discussion remains superficial—[5] already develops a continuous-time actor–critic framework that learns a deterministic policy gradient under ODE dynamics, conceptually similar to this work.
   
   The paper does not clearly articulate how its CT-DDPG algorithm differs from or improves upon these established baselines, many of which already achieve comparable goals such as variance control, discretization invariance, and stability. A more thorough comparative discussion and empirical ablation referencing these works would be necessary to substantiate the claimed contributions.

   [1] Treven, L., Hübotter, J., Sukhija, B., Dorfler, F., & Krause, A. (2023). Efficient exploration in continuous-time model-based reinforcement learning. _Advances in Neural Information Processing Systems_, _36_, 42119-42147.

   [2] Treven, Lenart, et al. "When to sense and control? a time-adaptive approach for continuous-time RL." _Advances in Neural Information Processing Systems_ 37 (2024): 63654-63685.

   [3] Zhao, R., Yu, Y., Zhu, A. Y., Yang, C., & Zhou, D. Sample and Computationally Efficient Continuous-Time Reinforcement Learning with General Function Approximation. In _The 41st Conference on Uncertainty in Artificial Intelligence_.

   [4] Zhao, R., Yu, Y., Wang, R., Huang, C., & Zhou, D. (2025). Instance-Dependent Continuous-Time Reinforcement Learning via Maximum Likelihood Estimation. _arXiv preprint arXiv:2508.02103_.

   [5] Yildiz, C., Heinonen, M., & Lähdesmäki, H. (2021, July). Continuous-time model-based reinforcement learning. In _International Conference on Machine Learning_ (pp. 12009-12018). PMLR.

2. The set of stochastic baselines is insufficient. It is unclear why the paper does not include comparisons with ctrl methods such as those proposed in [2] and [5]. In Figure 2, the only continuous-time baseline shown is the q-learning algorithm. Since the authors already cite [5] in the main paper, it would be natural—and important—to include it as a baseline to better demonstrate the claimed advantages of the proposed deterministic approach.
3. The reported results also appear somewhat counter-intuitive. In Figure 2(c) (Hopper, h = 0.008, $\sigma$ = 0), the q-learning (L = 1) baseline exhibits a substantially larger error bar than in panels (g) and (k), even though the noise level is lower. This behavior is unexpected and not explained in the paper—it raises questions about potential implementation differences, instability in training, or inconsistent experimental conditions.

### Questions
Please compare with the papers listed in the weakness part. I would be happy to raise the score if the authors can address the concerns as listed in the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper formulates RL directly in continuous time (finite‑horizon SDE dynamics) and derives a DPG (Deterministic Policy Gradient) formula using an *advantage rate* (A_\phi(t,x,a)\coloneqq \mathcal{L}[V_\phi](t,x,a)+r(t,x,a)), where (\mathcal{L}) is the generator of the diffusion. The main result (Thm. 3.1) expresses (\partial_\phi V_\phi) as an expectation over (\partial_\phi \mu_\phi^\top \partial_a A_\phi), i.e., a clean continuous‑time analogue of discrete‑time DPG.  Thm. 3.2 characterizes both the value (V_\phi) and the advantage rate (A_\phi) via a *martingale orthogonality* condition. Practically, the authors enforce this by minimizing a “martingale loss” with test functions chosen as the networks’ parameter gradients and by reparameterizing the critic to satisfy (q_\psi(t,x,\mu_\phi(t,x))=0) (Eq. 4.1). This yields an implementable objective without sampling over actions.  Building on this the paper proposes **Continuous‑Time DDPG (CT‑DDPG)** (Alg. 1). Key ingredients: (i) a *multi‑step* TD objective over a short fixed window (Lh=\delta), (ii) a target value network, replay buffer, and Gaussian exploration noise, and (iii) a discrete‑time implementation robust as (h\to0). The “martingale loss” is the core critic objective (Eq. 4.2).

### Strengths
The DPG formula (Thm. 3.1) and the martingale identification (Thm. 3.2) are clean and connect continuous‑time stochastic control with practical deep RL estimators. The reparameterization (q_\psi=\bar q_\psi-\bar q_\psi(\cdot,\mu_\phi)) enforces the Bellman side condition  (Eq. 4.1). 

The variance analysis (Props. 4.1–4.2) explains a widely observed pathology: one‑step TD becomes unusable as (h) shrinks, whereas multi‑step with fixed physical window (\delta) remains well‑behaved. 

CT‑DDPG combines these ideas into a simple recipe (Alg. 1, p. 7–8): multi‑step critic, target network, replay, plus an actor trained by the estimated (q). The experiments consistently show improved stability under smaller (h) and higher noise, the regimes where discrete‑time baselines degrade (Fig. 1).

### Weaknesses
The paper proves bounded‑variance gradients and non‑vanishing signal for the critic objective (Props. 4.1–4.2), but it does **not** provide convergence or stability guarantees for the *actor–critic* recursion as a whole (with function approximation, target nets, replay, and exploration noise). The term “provable stability and efficiency” (Sec. 4) overstates what is established. 

Clarifying the exact scope of guarantees would help. Discrete‑time baselines use standard one‑step updates; the paper does not include multi‑step or TD(λ) versions of DDPG/SAC that might mitigate the very issue analyzed here.
 
Among continuous‑time methods, the comparison is to a stochastic‑policy q‑learning implementation; there is no head‑to‑head with continuous‑time PPO/TRPO‑style approaches cited in Sec. A (related work). A stronger baseline suite would isolate the benefits of deterministic policies from those of *multi‑step TD*. 

 Experiments inject i.i.d. Gaussian generalized forces at the MuJoCo level.  While sensible, this perturbation may not match the SDE regularity assumptions used in analysis (e.g., uniform ellipticity in Prop. 4.1’s setup). Some discussion of this modeling gap would be needed.

Theorems assume (V_\phi\in C^{1,2}) and smooth dependence on parameters/actions (Assumptions 1–2, Sec. 3). In practice, ReLU networks violate (C^1) smoothness; the paper informally argues Lipschitzness suffices (Asp. 1), but it is unclear whether subgradient analogues cover the results used in Thm. 3.1/3.2. A brief justification or reference would strengthen the story. 

Identification of (A_\phi) holds in a *neighborhood* (O_{\mu_\phi(t,x)}) of the policy action (Eq. 3.7). This entails a coverage condition on exploratory actions and suggests potential brittleness far from the current policy. The algorithm’s off‑policy robustness as exploration grows is not directly tested. 

 CT‑DDPG hinges on window length (\delta=Lh) and the choice of test functions in the martingale loss. Ablations over (L), (\delta), the terminal‑value penalty weight, and the set of test functions would clarify how performance depends on these choices. (Alg. 1; Eq. 4.2.) 

 Tasks are standard but mid‑scale; there’s no Ant/Humanoid evaluation. The return curves (Figs. 1–2) demonstrate trends, but sample‑efficiency comparisons in terms of environment interactions (not just episodes) would make claims about efficiency more concrete.

### Questions
Add *n‑step* or TD(λ) DDPG/SAC and TD3; test whether multi‑step alone narrows the gap to CT‑DDPG when (h) is small. 

Show learning speed/variance vs. (\delta) at fixed (h) and vs. (L) at different (h). 

 Probe how performance changes with exploration noise magnitude; does the martingale identification break when actions stray far from (O_{\mu_\phi})? (Eq. 3.7.) 

Beyond (\zeta_t=\partial_\theta V_\theta) or (\partial_\psi q_\psi), try basis functions in (t) and (x) to assess bias/variance of the martingale loss. (Sec. 4.1.) 

Include continuous‑time TRPO/PPO‑style methods from the related‑work section and report stability under small (h). 

Compare the current MuJoCo force‑noise to process‑noise injected at the state‑equation level to align more closely with the SDE assumptions. (Sec. 5.)

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
4

### Summary
This paper proposes CT-DDPG, an discrete RL algorithm in the continuous RL paradigm. The main idea is to use the martingale orthogonality characterization. Experiments have been conducted on Pendulum-v1, HalfCheetah-v5, Hopper-v5, and Walker2d-v5.

### Strengths
The paper is well-written, and I mostly enjoyed reading it. The theoretical results are solid. The numerical experiments also support the proposed approach.

### Weaknesses
Below are a few comments:

(1) The introductory part seems to be too long, e.g., Thm 3.1 and 3.2 are basically known from the literature. It isn't until Section 4 (p.5) that new results come. The authors may think of a bit reorganization.

(2) The idea of applying Robbins-Monroe in continuous RL was also explored in Regret of exploratory policy improvement and q-learning, arXiv:2411.01302, Tang and Zhao -- but using the martingale loss approach. This paper considers a different martingale orthogonality approach. See also On Optimal Tracking Portfolio in Incomplete Markets: The Reinforcement Learning Approach, SICON, Bo, Huang and Yu for a similar analysis in a specific setting.

(3) Proposition 4.2 (and also 4.1): the authors proved the variance blow-up/no blow-up. However, there is no analysis on the proposed algorithm. What is (roughly) the value function gap in terms of h? Also the proposed algorithm uses the number of steps of order 1/h -- which is exploding. The authors may provide some explanations.

(4) Is there any insight how to choose h in practical problems?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper studies the setting of policy gradient in continuous time and state-action spaces. The authors essentially rely on a martingale characterization to implement PG in a continuous setting. The PG theorem derived is analogous to the discrete counterpart with the advantage rate function taking the place of the advantage function. The authors list a bunch of regularity conditions to invoke the existence of Bellman equation and so on. The policies are deterministic mappings from state to action. The authors then argue that TD(0) is fundamentally incompatible in the continuous setting and propose a multi step TD approach in its place. The proposed algorithm is experimentally validated.

### Strengths
The paper considers continuous RL which is technically challenging and not that well studied. 

Identifies the fundamental failure of TD(0) in this setting. 

Considers deterministic policies which reduces computational cost that is incurred due to sampling.

### Weaknesses
Considers the finite horizon setting only. What are some challenges when it comes to extending the theory to the infinite horizon settings?

The regularity assumptions _can_ be strong and may not always be met.

### Questions
What is the form of Q and V in line 182?

How is the neighbourhood O determined in line 196?

### Soundness
3

### Presentation
3

### Contribution
3
