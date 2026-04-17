000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Constraint-Aware Reward Relabeling For Offline Safe Reinforcement Learning

Anonymous authors Paper under double-blind review

## Abstract

Offline safe reinforcement learning (OSRL) considers the problem of learning reward-maximizing policies for a pre-defined cost constraint from a fixed dataset.

This paper proposes a simple and effective approach referred to as Constraintaware Reward (Re)Labeling (CARL), that can be wrapped around existing offline RL algorithms. CARL is an iterative approach that alternates between two steps for each sampled batch of data to ensure state-action-wise safety constraints. First, update cost evaluation function using an off-policy evaluation procedure. Second, update policy using relabeled rewards (assign large penalty) for state-action pairs which are detected unsafe based on cost estimates. CARL is a minimalist approach, doesn't introduce any additional task-specific hyperparameters, and allows us to leverage strong off-the-shelf offline RL algorithms to solve OSRL problems. Experimental results on the DSRL benchmark tasks demonstrate that CARL reliably enforces safety constraints under small cost budgets, while achieving high rewards. The code is available at https://anonymous.4open.science/
r/CARL-6F11.

## 1 Introduction

Reinforcement learning (RL) has achieved remarkable success across diverse domains, from game playing (Silver et al., 2016) to robotic control (Siekmann et al., 2021). However, in safety-critical applications such as healthcare, autonomous systems, and industrial control, online exploration poses unacceptable risks. Offline reinforcement learning addresses this limitation by learning policies exclusively from pre-collected datasets without any additional interaction with the environment (Levine et al., 2020). However, in safety-critical domains, it is insufficient to focus on maximizing rewards alone and policies must also respect explicit safety constraints during deployment. This double requirement motivates offline safe reinforcement learning (OSRL), where agents must simultaneously maximize expected returns while satisfying user-specified cost budgets or safety constraints. OSRL inherits fundamental challenges from both offline RL and safe RL: handling distributional shift from fixed datasets, and ensuring that the policy behavior remains within safety constraints after deployment. This problem becomes especially challenging under tight cost budgets, where even small cost constraint violations may be unacceptable. As explained in the related work section, prior approaches for OSRL typically employ constrained optimization frameworks. These approaches rely on dual-gradient updates, Lagrangian multipliers, or policy regularization techniques (Polosky et al., 2022; Lee et al., 2022; Xu et al., 2022b). While theoretically principled, such methods introduce substantial algorithmic complexity. In practice, they can also be sensitive to hyperparameter configurations and require careful tuning to avoid training instability or collapsing to overly conservative solutions when the cost budget is small (Zheng et al., 2024).

This paper develops a novel approach referred to as *Constraint-aware Reward (Re)Labeling (CARL)*
to solve OSRL problems. There are two key innovations behind CARL. First, we formulate an unconstrained policy optimization problem to enforce state-action-wise safety constraints. This formulation naturally motivates an iterative policy improvement algorithm that doesn't require tuning Lagrange multipliers. Second, we develop a simple iterative method that can be wrapped around offline RL algorithms with batch updates. It alternates between two steps for each sampled batch of data: update cost evaluation function using an off-policy evaluation procedure and update policy 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 using relabeled rewards (assigns a large penalty) for state-action pairs whose cost-to-go estimates violate cost constraint. CARL is a minimalist method requiring no additional hyperparameters, leveraging off-the-shelf offline RL algorithms to effectively address OSRL problems. The main findings from our evaluation of CARL on DSRL benchmark tasks (Liu et al., 2024) are as follows. First, CARL produces safe policies with high returns on most tasks and outperforms prior methods on a greater number of tasks. Second, CARL achieves excellent performance for the challenging setting of small cost budgets. Third, when trained only on unsafe trajectories, CARL remarkably learns safe policies. Finally, CARL achieves good performance with different offline RL algorithms. Contributions. The key contribution of this paper is the development and evaluation of the Constraint-aware Reward Relabeling approach for OSRL problems. Specific contributions include:
- Formulation of an unconstrained optimization problem for state-action-wise safety constraints.

- Development of the iterative constraint-aware reward relabeling (CARL) method that can be wrapped around existing offline RL algorithms.

- Experimental evaluation of the proposed CARL algorithm on DSRL benchmark tasks to demonstrate its strong effectiveness over state-of-the-art methods.

## 2 Problem Setup

We consider the problem of offline reinforcement learning under safety constraints, modeled using the Constrained Markov Decision Process (CMDP) framework. A CMDP is defined by the tuple
(S, A*, P, r, c, γ, µ*0), where S and A denote the state and action spaces, respectively; P : *S × A ×*
S → [0, 1] represents the unknown stochastic transition dynamics; r : *S × A →* R is the reward function; c : S × A → [0, Cmax] is a non-negative cost function; γ ∈ (0, 1) is the discount factor; and µ0 is the initial state distribution. Let π : *S → A* represent a policy that maps states to actions. Given π and µ0, we can define a distribution over trajectories τ = {(st, at, rt, ct)}
T
t=1 generated by rolling out π from initial states drawn from µ0. We define

$$V_{r}^{\pi}(s)=\ \mathbb{E}_{r\sim\pi}\left[\sum_{t=1}^{T}\gamma^{t}r_{t}\ |\ s_{1}=s\right],\qquad Q_{r}^{\pi}(s,a)=\ \mathbb{E}_{r\sim\pi}[V_{r}^{\pi}(s^{\prime})\ |\ s_{1}=s,a_{1}=a].$$

to be the standard reward state- and action-value functions. Similarly V
π c(s) and Qπ c(*s, a*) denote the cost state- and ation-value functions respectively. In the offline setting, the agent has access only to a static dataset D = {(si, ai, ri, ci, s′i)}
n i=1 collected from one or more unknown behavior policies, without further interaction with the environment. The goal of offline safe RL (OSRL) is to learn a policy from the given offline dataset D that maximizes the expected return while satisfying a given cost constraint:
max π Es∼µ0[V
π r(s)] subject to V
π c ≤ κ, (1)
where κ ≥ 0 is a user-specified safety cost threshold (also referred to as a cost budget or limit). Tight cost limits (i.e., small κ values) present an especially demanding regime where many existing OSRL methods falter. In such safety-critical settings, such as autonomous driving, robotics, or industrial control, even minor constraint violations can be unacceptable. Common approaches based on constrained optimization or dual updates often face practical implementation challenges, such as high sensitivity to tuning, which can result in unstable training dynamics or overly conservative policies. Our goal is to develop a minimalist approach that can be wrapped around existing offline RL methods to solve OSRL problems under the challenging setting of small cost budgets.

## 3 Related Work

Online Safe RL. Safety in online RL has been widely studied (Garcıa & Fernandez, 2015; Gu et al., ´ 2024; Wachi et al., 2024). In this setting, the agent interacts with the environment while adhering to safety constraints. A common mechanism is penalty-based control, shaping the reward as

$$(\mathbf{l})$$

r
′ = r − λ · c. While simple, fixed penalties introduce a sensitive trade-off between reward and constraint satisfaction, and often require extensive tuning; e.g., RCPO (Tessler et al., 2018) and Safety Gym (Ray et al., 2019) include baselines with fixed λ values, with main results based on adaptive Lagrangian updates. ROSARL formalizes penalty selection via the "minmax penalty" (Tasse et al., 2023). Within this penalty-based family, Saute RL (Sootla et al., 2022) augments the state with a ´ safety budget, and MASE (Wachi et al., 2023) casts online safe exploration as a generalized problem that combines uncertainty quantification with a reset action to prevent violations during training. Recent work also targets adaptability under changing constraints, e.g., constraint-conditioned value function approximation (Yao et al., 2023). Offline RL. Offline reinforcement learning focuses on learning policies purely from fixed datasets without additional interaction with the environment (Levine et al., 2020; Figueiredo Prudencio et al., 2024). A key challenge in this setting is distributional shift between the behavior policy and the learned policy. Approaches addressing this include value estimation regularization (Fujimoto & Gu, 2021; Kostrikov et al., 2022; Kumar et al., 2019; Lyu et al., 2022; Yang et al., 2022), generative or sequential modeling (Janner et al., 2021; Wang et al., 2022), and uncertainty-aware learning (An et al., 2021; Bai et al., 2022). Some techniques utilize Q-function based action selection via filtering and reweighting, including SfBC and IDQL (Chen et al., 2023; Hansen-Estruch et al., 2023). Other methods constrain policy updates using divergence measures (Wu et al., 2020; Jaques et al., 2020; Wu et al., 2022), or leverage advances in model-based RL (Kidambi et al., 2020; Yu et al., 2020; Rigter et al., 2022) and imitation learning (Xu et al., 2022a). Offline Safe RL. OSRL extends the offline RL setting to include user-defined cost constraints (Liu et al., 2024). Many OSRL methods rely on constrained optimization, often using Lagrangian relaxation (Xu et al., 2022b; Lee et al., 2022; Polosky et al., 2022), or exploiting convexity assumptions in cost and reward trade-offs (Zhang et al., 2024). However, they often require solving interdependent optimization problems and are prone to instability, particularly under strict cost limits. Beyond constrained optimization, several recent methods propose alternative strategies. FISOR (Zheng et al., 2024) enforces safety via diffusion models to select only feasible actions and is the only method designed to handle the challenging setting of small cost budgets. TraC (Gong et al., 2025) introduces a trajectory-based classification method for safe policy learning. Latent safety modeling was also explored in (Koirala et al., 2024). LSPC employs a conditional VAE to encode conservative safety constraints into a latent space and perform reward optimization via advantageweighted regression in that space. CAPS (Chemingui et al., 2025) switches between pre-trained policies to adapt to different test-time cost constraints. (Guo et al., 2025) propose a constraintconditioned actor-critic (CCAC) method that explicitly models the relationship between state-action distributions and constraints, to improve generalization to unseen cost thresholds. Diffusion-based generative models have also gained traction for OSRL. TREBI (Lin et al., 2023) employs trajectory-level diffusion sampling guided by safety classifiers, while other works adapt models such as Decision Transformer (Chen et al., 2021) and Diffuser (Janner et al., 2022) to the constrained setting (Liu et al., 2023; Lin et al., 2023). While powerful, these methods often require additional architectural components, hyperparameter tuning, or auxiliary optimization targets.

The overall goal of this paper is to address two key limitations of prior work. First, there is very little work on OSRL under small cost budgets. FISOR produces safe policies in this regime, but it achieves low reward. Second, Lagrangian-based constrained policy optimization methods can be difficult to stabilize in offline settings, often requiring extensive tuning to achieve optimal performance. The proposed constraint-aware reward relabeling approach is minimalist, can be wrapped around offline RL algorithms.

## 4 State-Action-Wise Constraints For Safety

This section first describes an alternative formulation which enforces safety for all states and theoretically shows that its solution also solves the original problem which enforces safety constraint in expectation. Next, we provide the sketch of an iterative policy improvement approach that is most natural to solve our formulation and forms the basis for our proposed algorithm. Formulation for state-action-wise safety. Solving the constrained MDP in Equation (1) often involves reformulating it as an unconstrained optimization problem via the Lagrangian method.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Although this reformulation offers a principled approach, it requires an intricate tuning of the Lagrangian multiplier, where the final performance is sensitive to it. To overcome this challenge, we consider a stronger formulation that aims to ensure constraint satisfaction for all state-action pairs a policy will encounter which is particularly beneficial when the cost limit κ is small.

$$\operatorname*{max}_{\pi}\;V_{r}^{\pi}{\mathrm{~subject~to~}}Q_{c}^{\pi}(s,\pi(s))\leq\kappa,\forall s.$$
(s, π(s)) ≤ κ, ∀s. (2)
where the max operator over value functions considers V1 ≤ V2 if V1(s) ≤ V2(s), ∀s. The above pointwise constraints ensure that we enforce safety across all states and actions that the policy will select, not simply over the expected value of the cost function as in the optimization problem from Equation (1). A solution to the optimization problem in Equation (2), if it exists, immediately yields a solution to the problem in Equation (1), but not vice versa. The point-wise constraints formulation offers various benefits. First, to a certain extent, the pointwise constraint should be a preferred solution to many real-world applications that require safety. It requires that, no matter where we start within the system, the expected cumulative cost will be within the safety threshold. On the other hand, the problem in Equation (1) only requires that a policy is safe in the expectation with respect to the initial state and action. This might not be desirable. In the deployment of safe RL for real safety-critical applications, we are often in a one-shot setting, instead of performing repeated experiments. In such a case, only the solution for Equation (2) but not from Equation (1) guarantees safety every time during deployment. Second, and most importantly, the point-wise constraints allow us to turn the offline constrained RL problem into an unconstrained optimization problem where it is free of tuning Lagrangian multipliers. Additionally, this unconstrained optimization naturally motivates a simple iterative algorithm that allows us to leverages powerful off-the-shelf solvers. Let us consider the following problem:

* [10] M. C.  
max πV
π rπ where rπ(*s, a*) := 1{Qπ c
(s,a)≤κ} · r(*s, a*) − 1{Qπ c
(s,a)>κ}Vmax (3)
with Vmax = Rmax/(1 − γ) being the maximum possible infinite-horizon value. We show in the Theorem below that it suffices to solve the unconstrained optimization in Equation (3). Theorem 1. Assume there exists a solution to Problem (2). Then a policy π
∗is an optimal solution to Problem (2) if and only if it is an optimal solution to the unconstrained optimization in (3).

Proof. Consider any safe policy π according to the constraint in Problem (2). By definition we know that for each state s, Qπ c
(s, π(s)) ≤ κ. This means that for any state-action pair (st, at) along a trajectory τ generated by π, we have r(st, at) = rπ(st, at). Thus, for any safe policy π we have V
π r = V
π rπ
. To complete the proof we show that any solution to Problem (3) must be safe.

Let π˜
∗ be a solution to Problem (2). Let π
∗ be a solution to Problem (3). We will show that π
∗ must satisfy the point-wise safety, i.e., Qπ
∗
c(*s, π*∗(s)) ≤ κ, ∀s. Indeed, assume that Qπ
∗
c(s, π∗(s)) > κ for some s. Then, rπ∗ (*s, π*∗(s)) = −Vmax. Thus, V
π
∗
rπ∗(s) = −Vmax +
Eπ∗,P [P∞
t=1 γ trπ∗ (st, at)] < 0 < V π˜
∗
r(s) = V
π˜
∗
rπ˜∗(s), where the last equality follows from the safety of π˜
∗. This contradicts that π
∗is an optimal policy to Problem (3). Thus, Qπ
∗
c(*s, π*∗(s)) ≤
κ, ∀s and hence is safe according to Problem (2). □
Sketch of an Iterative Policy Improvement Algorithm. Motivated by the classical policy iteration method, we can design a policy iteration variant to solve the unconstrained optimization in Equation (3) as summarized below:
πt offline policy evaluation
−→ Q
πt c reward relabeling
−→ rπt offline policy optimization
−→ πt+1 (4)
The goal of this iterative method is to improve utility and safety trade-offs of the policy incrementally from one iteration to the next. Specifically, in each iteration t, given the current policy πt, we estimate the cost function Qπt c using an offline policy evaluation (OPE) solver. Next, we compute the reshaped reward rπtbased on Qπt c. Finally, we perform offline policy optimization (OPO)
by calling an offline RL algorithm with the reward-relabeled offline data (s, a, rπt, s′) to obtain a improved policy πt+1. Building on this general principle, we describe a concrete algorithmic approach next.

## 5 Carl: Constraint-Aware Reward Relabeling

Our solution approach is based on designing a wrapper around any *batch-update* offline RL algorithm. In particular, batch update algorithms can be described via batch update rules, where AOPE denotes an offline policy evaluation update rule, and AOPO an offline policy optimization update rule. Each accepts a mini-batch B = {(si, ai, ri, s′i)} and performs:
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 1. **Cost evaluation:** M iterations of OPE updates on (*s, a, c, s*′) to refine Qπ c.

2. **Policy optimization:** K iterations of OPO updates on CARL(*s, a, r, s*′) given by 5.

allowing iterative improvement by repeatedly sampling batches from a given offline training dataset. Most state-of-the-art offline RL algorithms—e.g., TD3-BC, IQL—can be expressed in this form. The goal is to wrap these existing update functions so that, given a safe offline RL problem in the form of a training dataset D and a cost budget κ, the resulting policy π maximizes reward while satisfying the safety constraint. Below we first describe the motivation for our approach, followed by the algorithmic details.

## 5.1 Action Filter Motivation

To motivate our approach, consider a finite MDP with discrete states and actions. We can view applying the iteration in Equation 4 to this MDP as integrating an *action filter* into a standard unconstrained solver (e.g., policy iteration): given the current policy π, estimate its cost-to-go Qπ c
(*s, a*),
and remove from the MDP all actions whose cost-to-go exceeds the budget κ. Solving this reduced MDP yields a new policy π
′, and the process can be repeated, always starting each iteration with the full action set. Across iterations, actions previously removed may be reintroduced if they become safe, and safe actions may be removed if they become unsafe. While intuitive, this process can be unstable. The root cause is that after each policy update, the cost-to-go function can change drastically, causing the set of filtered actions to vary arbitrarily from one iteration to the next. One way to mitigate this instability is to update the policy and cost-to-go *gradually*, so that they track each other closely throughout the optimization process. In the discrete MDP setting, this might mean updating only for a small batch of states at each iteration. Below we extend this idea to the more general setting of continuous state and action *offline* safe RL.

## 5.2 Filtering Via Batch Reward Relabeling

In continuous domains with function approximation, "removing" an individual unsafe action is illposed: we instead want to suppress an *entire neighborhood* of similar actions. A simple way to achieve this is via the reward relabeling approach implicit in Problem (3) where we replace the reward for any state–action pair predicted to be unsafe with a maximally negative constant. Function approximation then naturally generalizes this penalty to nearby actions, discouraging them without needing to identify and prune them explicitly. We use a simple approach to integrate reward relabeling and batch-update offline RL. Specifically, given the current Q-cost function Qπ c under the current policy π, we define the constraint-aware reward relabeling rule for a transition (*s, a, r, s*′) as:

$$\mathbf{C}\mathrm{{ARL}}(s,a,r,s^{\prime})={\begin{cases}(s,a,r,s^{\prime}),&{\mathrm{if~}}Q_{c}^{\pi}(s,a)\leq\kappa,\\ (s,a,-V_{\mathrm{max}},s^{\prime})&{\mathrm{otherwise}},\end{cases}}$$

Performing batch updates where rewards in each batch are relabeled naturally avoids actions currently considered unsafe and, by generalization, actions similar to them. The generic batch reward relabeling loop alternates between the following two steps:

$$(S)$$
$$\mathrm{I},\qquad(\pi^{\prime},Q^{\prime})\leftarrow{\mathcal{A}}$$

Q
′ ← AOPE(*Q, π,* B), (π
′, Q′) ← AOPO(*π, Q,* B),
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Intuitively, large M and K let Qc and π change substantially between phases. Thus, as described above for discrete MDPs, this can cause severe oscillations: the algorithm alternates between unsafe, high-reward policies and overly conservative safe policies, never exploring the many safe policies with high reward. Figure 1 illustrates an implementation based on TD3BC where we observe this oscillation for a standard benchmark problem AntRun.

To address this instability, a natural stabilizing choice is to keep M and K *small* so that Qc and π track each other closely. The most extreme case, M = K = 1, suggests a simple wrapper with no additional tunable hyperparameters (utilizing dataset-derived penalties) beyond the base OPE and OPO algorithms. Each iteration involves sampling a mini-batch B followed by:
Require: Offline dataset D, budget κ, backbone updates AOPE, AOPO, batch size m 1: Init cost critic Qc, policy π, and reward critic Qr 2: **while** not converged do 3: Sample mini-batch {(si, ai, ri, ci, s′i)}
m i=1 ⊂ D
4: Qc ← AOPEQc, π, {(si, ai, ci, s′i)}
m i=1 5: (π, Qr) ← AOPO*π, Q*r, {CARL(si, ai, ri, s′i)}
m i=1 6: **end while** 7: **return** π Figure 1: Oscillatory Performance of AntRun: re-

![5_image_0.png](5_image_0.png) ward (left) and cost (right) across training steps with a cost limit of 40.

Update Qπ c with B −→ Relabel B via Equation 5 −→ Update π with relabeled B.

The pseudo-code for this incremental variant called is shown in Algorithm 1. Despite the minimalism, we find that M = K = 1 consistently results in state-of-the-art performance. Further, while one can treat K and M as hyperparameters, we have not found values that consistently outperform CARL across benchmarks. Because the CARL algorithm modifies only the reward before passing data to the backbone, it inherits the underlying offline RL algorithms' sample efficiency and out-of-distribution handling.

While setting K = M = 1 greatly reduces drift compared to using large values of K and M, theoretical convergence guarantees are unclear. Formally analyzing whether K = M = 1 convergespossibly under assumptions on the MDP class, dataset coverage, or backbone stability—is an open problem. Empirically, we find the method to be stable across all tested benchmarks and initializations, and achieving significantly better performance than state-of-the-art offline safe RL methods. Summary of CARL's advantages. Our proposed CARL approach has the following advantages.

- It can be wrapped around existing offline RL algorithms to effectively solve offline safe RL
problems as we demonstrate in our experiments.

- It is a minimalist approach that doesn't introduce any additional tunable hyperparameters.

## 6 Experiments And Results 6.1 Experimental Setup

Benchmarks. We evaluate CARL across a variety of offline safe reinforcement learning tasks using the DSRL benchmark suite (Liu et al., 2024), which offers standardized datasets with diverse safety constraints. Our evaluation mainly spans both the Bullet-based environments (Car-Run, Drone-Circle etc.), as well as broader generalization through additional tasks drawn from Safety- Gymnasium (Ray et al., 2019; Ji et al., 2024). The combined benchmark suite includes diverse tasks spanning a range of difficulty levels and safety complexities.

## Algorithm 1 Constraint-Aware Reward Relabeling (Carl)

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Evaluation Protocol. Consistent with DSRL standards, we report two core metrics: normalized cumulative reward and normalized cumulative cost. For a task T, if rmax(T) and rmin(T) represent the maximum and minimum observed cumulative rewards, the normalized reward for a policy π is computed as:

$$R_{\mathrm{norm}}={\frac{R_{\pi}-r_{\mathrm{min}}(T)}{r_{\mathrm{max}}(T)-r_{\mathrm{min}}(T)}},\qquad C_{\mathrm{norm}}={\frac{C_{\pi}}{\kappa}}$$
.
where Cπ is the policy's cumulative cost and κ is the cost threshold. A policy is considered safe if Cnorm ≤ 1. This formulation differs from alternative evaluations in CCAC (Guo et al., 2025)
that normalize rewards only using trajectories satisfying the cost budget κ. Instead, we follow the standard DSRL evaluation protocol, which uses the full reward range per task. Our main results use stringent thresholds of κ = 5 and κ = 10 to test performance in highly constrained scenarios. Each policy is tested over 20 episodes and averaged over three random seeds. Baselines. We compare CARL against a range of state-of-the-art OSLR methods. BC-Safe is a behavior cloning baseline that trains on trajectories satisfying a predefined cost threshold. CPQ (Xu et al., 2022b) penalizes out-of-distribution actions and updates Q-values using only safe transitions. COptiDICE (Lee et al., 2022) addresses safety via stationary distribution correction. CDT (Liu et al., 2023) uses a decision transformer to learn policies conditioned on reward and cost enabling test-time constraint adaptation. CAPS (Chemingui et al., 2025) also supports test-time constraint handling by switching among multiple pre-trained policies. FISOR (Zheng et al., 2024) is a diffusion-based approach that directly optimizes for feasibility to produce zero-violation policies.

Finally, we include CCAC (Guo et al., 2025), a recent method that uses a constraint-conditioned actor–critic with generative modeling and OOD detection for adaptive, safe policy learning under varying constraints. We further evaluated Lagrangian variants of offline RL algorithms in Table 5. Our results use CARL as a wrapper around TD3-BC, with fitted-Q evaluation (FQE) employed for off-policy evaluation (Le et al., 2019). Implementation details are available in the Appendix.

## 6.2 Results And Discussion

CARL vs. Baselines. Table 1 compares CARL with a suite of offline safe RL baselines across 19 DSRL tasks. Performance is reported in terms of normalized reward (higher is better) and normalized cost (lower is better, with values ≤ 1 indicating constraint satisfaction) under low budgets of 5 or 10. For the main results, we set the penalty using Rmax = max(*s,a,r,*·) r from the offline data instead of Vmax; an ablation with the larger penalty Vmax is included in Table 5 in the appendix.

Remarkably, CARL is the *only* method that satisfies the cost constraint across all Bullet tasks. It is also safe on 8 out of 11 in the more challenging SafetyGym tasks. While other methods such as CAPS, FISOR, or CCAC manage to remain safe on a few tasks, none of them achieve the same level of consistency for safety. Importantly, CARL's safety does not come at the expense of reward performance. CARL consistently ranks as the best or second-best safe method in terms of reward. While other baselines occasionally achieve higher returns, the top-performing method varies across tasks and fails to consistently satisfy the cost constraint, unlike CARL which maintains stricter safety in most of the tasks. These results demonstrate that CARL strikes a strong balance between reward maximization and constraint satisfaction, without requiring special reward shaping, risk-sensitive training objectives, or task-specific tuning. Its consistent safety and competitive performance across tasks and cost budgets highlight its robustness and suitability for safety-critical settings. Backbone Offline RL Algorithm. CARL can be wrapped around any off-the-shelf offline RL algorithm without modifying the method's loss, targets, or regularizers. Our main results use TD3- BC (Fujimoto & Gu, 2021) as the backbone. To test generality, we also evaluate CARL with IQL (Kostrikov et al., 2022), which differs significantly in design. TD3-BC is an actor–critic method that queries the current policy to generate target actions and applies behavior cloning regularization on the actor. In contrast, IQL estimates Q-values purely from dataset actions and optimizes the policy separately via advantage-weighted regression, without querying the policy during value learning. As shown in Table 2, CARL maintains safety and achieves comparable rewards under both backbones. This indicates that our relabeling in Equation 5 is agnostic to the underlying backbone, confirming that CARL generalizes effectively across offline RL algorithms.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

Tasks BC Safe CPQ COptiDICE CDT CAPS CCAC FISOR CARL

Bullet Gym Tasks: κ = 5

BallRun Reward ↑ 0.16±0.11 0.09±0.26 0.53±0.10 0.27±0.09 0.07±0.05 0.31±**0.01** 0.09±0.07 0.28±**0.02**

Cost ↓ 4.50±3.38 2.20±2.88 10.83±1.68 2.57±3.23 0.00±0.00 0.00±**0.00** 1.28±1.70 0.00±**0.00**

CarRun Reward ↑ 0.92±0.01 0.93±0.01 0.91±0.04 0.99±0.00 0.97±**0.00** 1.82±0.84 0.74±0.01 0.97±**0.00**

Cost ↓ 0.26±0.23 0.20±0.18 0.00±0.00 0.90±0.34 0.11±**0.17** 24.57±20.94 0.00±0.00 0.02±**0.03**

DroneRun Reward ↑ 0.41±0.23 0.29±0.11 0.68±0.01 0.58±**0.00** 0.41±0.06 0.50±0.08 0.31±0.04 0.36±**0.12**

Cost ↓ 1.62±1.73 2.35±4.08 15.02±0.10 0.07±**0.07** 5.70±3.08 16.29±9.35 2.52±1.10 0.30±**0.52**

AntRun Reward ↑ 0.56±0.02 0.03±**0.05** 0.61±0.01 0.70±0.03 0.53±0.13 0.03±0.10 0.43±0.02 0.36±**0.09**

Cost ↓ 1.15±0.47 0.05±**0.08** 3.26±1.39 1.66±0.24 2.03±2.17 0.00±0.00 0.27±0.15 0.60±**0.41**

BallCircle Reward ↑ 0.45±0.03 0.56±0.10 0.71±0.01 0.61±0.17 0.33±**0.02** 0.47±0.38 0.32±0.05 0.69±**0.03**

Cost ↓ 1.21±0.23 1.11±1.92 9.41±0.69 2.03±1.39 0.01±**0.02** 10.19±17.65 0.00±0.00 0.33±**0.23**

CarCircle Reward ↑ 0.31±0.06 0.71±**0.02** 0.49±0.01 0.71±0.01 0.40±**0.03** 0.70±0.01 0.37±0.02 0.69±**0.01**

Cost ↓ 1.57±0.07 0.00±**0.00** 10.82±1.28 1.58±0.57 0.03±**0.03** 1.61±0.51 0.00±0.00 0.00±**0.00**

DroneCircle Reward ↑ 0.50±0.01 -0.21±**0.04** 0.26±0.02 0.55±0.01 0.36±0.02 0.43±0.04 0.48±0.01 0.53±**0.02**

Cost ↓ 1.22±0.31 0.70±**0.41** 3.68±0.51 1.14±0.06 0.00±0.00 0.06±0.07 0.17±0.15 0.00±**0.00**

AntCircle Reward ↑ 0.40±0.02 0.00±**0.00** 0.18±0.02 0.45±0.05 0.33±0.05 0.49±0.07 0.24±0.02 0.60±**0.01**

Cost ↓ 4.72±0.87 0.00±**0.00** 17.40±0.36 6.59±1.42 0.00±0.00 0.02±0.03 0.04±0.08 0.02±**0.03**

Safety Gym Tasks: κ = 10

CarCircle1 Reward ↑ 0.27±0.09 -0.03±0.07 0.68±0.05 0.53±0.07 0.42±0.04 0.32±0.10 0.69±0.03 0.46±0.03

Cost ↓ 3.82±1.01 10.17±7.26 20.70±1.15 8.58±2.04 2.79±1.04 21.96±4.14 10.52±0.10 4.15±0.83

CarCircle2 Reward ↑ 0.37±0.06 0.46±0.05 0.75±0.02 0.61±0.04 0.38±0.14 0.57±0.01 0.63±0.02 0.45±0.05

Cost ↓ 6.90±2.62 2.41±3.95 26.40±2.30 13.12±2.59 4.15±5.07 16.85±4.07 12.78±1.97 1.57±1.38

CarGoal1 Reward ↑ 0.20±**0.09** 0.67±0.07 0.40±0.03 0.62±0.04 0.27±0.00 0.82±0.06 0.42±0.07 0.26±**0.04**

Cost ↓ 0.40±**0.12** 4.41±0.46 2.50±0.36 3.12±0.69 1.15±0.10 5.27±0.92 1.70±0.58 0.92±**0.55**

CarGoal2 Reward ↑ 0.14±0.02 0.53±0.14 0.21±0.10 0.42±0.06 0.11±0.06 0.91±0.02 0.05±**0.01** 0.13±0.03

Cost ↓ 1.45±0.60 15.09±1.91 3.01±1.45 5.34±1.25 2.05±2.23 15.80±1.04 0.50±**0.71** 1.77±0.51

PointCircle1 Reward ↑ 0.30±0.11 0.46±**0.04** 0.87±0.01 0.54±0.01 0.31±**0.05** 0.57±0.08 0.43±0.05 0.52±**0.01**

Cost ↓ 0.26±0.24 0.73±**1.20** 19.16±0.38 0.54±0.61 0.94±**1.53** 6.65±3.51 14.93±4.24 0.04±**0.07**

PointCircle2 Reward ↑ 0.42±0.07 0.44±0.05 0.85±0.01 0.59±0.02 0.44±**0.06** 0.03±0.75 0.76±0.05 0.55±**0.01**

Cost ↓ 2.10±0.37 1.20±1.28 29.36±1.23 2.05±0.93 0.10±**0.09** 3.99±4.30 18.02±4.17 0.91±**1.46**

PointGoal1 Reward ↑ 0.22±**0.02** 0.55±0.11 0.49±0.02 0.63±0.08 0.30±0.09 0.74±0.03 0.64±0.03 0.06±**0.06**

Cost ↓ 0.91±**0.42** 3.33±1.68 5.27±0.91 2.97±0.82 1.45±0.74 3.90±0.50 5.38±1.05 0.09±**0.11**

PointGoal2 Reward ↑ 0.18±0.01 0.25±0.13 0.41±0.06 0.46±0.08 0.19±0.04 0.78±0.05 0.31±0.07 0.13±**0.05**

Cost ↓ 1.81±0.72 5.34±1.33 5.68±0.48 5.57±0.81 1.02±0.22 15.51±5.75 1.67±1.32 0.81±**0.31**

AntVelo Reward ↑ 0.92±0.03 -1.01±**0.00** 1.00±0.01 0.97±0.02 0.88±0.05 -1.01±0.00 0.89±0.01 0.99±**0.01**

Cost ↓ 0.33±0.19 0.00±**0.00** 11.82±5.56 0.59±0.02 0.21±0.07 0.00±0.00 0.00±0.00 0.43±**0.11**

HalfCheetahVelo Reward ↑ 0.86±**0.01** 0.67±0.05 0.63±0.02 0.98±0.00 0.86±0.01 0.91±0.04 0.89±0.01 0.96±**0.01**

Cost ↓ 0.30±**0.13** 8.56±6.19 0.00±0.00 0.65±0.34 0.27±0.06 0.97±0.12 0.00±0.00 0.14±**0.09**

SwimmerVelo Reward ↑ 0.46±**0.07** 0.20±0.20 0.59±0.07 0.66±0.01 0.45±0.06 0.06±0.25 -0.02±0.05 0.21±**0.19**

Cost ↓ 0.80±**0.19** 2.82±4.19 16.80±7.39 1.26±0.50 2.17±2.91 0.91±1.57 0.23±0.20 0.00±**0.00**

Method CarRun DroneRun CarCircle DroneCircle AntVelocity HalfCheetahVelo

TD3BC Reward ↑ 0.97±0.00 0.36±0.12 0.69±0.01 0.53±0.02 0.99±0.01 0.96±**0.01**

Cost ↓ 0.02±0.03 0.30±0.52 0.00±0.00 0.00±0.00 0.43±0.11 0.14±**0.09**

IQL Reward ↑ 0.96±0.01 0.46±0.03 0.68±0.01 0.35±0.02 0.97±0.01 0.92±**0.02**

Cost ↓ 0.12±0.14 0.71±1.06 0.08±0.10 0.00±0.00 0.38±0.09 0.26±**0.40**

Evaluation Across Varying Cost Limits. While our main results are for the strict cost thresholds (κ equals 5 or 10) representative of safety-critical settings, we also evaluate CARL under more relaxed safety constraints: cost limits of 20 and 40 for Bullet tasks, and 40 and 80 for Safety Gym tasks, following the setup introduced in DSRL. Based on the results shown in Table 1, the safest baseline methods after CARL are FISOR, CAPS, CPQ, and CCAC. However, FISOR is trained solely to minimize cost and does not adapt to different cost limits. CPQ, while safe in some tasks, achieves 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## Training Only On Unsafe

Trajectories. We perform an ablation where we train CARL using either the full dataset, or only unsafe trajectories (those with cumulative cost exceeding the threshold (κ = 5 or 10)).

Figures 3 shows results for training with only unsafe trajectories (blue). CARL generates 60 trajectories per task (20 from each of the three seeds), shown in red. Despite the absence of safe training examples, these trajectories remain within the cost threshold (dashed line) while achieving strong rewards. In BALLCIRCLE, CARL attains rewards around 600–650, enforcing safety with little loss in rewards. In ANTVE- LOCITY, it reaches near-optimal rewards of ∼3000 while staying safe, combining strict constraint satisfaction with top-level performance. In ANTCIRCLE, CARL shifts trajectories into the safe zone while still reaching rewards up to 300+, comparable to the best unsafe cases near the boundary, showing that meaningful reward performance can be retained under strict safety. Overall, these results highlight CARL's ability to recover safe and competitive behavior from purely unsafe data through reward relabeling: transforming unsafe dataset trajectories into safe ones and shifting behavior into the feasible region without substantial loss in rewards.

Figure 2: Results for CARL with varying cost budget limits on three

![8_image_0.png](8_image_0.png) tasks: rewards vs. cost budget (left) and costs vs. cost budget (right).

![8_image_1.png](8_image_1.png)

We also evaluate a naive hard-filtering variant (Appendix Table 8), which removes unsafe transitions entirely, and find that it fails on nearly all tasks, underscoring the importance of CARL's reward relabeling rather than simple data exclusion.

## 7 Summary

This paper introduces an embarrassingly simple framework for solving offline safe reinforcement learning (OSRL). It reformulates the OSRL problem into an unconstrained optimization problem that requires no tuning of Lagrangian multipliers when compared to the vast majority of the previous methods. This framework allows us to naturally leverage powerful off-the-shelf offline RL algorithms and advances directly to effectively solve OSRL problems via constraint-aware reward relabeling. Experimental results on the DSRL benchmark demonstrate the remarkably strong performance of the proposed framework compared to state-of-the-art OSRL methods. very low rewards in two of the safe results. Therefore, for the varying cost limit analysis, we focus our comparison on CAPS and CCAC, both of which show better balance for safety and reward performance. Figures 2 show that CARL improves rewards as the cost budget increases, while keeping normalized costs within the safety threshold (≤ 1). On the challenging **CarCircle2** task, all methods are unsafe at budget 10. When the budget rises to 40 or 80, CARL attains both safety and higher rewards, whereas CAPS and CCAC remain unsafe (Table 6, Appendix). This demonstrates CARL's ability to exploit larger budgets where other methods fail.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. Advances in neural information processing systems, 34:7436–7447, 2021.

Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, and Zhaoran Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=Y4cs1Z3HnqL.

Yassine Chemingui, Aryan Deshwal, Honghao Wei, Alan Fern, and Jana Doppa. Constraint-adaptive policy switching for offline safe reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 15722–15730, 2025.

Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, and Jun Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling. In The Eleventh International Conference on Learning Representations, 2023.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, 34:15084–15097, 2021.

Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, and Esther Luna Colombini. A survey on offline reinforcement learning: Taxonomy, review, and open problems. *IEEE Transactions on* Neural Networks and Learning Systems, 35(8):10237–10257, 2024. doi: 10.1109/TNNLS.2023. 3250269.

Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.

Advances in neural information processing systems, 34:20132–20145, 2021.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In *International conference on machine learning*, pp. 2052–2062. PMLR, 2019.

Javier Garcıa and Fernando Fernandez. A comprehensive curvey on safe reinforcement learning. ´
Journal of Machine Learning Research, 16(1):1437–1480, 2015.

Reproducibility Statement. We include full implementation details in Appendix C, including hyperparameters and training settings. All datasets used are publicly available. An anonymous link to the source code is provided in the abstract to support reproducibility of our experiments. Sven Gronauer. Bullet-safety-gym: A framework for constrained reinforcement learning. Technical report, mediaTUM, 2022.

Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, and Alois Knoll. A
review of safe reinforcement learning: Methods, theories and applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Zijian Guo, Weichao Zhou, Shengao Wang, and Wenchao Li. Constraint-conditioned actor-critic for offline safe reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025.

Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine.

Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv preprint arXiv:2304.10573, 2023.

Michael Janner, Qiyang Li, and Sergey Levine. Offline reinforcement learning as one big sequence modeling problem. *Advances in neural information processing systems*, 34:1273–1286, 2021.

Ze Gong, Akshat Kumar, and Pradeep Varakantham. Offline safe reinforcement learning using trajectory classification. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pp. 16880–16887, 2025.

Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. *arXiv preprint arXiv:2205.09991*, 2022.

Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement learning of human preferences in dialog, 2020. URL https://openreview.net/forum?id= rJl5rRVFvH.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, and Yaodong Yang. Omnisafe: An infrastructure for accelerating safe reinforcement learning research. *Journal of Machine Learning Research*, 25(285):1–6, 2024. URL http://jmlr.org/papers/v25/23-0681.html.

Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Modelbased offline reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 21810–21823. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_ files/paper/2020/file/f7efa4f864ae9b88d43527f4b14f750f-Paper.pdf.

Prajwal Koirala, Zhanhong Jiang, Soumik Sarkar, and Cody Fleming. Latent safety-constrained policy approach for safe offline reinforcement learning. *arXiv preprint arXiv:2412.08794*, 2024.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit qlearning. In *International Conference on Learning Representations*, 2022. URL https:// openreview.net/forum?id=68n2s9ZJWF8.

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. *Advances in neural information processing systems*,
32, 2019.

Hoang Le, Cameron Voloshin, and Yisong Yue. Batch policy learning under constraints. In International Conference on Machine Learning, pp. 3703–3712. PMLR, 2019.

Jongmin Lee, Cosmin Paduraru, Daniel J Mankowitz, Nicolas Heess, Doina Precup, Kee-Eung Kim, and Arthur Guez. COptiDICE: Offline constrained reinforcement learning via stationary distribution correction estimation. In *International Conference on Learning Representations*, 2022. URL
https://openreview.net/forum?id=FLA55mBee6Q.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv preprint arXiv:2005.01643*, 2020.

Qian Lin, Bo Tang, Zifan Wu, Chao Yu, Shangqin Mao, Qianlong Xie, Xingxing Wang, and Dong Wang. Safe offline reinforcement learning with real-time budget constraints. In International Conference on Machine Learning, pp. 21127–21152. PMLR, 2023.

Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, and Ding Zhao.

Constrained decision transformer for offline safe reinforcement learning. In International Conference on Machine Learning, pp. 21611–21630. PMLR, 2023.

Zuxin Liu, Zijian Guo, Haohong Lin, Yihang Yao, Jiacheng Zhu, Zhepeng Cen, Hanjiang Hu, Wenhao Yu, Tingnan Zhang, Jie Tan, and Ding Zhao. Datasets and benchmarks for offline safe reinforcement learning. *Journal of Data-centric Machine Learning Research*, 2024.

Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu. Mildly conservative q-learning for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 35:1711–1724, 2022.

Nicholas Polosky, Bruno C Da Silva, Madalina Fiterau, and Jithin Jagannath. Constrained offline policy optimization. In *International Conference on Machine Learning*, pp. 17801–17810. PMLR, 2022.

Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. *arXiv preprint arXiv:1910.01708*, 2019.

Marc Rigter, Bruno Lacerda, and Nick Hawes. Rambo-rl: Robust adversarial model-based offline reinforcement learning. *Advances in neural information processing systems*, 35:16082–16097, 2022.

Jonah Siekmann, Kevin Green, John Warila, Alan Fern, and Jonathan Hurst. Blind bipedal stair traversal via sim-to-real reinforcement learning. *arXiv preprint arXiv:2105.08328*, 2021.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.

Aivar Sootla, Alexander I Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David H Mguni, Jun Wang, and Haitham Ammar. Saute rl: Almost surely safe reinforcement learning using state augmenta- ´ tion. In *International Conference on Machine Learning*, pp. 20423–20443. PMLR, 2022.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Akifumi Wachi, Wataru Hashimoto, Xun Shen, and Kazumune Hashimoto. Safe exploration in reinforcement learning: A generalized formulation and algorithms. *Advances in Neural Information* Processing Systems, 36:29252–29272, 2023.

Akifumi Wachi, Xun Shen, and Yanan Sui. A survey of constraint formulations in safe reinforcement learning. *arXiv preprint arXiv:2402.02025*, 2024.

Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, and Dongsheng Li. Bootstrapped transformer for offline reinforcement learning. Advances in Neural Information Processing Systems, 35:34748–34761, 2022.

Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. RORL: Robust offline reinforcement learning via conservative smoothing. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=_QzJJGH_KE.

Yihang Yao, Zuxin Liu, Zhepeng Cen, Jiacheng Zhu, Wenhao Yu, Tingnan Zhang, and Ding Zhao. Constraint-conditioned policy optimization for versatile safe reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=FdtdjQpAwJ.

Geraud Nangue Tasse, Tamlin Love, Mark Nemecek, Steven James, and Benjamin Rosman. Rosarl:
Reward-only safe reinforcement learning. *arXiv preprint arXiv:2306.00035*, 2023.

Chen Tessler, Daniel J Mankowitz, and Shie Mannor. Reward constrained policy optimization. arXiv preprint arXiv:1805.11074, 2018.

Haoran Xu, Li Jiang, Li Jianxiong, and Xianyuan Zhan. A policy-guided imitation approach for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 35:4085– 4098, 2022a.

Haoran Xu, Xianyuan Zhan, and Xiangyu Zhu. Constraints penalized q-learning for safe offline reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pp. 8753–8760, 2022b.

Yihang Yao, Zhepeng Cen, Wenhao Ding, Haohong Lin, Shiqi Liu, Tingnan Zhang, Wenhao Yu, and Ding Zhao. Oasis: Conditional distribution shaping for offline safe reinforcement learning. Advances in Neural Information Processing Systems, 37:78451–78478, 2024.

Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning, 2020. URL https://openreview.net/forum?id=BJg9hTNKPH.

Jialong Wu, Haixu Wu, Zihan Qiu, Jianmin Wang, and Mingsheng Long. Supported policy optimization for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 35:31278–31291, 2022.