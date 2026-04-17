# Td-Jepa: Latent-Predictive Representations For Zero-Shot Reinforcement Learning

Marco Bagatella123∗, Matteo Pirotta1, Ahmed Touati1, Alessandro Lazaric1**, Andrea Tirinzoni**1 1 FAIR at Meta, 2 ETH Zurich, ¨
3 Max Planck Institute for Intelligent Systems, Tubingen ¨
mbagatella@ethz.ch, {pirotta,touati,lazaric,tirinzoni}@meta.com

## Abstract

Latent prediction–where agents learn by predicting their own latents–has emerged as a powerful paradigm for training general representations in machine learning. In reinforcement learning (RL), this approach has been explored to define auxiliary losses for a variety of settings, including reward-based and unsupervised RL, behavior cloning, and world modeling. While existing methods are typically limited to single-task learning, one-step prediction, or on-policy trajectory data, we show that temporal difference (TD) learning enables learning representations predictive of long-term latent dynamics across multiple policies/tasks from offline, reward-free transitions. Building on this, we introduce TD-JEPA, which leverages TD-based latent-predictive representations into unsupervised RL. TD-JEPA trains explicit state and task encoders, a policy-conditioned multi-step predictor, and a set of parameterized policies directly in latent space. This enables zeroshot optimization of any reward function at test time. Theoretically, we show that an idealized variant of TD-JEPA avoids collapse with proper initialization, and learns encoders that capture a low-rank factorization of long-term policy dynamics, while the predictor recovers their successor features in latent space. Empirically, TD-JEPA matches or outperforms state-of-the-art baselines on locomotion, navigation, and manipulation tasks across 13 datasets in ExoRL and OGBench, especially in the challenging setting of zero-shot RL from pixels.1

## 1 Introduction

Learning effective state representations is a core challenge in reinforcement learning (RL). Useful representations should capture the dynamics of the environment in a way that supports efficient value estimation and policy optimization across tasks (Watter et al., 2015; Silver et al., 2018; Hafner et al., 2019; Gelada et al., 2019). A promising line of work is latent-predictive (a.k.a. self-predictive) representation learning (Schwarzer et al., 2021; Grill et al., 2020; Guo et al., 2020; Tang et al., 2023), an instance of the joint-embedding predictive architecture (LeCun, 2022, JEPA) paradigm. These algorithms jointly learn a *state encoder* ϕ(s) and a *predictor* P, i.e., a latent dynamics model estimating the representation of a future state s
′: P(ϕ(s)) ≃ ϕ(s
′). Latent-predictive methods thus perform *self-supervised* learning entirely in latent space without any reward or reconstruction of (possibly high-dimensional) states. Several RL methods leverage latent prediction as an auxiliary loss to improve sample efficiency and generalization in reward-based learning (Schwarzer et al., 2021; Guo et al., 2020; Hansen et al., 2024), behavior cloning (Lawson et al., 2025), and curiosity-driven exploration (Guo et al., 2022). As latent-predictive losses do not require any reward, they have been recently used for unsupervised RL: Assran et al. (2025a), Zhou et al. (2025) and Sobal et al. (2025) learn latent world models that can solve goal-reaching tasks via test-time planning, whereas Jajoo et al. (2025) learn a state encoder from trajectory data to define the space of tasks used to optimize zero-shot unsupervised policies. This paper proposes a novel way to instantiate latent-predictive representations for unsupervised RL. While previous methods have largely focused on either one-step dynamics, single-task/single-
∗Work done at Meta.

1Code available at github.com/facebookresearch/td jepa.

1

![1_image_0.png](1_image_0.png)

Figure 1: TD-JEPA trains policies πz parameterized by latents z. The predictor, conditioned on z, predicts the representations of future states visited by πz (left). When trained via TD, the predictor (arrows on the *right*) approximates successor features for each policy, i.e., the weighted barycenter (stars) of representations of visited states (circles). policy training, or relied on on-policy data, we introduce a policy-conditioned, multi-step formulation based on a novel off-policy temporal-difference loss. This objective encourages representations that are predictive not only of immediate transitions, but also of long-term features relevant for value estimation across multiple policies. This property makes such representations and the associated predictors particularly well-suited for integration with off-policy, successor-feature based approaches to zero-shot unsupervised RL (Touati & Ollivier, 2021; Touati et al., 2023; Park et al., 2024), which have recently emerged as a promising solution for applications such as whole-body humanoid control in simulation (Tirinzoni et al., 2025) and on real hardware (Li et al., 2025). We thus instantiate temporal difference latent-predictive representation learning into TD-JEPA, a zero-shot unsupervised RL algorithm which pre-trains four components: a state encoder, a policyconditioned multi-step predictor, a task encoder, and a set of parameterized policies, all of which are learned end-to-end from offline, reward-free transitions. Departing from previous approaches, latent prediction is not merely an auxiliary loss, but rather the core objective that enables TD-JEPA to learn all the components needed to distill zero-shot policies. In fact, the predictor may be leveraged as an approximation of successor features (see Figure 1) to extract policies mapping encoded observations to optimal actions for all reward functions in the span of the learned features. This enables TD-JEPA to perform zero-shot policy optimization for any downstream reward, *entirely in latent space*. Theoretically, for an idealized version of TD-JEPA with linear predictors, we show that 1) the representations do not collapse with a suitable initialization; 2) they recover a low-rank factorization of the successor measures of the trained policies, while the predictor approximates successor features in latent space; 3) they minimize an upper bound on the policy evaluation error for any reward, thus making zero-shot optimization possible. These results build on a novel "gradient matching" argument that extends and generalizes existing theoretical analyses of latent-predictive representations, and connect TD-JEPA with other unsupervised RL methods such as forward-backward (Touati &
Ollivier, 2021) and intention-conditioned value functions (Ghosh et al., 2023). Empirically, we evaluate TD-JEPA on 65 tasks across 13 datasets from ExoRL (Yarats et al., 2022a) and OGBench (Park et al., 2025a), covering locomotion, navigation, and manipulation with both proprioceptive and pixel-based observations. TD-JEPA matches or outperforms state-of-the-art zeroshot baselines across these settings, in particular when learning from pixels, which has proven to be one of the most challenging settings for unsupervised RL so far. Moreover, we ablate several dimensions of the algorithm, demonstrating the importance of learning representations that are predictive of multi-step policy-dependent dynamics, and the advantage of training distinct state and task encoders. Finally, we show that learned representations can be easily reused for offline or online RL, improving over zero-shot policies and learning from scratch.

## 2 Preliminaries

We consider a reward-free Markov Decision ProcessM = (S, A*, P, γ*), where S and A are state and action spaces, P is the probability measure over next states when taking action a in state s as P(ds
′| s, a), and γ ∈ [0, 1) is a discount factor. Executing a Markov policy π : S → Prob(A) induces an unnormalized distribution over visited states, which is referred to as the *successor measure*:

$$M^{\pi}({\mathcal{X}}\mid s,a)=\sum_{t=0}^{\infty}\gamma^{t}\mathrm{Pr}(s_{t+1}\in{\mathcal{X}}|s,a,\pi)\quad\forall\,{\mathcal{X}}\subseteq{\mathcal{S}}.$$
tPr(st+1 ∈ X |s, a, π) *∀ X ⊆ S*. (1)
$$(\mathbb{I})$$

Given a reward function r : S → R and a policy π, the action-value function Qπ r(*s, a*) measures the cumulative discounted reward obtained by the policy over an infinite horizon, i.e., Qπ r(s, a) =
E

P∞
t=0 γ tr(st+1) | *s, a, π*. Action-value functions are connected to successor measures via

$$Q_{r}^{\pi}(s,a)=\int_{s^{+}\in\mathcal{S}}M^{\pi}(\mathrm{d}s^{+}\mid s,a)r(s^{+})=\mathbb{E}_{s^{+}\sim M^{\pi}(\cdot\mid s,a)}\big{[}r(s^{+})\big{]},\tag{2}$$
$$({\mathfrak{I}})$$

which shows a convenient linear decomposition of Qπ
rinto the reward function and the dynamics induced by π. Standard RL agents aim at finding reward-maximizing policies π
⋆ r
(s) ∈
arg maxa∈A Q⋆r
(s, a), where Q⋆
r
(*s, a*) := maxπ Qπ
r
(*s, a*).
Latent-predictive representations. In high-dimensional settings, state encoders ϕ : S → R
dϕ may
be learned to ease the estimation of action-value functions. For instance, if an encoder ϕ is such
that Qπ
r(s, a) = ϕ(s)
⊤w
π
a,r for some vector w
π
a,r ∈ R
dϕ , then the RL process reduces to learning
vectors in R
dϕ rather than high-dimensional functions Qπ
r
(*s, a*). Latent-predictive learning has been
shown to be an effective approach for this problem. In the simplest formulation, latent-predictive
representations capture the one-step latent dynamics of a policy π by minimizing the loss
$${\mathcal{L}}_{\mathrm{one-step}}(\phi,T)=\mathbb{E}_{s\sim\rho,a\sim\pi(\cdot|s),s^{\prime}\sim P(\cdot|s,a)}\left[\|T(\phi(s))-{\overline{{\phi(s^{\prime})}}}\|^{2}\right],$$
2, (3)
where T : R
dϕ → R
dϕ is a (possibly non-linear) predictor of the latent one-step dynamics induced by ϕ and policy π, and ϕ denotes stop-gradient. Notably, optimizing for this loss does not require any decoding or reconstruction, and it only relies on an *unsupervised* dataset D = {(*s, a, s*′)}. Different instantiations of this approach have been shown both empirically and theoretically to produce representations that accurately approximate action-value functions or policies (Guo et al., 2022; Tang et al., 2023; Voelcker et al., 2024; Lawson et al., 2025; Fujimoto et al., 2025).

Successor-features and zero-shot unsupervised RL. Considering a state encoder ψ : S → R
dψ and the associated space of linear rewards Rψ = {r(s) = ψ(s)
⊤z | z ∈ R
dψ }, Q-values for any reward function r(s) = ψ(s)
⊤zr ∈ Rψ can be written as

$$Q_{r}^{\pi}(s,a)=\int_{s^{+}\in\mathcal{S}}M^{\pi}(\mathrm{d}s^{+}\mid s,a)\psi(s^{+})^{\top}z_{r}=\mathbb{E}_{s^{+}\sim M^{\pi}\cap\{s,a\}}\left[\psi(s^{+})\right]^{\top}z_{r}:=F_{\psi}^{\pi}(s,a)^{\top}z_{r},\tag{4}$$  where $\mathbb{E}_{s^{+}\sim M^{\pi}\cap\{s,a\}}=\mathbb{E}_{s^{+}\sim M^{\pi}\cap\{s,a\}}$. The $\mathbb{E}_{s^{+}\sim M^{\pi}\cap\{s,a\}}$ is the 
where F
π ψ
(s, a) ∈ R
dψ captures the *successor features* of π (Barreto et al., 2017). The majority of unsupervised zero-shot RL methods (Touati & Ollivier, 2021; Park et al., 2024; Agarwal et al.,
2025; Jajoo et al., 2025) learn successor features F(*s, a*; z) ≈ F
πz ψ
(*s, a*) for a set of parameterized policies {πz(s)}z∈Z , with Z ⊆ R
d, that are trained to be optimal for all rewards in Rψ, i.e., πz(s) ≈
arg maxa F(*s, a*; z)
⊤z, where F(s, a; z)
⊤z is an approximation of Q⋆r
(*s, a*) for r(s) = ψ(s)
⊤z. At test time, given a reward function r, a vector zr ∈ R
dψ is first obtained by projecting r onto Rψ, and the associated policy πzris then returned.

Given the role played by ψ in defining the space of tasks of interest, with an abuse of terminology, we will refer to ψ as a *task encoder*. On the other hand, we shall call *state encoder* a map ϕ : S → R
dϕ that is used to embed states before feeding them into different networks (e.g., we will train successor features F
π ψ
(ϕ(s), a) and policies π(ϕ(s)) in the latent space given by ϕ). While the zero-shot methods cited so far train the task encoder ψ in different ways, and do not train any explicit state encoder ϕ, the next section will show how multi-step policy-dependent latent-predictive learning can be used to train both simultaneously.

## 3 Latent-Predictive Temporal-Difference Representations

We begin by showing how the latent-predictive loss of Eq. 3 can model multi-step and policydependent dynamics, and how temporal difference (TD) learning allows learning from offline transition data. We will then expand this idea to learn separate state and task embeddings, and finally show how it can be instantiated as a zero-shot unsupervised RL method.

## 3.1 Multi-Step Policy-Conditioned Latent Prediction

Let {πz}z∈Z be a family of policies parameterized by z ∈ Z, and D = {(*s, a, s*′)} be a dataset of transitions. We train a state encoder ϕ : S → R
dϕ and a *policy-dependent* predictor Tϕ :

```
R
 
 dϕ × A × Z → R
           dϕ to be latent-predictive of the long-term dynamics of the policies {πz}, i.e.,

```

$$(6)$$

LMC-JEPA(*ϕ, T*ϕ) = E(s,a)∼D,z∼Z,s+∼Mπz (·|s,a)
-∥Tϕ(ϕ(s)*, a, z*) − ϕ(s+)∥
2, (5)
where *MC-JEPA* stands for Monte-Carlo (MC) JEPA loss, as on-policy samples s
+ ∼ Mπz (·|*s, a*)
are needed for all policies of interest. Intuitively, Tϕ(ϕ(s)*, a, z*) tries to predict future latent states visited by the policy πz. More formally, predictors trained via minimization of LMC-JEPA(*ϕ, T*ϕ) approximate the successor features of ϕ in the latent space induced by ϕ itself. Proposition 1. For any ϕ and Tϕ*, we have the following equivalence* LMC-JEPA(*ϕ, T*ϕ) = E(s,a)∼D,z∼Z -∥Tϕ(ϕ(s)*, a, z*) − F

$$\mathbf{\Phi}(t)={\overline{{F_{\phi}^{\pi z}(s)}}}$$
$$\overline{{{\epsilon\left(s,a\right)}}}\left[\vphantom{\left(\frac{}{}\right)}\right]+\mathrm{const.}$$

Given the connection between Q-functions and successor features (Eq. 4), this result crucially relates multi-step latent prediction with value estimation across multiple policies. More precisely, it implies that the predictor enables policy evaluation and optimization of rewards in the span of ϕ, as we detail at the end of this section. Since F
πz ϕis the successor features of ϕ, with the terminology introduced in Sec. 2, ϕ is used both as a *state encoder*, i.e., to embed states passed to the predictor, and as a task encoder, i.e., defining a space of reward functions. Unfortunately, this loss cannot be estimated on off-policy data since it requires sampling from the successor measures of the given policies. We can however leverage the previous result and the fact that successor features admit a Bellman equation F
πz ϕ(*s, a*) = Es
′∼P (·|s,a),a′∼πz(s
′)[ϕ(s
′) +
γF πz ϕ
(s
′, a′)] (Barreto et al., 2017) to define a temporal-difference version of the previous loss:
LTD-JEPA(ϕ, Tϕ) = E(*s,a,s*′)∼D,z∼Z,a′∼πz(·|s
′)-∥Tϕ(ϕ(s)*, a, z*)−ϕ(s
′)−γTϕ(ϕ(s
′), a′, z)∥
2. (7)
Unlike the Monte Carlo loss of Eq. 5, LTD-JEPA only requires sampling one-step transitions and actions from the given policies, and it can thus be estimated from off-policy, offline datasets.

## 3.2 Training Separate State And Task Representations

While in Eq. 5 and 7 the same encoder ϕ is used for both state and task representations, these need not be the same in practice. Consider, for instance, a robot navigating a building: useful state representations may capture low-level dynamical information critical for control (e.g., joint positions and velocities), while task representations could abstract higher-level contextual features, such as the building's topology. In this case, a single representation might be either too complex, or too abstract: having flexibility over the dimensionality and content of each representation would be
desirable. We thus now introduce an asymmetric variant that trains a distinct encoder ψ : S → R
dψ
to define the set of reward functions of interest (i.e., as a *task* encoder). We first redefine the predictor
as Tϕ : R
dϕ *× A × Z →* R
dψ and the latent-predictive Monte-Carlo loss to train ϕ and Tϕ as
LMC-JEPA(ϕ, Tϕ, ψ) = E(s,a)∼D,z∼Z,s+∼Mπz (·|s,a)-∥Tϕ(ϕ(s)*, a, z*) − ψ(s+)∥
2, (8)
such that Tϕ maps states encoded through ϕ to the long-term dynamics of a policy πz in the latent space induced, this time, by ψ. Similar to Prop. 1, Tϕ approximates the successor features
F
πz
ψ
(*s, a*) of ψ in the latent space induced by ϕ. Symmetrically, we train ψ together with an additional predictor Tψ : R
dψ *× A × Z →* R
dϕ . To do so, we follow existing literature - according to
which joint representations should be predictive of each other (Guo et al., 2020; Tang et al., 2023) - and train ψ and Tψ through the same latent-predictive loss with the roles of ϕ and ψ inverted, i.e.,
LMC-JEPA(ψ, Tψ, ϕ).
2 As before, we can then design an off-policy TD variant of this loss,
LTD-JEPA(ϕ, Tϕ, ψ) = E (*s,a,s*′)∼D
z∼Z,a′∼πz(·|s
′)
$$\begin{array}{c}{{\left[\|T_{\phi}(\phi(s),a,z)-\frac{1}{\psi(s^{\prime})}-\gamma\overline{{{T_{\phi}(\phi(s^{\prime}),a^{\prime},z)}}}\|^{2}\right],}}\end{array}$$
2, (9)

so that ϕ and Tϕ are optimized via LTD-JEPA(ϕ, Tϕ, ψ), while ψ and Tψ via LTD-JEPA(ψ, Tψ, ϕ).

3.3 TD-JEPA REPRESENTATIONS FOR ZERO-SHOT RL

$$(9)$$

Algorithm 1 TD-JEPA for zero-shot RL
Inputs: Dataset D, batch size B, regularization coefficient λ, networks π, Tϕ, ϕ, Tψ, ψ Initialize target networks: T
−
ϕ ← Tϕ, ϕ
− ← ϕ, T
−
ψ ← Tψ, ψ
− ← ψ while not converged do
▷ Sample training batch
{(si, ai, s′i)}
B
i=1 ∼ D, {zi}
B
i=1 ∼ Z, {a
′
i}
B
i=1 ∼ {π(ϕ−(s
′i), zi)}
B i=1
▷ Compute latent-predictive losses LbTD-JEPA(ϕ, Tϕ, ψ) = 1 2B
Pi Tϕ(ϕ(si), ai, zi) − ψ−(s
′ i
) − γT
−
ϕ
(ϕ−(s
′ i
), a′i
, zi)

2 LbTD-JEPA(ψ, Tψ, ϕ) = 1 2B
Pi Tψ(ψ(si), ai, zi) − ϕ−(s
′
i) − γT
−
ψ(ψ−(s
′
i), a′i, zi)

2
▷ Compute orthonormality regularization losses LbREG(ϕ) = 1 2B(B−1)
Pi̸=j(ϕ(si)
⊤ϕ(sj ))2 −
1 B
Piϕ(si)
⊤ϕ(si)
LbREG(ψ) = 1 2B(B−1)
Pi̸=j
(ψ(si)
⊤ψ(sj ))2 −
1 B
Pi ψ(si)
⊤ψ(si)
▷ Compute actor loss {aˆi}
B
i=1 ∼ {π(ϕ(si), zi)}
B i=1 Lbactor(π) = −
1 B
PB
i=1 Tϕ(ϕ(si), aˆi, zi)
Tzi Update ϕ, Tϕ to minimize LbTD-JEPA(ϕ, Tϕ, ψ) + λLbREG(ϕ) Update ψ, Tψ to minimize LbTD-JEPA(ψ, Tψ, ϕ) + λLbREG(ψ) Update π to minimize Lbactor(π)
Update target networks ϕ
−, T
−
ϕ, ψ
−, T
−
ψvia EMA of ϕ, Tϕ, ψ, Tψ space Z as the task embedding space (i.e., Z ⊆ R
dψ ), we train latent policies such that πz(ϕ(s)) =
argmaxa Tϕ(ϕ(s)*, z, a*)
⊤z for all z ∈ Z3. Since Tϕ(ϕ(s)*, z, a*) ≃ F
πz ψ
(s, a) (Proposition 1), this produces optimal policies for all rewards in the span of ψ, learned directly from state representations ϕ(·). At test time, given an inference dataset of rewarded samples Drwd = {(*s, r*)}, the optimal policy πzrcan be retrieved by computing zr through linear regression, e.g. through the closed-form solution zr = argminz E(s,r)∼Drwd [(r − ψ(s)
⊤z)
2] = Es∼Drwd [ψ(s)ψ(s)
T]
−1E(s,r)∼Drwd [ψ(s)r(s)].

Alg. 1 describes TD-JEPA, which combines LTD-JEPA with stabilization strategies, e.g. target networks and covariance regularization. We remark that latent prediction is not auxiliary: it is the core objective that trains encoders and predictors, from which zero-shot policies can be directly distilled.

## 4 Theoretical Analysis

We now provide some theoretical arguments showing how latent-predictive temporal difference representations capture the long-term dynamics of a given set of policies in a way that makes them amenable to zero-shot RL. Following Tang et al. (2023), we consider a simplified tabular setting with linear predictors. We view the representation ϕ (resp. ψ) as a S ×dϕ (resp. S ×dψ) matrix, and consider action-free predictors Tϕ,z (resp. Tψ,z) as dϕ × dψ (resp. dψ × dϕ) matrices for all z. The expression Tϕ(ϕ(s)*, a, z*) in Eq. 8 and 9 thus reduces to T
T
ϕ,zϕ(s), while Mπ(s
′|*s, a*) and P(s
′|*s, a*)
are replaced by Mπz (s
′|s) = Mπz (s
′|*s, π*z(s)) and P
πz (s
′|s) = P(s
′|*s, π*z(s)).

Monte-Carlo losses. We define a (non-latent-predictive) successor measure approximation loss

$${\mathcal{L}}_{\mathrm{SM}}(\phi,\{T_{z}\}_{z},\psi):={\frac{1}{2}}\mathbb{E}_{z\sim Z}\|\phi T_{z}\psi^{\mathsf{T}}-M^{\pi_{z}}\|_{F}^{2}.$$
F . (10)
Minimizing LSM is equivalent to finding the best multilinear approximation to the successor measures Mπz. We prove the following connection with the Monte Carlo latent-predictive loss of Eq. 8. Theorem 1. For fixed ϕ and ψ*, let* T
⋆
z
, T ⋆ϕ,z, T ⋆ψ,z be the optimal predictors for LSM(ϕ, Tz, ψ)
(Eq. 10), LMC-JEPA(ϕ, Tϕ,z, ψ), LMC-JEPA(ψ, Tψ,z, ϕ) *(Eq. 8), respectively. If (A1)* ϕ Tϕ = ψ Tψ = I,
(A2) the state distribution is uniform, and (A3) for all z ∈ Z*, the matrix* P
πz*is symmetric, then*

$$(10)$$

1. *for all* z, ϕT ⋆
z = ϕT ⋆ϕ,z = ΠϕMπz ψ and ψT ⋆ψ,z = ψ(T
⋆
z)
T = ΠψMπz ϕ, where Πϕ *(resp.* Πψ)
is an orthogonal projection on the span of ϕ *(resp.* ψ);
2. ∇ϕLMC-JEPA(ϕ, Tz, ψ) = ∇ϕLSM(ϕ, Tz, ψ) and ∇ψLMC-JEPA(ψ, Tz, ϕ) = ∇ψLSM(*ϕ, T* T
z, ψ).

This result reveals that 1) the optimal predictors for the successor measure loss LSM and the latentpredictive loss LMC-JEPA match, and yield an orthogonal projection of the successor features Mπz ψ onto the ϕ space; 2) the gradients w.r.t. the representations ϕ and ψ, when evaluated at any predictor, match among these two losses, showing that gradient descent on LMC-JEPA would update representations in the direction that reduces LSM, hence improving the approximation of the successor measures. This result follows as a special case of a novel theorem (see App. C) generalizing and implying all previous guarantees for latent-predictive representations (Tang et al., 2023; Khetarpal et al., 2025; Voelcker et al., 2024; Lawson et al., 2025), which we believe is of independent interest. Finally, we remark that, while the assumptions A1-A3 have been considered in all these related works, they can be relaxed, at the price of more involved proofs and notation, as shown in App. C. Temporal-difference losses. We first derive a non-collapse guarantee. While a similar result was originally proved by Tang et al. (2023) for the one-step loss (Eq. 3), our case is more complex since TD latent-prediction can be seen as "doubly latent-predictive" (cf. Eq. 9): T
T
ϕ,zϕ(s) is optimized to match a representation being learned - ψ(s
+) - plus a bootstrapped version of itself - T
T
ϕ,zϕ(s
+).

Theorem 2. Let ϕt and ψt be the representations learned under a continuous-time relaxation of Eq. 9 where, at each step t, the optimal predictors for (ϕt, ψt*) are first computed and then a gradient* step on (ϕt, ψt*) is taken (see App. B.3 for the explicit formulation). Then, the covariance matrices* ϕ T
t ϕt and ψ T
t ψt *are constant over time, i.e.,* ϕ T
t ϕt = ϕ T
0 ϕ0 and ψ T
t ψt = ψ T
0 ψ0 *for all* t ≥ 0.

This result suggests that, if predictors are trained at a faster rate than representations, the overall dynamics preserve their covariance, thus preventing ϕ and ψ from collapsing to trivial solutions (e.g., ϕ = ψ = 0) when properly initialized, e.g., with unitary covariance. As done for MC objectives (Th. 1), we now show that the latent-predictive loss of TD-JEPA is related to forward and backward TD losses for approximating the successor measure (Blier et al., 2021). Theorem 3. *Consider the following TD losses for approximating the successor measure*

$$\mathcal{L}_{\text{fw}}(\phi,T_{z},\psi):=\frac{1}{2}\mathbb{E}_{z\sim Z}\left[\|\phi T_{z}\psi^{\mathsf{T}}-P^{\pi_{z}}-\gamma\overline{P^{\pi_{z}}\phi T_{z}\psi^{\mathsf{T}}}\|_{F}^{2}\right],\tag{11}$$ $$\mathcal{L}_{\text{bw}}(\phi,T_{z},\psi):=\frac{1}{2}\mathbb{E}_{z\sim Z}\left[\|\psi T_{z}\phi^{\mathsf{T}}-(P^{\pi_{z}})^{\mathsf{T}}-\gamma(P^{\pi_{z}})^{\mathsf{T}}\overline{\psi T_{z}\phi^{\mathsf{T}}}\|_{F}^{2}\right].\tag{12}$$

For fixed (ϕ, ψ)*, let* T
⋆
z,fw, T ⋆
z,bw, T ⋆
ϕ,z, T ⋆ψ,z respectively be the optimal predictors for Lfw(ϕ, Tz, ψ),
Lbw(ϕ, Tz, ψ), LTD-JEPA(ϕ, Tz, ψ), LTD-JEPA(ψ, Tz, ϕ)*. Under the same assumptions as Th. 1,*

_for all $z$, $\phi T^{\star}_{\phi,z}=\phi T^{\star}_{z,\mathrm{fw}}=\tilde{\Pi}_{\phi,z}M^{\pi z}\psi$ and $\psi T^{\star}_{\psi,z}=\psi T^{\star}_{z,\mathrm{bw}}$ (resp. $\tilde{\Pi}_{\psi,z}$) is an oblique projection on the span of $\phi$ (resp. $\psi$); $\nabla_{\phi}\mathcal{L}_{\mathrm{TD},\mathrm{IFPA}}(\phi,T_{z},\psi)=\nabla_{\phi}\mathcal{L}_{\mathrm{fw}}(\phi,T_{z},\psi)$ and $\nabla_{\psi}\mathcal{L}_{\mathrm{TD},\mathrm{IFPA}}(\psi,T_{z},\psi)$._
= Π˜ψ,zMπz ϕ, where Π˜ϕ,z 2. ∇ϕLTD-JEPA(ϕ, Tz, ψ) = ∇ϕLfw(ϕ, Tz, ψ) and ∇ψLTD-JEPA(ψ, Tz, ϕ) = ∇ψLfw(ϕ, Tz, ψ).

Similar to Th. 1, the optimal predictors and gradients of TD-JEPA match those of the non-latentpredictive TD losses of Eq. 11 and 12, which are known to recover an approximation of the successor measure for bilinear parameterizations of the form F
T
z B (Blier et al., 2021). Unlike in the Monte Carlo case, here the optimal predictors solve a least-squares TD problem (Boyan, 1999; Precup et al., 2001), yielding the fixed point of a projected Bellman operator whose closed-form expression is an oblique projection (Scherrer, 2010). Policy evaluation and zero-shot RL. Finally, the following result motivates the significance of optimizing the successor measure losses of Eq. 10, 11, and 12. Theorem 4. Let ϕ, ψ have identity covariance matrices. For any reward function r*, let* ωr := (ψ Tψ)
−1ψ Tr be the linear regression weight for representation ψ*. Then, for any* Tz,

$$\max_{r\in\mathbb{R}^{n};\|\tau\|_{2}\leq1}\mathbb{E}_{z\in\mathcal{Z}}\left[\sum_{s\in\mathcal{S}}\left(V_{r}^{\tau_{z}}(s)-\phi(s)^{\mathrm{T}}T_{z}\omega_{r}\right)^{2}\right]\leq2\mathcal{L}_{\mathrm{SM}}(\phi,T_{z},\psi).$$  _Moreover, $\mathcal{L}_{\mathrm{SM}}(\phi,T_{z},\psi)\leq c\mathcal{L}_{\mathrm{tw}}(\phi,T_{z},\psi)$ and $\mathcal{L}_{\mathrm{SM}}(\phi,T_{z},\psi)\leq c\mathcal{L}_{\mathrm{bw}}(\phi,T_{z},\psi)$ for some $c$._
$$\mathbf{\partial})=\nabla_{\psi}{\mathcal{L}}_{\mathrm{fw}}(\phi,T_{z},\psi).$$

Paraphrasing, the policy evaluation error of the technique in Section 3.3 (i.e., embed r into a vector ω through linear regression on ψ, and compute Tϕ(ϕ(s), z)
Tω) is bounded by the successor measure approximation loss and the corresponding TD errors. Both these quantities are indirectly optimized by TD-JEPA (Th. 1, 3), which is thus a sound approach for zero-shot policy evaluation. Moreover, Th. 4 leads to a zero-shot optimality result analogous to Theorem 2 of (Touati & Ollivier, 2021): if the approximation of Mπzis perfect (i.e., Mπz = ϕTzψ T for all z or, equivalently, the TD errors in Eq. 11 and 12 are zero) and the policies πz are optimal for all linear rewards in ψ, then the inference procedure above recovers optimal policies for any (even non-linear) reward function.

## 5 Related Work

Zero-shot RL algorithms. Methods that pre-train agents on unsupervised data to enable zero-shot solution of a wide range of downstream tasks have achieved impressive results, yielding so-called behavioral foundation models (Pirotta et al., 2024; Tirinzoni et al., 2025). The forward-backward algorithm (FB, Touati & Ollivier (2021); Touati et al. (2023)) is an established method, and perhaps the most related to TD-JEPA. FB learns a task encoder and estimates its successor features, essentially finding a bilinear decomposition of policy-conditional successor measures (e.g., Mπz ≈ FzBT). On the other hand, TD-JEPA uses the parameterization Mπz ≈ ϕTzψ T, thus training shared (across tasks) state representations. Moreover, FB adopts a *contrastive* loss, which computes pairwise dot products across each training batch, while TD-JEPA is non-contrastive at its core. FB has been further shown capable of zero-shot imitation (Pirotta et al., 2024) and extended to several settings, including online training regularized by action-free expert data (Tirinzoni et al., 2025), offline training on low-quality data (Jeen et al., 2024b), training on environments with different dynamics (Bobrin et al., 2025), online fine-tuning (Sikchi et al., 2025) and pure exploration (Urp´ı et al., 2025). Other methods, like HILP, PSM, and RLDP, can also be seen as training a task encoder ψ plus successor features on top. HILP (Park et al., 2024) trains ψ through a distance-preserving "goal-reaching" loss, while PSM (Agarwal et al., 2025) learns an affine decomposition of the successor measure for a discrete codebook of policies, and RLDP (Jajoo et al., 2025) trains ψ using chained multi-step latent prediction (Hansen et al., 2024). Jajoo et al. (2025) also observe that regularizing the representation to be orthonormal is crucial to avoid collapse, which we also observe in TD-JEPA (see Alg. 1). Latent-predictive methods. Latent-predictive methods have mostly been applied to define auxiliary losses for a variety of RL settings. Schwarzer et al. (2021) use a latent-predictive loss to enhance state representations learned through a deep Q network. Guo et al. (2020) use latent prediction in POMDPs to encourage two representations (of observations and histories) to be self-predictive of each other, similarly to the asymmetric variant of TD-JEPA and the method explained in Appendix C. Hansen et al. (2024) and Sobal et al. (2025) train a latent dynamics model that enables test-time planning to improve a pre-trained policy or solve goal-reaching tasks, respectively. BYOL-γ (Lawson et al., 2025) trains representations by predicting discounted future latent states visited by the behavior policy. BYOL-γ may thus be seen as an unconditional, Monte Carlo version of TD-JEPA, which is instead policy-conditional and off-policy. The on-policy nature of the algorithm enables Lawson et al. (2025) to implement a bi-directional update of asymmetric representations. TD-JEPA can also recover an asymmetric parameterization, but its practical objective is not bi-directional (i.e., it only implements forward TD prediction, cf. Appendix C.3 for a formal definition of the bi-directional objective). Crucially, BYOL-γ is not proposed as a zero-shot method: the version we evaluate is a novel instantiation in a successor-feature framework. Theory of latent-predictive representations. The theory of latent-predictive representations has been previously studied in several works (Tang et al., 2023; Voelcker et al., 2024; Khetarpal et al., 2025; Lawson et al., 2025), with a particular focus on single-policy, single-step prediction (potentially, bi-directional). Our analysis of MC-JEPA (Section 4) largely takes place in a multi-policy setting, with generic transition kernels over states; as such, it subsumes and expands on several existing results (see Appendix C.2). On the other hand, representation learning through TD losses, as in TD-JEPA, is largely understudied. The closest studies (Blier et al., 2021; Lan et al., 2023) show that, under certain parameterizations and assumptions, TD representation learning can recover low-rank decompositions of the successor measure, (i.e. it optimizes the corresponding approximation loss). While these works rely on having a single policy, we provide a first result connecting latent-predictive TD learning with TD learning over the successor measure for multiple policies.

Laplacian ICVF* HILP FB RLDP BYOL* BYOL-γ* TD-JEPA

DMCRGB (avg) 293.1 ± 15.1 438.7 ± 14.9 391.2 ± 23.8 456.2 ± 8.6 525.7 ± 13.3 513.8 ± 11.6 582.4 ± 9.8 628.8 ± 5.5

walker 309.4 ± 50.0 534.9 ± 61.3 422.8 ± 32.5 324.4 ± 16.6 576.1 ± 35.3 595.2 ± 9.0 648.3 ± 36.5 738.9 ± 3.5 cheetah 242.4 ± 29.6 394.9 ± 30.1 333.0 ± 86.6 622.4 ± 23.1 605.3 ± 23.5 468.0 ± 46.7 679.8 ± 17.1 706.0 ± 4.1

quadruped 430.1 ± 32.3 583.3 ± 17.2 513.9 ± 10.8 475.4 ± 16.7 551.1 ± 23.4 581.8 ± 16.6 570.0 ± 6.6 626.7 ± **13.6** pointmass 190.4 ± 12.4 241.6 ± 35.6 294.9 ± 33.4 402.8 ± 16.8 370.3 ± 12.0 410.3 ± 8.5 431.6 ± 17.4 443.7 ± **10.9**

DMC (avg) 591.1 ± 10.7 619.3 ± 10.3 620.1 ± 8.4 648.2 ± 4.1 610.2 ± 13.5 618.6 ± 10.5 645.4 ± 10.5 661.2 ± 6.3

walker 769.7 ± 4.7 727.0 ± 16.2 796.4 ± 7.7 811.5 ± 5.9 723.9 ± 18.3 746.8 ± 11.0 786.1 ± 9.6 785.2 ± 6.7

cheetah 614.5 ± 18.9 606.3 ± 16.8 618.3 ± 5.8 672.7 ± 4.9 575.6 ± 44.9 622.8 ± 23.9 647.2 ± 9.0 688.7 ± 6.7 quadruped 635.0 ± 38.7 708.5 ± 14.2 694.8 ± **11.0** 595.6 ± 9.1 665.0 ± 13.9 611.8 ± 28.1 683.1 ± 26.1 691.4 ± 5.0 pointmass 345.1 ± 22.4 435.5 ± 11.1 371.0 ± 37.1 513.0 ± 20.0 476.3 ± 39.4 493.0 ± **41.3** 465.1 ± 17.6 479.3 ± **23.6**

OGBenchRGB (avg) 30.58 ± 0.81 25.22 ± 0.55 32.56 ± 0.92 39.89 ± 0.47 39.09 ± 0.59 40.33 ± 0.52 41.58 ± 0.64 41.34 ± **0.45**

antmaze-mn 92.20 ± 2.91 85.80 ± 3.02 84.60 ± 3.59 96.80 ± 0.74 97.60 ± **0.50** 94.40 ± 1.48 98.00 ± 0.73 96.67 ± **1.11** antmaze-ln 35.40 ± 2.97 42.60 ± 2.84 47.00 ± 4.04 76.80 ± **2.33** 63.60 ± 3.89 62.20 ± 3.42 68.80 ± 2.70 74.60 ± **3.35**

antmaze-ms 60.20 ± 3.88 46.20 ± 2.74 71.80 ± 2.22 86.20 ± 2.05 90.60 ± 1.91 90.40 ± 1.97 86.00 ± **3.10** 84.40 ± 3.85 antmaze-ls 7.20 ± 1.98 7.20 ± 1.20 23.60 ± 1.83 27.40 ± **2.78** 21.80 ± 1.01 26.60 ± 2.23 28.60 ± 1.71 28.80 ± **2.50**

antmaze-me 0.00 ± 0.00 0.00 ± 0.00 0.20 ± 0.20 1.80 ± 1.09 0.80 ± 0.44 1.20 ± 1.00 3.20 ± **1.98** 0.20 ± 0.20 cube-single 73.80 ± **3.53** 34.80 ± 7.03 56.40 ± 3.82 62.00 ± 2.27 63.20 ± 3.91 75.40 ± 2.58 76.40 ± **3.24** 67.80 ± 3.67 cube-double 1.60 ± **0.72** 0.80 ± 0.44 1.60 ± **0.58** 1.20 ± 0.61 2.20 ± 1.31 2.40 ± **0.65** 1.40 ± 0.67 3.00 ± **0.91**

scene 2.80 ± 1.12 8.40 ± 1.45 5.40 ± 1.63 4.20 ± 0.87 9.40 ± 1.33 8.80 ± 1.64 11.20 ± 1.82 14.20 ± **2.22** puzzle-3x3 2.00 ± **1.40** 1.20 ± 0.44 2.44 ± 0.99 2.60 ± 0.79 2.60 ± 0.79 1.60 ± **0.40** 0.60 ± 0.31 2.40 ± **0.83**

OGBench (avg) 14.81 ± 1.32 30.87 ± 0.58 37.98 ± 1.11 39.04 ± **0.66** 27.07 ± 0.83 26.42 ± 0.83 30.42 ± 0.94 37.98 ± **0.77**

antmaze-mn 50.00 ± 4.94 79.80 ± 2.62 83.60 ± **2.63** 73.00 ± 2.72 74.60 ± 4.15 58.40 ± 2.00 51.40 ± 1.55 70.40 ± 3.72

antmaze-ln 21.60 ± 3.90 58.40 ± **1.90** 52.60 ± 3.86 36.80 ± 4.28 36.40 ± 4.66 26.60 ± 3.03 21.80 ± 3.57 57.20 ± **4.25** antmaze-ms 21.40 ± 4.32 39.00 ± 3.30 50.60 ± 2.46 70.40 ± **3.95** 58.40 ± 3.29 60.60 ± 5.07 45.60 ± 2.84 61.56 ± 4.53

antmaze-ls 11.80 ± 1.47 13.20 ± 1.64 12.20 ± 1.75 49.80 ± **5.64** 19.60 ± 2.73 25.80 ± 4.28 20.20 ± 1.80 40.60 ± 2.51 antmaze-me 0.80 ± 0.61 0.00 ± 0.00 2.00 ± 0.84 51.60 ± **2.65** 4.80 ± 2.35 11.40 ± 2.29 19.60 ± 2.53 20.20 ± 2.39 cube-single 15.11 ± 1.49 20.40 ± 1.93 74.20 ± **3.53** 49.60 ± 3.83 19.80 ± 2.41 22.00 ± 3.16 79.40 ± **2.83** 34.20 ± 2.88

cube-double 2.00 ± 0.42 5.00 ± 0.80 20.00 ± **2.72** 2.60 ± 0.43 3.80 ± 0.76 4.40 ± 0.72 2.60 ± 0.67 3.60 ± 0.78 scene 7.80 ± 1.28 45.40 ± 2.29 43.80 ± **1.90** 12.80 ± 1.61 11.60 ± 1.57 15.40 ± 1.37 14.40 ± 2.32 38.44 ± 1.37 puzzle-3x3 2.80 ± 0.68 16.60 ± 0.73 2.80 ± 0.68 4.80 ± 0.68 14.60 ± 0.90 13.20 ± 1.91 18.80 ± **0.44** 15.60 ± 1.11

Table 1: Performance of zero-shot algorithms for DMC (reward) and OGBench (success rate) with either proprioception or RGB inputs. We report means and standard errors across seeds. Numbers are bold for top algorithms if confidence intervals overlap.

## 6 Experiments

We benchmark zero-shot performance across a diverse set of problems, including 4 locomotion/navigation domains from ExoRL/DMC (Tassa et al., 2018; Yarats et al., 2022a), as well as 9 navigation/-
manipulation domains from OGBench (Park et al., 2025a). The former suite involves reward-based tasks and high-coverage data, while the latter evaluates goal-reaching and provides low-coverage datasets4. We consider both proprioceptive and pixel-based variants of all domains, and report expected returns/success rates across a set of tasks (4-8 depending on the domain) as main evaluation metric. In DMC, we often normalize returns by the maximum achievable (1000). We structure our evaluation in four parts: (i) a comprehensive evaluation of TD-JEPA with respect to existing zero-shot methods; **(ii)** an ablation over the prediction target, measuring the impact of multistep, policy-aware dynamics modeling; **(iii)** a comparison of TD-JEPA to its symmetric variant that learns a shared state-task encoder ϕ; and **(iv)** a demonstration of fast adaptation from pre-trained state representations. Further results are presented in App. D, and implementation details in App. E.

How does TD-JEPA compare to zero-shot RL algorithms? We first compare TD-JEPA to three groups of successor-feature-based zero-shot RL baselines:5
- *Laplacian* (Wu et al., 2019), *HILP* (Park et al., 2024), and FB (Touati & Ollivier, 2021) are established zero-shot methods that train a task encoder ψ, without specific learning objectives for a state encoder.

- *BYOL*⋆(Grill et al., 2020), *BYOL-*γ
⋆(Lawson et al., 2025) and *RLDP* (Jajoo et al., 2025) learn a state encoder ϕ via latent-predictive learning, which we then use as a task encoder for successor features (learned through a contrastive loss in the case of RLDP).

- *ICVF*⋆(Ghosh et al., 2023) learns a multilinear decomposition of the successor measure via expectile regression, yielding both state and task encoders on top of which we train successor features.

For a fair comparison, each method is tuned over comparable hyperparameter grids and adopts the same architecture: in particular, the state input is always passed through an explicit state encoder 4We additionally apply BC regularization in OGBench based on Park et al. (2025b), as detailed in App. E.6 5Notice that only Laplacian, *HILP*, FB and *RLDP* are standard zero-shot unsupervised RL algorithms, while BYOL, *BYOL-*γ, and *ICVF* (henceforth marked with a ∗) are representation learning methods: their instantiation in a zero-shot framework is novel and designed to investigate the impact of different representations.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

before being fed into, e.g., the successor features estimator F(*s, a*; z)
6. We find that this protocol results in significant improvements in zero-shot performances, even for existing methods (e.g., 1.3× and 2.4× higher than overlapping pixel-based results for the methods presented in Park et al. (2024) and Jajoo et al. (2025), respectively), as displayed in Tab 1. When considering suite-aggregated performance, we find that TD-JEPA is on par or better than the best performing baseline in each suite. Given the diverse nature of suites (proprioception vs pixels), domains (locomotion, navigation, manipulation) and datasets (high- vs low-coverage), many algorithms unsurprisingly achieve strong performance in some configurations while under-performing in others. We thus additionally measure how consistently well each algorithm performs by computing the probability of improvement (Agarwal et al., 2021) across all domains in Fig. 2. We find that TD-JEPA is consistently among the top performing algorithms, whereas most baselines perform well on a narrow subset of problems. For instance, while TD-JEPA is only slightly preferable to FB and HILP from proprioception, it is significantly better than them in visual domains. Similarly, BYOL-γ is slightly better than TD-JEPA in OGBenchRGB, but it is significantly worse in DMCRGB and OGBench. Finally, we note that latent-predictive methods tend to be generally preferrable in pixel-based domains.

Which dynamics should latent-predictive zero-shot algorithms model? The baselines based on BYOL and BYOL-γ are algorithmically closest to TD-JEPA, and allow a precise investigation on the dynamics to model. While BYOL⋆and BYOL-γ
⋆approximate one-step and multi-step transitions of the behavioral policy, respectively, TD-JEPA models multi-step transitions *of the zero-shot* policies. While approximating the behavioral dynamics can be effective for expert-like data (i.e., in OGBench), we observe a general pattern suggesting that directly modeling policy-conditional successor measures is on average beneficial, as reported in Fig. 3 (left).

Should state and task representations differ? TD-JEPA trains separate state and task encoders:
while this may grant a better approximation of successor measures, sharing state and task representations while optimizing a single objective (see Section 3.1) may in practice be more efficient. We measure the difference in per-task normalized performance between TD-JEPA and a symmetric 6On average, explicit state encoders actually improve the performance for existing methods, see App. D.1.

![9_image_0.png](9_image_0.png)

variant in Figure 3 (*right*): we observe that this variant performs comparatively rather well, while relying on a single predictor-encoder pair. However, using distinct state and task embeddings tends to improve empirical performance more often than not. Are state representations beneficial for fast adaptation? While the previous evaluations have focused on aggregated zero-shot performance, we now investigate an additional benefit of explicit state representations: fast adaptation at test-time. Given a pixel-based task, we initialize the agent with the zero-shot policy πz and critic learned at pre-training, and we either *fine-tune* the whole model via TD3 (Fujimoto et al., 2018) or keep the pre-trained state encoder *frozen*. We consider two RL adaptation protocols (i) **Offline**: a transition-reward dataset is provided Drew = {(s, a, s′, r)}
and TD3 updates are applied offline; (ii) **Online**: an online buffer is additionally collected over time and batches are sampled by mixing it with the offline buffer mentioned above (following the unsupervised-to-online protocol of Kim et al. (2024)). Figure 4 reports results for each DMC domain for the task in which the gap between online and zero-shot algorithms is largest; we consider TD- JEPA and FB as strong, representative algorithms among self-predictive and contrastive methods. We first observe that fine-tuning pre-trained agents leads to large gains in sample efficiency w.r.t. training from scratch, and reaches the asymptotic performance of TD3. More interestingly, frozen representations are often sufficient for downstream learning, and do not need further fine-tuning. We refer to App. D.3 and App. E.7 for further results and details, respectively.

## 7 Conclusion

Through the introduction of a novel temporal-difference latent-predictive loss, we presented a zeroshot unsupervised RL method that operates entirely in latent space and can be shown to recover a factorization of the successor measures of multiple policies. Our method tackles a fundamental representation learning problem for control, and highlights a connection between downstream performance and accurate modeling of the successor measure. We thus suggest that flexible representations for RL, particularly for value estimation and optimization on downstream tasks, should be predictive of future behaviors, and precisely capture their *diverse* and *long-term* nature. Empirically, we found that TD-JEPA matches the best zero-shot methods when learning from proprioception, and exceeds them when learning from pixels, while also retrieving state representations that allow fast downstream adaptation. As formal guarantees rely on an assumption of symmetry, one exciting direction for future work may study learning objectives that are compatible with asymmetric successor measures, yet remain amenable to practical optimization. On a practical note, we believe that benchmarking latent-predictive zero-shot objectives on large-scale, real robotic dataset can shed further light on opportunities and limitations of this promising framework.

## References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc Bellemare.

Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*, 2021.

Siddhant Agarwal, Harshit Sikchi, Peter Stone, and Amy Zhang. Proto successor measure: Representing the behavior space of an rl agent. *ICML*, 2025.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025a.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. *arXiv preprint arXiv:2506.09985*, 2025b.

Andre Barreto, Will Dabney, R ´ emi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and ´
David Silver. Successor features for transfer in reinforcement learning. *NeurIPS*, 2017.

Chethan Bhateja, Derek Guo, Dibya Ghosh, Anikait Singh, Manan Tomar, Quan Vuong, Yevgen Chebotar, Sergey Levine, and Aviral Kumar. Robotic offline rl from internet videos via valuefunction pre-training. *arXiv preprint arXiv:2309.13041*, 2023.

Leonard Blier, Corentin Tallec, and Yann Ollivier. Learning successor states and goal-dependent ´
values: A mathematical viewpoint. *arXiv preprint arXiv:2101.07123*, 2021.

Maksim Bobrin, Ilya Zisman, Alexander Nikulin, Vladislav Kurenkov, and Dmitry Dylov.

Zero-shot adaptation of behavioral foundation models to unseen dynamics. *arXiv preprint* arXiv:2505.13150, 2025.

Justin A Boyan. Least-squares temporal difference learning. *ICML*, 1999. Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. *ICML*, 2018.

Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.

Advances in neural information processing systems, 34:20132–20145, 2021.

Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actorcritic methods. *ICML*, 2018.

Scott Fujimoto, Pierluca D'Oro, Amy Zhang, Yuandong Tian, and Michael Rabbat. Towards general-purpose model-free reinforcement learning. *ICLR*, 2025.

Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G Bellemare. Deepmdp:
Learning continuous latent space models for representation learning. *ICML*, 2019.

Dibya Ghosh, Chethan Anand Bhateja, and Sergey Levine. Reinforcement learning from passive data via latent intentions. *ICML*, 2023.

Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond, Elena ´
Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. *NeurIPS*, 2020.

Zhaohan Guo, Shantanu Thakoor, Miruna Pˆıslar, Bernardo Avila Pires, Florent Altche, Corentin ´
Tallec, Alaa Saade, Daniele Calandriello, Jean-Bastien Grill, Yunhao Tang, et al. Byol-explore: Exploration by bootstrapped prediction. *NeurIPS*, 2022.

Zhaohan Daniel Guo, Bernardo Avila Pires, Bilal Piot, Jean-Bastien Grill, Florent Altche, R ´ emi ´
Munos, and Mohammad Gheshlaghi Azar. Bootstrap latent-predictive representations for multitask reinforcement learning. *ICML*, 2020.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. *ICML*, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. *ICLR*, 2020.

Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. *ICLR*, 2024.

Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, et al. Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. *arXiv preprint arXiv:2502.01143*, 2025.

Pranaya Jajoo, Harshit Sikchi, Siddhant Agarwal, Amy Zhang, Scott Niekum, and Martha White.

Regularized latent dynamics prediction is a strong baseline for behavioral foundation models. Workshop on Reinforcement Learning Beyond Rewards @ RLC 2025, 2025.

Scott Jeen, Tom Bewley, and Jonathan Cullen. Zero-shot reinforcement learning from low quality data. *Advances in Neural Information Processing Systems*, 37:16894–16942, 2024a.

Scott Jeen, Tom Bewley, and Jonathan Cullen. Zero-shot reinforcement learning from low quality data. *NeurIPS*, 2024b.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. RSS 2024 Workshop: Data Generation for Robotics, 2025.

Khimya Khetarpal, Zhaohan Daniel Guo, Bernardo Avila Pires, Yunhao Tang, Clare Lyle, Mark Rowland, Nicolas Heess, Diana Borsa, Arthur Guez, and Will Dabney. A unifying framework for action-conditional self-predictive reinforcement learning. *AISTATS*, 2025.

Junsu Kim, Seohong Park, and Sergey Levine. Unsupervised-to-online reinforcement learning.

arXiv preprint arXiv:2408.14785, 2024.

Diederik P Kingma. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*,
2014.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. *NeurIPS*, 2020.

Charline Le Lan, Stephen Tu, Mark Rowland, Anna Harutyunyan, Rishabh Agarwal, Marc G Bellemare, and Will Dabney. Bootstrapped representations in reinforcement learning. *arXiv preprint* arXiv:2306.10171, 2023.

Daniel Lawson, Adriana Hugessen, Charlotte Cloutier, Glen Berseth, and Khimya Khetarpal. Selfpredictive representations for combinatorial generalization in behavioral cloning. arXiv preprint arXiv:2506.10137, 2025.

Yann LeCun. A path towards autonomous machine intelligence. *Open Review*, 2022.

Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, and Guanya Shi. Bfm-zero: A promptable behavioral foundation model for humanoid control using unsupervised reinforcement learning. *arXiv preprint arXiv:2511.04131*, 2025.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. ICLR, 2023.