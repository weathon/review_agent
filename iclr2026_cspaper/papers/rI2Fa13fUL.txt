000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Offline Reinforcement Learning With Genera- Tive Trajectory Policies

Anonymous authors Paper under double-blind review

## Abstract

Generative models have emerged as a powerful class of policies for offline reinforcement learning (RL) due to their ability to capture complex, multi-modal behaviors. However, existing methods face a stark trade-off: slow, iterative models like diffusion policies are computationally expensive, while fast, single-step models like consistency policies often suffer from degraded performance. In this paper, we demonstrate that it is possible to bridge this gap. The key to moving beyond the limitations of individual methods, we argue, lies in a unifying perspective that views modern generative models—including diffusion, flow matching, and consistency models—as specific instances of learning a continuous-time generative trajectory governed by an Ordinary Differential Equation (ODE). This principled foundation provides a clearer design space for generative policies in RL and allows us to propose *Generative Trajectory Policies* (GTPs), a new and more general policy paradigm that learns the entire solution map of the underlying ODE. To make this paradigm practical for offline RL, we further introduce two key theoretically principled adaptations. Empirical results demonstrate that GTP achieves state-of-the-art performance on D4RL benchmarks - it significantly outperforms prior generative policies, achieving perfect scores on several notoriously hard AntMaze tasks.

## 1 Introduction

In offline reinforcement learning (RL), an agent needs to learn a policy from a pre-collected dataset without any further interaction with the environment. This setting creates a fundamental challenge: the agent is asked to generalize from limited, often narrow, experience to an unpredictable world. At the heart of this challenge lies the need for policy expressiveness - the capacity to capture rich, often multi-modal patterns of behavior present in real-world datasets. Traditional offline RL methods rely on simple function approximators and are prone to distribution shift, as the learned policy may choose actions not present in the dataset, leading to inaccurate value estimates (Fujimoto et al., 2019; Wu et al., 2019; Kumar et al., 2020). This has sparked growing interest in generative models - from generative adversarial networks (GANs), variational autoencoders (VAEs), to Energy-Based Models (EBMs) - as powerful tools to model the full complexity and diversity of RL policies (Ho & Ermon, 2016; Ha & Schmidhuber, 2018; Ho et al., 2020; Brahmanage et al., 2023; Messaoud et al., 2024). Most recently, diffusion-based policies have emerged as a powerful paradigm due to their exceptional ability to represent complex, multi-modal distributions (Wang et al., 2023; Janner et al., 2022; Pearce et al., 2023). However, their expressive power comes at a steep price: the slow, iterative sampling process required for generation imposes a significant computational burden, hindering their practical utility. To resolve this, subsequent work has employed consistency-based models to accelerate inference, often enabling one or two-step generation (Ding & Jin, 2024). While remarkably fast, this simplification frequently leads to degraded policy quality, with performance saturating quickly.

This demonstrates a fundamental trade-off between expressiveness and efficiency for generative policies. The research question in this work is: Is it possible to design a policy class that can achieve both policy expressiveness and computational efficiency? A key insight of our work is that the path to resolving this trade-off lies in a general principle that unifies a family of powerful modern generative models. We observe that a spectrum of recent advancements, including diffusion models (Song & Ermon, 2019; Song et al., 2021b), Consistency Models (Song et al., 2023), Consistency Trajectory Models (CTMs) (Kim et al., 2024), and various 1 forms of Flow Matching (Frans et al., 2025; Geng et al., 2025), can all be understood through the lens of a continuous-time generative trajectory governed by an Ordinary Differential Equation (ODE). This unified perspective provides the theoretical foundation for our work, enabling us to conceptualize a policy itself as a full trajectory and thereby design a new class of expressive and efficient policies. Building on this foundation, we introduce *Generative Trajectory Policies* (GTPs), a new policy paradigm that learns the entire solution map of the underlying ODE. By learning the full trajectory, GTPs are not confined to either slow, high-fidelity sampling or fast, low-fidelity shortcuts. Instead, they enable flexible, multi-step, deterministic generation that can achieve high performance even with a few sampling steps. Our key contributions include: i) We propose GTP, a new and highly expressive policy paradigm for offline RL, derived from a unifying framework that connects a family of modern generative models to continuous-time ODE trajectories. ii) We make a practical implementation of the GTP paradigm by developing two key *theoretically-grounded adaptations* that address computational cost, training instability, and misaligned objectives, including a score approximation and a variational framework for value-driven policy improvement. iii) We empirically validate GTP on the D4RL benchmarks, where it achieves state-of-the-art performance, outperforming prior generative and offline RL methods. Notably, our approach achieves perfect scores on several notoriously challenging AntMaze tasks, demonstrating its ability to strike a more favorable balance between expressiveness and efficiency. Our code is included in the supplementary and will be released upon paper acceptance.

## 2 Related Work

We briefly introduce the related work. A more detailed discussion is provided in Appendix A.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 3 A Unified Ode Framework For Generative Models

A cornerstone of many modern generative models is the idea of reversing a process that gradually perturbs data into noise. While prior work typically treats diffusion models, consistency models, and Expressive Policies in Offline RL. Offline RL depends on policies that are expressive enough to capture the diverse, often multi-modal behaviors present in datasets. Conventional choices like Gaussian policies are easy to train but struggle to represent such complexity. Much of the literature has instead advanced from the critic side, regularizing value functions to guard against overestimation (Fujimoto et al., 2019; Wu et al., 2019; Kumar et al., 2020; Kostrikov et al., 2022). While effective, these methods leave the policy class itself underpowered, motivating a complementary line of research: actor-centric approaches that adopt generative models. Early explorations with GANs/VAEs (Ho & Ermon, 2016) and energy-based policies (Messaoud et al., 2024) showed promise, but were often hampered by training instabilities and did not achieve the sample quality of modern generative paradigms, leaving the need for a truly robust and expressive policy class as a key open problem. Continuous-Time Generative Models. A new generation of powerful tools for this task has emerged from the generative modeling community. A spectrum of recent advancements can be understood through the unifying lens of learning a continuous-time trajectory governed by an Ordinary Differential Equation (ODE). This includes score-based diffusion models (Song et al., 2021b), Flow Matching (FM) (Lipman et al., 2023; Frans et al., 2025), Consistency Models (CMs) (Song et al., 2023) and Consistency Trajectory Models (CTMs) (Kim et al., 2024). While this evolution has produced a powerful toolbox of trajectory-based generative models, their potential has not yet been fully realized in the RL domain. The challenge of adapting these powerful but complex models to the specific constraints and objectives of offline RL remains a significant barrier. The Trade-off in Generative Policies. Researchers have begun to apply these powerful generative tools as policies in offline RL. Early work with diffusion-based policies demonstrated their immense potential to model complex action distributions (Wang et al., 2023; Janner et al., 2022; Pearce et al., 2023), but at the cost of slow, iterative inference. In response, consistency-based policies were introduced to accelerate sampling, often to one or two steps (Ding & Jin, 2024), but this frequently resulted in degraded policy performance. This work has established a new and critical trade-off between expressiveness and efficiency. How to properly adapt the underlying principles of these powerful generative models to create a policy class that is both high-performing and efficient in the demanding offline RL setting remains a key open problem that our work aims to address.

![2_image_0.png](2_image_0.png)

$$(1)$$

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

flow matching as separate families, we propose a single unified ODE framework that reveals all these models as instances of the same underlying formulation. The reverse process can be described by a general ODE:
$${\frac{\mathrm{d}\mathbf{x}_{t}}{\mathrm{d}t}}=f(\mathbf{x}_{t},t),$$

## Dt= F(Xt, T), (1)
Where The Vector Field F(X, T) Defines A Deterministic Trajectory From A Point Xt Sampled From A Simple Prior Distribution To A Data Sample X0, And T ∈ [0, T].
Innovation Within This Framework Has Advanced Along Two Complementary Axes: (1) Defining The Vector Field F, As In Diffusion-Based Approaches (Song Et Al., 2021B) And Flow Matching (Lipman Et Al., 2023); And (2) Solving The Ode Efficiently, Which Is A Central Challenge Since Standard Numerical Solvers Require Hundreds Of Discretization Steps, Leading To Slow Inference And Accumulated Errors (Song Et Al., 2023; Kim Et Al., 2024). While The First Axis Has Largely Matured, The Second Remains A Bottleneck And Has Inspired A New Class Of Methods That Directly Learn The Ode'S Solution Map. 3.1 Defining The Vector Field

The choice of the vector field f(xt, t) is crucial, as it determines the exact generative path from noise to data. Prominent methods for defining these dynamics include:
Diffusion Models. Originating from diffusion-based modeling, this class of methods defines the vector field indirectly. The dynamics are determined by the score function, ∇xt log pt(xt), which is the gradient of the log-density of the noisy data distribution. In the corresponding Probability Flow (PF) ODE (Song et al., 2021b), a neural network is trained to approximate this score (or an equivalent denoising function), thereby implicitly specifying the vector field that governs the generative process. Flow Matching. In contrast, Flow Matching (FM) (Lipman et al., 2023) provides a more direct and general framework for learning the vector field. This method involves training a neural network fθ(xt, t) by directly regressing it against a known target vector field that connects the data and prior distributions. This direct regression offers a stable and often more efficient training objective.

## 3.2 Efficiently Solving The Ode By Learning The Solution Map

Instead of relying on numerical integration, a powerful alternative is to model the ODE's solution map directly. We highlight that the true ODE flow map, Φ(xt*, t, s*), which maps a state at time t to its corresponding state at time s, naturally provides a *unifying representation* for a wide family of
generative models:
$$\mathbf{x}_{s}=\Phi(\mathbf{x}_{t},t,\mathbf{s})=\mathbf{x}_{t}+\int_{t}^{t}f(\mathbf{x}_{\tau},\tau)\mathrm{d}\tau.$$
t
f(xτ , τ )dτ. (2)
Figure 1 illustrates this unified viewpoint. Under this formulation, classic approaches such as Consistency Models (Song et al., 2023), Consistency Trajectory Models (Kim et al., 2024), Shortcut Models (Frans et al., 2025), and Mean Flows (Geng et al., 2025) can all be interpreted as approximating specific aspects or limits of the same flow map Φ. For instance, diffusion denoisers estimate its infinitesimal form, whereas consistency models enforce its compositional structure.

$$(2)$$

## 3.3 Learning The Flow Map: A General Parameterization

The exact flow map can be recovered via linear interpolation:

$$\phi(\mathbf{x}_{t},t,s)=\mathbf{x}_{t}+{\frac{t}{t-s}}\int_{t}^{s}f(\mathbf{x}_{\tau},\tau)\mathrm{d}\tau.$$
$$({\mathfrak{I}})$$
$$(4)$$

$$({\boldsymbol{S}})$$

($\small\sf0$). 
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

This parameterization has a natural interpretation: ϕ(xt*, t, s*) serves as an estimate of the endpoint x0, extrapolated from xt using the *average velocity* over [s, t]. Importantly, it allows us to define two
complementary training objectives that together form the core of our unified ODE framework.
1. The Instantaneous Flow Loss (Local Anchor). This objective ensures the learned map is correct
for *infinitesimal steps* by enforcing a boundary condition at the limit s → t:
$$\operatorname*{lim}_{s\to t}\phi(\mathbf{x}_{t},t,s)=\mathbf{x}_{t}-t f(\mathbf{x}_{t},t).$$
ϕ(xt*, t, s*) = xt − tf(xt, t). (5)
$$\Phi(\mathbf{x}_{t},t,s)\approx\Phi(\Phi(\mathbf{x}_{t},t,u),u,s),$$
For convenience, we denote ϕ inst(xt, t) := ϕ(xt*, t, t*) and refer to it as the *Inst Map*. This condition provides a powerful connection to prominent generative modeling paradigms: the right-hand side recovers the denoiser D(xt, t) (i.e., E[x0 | xt]) in diffusion models and the velocity field target in flow matching, f(xt, t) = (xt − ϕ inst(xt, t))/t. In practice, ϕ inst θ
(xt, t) is the model prediction, trained with task-specific targets: for diffusion, the target is the clean sample x0; for flow matching, the target becomes xt − t(x1 − x0). In this sense, the Inst Map acts as a *local anchor*, unifying diffusion-style denoising and flow-matching velocity estimation under a single principle. 2. The Trajectory Consistency Loss (Global Regulator). This objective enforces correctness across long, *multi-step jumps* by requiring self-consistency:
Φ(xt*, t, s*) ≈ Φ(Φ(xt, t, u)*, u, s*), for *t > u > s.* (6)
where u denotes an intermediate time between t and s. Here, the displacement over [*t, s*] must equal the sum of displacements over [*t, u*] and [*u, s*]. In practice, the right-hand side is treated as the target:
Φ(xt*, t, u*) is obtained using an ODE solver (or its learned approximation), and then composed forward to s. The loss is then defined by the discrepancy between the left- and right-hand sides of Eq. (6). This serves as a *global regulator*, enforcing coherence of long trajectories with the additive structure of ODEs. Taken together, the two objectives are complementary: the instantaneous loss enforces fidelity in local dynamics, while the trajectory consistency loss guarantees global coherence across time. We next show how several prominent generative models emerge as a concrete instances of this unified ODE framework.

## 3.4 Prior Models As Special Cases Of The Unified Ode Framework

With the flow map Φ, the reparameterization ϕ, and the two training objectives, many existing generative models naturally emerge as a special case of our unified ODE framework. Below we
summarize the key correspondences; additional details are provided in Appendix B.1 Consistency Models (CMs). CMs (Song et al., 2023) effectively restrict the flow map to the terminaltime evaluation Φ(xt*, t,* 0) in Eq. (2). Their core objective enforces a discrete approximation of the
ODE's compositional property:
$\Phi(\mathbf{c}_{t},t,0)\approx\Phi(\Phi(\mathbf{c}_{t},t,t-\mathbf{A}t),t-\mathbf{A}t,0)$,
which is directly aligned with our Trajectory Consistency Loss in Eq. (6). The EMA bootstrapping operator is a practical implementation of enforcing the same flow-map identity with a one-step target. Consistency Trajectory Models (CTMs). CTMs (Kim et al., 2024) explicitly parameterize
Φ(xt*, t, s*) and train it using a form of trajectory self-consistency. This corresponds exactly to our Guided by this unified perspective, we introduce a general parameterization for learning the ODE flow map. Although Φ provides the ideal target, its integral form is not directly suitable for training.

We therefore adopt a surrogate function ϕ(xt*, t, s*) inspired by (Kim et al., 2024):

$$\Phi(\mathbf{x}_{t},t,s)={\Big(}1-{\frac{s}{t}}{\Big)}\phi(\mathbf{x}_{t},t,s)+{\frac{s}{t}}\mathbf{x}_{t}.$$
$${\mathrm{for~}}t>u>s.$$
xt. (4)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Trajectory Consistency Loss, while their auxiliary diffusion loss plays the role of our Instantaneous Flow Loss. Thus CTMs instantiate both core components of our unified framework. Shortcut Models. Shortcut Models (Frans et al., 2025) learn a finite-time "average velocity" over the interval [*t, t* + d], which corresponds to estimating the integral term in the reparameterized form ϕ of Eq. (3). Taking the limit d → 0 yields the instantaneous velocity f(xt, t) and therefore aligns with Eq. (5), while imposing compatibility across different d values realizes a discrete form of the Trajectory Consistency condition in Eq. (6). Hence Shortcut Models can be viewed as learning finite-time approximations to the same flow map Φ. Mean Flows. Mean Flows (Geng et al., 2025) reparameterize the average velocity

$$u(\mathbf{x}_{t},t,\mathbf{s})={\frac{\mathbf{x}_{t}-\phi(\mathbf{x}_{t},t,\mathbf{s})}{t}}$$

which follows directly from the flow-map definition in Eq. (2) and aligns with our ϕ parameterization in Eq. (3). Instead of applying an explicit multi-step consistency loss, Mean Flows enforce a differential "MeanFlow Identity" relating u and the instantaneous velocity f, serving as an implicit analog of the Trajectory Consistency condition in Eq. (6). Despite using a distinct training mechanism, the learned object is mathematically a special case of our reparameterized flow representation.

## 4 Generative Trajectory Policies For Offline Rl

In the previous section, we established a unified ODE trajectory framework that offers an elegant lens for understanding a family of modern generative models. This lays a theoretical foundation for designing expressive generative trajectory policies. We define a Generative Trajectory Policy (GTP) as a policy class that generates actions by learning the solution map of a continuous-time generative ODE. However, translating these insights into a functional offline RL algorithm is hindered by three practical challenges: Prohibitive Computational Burden. Learning an ODE trajectory requires on-trajectory supervision. As discussed in Section 3.3, this is obtained by numerically solving the ODE backward from t to an intermediate point u using multiple discrete steps (e.g., Euler, Heun), an operation we denote as Solver(xt*, t, u*). When scaled to offline RL, where millions of updates are needed, repeatedly performing this inner-loop solving for every sample makes the overall computation quickly intractable.

Inherent Training Instability. Unlike distillation methods, our framework must learn the entire ODE trajectory *from scratch*. Central to this process is the Inst Map ϕ inst(xt, t), which specifies the ODE's right-hand side through f(xt, t) = (xt − ϕ inst(xt, t))/t. Early in training, the Inst Map is highly inaccurate; yet its outputs are immediately fed back into the solver to generate supervision. This bootstrapping quickly forms a vicious cycle that resembles TD learning (Sutton, 1988)—bad targets yield bad updates—that destabilizes the Actor–Critic loop and often hinders convergence. Misaligned Generative Objective. The default objective of generative models is to match the data distribution, which in offline RL reduces to behavior cloning (BC). While BC is a reasonable baseline, it cannot achieve policy improvement—the central goal of offline RL. Thus, a key challenge is to design a value-aware objective that leverages the generative process not only to imitate observed actions but also to emphasize those leading to higher returns. To address these challenges, we introduce two key techniques tailored to the practical implementation of GTP, as illustrated in Figure 2. The following subsections detail these techniques and show how they jointly enable stable, efficient, and value-driven training.

## 4.1 Efficient And Stable Training Via Score Approximation

A central difficulty in our framework is the reliance on self-referential supervision: the model must repeatedly supply ϕ inst(xt, t) (score estimates) at each solver time point1, which the ODE solver integrates over many iterations. This approach is not only computationally demanding, but also fragile—early-stage errors in the learned vector field immediately corrupt the supervision signals.

1Throughout the paper we use the term *score* for consistency with prior literature, although in our framework it is formally the Inst Map ϕ inst.

In particular, as h → 0, the two objectives coincide in expectation.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

![5_image_1.png](5_image_1.png)

![5_image_2.png](5_image_2.png)

Figure 2: The two core techniques of the GTP implementation: (a) Stable Score Approximation: the target trajectory (green) is contrasted with a reference (red) computed by a multi-step ODE solver (red dashed arrow). The blue dashed arrow denotes a single-step update obtained from our approximate score, which yields the blue trajectory without multi-step integration. (b) Value-Driven Guidance: the BC trajectory (green) is shifted toward high-value regions so that the learned GTP trajectory (blue) approaches the optimal action while remaining aligned with the data. To address it, we replace ϕ inst(xt, t) with a closed-form surrogate anchored to the offline sample,
˜f(xt, t) = (xt − x)/t. The theorem below shows that this yields a training loss asymptotically equivalent to the ideal one.

Theorem 1. Fix a time horizon T > 0*, let* x ∼ pdata, z ∼ N (0, I)*, and define* xt = x + tz.

Define the vector fields f
⋆,˜f : R
d × (0, T] → R
d by f
⋆(xt, t) := xt−E[x|xt]
tand ˜f(xt, t) := xt−x t.

Assume f
⋆(·, t) is Lipschitz in x. Let t = τ0 > τ1 > · · · > τK = u *be a sequence of time points with* step sizes ∆k = τk+1 − τk *and maximal step* h = maxk |∆k|. For a p-th order, zero-stable one-step solver S∆k
[f] : R
d → R
d*, define the multi-step propagation from* t to u as

$$\Psi_{t\to u}^{\mathrm{sol}}[f]:=S_{\Delta_{K-1}}[f]\circ\cdot\cdot\circ S_{\Delta_{0}}[f].$$
$$({\mathfrak{s}})$$
$$({\mathfrak{g}})$$
$$(10)^{\frac{1}{2}}$$
[f]. (7)
Assume further that for each u > s, Φθ(·*, u, s*) : R
d → R
dis Lipschitz in x*, and that solver states* admit bounded second moments independent of h. Define the **ideal** and **practical** *training objectives*

$${\mathcal{L}}_{\mathrm{ideal}}(\mathbf{\theta}):=\mathbb{E}\Big[\big|\Phi_{\mathbf{\theta}}(\mathbf{x}_{t},t,s)-\Phi_{\mathbf{\theta}-}\big(\Psi_{t\to u}^{\mathrm{sol}}[f^{\star}](\mathbf{x}_{t}),u,s)\big|\big|^{2}\Big],$$
2i, (8)
$${\mathcal{L}}_{\mathrm{prac}}(\mathbf{\theta}):=\mathbb{E}\Big[\big\|\Phi_{\mathbf{\theta}}(\mathbf{x}_{t},t,s)-\Phi_{\mathbf{\theta}-}\big(\Psi_{t\to u}^{\mathrm{sol}}[{\tilde{f}}](\mathbf{x}_{t}),u,s)\big\|^{2}\Big],$$
2i, (9)
where Φθ− *denotes the exponentially moving averaged model. Then*
$$\left|\,{\mathcal{L}}_{\mathrm{prac}}(\mathbf{\theta})-{\mathcal{L}}_{\mathrm{ideal}}(\mathbf{\theta})\,\right|=O(h^{p}).$$
p). (10)
Theorem 1 shows that using the closed-form surrogate ˜f changes the objective only by O(h p),
providing theoretical support for our formulation. This replacement makes GTP training both efficient and robust in offline RL, which is further validated empirically by our ablation study (Section 5.3). Further intuition is given in Appendix B.4, where we relate this formulation to consistency training and flow matching. Remark 1 (Computational Efficiency). Using the surrogate score removes the need for multi-step ODE integration. Intermediate points xu *for the trajectory consistency loss are obtained directly as* xu = x + u · z, z ∼ N (0, I), (11)
a one-step perturbation instead of a costly numerical solver.

Proof Sketch. The only difference between the two objectives is that the solver uses the surrogate ˜f instead of the true field f
⋆. Since both are Lipschitz and the solver is p-th order and zero-stable, the propagated states differ by O(h p) in mean square. By Lipschitz continuity of Φθ, this discrepancy transfers directly to the objectives, giving the stated bound. Details are deferred to Appendix B.3.

$$\cdot\,z,\quad z$$
$$z\sim{\mathcal{N}}(0,I),$$

![5_image_0.png](5_image_0.png)

## 4.2 Value-Driven Guidance For Policy Improvement

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Remark 2 (Training Stability). Anchoring supervision to offline data avoids the instability of selfgenerated targets. The model no longer relies on imperfect early-stage estimates of its own vector field, but instead receives a stable analytical signal tied directly to x*. This breaks the cycle of error* propagation and ensures consistent learning from the very beginning of training. To address the misaligned generative objective and unify generative imitation with value-based policy improvement, we formalize a value-weighted training objective for our GTP in Theorem 2, with the detailed derivation provided in Appendix B.5.

Theorem 2 (Advantage-Weighted Objective). Consider the KL-regularized policy optimization problem in offline RL. Its optimal solution can be written as

$$\pi^{*}(a|s)\propto\pi_{\mathrm{BC}}(a|s)\exp\bigl(\eta A(s,a)\bigr),$$
$$(12)$$

$$(13)$$
∗(a|s) ∝ πBC(a|s) expηA(*s, a*), (12)
where A(*s, a*) = Q(s, a) − V (s) is the advantage. Training a generative policy πθ *to match* π
∗is therefore equivalent to solving the weighted generative training objective

$$\operatorname*{max}_{\theta}\,\mathbb{E}_{(s,a)\sim{\mathcal{D}}}\big[\exp\big(\eta A(s,a)\big)\,\ell_{g e n}(\pi_{\theta};a|s)\big],$$

where ℓgen *denotes the standard generative loss (e.g., diffusion loss, or flow-matching loss).*
Theorem 2 confirms that exponential advantage weighting is the theoretically correct way to incorporate value guidance into generative training. Remark 3 (Practical Implementation). For numerical stability, we normalize the advantage weights and truncate negatives:

$$w(s,a)=\exp\biggl(\eta\cdot{\frac{\operatorname*{max}(0,A(s,a))}{\operatorname*{std}(A)+\epsilon}}\biggr).$$
$$(14)$$

This ensures stable optimization while allowing GTP to preferentially imitate high-advantage actions, thereby preserving the robustness of standard generative training.

## 4.3 The Gtp Optimization Framework

Having introduced the two key techniques for the practical implementation of our GTP paradigm, we now integrate them into a complete actor-critic algorithm. The actor is our Generative Trajectory Policy, πθ, represented by the learned solution map Φθ. The critic is a standard double Q-network, Qφ, trained to estimate state-action values. Policy Representation and Action Sampling. The actor Φθ(s, at*, t, τ* ) learns to map a noisy action at at time t to a cleaner action aτ at time τ ≤ t, conditioned on a state s. At inference time, an action is generated by starting with pure Gaussian noise aT ∼ N (0, T2I) and iteratively applying the learned map over a sequence of timesteps T = t0 > t1 *> ... > t*K = 0:
ati+1 = Φθ(s, ati, ti, ti+1), for i = 0*, . . . , K* − 1. (15)
The final denoised sample a0 is the action executed by the policy, i.e., πθ(s*) :=* a0.

Critic Training. We use a standard double Q-network parameterized by ϕ to mitigate the overestimation bias. The critic is trained to minimize the temporal-difference (TD) error using a batch of transitions (*s, a, r, s*′) from the offline dataset:

$a_{t_{i+1}}=\Phi_{\theta}(s,a_{t_{i}},t_{i},t_{i+1}),\quad\mbox{for}\;i=0,\ldots,K-1.$
$${\mathcal{L}}_{\mathrm{{crtic}}}=\mathbb{E}\biggl[\Bigl(r+\gamma\cdot\operatorname*{min}_{j=1,2}Q_{\varphi_{j}^{-}}(s^{\prime},\pi_{\theta^{\prime}}(s^{\prime}))-Q_{\varphi_{j}}(s,a)\Bigr)^{2}\biggr]$$
where φ− and θ
− are the target networks for the critic and actor, respectively, updated via exponential moving average (EMA). Actor Training. The GTP actor is trained by combining the two fundamental objectives in Eqs.(5)- (6) introduced in our unified framework. These objectives are directly modified to incorporate our key adaptations for offline RL. To enable policy improvement, both loss components are weighted by the advantage-based term w(*s, a*), thereby prioritizing high-value actions. Simultaneously, to

$$(15)$$

$$(16)$$

ensure computational feasibility and training stability, the supervision targets are generated using our efficient score approximation instead of a costly ODE solver. First, the Trajectory Consistency Loss, LConsistency, enforces the global self-consistency of the learned flow map Φθ:

$${\mathcal{L}}_{\mathrm{Consistency}}=\mathbb{E}_{(s,a)\sim{\mathcal{D}}}\ \mathbb{E}_{t,\tau,u}\ \mathbb{E}_{\pi\sim{\mathcal{N}}(0,I)}\Big[w(s,a)\,\|\Phi_{\theta}(s,a_{t},t,\tau)-\Phi_{\theta^{-}}(s,\tilde{a}_{u},u,\tau)\|_{2}^{2}\Big]\,.$$
i. (17)
where at = a + t · z, and the teacher's intermediate action a˜u = a + u · z. Second, the Instantaneous Flow Loss, LFlow, anchors the model's local dynamics. As established in Section 3.3, this objective enforces that the learned Inst Map behaves as a correct denoiser in the infinitesimal limit. We implement it by penalizing the prediction error of ϕ inst θ:

$${\mathcal{L}}_{\mathrm{Flow}}=\mathbb{E}_{(s,a)\sim{\mathcal{D}}}\ \mathbb{E}_{t}\bigg[w(s,a)\,\big\|\,a-\phi_{\boldsymbol{\theta}}^{\mathrm{inst}}(s,a_{t},t)\big\|_{2}^{2}\bigg].$$
i. (18)
The total actor loss is then a weighted sum of the two components:

$$(17)$$
$$(18)$$

$\eqref{eq:walpha}$
Lactor = LConsistency + λFlow · LFlow. (19)
The full training pipeline is outlined in Algorithm 1.

Algorithm 1 Training Generative Trajectory Policy (GTP)
1: Initialize actor Φθ, critic Qφ, target networks θ
− ← θ, φ− ← φ 2: for iteration i = 1 to Niter do 3: Sample batch (s, a, r, s′) ∼ D
4: Update critic Qφ using Eq. (16) 5: Compute advantage weights w(*s, a*) using the trained critic 6: Sample time pairs *t > u > τ* , and noise z ∼ N (0, I) 7: Generate noisy actions via score approx.: at = a + t · z, a˜u = a + u · z 8: Update actor Φθ using the weighted loss in Eq. (19)
9: Update target networks: θ
− ← τθ + (1 − τ )θ
−, φ− ← τφ + (1 − τ )φ−
10: **end for**

## 5 Experimental Results

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 In this section, we empirically validate our central claims through experiments. Our evaluation is designed to answer three core questions: (i) whether GTP provides a more expressive generative model for imitating complex behaviors than prior approaches; (ii) whether our two key techniques (Section 4) effectively translate into stable policy improvement that surpasses state-of-the-art offline RL algorithms; and (iii) whether GTP resolves the tension between expressiveness and efficiency. We evaluate our method on a suite of challenging offline reinforcement learning tasks from the D4RL
benchmark (Fu et al., 2020), including the Gym and AntMaze domains. Following the standard setting of Ding & Jin (2024), we evaluate each policy over 10 episodes for Gym tasks and 100 episodes for all other tasks. Unless otherwise noted, diffusion policies and our GTP use K = 5 sampling steps, and consistency policies use K = 2. Hyperparameters are provided in Appendix C.1. Due to space limit, we only show major results in the following. Additional ablations and visualizations in a multi-goal environment are deferred to Appendix D, which provide further evidence of the effectiveness and efficiency of GTP.

## 5.1 Expressiveness As A Behavior Cloning Policy

To assess the intrinsic modeling capacity of our policy architecture, we first conduct experiments in a pure behavior cloning (BC) setting. By setting the value-guidance coefficient η = 0, the objective reduces to a purely generative supervised loss, so the policy is trained only to match the data distribution without policy improvement. Baselines. We compare our method, GTP-BC, against a diverse set of baselines, which includes classic behavior cloning (a Gaussian policy), several strong offline RL methods such as AWAC (Nair et al., 2020) and TD3+BC (Fujimoto & Gu, 2021), and importantly, other generative policies in a BC setting: Diffusion-BC (D-BC) (Wang et al., 2023) and Consistency-BC (C-BC) (Ding & Jin, 2024).

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Results and Analysis. As shown in Table 1, our method achieves strong results across a broad spectrum of tasks, from basic locomotion to complex sparse-reward environments, achieving state-ofthe-art performances in 11 out of 15 tasks. This strong overall performance is reflected in the average scores across both major task suites. In the Gym tasks, our model's average return of 82.3 significantly surpasses both D-BC (76.3) and C-BC (69.7). This highlights the superior modeling capacity of learning the full trajectory map. The performance is even more striking in the notoriously difficult AntMaze suite, where long-horizon planning and multimodality are critical. Here, GTP-BC (66.3) dramatically outperforms all other methods, including the next-best generative approach, C-BC (44.1). This substantial gap suggests that our model's ability to learn the full continuous-time trajectory provides a powerful inductive bias for capturing the complex, temporally extended behaviors required for success. These results confirm the strong expressiveness inherent to the GTP architecture itself. Table 1: Behavior cloning performances on D4RL. We report the mean and standard deviation of normalized scores over 5 random seeds. Bold indicates the best performance among all methods.

Gym BC AWAC Diffuser MoRel Onestep RL TD3+BC DT D-BC C-BC **GTP-BC (Ours)** halfcheetah-m 42.6 43.5 44.2 42.1 48.4 48.3 42.6 45.4 31.0 48.6±0.3 hopper-m 52.9 57.0 58.5 **95.4** 59.6 59.3 67.6 65.3 71.7 83.7±4.0 walker2d-m 75.3 72.4 79.7 77.8 81.8 **83.7** 74.0 81.2 83.1 77.1±1.7 halfcheetah-mr 36.6 40.5 42.2 40.2 38.1 44.6 36.6 41.7 34.4 46.3±0.6 hopper-mr 18.1 37.2 96.8 93.6 97.5 60.9 82.7 67.3 99.7 100.5±0.3

walker2d-mr 26.0 27.0 61.2 49.8 49.5 81.8 66.6 77.5 73.3 83.4±1.8

halfcheetah-me 55.2 42.8 79.8 53.3 **93.4** 90.7 86.8 90.8 32.7 91.3±0.5 hopper-me 52.5 55.8 107.2 108.7 103.3 98.0 107.6 107.6 90.6 109.6±1.9 walker2d-me 107.5 74.5 108.4 95.6 **113.0** 110.1 108.1 108.9 110.4 100.2±2.1 Average 51.9 50.1 75.3 72.9 76.1 75.3 74.7 76.3 69.7 **82.3** AntMaze BC AWAC Diffuser MoRel Onestep RL TD3+BC DT D-BC C-BC **GTP-BC (Ours)** antmaze-u 54.6 56.7 78.9 73.0 64.3 78.6 59.2 71.8 75.8 84.2±6.6 antmaze-ud 45.6 49.3 55.0 61.0 60.7 71.4 53.0 61.2 77.6 79.2±3.2 antmaze-mp 0.0 0.0 0.0 0.0 0.3 10.6 0.0 43.4 56.8 74.4±6.5 antmaze-md 0.0 0.7 0.0 8.0 0.0 3.0 0.0 29.8 31.6 85.0±6.6 antmaze-lp 0.0 0.0 6.7 0.0 0.0 0.2 0.0 14.6 10.2 34.4±5.1 antmaze-ld 0.0 1.0 2.2 0.0 0.0 0.0 0.0 26.6 12.8 40.8 ±6.3 Average 16.7 18.0 23.8 23.7 20.9 27.3 18.7 41.2 44.1 **66.3**

## 5.2 From Imitation To Improvement: Gtp In Offline Rl

Having established GTP's strong performance as an imitation learning agent, we now evaluate the full actor-critic algorithm, GTP, to assess whether our variational policy optimization framework (Section 4.2) can effectively translate this expressiveness into state-of-the-art policy improvement. Baselines. We compare GTP against a suite of strong offline RL algorithms, including CQL (Kumar et al., 2020), IQL (Kostrikov et al., 2021), χ-QL (Garg et al., 2023), ARQ (Goo & Niekum, 2022), IDQL-A (Hansen-Estruch et al., 2023), and the two most relevant generative policy competitors:
Diffusion-QL (D-QL) (Wang et al., 2023), QGPO (Lu et al., 2023), BDM (Chen et al., 2024b), and Consistency-AC (C-AC) (Ding & Jin, 2024). Results and Analysis. Table 2 demonstrates that GTP sets a new state-of-the-art for generative policies in offline RL. On the Gym tasks, our method achieves the highest average return (89.0), outperforming the previous best, D-QL (87.9). The gains are even more pronounced in the challenging AntMaze suite, where GTP (80.6) significantly surpasses both Diffusion-QL (69.6) and QGPO (78.3). Notably, on the antmaze-umaze task, our method achieves a perfect score of 100.0. These results provide strong evidence that our principled, advantage-weighted learning objective successfully leverages the critic's signal to guide the powerful generative policy beyond simple imitation, enabling robust and effective policy improvement.

## 5.3 Ablation Study

We conduct ablations to evaluate the contribution of two key components of GTP: the score approximation scheme (Section 4.1) and the variational value guidance mechanism (Section 4.2). Score Approximation. Replacing our score approximation with signals generated directly by an ODE solver leads to substantially longer training time and weaker performance, even when the solver is limited to at most three steps. Without approximation, training suffers from high variance and slow convergence due to the need for numerical integration at each iteration. In contrast, our approximation

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

Gym CQL IQL χ-QL ARQ IDQL-A D-QL QGPO BDM C-AC **GTP (Ours)**

halfcheetah-m 44.0 47.4 48.3 45 51.0 51.1 54.1 57.0 **69.1** 53.9±0.1 hopper-m 58.5 66.3 74.2 61 65.4 90.5 98.0 **98.4** 80.7 90.3±2.7 walker2d-m 72.5 78.3 84.2 81 82.5 87.0 86.0 87.4 83.1 89.5±0.6 halfcheetah-mr 45.5 44.2 45.2 42 45.9 47.8 47.6 51.6 **58.7** 50.8±0.4

hopper-mr 95.0 94.7 100.7 81 92.1 101.3 96.9 92.7 99.7 101.7±0.3

walker2d-mr 77.2 73.9 82.2 66 85.1 **95.5** 84.4 89.2 79.5 94.2±0.3 halfcheetah-me 91.6 86.7 94.2 91 95.9 **96.8** 93.5 93.2 84.3 93.8±0.8 hopper-me 105.4 91.5 111.2 110 108.6 111.1 108.0 104.9 100.4 112.2±0.6 walker2d-me 108.8 109.6 112.7 109 112.7 110.1 110.7 111.1 110.4 114.2±0.3 Average 77.6 77.0 83.7 76.2 82.1 87.9 86.6 87.3 85.1 **89.0** AntMaze CQL IQL χ-QL ARQ IDQL-A D-QL QGPO BDM C-AC **GTP (Ours)** antmaze-u 74.0 87.5 93.8 97 94.0 93.4 96.4 93.0 75.8 100±0 antmaze-ud **84.0** 62.2 82.0 62 80.2 66.2 74.4 81.0 77.6 81.9±4.4 antmaze-mp 61.2 71.2 76.0 80 **84.2** 76.6 83.6 79.0 56.8 83.3±8.1 antmaze-md 53.7 70.0 73.6 82 84.8 78.6 83.8 84.0 - 94.2±2.0 antmaze-lp 15.8 39.6 46.5 37 63.5 46.4 **66.6** - - 53.5±2.2 antmaze-ld 14.9 47.5 49.0 58 67.9 56.6 64.8 - - 71.0 ±4.9 Average 50.6 63.0 70.1 69.3 79.1 69.6 78.3 - - **80.6**

provides an efficient surrogate that closely aligns with the desired consistency condition, enabling faster optimization and stronger policies. Variational Guidance. We compare GTP with a baseline that combines the generative loss with a linear Q-learning actor loss. As shown in Table 3, this baseline is highly brittle: for typical coefficients (λ = 0.1 or 1.0), training diverges due to exploding critic gradients. Even with λ = 0.01, the baseline occasionally achieves returns close to ours, but this setting is highly sensitive to the critic scale and does not transfer across tasks. In contrast, our variational guidance normalizes and clips critic signals into stable importance weights, yielding consistently high returns across seeds without per-task hyperparameter tuning. Further details and extended comparisons are provided in Appendix B.6. Table 3: Ablation results on hopper-medium-expert-v2 (mean ± std over 5 random seeds). Training time is wall-clock hours per run. Baselines with λ = 0.1 or 1.0 consistently diverged.

## 6 Conclusion

In this work, we introduced *Generative Trajectory Policies*, a new paradigm for offline RL that leverages our proposed unifying perspective of continuous-time generative ODEs. We show that while this framework offers immense expressive power, its direct application is hindered by critical challenges of computational cost, training instability, and objective misalignment. We overcame these obstacles through two theoretically principled adaptations: score approximation for efficient, stable training and a variational, advantage-weighted objective to bridge the gap between imitation and policy improvement. Our empirical results on the D4RL benchmarks validate this approach, showing that GTP establishes a new state-of-the-art for generative policies in offline RL. This work opens a promising direction for harnessing continuous-time dynamics in RL. While inference is fast, reducing the substantial training time of this model class remains an important avenue for future research.

## References

Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. In *International Conference on Learning*

| Method                               | Training Time   | Score       |
|--------------------------------------|-----------------|-------------|
| GTP (ours)                           | 4.26 h          | 112.2 ± 0.6 |
| w/o score approximation (ODE solver) | 5.23 h          | 99.7 ± 1.7  |
| GTP-BC + linear Q-term (λ = 0.01)    | 5.08 h          | 111.4 ± 0.9 |
| GTP-BC + linear Q-term (λ = 0.1)     | Diverged        | -           |
| GTP-BC + linear Q-term (λ = 1.0)     | Diverged        | -           |

Representations, 2018.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Janaka Brahmanage, Jiajing Ling, and Akshat Kumar. Flowpg: action-constrained policy gradient with normalizing flows. In *Advances in Neural Information Processing Systems*, volume 36, pp. 20118–20132, 2023.

Onur Celik, Zechu Li, Denis Blessing, Ge Li, Daniel Palenicek, Jan Peters, Georgia Chalvatzaki, and Gerhard Neumann. Dime: Diffusion-based maximum entropy reinforcement learning. In International Conference on Machine Learning, 2025.

Huayu Chen, Cheng Lu, Zhengyi Wang, Hang Su, and Jun Zhu. Score regularized policy optimization through diffusion behavior. In *International Conference on Learning Representations*, 2024a.

Huayu Chen, Kaiwen Zheng, Hang Su, and Jun Zhu. Aligning diffusion behaviors with q-functions for efficient continuous control. In *Advances in Neural Information Processing Systems*, volume 37, pp. 119949–119975, 2024b.

Tianyu Chen, Zhendong Wang, and Mingyuan Zhou. Diffusion policies creating a trust region for offline reinforcement learning. In *Advances in Neural Information Processing Systems*, 2024c.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 43(1):3–24, 2023. doi: 10.1177/02783649241273668.

Shutong Ding, Ke Hu, Zhenhao Zhang, Kan Ren, Weinan Zhang, Jingyi Yu, Jingya Wang, and Ye Shi. Diffusion-based reinforcement learning via q-weighted variational policy optimization. In Advances in Neural Information Processing Systems, 2024.

Zihan Ding and Chi Jin. Consistency models as a rich and efficient policy class for reinforcement learning. In *International Conference on Learning Representations*, 2024.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In *International Conference on Learning Representations*, 2025.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning, 2020.

Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pp. 20132–20145, 2021.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In *International conference on machine learning*, pp. 2052–2062, 2019.

Divyansh Garg, Joey Hejna, Matthieu Geist, and Stefano Ermon. Extreme q-learning: Maxent rl without entropy. In *International Conference on Learning Representations*, 2023.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In *Advances in Neural Information Processing Systems*, 2025.

Wonjoon Goo and Scott Niekum. Know your boundaries: The necessity of explicit behavioral cloning in offline rl. *arXiv preprint arXiv:2206.00695*, 2022.

David Ha and Jurgen Schmidhuber. World models. ¨ *arXiv preprint arXiv:1803.10122*, 2018. Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In *International conference on machine learning*, pp. 1352–1361, 2017.

Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine.

Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv preprint arXiv:2304.10573, 2023.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In *Advances in Neural* Information Processing Systems, volume 29, 2016.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in neural information processing systems, volume 33, pp. 6840–6851, 2020.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In *International Conference on Machine Learning*, pp. 9902–9915, 2022.

Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, and YAN Shuicheng. Efficient diffusion policies for offline reinforcement learning. In *Advances in Neural Information Processing Systems*, 2023.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In *International Conference on Learning Representations*, 2024.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems, volume 34, pp. 21696–21707, 2021.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In *Advances in Neural Information Processing Systems*, 2021.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In *International Conference on Learning Representations*, 2022.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. In *Advances in Neural Information Processing Systems*, volume 33, pp.

1179–1191, 2020.

Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In Reinforcement learning: State-of-the-art, pp. 45–73. Springer, 2012.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In *International Conference on Learning Representations*, 2023.

Xu-Hui Liu, Tian-Shuo Liu, Shengyi Jiang, Ruifeng Chen, Zhilong Zhang, Xinwei Chen, and Yang Yu. Energy-guided diffusion sampling for offline-to-online reinforcement learning. In International Conference on Machine Learning, pp. 31541–31565, 2024.

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models.

In *International Conference on Learning Representations*, 2025.

Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. In International Conference on Machine Learning, 2023.

Haitong Ma, Tianyi Chen, Kai Wang, Na Li, and Bo Dai. Soft diffusion actor-critic: Efficient online reinforcement learning for diffusion policy. In *International Conference on Machine Learning*,
2025.

Safa Messaoud, Billel Mokeddem, Zhenghai Xue, Linsey Pang, Bo An, Haipeng Chen, and Sanjay Chawla. S2ac: Energy-based reinforcement learning with stein soft actor critic. In International Conference on Learning Representations, 2024.

Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, and Sam Devlin. Imitating human behaviour with diffusion models. In *International Conference on Learning Representations*, 2023.

Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online reinforcement learning with offline datasets. *arXiv preprint arXiv:2006.09359*, 2020.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In *International Conference on Machine Learning*, pp. 8162–8171, 2021.