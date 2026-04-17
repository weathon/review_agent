000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Dynamical systems naturally evolve on structure-rich manifolds, yet naive machine learning models learn dynamics in flat Euclidean embeddings. This mismatch forces models to *implicitly* learn geometric constraints, resulting in dataintensive training and limited generalization across operating conditions. In this work, we demonstrate how leveraging geometry-informed inductive biases reduces the dependency on larger models to achieve robust generalisation. We investigate a dissipative and a conservative system as use-cases. In the dissipative case, we *identify* a 2-dimensional heat transfer system using a linear state-space formulation where the state operator is constrained to be symmetric positive definite via Riemannian optimization. In the conservative case, we *model* an 18-dimensional Fermi-Pasta-Ulam-Tsingou (FPUT) system on its native symplectic manifold using a symplectic Hamiltonian neural network (SHNN). In the latter case we reveal how structurally-naive models suffer from energy drift when referenced against the true energy surface leading to fragile roll-out generalization, unlike SHNNs which conserve phase-space volume along the correct energy level.

## 1 Introduction

Most real-world physical systems are governed by underlying dynamical systems. Whether modeling from first principles or learning from data, we can reasonably assume the system temporally evolves on some lower-dimensional manifold embedded in a high-dimensional physical coordinate representation. This manifold can be described as a generalized space whose geometric structure is time-invariant thereby preserving the system's structural properties (symmetries, invariances, conservation laws) even as trajectories are observed at different temporal or parametric scales. Thus, imposing structure-preserving inductive biases in machine-learning models that operate on such spaces improves generalization and reduces reliance on large models and volumes of data to implicitly recover the underlying structure. This is a necessary path forward for modeling real-world physical systems across engineering domains. In this paper, we reinforce this claim with a comparative study of structure-preserving versus structurally naive approaches on one dissipative system and one conservative system. In the former use-case, we present a system identification (SID) of a dynamical system via a structure-preserving, linear matrix model to learn the phase-space dynamics of a 2-dimensional heat transfer system, while in the latter, an established structure-preserving neural-network architecture is adopted to learn the conservative dynamics of an 18-dimensional system. Our overarching aim is to illustrate how superior generalization and stability can be achieved with smaller, yet structure-aware, models.

## 1.1 Physics-Informed Biasing

Encoding prior knowledge into machine learning through physics informed-biases is a growing research topic aimed at improving training parsimony and generalization. Most popular are physicsinformed neural networks (PINNs) which incorporate constraints and laws directly into the learning via the loss function (Raissi et al., 2020). PINNs overcome the computational expense often encountered with numerical solvers when solving forward and inverse problems for partial differential

## Abstract

Anonymous authors Paper under double-blind review

# Structure-Preserving Machine Learning Of Dynamical Systems: A Case For Smaller Mod- Els

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 equations (PDEs) for high-dimensional and non-linear applied problems, for example Raissi et al. (2019), Jagtap et al. (2022), Berardi et al. (2025). Traditionally, PINNs encode physics through loss penalties on the PDE residual and boundary conditions (BCs), rather than by encoding inductive biases into the model architecture. Consequently, the residual-based learning still conditions generalization on the representativeness of the training data, sampling, and the implicit regularization of stochastic optimization, where the optimizer resides in flat Euclidean parameter space and is not aware of the non-Euclidean structure of the phase-space (e.g., symplectic form) (Zhang et al., 2017). Once trained, structurally-naive models often struggle to extrapolate to unseen initial conditions or parameters. This has motivated structure-preserving approaches such as Hamiltonian/symplectic neural networks (HNNs/SHNNs) (Greydanus et al., 2019; David & Mehats, 2023) and SympNets (Jin et al., 2020) for learning con- ´ servative systems, symmetry-equivariant neural networks (Wang et al., 2022) and thermodynamicsinformed for learning non-conservative systems (Hernandez et al., 2023; Barbaresco, 2022), and ´ other structure-oriented inductive-biasing approaches for the data-driven discovery of intrinsic dynamics (Champion et al., 2019; Floryan & Graham, 2022). We argue that extensive knowledge of structure-rich spaces from classical mechanics and differential geometry remains an underutilized opportunity for developing data-driven models that can generalize at a structural level rather than merely fitting data with predefined constraints.

## 2 Geometric Underpinning Of Physical Systems

Dynamical systems underlying physical phenomena can be broadly classified by their energy behavior: some conserve energy (conservative), while others dissipate it (dissipative). All conservative systems can be represented in a Hamiltonian formulation whose dynamics evolve on a symplectic manifold, preserving volume form as they flow through phase space. On the other hand, dissipative systems lack symplectic structure; their natural geometries depend on the choice of model formulation. For example, when dynamics converge to stable, low-dimensional (non-chaotic) attractors, models such as linear state-space matrix representations are suitable, where the gradient flows on Riemannian manifolds. Conceptually, one can view mathematical models of dynamics as living in a hierarchy of increasingly general geometric spaces, where parameter perturbations deform the vector field and are observed as changes in flow trajectories on an underlying manifold (Figure 1). Most governing laws for physical systems are smooth, allowing us to work on smooth manifolds with well-defined tangent spaces and differentiable maps (see, Asselmeyer-Maluga & Brans 2007). We illustrate these ideas with the following two use-cases.

## 2.1 Dissipative Use-Case: Heat Transfer System

The conduction dynamics of a material system undergoing one-dimensional heat transfer laterally along its thickness can be described by:

$${\frac{\partial u(x,t)}{\partial t}}={\frac{k}{\rho c}}{\frac{\partial^{2}u(x,t)}{\partial x^{2}}}+{\frac{q(x,t)}{\rho c}},$$
$$(1)$$
ρc, (1)
where: k is the thermal conductivity of the material (W/mK), u(*x, t*) is the temperature as a function of spatial coordinate x and time t, ρ is the density of the material (*kg/m*3), c is the specific heat capacity of the material (J/kgK), *k/ρc* is the thermal diffusivity and q(*x, t*) is the internal heat generation per unit volume (W/m3).

Considering the discrete nature of physical systems, we must approximate the continuous temperature field in 1 by reducing it to a discrete domain. We adopt a linear time-invariant state space model
(LSSM) approach whose matrix representation offers a compact formulation while preserving the geometric structure that governs their evolution. We assume a discrete approximation of the material system as m=2 temperature states Text1, Text2 ∈ T (on either side of the material thickness) and an external forcing Text which represents the ambient temperature influencing the dynamics through convection, directly influencing Text1 only. These are represented in a continuous-time LSSM formulation as follows:
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 where, T ∈ R
2is a state vector, U ∈ R
2×1is input vector, A ∈ R
2×2contains the information about the unforced dynamics of all system states T while B ∈ R
2×1 determines how the input matrix U (forcing) influences the states T. Further, U*exti,extj* is the thermal transmittance (W/m2K). The structure of A reflects the physical topology of the discretized domain via the lumped parameter approach as described in Xuereb Conti et al. (2023) where, where A = VΛT
−1, Λ = diag(λ1, λ2*, . . . , λ*N ) is the diagonal matrix of eigenvalues and V is the matrix of eigenvectors of A. Let T ∈ R
2 be the state vector and U ∈ R
1×1the input vector. A remains invariant to the order of the system provided the topological connectivity between the states is preserved.

In order to solve the system in 2, we must convert time-continuous A to discrete-time ΦA via the matrix exponential expansion, as follows:

$$\Phi_{A}=e^{A\tau}\;\mathrm{and}\;\Phi_{B}=A^{-1}(e^{A\tau}-I)B$$
Aτ − I)B (3)
where τ is the time-step for discretization. Further expansion of 16 can be found in Appendix A.

$\eqref{eq:walpha}$. 

![2_image_0.png](2_image_0.png)

## 2.1.1 Stable Generalization On The Symmetric Positive Definite Manifold

Formulating dynamical systems in the state space matrix format offers advantages and opportunities for preserving natural structure related to geometry governed by the invariant and symmetry features of the matrices. In several instances, the formulation of system matrix A in equation 2 belongs to the symmetry matrix manifold Symn where A = ATand which is a Euclidean (flat) subspace of the space of all matrices R
n×n. Its time-discretization ΦA belongs to the symmetric positive definite
(SPD) manifold Sym+
n which is a non-Euclidean space (curved) and a submanifold of Symn where matrices are symmetric but specifically, positive definite. For a matrix to be positive definite, all its eigenvalues must be positive (i.e. Re(λi) > 0)∀i). The SPD manifold M is a smooth differentiable topological space equipped with an invariant Riemannian structure (i.e. Riemannian manifold). The structure facilitates a Riemannian metric that varies smoothly from point to point where every point is equivalent to a unique and valid physical system. For further reading on Riemannian metrics, see Sommer et al. (2020). For each system matrix ΦA ∈ M, it is possible to compute a tangent

$$\frac{d\text{T}}{dt}=A\text{T}+B\text{U}=\begin{bmatrix}-U_{ext11,ext2}&U_{ext11,ext2}\\ \frac{U_{ext11}}{C_{ext12}}&\frac{U_{ext11}}{C_{ext12}}\\ \frac{U_{ext11,ext2}}{C_{ext2}}&-\frac{U_{ext11,ext2}-U_{ext21,ext}}{C_{ext2}}\end{bmatrix}\begin{bmatrix}T_{ext1}\\ T_{ext2}\end{bmatrix}+\begin{bmatrix}0\\ \frac{U_{ext21,ext}}{C_{ext2}}\end{bmatrix}[T_{ext}]\tag{2}$$

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 2.1.2 Spd-Aware Machine Learning Via Riemannian Optimisation

Having illustrated the geometric connection between the phase space underlying physical descriptions of dynamical systems, it becomes natural to leverage these geometric representations for preserving structure when machine learning phase space dynamics from data. Assuming the availability of measurement data from the system, our mission is to uncover the underlying eigenstructure that governs a measured system's behavior, by perturbing or 'nudging' the physics-approximated state space model within the underlying *SP D* manifold, closer towards a dynamical system that represents the stable dynamics underlying the measured temperature data. In the control/dynamical systems community, this could be interpreted as manifold-constrained system identification (SID). We start with an initial state matrix A that is derived from Physics but misspecified (see Table 3), and which is used as an initial guess at the start of the optimization. Physical systems are measured at discrete time-steps, hence it is necessary to reformulate equation 2 into its discrete-time form via equation 3, as follows:

$$\mathbb{T}_{t+1}=\Phi_{A}\mathbf{T}_{t}+\Phi_{B}\mathbf{U}_{t}.$$
$$(4)$$
Tt+1 = ΦATt + ΦBUt. (4)
The optimization goal is to learn a new LSSM whose matrices, denoted Φˆ A and Φˆ B, better fit the target measurement data. The matrices are parameterized as tensors of size n × n and n × m, respectively. The optimization problem described above may be stated as:

$\hat{\Phi}_{A},\hat{\Phi}_{B}=\operatorname*{arg\,min}_{\Phi_{A},\Phi_{B}}\mathcal{J}(X|\Phi_{A},\Phi_{B}),$  s.t. $\Phi_{A}^{\top}=\Phi_{A}$ and $\mathbf{T}^{\top}\Phi_{A}\mathbf{T}>0\left\{\mathbf{T}|\mathbf{T}\in\mathbb{R}^{2}\right\}$,
2}, (6)
where the loss function, J , is defined as:

$${\mathcal{J}}(X|\Phi_{A},\Phi_{B})=\sum_{i=1}^{n-1}\left\|\Phi_{A}{\bf T}_{i}+\Phi_{B}{\bf T}_{i}-{\bf T}_{i+1}\right\|_{2}^{2}.\tag{1}$$
$$(S)$$
$$(6)$$
$$(7)$$

To preserve stability of ΦA via the symmetric positive structure, we adopt the Riemannian adaptive optimization method (RAdam) Becigneul & Ganea (2019) to estimate the ´ Φˆ A tensor where gradient updates follow the curved geodesic. The Riemannian gradient is given by:

$$\nabla_{\Phi_{A}}{\mathcal{I}}(\Phi_{A}^{(i)}),$$
A ), (8)
space TΦAM. The operator used to map from a point on the manifold to its tangent space is given by the logarithmic map LogΦA
(m) : *M → T*ΦAM while the inverse is given by the exponential map ExpΦA
(m) : TΦA*M → M*. Therefore, we can interpret the tangent space at a given system ΦA in equation 2 on the Sym+
n manifold M as the linearized space of all possible infinitesimal perturbations of ΦA that preserve the symmetric structure (1).

Thus, when interpreted geometrically, the latter is equivalent to 3 implying that time-discretization of A is a projection from continuous-time dynamics residing in the Euclidean space of all possible symmetric dynamical systems Symn to the non-Euclidean space Sym+
n of symmetric but stable discrete-time dynamical systems, by means of their positive eigenvalues implying positive definiteness. In further detail, e Aτ is a bilinear map that geometrically maps the complex s-plane to the complex unit circle in the z-plane where system stability is preserved by wrapping the stable eigenvalues located in the left half-plane (i.e., Re(λi) < 0)) within the unit circle in the s-plane where Re(λi) > 0). System matrices ΦA that lie on the surface of the SPD manifold are positive semidefinite attributed to their low-rank and are said to be *bistable* due to some eigenvalues Re(λi) ≥ 0).

In general, as you move towards the boundary in the stratified space composing the SPD manifold, the matrix loses rank, meaning that fewer independent eigendirections remain for the system trajectories to evolve in. For further reading on the role of symmetry in dynamical systems, see Marsden & Ratiu (2013).

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 and therefore, gradient updates follow the curved geodesic by projecting the gradient onto the tangent space as follows:

$$\Phi_{A}^{(i+1)}=\exp_{\Phi_{A}^{(i)}}\left(-\eta.\frac{\hat{m}_{i}}{\sqrt{\hat{v}_{i}}}\right),$$
$$(9)$$

where where η is a user determined learning rate, mˆi and vˆi are the bias-corrected first and second moment estimates, respectively which summarise the history of gradients of J to inform the adaptive direction of the geodesic update. The RAdam was implemented using the geoopt Python library (Kochurov et al., 2020). On the other hand, gradient updates for learning Φˆ B were computed in flat Euclidean space using Adam (Kingma & Ba, 2017) in torch.optim Python library (Paszke et al., 2019). In an alternative approach, Φˆ A, may also be parameterized by the lower Cholesky decomposition via Φˆ A = LLTto ensure optimization stays within the SPD manifold.

## 2.2 Conservative Use-Case: Fermi-Pasta-Ulam-Tsingou System

The FPUT chain (Fermi et al., 1955) provides a classic benchmark for studying nonlinear dynamics in many-body systems. It models a set of particles connected by springs, where nonlinearity arises from higher-order terms in the spring potential. We consider a fixed-end chain of N masses, with M = N − 1 interior degrees of freedom, leading to a 2M-dimensional canonical phase space z = (q1, . . . , qM, p1*, . . . , p*M). The Hamiltonian of the cubic FPUT–α model is:

$$H(q,p)=\sum_{i=1}^{M}\frac{1}{2}p_{i}^{2}+\sum_{i=0}^{M}\frac{1}{2}(q_{i+1}-q_{i})^{2}+\frac{\alpha}{3}(q_{i+1}-q_{i})^{3},\qquad q_{0}=q_{N}=0,$$
$$(10)$$

where qi and pi denote displacement and momentum of the ith mass, respectively, and α controls the nonlinear stiffness. For α = 0, the system reduces to a linear chain with nearly elliptical phase portraits, while α ̸= 0 produces asymmetric level sets (e.g. the 'teardrop' shapes in Figure 2). Here, q0 = qM+1 = 0.

The corresponding equations of motion follow from Hamilton's equations,

$$\dot{q}_{i}=\frac{\partial H}{\partial p_{i}}=p_{i},\qquad\dot{p}_{i}=-\frac{\partial H}{\partial q_{i}}=q_{i+1}-2q_{i}+q_{i-1}+\alpha[(q_{i+1}-q_{i})^{2}-(q_{i}-q_{i-1})^{2}].\tag{11}$$

where, i = 1*, . . . , M*. In compact form, the Hamiltonian flow can be expressed as:

$$X_{H}(z)=J\nabla H(z),\qquad J=\begin{bmatrix}0&I\\ -I&0\end{bmatrix},$$  where the canonical matrix $J$ defines the symplectic structure. The associated two-form
$$\omega=\sum_{i=1}^{M}dq_{i}\wedge dp_{i},\tag{1}$$
$$(12)^{\frac{1}{2}}$$
$$(13)$$

is exactly preserved under the dynamics. This invariance guarantees conservation of phase-space volume (Liouville's theorem) and energy, H(*q, p*). The Hamiltonian H can be viewed as a time-invariant surface defined over the phase space. While the full 18-dimensional energy surface over the full phase space cannot be visualized, in Figure 2
(bottom row) we illustrate two-dimensional projection slices through H, evolving over time alongside the trajectory of the first coordinate pair (q4, p4) evolving over time (t = 0*, . . . ,* 150). A slice at each time t, is achieved by varying (q4, p4) while holding all other coordinates fixed at their instantaneous values. In the plots, the white contour is the level set at which the invariant energy hypersurface intersects with the (q1, p1)–slice at time t. The energy level is time-invariant and only appears to change due to the sliced view of the 18-dimensional surface.

Since Hamiltonian flows satisfy XH(z) = J∇H, the flow is *tangent* to the (constant) energy level set in ever slice, as can be seen in Figure 2. Importantly, when a trajectory is visibly *jumping* between level sets in these plots, it is indicative of *energy drift* (non–conservation) arising from model discrepancy from the true energy surface.

## 2.2.1 Symplectic-Aware Machine Learning Via Shnns

The symplectic structure governing conservative systems can be leveraged to learn the conservative dynamics of Hamiltonian systems from data. Hamiltonian neural networks (HNNs) (Greydanus et al., 2019) take physical coordinates (q, p) as input and learn a single scalar Hamiltonian Hθ(*q, p*); the dynamics are obtained via the symplectic gradient fθ(z) = J∇Hθ(z) with z = (q, p).

Specifically, we use symplectic Hamiltonian neural networks (SHNN), which extend HNNs using a symplectic time discretization via the implicit midpoint rule, while retaining the Hamiltonian parameterization (David & Mehats, 2023). This setup ensures the learned vector field is Hamiltonian ´ by construction and that the roll-out map is symplectic due to the integrator, promoting energy and structure preservation.

## 3 Experiments And Results 3.1 Dissipative Use-Case

270

![5_image_0.png](5_image_0.png) 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 A sequence of one year's worth of synthetic, hourly measurement temperature data T (8,759 hours, τ = 1) where T ∈ R
8759×1 was generated via a high-fidelity numerical analysis of a homogeneous material system using EnergyPlus 1. The selected physical and thermodynamic properties are found in Table 3, while the ambient dry bulb temperature acting as a forcing U ∈ R
8759×2 was obtained from a historical weather file located in London and Chicago (ladybug tools, 2013), and used as an input for the numerical simulation. While the former was split for testing/training, the latter was used as secondary test set for testing out of distribution initial conditions. The data was obtained from an earlier study in Xuereb Conti et al. (2023). To highlight the benefit of leveraging structure-informed biasing, we repeated the same modeling task across three popular time-series modeling approaches built without structure-awareness, namely: Random forest (RF), extreme gradient boosting (XGBoost), and long short-term memory networks (LSTMs). Additionally, we repeat the system identification of the linear state space model where Φˆ A and Φˆ B tensor elements are learned using only Euclidean gradient updates, denoted as EucOpt, rather than through the proposed Riemannian optimization scheme, which we denote RieOpt.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

| Method                     | Text1    | Text2    |          |          |
|----------------------------|----------|----------|----------|----------|
| LSSM from Physics (ΦA, ΦB) | 2.86e+00 | 1.07e+01 | 6.06e-01 | 2.10e+00 |
| RieOpt (Φˆ A, Φˆ B)        | 4.00e-01 | 1.36e+00 | 5.07e-01 | 1.79e+00 |
| EucOpt (Φˆ A, Φˆ B)        | 1.28e+00 | 3.35e+00 | 5.80e-01 | 1.98e+00 |
| RF                         | 6.81e-01 | 2.41e+01 | 2.32e-01 | 1.63e+01 |
| XGBoost                    | 5.02e-01 | 2.23e+01 | 1.06e-01 | 1.33e+01 |
| LSTM                       | 2.57e+01 | 4.01e+01 | 6.10e+00 | 7.85e+00 |

Table 1: MSE error of models applied to test datasets.

## 3.1.1 Results

While Figure 5 suggests that the structure-naive models (RF, XGBoost and LSTM) seem to rollout the test segments accurately, as evidenced by their mean square error (MSE) loss for the unseen time-steps in Table 1, their training convergence is significantly slower as can be noted on comparing Figure 8 with the structure-preserving EucOpt and RieOpt in Figure 7. To evaluate generalisation for unseen initial conditions, we expose all trained models to an unseen sequence of hourly forcing temperatures Text across one year, located in Chicago (ladybug tools, 2013) where temperatures exhibit different seasonal extremes to London (Figure 6). The underlying thermal dynamics of the material system is invariant of the forcing, implying that if a model has successfully captured the relationship between the forcing and the unforced dynamics, it should generalize for the unseen initial condition. Observing both the MSE loss in Table 1 and the timeseries fit when predicting the indirectly forced state Text1 in Figure 5 we can instantly note how the structurally-naive approaches demonstrate instability whereas, RieOpt and EucOpt demonstrate global stability in capturing the dynamics, in particular the former, as illustrated by the nudged phase portrait in Figure 5 (bottom, left). The structure-aware approach has learned the phase space vector field decoupled dynamics, as opposed to the investigated model free approaches, that learn the forced response of the system as a time series.

## 3.2 Conservative Use-Case

Training data were generated by integrating the Hamiltonian FPU-α system 10 with the symplectic leapfrog (Stormer–Verlet) scheme, which preserves the symplectic form and offers good long- ¨
time energy behavior. We simulate a single long trajectory Z (t = 30, 000 steps, τ = 0.1) with fixed-end boundary conditions, initialized by exciting the first normal mode, qi(0) = siniπ N
and pi(0) = 0 for i = 1*, . . . , N* − 1. The resulting time series (Z ∈ R
30000×18) is split chronologically into 80/20% for training/testing (Ztr ∈ R
24000×18 / Zte ∈ R
6000×18). We benchmark structurepreserving SHNNs against a naive LSTM and a NeuralODE (Chen et al., 2019) baseline. LSTM and NeuralODE inputs (*p, q*) were standardized using the training split mean µ and standard deviation σ and de-standardized for evaluation. The same *µ, σ* were applied to the test split and to any evaluations on unseen initial conditions. SHNN models were trained directly in physical coordinates to preserve the canonical symplectic structure. To ensure fairness, all metrics used to compare models are computed in physical units.

For SHNNs and NeuralODEs we sweep over the number of hidden layers L ∈ {nf , 2nf , 4nf , 8nf } and hidden widths W ∈ {nf , 2nf , 4nf , 8nf }, where nf denotes the dimension of the dynamical state. For the LSTM, we sweep over W only. Each model is trained for 2, 000 epochs with the Adam optimizer (learning rate 3 × 10−3). For all models we evaluate: a) the average one-step update (one-stepMSE) across the test set, b) average autoregressive roll-out (1, 000 steps) prediction of the test set (roll-outMSE), and c) the average drift from the true Hamiltonian (driftRMS). The latter is measured by computing ∆Hk = H(ˆzt+k) − H(ˆzt) where each model is autoregressively rolled out for 1, 000 steps from the initial state (q0, p0) and where zˆt+k is the predicted state. The drift MSE is obtained as the mean loss from the true Hamiltonian (intial state) across the roll-out. The drift MSE provides implicit insight about the stability for longer horizon predictions where, a large drift implies that energy levels on the energy surface are being crossed and thus, the total energy is not conserved.

378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

SHNN NeuralODE LSTM

L W TestMSE DriftRMS Params TestMSE DriftRMS Params TestMSE DriftRMS Params

1 18 6.045e-08 3.697e-03 361 2.079e-07 3.141e+01 684 4.065e-05 5.090e+00 3078 1 36 1.908e-08 1.319e-03 721 7.991e-08 3.775e+02 1350 5.447e-06 5.702e+00 8730 1 72 **8.876e-09 1.322e-03 1441 7.430e-08 1.787e+00 2682** 1.329e-06 5.687e+00 27810 1 144 4.256e-09 5.035e-04 2881 5.472e-08 1.617e+00 5346 **1.694e-06 5.914e+00 97074** 2 18 4.064e-08 4.638e-03 703 2.160e-07 1.919e+00 1026 - - - 2 36 1.209e-08 7.420e-04 2053 9.488e-08 1.802e+03 2682 - - - 2 72 5.284e-09 3.982e-04 6697 7.794e-08 1.420e+00 7938 - - - 2 144 3.901e-09 5.654e-04 23761 5.982e-08 1.194e+00 26226 - - - 4 18 2.606e-08 9.681e-04 1387 2.437e-07 1.673e+00 1710 - - - 4 36 7.120e-09 1.178e-03 4717 1.229e-07 4.533e+01 5346 - - - 4 72 3.574e-09 5.463e-04 17209 1.391e-07 1.396e+00 18450 - - - 4 144 3.091e-09 3.445e-04 65521 1.707e-07 1.484e+01 67986 - - - 8 18 1.338e-08 1.073e-03 2755 2.221e-07 1.206e+00 3078 - - - 8 36 1.302e-08 2.453e-03 10045 1.009e-07 3.970e+00 10674 - - - 8 72 4.621e-09 9.373e-04 38233 1.910e-07 3.709e+00 39474 - - - 8 144 3.799e-09 1.995e-04 149041 2.296e-07 2.028e+00 151506 - - -

## 3.2.1 Results

Figure 3 illustrates the loss for one-step prediction (left panel), free roll-out over 1, 000 steps of the unseen test set (centre panel) and loss for the energy drift (right panel), for varying model sizes across all three models. As expected, increasing the model size improves one-step predictions across all three models but not necessary longer-horizon behaviour as can be seen in the centre panel. Compact symplectic models beat larger, structure-naive baselines on test rollout and drift: most notably, a small SHNN (1,441 params) achieves a significantly better roll-out than the best LSTM (97,074 parameters) which is justified by the lower drift loss, underscoring the benefit of structure Table (2). NeuralODEs vary widely where the best case still drifts significantly more than the SHNN. The impact of enforcing symplectic conservation is especially highlighted in Figure 4a where the overlayed phase trajectory (blue) in the projected phase space (q4, p4) remains close to the predicted energy level that aligns well with the true Hamiltonian. We visualise time-evolving snapshots of the trajectory. Further, when rolling out for perturbed unseen initial conditions in Figures 4b and 4c, the smaller SHNN demonstrates better stability than the best performing yet structure-naive LSTM whose trajectory drifts across the energy levels thus, energy is lost. This visualization helps explain why structurally-naive models such as LSTMs tend to generalize poorly for long roll-outs and outof-distribution conditions.

8

![8_image_0.png](8_image_0.png)

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 4 Conclusion

(a) Predicted (q4, p4) **unseen test data** trajectory (blue line) via SHNN L=1,W=72(1,441 parameters)
trained for 2K epochs. The predicted energy level is represented by the dashed ellipse and can be seen to ![8_image_1.png](8_image_1.png)

(b) Predicted (q4, p4) **unseen initial condition** trajectory (blue line) via SHNN L=1,W=72(1,441 parameters) ![8_image_2.png](8_image_2.png)

Figure 4: Illustrating phase space stability by means of predicted flows on the Hamiltonian.

We showed that structure-aware models can reduce dependence on model size while improving robustness. In two use cases: Riemannian optimization for system identification and symplectic Hamiltonian neural networks for conservative dynamics, varying model size revealed that stable generalization across initial conditions is achievable with models that are much smaller than equally robust, structure-naive baselines. By encoding geometric and physical priors (symmetric positive definite constraints for stable dissipative systems and symplectic structure for conservative systems), we obtain lower long-horizon rollout error and smaller energy drift, even when one-step accuracy alone might suggest simply making models larger. ETHICS STATEMENT

## References

ChatGPT and Google Gemini were used to polish the writing of the paper. All data and code will be made available in a public repository. T. Asselmeyer-Maluga and C.H. Brans. *Exotic Smoothness and Physics: Differential Topology and* Spacetime Models. World Scientific, 2007. ISBN 9789810241957. URL https://books. google.co.uk/books?id=RA8NrZcqejAC.

Fred´ eric Barbaresco. Chapter 4 - symplectic theory of heat and information geometry. In Frank ´
Nielsen, Arni S.R. Srinivasa Rao, and C.R. Rao (eds.), *Geometry and Statistics*, volume 46 of *Handbook of Statistics*, pp. 107–143. Elsevier, 2022. doi: https://doi.org/10.1016/bs.host.

2022.02.003. URL https://www.sciencedirect.com/science/article/pii/ S0169716122000062.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Marco Berardi, Fabio V. Difonzo, and Matteo Icardi. Inverse physics-informed neural networks for transport models in porous materials. Computer Methods in Applied Mechanics and Engineering, 435:117628, 2025. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.

2024.117628. URL https://www.sciencedirect.com/science/article/pii/ S004578252400882X.

Gary Becigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods, 2019. ´
URL https://arxiv.org/abs/1810.00760.

Kathleen Champion, Bethany Lusch, J Nathan Kutz, and Steven L Brunton. Data-driven discovery of coordinates and governing equations. *Proceedings of the National Academy of Sciences*, 116 (45):22445–22451, 2019. URL https://arxiv.org/abs/1904.02107.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations, 2019. URL https://arxiv.org/abs/1806.07366.

Marco David and Florian Mehats. Symplectic learning for hamiltonian neural networks. ´ Journal of Computational Physics, 494:112495, 2023. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp. 2023.112495. URL https://www.sciencedirect.com/science/article/pii/ S0021999123005909.

Enrico Fermi, John Pasta, and Stanisław M. Ulam. Studies of nonlinear problems. I. Technical Report LA-1940, May 1955. URL http://www.osti.gov/accomplishments/ documents/fullText/ACC0041.pdf. Also in Enrico Fermi: Collected Papers, volume 2, edited by Edoardo Amaldi, Herbert L. Anderson, Enrico Persico, Emilio Segre, and Albedo ´ Wattenberg. Chicago: University of Chicago Press, 1965, pages 978–988.

Daniel Floryan and Michael D. Graham. Data-driven discovery of intrinsic dynamics. *Nature* Machine Intelligence, 4(12):1113–1120, Dec 2022. doi: 10.1038/s42256-022-00575-4. URL
https://doi.org/10.1038/s42256-022-00575-4.

Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. Advances in neural information processing systems, 32, 2019.

Quercus Hernandez, Alberto Bad ´ ´ıas, Francisco Chinesta, and El´ıas Cueto. Port-metriplectic neural networks: thermodynamics-informed machine learning of complex physical systems. Computational Mechanics, 72(3):553–561, March 2023. ISSN 1432-0924. doi: 10.1007/ s00466-023-02296-w. URL http://dx.doi.org/10.1007/s00466-023-02296-w.

Ameya D. Jagtap, Zhiping Mao, Nikolaus Adams, and George Em Karniadakis. Physics-informed neural networks for inverse problems in supersonic flows. *Journal of Computational Physics*, 466:111402, October 2022. ISSN 0021-9991. doi: 10.1016/j.jcp.2022.111402. URL http: //dx.doi.org/10.1016/j.jcp.2022.111402.

Pengzhan Jin, Zhen Zhang, Aiqing Zhu, Yifa Tang, and George Em Karniadakis. Sympnets: Intrinsic structure-preserving symplectic networks for identifying hamiltonian systems, 2020. URL https://arxiv.org/abs/2001.03750.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL
https://arxiv.org/abs/1412.6980.

Max Kochurov, Rasul Karimov, and Serge Kozlukov. Geoopt: Riemannian optimization in pytorch, 2020. URL https://arxiv.org/abs/2005.02819.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593

## A Appendix: Dissipative Use-Case Supplementary Details

State Space Matrix model formulation: ladybug tools. Epwmap, 2013. URL https://github.com/ladybug-tools/epwmap?

tab=readme-ov-file.

J.E. Marsden and T.S. Ratiu. Introduction to Mechanics and Symmetry: A Basic Exposition of Classical Mechanical Systems. Texts in Applied Mathematics. Springer New York, 2013. ISBN 9780387217925. URL https://books.google.co.uk/books?id=k-7kBwAAQBAJ.

National Renewable Energy Laboratory NREL. Energyplus™, Sep 2017. URL https://www.

osti.gov/biblio/1395882.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In *Advances in Neural Information Processing Systems 32*, pp. 8024–8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/ 9015-pytorch-an-imperative-style-high-performance-deep-learning-library. pdf.

M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019. ISSN 0021-9991. doi: https://doi.

org/10.1016/j.jcp.2018.10.045. URL https://www.sciencedirect.com/science/ article/pii/S0021999118307125.

Maziar Raissi, Alireza Yazdani, and George Em Karniadakis. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. *Science*, 367(6481):1026–1030, 2020. doi: 10.1126/science.aaw4741. URL https://www.science.org/doi/abs/10. 1126/science.aaw4741.

Stefan Sommer, Tom Fletcher, and Xavier Pennec. Introduction to differential and Riemannian geometry. In *Riemannian Geometric Statistics in Medical Image Analysis*, number Chap. 1, pp. 3–37. Elsevier, 2020. doi: 10.1016/b978-0-12-814725-2.00008-x. URL https://inria. hal.science/hal-02341901.

Rui Wang, Robin Walters, and Rose Yu. Approximately equivariant networks for imperfectly symmetric dynamics, 2022. URL https://arxiv.org/abs/2201.11969.

Zack Xuereb Conti, Ruchi Choudhary, and Luca Magri. A physics-based domain adaptation framework for modeling and forecasting building energy systems. *Data-Centric Engineering*, 4:e10, 2023. doi: 10.1017/dce.2023.8.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization, 2017. URL https://arxiv.org/abs/ 1611.03530.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647

$$\begin{array}{l}{{\frac{d T_{\mathrm{{ext1}}}}{d t}=\frac{1}{C_{\mathrm{{ext1}}}}\left[\frac{1}{R_{\mathrm{{ext2,ext1}}}}\left(T_{\mathrm{{ext1}}}-T_{\mathrm{{ext2}}}\right)\right],}}\\ {{\frac{d T_{\mathrm{{ext2}}}}{d t}=\frac{1}{C_{\mathrm{{ext2}}}}\left[\frac{1}{R_{\mathrm{{ext2,ext1}}}}\left(T_{\mathrm{{ext1}}}-T_{\mathrm{{ext2}}}\right)\right]}}\\ {{+\frac{1}{R_{\mathrm{{ext,ext2}}}}\left(T_{\mathrm{{ext2}}}-T_{\mathrm{{ext}}}\right).}}\end{array}$$
$$(14)$$

$$(15)^{\frac{1}{2}}$$
$\eqref{eq:walpha}$. 
Matrix exponential expansion:

$$e^{A t}=\mathbf{I}+A t+{\frac{A^{2}t^{2}}{2!}}+{\frac{A^{3}t^{3}}{3!}}+\ldots+{\frac{A^{k}t^{k}}{k!}}+\ldots,$$
k!+ *...,* (15)
which leads to the following equation for the discrete time dynamics:
$${\dot{\mathbf{T}}}(t)=e^{A t}\mathbf{T}(t_{0})+A^{-1}(e^{A t}-I)B\mathbf{U}(t_{0}).$$
At − I)BU(t0). (16)
Table 3 displays the values of the various physical and thermophysical parameters in the dissipative use-case.

| Genre                              | Property           | Target (measurements)   | Misspecification (physics)   |
|------------------------------------|--------------------|-------------------------|------------------------------|
| Physical properties                | volume             | 1.8 m3                  | 3.6 m3                       |
| layer thickness                    | 0.2 m              | 0.4 m                   |                              |
| Thermophysical material properties | conductivity       | 0.72 W/mK               | 0.2 W/mK                     |
| density                            | 1920 kg/m3         | 1920 kg/m3              |                              |
| specific heat capacity             | 780 J/kgK          | 780 J/kgK               |                              |
| Convection coefficients            | outdoor convection | 25 W/m2K                | 20 W/m2K                     |
| Theremophysical air properties     | air density        | 1.2 kg/m3               | 1.2 kg/m3                    |
| air specific heat capacity         | 100 J/kgK          | 100 J/kgK               |                              |

Table 3: Physical and thermodynamic properties used to generate target data via numerical analysis in EnergyPlus and to initialize the state space model optimisation, respectively.

## B Appendix: Dissipative Use-Case Supplementary Details

Figure 5 displays the results for model training and testing on the London dataset (top and bottom right). Note that all model-free approaches demonstrate instability in contrast with, the model-based approaches Rie opt and Euc opt which demonstrate global stability in capturing the dynamics, in particular the former, as illustrated by the nudged phase portrait in the bottom left panel. Figure 6 displays the results of the models when applied autoregressively to the unseen Chicago dataset, which demonstrates different seasonal extremes compared to the London dataset. Figure 7 displays the MSE loss per epoch of the model-based approaches Rie opt and Euc opt during training on the London dataset. Note that Rie opt, trained by optimizing on the Riemannian manifold, converged significantly faster than the model optimized in Euclidean space. Figure 8a displays the five-fold cross validated (CV) MSE, for a sweep of forest sizes. The training portion of the London dataset was used to perform this sweep, with a forest size of 250 trees selected for the random forest and 60 for XGBoost. Figure 8b displays the convergence of the investigated LSTM architectures in training. Considerable instability was observed in LSTM training and testing on the London dataset, with best results achieved by learning Text1 and Text2 independently. Both LSTMs used 64 hidden layers, with a window size of 100. There is a strong seasonal variation to both the London and Chicago dataset. The poor performance of the LSTMs was attributed to the relatively small size of the training dataset, which limited the window size and made it difficult for the LSTMs to capture seasonal variations.