000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 In this paper, we establish a fundamental connection between the stability of gradient descent dynamics and the curvature of the underlying loss landscape from a continuous-time perspective. We show that the sign of the real parts of the Hessian's eigenvalues directly governs the convergence behavior of gradient-based optimization. Through analytically tractable, low-dimensional toy examples, we demonstrate that gradient descent can diverge even in simple convex settings. To address this issue, we formulate gradient descent as a second-order dynamical system and introduce a controller that guarantees *locally asymptotic stability* by regulating the system's eigen-structure. Notably, we show that the proposed controller admits a variational interpretation and can be realized as a gradient guidance term augmenting the original gradient. Empirical results on numerical examples with various curvatures and learning rate validate our theoretical findings and demonstrate our proposed method improves both stability and convergence behaviors.

## 1 Introduction

Gradient descent (GD) discretely iterates θt+1 = θt − η∇(θt) to optimize over the training loss f. GD based algorithm, such as stochastic gradient descent (SGD), is one of the fundamental optimization strategy for deep learning models. Existing works on the analysis of stability of GD have strong assumptions regarding the convexity, *sharpness* (the maximum eigenvalue of the Hessian of loss function), and *smoothness*. Traditional framework (Nesterov, 2013) analyze gradient descent under the assumption of training loss is convex and L-smooth (i.e. a function f is L-smooth if, ∀θ, θ
′, ||∇f(θ − ∇f(θ
′)*|| ≤*
L||θ − θ
′||,). Under this framework, Ahn et al. (2022) proves that gradient descent converges only if the learning rate η satisfies η < 2L
, and diverge otherwise. On the other side, Cohen et al.

(2021) empirically demonstrate that GD operates in the regime called the Edge of Stability (EoS),
in which the sharpness hovers just above 2η
, and the training loss behaves non-monotonically over short timescales, yet consistently decreases over long timescales. Meanwhile, Wu et al. (2018) prove GD is stable if ||H(θ)||2 ≤
2 η
, which draws the connection between sharpness, as measured by the spectral norm of Hessian, and the convergence behaviors of GD. Unfortunately, modern deep neural networks normally fail on these assumptions above, and exhibit non-convex or non-smoothness loss. It is still unclear for the convergence and stability behaviors in general loss function of neural network, yet very little is known on the theoretical side of solving the unstable convergence of general non-convex and non-smooth case. In our paper, we consider gradient descent as a dynamical system (Zhu et al., 2018; Wu et al., 2018), and analyze the stability of gradient descent without constrains on curvature. Specifically, we start with the first order training dynamics of gradient descent derived from gradient flow. We transform this first order training dynamics into a system of second order differential equation using functional derivative. We theoretically analyze the stability under various curvature setting. Then we propose a controller term and prove our modified *asymptotically stabilize* gradient descent regardless of the curvature and smoothness on loss function. Based upon the theoretical proofs, we propose our controlled gradient descent in algorithm 1 and conduct empirical experiments on numerical examples with various curvatures and learning rates to prove the effectiveness of our method. The key contributions can be summarized as:
Anonymous authors Paper under double-blind review

## Abstract

# Stabilizing Gradient Descent Via Second- Order Control-Theoretic Dynamics

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107
- **Stability of GD for various curvatures**: We show that the stability of gradient descent is related to the curvature of training loss. We formulate the training dynamics as a secondorder ODE and prove the connections between curvature and sharpness λ, where λ is the largest eigenvalue of Hessian at local minimum. We show that even if the learning rate η is properly bounded by η < 2λ
, gradient descent can still be unstable if the curvature of training loss is not strongly convex, see Table 1.

- **Asymptotically stabilized GD dynamics:** We design a controller term that regulating the eigen-structure of the dynamical system. We apply our controller term into the secondorder ODE of GD training dynamics. We prove our controlled second-order ODE is asymptotically stable regardless of the curvature in Theorem 3.

- **Higher tolerance on learning rate:** Empirically, we observe our controlled gradient descent not only stabilize the training dynamics but also have higher tolerance on the learning rate than GD since our controller term alternate the eigen-structure of training Hessian, therefore increases the 2/sharpness threshold for a stable learning rate as in Figure 1.

- **Our Controlled gradient descent algorithm:** We convert our theoretical controller term on the second-order ODE into an extra term on dθ dt in Equation 5 to modify the gradient update of GD. We formulate our controlled gradient descent algorithm in Algorithm 1.

![1_image_0.png](1_image_0.png)

## 1.1 Related Work

Many works observe the unstable convergence of GD. Jastrzebski et al. (2018); Xing et al. (2018) find that during GD, the training loss decreases non-monotonically, and than oscillating in between "valley walls". Lewkowycz et al. (2020) note that if the GD initialization has sharpness greater than 2 η during neural network training, GD will become initially destabilized, then exhibits "catapults" behavior into a flat region. Even though existing literature has observed the unstable training behaviors of GD and attempted to explain their underlying nature, it remains unclear how to resolve such instabilities. To date, no theoretically characterized algorithm exists that guarantees stabilized convergence of GD in general setting. The Edge of Stability (EoS) is also a crucial view analyzing GD training dynamics. Cohen et al. (2021) formally point out the phenomenon of EoS where the largest eigenvalue of the Hessian hovers around threshold 2η
. Furthermore, Long & Bartlett (2024); Dai et al. (2023) show that the Sharpness-
Aware Minimization (SAM) stabilizes training dynamics and promote flatter minima with dynamically adjusted Hessian norm. Grimmer (2024) use periodically long step to speed up the rate of convergence for smooth and convex training loss. However, for other curvature cases, the stability analysis is insufficient and no existing method stabilize GD for training loss with general curvature. Several works have considered GD training as a dynamical system. Zhu et al. (2018) formulate the training dynamics of stochastic gradient descent (SGD) as a stochastic differential equation (SDE) to analyze the behavior of SGD on escaping from minima and its regularization effects. Wu et al. (2018) shows that learning rate and batch size play different roles in minima selection from the dynamical stability perspective. In this paper, we follow the dynamical system framework to analyze GD training as a second-order ODE, design controlling method accordingly, and propose our controlled gradient descent algorithm.

## 2 Preliminary

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 In this section, we present necessary background knowledge on control theory of a dynamical system and gradient flow of gradient descent algorithm. We define the equilibrium of a dynamical system:
Definition 1 (Equilibrium Point). (Khalil, 2002) Let dx dt = f(x) be a nonlinear dynamical system.

A point x
∗*is called an equilibrium point if* f(x
∗) = 0.

For a dynamical system with a desired equilibrium, we can characterize its stability toward this equilibrium. Specifically, we define the following types of stability:
Definition 2. (Glendinning, 1994) Let x(t) be a solution to a dynamical system dx dt = f(x) with equilibrium point x
∗*. This system is*
- **Locally Lyapunov stable** if for every ε > 0, there exists a δ > 0*, such that when* ∥x(0) −
x
∗∥ < δ, ∥x(t) − x
∗∥ < ε *for all* t ≥ 0.

- **Locally asymptotically stable** if it is Lyapunov stable and in addition, there exists a δ > 0, such that when ∥x(0) − x
∗∥ < δ, limt→∞ x(t) = x
∗.

## - **Unstable** Otherwise.

Remark 1. (Ak Gum¨ *us¸, 2014) For locally (asymptotic) stability, solutions must approach an equilib-* ¨ rium point under initial conditions close to the equilibrium point. In globally (asymptotic) stability, solutions must approach to an equilibrium point under all initial conditions.

1. If all eigenvalues satisfy Re(λi) ≤ 0 and every eigenvalue λi *with Re*(λi) = 0 must have jordan blocks of size 1 × 1, then the system is said to be *locally Lyapunov stable*.

2. If all eigenvalues of J have strictly negative real parts, i.e., Re(λi) < 0 for all i*, then* x
∗is locally asymptotically stable.

with initial condition θ(0) = θ0

## 3 Gradient Flow And Second-Order Dynamics

We begin with the standard gradient flow dynamics (Eq. 1) to model gradient descent in continuous time setting, where θ ∈ R
drepresents the parameters of neural networks and L(θ) ∈ R
d → R is the corresponding loss function. Taking the time derivative of both sides yields the second-order dynamics:

$${\frac{d^{2}\mathbf{\theta}}{d t^{2}}}=-{\frac{d}{d t}}\nabla L(\mathbf{\theta})=-\nabla^{2}L(\mathbf{\theta})\cdot{\frac{d\mathbf{\theta}}{d t}}.$$

When dealing with a complicated non-linear dynamical system, we can use the local linearization method to analyze its local stability around equilibrium with the following theorem:
Theorem 1. *[Local Stability via Linearization] (Perko, 2008) Let* dx dt = f(x) be a continuously differentiable vector field, and let x
∗ *be an equilibrium point. Consider the Jacobian matrix* J =
Df(x
∗)*. Then the local stability of* x
∗*is characterized as follows:*

$$(1)$$

3 3. If at least one eigenvalue satisfies Re(λi) > 0*, then* x
∗is *unstable*.

Definition 3 (Gradient Flow). *(Poliak, 1987) Let* L : R
n → R be a continuously differentiable
function. The gradient flow associated with L *is the solution to the first-order ODE:*
$${\frac{d\theta}{d t}}=-\nabla L(\theta),$$
= −∇L(θ), (1)
Hence, the second-order ODE is:
dt (2)
$${\frac{d^{2}\mathbf{\theta}}{d t^{2}}}=-H(\mathbf{\theta})\cdot{\frac{d\mathbf{\theta}}{d t}}$$
where H(θ) = ∇2L
(θ) is the Hessian matrix of the loss function.

$${\mathrm{ND~LOC}}.$$

## 4 Reformulation And Local Linearization Of Second-Order Ode 4.1 First-Order System Reformulation

To express our second-order ODE as a first-order system, define the auxiliary variable: x =
dθ dt .

Then define the state vector: z =
θ x
∈ R
2n, and define the dynamics:

$$(2)^{\frac{1}{2}}$$
$$\begin{array}{c}\left[\mathbf{z}\right]\\ \frac{d\mathbf{z}}{dt}=f(\mathbf{z})=\left[\frac{d\mathbf{\theta}}{dt}\right]=\left[-H(\mathbf{\theta})\cdot\mathbf{x}\right].\end{array}$$

4.2 LOCAL STABILITY AT EQUILIBRIUM During the training process of gradient descent, our goal is to reach the condition that θ = θ
∗and x = 0, which is when z
∗ =
θ
∗
0
. This goal is an equilibrium point, since z
∗satisfies: f(z
∗) = 0 At equilibrium z
∗, we have: J(z
∗) = 
0 I
0 −H(θ
∗)
. To investigate the stability, consider the characteristic equation of J(z
∗). Let λ ∈ C be an eigenvalue, and we write H = H(θ
∗) for simplicity. Then:

implicitly. Then:  $$\det\left(\begin{bmatrix}0&I\\ 0&-H\end{bmatrix}-\lambda I\right)=\det\left(\begin{bmatrix}-\lambda I&I\\ 0&-H-\lambda I\end{bmatrix}\right)=\det(\lambda^{2}I+\lambda H)=\prod_{i=1}^{n}\lambda(\lambda+\lambda_{i}),$$  where $\lambda>0$ are the eigenvalues of $H$. Therefore, the property of the eigenvalues of the Jacobi
$$({\mathfrak{I}})$$
$f_1\,$ 7. 
where λi > 0 are the eigenvalues of H. Therefore, the property of the eigenvalues of the Jacobian matrix is determined by the eigenvalues of the training loss Hessian.

Theorem 2. The first order dynamic dz dt = f(z) is:
- locally Lyapunov stable if the loss function L *is strongly convex (proof in Section 4.2.1)* - unstable if the loss function L *is convex but not strongly concave (proof in Section 4.2.3)*
4.2.1 STRONGLY CONVEX CASE
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Lemma 1. *(Proof in Appendix A) [Strong Convexity and Positive Definiteness of the Hessian] Let* L : R
n → R be a twice continuously differentiable function. Then, L is strongly convex if and only if there exists a constant m > 0 *such that the Hessian satisfies*
∇2L(θ) ⪰ mI *for all* θ ∈ R
n.

Equivalently, L is strongly convex if and only if ∇2L(θ) *is positive definite for all* θ ∈ R
n.

We now compute the Jacobian J(z) = ∂f
∂z
, which has the block structure J(z) = 
 ∂f1
∂θ
∂f1
∂x
∂f2
∂θ∂f2
∂x
,
where: f1(θ, x) = x, f2(θ, x) = −H(θ)x. Thus, the full Jacobian is:

$$J(\mathbf{z})=\begin{bmatrix}0&I\\ -\sum_{i=1}^{n}\mathbf{x}_{i}{\frac{\partial H(\mathbf{\theta})}{\partial\mathbf{\theta}_{i}}}&-H(\mathbf{\theta})\end{bmatrix}$$

- unstable if the loss function L is convex but not strongly convex (proof in Section 4.2.2)
In this section, we transform our second order ODE into a first order one and analyze the stability of second order ODE derived from the gradient descent algorithm by cases dividing upon the curvature of the loss function. We utilize Theorem 1 and relate the eigenvalues of the Hessian to the curvature of the training loss, then derive the stability analysis based on different settings.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

| Original              | Our Controlled   |                |         |      |
|-----------------------|------------------|----------------|---------|------|
| Gradient Descent      | Gradient Descent |                |         |      |
| Strongly              |                  |                |         |      |
| Curvature             | Convex           |                |         |      |
| Assumption            | Convex           | (not strongly) | Concave | None |
| Stable                | ✓                | ×              | ×       | ✓    |
| Asymptotically Stable | ×                | ×              | ×       | ✓    |

Suppose L is *strongly convex*. Then there exists m > 0 such that H ⪰ mI, implying that H is symmetric positive definite. The characteristic polynomial of J is Qn i=1 λ(λ + λi), where λi > 0 are the eigenvalues of H. This yields n eigenvalues at λ = 0, and n eigenvalues at λ = −λi < 0.

To study the Jordan form of J, consider the eigenspace of λ = 0:

$$J\mathbf{v}=0\quad\Rightarrow\quad{\begin{bmatrix}0&I\\ 0&-H\end{bmatrix}}{\begin{bmatrix}\mathbf{v}_{1}\\ \mathbf{v}_{2}\end{bmatrix}}={\begin{bmatrix}\mathbf{v}_{2}\\ -H\mathbf{v}_{2}\end{bmatrix}}=\mathbf{0}.$$

This implies v2 = 0, and thus v1 ∈ R
n is arbitrary. Therefore, the nullspace has dimension n, and the geometric multiplicity of the zero eigenvalue is n, equal to its algebraic multiplicity. Hence, all Jordan blocks associated with λ = 0 are 1 × 1. Therefore by Theorem 1, the first order dynamic dz dt = f(z) is locally Lyapunov stable if the loss function L is strongly convex.

## 4.2.2 Convex But Not Strongly Convex Case

Lemma 2. *(Proof in Appendix B) [Convexity and Positive Semidefiniteness of the Hessian] Let* L : R
n → R be a twice continuously differentiable function. Then L *is convex if and only if the* Hessian satisfies
∇2L(θ) ⪰ 0 *for all* θ ∈ R
n.

Equivalently, L *is convex if and only if the Hessian is positive semidefinite at all points.*

Now assume L is convex but not strongly convex. Then H ⪰ 0, but H is only positive semidefinite, meaning it has at least one eigenvalue equal to zero. Suppose λ1 = 0 and v1 is the corresponding eigenvector. Then the characteristic polynomial becomes Qn i=1 λ(λ+λi), with at least one repeated root at λ = 0. The algebraic multiplicity of λ = 0 is greater than n. Therefore, the geometric multiplicity is strictly less than the algebraic multiplicity. This implies that the Jordan block associated with λ = 0 has size strictly greater than 1 × 1.

Although all eigenvalues satisfy Re(λ) ≤ 0, the existence of a Jordan block larger than 1 × 1 for an eigenvalue on the imaginary axis (specifically, λ = 0) violates the condition for marginal stability, resulting in solutions that grow linearly over time. Therefore by Theorem 1, the first order dynamic dz dt = f(z) is unstable if the loss function L convex but not strongly convex.

## 4.2.3 Concave Case

Lemma 3. *(Proof in Appendix C) [Concavity and Negative Semidefinite Hessian] Let* L : R
n → R
be twice continuously differentiable. Then L is concave if and only if the Hessian satisfies

$$\nabla^{2}L(\theta)\preceq0\,f o r\,a l l\,\theta\in\mathbb{R}^{n},$$

Equivalently, L *is concave if and only if the Hessian is negative semidefinite at all points.*
At a critical point, the characteristic polynomial is Qn i=1 λλ + λi
. Because λi ≤ 0, the spectrum of J is contained in {0} ∪ [0, ∞):

$\mathrm{spec}(J)=\{\underbrace{0,\ldots,0}_{n\text{times}}\}\ \cup\ \{-\lambda_{i}(H)\}_{i=1}^{n}\subseteq\{0\}\cup(0,\infty)$.  
Therefore by Theorem 1, the dynamical system of GD is unstable if the loss function L is concave.

## 5 Controlling And Stabilizing Second Order Ode

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 6 Controlled Gradient Descent

d
$${\frac{d^{2}\theta}{d t^{2}}}^{\prime}=-(H(\theta)+K_{2})\cdot{\frac{d\theta}{d t}}-K_{1}\theta.$$
We define θ˙ =
dθ dt convert this second order ODE into first order ODE such that

$${\frac{d}{d t}}\begin{bmatrix}\theta\\ \dot{\theta}\end{bmatrix}=\underbrace{\begin{bmatrix}0&I\\ -K_{1}&-(H(\theta)+K_{2})\end{bmatrix}}_{J(\theta)}\begin{bmatrix}\theta\\ \dot{\theta}\end{bmatrix}$$

We denote J(θ) as the Jacobian matrix of the system. By Theorem 1, system 4 is locally asymptotically stable around an equilibrium θ θ˙
=
θ
∗
0 if all eigenvalues of J(θ
∗) have strictly negative real parts. Lemma 4. *(Tisseur & Meerbergen, 2001) Let* Q(λ) = λ 2M+λC+K be a matrix-valued quadratic polynomial. Suppose M ≻ 0 , C ≻ 0, and K ≻ 0, then all eigenvalues λ of Q(λ) *have strictly* negative real parts. The characteristic equation for seeking the eigenvalues λ of the Jacobian equation above is det (λI − J) = 0. this leads to a matrix-valued quadratic eigenvalue problem (QEP):
Theorem 3. The controlled second order system of gradient descent in Equation 4 is locally asymptotically stable Substitute the controller term u in Definition 4 we get Definition 4. *Let the controller term* u = −K1θ − K2 dθ dt , where K1, K2 are Rd×d *matrix,* K1 ≻ 0 and H(θ) + K2 ≻ 0. Remark 2. Empirically, u can be selected by choosing K1 ≻ 0 by letting K1 = µI for some µ > 0 and choosing K2 such that K2 ≻ −H(θ) *for all* θ.

In this section, we build from the training dynamic of gradient descent in Equation 2 and propose a controller term to stabilize gradient descent regardless of the curvature of the training objective. Specifically, we formulate a controller function u and transform our original training dynamic into

$$\frac{d^{2}\mathbf{\theta}^{\,\prime}}{dt^{2}}=\frac{d^{2}\mathbf{\theta}}{dt^{2}}+\mathbf{u}=-H(\mathbf{\theta})\cdot\frac{d\mathbf{\theta}}{dt}+\mathbf{u}\tag{4}$$

Proof. In our system:
$$M=I,\quad C=H+K_{2},$$
Therefore by Lemma 4, all the eigenvalues λ have negative real parts. As a result, by Theorem 1, system 4 is locally asymptotically stable, whereas the original system is Lyapnuov stable only under strongly convex loss. Notice that the locally asymptotically stable guarantee for our controlled second order system is regardless of its curvature. We present a comparison of the theoretical stability guarantees for training dynamics between GD under various curvatures and our controlled gradient descent in general case in Table 1. Our controlled gradient descent not only relaxes the constrain on curvature, but also achieves better stability than GD for all curvature settings. In this section, we extend our theoretical analysis of the controlled dynamical system back to the gradient descent algorithm. Recall that we are imposing the controller term on the second order

$$Q(\lambda)=\lambda^{2}I$$
$\downarrow\quad K_{\rm d}$. 
2I + λ(H + K2) + K1.

derivative d 2θ dt2
′, which measures the acceleration of θ with respect to the training time t. Therefore, we can easily recover the gradient dθ dt
′of θ by taking an integration on the second derivative d 2θ dt2
′.

$${\frac{d\mathbf{\theta}^{\prime}}{d t}}=\int{\frac{d^{2}\mathbf{\theta}^{\prime}}{d t^{2}}}d t=\int{\frac{d^{2}\mathbf{\theta}}{d t^{2}}}d t+\int u d t={\frac{d\mathbf{\theta}}{d t}}-{\frac{1}{2}}K_{1}\mathbf{\theta}^{2}-K_{2}\mathbf{\theta},$$
$$(5)^{\frac{1}{2}}$$

where θ 2is the element-wise square and θ 2:= (θ 21, θ22*...., θ*2d
). Notice that dθ dt
′represents the gradient of θ in continuous setting, in which we can extend to discrete gradient descent by considering it as dθ dt
′|t=t , where t is the current training time and we use θt for evaluation. Specifically, we can apply modification into gradient computing process of gradient descent and propose our controlled gradient descent as: Algorithm 1: Controlled Gradient Descent for Neural Network Training Input: Neural network with parameters initialized as θ0, learning rate η > 0, training data
{(xi, yi)}
N
i=1, loss function L(θ; *x, y*), maximum epochs T
Output: Trained network parameters θT
for t = 0 to T − 1 do for each mini-batch B ⊂ {(xi, yi)} do Compute gradient of loss: gt =1 |B| P(xi,yi)∈B(∇θL(θt; xi, yi) − K1θ 2 t − K2θt);
Update network parameters: θt+1 ← θt − ηgt;

## Return Θt ;

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Intuitively, our controlled gradient descent can be considered as a gradient guidance toward the optimal equilibrium. In our theoretical analysis of GD, the stability of gradient descent is determined by the spectrum of the Hessian H(θ
⋆) at a local minimum θ
⋆. Cohen et al. (2021) shows that the discrete system is stable only if the learning rate satisfies η < 2*/sharpness*. This criterion highlights the sharpness barrier that constrains the allowable learning rate. We explain both stabilization and improved learning-rate tolerance from the eigenvalue shifting mechanism. In our controlled formulation, the update rule effectively replaces the original gradient
∇L(θ) with a modified direction ∇L(θ) − K1θ 2 − K2θ. The eigenvalues of the controlled Jacobian are therefore deriving from the shifted versions of those of H(θ
⋆). By choosing K1 and K2 following Definition 4, we guarantee local asymptotic stability of the continuous-time dynamics.

## 7 Experiments

In this section, we empirically validate the effectiveness of our controlled gradient descent (CGD) algorithm on synthetic numerical cases. Our experimental design serves two complementary purposes. Even in the strongly convex and smooth case, vanilla GD may diverge when the learning rate is chosen outside the narrow stability region bounded by 2*/sharpness*. Figure 1 highlights such instability in a toy quadratic problem, contrasting the divergent trajectory of GD with the stabilized dynamics achieved by CGD. This motivates a detailed study in the following subsections: (i) **Stability across various curvature regimes**, where we validate our CGD stabilize the training process regardless of curvatures; and (ii) Stability under learning rates around the edge of stability, where we show that CGD significantly enlarges the admissible step-size range compared to GD. Taken together, these experiments confirm our theoretical findings: CGD consistently stabilizes optimization across diverse curvature structures, demonstrates robustness to controller hyperparameters, and substantially improves tolerance to larger learning rates.

## 7.1 Stability For Various Curvature

We evaluate the stability of our controlled gradient descent comparing to GD across different curvature settings. Specifically, we consider three representative objective functions:
- Strongly convex ellipse: L(θ) = 2θ 2 1 + 0.5θ 22, initialized at θ0 = (2.0, 1.5) with η = 0.5.

- Strongly convex quartic: L(θ) = θ 41 + θ 42, initialized at θ0 = (1, 1) with η = 0.5.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

- Convex but not strongly convex sphere:L(θ) = θ 21 + θ 22, initialized at θ0 = (1, 1) with η = 0.995.

Figure 2 (a)–(c) show the optimization trajectories projected in three dimensions, while (d)–(f) depict the corresponding training curves. Across all cases, GD exhibits instability: oscillations on the ellipse, divergence on the quartic, and slow convergence or marginal instability on the sphere. In contrast, our contrast gradient descent consistently stabilizes the dynamics, ensuring convergence even when GD fails. This empirical evidence aligns with our theoretical analysis: the stability of GD is sensitive to curvature (both strong convexity and higher-order terms), while our control-theoretic modification guarantees asymptotic stability under all examined cases. Ablation on controller hyperparameters We further investigate the sensitivity of controlled gradient descent to the choices of K1 and K2. In Figure 2 we set K1 = k1*I, K*2 = k2I, where we plot three curves for k1 = k2 = 0.05, 0.1 and 0.2 respectively. We observe that our controlled gradident descent converges reliably regardless of the exact choice of hyperparameters. This indicates that the effectiveness of CGD does not hinge on fine-tuning K1 and K2, highlighting its robustness as a practical optimization method.

## 7.2 Stability For Various Learning Rate Around Eos

We analyze stability when the learning rate is close to the classical upper bound η < 2*/sharpness*. For the convex sphere loss L(θ) = θ 21 + θ 22(sharpness = 2), we vary the learning rate around the theoretical threshold η = 1.

Figure 3 presents the loss curves under η = 0.99, 1 and 1.01, respectively. We observe:
- For η = 0.99, GD converges slowly, while our method achieves faster and smoother convergence.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png) 
- At the critical point η = 1.0, GD fails to converge and oscillates around the sub-optimum, while our method maintains stability.

- For η = 1.01, GD diverges, but our method continues to converge reliably.

These results demonstrate that our controlled gradient descent remains stable beyond the edge of stability, validating its robustness with respect to learning rate selection.

## 8 Conclusion And Discussion

In this paper, we propose a controlled gradient descent method using control theory to stabilize the training dynamics of GD. We formulate GD as a second-order dynamical system and use this perspective to analyze its stability. Through this reformulation, we show that GD can diverge even when the learning rate satisfies the classical bound, highlighting fundamental limitations of existing stability analyses. We further characterize how stability behaviors differ under various curvature conditions, demonstrating that convergence cannot be guaranteed solely by bounding the learning rate. To address these issues, we introduce a controller that regulates the eigen-structure of the training loss Hessian. We prove that this controller guarantees local asymptotic stability under general curvature settings and interpret it as a gradient guidance term augmenting the original update rule. This control-theoretic lens opens a pathway for systematically designing stabilized variants of gradient descent that remain effective in highly non-convex or non-smooth landscapes. Empirical evaluations on synthetic problems confirm our controlled gradient descent improves stability, tolerates larger learning rates, and converges more reliably than standard GD. Limitations and Future Directions: Our analysis focuses on the continuous-time formulation of gradient descent, where the learning rate is assumed to be sufficiently small so that the discrete updates approximate the gradient flow. Within this setting, we show that gradient descent can still diverge under various curvature conditions, revealing instability that persists even in the idealized continuous case. However, a gap remains between continuous-time differential equations and the actual discrete gradient descent updates. This gap represents a limitation of our current analysis, as discretization effects may introduce additional sources of instability or alter the stability thresholds we derive. Future work includes conduction stability analysis directly in discrete setting. Extending the controller design to stochastic optimization, adaptive learning-rate methods, and large-scale nonconvex landscapes also represents an exciting direction for building more robust training algorithms.

## 9 Broader Impacts And Llm Usage

This work is primarily theoretical and focuses on the stability analysis of gradient descent from a control-theoretic perspective. Its contributions lie in advancing the understanding of optimization dynamics and in proposing more stable training methods. Any broader societal consequences would only arise indirectly through downstream applications of deep learning, which fall outside the scope of this study. We have used LLM to polish writing for this paper.

## References

Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. In *International conference on machine learning*, pp. 247–257. PMLR, 2022.

Ozlem Ak G ¨ um¨ us¸. Global and local stability analysis in a nonlinear discrete-time population model. ¨
Advances in Difference Equations, 2014(1):299, 2014.

Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. *arXiv preprint arXiv:2103.00065*, 2021.

Yan Dai, Kwangjun Ahn, and Suvrit Sra. The crucial role of normalization in sharpness-aware minimization. *Advances in Neural Information Processing Systems*, 36:67741–67770, 2023.

Paul Glendinning. Stability, instability and chaos: an introduction to the theory of nonlinear differential equations. Cambridge university press, 1994.

Benjamin Grimmer. Provably faster gradient descent via long steps. *SIAM Journal on Optimization*,
34(3):2588–2608, 2024.

Stanisław Jastrzebski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the relation between the sharpest directions of dnn loss and the sgd step length. *arXiv* preprint arXiv:1807.05031, 2018.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Hassan K Khalil. *Nonlinear systems*. Prentice Hall, Upper Saddle River, N.J., 2002. ISBN
0130673897 9780130673893 0131227408 9780131227408.

Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, and Guy Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. *arXiv preprint arXiv:2003.02218*, 2020.

Philip M Long and Peter L Bartlett. Sharpness-aware minimization and the edge of stability. *Journal* of Machine Learning Research, 25(179):1–20, 2024.

Yurii Nesterov. *Introductory lectures on convex optimization: A basic course*, volume 87. Springer Science & Business Media, 2013.

L. Perko. *Differential Equations and Dynamical Systems*. Texts in Applied Mathematics. Springer New York, 2008. ISBN 9780387951164. URL https://books.google.com.sg/
books?id=A7fvvz9Puf8C.

B.T. Poliak. *Introduction to Optimization*. Translations series in mathematics and engineering.

Optimization Software, Publications Division, 1987. ISBN 9780911575149. URL https:
//books.google.com.sg/books?id=gUXvAAAAMAAJ.

Franc¸oise Tisseur and Karl Meerbergen. The quadratic eigenvalue problem. *SIAM review*, 43(2):
235–286, 2001.

Lei Wu, Chao Ma, et al. How sgd selects the global minima in over-parameterized learning: A
dynamical stability perspective. *Advances in Neural Information Processing Systems*, 31, 2018.

Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A walk with sgd. arXiv preprint arXiv:1802.08770, 2018.

Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. arXiv preprint arXiv:1803.00195, 2018.

## A Appendix A: Proof Of Lemma 1

Proof. We prove both directions.

(⇒**) Strong convexity implies positive definiteness.** Assume L is m-strongly convex. By definition, for all *x, y* ∈ R
n,

$$L(y)\geq L(x)+\nabla L(x)^{\top}(y-x)+$$
2
∥y − x∥
2.
Since L is twice differentiable, we can apply the second-order Taylor expansion around x:
L(y) = L(x) + ∇L(x)
⊤(y − x) + 12
(y − x)
⊤∇2L(ξ)(y − x),
for some ξ on the line segment between x and y. Comparing with the strong convexity inequality, we obtain:1

$${\frac{1}{2}}(y-x)^{\top}\nabla^{2}L(\xi)(y-x)\geq{\frac{m}{2}}\|y-x\|^{2},$$

which implies:
(z)
⊤∇2L(ξ)z ≥ m∥z∥
2, ∀z = y − x ∈ R
n.

Therefore, ∇2L(ξ) ⪰ mI, which means ∇2L(θ) ≻ 0 for all θ ∈ R
n.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 B APPENDIX B: PROOF OF LEMMA 2 Proof. We prove both directions.

$$L(y)=L(x)+\nabla L(x)^{\top}$$
⊤(y − x) + 12
(y − x)
⊤∇2L(ξ)(y − x),
for some ξ on the segment joining x and y. Comparing this with the convexity inequality gives: which implies:
z
⊤∇2L(ξ)z ≥ 0 for all z = y − x ∈ R
n.

Thus, ∇2L(ξ) ⪰ 0, and since ξ is arbitrary, the Hessian is positive semidefinite everywhere. (⇐**) Positive definiteness implies strong convexity.** Assume ∇2L(θ) ⪰ mI for some m > 0, and all θ ∈ R
n. Using Taylor's expansion as above, we again write for all *x, y* ∈ R
n:

 - $L\left(x\right)$ + $\nabla L$

L(y) = L(x) + ∇L(x)
⊤(y − x) + 12
$$y-x)^{\top}\nabla^{2}L(\xi)(y-x),$$
for some ξ on the segment between x and y. Then:

$\mathcal{L}$
$\square$
(y − x)
⊤∇2L(ξ)(y − x) ≥ m∥y − x∥
2,

and so:
$${\frac{m}{2}}\|y-x\|^{2},$$
$$L(y)\geq L(x)+\nabla L(x)^{\top}(y-x)+{\frac{m}{2}}\|y\|$$
Let L : R
n → R be a twice continuously differentiable function. Then, L is strongly convex if and only if there exists a constant m > 0 such that the Hessian satisfies
∇2L(θ) ⪰ mI for all θ ∈ R
n.

Equivalently, L is strongly convex if and only if ∇2L(θ) is positive definite for all θ ∈ R
n.

(⇒**) Convexity implies positive semidefiniteness.** Assume L is convex. By the definition of convexity, for all *x, y* ∈ R
n, L(y) ≥ L(x) + ∇L(x)
⊤(y − x).

Using the second-order Taylor expansion at x, we have:

$=\;x^{\frac{1}{2}}$
$$\frac{1}{2}(y-x)^{\top}\nabla^{2}L(\xi)(y-x)\geq0,$$
2
which is the definition of m-strong convexity. Hence, L is strongly convex.

(⇐**) Positive semidefiniteness implies convexity.** Assume ∇2L(θ) ⪰ 0 for all θ ∈ R
n. Let x, y ∈ R
n, and consider:
ϕ(t) = L(x + t(y − x)), t ∈ [0, 1].

Then:
ϕ
′′(t) = (y − x)
⊤∇2L(x + t(y − x))(y − x) ≥ 0.

So ϕ is a convex function on [0, 1], and:
ϕ(1) ≥ ϕ(0) + ϕ
′(0) = L(x) + ∇L(x)
⊤(y − x).

This gives the first-order convexity condition, and hence L is convex.

C APPENDIX C: PROOF OF LEMMA 3 Let L : R
n → R be twice continuously differentiable. Then L is concave *⇐⇒ ∇*2L(θ) ⪯ 0 for all θ ∈ R
n, equivalently, every eigenvalue of H(θ) = ∇2L(θ) satisfies λi(H(θ)) ≤ 0.

Proof. We prove both directions.

(⇒**) Concavity implies negative semidefiniteness.** Assume L is concave. Fix θ ∈ R
n and h ∈
R

n and define the univariate function φ(t) := L(θ + th), t ∈ R.

Concavity of L implies φ is concave on R, hence φ
′′(t) ≤ 0 for all t. By the chain rule, φ
′′(t) = h
⊤∇2L(θ + th) h.

Evaluating at any t (in particular t = 0) yields h
⊤∇2L(θ) h ≤ 0 for all h ∈ R
n, 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 which is exactly ∇2L(θ) ⪯ 0.

(⇐**) Negative semidefiniteness implies concavity.** Assume ∇2L(θ) ⪯ 0 for all θ. Fix θ, h and define φ(t) := L(θ + th). Then φ
′′(t) = h
⊤∇2L(θ + th) h ≤ 0 for all t.

Thus φ is concave on R. Since L is concave along every line in R
n, it is concave on R
n.

Eigenvalue Corollary. From ∇2L(θ) ⪯ 0 it follows that all eigenvalues of H(θ) are nonpositive: λi(H(θ)) ≤ 0 for i = 1*, . . . , n*.