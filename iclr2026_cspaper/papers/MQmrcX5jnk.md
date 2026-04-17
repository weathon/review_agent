# Learning Boltzmann Generators Via Con- Strained Mass Transport

Christopher von Klitzing1 Denis Blessing1∗ **Henrik Schopmans**2∗
Pascal Friederich2 **Gerhard Neumann**1 1 Autonomous Learning Robots, Karlsruhe Institute of Technology 2 Artificial Intelligence for Materials Sciences, Karlsruhe Institute of Technology

## Abstract

Efficient sampling from high-dimensional and multimodal unnormalized probability distributions is a central challenge in many areas of science and machine learning. We focus on Boltzmann generators (BGs) that aim to sample the Boltzmann distribution of physical systems, such as molecules, at a given temperature. Classical variational approaches that minimize the reverse Kullback–Leibler divergence are prone to mode collapse, while annealing-based methods, commonly using geometric schedules, can suffer from mass teleportation and rely heavily on schedule tuning. We introduce *Constrained Mass Transport* (CMT), a variational framework that generates intermediate distributions under constraints on both the KL divergence and the entropy decay between successive steps. These constraints enhance distributional overlap, mitigate mass teleportation, and counteract premature convergence. Across standard BG benchmarks and the here introduced ELIL tetrapeptide, the largest system studied to date without access to samples from molecular dynamics, CMT consistently surpasses state-of-the-art variational methods, achieving more than 2.5× higher effective sample size while avoiding mode collapse.

## 1 Introduction

We consider the problem of sampling from a target probability measure p ∈ P(R
d) given by p(x) = p˜(x)/Z where p˜ ∈ C(R
d, R≥0) can be evaluated pointwise but the normalization constant Z =RRd p˜(x) dx is intractable. Sampling from unnormalized densities arises in many areas, including Bayesian statistics (Gelman et al., 1995), reinforcement learning (Celik et al., 2025), and the natural sciences (Stoltz et al., 2010). A promising alternative to classical Monte Carlo methods (Hammersley, 2013) is offered by variational approaches (Struwe & Struwe, 2000), which aim to minimize a statistical divergence between a variational probability measure q ∈ P(R
d) and the target p, commonly the reverse Kullback–Leibler
(KL) divergence q∗ = arg min q∈P(Rd)
DKL(q ∥ p), (1)
whose unique minimizer is q∗ = p. A prominent example is the variational learning of molecular Boltzmann generators (BGs) (Noe et al., 2019) ´ , for which p˜(x) = exp(−E(x)/kBT), with E being an energy function, T the temperature, and kB the Boltzmann constant. BGs enable efficient sampling of thermodynamic ensembles, thereby bypassing costly molecular dynamics (MD) simulations and accelerating the exploration of rare but physically important states. However, learning BGs is challenging as the state space is typically high-dimensional, the target distribution is often highly multimodal, and evaluating E(x) can be very costly, especially when using accurate energies such as those from density-functional theory (Argaman & Makov, 2000). Furthermore, directly minimizing the reverse KL divergence tends to suffer from mode collapse, ignoring low-probability modes of the target (Blessing et al., 2024; Soletskyi et al., 2025). To address this, a number of recent approaches have proposed to construct a sequence of intermediate
*Equal contribution. Correspondence to denis.blessing@kit.edu and henrik.schopmans@kit.edu.

1 distributions that transport probability mass from a tractable base distribution q0 to the target (Arbel et al., 2021; Matthews et al., 2022; Vargas et al., 2023; Tian et al., 2024; Albergo & Vanden-Eijnden, 2024). This idea, which dates back more than two decades to annealed importance sampling (Neal, 2001), is most often realized through a geometric annealing path, which is defined as a sequence of
(qi)
I
i=1 which follows qi ∝ q 1−βi 0p˜
βi, where the corresponding annealing schedule (βi)
I
i=1 ensures that qI = p. Despite its simplicity, geometric annealing can suffer from mass teleportation, where large portions of the probability mass shift to disjoint regions between successive steps, complicating mass transport (Akhound-Sadegh et al., 2025; Maurais et al., 2025; Chehab & Korba, 2024; Bereux ´ et al., 2024). Moreover, its performance critically depends on the choice of annealing schedule (Syed et al., 2024). By "mass teleportation", we refer to the failure mode of geometric annealing in which probability mass abruptly shifts to regions where the current intermediate has negligible density, blocking effective transport. This differs from the definition of Mat´ e & Fleuret (2023) ´ , who focus on preserving relative weights (often termed "mode switching" (Phillips et al., 2024)).

To counteract this, we build on ideas from reinforcement learning (Schulman et al., 2015) and use a trust-region constraint that modifies (1) by bounding the KL divergence between successive distributions, which results in the geometric annealing path with automatic schedule tuning. Adapting these ideas to sampling problems, we further introduce a constraint that explicitly controls the rate at which the entropy of the variational distribution decreases along the transport path. This added degree of freedom enables deviations from the standard geometric annealing path, mitigating issues such as mass teleportation and premature convergence, while fostering greater overlap between consecutive distributions. These contributions culminate in *Constrained Mass Transport* (CMT), a general framework for transporting variational densities along well-defined annealing paths. To highlight its practical utility, we instantiate the framework with normalizing flows and demonstrate that it consistently outperforms state-of-the-art approaches, often by a substantial margin when learning molecular Boltzmann generators directly from energy evaluations, without reliance on additional MD samples. Contributions. Our contributions can be summarized as follows: - We introduce a general framework for addressing sampling problems through a sequence of constrained variational problems, considering trust-region, entropy, and hybrid constraints.

- We establish a connection between these sequences and annealing paths, which interpolate between a tractable prior and the target distribution.

- We instantiate our framework in practice by employing normalizing flows (Papamakarios et al.,
2021) to learn molecular Boltzmann generators (Noe et al., 2019) ´ .

- We show that our method, *Constrained Mass Transport* (CMT), consistently surpasses state-of-theart approaches, often by a significant margin when learning molecular Boltzmann generators solely from energy evaluations, without relying on additional MD samples.

- We introduce a new benchmark, the *ELIL tetrapeptide*, which, to the best of our knowledge, is the largest system studied to date under the setting of learning exclusively from energy evaluations.

## 2 Constrained Mass Transport

Here, we denote by P(R
d) the space of probability measures on R
dthat are absolutely continuous with respect to Lebesgue measure and admit smooth densities. We approach the sampling problem by dividing (1) into a sequence of constrained optimization problems that result in an annealing path of intermediate densities (qi)
Ii=0 that bridge between a tractable prior q0 and the target p.

Trust-region constraint. Trust regions aim at dividing the problem (1) into subproblems by constraining the updated density to be close to the old density in terms of KL divergence. Formally, this is given by the iterative optimization scheme 1 qi+1 = arg min q∈P(Rd)
DKL(q∥p) s.t. DKL(q∥qi) ≤ εtr, (2)

![2_image_0.png](2_image_0.png)

$$({\mathfrak{I}})$$
$$(4)$$

Figure 1: Illustration of the annealing paths (AP) obtained by solving the variational problems (2), (7), or (9). Trust-region–based optimization (2) mitigates the irregularities of naive schedules (e.g., the linear schedule), but the resulting geometric AP suffers from mass teleportation as the right mode of the target distribution p emerges without overlap with earlier intermediate densities. Constraining the entropy decay between successive densities (7) prevents mass teleportation, yet fails to guarantee sufficient overlap between the initial distribution q0 and subsequent intermediate densities. In contrast, combining both constraints (9) yields APs that both maintain overlap between successive densities and avoid mass teleportation.

for i ∈ N, trust-region bound εtr > 0 and some q0 ∈ P(R
d). Due to the convexity of the KL
divergence, we can show that in all but the last step we actually have an equality constraint in (2);
see Appendix A. Thus, there exists an I ∈ N such that qI = q∗(= p). Under suitable regularity
assumptions, we can approach the above constrained optimization problem using a relaxed Lagrangian
formalism, i.e.,  $$\mathcal{L}_{\mathrm{tr}}^{(i+1)}(q,\lambda)=D_{\mathrm{KL}}(q\|p)+\lambda\left(D_{\mathrm{KL}}(q\|q_{i})-\varepsilon_{\mathrm{tr}}\right)$$  where $\lambda\geq0$ is a Lagrangian multiplier, and solve the saddle point problems
tr (*q, λ*) = DKL(q∥p) + λ (DKL(q∥qi) − εtr) (3)
tr (q, λ). (4)
$$\operatorname*{max}_{\lambda\geq0}\;\operatorname*{min}_{q\in{\mathcal{P}}(\mathbb{R}^{d})}{\mathcal{L}}_{\mathrm{tr}}^{(i)}(q,\lambda).$$
We note that L
 to that $\mathcal{L}_{\mathrm{tr}}^{(i)}$ is convex in . 
tr is convex in q by convexity of the KL divergence and the dual function
of the KL $\mathrm{d}\vec{\mathrm{i}}$. 
$$g_{\mathrm{tr}}^{(i+1)}(\lambda):=\operatorname*{min}_{q\in{\mathcal{P}}(\mathbb{R}^{d})}{\mathcal{L}}_{\mathrm{tr}}^{(i)}(q,\lambda)$$

$$\quad(5)$$
concave in λ since it is the pointwise minimum of a family of linear functions of λ. Thus, (4) has unique optima which we denote by qi+1 and λi, respectively. Indeed, (2) admits an analytical solution which is characterized by Proposition 2.1. We refer to Appendix A for a proof and further details on problem (2). Proposition 2.1 (Optimal intermediate trust-region densities). The intermediate optimal densities that solve (2) *satisfy*

$$q_{i+1}(x,\lambda)=\frac{q_{i}(x)^{\frac{1}{1+\lambda}}\tilde{p}(x)^{\frac{1}{1+\lambda}}}{\mathcal{Z}_{i+1}(\lambda)},\quad\text{with}\quad\mathcal{Z}_{i+1}(\lambda)=\int q_{i}(x)^{\frac{\lambda}{1+\lambda}}\tilde{p}(x)^{\frac{1}{1+\lambda}}\mathrm{d}x,$$  _where $q_{i+1}$ are the unique optimal of the Lagrangian corresponding to (2)._
1+λ dx, (5)
The optimal multiplier λithat solves (2) is obtained by plugging qi+1(λ) in the Lagrangian (3) to
obtain the dual function gtr ∈ C(R, R) given by
$$g_{\mathrm{tr}}^{(i+1)}(\lambda):={\mathcal{L}}_{\mathrm{tr}}^{(i+1)}(q_{i+1}(\lambda),\lambda)=-(1+\lambda)\log{\mathcal{Z}}_{i+1}(\lambda)-\lambda\varepsilon_{\mathrm{tr}}.$$
Assuming access to Zi+1(λ) one can solve λi = arg maxλ≥0g
(i+1)
tr (λ) to obtain the optimal
q ∈ P(R
d) that solves (2) as qi+1 := qi+1(λi).

Entropy constraint. In a similar fashion to (2), we can avoid premature convergence by regulating the entropy decay of the model by constructing a sequence of intermediate densities whose change in entropy is constrained. Formally, we aim to solve the following problem

$q_{i+1}=\operatorname*{arg\,min}_{q\in\mathcal{P}(\mathbb{R}^{d})}\;D_{\mathrm{KL}}(q\|p)\quad\text{s.t.}\quad H(q_{i})-H(q)\leq\varepsilon_{\mathrm{ent}},$
where H(q) = −Rq(x) log q(x)dx is the Shannon entropy and εent > 0 the entropy bound. We can again approach (7) using a Lagrangian formalism by introducing a Lagrangian multiplier η ≥ 0. The analytical solution to (7) is characterized by Proposition 2.2 whose proof can be found in Appendix A.

$$(6)$$
$$\left(T\right)$$

Proposition 2.2 (Optimal intermediate densities for entropy constraint). *The intermediate optimal* densities that solve (7) *satisfy*

qi+1(x, η) =  p˜(x) 1 1+η Zi+1(η) , with Zi+1(η) =  Zp˜(x) 1 1+η dx, (8)
$$({\mathfrak{s}})$$
where qi+1 are the unique optima of the Lagrangian corresponding to (7).

Despite the potential of (7) for counteracting premature convergence, we identify two challenges depending on the entropy of the initial density H(q0): First, if H(q0) < H(p) then the constraint is inactive resulting in η0 = 0, reducing (7) to the optimization problem as stated in (1). Second, if H(q0) ≫ H(p) then the KL divergence between q0 and q1 ∝ p 1/1+η0 can be arbitrarily large and therefore could cause instabilities due to a lack of overlap between the successive densities. While the former challenge can typically be addressed by initializing q0 with large entropy, the second can be more intricate. In the following, we explain how this challenge can be addressed by combining the trust-region and entropy constraint. Combining both constraints. One can straightforwardly combine the constraints in (2) and (7) into a single iterative optimization scheme defined as

$$q_{i+1}=\operatorname*{arg\,min}_{q\in\mathcal{P}(\mathbb{R}^{d})}\;D_{\mathrm{KL}}(q\|p)\quad\text{s.t.}\quad\begin{cases}D_{\mathrm{KL}}(q\|q_{i})\leq\varepsilon_{\mathrm{tr}},\\ H(q_{i})-H(q)\leq\varepsilon_{\mathrm{ent}}.\end{cases}$$
$$(9)$$

In analogy to the previous section, we introduce Lagrangian multipliers λ and η for the trust-region and entropy constraint, respectively. Indeed, one can again obtain an analytical expression for the evolution of the optimal densities, see Proposition 2.3 and Appendix A for a proof. Proposition 2.3 (Optimal intermediate densities for entropy and trust-region constraint). The intermediate optimal densities that solve (9) *satisfy*

$$q_{i+1}(x,\lambda,\eta)=\frac{q_{i}(x)^{\frac{\lambda}{1+\lambda+\eta}}\tilde{p}(x)^{\frac{1}{1+\lambda+\eta}}}{\mathcal{Z}_{i+1}(\lambda,\eta)}\quad\text{with}\quad\mathcal{Z}_{i+1}(\lambda,\eta)=\int q_{i}(x)^{\frac{\lambda}{1+\lambda+\eta}}\tilde{p}(x)^{\frac{1}{1+\lambda+\eta}}(x)\mathrm{d}x,\tag{19}$$
where qi+1 are the unique optima of the Lagrangian corresponding to (9).
$$(10)$$

Clearly, if H(q0) ≫ H(p), the trust-region constraint ensures that the KL divergence between q0 and
q1 is at most εtr and, therefore, for a suitable choice of εtr ensures that two consecutive densities
have sufficient overlap. Lastly, the Lagrangian dual function gtr-ent ∈ C(R
2, R) corresponding to (9),
that is,
$$g_{\mathrm{H-ent}}^{(i+1)}(\lambda,\eta):=-(1+\lambda+\eta)\log{\mathcal Z}_{i+1}(\lambda,\eta)-\lambda\varepsilon_{\mathrm{tr}}-\eta(H(q_{i})-\varepsilon_{\mathrm{ent}}),$$
requires solving a two-dimensional convex optimization problem to obtain λi, ηi which can be done efficiently in practice; see Section 3 for additional details. Connection to annealing paths. Iteratively solving (2), (7) or (9) induces an *annealing path*, that is, a sequence of densities (qi)i∈N that interpolates between q0 and p. We characterize these paths in Theorem 2.4; See Appendix A for a proof.

Theorem 2.4 (Annealing paths). Let p ∈ P(R
d) be the target density and q0 ∈ P(R
d) *some initial* density. The intermediate optimal densities that solve (2), (7) and (9) *satisfy*

$$q_{i}\propto q_{0}^{1-\beta_{i}}\bar{p}^{\beta_{i}},\quad q_{i}\propto\bar{p}^{\alpha_{i}}\;(i\geq1),\quad a n d\quad q_{i}\propto q_{0}^{1-\beta_{i}}\left(\bar{p}^{\alpha_{i}}\right)^{\beta_{i}},$$
βi, (12)
respectively, with β and α being functions of the corresponding Lagrangian multipliers. Moreover, the sequences (αi)i∈N0and (βi)i∈N0take values in [0, 1]*, satisfy* α0 = β0 = 0 and αI = βI = 1 for some I ∈ N+ and (βi)i∈N0*is monotonically increasing.*
In what follows, we refer to the annealing paths in (12) as geometric (G), tempered (T), and geometrictempered (GT) annealing paths, respectively. We further refer to Figure 1 for an illustration of the impact of the introduced constraints on the annealing paths.

$$(11)$$

$\mathbf{a}\cdot\mathbf{b}=-\mathbf{b}$. 
$$(12)$$

## 3 Learning The Intermediate Densities

A general recipe. So far, we discussed how one can construct a sequence of intermediate measures
(qi)i∈N using our constrained mass transport formulation. However, despite having access to the
analytical form of qi, it is typically not possible to sample from it directly. As such, we approximate
each qi by a distribution from a tractable class *Q ⊂ P*(R
d) that permits efficient sampling and density
evaluation. Given an approximation family Q, we select qˆi ∈ Q to approximate qi by solving
$$\hat{q}_{i}=\operatorname*{arg\,min}_{q\in\mathbb{Q}}\;D(q_{i},q),$$
D(qi, q), (13)
where D is an arbitrary statistical divergence between probability measures. This formulation is general: the choice of Q and D determines the trade-off between expressivity, computational cost, and statistical properties such as mode coverage or robustness.

Practical algorithm. In this work, we choose Q to be a normalizing flow family constructed via push-forwards of a simple base measure (Rezende & Mohamed, 2015; Dinh et al., 2016; Durkan et al., 2019; Kingma & Dhariwal, 2018; Gabrie et al., 2022; Kolesnikov et al., 2024; Zhai et al., 2024) ´ .

Let qz ∈ P(R
d) be an easy-to-sample base measure (e.g., a standard Gaussian), and let F be a class of smooth invertible maps f : R
d → R
d. We define

$$(13)$$
$$Q_{\rm NF}:=\{\,f_{\#}q_{z}\mid f\in{\cal F}\,\},\quad\mbox{with}\quad(f_{\#}q_{z})(z)=q_{z}\big{(}f^{-1}(z)\big{)}\,\Bigg{|}\det\frac{\partial f^{-1}(z)}{\partial z}\Bigg{|}.\tag{14}$$  a-forward $f_{\#}q_{z}$. To fit $\hat{q}_{i}$ within this family, we take $D$ to be the importance-weighted
with push-forward f\#qz. To fit qˆi within this family, we take D to be the importance-weighted forward KL divergence

$$\hat{q}_{i+1}=\operatorname*{arg\,min}_{q\in\operatorname{Qisp}}\;D_{\operatorname{KL}}(q_{i+1}\|q)\quad\text{with}\quad D_{\operatorname{KL}}(q_{i+1}\|q)\;=\;\mathbb{E}_{x\sim q_{i}}\left[\frac{q_{i+1}(x)}{q_{i}(x)}\log\left(\frac{q_{i+1}(x)}{q(x)}\right)\right]\tag{15}$$  This choice offers several advantages. First, forward KL strongly penalizes underestimating the 
This choice offers several advantages. First, forward KL strongly penalizes underestimating the support of qi+1, encouraging mode coverage and reducing the risk of mode collapse. Second, because qi+1 is available in closed form from the constrained transport updates (see Proposition 2.1, 2.2 and 2.3), the importance weights qi+1(x)/qi(x) can be computed solely from qi and p˜. Third, the importanceweighted formulation allows us to reuse samples drawn from qi, enabling a seamless integration of replay buffers, resulting in increased sample efficiency. Lastly, the trust-region constraint controls the variance of the importance weights, keeping it approximately constant, independent of the problem dimension d (see Appendix C.3), resulting in a highly scalable algorithm. Details regarding this specific form of CMT are provided in Appendix C.2.

Lagrangian dual optimization. Maximizing the concave dual function (11) requires evaluating intermediate normalization constants Zi+1. This can be done efficiently by expressing Zi+1 as an expectation under qi and using Monte Carlo estimation. For instance, the expression for Zi+1 in (10)
can be estimated as

$$\mathcal{Z}_{i+1}(\lambda,\eta)=\mathbb{E}_{x\sim q_{i}}\left[\left(\frac{\tilde{p}(x)}{q_{i}(x)^{1+\eta}}\right)^{\frac{1}{1+\lambda+\eta}}\right]\approx\frac{1}{N}\sum_{x_{n}\sim q_{i}}\left(\frac{\tilde{p}(x_{i})}{q_{i}(x_{i})^{1+\eta}}\right)^{\frac{1}{1+\lambda+\eta}}.$$
$$(16)$$

We note that, in contrast to estimating the normalization constant of the target Z, estimation of the normalization constant of the intermediate distributions Zi+1 can be performed with low variance since the trust-region constraint ensures sufficient overlap with the next intermediate distribution.

Furthermore, samples xn ∼ qi and the corresponding evaluations qi(xn) and p˜(xn) are typically already computed when solving (13), so the additional cost of determining the Lagrangian multipliers is negligible (see Appendix D.4). For example, on alanine dipeptide, it accounts for only about 0.01% of the total training time. Details of the dual optimization procedure are provided in Appendix C.1, including a code example. Lastly, we refer to Algorithm 1 for an algorithmic overview of the trained measure transport method.

## 4 Related Work

Boltzmann generators. Learning molecular Boltzmann generators (Noe et al., 2019) ´ purely from energy evaluations has been explored with both flow-based methods (Stimper et al., 2022b; Midgley Algorithm 1 Constrained mass transport Require: Initial measure q0, target measure p˜, divergence D, approximation family Q, buffer size N
for i ← 0*, . . . , I* − 1 do Draw N samples xn ∼ qi, evaluate qi(xn), p˜(xn)
Initialize buffer B
(i) = (xn, qi(xn), p˜(xn))N
n=1 Compute multipliers λi, ηi = arg maxλ,η∈R+ g
(i+1)
tr-ent (*λ, η*) using B
(i)
Compute qi+1 ≈ qˆi+1 = arg minq∈Q D(qi+1, q) using B
(i)
return qˆI ≈ p et al., 2022; Schopmans & Friederich, 2025) and diffusion-based methods (Liu et al., 2025; Choi et al., 2025; Kim et al., 2025). While flow-based approaches have demonstrated strong performance, their diffusion-based counterparts remain less competitive on molecular systems, often struggling with mode collapse, even on relatively small systems. Next to purely energy-based approaches, recent work showed success in leveraging samples from a higher temperature, which are typically easier to obtain, and transferring the distribution to a lower target temperature (Dibak et al., 2022a; Wahl et al., 2025; Schopmans & Friederich, 2025; Rissanen et al., 2025; Akhound-Sadegh et al., 2025). Alternative methods train on samples from molecular dynamics (Klein et al., 2023; Midgley et al., 2023; Tan et al., 2025a; Peng & Gao, 2025), allowing for amortized sampling due to transferability to unseen systems (Jing et al., 2022; Abdin & Kim, 2023; Klein & Noe, 2024; Jing et al., 2024; Lewis ´ et al., 2025; Tan et al., 2025b). Constrained optimization. Trust-region methods have a long history as robust optimization algorithms that iteratively minimize an objective within an adaptively sized "trust region"; see Conn et al. (2000) for an overview. Beyond classical optimization, these methods have been extended to operate over spaces of probability distributions, with applications in reinforcement learning (Peters et al., 2010; Schulman et al., 2015; 2017; Achiam et al., 2017; Pajarinen et al., 2019; Akrour et al., 2019; Yang et al., 2020; Otto et al., 2021; Xu et al., 2024; Wu et al., 2017; Abdolmaleki et al., 2018b;a; Meng et al., 2021), black-box optimization (Sun et al., 2009; Wierstra et al., 2014; Abdolmaleki et al., 2015), variational inference (Arenz et al., 2020; 2022), and path integral control (Gomez ´ et al., 2014; Thalmeier et al., 2020). The first explicit link between trust-region optimization and geometric annealing paths was established by Blessing et al. (2025) for path space measures in the setting of stochastic optimal control. Entropy constraints, often introduced as entropy regularization, have also been studied in policy optimization and reinforcement learning, either in the form of soft constraints (Ahmed et al., 2019; Mnih et al., 2016; O'Donoghue et al., 2016) or hard constraints (Abdolmaleki et al., 2015; Pajarinen et al., 2019; Akrour et al., 2016; 2018; 2019). However, prior work typically constrains the absolute entropy value, which is problematic for inference tasks, since it requires prior knowledge of the target density's entropy. To the best of our knowledge, such methods have not yet been extended to sampling problems. Furthermore, the connection between entropy-constrained optimization and annealing paths has not previously been established. Improved annealing paths. Research on improving annealing paths (APs) has largely focused on geometric APs in the context of annealed importance sampling (AIS) (Neal, 2001) and their extensions to sequential Monte Carlo (SMC) (Del Moral et al., 2006); see Jasra et al. (2011); Goshtasbpour et al. (2023); Chopin et al. (2023); Syed et al. (2024). Beyond the standard geometric AP, alternative constructions have been proposed, such as the moment-averaging path for exponential family distributions (Grosse et al., 2013) and the arithmetic mean path (Chen et al., 2021). The geometric path itself can be interpreted as a quasi-arithmetic mean (Kolmogorov & Castelnuovo, 1930) under the natural logarithm, which motivated Brekelmans et al. (2020) to propose APs based on the deformed logarithm transformation. A variational characterization of these paths was later analysed by Brekelmans & Nielsen (2024). Related work also explores improved schedules for parallel tempering Surjanovic et al. (2022); Syed et al. (2021) and for the diffusion coefficient in ergodic Ornstein–Uhlenbeck processes used to train denoising diffusion models (Ho et al., 2020; Song et al., 2020); see, e.g., Nichol & Dhariwal (2021); Williams et al. (2024); Benita et al. (2025); Zhang (2025).

## 5 Numerical Evaluation

Table 1: Results for all systems of varying dimensionality d. Evaluation criteria include the number of target evaluations (TARGET EVALS), the evidence upper bound (EUBO), the reverse effective sample size (ESS) and the average total variation distance to the Ramachandran plots generated from molecular dynamics samples (RAM TV). Details on all metrics can be found in Appendix D.3. Each value is shown as the mean ± standard error over four independent runs. An exception is TA-BG on the ELIL tetrapeptide, for which only two runs were successful due to numerical instabilities. The best results are highlighted in bold, except for the forward KL and reverse KL. Reverse KL is prone to mode collapse, which makes ESS values not directly comparable, and forward KL is trained from samples rather than from energy.

| SYSTEM METHOD              | TARGET EVALS ↓   | EUBO ↓                                               | ESS [%] ↑                             | RAM TV ↓             |                      |
|----------------------------|------------------|------------------------------------------------------|---------------------------------------|----------------------|----------------------|
| FORWARD KL                 | 5 × 109          | −174.92 ± 0.00                                       | (82.14 ± 0.08) %                      | (1.09 ± 0.01) × 10−2 |                      |
| REVERSE KL                 | 2.56 × 108       | −174.96 ± 0.00                                       | (94.13 ± 0.21) %                      | (1.36 ± 0.05) × 10−2 |                      |
| FAB                        | 2.13 × 108       | −174.98 ± 0.00                                       | (94.80 ± 0.04) %                      | (1.03 ± 0.01) × 10−2 |                      |
| TA-BG                      | 1 × 108          | −174.99 ± 0.00                                       | (95.76 ± 0.13) %                      | (1.24 ± 0.07) × 10−2 |                      |
| CMT (OURS)                 | 1 × 108          | −175.00 ± 0.00 (97.69 ± 0.03) % (9.43 ± 0.08) × 10−3 |                                       |                      |                      |
| FORWARD KL                 | 4.2 × 109        | −333.79 ± 0.00                                       | (45.29 ± 0.11) %                      | (1.47 ± 0.03) × 10−2 |                      |
| ALANINE REVERSE KL         | 2.56 × 108       | −332.96 ± 0.13                                       | (75.06 ± 3.50) %                      | (2.89 ± 0.02) × 10−2 |                      |
| TETRAPEPTIDE                            | FAB              | 2.13 × 108                                           | −333.93 ± 0.00                        | (63.59 ± 0.23) %     | (3.10 ± 0.04) × 10−2 |
| TA-BG                      | 1 × 108          | −333.99 ± 0.00                                       | (65.81 ± 0.24) %                      | (1.53 ± 0.09) × 10−2 |                      |
| (d = 120) CMT (OURS)       | 1 × 108          | −334.00 ± 0.00 (68.60 ± 0.21) % (1.43 ± 0.03) × 10−2 |                                       |                      |                      |
| FORWARD KL                 | 4.2 × 109        | −533.16 ± 0.01                                       | (10.98 ± 0.11) %                      | (1.88 ± 0.01) × 10−2 |                      |
| ALANINE HEXAREVERSE KL                            | 2.56 × 108       | −529.26 ± 0.26                                       | (21.83 ± 1.30) %                      | (7.73 ± 0.63) × 10−2 |                      |
| PEPTIDE                    | FAB              | 4.2 × 108                                            | −532.98 ± 0.01                        | (14.55 ± 0.05) %     | (6.43 ± 0.03) × 10−2 |
| (d = 180) TA-BG            | 4 × 108          | −533.43 ± 0.00                                       | (18.22 ± 0.15) %                      | (2.59 ± 0.03) × 10−2 |                      |
| CMT (OURS)                 | 4 × 108          | −533.51 ± 0.01 (29.63 ± 0.08) % (2.48 ± 0.02) × 10−2 |                                       |                      |                      |
| FORWARD KL                 | 4.2 × 109        | −276.76 ± 0.00                                       | (5.85 ± 0.03) %                       | (1.58 ± 0.01) × 10−2 |                      |
| ELIL                       | REVERSE KL       | 2.56 × 108                                           | −262.34 ± 3.48                        | (1.26 ± 0.53) %      | (2.61 ± 0.27) × 10−1 |
| TETRA                            | FAB              | 8.43 × 108                                           | −276.67 ± 0.01                        | (7.21 ± 0.08) %      | (7.54 ± 0.14) × 10−2 |
| PEPTIDE (d = 219) TA-BG    | 8 × 108          | −277.40 ± 0.06                                       | (13.75 ± 1.42) % (2.54 ± 0.13) × 10−2 |                      |                      |
| CMT (OURS)                 | 8 × 108          | −277.83 ± 0.00 (26.06 ± 0.26) % (3.13 ± 0.03) × 10−2 |                                       |                      |                      |
| ALANINE DIPEPTIDE (d = 60) |                  |                                                      |                                       |                      |                      |

![6_image_0.png](6_image_0.png)

In this section, we compare our approach against state-of-the-art methods on four challenging molecular systems. We provide a brief overview of the experimental setup here, with full details in Appendix D. Additional experimental results are provided in Appendix B, including extended performance metrics, an ablation study on the effect of both constraints, and an analysis of different trust-region bounds across systems of different dimensionality.

## 5.1 Experimental Setup

Benchmark problems. Our evaluation covers a range of molecular systems, beginning with the well-studied alanine dipeptide (d = 60) (Dibak et al., 2022b; Stimper et al., 2022b; Midgley et al., 2022; Tan et al., 2025a), and extending to the larger alanine tetrapeptide (d = 120) and alanine hexapeptide (d = 180), which have only recently been addressed using variational methods (Schopmans & Friederich, 2025). In addition, we introduce a new benchmark, the ELIL tetrapeptide (d = 219), which is higher-dimensional and which contains more complex side chain interactions compared to the alanine hexapeptide. To the best of our knowledge, this represents the largest and most complex molecular system investigated using variational approaches to date. A detailed description of all benchmark systems is provided in Appendix D.2, and visualizations of all systems can be found next to Table 1. While using Lagrangian multipliers as a stopping criterion is possible (as λ = η = 0 implies satisfied constraints), we use a fixed number of annealing steps ˜I to strictly control the computational budget for fair benchmarking (see Algorithm 2). Baseline methods. Our main baselines are Flow Annealed Importance Sampling Bootstrap (FAB) (Midgley et al., 2022) and Temperature-Annealed Boltzmann Generators (TA-BG) (Schopmans & Friederich, 2025), which currently define the state of the art for variational sampling of molecular systems. For reference, we also include reverse and forward KL training; the latter leverages ground truth samples obtained from molecular dynamics (MD) simulations (see Appendix D.2). To ensure a fair comparison, all methods use neural spline flows (Durkan et al., 2019) and identical architectures. Performance criteria. We evaluate methods primarily using three criteria. First, the evidence upper bound (EUBO), computed with ground truth MD samples. Up to an additive constant, the EUBO corresponds to the forward KL divergence and is therefore well suited for detecting mode collapse (Blessing et al., 2024). Second, we consider the effective sample size (ESS), defined as ESS(*q, p*) :=Ex∼q-(p(x)/q(x))
2−1. ESS is a common measure of sample quality, but it is known to be less reliable for assessing mode collapse (Blessing et al., 2024). Finally, we consider Ramachandran plots as a qualitative criterion for assessing mode collapse. These plots visualize the 2D log-density of the joint distribution of a pair of dihedral angles in a peptide's backbone. This is a low-dimensional representation of important molecular configurations, making it possible to assess whether the generated samples capture all relevant modes of the distribution or fail to represent certain regions of the state space. For more details on Ramachandran plots, we refer to Appendix D.3 and Schopmans & Friederich (2025). To assess the quality of the Ramachandran plots, we use the total variation distance (Ram TV) between the model-sampled and ground-truth (MD) Ramachandran histograms, as the TV distance is symmetric and more naturally reflects the bidirectional nature of matching generated and target Boltzmann distributions, thereby also penalizing overestimation of density by the model. For details on all metrics, we refer to Appendix D.3. Since evaluating the target density of molecular systems is typically expensive, we also report the number of target evaluations required by each method.

## 5.2 Results

Main results. The main findings are summarized in Table 1. Across all systems and metrics, our method outperforms the baselines while requiring the same or fewer target evaluations. It produces samples closer to the ground-truth distribution (EUBO), allows more efficient importance sampling (ESS), and provides superior mode coverage and resolution of metastable high-energy regions (RAM TV). While the performance gap between our method and the baselines is less pronounced for smaller systems, it widens substantially for the larger ones. In particular, on alanine hexapeptide and ELIL tetrapeptide, our method attains approximately twice the ESS of competing approaches, while also avoiding mode collapse, as reflected in improved EUBO and Ram TV values. In contrast, the reverse KL objective exhibits significant mode collapse, as evidenced by the widening gap in EUBO and Ram TV relative to the other methods, with the most pronounced discrepancy observed on the largest system, ELIL tetrapeptide. Taken together, the consistency of these trends across metrics and systems highlights the robustness of our method, particularly in challenging high-dimensional systems. Ablation study for constraints. Additionally, we investigate the effect of different constraint choices on the performance of the alanine hexapeptide system. Specifically, we compare four settings: using both constraints, each constraint individually, and no constraint (which corresponds to importance-weighted forward KL minimization). The results are summarized in Figure 2 and Figure 3. Figure 2a shows that omitting the trust-region constraint causes entropy to decrease rapidly, which leads to mode collapse during training. Moreover, using only the entropy constraint yields unstable training, as evidenced by violations of the prescribed linear entropy decay. In contrast, incorporating a trust-region constraint stabilizes training, as reflected in Figure 2b, where it produces a substantially higher ESS between successive intermediate densities. Figure 3 shows Ramachandran plots of alanine hexapeptide with the constraints selectively enabled or disabled. Visible signs of mode collapse appear in all cases except for the tempered (7) and geometric-tempered (9) variants, with the most accurate Ramachandran plot observed in the latter. Overall, our findings indicate that both constraints are necessary to achieve high ESS values while simultaneously avoiding mode collapse.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

## 6 Conclusion

We have introduced *Constrained Mass Transport* (CMT), a variational framework for constructing intermediate distributions that transport probability mass from a tractable base measure to a complex, unnormalized target. By enforcing constraints on both the KL divergence and the entropy decay between successive steps, CMT balances exploration and convergence, thereby mitigating mass teleportation, reducing mode collapse, and promoting smooth distributional overlap. Our empirical evaluation across established Boltzmann generator benchmarks and the here proposed ELIL tetrapeptide, learned purely from energy evaluations without access to molecular dynamics samples, demonstrates that CMT consistently outperforms existing annealing-based and variational baselines, achieving over 2.5× higher effective sample size while preserving mode diversity. Promising directions for future work include exploring alternative approximation families Q and divergences D for learning intermediate densities, which may lead to further performance improvements. Applying our method in Cartesian coordinate representations also presents an interesting avenue, as it facilitates transferability across different molecular systems (Klein & Noe, 2024; Tan ´ et al., 2025b). A key limitation of the current approach is the large number of gradient updates needed to approximate each intermediate target during training. Future work could investigate using more efficient loss functions, such as the log-variance loss (Richter et al., 2020), to reduce computational cost and improve scalability.

## Reproducibility Statement

The source code for all experiments is available at https://github.com/ ChristophervonKlitzing/CMT-Molecular. The ground-truth molecular dynamics data have been made publicly accessible at https://doi.org/10.5281/zenodo.18822445.

## Acknowledgments

The authors acknowledge support by the state of Baden-Wurttemberg through bwHPC. This work is ¨ supported by the Helmholtz Association Initiative and Networking Fund on the HAICORE@KIT partition. D.B. acknowledges support by funding from the pilot program Core Informatics of the Helmholtz Association (HGF). H.S. acknowledges financial support by the German Research Foundation (DFG) through the Research Training Group 2450 "Tailored Scale-Bridging Approaches to Computational Nanoscience". P.F. acknowledges funding from the Klaus Tschira Stiftung gGmbH (SIMPLAIX project) and the pilot program Core-Informatics of the Helmholtz Association (KiKIT project).

## References

Osama Abdin and Philip M Kim. Pepflow: direct conformational sampling from peptide energy landscapes through hypernetwork-conditioned diffusion. *bioRxiv*, pp. 2023–06, 2023.

Abbas Abdolmaleki, Rudolf Lioutikov, Jan R Peters, Nuno Lau, Luis Pualo Reis, and Gerhard Neumann. Model-based relative entropy stochastic search. Advances in Neural Information Processing Systems, 28, 2015.

Abbas Abdolmaleki, Jost Tobias Springenberg, Jonas Degrave, Steven Bohez, Yuval Tassa, Dan Belov, Nicolas Heess, and Martin Riedmiller. Relative entropy regularized policy iteration. arXiv preprint arXiv:1812.02256, 2018a.

Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. *arXiv preprint arXiv:1806.06920*, 2018b.

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pp. 22–31. PMLR, 2017.

Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In *International conference on machine learning*, pp.

151–160. PMLR, 2019.

Tara Akhound-Sadegh, Jungyoon Lee, Joey Bose, Valentin De Bortoli, Arnaud Doucet, Michael M.

Bronstein, Dominique Beaini, Siamak Ravanbakhsh, Kirill Neklyudov, and Alexander Tong. Progressive inference-time annealing of diffusion models for sampling from boltzmann densities. In *ICML 2025 Generative AI and Biology (GenBio) Workshop*, 2025. URL https:
//openreview.net/forum?id=uoU24LvcuV.

Riad Akrour, Gerhard Neumann, Hany Abdulsamad, and Abbas Abdolmaleki. Model-free trajectory optimization for reinforcement learning. In *International Conference on Machine Learning*, pp.

2961–2970. PMLR, 2016.

Riad Akrour, Abbas Abdolmaleki, Hany Abdulsamad, Jan Peters, and Gerhard Neumann. Model-free trajectory-based policy optimization with monotonic improvement. *Journal of machine learning* research, 19(14):1–25, 2018.

Riad Akrour, Joni Pajarinen, Jan Peters, and Gerhard Neumann. Projections for approximate policy iteration algorithms. In *International Conference on Machine Learning*, pp. 181–190. PMLR,
2019.

Michael S Albergo and Eric Vanden-Eijnden. Nets: A non-equilibrium transport sampler. arXiv preprint arXiv:2410.02711, 2024.

Michael Arbel, Alex Matthews, and Arnaud Doucet. Annealed flow transport monte carlo. In International Conference on Machine Learning, pp. 318–330. PMLR, 2021.

Oleg Arenz, Mingjun Zhong, and Gerhard Neumann. Trust-region variational inference with gaussian mixture models. *Journal of Machine Learning Research*, 21(163):1–60, 2020.

Oleg Arenz, Philipp Dahlinger, Zihan Ye, Michael Volpp, and Gerhard Neumann. A unified perspective on natural gradient variational inference with gaussian mixture models. *arXiv preprint* arXiv:2209.11533, 2022.

Nathan Argaman and Guy Makov. Density functional theory: An introduction. American Journal of Physics, 68(1):69–79, January 2000. ISSN 1943-2909. doi: 10.1119/1.19375. URL http:
//dx.doi.org/10.1119/1.19375.

Roi Benita, Michael Elad, and Joseph Keshet. Spectral analysis of diffusion models with application to schedule design. *arXiv preprint arXiv:2502.00180*, 2025.

Nicolas Bereux, Aur ´ elien Decelle, Cyril Furtlehner, Lorenzo Rosset, and Beatriz Seoane. Fast ´
training and sampling of restricted boltzmann machines. *arXiv preprint arXiv:2405.15376*, 2024.

Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond ELBOs: A large-scale evaluation of variational methods for sampling. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 4205–4229. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/blessing24a.html.

Denis Blessing, Julius Berner, Lorenz Richter, Carles Domingo-Enrich, Yuanqi Du, Arash Vahdat, and Gerhard Neumann. Trust region constrained measure transport in path space for stochastic optimal control and inference, 2025. URL https://arxiv.org/abs/2508.12511.

Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, Cambridge, UK, 2004. ISBN 9780521833783. Seventh printing with corrections, 2009.

Rob Brekelmans and Frank Nielsen. Variational representations of annealing paths: Bregman information under monotonic embedding. *Information Geometry*, 7(1):193–228, 2024.

Rob Brekelmans, Vaden Masrani, Thang Bui, Frank Wood, Aram Galstyan, Greg Ver Steeg, and Frank Nielsen. Annealed importance sampling with q-paths. *arXiv preprint arXiv:2012.07823*, 2020.

Richard P Brent. *Algorithms for minimization without derivatives*. Courier Corporation, 2013. Onur Celik, Zechu Li, Denis Blessing, Ge Li, Daniel Palenicek, Jan Peters, Georgia Chalvatzaki, and Gerhard Neumann. Dime: Diffusion-based maximum entropy reinforcement learning. arXiv preprint arXiv:2502.02316, 2025.

Omar Chehab and Anna Korba. A practical diffusion path for sampling. arXiv preprint arXiv:2406.14040, 2024.

Junya Chen, Danni Lu, Zidi Xiu, Ke Bai, Lawrence Carin, and Chenyang Tao. Variational inference with holder bounds. *arXiv preprint arXiv:2111.02947*, 2021.

Jaemoo Choi, Yongxin Chen, Molei Tao, and Guan-Horng Liu. Non-equilibrium annealed adjoint sampler. *arXiv preprint arXiv:2506.18165*, 2025.

Nicolas Chopin, Omiros Papaspiliopoulos, et al. *An introduction to sequential Monte Carlo*, volume 4.

Springer, 2020.

Nicolas Chopin, Francesca R Crucinio, and Anna Korba. A connection between tempering and entropic mirror descent. *arXiv preprint arXiv:2310.11914*, 2023.

Andrew R Conn, Nicholas IM Gould, and Philippe L Toint. *Trust region methods*. SIAM, 2000.

Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. *Journal of the* Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006.

Manuel Dibak, Leon Klein, Andreas Kramer, and Frank No ¨ e. Temperature steerable flows and ´
boltzmann generators. *Physical Review Research*, 4(4):L042005, 2022a.

Manuel Dibak, Leon Klein, Andreas Kramer, and Frank No ¨ e. Temperature steerable flows and ´
Boltzmann generators. *Physical Review Research*, 4(4):L042005, October 2022b. doi: 10.1103/
PhysRevResearch.4.L042005.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. *arXiv* preprint arXiv:1605.08803, 2016.

Simon Duane, A. D. Kennedy, Brian J. Pendleton, and Duncan Roweth. Hybrid Monte Carlo. Physics Letters B, 195(2):216–222, September 1987. ISSN 0370-2693. doi: 10.1016/0370-2693(87)
91197-X.

Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. Advances in neural information processing systems, 32, 2019.