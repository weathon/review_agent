# Learning Escorted Protocols For Multistate Free-Energy Estimation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Estimating relative free energy differences between multiple thermodynamic states lies at the core of numerous problems in computational biochemistry. Traditional estimators, such as Free Energy Perturbation and its non-equilibrium counterpart based on the Jarzynski equality, rely on defining a switching protocol between thermodynamic states and computing the free energy difference from the work performed during this process. In this work, we present a method for learning such switching protocols within the class of escorted protocols that combine deterministic and stochastic steps. For this purpose, we use Conditional Flow Matching, and  introduce Conditional Density Matching (CDM)  for the purpose of estimating the change in Free-Energy. We further reduce the variance in the multistate setting by coupling multiple flows between thermodynamic states into a Flow Graph, enforcing estimator consistency across different transition paths.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A paper proposing an original solution to an important computational physics problem, namely computing free energy changes in non-equilibrium thermodynamics. The proposed solution uses flow matching techniques in an innovative way to suggest a more efficient/ accurate algorithm to sample Monte Carlo trajectories over which to estimate such free energy changes.

### Strengths
I really enjoyed reading this paper, albeit as a somewhat knowledgeable outsider. The authors make a good effort to provide a comprehensive background accessible to non-specialists (although I wonder how well it fares with non-physicists). The idea of using ML to minimise the Jaczinsky lower bound is elegant and the proposed improvements in terms of adopting larger time steps/ extending to multi-state estimation are potentially important for the community. The empirical validation is well carried out albeit not extensive (similar in that to physics papers)

### Weaknesses
The main weakness to me is how much this paper could appeal outside of the computational physics community, which is but a (small) strand in the ICLR community. Some opportunities to broaden the appeal of the paper are listed below in the questions.

### Questions
- Is it conceivable that your algorithmic improvements to learning the escorting protocol could be extended to other common ML scenarios where distributions are to be matched, e.g. diffusion models?
- Your approach is a practical solution to minimising the bias/ variance of the Monte-Carlo Jaczinsky estimator, does it come with guarantees/ special scenarios in which it could turn out to be exact?
- I didn't get much rationale for the choice of flow architecture etc, which potentially could be a factor in determining efficiency gains. Did you just take an off-the-shelf approach or are there mileage in optimising that side?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper learns escorted non-equilibrium protocols for free-energy estimation by pairing Conditional Flow Matching for the escort field with a “Conditional Density Matching” objective for the time-dependent potential so the pair approximately satisfies the continuity equation. It also adds two pragmatic pieces: Lie–Trotter splitting to make work evaluation cheaper and an Escorted Protocol Flow Graph (EPFG) so K states only need K–1 learned protocols but still feed MBAR with all pairwise work values. On alanine dipeptide, the approach improves accuracy over TFEP and benefits from EPFG concatenation.

### Strengths
- Overall, well-written paper.
- Jointly learn the escort vector field and the time-dependent potential (CFM + CDM) so the learned pair targets the continuity equation, which is exactly what E-NEQ needs.
- Lie–Trotter splitting reduces the number of divergence evaluations in the work computation—practically important for MD-style forces.
- Train only $K–1$ protocols yet still populate MBAR with all $W_{i\to j}$, cutting training cost while improving multi-state consistency.

### Weaknesses
- The empirical validation is limited to ADP; it’s not obvious how the training and stability behave on larger biomolecular benchmarks. 
- Some integrator choices remain touchy on long concatenations.
- The CDM/DM construction is essentially an MLE on a time-indexed density $p_\theta(x,t)$; the paper introduces new terminology (“Density Matching”) but the core estimator is maximum likelihood on $(x,t)$ samples (later replaced by conditional sampling). That raises questions about identifiability (many $(b_\phi,U_\theta)$ pairs can satisfy the continuity equation approximately) and about how model misspecification in $p_\theta(x,t)$ propagates into work-estimate bias.

### Questions
- How is the density matching (DM) in Equation 15 different from maximum likelihood estimation? (I’m confused by the terminology—why call it “Density Matching” when Eq. (15) is a negative log-likelihood over \((x,t)\)?) In line 263, the authors say “MLE”—that’s maximum likelihood estimation, right (still need to define it!)? 
- The method learns both $b_\phi(x,t)$ and $U_\theta(x,t)$ to “collectively” solve the continuity equation. Is there an identifiability or degeneracy issue—i.e., multiple $(b,U)$ pairs giving similar likelihood but very different work statistics? What regularization or constraints help?
- In the Lie–Trotter split, how large can the time step be before the discrete E-NEQ estimator’s bias shows up meaningfully in MBAR? Any diagnostic or adaptivity you recommend during training/evaluation? 
- EPFG builds all pairwise work values from K–1 learned edges. Do we lose anything versus learning each pair directly (e.g., variance concentration on hard pairs), and could the graph topology (choice of hub k) be learned or adapted?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a neural formulation of the Escorted Jarzynski Equality (E-JE) to estimate free-energy differences with reduced variance. The authors reinterpret the E-JE framework using neural potentials $U_\theta$ and escort fields $b_\phi$, trained via conditional flow matching to approximate the optimal nonequilibrium protocol. The paper is well written, with clear theoretical exposition and solid mathematical grounding, providing an intuitive bridge between nonequilibrium statistical mechanics and neural flow–based modeling. Experiments on the alanine dipeptide system demonstrate the feasibility of the approach and compare its performance with classical methods such as Umbrella Sampling and TFEP.

### Strengths
1. Provides clear and pedagogical background on the Jarzynski Equality (JE) and its escorted extension, making a technically challenging topic highly accessible.
2. Builds on a solid theoretical foundation, with mathematically consistent derivations and well-motivated neural extensions.
3. The formulation elegantly connects non-equilibrium statistical mechanics with modern neural flow modeling, offering a promising bridge between physics-based and data-driven approaches.

### Weaknesses
**Limited discussion of related work and positioning**
The paper presents a neural formulation of the Escorted Jarzynski Equality (E-JE), but does not clearly situate it within the broader landscape of neural flow, Schrödinger bridge, or optimal control–based free-energy estimation methods. Moreover, the boundary between background explanation and related work discussion is somewhat blurred, making it difficult to discern which parts of the formulation are newly contributed by this paper versus prior theoretical developments. A clearer distinction between foundational background and the paper’s novel contributions would help readers better appreciate the originality of the work.

**Lack of verification for continuity-equation consistency**
A central theoretical condition of the proposed method is the continuity equation
$$\partial_t p_\theta + \nabla \cdot(p_\theta b_\phi)=0,$$
ensuring that $U_\theta$ and $b_\phi$ form a consistent escorted protocol. However, the paper does not quantitatively evaluate or visualize how well this condition is satisfied after training.

**Missing JE ($b$ = 0) baseline — variance reduction unquantified**
The main purpose of E-JE is to reduce variance relative to the standard JE, but the JE ($b$ = 0) baseline is entirely omitted from experiments.
Although both yield identical $\Delta F$ values, without reporting $\mathrm{Var}(e^{-\beta W})$ it is impossible to demonstrate the actual benefit of the learned escort field.

**Limited experimental diversity and missing ablations**
Experiments are restricted to a single system (alanine dipeptide) and comparisons are limited to classical methods such as TFEP and umbrella sampling. There is no evaluation against recent neural baselines or component-wise ablations.

**Insufficient explanation for separating $b_\phi$ and $U_\theta$**
It is unclear why $b_\phi$ (flow field) and $U_\theta$ (potential) must be trained as independent networks rather than being coupled through a gradient or consistency constraint. A brief discussion on the physical or algorithmic motivation for this design choice (e.g., Helmholtz decomposition or non-conservative flow representation) would improve clarity.

### Questions
1. **JE baseline and variance reduction**
Since the standard Jarzynski Equality (JE) corresponds to the case ($b$ = 0), could you include this baseline to quantify the actual variance reduction achieved by the learned escort field?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for learning escorted switching protocols to estimate free-energy differences between multiple thermodynamic states. The approach combines Conditional Flow Matching (CFM) to learn a deterministic escorting vector field and introduces Conditional Density Matching (CDM) to learn a time-dependent potential. The authors also propose Lie-Trotter splitting to reduce computational cost and a flow graph construction to scale to multiple states. Experiments on alanine dipeptide (ADP) show improvements over Targeted Free-Energy Perturbation (TFEP) in terms of estimation accuracy.

### Strengths
- The paper tackles an important problem in computational biochemistry: efficient and accurate free-energy estimation across multiple states.

- The idea of jointly learning the escorting vector field and the time-dependent potential via CFM and CDM is a meaningful attempt to reduce variance in free-energy estimators.

- The use of Lie-Trotter splitting and flow graph construction addresses practical challenges in computational cost and scalability.

- Experimental results on ADP show that the proposed E-NEQ estimator outperforms TFEP in several settings.

### Weaknesses
Overall, it is hard for me to follow this paper, and I think there are many related but not well discussed in this work. But I would be open to adjusting my score if the authors can address my concerns.

- It is very hard for me to identify what the actual contribution of this paper is.
  - If I understand the work correctly, Sections 2 and 3 both appear to be preliminary material rather than novel methods. 
  - Section 4 supposedly presents the proposed algorithm, but the description is vague and lacks a clear algorithmic definition. I would recommend a pseudo-code to present the algorithm clearly.
  - It is unclear to me what exactly is being optimized or learned, and what the goal of this paper is to address.
- The methodology is vague to me.
  - Section 3 mentions flow matching but does not clearly define the forward and backward processes, how the reverse flow is parameterized, or how training is performed.
  - Section 4 seems to involve a neural network trained with ideas similar to parallel tempering, but no pseudo-code or mathematical formulation is provided.
  - Section 5 discusses Overdamped & Underdamped Langevin and Hamiltonian Monte Carlo, which makes the algorithm a bit clearer. But it is unclear to show the continuous-time SDEs, their discretization schemes, or how stochastic gradients are estimated from data.
  - The use of Metropolis–Hastings correction seems questionable given the high computational cost in high-dimensional generative models, yet the paper does not justify this design choice.
- There is no dedicated related work section, and the paper does not discuss how it differs from prior research in replica exchange or parallel tempering methods. To me, the approach appears very similar in spirit to parallel tempering/replica exchange MCMC methods [1–4], which also sample across multiple states to balance exploration and exploitation. The paper does not clarify whether the proposed method offers any theoretical or empirical advantage over the related work.
- There are some undefined notations or some presentation issues. If I missed some, please let me know where they are. For example, I did not find some important quantities such as $\alpha_R$, $\alpha_L$, $\alpha_D$, $\beta$, $C_5$, and $\alpha'$ in Table 2, and variables $\phi$ and $\psi$ in Figure 4. This makes the results hard to interpret for readers unfamiliar with Alanine Dipeptide.

References:

[1] Accelerating Nonconvex Learning via Replica Exchange Langevin Diffusion. ICLR 2019

[2] Non-convex Learning via Replica Exchange Stochastic Gradient MCMC. ICML 2020

[3] Spectral Gap of Replica Exchange Langevin Diffusion on Mixture Distributions, Stochastic Processes and their Applications, 2022 

[4] Non-reversible Parallel Tempering: A Scalable Highly Parallel MCMC Scheme, Journal of the Royal Statistical Society Series B: Statistical Methodology, 2022

### Questions
- Why is the Metropolis–Hastings correction necessary here, and how is it made computationally feasible in high-dimensional settings?
- How are the Langevin dynamics discretized? What step size, friction, and stochastic gradient estimation schemes are used?

### Soundness
2

### Presentation
1

### Contribution
2
