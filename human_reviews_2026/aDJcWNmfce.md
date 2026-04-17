# Strictly Constrained Generative Modeling via Split Augmented Langevin Sampling

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 4

## Abstract
Deep generative models hold great promise for representing complex physical systems, but their deployment is currently limited by the lack of guarantees on the physical plausibility of the generated outputs. Ensuring that known physical constraints are enforced is therefore critical when applying generative models to scientific and engineering problems. We address this limitation by developing a principled framework for sampling from a target distribution while rigorously satisfying physical constraints. Leveraging the variational formulation of Langevin dynamics, we propose Split Augmented Langevin (CASAL), a novel primal-dual sampling algorithm that enforces constraints progressively through variable splitting, with convergence guarantees. While the method is developed theoretically for Langevin dynamics, we demonstrate its effective applicability to diffusion models. We apply our method to diffusion-based data assimilation on a complex physical system, where enforcing physical constraints substantially improves both forecast accuracy and the preservation of critical conserved quantities. We also demonstrate the potential of CASAL for challenging feasibility problems in optimal control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles the problem of sampling from generative models while strictly satisfying physical constraints like conservation laws. Existing approaches either fail to enforce constraints exactly (penalty methods) or become trapped in limited regions under non-convex constraints (projection methods). The authors propose Split Augmented Langevin (SAL), which uses variable splitting and augmented Lagrangian dynamics to progressively enforce constraints while preserving exploration capability. The method operates in a training-free manner on pre-trained diffusion models and provides convergence guarantees through duality analysis. Experiments on energy-preserving field generation, data assimilation for the Burgers equation, and optimal control problems demonstrate that SAL achieves both strict constraint satisfaction and accurate conditional sampling, outperforming existing baselines in maintaining physical plausibility and forecast accuracy.

### Strengths
**Principled framework with theoretical guarantees:** The variational formulation and duality analysis provide strong theoretical foundations, proving that the relaxed problem recovers the target distribution as coupling strength increases.

**Training-free and modular:** The method works as a drop-in replacement for standard Langevin steps in pre-trained diffusion models, requiring no retraining or additional data.

**Strong empirical validation:** Experiments on diverse physical systems demonstrate practical effectiveness, showing improved constraint satisfaction and forecast accuracy compared to existing baselines.

### Weaknesses
**Limited Complexity of Constraints.** While the experiments effectively demonstrate the method's core capabilities, the constraint types considered are relatively simple. The energy conservation constraint in Section 5.1 is a quadratic sphere, and the mass conservation in Section 5.2 is linear. Real-world physical systems often involve more complex constraints, such as coupled higher-order nonlinear conservation laws (e.g., multi-component energy functionals with nonlinear operators), or systems governed by multiple interacting PDEs with intricate coupling structures. It remains unclear how the projection step and the coupling parameter would perform when the constraint manifold has significantly more complex geometry or when the projection itself becomes computationally expensive. Evaluating SAL on such challenging constraints would strengthen confidence in its applicability to complex physical modeling tasks.

**Gap between Theoretical Analysis and Actual Implementation.** While the paper presents a practical and effective algorithm, there is a notable gap between the theoretical analysis and the actual implementation. The convergence analysis in Section 4.3 focuses on the optimization problem (Eq 4.4) in the space of probability measures, establishing strong duality (Proposition 5) and asymptotic recovery (Proposition 6). However, the actual algorithm (Eq 4.5-4.6) is a stochastic sampling procedure operating in sample space with Langevin noise. The missing piece is a rigorous convergence analysis of the proposed iterative scheme (4.6) itself, which would require establishing: (1) Non-asymptotic convergence rates in Wasserstein distance, (2) Finite-sample guarantees for fixed $\rho$ values, (3) Guidance on hyperparameter selection ($\rho, \tau, \eta$) based on convergence theory. The current theoretical results only guarantee that as $\rho \to \infty$, the solution to (4.4) approaches $p_C$, but do not characterize how the finite-step stochastic algorithm (4.6) approximates this solution. This is acknowledged by the authors as a limitation, but addressing it would significantly strengthen the contribution. Despite this gap, the empirical results convincingly demonstrate the method's effectiveness.

### Questions
While the theoretical contributions (Propositions 3-6) are rigorous, the connection between the mathematical advantages and the observed performance improvements in Section 5 is not clearly articulated. Specifically, it remains unclear how the duality gap in Proposition 3 directly explains the constraint violations in Figure 3, or how Proposition 5's attained duality resolves both the exploration bias (Figure 2) and projection artifacts (Figure 4) simultaneously. I recommend adding explicit statements or a summary that maps each theoretical result to the corresponding experimental observations, making the paper more accessible to readers outside convex optimization. A comparative table contrasting penalty methods, projected Langevin, and SAL across key mathematical properties (duality, constraint satisfaction, exploration) would significantly improve clarity.

### Soundness
3

### Presentation
2

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
The paper introduces SAL, a principled framework for strictly constrained sampling. By combining a variable-splitting formulation with stochastic primal–dual updates, SAL attains strict feasibility under nonconvex constraints while preserving the exploratory behavior. Theoretical analysis provides strong guarantees, and experiments on physical systems and control tasks demonstrate the method’s potential. However, the experiment results are somewhat limited, leaving some unconvincibility about the practical robustness of the proposed method. I would consider raising my rating if additional experimental results were provided.

### Strengths
1. The SAL algorithm achieves strict feasibility without sacrificing exploration in constrained generation, addressing the weakness of previous projected or penalty-based approaches through a well-motivated formulation.
2. The paper provides a clear variational interpretation of constrained sampling and offers theoretical guarantees supporting the soundness and effectiveness of the proposed method.
3. The experiments cover a broad range of tasks and settings, demonstrating that SAL maintains physical constraints while preserving sampling diversity and quality.

### Weaknesses
1. None of the three experiments report quantitative metrics or summary tables comparing different methods.
2. Lines 329–333 mention several baselines intended for comparison, yet the most relevant one, the Primal–Dual Langevin method, does not appear in the reported experimental results.
3. Lines 420–421 note that ADMM is a classical solver for obstacle-avoidance problems and highlight its limitations, but the experiments result does not provide a direct comparison between ADMM and the SAL, which is unreasonable.
4. Lines 320–322 discuss computational cost, claiming that SAL adds only the cost of a projected diffusion step compared to the unconstrained baseline. However, the additional variable-splitting updates also may introduce extra overhead. Additionaly, a quantitative runtime comparison with baselines would be necessary to substantiate the efficiency claim.

### Questions
1. I understand that the SAL combines the strengths of several existing baselines. Since prior work such as Primal–Dual Sampling already performs well, to what extent is SAL an incremental improvement that integrates ideas from Projected LMC, variable splitting, and augmented Lagrangian methods? It would be helpful to include a summary table comparing different baselines across key aspects (feasibility, stricticity, etc.) to make the distinctions clearer.
2. The paper mentions that a projection operator is required for SAL as well as for other methods. However, do such projection operators always exist for arbitrary physical constraints or control objectives? If not, could the authors discuss possible strategies or approximations?
3. Given that the authors are trying to model constrained sampling over physical field functions rather than simple random variables, and physical conservation laws also apply to fields, is it formally appropriate to represent the problem directly using random variables $x$ and $z$? Some clarification on this abstraction would make the formulation more convincing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Split Augmented Langevin (SAL), a training-free sampler that enforces hard constraints by splitting variables and projecting a companion variable onto the constraint set each step, with the final output $z_T \in C$. It provides a variational/duality view, consistency to the strictly constrained law as coupling increases, and a drop-in use within diffusion models.

Contributions:
1) A split augmented Langevin sampler with projection that enforces samplewise constraints while preserving exploration.
2) A relaxed duality analysis with attained strong duality and recovery of the strictly constrained law.
3) A practical integration recipe for diffusion/latent diffusion requiring only a projection operator $P_C$, demonstrated on three tasks.

### Strengths
1) A split augmented Langevin formulation that enforces per-sample constraints while retaining exploration; a practical combination of variable splitting and projection for sampling.\\
2) Coherent variational/duality setup with clear algorithms; sensible baselines demonstrating feasibility and distributional fidelity.
3) Motivation is explicit; notation and procedures are readable; appendices provide proofs, variants, and implementation notes.
4) Training-free and model-agnostic; integrates with diffusion pipelines, making strict constraints more accessible in scientific applications.

### Weaknesses
1) Projection robustness: many constraints need iterative or approximate $P_C$. Quantify how projection error affects feasibility and sampling bias; include experiments sweeping inner-iteration counts or projection tolerances.
2) Hyperparameter tuning: performance depends on $\rho$ and dual step $\eta$. Propose adaptive schedules driven by observed coupling gaps $\|x_t-z_t\|$ or constraint residuals, and compare against fixed/annealed baselines.
3) Experimental breadth: add compact, classical PDE tests where projections are standard and informative (Poisson $Au=b$, Darcy with mass/flux balance, incompressible NS via Hodge projection), reporting PDE residuals, divergence norms, spectra.
4) Latent diffusion details: when enforcing constraints in decoded space, discuss scaling $\rho$ through the decoder Jacobian and show sensitivity analyses to ensure stability and consistent enforcement.

### Questions
1) Problem setting and constraints. For physics, equality-in-expectation $\mathbb{E}_q[h(x)]=0$ seems insufficient. In many cases, the idealized requirement $\mathbb{E}_q[|h(x)|]=0$ would force $h(x)=0$ almost surely (i.e., strict feasibility), but this is typically impractical to enforce via smooth penalties. Please clarify your recommended formalization: average constraints (Section 3) versus support constraints $x\in C$ almost surely (Section 4). Do you envisage $h(x)\ge 0$ and $\lambda\ge 0$ anywhere, or do you recommend set membership with projection as the primary approach?


2) Finite-time behavior and stopping. Can you relate finite $\rho$ and step sizes to discrepancy from the target conditional $p_C$? A practical stopping rule based on $\|x_t - z_t\|$ or constraint residual thresholds would help practitioners decide when sampling is accurate enough.

3) Adaptive schedules for $\rho$ and $\eta$. Do you recommend an adaptive policy driven by observed coupling gaps or residuals (e.g., increase $\rho$ when violations plateau, decrease when exploration slows)? A short ablation comparing fixed, annealed, and adaptive schedules would be valuable.

4) Inexact projections and robustness. When $P_C$ is computed approximately (few inner iterations), what error conditions preserve convergence and strict feasibility in practice? Please include an experiment sweeping projection tolerance/iterations and reporting feasibility rates and sampling bias.

5) Intersections of multiple constraints. For $C=\cap_i C_i$, do you prefer a single projection onto the intersection, alternating projections onto $C_i$, or multiple split variables $z^{(i)}$ with separate dual updates? A small comparison on stability, wall-clock, and feasibility would inform users.

6) Classical PDE benchmarks. Consider adding concise tests where projections are standard and informative: Poisson with linear constraints $Au=b$, Darcy flow with mass/flux balance, and incompressible Navier–Stokes via Hodge projection. Reporting $\|Au-b\|_2$, divergence norms, spectra, and feasibility histograms would strengthen generality claims.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Split Augmented Langevin (SAL), a method for strictly enforcing equality constraints during sampling in generative methods such as score-based diffusion models. It introduces a primal–dual Langevin framework that couples unconstrained and projected variables to guarantee feasibility at every step.

### Strengths
**S1.** The paper effectively highlights the central difficulty in constrained generation, balancing feasibility with sample diversity, and offers a thoughtful critique of simple projection-based solutions.

**S2.** It presents a theoretically grounded primal–dual Langevin formulation that unifies ideas from constrained optimization and generative sampling in a principled way.

**S3.** The exposition is clear and precise, with mathematical reasoning and well-structured derivations that make the approach easy to follow.

### Weaknesses
**W1.** The paper does not include enough baselines or adequately discuss prior work on constrained generation. Comparable methods such as PCFM [2], D-Flow [3], DiffusionPDE [4], and ECI [1] are only briefly mentioned or omitted. ECI, for instance, also enforces hard constraints, while PCFM demonstrates constrained sampling under nonconvex conditions: both directly relevant for benchmarking.

**W2.** The empirical analysis is limited, lacking standard quantitative metrics such as MMSE, SMSE, or FID reported routinely in this literature. Reported results rely largely on qualitative visualizations and histograms, and even show non-zero constraint violations (e.g., Fig. 3) without quantifying their scale or assessing the degree of enforcement softens your claim of hard constraint.

**W3.** The evaluated constraints are relatively simple, and the computational claims appear optimistic. The method assumes analytic projectors, but more complex nonlinear or PDE-based constraints would require solving constrained least-squares problems at every step, which could become computationally expensive and limit scalability.

**W4.** The algorithm assumes that projecting Gaussian noise to obtain ($z_0$) yields feasible initial states. This assumption may hold for simple quadratic or linear constraints but lacks guarantees for challenging manifolds, such as PDE solutions with shocks and discontinuities. Testing on more complex physical systems, as in PCFM [2] or guided functional diffusion methods [5], would better support claims of generality.

**References**

[1] Cheng et al., *Gradient-Free Generation for Hard-Constrained Systems (ECI)*, arXiv:2412.01786 (2024).

[2] Utkarsh et al., *Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints*, arXiv:2506.04171 (2025).

[3] Ben-Hamu et al., *D-Flow: Differentiating through Flows for Controlled Generation*, arXiv:2402.14017 (2024).

[4] Huang et al., *DiffusionPDE: Generative PDE-Solving under Partial Observation*, NeurIPS 37 (2024).

[5] Yao et al., *Guided Diffusion Sampling on Function Spaces with Applications to PDEs*, arXiv:2505.17004 (2025).

[6] Chamon et al., *Constrained Sampling with Primal-Dual Langevin Monte Carlo*, NeurIPS 37 (2024).

### Questions
**Q1.** How does the proposed SAL method compare quantitatively to prior constrained generation frameworks such as PCFM [2], D-Flow [3], DiffusionPDE [4], or ECI [1] that also enforce hard or functional constraints?

**Q2.** What is the average magnitude of the constraint violation observed in Figure 3, and does the method guarantee strict satisfaction (i.e., ($||h(x)|| = 0$)) or only approximate feasibility?

**Q3.** How would the projection operator behave for complex or nonlinear constraints where analytical projectors are unavailable? Would this require solving an inner constrained least-squares problem at every step, and how does that impact computational cost?

**Q4.** Given that the initialization step projects Gaussian noise to obtain ($z_0$), what guarantees exist that this projection produces feasible or representative samples, especially for high-dimensional or nonconvex manifolds?

**Q5.** How does your method perform in high-dimensional or multi-modal constrained settings? Does it exhibit the slow mixing behavior often observed in Langevin-based samplers, and what evidence supports stable convergence in such regimes?

### Soundness
2

### Presentation
2

### Contribution
2
