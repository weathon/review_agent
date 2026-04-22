# Tensor Train Diffusion: A Fast Solver for High-Dimensional Sampling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Diffusion models offer a powerful framework for sampling from complex probability densities by learning to reverse a noising process. A common approach involves solving for the time-reversed stochastic differential equation (SDE), which requires the score function of the evolving sample distribution. The logarithm of this distribution's density is governed by a Hamilton-Jacobi-Bellman (HJB) type partial differential equation (PDE). However, current methods for solving this PDE, such as PINNs or trajectory-based techniques, often suffer from long training times and significant sensitivity to hyperparameter tuning. In this work, we introduce a novel and efficient solver for the underlying HJB equation based on the functional tensor train (FTT) format. The FTT representation leverages latent low-rank structures to efficiently approximate high-dimensional functions, enabling both model compression and rapid computation. By integrating this efficient representation with a backward-in-time iterative scheme derived from backward stochastic differential equations (BSDEs), we develop a fast, robust and accurate sampling method. Our approach overcomes primary bottlenecks of existing techniques, enabling high-fidelity sampling from challenging target distributions with improved efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Tensor Train Diffusion (TTD), a novel and efficient numerical solver for diffusion-based sampling. The core challenge in diffusion models is the estimation of the time-dependent score function, which is required to solve the time-reversed SDE. The authors correctly identify that the log-density of this process is governed by a Hamilton-Jacobi-Bellman (HJB) type PDE.

### Strengths
1. The paper is easy to follow 

2. The computational cost and instability of training samplers are well-known and critical bottlenecks for the widespread use of diffusion models, especially in scientific domains. This paper directly attacks this bottleneck and provides a compelling and effective alternative.

### Weaknesses
1. The entire premise of TTD's efficiency hinges on the assumption that the HJB equation's solution $V(x,t)$ possesses a low-rank structure that FTT can exploit The only benchmark used (Multiwell) is quasi-separable by construction  and is therefore an ideal, best-case scenario for tensor-train methods. It is highly questionable whether this low-rank assumption holds for the complex, highly-entangled densities of general datasets (e.g., natural images). If $V(x,t)$ is not low-rank, the required TT ranks $r$ could grow exponentially, making TTD less efficient than NNs. The paper provides no evidence for this more general case and completely omits this crucial discussion of its primary limitation.

2. This work replaces the familiar (though flawed) NN+SGD stack with a set of tools (FTT, ALS, BSDE solvers, HOSVD) that are highly non-trivial and unfamiliar to the vast majority of the NeurIPS community. This creates a significant practical barrier to adoption, reproduction, and extension by other researchers.

3. The main paper presents the method as more straightforward than it is. The appendix reveals that several highly-engineered components are essential for the method to work at all. Most notably, Appendix A.6 states that "a naive choice of $\tau_n$ ... did not yield a convergent scheme", meaning the adaptive regularization scheme is not a minor tweak but a critical, non-obvious part of the algorithm. The same applies to the choice of basis functions (A.3) and the moving approximation domains (A.4.4) .

4. Add a "Limitations and Future Work" section to the main paper. This section must explicitly address Weakness #1. It should state that the method's efficiency relies on the low-rank assumption and discuss which problem classes (like SciML) are a good fit, while acknowledging that generality to highly-entangled data (like images) is an open and challenging question. This makes the paper more rigorous.

5. Briefly allude to the necessity of the adaptive regularization scheme (from A.6) in the main method section (e.g., 3.1 or 4). This gives proper credit to a key part of the algorithm's success without spending too much space.

6. Add a brief ablation or sensitivity analysis in the appendix regarding key "new" hyperparameters, such as the number of basis functions, to aid reproducibility.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Tensor Train Diffusion (TTD), a novel method for sampling from unnormalized high-dimensional probability distributions. The approach solves the Hamilton-Jacobi-Bellman (HJB) PDE underlying diffusion-based sampling using functional tensor train (FTT) representations rather than neural networks. The key innovation is combining the FTT format—which efficiently represents high-dimensional functions by exploiting low-rank structure.

### Strengths
The paper makes a novel and well-motivated methodological contribution by applying tensor train representations to solve HJB equations for diffusion-based sampling, directly addressing real limitations of neural network-based methods such as long training times and hyperparameter sensitivity. The theoretical foundation is strong, with clear connections established between diffusion models, HJB PDEs, and BSDEs, and rigorous derivations from the BSDE formulation to the discrete loss function.

### Weaknesses
The experimental validation is limited, focusing exclusively on the Multiwell benchmark family, and would be significantly strengthened by demonstrating performance on other problem classes.

### Questions
How does the required TT rank scale with dimension $d$, mode separation $\delta$, and number of modes $m$? Can you characterize this relationship theoretically or empirically?

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
2

### Summary
The paper proposes Tensor Train Diffusion (TTD), a solver for the HJB-type PDE that underlies diffusion-based sampling from complex, high-dimensional unnormalized densities. Instead of training a large neural PDE surrogate, TTD represents the value/log-density function with a functional tensor train (FTT) and derives a backward-in-time scheme from the HJB–BSDE connection. This yields a regression objective based on a BSDE residual, and the FTT parameters are optimized by alternating least squares (ALS) sweeps.

### Strengths
1. It replaces large neural surrogates with a functional tensor-train (FTT) representation, which avoids costly target-density evaluations and reliance on higher-order automatic differentiations.

2. Each TT core is updated by solving a regularized linear least-squares subproblem, enabling stable, fast sweeps.

3. On 10D and 50D multi-modal targets, runtime scales linearly with steps, and TTD shows favorable ESS/log-Z behavior versus DIS; an additional plot also compares against PIS.

### Weaknesses
1. The core experimental results are on Multiwell only (two setups: 10D/50D).

### Questions
1. How sensitive are results to the rank-adaptation threshold and update frequency?

2. Under what structural conditions on V(x,t) do you expect a low TT rank?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes tensor train diffusion, a method for sampling from unnormalized probability densities, combining tensor train representations with SDEs. The authors frame their approach as a diffusion model innovation, positioning it within the recent generative modeling literature. The method employs tensor train decompositions for function approximation and solves Hamilton-Jacobi-Bellman equations through backward-in-time iterations. Numerical experiments on a multiwell synthetic function (with d=10 and d=50) are used to compare against two approaches that also leverage SDE frameworks for addressing this problem.

### Strengths
The paper addresses a fundamental problem in computational statistics and scientific computing that has significant practical importance across many domains from Bayesian inference to statistical physics. The methodology is documented in considerable detail in the main paper and the appendices. A detailed mathematical exposition of the tensor train decomposition framework, the BSDE formulation, and the connection to Hamilton-Jacobi-Bellman equations are provided.

### Weaknesses
The introduction conflates two fundamentally distinct problems by citing diffusion model literature (Ho et al., 2020; Song et al., 2021) as motivation for sampling from known unnormalized densities. This is a classical problem dating back to Metropolis (1953) and has been extensively studied in various fields since then. The papers by Ho et al. and Song et al. address a distinct problem: learning to generate samples from an unknown distribution given only empirical samples, without access to the functional form of the density. Presenting the classical problem of generating samples from unnormalized densities as a novel application of diffusion models misrepresents the problem's rich heritage.

The authors cite Dai et al. (2022) to support their claim that MCMC methods “typically require substantial tuning and long runtimes.” However, this citation is not appropriate as Dai et al.’s paper titled “An Invitation to Sequential Monte Carlo Samplers” advocates for wider adoption of sequential Monte Carlo methods in statistics emphasizing their ability to leverage parallel processing resources among other potential benefits. This reference actually undermines rather than supports the authors’ critique of Monte Carlo methods.

If I understand correctly, the proposed method appears to rely on annealed Langevin dynamics for initialization, as shown in Figure 1 where Langevin already discovers both modes. Given that the paper dismisses classical methods as inefficient, I find it somewhat contradictory that TTD requires a classical MCMC method to provide a near-solution before it can even begin? How would TTD perform without this classical initialization, and doesn't this dependency suggest that classical methods are actually doing most of the work? It is not clear to me if the baseline methods in the numerical studies were also allowed the benefit of this initialization strategy or if they started from scratch. 

The methodology writeup begins with an SDE in equation (2) that maps samples from the target distribution $p_{target}$ to samples from a known reference/prior distribution. However, in the problem setting where $\rho_{target}$ is known and samples from $p_{target}$ are not provided, this forward process cannot actually be simulated. This forward SDE appears to serve no computational purpose in the subsequent algorithm and seems to exist solely to maintain alignment with the diffusion model framework of Ho et al. and Song et al. that addresses a different problem statement. 

The paper appears to force-fit the diffusion model-based generative modeling framework of Song et al. The setting in Song et al. involving forward and time-reversed SDEs makes sense when we have access to samples drawn from the unknown target distribution. Here, since $\rho_{target}$ is given, the score function of the target distribution is also known. The elaborate construction through HJB equations and BSDEs essentially reduces to solving for a transport map that pushes samples from a known reference distribution to samples from the target distribution. Therefore, this problem can be approached more directly without the detours that the paper takes. 

I have several questions and concerns about the numerical studies - please see next section.

### Questions
- Given that the proposed method essentially applies tensor train decomposition to a transport problem, what specific advantages does the diffusion model framing provide over directly solving the optimal transport or Schrödinger bridge problem?
- Could the authors clarify the distinction between their "extended Tensor Train (xTT)" formulation and the standard functional tensor train (FTT) construction? Unless I am mistaken, the proposed construction appears identical to the  FTT format (Oseledets, 2013) that is widely used in the literature. If the proposed xTT introduces novel aspects beyond the established FTT framework, it would be helpful to explicitly highlight these contributions.
- The evaluation focuses solely on a synthetic multiwell test function with d=10 and d=50. Why aren't standard benchmarks from the literature included, such as those used in Zhang & Chen (2022), or the extensive test suites from the normalizing flow literature?
- The experiments compare only against DIS (Berner et al., 2024) and PIS (Zhang & Chen, 2022). Why are there no comparisons against well-established workhorses such as NUTS, HMC, pSMC, which have been developed to exactly address this problem class?
- The proposed method includes an initial regression of $\log \rho_{target}$ as mentioned in Remark 2.5. How sensitive is the overall algorithm performance to the quality of this initial regression? 
- Alternating least squares is known to suffer from slow convergence. How do the authors overcome this limitation in practice, and have they tested the method on problems beyond the relatively simple multiwell distributions? Was the initialization step motivated by the  poor convergence of ALS?
- Do all methods use the same initialization strategy? Specifically, does TTD's use of annealed Langevin initialization (which Figure 1 shows already discovers all modes) provide an unfair advantage over DIS and PIS, which appear to start from scratch? For fair comparison, either all methods should start from the same Langevin-initialized point, or all should start from a simple reference distribution.
- Could the authors provide detailed specifications of the hyperparameters and implementation choices used for the baseline methods, particularly DIS? What do the N values (50, 200, 400) represent for DIS? Additionally, were the hyperparameters for DIS tuned for these specific problems, or were default values from the original paper used? Given that Berner et al. report strong performance on similar problems with different settings, understanding whether the baselines received comparable optimization effort to the proposed method would help readers assess the fairness of the comparisons.
- Could the authors provide a detailed computational cost breakdown, specifically the number of evaluations of $\rho_{target}$  versus other computational steps? The plots show runtimes of up to 300 minutes for "two iterations", which I am unable to make sense of.  A breakdown showing number of evaluations of $\rho_{target}$, time spent in the initial regression/preprocessing (Remark 2.5), ALS iteration counts and costs, etc would help readers understand where the computational budget is being spent and whether the method would remain competitive for problems where density evaluation is expensive (e.g., problems requiring numerical integration or simulation).

### Soundness
2

### Presentation
2

### Contribution
2
