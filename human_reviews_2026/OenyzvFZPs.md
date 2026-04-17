# Adaptive Mamba Neural Operators

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Accurately solving partial differential equations (PDEs) on arbitrary geometries and a variety of meshes is an important task in science and engineering applications. In this paper, we propose Adaptive Mamba Neural Operators (AMO), which integrates reproducing kernels for state-space models (SSMs) rather than the kernel integral formulation of SSMs. This is achieved by constructing Takenaka-Malmquist systems for the PDEs. AMO offers new representations that align well with the adaptive Fourier decomposition (AFD) theory and can approximate the solution manifold of PDEs on a wide range of geometries and meshes. In several challenging benchmark PDE problems in the fields of fluid physics, solid physics, and finance on point clouds, structured meshes, regular grids, and irregular domains, AMO consistently outperforms state-of-the-art solvers in terms of relative $L^2$ error. Overall, this work presents a new paradigm for designing explainable neural operator frameworks. The code is available at https://github.com/checlams/AMO.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes Adaptive Fourier Mamba Operators (AFMO), a novel neural operator architecture that integrates:

Takenaka–Malmquist (TM) systems and Adaptive Fourier Decomposition (AFD) theory,

with state-space models (SSMs) in the frequency domain.

The key idea is to parameterize the SSM transfer function using adaptive poles learned by a small MLP, allowing state-free inference and orthogonal rational basis construction directly in the spectral domain.
AFMO is designed to combine orthogonality, adaptivity, and linear-time efficiency, aiming to outperform LaMO (Latent Mamba Operator) and FNO-family models on irregular geometries and singular PDEs.

### Strengths
To me it feels like blending method A to method B, which is not innovative enough

### Weaknesses
Empirical scope.

All benchmarks are 2D or low-dimensional. It would strengthen the paper to include a 3D PDE (e.g., Navier–Stokes 3D or Poisson 3D) or demonstrate scalability across resolution levels.

No robustness analysis to noise, parameter shifts, or out-of-distribution geometries.

Interpretability of RKHS mapping.
The “mapping operator R” from latent tokens to an RKHS is implemented as an MLP without clear physical meaning. What does the learned RKHS correspond to in PDE solution space? Some visualization of learned poles or spectral responses would improve interpretability.

Justification of using SSM is not grounded

### Questions
Please provide more motivations and ablation studies

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The AFMO paper presents an architecture that achieves state of the art results in solving partial differential equations (PDEs) on arbitrary geometries and diverse mesh types using neural operators. The authors propose Adaptive Fourier Mamba Operators (AFMO), which integrates Takenaka-Malmquist (TM) systems for constructing reproducing kernels within SSMs. The architecture follows a structured pipeline: a lifting operator compresses physical tokens via cross-attention, followed by mapping to a reproducing kernel Hilbert space (RKHS), then N processing blocks each containing a TM layer and bidirectional SSM, with aggregation layers incorporating skip connections, and finally a projection operator to output space. The key innovation lies in TM layers that construct orthogonal bases from adaptively learned poles, while SSMs operate in the frequency domain with transfer functions. The empirical evaluation demonstrates substantial improvements across diverse benchmarks. Compared against 11 baseline methods including FNO, LaMO, Transolver, and ONO on standard tasks spanning regular grids, irregular geometries, and problems with singularities. AFMO exhibits superior computational efficiency with approximately 46-51\% training time reduction and is 2.5$\times$ lighter than LaMO (ICML 2025).

### Strengths
1.  AFMO outperforms 11 baseline methods on all benchmarks with good margins. 
2. AFMO achieves better accuracy while being faster and more memory-efficient than competing methods
3. First work to explicitly incorporate TM systems into Mamba architecture with frequency-domain SSMs

### Weaknesses
1. A significant weakness lies in the limited conceptual novelty, as the core components are largely borrowed from existing work. Takenaka-Malmquist systems are well-established in signal processing (Qian 2010, 2012), while frequency-domain SSMs were recently explored by Parnichkun et al. (2024). The main contribution is engineering, integrating these established techniques into a neural operator framework. Although this combination is novel, the paper overclaims its innovation by positioning it as a fundamental advance. The connection to Adaptive Fourier Decomposition is somewhat obvious given the explicit use of TM bases, and the theoretical results presented largely follow from known AFD convergence properties rather than providing genuinely new mathematical insights. The work represents a competent integration effort, but lacks the conceptual depth claimed.

2. Some design choices appear empirically driven rather than theoretically motivated, contradicting claims of "fully guided by AFD theory". For example, bidirectional SSM (LaMO also uses this) seems to be empirically determined as AFD doesn't require bidirectionality. Another point is that aggregation by element wise product seems to be an arbitrary empirical choice, not theoretically grounded/motivated.

3. The experimental design, although it follows other published work in PDE solving, is weak. No confidence intervals, error bars and statistical significance testing is reported. While FNO was modified, AFNO is not reported. The modifications of FNO is also not reported (readers would expect to see that in an appendix). European option pricing is a 1D test case with no relevance to the irregular geometry arguments raised to motivate the state space model usage. Instead a 3D test case would have conviced readers better.

4. Crucial implementation and evaluation details are missing. I am listing a few here.
(i) How is the "small MLP" structured? What initialization? How are poles constrained to $|a_k| < 1$ during training?
(ii)  How are TM bases computed for poles near boundary? The required product terms seem numerically unstable.
(iii) Hyperparameter selection is not reported
(iv) For some test cases (such as pipe), the reported baseline numbers differ from those in the respective papers. The reason for this should be reported. What are the dimensions on which the evaluation is performed? Are these choices computationally driven?

5. The theoretical details in Appendix C raises a lot of concerns. 
(i) Theorem C.4 requires exact coefficient recovery. How is the approximation error in practice handled?
(ii) Theorem C.13 requires $s \in K_B$, but no analysis or arguments on when PDE satisfies this is provided.
(iii) Assumption C.1 about RKHS properties is strong and not verified for practical cases

### Questions
1. The claimed $O(N_s D) + O(NM \log M)$ complexity is confusing. Is $M$ constant? What about cases where M less than $N_s$ may not hold? What about adaptive cases where M must change with problem complexity? Do you include the TM basis construction overhead in the computational cost?

2. Can you report statistical significance of the results?

3. What are the evaluation conditions? Why are reported numbers slightly different from the respective papers?

4. How does memory scale with problem size during training vs. inference?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Adaptive Fourier Mamba Operator (AFMO), a novel neural operator architecture that combines Adaptive Fourier Decomposition (AFD) theory with the state-space efficiency of Mamba models. By leveraging Takenaka–Malmquist (TM) systems and reproducing kernels in the frequency domain, AFMO provides a mathematically interpretable, efficient, and accurate solver for PDEs across diverse geometries. The model demonstrates strong performance across physical and financial benchmarks, surpassing SOTA neural operators like FNO, Transolver, and LaMO in both accuracy and computational efficiency.

### Strengths
1) Introduces a mathematically grounded operator framework combining AFD theory and Mamba-based SSMs.

2) Provides strong empirical improvements across varied PDE benchmarks, including irregular and singular domains.

3) Demonstrates computational efficiency, achieving faster training and lower GPU memory than existing models.

### Weaknesses
1) Ablation studies could be expanded to include more operator baselines (e.g., Laplace Neural Operator).

2) Scalability to large 3D or time-dependent PDEs is not thoroughly evaluated.

3) The sensitivity of performance to adaptive pole selection and hyperparameters is not deeply analyzed. It would be valuable to visualize how adaptive poles evolve across layers for different PDEs — this could provide strong insights into how the model captures changing dynamics.

4) While the theoretical motivation is strong, a more intuitive discussion of adaptive pole behavior and dynamics would make the approach easier to interpret for a broader audience. Additionally, as shown in the appendix, the LaMO deviation in the propagation of high-frequency perturbations can also demonstrate the performance of AFMO on those datasets.

### Questions
1) How sensitive is the model’s performance to the number and spatial distribution of adaptive poles in the TM system? How do you determine the optimal number of poles for a given problem?

2) Can AFMO be extended to time-dependent PDEs or coupled multi-physics systems? For example, in the Navier–Stokes equations, how do the adaptive poles evolve over time, and can this dynamic behavior be visualized?

3) How does AFMO handle noisy or incomplete boundary conditions, especially when applied to real-world experimental or observational data?

4) Would incorporating spectral regularization or physics-based constraints further stabilize training or improve interpretability?

5) How does the model perform on very high-resolution 3D geometries, and what are the associated computational trade-offs? Have you considered performing individual ablation studies on the TM and SSM layers to quantify each component’s contribution?

6) It would also be interesting to evaluate AFMO on the CARS dataset, as done in Transolver, to provide a more direct comparison of generalization across complex flow scenarios.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the Adaptive Fourier Mamba Operator (AFMO), which is a neural operator that integrates Takenaka-Malmquist systems with state space models for solving PDEs on irregular geometries. The core novelty is in parameterizing SSM transfer functions using adaptive poles in a RKHS. AFMO constructs orthonormal bases from learned poles and proves that the output performs adaptive Fourier decomposition.  Experiments show performance gains on irregular geometries.

### Strengths
1. Strong theoretical grounding with explicit connection to AFD theory with convergence guarantees. 

2. Integrating TM systems into mamba SSMs via transfer functions is well motivated. 

3. Experiments are comprehensive with thorough ablations. 

4. The method is computationally and seems exaplanability.

### Weaknesses
1. Limited baseline comparisons in that recent methods, such as BENO, UNO, and new transolver variants. 

2.   Several gaps exist in theoretical formulations: Theorems assume s ∈ K_B (target in model space), but this is rarely true for real PDEs. What happens when this fails? Theorem C.6's bound includes Δ_pole(N) (suboptimality gap), but no analysis of when and why this is small. Convergence rates (Corollary C.7) require weak-ℓ^p decay of the coefficient. When does this hold for PDE solutions?

3. Illustrative examples seem cherry-picked. 

4. All experiments use relatively coarse grids. 

5. Computational complexity analysis is not convincing enough.

### Questions
1. When does s ∈ K_B hold for PDE solutions?  Characterize the class of PDEs where the convergence guarantees apply. 

2. Authors should provide architectural details of the small MLP predicting poles. You should show learned pole distributions for different problems. 

3. Resolution invariance experiments have to be conducted. 

4. Failure analysis of AFMO has to be described. 

5. How does AFMO scale to 3D problems? 

6. Please report time and memory as a function of grid size. 

7. How is orthogonality maintained during training?

8. How does AFMO compare to explicit AFD+preprocessing+standard neural operators? 

9. You claim LaMO has low-pass filtering bias, but doesn't your method also have similar frequency characteristics since its also has an SSM transfer function, which is also a rational function.

### Soundness
3

### Presentation
3

### Contribution
3
