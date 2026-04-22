# EFiGP: Eigen-Fourier Physics-Informed Gaussian Process for Inference of Dynamic Systems

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Parameter estimation and trajectory reconstruction for data-driven dynamical systems governed by ordinary differential equations (ODEs) are essential tasks in fields such as biology, engineering, and physics. These inverse problems -- estimating ODE parameters from observational data -- are particularly challenging when the data are noisy, sparse, and the dynamics are nonlinear. We propose the Eigen-Fourier Physics-Informed Gaussian Process (EFiGP), an algorithm that integrates Fourier transformation and eigen-decomposition into a physics-informed Gaussian Process framework. This approach eliminates the need for numerical integration, significantly enhancing computational efficiency and accuracy. Built on a principled Bayesian framework, EFiGP incorporates the ODE system through probabilistic conditioning, enforcing governing equations in the Fourier domain while truncating high-frequency terms to achieve denoising and computational savings. The use of eigen-decomposition further simplifies Gaussian Process covariance operations, enabling efficient recovery of trajectories and parameters even in dense-grid settings. We validate the practical effectiveness of EFiGP on three benchmark examples, demonstrating its potential for reliable and interpretable modeling of complex dynamical systems while addressing key challenges in trajectory recovery and computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The proposed Eigen-Fourier Physics-Informed Gaussian Process (EFiGP) framework demonstrates a certain degree of theoretical novelty by integrating Fourier transformation and eigen-decomposition into a physics-informed Gaussian process framework, aiming to enhance the efficiency and stability of ODE parameter inference and trajectory reconstruction.

### Strengths
1.Embedding the Fourier transform and eigen-decomposition into the GP framework is a non-trivial idea that strengthens the physical constraints from a frequency-domain perspective while reducing computational complexity.

2.The authors clearly identify the limitations of the MAGI framework—specifically, its high computational cost and poor convergence under dense discretization—and propose targeted solutions through frequency-domain and eigenspace dimensionality reduction. The problem statement is well articulated.

3.The experiments are conducted on three classical dynamical systems—FitzHugh-Nagumo, Lotka-Volterra, and Hes1—with clear comparative analyses. The results are quantitatively summarized in tables reporting RMSE and parameter errors, demonstrating that EFiGP achieves faster computation and more stable performance.

### Weaknesses
The paper mainly represents an engineering extension of the MAGI framework rather than a fundamental theoretical breakthrough. Although the idea of transferring ODE physical constraints into the Fourier domain is inspiring, its mathematical validity and convergence analysis are entirely absent. The paper provides no theoretical guarantees such as consistency, bias bounds, or truncation error analysis. Key derivations (e.g., Equations (9) and (11)) rely on statements like “by Lemma 2.2 we can obtain...” without rigorous justification of whether these transformations preserve the equivalence of the physical constraints. Moreover, there is no discussion of alignment with Bayesian inference theory (e.g., posterior consistency or marginal likelihood).

Although the proposed EFiGP framework shows computational advantages over MAGI, the experimental validation remains insufficient in both scope and depth.
All experiments are conducted on classical low-dimensional ODE benchmarks (FitzHugh–Nagumo, Lotka–Volterra, Hes1). These are toy systems with simple oscillatory dynamics, and the observation noise is i.i.d. Gaussian with uniform sampling—conditions that are relatively easy for GP-based methods. The paper does not test EFiGP on high-dimensional, chaotic, or real-world scientific datasets (e.g., climate models, fluid dynamics, or neural population dynamics), where scalability and robustness are critical. The comparison is restricted to MAGI and a simple numerical solver (Runge–Kutta). There are no experiments against more recent and competitive baselines such as FNO (Fourier Neural Operator, Li et al., ICLR 2021), AutoIP (Da Long et al., NeurIPS 2022), and Fenrir (Tronarp et al., ICML 2022), all of which address physics-informed kernel or operator learning in related contexts. The results mainly highlight faster computation, but accuracy improvements are marginal (Tables 2–5 show only small RMSE differences), and there is no systematic analysis of the trade-off between speed and inference accuracy as discretization increases.

The paper’s claims about computational complexity and scalability are also somewhat vague. The authors state that the complexity is reduced from O(n²) to a constant level (due to fixed j and l), but this is not theoretically rigorous—if j and l grow with system complexity, the computation may still scale linearly or quadratically. No detailed runtime environment or complexity–n scaling curves are provided; only static tables are shown.

The paper’s positioning and comparative analysis are also limited. It does not compare with recent works on physics-informed kernel learning (e.g., AutoIP) or FNO-style spectral operator methods. The Fourier domain truncation approach resembles the idea behind the Fourier Neural Operator, yet the paper does not clarify its relation or distinction from operator-learning frameworks. 

There are also issues in writing and academic presentation. Some notations are redundant or ambiguous (e.g., W_F^I, m are not clearly defined).

### Questions
1. How does the time complexity of EFiGP scale with the state dimension D and the number of discretization points n? Is the method still feasible for large-scale systems?
2. How are the two truncation hyperparameters (l for Fourier, j for eigen) selected automatically? Could their choices lead to underfitting?
3. How does the method perform on non-periodic systems ? 
4. Can you provide comparisons against Fenrir and AutoIP under identical settings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an algorithm called EFiGP (Eigen-Fourier Physics-Informed Gaussian Process), which is a Bayesian framework for parameters estimation and trajectory reconstruction of data-driven dynamical systems governed by ordinary differential equations (ODEs). The algorithm is mainly built on MAGI (Manifold-Constrained Gaussian Process Inference) but brings truncation with Fourier Transformation and Eigen-decomposition so that the computation time and cost are reduced, while computational efficiency and accuracy are enhanced. Simulations are done on three datasets.

### Strengths
- EFiGP’s most significant strength lies in its ability to bypass numerical integration during parameter estimation and trajectory inference without repeated ODE solving. 
 - The incorporation of eigen-decomposition and Fourier-domain truncation within a physics-informed Gaussian Process (GP) framework reduces computational complexity, resulting in near-constant runtime across increasing discretization levels. The proposed spectral and eigen-based reparameterization reduces computational complexity from O(n^2) to near-constant runtime with respect to discretization size, outperforming existing approaches such as MAGI. 
 - Moreover, EFiGP maintains convergence under dense discretization settings where the baseline MAGI algorithm fails to converge.

### Weaknesses
- Despite its efficiency gains, the claimed improvements in estimation accuracy over baseline MAGI remain modest. In particular, while EFiGP slightly outperforms MAGI for dense discretization (≥161 points), MAGI has comparable accuracy for sparser datasets, shown in Table 2 and Table 3, where EFiGP’s trajectory estimation accuracy improves only when the discretization becomes sufficiently dense. The algorithm did not deal with the challenge posed by data sparsity. The approach remains sensitive to data sparsity, and the paper’s results suggest that additional observations are required to mitigate degradation in parameter estimation performance.
 - Furthermore, parameter identifiability issues persist, especially for weakly identifiable parameters such as b in the FN system. Although EFiGP improves over MAGI in this respect, the mean of absolute error for b (0.176 at 41-point discretization) remains substantial compared to the true value of 0.2, and the authors do not propose any strategies to address this limitation.
 - The main evaluations are on three oscillatory systems (FN, LV, Hes1), with only a brief mention of a chaotic system in the supplement. Since the algorithm’s efficiency relies on Fourier-domain representations that are naturally suited to oscillatory behavior, broader testing on non-oscillatory or chaotic systems would be essential to substantiate claims of general applicability and robustness.

### Questions
For the reduce in computation cost, how much comes from eigen-decomposition and how much comes from Fourier truncation?

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
4

### Summary
This paper is about using Gaussian processes (GPs) for calibrating dynamical systems governed by ODEs in a Bayesian manner. Relying on the MAGI framework by Yang et al. (2021), physical information is included through a constraint on the residuals of the ODE for the GP solution. The contribution is to improve the scaling of the method by transposing the problem in Fourier space and relying on a Karhunen-Loeve decomposition of the constraint.

### Strengths
- the method improve greatly the computational efficiency over the default version
- ablation studies on the effect of truncation are provided

### Weaknesses
- spectral decomposition and Fourier features are typical tools for reducing the computational complexity. Some references may be worth adding here: such as Gauthier, B., & Pronzato, L. (2014). Spectral approximation of the IMSE criterion for optimal designs in kernel-based interpolation models. SIAM/ASA Journal on Uncertainty Quantification, 2(1), 805-825., or Mutny, M., & Krause, A. (2018). Efficient high dimensional bayesian optimization with additivity and quadrature fourier features. Advances in Neural Information Processing Systems, 31.
- the state of the art is missing some related works, such as Alvarez, M., Luengo, D., & Lawrence, N. D. (2009, April). Latent force models. In Artificial intelligence and statistics (pp. 9-16). PMLR.

### Questions
Can you give the values of W_I?
Can you provide a full pseudo-code for the proposed approach in appendix?
What would be the impact of training the GP hyperparameters including the physical information?

Minor points: 
P2: there is redundancy between the first and third paragraphs.
P3: As I understand it, the d-dimensional stochastic process is treated with independent GPs. Then it would be simpler to only put the 1d case here while the extension is direct.
P4L207: missing parenthesis in 2n-1x2n-1
Tables: please put best results in bold
L885: Kramer?
S3.3: what is the value of nu used here? Do you tune it?

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
3

### Summary
The paper proposes EFiGP (Eigen–Fourier Physics-Informed Gaussian
Process), an integration-free Bayesian framework for parameter
estimation and trajectory inference in ODE-based dynamical systems.
Building upon the MAGI framework, the authors introduce two
modifications:

*  enforcing the physics-informed constraint in the Fourier domain,
which allows truncation of high-frequency components for denoising
and efficiency; and

*  applying eigendecomposition truncation of the GP covariance to
reduce computational cost.

### Strengths
* The paper is well-writen and easy to follow
* The approach is simple yet effective, offering a clear
computational improvement over MAGI.
* Empirical results demonstrate faster inference and stable trajectory recovery.

### Weaknesses
* The weakness is that the experimental comparison is somewhat limited in scope. The paper would be strengthened by incorporating additional methods, e.g., non-GP-based methods, to better contextualize the proposed method’s advantages.

### Questions
* How scalable is the proposed method for high-dimensional ODE systems?

### Soundness
3

### Presentation
3

### Contribution
3
