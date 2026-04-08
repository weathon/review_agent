## Human Reviewer 1

### Summary
This paper proposes a variational family LBF-NPE that is specialized for neural posterior estimation, which is a simulation-based (i.e., likelihood-free) inference method of the Bayesian posterior. Specifically, the variational family is an exponential family where the natural parameter $\eta$ is given by a function $f_{\phi}(x)$, providing amortization across observations $x$, and the sufficient statistic is a vector of $K$ basis functions $s_{\psi}(z)$. This means the optimization depends only on the inner product between $f_{\phi}(x)$ and $s_{\psi}(z)$, leading to several properties, including affine gradients, convexity in $f$ and $s$, and the ability to fix the basis functions. Redundancy between $\phi$ and $\psi$ can be mitigated by a stereographic projection approach. The paper provides four experiments on four datasets, two are low dimensional toys and two are more realistic datasets related to astronomy, showing improvements in NLL and/or KL. The paper includes detailed explanations of the experiments in the appendix.

### Strengths
Overall this is a well-written and thorough paper that provides theoretical and empirical backing for the proposed method. Given recent interest in basis-expansion methods in VI (e.g., eigenVI), I think it provides a nice contribution. It’s interesting that LBF-NPE performs better than eigenVI with fewer basis functions. I appreciate the attempt at a middle-ground between, say, mixtures of Gaussians and normalizing flows. This gives the user a few levers (e.g., fixing the basis) to ease the complexity of the optimization problem. The experiments showed improvements over recent baselines.

### Weaknesses
- The paper aims to strike a balance between simplicity and expressivity in the variational family. However, since the amortization $f_\phi$ is a neural network in parameters $\phi$, one concern is that it doesn’t make the optimization problem that much easier (see my question below regarding the importance of convexity in the function $f$). 
- As the authors point out, sampling from the variational distribution in LBF-NPE is challenging.
- The results of the object detection experiment are buried in the appendix. As I understand it, the takeaway is that more basis functions reduces the KL to the chosen target but at a diminishing rate.

### Questions
- Regarding sections 4.1 and 4.2, can you comment on the importance of convexity in $f$ and $s$ given that they are potentially nonlinear functions of $\phi$ and $\psi$, respectively? For example, if you fix the basis functions, then the objective in Eq 6 may still be nonconvex in $\phi$ (e.g., if $f_\phi$ is a neural network), is that correct? If so, can you expand on how being convex in $f_\phi$  but not $\phi$ is still helpful? 
- In the Discussion & Limitations section, you write that regions of the latent space can be “zeroed-out”. I understand this to mean that some of the coefficients $\eta = f_\phi$ are zero after training. Can you comment on why this is useful and which experiments show this behavior? It’s interesting this can happen without anything encouraging sparsity.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces LBF-NPE, a neural posterior estimation method that parameterizes variational distributions using latent basis expansions. The method models log-densities as linear combinations of basis functions, forming an exponential family that balances expressivity with optimization stability. The authors propose several variants, including fixed basis functions and adaptive basis learning with stereographic projection for identifiability.

### Strengths
* Novel variational family that bridges the gap between simple distributions and complex black-box models.
* Strong theoretical foundation with convexity guarantees in certain settings.
* Comprehensive experiments across synthetic and real-world scientific problems.
* Outperforms MDNs and other baselines on multiple benchmarks.
* Effective handling of multimodal and complex posterior geometries, as demonstrated by the experiments.

### Weaknesses
* There seems to be no discussion of computational cost compared to baselines.
* If experiments are repeated several times with different random seeds, error bars should be reported in the plot and tables.

### Questions
* How to determine the number of samples and the number of basis $K$, and how sensitive is the performance to them?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper proposes a new variational family for Neural Posterior Estimation (NPE) that parameterizes the log-density of the variational distribution through basis expansions of latent variables. Instead of the traditional mixture of Gaussian or flow-based families, the authors parametrize the log density as a linear combination of latent basis functions, with coefficients modeled by an inference network and basis function network either fixed (e.g., B-splines, wavelets) or learned adaptively. The authors have claimed convexity and stable optimization in the fixed-basis case, and flexibility and expressivity comparable to normalizing flows in the adaptively learning case. The proposed method shows strong empirical results across both synthetic and real-world applications, consistently outperforming baselines such as MDN, RealNVP, and NSF on KL divergence and NLL.

### Strengths
1. The latent basis function formulation for the variational family is original, offering a middle ground between the simple Gaussian family and black-box flows.
2. Convexity and convergence properties are explicitly proven (App. B) and related to NTK theory, a rare level of rigor for NPE papers. The fixed-basis variant provides stable inference under certain conditions where global convexity is crucial.
3. Benchmarks cover synthetic, multimodal, and real astrophysical datasets, including redshift estimation (DC2 dataset). The proposed LBF-NPE achieves lower forward/reverse KL and NLL than baselines.

### Weaknesses
1. Most experiments are in ≤ 2D latent spaces. The high-dimensional example (App. E.4) is also synthetic (50-dimensional); stronger evidence on real higher-dimensional tasks would strengthen the claim of scalability.
2. While the impact of basis dimensionality and normalization is explored (App. E.1–E.3), ablations on the effect of stereographic projection and alternating optimization are somewhat fragmented.
3. Given that both the basis function $s_\psi$ and inference network $f_\phi$ are parameterized as deep neural networks, computational efficiency remains a concern for this algorithm.
4. Although Algorithm 1 provides a gradient estimator for the objective function, it lacks pseudocode for the entire algorithm. Including this would clarify the overall process.

### Questions
1. Could the authors elaborate on whether LBF-NPE can be extended to high-dimensional latent spaces (e.g., posteriors of Bayesian neural networks) without prohibitive integration cost? Given time constraints, you can illustrate this only using a small toy regression experiment.
2. How does the computational efficiency of the proposed LBF-NPE compare to other baselines? Can you provide inference time comparisons?
3. The current gradient estimator is biased but consistent. Have you quantified this bias, or explored control variates or reparameterization tricks to reduce it?
4. While Proposition 1 ensures convexity when either $s_\psi$ or $f_\phi$ is fixed, what guarantees or empirical evidence exist when both networks are trained jointly? Have the authors observed non-convex behavior or mode collapse in practice?
5. Can you provide quantitative results comparing fixed (B-spline/wavelet) versus adaptive bases in terms of convergence speed, stability, and overfitting? How do you decide whether to use B-splines, wavelets, or neural bases across different tasks? Is there an automatic basis selection mechanism?
6. The choice of the scaling factor $w$ is ad-hoc. Is it tuned per task, or could it be learned automatically? 
7. Section 7 and Appendix C mention inverse-transform and Langevin sampling. How efficient are these in high-dimensional cases, and have you compared them to Hamiltonian Monte Carlo or other samplers for posterior recovery?
8. Since each latent basis has an interpretable contribution to the posterior, can you comment on whether the learned bases correspond to meaningful latent modes (especially in the astrophysical tasks)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2