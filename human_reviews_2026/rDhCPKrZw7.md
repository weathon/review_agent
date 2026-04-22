# Convergence Guarantees for Gradient-Based Training of Neural PDE Solvers: From Linear to Nonlinear PDEs

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
We present a unified convergence theory for gradient-based training of neural network methods for partial differential equations (PDEs), covering both physics-informed neural networks (PINNs) and the Deep Ritz method. For linear PDEs, we extend the neural tangent kernel (NTK) framework for PINNs to establish global convergence guarantees for a broad class of linear operators. For nonlinear PDEs, we prove convergence to critical points via the Łojasiewicz inequality under the random feature model, eliminating the need for strong over-parameterization and encompassing both gradient flow and implicit gradient descent dynamics. Our results further reveal that the random feature model exhibits an implicit regularization effect, preventing parameter divergence to infinity. Theoretical findings are corroborated by numerical experiments, providing new insights into the training dynamics and robustness of neural network PDE solvers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper establishes convergence guarantees for neural network-based PDE solvers, specifically PINNs and the Deep Ritz method. For linear PDEs, the authors extend the NTK framework beyond second-order cases to general admissible linear operators. For nonlinear PDEs, the authors use the Lojasiewicz inequality under a random feature model to prove convergence to critical points for both gradient flow and implicit gradient descent, without requiring strong overparameterization. The theoretical results are supported by numerical experiments on the Burgers, Allen-Cahn, and Fisher-KPP equations.

### Strengths
1. The paper provides a unified treatment covering both linear and nonlinear PDEs, and both PINNs and Deep Ritz methods under a single theoretical framework.
2. The proofs are detailed, and the use of Lojasiewicz inequality for nonlinear PDEs is well-motivated. Lemma 1's linear independence result for tanh based networks is a useful technical contribution
3. The numerical experiments on Burgers', Allen-Cahn, and Fisher-KPP equations demonstrate that the theory captures relevant aspects of optimization behavior. 
4. Theorem 3 and 4's identification of implicit regularization in random feature models is a valuable observation that could inform practical algorithm design

### Weaknesses
1. The scope is a bit limited for random feature model. The main results for nonlinear PDEs (Theorems 3, 4) only apply when inner-layer parameters are frozen. This is a severe restriction since most practical neural PDE solvers train all parameters. The paper acknowledges this in Section H.2 but doesn't provide a clear path forward. How restrictive is this assumption in practice? Can the authors provide empirical evidence comparing random feature models to fully-trained networks on the test problems?
2.  The experiments use implicit gradient descent with specific hyperparameters, but the theory covers gradient flow and generic implicit gradient descent. What convergence rate does Proposition 1 predict for the specific experimental settings? Do the observed convergence rates match theoretical predictions?
3. The convergence results assume the neural network can represent the solution well, but this isn't rigorously analyzed. For nonlinear PDEs with complex solutions (say Allen-Cahn with sharp interfaces), is the two-layer tanh network architecture sufficient?
4. Assumption 1 is needed for Theorem 3. While the paper claims this is "naturally satisfied for evolutionary PDEs," it's unclear how restrictive this is for general domains. What happens for domains without flat boundary portions?

### Questions
1. In Theorem 1, how tight is the overparameterization requirement? The bound grows rapidly with dimension and operator order. Are there known lower bounds showing this is necessary?
2. For Theorem 3, what is the typical Lojasiewicz exponent for the loss functions in your experiments? How does this affect the convergence rate in practice?
3. The paper proves convergence under "almost sure" initialization conditions. How robust are the results to initialization in practice? What happens with common initialization schemes like Xavier or He initialization?
4. Section 4.3 considers specific nonlinear operators in Eq 11. Can the interior coercivity analysis be extended to other important nonlinear PDEs, such as Navier-Stokes equations?
5. Can the convergence analysis framework be extended to the Neural Galerkin method or TENG (https://proceedings.mlr.press/v235/chen24ad.html)? Both use sequential-in-time optimization with projections onto neural network manifolds. Would the Lojasiewicz approach apply, and what additional assumptions would be needed?
6. How does the implicit regularization effect in random feature models manifest in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents convergence guarantees for 2 layer PINNs for approximating linear and non-linear PDEs using gradient-descent based methods, where for nonlinear PDEs, the hidden layer parameters are initialized from fixed distributions and left untrained. Experiments are conducted with the Burgers' equation to support theoretical claims.

### Strengths
(1) The manuscript is well written with a good motivation on the missing convergence properties for non-linear PDEs for PINNs. Especially the PDE setting considered in the paper is concise and on point.
(2) The manuscript touches on the concrete missing points in the convergence theory and error bounds for PINNs / deep Ritz.

### Weaknesses
(1) The paper heavily relies on the random-feature weight initialization, but does not provide a through literature review on the random features one can use. 
 - Fourier random feature approaches should be cited for data-agnostic cases, e.g. 
(1.1) Li, Zhu, Jean-Francois Ton, Dino Oglic, and Dino Sejdinovic. 2021. “Towards a Unified Analysis of Random Fourier Features.” Journal of Machine Learning Research 22 (108): 1–51.
 - Data-driven approaches for random features are also missing; e.g.
(1.2) Bolager, Erik L, Iryna Burak, Chinmay Datar, Qing Sun, and Felix Dietrich. 2023. “Sampling Weights of Deep Neural Networks.” Advances in Neural Information Processing Systems 36: 63075–116.

(2) Only last layer parameters are trained using iterative gradient-descent methods, which turns the networks into linear models (e.g., like finite element method approaches in classical scientific computing). This class is provably separated form general neural networks, and thus the results in the paper are more about random feature methods (with problem-agnostic weights), rather than PINNs trained wtih gradient descent or random features with data-driven weights. This is not discussed, and should be cited, e.g.:
(2.1) Wu, Lei, and Jihao Long. 2022. A Spectral-Based Analysis of the Separation between Two-Layer Neural Networks and Linear Methods. Journal of Machine Learning Research, vol. 23 (1).

For the claim of "establishing a systematic convergence theory for neural PDE solvers across both linear and nonlinear regimes", the random feature model is far too restrictive.

(3) The claim: 'using random-features as regularization' is not supported with additional experiments. For example, data-driven random features may be effective to sample more basis-functions in the shock regions (see "Datar, Chinmay, Taniya Kapoor, Abhishek Chandra, et al. 2024. “Solving Partial Differential Equations with Sampled Neural Networks.” arXiv:2405.20836. Preprint, arXiv, May 31, 2025.")

(4) The operator $\mathcal{L}$ is introduced as "a differential operator that may be linear or nonlinear" in line 119, but then redefined as linear operator only in line 170.

(5) l 2025 "similar techniques apply to a broader class of linear operators. Further details are omitted for brevity." This is a claim that should not just "omitted for brevity", especially if there is an extensive appendix to the paper already. Either the claim is supported by additional results (in the appendix, to not make the main paper longer) or the claim must be removed. Brevity alone is not enough to omit results; this also holds for Remark 5. The paper is already 37 pages long, it is not clear why results were simply omitted. If they can be found in the appendix, this must be stated in the main text.

(6) Some of the proofs are not precise enough. Example: l.422, "Fix any choice of inner-layer parameters" while the proof then requires specific choices of inner-layer parameters (l427). Similarly, for l328, "for randomly initialized inner parameters" does not specify the distribution. The longer versions of the proofs in the appendix should be referenced each time, to clarify that these are not the entire proofs.

### Questions
(1) When does the random feature model become restrictive during training (of the last layer), or for which problems?
(2) Could you elaborate on the random-features acting as regularization and explain the limitations of this? It is a very uncommon type of regularization. Usually, the training algorithm (Adam, SGD) itself acts as the "implicit regularization", while training the inner weights.
(3) Do the convergence bounds also hold with a relaxation when optimizing the hidden layer parameters as well for PINNs with the 2-layer architecture?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper provides a systematic analytical framework to establish convergence rates and guarantees, bridging the gap between existing NTK-based results for linear problems and new, though restricted, theoretical results for the challenging non-convex landscapes of nonlinear PDE solvers.

### Strengths
1. The novel application of the Łojasiewicz inequality to analyze the convergence of the non-convex loss functions encountered when solving nonlinear PDEs with neural networks.
2. The article provides a  technically solid extension of the NTK theory for linear PDEs.
3. This theoretical article presents mathematically rigorous convergence results that span a wide class of PDEs.

### Weaknesses
1. The core weakness is proving convergence for nonlinear problems only under the RFM, where only the last layer weights are trained.
2. The empirical validation of the derived theory is very limited, focusing only on a single 1-D nonlinear PDE and lacking higher-dimensional or real-world benchmarks.
3. For nonlinear PDEs, the proven convergence is only to a critical point of the loss function. A critical point is merely a state where the gradient is zero, which could be a local minimum.
4. The entire analysis depends on the properties of a two-layer network with a $\tanh$ activation function. While this is common in theoretical work, practical high-performing solvers often use deeper architectures and $\mathrm{ReLU}$ or $\mathrm{SiLU}$ activations.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

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
This paper studies the training dynamics and convergence of neural PDE solvers using NTK and Łojasiewicz inequality under the random-feature model for linear and nonlinear PDEs, respectively.

### Strengths
- The paper is overall well written and easy to follow, though a few key notations could be introduced earlier for clarity.  
- The work approaches the study of training dynamics and convergence from a rigorous theoretical perspective.

### Weaknesses
- Convergence of two-layer MLPs under the NTK regime is already well established. Theorem 1 merely applies this known result under an additional regularization on the differential operator. Moreover, the theorem could fail if the smallest eigenvalues are zero, and the authors does not discuss this possibility and even not officially define the Gram matrices before or within the theorem statement until next subsection.

- Sections 4.1–4.3 train only the second layer while freezing the first-layer weights, making the model linear in parameters. The resulting analysis essentially reduces to somewhat least-squares convergence under mild regularity (coercivity or PL-type) conditions, offering only incremental extensions of classical optimization theory rather than new insights into nonlinear or deep neural PDE solvers.

- The paper provides no experiments or numerical evidence verifying whether the theoretical assumptions hold in practice, leaving the practical relevance of the results unclear.

- The residual $r_\theta$ and its associated NTK-based Gram matrix, both central to Theorem 1, are not formally defined until Section 3.2.2, and only through an illustrative example. This late introduction makes the theoretical study difficult to follow.

### Questions
- In Theorem 1, you assume positive smallest eigenvalues $\lambda_0, \tilde{\lambda}_0 > 0$. How realistic is this assumption in practice, and can you provide numerical evidence or bounds showing these eigenvalues are indeed positive for typical PINN setups?

- Since the paper focuses on implicit gradient descent, have you explored whether the same convergence behavior holds for explicit gradient descent, which is what practitioners actually use?

### Soundness
2

### Presentation
2

### Contribution
1
