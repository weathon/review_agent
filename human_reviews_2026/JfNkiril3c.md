# Learning Hamiltonian Dynamics at Scale:  A Differential-Geometric Approach

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
By embedding physical intuition, network architectures enforce fundamental properties, such as energy conservation laws, leading to plausible predictions. Yet, scaling these models to intrinsically high-dimensional systems remains a significant challenge. This paper introduces Geometric Reduced-order Hamiltonian Neural Network (RO-HNN), a novel physics-inspired neural network that combines the conservation laws of Hamiltonian mechanics with the scalability of model order reduction. RO-HNN is built on two core components: a novel geometrically-constrained symplectic autoencoder that learns a low-dimensional, structure-preserving symplectic submanifold, and a geometric Hamiltonian neural network that models the dynamics on the submanifold. Our experiments demonstrate that RO-HNN provides physically-consistent, stable, and generalizable predictions of complex high-dimensional dynamics, thereby effectively extending the scope of Hamiltonian neural networks to high-dimensional physical systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a dimension reduction that is faithful to symplectic geometry for Hamiltonian Neural Networks. Many methods have been proposed to efficiently learn high-dimensional dynamical systems by projecting them into a low-dimensional latent space with an autoencoder and then training HNNs or related models in that space (the original HNNs also do this). However, these reductions have not been guaranteed to be symplectically valid. The paper resolves this by proposing an autoencoder that is a symplectomorphism.

### Strengths
1. The proposed method prepares an autoencoder $(\phi_Q, \rho_Q)$ for the configuration, and then designs an autoencoder for the momentum using the pullback via this mapping. If the variable were velocity, one could use a pushforward, but because it is momentum, the pullback of the inverse map is required. This seems to be the key idea of the work.

2. The proposed framework incorporates many designs that respect intrinsic geometric structures, such as using an SPD network for the mass matrix and the dissipation matrix, and adopting a symplectic integrator for non-separable systems.

### Weaknesses
1. At line 342, observations are listed as position, velocity, and force, while the model is defined using momentum. Is this a typo for momentum? At line 366, the observations include momentum.

2. If the observations indeed include momentum, how would this be measured? For a point mass, one could multiply by mass, but in general dynamical systems, this is not sufficient, and a Legendre transform is required. Moreover, under the situation that mass can be observed, would the system still be meaningfully unknown? I am unsure about realistic scenarios where this framework would be useful. Speeding up simulation, as in Sec. 4.3, is conceivable, but then, the paper needs to clarify how much speedup is achieved and how large error is introduced, and it should compare against dimension reduction methods for known governing equations. The same goes for the force.

3. Similarly, the most direct comparison method seems to be Friedl et al. (2025), but this comparison is missing.

4. SPD network, constrained autoencoder, and the Strang-symplectic integrator are all existing components combined. The novelty appears somewhat limited. The definition of the symplectic autoencoder is very natural from the viewpoint of symplectic geometry.

5. Both the autoencoder design and the numerical integration place strong emphasis on symplectic geometry. However, the actual latent-space dynamics is something like a port-Hamiltonian system, with dissipation and external forces. While the experiments demonstrate the usefulness of the Strang-symplectic integrator, the discussion could be strengthened.

6. Prediction errors accumulate, so as the prediction time grows, errors can become extreme. A single divergent trajectory can drastically worsen the score. In fact, the standard deviation is about the same size as, or larger than, the average error. A three-times difference could be overturned. It would be better to include plots of error growth or adopt robust metrics.

Minor comments:

7. There is some confusion in the Introduction. PINNs use governing equations to define losses. Unless extended in combination with methods like SINDy, they are not intended for learning unknown dynamical systems, and their objective differs from HNNs and Neural Operators. Since this work is an extension of HNNs, discussing it separately from Neural Operators and PINNs might reduce confusion.

8. In Eq. (8), the left-hand side takes arguments, so the right-hand side should also take arguments.

9. Line 278

> Existing HNNs enforce the symmetric positive-definiteness of the inverse mass-inertia matrix via a Euclidean network encoding its Cholesky decomposition $L$

At least the original HNNs did not implement it that way. Perhaps this refers to Dissipative SymODEN? Additional explanation would help.

10. From the perspective of learning non-canonical Hamiltonian systems and dimension reduction with neural networks, in addition to neural symplectic forms, I recommend citing the following papers.

Jin et al., "Learning Poisson systems and trajectories of autonomous systems via Poisson neural networks," IEEE Transactions on Neural Networks and Learning Systems, 2023.

Sipka et al., "Direct Poisson neural networks: Learning non-symplectic mechanical systems," Journal of Physics A: Mathematical and Theoretical, 2023.

### Questions
1. The proposed autoencoder is implementable with vector-Jacobian products, so the computational overhead (compared to simple autoencoders) appears to be only a few times larger. Please clarify the computational overhead.

2. At line 306, is $\rho$ a typo for $\varphi$? Notations are also hard to read. Do subscripts $p$ indicate predicted quantities, while symbols without $p$ indicate encoder outputs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents RO-HNN (Geometric Reduced-order Hamiltonian Neural Network), which aims to extend Hamiltonian neural networks (HNNs) to high-dimensional systems. The model combines (1) a geometrically constrained symplectic autoencoder for structure-preserving dimensionality reduction and (2) a geometric HNN for learning reduced-order Hamiltonian dynamics in latent space. By enforcing symplecticity and biorthogonality constraints within the autoencoder and using SPD networks to parametrize the latent inertia and damping matrices, the authors claim to obtain stable and physically consistent dynamics. Experiments conducted on a series of physical models show improved reconstruction and long-term prediction compared to vanilla HNNs.

### Strengths
1. The paper systematically integrates geometric and symplectic constraints across model components, which guarantees the outperformance in preserving the physical laws during the learning process.

2. Empirical results demonstrate advantages of the proposed method over the baseline HNNs.

### Weaknesses
1. Lack of originality:The proposed RO-HNN essentially combines existing components. For example, using an autoencoder for latent-space projection of Hamiltonian systems is a well-established idea [R1],[R2]. The concept of symplectic or structure-preserving projections has been explored in SympNet [R3]. Modeling external forces and damping has already been discussed in [R4],[R5]. 
Thus, the method largely stitches together known techniques rather than introducing a new conceptual contribution.

2. The paper frames its “differential-geometric” perspective as new, but most theoretical results are restatements of standard symplectic and Riemannian geometry properties already available in textbooks or prior works [R6].

3. In the experiments, all datasets are clean and simulated, and the paper does not analyze performance under observation noise, though this is critical for real-world Hamiltonian systems (e.g., robotics, experimental physics).

4. Given that the framework aims to handle high-dimensional systems, it would be natural to compare against Koopman-based operator learning or neural operator baselines [R1],[R2],[R7]. The current paper mainly compares with vanilla HNN, which was been developed six years ago and is not the SOTA method in the related tasks.

__References__
[R1] Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9(1), 4950.

[R2] Zhang, J., Zhu, Q., & Lin, W. (2024). Learning hamiltonian neural koopman operator and simultaneously sustaining and discovering conservation laws. Physical Review Research, 6(1), L012031.

[R3] Jin, P., Zhang, Z., Zhu, A., Tang, Y., & Karniadakis, G. E. (2020). SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems. Neural Networks, 132, 166-179.

[R4] Zhong, Y. D., Dey, B., & Chakraborty, A. (2019). Symplectic ode-net: Learning hamiltonian dynamics with control. arXiv preprint arXiv:1909.12077.

[R5] Liu, Z., Wang, B., Meng, Q., Chen, W., Tegmark, M., & Liu, T. Y. (2021). Machine-learning nonconservative dynamics for new-physics detection. Physical Review E, 104(5), 055302.

[R6] Buchfink, P., Glas, S., & Haasdonk, B. (2023). Symplectic model reduction of Hamiltonian systems on nonlinear manifolds and approximation with weakly symplectic autoencoder. SIAM Journal on Scientific Computing, 45(2), A289-A311.

[R7] Xiong, W., Huang, X., Zhang, Z., Deng, R., Sun, P., & Tian, Y. (2024). Koopman neural operator as a mesh-free solver of non-linear partial differential equations. Journal of Computational Physics, 513, 113194.

### Questions
Q1: How does the proposed method compare to Koopman-based latent Hamiltonian learning [R1],[R2] in terms of reconstruction accuracy and scalability when the observational data is perturbed by noise?

Q2: Can the model handle partial observations (e.g., only $q$ is available)?

Q3: Since the method combines known components, what is the main conceptual novelty that differentiates it from prior structure-preserving model-reduction approaches?

Q4: Can the proposed framework handles irregular time series data?

Q5: In line 295, should it be 'constrain' instead of 'constraint'?

Q6: In line 350, the integrals of $\check{q}$ and $\check{p}$ should solve the IVP, what the initial value in this case? 

Q7: In line 423, it should be $\{6,10\}$ instead of $\{6.10\}$. 

Q8: In line 481, is the title of the 5th and 6th column in Table 4 right? Does the $\check{q}_p$ and $q$ have the same dimension?

Q9: In your experiments, why do you select the hyperparameter of latent dimension as $6,10$?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a geometric framework for learning Hamiltonian dynamics via a Reduced-Order Hamiltonian Neural Network (RO-HNN). The method combines a symplectic autoencoder—constructed through cotangent-lift mappings that enforce exact symplecticity and projection consistency—with a latent-space Hamiltonian network that parameterizes the inverse mass, damping, and potential functions using SPD-aware and MLP components. A Strang-type symplectic integrator is employed to evolve the latent dynamics, and the framework is trained end-to-end. Experiments cover systems up to 600 degrees of freedom, including pendula, vortex particles, and damped cloth simulations.

### Strengths
The paper introduces a cotangent-lift–based symplectic autoencoder that enforces exact symplecticity and projection consistency through hard constraints rather than soft penalties. This ensures that the learned encoder–decoder pair defines a true symplectic submanifold in the ambient phase space, maintaining the canonical two-form and the projection property.

The inverse mass matrix and damping operator in the latent Hamiltonian are parameterized using SPD-aware neural networks instead of Euclidean Cholesky factors. This geometrically consistent formulation improves numerical stability and preserves positive definiteness during long-time integration.

The method is demonstrated on large-scale systems with 15-, 90-, and 600-degree-of-freedom (DoF) dynamics—including multi-pendulum, vortex particle, and damped cloth systems—showing strong stability, energy preservation, and generalization across regimes.

The paper’s separation of geometry learning (symplectic autoencoder) and dynamics learning (latent-space Hamiltonian) is conceptually clean, interpretable, and aligns with the theoretical principles of symplectic reduction.

### Weaknesses
1. The definition in Equation (8) is not clear to me. It is stated that ( M = T^*Q ), but is every symplectic manifold assumed to possess such a cotangent-bundle structure? 

Moreover, in the definitions of ( \varphi_Q ) and ( \rho_Q ), the inputs appear to depend only on the configuration variable ( q ). What is the role of the momentum variable ( p ) in these mappings?  Given that the paper’s primary focus is on constructing a learnable, structure-preserving dimensionality-reduction network rather than developing new differential-geometric theory, I would strongly suggest abandoning the abstract manifold language and instead expressing the model directly in coordinate form. A clear coordinate-level presentation of how the encoder and decoder act on (q,p) would make the construction more transparent and easier to verify mathematically and computationally.

2. When dissipation and external forces are introduced, the system ceases to be Hamiltonian. The paper embeds damping and external forces via a learnable SPD operator, but it does not clarify the resulting geometric consistency. In particular, once dissipation is present, the underlying dynamics no longer preserve a symplectic structure. It is therefore unclear what the meaning or advantage of enforcing a symplectic reduction remains in this non-Hamiltonian setting.

If the proposed framework is intended to generalize to dissipative dynamics (e.g., via contact geometry or GENERIC formulations), this connection should be explicitly established. Otherwise, the rationale for retaining a symplectic encoder–decoder becomes questionable. Moreover, the numerical experiments involving dissipative systems do not appear to demonstrate a clear benefit of using a symplectic reduction compared with standard autoencoders or non-geometric models. The authors should clarify whether any theoretical or empirical advantage persists once energy dissipation breaks the canonical symplectic structure.

### Questions
pleasse see weakness above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this work, the authors propose a new neural network architecture for learning reduced-order Hamiltonian dynamics, consisting mainly of an dimensionality-reducing autoencoder with a specific parameterization that ensures a symplecticity property, and a Hamiltonian NN architecture on the lower-dimensional symplectic manifold. The method is demonstrated to perform well on three physical systems with degrees-of-freedom ranging from 15 to 600.

### Strengths
1. The "geometrically-constrained symplectic autoencoder" achieves theoretically-guaranteed symplecticity property through the novel architecture design while demonstrating good expressivity in the experiments, which is a commendable contribution in my opinion.

2. Other components of the RO-HNN model take good advantages of methods from prior literature (e.g. the SPD network for parameterizing the HNN, symplectic integrators, constrained autoencoders)

3. The experiments cover several Hamiltonian systems ranging from simple to higher-dimensional ones and also covering damped systems.

4. The paper is well-presented overall.

### Weaknesses
The method assumes that the system has a canonical symplectic form that is valid globally, as well as the existence of a low-dimensional latent Hamiltonian system that captures the high-dimensional dynamics well.

### Questions
In the experiments with the coupled pendulum system, is it the case that the learned latent system actually coincide with the 3 degrees of freedom of the underlying pendulum? For the more complex systems (particle vortex and cloth), do the latent dimensions have physical correspondences at all?

### Soundness
3

### Presentation
3

### Contribution
3
