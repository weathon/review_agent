# A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
By incorporating physical consistency as inductive bias, deep neural networks display increased generalization capabilities and data efficiency in learning nonlinear dynamic models. However, the complexity of these models generally increases with the system dimensionality, requiring larger datasets, more complex deep networks, and significant computational effort.
We propose a novel geometric network architecture to learn physically-consistent reduced-order dynamic parameters that accurately describe the original high-dimensional system behavior.
This is achieved by building on recent advances in model-order reduction and by adopting a Riemannian perspective to jointly learn a non-linear structure-preserving latent space and the associated low-dimensional dynamics.
Our approach enables accurate long-term predictions of the high-dimensional dynamics of rigid and deformable systems with increased data efficiency by inferring interpretable and physically-plausible reduced Lagrangian models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a reduced-order learning model with the preservation of the state geometry for high-dimensional Lagrangian systems.

### Strengths
The author proposes a dimensionality reduction framework with physical consistency and geometry preservation. The writing is clear and the experiment demonstrates the effectiveness of the model.

### Weaknesses
1.	In the Introduction, for readers unfamiliar with the high-dimensional Lagrangian system, the author should provide some references and emphasize their importance. The author is also suggested to provide some quantitative descriptions, e.g., the range of the dimensionality.

2.	Is preserving the geometry important between FOM and ROM? Can the author summarize what the benefits are of preserving the structure in physical systems and control? 

3.	What’s the difference between Eq. (3) and Eq. (4)? For example, in Eq. (3), you state $X|\_{\gamma(t)}\in \mathcal{T}\_{\gamma(t)}\mathcal{T}\mathcal{Q} $. In Eq. (4), however, the set is $\mathcal{T}_{\gamma(t)}\mathcal{Q}$. Why do smooth trajectory lie in $\mathcal{Q}$ rather than $\mathcal{T}\mathcal{Q}$, where is $\dot{q}$?

4.	What is $V(q;\theta_V)$ in Line 162? Since there are massive notations, the author is advised to complete the proofreading. 

5.	It’s unclear of the transition between Section 2.3 and 2.4. For example, to fix what problems in Section 2.3, should we introduce LNN in Section 2.4?

6.	The advantage of the proposed SPD network should be better illustrated. The authors claim that their model makes use of Riemannian geometry. However, it seems that other methods can also maintain this constraint. For example, Cholesky decomposition can also make sure the output belongs to SPD space. 

7.	Notations in Eq. (8) are unclear to the reader. What are $l$, $h$, $x$, and PT? Can the author align them with the previous notations? 

8.	It seems weird to state that computing the second-order derivative in Eq. (13) is expensive, but computing the integration of multi-step in Eq. (14) is cheap.  

9.	The legends in Fig. 2 are hard to align with the curves. Please refine this figure. 

10.	Why is the error in Table 1 very high? I am not sure if this is a validated result.

### Questions
Q1. Please refine the notation system in the paper. You may remove some unnecessary notations and descriptions. See my Weakness points 3, 4, and 7.

Q2. The background and design motivation should be emphasized for readers unfamiliar with Lagrangian systems. See my Weakness points 1, 2.

Q3. The writing logic should be improved. See my Weakness points 5, 6.

Q4. The computational cost should be illustrated and reported. See my Weakness point 8. 

Q5. The result presentation can be improved. See my Weakness points 9 and 10.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a geometric network architecture that learns reduced-order Lagrangian dynamics for high-dimensional systems, using a Riemannian approach to jointly learn a structure-preserving latent space and the associated low-dimensional dynamics. This method efficiently predicts long-term dynamics for rigid and deformable systems while improving interpretability based on physics.

### Strengths
The strengths are listed as followed
- Utilizes geometry and physics as inductive biases, leading to a physically consistent model to improve the interpretability. 
- Effectively learns reduced-order dynamics for high-dimensional rigid and deformable systems with accurate long-term predictions.
- Joint optimization of latent space and reduced dynamic parameters enhances performance, particularly in low-data regimes.
- Incorporates intrinsic geometry of the problem, aiding in the preservation of energy and interpretability of the model.
- Outperforms standard autoencoders by employing a constrained auto-encoder trained with Riemannian optimization.

### Weaknesses
### Contribution and Originality

- I question the novelty of this paper. The proposed approach closely resembles existing methods described in [1, 2]. Additionally, the authors have not cited reference [1], which appears to be highly relevant to their work. 

- The contribution of this paper is overstated, and the true contribution appears to be quite incremental. Based on my understanding, the theoretical foundation relies heavily on the published work in [2], with many of the equations directly derived from [2]. There seem to be no novel contributions to the methods or framework, as the use of an autoencoder for reduced-order modeling was already proposed in [2]. This paper primarily appears to be an implementation of the original idea from [2].

- For the learning framework, this paper replaces the Galerkin method (subspace method) with an autoencoder structure for learning Lagrangian dynamics. However, the autoencoder representation is fundamentally similar to the Galerkin (subspace) method in its underlying principles. The primary distinction from reference [1] lies in the representation of the mass-inertia matrix as a symmetric positive definite (SPD) manifold within the configuration space, whereas the loss functions and implementation details remain largely consistent with those in [1]. The authors should clearly elucidate the key differences from prior work and provide a justification for why reference [1] was not cited.


### Comparison

- There is no comparison with other existing algorithms [1, 3]. The authors claim scability of its algorithm, but the current work only tests up to 600 DoFs using deep learning. In contrast, the existing work in [1] conducted experiments with 259,744 DoFs, even using Matlab. This raises questions about the claimed scability and performance improvements.

### Computational Complexity

- The computational complexity is significantly higher compared to [1], mainly due to the learning of the SPD manifold. It does not seem necessary to optimize such a complex manifold for high DoF cases, as the constant symmetric positive-definite surrogate used in [1] performs adequately in a complex scenario. 


[1]. Sharma, Harsh, and Boris Kramer. "Preserving Lagrangian structure in data-driven reduced-order modeling of large-scale dynamical systems." Physica D: Nonlinear Phenomena 462 (2024): 134128.

[2] Buchfink, Patrick, et al. "Model reduction on manifolds: A differential geometric framework." Physica D: Nonlinear Phenomena 468 (2024): 134299.

[3] Harsh Sharma, Hongliang Mu, Patrick Buchfink, Rudy Geelen, Silke Glas, and Boris Kramer. Sym- plectic model reduction of Hamiltonian systems using data-driven quadratic manifolds. Computer Methods in Applied Mechanics and Engineering, 417:116402, 2023. doi: 10.1016/j.cma.2023. 116402.

### Questions
see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents an idea of a neural network to learn the dynamics of a system using Reduced Order Model (ROM). ROM is achieved by mapping the state-space dimensions into a lower dimension and learn the dynamics in the reduced lower dimensional space. The authors proposes a neural network that learns the mass matrix and the potential function in the lower dimensional, and compute the acceleration based on Lagrangian principle. The model is trained in 2 ways: (1) learning the acceleration, and (2) learning the trajectory (and velocity) in a roll-out dynamics. The results show significantly better results than Lagrangian Neural Networks.

### Strengths
1. The paper presents experimental results on increasing complexity of the case
2. It shows better results compared to Lagrangian Neural Network

### Weaknesses
Major:
1. It is unclear where the novelty of the idea lies and what advantage it brings with respect to the previous works. The use of auto-encoder to reduce the dimensionality of the system is not a new idea. In fact, several papers in the same field like Hamiltonian Neural Network, Lagrangian Neural Network also used auto-encoder to learn dynamics of systems from images. If I understand it correctly, the novelty of the idea lies in a neural network learning both the mass matrix and potential for the Lagrange term. However, it is unclear what benefits does this new formulation of Lagrange term has in comparison with, for example, DeLan and LNN.
2. Reducing the dynamics to lower order dimension does not mean lower computational demand. It is possible, for example, that the lower dimensional dynamics has stronger stiffness than the original dimension. The paper does not present any results comparing the wall-clock time of solving ODE in the original dimension vs solving learned ODE in the lower dimension using the same hardware (e.g., CPU and GPU) to be convincing that their method really achieve computational reduction.
3. Lack of comparisons with other architectures. It seems like the authors only comparing the experiment with LNN. There are a lot of other works in this area that the authors can compare their work with. For example, Hamiltonian Neural Networks (https://arxiv.org/abs/1906.01563), dissipative HNN (https://arxiv.org/abs/2201.10085), dissipative SymOden (https://arxiv.org/abs/2002.08860), Constant of motion network (https://arxiv.org/abs/2208.10387), etc.

Minor comments:
1. Please avoid using the unscientific notation like $1e^{-5}$. Instead, please use notations like $1\times 10^{-5}$.

### Questions
1. The idea of using Lagrangian principle is usually made to conserve the energy. However, some of the case (e.g., the cloth) does not seem to conserve the energy. How can the authors justify the use of Lagrangian principle in such cases?
2. The error in Table 1 seems to be very high (it reaches $1\times 10^8$). Why can this happen? Normally, a well-setup neural network train should be normalized so the error starts around 1. The error reaches $10^8$ seems to indicate that the error is not normalized. Does this mean the training is not normalized?
3. (Related to weakness #2) Computational complexity in solving the ODE of a system does not only depend on how many dimension it is, but also how stiff the ODE is. Is there any effort in reducing the stiffness of the equation to ensure lower computational demand?
4. How is the roll-out of the NN done? What kind of ODE solver is used? I found in the appendix that the simulation is done using RK4. Is the same method used to roll-out the NN for the ODE loss?
5. Why the number of rolled out time steps $H=8$ quite small in computing the ODE loss? What is preventing it from using higher number of time steps?
6. How was the time step $\Delta t=10^{-3}$ chosen? How would choosing larger time step affect the roll-out prediction error?

### Soundness
2

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
4

### Summary
This paper investigates the problem of efficiently learning high-dimensional system dynamics from trajectory data. It simultaneously learns a reduced-order model using constrained AE and a dynamics model in the latent space that satisfies Euler-Lagrange equations. Preserving this Lagrangian structure imposes an inductive bias in the latent space to increase the generalization capability of the learned model. The authors proposed a new SPD network architecture to parameterize the mass-inertia matrix in LNN and leveraged Riemannian optimization methods to train the model considering the parameters' geometry.

### Strengths
The overall structure of RO-LNN is visualized well in Fig. 1, and the authors put great effort into explaining their architectures and learning algorithms that contain lots of equations in the limited space.

### Weaknesses
It seems like the overall idea of learning ROM using constrained AE that preserves the Lagrangian structure is already presented in [Buchfink et al.] cited in the paper. However, this is not clearly pointed out in the paper, and the authors' claims in the introduction and their explanation of the methodology in Section 3&4 somehow mislead the readers to think that the idea is also the main contribution of this paper. This reviewer thinks this paper's contribution on top of the prior work is proposing a new SPD network architecture for parameterizing the mass-inertia matrix and using Riemannian optimization in training to consider the parameters' geometry.

This identification of the contribution should be explained clearly, and the performance of RO-LNN should be compared to the prior works in the experiments.

There are no experimental results that show the benefit of preserving the Lagrangian structure in the latent space. Adding a baseline that learns the model without LNN would help to understand this benefit.

For the proposed architecture of the SPD network in section 3.1, the role of the SPD layers is not explained well, and the experiments show better results without those layers. Justification of having that component is required.

While using Riemannian optimization methods in training RO-LNN is one of the main contributions of this paper, the process is not explained well in Section 3.2.

For eq (13), there is no description of how to compute $\ddot{\check{q}}\_i$. Also, it seems $\tilde{\ddot{q}}\_i = \tilde{\ddot{q}}\_{p,i}$ so that the third term of $\ell_\text{AE}$  should be disregarded, or it should be explained what $\tilde{\ddot{q}}\_i$ is.

In the experiments, the trajectories in fig.2,3 are too cluttered. It might be better to just plot the log-scale difference between the true and predicted trajectories. Also, the plots mislead the readers into thinking that the learned models are used to predict the long-term trajectories while they were fed with ground-truth initial points every 0.025s.

For the cloth experiments, it seems a much shorter timestep size is used for full-horizon prediction (25 steps for 0.025s vs 2500 steps for 0.25s). Then, fig. 6 is not a fair comparison, and this should be pointed out clearly.

### Questions
Most of this reviewer's questions is dealt with in the weakness section above.

Additionally, how could we connect the satisfaction of the Euler-Lagrange equations of the latent dynamics to that of the original dynamics?

### Soundness
3

### Presentation
2

### Contribution
2
