# Finite Element Operator Learning for Solving Parametric PDEs without Labeled Data

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Partial differential equations (PDEs) underlie our understanding and prediction of natural phenomena across numerous fields, including physics, engineering, and finance. However, solving parametric PDEs is a complex task that necessitates efficient numerical methods. In this paper, we propose a novel approach for solving parametric PDEs using a Finite Element Operator Network (FEONet). Our proposed method leverages the power of deep learning in conjunction with traditional numerical methods, specifically the finite element method, to solve parametric PDEs in the absence of any paired input-output training data. We demonstrate the effectiveness of our approach on several benchmark problems and show that it outperforms existing state-of-the-art methods in terms of accuracy, generalization, and computational flexibility. Our FEONet framework shows potential for application in various fields where PDEs play a crucial role in modeling complex domains with diverse boundary conditions and singular behavior. Furthermore, we provide theoretical convergence analysis to support our approach, utilizing finite element approximation in numerical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This article claims to propose a novel operator learning algorithm for the solution of PDEs by leveraging neural networks and the well-known finite element methods for numerical simulation of PDEs. The underlying idea is very simple. The PDE that is considered is a parametric PDE, with inputs such as forcing terms being completely described in terms of a (reshaped) *finite-dimensional* parameter vector $\omega$. The corresponding solution $u(x,\omega)$, with $x$ denoting the spatial coordinate is then approximated by a linear combination of the form Eqn (8) such that $u(x,\omega) \approx \sum_{k} \alpha_k(\omega) \phi_k(x)$, with $\alpha_k$ being neural networks and $\phi_k$ being finite-element basis functions. Then, the neural network is trained using a PINN-type loss function based on the residual of the underlying PDE. A Theorem is proved showing consistency of the method and some numerical experiments are presented to compare the method against DeepONet and Physics-informed DeepONet baselines.

### Strengths
- An interesting problem is addressed. State of the art operator learning algorithms are data intensive and reducing the amount of data required to train them can be advantageous. 

- Some theoretical results are provided.

- The algorithm is illustrated with numerical experiments.

### Weaknesses
- **Proposed Method is NOT operator learning**: As it has rapidly emerged in last 2-3 years, there can be some confusion about what exactly constitutes operator learning but there is a clear consensus in the community that, at the very minimum, an operator learning algorithm has to be able to evaluate arbitrary function-valued inputs and outputs (see Kovachki et al JMLR 2023 paper on the definition of neural operators, Section 2 and Table 1), plus some notion of discretization invariance or consistency (see also arXiv:2305.19913 for a more stringent definition). However, your method does not evaluate arbitrary function valued inputs, only those that are described by the parameters $\omega$. Let me provide a concrete example -- consider the Poisson equation benchmark from arXiv:2302.01178 (described in Eqn 4.1 with source-terms in Eqn 4.2. Your algorithm will automatically input the parameters $a_{ij}$ with $1 \leq i,j \leq K$ for its operation and then map it onto the neural network. Note that in arXiv:2302.01178, the training and in-distribution testing is performed with $K=16$ where as out-of-distribution testing is performed with $K=20$. Now, your algorithm cannot even evaluate the source Eqn (4.2) of arXiv:2302.01178 as the parameter vector $\omega$'s dimension as well as setting is changed. In other words, you are restricted to only consider parametrized functions with a fixed parameter domain rather than general $L^p$- functions that a genuine operator learning algorithm is supposed to process. Thus, your algorithm is not an operator learning algorithm or neural operator at all, to the best understanding of this reviewer and should not be viewed as such or compared to genuine neural operators such as FNO which can process any input-output function pairs.  


- **Super- and Sub-Resolution**: An essential element of operator learning is the ability of the model, trained on inputs and outputs at a certain grid resolution, to generalize well to other grid-resolutions. This *superresolution* and *subresolution* features of operator learning architectures are what distinguishes them from plain vanilla finite-dimensional neural networks, see Kovachki et al JMLR 2023 paper for an extensive discussion on this issue and  arXiv:2305.19913 for a state of the art take. It is unclear to this reviewer how the proposed FEOnet algorithm super resolves or sub-resolves and I find it unsatisfactory that the authors have not provided any discussion on this essential issue about operator learning. For instance, in the numerical experiments, something like Figure 2 right from  arXiv:2302.01178 should be shown with regards to FEOnet, see also  arXiv:2302.01178 Figure 24 (left) for the analogous results for the Poisson equation. Without super- and sub-resolution capabilities, an architecture is not really learning operators but just a particular finite-dimensional representation of it, further buttressing my contention in the above point. 

- **Context and Related Work**: As the above comments clearly bring out, the proposed algorithm cannot be viewed as an operator learning algorithm. So what is it related to ? In the opinion of this reviewer, it is more related to *Reduced Order Modeling* or *Model-order reduction* (see the excellent textbook of Quarteroni, Negri and Manzoni 2015 for a thorough introduction to this topic and the enormous amount of literature on it). MOR algorithms are also often restricted to a strict parametric setting and cannot be evaluated for arbitrary functional inputs. Moreover, using neural networks within the MOR framework has a well-established record, see for instance: *A deep learning approach to reduced order modelling of parameter dependent partial differential equations, 
N Franco, A Manzoni, P Zunino,  Mathematics of Computation 92 (340), 483-524* where similar ideas are investigated although a physics-informed loss was not considered. 

Moreover, even within the operator learning community, similar parametric representations have been used: see for instance Bhattacharya et al, SMAI JCM 7, 121-157, 2021 and more pertinently the NOMAD approach of Kissas et al, 2023 where the authors consider general functions but encode to a manifold described by the parameters. The authors should clearly contextualize their work in terms of these papers. 

- **Numerical results and Baselines**: The numerical benchmarks and baselines are far from adequate. Only one class of reaction-diffusion equation is considered and the authors take very weak baselines, particularly DeepONet. It is well known in the community that DeepONet is not a strong baseline and performs very poorly on numerical experiments, see for instance Table 1 of  arXiv:2302.01178 whereas DeepONet is consistently among the worst performing models. Similarly poor results are expected from Physics-informed DeepONets when the supervised version does poorly. To address these deficiencies, the authors should consider the following: 

   -- Add some modern MOR baselines for their methods. 
  --- Add a significantly more diverse set of experiments: for instance consider adding the Poisson Eqn benchmark from  arXiv:2302.01178 with sufficient number of frequencies say $K > 64$ to see how your method scales and consider addressing the limitation of being able to evaluate only for fixed parameter vectors. 
   --- Go beyond advection-diffusion by considering Stokes, Darcy (see FNO paper) and Helmholtz problems. 

The current set of experiments is not adequate.

### Questions
Please address the questions in the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The present manuscript proposes to leverage finite-element techniques to learn the solution to parametric PDEs using neural networks. Minimizing a classical Galerkin approximation, the method trains a neural network to predict the coefficients of a nodal FEM basis for a given PDE parameter, where the mesh and nodal basis generation is performed using FEniCS.
Moreover, building upon rich literature on FEM methods, corresponding theoretical guarantees are developed. Finally, numerical experiments on complex geometries show that the proposed approach can outperform supervised and unsupervised approaches based on DeepONets.

### Strengths
- The development of hybrid methods, combining existing FEM solvers with deep learning, is a promising research direction.
- On complex geometries, the method performs significantly better than existing physics-informed neural operators.

### Weaknesses
1) The stated contributions seem to "oversell" the method:
    - As discussed later, all physics-informed neural operators do not require any training data.
    - Most neural operators can deal with any form of PDE data (forcing, coefficients, boundary conditions, initial conditions). 
    - In contrast, the proposed method appears not to work on *any* form of PDE data but only on data given in a parametrized form, i.e., by a finite-dimensional parameter vector. In particular, one cannot use (discretizations of) arbitrary input functions, which is possible for, e.g., FNO. 

2) Further numerical results are needed:
    - The hyperparameters of the baselines seem not to have been optimized for the considered problems.
    - It would be good to present comparisons on problems that have been considered in the PIDeepONet or PINO papers since these methods have not been optimized for complex geometries. In this context, other approaches have been suggested; see, e.g., https://arxiv.org/pdf/2207.05209.pdf.
    - For the current problems, it would be more suitable to see comparisons against graph-based neural PDE solvers.
    - It is essential to compare runtimes of the considered methods.

3) Motivation of the work and comparisons with classical FEM methods:
    - It seems that the proposed approach is merely learning a surrogate model for solving the linear/linearized system of equations arising in FEM. It still requires carefully choosing basis functions and meshes and assembling stiffness matrices (i.e., in the specific case of the present work, it is heavily relying on FEniCS). While current operator learning methods can not yet achieve the same accuracies as specialized numerical solvers, they are more universal and do not need to be adapted to specific PDEs.
   - Considering training cost, what is the advantage of the proposed approach to just solving the linear system in (6) with a suitable solver? There is a single comparison of the runtime of FEONet and FEM in the appendix, but this seems to be a very critical point. Especially given that the achieved accuracy of FEONet seems to be orders of magnitude worse than the FEM solver. In the case of varying forcing functions, it seems that one could reuse the inverse of the stiffness matrix and just compute a single matrix-vector product to arrive at the solution (which could also be batched).
    - There should be more information in the main text on how the bilinear form $B$ is computed for varying PDE data.

### Questions
What is the advantage of minimizing (9) instead of solving the FEM system and directly regressing the optimal coefficients?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors proposed the Finite Element Operator Network to solve parametric PDEs. A neural network is taken to predict the FEM solution coefficient from the input parameters, such as forces, boundary conditions, diffusion coefficient, etc. The loss function is provided by the weak residual hence no data are needed. It is claimed that, compared to the current physics-informed neural operator methods, the proposed method can better handle boundary layer issues, and can provide better generalization capabilities.

### Strengths
- Inherited from FEM, the proposed model works for PDEs with complex geometry.
- Boundary conditions including both Dirichlet and Neumann types can be naturally satisfied.
- Because of the clear mechanism, some theoretical analysis on convergence can be obtained.

### Weaknesses
Major concerns:
- Limited novelty: PINN-style methods with a loss from the weak form of the PDE have been investigated in previous works. The neural network structure in this work is very basic. It is actually very straightforward to add the generalization capability to PINN-style methods. As the Related Works section has shown, there are also many previous works.
- The learned operator is fragile: it is sensitive to the geometry and meshing of the problem.
- Scalability issues: for a practical PDE problem, the number of elements will be tremendous.
- The experimental settings and the comparison baselines are relatively simple.

Minor issues:
- The citation format is mixed. I think it is because of the misuse of command \citet and \citep.

### Questions
None.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This manuscript introduced a Finite Element Operator Network (FEONet) for solving parametric PDEs. In particular, they leveraged a neural network to learn the nodal solution and train the network based on the variational loss constructed via the Garlerkin formulation. Overall, this is an interesting paper. However, there are some key issues which the authors need to address.

### Strengths
+ Integrating a neural network into the finite element formulation is interesting. The resulting model works well for solving low-dimensional parametric PDEs.

+ The results show efficacy of FEONet over two baseline models (e.g., DeepONet and PIDeepONet).

### Weaknesses
- The authors overly claimed the power of their approach and failed to compare their approach with other SOTA models. The baseline models considered in this paper are very limited.

- The experiments used to demonstrate the capability of the model are rather simple (mostly 1D problems). The authors should test the method on other complex systems, e.g., 2D/3D GS reaction-diffusion equations, wave propagation in an elastic media, 2D/3D material deformation with a nonlinear constitutive relationship, etc.

- The paper didn’t clearly mention how the BCs are handled. If the complete BCs are known (especially the Neumann BC), the Green’s theorem could be applied to Eq. (3) to impose the BC. 

- The forces, BCs and ICs are parameterized in simple forms governed by only a few parameters (e.g., $\boldsymbol{\omega}$). The generalization tests still fall into the same distribution of these parameters. A sounder generalization test on other parametric forms should be conducted (e.g., training based on one parametric form while testing for another different form).

### Questions
1. The authors did not discuss in detail how the integral loss function is evaluated (just simply mentioned by the Monte Carlo method). Why did not the authors use the Gauss quadrature approach for better accuracy and computational efficiency? 

2. The authors claimed “it outperforms existing state-of-the-art methods in terms of accuracy, generalization, and computational flexibility”. The two baselines (e.g., DeepONet and PIDeepONet) considered are not the SOTA models. There are many other models performing better than these two baselines, which the authors should consider to compare.

3. Like what has been mentioned in the weaknesses, the experiments used to demonstrate the capability of the model are rather simple (mostly 1D problems). The authors should test the method on other complex systems, e.g., 2D/3D GS reaction-diffusion equations, wave propagation in an elastic media, 2D/3D material deformation with a nonlinear constitutive relationship, etc.

4. The authors compared the computational time of FEONet with that of FEM in Appendix Section C.2. However, they did not mention the computing environment. Was the comparison done on the same hardware environment (e.g., the same CPU w/ or w/o parallel computing)? It seems that the inference speedup by FEONet is not significant compared with FEM. Considering the training time, the performance of FEONet in the context of computational might not be a great deal.

5. The paper didn’t clearly mention how the BCs are handled. If the complete BCs are known (especially the Neumann BC), the Green’s theorem could be applied to Eq. (3) to impose the BC. 

6. The forces, BCs and ICs are parameterized in simple forms governed by only a few parameters (e.g., $\boldsymbol{\omega}$). The generalization tests still fall into the same distribution of these parameters. A sounder generalization test on other parametric forms should be conducted (e.g., training based on one parametric form while testing for another different form).

7. Why the authors stated that the current method is not applicable to solving NS equations? I believe it is applicable.

I will consider to increase the score if these questions are well addressed.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
