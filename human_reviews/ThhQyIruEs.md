# MGCFNN: A Neural MultiGrid Solver with Novel Fourier Neural Network for High Wave Number Helmholtz Equations

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Solving high wavenumber Helmholtz equations is notoriously challenging. Traditional solvers have yet to yield satisfactory results, and most neural network methods struggle to accurately solve cases with extremely high wavenumbers within heterogeneous media. This paper presents an advanced multigrid-hierarchical AI solver, tailored specifically for high wavenumber Helmholtz equations. We adapt the MGCNN architecture to align with the problem setting and incorporate a novel Fourier neural network (FNN) to match the characteristics of Helmholtz equations. FNN, mathematically akin to the convolutional neural network (CNN), enables faster propagation of source influence during the solve phase, making it particularly suitable for handling large size, high wavenumber problems. We conduct supervised learning tests against numerous neural operator learning methods to demonstrate the superior learning capabilities of our solvers. Additionally, we perform scalability tests using an unsupervised strategy to highlight our solvers' significant speedup over the most recent specialized AI solver and AI-enhanced traditional solver for high wavenumber Helmholtz equations. We also carry out an ablation study to underscore the effectiveness of the multigrid hierarchy and the benefits of introducing FNN. Notably, our solvers exhibit optimal convergence of $\mathcal{O}(k)$ up to $k \approx 2000$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MGCFNN, an AI-based solver tailored for high wavenumber Helmholtz equations, which are  challenging, especially in heterogeneous media. Building on the MGCNN framework, the authors incorporate a Fourier neural network (FNN) to improve solution efficiency and accuracy, leveraging FNN's CNN-like structure for faster propagation of source influence. Experimental results reveal MGCFNN's promising performance in supervised learning compared to existing neural operator methods, as well as notable scalability improvements in unsupervised settings. An ablation study underscores the value of the multigrid hierarchy and FNN, with the solver achieving optimal convergence up to $k\approx 2000$.

### Strengths
1. The framework presented in this work, with its division into setup and solve phases, is both well-structured and innovative.
2. The paper introduces a novel Fourier neural network (FNN) that enhances the solver’s ability to manage high wavenumber problems efficiently, particularly within heterogeneous media, providing a significant advancement over traditional CNN-based approaches.
3. The manuscript includes multiple numerical examples that demonstrate improved performance over existing neural operator architectures. Additionally, the results in Tables 3 and 5 are particularly compelling and highlight the model's effectiveness.

### Weaknesses
1. The paper would benefit from a more detailed discussion of its novelty relative to prior work on MGCNN, explicitly highlighting the advancements MGCFNN brings over previous frameworks.
2. While the motivation appears intuitively reasonable, adding solid mathematical insights to support the design of MGCFNN would strengthen the rationale and deepen the reader’s understanding of the architectural choices.
3. The explanation in Section 3.1 lacks clarity, particularly regarding its connection to Sections 3.2 and 3.3. Providing clearer linkages between these sections would help readers follow the logic and flow of the methodology.

### Questions
In addition to the points noted in the "Weaknesses" section, I have the following questions:
1. Tables 1 and 2 present results with $\omega=80\pi$. How do the results compare at $\omega=640\pi$, given that the primary goal of this work is to address high-wavenumber cases?
2. The numerical results in Table 4 do not show a significant difference between MGFCNN and MGFNN. Additional arguments and comparisons between these two methods would help clarify their respective advantages.

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
4

### Summary
This work presents an AI-driven PDE solver that specifically utilizes a Fourier neural network. By independently processing the coefficient and source term using a modified spectral convolution operation, it effectively handles high wavenumber Helmholtz equations.

### Strengths
* The modification of spectral convolution successfully handles full-mode information
* MGCFNN effectively handles the linearity of source terms
* It demonstrates excellent scalability with high-wavenumber problems

### Weaknesses
* Does MGCFNN work well only for high-frequency problems?
* Can MGCFNN be applied to other PDEs in operator learning?

### Questions
* In Figure 2, the visualization of the network architecture is difficult to understand

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
The paper introduces MGCFNN, a multigrid-hierarchical neural solver designed to address high-wavenumber Helmholtz equations, which are challenging for both traditional and AI solvers. The authors propose a Fourier Neural Network (FNN), an enhancement of spectral convolution operations, specifically to manage full-mode information and accelerate wave propagation during solving. The authors evaluate MGCFNN on both supervised and unsupervised tasks, comparing its performance against various neural operator learning methods.

### Strengths
1. The paper is well written and easy to follow
2. The paper presents extensive experimental validation, including scalability tests and comparisons with recent operator learning methods.
3. This is interesting idea for generalizing FNO to a more generic architecture.

### Weaknesses
1. Comparison is limited. The paper mainly focuses on a few high-wavenumber scenarios for Helmholtz equations. There are a broad range of equations worth to be evaluated. 
2. Hyperparameters: the paper lacks discussion on the choice and tuning cost of their hyperparameters, such as number of multigrid levels, number of Fourier modes, etc.

### Questions
1. Can you provide more comparison experiments especially with other types of equations?

2. Can you provide details on your hyperparameter selection and cost of tuning and selection?

### Soundness
3

### Presentation
3

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
Authors develop a hybrid iterative solver for Helmholtz equation. A proposed technique is a combination of multigrid-like architecture with a specially modified Fourier Neural Operator and a classical modified Richardson iteration. As I understand (although the description is not complete as I discuss below), authors apply a neural network to produce parameters of linear operators from PDE data and later apply these linear operators within a multigrid-like architecture. The resulting solver shows promising results on several high frequency problems.

### Strengths
1. The problem authors consider is well motivated, since it is known that classical iterative methods suitable for elliptic problems perform poorly on the Helmholtz equation.
2. A proposed method is a combination of multigrid and machine learning techniques, so, potentially, it scales well, is consistent and may be more robust than purely ML-based solvers.
3. Benchmarks authors propose cover especially challenging problems with high wavenumbers.

### Weaknesses
An article has several weaknesses which I briefly describe below:
1. The description of the method is not entirely clear.
2. Not all relevant baselines are covered.
3. The dataset is chosen in a very specific manner, making generalization hard to access.
I elaborate on these and other issues in the next section.

### Questions
1. The description of the method is not entirely clear.
   Authors provide descriptions of their approach in various places in the article. In my opinion these descriptions are confusing and incomplete. It is possible that other readers will find descriptions insufficient as well. Below is a list of questions on architectures authors used:
   1. Appendix E contains parameters of MGCFNN and MGFNN. Are these parameters for the setup phase network or for the solve phase network (or both)?
   2. Does the solve phase network have other parameters than the one produced by the setup phase network? If the answer is yes, what are those parameters?
   3. Is it the case that authors explicitly used residual only on the fine level, i.e., in equation (9) network implement linear operator B?
   4. Authors write about inefficiency of linear transformation in appendix D. From the number of operations authors provide it seems that this is a convolution with kernel size 1. Is that the case? Can the authors provide a number of floating point operations their architectures perform? This is highly relevant for comparison with classical iterative methods.
1. Not all relevant baselines are covered.
   1. Many well developed classical methods are available for the Helmholtz equation. It looks problematic that authors do not compare with any of them. For some reason all baselines are other ML approaches. I suggest the authors provide results for at least shifted-Laplace preconditioners and ideally with the optimized Schwartz method https://arxiv.org/abs/1610.02270. It is especially interesting to consider time needed to achieve desired accuracy since ML-based preconditioners are typically more computationally involved.
   2. Since authors consider only 2D cases, sparse direct solvers are also relevant baseline (see section 4 in https://arxiv.org/abs/1610.02270). I suggest providing solution time for modern CUDA implementation of sparse LU, preferable using freely available SuiteSparse solver https://people.engr.tamu.edu/davis/suitesparse.html.
   3. Sparse algebraic ML-based preconditioners (often GNN-based) form another class of relevant baselines, for example, https://arxiv.org/abs/2405.15557, https://arxiv.org/abs/2305.16432, etc
   4. Completely nonlinear methods based on flexible Krylov methods are also a valid solution approach https://arxiv.org/abs/2401.02016v2, https://arxiv.org/abs/2402.05598.
2. Dataset construction and generalization.
   First, the dataset authors used is detached from practically relevant applications of the Helmholtz equation. Second, a main dataset is formed from CIFAR-10, which has only 32x32 resolution which is interpolated to access performance on grids with higher resolutions. The troubling part is that fields, produced in such a way, have fixed frequency content, which is already known at the lowest resolution. Given that, it does not allow one to test whether the preprocessing network used to construct linear operators can generalize well to the problems of other resolutions. I suggest authors provide generalization tests where $k(x)$ contains higher frequencies on fine grids.
4. The fixed number of levels is not completely satisfactory.
   Authors used multigrid with a fixed number of layers. This means, with the increase in resolution the performance of the solver will degrade. I can suggest authors to empirically demonstrate that this is not the case. A possible experiment is to train architecture with two levels (i.e., the smallest possible) on a small grid and after that apply it to grids with increasing resolutions (much larger than the original grid). If one observes no degradation, the fixed number of levels is not a problem.
5. Is a comparison with NO fair?
   Neural operators have only access to input-output pairs, while the method in the current article explicitly uses discretisation. I think it is appropriate to explicitly discuss this distinction.

### Soundness
2

### Presentation
2

### Contribution
2
