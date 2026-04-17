# Locally Subspace-Informed Neural Operators for Efficient Multiscale PDE Solving

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
We propose GMsFEM-NO, a novel hybrid framework that combines the robustness of the Generalized Multiscale Finite Element Method (GMsFEM) with the computational speed of neural operators (NOs) to  create an efficient method for solving heterogeneous partial differential equations (PDEs). GMsFEM builds localized spectral basis functions on coarse grids, allowing it to capture important multiscale features and solve PDEs accurately with less computational effort. However, computing these basis functions is costly. While NOs offer a fast alternative by learning the solution operator directly from data, they can lack robustness. Our approach trains a NO to instantly predict the GMsFEM basis by using a novel subspace-informed loss that learns the entire relevant subspace, not just individual functions. This strategy significantly accelerates the costly offline stage of GMsFEM while retaining its foundation in rigorous numerical analysis, resulting in a solution that is both fast and reliable. On standard multiscale benchmarks—including a linear elliptic diffusion problem and the nonlinear, steady-state Richards equation—our GMsFEM-NO method achieves a reduction in solution error compared to standalone NOs and other hybrid methods. The framework demonstrates effective performance for both 2D and 3D problems. A key advantage is its discretization flexibility: the NO can be trained on a small computational grid and evaluated on a larger one with minimal loss of accuracy, ensuring easy scalability. Furthermore, the resulting solver remains independent of forcing terms, preserving the generalization capabilities of the original GMsFEM approach. Our results prove that combining NO with  GMsFEM creates a powerful new type of solver that is both fast and accurate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The manuscript introduces the GMsFEM-NO framework, which integrates the Generalized Multiscale Finite Element Method with NOs to improve the efficiency of existing NO frameworks when solving multiscale and high-contrast PDEs. Legacy GMsFEM requires eigen solvers, which are computationally expensive. GMsFEM-NO legacy eigen solvers with a NO trained to efficiently predict the local multiscale basis subspaces. A Subspace Alignment Loss, with a regularized variant, is also proposed to enhance global reception and ensure geometric and physical consistency. Experiments on 2D and 3D benchmarks show significant speedup over legacy GMsFEM with a comparable accuracy. Other technical merits include better generalization to OOD enforcing terms (v.s. NOs) and resolution invariance.

### Strengths
- **Mathematical grounds**: Links subspace alignment to Grassmann manifold distance, enhancing theoretical credibility. The new SAL and SAL-PR losses are well-motivated and mathematically tied to Grassmannian geometry. 
- **Comprehensive experiments**: Includes linear/nonlinear PDEs, 2D/3D cases, various forcing terms, and extensive comparisons vs. multiple baselines.
- **Performance boost**: 60x speedup v.s. legacy GMsFEM with comparable accuracy; better OOD generalization vs. standalone NOs; resolution invariance.

### Weaknesses
- **Limited scope**: The experiments are limited to elliptic problems. The study does not include time-dependent PDEs, limiting the demonstrated generality. FEMs and NOs are known to be robust to hard problems, e.g., Navier-Stokes. This limits the scope of the proposed framework. 
- **Assumption of regular grids**: Current implementation works only for regular grids due to the use of F-FNO; performance on irregular geometries or adaptive meshes remains unexplored.
- **Dependence on GMsFEM setup**: The framework still relies on a coarse grid and precomputed spectral problems for training data. It does not remove the expensive stage, but only accelerates it.

### Questions
- Could the same SAL losses be extended to time-dependent GMsFEM formulations or to non-elliptic PDEs?
- Can GMsFEM-NO handle non-rectangular or unstructured meshes if implemented with GNOs or other NO variants, e.g., GNOT, Transolver, etc.?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends the GMSFEM by incorporating a neural operator to accelerate the process. The work proposes a subspace alignment loss function to learn coherent subspaces, and introduces projection regularization terms to enforce consistent projections. Experiments compare the proposed loss function with the conventional loss function, GMSFEM-NO with GMSFEM and F-FFO, for evaluations.

### Strengths
The combination of GMSFEM with NO is an interesting idea.

The proposed SAL and PR losses can train the NO effectively.

### Weaknesses
The method is only tested with m moderate-scale data in rectangular domains. Is it able to solve larger-scale problems in non-rectangle domains?

The generalization ability of the learned NO is not evaluated sufficiently. The usability of the method is unclear.

Only zero Dirichlet boundary condition equations are tested? Can it solve Dirichlet boundary condition equations with other values?

### Questions
Are there failed cases that don't converge to the solution?

Is it able to extend the method to time-dependent equations?

Is it possible to further reduce the error if more basis functions are used? What's the limits of the error?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **GMsFEM-NO**, a hybrid framework that uses a neural operator to accelerate the **offline stage** of the Generalized Multiscale Finite Element Method (GMsFEM). Instead of solving many local eigenvalue problems to construct multiscale basis functions, the method trains an NO to map a heterogeneous coefficient field (\kappa(x)) directly to the **subspace** spanned by the GMsFEM bases. The key idea is a **Subspace Alignment Loss** ((\mathcal{L}_{\text{SAL}})) that aligns predicted and true multiscale subspaces on a Grassmannian, avoiding the sign/permutation ambiguity of individual eigenvectors. The approach preserves two important GMsFEM properties: (i) independence from the forcing term (f(x)), so it generalizes to OOD right-hand sides, and (ii) **resolution invariance** (train on coarse, use on fine). Experiments on 2D/3D high-contrast diffusion and steady-state Richards equations show **>60× speedup** in basis generation while matching GMsFEM-level accuracy.

### Strengths
1. **Targets the real bottleneck.** It identifies the most expensive part of GMsFEM (local spectral problems in the offline stage) and replaces it with an NO, which is a practical and impactful acceleration.
2. **Subspace-level supervision.** The proposed $(\mathcal{L}_{\text{SAL}})$ is technically meaningful: learning the *space* of local bases is more robust than learning each basis function, and it naturally resolves sign ambiguity; this is the main methodological contribution.
3. **Strong empirical claim: fast but still GMsFEM.** The method demonstrates >60× speedup in basis construction while retaining essentially the same $(L_2/H^1)$ errors as classical GMsFEM on multiscale benchmarks, which makes it attractive for large runs or parameter sweeps. 
4. **OOD and resolution generalization.** Because the solver structure of GMsFEM is kept, the approach remains independent of $(f(x))$ and supports train–coarse / test–fine usage, which standard NOs typically fail to do.

### Weaknesses
1. **Marginal gain from $(\mathcal{L}_{\text{SAL-PR}})$.** The paper introduces a more complex variant with projection regularization, but the improvement over plain $(\mathcal{L}_{\text{SAL}})$ is small (e.g. 1.82% → 1.72% on 250×250), and only on a single setting; the extra term is not clearly justified. 
2. **Systematically “better than” the target.** In several tables, GMsFEM-NO slightly outperforms the original GMsFEM it is approximating; calling this “statistical variation” is not entirely convincing because it appears repeatedly, suggesting either measurement noise, data leakage, or a smoother inductive bias from the NO. This needs a clearer explanation. 
3. **Baselines not fully up to date.** Most comparisons are to F-FNO and classical reduced-order/ POD-style methods; stronger modern neural operators for multiscale/high-contrast settings are missing, so it’s hard to tell where GMsFEM-NO sits against the current SOTA.

### Questions
1. Several tables show GMsFEM-NO outperforming the classical GMsFEM it is meant to approximate. Can you provide a more concrete explanation than “statistical variation” (e.g. averaging over multiple local domains, extra smoothing from the NO, or differences in quadrature/projection)?

2. For $(\mathcal{L}_{\text{SAL-PR}})$, how sensitive is the reported improvement to the choice and number of random test vectors 
$v^{i}$? 

Would the simpler $\mathcal{L}_{\text{SAL}}$ suffice in most practical cases?

3. Can you add comparisons to more recent neural operators (beyond F-FNO / POD-style baselines) to better justify that GMsFEM-NO is competitive not only with classical GMsFEM but also with current NO-based multiscale solvers?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles multiscale, high-contrast PDEs by replacing the expensive local eigenvalue problems of GMsFEM with a neural operator that predicts the local coarse spaces directly. Instead of regressing individual basis functions, the method learns the subspace they span and trains with a Grassmann-geometry–aware subspace alignment loss (with an optional projection regularizer). The learned spaces assemble into a restriction operator that preserves the classical GMsFEM solve, yielding GMsFEM-level accuracy while reducing offline basis construction by more than an order of magnitude. Experiments on diffusion and steady Richards equations in 2D and 3D show strong accuracy, data efficiency, right-hand-side robustness, and resolution transfer.

### Strengths
The conceptual shift from matching basis functions to matching their span is clear and well motivated, aligning the learning objective with what GMsFEM truly needs. The subspace alignment loss directly optimizes principal angles on the Grassmann manifold, removing sign and ordering ambiguities and leading to stable training. The framework preserves the classical coarse-solve pipeline, so accuracy remains comparable to GMsFEM while offline cost is drastically reduced; the empirical speedups are practically meaningful. Results demonstrate robustness to right-hand-side changes and cross-resolution transfer, where direct solution-predicting NNs typically struggle. The paper is clearly structured, with a clean train→assemble→solve story that is easy to reuse in other PDEs.

### Weaknesses
Theoretical guarantees connect subspace error to final solution error only implicitly; a formal upper bound from principal angles to 
$𝐻^1$error would strengthen soundness. Evaluation focuses on structured grids with Dirichlet boundaries and steady problems; non-structured meshes, mixed or Robin boundaries, and time-dependent or strongly nonlinear systems remain open. Comparisons omit graph- or mesh-based neural operators and learning-augmented multigrid with learned prolongation/restriction, which are natural baselines here. The model zoo created by per-region-type networks increases operational burden at scale; an ablation on parameter sharing or conditional modulation would be helpful. Wall-clock reporting includes basis generation speedup but a comprehensive end-to-end budget (training time, energy, peak memory, coarse-solve cost) would improve practical transparency.

### Questions
Could you provide a bound that maps principal angles between true and predicted coarse spaces to a bound on the $𝐻^1$ relative error of the final solution, at least under simplifying assumptions?
For 3D large-scale runs, what are the peak memory, wall-clock breakdown, and parallelization strategy during inference and coarse solves?

### Soundness
3

### Presentation
3

### Contribution
3
