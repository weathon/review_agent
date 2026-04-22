# FD-Bench: A Modular and Fair Benchmark for Data-driven Fluid Simulation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Data-driven modeling of fluid dynamics has advanced rapidly with neural PDE solvers, yet a fair and strong benchmark remains fragmented due to the absence of unified PDE datasets and standardized evaluation protocols. Although architectural innovations are abundant, fair assessment is further impeded by the lack of a clear disentanglement between spatial, temporal and loss modules.
In this paper, we introduce FD-Bench, the first fair, modular, comprehensive and reproducible benchmark for data-driven fluid simulation. 
FD-Bench systematically reviews and decomposes 89 baseline models reported across recent publications, extracting and standardizing their key architectural and training components for fair, unified comparison across 10 representative flow scenarios.
It provides four key contributions: (1) a modular design enabling fair comparisons across spatial, temporal, and loss function modules; (2) the first systematic framework for direct comparison with traditional numerical solvers; (3) fine-grained generalization analysis across resolutions, initial conditions, and prediction time window]; 
and (4) a user-friendly, extensible codebase to support future research. Through rigorous empirical studies, FD-Bench establishes the most comprehensive leaderboard to date, resolving long-standing issues in reproducibility and comparability, and laying a foundation for robust evaluation of future data-driven fluid models. The code is open-sourced at https://anonymous.4open.science/r/FD-Bench-15BC.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces FD-Bench, a benchmark for data-driven fluid simulation aimed at addressing the lack of standardized evaluation in the field. The authors propose a modular decomposition of neural PDE solvers into spatial, temporal, and loss components, and evaluate 89 baseline models across 10 fluid flow scenarios. They also include comparisons with traditional numerical solvers and analyze generalization across resolutions, initial conditions, and rollout horizons.

### Strengths
- The modular design is a step toward disentangling the contributions of different components in neural PDE solvers.

- The effort to unify and standardize evaluation across a large number of models is commendable.

- The inclusion of traditional numerical solvers as baselines is a valuable addition.

- The codebase is publicly released, which may facilitate future research.

### Weaknesses
- Lack of Theoretical or Algorithmic Novelty: The paper presents a benchmarking effort rather than a methodological advance. While useful, it does not introduce new models, theoretical insights, or algorithmic improvements. The decomposition into spatial/temporal/loss modules is intuitive but not novel, and the paper does not justify why this particular decomposition is the most meaningful or complete.

- Superficial Model Decomposition: The decomposition of 89 models into modular components is largely based on a post-hoc categorization rather than a unified theoretical framework. This risks oversimplifying architectural differences and may misrepresent the original contributions of the models being compared. For instance, grouping models under “self-attention” or “Fourier” ignores important nuances in their design and implementation.

- Limited Justification for Dataset Selection: While the authors curate 10 flow scenarios, the rationale for their selection is not deeply motivated from a fluid dynamics perspective. The paper does not adequately address whether these scenarios are representative of real-world challenges or if they cover the full spectrum of fluid behaviors (e.g., multi-phase flows, high Mach number regimes, complex geometries).

- Evaluation Metrics Lack Physical Interpretability: The reliance on RMSE and nRMSE as primary metrics is insufficient for fluid dynamics, where quantities like vorticity preservation, energy spectra, or divergence errors are often more informative. The paper does not justify why these standard fluid diagnostics are omitted, limiting the utility of the benchmark for the CFD community.

- Insufficient Discussion of Limitations in Neural vs. Traditional Solvers: The comparison with traditional solvers focuses on error and runtime but overlooks critical aspects such as stability, convergence guarantees, and physical consistency. Neural solvers are known to suffer from error accumulation and lack of long-term stability—issues that are not sufficiently addressed in the experiments or discussion.

- Generalization Claims Are Overstated: The generalization studies (e.g., zero-shot initial conditions, resolution shifts) are limited to narrow variations and do not convincingly demonstrate robustness to realistic distribution shifts. The paper does not test on truly out-of-distribution scenarios such as different geometries, boundary conditions, or physical parameters outside the training range.

- Reproducibility and Scalability Concerns: Although the code is released, the work does not provide sufficient details on computational budgets, hyperparameter tuning procedures, or the feasibility of reproducing all 89 models. The scalability of the benchmark to 3D or more complex PDE systems is also not demonstrated or discussed.

### Questions
- How does your modular decomposition account for models that inherently couple spatial and temporal processing (e.g., neural ODEs or recurrent architectures)?

- Why were more physically meaningful metrics (e.g., enstrophy, divergence, spectral decay) not included in the evaluation?

- Can you provide evidence that the selected flow scenarios are sufficiently diverse to draw general conclusions about model performance?

- What steps were taken to ensure that the hyperparameter tuning for each model was fair and computationally feasible across 89 baselines?

Should you be able to satisfactorily address the points I've raised above, I will accordingly provide a positive rating.

### Soundness
2

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
5

### Summary
This paper introduces FD-Bench, a modular benchmark for data-driven fluid simulation that standardizes datasets, model components, and evaluation protocols. The core contribution is to decouple neural PDE solvers into three orthogonal modules (spatial representation, temporal representation, and loss) and to compare 89 baselines across 10 flow scenarios under a unified codebase and tuning budget. The benchmark also includes head-to-head comparisons with traditional numerical solvers, generalization tests (OOD initial conditions, resolution shifts, and rollout length), and discretization studies (Eulerian vs. mesh vs. particle). Empirically, the paper reports that self-attention/Transformers tend to yield the best accuracy, while Fourier-based operators are competitive and computationally efficient; temporal bundling is consistently strong among evolution strategies. The authors release an extensible codebase and claim improved fairness and reproducibility of comparisons for data-driven fluid simulation.

### Strengths
The modularization of neural PDE solvers into spatial/temporal/loss components, along with benchmarking controlled cross-comparisons of these choices, is a useful organizing principle that helps attribute where gains come from, rather than treating methods as monoliths. The inclusion of direct comparisons to classical solvers at coarser grids and lower-order schemes (with matched error targets) is a thoughtful step toward practicality.

The benchmark scope is broad (reported 89 baselines, 10 scenarios) with standardized tuning and clear metrics (RMSE/nRMSE, fRMSE bands, memory, GFLOPs). The paper transparently documents setups and seeds, and provides a public, modular codebase for reproduction.

The paper is well structured and clear to read. I enjoyed reading this paper!

A fair, reproducible reference point for data-driven CFD is valuable. The finding that Transformers dominate on accuracy while Fourier operators offer excellent cost-accuracy trade-offs, and that temporal bundling is a robust evolution strategy, can rationalize design choices in subsequent work. The numerical-solver comparisons and run-time/speed-up evidence increase the benchmark’s practical relevance.

### Weaknesses
*Novelty of questions vs. synthesis*: The three headline questions (which neural architecture; can neural replace numerical; which discretization; how well do models generalize) are useful but not conceptually new; several prior surveys/benchmarks have articulated similar axes. FD-Bench’s novelty is primarily scope and modular rigor, not new task formulations. (This aligns with the authors’ own positioning against prior benchmarks in Table 1.)

*Insights largely confirm established intuitions*: The empirical takeaways (Transformers (self-attention) excel in accuracy, Fourier operators in efficiency/long-range mixing) are directionally consistent with recent literature and community experience (e.g., work by Mishra group at ETH, reviews from Brown, etc); FD-Bench consolidates rather than surprises here.

*Geometry and boundary-condition coverage feel limited for real-world deployment*: Despite listing heterogeneity and even “irregular geometries” among selection criteria, much of the benchmark centers on canonical, mostly rectangular/periodic domains (e.g., incompressible/Compressible N-S on uniform grids, Burgers, advection, Kolmogorov flow). The Lagrangian subsets (Taylor–Green, lid-driven cavity, reverse Poiseuille) add variety, but the dominant training/evaluation remains on boxes with simple/periodic BCs, and several Lagrangian cases are aggregated back to regular grids for evaluation. This limits conclusions about performance under complex geometries (curved walls, obstacles, multi-body systems), mixed/moving boundaries, and non-Cartesian meshes, which are typical in engineering applications.

*Boundary-condition diversity*: The protocol does not yet stress diverse BC types (Dirichlet/Neumann/Robin inflow–outflow mixes, no-slip vs. partial slip, pressure outlets), CFL-critical transient regimes near complex boundaries, or wall models and source terms relevant to real devices. This weakens claims about readiness for practical deployment, even if trends are robust on boxes.

*Limited 3D and multiphysics breadth*: The benchmark mix skews toward 2D and single-physics PDEs; many industrial use cases hinge on 3D, moving interfaces, or coupled physics (e.g., conjugate heat transfer, reacting flows). As a result, the reported rankings may not carry over.

### Questions
I suggest these questions for consideration (with the understanding that many will be difficult to execute in the short response time, but with the hope that this will help the authors strengthen this good paper for the community)

*Geometry and BC realism*: Could the authors extend FD-Bench with non-rectilinear geometries (curved ducts, bluff-body wakes) and mesh-based Eulerian data (unstructured tri/tet, boundary-layer refinement) to stress spatial inductive biases outside regular grids? Can you introduce tasks with mixed boundary conditions (e.g., no-slip on walls, specified pressure at the outlet, and inflow profiles at the inlet) and report the per-boundary condition breakdown of errors? This would test if attention/Fourier advantages persist with wall physics.

*Generalization beyond periodic boxes*:  Several scenarios appear periodic or effectively box-confined. How do models fare under obstacle insertion or domain deformation (e.g., adding a cylinder/airfoil) without finetuning? 

*3D scaling/memory*: The efficiency section reports GFLOPs/memory on 2D. Do the memory/runtime advantages for Fourier vs. attention hold in 3D at comparable accuracy (e.g., (128^3), (256^3))? A 3D micro-suite could meaningfully extend the benchmark.

*Rollout step size & stability claims*: The paper argues neural solvers tolerate larger (\Delta t) with minor degradation compared to classical solvers constrained by CFL. Could you report absolute wall-clock vs. effective physical time advanced per step curves, plus failure cases where large (\Delta t) breaks stability/accuracy? This would help practitioners set safe schedules.

*Task difficulty calibration*: How sensitive are rankings to turbulence intensity (e.g., higher Reynolds for cavity/Poiseuille), shock strength in compressible cases, or stiff reaction rates? A difficulty-graded version of each task could probe robustness of the reported ordering.

### Soundness
3

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
2

### Summary
The current field of data-driven fluid simulation lacks unified datasets and standardized evaluation protocols, and the highly coupled design of spatial, temporal, and loss modules makes fair comparisons between methods difficult. This paper proposes FD-Bench, which establishes a comprehensive leaderboard, addressing long-standing issues of repeatability and comparability, and laying the foundation for robust evaluation of future data-driven fluid models.

### Strengths
1.This paper accurately identifies the core pain point of "fragmented evaluation" in the current field of neural PDE solvers, and the proposal of FD-Bench has clear practical significance.
2.This paper unifies and abstracts various existing methods into a quadruple of "spatial encoding + temporal encoding + loss function + tricks", and designs controlled variable experiments based on this to achieve "structure-performance" decoupling.
3.This paper provides an easy-to-use and reproducible codebase that addresses the long-standing lack of standardized evaluation protocols and helps to facilitate fair and reproducible evaluations in the future.

### Weaknesses
1.This paper emphasizes comparisons across 89 baselines. However, Table 3 appears to extract components from existing methods, categorize them, and then compare the methods after reorganizing the components, which doesn't seem entirely consistent with the description.
2.FD-Bench’s primary metrics are RMSE, nRMSE, and frequency‑domain RMSE. These capture “fit” but do not directly assess physical consistency.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
