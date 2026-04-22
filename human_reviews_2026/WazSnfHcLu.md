# Flow-Distorted Plane Waves

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
The plane wave basis is widely used in Galerkin approximation, due to its periodicity and computational advantage, where the fast Fourier transform (FFT) can be applied. However, since its spatial resolution is uniform, the number of basis functions required can be excessive for problems with rapidly varying local features. We propose an adaptive basis called flow-distorted plane wave (FDPW), where the bijection of a normalizing flow is used to distort the problem domain, hence achieving adaptive resolution. We apply FDPW to Kohn-Sham density functional theory (DFT) calculations to both molecular and solid-state systems, demonstrating improved speed and memory usage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Flow-Distorted Plane Waves (FDPW): a geometry-aware basis obtained by composing a lattice linear map $T$ with a learned torus diffeomorphism $g_\theta$ while retaining PW-style algebra (FFT/NUFFT and k-point decoupling via a modified Bloch phase). The only "learning" component is a one-shot KL fitting of $g_\theta$ to a prescribed density, after which $g_\theta$ is frozen and the electronic structure problem is solved on the distorted basis.

### Strengths
- **Algebraic elegance and clarity.**  
  Recasting the Bloch phase as $ \exp(i\,k^\top f^{-1}(r)) $ is a neat device that simultaneously addresses the $k$–space coupling due to $T$ and the nonlinear phase distortion due to $g$, yielding block-diagonal structure across $k$ after the warp.  
- **Computationally coherent PW replacement.**  
  Local operators are evaluated via two inverse FFTs and pointwise products; the kinetic energy becomes a “minimal‑coupling” quadratic form over the warped metric, avoiding dense $G{\times}G'$ matrices while preserving PW‑like efficiency.  
- **Compact warp parameterization.**  
  A circular RQS autoregressive flow on the torus offers sufficient expressivity with a small parameter count, suggesting a practical path to spatial adaptivity without abandoning spectral tooling.  
- **Potential for efficiency gains.**  
  The premise—achieving target accuracy with fewer modes/smaller grids by concentrating resolution near physically singular/rapid‑variation regions—addresses a real bottleneck in PW‑style solvers. Flag For Ethics Review

No ethics review needed

### Weaknesses
(W1) Empirical evidence

-	The present experimental scope appears limited (e.g., one crystalline material and one small molecule).
-	A quantitative, matched-accuracy comparison with recent adaptive bases/coordinate approaches, as well as strong PW+PAW/USPP setups, would help readers gauge relative merits in time/memory/accuracy.
-	Ablations seem absent for:
(i) NUFFT parameters (oversampling, kernel width) versus accuracy and wall-clock;
(ii) flow capacity/regularization;
(iii) the contribution of the prescribed-density fit versus simpler prescriptions/no fit;
(iv) Hartree/Poisson preconditioner robustness.
A compact table of matched-accuracy results and a few ablation curves would substantiate the efficiency claims more convincingly.

(W2) Learning is used as a pre-fit; $g_\theta$ is then fixed.
The ML component is currently a one-shot geometry pre-fit,
\\[\min_{\theta}\ \mathrm{KL}\\big(p_\theta \,\|\, \rho_{\mathrm{prescribed}}\big),\qquad p_\theta \text{ induced by } g_\theta,\\]
after which $g_\theta$ becomes a fixed, differentiable background for operator evaluation. In essence this is a basis change (a warped PW basis).

-	Why is learning necessary at all? Would an analytic/hand-crafted warp (e.g., radial crowding near nuclei plus lattice symmetries) achieve similar results? Please quantify.
-	Why not adapt $g_\theta$ during SCF/geometry optimization with stability safeguards (slow updates, penalties, bilevel schedules)? If attempted, please report stability and net benefit; otherwise the ML contribution may be viewed as just implementation convenience rather than learning.

(W3) Writing/positioning for an ML venue

The abstract, introduction, and related work are too terse for orienting ML readers: the problem, background, and significance are not sufficiently articulated, and the method exposition mixes core ideas with plumbing. It may help to separate “concept $\to$ algebraic consequences $\to$ implementation modules” and to add a brief “method-at-a-glance” figure.

### Questions
- **(Q1)** Clarify roles of $T$ vs $g$ in Eq. (6). Even with $g=\mathrm{Id}$, the lattice map $T$ generally couples $(G,k)$, breaking cross-$k$ orthogonality unless the Bloch phase is expressed through $f^{-1}$. The manuscript should explicitly separate: (a) the linear mismatch introduced by $T$ (which is induced by problems), and (b) the nonlinear distortion from $g$ (which is induced by methods).  
- **(Q2)** On the representation $g=\mathrm{Id}+g_p$ and its scope. Arguments that absorb the periodic part into the cell-periodic function rely on $g$ being an orientation-preserving torus diffeomorphism homotopic to identity (as realized by the circular-flow family). They do not cover flips such as $g=-\mathrm{Id}$. Please state this assumption explicitly and indicate which steps (e.g., triangular Jacobian/log-det structure) rely on it.  
- **(Q3)** There are several formatting issues.  
  - Lines 248 and 249 are too close.  
  - The format of the symbol "-" seems incorrect, e.g., in line 149.  
  - There is a typo in lines 90–91: “when we G in subscript we mean…”.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Flow‑Distorted Plane Waves (FDPW), a Galerkin basis that composes a periodic, bijective normalizing flow on the 3‑torus with plane‑wave coordinates to obtain adaptive spatial resolution while retaining much of the plane‑wave algebra (FFT/NUFFT structure). The authors provide a rigorous mathematical formulation grounded in differential geometry, treating wavefunctions as half-densities. A key theoretical contribution is the introduction of a modified Bloch phase factor, which elegantly preserves k-space decoupling in periodic systems—a necessity for efficient solid-state calculations.

However, this coordinate transformation introduces significant complexity. The Laplacian becomes the Laplace-Beltrami operator, requiring the evaluation of complex geometric quantities (Jacobian, metric tensor, connections) derived from the flow. Crucially, the Hartree potential can no longer be solved directly in Fourier space and requires an iterative Preconditioned Conjugate Gradient (PCG) solver. External potentials are handled via Non-Uniform FFTs (NUFFT).

The authors demonstrate the method on simple systems (Diamond, CO, H2), suggesting that FDPW can achieve comparable accuracy to standard PWs with a smaller basis set size (N). While this suggests potential memory savings, the analysis of the introduced computational overheads and the comparison against standard practices in the field are insufficient to establish practical superiority.

### Strengths
The paper presents a mathematically sophisticated approach to adaptive basis sets, with notable theoretical innovations.

1. Elegant Solution to K-Space Decoupling: The introduction of the modified Bloch phase (Sec 4.2) to maintain orthogonality between different k-points in a distorted coordinate system is a significant theoretical contribution. It elegantly resolves a fundamental barrier to applying DPW methods in periodic systems.

2. Rigorous Geometric Formulation: The authors provide a thorough derivation based on differential geometry (Appendices E, F, G). The treatment of wavefunctions as half-densities ensures the unitarity of the transformation and provides a sound basis for deriving the operators.

3. Novel Synthesis: Combining normalizing flows, a modern ML technique, with the established DPW framework offers a parameter-efficient and differentiable approach to defining the adaptive basis.

### Weaknesses
While the approach is theoretically strong, the paper has significant limitations regarding the scope of validation, analysis of computational overhead, and practical applicability.

1. Insufficient Comparisons to Standard Practices: The comparison is limited to vanilla Plane Waves. FDPW is presented as an all-electron method (using ANC potentials). To assess practical relevance, comparisons against highly optimized PW codes using standard pseudopotentials (e.g., PAW or ultrasoft), which inherently reduce the required basis size, are necessary. Comparisons with other modern adaptive basis methods (e.g., Lindsey & Sharma (2024)) are also missing.

2. Dependence on Heuristic Initialization: The method's performance relies on a two-stage process where the flow is first fitted to a heuristic "prescribed density" (Sec 4.4), involving several hyperparameters (Eq 13) and regularization weights (Eq 15). The paper does not sufficiently explore the robustness and sensitivity of the results to this initialization procedure.

3. Unanalyzed and Potentially Prohibitive Computational Overhead: FDPW introduces substantial overhead per iteration. This includes evaluating the flow and computing complex geometric quantities via automatic differentiation at every grid point. The kinetic energy evaluation (Eq. 18) and the iterative nature of the PCG solver also add significant cost. The paper lacks a breakdown of these costs and their scaling, making the true efficiency trade-off unclear.

4. Numerical Stability and Conditioning: There is a fundamental tension between adaptation (requiring strong distortions) and numerical stability. Strong distortions lead to ill-conditioned metric tensors, which can slow down the PCG solver and increase quadrature errors. This trade-off is not analyzed.

5. Limited Empirical Validation and Missing Features: The experiments are restricted to very simple systems (Diamond, H2, CO). Performance on realistic scenarios (metals, defects) remains unexplored. Furthermore, the framework lacks support for essential features like non-local pseudopotentials and efficient force calculations.

6. Accessibility: The paper heavily relies on advanced concepts from differential geometry (e.g., fiber bundles, pullbacks, connections) and solid-state physics. While rigorous, this presentation style makes the paper highly inaccessible to a general ICLR audience, with crucial details buried in extensive appendices.

### Questions
The following major concerns must be addressed to improve the assessment of this work:

1. Have you tested FDPW on more complex systems than Diamond or CO, such as metallic systems or systems with defects? How does the complexity of the required normalizing flow scale with the heterogeneity of the physical system?

2. Standard PW calculations typically use optimized pseudopotentials to significantly reduce the required energy cutoff. How does the efficiency of the proposed all-electron FDPW method compare to a standard PW implementation using efficient norm-conserving or PAW pseudopotentials? 

3. How sensitive are the final accuracy and convergence speed to the hyperparameters of the prescribed density initialization and the elastic regularization? Was significant tuning required for the examples shown?

4. In the finite system experiments (Table 3), the PW results seem very poorly converged, and the FDPW energies also differ significantly from the reference. Could you clarify these discrepancies and the convergence behavior of both PW and FDPW for this system?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Flow‑Distorted Plane Waves (FDPW), an adaptive Galerkin basis for Kohn–Sham Density Functional Theory (DFT). The FDPW basis is constructed by applying a periodic normalizing flow on the 3‑torus to standard plane-wave coordinates. A key innovation is a modified Bloch factor, $𝑒^{ikf^{-1}(r)}$, which is designed to preserve 𝑘-point orthogonality. A special care is taken to efficiently treat different terms of arising equations.

### Strengths
- The paper proposes a principled way to add adaptivity to plane‑wave methods while retaining their simple algebra and FFT‑friendly structure.
- Modifying the Bloch phase to work with the distortion is neat and keeps the usual \(k\)-space structure intact. 
- Using a compact normalizing‑flow map to enhance a classical solver is a sensible, lightweight use of ML that fits existing workflows.
- Preliminary experiments suggest the method maintains spectral‑like accuracy in practice while noticeably reducing parameters and resource use.

### Weaknesses
I see this paper as a first step in an interesting direction, but there remains substantial room for improvement. At the current stage, there is little theoretical or practical analysis of the approach’s limitations. In particular, there is no systematic study of accuracy versus cutoff or an examination of NUFFT and curvilinear discretization errors, which would help establish numerical reliability. The paper also lacks comparison with other adaptive bases, such as finite elements or wavelets, making it difficult to judge relative advantages. Despite these gaps, whether or not it becomes a powerful long-term method, in my opinion, it opens a valuable direction worth exploring, and I am leaning towards acceptance.

### Questions
1) Is there anything you pay for using $f^{-1}(r)$ or are there no noticiable drawbacks?
2) How does FDPW handle GGA gradients in practice? Any challenges with hybrid functionals or nonlocal pseudopotentials?
3) How sensitive are performance and accuracy to the prescribed‑density parameters $a_l,b_l,c$?

### Soundness
3

### Presentation
2

### Contribution
3
