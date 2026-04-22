# Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Neural operators provide resolution-equivariant deep learning models for learning mappings between function spaces. Among them, the Fourier Neural Operator (FNO) is particularly effective: its spectral convolution combines a low-dimensional Fourier representation with strong empirical performance, enabling generalization across resolutions. 
While this design aligns with settings where the Fourier basis diagonalizes the underlying operator, such as linear, constant-coefficient PDEs on periodic domains, in which Fourier modes evolve independently, nonlinear PDEs exhibit structured interactions between modes governed by polynomial nonlinearities. To capture this inductive bias, we introduce the **Higher-Order Spectral Convolution**, a spectral mixer that extends FNO from diagonal modulation to explicit $n$-linear mode mixing aligned with nonlinear PDE dynamics. Across benchmarks, including Burgers and Navier-Stokes equations, our method consistently improves accuracy in nonlinear regimes, achieving lower error while retaining the efficiency of FFT-based architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Higher-Order Fourier Neural Operators (HO-FNO), an extension of the popular Fourier Neural Operator that explicitly models nonlinear mode interactions in the spectral domain. The key insight is elegant: while standard FNO processes Fourier modes independently (suitable for linear PDEs), nonlinear PDEs exhibit structured interactions between modes through polynomial nonlinearities. The authors address this by introducing an m-linear spectral convolution that aggregates m-tuples of Fourier coefficients whose indices sum to each target mode, directly mirroring the triadic interactions seen in equations like Navier-Stokes. The method maintains FFT efficiency at O(N log N) complexity while achieving consistent accuracy improvements across multiple benchmarks including Burgers, Navier-Stokes, Diffusion-Reaction, and spherical Shallow Water equations.

### Strengths
I think the authors are trying to address a real limitation of standard FNO, and I appreciate that they're thinking carefully about the physical structure of PDEs rather than just throwing more parameters at the problem. The observation that nonlinear PDEs have structured frequency interactions is valid, and building this into the architecture as an inductive bias is reasonable in principle. The mathematical framework connecting polynomial nonlinearities to m-linear Fourier mixing is clearly presented in Sections 2-3, particularly the detailed Navier-Stokes derivation showing triadic interactions, even if I'm not fully convinced it provides practical benefits beyond what existing methods achieve. I also think it's good that the authors tested on multiple benchmarks rather than cherry-picking a single favorable result the diversity of PDEs (1D Burgers, 2D Navier-Stokes at different viscosities, reaction-diffusion systems, and spherical equations) does give some confidence that the approach has breadth, and the effort to extend the method to non-Euclidean geometries via generalized Fourier transforms shows the authors are thinking about broader applicability.

### Weaknesses
The paper has fundamental issues with both theoretical justification and experimental rigor. Theoretically, the contribution is unclea the authors motivate HO-FNO by pointing to triadic interactions in PDEs, but they never rigorously prove what additional expressivity this provides beyond standard FNO with pointwise nonlinearities, which already has universal approximation capabilities. The claim of "first spectral neural operators modeling exact mode interaction" feels like an overstatement when physics-informed methods already encode PDE structure directly. The experimental setup raises serious concerns: there's no statistical significance testing, no error bars reported. The "parameter-matched" claim is never substantiated with actual parameter counts or FLOPs comparisons. The computational complexity analysis is misleading—while FFT operations are O(N log N), the m-linear mixing in physical space adds overhead that isn't carefully quantified. Real wall-clock time comparisons are absent. The results themselves are inconsistent and sometimes contradictory: UNO achieves much better rollout on Diffusion-Reaction (1.59 vs 2.37) despite weaker single-step metrics, which the authors hand-wave away rather than properly investigate.

Moreover, the scope and validation are limited. The method only works for polynomial nonlinearities, excluding many important PDEs with exponential, logarithmic, or other non-polynomial terms. There are no ablation studies systematically varying m, the number of modes M, or the parameterization of the A_i operators. The claim that hyperparameters don't need tuning is convenient but completely unsupported. No comparison to simply making FNO deeper with the same parameter budget, and no comparison to higher-order attention mechanisms despite citing them as related work. The improvements shown are often marginal and may not be practically meaningful. The spherical harmonics extension (HO-SFNO) is straightforward and expected, adding little conceptual depth. The presentation also suffers from inconsistent notation, missing experimental details (initialization, learning rates, batch sizes), and visualizations that are too small to meaningfully evaluate. Overall, this feels like an early-stage idea that needs substantial additional work before it's ready for publication.

### Questions
1. **Theoretical Expressivity vs. Standard FNO**: Can you provide some theoretical analysis demonstrating what functions or PDE solutions HO-FNO can represent that standard FNO with sufficient depth cannot (I would assume they are the same though)? Given that both architectures have universal approximation capabilities, what is the formal advantage of explicit m-linear mixing versus implicit mixing through layered pointwise nonlinearities? Without this clarity, it's difficult to assess whether the improvements are fundamental or simply due to a different parameterization.

2. **Experimental Rigor and Fair Comparison**: Your claim of "parameter-matched" models is central to demonstrating that improvements come from the architectural innovation rather than simply having more capacity. Can you provide: (a) exact parameter counts for each model on each dataset, (b) actual wall-clock training and inference times, (c) results across multiple random seeds with error bars and statistical significance tests, and (d) a comparison against simply making FNO deeper/wider with the same parameter budget? The current experimental setup makes it impossible to determine whether HO-FNO's gains are meaningful or within noise margins.

3. **Inconsistent Rollout Performance and Metric Interpretation**: In Table 1, UNO achieves substantially better rollout performance on Diffusion-Reaction (1.59 vs your 2.37) despite much worse single-step metrics, yet your visual analysis suggests UNO's predictions are qualitatively poor. This contradiction raises concerns about what these metrics actually measure. Can you explain: (a) why rollout NRMSE and visual quality sometimes disagree so dramatically, (b) whether the normalization scheme in your rollout metric might be masking important error patterns, and (c) how practitioners should interpret these conflicting signals when choosing a model? This seems like a fundamental issue with the evaluation protocol that needs resolution.

### Soundness
2

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
5

### Summary
Summary.
This paper proposes Higher-Order Fourier Neural Operators (HO-FNO), which extend FNOs by introducing an $m$-linear spectral convolution that explicitly mixes Fourier modes reflecting the polynomial nonlinearities in PDEs. The method retains FFT-level $O(N\log N)$ complexity, generalizes to spherical harmonics (HO-SFNO), and achieves improved single-step accuracy and stable rollouts on Burgers, Diffusion--Reaction, Navier--Stokes, and spherical SWE benchmarks. 

Contributions.
A higher‑order spectral convolution that performs explicit $m$‑linear mode mixing inside neural‑operator layers; an FFT‑efficient implementation that preserves FNO‑class complexity without adding hyperparameters; and a geometry‑aware extension to spherical harmonics (HO‑SFNO) validated by empirical improvements across standard nonlinear PDE benchmarks.

### Strengths
1) Originality: Physics-aligned higher-order spectral convolution explicitly mixing modes with $k_1+\cdots+k_m=k$, introducing a new operator family beyond per-mode scaling.

2) Quality: Efficient layer design using one FFT and one IFFT with $\mathcal{O}(N\log N)$ complexity and no extra hyperparameters, delivering strong accuracy with comparable model size.

3) Clarity: Equations connect the physical $m$-linear product to Fourier-space mixing $k_1+\cdots+k_m=k$; the layer is simple to implement and its assumptions are clearly stated.

4) Significance: Gains on nonlinear flows and a spherical variant (HO--SFNO) that outperforms SFNO on $S^2$, indicating impact for operator learning and geophysical modeling.

### Weaknesses
1) Abstract over-claim. The abstract asserts that linear PDEs have Fourier modes that “evolve independently.” This only holds in special cases (e.g., constant coefficients on periodic domains where the operator is diagonal in the Fourier basis). In general (variable coefficients, non-periodic geometries, alternative bases), linear PDEs can couple modes; the claim should be qualified.

2) Resolution-equivariance phrasing. The introduction attributes “resolution-equivariant” behavior to neural operators broadly; in practice, this property is specific to FNO/SNO-style constructions under appropriate truncation/aliasing assumptions, not to all neural-operator families. Please narrow or define the scope precisely.

3) Novelty vs.\ prior TMLR work (DSFNO). The claimed contribution overlaps with prior work that already introduces non-diagonal frequency-domain interactions and formalizes mode mixing. The contribution should be positioned as a structured, higher-order ($m$-linear) mixer aligned with polynomial PDE couplings, with direct comparisons and discussion.

4) Parameters and fairness. Each HO-FNO layer adds $m$ per-point channel maps $A_i\in\mathbb{R}^{C\times C}$ plus pointwise products; at fixed depth/width this likely increases parameters relative to a standard FNO layer. Please provide explicit per-layer and total parameter counts (FNO vs.\ HO-FNO/HO-SFNO) and a table confirming parameter matching in the reported experiments.

5) Ablations are missing. The paper does not isolate (i) the interaction order $m$ (e.g., $m\!=\!1,2,3$), (ii) the role/number of $A_i$, or (iii) the number of retained modes $K$. Without these ablations, it is difficult to attribute gains specifically to higher-order mixing (vs.\ capacity).

6) Clarity and notation. The concurrent use of “$n$-linear” (PDE) and “$m$-linear” (model) can be confusing; add a brief notation paragraph. In Figure~1, denote products with $\times$ (instead of asterisks) and display the constraint $k_1+\cdots+k_m=k$ near the mixer to make the interaction explicit and improve figure quality.

References.
Gao, W., Luo, J., Xu, R., and Liu, Y. Dynamic Schwartz–Fourier Neural Operator for Enhanced Expressive Power. Transactions on Machine Learning Research (TMLR), 2025.

### Questions
1) Will you qualify the abstract’s “independent modes for linear PDEs” statement and specify the conditions under which operators are diagonal in Fourier space?

2) Can you provide a precise contrast to DSFNO: interaction form, symmetry implications, computational complexity, and regimes where each approach is preferable?

3) Please add ablations on $m$, on removing/tying $A_i$, and on the number of retained modes $K$ (and, if feasible, constant vs.\ learned spectral multipliers).

4) Please report closed-form parameter counts per layer (in $C$, $m$, groups, $K$) and a summary table verifying parameter matching across baselines.

5) Minor presentation: in Figure~1, switch to $\times$ for products and print the $k_1+\cdots+k_m=k$ constraint on the diagram for readability.

References.
Gao, W., Luo, J., Xu, R., and Liu, Y. Dynamic Schwartz–Fourier Neural Operator for Enhanced Expressive Power. Transactions on Machine Learning Research (TMLR), 2025.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Higher-Order Fourier Neural Operator (HO-FNO), an extension of the Fourier Neural Operator (FNO) designed to explicitly model multi-linear (m-linear) interactions among Fourier modes. While standard FNOs apply independent (diagonal) updates to Fourier coefficients, many PDEs inherently involve structured mode coupling (e.g., triadic interactions in Navier–Stokes). The authors propose a Higher-Order Spectral Convolution, which aggregates all m-tuples of modes and implements polynomial-like nonlinear mixing in the spectral domain. This mechanism remains computationally efficient at O(N log N) per layer using FFTs and pointwise multiplication.

### Strengths
1. Computational efficiency maintained: Despite introducing explicit multi-mode interactions, the method retains the same asymptotic cost as standard FNOs (both $O(n\log n)$ as in FFT). 

2. Easy integration into FNO: The modification adds no new hyperparameters and fits seamlessly into the existing FNO pipeline.

### Weaknesses
1. Insufficient empirical evaluation: The paper benchmarks against a limited and outdated set of models. Although many neural operators, including more recent FNO-based variants, have been proposed, the newest baseline considered is from 2023. Moreover, only two baselines (UNO and FNO/SFNO, I don't count U-Net as a baseline) are included in the comparison.

2. Although it maintains the same complexity as FNO, the computational constants (not just asymptotic complexity) might be significant. There is no report of runtime or memory scaling. 

3. A key claim of the paper is that standard FNOs fail to capture nonlinear mode coupling because they update Fourier modes independently. However, in FNO, pointwise nonlinearities between layers already induce implicit cross-mode coupling once signals are transformed back and forth between spatial and spectral domains.

4. The experimental setup appears to differ from prior FNO benchmarks. The reported results for the Navier–Stokes equation do not align with those in the original FNO paper, suggesting that datasets, preprocessing, or training configurations were redefined by the authors. I did not find any mentioning of this in the paper. Please advise if I missed them.

### Questions
1. How sensitive is the model to the choice of interaction order $m$?

2. Can you compare your method with FNO under the same setting as the original FNO paper?

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
5

### Summary
The paper extends FNO by introducing an explicit higher-order spectral mixing term to capture nonlinear mode interactions in PDEs. It demonstrates improved accuracy across several nonlinear PDE benchmarks under comparable training budgets. The proposed model aims to generalize Fourier Neural Operators to nonlinear dynamics while retaining their efficiency and resolution-equivariant properties.

### Strengths
1. The proposed HO-FNO achieves better performance than baseline FNOs across multiple benchmark problems, highlighting its strong capability to capture complex nonlinear dynamics.
2. The method introduces an insightful extension of FNO that explicitly models nonlinear mode coupling while preserving the spectral efficiency of the original framework.

### Weaknesses
1. The resolution-equivariance property, one of FNO’s core advantages, is not empirically demonstrated.
2. It would be more insightful to evaluate how performance changes with different interaction orders (m).

### Questions
1. Could the authors provide wall-clock runtime comparisons? Including such data would be helpful to substantiate the claim of computational efficiency and demonstrate the model’s practicality.

### Soundness
3

### Presentation
4

### Contribution
3
