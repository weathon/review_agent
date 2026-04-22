# EB-gMCR: Energy-Based Generative Modeling for Signal Unmixing and Multivariate Curve Resolution

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Signal unmixing analysis decomposes data into basic patterns and is widely applied in chemical and biological research. Multivariate curve resolution (MCR), a branch of signal unmixing, separates mixed signals into components (base patterns) and their concentrations (intensity), playing a key role in understanding composition. Classical MCR is typically framed as matrix factorization (MF) and requires a user-specified number of components, usually unknown in real data. Once data or component number increases, the scalability of these MCR approaches face significant challenges. This study reformulates MCR as a data generative process (gMCR), and introduces an Energy-Based solver, EB-gMCR, that automatically discovers the smallest component set and their concentrations for reconstructing the mixed signals faithfully. On synthetic benchmarks with up to 256 components, EB-gMCR attains high reconstruction fidelity and recovers the component number within 5% at 20dB noise and near-exact at 30dB. On two public spectral datasets, it identifies the correct component number and improves component separation over MF-based MCR approaches (non-negative variants [NMFs], ICA, MCR-ALS). EB-gMCR is a general solver for fixed-pattern signal unmixing (components remain invariant across mixtures). Domain priors (non-negativity, nonlinear mixing) enter as plug-in modules, enabling adaptation to new instruments or domains without altering the core selection learning step.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EB-gMCR (Energy-Based Generative Modelling for Signal Unmixing and Multivariate Curve Resolution). This framework reformulates classical multivariate curve resolution (MCR) as a generative process rather than a matrix factorisation problem. The authors introduce an energy-based adaptive gating module (EB-select) that automatically selects a minimal subset of source components while maintaining reconstruction fidelity. This approach aims to overcome limitations of conventional MCR methods, such as the need to predefine the number of components, lack of scalability, and difficulty in incorporating domain constraints.
Experiments on synthetic benchmarks (up to 256 components) and two real spectral datasets demonstrate strong reconstruction accuracy and correct component-count recovery compared with NMF, sparse-NMF, ICA, and MCR-ALS baselines. The authors also provide detailed convergence proofs, ablations, and reproducibility statements.

### Strengths
* Recasts MCR as a generative problem, conceptually unifying signal unmixing and energy-based learning.

* Introduces a differentiable hard-selection mechanism (EB-select) enabling data-driven component pruning.

* Provides comprehensive theoretical derivations and convergence analysis.

* Demonstrates strong empirical results on spectral datasets and includes full reproducibility materials.

* A meaningful step forward for scalable unmixing and spectral decomposition within chemistry and materials analysis.

### Weaknesses
*  The claimed novelty is modest; the paper applies energy-based modelling (EBM) and sparse gating techniques, both well-studied concepts. For example, the recent review “Hitchhiker’s guide on Energy-Based Models” (Carbone, 2024) surveys EBMs and situates them among VAEs, flows and GANs, showing energy-based methods are mature and widely applied. Meanwhile, sparse gating or mixture-of-experts style selection mechanisms are well established (see e.g., Mixture-of-Experts architecture surveys).  As a result, the contribution here is primarily domain adaptation (spectral unmixing) rather than a fundamentally new generative modelling algorithm or gating mechanism.

* Although the paper claims plug-in adaptation across domains, validation is confined to chemical spectra. There are no results for real hyperspectral imaging datasets or other modalities.

* The method claims to handle thousands of components, yet experiments only reach 256 latent sources.

* Metrics focus on 𝑅2 and component count recovery; no generative metrics (likelihood, uncertainty, sample diversity) or comparisons to VAE/score-based baselines are reported.

* Despite claims of “hands-free” component discovery, λ and temperature schedules require manual tuning.

* Dense notation, minimal intuition for EB-select and gating dynamics.

### Questions
* How sensitive is EB-gMCR to the choice of the λ coefficients and temperature decay schedule? Can the authors provide quantitative ablation on these hyperparameters?

* Can the proposed framework be extended to nonlinear or multimodal unmixing tasks (e.g., image or audio mixtures)?

* How does EB-gMCR perform under correlated component spectra or severe noise (>30 dB)?

* Could the authors clarify how EB-select differs fundamentally from previous sparse or Concrete-gated EBMs beyond its application domain?

* Would evaluating on benchmark generative datasets (e.g., MNIST mixtures, speech sources) help demonstrate generality

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
3

### Summary
The paper proposes a generative model for signal unmixing that includes a generic aggregation function, concentration generator, and component patterns, all of which are implemented by deep learning networks. The model is trained via minimization of an energy function that accounts for correlation of the signal components via a kernel embedding, component usage, and selection energy.

### Strengths
The generic nature of the components of the mixing model makes the approach amenable to a variety of applications.

### Weaknesses
The approach uses a combination of heuristics (sparsity prior, component activations, $\ell_1$ norm penalty on component weights) but does not provide much motivation for the choices. Furthermore they are sourced from existing approaches.

There are multiple instances of notation not being defined (as detailed in "Questions").

A comparison of computation time with unmixing baselines that do not require training is missing.

### Questions
What is $\omega$ in eqs. (2-3)? It does not appear to be considered anywhere else.

In lines 201 and later, what is $X_o$/$X_0$?

In line 226, is $|{\mathbf E}\|_2^2$ supposed to be $\|{\mathbf E}(\omega)\|_2^2$?

What is $E_{f_e}$ in eq. (6)? Is it different than ${\mathbf E}(\omega)$?

What is the "coefficient of determination $R^2$"? Why is it used as a metric in Section 5? What is the "EC" metric?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes EB-gMCR, an approach that reformulates the classical MCR problem as a data generative process (gMCR). The main innovation is an energy-based solver that automatically discovers the minimal component set needed for signal reconstruction without requiring a pre-specified number of components. The method starts with an oversized pool of candidate components and uses a differentiable "EB-select" gating network to retain only necessary components while estimating their concentrations. The authors validate their approach on synthetic benchmarks with up to 256 components and two real spectral datasets (carbohydrate Raman and NIR bioethanol).

### Strengths
* Overall, the writing is quite clear and logically structured (except for the mathematical formalism; see below).
* The reformulation of MCR as a generative process (gMCR) is innovative. Traditional MCR is typically framed as matrix factorization requiring a user-specified number of components, while this work provides a principled way to learn both the component set and their mixing patterns simultaneously.
* The ability to handle pools of 1000+ candidates is impressive and addresses real-world needs.
* The convergence analysis (Theorem B.6 and supporting lemmas) provides theoretical grounding for the two-phase learning dynamics, though the presentation could be clearer.
* The plug-in of domain constraints (non-negativity, nonlinear mixing) without requiring solver redesign is a  practical advantage over existing methods.

### Weaknesses
* Only two real datasets are tested, both relatively simple (N=3 and N=2 components). The method's performance on more complex real-world mixtures isn't shown.
* The method introduces several hyperparameters (λ weights, temperature τ, R² bands for checkpointing) whose selection process and sensitivity are not thoroughly discussed. Sensitivity or instability w.r.t. those parameters could be neck breaking for many more complex problems.
* The mathematical formalism is quite sloppy and therefore sometimes difficult to follow. E.g. what kind of elements are D, C, S and E? Assumptions and conclusions aren't strictly separated (see questions below).

### Questions
* All mathematical statements in the appendix contain proof sketches. Does that mean you have a checked proof for those statements, but you're not spelling out the proof? If so, why aren't you showing the full proof?
* Fig. 2 and 3: Are the R2 numbers on test or train data?
* Eq. 3: is this an assumption or a condition that is required? It seems like this does not follow from E. 2 in general.
* Eq. 5: Is the approximation an assumption or why would this be true? The expectation is in [0, 1] in general whereas the function maps to {0,1}.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an eneger-based method, EB-gMCR, to tackle multivariate curve resolution. 
EB-gMCR automatically discovers the minimal set of latent components needed to reconstruct mixed signals by using a hard selection mechanism (EB-select) and additional regularizations to prune an initially oversized component pool during training. The approach shines when to hundreds of candidate components and integrates chemical priors naturally. Experiments on synthetic datasets (up to 256 components) and real spectral mixtures show that EB-gMCR accurately recovers component numbers and achieves strong reconstruction fidelity, outperforming classical MCR-ALS, NMF variants, and ICA particularly in high-component regimes.

### Strengths
- The proposed formulation is conceptually natural for the problem, and as a result the method exhibits favorable scaling compared to traditional matrix-factorization–based approaches.
- The framework is flexible, allowing domain knowledge to be incorporated. 
- The empirical results are strong: the approach performs well in both synthetic benchmarks and real spectral datasets.

### Weaknesses
- The exposition is generally unclear. The core components of the method are spread across Section 4, with implementation details and conceptual justification interleaved, making it difficult to follow the full pipeline end-to-end. The presentation would benefit from consolidating the algorithmic steps (e.g., perhaps through a dedicated algorithm block or overview subsection) and then discussing each module in isolation in separate sections.
- In addition, there are several minor writing and notation issues, e.g. $\lambda_{\text{amb}}$ appears for the first time in Eq. 13 without a clear linkage to the ambiguity regularizer in Eq. 9, and $X_o$ is not introduced prior to line 201. A few other are listed below, but they are frequent enough that a careful pass is warranted. 
- One of the main benefits of the method is it's computation efficiency as number of components grow. However, the experiments on real datasets involve only 2-3 components. Could the authors demonstrate performance of EB-gMCR in a real, large-component setting?

### Questions
- In line 240, why $\tau = 0.9999994$?
- In line 302, what is PL? 
- In line 202, do you mean $X_o$, as opposed to $X_0$
- Could the authors provide wall-time comparison between EB-gMCR and baselines, particularly when component count is high.

### Soundness
2

### Presentation
1

### Contribution
3
