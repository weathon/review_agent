# Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Advanced deep learning-based approaches have been actively applied to forecast the spatiotemporal physical dynamics governed by partial differential equations (PDEs), which acts as a critical procedure in tackling many science and engineering problems. As real-world physical environments like PDE system parameters are always capricious, how to generalize across unseen out-of-distribution (OOD) forecasting scenarios using limited training data is of great importance. To bridge this barrier, existing methods focus on discovering domain-generalizable representations across various PDE dynamics trajectories. However, their zero-shot OOD generalization capability remains deficient, since extra test-time samples for domain-specific adaptation are still required. This is because the fundamental physical invariance in PDE dynamical systems are yet to be investigated or integrated. To this end, we first explicitly define a two-fold PDE invariance principle, which points out that ingredient operators and their composition relationships remain invariant across different domains and PDE system evolution. Next, to capture this two-fold PDE invariance, we propose a physics-guided invariant learning method termed iMOOE, featuring an Invariance-aligned Mixture Of Operator Expert architecture and a frequency-enriched invariant learning objective. Extensive experiments across simulated benchmarks and real-world applications validate iMOOE's superior in-distribution performance and zero-shot generalization capabilities on diverse OOD forecasting scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles zero-shot OOD generalization for PDE forecasting with **iMOOE**, a framework built on a **"two-fold PDE invariance principle"**. This principle posits that core physical operators and their compositional structure remain invariant despite parameter shifts. The framework captures this using an **invariance-aligned Mixture of Operator Expert (MOOE) architecture**  and a **frequency-enriched invariant learning objective**. This objective enforces risk equality while also using a novel regularizer to **mitigate the spectral bias** of neural operators, forcing them to learn essential high-frequency features. Experiments show this "plug-and-play" framework significantly boosts zero-shot OOD performance for various neural operators.

### Strengths
* The paper tackles the highly significant and challenging problem of zero-shot OOD generalization for PDE dynamics. It accurately pinpoints the failure of prior methods: they do not capture the underlying physical invariance required for true zero-shot generalization without test-time adaptation.

* The core originality lies in proposing a "two-fold PDE invariance principle" (operator and compositionality invariance). This provides a structured, physics-based hypothesis that reframes the problem from finding abstract features to learning specific, invariant physical components.

* The framework is technically novel and significant in two ways. First, its "plug-and-play" nature makes it a broadly applicable tool for enhancing various existing neural operators. Second, it originally introduces a **frequency-enrichment loss** ($\mathcal{L}_{freq}$) to mitigate the spectral bias of neural operators, which is a critical and novel step for learning the complete physical invariance.

* The paper's claims are supported by comprehensive experiments across simulated PDEs and a real-world dataset, using a strict OOD setup with non-overlapping parameters. The results consistently show SOTA performance. Crucially, the insightful ablation studies (e.g., on expert number $K$ and the $\mathcal{L}_{freq}$ loss) effectively validate that each core component of the framework contributes to the final success.

### Weaknesses
* The paper's central premise—that prior methods fail *because* they miss the "fundamental invariance principle"—is asserted without direct evidence. This creates a correlation-not-causation gap, as the paper only proves its own method works, not *why* others fail.

* The paper fails to validate its core hypothesis with interpretability. There is no analysis or visualization of *what* the specialized experts ($\sigma_i$) actually learn, nor any theoretical analysis of the decomposition's properties (e.g., completeness, uniqueness).

* The mechanism for learning the derivative selection masks ($m_i$) is ambiguous. It is unclear whether these masks are learnable parameters or the output of a separate network, which harms reproducibility.

* A crucial baseline is missing: a monolithic neural operator with an equivalent parameter count to the final $K$-expert MOOE model. Without this control, it is unclear if the performance gain comes from the principled MOOE *structure* or simply from increased model capacity.

### Questions
1.  The paper's premise is that prior OOD methods fail *because* they miss the "fundamental invariance principle." Is there direct evidence to support this causal claim about *why* baselines fail, beyond iMOOE's own success?

2.  To validate the core hypothesis, can you provide an interpretability analysis of *what* the individual experts $\sigma_i$? For example, in the DR system, do they measurably specialize into distinct diffusion and reaction operators?

3.  Could you clarify the exact implementation of the derivative selection masks $m_i$? Are they learnable parameter vectors for each expert, or the output of a separate routing/gating network?

4.  To isolate the benefit of the MOOE *structure* from simply having more parameters, could you add a baseline comparison against a single, monolithic operator with an equivalent total parameter count to the $K$-expert iMOOE?

5.  While iMOOE uses dense experts, does it face training challenges analogous to sparse LLM MoEs, such as expert collapse or redundancy (especially as $K$ increases)? How effective is $\mathcal{L}_{mask}$ at ensuring true specialization versus mere diversity?

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
3

### Summary
This paper proposes iMOOE, a physics-guided invariant learning framework for zero-shot out-of-distribution (OOD) generalization in PDE dynamics forecasting. The key idea is to exploit two levels of PDE invariance—operator invariance and compositional invariance—through a Mixture of Operator Experts (MOOE) architecture and a frequency-enriched invariant learning objective. The authors show improved ID and OOD performance across several simulated PDE systems (Diffusion–Reaction, Navier–Stokes, Burgers, Shallow-Water, and Heat-Conduction) compared with recent meta-learning and parameter-conditioned models.

### Strengths
1. The paper tackles an important and difficult problem “zero-shot OOD generalization in PDE dynamics” and does so with a physically motivated approach.

2. The proposed Mixture of Operator Experts (MOOE) architecture is intuitive and nicely aligned with operator-splitting ideas in classical PDE solvers.

3. The frequency-augmented loss and risk-equality objective are interesting design choices that improve empirical robustness.

4. Experiments on multiple PDE benchmarks show clear performance gains over meta-learning and conditioning-based methods.

### Weaknesses
1. The “two-level PDE invariance” principle feels more heuristic than theoretically grounded — there’s no rigorous justification or proof that these invariances hold across PDE families.

2. All experiments are on synthetic, noise-free datasets with simple geometries and periodic boundaries. No real or noisy data are tested, so generalization claims are not fully convincing.

3. The frequency-based regularization might amplify noise in realistic cases, but this is not examined.

4. Comparisons with baselines may not be entirely fair since meta-learning models are designed for few-shot adaptation rather than strict zero-shot settings.

### Questions
1. Can the authors provide more formal or causal justification for the “two-level invariance” concept?

2. How sensitive is the method to the way training environments are defined (by parameters vs. time steps)?

3. How does the model perform on noisy or irregular real-world PDE data, such as CFD or weather simulations?

4. Does the frequency loss harm stability when data are non-periodic or contain sensor noise?

5. What is the computational overhead of iMOOE compared to a standard FNO?

### Soundness
2

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
This paper addresses the challenge of out-of-distribution (OOD) generalization in PDE dynamics forecasting. The paper proposes iMOOE, consisting of: 
- A Mixture of Operator Experts (MOOE) architecture, where parallel neural operator experts model distinct physical processes, and a fusion network captures compositional relationships.
- A frequency-enriched invariant learning objective, combining prediction, risk equality, and frequency regularization losses to promote invariant representations across environments.
Extensive experiments on five PDE systems (Diffusion-Reaction, Navier–Stokes, Burgers, Shallow-Water, Heat-Conduction) show that iMOOE improves ID and OOD generalization over baselines.

### Strengths
- Architectural Innovation:
The MOOE architecture aligns with PDE operator decomposition, offering modularity and interpretability. Its parallel design is more efficient than serial operator splitting.

- Robust Empirical Validation:
The experiments are comprehensive—covering both synthetic PDE benchmarks and a real-world application (SST).

### Weaknesses
- Limited Diversity of PDE Systems:
Although the paper evaluates multiple canonical PDEs, they are all relatively low-dimensional and well-behaved (2D systems with periodic boundaries). The scalability to irregular geometries or 3D turbulent systems is unclear.

- Theoretical Rigor of Invariance Definition:
The proposed “two-level PDE invariance” is intuitive but lacks rigorous formal grounding. The connection between the invariance principle and the optimization objective could be better substantiated.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work is motivated by invariant learning theory and addresses the challenge of out-of-distribution (OOD) prediction, covering both temporal and parametric extrapolation tasks. The key idea is that partial differential equations (PDEs) possess two fundamental types of invariance:
* Operator invariance: individual physical operators (e.g., diffusion, advection, reaction) remain consistent across domains.
* Compositionality invariance: the relationships between these operators and external factors such as physical parameters and forcing terms remain fixed for a given PDE.
To capture these invariances, the authors propose iMOOE (Invariant Mixture of Operator Experts). The method first decomposes a PDE into a set of compositional operators, then uses a Mixture of Operator Experts architecture to model both the invariant operators and their compositional relationships.
The effectiveness of iMOOE is demonstrated across five canonical PDE systems, including diverse parametric and temporal OOD scenarios, as well as a real-world Sea Surface Temperature (SST) dataset. Across all tests, iMOOE consistently outperforms baseline approaches (such as CoDA, GEPS, CAPE, CNO, DPOT, and VCNeF), achieving state-of-the-art performance in challenging OOD prediction tasks.

### Strengths
1. The paper presents a framework for improving OOD generalization in PDE simulation, motivated by invariant learning theory.
2. Several test cases are provided, including both temporal and parametric OOD settings, validating the method’s potential on challenging extrapolation problems.
3. The paper includes comprehensive experiments, systematic comparisons with strong baselines, and sensitivity analyses (e.g., number of experts, loss weighting, domain partitioning).

### Weaknesses
1. The method assumes that PDEs can be decomposed into known invariant operators. In cases where the governing equations are unknown or highly complex, this assumption may not hold, reducing the method’s applicability.
2. The paper claims “zero-shot OOD generalization”, but the evaluated OOD shifts are primarily controlled parametric variations. It is unclear whether this corresponds to zero-shot generalization to entirely new physical equations.
3. The method’s reliance on multiple neural operator experts increases computational cost, especially for high-dimensional PDEs.

### Questions
1. How does iMOOE perform for irregular spatial domains or unstructured meshes? Can the operator decomposition and masking strategy be naturally extended to such cases?
2. Can the authors comment on the generalization between dimensions — for example, if trained on 2D heat conduction, how well would the approach extrapolate to 3D?
3. Beyond parametric shifts, can iMOOE handle structurally different PDEs (e.g., transitioning from Burgers to Navier–Stokes), or does operator invariance only hold within a single PDE family?
4. How sensitive is the method to the diversity of training domains? Is there a minimum level of variation required to achieve robust generalization?

### Soundness
3

### Presentation
2

### Contribution
2
