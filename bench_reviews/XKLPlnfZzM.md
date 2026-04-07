## Summary
This paper introduces the Temporal Deaggregation Diffusion Model (TDDM), a hierarchical framework for generating large-scale human trajectories. TDDM factors generation into spatial occupancy priors (marginal distributions over geographic cells) and temporal dynamics, using coordinate canonicalization to enable a single model to generalize across regions. It establishes a multi-city benchmark and shows improved fidelity and coverage over strong baselines, with compelling zero-shot generalization to unseen city areas and entirely new cities.

## Strengths
- **Novel factorization**: Separating spatial priors from temporal dynamics is an elegant and well-motivated solution to limitations of prior work (e.g., sample-specific conditioning), enabling controllability and reducing memorization risk.
- **Comprehensive evaluation**: Rigorous benchmarking across three diverse cities (Beijing, Porto, San Francisco) with a suite of 10 metrics covering fidelity, coverage, proportionality, usefulness, and generalization. TDDM consistently outperforms strong GAN, VAE, and diffusion baselines.
- **Impressive generalization**: Demonstrates compelling zero-shot generalization, both intra-city (trained on 25% of a city) and city-to-city (trained on one city and applied to another), validating the benefits of spatial-temporal factorization and canonicalization.
- **Strong reproducibility**: Provides detailed pseudocode, architecture diagrams, hyperparameters, preprocessing steps, and commits to releasing runnable code, facilitating replication and future work.

## Weaknesses
- **Mathematical clarity**: The mathematical formulation in Section 3 (Equations 1–5) is garbled and must be rewritten with correct notation to clearly convey the mixture model and the definition of the spatial prior \(H\).
- **Inconsistent descriptions**: Canonicalization is described as mapping to \([-1,1]^D\) in the text but to \([0,1]^D\) in Algorithm 1 and Appendix C.3. This inconsistency must be resolved.
- **Architectural ambiguity**: The tokenization of the spatial prior \(H\) (a 64×64 grid) into 64 tokens is not clearly explained in the main text. The exact method (e.g., patching) should be specified for reproducibility.
- **Algorithmic error**: Algorithm 2, line 4 contains garbled text for calculating \(N_{rc}\) and must be corrected.
- **Generalization requirements**: Zero-shot transfer requires access to aggregate target data to compute the spatial prior \(H\). The paper should explicitly discuss this requirement and its implications (e.g., what if only a coarse prior is available?).
- **Missing baseline comparisons**: The claim of state‑of‑the‑art would be stronger with direct comparisons to modern conditional baselines like ControlTraj and COLA, even if reimplementation is needed.
- **Lack of memorization analysis**: The claim that spatial priors reduce memorization risk is not substantiated by experiments (e.g., nearest‑neighbor distance or membership inference tests).
- **Incomplete ablation**: The contribution of canonicalization is not isolated; an ablation training without canonicalization (using absolute coordinates) is needed to validate its role in generalization.
- **Limited failure analysis**: The increased Length error in city‑to‑city transfer indicates a weakness. A deeper analysis of what temporal dynamics fail to transfer (e.g., speed profiles, turn patterns) is missing.
- **Sample‑level realism**: The evaluation relies on aggregate metrics; an analysis of physical plausibility (e.g., acceleration constraints, adherence to road networks) at the trajectory level would strengthen the fidelity claim.

## Nice-to-Haves
- Reporting uncertainty estimates (e.g., standard deviations over multiple runs) for key distributional metrics.
- A deeper investigation into why Porto serves as a particularly good source city for generalization.
- Sensitivity analysis on the robustness of the method to noisy or sparse spatial priors.
- A simple two‑stage baseline to disentangle the contribution of the spatial prior from the diffusion model.
- Computational cost comparison with baseline methods.
- Visualizations of failure cases and side‑by‑side comparisons in held‑out regions for generalization.
- A case study demonstrating controllability by editing the spatial prior \(H\).

## Removed Points
*These points are flagged to be removed, treat them with caution.*  
- None. All points raised by the reviewers are substantive, though some have been weakened or moved to nice‑to‑haves.

## Novel Insights
The paper’s core insight—that trajectory generation can be factorized into spatial occupancy priors and temporal dynamics, and that canonicalization enables a single model to generalize across cities—is novel and impactful. The finding that training on Porto generalizes well to other cities suggests that certain datasets may act as “universal sources” for trajectory generation, which is an interesting observation for future research.

## Suggestions
- Revise Section 3 to correct the mathematical notation, resolve the canonicalization inconsistency, and clearly explain the tokenization of \(H\).
- Correct Algorithm 2, line 4.
- Add a discussion on the requirements for zero‑shot transfer (aggregate target data for \(H\)).
- Include an ablation study on canonicalization and a memorization test.
- Analyze the failure modes for Length error in city‑to‑city transfer.
- Consider adding a comparison with ControlTraj and COLA, even if via reimplementation, to solidify the state‑of‑the‑art claim.