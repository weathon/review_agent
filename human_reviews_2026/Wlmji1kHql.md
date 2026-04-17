# Latent Space Learning for PDE Systems with Complex Boundary Conditions

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Latent space Reduced Order Models (ROMs) in Scientific Machine Learning (SciML) can enhance and accelerate Partial Differential Equation (PDE) simulations. However, they often struggle with complex boundary conditions (BCs) such as time-varying, nonlinear, or state-dependent ones. Current methods for handling BCs in latent space have limitations due to representation mismatch and projection difficulty, impacting predictive accuracy and physical consistency. To address this, we introduce BAROM (Boundary-Aware Attention ROM). BAROM integrates: (1) explicit, RBM-inspired boundary treatment using a modified ansatz and a learnable lifting network for complex BCs; and (2) a non-intrusive, attention-based mechanism, inspired by Galerkin Neural Operators, to learn internal field dynamics within a POD-initialized latent space. Evaluations show BAROM achieves superior accuracy and robustness on benchmark PDEs with diverse complex BCs compared to established SciML approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a boundary-aware surrogate model grounded on reduced order modeling over latent spaces. The core idea of the proposal is an attention-based mechanism, inspired by Galerkin Neural Operators, to learn internal field dynamics and a module aimed at injecting explicit boundary forcing. These proposals are well-grounded on ROM theory. The proposed model is evaluated against a couple of strong representative baseline models over challenging scenarios in the field.

### Strengths
- The model architecture is grounded well on reduced order modeling.
- The proposed model outperforms other baselines in (limited) challenging scenarios.

### Weaknesses
**Non-standard or redundant usage of symbols:** The organization of the paper is fine on a high level, but many symbols are used in non-standard ways and need to be reorganized. 
- The attention $\mathbf{a}$ is expressed with a couple of different symbols and is very distracting.
- I do not clearly understand the indication of (13) and (14). Do they mean they are equivalent? If that is the case, how are these equivalences assured that they are equivalent?
- Some functions in the equation (15) are introduced without their definition.

**Over-claimed performance over the hyperbolic PDEs:** Table 2 shows that the proposed model is at most competitive and inferior to the baseline models in most of the cases. Even in a few scenarios where the proposed model achieves the best performance, most of those scores are very close to those of the other models. In the Darcy flow experiment with $T=2$, the proposed model is also not the best. The table also somehow looks like being “hidden” in Appendix, but this table should be presented in the main text. The metric is also limited as opposed to those used in Table 1.

**Insufficient ablation study:** The same ablation study should be conducted over more scenarios, especially should include the same scenarios as reported in Table 2. The description of the configuration is largely missing. In particular, the random noise experiment misses the detail involving its problem setting and hyperparameter configuration. This experiment should be conducted varying the range of variance of noise distribution and the performance should be averaged over multiple runs with the different noises randomly picked from the distribution with a fixed variance. An ablation study for the loss function should be also conducted.

**Presentation:** The paper has typos here and there. The followings are a (not exhaustive) list of typos I found:
- Standard deviation is missing across all the tables and figures.
- Line 1097. Reference environment does not work.
- Typo: at line 1112 "Eq. equation 1"
- Caption in Figure 8.
- Figures of axis in Figure 8 are strange.
- At line 1387 "Section ??"

### Questions
All the questions are included in Weakness column.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BAROM, a novel latent-space reduced-order modeling framework for simulating PDE systems with complex boundary conditions (BCs), especially when BCs are coupled with the internal state. BAROM integrates classical reduced-order theory by decomposing the solution into a boundary field and an internal field, with the following key enhancements: a learnable lifting network explicitly handles the boundary conditions, while a three-branch updater including attention mechanism models the nonlinear dynamics of the internal field. The authors evaluate BAROM alongside a broad range of SciML baselines on seven challenging PDE benchmarks with complex boundary conditions, and report that BAROM outperforms state-of-the-art baselines particularly in systems with coupled internal-boundary feedback.

### Strengths
1. The authors propose a theoretically motivated end-to-end  latent-space modeling framework with each component (learnable lift, three-branch latent updater) directly derived from and justified by the classical RBM theory, representing a significant innovation.
2. The method emphasizes the importance of explicitly handling boundary conditions when using latent-space learning for PDE systems  with complex boundary conditions.
3. The paper provides a thorough empirical validation across a diverse set of seven PDEs, covering both externally prescribed and feedback-coupled BCs. The results show significant and consistent improvements over a strong set of modern baselines.
4. The appendices are comprehensive and easy to follow, including detailed hyper-parameter settings for all baselines.

### Weaknesses
1. In a few instances on non-feedback systems (Table 2), BAROM does not uniformly rank first, with methods like GNOT or UNISOLVER performing best on some metrics, but the authors do not analyze the underlying reasons.
2. The motivation for decomposing the latent update into three separate branches (attention, FFN and boundary forcing) is presented as intuitively reasonable, but the paper offers no ablation or theoretical justification that shows this particular split is superior to simpler merged architectures.
3. The study does not include a systematic comparison with traditional or physics-based reduced-order methods (such as POD-Galerkin), which limits a comprehensive assessment of its advantages.

### Questions
1. Why does BAROM underperform GNOT/Unisolver on some non-feedback cases?
2. Is the three-branch latent update actually better than a single-branch alternative? Perhaps provide ablation and evaluate it?
3. How does BAROM compare with classic methods (such as POD-Galerkin) under the same boundary conditions?

### Soundness
2

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
This paper introduces a latent-space framework for solving PDE systems with complex boundary conditions using deep learning. The proposed model encodes high-dimensional spatial fields into a latent representation, where the PDE solution operator is efficiently learned and enforced. The method effectively handles irregular domains and mixed boundary types, achieving accurate reconstructions and generalization across diverse geometries. The authors validate the approach on benchmark PDE problems, demonstrating improved stability, computational efficiency, and adaptability compared to standard neural operator baselines.

### Strengths
1) The paper presents a novel latent-space formulation that effectively simplifies PDE learning, particularly for irregular domains and mixed boundary conditions.

2) It demonstrates strong empirical performance and generalization across various PDE types and boundary scenarios, showing versatility and robustness.

3) The approach achieves computational efficiency by learning and operating in a compressed latent domain without compromising predictive accuracy.

4) The framework is conceptually sound and has potential for scaling to more complex PDE systems in scientific and engineering contexts.

### Weaknesses
1) The selection and sensitivity analysis of the latent dimensionality are not fully explored, making it difficult to assess the optimal representation scale.

2) Comparisons with recent neural operator models such as Transolver, Latent Mamba Operator, and GeoFNO are missing, which would strengthen the empirical evaluation.

3) The paper does not clearly report training time or computational cost relative to baselines, which is important for assessing efficiency claims.

4) The boundary enforcement strategy could be discussed in greater detail to explain how accuracy is maintained near complex or irregular domain boundaries.

### Questions
1) How does the proposed latent representation handle non-smooth or discontinuous boundary conditions in practice? Have you experimented with fixed basis functions such as Fourier or wavelets, and analyzed the learned latent bases? 

2) What methodology or criteria were used to determine the latent dimension size, and how does this choice impact both accuracy and generalization?

3) Could the framework be extended to time-dependent PDEs or multi-physics coupled systems, and if so, what modifications would be necessary?

4) How does the model scale in terms of computational efficiency when compared to established operator-learning approaches like FNO or PINO, especially for large-scale domains?

5) Would incorporating spectral or operator-based regularization in the latent space further improve stability and interpretability? Additionally, have you considered adding an explicit PINN-style physics loss to guide the latent learning process?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a model called BAROM, which is a reduced-order model for PDEs with complex boundary conditions. The main idea is about decomposing the solutions into boundary and internal fields, motivated by the Reduced Basis Methods. It uses a learnable lifting network, basis functions, and a three-branch attention mechanism for latent dynamics. Experiments demonstrate that BAROM outperform many methods on several benchmarks.

### Strengths
1. The explicit boundary/initial decomposition is principled and addresses a known limitation. 

2. Evaluation is good and shows strong empirical results. 

3. Ablations and fasirness discussions are well-executed.

### Weaknesses
1. The core-architectural novelty seems limited. The proposed architecture is essentially a standard transformer plus boundary-conditioned MLP.

2. BAROM_ExpBC uses next-step boundary information while baselines don't, which seems unfair. 

3. There seems to be a gap between theory and practice: ROM requires knowing A, B matrices that are unavailable, claiming that neural operators would learn is standard UAT.  Eq. 15 involves spatial integration which isn't bypassing intrusive methods as claimed. 

4. All experiments are on 1D or 2D grids, which are small. 

5. Design choices are adhoc (why is random \phi initialization better than POD as per fig. 8 in the appendix). 

6. Unclear computatiomal efficiency claims.

### Questions
1. Why have you boldfaced BAROM_ExpBC when it uses privilaged information?

2. Why does randomly initialized \phi outperformed POD initialized \phi? 

3. Baselines don't seem to be properly tuned. E.g., BENO shows a huge decrease (400X)l in performance. 

4. How does the method scale for nx>256 or more in 3D problems?

5. Can the method achieve resolution invariance?

6. You claim Φ^T U_B provides explicit boundary enforcement, but this requires spatial integration over the domain. How is this different from implicit feature concatenation?

7. Is attention necessary in the architecture? What if it is completely removed? 

8. Why don't you compare with other ROM based methods?

9. How does the method handle truly discontinuous or singular boundary data?

10. Does the method generalize to unseen boundary condition types?

11. Do you observe stability issues in long rollouts due to error compounding?

### Soundness
2

### Presentation
2

### Contribution
2
