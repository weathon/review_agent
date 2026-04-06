=== CALIBRATION EXAMPLE 86 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's focus on using special unitary matrices (SU(2)) for rotation estimation. The abstract clearly states the main contributions: reformulating Wahba's problem via SU(2), deriving linear constraints on quaternion parameters, and proposing two novel continuous representations for learning rotations. The claims of extensive experimental validation are supported in the later sections.

### Introduction & Motivation
The introduction effectively motivates the problem, highlighting the ubiquity of 3D rotations and the under-explored potential of SU(2) in machine learning and robotics. It provides a concise background on Wahba's problem and the challenges of learning rotations in neural networks (e.g., discontinuities in minimal representations). The contributions are clearly listed, setting appropriate expectations. The recommendation to review Appendix A is reasonable given the heavy mathematical reliance.

### Method / Approach
This is the core of the paper and is mathematically dense. The derivations appear sound, but the presentation is highly technical and assumes substantial familiarity with complex projective geometry, SU(2), and Möbius transformations. While the appendix provides detailed proofs, the main text could benefit from more intuitive explanations to improve accessibility.

**Key concerns:**
1. **Clarity and Reproducibility:** The step-by-step derivations are provided, but the complexity may hinder implementation. The mapping between SU(2), quaternions, and SO(3) is clearly defined, but the numerous equations and transformations (e.g., Eq. (45) for recovering **R**) make it easy to lose the thread. More narrative guidance would help.
2. **Logical Gaps:** The transition from the theoretical solutions to the proposed learning representations (2-vec and QuadMobius) is somewhat abrupt. For 2-vec, the connection to the two-point unweighted solution is clear, but the geometric intuition behind Eq. (23) could be expanded. For QuadMobius, the motivation for using a 16D output to parameterize **G_M** is not thoroughly justified beyond inspiration from prior work.
3. **Assumptions and Edge Cases:** The methods assume non-singular inputs (e.g., **M** non-singular for QuadMobius, **a**₁ × **a**₂ ≠ 0 for two-point solutions). While the appendix discusses some degenerate cases (Appendix B.4.4, D.2), the practical handling of these in learning scenarios is not addressed in the main text. The differentiability of the proposed maps is asserted and supported in Appendix E, but a discussion of potential gradient instability (e.g., near singularities) is missing.

### Experiments & Results
The experimental section is extensive and generally supports the claims. However, some aspects require deeper scrutiny.

**Wahba's Problem Validation (Sec. 5.1, Tables 4 & 5):**
- The proposed optimal solvers (**G_P**, **G_S**) match the accuracy of established methods (Q-method, QUEST), as expected from equivalent formulations. The reported timing differences are noted but not analyzed in depth (e.g., why is **G_P** slower than Q-method?).
- The Möbius approximation (**G_M**) shows significant sensitivity to noise, which the authors acknowledge. This raises questions about its utility outside the learning context, where noise is inherent.
- The two-point solutions are shown to be computationally more efficient (Table 5), a valuable contribution. The derivation of the weighted average of unnormalized quaternions (Appendix B.4.3) is elegant.

**Learning Experiments (Sec. 5.2, Tables 1, 2, Fig. 2, Table 6):**
- The benchmarks (ModelNet10-SO3, Inverse Kinematics, Camera Pose Estimation) are appropriate and commonly used. The proposed representations, especially QuadMobius variants, often achieve top or competitive performance.
- **However, the experimental scope is limited:** Only three object categories from ModelNet10-SO3 are used, following prior work but limiting generalizability. The inverse kinematics and camera pose experiments are on single datasets.
- **Missing Ablations:** While Table 6 (synthetic learning of Wahba's problem) provides a more controlled comparison, it lacks ablations on why QuadMobius works well. The theoretical investigations in Appendix F (gradient analysis, dropout sensitivity) are insightful but preliminary. A deeper analysis of the representation's properties (e.g., the effect of the 16D parameterization, the role of the eigendecomposition) would strengthen the paper.
- **Statistical Significance and Reporting:** Results are reported as mean/median errors, but no measures of variance or statistical significance tests are provided. The "leader count" (Ldr.) in Table 6 is an interesting convergence metric but is not standard and its calculation is ambiguous (how is a "leader" defined per epoch?).
- **Comparison to Baselines:** The baselines are well-chosen (Euler, Quat, GS, QCQP, SVD). The consistent outperformance of QuadMobius and the strong showing of 2-vec (often beating Gram-Schmidt) are convincing.

### Writing & Clarity
The paper is structurally sound but extremely dense. The heavy reliance on mathematical notation and derivations, while necessary, makes it challenging to read. Key insights are sometimes buried in equations. Figures 1, 4, 5, 6, and 7 are helpful, but their captions could more explicitly connect to the main claims. The appendix is massive (33 pages of content), suggesting that much of the critical detail is relegated to supplemental material, which may hinder accessibility.

### Limitations & Broader Impact
The paper does not have a dedicated limitations section. Important limitations that should be explicitly acknowledged include:
1. **Complexity and Accessibility:** The mathematical complexity of the methods may limit adoption.
2. **Computational Cost:** QuadMobius is significantly slower in inference (Table 7), which may be prohibitive for real-time applications.
3. **Learning-Specific Concerns:** The sensitivity of the Möbius approximation to noise might be a double-edged sword in learning; this trade-off is not discussed. The behavior of the representations in very high-noise regimes or with adversarial examples is unexplored.
4. **Scope of Experiments:** As noted, the learning experiments are on a relatively narrow set of tasks.
Broader societal impact is not discussed, which is acceptable for a technical paper of this nature.

### Overall Assessment
This paper presents a novel and theoretically rigorous exploration of rotation estimation using SU(2). The core contributions—new solutions to Wahba's problem and two novel neural network rotation representations—are substantiated by derivations and experiments. The 2-vec representation is a simple, efficient alternative to Gram-Schmidt, and QuadMobius demonstrates strong empirical performance. However, the paper's impact is tempered by its high mathematical density and somewhat limited experimental breadth. The work is suitable for ICLR if the authors can improve clarity and provide a more thorough empirical analysis. Key revisions should include: (1) a more accessible presentation of the core ideas, (2) expanded experiments (more datasets, ablations), (3) explicit discussion of limitations, and (4) proper statistical reporting.

# Neutral Reviewer
## Balanced Review

### Summary
This paper revisits the classic Wahba's problem in attitude estimation through the lens of special unitary matrices (SU(2)). It derives new, linear quaternion constraints by reformulating the problem in complex projective space, leading to efficient solvers and closed-form solutions for the two-point case. Building on this theory, the paper introduces two novel continuous representations for learning 3D rotations in neural networks: "2-vec" and "QuadMobius". These are evaluated on several standard benchmarks, demonstrating competitive or superior performance against established baselines.

### Strengths
1. **Novel Theoretical Formulation**: The paper provides a fresh, principled derivation of Wahba's problem using SU(2) and complex projective geometry. The reformulation yields linear constraints on quaternion parameters (Eqs. 11, 18), which is a distinct and theoretically interesting approach compared to traditional methods.
2. **Practical Algorithmic Contributions**: The derived constraints enable efficient solutions for Wahba's problem (matching optimal solvers in Table 4) and lead to simplified, closed-form solutions for the two-point case (Eqs. 21, 22), which offer computational advantages (Table 5).
3. **Effective New Learning Representations**: The proposed "2-vec" and "QuadMobius" representations are well-motivated from the theoretical framework. "2-vec" provides a balanced alternative to Gram-Schmidt with more stable gradients (Fig. 4), while "QuadMobius" shows strong empirical performance across multiple diverse benchmarks (Tables 1, 2, Fig. 2).
4. **Comprehensive Experimental Validation**: The paper validates the theoretical solvers on synthetic Wahba's problems and evaluates the learning representations on three established tasks (ModelNet10-SO3, inverse kinematics, camera pose estimation). The results are thorough and demonstrate the versatility and competitiveness of the proposed methods.
5. **Strong Supplementary Material**: The appendix provides detailed derivations, proofs, and additional experiments (e.g., gradient analysis, ablation studies), which greatly enhance reproducibility and understanding of the technical contributions.

### Weaknesses
1. **Limited Comparison to Recent Learning Methods**: While comparisons are made to SVD, QCQP, and Gram-Schmidt, the paper does not situate its learning representations within the broader, rapidly evolving landscape of rotation learning (e.g., more recent works on Procrustes, Riemannian, or implicit representations from the last 2-3 years). This makes it harder to assess the current significance of the contributions.
2. **Theoretical Justification for QuadMobius as a Learning Proxy**: The connection between the Möbius approximation (Section 2.2) and the final "QuadMobius" learning representation (Section 4) feels somewhat heuristic. While the empirical results are strong, a more rigorous justification for why this specific 16D parameterization and projection is effective for learning is lacking.
3. **Clarity and Accessibility of Complex Derivations**: The heavy reliance on complex numbers and projective geometry, while novel, makes the core theoretical sections (Section 2) difficult to follow for a general machine learning audience. Key intuitions are sometimes buried in algebraic derivations.
4. **Computational Cost of QuadMobius**: The "QuadMobius" representation requires a 16D output and complex eigendecomposition/SVD (Table 7), making it significantly more expensive than lower-dimensional representations. The paper does not deeply discuss the trade-off between this cost and the performance gains, which are sometimes marginal.
5. **Incomplete Discussion of Limitations**: The paper does not fully discuss the failure modes or limitations of the new representations (e.g., singularities for "2-vec", behavior of "QuadMobius" when `det(M)` is near zero, or the sensitivity of the Möbius approximation to noise noted in Table 4).

### Novelty & Significance
The theoretical reformulation of Wahba's problem using SU(2) is novel and provides a new perspective that yields practical algorithmic benefits, particularly for the two-point case. The translation of this theory into novel neural network representations ("2-vec" and "QuadMobius") is a creative and significant application. The empirical results are solid, showing that these representations can outperform or match state-of-the-art continuous rotation representations on several tasks. The work is a valuable contribution to the field of rotation estimation and learning. However, its impact is partially limited by the complexity of the derivations and the lack of comparison to the very latest learning techniques.

### Suggestions for Improvement
1. **Expand Related Work**: Add a subsection or paragraph discussing more recent advances in rotation representation learning (post-2020) to better contextualize where "2-vec" and "QuadMobius" stand in the current literature.
2. **Strengthen the Learning Motivation**: Provide a more intuitive, geometric, or probabilistic motivation for the "QuadMobius" representation. For instance, can the 16D parameter `Θ` be interpreted as encoding uncertainty or a distribution over transformations?
3. **Improve Exposition**: To improve clarity, consider adding a high-level schematic or algorithm box summarizing the key steps of the SU(2) formulation for Wahba's problem. Move some of the more intricate algebraic manipulations (e.g., Appendix B.1.1) entirely to the appendix and replace them with intuitive explanations in the main text.
4. **Analyze Complexity-Accuracy Trade-offs**: Include a more explicit discussion and experiment analyzing the inference/training time versus accuracy trade-off for "QuadMobius" compared to other representations. This is crucial for practitioners.
5. **Discuss Failure Cases and Robustness**: Explicitly discuss and, if possible, experimentally validate the robustness of the proposed methods to edge cases (e.g., nearly collinear vectors for "2-vec", ill-conditioned `G_M` for "QuadMobius"). Suggest potential mitigation strategies.
6. **Release Code**: To ensure reproducibility and foster adoption, the authors should commit to releasing well-documented code for all proposed methods and experiments.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison to recent state-of-the-art rotation representations in learning tasks.** The paper does not compare against recent, strong baselines like the Projective Manifold Gradient Layer (Chen et al., 2022) or comprehensive frameworks like Geist et al. (2024). Without these, the claim that the new representations are competitive is not fully substantiated.

2. **Missing controlled ablation on the QuadMobius components.** The paper does not isolate the impact of the Möbius transformation approximation versus the eigendecomposition step. An ablation (e.g., predicting an SU(2) matrix directly vs. full QuadMobius) is needed to justify the complexity of the proposed pipeline.

3. **No evaluation of the two-point methods in a practical robust estimation pipeline.** The paper derives closed-form solutions for the two-point case but does not integrate them into a RANSAC or outlier-robust framework to demonstrate real-world efficiency gains over existing solvers.

4. **Lack of systematic noise/outlier analysis in learning benchmarks.** The learning experiments report performance on clean datasets. There is no analysis of how the representations degrade with increasing label noise or outliers, which is critical for assessing robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical characterization of the Möbius approximation error.** The paper states Section 2.2 is an approximation but provides no bounds on the error or analysis of when it coincides with the optimal solution. This makes it impossible to assess the reliability of the method.

2. **Thorough gradient flow and optimization landscape analysis for the new representations.** While Appendix F includes a limited gradient analysis, a comprehensive study (e.g., gradient norms, conditioning, loss surfaces) comparable to Zhou et al. (2019) is missing. This is essential to trust the learning claims.

3. **Analysis of singularities and failure modes for 2-vec and QuadMobius.** The paper mentions singular regions but does not quantify how often they occur in practice during training or inference, nor their impact on convergence and stability.

### Visualizations & Case Studies
1. **Visualization of the Möbius approximation error.** For the Wahba problem, plots showing the distribution of error between the optimal rotation and the Möbius approximation under different noise levels would reveal when the approximation breaks down.

2. **t-SNE visualizations of the learned representation spaces.** For the learning tasks, projecting the network outputs (e.g., the 16D Θ for QuadMobius) would show if the representations structure the latent space meaningfully compared to baselines.

3. **Case studies of specific failure instances.** Showing concrete examples where the proposed methods underperform compared to baselines (e.g., on particular ModelNet objects or camera poses) would help identify limitations.

### Obvious Next Steps
1. **Incorporate the two-point solvers into a RANSAC framework for camera pose estimation.** This is a direct application mentioned in the text but not implemented. It would demonstrate practical utility.

2. **Extend QuadMobius to model uncertainty.** The paper notes a link to Bingham distributions but does not implement uncertainty estimation. Outputting a distribution over rotations is a natural and impactful extension.

3. **Apply the methods to direct point cloud registration tasks.** The theoretical formulations are geometric; testing on registration (e.g., ICP variants) would broaden the impact beyond learning-from-images.

# Final Consolidated Review
## Summary
This paper revisits rotation estimation through the lens of special unitary matrices (SU(2)). It reformulates Wahba's problem in complex projective space to derive linear constraints on quaternion parameters, leading to efficient solvers and novel closed-form solutions for the two-point case. Building on this theory, the paper introduces two new continuous representations for learning rotations in neural networks: "2-vec," a balanced 6D alternative to Gram-Schmidt, and "QuadMobius," a higher-dimensional representation based on Möbius transformations.

## Strengths
- **Novel Theoretical Formulation**: The paper provides a principled, fresh derivation of Wahba's problem using SU(2) and complex projective geometry, yielding linear quaternion constraints (Eqs. 11, 18). This theoretical perspective is distinct from traditional approaches.
- **Practical Algorithmic Contributions**: The derived constraints enable efficient optimal solvers for Wahba's problem and lead to simplified, closed-form solutions for the two-point case (Eqs. 21, 22), which offer computational advantages, as validated in Tables 4 and 5.
- **Effective New Learning Representations**: The proposed "2-vec" and "QuadMobius" representations are well-motivated from the theory. "2-vec" demonstrates more balanced gradient flow than Gram-Schmidt (Fig. 4), and "QuadMobius" shows strong, often state-of-the-art, empirical performance across multiple diverse benchmarks (3D shape alignment, inverse kinematics, camera pose estimation) in Tables 1, 2, and Figure 2.

## Weaknesses
- **Incomplete Empirical Contextualization**: While the paper compares to established baselines (SVD, QCQP, Gram-Schmidt), it does not include empirical comparisons to very recent state-of-the-art rotation learning methods (e.g., Chen et al., 2022, cited but not compared against). This makes it difficult to fully assess the current significance of the proposed representations.
- **Justification for QuadMobius as a Learning Proxy is Empirical**: The connection between the Möbius approximation (Section 2.2) and the final "QuadMobius" learning representation (Section 4) is motivated by prior work and shown to work well, but a more rigorous theoretical or intuitive justification for why this specific 16D parameterization is particularly effective for learning is lacking.
- **Computational Trade-off Underexplored**: "QuadMobius" is significantly more computationally expensive in inference than lower-dimensional representations (Table 7). The paper notes this but does not provide a substantive discussion or analysis of the performance-versus-cost trade-off, which is important for practitioners.

## Nice-to-Haves
- A more explicit discussion of potential failure modes or robustness limits (e.g., behavior when `det(M)` is near zero for QuadMobius), even if preliminary mitigation strategies are outlined in the appendix.
- An expanded experiment on the ModelNet10-SO3 benchmark using more object categories to strengthen the claim of generalizability beyond the three presented.

## Novel Insights
The paper's core novel insight is the application of the SU(2) representation—common in physics but underused in robotics and ML—to rotation estimation. This perspective yields linear quaternion constraints, which in turn inspire new learning representations. Beyond the performance results, the analysis reveals insightful properties: "2-vec" provides more stable gradients than Gram-Schmidt by design (Fig. 4), and "QuadMobius" creates a resilient intermediate representation (the Möbius transformation) that buffers against input corruption while enabling strong gradient flow (Figs. 6, 7). These observations connect the theoretical formulation to tangible benefits in neural network optimization.

## Suggestions
- Include empirical comparisons to recent strong baselines (e.g., Projective Manifold Gradient Layer) in the learning experiments to better situate the performance of the proposed representations.
- Conduct and report a controlled ablation study for "QuadMobius" (e.g., comparing the full pipeline to directly predicting an SU(2) matrix) to isolate the contribution of its components.
- Provide a more explicit discussion in the main text analyzing the accuracy versus inference-time trade-off presented by the "QuadMobius" representation.
- Commit to releasing well-documented code for all proposed methods and experiments to ensure reproducibility and foster adoption.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 10.0]
Average score: 8.5
Binary outcome: Accept
