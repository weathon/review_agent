=== CALIBRATION EXAMPLE 89 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title**: Appropriately reflects the core contribution: using special unitary matrices (SU(2)) for rotation estimation.
- **Abstract**: Clearly states the paper's goals: reformulating Wahba's problem via SU(2), deriving linear quaternion constraints, and proposing two novel continuous rotation representations for neural networks. Claims are supported by the experiments mentioned.

### Introduction & Motivation
- **Problem Motivation**: Well-motivated, with a clear explanation of Wahba's problem and the challenges in learning rotations. The link between classical attitude estimation and modern deep learning representations is effectively drawn.
- **Contributions**: Explicitly listed and match the content of the paper. However, the intuition for *why* SU(2) is particularly useful could be emphasized earlier (the linear constraints are a key benefit).

### Method / Approach
- **Structure**: The method is divided into theoretical solutions to Wahba's problem (Section 2) and derived optimization methods/representations (Sections 3-4). This is logical.
- **Section 2 (Solutions via SU(2))**: 
    - Derivation of linear constraints in the stereographic plane (Eqs. 9, 11) and on the 3D sphere (Eqs. 17, 18) is sound, though dense. The appendices provide necessary detail.
    - The Möbius approximation (Section 2.2) is interesting but presented as an approximate solution. Its motivation and relationship to the exact solution could be clearer.
    - **Potential Gap**: The paper claims the solution to Eq. (12) (eigenvector of **G_P** with smallest eigenvalue) is equivalent to the original Wahba problem. While empirically validated (Table 4), a more explicit theoretical justification in the main text would strengthen the claim.
- **Section 3 (Optimization Methods)**:
    - Applications like residual-based optimization and constrained optimization are sensible extensions of the linear constraints.
    - The two-point solutions (weighted and unweighted) are a significant contribution. The closed-form expressions (Eqs. 21, 22) appear simpler than prior work. However, the paper should discuss their numerical stability and singularities more directly (Appendix B.4.4 and D.2 handle this, but main text should note).
- **Section 4 (Representations for Learning)**:
    - **2-vec**: A novel 6D representation based on the two-point solution. The geometric intuition (balancing error from both axis predictions vs. Gram-Schmidt) is clear and supported by gradient analysis (Fig. 4).
    - **QuadMobius**: A 16D representation based on the Möbius approximation. The construction (from network output to Hermitian matrix **G_M** to eigenvector to Möbius transformation to SU(2)) is complex. While motivated by connections to prior work (QCQP, SVD), the specific advantages of this high-dimensional, complex parameterization are not fully justified. Appendix F provides some analysis, but an ablation study in the main text (e.g., comparing to predicting SU(2) directly) would help.
    - **Derivatives**: Appendix E provides formulas for backpropagation, which is necessary for implementation.

### Experiments & Results
- **Section 5.1 (Wahba's Problem)**:
    - Synthetic experiments are thorough (1M trials). Tables 4 and 5 validate that the proposed optimal solvers match existing methods, and the two-point methods are more efficient.
    - The Möbius approximation shows higher error, as noted. This is acceptable given its role as a learning representation component.
    - **Missing**: Discussion of numerical stability and conditioning. Timing measurements should specify hardware/software environment.
- **Section 5.2 (Learning Experiments)**:
    - Benchmarks are diverse and standard (ModelNet10-SO3, Inverse Kinematics, Camera Pose). Results in Tables 1, 2, and Fig. 2 show that the proposed representations are competitive, often achieving best or second-best performance.
    - **Concerns**:
        - While results are strong, the gains over strong baselines (SVD, QCQP) are sometimes marginal. The paper should discuss statistical significance or provide error bars.
        - The training details and architectures differ per task. It is unclear if the improvements are consistent or due to task-specific tuning. A more controlled synthetic learning experiment (like Appendix G.2.2) helps, but main text should summarize key insights.
        - Computational cost: Table 7 shows QuadMobius is significantly slower. A discussion of the accuracy/speed trade-off is needed.
    - **Appendix G.2.2 (Additional Learning Experiments)**: Provides extensive ablation across noise levels, loss functions, and real/complex domains. This is a strength, showing robustness. However, the main text should highlight key takeaways (e.g., QuadMobius often leads in convergence).

### Writing & Clarity
- **Overall**: The paper is technically dense but well-structured. The heavy reliance on appendices (over 30 pages) makes the main text less self-contained, which may hinder readability for a conference paper.
- **Equations and Derivation**: Many derivations are relegated to appendices. While this keeps the main text focused, some key steps (e.g., the form of Eq. (17)) appear without intuition. The main text could benefit from more high-level explanations.
- **Figures**: Figures 1, 4, 5, 6, 7 are helpful but some captions are brief (e.g., Fig. 1(c)-(d) are "conceptual illustrations" that could be better explained).

### Limitations & Broader Impact
- **Limitations**: No dedicated section. The paper should explicitly discuss:
    - Computational complexity of QuadMobius vs. other representations.
    - Numerical stability of the two-point solutions (singularities are handled in appendix, but main text should note).
    - The Möbius approximation is not exact; its role in learning is justified empirically but theoretical guarantees are limited.
- **Broader Impact**: Not discussed. While rotation estimation itself has many positive applications (robotics, AR/VR), the paper could briefly mention societal impact (likely minimal) and limitations.

## Overall Assessment
This paper makes a solid theoretical and practical contribution to rotation estimation. The use of SU(2) to derive linear quaternion constraints is novel and leads to new closed-form solutions for Wahba's problem (especially the two-point case) and two new neural network rotation representations (2-vec and QuadMobius). The experimental validation is extensive across both classical estimation and learning tasks, showing competitive or superior performance.

**Main strengths**: Theoretical novelty, thorough derivations, comprehensive experiments (including synthetic validation and multiple benchmarks), and the introduction of efficient two-point solutions.

**Main weaknesses**: The paper is dense and heavily reliant on appendices, which may affect accessibility. The QuadMobius representation, while empirically strong, is complex and its design choices are not fully ablated. The computational cost of QuadMobius is notable, and the trade-offs are not discussed. Some experimental results, while positive, show marginal gains.

For ICLR, the paper is above the acceptance bar due to its novel theoretical foundation and strong empirical results. However, revisions to improve clarity, discuss limitations, and provide more ablation analysis for QuadMobius would strengthen it significantly. I recommend acceptance conditional on addressing the major concerns outlined above.

# Neutral Reviewer
## Balanced Review

### Summary
This paper revisits rotation estimation through the lens of special unitary matrices (SU(2)). It reformulates Wahba’s problem in complex projective space, deriving novel linear constraints on quaternion parameters that lead to efficient solutions for rotation estimation. Building on this theoretical foundation, the paper proposes two new continuous representations for learning rotations in neural networks: "2-vec" (a 6D representation) and "QuadMobius" (a 16D representation). Extensive experiments on synthetic Wahba’s problem solvers and three learning benchmarks (3D shape alignment, inverse kinematics, and camera pose estimation) demonstrate competitive performance of the proposed methods.

### Strengths
1. **Theoretical contributions**: The paper provides a novel reformulation of Wahba’s problem using SU(2) and stereographic projections, leading to linear quaternion constraints (Eqs. 11, 18). This formulation unifies the treatment of rotation estimation in both 3D and complex projective spaces, offering new insights and algorithmic alternatives.
2. **Practical algorithms**: The paper derives efficient closed-form solutions for the two-point Wahba’s problem (Eqs. 21, 22) that are simpler and computationally cheaper than existing methods (Table 5). The proposed "2-vec" representation is shown to be more efficient than Gram-Schmidt and achieves competitive results with lower dimensionality.
3. **Empirical validation**: The proposed representations are thoroughly evaluated on multiple benchmarks (ModelNet10-SO3, inverse kinematics, camera pose estimation) and show strong performance, often outperforming or matching state-of-the-art methods like SVD and QCQP (Tables 1, 2, Fig. 2). The paper also includes synthetic experiments validating the new Wahba solvers (Tables 4, 5).
4. **Clarity and organization**: The paper is well-structured, with clear derivations in the main text and detailed proofs in the appendix. The figures (e.g., Fig. 1, 4) help illustrate the key ideas and differences between methods.

### Weaknesses
1. **Limited theoretical comparison**: While the paper derives new formulations, it does not deeply compare the theoretical properties (e.g., convergence, stability) of the proposed SU(2) approach with existing SO(3) or quaternion methods. The connection to prior work (e.g., Bingham distributions) is mentioned but not fully explored.
2. **Empirical limitations**: The learning experiments, while comprehensive, are limited to a few standard benchmarks. The paper does not include ablation studies on the sensitivity of QuadMobius to hyperparameters (e.g., network architecture, loss functions) or its behavior in more challenging scenarios (e.g., large-scale datasets, noisy labels).
3. **Complexity and practicality**: The QuadMobius representation, while effective, is computationally more expensive than alternatives (Table 7) and requires complex arithmetic, which may hinder adoption. The paper does not discuss the trade-offs between performance and computational cost in depth.
4. **Reproducibility concerns**: Although the paper provides derivations and experimental settings, key implementation details (e.g., handling of edge cases in the two-point solvers, initialization of complex-valued networks) are only briefly covered. The code is not provided, and the appendix, while detailed, may still leave gaps for full replication.

### Novelty & Significance
The paper introduces a novel perspective on rotation estimation by leveraging SU(2) matrices, which are less commonly used in robotics and machine learning compared to SO(3). The derived linear quaternion constraints and the resulting algorithms for Wahba’s problem are novel and offer computational advantages in certain cases. The proposed "2-vec" and "QuadMobius" representations are innovative contributions to the field of learning rotations, providing continuous, over-parameterized mappings that improve gradient flow and performance. The work is significant as it bridges theoretical rotation estimation with practical deep learning applications, offering new tools for a fundamental problem.

### Suggestions for Improvement
1. **Provide more theoretical analysis**: Compare the proposed SU(2) formulation with existing methods in terms of geometric interpretation, robustness to noise, and convergence properties. Discuss the relationship with Bingham distributions and other probabilistic models in more detail.
2. **Expand empirical evaluation**: Include more diverse tasks (e.g., robotic manipulation, SLAM) and datasets to demonstrate broader applicability. Conduct ablation studies on QuadMobius to understand the impact of its components (eigendecomposition, Möbius transformation, projection) and its sensitivity to hyperparameters.
3. **Address computational efficiency**: Discuss strategies to reduce the computational overhead of QuadMobius (e.g., approximation of eigendecomposition, use of real-valued equivalents) and provide a more thorough cost-benefit analysis compared to other representations.
4. **Improve reproducibility**: Release code and models to facilitate replication. Provide more explicit pseudocode or algorithm descriptions for the two-point solvers and the backpropagation through QuadMobius, especially for edge cases (e.g., degenerate inputs, singular matrices).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Statistical significance testing is missing.** The paper reports mean/median errors but provides no variance measures, confidence intervals, or statistical tests (e.g., paired t-tests) to confirm that performance differences between methods are significant. Without this, it is impossible to judge whether the reported improvements are meaningful or due to random chance.
2. **Comparison to recent, strong baselines is incomplete.** The experiments omit comparisons to several modern, high-performing rotation regression methods (e.g., ProHMR, Procrustes-based approaches, or the Bingham loss) that have been shown to work well on tasks like ModelNet-SO3 and camera pose estimation. This gap undermines the claim that the proposed methods are state-of-the-art.
3. **Insufficient ablation of the QuadMobius representation.** The paper does not ablate key design choices: Why use the eigenvector of the *smallest* eigenvalue? What is the impact of the intermediate Möbius transformation normalization step? An ablation comparing the SVD and algebraic projection variants, and testing the sensitivity to the eigenvalue selection, is needed to validate the core design.
4. **No experiment on the real impact of the theoretical SU(2) formulations.** The paper derives new solutions to Wahba's problem but only validates them in synthetic, noise-added scenarios. A critical test is missing: apply these solvers in a real iterative optimization loop (e.g., in a bundle adjustment or SLAM context) against standard solvers to demonstrate practical utility beyond synthetic checks.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of gradient flow and learning dynamics for QuadMobius is superficial.** Figure 7 and the associated text in Appendix F provide some gradient magnitude plots, but a rigorous analysis is missing. How do the gradients through the complex eigendecomposition and SVD compare in terms of variance and bias? Does the representation suffer from gradient explosion/vanishment in deep networks? This is critical for trust in a learnable representation.
2. **No clear analysis of the approximation error in the Möbius method.** Section 2.2 presents an *approximate* solution. The paper should quantify the approximation error theoretically or empirically as a function of noise level and number of points, explaining when and why this approximation might be beneficial or detrimental for learning.
3. **Lack of discussion on the double-cover ambiguity and its handling.** The paper briefly mentions (in Appendix F) that directly predicting *SU*(2) performs poorly due to double-cover ambiguity, but does not analyze how QuadMobius avoids or mitigates this issue. A clear explanation of how the mapping from **G_M** to a rotation resolves the ambiguity is necessary.
4. **The singular/degenerate cases for the proposed 2-point solutions are not thoroughly analyzed.** While Appendix B.4.4 and D.2 discuss degenerate cases, there is no empirical evaluation showing the failure rate or error behavior of the proposed robust selection scheme under near-degenerate configurations (e.g., nearly collinear points).

### Visualizations & Case Studies
1. **Visualizations of failure cases for the learned representations.** The paper shows only aggregated metrics. Visual case studies are needed: for example, show input images from ModelNet-SO3 or Cambridge Landmarks where 2-vec or QuadMobius produce large errors, and contrast with the predictions of baselines. This would expose systematic weaknesses.
2. **Visual demonstration of the "balanced gradient" claim for 2-vec.** Figure 4 shows a density plot, but a more intuitive visualization is needed. For example, visualize the Gram-Schmidt and 2-vec mappings for a set of perturbed 6D inputs on a 2D manifold, showing how the output rotation changes, to geometrically illustrate the improved gradient behavior.
3. **T-SNE/PCA visualization of the learned high-dimensional space for QuadMobius.** Since QuadMobius uses a 16D input Θ, visualizing how the network organizes this space (e.g., clustering by rotation type) could provide insight into why it works better than lower-dimensional representations.

### Obvious Next Steps
1. **Incorporate uncertainty estimation.** Given that QuadMobius is motivated by connections to Bingham distributions (mentioned briefly), a direct and evaluated method to extract prediction uncertainty (e.g., from the eigenvalues of **G_M**) should have been included. This is a standard expectation for learning rotation distributions.
2. **Apply the SU(2)-based Wahba solvers to a real-world sensor fusion task.** The paper should have included a small but real experiment (e.g., fusing IMU and visual data for attitude estimation) to demonstrate the practical advantage of the new linear constraint formulations (Eqs. 11, 18) in a residual-based optimization, as claimed in Section 3.1.
3. **Benchmark computational cost in an end-to-end learning pipeline.** Table 7 provides isolated timings, but the paper should analyze the total training time and memory footprint for each representation on a standard benchmark, discussing the trade-off between accuracy and efficiency for deployment.
4. **Provide an open-source, easy-to-use implementation of the solvers and layers.** For the community to adopt these methods, a well-tested PyTorch/TensorFlow implementation of the 2-vec and QuadMobius layers, along with the new Wahba solvers, is essential. The paper only mentions reimplementations in C++ for timing.

# Final Consolidated Review
## Summary
This paper revisits rotation estimation through the lens of special unitary matrices (SU(2)). It reformulates Wahba's problem in complex projective space, deriving linear constraints on quaternion parameters that yield new efficient solutions, including closed-form two-point methods. Building on this foundation, the paper proposes two novel continuous rotation representations for neural networks: 2-vec (6D) and QuadMobius (16D). Extensive experiments validate the methods on synthetic Wahba problems and multiple learning benchmarks.

## Strengths
- **Theoretical novelty:** The SU(2) reformulation unifies rotation estimation in 3D and complex projective space, producing linear quaternion constraints (Eqs. 11, 18) that enable new efficient algorithms and insights.
- **Algorithmic contributions:** The paper derives simplified, closed-form solutions for the two-point Wahba problem (Eqs. 21, 22) that are computationally more efficient than existing methods (Table 5).
- **Empirical validation:** The proposed representations are thoroughly evaluated on multiple benchmarks (ModelNet10-SO3, inverse kinematics, camera pose estimation) and consistently show competitive or superior performance against strong baselines (Tables 1, 2, Fig. 2). Additional synthetic experiments (Tables 4, 5, 6) comprehensively validate the solvers and learning behavior.

## Weaknesses
- **Lack of statistical significance reporting:** The learning experiments report mean/median errors but no variance measures, confidence intervals, or statistical tests. Without these, it is difficult to assess whether the performance differences are meaningful or due to random variation.
- **Incomplete ablation of QuadMobius:** While Appendix F provides some analysis, the paper does not systematically ablate key design choices of the QuadMobius representation (e.g., the role of the smallest eigenvalue, the impact of the Möbius normalization step) to validate the necessity of its components.
- **Computational cost trade-off not discussed:** QuadMobius is significantly slower in both training and inference than other representations (Table 7), but the paper does not discuss the accuracy-versus-speed trade-off, which is important for practical deployment.

## Nice-to-Haves
- Extend the evaluation to more diverse tasks (e.g., robotic manipulation, SLAM) to demonstrate broader applicability.
- Provide a more thorough theoretical comparison of the SU(2) formulation with existing methods in terms of geometric interpretation and robustness.
- Release code and models to facilitate replication and adoption.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"The paper is dense and heavily reliant on appendices."** This is a stylistic preference; the paper is well-structured and the appendices appropriately contain detailed derivations.
- **"The Möbius approximation lacks theoretical guarantees."** The paper explicitly notes it is an approximation and empirically shows its error (Table 4); its role in learning is justified by the results.
- **"Numerical stability of the two-point solutions is not addressed."** The paper handles singular/degenerate cases in Appendix B.4.4 and D.2.
- **"Gradient flow analysis for QuadMobius is superficial."** The paper provides gradient analysis in Appendix F (Figs. 6, 7), which is more than many papers offer.
- **"Approximation error of the Möbius method is not quantified."** Table 4 shows the error, and the paper discusses its sensitivity to noise.
- **"Double-cover ambiguity is not analyzed."** Appendix F explains why direct SU(2) prediction fails and how QuadMobius avoids the issue.
- **"Visualizations of failure cases or T-SNE plots are missing."** While insightful, these are not required for validation.
- **"Uncertainty estimation should be incorporated."** This is outside the paper's scope on deterministic rotation estimation.
- **"Real-world sensor fusion experiments are missing."** The paper's contributions are primarily theoretical and focused on learning representations, not on applied sensor fusion.
- **"End-to-end pipeline computational cost benchmarking."** Table 7 already provides isolated timings; deeper pipeline analysis is not essential.
- **"Open-source implementation is not provided."** Code release is encouraged but not a requirement for publication.

## Novel Insights
The paper's key novel insight is that reformulating Wahba's problem via SU(2) and stereographic projection yields linear constraints on quaternion parameters. This perspective enables efficient closed-form solutions (e.g., for the two-point case) and inspires new continuous rotation representations. Specifically, 2-vec provides a more balanced gradient flow than Gram-Schmidt by optimally combining two axis predictions, while QuadMobius leverages a high-dimensional complex eigendecomposition to create a stable intermediate representation that buffers against poor inputs and ensures predictable gradient flow.

## Suggestions
- Conduct statistical significance tests (e.g., paired t-tests) for the learning experiments to substantiate the reported performance differences.
- Perform an ablation study on QuadMobius to validate the necessity of its components (e.g., the choice of the smallest eigenvalue, the Möbius transformation, and the projection step).
- In the main text, discuss the accuracy-speed trade-off of QuadMobius given its higher computational cost, and suggest potential optimizations or use cases where the trade-off is justified.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 10.0]
Average score: 8.5
Binary outcome: Accept
