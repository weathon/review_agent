=== CALIBRATION EXAMPLE 67 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title accurately reflects the core contribution: an efficient, differentiable contact model that provides long-range gradient influence. The abstract clearly states the problem (erratic gradients in differentiable simulators), the proposed solution (a set of properties and a model satisfying them), and the experimental outcome (discovery of complex control signals from trivial initializations). All claims are supported in the paper.

**Introduction & Motivation**  
The introduction effectively motivates the need for improved contact models in differentiable physics, citing recent work that highlights gradient pitfalls. The contributions are clearly stated: (1) a set of theoretical properties for well-behaved contact models, and (2) a practical, efficient model satisfying these properties. The scope (rigid bodies, triangle meshes) is explicitly defined.

**Method / Approach**  
*Section 3: Properties* – The four properties (Barrier-Form, Smoothness, Non-prehensile, Non-vanishing) are well-defined and intuitively explained with Figure 2. Property 3.3 (Non-prehensile & Non-vanishing) is somewhat intricate but necessary to formalize long-range influence. Table 1 effectively summarizes prior work against these properties.

*Section 4: Contact Potential* – The potential based on separating planes (Equation 3) is a natural extension of prior work (Liang et al., 2024; Ye et al., 2025). The key modification is using a globally supported barrier (1/(●)⁺) instead of a locally supported one to achieve non-vanishing gradients. Lemma 4.1 claims this potential satisfies all properties; the proof is deferred to Appendix A.1 and appears sound. However, the nested minimization over \(p_{ij}\) introduces a computational burden and requires solving a 4D convex optimization per triangle pair. The authors mention Newton’s method, but the practical implementation details (e.g., initialization, convergence tolerance) are sparse; this could affect reproducibility.

*Section 5: Efficient Evaluation* – The use of a Bounding Sphere Hierarchy (BSH) and smooth blending (Equation 4) to reduce complexity is inspired by N-body methods. Theorem 5.4 states the hierarchical potential retains all properties. The complexity analysis (Appendix A.2) shows linear cost for a uniform grid, but general-case complexity is not analyzed; the efficiency claim for arbitrary meshes relies on empirical performance. The blending parameter \(\epsilon\) controls the transition between exact and approximated potentials; sensitivity to \(\epsilon\) is not studied, though it likely trades accuracy for speed.

**Experiments & Results**  
*Physics Accuracy* – The book stacking test (Figure 4, Table 2) validates that the model produces physically plausible results (stable stacking) for sufficiently small \(\mu\). However, the contact force error is reported only for a single \(\mu\); a broader sweep would strengthen the claim of convergence to exact contact as \(\mu \to 0\).

*Benchmark Tasks* – The five tasks (Billiards, Push, Ant-Push, Sort, Gather) are diverse and contact-rich. The results (Figures 5-9) clearly demonstrate that the proposed model enables gradient-based optimization from trivial initializations where baselines fail due to vanishing gradients. The comparison with IPC, SDRS, MuJoCo, and Gradient Bundle is fair, as these represent state-of-the-art alternatives. However, the experiments lack statistical robustness: each task appears to be run once, and variance across multiple seeds is not reported. This is important because gradient-based optimization can be sensitive to initialization and hyperparameters.

*Ablation Study* – Appendix A.6.3 includes an ablation on Smoothness (comparing to a modified IPC with global support), showing slower convergence without twice-differentiability. This supports the necessity of Smoothness. However, an ablation on the long-range property (e.g., by truncating the potential) would further isolate its benefit.

*Computational Efficiency* – Table 6 (in the appendix, not shown in the excerpt) compares per-frame simulation time; the BSH-based method is significantly faster than brute-force and competitive with IPC. However, the overhead of constructing and traversing the BSH, as well as solving the nested optimizations, is not broken down. The claim of “efficiency” is supported but could be more thoroughly analyzed.

**Writing & Clarity**  
The paper is generally well-written, but some sections require careful reading. The definition of Property 3.3 is dense and may be challenging to parse; a more intuitive explanation would help. The transition from the exact potential (Equation 3) to the hierarchical approximation (Section 5) is somewhat abrupt; a high-level overview of the algorithm (e.g., a pseudocode) in the main text would improve clarity. The appendix contains essential details (e.g., proofs, algorithm description, friction model), which is appropriate for an archival paper.

**Limitations & Broader Impact**  
The conclusion acknowledges two limitations: (1) the model is restricted to rigid bodies (not deformable), and (2) the recursive definition and nested optimizations incur overhead. Additional limitations include: the assumption of triangle meshes (no other primitives), the need to tune \(\mu\) and \(\epsilon\), and the lack of a theoretical complexity guarantee for general meshes. Broader impact is not discussed; the work enables better gradient-based optimization for robotics and graphics, which is positive, but potential misuse (e.g., in adversarial physical design) is not considered. Given the technical nature, this omission is acceptable.

### Overall Assessment
The paper presents a novel differentiable contact model that addresses a significant problem in differentiable simulation: erratic gradients. The theoretical contribution (four essential properties) is well-founded, and the practical model satisfies them via a carefully designed potential and efficient hierarchical evaluation. Experiments demonstrate clear advantages over baselines in contact-rich control tasks. The main concerns are the limited empirical robustness (single runs, no variance) and the need for more detailed analysis of computational efficiency and parameter sensitivity. Nonetheless, the core ideas are sound and the contribution is substantial, likely meeting ICLR’s standards for novelty, technical quality, and potential impact. With minor revisions (e.g., reporting multiple runs, clarifying the algorithm), the paper would be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a novel differentiable contact model for rigid-body simulators designed to provide well-behaved gradient information. Theoretically, it defines four key properties (Barrier-Form, Smoothness, Non-prehensile, and Non-vanishing) that a contact potential must satisfy to ensure stable and informative gradients, even for distant objects. Practically, it proposes an efficient implementation using a Bounding Sphere Hierarchy (BSH) to achieve near-linear computational cost. Experiments on contact-rich robotic tasks demonstrate that the model enables gradient-based optimizers to discover complex control policies from trivial initializations where prior methods suffer from vanishing or erratic gradients.

### Strengths
1. **Clear Theoretical Contribution**: The paper provides a well-motivated set of properties (Barrier-Form, Smoothness, Non-prehensile, Non-vanishing) that formally characterize a "well-behaved" contact model. This framework helps explain why prior models fail and guides the design of the proposed solution.
2. **Comprehensive Empirical Validation**: The experiments cover a diverse set of challenging contact-rich tasks (manipulation and locomotion), showing consistent improvement over strong baselines (IPC, SDRS, MuJoCo, Gradient Bundling). The ablation study (e.g., Table 2 on physical accuracy, Figure 13 on Smoothness) effectively isolates the impact of key properties.
3. **Practical Efficiency**: The BSH-based hierarchical evaluation scheme is a necessary and well-explained optimization. The complexity analysis for a uniform grid (Appendix A.2) demonstrates near-linear scaling, and the reported timings show substantial speedups over brute-force computation.

### Weaknesses
1. **Limited Scope to Rigid Bodies**: The method is explicitly limited to rigid bodies, as acknowledged in the conclusion. The BSH construction assumes non-deforming geometry, preventing application to soft robots or deformable object manipulation, which are important areas for differentiable simulation.
2. **Computational Overhead and Implementation Complexity**: Despite the BSH acceleration, the model requires solving a nested 4D optimization problem per triangle pair and a recursive blending scheme. The per-frame cost, while better than brute-force, remains non-negligible (e.g., ~3.89s for Gather-Bunny), and the implementation intricacy could hinder adoption.
3. **Sensitivity to Hyperparameters**: The contact model's physical accuracy and gradient behavior depend critically on the barrier coefficient \( \mu \) and the blending margin \( \epsilon \). While some analysis is provided (Table 2, Table 5), a more systematic study of their influence on optimization performance would strengthen the practical guidance.

### Novelty & Significance
**Novelty** is high. The identification of the *Non-vanishing* property as critical for long-range gradient influence is a key insight that distinguishes this work from prior differentiable contact models (e.g., IPC, SDRS). The formulation of a contact potential that provably satisfies all four properties via a globally supported barrier and hierarchical blending is novel.

**Clarity** is generally good. The paper is well-structured, and the illustrative figures (Fig. 1, 2, 3) help convey the core ideas. However, Section 3.2 and the property definitions are quite dense; a more intuitive explanation alongside the formal statements would improve accessibility.

**Reproducibility** is moderately supported. The appendix includes pseudocode (Algorithm 1), proof sketches, and experimental parameters (Table 3). However, the full implementation of the simulator and the BSH construction is non-trivial. Releasing code would greatly enhance reproducibility.

**Significance** is substantial for the fields of robotics, graphics, and machine learning where differentiable simulation is crucial. By providing reliable gradients over long distances, the model can enable more effective gradient-based optimization for contact-rich planning and control tasks, potentially reducing the need for global search heuristics.

### Suggestions for Improvement
1. **Extend Discussion on Limitations**: Expand the conclusion to discuss more concretely the challenges of extending the method to deformable bodies. Could a dynamically updated BSH or a different bounding primitive be a path forward?
2. **Simplify Theoretical Presentation**: Consider adding a more intuitive, narrative explanation of the four properties and their necessity in Section 3.2 before diving into formal definitions. A table (like Table 1) earlier in the paper could help orient the reader.
3. **Provide More Implementation and Parameter Details**: While Table 3 lists parameters, a discussion on how to select \( \mu \) and \( \epsilon \) for new tasks would be valuable for practitioners. Additionally, open-sourcing the simulator code would be a major contribution to the community.
4. **Strengthen Baseline Comparisons**: Include a more detailed runtime breakdown (e.g., time spent in contact potential vs. other simulator components) to better contextualize the efficiency claims. Also, compare against a broader set of recent differentiable simulators, if applicable.
5. **Clarify Gradient Propagation**: The paper focuses on the contact potential \( P \). A brief discussion in the main text on how the full gradient \( \partial x^{t+1} / \partial (x^t, x^{t-1}) \) is computed in the optimizer (beyond the implicit function theorem mention) would clarify the integration into a full differentiable pipeline.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison with the state-of-the-art differentiable simulator (Huang et al., 2024) that implements IPC.** The paper uses IPC as a baseline but does not directly compare against the full simulator from Huang et al. 2024, which is a key contemporary work. A head-to-head comparison on the same tasks is necessary to validate the claim of superior gradient behavior and optimization performance.
2. **Ablation study isolating the "Non-vanishing" property.** To prove this property is critical, create a variant of your model that artificially removes long-range influence (e.g., by truncating the potential beyond a distance) and test it. Showing its performance drops to the level of baselines (IPC/SDRS) would directly validate your core theoretical contribution.
3. **Evaluation on higher-dimensional, complex robotic control tasks.** The demonstrated tasks (e.g., Billiards, Push) have relatively low-dimensional control. To substantiate claims of enabling "complex, contact-rich control," test on tasks like full-body humanoid locomotion or dexterous manipulation with a high-DoF hand, where gradient-based optimization is notoriously difficult.
4. **Sensitivity analysis of the blending hyperparameter `ε`.** The parameter `ε` controls the transition between exact and approximate potentials. A systematic ablation across tasks is needed to show how it affects optimization convergence, physical accuracy, and computational cost. Without this, the practical usability of the method is unclear.
5. **Runtime and scaling comparison between brute-force and hierarchical (BSH) evaluations.** The paper claims near-linear complexity but does not show direct timing comparisons between evaluating the full pair-wise potential (Eq. 3) and the hierarchical version (P_BSH) on the same problems. This is needed to validate the efficiency claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of gradient conditioning and Lipschitz properties.** Smoothness (C²) alone does not guarantee well-behaved optimization. Plot gradient norms or Hessian condition numbers along optimization trajectories to demonstrate your gradient landscape is indeed less "rugged" than baselines. This is crucial for the claim of improved optimizer convergence.
2. **Mechanistic analysis of how non-vanishing gradients guide optimization.** Show visualizations (e.g., 2D slices of the loss landscape, gradient vector fields) for a simple task to illustrate how your model's gradients provide directional information from far away, while baselines' gradients vanish. This would directly support the narrative of escaping trivial minima.
3. **Quantitative evaluation of gradient accuracy vs. approximation error.** The hierarchical blending introduces approximations. Compare gradients from P_BSH against "exact" gradients from the full pair-wise potential (or finite differences) on a controlled setup to quantify the error introduced. This is necessary to trust that the efficient method still provides useful gradients.
4. **Broader analysis of physical accuracy trade-off due to long-range forces.** The non-vanishing property is non-physical. Beyond the book-stacking test, systematically measure the error in simulated trajectories (vs. a high-accuracy reference) for various `μ` and object separations. This would clarify the practical limits of the model's physical fidelity.

### Visualizations & Case Studies
1. **Plot of contact force magnitude vs. distance for a simple pair of objects.** Graph the force law of your model compared to IPC and SDRS. This would visually demonstrate the non-vanishing property and the smooth blending transition, making the theoretical contribution immediately clear.
2. **Visual case studies of optimizer failures with baselines.** For a task like Billiards with trivial initialization, render the simulation trajectories after optimization for IPC and your method. Show that with IPC, the controlled ball does not move because initial gradients are zero, while your method discovers a collision-rich path.
3. **Visualization of the BSH hierarchy and blending regions during a simulation.** For a complex scene, visualize bounding spheres at different levels and color-code triangle pairs based on whether their interaction is computed exactly or approximated. This would build intuition for how efficiency is maintained without breaking smoothness.

### Obvious Next Steps
1. **Open-source release and integration with a public differentiable simulation framework.** For impact and reproducibility, the contact model should be integrated into a library (e.g., as a plugin for DiffSim) and all experiment code released. ICLR strongly values open-source contributions.
2. **Discussion and preliminary exploration for deformable bodies.** The limitation to rigid bodies is acknowledged. A discussion on the challenges of extending the method to deformable objects (e.g., updating bounding spheres) and even simple preliminary results would significantly broaden the paper's relevance.
3. **Application to sim-to-real reinforcement learning.** The improved gradient landscape could benefit policy learning. A demonstration of training a policy in your simulator that transfers to a real robot (even in a simple pushing task) would greatly increase the practical impact, though a detailed experiment may be future work.
4. **Evaluation on established differentiable physics benchmarks.** Using custom tasks makes comparison difficult. Testing on existing benchmarks from prior work (e.g., from Huang et al. 2024 or robotic manipulation benchmarks) would provide a more standardized and convincing performance assessment.

# Final Consolidated Review
## Summary
This paper introduces a set of theoretical properties (Barrier-Form, Smoothness, Non-prehensile, Non-vanishing) that characterize a well-behaved differentiable contact model, addressing erratic gradients in rigid-body simulation. It proposes a practical model that satisfies all properties via a globally supported barrier potential and a bounding sphere hierarchy for efficient evaluation. Experiments on contact-rich robotic tasks show that the model enables gradient-based optimization from trivial initializations where prior methods fail due to vanishing gradients.

## Strengths
- **Theoretical framework defining essential contact model properties** – The paper clearly motivates and formalizes four properties that ensure reliable gradients, providing a lens to analyze prior work (Table 1) and guiding the design of the proposed model (Section 3).
- **Comprehensive empirical validation across diverse tasks** – The model is evaluated on five manipulation and locomotion benchmarks, demonstrating consistent improvement over strong baselines (IPC, SDRS, MuJoCo, Gradient Bundle) in discovering contact-rich policies from simple initializations (Figures 5-9).
- **Efficient implementation with near-linear scaling** – The use of a bounding sphere hierarchy and smooth blending reduces computational cost from quadratic to near-linear in practice, supported by complexity analysis for a uniform grid and timing comparisons (Appendix A.2, Table 6 in appendix).

## Weaknesses
- **Restricted to rigid bodies and triangle meshes** – The model assumes non-deforming geometry, limiting its applicability to deformable objects or other primitives, which are important for broader differentiable simulation (acknowledged in conclusion but affects generalizability).
- **Computational overhead and incomplete complexity guarantees** – Despite BSH acceleration, the model requires solving nested 4D optimizations per triangle pair and recursive blending, leading to non-negligible per-frame cost (e.g., ~3.89s for Gather-Bunny). Theoretical complexity is only analyzed for a uniform grid, not general meshes.
- **Sensitivity to hyperparameters not thoroughly analyzed** – Performance depends on the barrier coefficient μ and blending margin ε; while some analysis is provided (Tables 2, 5), no systematic study across tasks shows how these affect optimization convergence or physical accuracy.
- **Lack of statistical robustness in experiments** – Evaluations rely on single runs without reporting variance or multiple seeds, which is common but undermines claims about optimizer performance, especially given the stochastic nature of gradient-based methods.

## Nice-to-Haves
- An explicit ablation study truncating the long-range influence to directly validate the necessity of the Non-vanishing property.
- More detailed runtime breakdown (e.g., time spent in contact potential vs. BSH traversal) and scaling experiments on non-uniform meshes.
- Integration with established differentiable simulation benchmarks for standardized comparison.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Demand for direct comparison with Huang et al. 2024 simulator** – The paper already compares with IPC, which is the contact model used in that work; a full simulator comparison is not essential for evaluating the core contact model contribution.
- **Request for gradient conditioning or Lipschitz analysis** – While insightful, this goes beyond the paper’s focus on contact model properties and is not a standard requirement for empirical systems papers in this field.
- **Suggestion for sim-to-real reinforcement learning experiments** – This is outside the scope of a methodological paper on differentiable contact models.

## Novel Insights
The paper’s key novel insight is the identification of *Non-vanishing* as a critical property for differentiable contact models, enabling long-range gradient influence that guides optimization from distant initializations. This addresses a fundamental limitation in prior work where gradients vanish for non-contacting objects, and it is rigorously formalized alongside other properties (Barrier-Form, Smoothness, Non-prehensile) to define a well-behaved contact model.

## Suggestions
- Run experiments with multiple random seeds to report mean and standard deviation of performance metrics, strengthening empirical claims.
- Provide practical guidance on selecting hyperparameters μ and ε based on task characteristics, possibly via a sensitivity study in the appendix.
- Release open-source code to facilitate reproducibility and adoption by the community.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 8.0, 8.0]
Average score: 5.5
Binary outcome: Accept
