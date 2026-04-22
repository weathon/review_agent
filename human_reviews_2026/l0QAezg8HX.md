# Preserving Gradient Harmony: A Rotation-Based Gradient Balancing for Multi-Task Conflict Remedy

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Multi-task learning (MTL) enables knowledge sharing across tasks but often suffers from gradient conflicts, leading to performance imbalances among tasks. Existing weighting-based methods attempt to balance the directional conflicts by striving for the optimal weights computed from gradient or loss information. However, those indirect weighting operations face a limited balancing effect, as the gradient's per-dimensional sensitivities are omitted. Alternatively, gradient manipulation methods such as PCGrad, GradDrop, etc., directly control the task gradients to eliminate opposing gradient directions, but their over-aggressive operations potentially harm the gradient properties, leading to suboptimal updates. They are associated with the issues of over-correction, order dependence, and poor scalability in high-dimensional task settings. To overcome these limitations, we propose the Rotation-Based Gradient Balancing (RGB), a novel algorithm that rotates normalized task gradients toward a consensus direction using independently optimized per-task angle corrections. Unlike projections, rotations provide fine-grained control that preserves beneficial gradient components, reduces global conflicts holistically, and implicitly incorporates loss change information for balanced optimization. Empirical results demonstrate the effectiveness and consistency of RGB, achieving state-of-the-art performance in various datasets, where RGB is the first method on the QM9 dataset with 11 tasks to surpass single-task baselines on average, and its performance is consistent across various benchmarks ranging from 3–40 tasks. Moreover, we propose the concept of multi-task equilibrium relationship that is supported by our empirical experiment and inferring the phenomenon of miss-correction angular error. We also provide the theoretical global convergence of RGB to Pareto stationary under standard smoothness assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the Rotation-Based Gradient Balancing (RGB) algorithm, which rotates normalized task gradients toward a consensus direction by independently optimizing angle corrections for each task. The RGB algorithm offers fine-grained control that preserves beneficial gradient components, comprehensively reduces global gradient conflicts, and implicitly integrates loss change information to achieve balanced optimization. Empirical evaluations across four datasets show its effectiveness. However, there are still some aspects in this paper that require improvement or further clarification.

### Strengths
S1: The RGB algorithm offers precise control mechanisms that effectively preserve advantageous gradient components, systematically mitigate global gradient conflicts, and seamlessly integrate loss change information to facilitate balanced optimization.
S2: Empirical evaluations conducted across multiple datasets convincingly demonstrate the efficacy of the RGB algorithm.

### Weaknesses
W1: The innovation of the gradient rotation method is limited. Essentially, gradient rotation amounts to multiplying the original gradient by a rotated coordinate system. What is the fundamental difference between this approach and traditional gradient constraint or game-theoretic methods (such as CAGrad and Nash-MTL)?
W2: In practical training scenarios, gradients may be unstable or contain noise. How does the gradient rotation method address the issues of unstable gradients or noise?
W3: There is a lack of analysis on time and space complexity. A large number of gradient operations may pose challenges in terms of computational overhead when dealing with more tasks or larger models.
W4: The demonstrated effectiveness is not entirely robust. For instance, on the NYUv2 dataset, the gradient rotation algorithm only achieves state-of-the-art performance on the Segmentation task.
W5: There is a lack of visual analysis for visual tasks. For example, for the NYUv2 and Cityscapes datasets, it is recommended to provide visual results.
W6: There are doubts about the authenticity of the results. In Figure 6 of the appendix, why is the optimization trajectory plot of the Nash-MTL method labeled as "Ours"? Moreover, the optimization trajectory plots of the RGB method and the Nash-MTL method appear to be very similar. Are the presented results genuine?
W7: There are numerous formatting and layout issues. For example, the font size in Table 5 is too large, while the font sizes in Figures 4-5 are too small.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates multi-task learning (MTL) from the perspective of gradient manipulation. It argues that previous approaches often tend to over- or under-correct task gradients, thereby compromising task-specific gradient information. To address this issue, the paper proposes a rotation-based strategy that introduces an objective function designed to simultaneously minimize gradient conflicts and regularize gradient deviations. Extensive experiments conducted on four public datasets demonstrate the competitive performance of the proposed method.

### Strengths
1. The idea of de-conflicting task gradients through a rotation-based perspective appears to be novel in the context of MTL.

2. The proposed method achieves competitive performance across multiple mainstream MTL benchmarks.

3. The paper provides some theoretical insights that help support and motivate the proposed approach.

### Weaknesses
1. Figure 1 provides a conceptual illustration of a phenomenon. To move beyond a purely illustrative claim, the manuscript would be significantly strengthened by either empirical evidence or a formal theoretical analysis to verify and substantiate this depiction.
2. In Figure 1, how to derive such an optimal gradient direction? 
3. Can you provide some insights on why RGB is extremely effective on QM9? 
4. Why only report across a single seed on CelebA? And why is $\Delta m$ \% here different from previously reported in other literature [1]?
5. The notations $f_z(x)$ and $F(x)$ are employed without prior definition.

Reference:

[1] Fair resource allocation in multi-task learning. ICML 2024.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
RGB is a rotation-based gradient balancing method for multi-task learning: each task’s unit gradient is rotated toward an EMA reference via a global alignment-plus-proximity objective, then averaged. The paper proves convergence to Pareto-stationary points and reports consistent gains on NYUv2, Cityscapes, CelebA and QM9.

### Strengths
1. Principled, globally coordinated reconciliation. Uses per-task scalar rotations toward a shared reference to reduce conflict globally, avoiding PCGrad-style pairwise projections while preserving task-specific signal via a proximity term.

2.Theoretical grounding to Pareto stationarity. Provides existence of the inner minimizer and convergence to Pareto-stationary solutions under standard smoothness/stepsize assumptions, tying the geometric construction to multi-objective optimality rather than a heuristic.

### Weaknesses
1.Mathematical clarity in the rotation operator.
The text states that $r_i(\tilde\alpha_i)=d_t$ when $\tilde\alpha_i=\pi/2$, but by construction $r_i(\pi/2)=w_i$ (the component orthogonal to $\bar g_i$). Hitting $d_t$ requires $\tilde\alpha_i=\angle(\bar g_i,d_t)$.

2.Unquantified compute/memory overhead and scalability.
The method introduces an EMA reference direction and an inner loop for angle optimization, but the paper does not report peak memory, extra FLOPs, or wall-clock overhead vs. baselines, nor scaling with the number of tasks $T$.

3.Missing ablation on the proximity weight $\lambda$.
$\lambda$ governs the trade-off between conflict reduction and fidelity to original gradients, effectively bounding the rotation. There is no systematic sweep . A grid over $\lambda$ with curves for scores is needed to validate robustness and to show the method does not hinge on a narrow setting.

4.Missing ablation on the EMA coefficient $\mu$ and reference-direction design.
Because $d_t$ defines the feasible rotation plane, its construction is critical to stability. The paper lacks sensitivity analyses over $\mu$ and comparisons to alternatives.

5.Incomplete positioning relative to prior rotation methods (RotoGrad)[1].
The paper does not cite or empirically compare to RotoGrad, a closely related line that also uses rotation to harmonize multi-task gradients. A proper literature positioning and a head-to-head comparison under the same backbone/budget are needed.

[1] Javaloy, A. & Valera, I. RotoGrad: Gradient Homogenization in Multitask Learning. ICLR 2022.

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
