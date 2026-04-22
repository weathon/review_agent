# ViscoReg: Neural Signed Distance Functions via Viscosity Solutions

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Implicit Neural Representations (INRs) that learn Signed Distance Functions (SDFs) from point cloud data represent the state-of-the-art for geometrically accurate 3D scene reconstruction. However, training these Neural SDFs often requires enforcing the Eikonal equation, an ill-posed equation that also leads to unstable gradient flows. Numerical Eikonal solvers have relied on viscosity approaches for regularization and stability. Motivated by this well-established theory, we introduce ViscoReg, a novel regularizer that provably stabilizes Neural SDF training. Empirically, ViscoReg outperforms state-of-the-art approaches such as SIREN, DiGS, HotSpot and StEik on ShapeNet, the Surface Reconstruction Benchmark, and 3D scene reconstruction datasets. Additionally, we establish novel generalization error estimates for Neural SDFs in terms of the training error, using the theory of viscosity solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ViscoReg, a viscosity-based regularization method for neural signed distance function (SDF) overfitting. The authors ground their approach in the theory of viscosity solutions of the Eikonal equation and propose adding a dynamically decayed diffusion term to stabilize training and improve reconstruction. They provide theoretical generalization bounds, a derivation of the gradient-flow stability, and experiments on ShapeNet, and scene data. While the mathematical exposition is solid and the motivation is clearly articulated, the experimental validation is not yet convincing and lacks key modern comparisons.

### Strengths
The paper offers a clean theoretical formulation, linking Neural SDF training to viscosity solutions and providing a nontrivial generalization error bound.

The proposed regularizer is physically motivated, conceptually elegant, and integrates well with existing Eikonal-based losses.

### Weaknesses
The experimental evaluation is incomplete. The lack of comparisons with recent strong baselines such as NSH, NeurCADRecon, or other newly neural surface reconstruction methods that based on SIREN.

Figure 1 does not show ground-truth shapes and thus fails to convincingly demonstrate the improvement on the zero-level surface. Input point clouds are not visualized on other figures, making it difficult to assess robustness.

The paper lacks data-centric ablations: experiments on varying input density, noise, missing regions, or irregular point distributions are presented, which are crucial for validating the claimed “stability”.

### Questions
How does ViscoReg perform when input point clouds are extremely sparse, noisy, or contain holes?

How sensitive are results to the decay schedule of ε when applied to real-world, non-synthetic data?

Is there any evidence that the theoretical generalization bound correlates with empirical convergence or stability in training?

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
4

### Summary
The presented paper introduces ViscoReg, a method to stabilize neural signed distance function (SDF) training. Many neural SDF methods use the Eikonal equation, which is unstable and ill-posed. ViscoReg adds a viscosity-based regularization term, inspired by PDE theory, to ensure convergence of the method to the correct viscosity solution. Furthermore, the authors present an error analysis of the typical Eikonal approach. The included experimental evaluation shows that ViscoReg is able to outperform state-of-the-art methods.

### Strengths
*Exposition:*
The proposed method is clearly explained, such that I am confident I could implement the method after reading the paper. Furthermore, the experiments are described in sufficient detail, such that I could also reproduce them.

*Performance:* 
The method shows slightly improved performance on two benchmarks. Furthermore, Figure 1 demonstrate that it is even suitable to handle complicated examples with high-frequency details.

### Weaknesses
*Novelty:*
I am mostly concerned about the novelty of the method. The cited paper ViscoGrids follows essentially the same idea, by using the regularized Eikonal equation instead of the usual one. The main differences are that ViscoGrids uses, well, grids and does not modify the viscosity coefficient. (This is also discussed in l.120) For me, these two changes to not justify a new publication at ICLR. There would be the contribution of generalization estimate, but this brings me to my second point:

*Theory:*
Unfortunately, I do not see the value of Theorem 1. Maybe I am mistaken, but as I see it, it holds all u_\theta with gradient bounded away from zero. That would mean it does not give use a specific insight into the optimizer of (5). Furthermore, I am also missing experiments that should the impact of this theorem onto computations. What could one learn from it to improve the performance of the method?

### Questions
- I think the connection between Theorem 1 and the viscosity regularization needs to be explained better. Only that fact that latter might control M_\theta (for which there is no proof) does not seem sufficient to me to generate a consistent story for the paper.
- l. 274: In (12), what is M (without index)?
- l. 239: Here it should be \citep instead of \citet 
- Experiments: I think the validations should include Neural Singular Hessian (SIGGRAPH Asia 2023) and "Aligning Gradient and Hessian or Neural Signed Distance Function" (NeurIPS 2023) as comparisons, as they also include higher-order regularizations.
- I would be interesting to (quantitatively) assess the accuracy of the obtained SDF away from the surface. This could further highlight the impact of the viscosity regularization.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ViscoReg, a new regularization method for training neural signed distance functions (SDFs). Standard implicit neural representations suffer from instability and non-uniqueness when enforcing the Eikonal equation, which can lead to poor surface reconstruction. The authors use viscosity theory from partial differential equations to address this issue. They provide a theoretical generalization bound showing that minimizing the manifold and Eikonal losses yields solutions close to the unique viscosity solution of the Eikonal equation. They then propose ViscoReg, which adds a viscosity-based term to the Eikonal loss with a decaying coefficient to improve stability and convergence. Experiments on several benchmarks, including the Surface Reconstruction Benchmark, a 3D scene dataset, and ShapeNet, demonstrate that ViscoReg outperforms methods such as DiGS and StEik, producing more stable and detailed reconstructions.

### Strengths
Originality: The method introduces a viscosity-based regularizer for neural SDFs, connecting numerical PDE theory and neural implicit representations. This is a novel way to stabilize training using a mathematically grounded approach instead of ad hoc methods.

Quality: The paper includes both theoretical and empirical contributions. The generalization bound links training errors to the viscosity solution, and the gradient flow analysis clearly explains how the viscosity term improves stability. The experiments are thorough and demonstrate consistent improvements across multiple datasets.

Clarity: The paper is clearly written, well-structured, and easy to follow. The motivation, method, and theory are explained logically, and figures effectively illustrate the improvements of ViscoReg.

### Weaknesses
My main concern is that the paper treats Sitzman et al from 2020 as its baseline. This is an old old paper - especially in today's world where things move so rapidly. Could the authors please comment on this
Also most of the analysis is done on scenes/objects without significant noise.The theoretical analysis depends on strong assumptions, such as bounded gradients and smoothness, which may not hold in practice. It is unclear how these assumptions relate to realistic neural SDF training dynamics.

How would these theoretical guarantees translate to the real world. I come from a vision/graphics background so to me it always feels like that these theoretically grounded algorithms work on very isolated scenarios. Its a personal preferences and I am open to hearing other reviewers if they have different opinion; I would be curious to see how this algorithm performs on real-world captures scans from scannet for example

The link between the theory (which analyzes the standard Eikonal equation) and the practical ViscoReg implementation (which includes a decaying viscosity term) is not fully formalized. Theoretical support for the time-dependent decay is missing.

The experiments do not discuss computational costs. Since the viscosity term involves computing the Laplacian, training may be more expensive, but runtime or memory overhead is not reported.

The evaluation focuses mostly on SIREN-based architectures. It is unclear whether the proposed method generalizes to other architectures such as ReLU MLPs or hash-based encodings.

There is limited analysis of robustness to noisy or irregular point clouds, which could test the generalization claims more strongly.

### Questions
How does this method work on real world datasets?

Is there theoretical or empirical justification for how fast the viscosity parameter should decay over time?

Which loss exponent (p=1 or p=2) was used in the main experiments, and how sensitive are the results to this choice?

What is the computational overhead of ViscoReg compared to Eikonal or DiGS in terms of runtime and memory?

Have you tested the method with other network architectures beyond SIREN?

How does ViscoReg perform when training data includes strong noise or non-uniform sampling?

Could you include quantitative metrics for the 3D scene reconstruction example instead of only qualitative images?

Are all compared methods trained under the same number of iterations, sample counts, and parameter counts?

Could you show curvature or smoothness statistics to demonstrate that ViscoReg maintains fine geometric details rather than over-smoothing surfaces?

What are common failure cases for ViscoReg, and how does the viscosity schedule affect them?

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
3

### Summary
This work aims to address the core challenges encountered when learning SDF with INRs: the ill-posedness of the Eikonal equation (which admits infinitely many solutions) and the instability of gradient flows during training.

Inspired by the “viscosity” approach commonly used in numerical Eikonal solvers, the authors propose a novel regularization term, **ViscoReg**. This term modifies the Eikonal loss into a viscous Eikonal loss by introducing a decaying Laplacian term during the early stages of training. This stabilizes the training process and guides the network to converge to a unique, physically meaningful viscous solution.

Theoretically, this work leverages properties of viscous solutions to establish a generalization error bound for neural SDFs, proving that a finite training error guarantees convergence of the learned function to the correct viscous solution. In experiments, ViscoReg demonstrates superior reconstruction quality and stability over state-of-the-art methods (e.g., DiGS, StEik), particularly when handling fine structures and high-frequency details in tasks such as surface and scene reconstruction.

### Strengths
1. This paper provides a generalization error bound for Neural SDFs, guaranteeing in the $L^{\infty}$ sense that the error between the learned function and the true SDF is controlled by the training loss. This addresses the theoretical challenge posed by the ill-posedness of the Eikonal equation.
2. On multiple public benchmarks, ViscoReg significantly outperforms SOTA methods. For instance, on ShapeNet, the mean squared Chamfer distance is reduced by approximately 35%, demonstrating its effectiveness in practical applications and its particular strength in faithfully reconstructing complex and fine-grained geometric structures.
3. Through gradient flow analysis, the paper shows that the viscous term introduced by ViscoReg suppresses high-frequency noise, stabilizing the training process and effectively preventing common artifacts in other methods, such as self-intersections, disconnections, or over-smoothing.

Overall, the core contribution of this work lies in introducing viscosity theory into Neural SDFs. The theoretical analysis appears comprehensive, though I cannot verify it in full. Nevertheless, the experimental results demonstrate its practical effectiveness. While viscosity theory has been proposed and applied in ViscoGrid for the same task, this paper presents a more theoretically sound and insightful treatment.

### Weaknesses
**Dependence on the decay strategy of the viscosity coefficient ($\epsilon$)**: The paper notes that the decay strategy of $\epsilon$ affects performance, and certain shapes (e.g., anchor and gargoyle) require adjustments to the decay rate. Although a baseline decay schedule is provided, its complex, piecewise-linear design means that applying the method to new datasets or novel shapes may require additional hyperparameter tuning, increasing the overall usage complexity.

**Experimentally**:

1. Although ViscoGrids may not be the current SOTA Neural SDF method, it is the SOTA “viscosity”-related method. Therefore, visual comparisons with it are highly valuable for clarifying the methodological innovations and physical significance of ViscoReg.
2. Similarly, the paper mentions the latest method **HotSpot**, whose experimental results are highly competitive. I therefore believe it would be worthwhile to further include comparative analysis and qualitative shape comparisons with this method.

### Questions
**Constants in the generalization error bound:** In Theorem 1, the generalization error bound depends on the Sobolev norm bound $M_{\theta^{\*}} $ and the gradient bound $C_{\theta^{\*}}$ of the network. Could the authors provide a way to track or control these constants during training? For example, is it possible to design an auxiliary loss term that indirectly constrains $M_{\theta^{\*}}$, thereby offering a more explicit guarantee on generalization performance?

**Universality of the gradient flow analysis:** The paper’s linearized analysis of the gradient flow (Eq. 17) is conducted around a linear steady-state solution $ u_0 = a \cdot x$. Is this local linear analysis sufficient to capture and explain the empirical stability provided by ViscoReg in regions with high curvature, fine details, or non-smooth points?

**Adaptive decay strategy:** Given the sensitivity of the $ \epsilon$ decay strategy, have the authors considered an adaptive $ \epsilon$ approach? For instance, dynamically adjusting $ \epsilon$ based on the current residual of the Eikonal loss, the energy of the network’s high-frequency components, or local instabilities in the gradient flow, rather than using a preset piecewise-linear schedule?

### Soundness
3

### Presentation
3

### Contribution
2
