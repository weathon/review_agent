# Latent PDE Mapping for Shape-Generalizable Physics-Informed Neural Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Physics-Informed Neural Networks (PINNs) have shown strong potential for learning physically consistent representations from sparse data, but often struggle to generalize to geometries with varying shapes. To address this challenge, we introduce $\textit{latent PDE mapping}$, a technique for mapping geometry-specific partial differential equations (PDEs) to a shared latent PDE representation using the deformation gradient. We embed latent PDE mapping into the PINN framework (LPM-PINN), enabling PINNs to capture geometric variability while preserving the governing physics. This integration facilitates accurate predictions of nonlinear, time-dependent systems even in geometries well beyond the training distribution. We demonstrate LPM-PINN on a challenging nonlinear time-dependent PDE with sharp gradients, the Aliev–Panfilov model of cardiac electrophysiology, in both 2D and 3D. Our results show that LPM-PINN generalizes robustly across diverse geometries, including shapes with drastically changing boundaries that lie outside the training distribution. These findings establish latent PDE mapping as a promising approach for boosting the geometric generalizability of physics-informed neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
A physics-informed neural network training for partial differential equations defined on varying domains.

### Strengths
- Easy to read and understand; e.g., input concatenation approach for dealing with parametric dependence.
- An uncommon benchmark problem (Aliev--Panvilov).

### Weaknesses
- Pulling back to the reference geometry is standard, not new (e.g., shell-FEM). Once pulled back to the reference geometry, the problem becomes parametric PDEs. Therefore, more appropriate baselines would be methods to solve parametric PDEs, such as DeepONet.
- No data regime? If there is data, then it is necessary to present a useful application.
- Comparison with other PINN methods for varying geometries? Mentioned that other methods are limited to linear and static PDEs, yet that does not mean inapplicability of such methods to the Aliev-Panfilov model.
- Somewhat overselling—the deformation gradient F depends on “t” (eq. (5)), but the benchmark problem considers static (in time) geometries.
- No thorough investigation of why the method generalizes to out-of-range cases.

### Questions
- How did you sample collocation points for PDE residuals?
- line 259: What does it mean by “tanh for handling second-order derivatives?”
- For nonlinear geometry, F depends on x. How did you compute PDE residuals with AD in this case?
- What are the values for D, mu_1, mu_2, epsilon_0, k, and a? Can’t find in Table 7.
- Why are snapshot times varying? Difficult to compare. Fix them for the final time.
- How are FEM solutions stored? In the reference geometry?
- More explanation is necessary for the two geometry cases: G_exp + G_rot.
- In Table 1, why did Affine-PINN perform well?
- Given that Affine-PINN performed well in Table 1, Figure 4 may indicate that missing shape gradient information is not so informative?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes latent PDE mapping for PINNs to solve PDEs of varying geometries. The method maps parametric geometries to a common latent geometry, and derives a general physics loss formula to train PINN. Experiment results on 2D and 3D Aliev-Panfilov equations are presented.

### Strengths
1. The paper is clearly written and easy to follow.

### Weaknesses
1. The contribution of the paper appears weak. It completely omits closely related works in the literature, which already address the same problem. Existing work such as [1] has demonstrated the possibility of mapping arbitrary nonparametric geometries to certain latent space, to learn large scale complex 3D PDEs. In comparison, the proposed method can only handle simple parametric cases.

2. The experiments are weak in terms of both complexity and baselines. Only rectangular domains and their affine transformations are considered. Moreover, only naive PINN and affine PINN are compared with, missing state of the art methods such as [1-3]. 

3. While the author did not mention, I believe the method only applies to domains that are diffeomorphic to the latent domain, as otherwise the gradient information can not be mapped. This will heavily limit the applicability of the method.

4. As a side note, I don’t think using inverted allocation of train-val-test split can demonstrate the model performance in data-limit scenarios. Few shot generalization ability only depends on the sufficiency of training data, not on how all data are partitioned.


[1] Li, Zongyi, et al. "Geometry-informed neural operator for large-scale 3d pdes." Advances in Neural Information Processing Systems 36 (2023): 35836-35854.

[2] Li, Zongyi, et al. "Fourier neural operator with learned deformations for pdes on general geometries." Journal of Machine Learning Research 24.388 (2023): 1-26.

[3] Li, Zijie, Anthony Zhou, and Amir Barati Farimani. "Generative Latent Neural PDE Solver using Flow Matching." arXiv preprint arXiv:2503.22600 (2025).

### Questions
1. How does the proposed method compare with state of the art methods such as [1-3]?

2. How does the proposed method work under more challenging benchmark problems such as the ones considered in [1-3]?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces latent PDE mapping to enhance the generalization of PINNs in different geometries. It maps geometry-specific PDEs to a shared latent space using the deformation gradient and affine shape parameterization. The method is tested on Aliev-Panfilov model of cardiac electrophysiology in 2D and 3D and shows improvement.

### Strengths
- It is an important problem to generalize PINNs across geometries without retraining. The use of a simple MSE loss on collocation points neglects the gradient component from the movement of the domain boundary. The paper provides both theoretical and empirical evidence that this is suboptimal.
- The focus on cardiac electrophysiology adds practical relevance.

### Weaknesses
- The geometric representation through affine transformations is limited. The authors acknowledge this limitation, but it remains unclear whether latent PDE mapping can be extended to a more general setting.
- The PDE dynamics tested are simplified for the stated application domain. The paper only considers isotropic diffusion.
- Lack of computational efficiency analysis. The method requires computing the deformation gradients but he paper provides no timing comparisons.
- The experimental comparisons are incomplete, particularly recent geometry-aware neural operators, for example GNO and DeepONet. The baselines (vanilla PINNs and APINN) are not adequate.

### Questions
Mentioned in Weaknesses.

### Soundness
3

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
4

### Summary
The paper proposes latent PDE mapping (LPM): it maps geometry-specific PDEs onto a shared latent PDE defined on a reference domain via the deformation gradient, and embeds this into a PINN (LPM-PINN). By moving shape dependence from the geometry into the PDE coefficients, the method aims to capture geometric variability while preserving the underlying physics and enabling correct shape gradients. They validate LPM-PINN on synthetic cardiac electrophysiology (Aliev–Panfilov) in 2D and 3D over rich families of deformed domains.

### Strengths
1.  Using deformation-gradient–based PDE mapping directly inside a PINN to create a shared latent PDE for varying geometries is conceptually new.
2. Performance is good compared with Affine-PINN and naive PINNs.
3. Key ideas (latent PDE, deformation gradient, missing boundary gradients) are explained at a level that a PINN/PDE reader can follow.

### Weaknesses
1. Problem scope is narrow:
All results are on synthetic cardiac PDE data with parameterized (mostly affine) deformations and simplified Aliev–Panfilov dynamics.

2. Limited ablation on: the necessity of the full deformation-gradient-based mapping vs simpler coordinate transforms, sensitivity to the choice of latent domain, computational overhead trade-offs.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4
