# HoP: Homeomorphic Polar Learning for Hard Constrained Optimization

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Constrained optimization demands highly efficient solvers, which promotes the development of learn-to-optimize (L2O) approaches. As a data-driven method, L2O leverages neural networks to efficiently produce approximate solutions. However, a significant challenge remains in ensuring both optimality and feasibility of neural network's output. To tackle this issue, we introduce Homeomorphic Polar Learning (HoP) to solve the hard-constrained optimization by embedding a homeomorphic mapping in neural networks. The bijective structure enables end-to-end training without extra penalty or correction. For performance evaluation, we evaluate HoP's performance across a variety of synthetic optimization tasks and real-world applications in wireless communications. Across synthetic and real tasks, HoP achieves zero violations while remaining competitive in optimality and significantly faster than classical solvers: on polygon-constrained sinusoidal QP it matches SLSQP within $16\times$ speedup; on high-dimensional semi-unbounded problems it is tens of times faster than the optimizer with comparable or better objectives; and on QoS-MISO WSR it preserves $0\%$ violations with $11\times$ speedup over SCS+FP. These results demonstrate that HoP provides a practical, general, and strictly feasible alternative to penalty-based or projection-based L2O methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Homeomorphic Polar Learning (HoP), an approach for solving hard-constrained optimization problems with feasibility guarantees. The authors design a polar parameterization that transforms complex constraint spaces into simple (box) constraints, which can be enforced in neural networks through bounded activation functions like sigmoid. To address the singularity issues inherent in polar-based transformations, they develop a geometric reconnection mechanism. Their experimental results demonstrate both feasibility guarantees and improved optimality compared to baseline methods across diverse applications, including synthetic optimization over polygons, L_p norm balls, and the real-world QoS-MISO WSR problem.

### Strengths
1. Ensuring neural network solution feasibility is indeed critical for real-world applications.
2. Polar-based transformations for handling constraints appear to be new in the literature.

### Weaknesses
1. Inaccuracy in Homeomorphic Mapping Claims: The author asserts a homeomorphism between $(0,1)\times(0,2\pi)$ and $int(Y_x)$ in the 2D case (line 178). However, this claim is mathematically flawed. Points on the positive X-axis in the original space, which are of $\theta = 0$ or $\theta = 2\pi$ in the polar space, are excluded, violating the bijection requirement for homeomorphism. This fundamental error undermines the theoretical foundation of the approach. The author should provide **formal** statements for the homeomorphic properties of the proposed transformations in different cases.

2. Ambiguity in Interior Reference Point Computation: While the constraint sets are explicitly defined as input-dependent, the setting of the interior reference point remains unclear. If this point is designed to be input-invariant, this should be stated as an explicit assumption. Conversely, if it is input-dependent, the computational expense of recalculating it for each new input would be significant and warrants detailed efficiency analysis (computing the Chebyshev center for general convex constraints beyond a polytope is non-trivial), which is currently absent. For example, how to compute the interior point for Ay<b in (14) under different b.

3. Insufficient Specification of Support Radius Calculation: Despite focusing on convex constraints, the paper fails to provide explicit formulas for calculating the support radius across different constraint types. For a method claiming practical applicability, precise formulations of the support radius for convex sets should be presented along with a thorough complexity analysis. This omission raises concerns about the method's implementability.

### Questions
1. How to obtain an interior point for an input-dependent convex constraint.
2. What is the impact of the choice of interior points on the performance of these methods?
3. What is the type of constraint for the QoS-MISO problem? How to compute the Support Radius for this case? Could the author give a clear characterization of which kind of constraints admit such a polar parametization?

### Soundness
2

### Presentation
1

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
This paper is concerned with learning to optimize subject to hard constraints. Many NN-based approaches struggle to satisfy such constraints, whereas methods that guarantee constraint satisfaction often fail to achieve optimal solutions or suffer from performance issues. In essence, the proposed approach relies on a smooth mapping of the feasible set via a radial transformation with respect to a prescribed interior point. Subsequently, composition with an appropriate mapping (e.g., sigmoid or arctan) smoothly maps the feasible set to the entire domain. Using this mapping (or its inverse), the arbitrary output of a NN is guaranteed to be mapped into the feasible set, thus transforming the optimization problem into an equivalent unconstrained one. The approach is demonstrated on several synthetic problems and an application in communications.

### Strengths
The paper addresses the important problem of learning to optimize under constraints. The experimental results, on synthetic problems and a communications application, are positive. The idea of transforming a constrained problem into an unconstrained one using a radial (polar) mapping is mathematically elegant.

### Weaknesses
Despite the strengths described above, the paper raises several concerns that I believe must be addressed before publication. I have reviewed a previous version of this paper, and while it has been moderately revised, many of the major concerns raised earlier remain unresolved.

**(1) Contribution and Novelty:**  
The ideas presented in this paper appear closely related to Liang et al. (2023). Although the authors cite this prior work, the paper lacks an explicit and careful discussion of how their approach differs. The authors also do not cite the extended version of that paper ([https://jmlr.org/papers/v25/23-1577.html](https://jmlr.org/papers/v25/23-1577.html)). These omissions make it difficult to assess the originality and scope of this work’s contribution.

**(2) Limitations:**  
One of the most challenging aspects of the proposed method seems to be the construction of the homeomorphism $H$, and specifically the term $R(v_\theta, Y_x)$. Even in the 2D convex case, computing Equation (3) explicitly may be intractable for anything but the simplest sets (e.g., norm balls or polyhedra).  
The paper does not discuss the implementation details of $H$ (e.g., whether an analytical expression is required) beyond the examples shown. I find this construction questionable, especially in higher-dimensional or non-convex settings, where constructing such a homeomorphism may become impractical.

**(3) Exposition, Organization, and Reproducibility:**  
Sections 3 and 4 provide a somewhat superficial description of the approach and experiments, making it difficult to follow key details. Many important points are deferred to the appendix, sometimes without sufficient context or cross-referencing. Even with the appendix, reproducing the proposed method for anything beyond basic problems would likely be difficult. The authors’ code does not appear to be available.  
I also find the discussion in Section 3.2 hard to follow. Conceptually, it makes sense that optimization over highly distorted mappings (as can be expected for maps of an arbitrary set to the entire domain) may suffer from ill-conditioning, potentially leading to stagnation. However, the discussion lacks clarity and context. Moreover, this issue was also discussed in Liang et al. (2023), yet is not cited here, again making it difficult to assess the authors’ contributions.

**(4) High Dimensions and Non-Convexity:**  
The generalization of the proposed approach to high-dimensional and non-convex constraints is mentioned throughout the paper somewhat superficially, even though such extensions present significant challenges. The appendices discuss several related issues (e.g., “Toward the Non-Convex Case”), but it is unclear whether these were implemented, for which problems, or how practical they are in more complex scenarios.  
The examples presented in the paper are relatively simple: for the non-convex $\ell_p$-norm constraint, the set is star-convex with respect to the origin (Appendix D), effectively reducing it to a convex case under the mapping. The parabolic constraint appears effectively linear due to small quadratic terms (mentioned only in Appendix C.1.4). The non-convex high-dimensional example in Appendix G does not provide enough evidence to support the strong claims of generalizability made in lines 410–423.

### Questions
See main Questions (1)–(4) above.  

**Additional Comments and Questions (non-comprehensive list):**  
* **Line 116:** “according to the designing” — unclear; please revise.  
* **Line 123:** “the stagnation endemic to polar parameterizations” — unclear; please revise.  
* **Lines 133–134:** “To facilitate that the outputs of a learning model satisfy the given constraints, we construct a mapping from the raw output space to a constrained space with homeomorphic transformation.” — this central sentence is difficult to follow; consider revising for clarity.  
* **Line 204:** “a single scalar angle for range” — unclear; please revise.  
* **Line 212:** Why is $\varepsilon > 0$ needed if $z_0$ lies in the interior of the constraint set?  
* **Line 251:** “In order to reveal the stagnation phenomenon in polar optimization” — unclear; please revise.  
* **Lines 273–285:** Phrases such as “the projection ... prevents this passage,” “enters an absorbing state,” “geometric artifact stagnation,” and “After crossing with the flip” are difficult to interpret; please consider revising for clarity.  
* **Lines 303–306:** “Since other NN-based methods such as DC3 typically rely on penalty functions to handle constraints, ..., where ... indicates constraint violations, and ... quantifies the distance to the constraint set.” — this sentence is incomplete; please revise.  
* **Line 355:** “Table 1 reports: ...” — unclear; at this point in the paper, the experiments have not been clearly introduced.  
* **Line 419:** “This enables the our framework is able to apply on these complex constraints.” — unclear; please revise.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a learning-to-optimize framework for constrained optimization problems, aiming to ensure both optimality and strict feasibility. To guarantee feasibility, the authors introduce a polar coordinate representation combined with a homeomorphic mapping that transforms neural network outputs into a deterministic ball, thereby satisfying hard constraints. Furthermore, they develop geometric reconnection strategies to promote stable convergence during backpropagation. The proposed method, HoP, is evaluated through both synthetic experiments and a real-world QoS-MISO WSR optimization task, demonstrating superior performance compared to existing baselines.

### Strengths
The strengths of the paper are listed below.
- The paper considers the important and fundamental problem of enforcing hard constraints for L2O.

- The proposed use of polar coordinates to reformulate the neural network outputs is interesting.

- The paper includes experimental evaluations on multiple benchmark problems.

### Weaknesses
The weaknesses are given below.
- The proposed method is limited to problems with convex constraints. However, learning-to-optimize (L2O) approaches are often expected to handle broader classes of non-convex and complex optimization problems, which restricts the general applicability of this work.
- The approach requires redefining the semantics of neural network outputs through coordinate transformation and mapping, making it difficult to integrate with existing neural network architectures. Consequently, applying this method to pre-trained models would require extensive retraining, increasing implementation complexity.
- The experimental validation is restricted to relatively simple, convex and well-studied optimization problems (e.g., QoS-MISO WSR), where efficient solutions already exist. This limits the strength of empirical evidence supporting the proposed method’s advantages in more challenging or practical settings.

### Questions
I am not convinced of the necessity of the proposed HoP method, given that the feasible region in this work is restricted to a convex space. In such cases, projecting the neural network outputs onto the convex feasible set can be done efficiently using standard convex optimization solvers. Moreover, end-to-end training with projection-based feasibility enforcement can also improve optimality while maintaining hard constraint satisfaction. Therefore, it remains unclear what practical or theoretical advantage the proposed HoP framework offers on top of the existing methods.

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
5

### Summary
This paper presents Homeomorphic Polar Learning (HoP), a framework for machine learning with hard constraints. The paper:
* Presents a method for defining exact bijective mappings between convex (and some nonconvex) constraint sets, using polar coordinate representations.
* Integrates an approach allowing the use of signed radii for addressing challenges in gradient-based optimization of polar coordinates.
* Conducts experiments on three synthetic problems and one realistic setting (QoS-MISO WSR), with comparison against traditional optimizers, standard neural network baselines, and two state of the art methods.

### Strengths
* The proposed method is strong in premise and innovative, providing a clean and cost-effective way to enable feasibility enforcement in neural networks (for certain classes of constraint sets) by leveraging homeomorphic mappings. 
* The method is very well-presented, with helpful pedagogical examples for the 1D and 2D case before extending to the semi-unbounded and high-dimensional cases. In general, the writing is strong.
* The trick to resolve stagnation of gradient descent with polar coordinates improves the efficacy of the framework (as shown via the ablation) and is also potentially of wider general interest.
* Experimental results are impressive. HoP demonstrates comparable objective value and feasibility to traditional optimizers, while being orders of magnitude faster. Compared to DC3 and HomeoProj, HoP generally does better with respect to objective value (and is also applicable in certain settings where HomeoProj is not).

### Weaknesses
Major:
* The method is limited to optimization problems with convex constraint sets, or specialized nonconvex structure (ray intersects the boundary exactly once), but this limitation is not stated up-front. The scope of the method should be stated much more clearly early on. (Appendix G does provide some exploration into the non-convex case, but it is very initial.)
* Some of the offline costs associated with the framework are not transparently disclosed. This also affects what sets of parametric constraints can be handled. For instance, in Section 4.1, only $b$ is varied, but $A$ is not -- it is unclear whether this is due to a limitation of HoP that makes varying $A$ difficult. (Note: While the experiments mention this choice is consistent with the DC3 paper, it is not -- the DC3 paper varies far more problem parameters in its experiments than this paper does.)
* In the experiments, the number of replicates is not provided, and error bars are not provided. It is therefore unclear which differences between methods are statistically significant.

Minor/medium:
* Some of the related work in the Introduction and Related Work is mischaracterized. Please double-check and fix these descriptions and references.
* The relationship of HoP to HomeoProj (Liang et al. 2023) is not sufficiently described in the Introduction or Related Work, which is surprising given its close relationship in toolkits to HoP. (To be fair, this method is at least compared against in the experiments.)
* Differentiable orthogonal projection is another baseline that should likely be compared against -- it will be slower than HoP, but it may have similar objective value performance, and this is worth noting. In general, it is important to have comparisons against another strong methods where -- like HoP -- the feasibility mapping is internalized to end-to-end training.

### Questions
* Could you please succinctly characterize the kinds of constraints that this method is able to handle, in theory and in practice?
* In the experiments in Section 4.1, does varying $A$ pose additional costs or difficulties for HoP? In general, what parameters are able to vary without requiring recomputation of the homeomorphic mappings?
* DC3 is consistently characterized in the submission as a method with post-correction mechanisms; however, in reality, the correction mechanism is internalized to the end-to-end training procedure (i.e., occurs before evaluation of the loss function, and is differentiated through), so it is not really a "post"-correction. Could the authors verify that they indeed implemented the correction procedure within their DC3 baseline such that it is part of end-to-end training?

### Soundness
2

### Presentation
3

### Contribution
3
