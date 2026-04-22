# Scalable and Adaptive Trust-Region Learning via Projection Convex Hull

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Learning compact and reliable convex hulls from data is a fundamental yet challenging problem with broad applications in classification, constraint learning, and decision optimization. We propose Projection Convex Hull (PCH), a scalable framework for learning polyhedral trust regions in high-dimensional spaces. Starting from an exact MINLP formulation, we derive an unconstrained surrogate objective and show that, under suitable weight assignments, the optimal hyperplanes of the MINLP are recovered as stationary points of the surrogate. Building on this theoretical foundation, PCH adaptively constructs and refines hyperplanes by subregion partition, strategic weight assignment, and gradient-based updates, yielding convex hulls that tightly enclose the positive class while excluding negatives. The learned polyhedra can serve as geometric trust regions to enhance selective classification and constraint learning. Extensive experiments on synthetic and real-world datasets demonstrate that PCH achieves strong performance in accuracy, scalability, and model compactness, outperforming classical geometric algorithms and recent optimization-based approaches, especially in high-dimensional and large-scale settings. These results confirm the value of PCH as a theoretically grounded and practically effective framework for trust-region learning. Codes are available at https://github.com/IDO-Lab/trust-region-pch.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the Projection Convex Hull (PCH) method, a scalable framework for learning polyhedral trust regions in high-dimensional spaces. The link between the intractable MINLP for tight convex hull learning and an unconstrained surrogate objective is established. The PCH framework incorporates (i) partition-based subregion assignment, (ii) a surrogate objective to align hyperplanes with class boundaries,and (iii) adaptive hyperplane pruning and addition. Experiments on synthetic and real-world datasets show that
 PCH achieves high accuracy and superior scalability.

### Strengths
1. PCH is proposed with sound theoretical foundation.
2. PCH scales to high-dimensional and large-scale problems.
3. Besides convex hull learning, downstream tasks including selective classification and constraint learning are considered.
4. Figure 1 and Figure 2 intuitively illustrates the main steps of PCH .

### Weaknesses
1. Whether trapping into local minima is possible during the optimization is not discussed.  
2. How is the robustness of PCH in the presence of data noise?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes an algorithm for learning convex hulls (polyhedral trust regions) from labeled data, ensuring that all positive samples are enclosed while excluding negatives as much as possible. Traditional geometric or MILP approaches fail to scale in high dimensions and lack flexibility, and it aims to design a scalable, theoretically grounded, and adaptive framework to learn boundary-tight convex hulls efficiently. The main contribution is that, the authors propose a divide-and-conquer learning framework that links an exact MINLP problem to an unconstrained surrogate objective, making gradient-based optimization feasible.

### Strengths
The paper presents a novel gradient-based surrogate formulation for convex hull learning, bridging discrete MINLPs with differentiable optimization. Moreover, its adaptive structure and surrogate update scheme enable practical training in high-dimensional settings, as demonstrated empirically.

### Weaknesses
1. It is unclear how the main theoretical result (Theorem 1) relates to the practical algorithm. The theorem establishes the existence of a certain weight assignment, but it is not evident whether this weight can be computed or approximated by Algorithm 1 or the procedure described in Section 5.2. Although there is some discussion in lines 354–356, the connection remains ambiguous, especially since Equation 3 does not explicitly include a weight term.

2. In addition, no convergence analysis or guarantee (e.g., monotonic improvement) is provided for the surrogate updates.



3. Since the algorithm is somewhat unintuitive, it would be helpful to include a simple 2D toy example illustrating the evolution of the hyperplane and weight updates over iterations. This visualization would greatly enhance readers’ understanding of the surrogate optimization process.

### Questions
1. A main contribution concerns the algorithm’s computational complexity. However, it appears to depend on the number of iterations—what is the typical order or empirical scaling of this term?

2. The authors should discuss potential failure cases, such as when the positive region is nonconvex or when the hyperplane budget is insufficient to capture the desired boundary.

3. Is that any small example showing that the surrogate formulation indeed approxiamte the discrete MINLP problem?

### Soundness
3

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
2

### Summary
Motivated by the observation that complex and nonlinear classification boundaries often exhibit regional linearity and can be locally approximated by hyperplanes with small classification error, the paper constructs a convex hull that is scalable,
structurally adaptive, and geometrically tight. Specifically, from the MINLP problem formulation, the work establishes its connection to a family of unconstrained surrogate objectives. This enables gradient-based learning of convex hulls that are both compact and theoretically motivated.

### Strengths
The paper effectively addresses key limitations of prior works in terms of scalability, adaptivity, and boundary tightness, and provides theoretical guarantees to support its claims. Given the intrinsic complexity of the MINLP formulation, it is particularly commendable that the authors decompose the original problem into a series of per-hyperplane subproblems with unconstrained surrogate objectives, making the optimization process more tractable. 
The final solution is obtained through gradient-based optimization, which further enhances computational efficiency. Moreover, the proposed adaptive structural adjustment introduces principled hyperplane addition and pruning criteria, enabling the model to maintain scalability without sacrificing geometric tightness.

### Weaknesses
The experimental evaluation appears somewhat limited. While the results on the BreastCancer, Spambase, Bace, and HIV datasets demonstrate the feasibility of the proposed approach, these datasets are relatively simple and well-studied binary classification benchmarks. To better assess the generalization ability and robustness of the proposed method, it would be valuable to include experiments on more challenging and diverse datasets, such as those with higher-dimensional features, class imbalance, or more complex data manifolds.

Additionally, it is not entirely clear whether the optimization process is guaranteed to converge. Since the proposed formulation involves nonconvex components and relies on surrogate objectives and iterative weight updates, a brief theoretical or empirical discussion of the convergence behavior (e.g., monotonicity of the objective, stability of fixed points, or stopping criteria) would strengthen the work and improve the reader’s understanding of the algorithm’s reliability.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the Projection Convex Hull (PCH), a scalable and adaptive framework for learning compact, polyhedral trust regions from data consisting of positive and negative points. The fundamental problem of computing the tight convex hull is actually a Mixed-Integer Nonlinear Program (MINLP). Classical geometric methods suffer from complexity that grows exponentially with dimension, and existing optimization-based approaches lack practical scalability.

The authors overcome this challenge through a rigorous theoretical decomposition. They first formally express the convex hull problem as a MINLP that explicitly accounts for model compactness. Then, they reduce the problem into multiple subproblems that handle one hyperplane at once. The authors prove that the solution for finding the optimal hyperplanes can be recovered as a stationary point of an unconstrained surrogate objective function under suitable weights. This surrogate loss function is minimized using gradient steps. The weight for each negative point is carefully designed and updated during this process to ensure that minimizing the weighted surrogate function is equivalent to maximizing the intended separation margin of the original MINLP.

In practice, the PCH framework achieves structural adaptivity: it dynamically manages the model complexity by pruning redundant hyperplanes and adding new ones if the current hull still incorrectly encloses negative samples. This ensures the final model is as compact and accurate as possible.

### Strengths
- The paper is very well written. It solves the problem gradually by breaking it down into simpler, smaller subproblems, makes the algorithm easy to follow.
- The method is strong in both theory and practice. PCH maintained high accuracy and low running time and its usage benefits the real-world downstream tasks.

### Weaknesses
- More detailed comparison to related gradient-based works are needed.
- see the question below

### Questions
- Lemma 1 does not ensure that the collection of all optimal solutions of the subproblem is the optimal solution of the original problem. How does the proposed algorithm ensure its global optimality?
- The adaptive strategy adds hyperplane when contains negative points and removes when encounter redundancy. Then how to decide the number S in practice? And what happened when the positive points are not convex (where the convex hull containing negative points are inevitable) ?

### Soundness
3

### Presentation
3

### Contribution
3
