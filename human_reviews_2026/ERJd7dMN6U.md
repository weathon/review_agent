# Riemannian Optimization on Relaxed Indicator Matrix Manifold

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
The indicator matrix plays an important role in machine learning, but optimizing it is an NP-hard problem. We propose a new relaxation of the indicator matrix and compared with other existing relaxations, it can flexibly incorporate class information. We prove that this relaxation forms a manifold, which we call the Relaxed Indicator Matrix Manifold (RIM manifold). Based on Riemannian geometry, we develop a Riemannian toolbox for optimization on the RIM manifold. Specifically, we provide several methods of Retraction, including a fast Retraction method to obtain geodesics. We point out that the RIM manifold is a generalization of the double stochastic manifold, and it is much faster than existing methods on the double stochastic manifold, which has a complexity of \( \mathcal{O}(n^3) \), while RIM manifold optimization is \( \mathcal{O}(n) \) and often yields better results. We conducted extensive experiments, including image denoising, with millions of variables to support our conclusion, and applied the RIM manifold to Ratio Cut, we provide a rigorous convergence proof and achieve clustering results that outperform the state-of-the-art methods. Our Code is presented in Appendix H.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the RIM (Relaxed Indicator Matrix) manifold as a generalization of the doubly stochastic and single stochastic manifolds. It proves Riemannian geometric properties of this set and develops a full Riemannian toolbox for optimization, including gradient, Hessian, and multiple retraction methods. The method is applied to large-scale tasks like Ratio Cut clustering and image denoising.

### Strengths
1. This paper proposes a manifold that interpolates between existing relaxations of indicator matrices.

2. The authors develops practical Riemannian algorithms (including a retraction via projection).

3. The authors shows speedups over doubly stochastic manifold optimization in experiments.

### Weaknesses
1. The paper defines the RIM manifold as a subset of Euclidean space with strict inequalities: M = {X | X1 = 1, l < Xᵀ1 < u, X > 0}
However, this is an open set, and hence optimization problems posed over this set may have no solution, since the infimum may lie on the boundary (e.g., when X^T 1=u). The paper does not discuss this issue, and Theorem 1 claims it's an "embedded submanifold" without clarifying what happens near the boundary. This has major implications — e.g., the projection-based retraction (Theorem 5) may not be well-defined if the minimum lies at the boundary where feasibility breaks.

2. The paper claims that optimization on the Stiefel manifold has time complexity 
O(n^3), but this is misleading. For typical tall matrices (n≫c), the complexity is only O(nc^2), as shown in Wen and Yin (2013). Please correct this and cite:

       -Wen and Yin, A feasible method for optimization with orthogonality constraints, Math. Program. (2013)

Also, for retraction definitions, please cite standard sources:

       -Boumal, “An Introduction to Optimization on Smooth Manifolds”, 2023, or

       -Absil et al., “Optimization Algorithms on Matrix Manifolds”, Princeton, 2008

In Theorem 5, the projection used as a retraction may not always be well-defined — the feasibility set is open, and projection may fail to produce a point inside. The paper should acknowledge this and cite related projection-based retractions such as:

       Absil & Malick, Projection-like retractions on matrix manifolds, SIAM J. Optimization, 2012.

3. Every minor step is called a "Theorem", including obvious projections and inner product definitions. This makes the paper harder to read. Please distinguish between core theoretical contributions and auxiliary results.

### Questions
1. How do you guarantee the solution remains in the interior of the RIM manifold? Do you clip or project back when the optimizer moves outside?

2. Can the projection in Theorem 5 fail if the retracted point lies on the boundary (e.g., due to positivity or sum constraints)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces the Relaxed Indicator Matrix Manifold (RIM). The authors prove (M) is an embedded submanifold. They equip (M) with the Euclidean metric restricted to the manifold and derive a simple projection formula for the Riemannian gradient. 
For retraction, they give two options: (1) a norm‑minimizing projection that they show is a geodesic for small steps (Dykstra‑style algorithm, Theorem 5, and (2) a Sinkhorn‑style mapping characterized by diagonal scalings.

A central claim is lower per‑step complexity on RIM vs the doubly stochastic manifold (DSM) (Table 1).
  
Experiments: 
(i) retraction timing favors Dykstra as size grows (Table 2). 
(ii) “Experiment 2” compares RIM vs DSM on two problems: a convex norm‑approximation and TV image denoising.   
(iii) “Experiment 3” applies RIM to Ratio‑Cut with closed‑form Euclidean gradient/Hessian (Appendix A.9) and shows losses/times vs several baselines
(iv) “Experiment 4” reports clustering metrics (ACC/NMI/ARI) across datasets (Table 5). 

Authors also provide a convergence proof of RIM‑RGD to stationarity at (O(1/T)) under standard smoothness and Armijo/Wolfe conditions.

### Strengths
1. The RIM construction interpolates between single‑stochastic and doubly‑stochastic models via ([l,u]) bounds, letting practitioners encode prior class information. 
2. Clean Riemannian operators on RIM reduce to simple column‑mean corrections for the gradient and avoid DSM’s pseudoinverses, yielding the (O(nc)) vs (O(n^3)) complexity gap (Table 1).  
3. Multiple retractions are implemented and compared; Dykstra is both fast at large sizes and, by Theorem 5, induces a geodesic in the small‑step regime.  
4. Useful coverage of Ratio‑Cut with explicit Euclidean gradient and Hessian (Appendix A.9) that can be projected to RIM. 
5. Convergence guarantees for RIM‑RGD with Armijo/Wolfe steps are provided.

### Weaknesses
1. No approximation guarantees to the discrete indicator problem. The paper proves geometry and algorithmic convergence, not approximation quality: there is no integrality gap, rounding guarantee, or conditions under which RIM recovers the discrete optimum. The main theory sections and appendices focus on the toolbox, complexity, convergence, not on approximation bounds. A relaxation cannot just be fast, it must actually be a good approximation to the original problem (discrete indicator).
2. TV denoising evidence is weak. The claim relies on an ~10% objective gap (1.05e5 vs 1.17e5) and “zoomed‑in” visual inspection; no PSNR/SSIM/LPIPS or even data‑term MSE are reported near those figures/tables [p.8, text around Table 3; Fig.4]. 
3. Compute fairness needs clarification. Speedups depend on stopping rules, retraction choice, and any hyperparameter tuning (e.g., (\xi) in TV); these details are not fully standardized across RIM vs DSM in the description of Experiment 2. 
4. Choice of ([l,u]) is heuristic and dataset‑dependent. Appendix D.3.5 admits the difficulty, proposes using (n/c) or K‑means proportions, and shows a sensitivity table only for one dataset (MnistData05). 
5. Claims that “RIM images are clearer (regardless of noise level)” (page 47) are again qualitative and not well justified.

### Questions
1. Can you provide any approximation or rounding guarantee that connects a stationary point on RIM to a discrete indicator solution (or to DSM) with a bounded increase in the objective?
2. How should ([l,u]) be picked in practice without prior labels Do you recommend a data‑driven estimator beyond K‑means, and how sensitive are results across datasets?
3. In Experiment 2, did both manifolds use identical stopping rules and the same retraction and line‑search settings If not, how do the results change when matched?
4. Can authors provide concrete denoising metrics such as PSNR/SSIM/LPIPS/MSE for RIM vs DSM denoising experiments?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A new relaxation is introduced for the indicator matrix optimization problem, leading to a manifold with a simple structure. Algorithms are designed based on this manifold.

### Strengths
New retractions proposed and a class of efficient manifold algorithms are developed. Extensive numerical experiments have been conducted to validate the effectiveness of the proposed algorithms.

### Weaknesses
Presentation can be improved, e.g., "Our Code is presented in Appendix H", "The proof is included in A.5", etc.

### Questions
It is claimed after (1) that when $l=u=r$, the relaxation becomes $\\{X | X1_c=1_, X^T1_n=r, X>0\\}$. However, it seems to me that when $l=u=r$, the set in (1) is empty. What the tangent space of the manifold when $l=u$? Can you verify that the proposed retraction is still valid for this case?

### Soundness
3

### Presentation
2

### Contribution
3
