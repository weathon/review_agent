# Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
This paper introduces a new constraint-free concave dual formulation for the Wasserstein barycenter. Tailoring the vanilla dual gradient ascent algorithm to the Sobolev geometry, we derive a scalable Sobolev gradient ascent (SGA) algorithm to compute the barycenter for input distributions supported on a regular grid. Despite the algorithmic simplicity, we provide a global convergence analysis that achieves the same rate as the classical subgradient descent methods for minimizing nonsmooth convex functions in the Euclidean space. A central feature of our SGA algorithm is that the computationally expensive $c$-concavity projection operator enforced on the Kantorovich dual potentials is unnecessary to guarantee convergence, leading to significant algorithmic and theoretical simplifications over all existing primal and dual methods for computing the exact barycenter. Our numerical experiments demonstrate the superior empirical performance of SGA over the existing optimal transport barycenter solvers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper,  the authors propose an algorithm for computation of the exact Wasserstein barycenter for a collection of input distributions discretized over a regular grid. The algorithm is based on the idea of Sobolev gradient descent and is tested in several experimental setups.

### Strengths
In general, the paper is well-written and provides an algorithm supported with theoretical analysis and experimental illustrations in several setups.

### Weaknesses
My main concerns are related to the limited practical usefulness of the developed algorithm. For example, as the authors claim themself in lines 101-105, the developed algorithm is not suitable for the distributions of the dimensions higher than 3D. Besides, the algorithm is designed for the computation of the optimal transport barycenter under the quadratic cost function assumption. This point also could be listed among the limitations of the developed methodology. While I understand that the listed limitations might correspond not to the algorithm itself, but rather to the particular  class of the algorithms for exact barycenter computation, I have other concerns regarding the demonstration of the practicality of the proposed algorithm.

Specifically, all of the experimental results only show the ability of the method to compute the barycenter of distributions, but do not assess its ability to approximate well the optimal transport (OT) maps between each of the distributions and barycenter. I think that it is an important aspect since the learned maps have more practical use cases than the barycenter itself. Meanwhile, the authors do not perform comparison with the ground-truth barycenters which are known, e.g., for Gaussian distributions. Thus, I kindly suggest the authors to perform comparison of their approach with the ground-truth barycenter and OT maps in the setting with Gaussian distributions which are known thanks to (Chewi et al., 2020). 

The related work section lacks the overview of recent methods for approximating the barycenter of continuous distributions (Fan et al, 2021; Chi et al., 2023; Noble et al., 2023; Kolesov et al, 2024a,b). I think these algorithms deserve mentioning since in lines 106-107 you write that “almost all existing barycenter algorithms are limited to 2D problems” which is not true when we are talking about the methods listed above. Thus, I kindly suggest the authors add the references in the related work sections and make the phrase in lines 106-107 more clear.

**Overall**, I think that the experimental evaluation of the proposed approach  should be improved since now it lacks comparison with the ground-truth barycenters and OT maps which is important for clearly revealing the performance of the developed approach.

**References.** 

Fan, J., Taghvaei, A., and Chen, Y. Scalable computations of wasserstein barycenter via input convex neu- ral networks. In Meila, M. and Zhang, T. (eds.), Pro- ceedings of the 38th International Conference on Ma- chine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 1571–1581. PMLR, 18–24 Jul 2021.

Chi, J., Yang, Z., Li, X., Ouyang, J., and Guan, R. Varia- tional wasserstein barycenters with c-cyclical monotonic- ity regularization. In Proceedings of the AAAI Confer- ence on Artificial Intelligence, volume 37, pp. 7157–7165, 2023.

Noble, M., Bortoli, V. D., Doucet, A., and Durmus, A. Tree-based diffusion schro ̈dinger bridge with applica- tions to wasserstein barycenters. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Kolesov, A., Mokrov, P., Udovichenko, I., Gazdieva, M., Pammer, G., Burnaev, E., and Korotin, A. Energy-guided continuous entropic barycenter estimation for general costs. Advances in Neural Information Processing Systems, 2024a

Kolesov, A., Mokrov, P., Udovichenko, I., Gazdieva, M., Pammer, G., Burnaev, E., and Korotin, A. Estimating barycenters of distributions with neural optimal transport. In Forty-first International Conference on Machine Learning, 2024b

Sinho Chewi, Tyler Maunu, Philippe Rigollet, and Austin J. Stromme. Gradient descent algorithms for Bures-Wasserstein barycenters. In Jacob Abernethy and Shivani Agarwal (eds.), Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pp. 1276–1304. PMLR, 09–12 Jul 2020.

### Questions
- Could you perform comparison with the ground-truth barycenter and OT maps in the Gaussian setup?
- Could your approach be extended to the case of more general cost functions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel, constraint-free, and concave dual formulation for the Wasserstein barycenter problem. By removing the constraints present in previous dual formulations, the authors develop a scalable Sobolev Gradient Ascent (SGA) algorithm. A key theoretical contribution is a global convergence analysis for the proposed method. The efficacy of the algorithm is validated through numerical experiments that show superior performance over existing methods.

### Strengths
The paper focuses on a fundamental problem of Barycenter Optimization. Theorem 4 is the paper's cornerstone. It introduces a constraint-free concave dual formulation for the Wasserstein barycenter problem. This key result makes the problem amenable to optimization via SGA and allows the authors to establish a global convergence guarantee.

### Weaknesses
1. The method's efficiency hinges on a regular grid, which allows for an efficient FFT-based solver for the inverse Laplacian. It would significantly strengthen the paper to discuss the adaptations required for irregular meshes, where one would need to employ more computationally expensive methods.
2. The grid-based approach is subject to the curse of dimensionality, meaning its computational cost grows exponentially with the number of dimensions. To their credit, the authors explicitly acknowledge this and correctly position their method for low-dimensional (2D and 3D) problems.

### Questions
Please fix a hanging citation "Matthew Jacobs and Bohan Zhou. The Signed Wasserstein Barycenters, 2025". It does not seem to exist?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a constraint-free concave dual formulation for the Wasserstein barycenter and a dual ascent algorithm adapted to the Sobolev geometry. The algorithm matches the complexity rate of the subgradient method. A key advantage is that the new dual algorithm bypasses the expensive c-concavity projection and achieves high efficiency. The experiments demonstrate the advantages of the proposed method.

### Strengths
This paper presents a novel concave dual formulation for the Wasserstein barycenter problem that is unconstrained yet achieves strong duality. The theoretical analysis appears reasonable and expected. Empirically, the authors demonstrate that SGA is significantly faster and more numerically stable.

### Weaknesses
The stepsize rule seems impractical—it requires an M that depends on all iterates. It's somewhat bizarre that the stepsize depends on future iterates. If the stepsize can be chosen more freely, please provide a detailed convergence rate analysis.

Additionally, would an adaptive gradient method (such as AdaGrad) perform well in your examples?

As the authors pointed out, the algorithm's complexity scales with the grid size $n$, which scales *exponentially* with the dimension $d$. This effectively limits the current method to low-dimensional problems ($d=2, 3$). This could be a practical constraint for many machine learning applications.

The real-world example using electric scooter tracking data appears quite simple. Are there more challenging datasets in this area?

### Questions
See the weakness part

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed a constraint-free and concave formulation of the Wasserstein barycenter optimization problem which achieves strong duality.  The paper derived a scalable Sobolev gradient ascent (SGA) algorithm without the computationally expensive c-concave projection steps. Then, it showed that the proposed SGA algorithm achieves the same rate as the classical subgradient descent methods for minimizing nonsmooth convex functions in the Euclidean space. Numerical experiments demonstrate the effectiveness of SGA over the existing optimal transport barycenter solvers.

### Strengths
Overall I found this paper to be interesting and provided a good theoretical contribution in computing Wasserstein barycenter efficiently:
-  the constraint-free concave dual formulation allows the algorithm to avoid the OT map computations in the primal problem
-  the proposed Sobolev gradient ascent (SGA) algorithm is simple and straightforward, but matches the rate of classical subgradient descent for minimizing nonsmooth convex functions in the Euclidean space.
- with synthetic and real world data, it was shown that SGA achieved comparable or better performance than existing baselines.

### Weaknesses
- In comparison to the theoretical results, I found the numerical experiments to be on the weaker side, with limited data involved in testing SGA. For real-world data, only one set of video frames was used to demonstrate SGA's performance, and the comparison between WDHA and SGA's performance was not very obvious. WDHA also captured the trajectory of the moving object, but for some reason the picture appeared to be darker. The paper will be strengthened if the real-world data experiments part can be more comprehensive and quantitative.

### Questions
Is the convergence rate for SGA in theorem 6 optimal?

### Soundness
3

### Presentation
3

### Contribution
2
