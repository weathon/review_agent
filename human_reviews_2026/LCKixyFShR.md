# Soft Quality-Diversity Optimization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Quality-Diversity (QD) algorithms constitute a branch of optimization that is concerned with discovering a diverse and high-quality set of solutions to an optimization problem.
Current QD methods commonly maintain diversity by dividing the behavior space into discrete regions, ensuring that solutions are distributed across different parts of the space.
The QD problem is then solved by searching for the best solution in each region.
This approach to QD optimization poses challenges in large solution spaces, where storing many solutions is impractical, and in high-dimensional behavior spaces, where discretization becomes ineffective due to the curse of dimensionality.
We present an alternative framing of the QD problem, called \emph{Soft QD}, that sidesteps the need for discretizations.
We validate this formulation by demonstrating its desirable properties, such as monotonicity, and by relating its limiting behavior to the widely used QD Score metric.
Furthermore, we leverage it to derive a novel differentiable QD algorithm, \emph{Soft QD Using Approximated Diversity (SQUAD)}, and demonstrate empirically that it is competitive with current state of the art methods on standard benchmarks while offering better scalability to higher dimensional problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Soft Quality-Diversity (QD) optimization framework that avoids discretizing the behavior space. It defines the Soft QD Score, analyzes its properties (monotonicity, submodularity, limiting equivalence to traditional QD Score), and derives SQUAD, a differentiable algorithm based on a lower bound of the Soft QD Score. SQUAD optimizes via a quality term (maximizing solution quality) and a repulsive diversity term (spreading solutions). Experiments on Linear Projection, Image Composition, and Latent Space Illumination benchmarks show SQUAD outperforms baselines in high-dimensional scenarios.

### Strengths
The idea of "illumination" for addressing the high-dimensionality is interesting. A differentiable method of SQUAD is appreciated.

### Weaknesses
1. The core idea of continuous behavior space “illumination” overlaps with Kent et al. (2022)’s continuous QD Score. However, the paper only notes Kent et al.’s work as an evaluation tool, a more indepth discussion should be provided to support its novelty.
2. The paper claims SQUAD scales to high-dimensional spaces, but LP domain tests only go up to 16-dimensional behavior spaces. The impact of solution count  on computational efficiency and optimization performance is unexamined.
3. For baselines like CMA-MAEGA, grid search is used for LP/IC domains, but default parameters are adopted for LSI (due to computational constraints). Additionally, SQUAD uses a smaller population size (256 vs. default 1024) in LSI. This inconsistent tuning may prevent baselines from performing optimally.
4. QD often claims the diversity. In LSI, SQUAD outperforms baselines in QVS, but the paper does not validate if the “diversity” is semantically meaningful.

### Questions
1. what is the technical difference with Kent et al. (2022)’s continuous QD Score design?
2.  If generated "diverse" solutions differ in real-world attributes rather than just behavioral space coordinates?
3.  Why the logit transformation is selected for SQUAD, instead of other transformations  (e.g., Box-Cox, standardization)?

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
This paper identifies a key limitation in current Quality-Diversity (QD) optimization methods: reliance on discrete archives (tessellations) to maintain diversity, which suffers from the curse of dimensionality in high-dimensional behavior spaces. To address this, the authors propose "Soft QD," a continuous relaxation of the QD problem. They define a new objective, the "Soft QD Score," which models each solution as a light source with a Gaussian field of influence, integrating the maximum influence over the entire behavior space.

Because directly maximizing this integral is intractable, the authors derive a differentiable lower bound that simplifies to a standard quality maximization term minus a pairwise repulsion term (weighted by quality and proximity). This leads to a new algorithm, SQUAD (Soft QD Using Approximated Diversity). Empirical results on Linear Projection (up to 16D behavior space), Image Composition, and Latent Space Illumination benchmarks demonstrate that SQUAD scales significantly better to high-dimensional behavior spaces than state-of-the-art differentiable QD baselines (like CMA-MAEGA) and offers a controllable trade-off between quality and diversity via its kernel bandwidth hyperparameter.

### Strengths
- Novel Problem Formulation: Shifting the QD objective from a discrete, archive-based metric to a continuous, integral-based "illumination" field is a significant and original theoretical contribution. It elegantly sidesteps the arbitrary nature of defining grid resolutions or pre-computing tessellations.

- Scalability to High-Dimensional Behavior Spaces: The empirical results strongly support the core claim that this method handles higher-dimensional behavior spaces better than archive-based counterparts. Figure 3 shows standard methods (CMA-MEGA/MAEGA) degrading at $d=16$ while SQUAD maintains performance. Besides, by framing the problem as a unified differentiable objective, SQUAD can seamlessly leverage modern optimizers like Adam, simplifying the typical QD loop.

- Theoretical Grounding: The derivation of SQUAD from the Soft QD Score via Bonferroni inequalities and geometric mean approximations (Theorem 2 and Appendix A) provides a solid theoretical foundation for the proposed pairwise repulsion objective, rather than just proposing it as a heuristic.

- Clarity and Presentation: The paper is exceptionally well-written. The conceptual difference between hard and soft archives is intuitively visualized in Figure 1.

### Weaknesses
- Hyperparameter Sensitivity ($\gamma$): The method's performance and its trade-off between quality and diversity are heavily dependent on the kernel bandwidth $\gamma^2$ (Section 5.3). While this provides controllability, it also introduces a critical hyperparameter that might be difficult to tune a priori for new domains compared to setting a grid resolution.

- Bounded Space Reliance: Appendix C.3 highlights that the logit transformation for bounded spaces is "critical" for success. This suggests the method might struggle with intrinsically unbounded behavior spaces or spaces where appropriate transformations are unknown, limiting generality slightly compared to robust binning strategies.

- Computational complexity of pairwise interactions: While standard QD archive insertions are typically $O(1)$ (grids) or roughly logarithmic (CVT trees) per solution, SQUAD's objective involves pairwise interactions. Although mitigated by KNN ($O(Nk)$), this could potentially become a bottleneck for very large populations compared to purely archive-based approaches, though it is likely an acceptable trade-off for high-dimensional capabilities.

### Questions
1. Adaptive Bandwidth: Given the strong influence of $\gamma^2$ on the quality-diversity trade-off (Figure 4), did you explore any mechanisms for adapting $\gamma^2$ online? For instance, starting with a larger $\gamma$ for broad exploration and annealing it for fine-grained illumination later in training?

2. Wall-clock time: Could you provide more details on the wall-clock time comparison between SQUAD (using its batch/KNN approximations) and the highly optimized pyribs baselines, particularly as the population size $N$ scales up?

3. Deceptive Landscapes: How does SQUAD behave in highly deceptive behavior landscapes? Does the continuous repulsion field ever prevent it from traversing narrow "corridors" in behavior space that a discrete archive might serendipitously cross?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Quality-Diversity (QD) algorithms typically discretize the behavior space into cells, find the best solution in each cell, and use the QD Score (i.e., the sum of the fitness value of the solution in each cell) as the objective. However, the discretizing mechanism makes it difficult to optimize end-to-end with gradient-based optimizers, and the number of cells grows exponentially when the dimension of the behavior space is large. For QD problems with differentiable fitness and behavior functions, this paper introduces a new objective, the Soft QD Score, to solve these issues. The behavior value is defined as the maximum fitness exponentially decreasing with distance among the population, and the Soft QD Score is defined as the integral of the behavior value over the behavior space. The Soft QD Score is then approximated by a computable and differentiable form, consisting of a quality term and a distance-based diversity term, which can be optimized directly with gradient-based optimizers. Experimental results on three differentiable QD (DQD) problems demonstrate that the proposed method outperforms other DQD methods significantly when the dimensions of behavior spaces are large.

### Strengths
- The proposed new objective bypasses the issues arising from behavior space discretizing, and enables it to be optimized by gradient-based optimizers easily.
- The proposed method performs well on DQD tasks with high-dimensional behavior spaces.
- The paper is well-organized and easy to follow. The code is available, contributing to reproducibility.

### Weaknesses
- The experiments are limited. The proposed method is not evaluated on high-dimensional reinforcement learning tasks, which are a more important type of DQD problem. Consequently, it is not compared with state-of-the-art methods that utilize policy gradients effectively.
- There are also other QD methods that do not discretize the behavior space (e.g., novelty search-based methods). They are not compared in the experiments.
- It is unclear whether the outstanding performance comes from the definition of the Soft QD Score, the end-to-end gradient-based optimization, or the population management.

### Questions
- How will the baselines perform if they use the Soft QD Score as the objective?
- What hyperparameter values do the baselines use in the experiments? These values may affect the performance of the baselines.
- Note that neighbor computing of batches might be time-consuming, since it applies $O(MNk)$ pairwise repulsions for each batch. Can you compare the running time of the algorithms?

### Soundness
3

### Presentation
4

### Contribution
3
