# Revisiting Tree-Sliced Wasserstein Distance Through the Lens of the Fermat–Weber Problem

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Tree-Sliced methods have emerged as an efficient and expressive alternative to the traditional Sliced Wasserstein distance, replacing one-dimensional projections with tree-structured metric spaces and leveraging a splitting mechanism to better capture the underlying topological structure of integration domains while maintaining low computational cost. At the core of this framework is the Tree-Sliced Wasserstein (TSW) distance, defined over probability measures in Euclidean spaces, along with several variants designed to enhance its performance. A fundamental distinction between SW and TSW lies in their sampling strategies—a component explored in the context of SW but often overlooked in comparisons. This omission is significant: whereas SW relies exclusively on directional projections, TSW incorporates both directional and positional information through its tree-based construction. This enhanced spatial sensitivity enables TSW to reflect the geometric structure of the underlying data more accurately. Building on this insight, we propose a novel variant of TSW that explicitly leverages positional information in its design. Inspired by the classical Fermat–Weber problem—which seeks a point minimizing the sum of distances to a given set of points—we introduce the Fermat–Weber Tree-Sliced Wasserstein (FW-TSW) distance. By incorporating geometric median principles into the tree construction process, FW-TSW notably further improves the performance of TSW while preserving its low computational cost. These improvements are empirically validated across diverse experiments, including diffusion model training and gradient flow. Our code is available at [https://github.com/thanhquangtran/FW-TSW](https://github.com/thanhquangtran/FW-TSW).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
While Optimal Transport (OT) allows to compare data in a 'meaningful' way, it is computionally costly. Slicing strategies allow for the definition of more efficient OT-based metrics.
In particular, this paper introduces a new variant of the Tree-Sliced Wasserstein (TSW) distance, called the Fermat-Weber Tree-Sliced Wasserstein (FW-TSW). It is presented as an efficient alternative to standard Sliced Wasserstein (SW), utilizing tree-structured metric spaces to better capture data geometry while maintaining low computational cost. The key insight leveraged is that TSW incorporates both directional and positional information distinguishing it from the purely directional SW. A key question for TSW is the generation of random trees. The authors propose FW-TSW, which explicitly integrates the geometric median (based on the Fermat-Weber problem) into the tree construction process to enhance this positional sensitivity. They claim that this leads to further performance improvements over TSW, which is empirically validated in experiments involving diffusion model training and gradient flows.

### Strengths
Novelty: Sampling strategies for sliced Wasserstein and kernel distances are an active field. Beyond projections onto the line, it is valuable to study sampling strategies for TSW, as the combined use of directional and positional information has been relatively underexplored. 
 
Structural Design: The integration of the geometric median into the tree construction mechanism is an intuitive apprach for sampling meaningful tree projections.
 
Theoretical Contribution: Since there are limited theoretical results for TSW, the bound presented in Theorem 4.5 is an interesting result that provides new theoretical insight into the properties of Tree-Sliced distances.
 
Empirical Validation: The experiments are targeted at typical applications of sliced OT and provide good support for the proposed FW-TSW method. There are extensive ablation studies in the supplementary material.

### Weaknesses
Limited Impact: While the question for 'smart' tree sampling is interesting and deserves further studiy, the contribution seems rather incremental.
 
Theoretical Complexity: Most of the theoretical results are assessed are  straightforward with the exception of  Theorem 4.5.
 
Doubts about empirical validation: Much of the experiments focus on comparing SW and TSW for a predefined number of projections. However, SW is faster, so it might be fairer to compare the two for a predefined run time. After all, the number of projections is very important for the Monte Carlo estimates.
 
Bound Limitation: The utility of the Theorem 4.5 bound is questionable due to a potentially large additive constant, especially for a high number of splits $k$. 
 
Applicability: The practical relevance of the demonstrated applications is limited outside of a dedicated Computational Optimal Transport research area. To my knowledge, TSW is not widely used for practical application (unlike standard OT or SW).
 
Clarity and Notation: Section 2.2 suffers from inconsistent and confusing notation. Key terms like $\mathbb T_k^d$ are undefined, and the subscript $l$ in Equation (4) is unexplained, hindering comprehension. The rather central closed-form Wasserstein formula for trees is not provided in the paper, but only in the references. It is also confusing that the supplementary material around (24) uses partly different notation.
 
References: Relevant references on sampling strategies for sliced divergences, e.g. (Hertrich et al., 2025) and (Sisouk et al., 2025) are missing.

### Questions
1. Scaling in (14): In Equation (14), why is the identity matrix $\mathbf{I}_d$ not scaled according to the data's covariance or variance?
 
2. Subscripts in (15): The measure dependence should be made explicit: the subscripts $\mu, \nu$ of $\Sigma$ should already appear in Equation (15).
 
3. Choice of Central Point: Why was the geometric median chosen over simply using the barycenter of the measures in the tree construction? Is it because of Wasserstein-1 instead of Wasserstein-2? Did you try using the barycenter?
 
4. $k$-Disjoint Union: In Equation (6), what is the precise definition and calculation of the distance on the $k$-times disjoint union of $\mathbb R$?
 
5. Typos and Formatting: Please correct the formatting error "Equation equation 4" (bottom of p. 6 and elsewhere). Ensure distinct notation is used for the exact distance (6) versus its approximation (8). Also, correct Equation (28) by removing the spurious subscript $l$ on the left-hand side.

### Soundness
3

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
This paper proposes the Fermat-Weber Tree-Sliced Wasserstein (FW-TSW) to solve a geometric inconsistency in the standard tree sliced Wasserstein (TSW) framework. While TSW is an $L_1$-based metric ($W_1$) and its tree-sampling is position-dependent, existing methods sample this position (the tree root $x$) from a Gaussian centered at the $L_2$ data mean. FW-TSW corrects this mismatch by centering its sampling distribution $\sigma_{FW}$ at the $L_1$ geometric median ($x^*$) which is the solution to the Fermat-Weber problem. This scheme is claimed to better capture data geometry and improve performance, adding only a negligible upfront cost to compute the median.

### Strengths
- This paper’s proposal to center the sampling distribution at the $L_1$ geometric median (the Fermat-Weber point) is the principled solution, aligning the geometry of the sampling space with the geometry of the metric.

- Experiment results, particularly in generative modeling (Table 3), are significant. The FW-TSW* variant achieves FID score (2.315) by making both the position ($x^*$) and directions ($\theta_i$) dependent on the measures being compared. This provides strong evidence that for high-dimensional generative tasks, data-agnostic slicing (like uniform sampling) is suboptimal, and a data-dependent discrepancy is superior.

- The cost of solving for the geometric median ($O(Tnd)$) is paid once per batch and does not dominate $O(Lkn \log n + Lkdn)$ TSW computation. The paper shows that this significant performance gain (e.g., in Table 3) is achieved with a negligible increase in wall-clock time, making it a drop-in replacement.

### Weaknesses
- FW-TSW is no longer a metric.

- The paper introduces two ideas: (1) data-dependent *positional* sampling (the core FW insight) and (2) data-dependent *directional* sampling (the FW-TSW* variant, eq 17). The paper’s narrative is built around the principled Fermat-Weber contribution, but the best empirical result (FID in Table 3) relies on the FW-TSW* variant, which includes the more heuristic directional sampling. This makes it difficult to attribute the performance. It is not clear if the central insight (the geometric median) is the key driver, or if the (less-justified) directional sampling is doing most of the work.

### Questions
- In the 25 Gaussians experiment (Table 1), TSW and FW-TSW methods achieve a better final error compared to SWGG, which converges faster initially. What properties of tree-slicing give this better final convergence, and does the Fermat-Weber centering improve the global optimization landscape beyond the standard TSW's mean-centering?

- The geometric median computation adds an $O(Tnd)$ cost (with $T = 100$) per batch, compared to the baseline's $O(nd)$ mean calculation. Is this iterative cost truly negligible for large $n$ and $d$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Fermat–Weber Tree-Sliced Wasserstein (FW-TSW) and FW-TSW* — new variants of the Tree-Sliced Wasserstein (TSW) distance that integrate positional information via the geometric median (Fermat–Weber point).

The main idea: previous TSW approaches sample tree intersection points (roots) heuristically (typically from a Gaussian centered at the data mean). In contrast, FW-TSW replaces this with a data-adaptive distribution centered at the geometric median of the combined source–target supports, computed via Weiszfeld’s algorithm.

This modification yields:

Better alignment between positional sampling and data geometry;

Theoretical guarantees: semi-metricity, Euclidean invariance, and boundedness;

Empirical gains across gradient flows, topic modeling, and diffusion model training — with comparable runtime to TSW and Db-TSW.

### Strengths
### **Strengths**

- **Novel conceptual link:**  
  Clever integration of *location theory* (Fermat–Weber / geometric median) into OT sampling, yielding a clean geometric interpretation that grounds the stochastic tree construction in data geometry.

- **Sound theoretical guarantees:**  
  The paper rigorously proves semi-metricity, symmetry, Euclidean invariance, and boundedness of the proposed FW-TSW distance. These properties ensure mathematical consistency while extending classical TSW theory.

- **Efficiency preserved:**  
  Despite adding geometric-median computation via Weiszfeld’s algorithm, the complexity remains $O(Lkn\log n + Lkdn)$, with only minor $O(Tnd)$ overhead — maintaining the practical efficiency of TSW.

- **Strong empirical performance:**  
  FW-TSW and FW-TSW consistently outperform SW, TSW, and Db-TSW baselines on gradient flow, topic modeling, and diffusion model tasks. Results show improved convergence, stability, and lower FID without sacrificing runtime.

### Weaknesses
### **Weaknesses**

- **Limited generality of the tree structure.**  
  The proposed FW-TSW and FW-TSW* are defined only on *star-shaped* tree systems, where all branches share a single root point (Eq. 15 / 18).  
  More general tree geometries—e.g., unions of multiple disjoint lines or hierarchical branching—cannot be represented within this framework.  
  This limits the flexibility of the model and its ability to capture more complex spatial relationships.

- **Unclear computational overhead.**  
  The complexity of Weiszfeld’s algorithm (used to compute the geometric median) is not analyzed or included in the overall runtime discussion.  
  While each iteration is \(O(nd)\), the number of iterations and convergence behavior can vary depending on data geometry, which may affect scalability in high dimensions.

- **Loss of metric property.**  
  FW-TSW and FW-TSW* do not satisfy the triangle inequality because the sampling distribution \(\sigma_{\text{FW},\mu,\nu}\) depends on the data pair \((\mu,\nu)\).  
  As discussed around Eq. (20), the distance is only a *semi-metric* when the sampling distribution is fixed.  
  This data-dependent design improves adaptivity but sacrifices the strict metric property of the original TSW.

- **Minor issue:**  
  Line 349 contains a small typo: “Equation equation 15.”

### Questions
1. Could you clarify whether the FW-TSW framework can be extended beyond the star-shaped tree assumption?  
   In particular, is it possible to construct tree systems with multiple intersection points or hierarchical branches under the same theoretical formulation?

2. How many iterations are typically required for the Weiszfeld algorithm in your experiments?  
   Did you observe any stability or convergence issues for high-dimensional datasets?

3. In particular, what is the benefit of \sigma_{dir,mu,\nu}  compared with U(S^{d-1})?

### Soundness
3

### Presentation
2

### Contribution
2
