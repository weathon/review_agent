# GeoDM: Geometry-aware Distribution Matching for Dataset Distillation

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Dataset distillation aims to synthesize a compact subset of the original data, enabling models trained on it to achieve performance comparable to those trained on the original large dataset. Existing distribution-matching methods are confined to Euclidean spaces, making them only capture linear structures and overlook the intrinsic geometry of real data, e.g., curvature. 
However, high-dimensional data often lie on low-dimensional manifolds, suggesting that dataset distillation should have the distilled data manifold aligned with the original data manifold. 
In this work, we propose a geometry-aware distribution-matching framework, called **GeoDM**, which operates in the Cartesian product of Euclidean, hyperbolic, and spherical manifolds, with flat, hierarchical, and cyclical structures all captured by a unified representation. 
To adapt to the underlying data geometry, we introduce learnable curvature and weight parameters for three kinds of geometries. At the same time, we design an optimal transport loss to enhance the distribution fidelity. 
Our theoretical analysis shows that the geometry-aware distribution matching in a product space yields a smaller generalization error bound than the Euclidean counterparts. Extensive experiments conducted on standard benchmarks demonstrate that our algorithm outperforms state-of-the-art data distillation methods and remains effective across various distribution-matching strategies for the single geometries.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces GeoDM, a geometry-aware dataset distillation method that performs distribution matching in a product space composed of Euclidean, hyperbolic, and spherical components. By learning per-geometry curvatures and weights and by adding a geometry-consistent optimal transport loss, the method aims to better preserve the intrinsic structure of real data than standard Euclidean distribution matching.

### Strengths
1. The paper argues clearly that real datasets may contain mixed geometric structures (flat, hierarchical, angular) and that matching only in Euclidean space can be limiting. Modeling in a mixed-curvature/product space is a natural response to this.

2. The combination of multiple geometries, learnable curvature, and an OT-based alignment term is conceptually clean and aligned with the stated goal of structure-preserving distillation.

3. The work provides a generalization argument showing that matching in a richer geometry can, under reasonable assumptions, lead to tighter approximation than purely Euclidean matching.

### Weaknesses
1. All results are reported on classic benchmarks such as MNIST and CIFAR (10/100), mostly under very low IPC. There is no evidence on medium- or large-scale, higher-resolution, or more diverse datasets. As a result, it is unclear whether the proposed geometric modeling remains useful or practical when the data manifold is more complex, when backbones are deeper, or when the distilled set must support stronger augmentation.

2. Running multiple geometries in parallel, learning curvatures, and adding an OT loss are nontrivial additions to a standard distillation pipeline. The paper does not provide a clear comparison of training time and memory usage against recent, efficient distillation methods.

3. The method uses a fixed-dimensional split across Euclidean, hyperbolic, and spherical parts for all datasets. This makes the method stable on small benchmarks but raises questions about how well it adapts to larger, more structured data.

4. The paper suggests that many real datasets are non-Euclidean, but it does not include experiments on clearly hierarchical, directional, or cross-domain data where the proposed product space would be most justified.

5. It is difficult to tell how much of the gain comes from the geometry-aware OT term itself versus the move to a product manifold. Simpler variants (e.g., OT in the dominant geometry only) are not explored.

### Questions
1. Can you provide results on at least one medium- or large-scale dataset (e.g., a 224×224 ImageNet setting) to show that the method does not break down or become too expensive at scale?

2. What is the actual wall-clock time and memory overhead of GeoDM compared with a strong Euclidean distribution-matching baseline when using the same backbone and training schedule?

3. Did you try making the allocation of dimensions to the three geometries learnable or data-dependent, and if so, what stability or performance issues arose?

4. For datasets without obvious hierarchical or angular structure, do the learned weights tend to collapse to the Euclidean component, and if that happens, can the method skip the curved parts to save computation?

5. Could a simpler OT variant (for example, OT only in the geometry with the largest learned weight) achieve similar performance with a lower cost?

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
Problem: The paper tries to tackle the limitation of dataset-distillation methods that perform distribution matching only in Euclidean latent spaces, which can miss non-Euclidean structure (e.g., hierarchical or directional/cyclical patterns) present in real data.

Motivation: Under the manifold hypothesis, data may live on curved manifolds; embedding and matching in a space that can express Euclidean, hyperbolic, and spherical geometry should better preserve task-relevant structure than a flat space.

Proposed solution: (1) Perform distribution matching in a product Riemannian space combining Euclidean, hyperbolic, and spherical factors, implemented with a Riemannian CNN to produce geometry-aware features. (2) Use learnable curvatures for the non-Euclidean branches and map real/synthetic data into the weighted product space; align them with a DM objective (e.g., NCFM). (3) Add a geometry-aware optimal-transport loss computed in the product space to couple the factors and preserve class-conditional mass; include curvature regularization in the total loss.

Experiments: On standard benchmarks, the method outperforms distillation baselines and include ablation studies.

### Strengths
- It explains why doing dataset distillation only in Euclidean space can miss real data geometry.
- It proposes distribution matching in a product space (Euclidean + hyperbolic + spherical) so each type of structure can be represented.
- The curvatures and the weights of the three geometries are learnable, letting the method adapt to each dataset.
- A geometry-aware OT loss aligns real and synthetic data across the three components and avoids one component dominating.
- Theory sounds, which tried to state that doing distribution matching in a product space (Euclidean × hyperbolic × spherical) gives a strictly tighter generalization-error bound than doing it in a single Euclidean space.
- Experiment includes ablation studies.

### Weaknesses
- Theory rests on specific assumptions. The analysis relies on “mild regularity” assumptions and constant-curvature product spaces (Euclidean, hyperbolic, spherical); real data may not fit these perfectly.
- The model fixes the dimensionality of each manifold factor.
- The method introduces learnable curvature, geometry weights, and an OT term with its own coefficient/regularization, hence more components and hyperparameters to manage.
- Added complexity. The approach uses a product of three geometries with a Riemannian CNN plus an OT loss, which increases modeling and training complexity compared to standard Euclidean DM. Please provide experiments on the tradeoff between performance and complexity among proposed method and baselines.
- Experiments are on MNIST (1, 28, 28), CIFAR-10 (3, 32, 32), and CIFAR-100 (3, 32, 32) only (small/medium scale). Please provide experiments on other large scale datasets.
- Theoretical results are upper-bound guarantees (tighter than Euclidean); they do not directly quantify runtime/compute or guarantee gains on tasks beyond those tested.
- The gain of using OT vs baseline is insignificant (71.8 vs 72.3) but the tradeoff is more complexities and higher training time because OT is not scalable. Please provide experiments to show the tradeoff clearly so readers can judge whether the ~0.5 pp gain justifies the added complexity.

### Questions
Please address all my concerns in Weaknesses section. Besides, I have some additional questions:
- How were manifold dimensions per branch chosen, and did you test making them learnable?
- How sensitive are results to the curvature initialization and to the geometry-weight initialization?

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
4

### Summary
The paper extends dataset distillation by embedding real and synthetic data into a learned product space combining Euclidean, hyperbolic, and spherical geometries. It learns curvature and geometry weights and adds an OT term to preserve class-level mass. The goal is to respect non-Euclidean data structure during distribution matching. Experiments on several small benchmarks show consistent improvements.

### Strengths
- the motivation of the paper is very clear and intuitive, Euclidean latent spaces likely miss curvature.

- the main idea is conceptually very intuitive to follow, a combination of several similar modules, and learnt weights.

### Weaknesses
- The paper compares single geometry vs three, but omits two-geometry combinations (E+H, E+S, H+S) in ablation studies. Without this, the claim that all three curvatures matter remains unverified. 

- Some of the assumptions might be too unrealistic, for example, uniform algorithmic stability is unlikely satisfied by deep non-convex training. Empirical check to support the relevance of the theoretical terms will be necessary. 

- The method introduces many complicated components, and whether the performance gain is worth the complexities is of question. runtime or memory comparison is better provided for context of the practical gain introduced by the complicated method.

### Questions
It will be good to offer empirical evidence for the three questions listed in weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GeoDM, a geometry-aware dataset distillation framework that performs distribution matching in a product manifold combining Euclidean, hyperbolic, and spherical spaces. The main motivation is that existing distribution-matching methods operate solely in Euclidean spaces and fail to capture intrinsic geometric structures of data like hierarchical or cyclical patterns. The authors introduce learnable curvature parameters and weights for different geometries, along with an optimal transport loss to align real and synthetic data distributions. Theoretical analysis shows tighter generalization bounds versus Euclidean-only approaches, and experiments on several datasets demonstrate consistent improvements over state-of-the-art baselines.

### Strengths
1. The connection between manifold hypothesis and dataset distillation is intuitive and clearly articulated. Figure 1 effectively demonstrates that data exhibits non-Euclidean geometric structure that Euclidean spaces fail to capture.

2. The experiments cover multiple datasets, baselines, and ablation studies. The robustness across different distribution matching methods (DM, DSDM) and cross-architecture evaluation demonstrate generalizability.

3. Theorems 4.1 and 4.2 provide mathematical justification for the approach, decomposing the error into statistical, stability, and geometric components, which offers insight into why geometry matters.

### Weaknesses
1. The use of product manifolds, Riemannian CNNs, hyperbolic/spherical embeddings, and optimal transport are all well-established techniques. The main contribution is combining them for dataset distillation, which feels somewhat incremental. The paper would benefit from deeper insights into why this particular combination works.

2. While GeoDM consistently outperforms baselines, the gains are often 1-3%, which may not justify the substantial increase in complexity (three geometry branches, learnable curvatures, OT loss). The computational cost is not discussed, but the method likely requires significantly more resources than Euclidean baselines.

3. Assumption 4.1 is quite strong (e.g., assuming data lies on a mixed-curvature product manifold), and it's unclear how realistic this is for vision datasets like CIFAR-10. The paper claims this is "empirically grounded" but doesn't provide evidence that CIFAR-10 actually exhibits this structure beyond the 3D visualization in Figure 1.

4. Several decisions appear arbitrary: Why fix dimensions (dE, dH, dS) rather than learn them? How sensitive is performance to these choices? The curvature regularization terms (Eq. 3) seem ad-hoc—why penalize deviation from the radius in these specific ways? What happens with different regularizers?

5. How are dimensions allocated across geometries? What is the computational overhead compared to baselines? How does performance vary with different dimension allocations? The paper mentions fixing dimensions "as varying dimensionality often introduces extra degrees of freedom" but provides no empirical support.

### Questions
1. How do you determine the split of dimensions across Euclidean, hyperbolic, and spherical components? Is there a principled way to set dE, dH, dS, or is it purely empirical? An ablation study on different dimension configurations would strengthen the paper.

2. What is the training time and memory overhead of GeoDM compared to NCFM or other baselines? Given the modest accuracy improvements, understanding the cost-benefit tradeoff is important for practical adoption. Can you provide wall-clock time comparisons and discuss whether the gains justify the added complexity?

### Soundness
3

### Presentation
3

### Contribution
3
