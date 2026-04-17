# Slicing Wasserstein over Wasserstein via Functional Optimal Transport

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Wasserstein distances define a metric between probability measures on arbitrary metric spaces, 
including *meta-measures* (measures over measures).
The resulting *Wasserstein over Wasserstein* (WoW) distance is a powerful, but computationally costly tool for comparing datasets or distributions over images and shapes.
Existing sliced WoW accelerations rely on parametric meta-measures or the existence of high-order moments, leading to numerical instability. As an alternative, we propose to leverage the isometry between the 1d Wasserstein space and the quantile functions in the function space $L_2([0,1])$.
For this purpose, we introduce a general sliced Wasserstein framework for arbitrary Banach spaces. 
Due to the 1d Wasserstein isometry, 
this framework defines a sliced distance between 1d meta-measures via infinite-dimensional $L_2$-projections, 
parametrized by Gaussian processes. 
Combining this 1d construction with classical integration over the Euclidean unit sphere yields the *double-sliced Wasserstein* (DSW) metric for general meta-measures. We show that DSW minimization is equivalent to WoW minimization for discretized meta-measures, while avoiding unstable higher-order moments and  computational savings. Numerical experiments on datasets, shapes, and images validate DSW as a scalable substitute for the WoW distance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a new computationally efficient way to compare distributions of distributions using sliced optimal transport (SOT). Traditional Wasserstein over Wasserstein (WoW) distances are powerful but computationally expensive and unstable when applied to high-dimensional or hierarchical data. To overcome this, the authors propose the Double-Sliced Wasserstein (DSW) metric, which generalizes sliced Wasserstein distances to arbitrary Banach spaces. Leveraging the isometry between one-dimensional Wasserstein and 
$L_2([0,1])$ spaces, they define infinite-dimensional slicing using Gaussian-process–based projections, followed by classical Euclidean slicing. Theoretically, DSW is shown to be topologically equivalent to WoW for discretized meta-measures, offering similar accuracy with far less computational cost. Experiments on datasets, 3D shapes, point clouds, and image distributions demonstrate that DSW achieves comparable or superior results to existing WoW and sliced-OT methods, making it a scalable and robust replacement for hierarchical OT computations.

### Strengths
1. The paper rigorously builds upon optimal transport theory, generalizing sliced Wasserstein distances to arbitrary Banach spaces and proving metric properties and equivalence to the original WoW distance.

2. The proposed Double-Sliced Wasserstein (DSW) metric is conceptually original, combining functional (infinite-dimensional) slicing via Gaussian processes with standard geometric slicing, which is both elegant and theoretically justified.

3. DSW substantially reduces the cost of computing hierarchical OT distances compared to WoW and OTDD, making it scalable to large and high-dimensional datasets.

4. By avoiding moment-based approximations (used in previous sliced WoW methods), DSW eliminates numerical instability issues that occur when high-order moments are poorly estimated.

5. Demonstrations across domains (shape analysis, dataset comparison, point clouds, and image distributions) show that the method is general-purpose and effective for both geometric and perceptual tasks.

6.  Experiments are well-chosen and show DSW’s performance to be comparable to or better than established baselines, often with much lower runtime.

### Weaknesses
1. While equivalence to WoW is shown for discretized measures, the paper lacks deeper analysis of approximation error, sample complexity, or convergence rates of Monte Carlo integration in DSW. However, these investigation seems to be challenging which might be quite beyond of the scope of the paper. 

2.  DSW requires choices such as the Gaussian process kernel width, number of projections (S, R), and quadrature parameters, yet sensitivity analysis or principled tuning guidelines are limited.

3. Despite lower asymptotic cost, the implementation still involves multiple nested integrations (spherical and functional), which may be nontrivial to optimize or parallelize in practice.

### Questions
1.  The approach is closely related to  sliced Wasserstein embedding (see Section 4.6 in [1]) which exploits the relationship between sliced Wasserstein space and $L_2$ function space. There are some related aspect should be discussed. For example, instead of double-slicing, I can define SW over SW embeddings (a function or vectors if quantization is used).

2. Why $L_2([0,1])$ is used? As in SW embedding, we can map to $L_2(\mathbb{R})$ using a fixed reference function. 

3. A minor comment is that the acronym “DSW” overlaps with “Distributional Sliced Wasserstein” in [2], which also involves modifying the slicing distribution.

[1] An Introduction to Sliced Optimal Transport, Khai Nguyen.

[2] Distributional Sliced-Wasserstein and Applications to Generative Modeling, Nguyen et al

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
3

### Summary
In this paper the authors propose an alternative to the Wasserstein
over Wasserstein (WoW) distance between measures over measures. This amounts of
slicing once in the original space all the distributions and then slicing a
second time the resulting 1D distribution in the set of quantile functions using 
directions corresponding to 1D Gaussian distributions. The resulting approximation
is called the Double Sliced Wasserstein (DSW) distance. The authors also show 
that DSW can be computed efficiently and converges to 0 similarly to WoW.
Numerical experiments on shape classification, dataset distances, distributions
of point clouds and image datasets seen as distributions of patches show the
potential of the proposed approach.

### Strengths
+ The paper is well written and clear. 

+ It provides an efficient way to compute similarity between meta distributions,
  and will scale much better to large distributions than existing approaches.

+ Nice theoretical results and discussion on slicing Banach space and 1D
  Wasserstein space.

+ Interesting experiments on meta distribution learning tasks that suggests that the
  proposed approach can be interesting in practice.

+ The method can be directly used to approximate the Third Lower Bound (TLB) of
  Memoli which is a very interesting application on challenging data.

### Weaknesses
+ The paper is missing a sensitivity analysis to important parameters such as S
  (number of slices in the original space) and R (number of slices in the
  quantile space) or even the variance $\sigma$ in the Gaussian kernel. This is
  important to understand the tradeoff between computational time and accuracy
  of the approximation. Illustrating variances over multiple runs would also be
  useful since the method is stochastic and Monte Carlo based approximation is
  done here twice and in very large spaces.

+ Application to shape distributions in 5.1 is very nice but could be made
  stronger. Why is K=3 chosen for KNN? For a fair comparison, K should be
  cross-validated for each metric. Also how correlated in DSW to TLB? to GW?
  This is an efficient alternative so how close it is to other metrics is
  important. Also the computational time is actually a bit underwhelming where
  DSW is faster than TLB only on 2 of the 4 datasets. The authors should discuss
  this.

+ Experiments for OTDD in 5.2 are rather limited and do not focus on what is
  very importante for OTDD, which is the estimation of the performance gap
  between models trained on different datasets. It is possible that the proposed
  approach actually works better than OTDD or s-OTDD in this context but this is
  not investigated. Also R=10 is very small for slicing an infinite dimensional
  space, and the method could be more correlated with larger R but this was not
  investigated.

### Questions
Please address the weakness comments above and the following questions:

+ In Table 1, what does "Time" refer to? Is it the average time to compute the
  distance between a pair of meta distributions? In this case could you also report
  the variance?

+ Could you be more consistent and use DSW everywhere instead of "Ours"?

+ The proposed Gaussian slicing is important but could faster approach work well
  (theoretically or in practice)? Could we use 0 variance, which is the step
  function as a more efficient a basis not requiring quadrature ? 

+ On OTDD experiments It would be interesting to see how DSW compares to other
  scalable OT methods such as OT on GMM where the components of the mixture can
  be estimated on each of the sub-distributions (e.g. Delon 2020 "A
  Wasserstein-type distance in the space of Gaussian Mixture Models").

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The objective of this work is to study Optimal Transport (OT) on measures over measures, so called meta-measures, aiming for computational efficiency. As noted by the authors, "a key feature of OT is its applicability to non-Euclidean spaces, even allowing the definition of Wasserstein distances on Wasserstein spaces". This leads to the Wasserstein over Wasserstein (WoW) distance, which is, however, computationally very costly.
Another important existing dissimilarity measure in this "multilevel OT" paradigm is the OT dataset distance (OTDD). OTDD was developed to quantify similarities between labeled datasets in a model-agnostic manner, making it particularly useful for transfer learning. However, OTDD is also highly expensive to compute, and slicing-based variants (s-OTDD) have been proposed to mitigate this. In addition, in applications such as evaluating point cloud generative models, the OT nearest-neighbor accuracy (OT-NNA) test is widely used, but it is computationally demanding too. Also, the Gromov-Wasserstein distance falls under this underlying framework, by treating metric measure spaces as measures over measures. However, one more time, its computation is contly and also it is a non-convex optimization.  

In response, the authors propose a general sliced Wasserstein framework for measures on Banach spaces. The theoretical foundation relies on the isometry between the one-dimensional Wasserstein distance and L2([0,1]) via the quantile mapping, as well as slicing techniques in Hilbert spaces (generalizing the inner product to Banach space by utilizing the pairing with linear functionals on its dual space). This allows them to introduce first the slice-quantile WoW (SQW) metric, which roughly speaking corresponds to 1d meta-measures, and then, more importantly, the double-sliced Wasserstein distance (DSW). On the practical side, for particularly sampling, they employ parameterized Gaussian-processes. They demonstrate that DSW can serve as an efficient surrogate for the previously mentioned meta-measures dissimilarities, such as,  WoW, OTDD, s-OTDD, OT-NNA, and GW.

### Strengths
- The paper is very well organized. 
- Previous works are clearly discussed. 
- The aims and problems are stated in a clear way both from the theoretical and practical perspectives. 
- The results, formulated as theorems, propositions, and corollaries, highlight important properties of the proposed dissimilarities. For the generalization of the sliced Wasserstein (SW) distance in Banach spaces, since the unit ball is no longer compact in the infinite-dimensional setting, arbitrary probability measures are considered over the dual space of linear functionals and comparisons are then provided: Proposition 3.2 shows that the definition of the new SW is independent of the choice of measure, while Proposition 3.3 establishes that it is a genuine generalization of the classical SW. Other results demonstrate that the proposed dissimilarities are in fact rigorous metrics. Finally, comparisons between the proposed DSW and WoW are presented, showing weak equivalence.
- Comparisons/Experiments with existing meta-measure dissimilarities are presented across different modalities (datasets, shapes, images).

### Weaknesses
- A dynamical perspective on the proposed methods is not discussed.
- Sketches of the proofs of the theorems, propositions, and corollaries are not included in the main text, which would provide the reader with at least a vague idea of the technicalities and main steps needed.
- Section 5.1 could benefit from additional comparisons, for example by adding a few more columns in Table 1.
- A discussion of possible reasons explaining the clear discrimination achieved by the proposed distance, as opposed to more traditional methods (Section 5.4, Figure 3), would be interesting.

### Questions
- No clousure in span(v) in Thm 3.1 is needed?
- I would appreciate if the authors write the precise definition of $\mathcal P_2(U^*)$.
- Proposition 3.3: I am confused between $\eta$ and the uniform measure $d\mathbb S^{d-1}(\theta)$.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a general sliced Wasserstein framework for arbitrary Banach spaces using quantile functions in the function space $L_2([0,1])$. Building on this, the authors present the sliced quantile WoW and double-sliced WoW distance.  Theoretical properties are presented to support the proposed framework.

### Strengths
- The paper is well-written and easy to follow. The writing flow is clear and presents the relevant work in a nice and elegant way
- The theoretical results are solid, and the proof in the appendix is organized
- Extending sliced OT to general Banach spaces via quantile maps is interesting and potentially impactful

### Weaknesses
- The sliced Wasserstein distance defined in Eq.(4) - there are no experimental results on this distance. How does it compare to other sliced Wasserstein distance empirically?
- SQW’s accuracy gains over competing methods appear modest, aside from FAUST-1000, performance is largely on par with STLB despite comparable runtime. It seems that the proposed method may be beneficial for large-scale datasets, but it is hard to judge based on merely one large dataset that is evaluated.

### Questions
- How sensitive are results to the number of projections of DSW in the experiment of optimal transport dataset distance? For example, using 500 projects in s-OTDD varies from using 10000 projections. What would be the correlation in DSW using 500, 1000, or 5000 projections?
- While it can be considered as concurrent work, it would be beneficial if the authors could compare their method conceptually with the work Busemann Functions in the Wasserstein Space: Existence, Closed-Forms, and Applications to Slicing. What are the key differences?
- How does the proposed method perform in transfer learning tasks?

### Soundness
3

### Presentation
3

### Contribution
3
