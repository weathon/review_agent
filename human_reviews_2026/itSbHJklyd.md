# Quantization bounds for Wasserstein metrics

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
The Wasserstein metric is becoming increasingly important in many machine learning applications such as generative modeling, image retrieval and domain adaptation. Despite its appeal, it is often too costly to compute. This has motivated approximation methods like entropy-regularized optimal transport, downsampling, and subsampling, which trade accuracy for computational efficiency.  In this paper, we consider the challenge of computing efficient approximations to the Wasserstein metric that also serve as strict upper or lower bounds, as these are essential components of branch-and-bound, A$^*$ path finding, and heuristic search techniques in tasks such as trajectory inference, alignment, and clustering.  Focusing on discrete measures on regular grids, our approach involves formulating and exactly solving a Kantorovich problem on a coarse grid using a quantized measure with a tailored cost matrix, followed by an upscaling and correction stage.  This is done either in the primal or dual space to obtain valid upper and lower bounds on the Wasserstein metric of the full-resolution inputs.  We evaluate our methods on the DOTmark optimal transport images benchmark as well as alignment tasks on volumetric dataset of macromolecules, demonstrating a 10×–100× speedup compared to entropy-regularized OT while keeping the approximation error well below 5\% at 2D, and 30\% width bounding regions at 3D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper concerns discrete probability measures on uniform grids. The authors develop efficient algorithms for bounding Wasserstein distance $\mathcal{W}_p$ (e.g., earth mover's distance) between such measures. The search for methods that compute exact upper and lower bounds (rather than just approximations) is motivated by the fact that such bounds enable the use of branch-and-bound, A* path finding and other search techniques for optimization problems involving $\mathcal{W}_p$. The proposed methods work as follows. First, a coarser grid is defined, and measures are downscaled. Second, a special cost matrix for the coarse grid is constructed (possibly depending on original measures), and the downscaled Kantorovich problem is solved exactly. Finally, the solution is upscaled and adjusted to obtain a valid bound. In numerical experiments, bounds based on the Sinkhorn algorithm are used as a baseline. The proposed methods outperform the baseline in these experiments both in terms of speed and accuracy.

### Strengths
1. The paper contributes to a relatively underexplored yet valuable area of research.
2. It presents theorems establishing the validity of the proposed bounds, along with a complexity analysis.
3. The theoretical results are supported by experimental evaluation.
4. The paper is relatively easy to follow.

### Weaknesses
1. Overview of existing bounds is lacking, see questions
2. Quality of the presentation should be improved
3. Numerical comparison to other bounds (apart from Sinkhorn-based) could also be added

### Questions
1. Could you please mention existing bounds in the "Related work" section? For example, Table 1 in the paper *Fast Dataset Search with Earth Mover’s Distance* by W. Yang et al. provides references to several bounds for the earth mover’s distance that would be worth citing and, where applicable, comparing against in the experiments.
2. In Figure 2, "Dual Entropic Reg." bounds are present in the legend but not on the plots. Am I missing something?
3. In formula (7), what do you mean by $|X_i|$? Also, please define SumPool and AvgPool more clearly as a function of 2 arguments (as you use it subsequently). Moreover, please define it for both usual measures and couplings.
4. In formula (8), you use entropy. Please define it explicitly.
5. Lines 259-260: please define a *normalized* kernel.
6. You write both "upscaling" and "up-scaling". I suggest that you settle on the first option, and do same for "downscaling".

Addressing the concerns is important for keeping (or possibly raising) the score.

Typos and minor mistakes:
- Line 25: "at 2D/3D" should be changed to "in 2D/3D".
- Line 135: "section 3.1" should be changed to "Section 3.1".
- Line 175: it seems that the period between "not" and "since" was placed by mistake.
- Line 195: "Proofs for 3.2" should be changed to "Proofs for Lemma 3.2 and Proposition 3.3".
- Lines 260-261: did you mean to place ":" instead of the period before the formula for $\hat{\mathbf{P}}$?
- Line 269: it seems that the period between "factors" and "yielding" was placed by mistake.
- Line 277: "than" should be removed.
- Line 282: please turn the part that starts with "Negligible" into a proper sentence.
- Line 306: "Since by" sounds unnatural.
- Figure 2: change "Upsacling" to "Upscaling" in the legend (4 times).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a family of quantization-based methods to efficiently approximate the p-Wasserstein distance between discrete probability measures defined on regular grids. The main goal is to produce upper and lower bounds on Wasserstein distance that are significantly faster to compute.

### Strengths
1, fast upper/lower bounds for Wasserstein distance on regular grids are provided.
2, Four complementary estimators (weighted-cost UB, min-cost LB, primal upscaling UB, dual upscaling LB) are introduced.

### Weaknesses
1, The proposed method primarily reduces computational cost by down-scaling the sample size through grid coarsening, rather than introducing a fundamentally new optimal-transport formulation or theoretical advance. While the quantization framework offers practical acceleration, it largely depends on existing solvers applied to smaller grids, with modest algorithmic novelty. As such, the broader impact on the optimal-transport or machine-learning community may be limited. The approach trades accuracy for efficiency in a relatively straightforward way and does not substantially deepen theoretical understanding or extend applicability to new data types.

2, The paper reports the average relative bounding region only empirically, with no theoretical justification or convergence analysis.

3, A more useful theory could be bounding the relative error (Upper Bound − Wasserstein distance)/Wasserstein distance, which directly measures overestimation of the true distance. This could be theoretically bounded under smoothness or regularity assumptions on the underlying distributions (e.g., Lipschitz or Holder continuity of the densities).

4, It’s unclear how to extend the bounds to irregular meshes or point clouds.

### Questions
1, Can you provide tightness rates for each bound as a function of the coarse factor, dimension , and simple distribution classes?

2, It would be nice to include a sensitivity analysis of how accuracy depends on grid resolution.

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
This paper introduces a family of grid-based techniques for computing upper and lower bounds on the Wasserstein distance between discrete distributions. The upper bounds are obtained by coarsening the input distributions onto a regular grid and solving an exact optimal transport problem using block-averaged costs or coarse-grid couplings that are then upscaled to the original resolution, with marginal mismatches corrected via weighted total variation. The constructions are designed to ensure that the resulting estimates always overestimate the true transport cost. The lower bounds are derived either from nearest-neighbor mappings, either by coarse OT with blockwise minimal costs or by interpolating duals refined by a c-transform to enforce dual feasibility.

The authors do not establish any approximation ratio or error bound quantifying how close their upper and lower bounds are to the exact value as a function of the grid resolution, dimension, or scaling factor $\kappa$. Consequently, the framework ensures correctness but offers no provable tightness guarantees, relying instead on empirical evidence to demonstrate that the bounds are often reasonably close in practice.

Furthermore, several existing methods already provide provable upper and lower bounds on the Wasserstein distance. For instance, quad-tree or hierarchical constructions yield an $O(\log ⁡n)$-approximation upper bound for $W_1$(see, e.g., Indyk & Thaper 2003; Backurs et al. 2022), while the nearest-neighbor mapping, also known as the Chamfer distance, serves as a simple and widely used lower bound. The paper does not directly compare its grid-based bounds to these established techniques. This omission makes it difficult to assess the theoretical significance of the proposed methods.

### Strengths
The claims appear correct, and the implementations are careful and well-executed. The experiments demonstrate meaningful computational speed-ups, and the writing is clear and well structured.

### Weaknesses
The paper lacks theoretical novelty and does not provide provable approximation guarantees. Existing hierarchical and Chamfer-based methods already achieve provable bounds, yet no direct comparison is made. The contribution is therefore primarily empirical, with limited new theoretical insight.

### Questions
Please address the concerns I have raised in my review.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces efficient methods for computing fast approximations that bound the Wasserstein metric between discrete distributions on a regular grid. Experiments on 2D and 3D data show the proposed approach achieves greater computational efficiency and accuracy than entropic optimal transport-based bounds.

### Strengths
1) This approach outperforms bounds based OT entropic methods, demonstrating less deviation compared to other methods.
2) The authors propose four tools for bounding from weighted-cost upper bound to dual upscaling lower bound
3) This finding allows developing new type of generative models based on OT

### Weaknesses
1) If I am not mistaken, the authors validated their method only for 2D and 3D dimensional setups. Unfortunately, there is no information about scalabity of the method in high-dimensional spaces (for example:100 or 1000, experiments even with Gaussian distributions seem enough).
2) see questions

### Questions
1) Is there any opportunity to continue this method in continuous space?
2) Could you provide more high-dimensional experiments with discrete Gaussian distributions?
3) Your method performs equally with independent transport plan and true transport plan, doesn't it?
4) How does the behaviour of convergence change depending on number of samples?

### Soundness
3

### Presentation
2

### Contribution
2
