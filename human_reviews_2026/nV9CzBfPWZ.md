# QDOT: An Efficient Quantile-weighted Distance Metric for Geometric Comparison via Optimal Transport

- Decision: Reject
- Scores: 6, 2, 6, 4, 6

## Abstract
Measuring the discrepancy between data distributions in heterogeneous metric spaces is a fundamental challenge. 
Existing methods, typically based on geometric structures, address this by embedding distributions into a shared space. 
However, these approaches face fundamental limitations, including the loss of geometric information, computationally intractable representations, and inability to preserve essential structural features. 
In this work, we introduce the Quantile-weighted Distance Optimal Transport (QDOT), a novel and efficient metric for geometric comparison. 
QDOT constructs a family of isometry-invariant distance representations by leveraging distance quantiles as structural weights in Euclidean space, thereby preserving essential geometric characteristics and enabling optimal transport coupling within a common space. 
We prove that, under mild conditions, QDOT is a well-defined metric with a convergence rate no slower than the classical Wasserstein distance. Moreover, we present an integral version that computes the loss in complexity of $\mathcal{O}(n\log n)$. 
Extensive experiments demonstrate that our methods achieves strong performance across diverse applications, including cross-space comparison, transfer learning, and molecule generation, while also achieving state-of-the-art results on several key metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper “QDOT: An Efficient Quantile-Weighted Distance Metric for Geometric Comparison via Optimal Transport” proposes a new geometric metric called QDOT (Quantile-weighted Distance Optimal Transport) to efficiently compare data distributions across different metric spaces. Traditional methods like the Wasserstein and Gromov-Wasserstein distances either lose geometric information or are computationally expensive. QDOT overcomes these issues by using quantile-weighted distance means (QDMs) as isometry-invariant anchors and constructing a quantile-based representation that preserves geometric structure. The method achieves theoretical guarantees, it is a valid metric on isometry classes, has convergence rates comparable to the Wasserstein distance, and supports rotation and translation invariance. The authors also propose an integral variant (IQDOT) with quasi-linear time complexity $\mathcal{O}(n\log n). Experiments on tasks like cross-space point cloud matching, transfer learning, and molecule generation show that QDOT offers empirical performance and computational efficiency, outperforming existing geometric comparison methods while maintaining theoretical soundness.

### Strengths
1. QDOT is rigorously defined as a proper metric on isometry classes, with proofs ensuring identity, symmetry, and triangle inequality properties.

2. By using quantile-weighted anchors, it naturally handles transformations like rotation, translation, and reflection, crucial for geometric data.

3. The integral version (IQDOT) achieves quasi-linear complexity $\mathcal{O}(n\log n)$, significantly faster than Gromov–Wasserstein and similar metrics.

4. Demonstrated good performance across diverse tasks, cross-space alignment, transfer learning, and molecular generation, showing good generalization ability.

5. Experiments and comparisons with baselines (GW, EGW, SGW, etc.) confirm both efficiency and accuracy advantages.

6. Integrating QDOT as a loss function improves deep learning models (e.g., molecule generation), showing its usefulness beyond theoretical contexts.

### Weaknesses
1. The quantile-weighted representation and multiple layers of transformation make the method conceptually and computationally more intricate to implement.

2.  Performance may depend on the choice of quantile levels and the Gaussian kernel bandwidth, which require tuning.

3. The sample complexity is bad i.e., $n^{-1/d}$ which scales exponentially with dimension.

4. The quantile-weighted features, though powerful, may obscure intuitive geometric interpretation compared to classical OT or GW frameworks.

### Questions
1. Does Integral-QDOT has good sample complexity as sliced optimal transport [1]?

2. How does QDOT compare theoretically to Sliced Gromov–Wasserstein (SGW) beyond computational complexity?

[1] An Introduction to Sliced Optimal Transport, Nguyen.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose the Quantile-Weighted Distance Metric for Geometric Comparison via Optimal Transport (QDOT) and Integral QDOT to address the structural alignment problem. This problem is closely related to the Gromov–Wasserstein problem, as both aim to match the geometric structures of two distributions that may differ in dimensionality. The proposed QDOT method first computes $k$ quantile means for a source and a target discrete distribution, then constructs cost matrices between the samples and the quantile means to finally use them to compute their Wasserstein distance and corresponding coupling matrix. To mitigate the computational limitations of QDOT during the Wasserstein distance computation, the authors introduce Integral QDOT, which replaces the full Wasserstein distance with the average of $d$ one-dimensional Wasserstein distances. Both methods are evaluated on two shape matching tasks (camell-galop and ModelNet40-ShapeNet) as well as a molecule generation task. The setups are evaluated using different metrics such as TMSE and accuracy.

### Strengths
- The structure and flow of the paper is good, this allows one to easily understand the proposed methods.
- The idea behind aligning quantiles instead of samples is novel and it seems it was not explored in the literature.

### Weaknesses
My main concerns about the paper are summarized below:
- The use of one-dimensional Wasserstein distance in integral QDOT is not well supported. As a result, Integral QDOT may fail to fully reflect the structural alignment between distributions as it does account inter dimensional relationship, this can potentially oversimplify complex dependencies. This would make the method not suitable for several, if not most, of the applications.
- There is a lack of discrete methods tested in the experimental section, methods such as (Alvarez, 2018), (Sebbouh, 2024) and (Klein, 2023) are also related and should be included to have a better understanding about the method’s performance and how it is positioned with respect to state-of-the-art approaches.
- In Subsection 4.1, it is unclear for me why TMSE and IR were chosen as metrics as they require a known transport map $\mathcal{T}$ (as pointed out in Appendix D.2). The details about the construction of this function are not explored in the paper and I think it is crucial to understand to which extent the reported metrics represent the alignment capabilities of the methods. More precisely, these metrics can give misleading results if $\mathcal{T}$ only ensures that points are close in space but not truly matched. In that case, even a poor coupling $\Pi$ could produce low TMSE or high IR scores without reflecting real alignment accuracy. 
- In addition to the previous point, no ground truth alignment or true baseline is provided; therefore it is difficult to know if the values in Table 1 actually represent a good alignment or not. I am also not sure about how the 2D projections in this experiment were obtained and why they should be used to assess the alignment abilities of the methods.
- In addition to the previous point, no ground truth alignment or true baseline is provided; therefore it is difficult to know if the values in Table 1 actually represent a good alignment or not. I am also not sure about how the 2D projections in this experiment were obtained and why they should be used to assess the alignment abilities of the methods.
- I believe the plot with losses (Figure 4) is not informative and I think it would be better to replace it with a figure containing the visual results of the obtained alignments for this experiment.
- In subsection 4.3, the IQDOT method’s performance is not consistent. This is mainly seen in the experiments denoted as Mo$\rightarrow$Sh. As the rest of the experiments (4.1 and 4.4) do not include results for IQDOT, it is difficult to ensure that this approach is suitable for them. I raise this concern as the computation of one-dimensional Wasserstein distances to obtain the coupling may restrict its applicability (as previously mentioned).
- I have some concerns and observations regarding the experimental setup in subsection 4.4. From Table 3, it is difficult to interpret the obtained results as the metrics and the problem to solve are not properly introduced. Therefore, it is hard to say whether the obtained results represent a good alignment or not. It is important to clearly explain the metrics as they may be meaningful for different applications, but irrelevant in the case of structural alignment. Additionally, as the paper is oriented to a community mostly familiar with machine learning and artificial intelligence, I believe it is necessary to introduce the problem with more details as well as the computed metrics.
- I think the experimental section should also report the distortion and see if it correlates with the value of QDOT. I also suggest and I think it would also be interesting to report it after step 9 and also after step 10 in Algorithm 1, this would help to understand how impactful is the use of quantile alignment while finding the optimal coupling.
- I think the paper would benefit from including images of the obtained aligned point clouds in the case of experiments in 4.1, 4.2 and Appendix D.1. In this last case, only the noised distribution is shown and I think it is important to also add the source, target and the predicted points.
- The paper lacks experiments on uncorrelated setups, or at least, this point is never specified in the paper. Testing the methods on source and target distributions with unknown optimal pairs is an insightful way to show the performance and limitations of a GW-like solver that aims to capture and keep the geometric structure of the distributions. The concept of (un)correlatedness has been extensively explored in (Aramayo, 2025).

**Minor comments**
- In Table 1 it is unclear for me why the Entropic Gromov-Wasserstein method is slower than QDOT if both use the same Sinkhorn solver.
- No reference for Algorithm 2 in the main text.
- The notation for $\phi$ is currently confusing. In certain instances, it appears as $\phi_{\sharp\mu_X}^X$, but based on the definition in Equation 6, it is unclear how the pushforward operation is being applied.
- I believe the current notation for dimensions and number of samples is confusing and since it is used everywhere in the paper to understand the proposed ideas, I would suggest modifying it to improve the readability of the paper. One example about the importance of this appears in Definition 4 where $q$ is used to denote the quantiles while in 3.3, $q$ represents the dimensionality of the target distribution.
- From Algorithm 1, it is unclear how the optimal coupling $\Pi$ is computed, as the algorithm appears to only output the QDOT distance/metric. However, in Appendix D.2 (Lines 1234–1235), it is stated that the algorithm computes the coupling. Therefore, it is necessary to explicitly specify the coupling as an output in the algorithm to avoid ambiguity.

**To summarize,** I believe the paper has some interesting theoretical contributions regarding the novelty of the proposed QDOT method; however, the practical aspects of the paper require major revisions. More precisely, the reported metrics seem to be not the most adequate to show the true performance of the solver and the authors should also report more insightful metrics such as the GW distortion, SInkhorn divergence or MMD. On the other hand, the proposed IQDOT method seems to be weaker as it uses an average of one-dimensional Wasserstein distances, which does not consider inter dimensional relationships and it may not be useful for more complex tasks. For these reasons, I recommend the paper to be rejected.

(Alvarez, 2018) David Alvarez-Melis and Tommi Jaakkola. Gromov-wasserstein alignment of word embedding spaces. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1881–1890, 2018.

(Sebbouh, 2024) Othmane Sebbouh, Marco Cuturi, and Gabriel Peyre. Structured transforms across spaces with ´ cost-regularized optimal transport. In International Conference on Artificial Intelligence and Statistics, pp. 586–594. PMLR, 2024.

(Klein, 2023) Dominik Klein, Theo Uscidda, Fabian Theis, and Marco Cuturi. Generative entropic neural optimal ´ transport to map within and across spaces. arXiv preprint arXiv:2310.09254, 2023.

(Aramayo, 2025) Aramayo, X., Nekrashevich, M., Mokrov, P., Burnaev, E., & Korotin, A. (2025). Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem.

### Questions
- In subsection 4.3, what are the dimensionalities of the source and the target distributions?
- How are the 2D projections obtained for the experiments in Subsection 4.1?
- Why are experiments with IQDOT not present in Table 1 and Table 3?
- In Algorithm 1, the norm $\|x\|_2$ represents the Euclidean norm? If yes, I would suggest keeping the same notation as in Equation 4, as it is confusing in the current state.
- Was the method tested on $q\ge d$? In other words, was it tested for low to high dimensional setups?
- Does the QDOT metric align with other metrics such as distortion from the GW problem?
- What loss is reported in Figure 4?

### Soundness
1

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
4

### Summary
This paper introduces QDOT (Quantile-weighted Distance Optimal Transport), a new metric for geometric comparison of data distributions, particularly across heterogeneous metric spaces. The method constructs isometry-invariant representations by leveraging quantile-weighted distance means (QDMs), then computes Wasserstein distances between transformed representations. This approach preserves intrinsic geometric information while achieving favorable theoretical and computational properties. The authors prove that QDOT and its integral variant (IQDOT) are valid metrics on the space of isometry classes of metric measure spaces, with empirical convergence rates no worse than Wasserstein distance, and O(n log n) computational complexity. Experiments on cross-space alignment, transfer learning, and molecular generation demonstrate strong performance and practical relevance.

### Strengths
1). Strong conceptual novelty and theoretical grounding. The introduction of quantile-weighted distances as canonical anchors represents an elegant and original solution to the long-standing trade-off between isometry invariance and information preservation in geometric metrics. The theoretical results (Theorems 1–3) are rigorous, clearly presented, and address fundamental properties such as identity, symmetry, triangle inequality, and convergence.

2). Efficient and scalable computation. The proposed IQDOT variant achieves quasi-linear complexity (O(n log n)), which is a substantial improvement over the cubic or quadratic complexity of Gromov–Wasserstein and related metrics. The empirical runtime results confirm this advantage convincingly.

3). Comprehensive and meaningful experiments. In cross-space alignment (camel-gallop dataset), QDOT matches or surpasses GW and EGW while being over 30× faster.

### Weaknesses
1). While the quantile-based weighting is conceptually appealing, additional intuition or ablation on the choice of quantile levels and bandwidth σ would strengthen the understanding of sensitivity.

2). The theoretical section could briefly relate QDOT to classical results in empirical OT convergence (e.g., emphasizing where quantile weighting introduces regularization).

3). Molecular generation experiments could include visual examples to illustrate the qualitative improvements achieved with QDOT.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes QDOT - a Quantile‑weighted Distance Optimal Transport metric for comparing probability distributions defined in (potentially different) Euclidean spaces. The core idea is to build isometry‑invariant anchors called Quantile‑weighted Distance Means (QDMs) by weighting points with a Gaussian centered at quantiles of the norm distribution; distances from points to these anchors produce a representation (QDMD) on which a standard Wasserstein distance is computed. An “integral” variant (IQDOT) aggregates 1‑D Wasserstein distances across quantiles to achieve quasi‑linear time. The paper proves that (under dimensionality conditions on the anchor set) both QDOT and IQDOT are metrics on isometry classes and that empirical convergence is at least as fast as the classical Wasserstein rate $n^{-\frac{1}{d}}$. Experiments cover cross‑space point‑cloud alignment, point‑cloud transfer learning and molecule generation where QDOT is used as a loss in diffusion models. Experimental results show comparable or improved alignment quality over Gromov Wassesrein and Entropic Gromov Wassesrstin with orders of magnitude speedups, quasi‑linear scaling for IQDOT, strong transfer accuracy and improved stability/validity for molecular generation.

### Strengths
The primary methodology of the paper is based on defining anchors by norm‑quantiles and building representations from distances to those anchors, which is conceptually simple and gives immediate invariance to orthogonal transforms. The metric properties together with rotational invariance of QDOT and IQDOT, under centering assumptions, make sense to me, together with the fact that the distances retain the Wasserstein convergence rates .The computational complexity is perhaps the most practically relevant and useful aspect of the proposed metrics.

### Weaknesses
1. The metric property of QDOT relies on the dimensionaility condition presented in Theorem 1 (Lines 223-224), and the a similar scenario is true for IQDOT based on Theorem 3. For any centered, radially (or centrally) symmetric distribution (for e.g., an isotropic Gaussian distribution ) one has that $\mathbb{E}(f(\||X\||_{2})X)$, and hence all QDMs, as defined in Equation 5 (Lines 195-195) will trivially reduce to the same value 0. In that case, Equation 6 shows that $\phi^{X}(x,q)=\||x\||_2$ for any $q \in (0,1)$, so QDOT compares only the radial norm distributions and can identify non‑isometric spaces with identical norm distributions, violating identity of indiscernibles without the assumption that all distributions are first centered around the origin. So the centering assumption around 0 is very crucial.

2. Corollary 1 claims location invariance (Lines 252-253), yet the representation explicitly includes $\varphi_{0}(x) = \||x\||_{2}$, which by itself is not translation‑invariant. The paper suggests centering as "recommended” pre‑processing (Appendix C.1 Lines 1105-1106), but then the corollary does not make this centering explicit. The paper should either: (i) prove translation invariance with $\varphi_0$ and without pre‑centering, or (ii) make centering a required step in the definition/algorithms and restate invariance accordingly.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QDOT (Quantile-weighted Distance Optimal Transport), a novel metric for comparing probability distributions across heterogeneous metric spaces. The key idea is to construct isometry-invariant anchor points (QDMs) using quantile-weighted distance means, then represent each distribution by distances to these anchors, enabling comparison via standard optimal transport. The authors prove QDOT satisfies metric properties on isometry classes and achieves convergence rates comparable to Wasserstein distance. An efficient integral variant (IQDOT) reduces complexity to O(n log n). Experiments demonstrate strong performance on cross-space alignment, transfer learning, and molecular generation tasks.

### Strengths
1. The paper provides rigorous proofs that QDOT is a well-defined metric on isometry classes (Theorem 1) and achieves favorable convergence rates (Theorem 2). The integral variant achieves O(n log n) complexity, making it significantly more practical than GW-based methods while maintaining theoretical guarantees, a rare combination in this space.

2. The quantile-weighted distance mean construction is elegant and well-motivated by trilateration principles. Unlike prior work, QDOT explicitly constructs the shared representation space rather than relying on implicit embeddings, addressing a fundamental limitation of existing geometric comparison methods.

3. The experiments span multiple domains (point cloud alignment, transfer learning, molecule generation) and consistently demonstrate both efficiency gains and competitive or superior performance. The molecular generation results are particularly impressive, achieving new state-of-the-art stability metrics while accelerating training convergence.

### Weaknesses
1. The requirement that QDMs span the full dimension is crucial for the metric property but receives insufficient treatment. When might this fail in practice? How sensitive is the method to near-degenerate cases? The paper claims this is "generally satisfied" but provides no empirical analysis or failure cases.

2.  The choice of quantile levels (the vector q) and bandwidth parameter σ appear critical but are not thoroughly investigated. How should practitioners select these? The experiments use different settings across tasks without clear justification, and there's no sensitivity analysis to understand robustness.

3. In the transfer learning experiments, only SGW is compared due to computational constraints, but recent efficient baselines like sliced methods are missing. The molecule generation ablation with MSE-reweighting is helpful but doesn't fully isolate QDOT's geometric properties from other potential effects.

4. While both are proven to be metrics, the relationship between them is unclear. Does IQDOT approximate QDOT, or are they fundamentally different? The paper would benefit from bounds relating the two or empirical analysis of when they agree/disagree.

5. The transition from intuition (trilateration) to formal definition could be smoother. Section 3.1's centering assumption is introduced casually but affects the entire framework. Some notation is inconsistent (e.g., switching between X and X for the space vs. sample matrix).

### Questions
1. Can you provide guidance on hyperparameter selection? Specifically, how should the number of quantiles k, their positions q, and bandwidth σ be chosen for new applications? Are there theoretical principles or heuristics that practitioners can follow?

2. How does performance degrade when the dimensionality condition is violated? It would be valuable to see experiments with distributions where the QDMs don't span the full space, to understand the practical robustness of the method and whether approximate metric properties still hold.

### Soundness
3

### Presentation
2

### Contribution
3
