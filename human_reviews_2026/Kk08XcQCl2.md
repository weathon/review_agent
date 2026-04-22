# The data manifold under the microscope

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
A significant gap exists between theory and practice in deep learning. One example is given by generalization and approximation error bounds, which are often derived for overly simplified models or yield guarantees that are too loose to be informative. Many such bounds rely on the manifold hypothesis and depend on geometric regularity properties, including intrinsic dimension, curvature, and reach of the data manifold or target functions. To make progress on improving these bounds, one needs detailed insight into data manifold geometry and suitable benchmarks on simple datasets. However, existing datasets and analysis tools typically fall into two extremes: analytically defined manifolds with precisely known geometry but limited realism, or real-world datasets where bounds are assessed only through downstream performance and geometric properties can be estimated only coarsely and with hard-to-quantify error.

To address this lack of simple yet realistic datasets and accompanying geometric tools, we introduce a benchmarking framework for studying data geometry. We repurpose and extend the dSprites and COIL-20 datasets with additional transformation dimensions and finer sampling resolution. This enables accurate finite-difference estimates of geometric quantities such as curvature, reach, and volume, yielding a flexible benchmark for evaluating manifold learning methods. As illustrative applications, we assess two established manifold learning bounds by Genovese et al. and Fefferman et al., and analyze how manifold geometry evolves across network layers in $\beta$-VAEs. Our results highlight both the limitations of existing bounds and the value of controlled benchmarks for guiding future theoretical developments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper purports to bridge the gap between theory and practice in deep learning for data satisfying the manifold hypothesis. The paper is based on empirical studies that aim to test whether theoretical upper (and lower) bounds for manifold approximation are sharp, i.e., overly not overly pessimistic (or optimistic).

The method is based on introducing a uniform sampling grid for simple, low-dimensional manifolds which are homeomorphic to [0,1]^r \times (S^1)^s, r+s=d. They then use existing manifold approximations. They consider a couple data sets and empirically observe results similar to the upper bound in one case and and the lower bound in the other.

Overall, this paper falls short of its goal of bridging theory and practice. There is little new here. Furthermore, the method relies on finite difference schemes which in turn rely on uniform sampling which limits the applicability of this method to real data. The "axis aligned" assumption is also quite limiting. Additionally, for this paper to make a more sizeable contribution, it should either a) introduce new theory or b) provide insights into model performance under complex manifolds, preferably under less than idealized sampling conditions.

Minor

line 43, extra space after hypothesis
equation 3.14 \mathcal{Q} does not appear to be defined

### Strengths
The background on differential geometry is mostly well written and there is some decent discussion of relevant literature

### Weaknesses
See above

### Questions
Do you have any insights as to when the upper or lower bounds will be tighter?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates how to compute key geometric properties of a data manifold (volume element, scalar curvature, and reach) directly from sampled data points. The authors focus on modeling the manifold as the product of a 1-manifold and a low dimensional ball. They utilize two manifold fitting approaches (MMLS and β-VAE) to empirically test the theoretical bounds proposed by Genovese et al. and Fefferman et al.

### Strengths
The paper addresses a critical challenge in learning theory: the difficulty of empirically validating theoretical bounds based on manifold assumptions due to the unknown properties of real-world data manifolds. As a first attempt to bridge this gap, the authors propose a straightforward experimental framework designed to test such theoretical bounds. This initiative provides a valuable starting point for experimentally probing the applicability and limitations of manifold-based theoretical results.

### Weaknesses
1. Limited novelty. While this appears to be the first framework for empirically testing manifold bounds, it primarily integrates well-established techniques, ranging from manifold fitting to computing geometric metrics. The potential for reproducibility and generalization to different types of manifolds is limited, which further restricts the novelty of the approach.

2. Low generalizability: The proposed method is confined to manifolds that are the product of a 1-manifold and a low-dimensional ball. This is a strong assumption that does not extend well to the diversity encountered in real-world datasets.

3. Relevant prior work not discussed. he paper omits discussion of several important lines of related research, such as bounds on manifold reconstruction [1,2,3] and bounds built on similar manifold assumptions[4].

[1] Partha Niyogi, Stephen Smale, and Shmuel Weinberger. Finding the homology of submanifolds with high confidence from random samples. Discrete & Computational Geometry, 2008.

[2] Hariharan Narayanan and Sanjoy Mitter. Sample complexity of testing the manifold hypothesis. In Advances in Neural Information Processing Systems, 2010.

[3] Stefan C. Schonsheck, Jie Chen, and Rongjie Lai. Chartauto-encoders for manifoldstructured data. CoRR, abs/1912.10094, 2019.

[4] Yao, J., Goswami, M., & Chen, C. (2024). A theoretical study of neural network expressive power via manifold topology. arXiv preprint arXiv:2410.16542.

### Questions
Clarification needed regarding bound evaluation. The paper does not explicitly present the two evaluated bounds (expand the computation of constants), neither in the main text nor the appendix. since the role and value of the constant C are not well explained, it is difficult to interpret the computational details. It is unclear how the authors compute Genovese's bound and plot it alongside Fefferman's bound. Are these bounds presented on the same scale for direct comparison?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper brings forward a new mechanism to evaluate theoretical bounds in a rich empirical setup. They first present two previous theoretical results on the sample complexity of learning data distributions with support on low-dimensional manifolds embedded in higher-dimensional Euclidean spaces. They introduce a set of finite-difference methods to estimate three measures: curvature, reach and volume. They finally present quantitative results of their verification pipeline which lets them comment on which of the two theoretical bounds are tighter.

### Strengths
The paper presents a novel idea and does a good job developing a framework to verify theoretical results using empirical methods on complex manifolds. Their empirical validation of the two bounds is impressive and novel, as far as I can tell.

### Weaknesses
One of the main weaknesses of the paper is that it is missing some key details. I am unclear as to how MMLS works or how $\beta$-VAE fit syntehtic data from the content of the paper. How does the sampling technique described in lines 266-272 ensure that you "obtain more uniform subsets"? What is dist in "To compare with theoretical rates, we regress log(dist) against log(n)"? (Line 359). This is a key detail because it tells us how your comparison to theoretical rates is valid. Please point me to the regression loss curve. Your second application of Manifold analysis (line 414) is not listed as a key result and I am not certain how this contributes to your goal of bridging the gap between theory and practice. I am also not able to grasp your conclusion on why $\beta$-VAE struggles to converge: "latent manifold emphasizes semantic clustering at the cost of geometric distortion". These are some issues that could be addressed to help me understand the manuscript better. Could you also comment on the hyper-parameters you searched for $\beta$-VAE?

Could you show independence of the Hausdorff Distance wrt the ambient dimension as claimed by the two bounds you have presented? This discussion was lost even though this is mentioned in your review of the theoretical results. I would also be curious in how the differences in $r,s$ manifest in the Hausdorff distance.

Another issue I found was that comparison to existing methods is a bit lacking. The finite-difference estimators are a major contribution of yours, I believe that the reach estimate follows from Aamari et al 2019 and similarly there has been past work on curvature estimates in the field of complex networks [1]. I am not discouraging the authors from using their suite to make these measurements but there ought to be a comparison to existing techniques.

I encourage the authors to rectify these issues because I find their approach of connecting theory to practice incredibly promising. I believe some key details are missing and the broader community would benefit from the authors presenting them in the main body of the paper.

----------------------------------------

References:

[1] Comparative analysis of two discretizations of Ricci curvature for complex networks, Areejit Samal, R. P. Sreejith, Jiao Gu, Shiping Liu, Emil Saucan & Jürgen Jost

### Questions
I have the following minor issues and questions:

Line 130: In definition of G what are $n_1, ..., n_d$? I would introduce these here if these are some form of scaling parameters.

Line 178: You might be overloading $d$ here, using it both for distance and dimensionality. Furthermore, how is this distance defined?

Appendix A.1 and A.2 seem to be duplicates of the geometry background section (Section 3) in the main paper. I would expect the authors to de-duplicate it for a concise presentation of their work.

Line 180: what is "scalar curvature integral"? I dont see this defined.

Line 163-167: I believe these lines are using the Einstein notation, I think the authors should at least mention this in their work. If it has been specified and I missed it please point me to the section.

Thoerem 1: where is the idea of fiber introduced in the paper? I understand that the some of the intended audience might know this but I believe it might be helpful for the broader audience for the authors to define it in their work or explain it informally.

Theorem 1: What is $\mathcal Q$ here? Are Genovese et al reasoning about the minimax risk over a family of distribution?

Line 347: sub-section 3 -> section 3

Line 304-305: "All three estimators have been tested on families of manifolds with known closed-form quantities" could you please point to these plots and description of the experiments?

Figures 1 and 2 seem to be very similar, I am not sure what is the value addition from adding this schematic in Figure 2.

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
3

### Summary
This paper presents a systematic empirical–geometric framework for studying the data manifold hypothesis under controlled settings. The authors build dense-grid versions of common benchmark datasets (extended dSprites, extended COIL-20) that allow near-continuous sampling of transformation parameters, enabling finite-difference (FD) estimation of geometric quantities such as volume, curvature, and reach. These ground-truth-like quantities are then used to (i) benchmark classical manifold-fitting bounds (Fefferman, Genovese, etc.) and (ii) analyze layer-wise geometry evolution in β-VAEs.

Overall, the paper is well executed and the results are clear and interpretable. The main contribution lies in the framework—combining dense-grid datasets with FD-based geometric estimation and empirical validation of theoretical bounds. While each element builds on existing ideas, the integration into a reproducible benchmark pipeline is valuable.

This is a well-written and methodologically sound paper whose main contribution is an integrated, reproducible pipeline for empirically probing data-manifold geometry and testing theoretical bounds. While the individual ingredients are not new, the overall framework fills a gap between theory and empirical practice. Clarifying the treatment of theoretical constants, explicitly positioning the work against prior reach/curvature studies, and open-sourcing the pipeline would strengthen both its novelty and community value.

### Strengths
* Solid motivation and clear presentation: The paper provides a coherent workflow from dataset construction to theoretical benchmarking. Figures are easy to interpret, and the connection between geometric theory and empirical practice is well articulated.
* Technically sound methodology: The finite-difference approach and reach estimation procedures are standard but implemented carefully, with convincing O(h^2) convergence plots.
* Bridging theory and experiment: Using real image manifolds to test reach/volume-based bounds is a refreshing way to connect classical geometry with representation learning.
* Potential for community impact: If released as a package, the proposed framework could serve as a valuable benchmark for manifold estimation and representation-geometry research.

### Weaknesses
* Limited novelty beyond integration:
The key contribution appears to be the framework rather than a fundamentally new theoretical or algorithmic idea. The paper would benefit from a more explicit discussion of how it extends beyond existing works on curvature or reach estimation (e.g., Aamari et al. 2019), theoretical bounds (Fefferman et al.), and prior controlled datasets (dSprites, COIL-20).
* Choice of constants in theoretical comparisons:
In Figures 5–7, empirical curves are plotted alongside theoretical upper/lower bounds. However, it is unclear how the constants in those bounds were chosen or fitted. Were they analytically derived, or manually scaled for visualization? Explicitly stating this would clarify the strength of the comparison.
* Open-source and reproducibility:
Since the main value lies in the pipeline, the paper would be significantly strengthened by releasing the dataset generators, FD estimators, and plotting utilities as an open-source package.
* Interpretation of mixed bound-matching behavior:
In some regimes, the empirical curves align with the theoretical upper bound, while in others they hug the lower bound. It would be interesting to discuss why this happens—e.g., differences in curvature concentration, sampling density, or estimator bias.
* Scalability and scope:
The FD approach depends on dense grids, which may not scale beyond low-dimensional transformations. A short discussion of this limitation would help readers understand the generality of the framework.

### Questions
1. How were the constants in the theoretical bounds chosen when overlaying them on the empirical results?
2. Do you plan to release the dataset-generation and geometry-estimation code as an open-source package?
3. Can you provide intuition for why the empirical scaling sometimes matches the upper bound and sometimes the lower bound?
4. How practical is the dense-grid FD approach for manifolds beyond two or three intrinsic dimensions?
5. How does your framework compare to prior curvature/reach-estimation and manifold-fitting pipelines in terms of computational efficiency and required sample density?

### Soundness
2

### Presentation
2

### Contribution
2
