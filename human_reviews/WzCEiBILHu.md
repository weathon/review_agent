# Topological Schrödinger Bridge Matching

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
Given two boundary distributions, the \emph{Schrödinger Bridge} (SB) problem seeks the “most likely” random evolution between them with respect to a reference process. It has revealed rich connections to recent machine learning methods for generative modeling and distribution matching. While these methods perform well in Euclidean domains, they are not directly applicable to topological domains such as graphs and simplicial complexes, which are crucial for data defined over network entities, such as node signals and edge flows. In this work, we propose the \emph{Topological Schrödinger Bridge problem} ($\mathcal{T}$SBP) for matching signal distributions on a topological domain. We set the reference process to follow some linear tractable \emph{topology-aware} stochastic dynamics such as topological heat diffusion. For the case of Gaussian boundary distributions, we derive a \emph{closed-form} topological SB ($\mathcal{T}$SB) in terms of its time-marginal and stochastic differential. In the general case, leveraging the well-known result, we show that the optimal process follows the forward-backward topological dynamics governed by some unknowns. Building on these results, we develop $\mathcal{T}$SB-based models for matching topological signals by parameterizing the unknowns in the optimal process as \emph{(topological) neural networks} and learning them through \emph{likelihood training}. We validate the theoretical results and demonstrate the practical applications of $\mathcal{T}$SB-based models on both synthetic and real-world networks, emphasizing the role of topology. Additionally, we discuss the connections of $\mathcal{T}$SB-based models to other emerging models, and outline future directions for topological signal matching.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the topological Schrödinger bridge problem (TSBP), extending Schrödinger bridge (SB) methods to non-Euclidean domains, specifically graphs and simplicial complexes. Traditional SB approaches face limitations on topological domains. TSBP is proposed to address these by defining a topologically aware reference process based on topological stochastic differential equations (TSDEs). For cases with Gaussian boundary conditions, the authors derive closed-form solutions, then construct TSB-based models for generative modeling and signal matching.

### Strengths
- The derivation of closed-form solutions for Gaussian boundaries is rigorous, and the detailed handling of topological SDEs adds significant depth to the method.
- The paper is well-structured, guiding readers through the theoretical framework, derivations, and experimental applications, though some dense mathematical notation may hinder readability.
- By addressing topological signal matching, this work is relevant to fields where data resides on complex networks, such as biological and social networks.

### Weaknesses
- Replicability is a major limitation. The lack of code or implementation details limits reproducibility, as replicating the results would require substantial independent effort, particularly in replicating TSDE parameterization and model training. Key training processes (e.g., likelihood optimization for neural networks on topological data) and parameter settings are not transparent, which might impede others from achieving comparable results.
- While synthetic and specific real-world network tasks are covered, the experimental scope could be broadened. Testing across a broader range of real-world network types (e.g., biological, transportation, or communication networks) would better demonstrate the broad applicability of TSBP and strengthen the claims of generalizability. If extending the experimental coverage is impractical, the paper should provide code to enable other researchers to explore the applicability and limitations of TSBP across different types of network-structured data.
- The primary focus on Gaussian boundary distributions restricts practical generalizability, as many real-world networked data scenarios involve non-Gaussian or mixed-distribution boundary conditions.
-  The method’s computational demands are high, particularly in large-scale networks. Although some efficiency measures are discussed, handling matrix exponentials in large or complex network structures remains challenging.

### Questions
- Can the authors provide further insight into adapting TSBP for boundary conditions that are not Gaussian?
- Could the authors expand on strategies to manage computational complexity for larger networks, especially in real-world scenarios?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work aims to generalize diffusion Schrodinger Bridge generative models from Euclidean domains to general topological spaces, in order to perform topological signal matching and generation. The new method is called topological Schrodinger Bridge Matching (TSBM). The authors develop a topological version of the Schrodinger bridge using a stochastic differential equation driven by a topological convolution. In the case of graphs, this can be driven by the graph heat equation with graph Laplacian. In the case where both distributions to be matched are Gaussian distributions, they give closed form solutions that generalize previous work on Schrodinger bridges. Using this framework, the authors developed a TSB-based method for topological signal generation and matching, based on parameterizing the policy with a topological neural network. The authors also provided the results of some numerical experiments on ocean current matching and modeling seismic events, along with some synthetic experiments.

A couple of typos I noticed: " a diffusion ndoe GP", "Vairous topological", "the spacial case", "we showed that the optimal process satisfy a pair"

### Strengths
Deep learning on topological spaces, like graphs, is an important area of research. The paper provides an original and mathematically well-grounded method for topological signal matching and generation. The mathematics is very well-developed and clearly laid out, and the paper is very well-written.

### Weaknesses
The numerical experiments are rather limited. I would have liked to see more large scale numerical experiments to validate the method. It is not clear to me if the methods scale up to large data sets well. Some of the future work problems (i.e., "On model training") indicate that it might not.

### Questions
On a graph, there are various normalizations of the graph Laplacian. The unnormalized (combinatorial), random walk and symmetric normalized are the three most common, but there are a wide range of others depending on a continuous family of parameters. Could the authors please clarify which graph Laplacian they are using? There is a substantial difference in terms of the diffusion process generated on the graph if one uses the unnormalized versus the random walk Laplacian (with the latter being the natural choice for diffusion on graphs). I think the paper would benefit from a discussion about the type of graph Laplacian that is used, and possibly numerical experiments with different graph Laplacian normalizations.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes to develop a method for solving the Schrödinger Bridge problem when there is an inherent topological structure to the underlying matching space.  Here, topological structure refers to discrete geometric structure, such as graphs and their higher order generalizations.  Theoretical guarantees are given and numerical demonstrations are given on both simulated and real data.

### Strengths
Overall, I found this paper to be a strong contribution.  I would have probably given/would have liked to give higher ratings but I am not very confident in some of the technical areas that this paper covers.

The experiments were good. They were carried out on a range of sizes where smaller ones were able to illustrate the impact of the proposed method but also showcase the computational ability of the method on larger networks.

Good overview on the limitations of the work, including future directions of research that can be carried to overcome them.

### Weaknesses
I think that some of the experiments could be more illustrative in why a topological approach is needed and what is missing if the underlying topology is ignored.

Also, other details on the computational aspects appear to be missing, such as runtime, complexity, convergence analyses, etc.

### Questions
- I am not familiar at all with the work of Deng et al. (2024) cited in in the introduction, but point clouds were cited as a Euclidean space.  Point clouds are in fact quite a challenging and general data structure (as a finite metric space) and are a key setting in which topological procedures are studied (especially in topological data analysis, which is subsequently cited in the introduction asa. basis for topological machine learning.  Wouldn't the approaches proposed in this work also apply to point clouds?
- Would it be possible to provide a performance comparison of the TSB to SB methods on point clouds?
- Would it be possible to have an example experiment that clearly shows what goes really wrong if the underlying topological structure is ignored and only Euclidean approaches are used?
- Check the paper for typos (e.g., "benefitial" on page 8).

### Soundness
3

### Presentation
3

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
The manuscript is on the Schrödinger bridge problem for special topologies like graphs. The authors define the reference process akin to the heat diffusion on said topology. For specific boundary distributions, they present closed for solutions.

### Strengths
- The introduction and background are really a pleasure to read; the SBP is very well introduced a rigorously defined. 
- The manuscript present strong theoretical contributions and a novel method to solve the SBP.

### Weaknesses
*Experiments.*
I believe the authors should conduct additional experiments to show the advantages and limitations of their method. There are many toy datasets or graph that could be used in addition to this one, for instance a Swiss roll (potentially rotated in higher dimensions) on which you can generate from a distribution or interpolate between distributions. I also think that reporting additional metrics for evaluation would be beneficial, for instance MMD, Wassertein etc, between predicted and ground truth distributions. 

It is difficult to assess the improvement of the method on figure 2 or 3. Quantitative results along the lines of table 1 are much better. Since this data appears to have multiple timepoints, the authors could validate the quality of interpolation by training with holding out one timepoint and predicting it at inference. The same experiment, should be conducted on multiple datasets, e.g. datasets from [3,4].

*On solution complexity.*
It is mentioned that $\Psi_t$ can be computed efficiently, but this requires the eigendecomposition of L, which scales at O(n^3) for an $n \times n$ L. Did the authors try other approximation methods such as Chebyshev (e.g. used in [1]) or Cholesky (e.g. used in [2]) ? These methods scale much better for on a sparse graph. Additionally,, it would be beneficial to add experiments on complexity and training time with respect to the size of the Laplacian.

### Questions
Questions and minor comments.

- On line 146, and the definition of the kernel $\Sigma = \exp(-\kappa^2/2L)$ is L at the numerator or denominator ? If it is like a heat kernel, it should be at the numerator.
- On line 225 (and other sections), $\exp(-cLt)$ is a matrix exponential ?
- I may have missed it, but what is $\xi_1$ on line 248 ?
- In equations 3a and 3b, it looks like 3a goes from t=0 to t=1, and 3b goes from t=1 to t=0, but it is not clear from these equations. Do you also need to define the backward Wiener process in terms of the forward one ?

Potential typo:
- Line 133: "concerns"
- Line 141: "high-dim"
- Line 146 "ndoe"
- line 446: "borh"

[1] Huguet, G., Tong, A., Zapatero, M. R., Tape, C. J., Wolf, G., & Krishnaswamy, S. (2023, September). Geodesic sinkhorn for fast and accurate optimal transport on manifolds. In 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1-6). IEEE.

[2] J. Solomon, F. De Goes, G. Peyre,´ et al., “Convolutional
wasserstein distances: Efficient optimal transportation
on geometric domains,” ACM Transactions on Graphics
(ToG), 2015

[3] Charlotte Bunne, Laetitia Meng-Papaxanthos, Andreas Krause, and Marco Cuturi. Proximal Optimal Transport Modeling of Population Dynamics, February 2022. arXiv:2106.06345 [cs]. Cited
on page

[4] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models
with minibatch optimal transport, March 2024. arXiv:2302.00482

### Soundness
2

### Presentation
4

### Contribution
3
