# PERSISTENCE SPHERES: BI-CONTINUOUS REPRESENTATIONS OF PERSISTENCE DIAGRAMS.

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 0, 6

## Abstract
Persistence spheres are a new functional representation of persistence diagrams. In contrast to existing embeddings such as persistence images, landscapes, or kernel-based methods, persistence spheres define a bi-continuous mapping: they are Lipschitz continuous with respect to the 1-Wasserstein distance and admit a continuous inverse on their image. This provides both stability and geometric fidelity, placing persistence spheres among the few representations of persistence diagrams that offer an inverse-continuity guarantee.
We derive explicit formulas for persistence spheres and show that they can be computed efficiently with minimal parallelization overhead. Empirically, we evaluate them on clustering, regression, and classification tasks involving functional data, time series, graphs, meshes, and point clouds. Across these benchmarks, persistence spheres are competitive with, and often improve upon, standard baselines including persistence images, persistence landscapes, persistence splines, and the sliced Wasserstein kernel. Additional simulations in the appendices further support the method and provide practical guidance for tuning its parameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a bi-continuous representation (with respect to the 1-Wasserstein distance) of Persistence Diagrams (PDs) via the restricted support of lift zonoids, i.e. Persistence Sphere (PS). The property of being bi-continuous is theoretically proved. The experiments focus on regression and classification tasks.

### Strengths
Strengths:   

- The bi-continuity w.r.t. Wasserstein distance is very interesting and important for ensuring PD correspondence, the key component in Topological Machine Learning.
- The background definitions and proofs are through and self-contained.
- The bi-continuity holds for different weight functions ws. This provides flexibility for the representation.
- The theoretical proofs for continuity theorems are sound.
- The datasets in experiments are extensive and the results are convincing.

### Weaknesses
On the presentation aspect:
- The discretization method of the Persistence Sphere, a restricted support function of lift zonoids is unclear.
- The paper focuses too much on the math notations. Only one visualization (Fig c) on the proposed method is provided. The lack of visual examples in $\mathbb{R}^3$ of the key components such as lift zonoid of a PD (definition 10) and PS (definition 11) makes these concepts difficult to understand intuitively and reduces readability.
- The background of Persistence Diagram, a multiset on the open half plane, is very limited. Maybe a detailed one including the filtration process could be included in the Appendix.

On the experiments aspect:
- The baselines (PI, PL, SWK) are very limited. More recent methods such as Persistence B-spline grids [1], an unsupervised method, should be included.
- The baselines are all unsupervised vectorization methods. Since the paper focus on supervised task like classification, I think it is necessary to compare PS with supervised vectorization like PersLay [2] to show whether PS can outperform supervised baselines. 
- The paper gives computational analysis without showing the actual time cost of PS. A scale-up test would make the analysis more convincing.
- Although it is nice that the bi-continuity holds for different weight functions $w$s, when it comes to experiments, it would be necessary to show how the choice of weight function affects the results. An ablation study on the choice of weight function would be nice.

[1] Dong Z, Lin H, Zhou C, et al. Persistence B-spline grids: stable vector representation of persistence diagrams based on data fitting[J]. Machine Learning, 2024, 113(3): 1373-1420.  

[2] Carrière M, Chazal F, Ike Y, et al. Perslay: A neural network layer for persistence diagrams and new graph topological signatures[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2020: 2786-2796.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

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
The paper proposes a new way of vectorizing persistence diagrams (PD). The idea of the vectorization procedure is the following: for each point $(b, d)$ of a PD with multiplicity $k$, consider a line segment from the origin to the point $(k, kb, kd)$. Take the Minkowski sum of all these segments, it is a convex polytope in $\mathbb{R}^3$. Now map the PD to the support function of this polytope (restricted onto the unit sphere). In fact, one has to apply weighting (points that are closer to the diagonal must contribute less). The paper establishes conditions for the weighting function that guarantee that the mapping PD -> support function is a Lipschitz embedding with contiuous inverse. The experimental part demonstrates the efficiency of this vectorization technique on various classification problems, by comparing it to many other known methods.
NB: the unweighted vectorization scheme was proposed in the literature, and the paper contains the appropriate references and attribution; the actual contribution of the paper is weighted case, its theoretical properties and experimental evaluation.

### Strengths
- The paper is rather well-written and very clear, being also practically self-contained.
- The properties that the embedding needs are certainly important, not just from a theoretical viewpoint, but also practically. Therefore the criteria on the weighting established in the paper are a solid contribution.
- The experiments cover various problems and look convincing.
- The proposed vectorization method is embarassingly parallel and can be expressed by a very simple formula.

### Weaknesses
- functions on a sphere are hard to work with, unlike, say, persistent images which are honest-to-goodness 2D numerical arrays. This fact is acknowledged in the paper.
- the paper could include the results for some of the experiments that the unweighted case paper made.

### Questions
Why not use spherical harmonics? I mean, to represent the vectorization instead of splines one could use the Fourier decomposition. Or has it been tried and the splines perform better?

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
4

### Summary
This paper proposes a novel functional representation for Topological Data
Analysis, specifically for Persistence Diagrams (PDs).

The approach builds upon the theory connecting persistence diagrams with
measure theory and optimal transport. The functional associated with a
persistence diagram (viewed as a discrete measure) is then derived from the
support function of a zonoid (a convex set) constructed from this measure.

After the necessary background, the authors present the novel functionnal :
Persistence Sphere, and show that it satisfied multiple desirable properties
for a TDA representation, namely: stability, and separability.

The paper concludes with an experimental section, showcasing the performances
of this approach on various datasets.

### Strengths
- Well written. 
 - Proofs seem to be sound 
 -  The proposed construction sounds reasonable : stability (w.r.t.
 Wasserstein 1) is a must have for persistence diagram representations, and
 the separability property is very nice
 -  Computational complexity is linear
 - The code is available, and the code is reproducible in notebooks, with
 clear parameter selection.

### Weaknesses
- This paper is hard to read for someone who is not familiar with measure
 theory / TDA. For instance, there is no real intuition on what is a
 persistence diagram, how to interpret the matching cost, etc. As the paper
 presents a representation for PDs though, I think this approach can be argued
 to be fair. 
 - The experiment section is weak. IIUC the SOTA is roughly from 2017 (sliced
 waserstein kernel).
 - l60-70 : Unless I misunderstood something, SWK is a stable and bicontinuous
 representations as well, with stronger separability properties (cf [SWK for
 PD, theorem 3.3]). The comparison to SWK is still on your favor though, since
 SWK is not an unsupervied vectorization method.
 - Def. 10 : a little picture/explanation could help
 - l193 : uniformly integrable is not def
 - l225 : "and p \in B_r^c" 
 - l232 : link with uniform integrability?
 - l240 : "and α ≥ 1. They are also effective weightings for α = 1" what is
 the point of the last sentence ?
 - l244 : This is a bit fast here. I suppose the ReLU comes from the fact that
 the segments of the zonoid are based on 0.
 - l257 : the [Carrière Bauer] paper mentionned in the "Contribution"
 paragraph fits here ?
 - l273 : this is a bit fast as well here. I assume that $Z_\varnothing =
 \{0\}$?
 - l477-483 : some references should be added
 - l647 : K
 - l740 : I'm not sure the notation with \Phi help a lot
 - l777 : By prop. 2, [...] to conclude with thm 1
 - l787 : effective weighting
 - l793, 798: a factor 2 is missing

### Questions
- See weaknesses.
 - When comparing with the SWK, two question naturally arise : 1) is it
 possible to have a better separability bound ? e.g., f(W_1(...)) <= || PS...
 ||_p  for a non-trivial non-decreasing function f ? and 2) The sliced
 wasserstein distance can be exactly computed on small enough diagrams, since
 the combinatorix is finite, and can thus be decomposed into a finite number
 of cells. What about the || PS_1 - PS_2 ||_p of two diagrams ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper presents persistence spheres, a new way to represent persistence diagrams as functions. The method provides a bi-continuous mapping that is Lipschitz continuous with respect to the 1-Wasserstein distance and has a continuous inverse on its image. These properties ensure both stability, meaning similar diagrams produce similar functions, and geometric fidelity, meaning that similar functions reflect similar diagrams. Together, they offer an optimal balance between robustness and faithfulness to the original geometry of persistence diagrams.

### Strengths
The experiments demonstrate that persistence spheres perform consistently well across a wide range of data types.

### Weaknesses
The paper as a whole reads more like a sequence of definitions and technical explanations than a cohesive narrative, which makes it difficult for readers to grasp the overarching motivation and significance of the work. While the theoretical foundations are clearly laid out, the presentation lacks a guiding storyline that connects these formal definitions to the broader research goals and practical implications. Nearly two pages are devoted solely to dataset descriptions, and almost another full page focuses on hyperparameter settings and implementation details. While such information supports reproducibility, it disrupts the flow of the paper and could be moved to an appendix or supplementary material. This space could instead be used to strengthen the narrative by clarifying research goals, providing interpretive discussion, and highlighting the significance of the findings. Moreover, several analytical gaps limit the depth of the study. There are no direct runtime benchmarks, so although computational efficiency is discussed qualitatively, explicit timing comparisons with existing topological representations such as persistence images, persistence landscapes, or the sliced Wasserstein kernel are missing. Hyperparameter sensitivity is not analyzed, leaving unclear how robust the method is to parameter changes, and no ablation studies are provided to isolate the effect of specific design choices such as the weighting function or spline grid size. Scalability testing is also limited to moderate-sized datasets, offering no insight into how the approach performs on large or high-dimensional data. Finally, the reported results rely on averages over runs but include no statistical significance tests or confidence intervals, making it difficult to assess whether performance differences are meaningful.

### Questions
How do persistence spheres advance the broader understanding of how topological information can be integrated into machine learning, beyond simply providing another competitive embedding for persistence diagrams?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes persistence spheres (PS), a new functional representation of persistence diagrams (PDs) for topological data analysis. Unlike existing embeddings, persistence spheres yield a bi-continuous embedding, which guarantees both stability and geometric fidelity. The paper defines persistence spheres via the support functions of lift zonoids of re-weighted persistence diagrams, proves continuity theorems, and provides explicit, efficiently computable formulas. Experiments across several benchmark datasets and show that persistence spheres are comeptitive or superior to established methods in regression and classification.

### Strengths
The paper demonstrates originality by introducing a mathematically rigorous yet computable embedding that enhances geometric fidelity compared with existing topological data analysis (TDA) vectorizations. Its quality is supported by solid theoretical foundations, complete proofs, and carefully reproducible experiments. The empirical validation spans diverse application domains, including functional, graph, mesh, and point-cloud data, underscoring the versatility of the proposed method. In terms of scalability, the approach achieves linear complexity with respect to the number of persistence diagram points and allows straightforward parallelization.

### Weaknesses
The paper lacks an ablation or sensitivity analysis on key hyperparameters such as the weighting function $\omega$ or the number of basis functions. These parameters likely influence both the expressiveness and stability of the representation, yet their impact is neither theoretically discussed nor empirically evaluated.
Grammar errors:
Line 465: “due the high variability” → should be “due to the high variability.”

### Questions
1. How sensitive are the results to the specific hyperparameter choices—particularly the weighting function $\omega$, the number of basis functions — and could the authors provide an ablation or sensitivity analysis to clarify their impact on performance and stability?
2. Runtime comparisons: the paper claims linear complexity, but actual wall-clock runtimes versus PIs/PLs are not reported—can authors provide them?

### Soundness
3

### Presentation
3

### Contribution
3
