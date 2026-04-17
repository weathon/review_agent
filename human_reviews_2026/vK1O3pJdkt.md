# One cluster or two? A Manifold-Based Approach

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
The manifold hypothesis suggests a natural criterion for clustering: partition data according to the manifold component from which they are drawn. This criterion is useful because, intuitively, the separability of manifold components is governed by the ambient separation between components relative to the largest gap in the sample’s coverage. The analysis integrates topology (e.g., manifold volume and reach) with estimation (e.g., fill radius and sample density). Formally it identifies a criticality: when a threshold is exceeded, nearest-neighbor data graphs avoid bridging edges and clusters are preserved; otherwise, bridges appear and components fuse. Practically, criticality is sandwiched between bounds that imply a measure of cluster confidence, and motivates an algorithm-Manifold-Based Clustering (MBC)-that constructs a candidate neighborhood graph. MBC is parameter-light and, unlike density-based methods (e.g., HDBSCAN), avoids hand-tuned scale thresholds. Instead, MBC yields a monotone bracket on the number of components by a natural sweep of neighborhood size. Across curved and high-dimensional benchmarks, MBC matches state-of-the-art accuracy and exposes ambiguity near the critical thresholds.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method for determining whether samples are "truly" from separate manifold components (under the manifold hypothesis). It then presents a way to cluster based on this notion and evaluates this clustering against density-based methods on toy datasets.

### Strengths
The paper proposes a novel way to make clustering and high-dimensional point evaluations based on the manifold hypothesis. I appreciate the efforts to formalize elements of the manifold hypothesis and to transfer this formalization into a practical set of decision-making criteria. I also find that the exposition and motivation are well-structured and that the algorithmic choices are well motivated. In short, I commend the paper's goal and believe the authors approached this goal in an appropriate way.

### Weaknesses
Unfortunately, I also think the authors are too focused on the specifics of making things sound nice and well-motivated. This comes at the expense of a practically sound and effective method. This is evidenced first in the experimental results. Specifically, the proposed algorithm only works in toy settings where the separation between clusters is extremely clear. In these settings, there is a large number of algorithms which will find identical clusterings (single-linkage clustering, ward algorithm, spectral clustering, etc.). Indeed, showing effectiveness on 2-moons and concentric circles is a common first step when showing experimentally that a non-convex clustering works as intended.

However, the authors' chosen algorithm immediately stops working once we reach "real" data. I would note that these real datasets are still fairly simple and much cleaner than the *actual* messiness of large, high-dimensional datasets. Indeed, I am confident that an effective method would be to run UMAP on the high-dimensional datasets and then apply HDBSCAN to the outputs. The effectiveness of these standard clustering procedures suggests that there *is* structure to cluster along.

Nonetheless, the authors' proposed algorithm stops working effectively on these real-world datasets. The authors suggest that this is because, for real datasets, the assumption of multiple distinct manifolds is broken. I agree with this conjecture, but I also argue that this significantly weakens the authors' proposed algorithm. Namely, *all* reasonably sophisticated real-world datasets will violate the assumptions of having multiple distinct, well-separated and well-sampled manifolds. Consequently, the authors' theoretical analysis, while interesting, does not naturally transfer to real-world settings.

What I would find more compelling is to find an analysis which explains why, for example, doing tSNE/UMAP into density-based clustering (or spectral, single-linkage, etc.) works as well as it does. It seems the authors' work could be naturally extended to such an analysis, and this would make the work more practically valuable.

### Questions
These are smaller questions than those implied in the weaknesses section:
- Why is the first theorem statement so long? Ideally, the assumptions/requirements should be stated outside the theorem environment so that the theorem statement itself can be made simple to interpret.
- Why do the authors state that their clustering algorithm does not require hyperparameters? It seems that, later on that page, they specify several hyperparameters which they've chosen.
- Some words are used without being introduced. For example, I am not familiar with the concept of "tubular" noise. Similarly, I'm not sure that manifolds are ever formally introduced. In general, although I could follow the statements made in the paper, I fear that somebody outside of our narrow subfield would struggle interpreting the statements.
- Is it clear why we are applying random geometric graph asymptotics to the analysis? The assumption that the samples exist on manifolds with varying density seems to contradict the fact that these are random geometric graphs. Consequently, none of the connectivity properties of random graphs should apply.

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
2

### Summary
The paper introduces a theoretical and algorithmic framework for clustering based on the manifold hypothesis. It defines a geometric separability criterion using the ratio between inter-manifold offset and sampling fill distance, proving threshold theorems for when kNN graphs connect or separate manifold components. Building on this criterion, the authors propose a Manifold-Based Clustering algorithm that avoids manual density thresholds and provides a confidence bracket on the number of clusters. Experimental studies on synthetic, high-dimensional, and biological datasets demonstrate that MBC achieves accuracy comparable to state-of-the-art methods.

### Strengths
- The paper provides a rigorous analysis connecting geometric properties (reach, fill distance) with random graph connectivity, establishing explicit separation thresholds with proofs.
- The proposed MBC directly translates theory into a practical algorithm that is nearly parameter-free and interpretable through the offset/distance ratio, which is much more preferred in practice compared to density-based methods (DBSCAN, HDBSCAN) that require hand-tuned thresholds.
- Experimental results on different datasets show the effectiveness of the theory and the proposed method. Especially on the challenging data: the neural case study effectively demonstrates the method's utility in real scientific contexts, where MBC's brackets correctly distinguish between genuinely clustered and unclustered (V1) data while baselines force spurious partitions.

### Weaknesses
- There is limited scalability analysis in this paper. While computational complexity is briefly mentioned (O(n^2D) in high dimensions with brute-force k-NN), there's insufficient empirical evaluation of runtime scaling. The largest dataset appears to be MNIST (n~60k), leaving questions about performance on modern large-scale applications (for example, n > 1M)
- More baseline methods can be compared. In particular, while DBSCAN, OPTICS, and HDBSCAN are tested, the paper omits comparisons to spectral or graph neural clustering methods that could be more competitive. Besides, the comparison focuses primarily on density-based methods; other manifold-aware approaches (e.g., recent diffusion-based or topological clustering methods) are not evaluated.
- There is limited exploration of sensitivity. Although claimed parameter-light, practical sensitivity to \alpha, and the rescue-step constants (\theta, q, etc.) is not systematically analyzed.

### Questions
- Could the authors elaborate on the computational cost of the method (time and memory) relative to HDBSCAN or spectral clustering, especially in high dimensions?
- How sensitive is the reported confidence bracket to the choice of parameters?

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
3

### Summary
The paper addresses a fundamental question in unsupervised learning: under what conditions does data naturally form multiple clusters rather than a single connected structure?

Leveraging the manifold hypothesis, the authors model the data as samples drawn from a manifold which may consist of multiple disconnected components. In this framework, clustering becomes a topological decision problem—determining whether tghe manifold M is connected or separable by a measurable gap.

Using tools from random geometric graph theory, the paper derives sharp connectivity thresholds for k-nearest-neighbor (kNN) graphs: above a critical ratio Δ/h, components remain disconnected with high probability; below this threshold, cross-edges emerge and the structure becomes connected.

Building on this theoretical insight, the authors propose MBC, a parameter-efficient geometric clustering algorithm that exploits these connectivity properties for robust unsupervised partitioning.

### Strengths
The paper elegantly bridges topological geometry concepts—such as reach and fill distance—with graph-based clustering through explicit probabilistic threshold analysis.

Its theoretical framework unifies manifold geometry, random geometric graph theory, and noise analysis into a coherent and rigorous foundation for understanding cluster connectivity.

### Weaknesses
The theoretical results may have limited relevance in high-dimensional settings (d large), which could restrict the scalability of the approach for large or complex datasets.

Algorithm 1 involves several parameters that are fixed without clear justification—their selection appears somewhat ad hoc (see Step 1 of the algorithm). A discussion or sensitivity analysis of these parameter choices would strengthen the paper.

In addition, the presentation needs improvement. The TwoScaleDTM function in Step 14 is not a standard algorithm and should be explained more clearly. Although a definition is briefly provided in lines 251–253, the formula used in Step 14 differs from that description. Moreover, several components are undefined or ambiguous: what does CC represent in Step 19? what is GM in Step 20? and where does K in Step 20 come from? Finally, Section A.2 is currently empty and should either be completed or removed.

### Questions
Since the proposed algorithm is built upon the k-nearest-neighbor (kNN) framework, it would be reasonable to include a standard kNN-based clustering method as a baseline for comparison. This would help clarify the performance gains introduced by the proposed modifications.

The implication of Theorem 1 is also not entirely clear. It guarantees only that the “Euclidean geometric-mean gate followed by triangle support” introduces no cross-component edges, which is a relatively weak form of assurance. One would expect a stronger theoretical result—such as a bound on the clustering error rate or guarantees on correctly identifying the number of clusters—to better demonstrate the algorithm’s reliability and effectiveness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new approach to clustering based on the manifold hypothesis, which assumes high-dimensional data lies on or near lower-dimensional manifolds. The core idea is that instead of forcing data into clusters, it is better to first understand whether distinct clusters exist at all at the available sampling scale. The clustering criterion is based on intuition similar to Gaussian Mixture Models—comparing separation between groups to dispersion within groups—but generalized to the case of Riemannian manifolds. Specifically, the authors propose comparing the offset Δ (minimal distance between manifold components) to the fill distance h (worst-case sampling gap), forming the ratio ρ = Δ/h that governs separability. When this ratio exceeds a threshold, manifold components remain disconnected in a k-nearest-neighbor graph; otherwise, they fuse together. The method is validated through experiments on both synthetic benchmarks and real-world neural data, demonstrating its ability to recover clusters when they are clearly separated and to report uncertainty (via a "bracket" on the number of components) when the data is ambiguous.

### Strengths
The main ideas of the method are interesting and promising: assuming clusters belong to different manifolds is reasonable and the kNN are usually a very good basis for clustering. The core concepts are rigorously stated and proof sketches are given. 

1. Originality. The extension of the Gaussian Mixture Model framework to Riemannian manifolds is an elegant contribution and the core conceptual framework appears sound and well-motivated. The authors adapt the variance-to-separation intuition from GMMs to the manifold setting by replacing variance with fill distance and inter-cluster distance with manifold offset.

2. Quality. The paper provides theoretical rigor and empirical validation. Key claims are supported by formal proofs (Theorems 3.3, 4.1), however, the empirical evaluation is weak. 

3. Significance. In my opinion, the contribution addresses an important gap in clustering methodology by reducing hyperparameter dependence (line 272). Unlike density-based methods that require manual scale thresholds, MBC provides a principled, geometry-driven decision criterion and reports uncertainty through a monotone bracket on the cluster count. This reduction in required prior knowledge makes the method more accessible and robust across application domains.

### Weaknesses
W1) Empirical evaluation: Unfortunately, the experimental results are not convincing. There are only few datasets tested. Most of them are low-dimensional toy datasets, where MBC works well, however, on all higher-dimensional datasets it yields NMI and ARI value of 0.0 -- This holds for Digits, MNIST, Retina (subset as well as full dataset). So I do not see where MBC would be used in practice. 

Furthermore,  the parameter settings of competitive methods seem to have not been chosen well: in Table 1, OPTICS and HDBSCAN are reported to not find the correct clusterings for Two Moons and Concentric Circles, which both are datasets that are known to work well together with those algorithms. 

W2) The competitive methods are only basic clustering methods that all work very similar. Please also compare to similar but different methods (e.g., Spectral clustering, Density Peaks), and newer methods and improvements on DBSCAN (e.g., [0,1,2])

[0] Chazal, F., Guibas, L. J., Oudot, S. Y., & Skraba, P. (2013). Persistence-based clustering in Riemannian manifolds. Journal of the ACM (JACM), 60(6), 1-38.

[1] Xu, X., Ju, Y., Liang, Y., & He, P. (2015, October). Manifold density peaks clustering algorithm. In 2015 Third international conference on advanced cloud and big data (pp. 311-318). IEEE.

[2] Deng, C., Zhang, Q., Zhou, X., Zhang, S., Wang, G., & Xu, W. (2025). Density peaks clustering algorithm integrating manifold distance and mutual nearest neighbors. Pattern Recognition, 112554.


W3) The paper is quite hard to follow and could benefit from figures guiding the readers.

### Questions
Q1) Why were baseline methods used mostly with default settings and were any tuned versions tried?

Q2) How complex can the data get before no meaningful clustering is possible anymore with your method ? What is the expected runtime for each data complexity factor?

Q3) Why did you not compare to newer methods?

### Soundness
2

### Presentation
2

### Contribution
2
