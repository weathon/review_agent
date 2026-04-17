# Fractal Graph Contrastive Learning

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
While Graph Contrastive Learning (GCL) has attracted considerable attention in the field of graph self‑supervised learning, its performance heavily relies on data augmentations that are expected to generate semantically consistent positive pairs. Existing strategies typically resort to random perturbations or local structure preservation, yet lack explicit control over global structural consistency between augmented views. To address this limitation, we propose Fractal Graph Contrastive Learning (FractalGCL), a theory-driven framework introducing two key innovations: a renormalisation-based augmentation that generates structurally aligned positive views via box coverings; and a fractal-dimension-aware contrastive loss that aligns graph embeddings according to their fractal dimensions, equipping the method with a fallback mechanism guaranteeing a performance lower bound even on non-fractal graphs. While combining the two innovations markedly boosts graph-representation quality, it also adds non-trivial computational overhead. To mitigate the computational overhead of fractal dimension estimation, we derive a one-shot estimator by proving that the dimension discrepancy between original and renormalised graphs converges weakly to a centred Gaussian distribution. This theoretical insight enables a reduction in dimension computation cost by an order of magnitude, cutting overall training time by approximately 61\%. The experiments show that FractalGCL not only delivers state-of-the-art results on standard benchmarks but also outperforms traditional and latest baselines on traffic networks by an average margin of about remarkably 4\%. Codes are available at (\url{https://anonymous.4open.science/r/FractalGCL-0511/}).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes FractalGCL, a theory-driven framework that enhances graph contrastive learning by explicitly maintaining global structural consistency through fractal-based augmentation. The method introduces a renormalisation-based augmentation and a fractal-dimension-aware contrastive loss, providing both improved representation quality and a theoretical performance lower bound even for non-fractal graphs. Moreover, the authors derive a one-shot estimator to efficiently approximate fractal dimensions, achieving training time reduction while attaining state-of-the-art performance across benchmark and traffic network datasets.

### Strengths
- The writing is good.
- The work is theoretically grounded.
- The performance is superior to the baselines.

### Weaknesses
* The overall design lacks consistency and coherence.
* The selection of baselines is not sufficiently comprehensive.
* There is performance inconsistency.
* The motivation experiment (Table 5) appears to contain methodological or statistical errors.

### Questions
* In the Introduction, the stated motivation focuses on ensuring *global semantic consistency* between positive pairs (inter-view similarity). However, the subsequent discussion emphasizes *self-similarity* and *hierarchical complexity*, both are intra-view properties. These two objectives operate at different conceptual levels and lack a unified logical connection.
* The renormalisation algorithm currently considers only graph topology. How can it be extended to handle graphs with node and/or edge attributes?
* In Section 3.2, the notation $z_n$ is overloaded: in *Mapping from Graph G to Representation z*, it refers to the embedding of the renormalised graph, while in *Contrastive Loss with Fractal Weight*, it denotes the embedding of the original graph. This inconsistency confuses the reader.
* In Section 3.2 and Figure 1, the augmented views are described as disjoint unions from two independent runs. However, in Section 3.3, the contrastive loss is defined between the original and renormalised graphs, not between two different unions. This discrepancy makes it difficult to assess the correctness and validity of the overall framework.
* The assumptions (A1–A4) referenced in Lemma 3.8 are not explicitly stated.
* What is the model’s performance under a transfer learning setting (e.g., MoleculeNet)?
* Please provide details on computational overhead (GPU memory usage and training time) compared to the baselines.
* What are the runtime complexities of Algorithms 1 and 2?
* In Table 4, the result for D&D (80.14) is notably lower than that in Table 1 (81.71). Please explain this inconsistency.
* A sensitivity analysis on the radius parameter used in the greedy box covering should be included.
* In Table 5, the authors report the number of graphs whose box-counting $R^2$ exceeds various thresholds. According to Algorithm 1, when the graph diameter ≤ 9, $R^2 = 0$. Based on my statistical check, the number of graphs with diameter > 9 defines the upper bound for Table 5 values. However, the reported ratios for $R^2 > 0.5$ exceed these bounds, raising serious concerns about the validity and authenticity of the results.


| Dataset    | #Graph | #Diameter<=9 | #Diameter>9 | Max-possible-$R^2>*$-ratio(%) |
|------------|--------|--------------|-------------|-----------------------------|
| PROTEINS   | 1113   | 521          | 592         | 53.19                       |
| MUTAG      | 188    | 143          | 45          | 23.94                       |
| NCI1       | 4110   | 889          | 3221        | 78.37                       |
| D&D        | 1178   | 31           | 1147        | 97.37                       |
| REDDIT-B   | 2000   | 910          | 1090        | 54.50                       |
| REDDIT-M5K | 4999   | 824          | 4175        | 83.52                       |


* Add more recent SOTAs, including TopoGCL[1], GCS[2], SEGA[3], CI-GCL[4], StructPosGSSL[5].



[1] Chen Y, Frias J, Gel Y R. TopoGCL: Topological graph contrastive learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2024, 38(10): 11453-11461.

[2] Wei C, Wang Y, Bai B, et al. Boosting graph contrastive learning via graph contrastive saliency[C]//International conference on machine learning. PMLR, 2023: 36839-36855.

[3] Wu J, Chen X, Shi B, et al. Sega: Structural entropy guided anchor view for graph contrastive learning[C]//International Conference on Machine Learning. PMLR, 2023: 37293-37312.

[4] Tan S, Li D, Jiang R, et al. Community-Invariant Graph Contrastive Learning[C]//International Conference on Machine Learning. PMLR, 2024: 47579-47606.

[5] Wijesinghe A, Zhu H, Koniusz P. Graph Self-Supervised Learning with Learnable Structural and Positional Encodings[C]//Proceedings of the ACM on Web Conference 2025. 2025: 4053-4067.

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
This paper proposes FractalGCL, a graph contrastive learning framework that leverages fractal geometry for improved graph representation learning. The method introduces a renormalization-based augmentation strategy that generates structurally aligned positive pairs via box coverings, and a fractal-dimension-aware contrastive loss that aligns embeddings according to their fractal dimensions. Experiments demonstrate state-of-the-art results on benchmark datasets.

### Strengths
1. The paper is among the first to rigorously integrate fractal geometry into graph contrastive learning.
2. The preliminary experiments convincingly demonstrate that fractal structures are prevalent in real-world graphs.

### Weaknesses
1. The paper introduces the concept of fractal geometry into GCL, but this motivation is not well aligned with the core challenges of GCL. A lot of work have consistently shown that the quality and hardness of negative samples, or hard negatives, are the dominant factors affecting GCL performance. However, this work focuses almost exclusively on generating or weighting positive pairs through fractal transformations, which addresses a secondary factor rather than the main bottleneck of GCL. In my view, for GCLs, discriminability is far more important than consistency. As a result, the proposed motivation feels tangential to the central problem and may offer only marginal benefit compared to methods that explicitly handle hard negatives or contrastive sampling bias. 
2. The theoretical results assume infinite graphs with well-defined limiting behavior, but all experiments use finite graphs with relatively small diameters. 
3. The method assumes graphs exhibit strong fractal properties, but this fundamentally limits applicability. While preliminary experiments show high fractal prevalence in some benchmarks, many real-world graphs (social networks, citation networks) may not satisfy this assumption.  The proposed safe fallback mechanism essentially disables FractalGCL components for non-fractal graphs, reducing the method to baseline GCL. This is not a feature but rather an admission of limited scope.
4. The contrastive fractal weight depends on a noisy estimator $dim_B$, whose sensitivity is not studied.
5. The method only activates fractal machinery when an $R^2$ test passes (and uses tuned thresholds), effectively admitting limited applicability and per-dataset calibration.
6. The contrastive setup treats negatives as other graphs in the batch, and the experimental protocol contains no comparisons with hard-negative mining or debiasing methods. This makes it hard to judge whether the proposed positive-side modifications actually outperform stronger negative-side baselines.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the FractalGCL framework, which introduces fractal geometry theory into graph contrastive learning to enhance global structural consistency. It generates structurally similar positive sample views through graph renormalization and designs a fractal dimension-aware contrastive loss function. To reduce computational overhead, the authors theoretically derive the dimensional difference as a Gaussian perturbation, enabling efficient approximation. Experiments on multiple graph classification tasks and real-world traffic networks demonstrate the superior performance of FractalGCL, validating its theoretical innovation and practical effectiveness.

### Strengths
1.The authors introduce a novel perspective based on fractal geometry and propose an original approach that enhances and aligns graph representations through graph renormalization and a fractal dimension-aware loss. 

2.They mathematically prove that graph renormalization does not alter the fractal dimension and employ the Central Limit Theorem to derive a Gaussian approximation of dimension differences, thereby reducing computational complexity.    
                             
3.The method demonstrates superior performance over existing approaches on standard graph classification benchmarks and real-world traffic networks, validating the effectiveness and practicality of the FractalGCL framework.

### Weaknesses
1. This method relies on the assumption of graph "fractality," but not all graphs exhibit strong fractal structures. For instance, its significantly lower performance on REDDIT-B suggests that the method may not be suitable for non-fractal graphs.

2. FractalGCL currently focuses on graph-level representation learning and lacks experimental results or theoretical justification for node-level contrastive learning.

3. Although approximate derivations are adopted to reduce computational cost, the overall method remains more complex than mainstream baselines, especially due to the need to compute graph diameter and box dimension during preprocessing.  

4. Proposition 3.6 provides only the asymptotic complexity, but there are no empirical results showing how the actual runtime or memory usage scales with the number of nodes, and the paper lacks corresponding experiments.

### Questions
1. When the graph does not exhibit clear fractal characteristics (such as social or citation networks), can FractalGCL still maintain its performance, or will it degenerate into a standard GCL?
2. The fractal dimension can be regarded as a measure of multi-scale structural complexity. What is its relationship with spectral graph features (e.g., Laplacian eigenvalue distributions)? Can the two be combined?

### Soundness
2

### Presentation
2

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
In response to the issue of the lack of global structural consistency control in traditional graph contrastive learning methods during data augmentation, this paper proposes Fractal Graph Contrastive Learning (FractalGCL). First, a re-normalization enhancement method is employed, utilizing box-covering techniques to generate structurally consistent positive sample views, ensuring global structural consistency across the augmented views. Secondly, a fractal dimension-aware contrastive loss is designed, which aligns graph embeddings based on the fractal dimension of the graph, ensuring that the method maintains a minimum performance lower bound even on non-fractal graphs. To address computational overhead, the paper further proposes a low-complexity fractal dimension estimation method based on theoretical analysis, significantly reducing the computational burden, with a 61% reduction in training time. Experimental results demonstrate that FractalGCL not only performs exceptionally well on standard benchmark tests but also improves performance by approximately 4% on complex traffic network data compared to traditional methods and the latest baselines, validating its superior advantages in graph representation learning.

### Strengths
1、This paper explores the issue of structural consistency in graph contrastive learning (GCL) from the mathematical theory of fractal geometry. This approach breaks through the traditional GCL paradigm, which relies solely on random perturbation or heuristic augmentation, by introducing theoretical explanation and structural constraints into the augmentation process.

2、In the design of contrastive loss, FractalGCL introduces a fractal dimension-aware contrastive loss function (Fractal-Dimension-Aware Loss), explicitly incorporating global self-similarity as a structural consistency signal into the learning objective. This method not only focuses on semantic proximity in the embedding space but also ensures that the model maintains consistent fractal structural features across different scales, thereby providing cross-scale robustness for the learned graph representations.

3、Through experiments, this paper verifies the widespread existence of fractal structures in real graph data and demonstrates that fractal features can effectively enhance performance.

4、To avoid the high cost of traditional full-graph covering algorithms, this paper theoretically proves that the box dimension difference between the original graph and the augmented graph follows a Gaussian weak convergence law. Based on this, Gaussian perturbation approximation is used to replace the high-cost dimension estimation, allowing FractalGCL to maintain structural consistency constraints at a lower computational cost and translating fractal theory into a computable and scalable optimization strategy.

### Weaknesses
1、The theoretical foundation of FractalGCL is based on the fractal characteristics of graphs, but the paper does not strictly define the assumption that "fractal features are universally present in graph structures." The method assumes that all graphs can be characterized by fractal dimensions but does not discuss its effectiveness or failure conditions in non-fractal structures.

2、The paper's ablation experiments are conducted only on the MUTAG dataset, which is very small (only 188 graphs), with considerable statistical fluctuations, making the results susceptible to random factors. There is a lack of repeated validation on medium to large datasets (e.g., D&D or NCI1), and no standard deviation or significance tests (e.g., t-test) are reported.

3、The paper highlights that FractalGCL achieves a 61% increase in training speed by replacing precise dimension calculations with Gaussian approximation, but this conclusion is based solely on comparisons within its own versions and lacks a comprehensive comparison with existing mainstream contrastive learning models (such as GraphCL, BGRL, and GRACE).

4、There are inconsistencies in notation within the paper, such as the use of "FractalGCL" and "Fractal GCL" interchangeably.

### Questions
1、Is the low-complexity fractal dimension estimation method proposed in the paper equally stable on graphs with different scales, densities, and topological structures? Has its variance and error control capability been thoroughly validated across multi-scale graph data?

2、Can the experimental results conducted on small sample datasets like MUTAG eliminate the influence of random factors? In the absence of significance tests or standard deviation analysis, how can we confirm that the performance differences are indeed due to method improvements rather than random fluctuations?

3、FractalGCL achieves a 61% improvement in training speed on small datasets. However, does this efficiency advantage hold true in large-scale or dynamic graph scenarios?

### Soundness
3

### Presentation
3

### Contribution
4
