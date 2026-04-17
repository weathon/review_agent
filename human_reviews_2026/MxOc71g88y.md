# Metric Graph Kernels via the Tropical Torelli Map

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
We introduce the first graph kernels for metric graphs via tropical algebraic geometry. In contrast to conventional graph kernels based on graph combinatorics such as nodes, edges, and subgraphs, our metric graph kernels are purely based on the geometry and topology of the underlying metric space. A key characterizing property of our construction is its invariance under edge subdivision, making the kernels intrinsically well-suited for comparing graphs representing different underlying spaces. We develop efficient algorithms to compute our kernels and analyze their complexity, which depends primarily on the genus of the input graphs. Empirically, our kernels outperform existing methods in label-free settings, as demonstrated on both synthetic and real-world benchmark datasets. We further showcase their practical utility with an urban road network classification task.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper builds on ideas from tropical geometry and presents a graph kernel for metric graphs. The kernel computes the matrix of the tropical Torelli map and is designed to be invariant under edge subdivisions. The authors show that the matrix of the tropical Torelli map is unique if the input graph has a unique minimal spanning tree and the lengths of edges are distinct. The matrix of the tropical Torelli map is then modified such that all graphs have matrices of equal size, and two variants of the kernel are produced, one that computes the Euclidean distance of the matrices and one that computes the Bures–Wasserstein distance. Distances are converted into similarities via an RBF function. The two variants are evaluated on synthetic and real-world datasets, including an urban road network classification task, and the results show that these kernels can outperform existing methods.

### Strengths
- To the best of my knowledge, there exist no graph kernels that are invariant under edge subdivisions. The proposed kernel is thus novel in this regard. In addition, no other kernels have been inspired by the field of tropical geometry. Therefore, it is my view that the kernel that is presented in the paper is novel.

- The authors have made a clear effort to motivate the proposed kernel and its properties (such as invariance under edge subdivisions) by discussing potential applications to road networks. 

- The two variants of the proposed kernel seem to lead to performance improvements on most of the considered datasets. However, there are several issues with the experimental evaluation, which are discussed below.

### Weaknesses
- The paper's main weakness is that there are several issues with the empirical evaluation of the proposed method.
    - No hyperparameter tuning was performed. The default hyperparameters were used both for the baseline kernels and also for the SVM classifier (its hyperparameter $C$ was not tuned). This raises doubts about the validity of the results. For a fair comparison, the hyperparameters of all kernels along with those of the SVM classifier should be optimized on some validation set.

    - I do not understand why the authors did not include the Weisfeiler–Lehman graph kernel, which is known to be state-of-the-art for many problems, in their experiments. This kernel can be actually applied to unlabeled graphs if all nodes are assigned the same label.
    
    - The results reported in Table 5 of the Appendix are far from state-of-the-art. I understand that this is because the discrete or continuous features of the nodes were ignored. However, it is not clear to me whether the overall comparison remains meaningful.


- If the genus of graph is much greater than $g_0$, a large number of rows and columns of the matrix of the tropical Torelli map are removed. This can lead to substantial information loss, and the kernel might fail to properly capture the similarity between graphs, thus affecting its overall performance.

- The TTW kernel is not actually a valid kernel. Therefore, for classification problems, the objective of SVM is non-convex, which might lead to instabilities. While in the experimental evaluation, this does not seem to happen, there might exist problem instances where the solver might fail to converge.

- The TTW is computationally very complex. On MSRC-21 and ER-MD, it is significantly slower than most of the competing kernels.

- The output of the kernel is unique only for graphs that have a unique minimal spanning tree and the lengths of the edges are distinct. These two conditions are not satisfied by most real-world graphs, and therefore, the kernel might produce different output when isomorphic graphs are given as input along with other graphs.

- Typo: l.277: "such the log-Euclidean metric" -> "such as the log-Euclidean metric"

### Questions
- What features of the graph does the kernel capture? For other kernels, it is generally known what features they compute (e.g., shortest path lengths for the shortest path kernel). The cycle–edge incidence matrix captures the edges from the minimal spanning tree which, together with the edge that does not belong to the tree, form a cycle. Can you provide more details about those features and why they are important?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The Tropical Torelli Map provides the definitive algorithm for transforming a metric graph into a Symmetric Positive Definite (SPD) matrix to create graph kernels that are uniquely invariant under edge subdivision. This method successfully embeds the graph's intrinsic geometric and topological properties into the matrix space, which is essential for metric graph comparison. The authors showed the advantages of this embedding in several setups.

### Strengths
A key strength is the new usage of the Tropical Torelli Map to create metric graph kernels, which are fundamentally different from conventional methods because they are purely based on the geometry and topology of the underlying metric space.
This construction results in kernels that are invariant under edge subdivision. This new geometric approach establishes the first framework specifically designed to compare and classify metric graphs based on their intrinsic structure.

### Weaknesses
The two main weaknesses I see are 1. concern the novelty of the underlying mathematical concepts and 2. the availability of robust, large-scale experiments. While the application to graph kernels for metric graphs is new, the core components are built on established fields: the embedding targets a space of SPD (Symmetric Positive Definite) matrices, which is a heavily studied manifold in information geometry. Similarly, the distance function used for the Tropical Torelli-Wasserstein (TTW) kernel, the Bures-Wasserstein distance, is a known geometric metric on this space. Therefore, the paper's novelty rests primarily on merging existing mathematical tools from tropical geometry and information geometry, rather than inventing entirely new core components or distance metrics. This is indeed a worthy result but I believe that showing its usefulness beyond toy problems is mendatory.

### Questions
1. Please show advantages on real world problem that allow us to compare the advantages of the proposed method
2. Elaborate on the weaknesses of this approach versus other kernels. Hopefully with examples.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes two new graph kernels TTE and TTW, based on the tropical Torelli map from tropical geometry. The map converts a metric (i.e., weighted) graph into a symmetric positive-definite matrix that captures its geometric and topological structure. The kernels are invariant under edge subdivision, making them suitable for comparing metric graphs that represent the same underlying space. The authors provide theoretical proofs, computational analysis, and experiments on synthetic graphs, standard benchmarks, and urban road networks. Results show strong performance, especially on sparse datasets.

### Strengths
- The idea of linking tropical geometry and graph kernels is original and well-grounded mathematically.

- Theoretical properties, including invariance under refinements, are clearly presented and proven.

- The algorithmic formulation is concrete and supported by detailed complexity analysis.

- Experiments are broad and carefully executed, with strong performance in label-free settings.

### Weaknesses
- There is no comparison with the Weisfeiler–Lehman (WL) kernel, which is the main baseline in graph kernel research. For example, Wasserstein WL (WWL) (Togninalli et al., NeurIPS 2019) would make the results more complete.
- Runtime grows fast with graph density; scalability to large or dense graphs remains unclear.

Togninalli M., Ghisu E., Llinares-López F., Rieck B., and Borgwardt K. M. (2019). Wasserstein Weisfeiler–Lehman graph kernels. NeurIPS 2019.

### Questions
Are there other real-world cases (besides road networks) where subdivision invariance improves performance or interpretability?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new class of graph kernels for metric graphs based on the tropical Torelli map from tropical algebraic geometry. The key idea is to represent a metric graph as a symmetric positive definite (SPD) matrix capturing its cycle structure and edge lengths. This mapping is invariant under edge subdivision, ensuring that the kernel depends only on the intrinsic geometry of the metric graph. Once each graph is mapped to an SPD matrix , two kernels are defined: 1. Tropical Torelli–Euclidean (TTE) -- a Gaussian RBF using Frobenius distance between matrices, adn  2. Tropical Torelli–Wasserstein (TTW) -- a Gaussian RBF using the Bures–Wasserstein distance. The paper provides an algorithm for computing the tropical Torelli matrix, proves its invariance under graph refinements, analyzes complexity in terms of graph genus, and evaluates both kernels on benchmark and real-world datasets. Empirically, TTE and TTW outperform existing label-free graph kernels on standard benchmarks and road-network classification tasks.

### Strengths
* Novelty and rigor: Introduces a kernel framework grounded in tropical geometry. Bridges algebraic topology (graph homology) and SPD information geometry. The paper rigorously shows that the tropical Torelli map yields a unique and refinement-invariant SPD representation for graphs with generic edge lengths.
* Empirical performance: Across 23 benchmark datasets, TTE and TTW match or slightly outperform existing unlabeled kernels (typically by 2–8 pp). Results on urban road network (URN) classification reach 80–94% accuracy and demonstrate practical utility.
* Efficient for sparse graphs. For graphs of low genus, runtime grows roughly linearly with the number of nodes, and the Euclidean version (TTE) scales well in practice.
* Clear exposition: The paper is well written and easy to read.

### Weaknesses
* Limited theoretical depth as a kernel paper: Beyond the refinement-invariance theorem, the work does not establish standard kernel properties (e.g., conditional positive definiteness, universality, or injectivity). The kernels are defined rather than theoretically characterized.
*  Modest empirical gains: The improvements over strong label-free baselines (Graphlet, Shortest Path, k-Core) are consistent but small. Results are competitive rather than outstanding.
* Scalability constraints: The method is only efficient on sparse graphs.
* Strictly label-free formulation: The framework cannot incorporate node or edge attributes, limiting its applicability to real-world graph learning tasks where features are central.
* Incremental novelty: While the use of tropical geometry is original, the kernel design (Euclidean and Wasserstein RBFs on SPD matrices) relies on well-known metrics from information geometry. The empirical results are in the same ballpark as existing methods.

### Questions
1. Can you formalize any properties of the TTW kernel (e.g., conditional positive definiteness or RKKS characterization) rather than relying on empirical stability?
2. Is there a principled way to extend your construction to labeled or attributed graphs?
3. How sensitive is the kernel to the genus truncation or subsampling of SPD matrices when graphs differ significantly in size?
4. Would a Power–Euclidean or Log–Euclidean variant offer a better tradeoff between invariance and scalability?

### Soundness
3

### Presentation
3

### Contribution
2
