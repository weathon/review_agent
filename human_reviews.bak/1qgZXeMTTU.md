# Coreset Spectral Clustering

- Decision: Accept (Poster)
- Scores: 3, 6, 8, 10, 6

## Abstract
Coresets have become an invaluable tool for solving $k$-means and kernel $k$-means clustering problems on large datasets with small numbers of clusters. On the other hand, spectral clustering works well on sparse graphs and has recently been extended to scale efficiently to large numbers of clusters. We exploit the connection between kernel $k$-means and the normalised cut problem to combine the benefits of both. Our main result is a coreset spectral clustering algorithm for graphs that clusters a coreset graph to infer a good labelling of the original graph. We prove that an $\alpha$-approximation for the normalised cut problem on the coreset graph is an $O(\alpha)$-approximation on the original. We also improve the running time of the state-of-the-art coreset algorithm for kernel $k$-means on sparse kernels, from $\tilde{O}(nk)$ to $\tilde{O}(n\cdot \min (k, d_{avg}))$, where $d_{avg}$ is the average number of non-zero entries in each row of the $n\times n$ kernel matrix. Our experiments confirm our coreset algorithm is asymptotically faster on large real-world graphs with many clusters, and show that our clustering algorithm overcomes the main challenge faced by coreset kernel $k$-means on sparse kernels which is getting stuck in local optima.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a refined algorithm of constructing a coreset for kernel k-means problem. They improves the time complexity from $\tilde{O}(nk)$ [Jiang et al. ML' 24] to $\tilde{O}(nd_{avg})$, where $d_{avg}$ is the average number of neighbors of a single vertex on the graph defined by the given similarity matrix. They also showed how to use their technique to improve spectral clustering and obtain a approximate solution for normalized cut problem. The experiments are designed to support their theoretical results.

### Strengths
The proposed technique of constructing a coreset is quite useful when k is large and the similarity is sufficiently sparse.

### Weaknesses
1. Limited contribution. The proposed method highly depends on the former work [Jiang et al. ML' 24]. And their claimed improvements seems trivial. Theorem 1 is also easy to obtain. 
2. This paper assumes that the similarity matrix is sparse, which means a vertex has only few neighbors. So when a vertex is sampled, only its neighbors ($d_{avg}$ neighbors on average) need to update the their distance to the sampled set. Therefore, the time complexity of $\tilde{O}(nd_{avg})$ is straightforward.
3. The experimental on the Appendix A seems not ideal. For example, in Figure 5,6,7, the proposed method does not obtain the best ARI; Figure 7 also shows that the green baseline is actually faster. And there is no explanation for that.

### Questions
1. In the experimental part, the ARI performance of yours is much better than the green baseline (which is the method of [Jiang et al. ML' 24]). But I think your result is mainly based on the green baseline and you improve their running time. So it makes sense that your method is faster. But why your ARI is so much better than the green baseline?
2. In the experimental part, you mention that you use the nearest neighbor graphs of MNIST. How to construct such a graph on MNIST? Is it a nearest neighbor graphs based on Euclidean distance?

### Soundness
2

### Presentation
3

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
The paper tackles the challenges of clustering large, sparse datasets, where traditional spectral clustering methods can be computationally demanding. While spectral clustering is widely used for identifying non-linear cluster boundaries, its dependence on dense similarity matrices restricts scalability, particularly when dealing with numerous clusters. The authors introduce Coreset Spectral Clustering (CSC), a method that merges the efficiency of coreset sampling with the accuracy of spectral clustering, achieving a substantial speedup while maintaining clustering precision.

### Strengths
- CSC is optimized for sparse graphs, where the sparsity structure significantly reduces both computation and memory usage. By using a small, representative subset of data (the coreset), CSC scales well with data size and can handle graphs with millions of nodes and thousands of clusters. This scalability makes CSC suitable for large datasets in social networks, biological clustering, and sensor network analysis, where traditional methods would struggle.

- Standard spectral clustering can become infeasible with large, dense similarity matrices due to the high demands on computation and memory. CSC addresses this by working with a sparse kernel matrix and clustering only on the coreset, significantly reducing matrix size and computational cost. This efficiency enables CSC to process large datasets on standard hardware, which would otherwise require extensive resources for traditional spectral clustering.

- A smaller coreset speeds up computation, while a larger coreset captures more nuances in the data structure. This adaptability is useful for applications with specific accuracy or runtime needs, making CSC versatile across different types of data and clustering goals.

### Weaknesses
- The accuracy of CSC’s clustering largely depends on the representativeness of the coreset. To achieve high-quality clusters, the coreset need to accurately capture key structural and distributional aspects of the dataset. In datasets with uneven distributions or subtle data patterns, it could be difficult to create a coreset that fully represents the original data, and even minor inaccuracies could impact clustering results.

- CSC relies on an initial similarity or nearest-neighbor graph, and parameters such as the number of neighbors (k) or distance threshold (ϵ) can significantly affect clustering performance. Choosing suboptimal values for these parameters may lead to an inaccurate initial graph structure, impacting the quality of the final clusters.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an algorithm coreset spectral clustering algorithm for k-means clustering. This is done by first converting the input graph into a k-means problem instance, constructing and $\epsilon$-coreset for this instance, then solving the spectral clustering problem on the reduced graph. A second contribution is an algorithm for fast $D^2$-sampling utilized in coreset construction, which results in an coreset construction algorithm with running time $\widetilde{O}(n d_{avg})$.

### Strengths
- The contribution of the paper is solid, with the main idea being combining the approaches of coreset construction and spectral clustering. The utilization of sparsity to improve the running time of clustering algorithm is also well-executed.
- The presentation is overall excellent, with all of the contributions stated clearly. Schemes and easy-to-read pseudocode are very helpful with understanding the approach.
- The experimental section is detailed and well-organised.

### Weaknesses
It is somewhat unclear how often it is desired to solve spectral clustering on sparse data, or whether settings of interest have $d_{avg} < k$. I would like the authors to add an overview on how clustering methods are used in the empirical research, for example social network analysis, in the introduction or related work.

### Questions
Please address the concern that I raised in the weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper leverages the equivalence between kernel kmeans and spectral
clustering to improve spectral clustering. As a secondary result they also
improve coreset construction for sparse matrices.

The equivalence between kernel kmeans and spectral clustering is well known.
It is, therefore, natural to expect improvements in kernel kmeans
algorithms to produce better spectral clustering. Results along this line
were recently described by Jiang24, performing kernel kmeans on
weighted sampled data (coreset).

The paper argues that improving coresets do not necessarily lead to
improved spectral clustering because the kernel kmeans typically gets
stuck in a local minimum. By contrast, spectral clustering computes
an approximation to the global optimum, and does not gets stuck in local
minima.

Using this key observation the authors propose a novel framework of going
back and forth between 
the graph and the points in high dimensional space that are represented
by the coreset. This improves the speed, but not the quality of the clustering 
(as measured by NCUT). The paper shows that the reduction in quality
is linear.

### Strengths
The paper is very nicely written. It describes a result that appear interesting
in theory and useful in practice.

### Weaknesses
An important side result is the fast construction of a coreset that can be used
for kernel k-means clustering. The improvement comes from a fast
$D^2$ sampling technique. I believe that there are other, competitive
fast sampling techniques and I was missing a comparison.
Here is an example:

Chib and Greenberg, 1995, Understanding the Metropolis-Hastings algorithm,
The American Statistician.


In addition please see the questions below.

### Questions
The result: Why are the derivation and experiments discussing only NCUT?
The equivalence of Dhillon04 was extended in Dhillon07 to other criteria,
in particular RatioCut. It should also apply to the newer stochastic box model.


Experimental results: why is there no comparison of the NCUT values
that were obtained in the experiments? The current evaluation is
in terms of ARI, but this is not what the algorithms attempt to maximize.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper develops new tools in coreset construction merging ideas from two different problems: one is the coresets for k-means and kernel k-means clsutering and the other is spectral clustering, hence the name Coreset Spectral Clustering of the paper.

The main result is to give an approximation algorithm for the problem of normalized cut based on coresets. Specifically, they can approximately solve the problem on the coreset graph and prove that this is enough to get a reasonable approximation on the original input graph. The authors also perform experiments and demonstrate that their approach leads to asymptotically faster results on large real-world graphs with many clusters beating prior coreset kernel k-means approaches for sparse kernels.

The second result of the paper is to speed up the running time of the current state-of-the-art coreset algorithm for the problem of kernel k-means on sparse kernels, where the speed up depends on the average degree of the graph.

### Strengths
-nice idea to rely on kernel sparsity that yields the first coreset construction for kernel spaces and leads to speed up which is especially useful for large graphs with many clusters.

-the two main protagonists here which are spectral clustering and kernel k-means are studied often separately, and I view this approach of merging ideas/techniques interesting.

-the coreset spectral clustering algorithm is interesting and gives a clean result statement: an \alpha-approximation of the normalized cut problem on the coreset graph will  in fact give an O(\alpha)-approximation of the normalized cut problem on the original graph. To me this is a very useful and interesting statement as it can be used as a black box and lead to practical results as well.

### Weaknesses
-novely: while the paper draws inspiration and combines cleverly prior works on normalized cut, kernel clustering and coresets, I wanted to point out that the current paper seems to heavily rely on ideas and techniques that were developed before. Of course the authors had to cleverly combine them in order to get the clean statement as their main result. I also read parts of the technical proofs in the appendix, and I believe that in terms of techniques the paper is a bit weak. Perhaps the authors could elaborate on what crucial ideas in terms of techniques were the novel aspects of this work. Specifically the analysis of Jiang et al. seems to be doing the heavy lifting in many parts of the paper, and conditioned on that  paper, I believe  the current technical contribution appears to be slightly less solid. This is my only concern about the paper, otherwise I do like the paper.

### Questions
-The authors say their speed up is from nk to nd where d is the average degree in some sense. In the abstract this is a bit confusing: but is this necessarily a speedup; what if k is relatively small but the average degree in the kernel matrix leads to more non-zero entries? Perhaps this is good to clarify early on as you do later in the main body cause the reader might be confused.

-While reading the paper, many ideas used in the analysis are actually coming from (and are cited) from the prior work by Jiang et al. I was curious if the authors could elaborate on what ideas were the novel part of the paper?

### Soundness
3

### Presentation
3

### Contribution
3
