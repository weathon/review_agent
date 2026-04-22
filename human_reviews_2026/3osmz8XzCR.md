# Do you know what k-means? Clustering with constant number of samples

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Clustering is one of the most important tools for analysis of large datasets, and perhaps the most popular clustering algorithm is Lloyd's algorithm for $k$-means.
This algorithm takes $n$ vectors $V=[v_1,\dots,v_n]\in\mathbb{R}^{d\times n}$ and outputs $k$ centroids $c_1,\dots,c_k\in\mathbb{R}^d$; these partition the vectors into clusters based on which centroid is closest to a particular vector. We present a classical $\varepsilon$-$k$-means algorithm that performs an approximate version of one iteration of Lloyd's algorithm with time complexity $\widetilde{O}\big(\frac{\|V\|_F^2}{n}\frac{k^{2}d}{\varepsilon^2}(k + \log{n})\big)$, exponentially improving the dependence on the data size $n$ and matching that of the "$q$-means" quantum algorithm originally proposed by Kerenidis, Landman, Luongo, and Prakash (NeurIPS'19). Moreover, we propose an improved $q$-means quantum algorithm with time complexity $\widetilde{O}\big(\frac{\|V\|_F}{\sqrt{n}}\frac{k^{3/2}d}{\varepsilon}(\sqrt{k}+\sqrt{d})(\sqrt{k} + \log{n})\big)$ that quadratically improves the runtime of our classical $\varepsilon$-$k$-means algorithm in several parameters.
Our quantum algorithm does not rely on quantum linear algebra primitives of prior work, but instead only uses QRAM to prepare simple states based on the current iteration's clusters and multivariate quantum mean estimation. Our upper bounds are complemented with classical and quantum query lower bounds, showing that our algorithms are optimal in most parameters.
Finally, we conduct numerical experiments that evidence the substantially improved runtime our classical algorithm over the standard Lloyd's algorithm, thus being one of the first cases of a practical dequantised algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ε-k-means, a classical approximation of Lloyd’s k-means algorithm that achieves exponential improvements in runtime dependence on data size, matching the efficiency of the q-means quantum algorithm by Kerenidis et al. (NeurIPS 2019). It also proposes an enhanced q-means algorithm that achieves a quadratic speedup over the new ε-k-means in several parameters. The quantum version relies only on QRAM-based state preparation rather than full quantum linear algebra primitives.

### Strengths
The paper addresses an interesting and worthwhile topic. The reasoning and justification behind the proposed approach appear well-founded and convincing.

### Weaknesses
The paper is difficult to follow due to its dense mathematical notation. The experimental evaluation is limited, as the proposed algorithm is compared to the original version only on synthetic datasets with fixed numbers of features and clusters. These parameters should have been varied to thoroughly assess effectiveness. Additionally, many standard synthetic datasets designed for clustering evaluation are omitted. The evaluation relies solely on RSS and runtime, neglecting other important metrics such as silhouette score, cluster sizes correlation, or external measures like Adjusted Rand Index (ARI) and Normalized Adjusted Rand Index (NARI) when ground truth labels are available (or against the original k-means result). Since each metric captures different aspects of clustering quality, this narrow evaluation is insufficient. Moreover, experiments involving quantum implementations are entirely missing. Finally, the paper omits relevant related work, including Poggiali, A., Berti, A., Bernasconi, A., Del Corso, G. M., & Guidotti, R. (2024). Quantum clustering with k-means: A hybrid approach. Theoretical Computer Science, 992, 114466.

### Questions
Questions can be derived from the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
I am not familiar with quantum algorithms, so this review will only cover the classical results presented in the paper.

This paper presents a quantum-inspired algorithm for k-means. The authors claim that the running time of their algorithm achieves an exponential improvement in the dependence in n (number of data points) compared to classical k-means. They complement this with lower bounds claiming optimality for most parameters. Finally they experimentally compare their algorithm to the classical k-means algorithm.

### Strengths
The paper’s main strength is the introduction of another quantum-inspired algorithm, which enriches this promising research area.

### Weaknesses
There are several critical issues with this paper.

The running times as stated require a preprocessing step that requires $\tilde{O}(nd)$ time. So there is no “exponential speed up”.

The algorithm is essentially a mini-batch style algorithm, but there is no discussion of existing mini-batch approaches.

The experiments are lacking in several ways:
1) The paper only considers synthetic datasets - it should consider real world datasets such as mnist etc…
2) The paper only reports residual sum of squares - it should report ARI and NMI
3) The algorithm should be benchmarked against mini-batch k-means at the very least. Other approaches such as coresets would also be nice. 
4) No code is provided

### Questions
See Weaknesses. Additional questions:
- The authors consider a version of k-means which converges if the movement of the centers falls below a certain threshold $\tau$. This is fine, but shouldn’t the parameter $\tau$ appear somewhere in the running time? Do your results work without this assumption?
- How does scaling the input vectors affect your running times?

### Soundness
2

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
2

### Summary
This paper presents a classical $\varepsilon$-$k$-means (EKMeans)
algorithm that performs an exponential reduction in dependence on
data size $n$ while maintaining comparable clustering quality. The
algorithm estimates cluster sizes and centroids by drawing only a
constant number of sampled data points in each iteration, making the
time complexity of each iteration almost independent of $n$. In addition,
this paper presents an improved $q$-means quantum algorithm that
quadratically improves the runtime of EKMeans algorithm in several
parameters. This algorithm avoids complex quantum linear-algebra operations,
relying solely on QRAM access and quantum mean estimation to efficiently
update cluster centers. Numerical experiments show that EKMeans achieves
a significant speedup over the standard Lloyd's algorithm
on large-scale datasets while maintaining stable clustering quality.
This research is the first time to demonstrate. how the core ideas of quantum algorithms
can be transformed into efficient classical algorithms through dequantization.

### Strengths
1. This paper proposes a constant-sample, approximate $k$-means algorithm
(EKMeans) that maintains clustering quality while making the iteration
time almost independent of the data size $n$; and it also proposes
an improved quantum algorithm, forming a theoretical system that mutually
reinforces classical and quantum approaches.

2. $k$-means is one of the most commonly used clustering algorithms, so the related research has both theoretical and practical values.

3. The EKMeans algorithm in this paper provides a constant-time iterative
k-means version that can be implemented on conventional hardware without
relying on quantum hardware. It serves as a key example bridging QML
and classical randomized algorithms, deepening the understanding of
the boundaries between classical and quantum computing capabilities.

4. This paper is well structured and logically rigorous. It clearly defines
the problem and its background.

### Weaknesses
1. All experimental designs are based on synthetic data for comparison. Why not use real-world datasets?

2. The figures can be improved. E.g., the color scheme of the curves does not adequately distinguish certain parameter settings, such as  $\varepsilon=0.0$ vs. $\varepsilon=0.2$. 

3. The title of the paper could be improved.

### Questions
1. In the experitmental section,  the authors use only synthetic data. What is the reason for not using real-world datasets?

2. Each iteration of k-means modifies the cluster partition (assigning different points to different clusters). Theoretically, each resampling iteration should reflect the latest cluster structure changes. However, if the same batch of P and Q is consistently used, these samples may primarily represent the structure at the initial partition. As the centroids gradually shift and boundaries adjust, the sampled points may no longer accurately represent the current cluster distribution, and the final updated centroids may be slow to react to the  real changes in some clusters. The algorithm may prematurely fall into a  locally stable solution (appearing convergent but with biases), especially when the initial sampling is not uniform or the cluster
distribution is complex. Do you have any theoretical guarantee or analysis against this issue?

### Soundness
2

### Presentation
2

### Contribution
3
