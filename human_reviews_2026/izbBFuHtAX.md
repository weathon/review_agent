# Internal Evaluation of Density-Based Clusterings with Noise

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 2, 4, 4

## Abstract
Evaluating the quality of a clustering result without access to ground truth labels is fundamental for research in data mining.
However, most cluster validation indices (CVIs) do not consider the noise assignments by density-based clustering methods like DBSCAN or HDBSCAN, even though the ability to correctly determine noise is paramount to successful clustering. 
In this paper, we propose DISCO, a **D**ensity-based **I**nternal **S**core for **C**lusterings with n**O**ise, the first CVI to explicitly assess the *quality* of noise assignments rather than merely counting them.
DISCO is based on the Silhouette Coefficient, but adopts density-connectivity to evaluate clusters of arbitrary shapes, and proposes explicit noise evaluation: it rewards correctly assigned noise labels and penalizes noise labels where a cluster label would have been more appropriate.
The pointwise definition of DISCO allows for the seamless integration of noise evaluation into the final clustering evaluation, while also enabling explainable evaluations of the clustered data.
In contrast to most state-of-the-art methods, DISCO is well-defined and also covers edge cases that regularly appear as output from clustering algorithms, such as singleton clusters or a single cluster plus noise.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DISCO, a new internal cluster validity index (CVI) designed to evaluate density-based clusterings (like DBSCAN or HDBSCAN), especially when noise points are present and when clusters have irregular (non-spherical) shapes. The authors tested the new CVI on 20+ synthetic and real-world datasets, including 2D toy data (rings, spirals) and high-dimensional sets (COIL20, Pendigits). The paper also shows that DISCO is robust to parameter changes, computationally efficient (O(n²)), and capable of evaluating all edge cases including single-cluster or all-noise situations.

While the paper makes a novel contribution, its main weaknesses lie in computational scalability, limited generality beyond density-based clustering, limited comparison with other CVIs, and modest theoretical depth, leaving room for future improvements and broader extensions.

### Strengths
1. The method provides a pointwise evaluation that treats both cluster points and noise points consistently, allowing fine-grained interpretability and explainable assessment of individual point assignments.

2. DISCO remains well-defined even for special scenarios, such as having only one cluster, only noise, or singleton clusters—cases that typically break other indices.

3. Its output is normalized between −1 and 1, making scores easy to interpret and compare across datasets and algorithms.

4. The authors conduct extensive experiments across many benchmark and real-world datasets, demonstrating that DISCO consistently aligns with external metrics (like ARI) and correctly identifies optimal clustering results.

### Weaknesses
1. Although comparable to other density-based CVIs, DISCO still requires quadratic time with respect to the number of data points, which may become computationally expensive for very large datasets (e.g., millions of points), limiting scalability.

2. While the authors claim robustness to the hyperparameter μ, the paper provides limited theoretical justification or adaptive mechanism for choosing it automatically; its effect is shown empirically but not deeply analyzed.

3. The experiments focus on accuracy and robustness but lack detailed runtime or memory usage comparisons against lightweight metrics like the Silhouette or Davies–Bouldin indices, which could highlight performance trade-offs more clearly.

4. Although code is shared, I have encountered difficulties when reproducing the experiments in "DISCO-E358/src/Experiments" folder.

5. The paper does not compare with other powerful CVIs like MMJ-SC and MMJ-CH. See:

 https://arxiv.org/abs/2301.05994
 
Python code of MMJ-SC and MMJ-CH can be found at:

https://github.com/mike-liuliu/Min-Max-Jump-distance/blob/main/test%20MMJ-based%20%20Silhouette%20coefficient%20(MMJ-SC).ipynb


https://github.com/mike-liuliu/Min-Max-Jump-distance/blob/main/test%20MMJ-based%20Calinski-Harabasz%20index%20(MMJ-CH).ipynb

### Questions
I have encountered difficulties when reproducing the experiments in "DISCO-E358/src/Experiments" folder. Could you provide a jupyter notebook file to reproduce the values in Figure 1? It is preferable to include all the necessary code in one or two files. E.g., using one file to define all the necessary functions and classes. It is a pain to check different functions and classes in multiple files.

You can provide the jupyter notebook file in an anonymous URL.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces DISCO, a novel internal cluster validity index (CVI) designed for evaluating density-based clusterings that include noise points. The authors identify a critical gap in existing internal CVIs: most either ignore noise points or penalize their mere presence without assessing the quality of their assignment. DISCO addresses this by extending the concept of the Silhouette Coefficient to use density-connectivity distance, making it suitable for arbitrary-shaped clusters, and by introducing a pointwise noise evaluation mechanism that rewards correct noise labels and penalizes incorrect ones. Through extensive experiments, the paper demonstrates that DISCO outperforms existing methods in evaluating density-based clusterings, selecting optimal parameters, and aligning with external validation measures, while also being deterministic and handling edge cases robustly.

### Strengths
1. The paper tackles a long-overlooked problem in clustering validation: evaluating the quality of noise assignments. DISCO is the first internal CVI to explicitly assess whether a noise label is appropriate, moving beyond simple counting.
2. The experimental section is thorough and systematic. The authors not only compare DISCO against a wide range of baselines on standard tasks like parameter selection but also design specific experiments to highlight its unique capability in noise assessment.
3. The identification and empirical demonstration of non-determinism in DBCV is a valuable finding for the community.

### Weaknesses
1. While the O(n²) time complexity is stated and argued to be comparable to other density-based CVIs like DBCV, no actual runtime comparisons are provided.
2. The evaluation of DISCO's noise-handling capability relies heavily on well-structured synthetic datasets. While effective for proof-of-concept, demonstrating its performance on real-world datasets with complex, real noise would greatly enhance the generalizability and impact of the claims.
3. Parameter μ is not fully discussed.

### Questions
1. Given the O(n²) complexity, what are the practical limits of DISCO on large-scale datasets? Do the authors have plans or can they discuss potential strategies for approximate computation of the dc-dist or MST to improve scalability? Could you add a runtime comparison table against key competitors like DBCV and LCCV?
2. Could the authors design a supplementary experiment that quantifies the risk posed by DBCV's non-determinism in a model selection scenario? For instance, when comparing two clusterings, how frequently does the random variation in DBCV's output lead to the selection of the objectively worse clustering?

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
4

### Summary
This paper introduces DICSCO (Density-based Internal Score for Clustering with Noise), a new validation metric designed to assess the quality of noise assignments in density-based clustering. The work addresses the limitation in existing cluster validation indices (CVIs) with improved noise condition modeling. Results indicate that DICSCO can better represent clustering quality in some cases presented.

### Strengths
1. The evaluation of density-based clustering quality with noise is an important topic.
2. The paper is clearly presented and well-organized. The concept of “bad noise examples” is effectively illustrated with a toy example
3. The experimental results show its improvements over the baselines.

### Weaknesses
1. The method only considers the uniformly distributed noises, which is limited in real-world situations.
2. The evaluation uses a few 2D synthetic datasets except for COIL20 and Pendigits. The few examples feel arbitrary and it is unclear how results will change if layouts are changed. While the method performs well in this setting, including more real-world or high-dimensional datasets will help to see if the proposed CVI performs consistently beyond artificial 2D cases. For example, more existing datasets with labels can be used to evaluate the scoring quality by comparing to ground truth.
3. The comparison uses some very bad clustering results to show the strength. But because the paper focuses on density-based clustering, it is unclear if those "very bad" clustering results  are useful for experiments as methods like HDBSCAN likely won't generate those.

### Questions
How well can the method handle other noise distributions?

### Soundness
3

### Presentation
3

### Contribution
2
