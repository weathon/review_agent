# AuToMATo: An Out-Of-The-Box Persistence-Based Clustering Algorithm

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 6

## Abstract
We present AuToMATo, a novel clustering algorithm based on persistent homology. While AuToMATo is not parameter-free per se, we provide default choices for its parameters that make it into an out-of-the-box clustering algorithm that performs well across the board. AuToMATo combines the existing ToMATo clustering algorithm with a bootstrapping procedure in order to separate significant peaks of an estimated density function from non-significant ones. We perform a thorough comparison of AuToMATo (with its parameters fixed to their defaults) against many other state-of-the-art clustering algorithms. We find not only that AuToMATo compares favorably against parameter-free clustering algorithms, but in many instances also significantly outperforms even the best selection of parameters for other algorithms. AuToMATo is motivated by applications in topological data analysis, in particular the Mapper algorithm, where it is desirable to work with a clustering algorithm that does not need tuning of its parameters. Indeed, we provide evidence that AuToMATo performs well when used with Mapper. Finally, we provide an open-source implementation of AuToMATo in Python that is fully compatible with the standard scikit-learn architecture.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies the ToMATo clustering algorithm (Chazal et al. 2013) in high dimensions.
ToMATo input includes a neighbor graph (e.g. kNN graph), an estimated density for each point derived from the neighbor graph, and a threshold parameter \tau to decide merging clustering.
In principle, ToMATo first forms several clusters by connecting points with high density to its neighbor in lower density.
After that ToMATo constructs a persistence diagram which measures the size of the formed clusters (via the prominence notion reflecting number of points in the same high density region).
Using the persistent diagaram and the parameter \tau, ToMATo can provide a hierarchical clustering structure, which is similar to HDBSCAN [a].

Since the setting of \tau is sensitive, Chazal et al. 2016 proposed to use boostrap, that samples several subset X' of X to estimate the relevant value for \tau.

This works implements ToMATo algorithm with the boostrap method, and carries out experiments to demonstrate the performance of ToMATo compared to other (mainly density-based) clustering algorithms.

[a] Density-Based Clustering Based on Hierarchical Density Estimates - PKDD 2013

### Strengths
- The paper presents an implementation of a hierarchical clustering ToMATo. 
- Some experiments and ablation study on ToMATo with Mapper, that approximates the Reeb graph of a manifold based on the sampled points, show some potentials of the implementation.

### Weaknesses
I feel that the novelty of the work is limited in the sense that the paper implements a clustering algorithm and presents some comparison with other clustering competitors. 
Regarding the clustering accuracy, the paper use FMI scores of clustering competitors though the improvement of ToMATo is quite maginal. It would be better to use other popular measures, including AMI or NMI, since these measures are less sensitive to different number of clusters and cluster sizes.  Also, there should be the reported running time of different clustering algorithms.

The main contribution of AutoToMATo (an implementation of Tomato with boostrap) is to replace the parameter \tau by the parameter \alpha which is easier to set up in several data sets. Since the foundation of such bootstrap theory was proposed in Chazal et al. 2016, this limits the contribution of the work.

It seems that the running time complexity in Line 263 does not consider the graph construction, which require O(n^2) time in high dimensional space. Will the graph construction complexity be part of time complexity of the algorithm?

Some typos:
- Line 035: a parameters
- Line 11: must clear

### Questions
Q1) Could the author clarify further regarding
- The sensitity of parameter of neighbor graph (e.g k or \delta) regarding the setting of parameter \tau.
- The size and dimensionality of data sets

Q2) Are there any novel aspects or improvements in the implementation of AutoMATo that could be highlighted?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces a nice clustering algorithm based on density and persistence. It uses similar concepts as HDBSCAN and Density peaks and works with a set of predefined default parameters.

### Strengths
S1) Easy to follow, written clearly. 
S2) Good idea and important concepts that are used
S3) Good reasoning and background information behind the choices (for experiments as well as design-choices in the algorithm development)

### Weaknesses
W1) Experiments are not sufficient. 
a) Even though Fowlkes Mallows is a good evaluation measure, state-of-the-art papers for clustering usually include the NMI and ARI values, so please include at least one of them additionally in the appendix.
b) Presentation of the experiments is hard to follow: In the main paper, only the average over all datasets is given, which is not enough to get an idea where the algorithm is good and where not. Instead of comparing AuToMATo with competitors individually, please give an overview of all methods, but for the individual datasets. 
c) Please provide an overview of the properties of the datasets. Which dimensionality, which size? 
d) Regarding Figure 2 : The worst clustering result that DBSCAN returns is completely irrelevant, leave it away. You can always find an epsilon or minpts such that all points are clustered together into one cluster.
e) Include an evaluation of noise/outlier labels given by your method or exclude it from the paper entirely. 
f) Do not set a seed for the experiments. That hinders evaluation of robustness. 
g) In line 440 you write DBSCAN "sometimes" outperforms AuToMATo, even though it does so for around a third of the datasets. Plus, the range of epsilon values is not chosen according to best practices (see, e.g., Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. ACM Transactions on Database Systems (TODS), 42(3), 1-21.) 

W2) Lack of novelty and discussion. Even though the method has some promising ideas, it is very similar to HDBSCAN and Density Peaks without discussing the differences enough or showing where exactly they make a difference in practice. Synthetic datasets could help to show these differences. Furthermore, AuToMATo does not significantly outperform any of the competitors. Please elaborate for which type of datasets one should use AuToMATo over existing methods. 

W3) The related work section is missing. Please add additionally to the background about ToMATo in Section 2 also background about other related methods. Elaborate on similarities, e.g., your concept of persistence is very similar to the stability used in HDBSCAN and the hill climbing approach is similar to density-peaks. 

W4) Presentation: Increase font size for Fig.2 . Instead of listing all the Tables in the appendix, make a visual overview where readers can compare all your competitors for the different datasets. 

Typos: line 060, 354/355, 420,

### Questions
See weak points. 

Q1) In which range are the dimensionalities of your tested datasets?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces AuToMATo, a persistence-based clustering algorithm based on the existing topological clustering approach, ToMATo. The methodology involves executing a bottleneck bootstrap on the persistence diagram generated by ToMATo to distinguish significant from non-significant peaks of the estimated density function. 
Although AuToMATo is not entirely parameter-free, the authors propose default settings with great performance without tuning the parameters. For evaluation, the algorithm was benchmarked using data sets from the Clustering Benchmarks suite, comparing it with DBSCAN, HDBSCAN, hierarchical clustering (employing Ward, single, complete, and average linkage strategies), the FINCH clustering algorithm, and a TTK-based algorithm stemming from the Topology ToolKit suite. Experiments showed that AuToMATo performs better than parameter-free clustering algorithms and other algorithms. The authors also presented an application of AuToMATo in combination with Mapper using synthetic two-dimensional data and the Miller-Reaven diabetes dataset.

### Strengths
The paper presents AuToMATo, a new and improved version of the topological clustering algorithm ToMATo.

The authors provide a rigorous explanation of the algorithm, including detailed mathematical definitions.

The paper is generally well-written.

AuToMATo achieves competitive performance without the need for manual parameter tuning.

### Weaknesses
The experiments only use datasets from the Clustering Benchmarks suite. Including more high-dimensional and real-world datasets would better evaluate the AuToMATo algorithm's scalability and performance.

The paper does not provide enough experiments and discussion comparing AuToMATo to other parameter-free clustering algorithms, which is necessary to demonstrate its effectiveness for this contribution.

->The paper does not sufficiently examine how changes in parameters affect the algorithm. While it's mentioned that the choice of alpha and B is justified by experiments, there should be a discussion that provides more evidence and reasoning. Including additional experiments that demonstrate how different values of alpha and B influence the clustering results would improve the discussion.

->Relying only on the Fowlkes-Mallows score is not sufficient. Including other evaluation metrics (internal and external) would provide a more complete evaluation and help compare with other algorithms.

-> The application of AuToMATo to Mapper is briefly mentioned but needs a more detailed explanation of the methods and the results obtained.

### Questions
page 5 -> Anonymized GitHub-link but no link. Where is the code?

Relying only on the Fowlkes-Mallows score is not sufficient. Including other evaluation metrics (internal as DB and external as NMI...) would provide a more complete evaluation and help compare with other algorithms.

In the context of the proposed approach, there have been no comparisons with non-parametric algorithms. Given the search for topological structure, I think it would be interesting to compare with topological algorithms such as SOM and spectral clustering.

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
3

### Summary
This paper considers and extends the ToMATo clustering algorithm, which finds clusters based on the persistent homology of a density estimate of the data. This paper proposes a novel method to automate the selection of the $\tau$ parameter of the ToMATo algorithm, giving a parameter-free, out-of-the-box clustering algorithm.

The new algorithm is based on the 'bottleneck bootstrap' process in order to build a persistence diagram in which the prominent connected components can be easily identified.

### Strengths
This paper is extremely clear, and presents a novel algorithm which successfully solves the problem of parameter selection in the ToMATo algorithm. The newly presented algorithm is easy to use and likely to be an effective drop-in replacement for the ToMATo algorithm.

### Weaknesses
The precise novel contribution of this paper is not completely clear. Given that the bottleneck bootstrap process was defined by Chazal et al. (2017), what is the key new insight that enables the AuToMATo algorithm to work? This should be made more clear in the write-up.

The datasets used for the experimental evaluation seem to be generally quite small (< 10000 data points), low dimensional synthetic datasets. It would be interesting to see a comparison on some larger, real world datasets. For example, the mnist dataset (which is one of the datasets in the benchmark used) seems to have been excluded.

Additionally, the running time of the algorithms are not reported in the experimental section. While I understand that there is often a trade-off between running time and performance, it would be interesting to see this trade-off explicitly discussed.

Finally, although the choice to use the Fowkes-Mellows score in the evaluation is justified in the paper, given that the Adjusted Rand Index and Normalised Mutual Information are very standard in the literature, in my view it would be better to compare on all metrics.

### Questions
* What is the key novel insight behind the AuToMATo algorithm?
* How does the empirical running time of AuToMATo compare with the alternative algorithms?
* Is there hope to scale AuToMATo to large, high-dimensional real-world datasets?

### Soundness
3

### Presentation
3

### Contribution
2
