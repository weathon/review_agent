# Active Probabilistic Clustering

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Active Constrained Clustering (ACC) is a widely used semi-supervised clustering framework to improve clustering quality through progressive annotation of informative pairwise constraints. However, the application of existing ACC methods to large datasets with numerous classes incurs high computational or query expenses. 
In this paper, we conduct a theoretical analysis of the inefficiency of sample-based ACC and the rationale behind cluster-based ACC. Moreover, we provide the theoretical guarantee for cluster fusion under a certain purity constraint and a clustering quality constraint with respect to normalized mutual information (NMI).
Drawing on these theoretical insights, we introduce a novel Active Probabilistic Clustering (APC) framework designed to scale effectively with large datasets. Compared to previous methods, APC demonstrates superior performance across eight datasets of varying sizes (ranging from 350 to 100,000 samples) in terms of clustering quality, query cost, and computational expense. Specifically, APC accomplishes satisfactory clustering outcomes (e.g., NMI $>0.95$) using 3,920 queries on a dataset with 100,000 samples, while baseline methods yield inferior clustering results (e.g., NMI $\leq0.85$) with 10,000 queries. Concurrently, APC operates at a speed 100x faster than baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies active constrained clustering, a semi-supervised learning problem in which the clustering method can query pairs of samples $(x_i, x_j)$ to learn if their underlying class labels $y_i, y_j$ are the same or different.  The goal seems to be to reduce the number of queries needed in order to learn the ground truth labels.

After developing a merging criterion based on normalized mutual information, a new active clustering method is proposed which the authors call active probabilistic clustering.  The idea is to construct appropriate sampling probabilities for pairs of clusters and to sample several pairs of clusters and query representatives from each cluster in order to make merging/splitting decisions.  Many experiments are carried out for this method, showing improvements to normalized mutual information and adjusted rand index using fewer queries than baseline methods.

### Strengths
Aside from the mathematics (see below), the writing and presentation is clear and seems to be well motivated.  The proposed method performs well in the experiments that were considered.

### Weaknesses
The problem formulation isn't clear at all aside from "the clustering method can query pairs of samples ...".  The actual objective and performance metrics are not clearly described or motivated in Sections 1 or 2.  I think it would help the reader to develop the objective for the problem more formally and contextualize prior work within this objective.

Additionally, without more care towards formalizing the problem setting and any mathematical models, I believe there are serious issues with how this paper is written.  In particular more care is needed to define the probability space and random variables that are being worked with to clarify the equations.  For example, Equation (2), seems to be wrong as written.  The LHS of this equation, ${\mathbb P}(w_i = w_j \mid w_i,w_j)$  has to either be 1 or 0, while the RHS does not.  The reason the LHS should be 1 or 0 is that once we've conditioned on the two random variables $w_i, w_j$, they are deterministically either the same or different.  If something else is meant, then this has to be clearly defined and communicated.  Similar issues hold for equations (3) and (4) as well.

### Questions
- Please clarify and motivate the objective (metrics) for the problem up front.
- Please clarify the terminology.  What is meant by splitting and clustering (e.g., in Theorem 2)?  The notations for these seem similar but I can't tell if splitting is just being used as a synonym for clustering or not.
- Please clarify what is meant by equations (2), (3), and (4).  What is the probability space, random variables, etc.?
- Please clarify the notation used.  What is meant by $w_j$?  Properly define entropy and mutual information, etc.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper suggests a query-based clustering algorithm. The queries are: "are two points in the same target cluster." Human intervention is used to be able to answer such queries. The goal is to get close to a target clustering while making a small number of such queries. The paper suggests ideas to start with clustering and improve it by using queries to fuse and split clusters. Certain theoretical conditions are given for effective cluster fusion, and experimental results are given to show the utility of the suggested clustering ideas.

### Strengths
Query-based clustering is a relevant topic in the theory of clustering.

### Weaknesses
- The paper is not easy to read and understand. Even though the paper's main contribution is an algorithm, a clear description of the algorithm is missing in the main write-up. Multiple aspects of the algorithm have been deferred to the Appendix, and the writeup keeps pointing to the Appendix. For instance, consider the description of Algorithm 1 -- Algorithms 2 and 3 are deferred to the Appendix without giving the intuition regarding what they do. It is unclear what "Implement Human Test on w_1 and w_2" means.
- Theorem 2 gives some conditions under which cluster fusion gives an improvement. Are there reasons to believe such conditions could hold in natural clustering settings? Can these conditions be tested? Do these conditions continue to hold after a sequence of fusion operations?
- Does the initial clustering algorithm (FPC) use any queries? Is there some reason to believe that the initial clustering has some correlation with the target clustering? If so, what is the correlation, and how does this impact the number of queries? If not, what does FPC help? what if the target clustering is an arbitrary partition of the dataset and has nothing to do with geometric clustering ideas that place closer points in the same cluster? If the target clustering is an arbitrary partition of the dataset, what are the number of queries required to cluster?

With a lack of discussion on various issues and a lack of clarity on the suggested algorithm, it is difficult to form an informed opinion about the paper. The write-up should be improved to enable a fair review of the paper.

### Questions
Some of the questions are mentioned in the weakness section.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the problem of active clustering. Pairs of points are given to a human to label whether or not the pair belongs to the same or different clusters. This paper uses an approximation to expected improvement of NMI to select pairs of clusters in the current clustering to use in the human query. Then representative samples from the pair of clusters is given to the human to label as same cluster or not. The human provided constraints are then incorporated by the clustering algorithm to improve the clustering. The empirical performance of the algorithm is shown on open source image datasets.

### Strengths
* The empirical results are very strong in comparison to the baselines
* The use of approximate expected NMI improvement to select queries for human labeling seems to be novel and intuitive.

### Weaknesses
* The writing needs some work in places. The authors use essential terminology and notation without definition: e.g. dominant class, purity, etc. There are a few minor typos as well throughout.
* There are some essential missing details of the method in Section 3. What exactly is the human answering in the Human Test? The lack of details in this section make it difficult to fully understand the proposed method
* It is unclear if the method is fair and could be applied in the real world. What information does the algorithm have access to? It seems like the method might have access to ground truth information and this is the reason it is performing so well.

### Questions
* Why is NMI of 0.95 enough? Is it possible in some applications that we would want say NMI of 1.0? Why not extend the results?
* What is a practical application in which we might want to utilize this method?
* Should the term *clustering* be used instead of *cluster* in Definition 2.1 and throughout the rest of the paper? 
* What is the clustering algorithm used? The authors state the they use FPC, but how does this algorithm work?
* How are the constraints enforced by the clustering algorithm over time?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
