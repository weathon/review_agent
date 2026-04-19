# Learning Graph Representations in Normed Spaces

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 3, 3

## Abstract
Theoretical results from discrete geometry suggest that normed spaces can abstractly embed finite metric spaces with surprisingly low theoretical bounds on distortion in low dimensions. 
In this paper, inspired by this theoretical insight, we propose normed spaces as a more flexible and computationally efficient alternative to several popular Riemannian manifolds for learning graph embeddings. 
Our normed space embeddings significantly outperform several popular manifolds on a large range of synthetic and real-world graph reconstruction benchmark datasets while requiring significantly fewer computational resources. 
We also empirically verify the superiority of normed space embeddings on growing families of graphs associated with negative, zero, and positive curvature, further reinforcing the flexibility of normed spaces in capturing diverse graph structures as graph sizes increase. 
Lastly, we demonstrate the utility of normed space embeddings on two applied graph embedding tasks, namely, link prediction and recommender systems. 
Our work highlights the potential of normed spaces for geometric graph representation learning, raises new research questions, and offers a valuable tool for experimental mathematics in the field of finite metric space embeddings.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose embedding using norm spaces. The authors also show the experimental results in many situations, both on synthetic and real datasets.

### Strengths
1. This paper provides many theoretical results about existing embedding in normed space.
1. This paper provides empirical results on many synthetic and real datasets of many types of embedding methods in Riemannian manifolds where the distance is given by geodesics and the normed space where the distance is given by a normed function.
1. The motivation for using the normed space is clearly stated in the paper.
1. The basic idea is simple and effective.

### Weaknesses
1. I do not see the novelty of the paper. Embedding in the normed space is not a new topic. Indeed, the authors themselves refer to many relevant papers in Section 3. Although the authors did not refer to papers in the machine learning community, the embedding in the normed space has been discussed even in the machine learning community e.g., [A]. Hence, the authors need to clarify the novelty of the paper.

1. While the motivation for using norm spaces is somewhat clear if we read Section 3, it is frustrating if we need to wait for it. It would be preferable to clarify what types of graphs are your paper's target in the Introduction section. In other words, it should be clarified in the introduction section for which graphs each of $l^1$ and $l^\infty$ is better than existing Riemannian manifold methods.

1. What is the benefit of the normed space embedding in applications? My concern is that although the normed space is simple in its formulation, it is not intuitive when we visualize the results even compared to those in hyperbolic space. The $l^1$ distance case might be acceptable since we could imagine Manhattan-like road structure to see the distance between two points when we see the visualization. However, I do not know how we visually understand them in the $l^\infty$, although we know that a ball is given as a hypercube there. 

1. Similar concerns related to the applications. I am concerned that it is not obvious whether we can connect the embedding results to another machine learning method since I do not know any machine learning methods that the input is the normed space, although we can forcibly input it as a vector. Graph reconstruction is a good benchmark but usually not an ultimate goal. Or if the graph reconstruction is an ultimate goal, we do not need to stick to embedding.

1. The objective function (1) is unfair. It gives strong advantages to the normed space. The objective function forces the ratio between the graph distance and the distance in the embedding space to be 1:1. In practice, it can be 1:c since we usually only care the relative distance. The norm space does not care about this difference since it is linear and we can scale any shape by any factor. Specifically, for a normed space $Y$ and for any embedding function $f: \\mathcal{V} \to Y$, we always have another embedding function $f\_c$ that scales the distances among the points $(f(v))\_{v \in \\mathcal{V}}$ with the factor $c$, i.e., the embedding function $f\_{c}$ such that $d\_{Y}(f\_{c}(u),f\_{c}(v)) = c d\_{Y}(f(u),f(v))$ for all $u,v \in \\mathcal{V}$. In general metric space, it does not hold. Rather, lacking this property gives some metric spaces attractive characteristics. The objective function (1) kills those attractive characteristics. This technical issue is the strongest concern to me.

[A] Atsushi Suzuki, Atsushi Nitanda, Taiji Suzuki, Jing Wang, Feng Tian, and Kenji Yamanishi.
"Tight and fast generalization error bound of graph embedding in metric space."
In Proceedings of the 40th International Conference on Machine Learning, 2023

### Questions
1. What is the novelty of the paper?
1. What is the usage of the embeddings in the normed space?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates embedding graphs into lower dimensions using norms, specifically the 1- and infinity- norms. Motivated by theoretical results that any metric space can be embedded into about O(logn) space with O(logn) distortion, the paper experimentally studies learned embeddings that minimize some average of (functions of) edge-wise distortions. Through fairly extensive experiments on well studied tasks, the paper demonstrates that this method can often give significant advantages.

### Strengths
Both the embedding goal, and the method for learning these embeddings, are natural. The experiments are extensive, and cover both important problems as well as important algorithms for doing predictions / recommendations. The results obtained also suggest that there are significant advantages to using these non-Euclidean norms to embed graphs.

### Weaknesses
My main concern is that the algorithms for using these embeddings are tailored to each of the applications, and doesn't seem to lead to a general purpose graph embedding algorithm. The algorithms also seem to assume that distances are the natural values to bound distortions against: I had trouble seeing why would the downstream tasks using the embeddings `pick up' the fact that the distances are encoded in 1-norm.

### Questions
None: I feel most of my concerns are outside of the scope of this work.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work the authors embed nodes of graphs in $\mathbb R^d$ and train these representations such that induced metric from $\ell^1$ or $\ell^\infty$ norms captures relevant graph structure. The first set of experiments is focused on graph reconstruction, wherein, for an undirected graph $G = (V, E)$ and nodes $a, b\in V$ the distance $\lVert f(a) - f(b) \rVert_{\ell^p}$ is trained to capture the shortest path distance in $G$. This is explored empirically on synthetic graphs including grids, trees, cartesian and rooted products of grids and trees, several expander graphs, as well as five real-world datasets. The authors evaluate the average distortion as well as mAP, and find that $\ell^\infty$ generally performs the best, and both $\ell^1$ and $\ell^\infty$ outperform other embedding spaces, including both Euclidean and non-Euclidean variants, such as hyperbolic and spherical spaces.

The authors also perform a set of experiments on link prediction, wherein $f$ is parameterized by a GNN architecture, and an edge exists between nodes which are sufficiently close, and conclude that $\ell^1$ performs best in this setting. Finally, the authors evaluate the normed spaces as embeddings for recommender systems, wherein the problem is formulated as binary classification of edges on a bipartite graph between users and items, finding that $\ell^\infty$ works best.

### Strengths
This paper's greatest strength is in the quality of exposition. Everything is explained quite clearly, with sufficient background information to be self-contained, clear motivation for the proposed experiments, and relevant aspects of those experiments highlighted in the discussion.

The proposed method is also exceedingly simple. The $\ell^1$ and $\ell^\infty$ norms are quite common and well understood by machine learning practitioners, and the authors do not seem to need any special training tricks to use the induced metric in their training, despite the fact that $\ell^1$ has a discontinuous gradient and $\ell^\infty$ has zero gradients in much of the admissible parameter space.

### Weaknesses
The greatest weakness of this paper is in the scale of the empirical evidence. Most of the graphs are <= 1000 nodes, and it is well-known by machine learning practitioners that results which hold at such small scales do not necessarily scale-up in a reasonable way, particularly those methods based on gradient-descent. For a specific example, the limited performance of methods in the small-data setting in of the experiments of Figure 2 are likely explained by training issues which are problematic for such small graphs but do not present issues for larger graphs. It is particularly surprising to me that the authors have focused on such small-scale data, as one of the intended selling points of their work is the computational efficiency of the induced distance metrics.

An additional weakness is the lack of theoretical statements related to the proposed metrics. The absence of such theoretical claims is not a deal-breaker for the paper, but this makes the limited scale of the empirical evidence all the more problematic, as theoretical results which might suggest (for example) greater representational capacity of the proposed metrics at larger scales are not available.

As a final weakness, there is limited *understanding* presented in the work - for example, intuition as to *why* these particular choices of norm work better than alternative approaches. The authors do present some motivation in Section 3 - "Theoretical Inspiration" - however the collection of facts here do not suggest to me any particular reason to favor $\ell^1$ or $\ell^\infty$ over $\ell^2$. For a more specific critique, the authors do not present any particularly compelling reason why $\ell^\infty$ works best for shallow embeddings, while $\ell^1$ works best when using a GNN.

### Questions
1. How do these results change as the number of nodes increase to 10K, 100K, and more?
2. Can you clarify the size (number of nodes and edges) of the graphs considered in the "Large Graph Representational Capacity" section? (My apologies if I missed them.)
3. Why did you use Remannian gradient descent based methods for optimization, as opposed to standard SGD or Adam?
4. Could you make more clear the direct comparisons between $\ell^1$, $\ell^2$, and $\ell^\infty$ which are possible based on the theorems mentioned in Section 3? As it currently stands, the results outlined in this section seem to be a collection of incomparable facts without any clear conclusions which would suggest a benefit for any particular $\ell^p$.
5. What is the reason that $\ell^\infty$ performs better for graph representation and reconstruction, while $\ell^1$ performs better for the GNN encoder? Is the reason something to do with representational capacity or ease of training (or is it something else entirely)? Does this suggest that, while $\ell^\infty$ may work well for encoding edges, it does not necessarily generate representations which are as useful for downstream applications?
6. How does the representational capacity of $\ell^1$ differ from $\ell^\infty$? A ball of a given size in $\ell^1$ is related to the ball of equal size in $\ell^\infty$ by a rotation and scaling, which makes me think the representational capacity of these models are likely equivalent.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates the use of normed spaces as a geometrically more elaborate alternative to the standard Euclidean choice. This sidesteps the complications and computational costs arising from using more general manifolds for more general geometries than Euclidean, while providing similar benefits to those.

Experimentally, the paper confirms this choice leads to lower computational costs and overall better link prediction and recommendation results than the standard Euclidean space and other standard choices for manifolds.

### Strengths
The paper is well written, all details are clearly explained and organised. I also appreciate the related work and theoretical inspiration sections. Specifically for the latter, the paper carefully highlights its main motivation, bringing together both intuition and existing theoretical results in a didactic way.

The problem definition and presented solution are simple and elegant, aiming to put the motivation to practice.

### Weaknesses
I disagree with some of the stated contributions. Quoting directly from the abstract "we propose normed spaces as a more flexible and computationally efficient [...]". I agree that the paper directly compares against popular Riemannian manifolds as a way to show the advantages of normed spaces, but they are not introduced here or are a new concept per se. I see the contributions are purely empirical, which, in my opinion, does not necessarily mean a negative point.

With this in mind, given that the main contributions of the paper are empirical, it would make the paper more of a "keep this in mind" when considering graph embedding problems. I have also listed this as a question, but considering the link prediction experiments: if the embeddings are learned and then used by the model, how does this compare to letting the model learn the embeddings itself in a way that best solves the task? I think it is crucial here to consider how the experimental setup is done to ensure the soundness of the claim that normed spaces are a better choice for link prediction using GNNs.

Further, in low-dimensional embeddings (e.g., t-SNE) where the goal is visualisation, the asymmetry pointed out in the paper for Eq. (1) is considered a problem, as the 2D Euclidean space is not adequate for handling it. Why is that? Appendix B was not clear in explaining this. I think one way to explore it could include distortion as computed over the "local neighborhoods" (e.g., 2-hop) and distortion globally (where the number of distance comparisons could dominate the average as they are much more numerous in cases where $E << N^2$, where $E$ is number of edges and $N$ is the number of nodes. As a side note, I believe exploring why this is not an issue is a theoretical result that is of interest to the wider ICLR community.

### Questions
1. As mentioned above, could it be that the asymmetry of Eq. (1) is not a problem simply because of the chosen evaluation metrics?

2. The experimental setup for the link prediction section is still unclear to me. Were the embeddings first learned by optimising (1) and then "frozen" to be used with the GCN?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
