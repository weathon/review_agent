# Efficient Sparsification of Densely Connected Clusters

- Decision: Reject
- Scores: 5, 5, 3, 5, 8

## Abstract
When modelling a real-world dataset as a graph, groups of highly correlated data items correspond to densely connected vertex sets (clusters), and efficient algorithms that find these clusters have broad applications in various data analysis tasks. In this paper we study densely connected clusters in graphs and introduce two sparsification algorithms that preserve the structure of  these clusters in both undirected graphs and directed ones. We show that our algorithms significantly speedup the running time of existing clustering algorithms while preserving their effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a sparsification method that maintain the strucutre of clusters for undirected and directed graphs. Specifically, the authors present two nearly-linear algorithms. Also, the authors provide a thorough theoretical analysis and expereimental results using synthetic datasets.

### Strengths
- Thorough theoretical analysis.

### Weaknesses
- What are the practical advantages of addressing the considered problem? What is the motivation?
- Some notations such as $\omega_G$ and $vol_G$ are used without being clearly defined.
- The algorithms assume that the input graphs exhibit clearly defined cluster structures. What if the clusters are less well-defined? Does it impact the effectiveness of sparsification?
- The experiments are conducted using only synthetic datasets. The effectiveness of real-world datasets are not tested.
- How sensitive are the algorithms with respect to the parameters?

### Questions
Please find the questions in the **weaknesses**.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The sparsification algorithm presented in this paper effectively handles densely connected clusters in both undirected and directed graphs. The article provides sufficient proofs demonstrating the method's time efficiency. However, the writing is somewhat confusing, making it difficult to clearly explain the approach used. Additionally, the experiments lack relevant data validation and comparative analysis.

### Strengths
1.The paper offers sufficient theoretical foundation with detailed analysis of Cheeger constants and spectral theory, providing solid proof of the algorithm's effectiveness.

2.The sparsification algorithm significantly speeds up existing clustering methods while effectively handling densely connected clusters in both undirected and directed graphs.

3.The algorithm is applicable to both undirected and directed graphs, demonstrating its flexibility and usefulness across different types of graph structures.

### Weaknesses
1. The structure of the article is confusing, and the method is not clearly explained. The article should highlight its contributions and clearly describe the solution ideas and working methods. Additionally, the use of symbols and lemmas, as well as the arrangement of proofs, should be more detailed.
    - For example, what’s vol_{G} and w_{G} is not explained in the intro when Equations 1.1 is introduced.
    - A_i \cup B_i \neq \emptyset for different i,j \in [k], typo?
2. The experiments lack relevant work. The absence of extensive testing with different algorithms on multiple datasets may impact the algorithm's universality and reliability.
3. The experiments are only conducted on synthetic datasets without real-world datasets. It would be better to provide more experiments on real-world datasets.  
4. While the algorithm shows improved efficiency empirically, the trends may need further explanation. For example, the improvement seems larger when the dataset is larger, while on directed graphs the improvement seems smaller when the dataset is larger (with 2000 and 5000 vertices, respectively). Further explanations are needed here, whether there are some gaps between the implementation and theoretical analysis.

### Questions
How do you plan to improve the structure of the article to enhance clarity? Will you include more detailed descriptions of the solution ideas and working methods?

Can you explain why the 5000-vertex setting in Figure 3 (a) takes less time compared to 4000-vertex setting? Will you consider testing the effectiveness of this method on a real-world dense dataset?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes efficient sparsification algorithms to preserve densely connected vertex sets in undirected and directed graphs.

### Strengths
The paper proposes efficient sparsification algorithms to preserve densely connected vertex sets in both undirected and directed graphs.

### Weaknesses
1. Notations (e.g., w_G(V_1, V_2), vol_G(V_1 \union V_2)) and concepts (e.g., k pairs of densely connected clusters) are introduced in the introduction but are not defined.

2. The significance of the proposed algorithms is not clear.

3. The experimental results are not convincing.

### Questions
Notations (e.g., w_G(V_1, V_2), vol_G(V_1 \union V_2)) and concepts (e.g., k pairs of densely connected clusters) are introduced in the introduction but are not defined, and therefore I was confused right from the beginning when reading the paper. In fact, I still do not fully understand what "k pairs of densely connected clusters" actually means, I can understand there are k disjoint subsets of vertices in the graph and the induced subgraph of each subset of vertex may form a cluster (i.e., a densely connected subgraph), but where do you get k pairs of densely connected clusters from k disjoint subsets of vertices? Therefore, it is very difficult for me to understand the paper when the important concepts are not properly defined. I suggest the authors to clearly define all notations and concepts the first time they are introduced in the paper. 

The significance of the proposed algorithms is not clear, i.e., why the results are important theoretically and practically?  It would be good if the authors could clearly explain the significance of the proposed method using a running example, e.g., given an input graph, what does the algorith output and show that the output is important.

The experiments were conducted on synthetic graphs only with k = 2. It would be more convincing to show the results on representative real graphs and large values of k. Another question is that k is actually unknown in practice, how do the authors address this problem?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a sparsification algorithm for dense clusters that approximately preserves the clustering the sparse representation of the underlying graph.

### Strengths
The authors study an interesting and well-studied problem.

### Weaknesses
1. A good fraction of the paper is rehashing well-known definitions.
2. This is a well-studied problem.  There are a lot of missing references including papers that compare different sparsification techniques (e.g., see https://arxiv.org/pdf/2311.12314 and references within).  Related references include : 
        1) D.A. Spielman and N. Srivastava. Graph sparsification by effective resistances. SIAM Journal on Computing, 40(6), pp.1913-1926. 2011.
        2) http://cs-www.cs.yale.edu/homes/spielman/PAPERS/CACMsparse.pdf
        3) https://www.cs.ubc.ca/~nickhar/papers/Sparsifier/Sparsifier-Long.pdf. and references within.
        4) https://www.cis.upenn.edu/~sanjeev/papers/focs22_weighted_sparsification.pdf

### Questions
1. The papers I referred to above address the questions of the sparse graph satisfying some properties including community detection. Please differentiate how the proposed work differs from the earlier work.

Update:  The authors have sufficiently addressed my concerns.  I have raised my score. Even as the authors claim their approach works well for the directed case as well, they have not conclusively demonstrated the effectiveness of their approach practically. I am not able to strongly back the paper as the work still seems lacking sufficiently new ideas for a well-studied problem.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduced sparsification algorithms and the corresponding theoretical analysis that preserve the structure of densely connected vertex sets in both undirected and directed scenario. The experiments on synthesis dataset indicates the efficiency of the proposed sparsification and the effectiveness for clustering.

### Strengths
1. The algorithm description is straightforward and easy to follow. For undirected graph, local sampling and reweighing the edges are the necessary steps for sparsification; for directed graph, the process to convert to undirected graph and converted back are the extra steps.
2. The theoretical analysis and the probability guarantee makes the methodology promising; the nearly-linear running time makes the algorithm applicable. 
3. The specific challenges in directed graph case are discussed in detail.

### Weaknesses
1. Some of the notations could be more clear: 
    i.  The w_G and vol_G in line 37 are not defined.
    ii. Although it mentions C is a universal constant on Lemma 3 and Lemma 4, it is still not clear how to determine the value of C in formula 3.1 for the experiment. 
2. There is no experiment conducted on multiple cluster case or real-world dataset.

### Questions
1. Is there a way to adjust the level of sparsification? How will the efficiency and effectively be affected?

### Soundness
4

### Presentation
3

### Contribution
3
