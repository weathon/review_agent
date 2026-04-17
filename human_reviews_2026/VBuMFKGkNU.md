# Edge-Based WL and Message Passing

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
We propose EB-1WL, an edge-based color-refinement test, and a corresponding GNN architecture, EB-GNN.   Our architecture is inspired by a classic triangle counting algorithm by Chiba and Nishizeki, and explicitly uses triangles during message passing.
We achieve the following results:
(1) EB-1WL is significantly more expressive than 1-WL. Further, we provide a complete logical characterization of EB-1WL based on first-order logic, and matching distinguishability results based on homomorphism counting. 
(2) In an important distinction from previous proposals for more expressive GNN architectures, EB-1WL and EB-GNN require near-linear time and memory on practical graph learning tasks.
(3) Empirically, we show that EB-GNN is a highly-efficient general-purpose architecture: It substantially outperforms simple MPNNs, and remains competitive with task-specialized GNNs while being significantly more computationally efficient.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The article proposes an altered Weisfeiler-Lehman approach that operates on the edges instead of nodes. Based on this, the authors design a new GNN architecture. The article discusses the expressive power by showing that EB-1WL lies between 1-WL and 2-WL. More precisely, they provide a lower bound using homomorphism counts from the class of treewidth-2 chordal graphs. Furthermore, the authors provide a first-order-logic equivalent formulation. Finally, an experimental evaluation shows a favourable performance of their GNN architecture on graph classification and edge-level tasks.

### Strengths
1. The article is easy to follow
2. EB-1WL gains expressivity over NC-1WL with a quite simple tweak
3. EB-1WL achieves good results on the QM9 dataset

### Weaknesses
1. The novelty is limited. The authors propose only a slight alteration of the NC-1WL method. 
2. The expressivity discussion is limited. While the authors formulate some lower bound (Thm.5), the experimental results on e.g. BREC are not put into perspective. It is unclear what the increased expressivity can be actually attributed to. 
3. It is unclear what the role of the first-order-logic formulation is. I don't see any implications of this. 
4. While the authors' approach is based on NC-1WL, it is never compared to in Section 6. 
5. Table 1 contains only very few competitor methods. It would be interesting to compare to the methods from Table 4 on synthetic datasets as well. 
6. Theorem 3 is incomplete. The second sentence is not a consequence of the first.  
7. Lines 178-181 need to be explained in more detail. While the Chiba&Nishizeki algorithm is even mentioned in the abstract, it is never discussed. I would recommend to at least mention what the arboricity is. Otherwise, Proposition 1 is not understandable.

### Questions
1. How does EB-1WL compare to ordinary 1-WL on graphs where nodes AND edges have triangle count features? 
2. Concerning W3 above: Are there any implications of the first-order-logic formulation?
3. On molecular graphs, expressivity beyond 1-WL is rarely needed. How do you explain the performance improvements? 
4. Can some of the experimental results be explained by Theorem 5?
5. Theorem 3 is proven using two CSL graphs (though with 16 nodes). Do you have an intuition why in Table 1 the CSL results for EB-GNN do not improve over MPNN+C3?

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
3

### Summary
This paper proposes edge-based WL (EB-1WL) test which has been proved to be strictly power than 1-WL test,  NC-WL-test but less than 2-WL test. Based on the proposed edge-based WL test, edge-based GNN (EB-GNN) is builted to achieve EB-1WL test.

### Strengths
1. It is interesting to see a newly proposed WL test algorithm that demonstrates stronger expressive power than existing WL tests, while still maintaining reasonable computational efficiency.

2. The writing and organization of the paper are clear and easy to follow.

### Weaknesses
1. Although the authors claim that the proposed method is more powerful than NC-1WL, the experimental results of NC-GNN and its efficiency are not included as a baseline in the experiments.
In addition, the experimental section lacks evaluations on large-scale graph datasets such as COLLAB, which would better demonstrate the efficiency and scalability of the model. 
It would also be helpful to report the GPU memory usage of each model to provide a more complete comparison of computational cost.

2. A comparison between the proposed EB-1WL test and other WL variants beyond NC-1WL is not discussed. For instance, 
	the baseline model 5-ℓGIN introduces the r-ℓWL test—what is the relationship between this and EB-1WL? 
	Moreover, is EB-1WL theoretically or empirically more powerful, leading to better performance on the QM9 dataset?

### Questions
N/A.

### Soundness
3

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
5

### Summary
The authors propose the EB-1WL test for graph isomorphism, from which they define EB GNNs, motivated by a triangle counting algorithm in the TCS literature. 
They show that the EB-1WL test is more expressive compared to 1-WL, and study its logical expressiveness and hormorphism counting power. 

For the logical part, they characterize the power in terms of the so-called clique-based finite-variable fragments (Theorem 4).  For the homomorphism count, they show that EB GNN can distinguish chordal graphs of treewidth two, which is something between 1-WL and 2-WL  (Theorem 5)


Moreover, they show that the proposed method, EB GNN, needs only linear time and memory for specific graph tasks. 
Indeed, if some parameter called the absorbcity of the graph is bounded, then each iteration of their algorithm only runs in almost linear time in the number of edges. 
The method is formulated in Equations 1-4. They conclude the paper with supporting experiments.

### Strengths
- The new method based on message passing on edges, with provable expressivity, is of potential interest to the graph ML community 

- The paper is well-written and clean

### Weaknesses
- The computation complexity reduction is restricted to some assumptions. And it is not clear whether those assumptions will break the symmetry/power of the method or not.

### Questions
This is an interesting, well-written paper on graph ML. The authors introduced EB GNNs, based a novel idea on message passing on edges. The authors supported the introduction of their method by proving that it's better than 1-WL in terms of expressive power, as well as showing (theoretically) how it can count homomorphisms and how it can express logical formulae for graphs. They are also supported by experiments. 

I have a number of questions/comments:

 - Line 178: 'If we iterate..." The question i,s why are you allowed to do this? Did this keep the permutation invariance, and will it break it? If you only do the iteration over the low-degree nodes, then do you still get the full expressivity (proven in the paper)? This is a main weakness of the paper. I ask the authors to provide more explanation on this.

- The improvement in time complexity is exciting; however, there are also some lower bounds that GNNs cannot break. If the average degree of the graph is small, then one cannot count subgraphs in linear time (I found it here [1], but there might be other theoretical papers you can find on the web, too). So please specify exactly when the time complexity of your method is small. If $m$ is of order $n^2$ (dense graph), then can't you just do 2 or higher order WL? If $m$ is of order $n$ (sparse graph), then the average degree is of course $O(1)$, so you can just search over neighborhoods (similar to [1]) and you are probably fine?


I feel my concern is fully addressed if you can explain more and compare with known TCS complexity bounds. I provided one reference just for instance, but please do search, as there might be other ones. 






[1] Tahmasebi, Behrooz, Derek Lim, and Stefanie Jegelka. "The power of recursion in graph neural networks for counting substructures." In International Conference on Artificial Intelligence and Statistics, pp. 11023-11042. PMLR, 2023.


---

Overall, this is an interesting paper and I'm happy to increase my score if the authors provide answers to my concerns.

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
4

### Summary
This paper proposed a special case of 2-WL variant, which only focusing on location connection and only using existing edges as the 2-tuple nodes. Nevertheless, the proposed method is largely overlapped with existing papers. Also some proposed theorems are not interesting as they are special case of existing papers.

### Strengths
Presentation is good. Writing is easy to follow.

Also, Figure 3 is interesting. I think Theorem 4 and Theorem 5 are the most important contribution.
Not sure whether any is overlapping with https://arxiv.org/abs/2503.00485

### Weaknesses
The biggest issue is that the proposed method is not new. 

1. The local-neighbor based extension is a special case of https://arxiv.org/pdf/1904.01543
2. Using existing edges as restricted 2-tuple is a special case of https://arxiv.org/abs/2210.09521. See figure 1 and appendix A.12. Notice that in figure 1, the equation (1) (2) (3) (4) corresponds to the connection shown in solid line and yellow nodes. The only difference, is that this paper changes the study from set to tuple, which actually is designed to be removed by the paper. See section 3.2 for the discussion of moving from tuple to set. While I do agree that order somehow has certain expressivity, but the paper is not interesting given the proof does not prove the importance of using tuple instead of set. In general, while being slightly different, I think the small modification does not worth a paper unless the author proof the expressivity improvement. (Notice that the computation cost should be the same) 

In general, I feel that the author lacks significant amount of literature review, especially regarding to efficient higher-order GNNs, and also subgraph GNNs. Papers from Haggai Maron, Bohang Zhang, Muhan Zhang, and others expressivity researchers are necessary to review and study.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1
