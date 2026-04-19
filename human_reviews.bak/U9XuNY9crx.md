# Complete and continuous representations of Euclidean graphs

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 5, 3, 6

## Abstract
Euclidean graphs have unordered vertices and non-intersecting straight-line edges in any Euclidean space. The main application is for molecular graphs with vertices at atomic centers and edges representing inter-atomic bonds. Euclidean graphs are considered equivalent if they are related by isometry (any distance-preserving transformation). This paper introduces the strongest descriptors that are provably (1) invariant under any isometry, (2) complete and sufficient to reconstruct any Euclidean graph up to isometry,  (3) Lipschitz continuous so that perturbations of all vertices within their epsilon-neighborhoods change the complete invariant up to a constant multiple of epsilon in a suitable metric, and (4) computable (both invariant and metric) in a polynomial time in the number of vertices for a fixed dimension. These strongest invariants transparently explained a continuous structure-property landscape for molecular graphs from the QM9 database of 130K+ molecules.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper describes descriptors for geometric Euclidean graphs that are invariant under isometry and rigid motion. These descriptors are robust against perturbations in the coordinates of the nodes. Furthermore, the descriptors that are invariant to rigid motion are also complete, meaning these descriptors are injective, and are sufficient to reconstruct the Euclidean graph up to isometry. In addition, the authors show the benefits of their descriptors for the study of molecular graphs.

### Strengths
The problem addresses a highly relevant problem within its field. The authors introduce a unique set of descriptors that offer theoretical guarantees in characterizing Euclidean graphs. This contribution underscores the paper's strong theoretical foundation, highlighting its significance in the field. Moreover, the authors do an excellent work in presenting and motivating the problem.

### Weaknesses
One notable weakness of the paper lies in the brief and challenging-to-understand experiments section. The lack of clarity in this section makes it difficult to discern the intended purpose of the figures presented. A more detailed explanation or elaboration of the experimental results would greatly enhance the overall comprehensibility of this section.

In addition, the descriptors do not scale effectively with the dimensionality of the Euclidean space and do not consider node or edge attributes, potentially overlooking crucial information in real-world applications where such attributes play a significant role. Moreover, these descriptors can only be utilized by models incorporating some sort of permutation invariance, as they require a "subrepresentation" for each possible permutation of the nodes.

### Questions
- In Section 5, the purpose of each figure is unclear. Could you please elaborate on what each figure represents? Additionally, in Table 2, what does SPF stand for?
- Lemma C.2 explicitly states that given the OCD descriptor, the Euclidean graph can be reconstructed up to isometry. However, I could not find a similar result for the SDD descriptor. Does this mean the same principle does not apply to SDD? If that's the case, could you provide the reason for this difference?
- Analogously, it is conjectured that the SDD descriptor  achieves completeness for $h>n-1$, in contrast to the OCD descriptor, which has been proven. What specific factors contribute to the absence of a proof for the SDD descriptor?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper gives a new signature suitable for representing geometric graphs that's invariant under isomorphism (permutations of vertices). It first gives desired properties of such invariants, in particular, stability under vertex perturbations, and then gives one based on distances after suitable shifts. The efficiency of the algorithm is analyzed, and experiments were done on finding (near) duplicates in the QM9 molecules dataset.

### Strengths
The algorithm given is natural, and has a high degree of interpretability. The experiments were performed on real data, and give direct comparisons with previous works (using significantly different invariants).

### Weaknesses
The theoretical running times of the algorithms still have exponential dependences (albeit on a different parameters). The requirements given in Problem 1.1, while natural, is also quite complex. These are understandable to me due to the complex domain-specific info of the molecules though.

### Questions
My background is in combinatorial/algebraic graph algorithms, so am accustomed to features based on eigenvalues. Are there graphs with same/similar list of eigenvalues where the invariants introduced here differ?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper defines Euclidean graphs as collections of m points in R^n, some of which are connected by straight non-intersecting lines (edges). The paper searches and finds descriptors for such graphs which satisfy several desirable axioms: invariance to rigid motions and permutations, completeness, LIpschitzness with respect to some known metric. 

The authors provide experiments which show the QM9 dataset contains some multiple molecules (and some other experiments which were not very well explained)

### Strengths
There have been several papers on similar topics in the last 1-2 years. This paper seems to be the first to 
(i) show completeness in a setting where edges are given. Many previous works only focused on the setting where there is no graph structures.
(ii) Show that the invariants described are Lipschitz.

### Weaknesses
(a) The proof of the first contribution is very simple and I'm not sure it is correct, see questions below.
(b) While previous papers have not considered Lipschitzness, this is probably not so difficult to achieve with other methods. The more interesting challenge IMHO lies in being bi-Lipschtz with respect to an appropriate invariant metric on the input space. For a definition of bi-Lipschtizness and a technique which is helpful for obtaining Lipschitz invariants from non-Lipschitz invariants See e.g.
[COMPLETE SET OF TRANSLATION INVARIANT MEASUREMENTS WITH LIPSCHITZ BOUNDS by Cahill et al.]
(c) The experiments were not well explained. They do not seem to point to an advantage of the described descriptor over other descriptors discusses in the literature. 
(d) the paper is overall well written but overemphasizes things which were already discussed in the literature (invariance, completeness) and doesn't talk enough about the unique contributions of the paper: The metric used is not defined in the paper but only in the appendix. The experiments are not explained as stated above. No intuition regarding proofs is provided- and in particular in the technical novelty with respect to earlier work.

### Questions
Regarding the proof of completeness in Lemma C.2 part (c). You say
"the presence of an edge... is determined... by the matrices D(A \cup 0) and M(G;A \cup 0)"
I'm not sure about this argument. Say you have two points p,p' which are both not in A. How would you know if there is an edge between them from the matrices D(A \cup 0) and M(G;A \cup 0)?

Other questions:
* Where in your proof do you need straight non-intersecting edges?
* Also is this assumption always valid in chemistry? If so could you explain/ give reference?
* Can you explain in more detail what problem 2.1 is? Is this a problem you solve in this paper?
* Page 5: what is the `main paper'? There is only one paper. How are these definitions related to the definitions above when you defined 
  IGS and RGS?
* A paper you could consider adding to your refs:
[Is Distance Matrix Enough for Geometric Deep Learning? Li et al]


Local remarks and suggestions regarding writing (no need to address in rebuttal):
* Last paragraph in page 2 is confusing I would rephrase
* In the past work section: Why do you say Problem 1.1 is much simpler? The explanation gives is that there is an  algorithm with exponential complexity to solve the problem (note also this algorithm has issues when there are eigenvalue multipliciities)
* There are two  things which seem to me make the presentation unnecessarily too complicated (this is not essential). Firstly, why do you decide to have a (n-1) by (n-1) distance matrix instead of n by n like the rest of the world uses? It makes the notation cumbersume. Secondly, why do you bother to lexicographically sort things when later you will define a permutation invariant distance? You can just store things in an arbitrary order.
* I think the first paragraph is Section 5 is unnecessarily aggressive. Firstly, I didn't attend Isabelle Guyon's talk but I would guess the remark you are quoting was made half-jestingly. Secondly, experiments on QM9 use the standard train/test paradigm and results are evaluated on the test. This is a reasonable and standard practice. The limitations of this practice regarding generalizing to out of distribution data are also known. Since you are not addressing these issues in the experiments I don't see why you need to bring up the whole issue. Just say what your experiments are. 
* Page 12 just before the beginning of the appendix there is a paragraph which maybe is misplaced?

### Soundness
3 good

### Presentation
3 good

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
This paper focuses on the construction of descriptors for Euclidean graphs. A Euclidean graph is a point cloud in $\mathbb{R}^n$, some of which are connected by (non-intersecting) straight lines.

The descriptors under consideration should be complete invariants, i.e. they should allow the identification of any graph up to rigid motion. Furthermore, they should be robust to noise in the following sense: if a vertex is moved by $\epsilon$, then the descriptor should be perturbed by $\epsilon$ up to a multiplicative constant, for some metric to be determined. Finally, both the descriptor and the metric should be computable in polynomial time.

The authors propose such an invariant for graphs in metric space then Euclidean graphs, and derive its time complexity.

The entire methodology has been developed for the analysis of molecular graphs whose vertices are atomic centers and edges represent inter-atomic bonds. A numerical application to a database of 130,808 molecules illustrates the behavior of the proposed algorithms and highlights their relevance.

### Strengths
The question raised by the paper is an interesting one, and goes beyond the application to molecular graphs. In addition to control its complexity, requiring the descriptor to be Lipschitz is, in my opinion, relevant in many contexts. The solution developed in the paper looks promising and is supported by rigorous theoretical arguments.

The introductory section is highly pedagogical and provides a clear understanding of the issues at stake in the paper.

### Weaknesses
When we read the motivations of the paper, we expect the graphs considered to have a large number of vertices, which could exclude some algorithms from the literature. In particular, the authors insist on the need for polynomial complexity in the number $m$ of vertices.

However, if the complexities are indeed polynomial, the method may turn out to be too slow for large graphs with a term in $m^5$ in Theorem 4.4 (in dimension 3).

The application considered in the numerical part actually deals only with small graphs ($m<30$) for which many other methods (even with worse complexities) could be employed.
It seems to me that this limits the interest of the proposed method, both for the example considered in the paper (where other algorithms could be applied) and for application to larger graphs (where the time complexity could be too bad).

In addition, I think it is a good point to control descriptor variations, but limiting them to a single vertex seems restrictive: in real data, all the vertices are subject to noise. The result obtained is interesting but should be generalized to metrics that take into account all the vertices of the graph.

Finally, the paper lacks a numerical section that illustrates the algorithm's behavior on synthetic graphs of varying size, both in terms of computation time and robustness to noise, with comparisons with others methods from the literature, e.g. graph distances or kernels).

### Questions
From my point of view, a numerical part with computation times should illustrate the theoretical results of complexity to highlight which size of graph can be handled by the proposed descriptors. Do the authors have an answer to this point?

I do not understand the role of $h$ in section 3 and more precisely in Theorem 3.4. To minimize complexity, we want to take $h=1$, but is the descriptor still relevant (in relation with the remark at the end of page 6)? How $h$ is selected and what happens to complexity in that case?

In the same vein, what is the exact role of $l$ and how should it be selected, both in Theorems 3.4 and 4.4? Is it a way to control the time complexity?

How were these two parameters selected in the numerical section of the paper?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into the representation and comparison of Euclidean graphs using concepts developed for point clouds. These point clouds are often represented by finite sets of unlabeled points, and the most natural equivalence for these point clouds is rigid motion or isometry, which maintains all inter-point distances. 

The paper emphasizes the importance of complete isometry invariants, which are descriptors that can reliably compare Euclidean graphs without ambiguity. The authors propose the first continuous and complete invariant for Euclidean graphs, which is computable in polynomial time based on the number of points.

Furthermore, the paper also touches upon the challenges posed by real data, which is often noisy, and provides implementation details and code for verification.

### Strengths
The paper addresses an important problem of checking invariance among Euclidean graphs and presents a pioneering approach to the challenge of recognizing patterns in Euclidean graphs. 

One of the standout strengths is the introduction of the first continuous and complete invariant for Euclidean graphs This invariant not only offers a reliable comparison but is also computable in polynomial time based on the number of points, showcasing its potential efficiency. 

The authors provide a sufficient discussion on the significance of distinguishing graphs related by isometry or having similar properties.

Additionally, the paper's acknowledgment of real-world challenges, such as noisy data demonstrates a comprehensive understanding of practical scenarios, making the proposed solutions more applicable and relevant.

### Weaknesses
Following would be some of the central points of discussion :

1) **Scalability \& Approximation**: While this is a pioneering effort for complete invariance in graphs, to be applicable to a much broader application class the question of scalability is crucial. Graphs typically comprise thousands or millions of nodes. Having a polynomial complexity on the nodes makes it harder to execute. To this end, would the authors think of some approximations that could be possible? or as a reverse question can comment on the impossibility of such a scenario?

2) **Other graph types**: I am assuming the graphs in the paper are undirected and consist of an edge weight of 1. Can the theory developed extend easily to weighted and/or undirected graphs? What would be the challenges in doing so?

3) **Robustness to noise**: This is a minor point, but is there some theoretical result possible on the tolerance to outliers or Gaussian noise levels?

### Questions
Please refer to **weaknesses** section

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
