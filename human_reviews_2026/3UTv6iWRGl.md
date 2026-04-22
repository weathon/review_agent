# Hierarchical Epsilon-Net Graphs: Time Guarantees for HNSW in Approximate Nearest Neighbor Search

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4, 2

## Abstract
Hierarchical graph-based algorithms such as HNSW achieve state-of-the-art performance for Approximate Nearest Neighbor (ANN) search in practice, but they often lack theoretical guarantees on query time or recall due to their heavy use of randomized heuristic constructions. In contrast, existing theoretically grounded structures are typically difficult to implement and struggle to scale in real-world scenarios.
We introduce a property of hierarchical graphs called Hierarchical $\varepsilon$-Net Navigation (HENN), grounded in $\varepsilon$-net theory from computational geometry. This framework allows us to establish time bounds for ANN search on graphs that satisfy the HENN property. The design of HENN is agnostic to the underlying proximity graph used at each layer, treating it as a black box. We further show that HNSW satisfies the HENN property with high probability, enabling us to derive formal time guarantees for HNSW.
Constructing a HENN graph relies on finding $\varepsilon$-nets. Existing methods for finding $\varepsilon$-nets are either probabilistic or, when deterministic, become impractical in high dimensions. To address this, we propose a budget-aware practical algorithm for building $\varepsilon$-nets, under a user-specified preprocessing time budget.
Empirical evaluations confirm our theoretical guarantees for both HENN and HNSW, and demonstrate the effectiveness of the proposed budget-aware algorithm for constructing HENN and, more generally, $\varepsilon$-nets. This flexibility allows practitioners to select the method that best fits their specific use case.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces _Hierarchical $\epsilon$-net Navigation_ (HENN), a property for hierarchical graph data structures such as the ever-popular HNSW. This property revolves around $\epsilon$-nets, a common tool in computational geometry. For a HENN graph, one can think of each layer in a hierarchical data structure as an $\epsilon$-net of the larger prior layer. We connect the hierarchy by drawing edges between common points across layers. Producing this data structure thus gives provable guarantees on HNSW:
1. Establishing that HNSW is a HENN graph with high probability
2. HNSW thus has logarithmic query times, conditioned on a recall bound parametrized by a success probability $\gamma$

The authors also provide a practice-friendly algorithm to compute a HENN graph, which offers a tradeoff between speed and quality (which existing algorithms cannot give). Finally, the authors provide empirical comparisons on standard retrieval datasets, such as SIFT, NYTimes, and GIST. The results are meant to be complementary to the theoretical results and show (nearly) matching performance between HNSW and HENN + NSW.

### Strengths
1. Theoretical understanding of graph-based ANNS is lacking and this submission contributes a very necessary result in that regard.
2. The main body is well-written, concise, and easy to follow.
3. The idea of applying $\epsilon$-nets to model graph-based ANNS is novel. I particularly like that it provides a model for multiple graph approaches.

### Weaknesses
1. The budget algorithm for computing $\epsilon$-nets is an interesting contribution, but it seems hard to justify using it in practice when we could just use the well-optimized HNSW. Not to mention, it's still probabilistic.
2. As an extension, I think the empirical results feel like an afterthought. My takeaway from this paper was that we could get theoretical guarantees for HNSW and other hierarchical graphs using $\epsilon$-nets, but it would've been nice to see a wider empirical study of HENN to complement it. For example, I'd have liked to have seen results on how HENN scales with dataset size and a series of plots tracking the recall bound (or an appropriate surrogate) with respect to QPS. 
3. I think a larger discussion about the place of this result among other recent works in theoretical graph-based search would be useful, simply to compare approaches and to help the reader understand the merits of this paper's approach.
4. As this is primarily a theoretical submission, I don't know if I agree with the discussion in appendix A: I believe some kind of some kind of worst-case analysis on the output quality of HENN/HNSW would have been nice. I think such a result would've made for a more complete submission.

### Questions
1. How difficult would it be to extend the results of this work to a beam search setting? i.e. can we obtain provable guarantees for outputting k > 1 nearest neighbors?

### Soundness
4

### Presentation
4

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
The authors introduce a property of hierarchical graphs called Hierarchical ε-Net Navigation (HENN) and analyze the time bounds for ANN search on graphs that satisfy the HENN property. The authors also claim that HNSW satisfies the HENN property with high probability and derive time guarantees for HNSW. The authors also conduct the experiments to show the effectiveness of HENN.

### Strengths
S1: The theoretical analysis of HENN is appreciated.

S2: The experimental evaluation of HENN is extensive.

### Weaknesses
W1: The theoretical analysis seems not very robust. The parameter $\rho_{\gamma}$ plays a key role in the theoretical results, and the authors claim that, for most existing navigable graphs, $\rho_{0.9} = O(1)$ (line 252). However, this claim is not rigorous. In fact, for some hard datasets, the recall of HNSW cannot even reach 0.9. Therefore, I wonder whether the theoretical analysis can truly explain why HNSW performs well in practice.

W2: I could not find an ablation study. The authors test several similarity graphs equipped with HENN, but I would like to see a direct performance comparison between using HENN and not using HENN.

W3. From the results in Figure 7, HENN+NSW does not show any performance improvement over HNSW. Therefore, I wonder whether it is meaningful to use HENN in practice.
 
W4: There exist several other state-of-the-art ANNS methods [a][b][c] that are not discussed in the related work. The authors are encouraged to discuss and compare their method with these approaches. For readers, it would be more informative to see a comparison between HENN-based similarity graphs and these SOTA approaches, as the hierarchical structure is not indispensable for many SOTA ANNS solvers.

[a]. Accelerating large-scale inference with anisotropic vector quantization. ICML 2020

[b]. Probabilistic routing for graph-based approximate nearest neighbor search. ICML 2024

[c]. Rabitq: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search. SIGMOD 2024

### Questions
My comments and suggestions are as follows:

C1: Could the authors provide a more rigorous analysis of $\rho_{\gamma}$ . The current explanation is not convincing. (See W1)

C2: The ablation study is important, and the authors are encouraged to include the ablation experiment mentioned in W2.

C3: Could the authors clarify the practical contribution of this paper? (See W3)

C4: The discussion of related work should be expanded to provide a more comprehensive comparison with existing studies so that readers can better understand the position of this paper. (See W4)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a graph property called Hierarchical epsilon Net Navigation(HENN). They provide probabilistic query-time bounds for the graphs that satisfy HENN property, that are logarithmic in input parameters. Furthermore, they show that the famous and widely used HNSW graphs satisfy HENN property with high probability. Finally, they design a novel budget-aware algorithm that with more preprocessing time increases the probability of successfully constructing an epsilon-nets, which is a key-subroutine to construct HENN graphs.


Decision: I feel the paper should be accepted.

### Strengths
1) Paper is well very written and clean. It is easily readable, and quite fun to read. Proofs are well written, loved reading the paper in general.

2) The paper shows nice theoretical guarantees for HENN graphs, which covers popularly and widely used HNSW.

3) Building relation between epsilon nets and recall bound is nice.

4) Connection between HENN and HNSW is nice.

5) Figure 6, showing reduction of index size with higher processing time, is extremely nice. This is happening because with higher processing time one is able to construct smaller sized epsilon nets.

### Weaknesses
1) The guarantees are in terms of recall bounds definition which is non-traditional.

2) Given the definition of recall bound, I felt the proofs are simple. 

3) The success probability is exponential in log n, that is gamma^{log n}, and gamma is something like 0.9 in practice for constant recall bound. Together this implies that success probability is very tiny.

### Questions
1) Please state the running time in terms of the maximum degree of the proximity graph and the recall bound parameter.

2) Can you please elaborate a more on "For most existing navigable graphs, Malkovetal.[36]; Malkov&Yashunin[35] show
 that ρ_{0.9}=O(1). Particularly, is this a theoretical result or just emprical finding? Would you please also mention the degree of these proximity graphs. In particular, I wish to know if we know of proximity graphs, that achieve constant recall bound with constant degree proximity graphs.

3) Index size: I believe roughly captures the number of edges? It would be interesting to see an analogous to Figure 6, where the instead of index size, you show the reduction in just the number of nodes.

4) What are practical implications of these theoretical findings? One is reduction in index size, by using your budget-aware procedure, is there anything else? You can be very speculative here.

5) Also, when you define recall bound, you need to specify the definition of distribution of choice for start note s. For theorem 4 to hold, what would this distribution be. I feel you may want to change the definition of recall bound where instead of doing a random choice s, you write recall bound over worst case start node s, which will change definition as follows: min{k|\max_{s} Pr_{q}[GS_{q}(G,s) \in NN_{k,X}(q)] >= \gamma}. Without this, your distribution of s, will need to depend on the query q itself. Happy to elaborate further if this doesn't make sense.

6) In Theorem 4, can you also write the expression for recall bound of HENN graph in terms of the recall bound of PG at each level.

7) Would you please talk more about the success probability which is exponential in log n. In particular, why is this okay?

### Soundness
4

### Presentation
4

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
This paper studies the problem of developing theoretical guarantees on the latency and search quality of hierarchical graph-based approximate near-neighbor search algorithms like HNSW. This is a very important and timely research question since these algorithms have achieved widespread adoption and strong performance in practice, but still largely lack meaningful theoretical guarantees. This paper makes progress in closing this gap between theory and practice by drawing a connection between computational geometry and learning theory. In particular, the authors introduction the notion of hierarchical epsilon-net navigation graphs (HENN), show that HNSW is an instantiation of HENN, and utilize this framework to prove a probabilistic guarantee on the runtime of HNSW, which they show is poly-logarithmic in the number of hierarchical layers and the vector dimensionality with high probability. The authors also experimentally validate their theorems where they find strong empirical evidence in support of their theoretical claims.

### Strengths
This paper makes progress on an important research problem in providing principled theoretical guarantees on the query complexity of graph-based near neighbor search algorithms like HNSW. Moreover, the authors experimentally validate their theory which strengthens their claims considerably. The paper is also well-written with clear plots, figures, and theorem statements while deferring the appropriate amount of detail to the appendix. The core insight at the heart of the paper in connecting ideas in computational geometry, namely epsilon-nets to concepts in learning theory such as VC dimension is creative and may inspire future work within this direction as well. The authors also use their theory to design an improved indexing scheme that reduces memory by spending more time preprocessing. This is also a novel insight and may have practical implications as well.

### Weaknesses
1. I think the experimental section is currently weak in that it does not consider large scale datasets beyond roughly 1 million points. I would strongly encourage the authors to consider running experiments on datasets such as Big ANN Benchmarks, which include benchmarks at the 10M, 100M, and 1B scale. I believe that running experiments at scale is especially important for this work because the authors claim that the hierarchical layers improve scalability, but this claim is difficult to verify without large-scale validation. 

2. The current discussion of related work is too brief and should not be completely relegated to the appendix because acknowledging the relevant pieces of related work is important for understanding the contributions of the paper. In particular, the authors discuss related work on hierarchical graph-based ANN algorithms, but do not really mention any non-hierarchical graph-based techniques that are also achieve state-of-the-art performance, such as [Vamana](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). In addition, multiple works in the literature seem to have recently demonstrated the hierarchical layers of HNSW are not necessary ([Lin & Zhao, 2019](https://arxiv.org/pdf/1904.02077), [Coleman, et al, 2022](https://arxiv.org/pdf/2104.03221), and [Munyampirwa, et al. 2025](https://openreview.net/pdf?id=OJwITuuU3h). This large body of literature on non-hierarchical graph-based ANN likely merits some discussion in this paper, and the authors might also want to address whether their theorems are consistent with these results as well. 

3. The citations in the paper are currently in the wrong format. The authors should follow the ICLR guidelines and use the natbib package and apply citations with the \citep{} and \citet{} commands as appropriate.

### Questions
1. As mentioned above, I would strongly suggest that the authors consider experimentally validating the core claims of the paper on larger-scale benchmark datasets, such as those from Big ANN benchmarks since scalability is a core component of the narrative in the paper. 

2. How does Theorem 4 of the paper change when the number of layers is 1? Is Theorem 4 consistent with the growing body of research in the literature suggesting that hierarchical layers are not required for state-of-the-art performance in graph-based near neighbor search (particularly in the context of [Vamana](https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) and the findings of [Munyampirwa, et al. 2025](https://openreview.net/pdf?id=OJwITuuU3h). Can the theory in the paper be naturally extended to handle the case of a single layer graph?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a theoretical analysis of the running time and search quality of the Hierarchical Navigable Small World (HNSW) algorithm. 
The main idea is that, with high probability, the randomly sampled upper layers in HNSW form an $\varepsilon$-net of their corresponding lower layers. 
Under the "navigable graph assumption", the search result from the upper layer is already an approximate nearest neighbor for the next layer, allowing the algorithm to refine results through the hierarchy. 
By analyzing the running time, the authors argue that the optimal hierarchical structure reduces data size by a constant factor c per layer, leading to a total of $O(\log n)$ layers and a query time of $O(d \log^2 n)$ 
The paper also includes empirical evaluations demonstrating that the theoretical predictions align with practical observations on HNSW and other popular navigable graph structures.

### Strengths
1. The theoretical analysis of HNSW’s performance addresses a key open question given the algorithm’s widespread use in large-scale ANN search.
2. The authors present both mathematical analysis and comprehensive experiments, with proofs that are relatively accessible and experiments that are detailed.
3. The use of $\varepsilon$-net to connect the optimality of the approximate neighbors across hierarchical layers is an elegant idea.

### Weaknesses
1. The theoretical results depend heavily on the assumption that $\rho_{\gamma}$ (the recall bound of a navigable graph) is a constant and that greedy search on a navigable graph with constant degree reliably returns a top-$\rho_{\gamma}$ neighbor. While this assumption may hold empirically, it fails in worst-case scenarios (as shown by Dian et al. [9]), which limits the theoretical soundness of the claimed guarantees for HNSW.
2. Even if the greedy search can consistently find a top $\rho_{\gamma}$ neighbor, the overall success rate of the hierarchical search decays exponentially with the number of layers—approximately $\gamma^{\log n}$. Thus, even with $\gamma \approx 0.99$, the success rate tends toward zero as $n$ grows...

### Questions
1. Could the authors clarify Algorithm 4 $\varepsilon$-net construction, particularly the step FindUnhitRange? 
    It is stated to run in $O(n)$ time (around line 869), but I'm still unclear about the details.
2. Minor typo: $L_L$ on line 1108

### Soundness
2

### Presentation
4

### Contribution
2
