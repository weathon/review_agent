# Efficient Learning on Large Graphs using a Densifying Regularity Lemma

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Learning on large graphs presents significant challenges, with traditional Message Passing Neural Networks suffering from computational and memory costs scaling linearly with the number of edges. We introduce the Intersecting Block Graph (IBG), a low-rank factorization of large directed graphs based on combinations of intersecting bipartite components, each consisting of a pair of communities, for source and target nodes. By giving less weight to non-edges, we show how an IBG can efficiently approximate any graph, sparse or dense. Specifically, we prove a constructive version of the weak regularity lemma: for any chosen accuracy, every graph can be approximated by a dense IBG whose rank depends only on that accuracy. This improves over prior versions of the lemma, where the rank depended on the number of nodes for sparse graphs. Our method allows for efficient approximation of large graphs that are both directed and sparse, a crucial capability for many real-world applications. We then introduce a graph neural network architecture operating on the IBG representation of the graph and demonstrating competitive performance on node classification, spatio-temporal graph analysis, and knowledge graph completion, while having memory and computational complexity linear in the number of nodes rather than edges.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper works with graph approximations roughly based on the stochastic block model, i.e. approximating the adjacency matrix of the original graph as the sum of multiple blocks, i.e. bipartite subgraphs where all source nodes are connected to all target nodes.
In contrast to previous work where the number of blocks needed to achieve a given approximation quality depends on the sparsity of the graph, the paper guarantees this depending only on the approximation quality itself.
They also showed that good enough approximations can practically be found in reasonable time and thus a GNN based on the approximation is able to run in time depending only on the number of nodes (and blocks) instead of linear in the number of edges compared to message passing. 
The implemented GNN achieves very good performance over a number of tasks on relatively large graphs.

### Strengths
- a much improved regularity lemma and corresponding metric that greatly improves over the predecessor architecture
 - a fast enough algorithm to compute the graph approximation
 - complemented by a GNN building on the graph approximation achieving SOTA performance on a variety of datasets
 - there is a vast and clean appendix explaining all the details of the paper and sufficient background.
 - The paper is easy to read for a theory paper

### Weaknesses
Overall, the paper is very easy to read (especially for a theory paper). I only have a few smaller remarks:

- It could be more clear whether the representation in terms of blocks for IBG and ICG is actually the same. If there is a discussion about it in the paper, I missed it and glancing over the ICG paper the difference is not immediately clear. (As, following this question is the exact role of $\textbf{b}$ in the definition of $\textbf P$, as this seems to be the main modification). The description in the appendix could also include pointers to how exactly the differences show themselves in the definitions.
- mapping the statement of Thm 4.1 to the claims about the contributions roughly works, but a 2-line explaination below would have been nice as well. But I guess due to space, this won't be possible.
- in 163 in the definition of the weighted cut: what does the i,j in the normalization factor range over? Is it all of Q or just U,V? (potentially use different indices within and outside of the max to make the difference more clear, given that its indeed ranging over all of Q)
- the workings of the IBG NN compared to a MPGNN is not clear (344ff). Is it doing something else than running a plain directed GNN on the approximated graph, taking into account the low-rank decomposition to make it fast? Or does the model behave differently? This is also relevant for the context in which the "simple and efficient operations" in 411 are to be understood.

Minor:  
- would it be possible to add references to the sources of each line in table 1? This would also make it clear which of the computations are new. (I know that it is described in the text later on, but simple hyperlinks could help as well and in the text)

Typos:  
248: soft indicator model -> soft affilitation model?  
444: kipf and welling -> citep  
461: performance

### Questions
see weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes **Intersecting Block Graphs (IBGs),** low-rank directional graph representation allowing the efficient learning on large and even sparse graphs, by squeezing the adjacency matrix into a smaller matrix regarding communities. Additionally, the authors introduce a densifying cut similarity metric and prove a corresponding weak regularity lemma for their claim. The method is validated with various benchmarks with complexity analysis.

### Strengths
1. Theorems and lemmas introduced (quality, clarity)

Section 4 introduces the semi-constructive weak regularity lemma and its proof, developed upon prior work.

2. Competitive performance (significance)

For node classification, the spatio-temporal graph, and knowledge graph, show competitive performance against the baselines. Additionally, it is relevant for sparse and dense graphs.

3. Efficient architecture (significance)

The proposed method achieves a time complexity proportional to the number of nodes instead of edges, i.e., the complexity of MPNNs.

### Weaknesses
I do not find any critical weakness

Minor

- figure 1 is not mentioned
- line 218 - IBG, IGBS are both used

### Questions
1. SVD initialization (section M)

While the SBD initialization was used against random initialization for faster convergence, I don’t see any performance comparison. Does the performance of two initialization eventually end up in the same?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a method for approximating a large graph that makes the GNN computation on the approximated graph linear with respect to the number of edges, representing a significant improvement over existing graph reduction methods. The approach compresses the original graph by extracting several communities that may overlap and have internal connections represented as bipartite graphs, referred to as blocks or directed communities. Specifically, it employs an extended version of WRL to solve the approximation with strong practical performance, although the theoretical guarantee may be limited.

### Strengths
1. The large computation required by MPNN or GNN on large graphs is indeed a problem, which hinders our understanding and analysis of large social networks, user item interactions in large recommendation networks, and similar systems.
2. The proposed method is not restricted to undirected graphs, unlike ICG. The objective function can be efficiently optimized using gradient descent.
3. The number of communities in the method is independent of any property of the graph, including the number of nodes and the sparsity level.

### Weaknesses
1. The semi constructive nature of the optimization is a problem. Then what guarantees do you have, or at least under what conditions of the graph structure, for instance, would you have confidence in the optimization results with guarantees? This would be essential, even though you claim in my comment strength 3) 
2. Explicitly encouraging density is counterintuitive to me, since it is usually natural to think that in real world graphs, many edges are consequences of other edges. This makes the sparsification of graphs under certain principles appealing. Could you elaborate on your thoughts about this?
3. Does the proposed IBC happen to align with some graph assumptions in real world data? Since in various real world tasks, IBC performs well with its ability to compress the graph by extracting community wise bipartite graphs. Beyond the numerical improvements, it would be very interesting and necessary to include some real world examples in those datasets explaining why condensing the graph in this way would improve performance even beyond the original one, for both heterophilic (I think the performance is very good) and homophilic graphs.

### Questions
1. Typo of the number of the edges in line 38 (not it’s smaller than the number of nodes)?
2. Elaborate on `special interpretation' in line 72.
3. The intuition of the weight cut norm can be introduced in line 83 to improve the readability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel low-rank approximation method for general directed graphs, called Intersecting Block Graph (IBG), designed for efficient graph signal processing. IBG is a non-trivial extension of the Intersecting Community Graph (ICG), overcoming key limitations such as poor performance on sparse graphs and the restriction to undirected graphs. By introducing a densifying cut similarity and an efficient semi-constructive weak regularity lemma, IBG achieves accurate approximations with rank independent of graph size or sparsity. The paper also presents an efficient gradient-based algorithm for fitting IBGs to large directed graphs. Building on this representation, the proposed IBG-NN architecture delivers state-of-the-art results across multiple domains, including node classification, spatio-temporal graph analysis, and knowledge graph completion, while significantly reducing computational complexity.

### Strengths
Overall the work is well motivated and demonstrates strong empirical results. The paper introduces a densifying weak regularity lemma for directed graphs, which improves upon prior formulations and is supported by sound proofs. It demonstrates robustness to sparsity through the use of a weighted Frobenius norm and densifying cut similarity, enabling accurate approximation of sparse graphs as validated by experimental results. Empirically, the proposed IBG-NN architecture achieves state-of-the-art performance across diverse tasks, including node classification, spatio-temporal graph analysis, and knowledge graph completion. Furthermore, the work emphasizes reproducibility by providing a public codebase and detailed hyperparameter settings, ensuring that the results can be reliably replicated.

### Weaknesses
Performance seems depend on the hyperparameters such as $\Gamma$ and $K$, but tuning guidelines are minimal.

### Questions
How sensitive is the performance to the choice of hyperparameters such as $\Gamma$ and $K$ in practice? Any guidelines for setting these?

### Soundness
3

### Presentation
3

### Contribution
3
