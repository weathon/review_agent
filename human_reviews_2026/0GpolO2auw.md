# Sublinear Spectral Clustering Oracle with Little Memory

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
We study the problem of designing *sublinear spectral clustering oracles* for well-clusterable graphs. Such an oracle is an algorithm that, given query access to the adjacency list of a graph $G$, first constructs a compact data structure $\mathcal{D}$ that captures the clustering structure of $G$. Once built, $\mathcal{D}$ enables sublinear time responses to \textsc{WhichCluster}$(G,x)$ queries for any vertex $x$. A major limitation of existing oracles is that constructing $\mathcal{D}$ requires $\Omega(\sqrt{n})$ memory, which becomes a bottleneck for massive graphs and memory-limited settings. In this paper, we break this barrier and establish a memory-time trade-off for sublinear spectral clustering oracles. Specifically, for well-clusterable graphs, we present oracles that construct $\mathcal{D}$ using much smaller than $O(\sqrt{n})$ memory (e.g., $O(n^{0.01})$) while still answering membership queries in sublinear time. We also characterize the trade-off frontier between memory usage $S$ and query time $T$, showing, for example, that $S\cdot T=\widetilde{O}(n)$ for clusterable graphs with a logarithmic conductance gap, and we show that this trade-off is nearly optimal (up to logarithmic factors) for a natural class of approaches. Finally, to complement our theory, we validate the performance of our oracles through experiments on synthetic networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the task of community detection in well-clusterable graphs with sublinear space. The goal is to design a data structure D that fits in sublinear memory, and that enables one to query the cluster assignment for each node in sublinear time. Previous approaches all require $\Omega(\sqrt{n})$ space for such a datastructure D, but this paper overcomes that. In particular, it is able to design a data structure with a much smaller memory requirement that still allow for sublinear time which-cluster queries. The paper also provides new insights into the time-space tradeoff for this problem, by designing oracles there memory usage S and query time $T$ satisfy $S\cdot T \approx \tilde{O}(n)$. Again, this holds for a class of graphs with good clustering structure. The paper also proves that this is optimal up to logarithmic factors for a certain class of techniques.

The notion of well-clusterable graphs corresponds (roughly) to graphs that have a k partition where clusters are roughly balanced in size and have small conductance (and large inner conductance, which measures internal connectivity of clusters).

The paper also proves new results (sublinear algorithms and lower bounds) for the 1-cluster/2-cluster problem, which seeks to tell the difference between graphs that are expenders on n nodes or that are disjoint unions of two identical expanders on n/2 nodes. 

For the clustering results, the key technical advance is to provide a new way to estimate the dot product between the spectral embedding of two nodes in sublinear space and time (the spectral embedding for a node comes from the node's entries in the first few eigenvectors of the normalized Laplacian). This primitive is combined with a previously observation that if two nodes are from the same cluster (under the well-clusterability assumptions), then the dot product of their embeddings with be large, and otherwise the dot product will be zero.

### Strengths
* This paper provides good motivation and justification for sublinear time and space algorithms, and tackles and interesting problem in graph analysis. 
* The main theoretical result is interesting and represents a substantial improvement over previous techniques for sublinear space clustering oracles. In particular, it breaks the $\Omega(\sqrt{n})$ space barrier limiting previous approaches. The results on algorithms and the lower bound for the 1-cluster/2-cluster problem are a nice bonus addition to the paper. The theoretical results are highly non-trivial. 
* The structure and writing of the paper are excellent; about as good as one could hope for a paper whose core contribution is so dense and complicated. The paper did a very good job motivating the problem, explaining the key contributions and their significance, and communicating the main technical components that made them work. I learned a lot by reading this paper, despite how theoretically dense it is. The presentation is very, very good. 
* I really appreciate that the authors have even provided an implementation of the algorithm and accompanied their theory with numerical experiments. Even if it's only on synthetic data, it is impressive to have an implementation of any kind for an algorithm that is so detailed and complicated.

### Weaknesses
The main downsides of this paper, when considering it for publication at ICLR specifically are: (1) the algorithm is extremely complicated and detailed, and (2) the algorithm is very impractical for real-world clustering problems. I don't this at all disqualifies the article from being accepted and published somewhere. The contribution here seems very impressive. I do wonder though whether ICLR is the right venue for this work, and instead of a core theory conference. For example, the main work this improves upon (Peng 2020; Gluche et al 2021) are both SODA papers.

Expounding more on the two weaknesses above:

(1) Although the writing does an impressive job making the pieces of the algorithm make sense at a high level, there is still no way around how intricate the statements of the results are. Many many different functions and parameters need to be defined---all with complicated dependencies on each other---even just to present the statement of the results, without even considering what it takes to prove the results. 

(2) The theory assumes d-regular graphs with a very specific clustering structure that is not going to be satisfied by pretty much any real world graph. Expanding this to d-bounded graphs doesn't make it much more practical. Even if a graph does satisfy the well-clusterability assumptions for certain choices of $k, \varepsilon, \varphi$, we wouldn't know these a priori, and the algorithm make strong assumptions about these parameter (just one among many examples: the specific need in Theorem 3.1 for $\varepsilon /\varphi \leq 1/10^5$). 

This makes it all the more impressive that the paper includes an experimental result of any kind, but in order for this to work (even in a very carefully controlled synthetic setting), one needs to try out many different parameters for the algorithms. 

I'm still overall in favor of seeing this paper accepted, given the significance of its core technical contribution and its many other strengths. Graph clustering in general seems well within scope at the conference, and the fact that there are at least some numerical experiments is a plus.

### Questions
Is there hope for something like this to be practical for real world graph clustering? What would it take to make that happen?

How long did it take to construct $\mathcal{D}$ in practice for your numerical experiments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studied the construction of spectral clustering oracles on well-clustered graphs with limited memory. The problem has recently attracted a flurry of work due to its applications in sublinear clustering algorithms. Here, we are given a graph that could be partitioned into $k$ clusters where the conductance between the clusters are high and the conductance inside the clusters are low. As such, we could label the vertices to generate a ‘ground truth’ clustering. The goal for the algorithm is to compute a data structure such that upon querying a vertex $x$, the algorithm can answer the cluster label of $x$ with high efficiency. The metrics for good algorithms in this application include:
- Pre-processing time: the time to construct the data structure
- Querying time: the time complexity needed to return the answer for each cluster
- Accuracy: The answer for most of the vertex queries should be correct
- Memory efficiency: the memory used by the data structure should be small

The last aspect was the main contribution of this paper. The paper discussed that all previous algorithms require $\Omega(\sqrt{n})$ space for such applications; in contrast, this algorithm is able to design an algorithm with only $n^{O(\varepsilon/\phi^2)}$ space, where $\varepsilon$ and $\phi$ are parameters that characterize the clusterability of the graph. The query time will be affected, which is now $n^{1+O(\varepsilon/\phi^2)}$ time. In fact, the trade-off could be made general with $n^{O(\varepsilon/\phi^2)}M$ space and $n^{1+O(\varepsilon/\phi^2)}/M$ time.

**Main techniques.** The main techniques of the paper follow from the construction in Shen and Peng [NeurIPS’23]. In a nutshell, this line of techniques reduces the algorithm for the spectral clustering oracle to the approximation of the dot products of vertex embeddings. The previous space lower bound is due to the computation of the approximation dot product using random walks, and this paper adopted the simple idea to conduct the walk in batches to trade time efficiency for space efficiency.

### Strengths
I’m generally supportive of the paper. The spectral clustering oracle problem has attracted quite some attention over the past few years, and it is great that the space aspect is taken into consideration in this paper. I did not get the time to verify the correctness of the results, but the technique overview provided some good justifications for first-time readers to believe the correctness. Some experiments are also provided besides the theoretical results.

### Weaknesses
On the flip side, I think the paper could do a better job in terms of the comparison between their results and existing algorithms.

Judging from the presentation of the main results, it is not entirely clear whether certain restrictions (e.g., $d$-bounded graph) are also used in previous results, and how the misclustering error would compare with oracles with no space limits. 

Furthermore, it is not always clear which part of the algorithm follows from existing work, and which part is the contribution of this paper. The techniques are not thoroughly compared with similar papers. Since I do not know the techniques in those papers, it is harder for me to evaluate the technical novelty.

Overall, I think it’s a solid paper with a good set of results. However, the writing issues and the fact that I’m not very familiar with related techniques make it hard for me to champion for it.

### Questions
Most of the questions are embedded in the weakness comments. Some additional questions and comments:

Line 150: The notion of ‘random walk queries’ is not defined. Also, what does Theorem 1.3 mean? The lower bound works only against algorithms that rely exclusively on random walk queries, and cannot make, e.g., degree or neighborhood queries, on adjacency lists? This family of algorithms seems to be extremely restrictive. 

Line 262-263: We use $O_\phi$ *to suppress*; similarly, $\tilde{O}$ *to hide*.

The leading constants in $n^O(\varepsilon)$ appear to be crazily large for any interesting applications. In your experiment, did you conduct some type of algorithm engineering to bring down the actual time and space?

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
This paper considered the problem of designing sublinear spectral clustering oracles for well-clustered graphs. The authors assume query access to the adjacency list of the graph. They have given a space-time tradeoff for this problem, and also showed this tradeoff is tight for approaches using only random walk oracles. One of the interesting feature of this work is that their algorithm has space complexity of $o(\sqrt{n})$, in contrast to previous algorithms which require $\Omega(\sqrt{n})$ space.

### Strengths
Strengths:

1. They have given a space-time tradeoff for designing sublinear spectral clustering oracles. 

2. They have shown that this tradeoff is tight for approaches that use only random walk oracles

### Weaknesses
1. (Upper bound) The main technical contribution of this work is the construction of an efficient inner-product oracle (Theorem 3.2) for spectrally embedded vertex vectors. However, the details of this construction are not presented in the main text. The authors only sketch those proofs that primarily follow the approach of Shen & Peng (2023).

2. (Lower bound) While the proof of the space–time tradeoff lower bound is nice, it would have been much more interesting if the lower bound had been established under adjacency list query access, which is the more natural and standard setting.

3. I suggest moving the experimental section to the appendix and instead including a more detailed proof outline for Theorem 3.2 in the main text.

4. Although the tradeoff result is nice, it is unclear whether the resolved question was explicitly open in the prior literature. Clarification on this point would be helpful.

5. The authors did not adequately highlight the novel ideas distinguishing their techniques from previous works.

### Questions
1. The authors should highlight the new ideas in their technique compared to previous works. 

2. What is the main bottleneck to extending the lower bound to adjacency oracle query access?

3. Was the tradeoff an explicit open problem in the prior literature?

### Soundness
2

### Presentation
2

### Contribution
2
