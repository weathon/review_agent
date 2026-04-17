# Efficient algorithms for Incremental Metric Bipartite Matching

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
The minimum-cost bipartite matching between two sets of points $R$ and $S$ in a metric space has a wide range of applications in machine learning, computer vision, and logistics. For instance, it can be used to estimate the $1$-Wasserstein distance between continuous probability distributions and for efficiently matching requests to servers while minimizing cost.  However, the computational cost of determining the minimum-cost matching for general metrics spaces, poses a significant challenge, particularly in dynamic settings where points arrive over time and each update requires re-executing the algorithm. In this paper, given a fixed set $S$, we describe a deterministic algorithm that maintains, after $i$ additions to $R$, an $O(1/\delta^{0.631})$-approximate minimum-cost matching of cardinality $i$ between sets $R$ and $S$ in any metric space, with an amortized insertion time of $\widetilde{O}(n^{1+\delta})$ for adding points in $R$. To the best of our knowledge, this is the first algorithm for incremental minimum-cost matching that applies to arbitrary metric spaces.

Interestingly, an important subroutine of our algorithm lends itself to efficient parallelization. We provide both a CPU implementation and a GPU implementation that leverages parallelism. Extensive experiments on both synthetic and real world datasets showcase that our algorithm either matches or outperforms all benchmarks in terms of speed while significantly improving upon the accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the incremental metric bipartite matching problem, where a fixed set of servers S (in a metric space) needs to be dynamically matched to incrementally arriving requests R. The core challenge it addresses is maintaining an approximate minimum-cost matching with efficient update times.

### Strengths
It breaks a key barrier of prior incremental matching algorithms by supporting arbitrary metric spaces, which is a critical generalization for applications like high-dimensional ML (e.g., dynamic alignment of MRI scans) and unbalanced logistics. The push-relabel framework is a clever adaptation: replacing augmenting paths with local push/relabel steps reduces per-insertion overhead, making the algorithm feasible for large-scale systems.

The theoretical analysis is comprehensive. The results are not just theoretical, since practical versions are optimized for real-world use, with batch processing that leverages parallelism to handle high-throughput requests.

### Weaknesses
The main problem is the experiment design. The authors fix server size and delta. And, logistics scenarios require unbalanced R and S, but in the experiments, |S| = |R|. 

One crucial invariant of the algorithm is that a server that is matched at any level 0 ≤ i ≤ µ is only available to requests that are at level i or higher, but this is under-motivated.

### Questions
Your guess-and-double trick requires reprocessing all historical requests when ω  is doubled. For large n (e.g., 100k requests), this could add significant runtime. How many such resets occur in practice for your real-world datasets (MNIST, NYC-Taxi)?

Your GPU implementation is mentioned but not detailed. Which submodules (e.g., distance matrix computation, admissible edge search, slack updates) are parallelized? What speedup do you see over the CPU implementation for each submodule, and is a batch size of 200 optimal (or would larger batches improve GPU utilization)?

Logistics scenarios often have bursty request arrivals (e.g., rush hour for taxis). How does your algorithm handle queue buildup under bursty arrivals? Does batch parallelism mitigate this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies the bipartite matching problem for general metrics in the incremental setting. Specifically, given a fixed point set $S$ and an initially empty point set $R$, the goal is to maintain a matching between $S$ and the current $R$ as new points are incrementally inserted into $R$. The paper presents an algorithm that maintains an $O(1/\delta^{0.631})$-approximate minimum-cost matching in amortized time $O(n^{1+\delta})$, ignoring lower-order terms and assuming that the aspect ratio of the underlying metric space is polynomially bounded.

In the static setting, this problem closely relates to estimating the discrete 1-Wasserstein distance, a fundamental quantity in optimization, applied mathematics, machine learning, and theoretical computer science. It has been extensively studied across many computational models. The dynamic (or incremental) setting considered here—where the underlying point sets evolve over time—is both practically motivated and theoretically rich. The only prior work in this area is by Goranci et al.~(ICML'25), which addresses the more general fully dynamic case (supporting both insertions and deletions) but is restricted to low-dimensional Euclidean metrics.

*Technical Contribution*: The central idea builds on the static approximation algorithm of Agarwal and Sharathkumar~(STOC'14) (derived from the classical Gabow--Tarjan framework). However, it is far from obvious how to \emph{dynamize} that algorithm, even in the incremental setting. Skipping many standard details, the paper’s main novelty lies in replacing the traditional *augmenting-path* step with a *push--relabel* step. While individual point insertions may still be expensive in the worst case, the push--relabel mechanism enables strong amortized guarantees. In hindsight, this is a particularly elegant and insightful design choice. This contribution alone, in my view, justifies acceptance at ICLR.

Last but not least, the paper demonstrates that the conceptual simplicity of the proposed approach translates into competitive empirical performance on both real-world and synthetic datasets.

### Strengths
-- The paper studies a central problem that lies at the intersection of several fields, and is highly relevant to modern machine learning, especially in the context of emerging computational models.

-- The use of push–relabel techniques to design incremental algorithms, in contrast to the traditional approach of searching for augmenting paths, represents the most novel and technically interesting aspect of the paper.

-- The presentation is generally clear and well-organized, with sufficient intuition and explanation provided beyond the core algorithmic details.

### Weaknesses
-- I would expect some better discussion of state-of-the-art resutls on this space (see my comments below:)
-- The result is advertised as applying to general metrics. However, the experiments involve Euclidean data sets. I would expect a bit more effort on this aspect.

### Questions
-- The algorithm by Goranci et al. 2025 does work for arbitrary point insertions and deletions, and it's not restricted to pairs of point insertions or deletions (even though it's not explicitly stated); so claiming that as a weakness doesn't seem appropriate.

-- In related works, the paper says that adapting recent breakthrough fast max-flow techniques in the incremental setting seems challenging; however, this has been done: https://arxiv.org/pdf/2311.03174 in the approximate setting (and the paper under review also works in the approximate regime); it still doesn't solve the problem considered in the paper, but some discussion would be due here.

-- Can you explain why you decided to consider only Euclidean data sets in the experiments?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers the \emph{metric bipartite matching problem} in the incremental setting: given two sets of points in a metric space undergoing insertions, the goal is to maintain a minimum-cost matching of one set to the other, where the cost is measured in terms of distances in the metric space, while minimizing the time spent updating the output. The authors present an algorithm achieving an $O(1/\delta^{\sim 0.631})$-approximation to the problem with $\tilde{O}(n^{1+\delta})$ update time.

The only comparable result is that of Goranci et al. (ICML'25), who obtain an $O(1/\epsilon)$-approximation for the problem with $\tilde{O}(n^{O(\epsilon)})$ update time under both point insertions and deletions. However, their algorithm is restricted to low-dimensional Euclidean metrics. Minimum-cost bipartite matching is much more challenging in general metrics, even in the static setting. The result presented by the authors is, to the best of my knowledge, the first algorithm addressing the problem in the dynamic setting for general metrics.

The algorithm presented by the authors closely follows that of Agarwal and Sharathkumar (STOC'14), which achieves the same approximation ratio for the static problem in $O(n^{2+\delta})$ running time (roughly equivalent to the total running time of the algorithm presented by the authors over the entire insertion sequence). The algorithm of Agarwal and Sharathkumar, in turn, builds on the influential work of Gabow and Tarjan (SIAM), who developed a scaling algorithm for exact minimum-cost bipartite perfect matching in graphs, running in $\tilde{O}(n^{2.5})$ time and based on augmenting path elimination techniques. At a high level, Agarwal and Sharathkumar modify this algorithm by approximately completing its scaling steps, resulting in an approximate solution with an almost $n^2$ running time.

The algorithm presented by the authors can be viewed as an incremental implementation of that of Agarwal and Sharathkumar. A crucial challenge in this approach is that the underlying algorithm relies on augmenting path elimination techniques, which are difficult to implement in the dynamic setting, even for maximum cardinality graph matching, when the cost of the optimal solution does not evolve monotonically, as is the case in Euclidean matching. The main technical contribution of the paper lies in overcoming this challenge by formalizing the algorithms of Agarwal and Sharathkumar (and Gabow and Tarjan) as a push-relabel framework, which naturally adapts to the dynamic setting.

The authors also present experiments comparing the performance of their algorithm to greedy and quad-tree-based implementations (Sariel Har-Peled) with respect to Euclidean norms on the MNIST, NYC-TAXI, and synthetic datasets. The results show predictable behaviour in terms of approximation: the algorithm proposed in the paper outperforms both the greedy and quad-tree-based implementations (although it is unclear how it compares to the optimal solution). In terms of update time, the batch-update variant of the proposed algorithm significantly outperforms the greedy approach and slightly underperforms the quad-tree-based implementation.

Overall, I consider the paper to be an important contribution to the study of the widely applied metric minimum-cost bipartite matching problem in the dynamic setting, which has recently received growing attention, due to being the first paper considering the problem in general metrics.

### Strengths
The formulation of a variant of the Agarwal and Sharathkumar (and Gabow and Tarjan) algorithm as a push-relabel framework is interesting and, I suspect, will be of independent research interest.

### Weaknesses
The algorithm (in contrast to that of Goranci \emph{et al.}) is designed to handle only point insertions. This is somewhat surprising. While for most problems the incremental and decremental settings have proven to be simpler than the fully dynamic one, one might expect this not to be the case for metric minimum-cost matching, as the objective value evolves non-monotonically under partially dynamic updates.

In terms of experimental results, a comparison to the work of Goranci \emph{et al.} would be interesting to see. Based on the theoretical guarantees, with the same slack parameter, one would expect the latter to be more time-efficient at the cost of some loss in approximation ratio relative to the algorithm proposed in this paper. However, since quad-tree-based techniques (such as those of Goranci \emph{et al.}) tend to perform better in practice than in theory, it would be important to see that (at least on low-dimensional Euclidean datasets) the algorithm presented in this paper strictly outperforms the previous approach in at least one aspect.

Furthermore, an experiment not restricted to Euclidean distances (preferably on non-synthetic data) would intuitively should demonstrate a significant improvement over the greedy algorithm in terms of approximation ratio.

### Questions
My questions would concern the performance of the algorithm for the test cases described in the weaknesses section.

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
2

### Summary
The paper presents the first streaming and batch-adaptive algorithm maintaining a constant-factor approximation the metric bipartite matching problem. The described algorithm builds upon a previously known algorithm for the offline setting, while making sure that update steps can be executed quickly (when one does amortized time analysis), whereas the worst case is of higher update time. The authors achieve this by utilizing a data structure tailor made for these updates, and by maintaining a collection of partial matchings with a specific property that are a good approximation of the desired matching. Each partial matching corresponds to a different "discretization" of the metric space, at different degrees of refinement. The authors support their theoretical results by empirical experiments.

### Strengths
The paper is an important contribution in the field of online-algorithms and metric-driven problems. It is well motivated, and builds upon interesting known ideas in a novel way. The introduction of the "push-relable" subroutine instead of augmenting paths, as well as the careful analysis of the "Find-Admissable-Edge" subroutine allow a clean generalization of the static algorithm. The authors present these deviations well, which contributes to the merit of the paper. The analysis is relatively straightforward, but the ideas are non trivial.

### Weaknesses
The paper does not stress in the main body where the assumption of Metricity is necessary, only in the analysis. Since this assumption is critical - a mention is in need.
The experiments need a little more content. The synthetic data analysis seems to be missing (or I have missed it). More importantly, the authors stress that this algorithm is not restricted to the assumption of a Euclidean metric. A comparison with a synthetic dataset that is not Euclidean would be interesting, and will support the strengths of the paper as well.

### Questions
Where does the algorithm break if the distance function is not metric?
A couple of typos:
page 4, line 189: $\theta$ should be $\Theta$.
page 5, line 247: $B^i$ should be $B^j$.

### Soundness
4

### Presentation
4

### Contribution
3
