# Learning to Segment for Vehicle Routing Problems

- Avg Score: 5.00
- Decision: Accept (Oral)
- Scores: 4, 4, 6, 6

## Abstract
Iterative heuristics are widely recognized as state-of-the-art for Vehicle Routing Problems (VRPs). In this work, we exploit a critical observation: a large portion of the solution remains stable, i.e., unchanged across search iterations, causing redundant computations, especially for large-scale VRPs with long subtours. To address this, we pioneer the formal study of the First-Segment-Then-Aggregate
(FSTA) decomposition technique to accelerate iterative solvers. FSTA preserves stable solution segments during the search, aggregates nodes within each segment into fixed hypernodes, and focuses the search only on unstable portions. Yet, a key challenge lies in identifying which segments should be aggregated. To this end, we introduce Learning-to-Segment (L2Seg), a novel neural framework to intelligently
differentiate potentially stable and unstable portions for FSTA decomposition. We present three L2Seg variants: non-autoregressive (globally comprehensive but locally indiscriminate), autoregressive (locally refined but globally deficient), and their synergy. Empirical results on CVRP and VRPTW show that L2Seg accelerates state-of-the-art solvers by 2x to 7x. We further provide in-depth analysis showing why synergy achieves the best performance. Notably, L2Seg is compatible with traditional, learning-based, and hybrid solvers, while supporting various VRPs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes First-Segment-Then-Aggregate (FSTA), a decomposition framework for large-scale VRPs, motivated by the observation that in iterative optimization methods, a large portion of the intermediate solution structure tends to remain stable across iterations.
To effectively identify which parts of the solution should be modified and which should remain fixed, the authors introduce a neural network–based module called Learning-to-Segment (L2Seg), which enables efficient segmentation within FSTA.
Through experiments on CVRP and TSPTW, the authors demonstrate that integrating L2Seg can accelerate existing iterative solvers by a factor of approximately 2×–7×, while maintaining or improving solution quality.

### Strengths
- The proposed method successfully restricts the search space and achieves higher solution quality within the same computational budget on large-scale CVRP and TSPTW instances.

- The framework is solver-agnostic and can be applied to multiple backbone iterative solvers, demonstrating its flexibility.

- The experimental evaluation is comprehensive, including comparisons against diverse baselines from different methodological categories.

### Weaknesses
- Since L2Seg is trained in a supervised manner, its performance may degrade when the distribution of intermediate solutions during inference differs significantly from that of the training data.

- The behavior of L2Seg appears conceptually closer to search-space restriction rather than genuine problem decomposition. Consequently, there is a risk that the restricted search could lead to premature convergence to local optima, preventing the discovery of globally superior solutions.

- The model is trained and evaluated on problem instances of the same size distribution, leaving it unclear whether L2Seg generalizes to unseen problem scales. If it fails to generalize, collecting sufficient training data for larger instances could become computationally prohibitive, limiting the practical applicability of the approach.

- (Minor) Typo found at line 130.

- (Minor) References in lines 73–75 should be enclosed in parentheses.

### Questions
- How do the authors precisely define “iterative solvers” in the context of this paper?

- The paper states that “a large portion of the solution remains stable.” Does this mean that most parts of the solution are not updated because they do not significantly contribute to objective improvement, or because only a small subset of solution elements is actually targeted for update by the iterative process?

- Does L2Seg generalize to problem sizes that were not seen during training? If not, how feasible is it to train the model for significantly larger problem instances?

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
This paper proposes Learning-to-Segment (L2Seg), a learning-based framework that accelerates iterative heuristics for Vehicle Routing Problems (VRPs). The key idea is to detect stable and unstable segments in existing solutions, aggregate stable portions into hypernodes via a First-Segment-Then-Aggregate (FSTA) decomposition, and then re-optimize only the unstable parts.
The authors design three neural variants (NAR, AR, and SYN)  and demonstrate empirical speedups over baseline solvers such as LKH-3, LNS, and L2D, on both CVRP and VRPTW benchmarks.

### Strengths
- The empirical study is well executed, covering multiple VRP variants, backbone solvers, and problem scales (1k–5k). Ablation studies, oracle comparisons, and visual analyses provide convincing evidence that the proposed framework accelerates iterative search.

- The proposed framework is well documented, including pseudocode and architectural details. The authors have made strong efforts to ensure reproducibility and practical relevance.

### Weaknesses
- Limited conceptual novelty: the proposed framework can be viewed as a natural neural extension of existing decomposition-based heuristics. While the idea of stability is interesting, the framework essentially replaces a search space of LNS.
- Restricted theoretical contribution: the theoretical analysis is limited to the monotonicity of the FSTA reduction — that improving a reduced problem implies improving the original one. However, the paper lacks theoretical analysis regarding solution quality guarantees or learnability.
- Dependence on heuristic supervision: the training labels rely on the lookahead procedure with heuristic solvers. This raises concerns about the correctness of the label.

### Questions
- Are there any justifications of the label about stability? It is not entirely clear whether the notion of stability—as defined by differences between consecutive heuristic solutions—is reliable.
- Are there any baselines using LNS with ML-based neighborhood selection? If such a baseline exists, the efficiency of proposed method can be more convincingly positioned as an improved (or generalized) version of ML-enhanced LNS.

### Soundness
3

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
4

### Summary
This paper aims to address the problem of redundant computation in iterative VRP solvers. The authors observe that during the iterative search process, a large portion of the solution stabilizes and no longer changes, yet the solver still consumes computational resources on these stable parts. To address this, the paper proposes FSTA (First-Segment-Then-Aggregate), a formalized decomposition framework with a full theoretical proof, to identify stable segments in the solution, aggregating their nodes into fixed hypernodes and then focusing the search only on the reduced unstable parts. Furthermore, this paper proposes L2Seg (Learning-to-Segment), a novel neural network framework for segmentation identification, including network architecture, training, and inference processes. Empirical results show that L2Seg can achieve speedups of 2 to 7 times for existing classical and learning-based solvers.

### Strengths
1. The paper's proposed method of learning to identify and freeze stable segments to accelerate iterative search is intuitive and novel. The proposed method is promising and achieves SOTA performance on most of the testing cases.
2. The FSTA framework is technically sound and empirically robust. It is formalized, and the authors provide theoretical proofs of its feasibility and monotonicity across various VRP variants (CVRP, VRPTW, VRPB, etc.).
3. The authors tested L2Seg on three different and representative backbone solvers (LKH-3, LNS, L2D), demonstrating its flexibility and versatility. Ablation experiments strongly support the necessity of the learned component (compared to stochastic FSTA).

### Weaknesses
1. The L2Seg-SYN process seems a bit complicated. It is unclear how much time is consumed by the L2Seg-SYN prediction step. If the L2Seg-SYN prediction step itself is costly, the 2x-7x speedup may only be noticeable over long runs.
2. While L2Seg has been successfully applied to LKH-3, LNS, and L2D, Appendix B.1.4 mentions that applying it to HGS (another top-level solver) requires modifying the HGS source code, which is left for future work. This is a reasonable limitation, but it means that L2Seg is not "plug and play" for all iterative solvers.

### Questions
1. Would better look-ahead heuristics lead to a better L2Seg model and ultimately better solution quality?
2. How does the time spent on the L2Seg-SYN prediction step compare to the time spent running the backbone solver in a single iteration?

Minor:

3. Some double quotes are different, such as the one in line 252.
4. Lines 47-48 write: FSTA identifies stable segments and then aggregates them as fixed hypernodes. But lines 146-147 write: FSTA segments the VRP solutions by identifying unstable portions, and then groups them into hypernodes. It seems there is a typo.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Learning-to-Segment (L2Seg), a novel learning-guided framework designed to accelerate iterative solvers for large-scale VRPs. It first formalizes a First-Segment-Then-Aggregate (FSTA) decomposition technique, which identifies stable segments in a solution and aggregates them into hypernodes, thereby reducing the problem size for re-optimization. It also employs neural models to predict unstable edges. The paper presents three variants of L2Seg: a non-autoregressive (NAR) model for global prediction, an autoregressive (AR) model for local precision, and a synergized (SYN) model that combines their strengths. Extensive experiments on CVRP and VRPTW show that L2Seg accelerates state-of-the-art solvers by 2x to 7x while maintaining or improving solution quality.

### Strengths
(1) The paper is generally well-written and structured.

(2) The novel FSTA framework and the specific problem of learning to segment for decomposition are a fresh perspective.

(3) The FSTA framework is well-motivated, and its theoretical properties (feasibility and monotonicity) are formally proven for multiple VRP variants. 

(4) Experiments are comprehensive, testing on large-scale problems, multiple backbone solvers (classic, neural, hybrid), and various VRP types, demonstrating robust performance and generalizability.

### Weaknesses
1. While the paper demonstrates broad applicability, a more explicit discussion of the boundaries of FSTA/L2Seg's effectiveness would be beneficial. For example, under what conditions (e.g., problem size, structure, solver type) might the overhead of segmentation and aggregation outweigh the benefits?
2. It is unclear that the boundary of the acceleration, as I notice that the HGS and LKH3 only run for a short time (5m and 10m). If the solving time extends, how will the acceleration benefit change in terms of both solution quality and time?
3. Some works are also based on hypergraphs, but this paper does not discuss [1, 2].


[1] A hierarchical destroy and repair approach for solving very large-scale travelling salesman problem. https://arxiv.org/pdf/2308.04639.

[2] Destroy and Repair Using Hyper-Graphs for Routing. https://ojs.aaai.org/index.php/AAAI/article/view/34018

### Questions
1. Can the author explain how to deal with the disconnected node after the insertion (shown as the initial node in the last subfigure in Figure 3)?

2. Can the author explain why not evaluate the proposed L2Segment on another two problems, VRPB and 1-VRPPD, as the FSTA is already theoretically verified on these two problems?

3. Please also see the weaknesses and clarify them.

### Soundness
3

### Presentation
3

### Contribution
3
