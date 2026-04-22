# Canonical Tree Cover Neural Networks for Expressive and Invariant Graph Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
While message-passing NNs (MPNNs) are naturally invariant on graphs, they are fundamentally limited in expressive power, oversmooth, and oversquash. Canonicalization offers a powerful alternative by mapping each graph to a unique, invariant representation on which expressive non-invariant encoders can operate. However, existing approaches rely on a single canonical sequence that distorts graph distances and restricts expressivity. To address these limitations, we introduce Canonical Tree Cover Neural Networks (CTNNs), which represent the graph with a canonical spanning tree cover. Each tree is then processed with an expressive tree encoder. Theoretically, tree covers better preserve graph distances in comparison to sequences, and on sparse graphs, the cover recovers all edges with a logarithmic number of trees in the graph size, making CTNNs strictly more expressive than sequence-based canonicalization approaches. Empirically, CTNNs consistently outperform invariant GNNs and sequence-based canonical GNNs across sparse molecular and protein graph classification benchmarks. Overall, CTNNs advance graph learning by providing an efficient, invariant, and expressive representation learning framework on sparse graphs via tree cover-based canonicalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Canonical Tree Cover Neural Networks (CTNNs) as a new approach to graph canonicalization. Instead of flattening graphs into a single sequence (which causes distortion and expressivity loss), CTNNs construct a canonical spanning tree cover and process each tree with expressive tree encoders, aggregating their outputs into an invariant representation. The authors provide theoretical results (distance preservation, expressivity beyond 1-WL, universality under certain conditions) and empirical evaluation on molecular and protein benchmarks, showing improvements over message-passing GNNs, sampling approaches, and sequence canonicalization baselines.

### Strengths
1. The paper clearly explains the limitations of sequence canonicalization, illustrating them with intuitive examples such as star graphs.
2. It introduces a tree cover method that better preserves structural information and invariance, supported by formal analyses of distortion, expressivity, and coverage guarantees.
3. On multiple benchmarks, CTNNs consistently surpass strong baselines, achieving notable gains on molecular tasks and competitive results on protein tasks.

### Weaknesses
1. The key idea (use a set of trees instead of one sequence) feels like an incremental extension rather than a major conceptual advance. Many theoretical results (universality, expressivity boost by multiple views, distortion comparisons) are natural consequences of existing work. Frequent use of terms like “strictly more expressive,” “provably invariant,” and “universal” gives an impression of overselling. Some proofs (e.g., universality of CTNNs) rely on standard functional approximation arguments, which are not truly new.

2. The guarantees, such as using only a logarithmic number of trees and achieving low distortion, mainly hold for sparse graphs. Dense graphs, such as proteins, show weaker improvements and less favorable theoretical bounds. Although the paper claims MST construction is efficient, it does not provide runtime or memory benchmarks. For dense graphs, the (O(Km \log n)) cost could be significant, and the preprocessing advantage is asserted but not supported with quantitative evidence.

3. Improvements on molecular benchmarks are often only 1–2 AUC points. On protein tasks, performance is inconsistent, sometimes close to baselines. No large-scale or real-world datasets beyond the standard benchmarks are tested. The ablation studies show predictable drops (removing key components hurts), but provide little insight into why certain parts matter. No analysis of failure cases or adversarial graph structures.

4. While molecular and protein benchmarks are reasonable, the method is not shown on social networks, knowledge graphs, or large heterogeneous graphs. This raises questions about generality beyond biochemical datasets.

### Questions
I will raise my score if the authors address W2.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper critiques single-sequence canonicalization methods in graph learning for causing distance distortion and having limited expressivity. To address this, it introduces Canonical Tree Cover Neural Networks (CTNNs), a framework that represents graphs using a small set of canonical spanning trees that cover all edges. Each tree is processed by a tree encoder, and the results are aggregated. The authors provide theoretical guarantees that CTNNs are probabilistically invariant, better preserve distances, and are more expressive than sequence-based methods. Empirically, CTNNs are shown to outperform standard GNNs, sampling approaches, and canonical sequence baselines on graph classification benchmarks.

### Strengths
1. The paper addresses a fundamental problem in graph learning: the trade-off between expressivity and isomorphism invariance.
    
2. It proposes a framework (CTNN) that uses a tree cover to represent graph structure, addressing the identified high distortion of sequence-based methods.
    
3. Theoretical analysis and empirical results across multiple benchmarks provide support.

### Weaknesses
1.  The experimental setup (e.g., using $\tau=1$) likely fails to meet the theoretical condition required by Lemma 5.3 for guaranteed logarithmic edge coverage. This gap between theory and practice undermines the paper's claims about efficient coverage and the universality (Thm 5.5) that depends on it.

2.  The empirical evaluation is missing the most critical baselines. As a method based on subgraph representations, CTNN must be compared against other state-of-the-art, subgraph-based GNNs (like GSN or ESAN) that also achieve high expressivity, not just standard 1-WL models.

3.  The universality claim (Thm 5.5) is trivial and not a unique advantage. This property holds for any model that completely decomposes a graph and uses universal encoders and aggregators. The paper does not sufficiently prove the *efficiency* of CTNN's universality (i.e., that it can be achieved with a small $K$).

### Questions
See in Weaknesses.

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
3

### Summary
The paper introduces Canonical Tree Cover Neural Networks (CTNNs), a new framework for graph representation learning that generalizes canonical sequence models by replacing a single canonical representation with a set of canonical spanning trees (MSTs).
Each tree is processed with a recurrent tree encoder, and message passing is applied over residual (non-tree) edges to capture local connectivity missed by individual trees.
The authors further provide:
(1) Theoretical results on probabilistic invariance and universality of CTNNs,
(2) Distortion and expressivity bounds comparing CTNNs to sequence-based canonicalization,
(3) Empirical evaluations on molecular and protein benchmarks, demonstrating improved performance.

### Strengths
(1) The paper presents a structured extension of canonical graph neural networks by introducing Canonical Tree Cover Neural Networks (CTNNs), which replace a single canonical ordering with a collection of spanning trees. The proposed method alleviates the structural distortion and expressivity limitations commonly observed in sequence-based canonicalization approaches.
(2) The paper provides a series of theoretical analyses, including probabilistic invariance, expected distortion bounds, and expressivity results, demonstrating a well-grounded theoretical foundation.

### Weaknesses
(1) Although the paper presents CTNN as an innovation grounded in canonicalization, its underlying modeling paradigm bears conceptual similarity to prior tree-structured graph neural networks (GNNs), such as Neural Trees for Learning on Graphs (Talak et al., 2021). Both approaches transform a graph into a hierarchy of trees for recursive message aggregation. As a result, the conceptual novelty of CTNN appears limited. 
(2) Dependence on root and tree structure without theoretical guarantees. The model’s expressive capacity and stability appear highly sensitive to the root selection and tree shape. Since the recursive encoder (e.g., Tree-LSTM) is order- and hierarchy-dependent, different rootings or unbalanced spanning trees may yield substantially different representations. The paper currently provides no theoretical or empirical analysis of this effect.
(3) Dependency on the Initial Labeler ${\pi}_{V}$: The paper criticizes sequence methods (Prop 3.3) for being limited by ${\pi}_{V}$'s expressivity. However, CTNN's own tree generation (Alg. 1) is also initialized by ${\pi}_{V}$. If ${\pi}_{V}$ is weak (e.g., cannot distinguish 1-WL-equivalent nodes), the initial MST selection will also be "blind.
(4) Missing/Failed Key Baselines: A core claim is surpassing MPNN expressivity. A fair comparison requires stronger baselines like k-WL GNNs or Graph Transformers (GT). The paper includes GT, but it timed out (OOT) on all protein datasets where CTNN excelled. The lack of results from this key high-expressivity baseline makes CTNN's victory less convincing.

### Questions
(1) Considering the conceptual resemblance between CTNN and Neural Trees for Learning on Graphs (Talak et al., 2021), both of which convert graphs into hierarchical tree representations for recursive message aggregation, it would be valuable for the authors to include a direct conceptual or empirical comparison in the paper.
(2) Why were GNNs with proven higher expressivity than 1-WL (e.g., k-WL GNNs like GSN, or subgraph GNNs like PPGN) not included as baselines? These seem like the most relevant competitors for a method claiming to surpass 1-WL limitations.
(3) Given that the paper's core motivation is to overcome the expressivity limits of 1-WL, why were synthetic benchmarks (like CSL or EXP) completely avoided? These datasets are specifically designed to demonstrate 1-WL failures and would be the clearest way to validate the superior expressivity of CTNN.
(4) Given that recursive encoders (e.g., Tree-LSTM) are inherently order- and hierarchy-sensitive, how robust is CTNN to different root node selections or unbalanced tree shapes? Do the authors have any empirical results on performance variance under random root permutations or tree rebalancing?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies limitations in existing graph canonicalization methods, particularly those that flatten graphs into sequences, arguing they introduce significant distance distortion and are bottlenecked by the expressivity of their node labelers. To address this, the authors propose Canonical Tree Cover Neural Networks (CTNNs), an invariant framework that represents each graph as a small set of canonical spanning trees. This tree cover is generated using an iterative, coverage-aware Minimum Spanning Tree (MST) algorithm. Each tree is then processed by an expressive tree encoder, and the results are aggregated. The authors provide theoretical contributions showing their method is (probabilistically) invariant, better preserves graph distances than sequence-based methods, and is strictly more expressive than both MPNNs and sequence canonicalization. Empirically, the paper demonstrates that CTNNs outperform standard GNNs, sampling approaches, and canonical sequence baselines on several graph classification benchmarks.

### Strengths
1. The paper is well-organized and clearly articulates the limitations of existing methods.
    
2. The proposed CTNN framework is a novel and intuitive solution that aims to tackle the issues by replacing the single-sequence representation with a more structurally tree cover.
    
3. The claims are supported by a combination of theoretical analysis and robust empirical results across diverse benchmarks.

### Weaknesses
1.  The term "canonicalization" is misleading. The method is not deterministic; it is a "probabilistically invariant sampling" algorithm that relies on random tie-breaking, making it conceptually closer to the sampling-based (e.g., RWNN) paradigms it critiques.
2.  There is a gap between the theoretical requirement for invariance (taking an expectation $\mathbb{E}[\cdot]$ over all possible random choices) and the practical implementation (aggregating a small sample of $K=4$ or $K=8$ trees). This small $K$ may be insufficient to approximate the expectation, leading to unstable representations for the same graph across different runs.
3.  A key theoretical justification (Thm 5.2) for the method's low distance distortion is based on Uniform Spanning Trees (USTs). However, the proposed algorithm (Alg 1) generates Minimum Spanning Trees (MSTs) from a different, non-uniform distribution, meaning the core distortion theory does not actually apply to the method as practiced.
4.  For highly regular graphs (e.g., where all nodes have the same degree $\pi_V$, resulting in identical initial edge weights ${\pi}_{E}^{(0)}$), the selection of the MST in Algorithm 1 will be determined entirely by the random tie-breaker $\zeta$. This suggests the process is not learning a "canonical" structure but rather performing a structured random decomposition of the graph.

### Questions
See in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
