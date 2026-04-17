# Contraction and Hourglass Persistence for Learning on Graphs, Simplices, and Cells

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 2

## Abstract
Persistent homology (PH) encodes global information, such as cycles, and is thus increasingly integrated into graph neural networks (GNNs). PH methods in GNNs typically traverse an increasing sequence of subgraphs. In this work, we first expose limitations of this inclusion procedure. To remedy these shortcomings, we analyze contractions as a principled topological operation, in particular, for graph representation learning. We study the persistence of contraction sequences, which we call Contraction Homology (CH). We establish that forward PH and CH differ in expressivity. We then introduce Hourglass Persistence, a class of topological descriptors that interleave a sequence of inclusions and contractions to boost expressivity, learnability, and stability. We also study related families parametrized by two paradigms. We also discuss how our framework extends to simplicial and cellular networks. We further design efficient algorithms that are pluggable into end-to-end differentiable GNN pipelines, enabling consistent empirical improvements over many PH methods across standard real-world graph datasets. Code is available at https://github.com/Aalto-QuML/Hourglass.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Hourglass Persistence (HP), a novel topological descriptor that generalizes traditional persistent homology (PH) through alternating inclusion and contraction steps. This leads to a more expressive and metrizable topological representation, which is further extended to Hourglass Persistence, where inclusions and contractions can interleave multiple times.

### Strengths
1. Conceptual novelty and clear motivation.
    
    The idea of introducing contraction-based backward persistence and merging it with inclusion-based persistence provides a genuinely new perspective beyond traditional or extended PH.
    
2. Strong theoretical grounding.
    
    The paper gives clear definitions for backward, forward–backward, and hourglass persistence, proving expressive power hierarchies (Proposition 1–2) and linking them to extended PH via the general ((f,g))-FB formulation. Stability and metrizability are discussed rigorously, addressing key limitations of standard PH.
    
3. Generalizability.
    
    The framework is elegantly extended from graphs to higher-dimensional topological spaces, including simplicial and cellular complexes, with appropriate handling of quotient and contraction operations.
    
4. Algorithmic contribution.
    
    The paper provides clear computational procedures using union–find and cycle basis bookkeeping, making the method implementable and efficient.

### Weaknesses
1. Limited experimental validation
- Although Hourglass persistence appears highly competitive, the current experiments are not strong enough to convincingly demonstrate its superiority. Is combining Hourglass persistence with neural networks truly the best application scenario?
- The authors do not provide sufficient experimental details. From the description, it seems that the PH module in RePHINE was simply replaced with the forward–backward persistence. The authors should better clarify and justify this design choice.
- Could the authors include an ablation study for the Backward-only variant?
- Since Hourglass persistence introduces additional computational overhead, could the authors report runtime or provide a complexity analysis?
2. Related work
    
In my view, the idea most closely related to Hourglass persistence is zigzag persistence [1], which also models both the appearance and disappearance of complexes during the filtration process. Could the authors discuss this connection in more detail?
   
[1] Carlsson G, De Silva V. Zigzag persistence[J]. Foundations of computational mathematics, 2010, 10(4): 367-405.

### Questions
1. Please provide additional experimental evidence.
2. Please include a comparison with zigzag persistence.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel filtration for feature extraction based on persistent homology. Unlike conventional approaches that employ forward filtration, the proposed method leverages a backward filtration. This design enables the method to achieve effective performance on graph datasets.

### Strengths
This paper proposes a new type of filtration from a novel perspective of persistent homology. While conventional approaches have primarily relied on forward filtration, the authors construct an interesting framework based on a backward filtration. As a result, the proposed method achieves improved performance in the experiments.

### Weaknesses
The main concerns are as follows:

- The central claims of this paper are the use of backward filtration and the introduction of (f,g)(f,g)-FB persistence. While the proposed method combines forward and backward (f,g)(f,g)-FB persistence, the actual contribution of the core idea — the backward filtration itself — remains unclear. Moreover, it is not evident how (f,g)(f,g)-FB persistence fundamentally differs from simply combining features derived from two filtrations. In general, combining multiple heterogeneous features often leads to performance gains, so the improvement shown here may not be specific to the proposed formulation.
    
- Although the notion of backward filtration is not identical, similar ideas have been explored in approaches such as Zigzag filtration [1]. The distinction between these existing methods and the proposed one is insufficiently clarified.
    
- The combination of two filtrations has also been proposed in prior work [2]. While the meaning differs due to the reversed filtration direction, the proposed combination seems to follow a rather straightforward idea, making it difficult to consider the novelty as significant.
    
- Since this conference focuses on machine learning and deep learning, introducing a new idea in persistent homology alone is not sufficient to meet the scope. It is therefore crucial to demonstrate the effectiveness of the approach in machine learning applications, but the current experimental evidence is not sufficient. In addition, Theorem 1 appears to be merely the authors’ claim and does not constitute a theorem in a rigorous sense. A theorem should be written in a mathematically precise manner that clearly states in what sense the proposed method is effective.

[1] G. Carlsson et al., Zigzag Persistence, Foundations of Computational Mathematics, Volume 10, pages 367–405, (2010)

[2] T. Aoki et al., Bipath persistence, Japan Journal of Industrial and Applied Mathematics, - Volume 42, pages 453–486, (2025)

### Questions
- Please include an ablation study comparing **forward-only**, **backward-only**, and **(f,g)-FB persistence** variants to quantify the contribution of the backward filtration (e.g., absolute/relative gains, with statistical significance).
    
- Please clarify how (f,g)-FB persistence differs from a simple concatenation/union of features derived from two filtrations. A theoretical distinction and an empirical comparison (which can be incorporated into the ablation) would be helpful.
    
- Either explain why zigzag filtration cannot achieve the same effect as the proposed method, or demonstrate empirically that your approach outperforms zigzag-based baselines.

### Soundness
2

### Presentation
2

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
This paper introduces a new family of topological descriptors for graphs, simplicial complexes, and cellular complexes, termed hourglass persistence, which interleaves inclusion and contraction operations to capture richer topological signals than traditional persistent homology (PH). The authors generalize beyond forward PH by defining backward PH, forward-backward (FB) persistence, and an expressive hourglass framework that allows arbitrary interleavings under a causal constraint. They provide theoretical separations demonstrating increased expressivity over classical PH and extended persistence, introduce a unifying (f,g)-FB perspective, and establish stability guarantees. Practical algorithms for computing these descriptors are developed, and empirical results show improvements over baseline PH and RePHINE on specific benchmarks.

### Strengths
1. The paper defines backward persistence and hourglass persistence, expanding the PH toolkit beyond inclusion-based filtrations.

2. The paper proves the expressiveness and the stability of the proposed framework.

3. The paper provides practical algorithms to compute inclusion–contraction PH with cycle-basis tracking and supernode maintenance.

### Weaknesses
1. While mathematically rigorous, the intuition behind when hourglass persistence most benefits learning tasks could be elaborated.

2. Empirical validation is limited to small graph-classification datasets (NCI109, PROTEINS, IMDB-BINARY, NCI1); scalability to large-scale benchmarks (e.g., OGB, ZINC) remains untested. In addition, the improvement compared to baselines is not significant.

3. Although extended persistence is mentioned, the empirical study does not include an extended-persistence baseline, and adding such a comparison would clarify practical differences

4. Computational overhead vs. benefit is not quantified; contraction bookkeeping may introduce non-trivial runtime/memory costs. Please include the efficiency analysis between FB-persistence with traditional PH and extended PH.

5. There are some missing related works, e.g., [1] also provides theoretical results on the expressiveness of PH.

[1] Yan et al. "Enhancing graph representation learning with localized topological features." JMLR 2025.

### Questions
1. How does the runtime and memory footprint scale for hourglass persistence relative to standard PH and extended PH? Can you provide empirical cost comparisons?

2. In practice, how are inclusion and contraction orders chosen? Are learnable schedules feasible, and if so, how do they interact with stability guarantees?

3. Can you provide case studies showing specific structural motifs uniquely captured by backward or hourglass persistence?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new persistent homology approach that combines forward and backward passes in creating persistent homologies.

### Strengths
The paper has deep mathematical discussion describing the properties of the forward backward persistence. 

The mathematical descriptions are well discussed. As a person who applied TDA for various application settings, I followed most of the discussion very carefully.

### Weaknesses
As a practitioner of topological data analysis (TDA), it is not clear to me whether the proposed forward–backward approach offers practical utility. The experimental evaluation appears limited, as it compares the method with only a single PH  approach used in GNN). Consequently, it is difficult to assess whether this work provides a substantive contribution beyond a theoretical exercise.

Moreover, based on my experience, many PH-based methods struggle to scale to large graphs. In this case, the computational cost of the proposed approach seems likely to increase at least twofold, without a clearly demonstrated benefit to justify the added complexity.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
2
