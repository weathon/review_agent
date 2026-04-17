# AH-UGC: $\underline{\text{A}}$daptive and $\underline{\text{H}}$eterogeneous-$\underline{\text{U}}$niversal $\underline{\text{G}}$raph $\underline{\text{C}}$oarsening

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
$\textbf{Graph Coarsening (GC)}$ is a prominent graph reduction technique that compresses large graphs to enable efficient learning on graphs. However, existing GC methods generate only one coarsened graph per run and must recompute from scratch for each new coarsening ratio, resulting in unnecessary overhead. Moreover, most prior approaches are tailored to $\textit{homogeneous}$ graphs and fail to accommodate the semantic constraints of $\textit{heterogeneous}$ graphs, which comprise multiple node and edge types. To overcome these limitations, we introduce a novel framework that combines Locality-Sensitive Hashing (LSH) with Consistent Hashing (CH) to enable $\textit{adaptive graph coarsening}$. Leveraging hashing techniques, our method is inherently fast and scalable. For heterogeneous graphs, we propose a $\textit{type-isolated coarsening}$ strategy that ensures semantic consistency by restricting merges to nodes of the same type. Our approach is the first unified framework to support both adaptive and heterogeneous coarsening. Extensive evaluations on 23 real-world datasets including homophilic, heterophilic, homogeneous, and heterogeneous graphs demonstrate that our method achieves superior scalability while preserving the structural and semantic integrity of the original graph.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes "Adaptive Coarsening via Consistent and LSH Hashing" (AH-UGC), an extension of the previous work UGC. The method aims to provide a unified graph coarsening framework that supports adaptive, streaming, expanding, heterophilic, and heterogeneous graphs.

### Strengths
- The goal of creating a unified framework for diverse graph types (heterophilous, heterogeneous, streaming) and tasks (link prediction) is ambitious and highly relevant.
- The experimental evaluation is comprehensive, and the authors compare against up-to-date baselines.

### Weaknesses
1. The primary novelty appears to be the "consistency hash." The justification for this module is weak. Why does a balanced supernode formation necessarily benefit both GNN performance and spectral preservation? This claim is counter-intuitive and requires stronger theoretical guarantees or empirical analysis.
2. The method addresses heterophily by using a *global* edge heterophily factor during a *local* concatenation of $X$ and $A$. This design choice seems inconsistent. Is this approach theoretically sound? Please provide more ablation studies to justify this specific design and cite prior works [1][2] that use a similar metric or approach.
3. The "type-isolated coarsening" is a key component for handling heterogeneous graphs. To fairly assess its contribution, can this module also be applied to baselines like UGC? This would help clarify if the impurity reduction is due to the new module or other aspects of AH-UGC.
4. In Table 1, FGC and LAGC are reported as "OOT" (Out-of-Time) on PubMed, yet their original papers report results for this dataset. Please provide implementation details and analyze the bottleneck causing this issue. Is it due to a difference in hardware, setup, or implementation?
5. The paper notes that the method does not perform well with Transformer-based GNNs. Please provide a deeper analysis of why this limitation exists.
6. The paper fails to introduce or define all baseline methods. For instance, what are "aJC" and "aGS"?
7. The related work section is incomplete. Please include and discuss relevant works on:
    ◦ Streaming graph condensation [3][4]
    ◦ Heterogeneous graph condensation [5]
8. I deeply suggest the authors to narrow the scope and focus on one dimension. It's ambitious to propose a unified framework but it'd be better to do it in a dissertation. Putting every dimension in one conference paper requires the method to be very solid, self-contained and consistent with the tasks you want to solve. But this proposed method (especially the novel part) is only targeted for balanced coarsening, which is not related to heterophily, heterogeneousity or streaminig.

Minor:

- **Table 1:** Please right-align the numbers in Table 1 for better readability.
- **Figure 2:** The figure quality is low. Please use a vector format (e.g., PDF) and enlarge the font size.
- **Table 4 & 9:** Highlights for the best-performing methods are missing in Table 4 and for the "Cora" rows in Table 9.
- **Table 10:** The GCN model is set to 3 convolution layers, which is non-standard (typically 2). Please justify this choice.
- **Figure 4:** Please define "hDBLP" in the figure caption or text.

## References:

[1] Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs, In NeurIPS 2020

[2] Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, In NeurIPS 2021

[3] Scalable graph condensation with evolving capabilities, arxiv 2025

[4] Graph Condensation for Open-World Graph Learning, In KDD 2024

[5] Training-Free Heterogeneous Graph Condensation via Data Selection, In ICDE 2025

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes AH-UGC, a unified framework for adaptive and heterogeneous graph coarsening. It integrates Locality-Sensitive Hashing (LSH) and Consistent Hashing (CH) to enable fast, scalable, and ratio-adaptive coarsening, while ensuring type-isolated merging for heterogeneous graphs. Experiments on 23 datasets show superior scalability and strong structural and downstream performance.

### Strengths
1. Novel adaptive mechanism combining LSH and CH for multi-resolution coarsening.
2. Type-consistent design effectively supports heterogeneous graphs.
3. Comprehensive experiments demonstrate scalability and robustness across diverse settings.

### Weaknesses
1. The paper does not include or discuss several recent graph coarsening or condensation methods from 2024–2025, which limits the completeness and fairness of the comparison.
2. The claim around line 177 that graph condensation methods are inefficient is not accurate. Recent studies such as [1], [2], and many more, have proposed efficient condensation techniques, and the paper would benefit from acknowledging and comparing to these works.

[1] Rethinking and Accelerating Graph Condensation: A Training-Free Approach with Class Partition, WWW 2025

[2] Adapting Precomputed Features for Efficient Graph Condensation, ICML 2025

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
As large-scale data core sets and sketching gained significant attention some time ago, graph coarsening has become a crucial challenge in computer science for increasingly massive graphs. This paper addresses two research challenges as the two promised advances for recent breakthroughs: (1) achieving consistent and simultaneous coarsening across multiple resolutions, and (2) ensuring applicability even in scenarios where input and observed graphs are heterogeneous.

### Strengths
- The two problem settings in this paper (i.e., simultaneously achieving multiple levels of coarsening and handling heterogeneous data) appear to be both highly significant and novel within the context of graph coarsening. And very happily, the authors' method provides a framework that can unify and solve these two challenges simultaneously.

- The experimental investigations in this paper are exceptionally large-scale and comprehensive. As they are shared with the community alongside open-source code, they provide crucial assistance for subsequent research.

- The presentation of this paper is highly clear and polished. (However, I do feel there is a little room for improvement regarding the mathematical notation, and I have commented on this briefly in the questions section.)

### Weaknesses
- This paper represents a solid advancement of existing research [Kataria+, NeurIPS2024]. However, its improvements (in my personal, subjective view) may be relatively minor compared to the breakthroughs in the foundational research [Kataria+, NeurIPS2024] itself. Specifically, the paper certainly adds two functionalities: (1) simultaneously acquiring coarsening at multiple hierarchical levels, and (2) handling heterogeneous graphs. However, these additions appear to be straightforward extensions of existing work, and I personally do not perceive them as constituting the ‘an innovative and flexible approach’ claimed by the authors in line 192.


- I have remaining concerns regarding the novelty of the theoretical contributions of the proposed method discussed in Section 3.2 of this paper. I remain skeptical about the novelty of the theoretical contributions of the proposed method discussed in Section 3.2 of this paper. Theorem 3.1 and Lemma 1 appear to state general properties of Gaussian variables within a rather broad context. In fact, the h(x) used in these seems to be employed in a different sense than the score h used by the authors in lines 215 and elsewhere. While the facts stated in Section 3.3 suggest the (high-probability) correctness of the authors' AH-UGC behavior, I question whether these theoretical analyses themselves constitute the authors' contribution. I find the positioning of these theoretical analyses within the paper somewhat unclear.

### Questions
I think there is some room for improvement in the handling of notation. I will give one example here, but reconsidering the overall notation for similar reasons would likely enhance clarity of this excellent paper.

For example, line 198 first mentions the augmented vector F_{i}. Later, on line 202, the expression ‘Let F_{I}’ appears. As a reader, I would greatly appreciate it if a symbol were properly defined upon its first appearance and then consistently reused within the same context. Considering this, I personally find the following style easier to read:

- It would be helpful if line 198 explicitly stated that it provides the definition of F_{i}. For example, a simple expression like ‘F_{I} :=‘ might suffice.

- Line 202 could be made much clearer by emphasizing that it is merely reintroducing a variable already defined once before. For instance, writing something like ‘Recall that F_{i} is defined at Line 198’ would greatly improve readability.

Is the equation in line 206 correctly written? I believe that with this notation, the right-hand side does not appear to be a scalar. More precisely, is it W_{k}^{\top}\cdot F_{I}+b_{k}?

Since the heading in Section 3.1 is labeled ‘Goal1,’ to maintain consistency in this notation, it would be clearer if ‘Goal2’ were also explicitly stated. Specifically, line 239 would be ‘Goal2,’ correct?

Is the equation on line 266 a period (.) rather than a comma (,)?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a unified graph coarsening framework that simultaneously supports adaptive coarsening ratio and heterogeneous graph structures.
The method combines Locally Sensitive Hashing (LSH) with Consistency Hashing (CH) to rapidly generate multi-level coarsened graphs without recomputation, while preserving semantic consistency in heterogeneous graphs through a type-isolation strategy.
Experiments demonstrate that AH-UGC outperforms existing methods in efficiency, structural preservation, and downstream task performance.

### Strengths
This paper addresses two issues crucial to graph coarsening: adaptive coarsening ratios and heterogeneous graphs. The experimental results suggest that this method appears to be quite effective. The boundaries provided by the two theorems are quite interesting.

### Weaknesses
1. The writing of this paper need to be improved.It took me a long time to figure out what the author was writing about. After reading the paper multiple times, I still don't understand what the author is trying to convey in Goal $2$, and I couldn't find the definitions for some notations at all. Notation in this paper appears poorly organized; for example, matrices are represented using three distinct forms: uppercase letters, bold uppercase letters, and calligraphic letters. $r$ has multiple meanings in this paper (the r in Theorem 3.1 is defined differently from $r$ elsewhere). Given the abundance of symbols in this paper, I recommend the author create a notation table for readers to consult quickly. Another small suggestion is to use $\top$ (\top) instead of $T$ as the symbol for matrix transposition.

2. Apart from writing, this paper is a bit too incremental in terms of innovation relative to UGC. Especially when it comes to handling heterogeneous graphs, the overall approach is not novel for me.

3. The authors state that their method can be applied in streaming settings, but no relevant experiments have been conducted to validate this claim.

I like the methodology of this paper and I think that if the paper can improve their writing, the technical contribution of the paper should be a borderline accept.

### Questions
1. What is Goal $2$ trying to convey?
2. What is the time complexity of AH-UGC? 
3. Is $l$ a hyperparameter? If so, how is it determined?

### Soundness
2

### Presentation
1

### Contribution
3
