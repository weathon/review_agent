# Schreier-Coset Graph Rewiring

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Graph Neural Networks (GNNs) provide a principled framework for learning on graph-structured data, yet their expressiveness is fundamentally limited by over-squashing-the exponential compression of information from distant nodes into fixed size vectors. While graph rewiring methods attempt to alleviate this issue by modifying topology, existing approaches can introduce prohibitive computational bottlenecks. We propose Schreier-Coset Graph Rewiring (SCGR), a group-theoretic rewiring method that augments the input graph with a Schreier-coset graph derived from a special linear group $\mathrm{SL}(2,\mathbb{Z}_n)$. Unlike heuristic rewiring, SCGR provides  $\textit{provable}$ theoretical guarantees: the auxiliary graph exhibits a spectral gap and a bounded effective resistance, creating low-resistance bypasses for long-range communication. By coupling these two graphs with strength, we ensure that effective resistance between any node pair is bounded, directly mitigating over-squashing. Empirical evaluations demonstrate that SCGR reduces effective resistance by 15-40\% across benchmark datasets while maintaining competitive accuracy and lower computational overhead, making it practical for both large-scale and diverse applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a group-theoretic framework to alleviate the *over-squashing* problem in Graph Neural Networks (GNNs). Instead of heuristic or computationally heavy rewiring, SCGR augments an input graph with a **Schreier-coset graph** derived from the special linear group, which has guaranteed spectral expansion and bounded effective resistance. A locality-preserving mapping aligns the two graphs, and edges are added based on proximity in the Schreier space but distance in the original graph, ensuring low-resistance pathways for long-range communication. Theoretical analysis establishes explicit bounds on spectral gap, effective resistance, and over-squashing reduction, while experiments on standard node and graph classification benchmarks, molecular property prediction, and stochastic block models show **15-40% reductions in effective resistance** and consistent accuracy gains, confirming that SCGR provides both **provable and practical mitigation** of topological bottlenecks in GNNs.

### Strengths
- **\[S1] Important problem.** Addresses the fundamental _over-squashing_ issue in GNNs, a key bottleneck limiting long-range information propagation.

- **\[S2] Novel group-theoretic rewiring.** Introduces _Schreier–Coset Graph Rewiring (SCGR)_—a new use of Schreier–coset graphs from $_SL(2, Z\_n)$_—bringing group theory into graph rewiring for the first time in this context.

- **\[S3] Theoretical guarantees.** Provides formal bounds on spectral gap, effective resistance, and over-squashing mitigation, offering provable improvements over heuristic methods.

### Weaknesses
See “Questions” below.

### Questions
- **\[Q1] Missing related work on spectral-property-preserving rewiring.** I noticed that a related paper, _Liang et al. "Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification." ICML’25_, also studies graph rewiring methods that maintain spectral characteristics of the original graph. Can you discuss how SCGR compares or complements such approaches—both conceptually and, if feasible, empirically.

- **\[Q2] Theory–practice discrepancy in spectral guarantees.** The central premise of this work is that SCGR preserves or improves spectral properties through a theoretically guaranteed construction. However, the empirical results focus only on downstream task performance and effective resistance, without demonstrating that these spectral guarantees actually hold in practice. Could the authors provide empirical evidence—such as spectral similarity measures (e.g., eigenvalue correlation, Laplacian distance) or structural overlap—showing that the rewired graphs indeed preserve the spectral characteristics of the originals? Otherwise, the claimed “spectral-property-preserving” aspect remains largely theoretical.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Schreier-Coset Graph Rewiring (SCGR), a method to reduce over-squashing in graph neural networks by adding carefully chosen shortcut edges to the input graph. The method first builds an auxiliary Schreier–coset graph with expander-like connectivity. Each node in the original graph is mapped to a node in this auxiliary graph, and then new edges are added between original nodes that are close in the Schreier–coset graph but far apart in the original graph. The paper argues this improves information flow and gives empirical gains on graph learning benchmarks.

### Strengths
1. The paper leverages expander-inspired auxiliary graph (the Schreier–coset graph) to guide which long-range connections to add is conceptually feasible.
2. The method aims to mitigate over-squashing while preserving sparsity and locality, which is desired for graph rewiring.

### Weaknesses
- Line 128: the expression $\mathcal{G} = SL(2,\mathbb{Z}_n$ is missing a closing parenthesis.

- Spectral Mapping Construction is not clearly defined. How exactly is 
  $\Phi_{\text{in}} : V_{\text{in}} \to \mathbb{R}^r$ 
  computed from the top $r$ eigenvectors? Please specify the construction.

- The optimization
  $
  \min_{\phi : V_{\text{in}} \hookrightarrow V_{\Gamma}}
  \sum_{(u,v)\in E_{\text{in}}} \operatorname{dist}_{\Gamma}(\phi(u), \phi(v))
  $
  is underspecified. How do you solve it?

- The constraint 
  $
  \left\lVert \Phi_{\Gamma}(\phi(v)) - \Phi_{\text{in}}(v) \right\rVert_2
  $
  is only described as “small,” but there is no quantitative definition.

- The “Mapping and Rewiring” section relies on several simplifying assumptions (many of which are only stated informally). The paper does not analyze how these simplifications affect the claimed guarantees or the empirical performance of the method.

There is no direct theoretical or empirical analysis of how close the rewired graph is to the original graph, or to what extent it preserves the original graph’s topology.

**Regarding Theorem 4.1:**
- Theorem 4.1 compares the Schreier–coset graph to the original graph. However, the final graph used in the method is not the Schreier–coset graph; it is a selective merge of edges from the Schreier–coset graph and the original graph. Even if the Schreier–coset graph were “similar” to the original graph, that does not imply that this selectively rewired graph is also similar.

- Theorem 4.1 does not provide a meaningful quantitative guarantee. The constant $c$ is left unbounded, and the claim is essentially intuitive. Stating that node distances in the Schreier–coset graph are (in expectation) shorter than in the original graph is just restating a standard property of expander-like graphs and does not, on its own, establish the usefulness of the proposed rewiring.

**Regarding experiments:**
- The evaluation is limited to six graph classification benchmarks, which is not sufficient to support broad claims. The method should also be tested on node classification tasks,.

- There is no empirical runtime or memory analysis. The paper claims near-linear complexity, but does not provide wall-clock time or scaling experiments. This raises concerns about practical computational cost.

**Clarity and positioning:**
- The paper is not fully self-contained. Several central concepts are only briefly described, and important background is omitted. Even if these ideas are not conceptually deep, a strong paper should provide precise definitions and intuition in the appendix to ensure readability.

- The literature review is incomplete. Important related work is missing or only mentioned superficially. The paper should better position its contribution relative to prior graph rewiring / expander-based approaches, and include additional background and ablations in the appendix to make the contribution and empirical claims more convincing.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Schreier-Coset Graph Rewiring, a topology-augmentation scheme for GNNs that overlays the input graph with a constant-degree Schreier-coset graph and connects each original node to a coset representative. The goal is to open low-resistance, long-range communication to mitigate over-squashing.

The theory argues that coupling the Schreier layer to the input with strength $\varepsilon$ gives a uniform upper bound on pairwise effective resistance in the rewired graph, and the improvement factor can be large, especially for distant pairs.

Empirically, SCGR is reported to reduce effective resistance and improve accuracy across standard node/graph benchmarks (e.g., Amazon, Coauthor-CS, TU datasets, OGB-MOLHIV/PCBA, Long-Range Graph Benchmark peptides). The method is positioned as a principled, lower-overhead alternative to expander-rewiring.

### Strengths
* Using Schreier-coset graphs to get constant degrees is an interesting design point that directly targets topological oversquashing via effective resistance analysis.
*  The theoretical section explicitly relates the spectral gap of the Schreier layer to resistance bounds, and shows how the coupling propagates this benefit to the original graph. 
*  The empirical validation seems strong, although the code has not been released. Results cover node classification (Amazon, Coauthor-CS, Cora/CiteSeer/PubMed), TU graph classification (REDDIT, IMDB, MUTAG, ENZYMES, PROTEINS, COLLAB), OGB molecular tasks, and OGBG peptides, with consistent ER reductions and multiple SOTA-competitive wins (for example, large gains on ENZYMES and solid ROC-AUC on MOLHIV).

### Weaknesses
- The **writing clarity** throughout the document is quite poor and needs, in my opinion, substantial improvements to reach the bar of ICLR. The manuscript is hard to follow in some sections due to its lack of readability and structure. 
  - For example, in the related work section, the sentence "In Expander Graphs-Expanders provide favorable spectral gap and resistance" is, first, not well-written and, second, seems vague because it lacks analytical depth. Also, "In message-passing networks, the gradient flow between distant nodes is inversely proportional to their effective resistance," is not supported by citations, and *gradient flow* is neither defined nor mentioned again. I also found it strange to read "By coupling these two graphs with strength,.." in the abstract, where strength has not been introduced yet nor is it common terminology in the literature.
  - The related work section provides an insufficient and unclear explanation of previous methods' limitations and how SCGR addresses them. 
  - Additionally, in the preliminaries, many concepts are introduced without sufficient explanation or connection, appearing as isolated elements. The readability is poor and can also be improved in sections 3.3 and 4. Overall, careful proofreading and restructuring would significantly enhance the document’s readability and clarity.
  - The clarity in the methodological explanation could also be improved. For instance, the mapping ($\phi: V_{\text{in}}\to V_\Gamma$) is not defined in detail. I understand (iiuc) that the "spectral mapping construction" minimizes a distance objective with a closeness constraint to spectral embeddings, but the optimization problem is not fully formalized (objective, constraints, solver), and its complexity and approximation quality are unclear.
  - Precisely define *how ER is computed (exact or approximation) and aggregated* (average over all pairs? over a sample? normalized by (n)?). The very large absolute ER values in Table 3 need units/normalization and a reproducible computation recipe. Are you talking about the total effective resistance (sum over all pairs) or the average effective resistance (average over all pairs)?
  - There are no links or appendix tables for hyperparameters per dataset, and no ablations on $\ell$ or $\varepsilon$ and added-edge budgets, limiting the interpretability of the results.
  
- Recent work shows the community is using two distinct notions under the same term: (i) a computational bottleneck view (over-squashing as compressing exponentially growing messages through fixed-width vectors along long dependency paths), and (ii) a topological bottleneck view (over-squashing as poor connectivity / high effective resistance / small spectral gap). The paper implicitly adopts the topological lens (effective resistance, expansion), but never states this upfront. Please disambiguate early (Intro/Preliminaries) which definition you use, cite both lines of work, and acknowledge limitations / potential tensions: e.g., rewiring that adds edges can reduce effective resistance yet may widen message fan-in and thus increase the computational bottleneck, whereas deleting/sparsifying edges has the opposite trade-off. Position your claims and experiments accordingly. 

- The code is not public; thus, reproducibility is limited. For instance, we cannot check how the hyperparameter search for the baselines was conducted or how the authors specifically split the datasets (they do not provide detailed information about it). Please release code and configs to facilitate verification and adoption.

- Please fix some citations of very important works:
  - *Kipf and Welling* is not in arXiv but in [ICLR 2017](https://openreview.net/forum?id=SJU4ayYgl).
  - *Alon and Yahav* is not in arxiv but in [ICLR 2021](https://openreview.net/forum?id=i80OPhOCVH2).
  - *Arnaiz-Rodriguez et al*  is not in arXiv but in [LoG 2022](https://proceedings.mlr.press/v198/arnaiz-rodri-guez22a.html).
  - *Karhadkar, et al* is not in arXiv but in [ICLR 2023](https://openreview.net/forum?id=3YjQfCLdrzz).
  - *Topping et al* is not in arXiv but in [ICLR 2022](https://openreview.net/forum?id=7UmjRGzp-A).
  - *Wilson et al* is not in arXiv but in [LoG 2024](https://openreview.net/forum?id=VaTfEDs6lE).
  - *Morris et al* is not in arXiv but in [ICML 2020 Workshop on 'Graph Representation Learning and Beyond'](https://chrsmrrs.github.io/datasets/).
  - *Xu et al* is not in arxiv but in [ICLR 2019](https://openreview.net/forum?id=ryGs6iA5Km).
  - There are even more citations that appear in arXiv but should be fixed to the actual venue.

- Minor: l.62: SCHREIER-COSET is misspelled as SCHRIER-COSET. 

*Refs*

Arnaiz-Rodriguez & Errica, *Oversmoothing, Oversquashing, Heterophily, Long-Range, and more: Demystifying Common Beliefs in Graph ML*, [MLG @ ECML-PKDD 2025.](https://arxiv.org/abs/2505.15547)

### Questions
In addition to previous comments, I have the following specific questions for the authors:

1. Please state explicitly which over-squashing notion you adopt (computational vs. topological) from the differentiation of (Arnaiz-Rodriguez & Errica, 2025), in the intro and related work. Update Related Work to reflect both strands (e.g., the already cited works: Alon & Yahav, 2021 for the computational view; Topping et al; Arnaiz-Rodriguez et al, Karhadkar et al, Black et al., and follow-ups for the gap/effective-resistance/topological view; and finally Arnaiz-Rodriguez & Errica, 2025 for a taxonomy/critique).
   
2. Based on the previous distinction, could you analyze its implications under the computational bottleneck definition? Can you add an experiment or diagnostic that speaks to the *computational bottleneck* alongside effective-resistance metrics?  In particular, under fixed hidden width and aggregator, does adding SCGR edges increase or decrease the per-layer message bottleneck (size of the computational tree, receptive field, average in-degree...), and what evidence supports this? In addition, quantify information compression proxies (e.g., Jacobian spectrum / gradient flow across hops) to see if SCGR helps or hurts this aspect.

3. How many edges are added per node on average for each dataset? Do you enforce a budget (e.g., (k) added edges per node) to keep ($|E^{\text{rwd}}|$) linear?

### Soundness
2

### Presentation
1

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
This paper addresses the well-known problem of over-squashing in Graph Neural Networks (GNNs), where information from distant nodes is exponentially compressed, limiting the model's ability to capture long-range dependencies. The authors propose a novel graph rewiring method called Schreier-Coset Graph Rewiring (SCGR). The authors provide theoretical guarantees that this graph has a spectral gap and, consequently, a uniformly bounded effective resistance.

### Strengths
1. This work is novel. While expander graphs and Cayley graphs have been explored, the shift to Schreier-coset graphs is a creative and elegant mathematical idea that appears to solve the scalability issues of previous group-theoretic approaches.

2. The paper is built on a solid theoretical foundation. Instead of proposing a purely empirical heuristic, the authors provide a principled, proof-backed framework. They formally link the properties of the Schreier-coset graph to a bounded effective resistance.

3. The experimental results are comprehensive. The method is tested on standard node classification (Cora, PubMed, etc.) , graph classification, and large-scale OGB datasets.

### Weaknesses
1. Section 3.2 provides the formal definition for the number of vertices in the Schreier-coset graph  $\frac{n(n^2-1)}{\phi(n)}$, which is $\Omega(n^2)$ since $\phi(n) \le n-1$. However, in line 200, the authors claim that the Schreier-coset graph has $O(n)$ vertices. There is a significant contradiction here.

2. The rewiring strategy is critically dependent on the locality-preserving mapping. It is unclear about the details of the solution to this mapping. This mapping is defined as the solution to an optimization problem. This appears to be a form of graph matching or alignment, which is often NP-hard. The time complexity for this step seems to be missing as well.

3. The rewiring strategy introduces at least two key hyperparameters, but their impact is never discussed.

### Questions
1. Please resolve the apparent contradiction regarding the size of the Schreier-coset graph. 

2. How is the locality-preserving mapping computed in practice? What algorithm is used to solve this minimization problem, and what is its computational complexity?

3. Could you provide a sensitivity analysis for the key hyperparameters like Schreier distance threshold and coupling strength?

4. What was the motivation for this specific choice for using $G = SL(2,\mathbb{Z}_{n})$? Would other groups or subgroups also yield scalable expander graphs?

5.

### Soundness
2

### Presentation
2

### Contribution
3
