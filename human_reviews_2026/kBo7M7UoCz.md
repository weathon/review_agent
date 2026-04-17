# Difference-Based Graph Attention Networks: A Dual Attention Mechanism for Similarity and Dissimilarity in Graph Learning

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Most Graph Neural Networks (GNNs) rely solely on similarity-based attention
mechanisms, limiting their ability to distinguish nodes that are structurally similar
but semantically distinct. We introduce Difference-Based Graph Attention Net-
work (DGAT), a novel architecture that integrates both similarity and dissimilarity
attention within a unified geometric framework. DGAT models contrastive rela-
tionships using orthogonal projections and wedge-product approximations, cap-
turing richer feature interactions beyond alignment. Our formulation is grounded
in a generalized Iwasawa–Cayley decomposition, where the combination of sim-
ilarity and dissimilarity attention correspond to orthogonal, scaling, and shifting
operations. We also connect its behavior to discrete analogs of differential opera-
tors and function orthogonality, establishing a principled geometric interpretation.
Experiments across homophilic OGB graphs, specially in OGBl-PPA, and het-
erophilic benchmarks show that DGAT consistently outperforms GAT, GATv2,
and Graph Transformer architectures, especially in settings requiring fine-grained
representational contrast or role differentiation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Difference-Based Graph Attention Network (DGAT), which is a novel GNN architecture that extends traditional similarity-based attention with an additional dissimilarity-aware attention pathway. DGAT models both similarity (via cosine or additive attention) and dissimilarity (via orthogonal projection and wedge-product approximations) within a unified geometric framework. The authors provide a theoretical grounding based on the Iwasawa-Cayley decomposition from Lie group theory, showing that similarity and dissimilarity components correspond to orthogonal, scaling, and shifting operations.

### Strengths
This paper introduce a dissimilarity-based attention mechanism into graph attention networks for the first time, addressing the long-standing limitation of relying solely on feature similarity. By modeling both similarity and dissimilarity between nodes, the proposed DGAT significantly enhances representation expressiveness. The paper provides a geometric interpretation by linking DGAT to the Iwasawa-Cayley decomposition, offering strong theoretical grounding and interpretability. Moreover, the DGAT framework is highly flexible and extensible, integrating seamlessly with existing architectures such as GAT, GATv2, and Graph Transformers.

### Weaknesses
1. The paper lacks a clear and intuitive diagram or framework illustration that helps readers better understand how the proposed approach works.

2. The paper should clarify how the proposed method differs from or relates to graph contrastive learning, which also uses node similarity and dissimilarity.

3. The motivation is unclear, the paper only mentions that previous models ignored node dissimilarity but does not analyze why dissimilarity was not utilized. Moreover, since the experiments include both homophilic and heterophilic graphs, it is not clear which type the study primarily focuses on. Some baseline models are designed for homophilic graphs, so applying them directly to heterophilic settings may raise questions about the fairness and validity of the comparisons.

4. The baseline methods are outdated and lack comparisons with more recent GNN models such as GARN[1]. In addition, the baselines mainly include attention-based architectures, so it is unclear why other classic methods such as GCN[2], GraphSAGE[3], and JKNet[4] were not included for comparison.

[1] Wang Y, Wen J, Zhang C, et al. Graph aggregating-repelling network: Do not trust all neighbors in heterophilic graphs. Neural Networks, 2024, 178: 106484.

[2] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, 2017.

[3] W. L. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In Proceedings of Advances in Neural Information Processing Systems, 2017.

[4] Xu K, Li C, Tian Y, et al. Representation learning on graphs with jumping knowledge networks. In Proceedings of the 35th International Conference on Machine Learning, 2018.

5.The paper lacks an analysis of time and space complexity.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper develops a new graph neural network called DGAT. The authors argue that existing attention-based GNNs are built on similarity-based attention mechanism, which is inefficient in capture complex graph information. To this end, the authors develop a new attention module based on orthogonal projections and wedge-prodcut approximations to preserve contrastive relationships between nodes. Experimental results on various datasets show the effectiveness of DGAT on graph data mining tasks.

### Strengths
1.This paper is well-organized and easy to follow.

2.The authors provide the theoretical analysis of the proposed method.

3.The proposed DGAT provides new insights for attention-based GRL methods.

### Weaknesses
1.The research gap is overclaimed.

2.Mainstream baselines are missing.

3.Some important experiments are missing.

### Questions
1.The authors claim the limitation in existing attention-based GRL methods which lack objectivity. There are also several works, such as FAGCN and ACMGNN, which are designed to capture both low-frequency information (similarity) and high-frequency information (dissimilarity) in graph representation learning.

2.In the experiment part, I suggest the authors add more recent graph Transformers as baselines for performance comparison.

3.Moreover, necessary experimental designs such as ablation study and parameter analysis are also required for strengthening the experiment part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript introduces Difference-Based Graph Attention Networks (DGAT), a novel GNN architecture that extends standard attention mechanisms to jointly capture similarity and dissimilarity between nodes. The method integrates a dual attention pathway, which contains a similarity-based component using cosine or additive attention and a dissimilarity-based component using orthogonal projections and wedge-product approximations. The final representation is obtained via a learned gating mechanism that balances both components. DGAT is grounded in the Iwasawa–Cayley decomposition, offering geometric interpretability connecting orthogonal, scaling, and shifting operations. Experiments are conducted on multiple homophilic (OGBg-MolHIV, OGBn-Proteins, OGBl-PPA, OGBl-DDI) and heterophilic (Minesweeper, Roman-Empire, Amazon-Ratings, Questions) benchmarks, showing consistent performance improvements over GAT, GATv2, and Graph Transformer baselines

### Strengths
1. The proposed dual-path mechanism and gating function of DGAT address an underexplored limitation of GAT-like models that emphasize only similarity

2. DGAT based on the theoretical grounding via the Iwasawa–Cayley decomposition. This linkage provides interpretability and situates DGAT in a geometric-algebraic context.

3. The authors conduct comprehensive experiments, covering both homophilic and heterophilic benchmarks. The results show the improvements of DGAT compared with baselines.

### Weaknesses
1.The authors claim the convergence guarantees of DGAT in the introduction. However, I do not find the specific proof or lemma to quantify this property. If boundedness can lead to convergence, the author needs to clearly point this out and provide a proof.

2.Ablation studies are missing. For example, the selection of $\lambda$ and similarity v.s. dissimilarity isolation are not empirically separated.

3.Computational overhead and training stability. Claims about efficiency and non‑expansiveness lack runtime or convergence curves.

### Questions
1.In Section 1, the authors claim that DGAT is “non-expansive” with “convergence guarantees.” However, no formal proof, theorem, or empirical demonstration is included.

Question: Could you please provide the precise mathematical derivation or an outline of the argument that establishes non-expansiveness and convergence guarantees? Are these guarantees derived from the gating mechanism, the orthogonal projection, or both?

2.Eq. (4) introduces different versions of the gating function (i.e., w‑orthogonal, torsion, and default). The manuscript names them but provides no training dynamics or comparison. 

Question: How do these gating variants quantitatively differ in learned behavior or performance? Have you tested their contributions through ablations, and if not, could you clarify why certain gates are only conceptually presented but not empirically analyzed?

3.I observe that the authors claim the efficiency related to head compensation and GPU-based orthogonal projection (in section A.5), yet no timing results are presented.

Question: What is the relative runtime cost of DGAT compared to GATv2 and Graph Transformer? Does the dissimilarity computation introduce additional latency or memory bottlenecks?

### Soundness
2

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
3

### Summary
The paper proposes DGAT, a dual-path attention layer that couples a standard similarity path with a dissimilarity/orthogonal path computed via Gram-projection / wedge-product surrogates and combined with a learned gate (including an orthogonality-enforcing “w-orth” and an optional torsion/Lie-bracket-inspired variant). Core updates are given in Eqs. (1)–(3) and the gate in Eq. (4).

### Strengths
1. Clear, principled idea. The orthogonal/difference channel complements similarity attention to encode contrast rather than only alignment; the equations and gate make the mechanism explicit and modular.
2. Geometric grounding. The Iwasawa perspective provides an interpretable decomposition (K/A/N) mapping cleanly onto (similarity / difference / gating). This is rare in GNN attention papers and helps motivate design choices.
3. Empirical signal. On OGB tasks and heterophilic benchmarks, DGAT variants are reported to outperform GAT/GATv2/Graph Transformers under matched settings (with a head-count compensation to control params).

### Weaknesses
1. While the geometric story is appealing, some claims (e.g., “non-expansive and convergent operator”) are mentioned in the intro but I did not see full proofs in the provided snippets; if they exist in the appendix, make them crisp with assumptions and operator norms (Lipschitz constants, spectral bounds). (Pointer to tighten: Sections A.3–A.4 already frame energy/orthogonality—turn these into formal theorems.)
2. The paper introduces w-orth and torsion gates; more ablation would clarify when each helps, sensitivity to λ, and whether improvements persist if gates are simplified. (Some hyperparameter tables appear, but targeted gate ablations would strengthen claims.)
3. It seems that results are described as best-of-run with std as “difference between best results,” which is non-standard. Prefer mean±std over many seeds.

### Questions
1. Can the w-orth constraint hurt when neighborhoods are tiny/noisy?

### Soundness
3

### Presentation
3

### Contribution
3
