## Summary
STAGE introduces a novel method for zero-shot generalization of graph neural networks across distinct node attribute domains by transforming raw features into edge-level statistical dependency graphs (STAGE-edge-graphs) and applying a two-stage GNN. It claims provable invariance to a class of domain shifts (COGGs) and demonstrates large empirical gains (40–103% relative improvement) on link prediction and node classification tasks.

## Strengths
- **Novel representation via statistical dependencies** — STAGE constructs conditional probability matrices (Sᵘᵛ) that capture feature dependencies rather than absolute values, enabling invariance to attribute domain shifts. (Section 2, Equations 2–3, Definition 2.1)
- **Strong and consistent empirical results** — STAGE outperforms all baselines by substantial margins on zero‑shot link prediction (E‑Commerce Stores, H&M) and node classification (Pokec), with gains up to 103% in Hits@1 and 10.88% in accuracy, and performance improves with more training domains. (Tables 1–2, Figure 4)
- **Theoretical guarantee of COGG invariance** — Theorem 3.4 proves that STAGE is invariant to a class of transformations covering feature value shifts, feature permutations, and node permutations, under a fixed‑dimensional feature assumption. (Section 3.2)
- **Clear problem framing and motivation** — The introduction and Figure 1 effectively illustrate the zero‑shot cross‑domain challenge and why existing strategies (ignoring features, using LLMs) may fail. (Section 1)
- **Comprehensive evaluation** — The paper covers both link prediction and node classification across multiple datasets and provides detailed statistical reporting (means, standard deviations). (Section 4, Tables 1–2)

## Weaknesses

### Fatal
None.

### Major
- **Inadequate baseline specification for zero‑shot generalization** — The paper compares STAGE against several baselines (NBFNet‑raw, gaussian, structural, llm, normalized) but does not clearly explain how these baselines handle graphs with node attribute spaces of different dimensions at test time. For example, NBFNet‑raw “projects each raw node feature into a fixed‑dimensional space via linear transformations,” which typically requires a fixed input dimension; it is ambiguous whether and how this baseline can process test graphs with entirely new feature types and dimensions. Without detailed, verified zero‑shot‑capable implementations, the reported superiority of STAGE rests on shaky ground. The near‑zero Hits@1 of NBFNet‑raw on all E‑Commerce domains (Table 1) further suggests potential implementation flaws rather than a genuine empirical gap. Fairness of comparisons is therefore in question. (Section 4.1, Table 1, baseline descriptions)
- **Theoretical guarantees do not match experimental setting** — The theoretical results (Theorems 3.2–3.4) are explicitly “restricted to domains with a fixed number of features… extending them to variable size spaces is left as future work” (Section 3). However, the experiments involve datasets with varying feature dimensions across domains (e.g., smartphones vs. shoes in E‑Commerce; H&M vs. E‑Commerce; Friendster vs. Pokec). The high‑level claim that STAGE “provably generalizes to unseen feature domains” is therefore not directly supported by the proofs for the actual setting. This disconnect weakens the theoretical contribution and overstates what is actually proven. (Section 3, before §3.2; also abstract/intro)

### Minor
- **Ablation studies confined to the appendix** — While Appendix E contains ablation experiments, the main text presents none. This forces readers to consult the appendix to understand which components (e.g., the specific form of Sᵘᵛ, the two‑stage GNN) are essential, reducing accessibility and immediate interpretability. (Section 4 reference to Appendix E)
- **Sparse baseline descriptions** — The one‑line descriptions of baselines are insufficient to assess implementation details, particularly regarding zero‑shot capability. More extensive explanations (or code release) would strengthen confidence in the comparative evaluation. (Section 4.1)

### Trivial
None.

## Nice‑to‑Haves
- Release code and data splits (especially for the E‑Commerce Stores dataset) to ensure full reproducibility.
- Visualize example STAGE‑edge‑graphs from both training and test domains to illustrate analogous dependency structures.
- Perform embedding‑space analysis (e.g., t‑SNE) to verify that STAGE clusters edges by task rather than by domain.

## Removed Points
These points are flagged to be removed, treat them with caution:
- Criticism about “missing ablation studies” is mitigated by the existence of Appendix E; the appendix is part of the paper and contains ablations, so the claim of missing ablations is inaccurate.
- Criticism about unclear handling of nodes without features is invalid; the paper explicitly states that edges are built between featured nodes of the same type and these are provided to all baselines (Section 4).
- Any nitpicks about presentation or minor notation issues are omitted as trivial.

## Novel Insights
The paper’s key insight is that statistical dependencies between node features, encoded as conditional probabilities and represented through a dedicated edge‑graph, serve as domain‑invariant signatures; this bridges maximal invariant statistical tests with GNN design, enabling a model to generalize across entirely new attribute spaces by recognizing analogous dependency patterns.

## Suggestions
- In a rebuttal, provide full implementation details for each baseline, clarifying exactly how they process graphs with unseen feature dimensions (e.g., does NBFNet‑raw use a per‑feature embedding or a fixed input layer? How are new feature names handled?). Consider adding a baseline that is explicitly permutation‑invariant (e.g., set encoder) to strengthen the zero‑shot comparison.
- Soften high‑level claims about provable generalization in the abstract and introduction to match the fixed‑d assumption stated in Section 3 (e.g., “provably generalizes under a fixed feature‑dimensional setting”).
- Move the most critical ablation results from Appendix E into the main paper to clearly demonstrate the contribution of each design choice (e.g., STAGE‑edge‑graph vs. a simpler interaction like concatenation).

## Score and Decision

I calibrated against multiple anchor papers:
- High‑scoring (6–7): OFA (7.0, accepted), GAA (6.0, accepted) – strong empirical gains, some weaknesses about missing recent baselines or complexity, but still accepted. Our paper shares strong gains but suffers from more fundamental baseline fairness concerns.
- Medium (5–5.5): SFDA (5.5, rejected) – fairness concerns about missing ablations and generic techniques; indoor 3D detection (5.0, rejected) – limited novelty and dataset concerns. Our paper’s baseline issue is more directly threatening to validity.
- Low (≤4): CGIL (3.0, rejected) – flawed claims and lack of theory; not comparable.

The STAGE paper presents a novel method with compelling empirical evidence and an interesting theoretical link. However, the inadequate specification of baseline implementations and the mismatch between theory (fixed‑d) and experiments (variable‑d) are major weaknesses that undermine confidence in the reported superiority. These issues are substantive and would require substantial revisions (re‑running or clarifying baselines, aligning theoretical claims). As such, the paper falls below the acceptance threshold.

**MY FINAL SCORE: <pineapple>5.0</pineapple>**
**MY FINAL DECISION: <orange>Reject</orange>**