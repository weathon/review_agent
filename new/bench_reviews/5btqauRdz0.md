## Summary

This paper introduces STAGE, a method for zero-shot generalization across graphs with entirely distinct node attribute domains. Instead of embedding raw feature values, STAGE constructs a fully-connected “STAGE-edge-graph” for every original edge, whose nodes correspond to individual endpoint features and whose attributes are empirical marginal and conditional probabilities (or CDFs). A two-level GNN architecture first embeds these edge-graphs and then performs message passing on the original graph topology. The paper claims both theoretical guarantees (invariance to Component-wise Order-preserving Groupoids, or COGGs) and strong empirical results on link prediction across disjoint e-commerce categories and on the distinct H&M fashion dataset, as well as node classification from Friendster to Pokec.

## Strengths

- **Novel dependency-based encoding.** The STAGE-edge-graph is a creative and principled mechanism for handling heterogeneous, misaligned, and mixed continuous/categorical feature spaces by transforming them into a unified density-based representation (Definition 2.1, Equations 2–3). This design is not present in prior GNN adaptation or textification methods.
- **Strong link-prediction gains under extreme domain shift.** When pretrained on five disparate e-commerce categories and evaluated zero-shot on H&M—with entirely different customers, products, and attribute semantics—STAGE achieves 0.4666 Hits@1 versus 0.2302 for the best zero-shot baseline (NBFNet-llm) and 0.2231 for a zero-shot structural baseline (Table 1). These are large absolute margins that are practically interesting.
- **Positive scaling with pretraining domains.** Figure 4 shows that STAGE’s zero-shot Hits@1 and MRR improve as more distinct training domains are added, while other methods plateau, suggesting it learns compositional rather than spurious patterns.
- **Clean architectural decomposition.** Separating intra-edge dependency encoding (via edge-graph GNN \(M_1\)) from inter-edge topological message passing (via \(M_2\)) is a well-motivated way to decouple feature-space variability from graph structure.

## Weaknesses

### Fatal
None.

### Major

- **Central theoretical guarantee does not cover the paper’s headline empirical claim.** The abstract states that STAGE can “provably generalize to unseen feature domains for a family of domain shifts,” and Section 3.2 frames Theorem 3.4 as enabling “zero-shot transferability” to feature domain shifts. However, Theorem 3.4 proves invariance only to COGGs (Definition A.5), whose actions permute nodes and feature dimensions and apply order-preserving transforms *within a fixed feature space*. The paper explicitly admits that “our theoretical results … are restricted to domains with a fixed number of features” (Section 3). COGG invariance therefore does not imply generalization to domains with different feature counts, semantics, or distributions (e.g., smartphone RAM vs. clothing size). The abstract and Section 3.2 repeatedly conflate invariance to within-domain rescaling/reordering with generalization to qualitatively new attribute spaces, which the theory—as presented—is mathematically incapable of supporting.
- **Suspicious baseline undermines relative improvement claims.** NBFNet-raw achieves exactly **0.0000 ± 0.0000** Hits@1 on held-out e-commerce stores and near-random performance on H&M (Table 1), well below the random baseline (~0.0026). An exactly zero score with zero variance across seeds strongly suggests an implementation bug rather than a fair baseline failure. This taints the reported percentage gains, several of which are computed relative to this broken baseline. While the absolute comparison against non-broken baselines (structural, LLM, normalized) remains favorable, the presence of a clearly malfunctioning baseline raises concerns about the overall baseline audit.
- **Node-classification evidence is critically thin.** Only a single transfer task (Friendster→Pokec, gender prediction) is reported in Table 2 after the authors dropped the age task because it was “not predictable” (Section 4.3, Appendix D). Reporting one cherry-picked task provides no meaningful evidence that the method generalizes broadly across node-classification domains, undermining the cross-task generality claims.

### Minor

- **Experiments confound unseen attributes with unseen nodes.** The held-out e-commerce link-prediction experiments use disjoint customer sets across product categories (Section 4.1), so the test domain introduces both novel attributes and novel node identities. The paper never disentangles these effects, e.g., via a controlled setting where node sets are shared but attributes are permuted or replaced. The H&M experiment partially mitigates this by using an entirely different data provider, but a cleaner ablation would strengthen causal interpretation.
- **Scalability with high-dimensional features is unresolved.** STAGE builds a fully connected edge-graph of size \(2d\) for every original edge. For graphs with moderate feature dimensionality and millions of edges, this is expensive. The paper delegates complexity and runtime analysis to Appendix F and relegates high-dimensional features to future work (Section 6), leaving the method’s scalability largely unaddressed.
- **Figure 4 references an undefined baseline “NBFNet-cw.”** This baseline is not described elsewhere in the paper, creating confusion about which method is being plotted.

### Trivial
None.

## Nice-to-Have

- A controlled ablation applying standard GNNs to rank-transformed (empirical CDF) node features *without* the two-level edge-graph construction, to isolate whether gains come from the dependency encoding or simply from order statistics.
- Inspection or visualization of \(M_1\) embeddings to verify that they encode dependency strength (e.g., positive/negative correlation, independence) rather than spurious artifacts.
- Additional node-classification tasks and dataset pairs to support broad node-classification claims.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **“Mixing empirical CDFs and conditional PMFs without normalization.”** The harsh reviewer claimed \(S^{uv}\) mixes CDFs and PMFs without accounting for “different dynamic ranges.” Both CDFs and PMFs are bounded in \([0,1]\), so they share the same dynamic range. The criticism is factually incorrect.
- **“Theorems rely on most-expressive encoders.”** Relying on most-expressive (theoretically ideal) encoders is standard practice in GNN expressivity literature and does not invalidate the theoretical framing.
- **“Comparison against Structural-Supervised is unfairly framed.”** The paper compares STAGE against both zero-shot and supervised structural baselines. The supervised comparison is presented as an additional demonstration of strength, not as the primary controlled comparison, and the zero-shot structural baseline is already included.
- **“Missing related work on copulas and rank-based domain adaptation.”** Per review guidelines, missing related works should not be flagged without external verification.
- **“Proof sketch admits sacrificing maximal expressivity.”** The paper explicitly states this trade-off in the proof sketch (Section 3.2, line 177). It is not hidden.
- **Formatting/style nitpicks, typos, and appendix-deferred proofs.** These are either parser artifacts or standard deferrals.

## Novel Insights

The observation that empirical conditional CDFs/PMFs can serve as a *universal interlingua* for cross-domain graph edges—decoupling the absolute meaning of features from their relational dependencies—is genuinely novel. If the authors honestly reframe the theory as motivating design rather than as a formal guarantee of cross-semantic transfer, and if they audit and correct the suspicious baselines, this work could represent a meaningful advance in graph foundation-model design. The positive scaling with the number of pretraining domains (Figure 4) is particularly promising, as it suggests the method is learning transferable abstractions rather than memorizing domain-specific spurious correlations.

## Suggestions

- **Reframe theoretical claims.** The abstract and Section 3 should clearly distinguish between (a) provable invariance to COGGs within a fixed feature space, and (b) the empirically motivated hypothesis that analogous dependency patterns enable generalization across distinct semantic domains. Stop claiming that Theorem 3.4 “proves” zero-shot transferability to unseen feature spaces with different semantics.
- **Audit and fix baselines.** Explain or correct the NBFNet-raw 0.0000 result. If it is due to a porting bug, report the corrected numbers and recalculate percentage gains only against valid baselines.
- **Expand node-classification evaluation.** Add at least 2–3 tasks across different dataset pairs, or scope the node-classification claims to the single demonstrated task.

## Score and Decision

**Calibration anchors used:**
- **One For All** (`/home/wg25r/review_agent/human_reviews/4IT2pgc9v6.md`, avg 7.0, Accept spotlight): A more comprehensive cross-domain graph foundation model with extensive experiments and fewer theory–empire gaps. STAGE is below this anchor due to its narrower evaluation, theory overclaim, and baseline issues.
- **IDEA** (`/home/wg25r/review_agent/human_reviews/FPpLTTvzR0.md`, avg 6.25, Reject): A paper with provable invariance claims that reviewers found overstated; strong experiments but still rejected. STAGE has similarly overstated theory but stronger absolute empirical gains on its main task.
- **TT-GREB** (`/home/wg25r/review_agent/human_reviews/rW3NVhKtQ2.md`, avg 4.5, Reject): A GNN generalization paper with theoretical concerns and limited experimental improvements. STAGE is above this anchor because its core idea is more novel and its absolute gains are substantially larger.
- **GraphFM** (`/home/wg25r/review_agent/human_reviews/zaxyuX8eqw.md`, avg 3.4, Withdrawn/Reject): A cross-domain graph pretraining paper with weak baselines and limited novelty. STAGE is well above this anchor due to its novel architecture and stronger empirical results.

STAGE has a genuinely novel architectural contribution and large absolute improvements on link prediction under extreme domain shift. However, the disconnect between the abstract’s “provable generalization to unseen feature domains” and what Theorem 3.4 actually proves (COGG invariance within fixed feature spaces) is a significant overclaim. The exactly-zero NBFNet-raw baseline and the single reported node-classification task further erode confidence. The paper sits between the medium and low anchors: it is stronger empirically than TT-GREB and GraphFM, but its theory–experiment gap and baseline issues place it below the threshold of reliable acceptability without revision.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>