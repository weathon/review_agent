## Summary

The paper proposes Forest-based Graph Learning (FGL), a paradigm that decomposes graph message passing into propagation over a forest of spanning trees. The key insight is that spanning trees are the minimal structures connecting all nodes, offering global coverage at low cost per structure. The framework includes: (1) pre-processing that augments the graph with k-NN edges from pseudo-labels; (2) a homophily-guided tree sampler using weighted Wilson sampling; (3) a linear-time tree aggregator derived from two recursions (Theorem 1); and (4) a tree fuser combining local and global information. Theoretically, Theorem 2 shows that improving edge-homophily estimates shifts the tree distribution toward higher-homophily trees. Empirically, FGL achieves strong results on 9 benchmarks, particularly on heterophilous graphs.

## Strengths

- **Conceptually novel paradigm**: Modeling graph learning as transport over spanning trees is an elegant alternative to deep local stacking and shallow global attention. The cost analysis (Eq. 1) clearly articulates why trees occupy a sweet spot between per-structure cost and structural count. The general tree aggregator in Theorem 1 (applicable to any aggregator satisfying combine/disentangle properties) is a clean theoretical contribution.

- **Strong results on challenging heterophilous benchmarks**: Large margins on Texas, Wisconsin, and Cornell (e.g., 91.89% on Texas vs. next-best 83.78%) demonstrate practical value for settings where standard GNNs struggle.

- **Thorough ablations and interpretability analyses**: Tables 3 and 4 systematically isolate component contributions. Figures 5–6 empirically correlate estimator accuracy, tree homophily, and final performance, providing useful diagnostic insight.

- **Efficiency advantages**: Table 2 shows consistently lower per-epoch time compared to most baselines. The linear-time tree aggregator is practically appealing.

## Weaknesses

### Major

- **Pre-processing confound undermines paradigm-level evaluation**: Section 4.1 adds k-NN edges based on pseudo-labels to create Ĝ, explicitly increasing homophily and ensuring connectivity. Baselines almost certainly operate on the original graph G, not the augmented Ĝ. This is critical because: (a) on heterophilous datasets where the largest gains appear, this augmentation can substantially alter the task; (b) no ablation runs FGL on the original graph (without augmentation) to isolate the forest contribution; (c) no ablation applies the same augmentation to baselines. The large improvements on Texas (+8% over two-stage estimator standalone, Table 4) and Wisconsin may partly reflect the difficulty advantage from graph rewiring. Without controlling for this, the claim that the forest paradigm itself drives the gains is not adequately supported.

- **Overclaiming "quadratic node-pair interactions"**: The abstract and contributions state the tree aggregator "realizes quadratic node-pair interactions" with linear complexity. In reality, the mechanism (Eqs. 7–8) conducts O(n) edge-level linear operations with shared parameters on a connected tree—it achieves a *global receptive field*, not true quadratic pairwise modeling. The information flow between any two nodes is mediated through shared linear transforms along tree edges, with no node-pair-specific computation. Every connected message-passing system with sufficient depth has the property that "all nodes influence all other nodes." Calling this "quadratic node-pair interactions" is misleading and conflates receptive field coverage with explicit pair-wise interaction modeling. The claim should be reframed as achieving global receptive field in linear time.

- **Missing heterophily-specialized baselines**: The paper reports results on 5 heterophilous benchmark datasets, where the largest gains appear, yet omits established heterophily-specialized methods like H2GCN (NeurIPS 2022), ACM-GCN, and GPR-GNN (ICLR 2021). The latter is particularly notable since Chien et al. (2021) is cited in Section 4.1 for homophily-related findings, making its absence conspicuous. Without these, it is unclear whether FGL genuinely advances the state of the art on heterophilous graphs or merely outperforms homophily-oriented baselines.

- **Theoretical gap between Theorem 2 and the actual algorithm**: Theorem 2 establishes monotonicity for an idealized model where edge scores take deterministic binary values (p for homophilous, q for heterophilous edges) based on ground-truth labels. In practice, edge scores are continuous values from a learned local attention mechanism (Eq. 3) trained on pseudo-labels. The paper provides no formal or even informal bridge between these settings—no discussion of how noise in estimator scores, continuous-vs-binary scoring, or label sparsity affect the monotonicity guarantee. This gap means the main theoretical narrative ("refining the estimator provably yields a better tree distribution") is only rigorously true for the idealized model, not for the actual learning dynamics.

### Minor

- **Small heterophilous datasets**: Texas (183 nodes), Wisconsin (251 nodes), and Cornell (183 nodes) are known to be extremely small and sensitive to hyperparameter choices and random seeds. The large performance variations on these datasets should be contextualized accordingly, and results on larger heterophilous benchmarks (e.g., Squirrel, Chameleon) would strengthen the evaluation.

- **Local module carries substantial weight**: In Table 3, removing the global submodule drops Texas from 91.89% to 82.88% and Wisconsin from 86.27% to 83.92%, but removing the local submodule drops Texas to 69.93% and Wisconsin to 75.49%. This asymmetry—where the local module (a multi-step propagation on Ĝ, Eq. 9) contributes more than the tree-based global module—raises questions about the relative importance of the novel paradigm vs. the strong local baseline it is paired with.

- **Efficiency comparison does not account for preprocessing overhead**: Table 2 reports per-epoch time for the student model only, excluding the cost of pseudolabel pretraining, homophily estimator training, and tree sampling. For a fair end-to-end efficiency comparison, total wall-clock time including all stages should be reported or at least discussed.

- **Structural information loss from tree decomposition is under-analyzed**: A forest of N_T spanning trees on an n-node, m-edge graph preserves at most N_T(n−1) of m edges. For sparse graphs this may be sufficient, but for dense graphs the fraction preserved can be very small. No analysis quantifies how many trees are needed to adequately represent different graph densities, nor what structural properties are lost.

### Trivial

- The "optimal range of 6 to 10 trees" (Fig. 4 discussion) is observed empirically but not explained. Minor, as this is a reasonable design choice that is empirically validated.

- The α_{Fix(v)→v} notation in Eq. 8 is not clearly defined in the main text (referenced only indirectly through Fig. 3), slightly hurting local readability.

## Nice-to-Haves

- Run FGL on original graphs (without k-NN augmentation) and/or apply the same preprocessing to a few baselines to isolate the contribution of the forest paradigm from graph rewiring. This single experiment would significantly strengthen the paper.

- Evaluate on larger heterophilous benchmarks (Squirrel, Chameleon) and include heterophily-specialized baselines (H2GCN, ACM-GCN, GPR-GNN).

- Extend or discuss Theorem 2 for continuous edge scores and noisy estimators to bridge the theory-practice gap.

- Report total end-to-end training time including all auxiliary stages.

## Removed Points

These points are flagged to be removed; treat them with caution:

1. **"Preprocessing creates circular dependency" (Neutral Reviewer)**: The pseudo-label step uses labeled nodes (which exist by assumption in semi-supervised learning) to train a simple predictor, then augments the graph. This is not circular—it's a standard self-training/bootstrap approach common in semi-supervised learning. The concern about noisy pseudo-labels is valid but doesn't constitute a "circular dependency."

2. **"Marginal improvements on homophilous benchmarks" (Neutral Reviewer)**: On Cora (85.46 vs. 85.34), Citeseer (74.42 vs. 74.46), and Pubmed (81.00 vs. 81.30), FGL is essentially competitive with the best prior methods. This is not a weakness—matching SOTA on homophilous graphs while substantially improving on heterophilous ones is a valid contribution profile.

3. **"No comparison with other spanning-tree or subgraph methods" (Neutral Reviewer)**: This is a missing-related-work concern; per the rules, I cannot confirm such works exist and should not flag their absence.

4. **"Not evaluated on truly large-scale graphs" (Spark)**: The paper evaluates on ArXiv (~169K nodes), which is a mainstream benchmark in this area. Requesting ogbn-products or Papers100M is scope creep beyond what is standard for this type of contribution.

5. **"No controlled experiment varying homophily estimator quality" (Spark)**: Figure 5 already shows performance vs. estimator accuracy, which partly addresses this, and the idealized experiment of injecting calibrated noise is a narrow diagnostic rather than a core flaw.

6. **"Novelty concerns—components are individually established" (Human Finder)**: The same could be said of most papers that combine known ingredients in a new way. The specific combination and the theoretical framework are the contribution.

## Novel Insights

The relationship between Theorem 2 and the practical algorithm is notably looser than the paper presents. The theorem proves that under idealized binary edge scoring, increasing the homophily/heterophily score ratio monotonically improves the expected tree homophily—bound by the graph's homophilous connected components. This structural upper bound (1 − (NHCC−1)/(n−1)) is an interesting graph-theoretic insight: it means that even with perfect estimation, the achievable tree homophily is limited by the graph's label structure. For heterophilous graphs with many small homophilous connected components, this bound could be quite low, which may explain why the pre-processing step (adding kNN edges) is necessary—it doesn't just ensure connectivity but also improves this theoretical bound. The paper misses this connection between its own theory and the practical necessity of augmentation.

## Suggestions

- **Most impactful**: Add one experiment running FGL on original (non-augmented) graphs to show the marginal contribution of the forest paradigm. If possible, also run 1–2 strong baselines on the augmented graphs. This would decisively address the biggest concern.

- **Important for completeness**: Add GPR-GNN, ACM-GCN, and H2GCN as baselines on the heterophilous datasets; these are standard in the community.

- **Framing**: Replace "quadratic node-pair interactions" with "global receptive field" or "quadratic effective receptive field in linear time" throughout the paper. The current phrasing misrepresents the mechanism.

- **Theory**: Add a paragraph discussing how Theorem 2's guarantee degrades under noisy/continuous edge scores, or at minimum acknowledge the gap explicitly.

- **Reporting**: Report total training time (including all auxiliary stages) in Table 2 or a supplementary table.

## Score and Decision

I compared this paper against several calibration anchors:

- **NeuralWalker** (8,6,8,6 → Accept Poster): Walk-based graph method with strong results and similar evaluation breadth. FGL has weaker evaluation controls (preprocessing confound) but comparable novelty.
- **S4G** (8,3,3 → Reject): Structured state-space approach with similar theory-practice gaps. FGL has stronger empirical results but similar overclaiming issues.
- **DeltaGNN** (3,5,5,6,5 → Reject): Efficiency claims, small datasets, missing heterophily baselines. FGL shares these issues but has more genuine novelty.
- **SoLAR** (3,6,3,5,5 → Withdrawn/Reject): Graph rewiring with pseudo-labels, similar preprocessing pipeline. FGL is more novel in its forest mechanism but shares the rewiring confound.

FGL sits above SoLAR (which had a near-identical preprocessing concern and weaker novelty) and DeltaGNN (weaker contribution), but below NeuralWalker (which had cleaner evaluation). The preprocessing confound and the "quadratic interactions" overclaim are significant but not fatal—the core idea is genuinely novel and the framework is well-constructed. However, the evaluation does not convincingly isolate the forest paradigm's contribution from the preprocessing effect, and major baselines for the most important setting (heterophily) are missing.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>