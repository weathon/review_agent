=== CALIBRATION EXAMPLE 26 ===

# Final Consolidated Review
## Summary

SubDyve is a network-based virtual screening framework for low-label regimes that constructs a subgraph-aware similarity network from class-discriminative substructures and performs iterative seed refinement guided by Local False Discovery Rate (LFDR) estimation. It aims to control false positives from topological bias while propagating activity signals from few known actives, evaluated on ten DUD-E targets and a 10-million-compound ZINC dataset for CDK7.

## Strengths

- **Strong empirical performance under label scarcity:** SubDyve achieves the best average rank (1.6) across BEDROC and EF1% on 10 DUD-E targets and leads all metrics on the PU dataset (BEDROC 83.44, EF1% 97.59), with large margins over foundation models and docking baselines. The varying-seed-size ablation (Tables 4, 20, 21) showing robustness down to 5 seeds is particularly convincing for the low-data claim.
- **Principled statistical calibration integrated into network propagation:** The LFDR-based seed refinement directly addresses the known problem of topological bias in network propagation (Picart-Armada et al., 2021). Proposition 1 provides a theoretical FDR bound, and Figure 4 empirically demonstrates substantially better calibration (ECE 0.204 vs. 0.511) compared to probability-based thresholding across 10 targets.
- **Comprehensive ablation and interpretability analysis:** The paper provides extensive ablations isolating subgraph construction, LFDR refinement, GNN features (Table 24), subgraph pattern dimensions (Figure 9), and LFDR sensitivity (Tables 5, 22). The pharmacophoric relevance analysis (Appendix G.4) and active-decoy ranking gap analysis (Section 4.3.2) demonstrate that mined subgraphs align with known binding mechanisms rather than capturing spurious correlations.

## Weaknesses

### Major:

- **"Zero-shot" terminology mischaracterizes the actual experimental setup.** The DUD-E evaluation sources seed compounds from homologous proteins with up to 90% sequence identity, with seed counts ranging from 174 to 1,277 per target (Table 7). This is more accurately described as "homolog transfer" or "cross-target few-shot" learning. While the stricter 0.5 identity experiment (Table 15) partially mitigates this, the paper still frames the main results as "zero-shot." For truly novel targets without close homologs in public databases, the method's applicability is untested. This matters because the claimed generalization strength is overstated relative to what the experiments actually demonstrate.

- **Subgraph-based library filtering may systematically exclude novel chemotypes.** Section 3.2 filters Q to retain only compounds matching at least one mined subgraph pattern, forming Q′. In the PU dataset, this reduces 10M compounds to ~30K (a 99.7% reduction). Compounds with novel binding mechanisms that don't share substructures with known actives—the very compounds virtual screening should discover—are excluded before propagation begins. The paper neither quantifies what fraction of true actives are lost to this filtering nor discusses the risk of reinforcing existing chemical bias. This is a fundamental limitation for a method whose goal is identifying new bioactive compounds.

- **FDR control guarantee relies on unverified distributional assumptions.** Proposition 1 assumes test statistics follow the two-group mixture f(z) = π₀f₀(z) + π₁f₁(z) and that LFDR estimates are accurate. However, GNN logits on molecular similarity graphs are likely correlated due to graph structure, violating the independence assumptions underlying the standard LFDR framework. The paper provides no empirical validation of observed FDR at various LFDR thresholds—only calibration (ECE) against probability thresholding is shown (Figure 4). Without reporting actual false discovery rates among promoted seeds, the practical reliability of FDR control remains unsubstantiated.

- **Ablation results undermine the stated contribution balance.** Table 3 shows Subgraph alone (no LFDR) achieves BEDROC 78.68, while LFDR alone (no subgraph) achieves only 63.78, and both together achieve 83.44. The paper claims "most gains come from their interaction," but subgraph construction alone accounts for ~74% of the improvement over baseline (78.68 − 63.78 = 14.9 out of 83.44 − 63.78 = 19.66). The LFDR refinement adds meaningful but incremental value (~5 points). This isn't reflected in how the contributions are framed, which gives equal emphasis to both components.

### Minor:

- **Asymmetric seed removal in Algorithm 2:** S1 seeds can never be removed regardless of their LFDR value (Algorithm 2, lines 4–12), while augmented seeds can be removed if their LFDR exceeds τ_FDR. This anchoring assumption needs justification—if an S1 seed receives a high LFDR due to propagation dynamics, retaining it could introduce persistent bias into subsequent iterations.

- **Max pooling for ensemble aggregation not ablated:** Section 3.4 uses element-wise max pooling across N stratified splits to construct the final seed vector. Max pooling is aggressive—it selects the highest weight per compound across splits, potentially amplifying split-specific false positives. Mean pooling or trimmed mean could be more conservative. No ablation of this design choice is provided.

- **π₀ estimation unclear:** Algorithm 3 lists π₀ as a required input for LFDR estimation, and Equation 4 uses it to update seed weights. The paper does not explain how π₀ is determined in practice—whether it's estimated from the data, set heuristically, or treated as a hyperparameter. This is important because seed weight updates are directly modulated by π₀ (Eq. 4: w_i ← w_i + β(σ(z_i) − π₀)).

- **Pretrained ChemBERTa features contribute substantially but aren't discussed as such:** Table 24 shows that removing ChemBERTa embeddings causes the largest single-component drop (BEDROC 83.44 → 79.82, Δ = 3.62), comparable to removing the NP score or seed weight. Yet the paper's narrative emphasizes subgraph patterns and LFDR as the key innovations. The relative importance of pretrained embeddings should be acknowledged more transparently.

- **PU dataset baseline fairness ambiguity:** It is unclear whether deep learning baselines (BIND, DrugCLIP, PSICHIC) were evaluated on the same ~30K filtered subset used by SubDyve, or on the full 10M library. If baselines screened the full library while SubDyve screened only the pre-filtered subset, the comparison is confounded by library size. This should be explicitly stated.

### Trivial:

- **Notation inconsistency between Algorithm 2 and Equation 4:** Algorithm 2 uses parameter name "b" (baseline) while Equation 4 uses π₀ for the same quantity. These should be unified for clarity.

## Nice-to-Haves

- **Empirical FDR validation:** Report the observed false discovery rate among promoted seeds at various τ_FDR thresholds to empirically verify Proposition 1's theoretical bound under the method's actual operating conditions.
- **Scaffold diversity analysis:** Quantify Bemis-Murcko scaffold diversity of retrieved hits relative to seeds, directly testing whether the subgraph filtering biases results toward close analogs rather than novel chemotypes.
- **Evaluation on targets without close homologs:** Test SubDyve on targets with no known homologs (sequence identity < 0.3) in public databases to establish the method's true minimal data requirements and practical limits.
- **Temporal validation split:** Evaluate using a temporal split (older vs. newer ChEMBL releases) to simulate prospective discovery rather than random masking.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing computational cost analysis"** (from harsh critic and transferred weakness): The paper actually provides detailed timing in Appendix E (Tables 11, 12) and runtime/memory comparisons in Appendix F.9 (Table 25, Figure 10). This criticism is factually wrong.
- **"Limited target diversity—all experiments on kinases"** (from harsh critic): DUD-E targets include ACES (hydrolase), FA10 (transferase), UROK (protease), THRB (serine protease), ANDR (nuclear receptor), etc. The targets are diverse, not just kinases.
- **"Proposition 1 proof is incomplete"** (from harsh critic): The proof in Appendix A is standard. The step FDR ≤ mFDR via V_α ≤ R_α is a well-known result (Efron, 2005). The proof is correct as stated.
- **"Fair comparison concerns about baseline evaluation protocol"** (from transferred weakness): The paper explicitly states baselines use best reported hyperparameters and evaluates them on the same datasets. This is standard practice.
- **"Insufficient justification for hyperparameter choices"** (from transferred weakness): The paper provides search spaces and selected values in Tables 5–6 and sensitivity analysis in Tables 5, 22. This is standard practice for the field.
- **"Parser artifacts / formatting issues"** (from harsh critic): Explicitly excluded per rules.
- **"Random seeds not reported"** (from harsh critic): Reproducibility nitpick, excluded per rules.

## Novel Insights

The interaction between subgraph-aware graph construction and LFDR refinement reveals a subtle dependency: Table 3 shows that LFDR refinement *without* the subgraph network actually *degrades* BEDROC from 79.04 to 63.78 on the PU dataset—a negative contribution. This suggests that LFDR's statistical calibration is only effective when operating on a graph whose edges encode task-relevant structure; on a generic fingerprint graph, the logits lack sufficient signal for LFDR to separate true and false discoveries. The practical implication is that subgraph-aware graph construction is not merely an incremental improvement but a prerequisite for LFDR refinement to function at all, making the two components co-dependent rather than independently additive.

## Suggestions

- **Reframe "zero-shot" as "homolog transfer"** throughout the paper, or add a clear definition of what "zero-shot" means in your specific context (i.e., no actives from the test target itself, but actives from homologous targets are available). This prevents reader confusion and aligns claims with evidence.
- **Report empirical FDR among promoted seeds** at each LFDR threshold. Add a table or plot showing observed false positive rate vs. nominal LFDR threshold for the DUD-E targets, directly testing whether Proposition 1's guarantee holds in practice.
- **Quantify the filtering loss:** Report what fraction of known actives (from the 70% held-out set in the PU experiment) are excluded by the subgraph matching filter. If few true actives are lost, this allays the novel-chemotype concern; if many are lost, the method's "large-scale" screening claim needs qualification.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
