=== CALIBRATION EXAMPLE 69 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title—"Subgraph-Driven Dynamic Propagation for Virtual Screening Enhancement Controlling False Positive"—is ungainly and somewhat imprecise: the method controls FDR at the seed-expansion level, which is not the same as controlling false positives in the final ranked list.

More critically, the abstract's framing of "zero-shot conditions" is misleading. SubDyve is not zero-shot in any standard sense. It requires dozens to over a thousand target-specific seed molecules curated from PubChem (Table 7 reports 174–1,277 seeds per DUD-E target). What the authors mean is "zero-shot with respect to direct training on target labels," but baseline comparisons (PharmacoMatch, CDPKit, DrugCLIP) are genuinely zero-shot. Labeling the paper's setting "zero-shot" obscures this asymmetry.

The performance margins cited in the abstract (+34.0 BEDROC, +24.6 EF1%) reflect cherry-picked comparisons (primarily vs. PharmacoMatch on ACES/EGFR). Against the more competitive baseline DrugCLIP, margins are considerably smaller and SubDyve does not outperform on FA10, THRB, or UROK.

---

### Introduction & Motivation

The motivation is well-constructed and the two limitations of NP highlighted (substructure blindness, topological bias inflating FP) are genuine. The related work summary is adequate, though SPRINT (McNutt et al., 2024) and similar large-scale ligand-only VS methods are mentioned only in passing without direct comparison.

**Concern:** The contributions as stated mix method and result claims. "State-of-the-art performance" is listed as a contribution, which is inappropriate—performance is evidence for contributions, not a contribution itself.

---

### Method (Sections 3.1–3.4 and Appendix B)

**The "zero-shot" and data-leakage framing (critical):** The comparison in Section 4.1.1 is framed as zero-shot, but SubDyve seeds are drawn from PubChem proteins at sequence identity ≤0.9 to the DUD-E target. The ADA target is treated inconsistently—it uses a looser threshold of 0.953 "to enable sufficient subgraph extraction." This ad hoc loosening of the criterion for a specific target is not justified and undermines the experimental rigor.

**Selection bias from 99.7% candidate filtering (serious):** In the PU/ZINC experiment, SubDyve filters 10,082,034 ZINC compounds down to ~30,000 (Section 4.1.2, Appendix D.2). This 99.7% reduction retains only compounds matching at least one discriminative subgraph from the seed actives. This is a strong structural pre-selection that effectively biases the candidate pool toward compounds similar to known actives. Baseline methods (GRAB, DrugCLIP, PSICHIC, NP variants) presumably screen more or all compounds—this must be clarified. The comparison in Table 2 may be evaluating fundamentally different tasks.

**Use of S₂ for both training and early stopping (methodological concern):** In Section 3.3.2, S₂ is used as the supervision signal for all three loss terms (BCE, RankNet, Contrastive) *and* as the criterion for early stopping (via enrichment scores). The authors state that "early stopping relies solely on intermediate enrichment scores, preventing bias from direct training objectives," but this is unconvincing—the GNN parameters are being updated to minimize L_total computed over S₂ at every iteration, so enrichment on S₂ is not an independent validation. With very small S₂ (10% of 30% of 1,468 ≈ 44 compounds), this risks severe overfitting to S₂.

**LFDR assumptions not validated (theoretical concern):** Algorithm 3 applies Efron's two-group mixture model (f(z) = π₀f₀(z) + π₁f₁(z)) where z-scores are GNN logits standardized by their mean and variance. The null density f₀ is taken to be a Gaussian (null_pdf). This is a strong distributional assumption. GNN logits after training are not guaranteed to be approximately normally distributed—especially in severely imbalanced settings (30,000 candidates, ~44 positives in S₂). The paper never validates whether the empirical z-score distribution is well-described by this mixture model. Without this, Proposition 1's FDR bound is a theoretical statement that may not apply in practice.

**Proposition 1 is not a novel contribution:** The proposition is a direct restatement of Efron (2005, Theorem 1). The proof (Appendix A) reproduces Efron's standard argument. Presenting it as "Proposition 1" with a proof implies novelty that is not present.

**Feature redundancy in Eq. (1):** The GNN input includes w_i (seed propagation weight), n_i^NP (NP score derived from w_i), s_i^PCA (PCA-based seed similarity), and h_i^hyb (average of NP rank and PCA rank). These four components capture highly correlated information. No ablation evaluates which features are actually informative (Appendix F.8 mentions a GNN feature ablation, but the main text does not summarize findings). The design rationale is not clearly motivated.

**Max pooling for ensemble aggregation (unmotivated):** Section 3.4 uses element-wise max pooling to combine N seed weight vectors. No justification is provided for max vs. mean or voting. This is a non-trivial design choice with potential pathological behavior (a single split's high-confidence false positive can dominate).

**Subgraph mining requires negative compounds:** Section 3.2 mentions that the SSM algorithm mines patterns from "labeled seed set S_train using curated negative molecules." The source and curation process for these negatives in the DUD-E experiment is not sufficiently described. For the PU/ZINC experiment, where negatives are treated as unlabeled, the origin of negatives for SSM is unclear.

---

### Experiments & Results

**Unfair comparison in DUD-E zero-shot benchmark (major concern):** PharmacoMatch, CDPKit, DrugCLIP, MoLFormer, AutoDock Vina—all true zero-shot or structure-based methods—are compared against SubDyve which uses 174–1,277 target-specific seed molecules (Table 7). This is not a zero-shot comparison. The authors attempt to justify this by calling it "zero-shot with respect to target-specific training," but this is a reinterpretation of the term that differs from every other method in the table. An honest comparison should label SubDyve's setting as "few-shot" and compare it to few-shot baselines.

**Ablation in Table 3 reveals a counterintuitive result that is not explained:** The "Subgraph only" row (BEDROC 63.78 ± 11.43, EF1% 67.22 ± 16.61) is worse than the "no subgraph, no LFDR" baseline rdkit+NP (BEDROC 79.04, EF1% 89.24). The subgraph fingerprint network *alone* hurts performance substantially compared to the ECFP/RDKit baseline. The authors assert the gains come from the "interaction" between subgraph and LFDR, but this finding—that the subgraph network by itself is actually worse—directly undermines the claim that subgraph-discriminative features are the primary innovation. The paper provides no analysis of why this happens.

**Statistical significance not reported consistently (Table 2):** Significance markers are present for BEDROC and EF1% but "—" for EF0.5%, EF5%, EF10%. This is unexplained. If SubDyve's improvement is not statistically significant at certain thresholds, this should be discussed rather than silently omitted.

**DUD-E performance on KIT is weak and not analyzed:** SubDyve achieves BEDROC 44 ± 3 and EF1% 13.8 ± 2.6 on KIT, while MoLFormer achieves 66 ± 1 and 36.8 ± 0.9. This is a substantial underperformance that is not discussed. Including one genuinely difficult case where the method fails, without commentary, creates an incomplete picture.

**Runtime comparison is not on equal footing (misleading):** The paper reports SubDyve (1,088s) vs. AutoDock Vina (13,343s), claiming 12.3× speedup. AutoDock Vina docks into explicit protein structures and is performing a fundamentally different task (pose generation + scoring). SubDyve operates only on ligand-side features. The comparison is inherently misleading.

**Sensitivity analysis of LFDR (Table 5) shows essentially no sensitivity:** BEDROC variation across τ ∈ {0.05, 0.10, 0.30, 0.50} is ≤0.44%, and EF variation ≤0.48%. This is unusually flat. Either the method is remarkably robust (a positive result, but it should be explained mechanistically), or the LFDR threshold doesn't meaningfully control the seed expansion in practice—which would raise questions about whether LFDR is doing anything substantive.

**PU dataset: the 10% EF is capped at 10.00 for multiple methods:** Both GRAB and SubDyve achieve EF10% = 10.00 ± 0.00, which is the theoretical maximum in a dataset where true actives constitute exactly 10% of the ranked list. No analysis of this ceiling effect is provided.

---

### Limitations & Broader Impact

The conclusion section acknowledges that the method is ligand-only and that integration of protein structure is future work. However, the following limitations are not acknowledged:

1. **Scalability for pairwise similarity:** The paper notes that computing similarity for 1M compound pairs takes ~5 hours (Appendix E). For a genuine 10M compound library without pre-filtering, this would be computationally prohibitive. The current evaluation only works because of the 99.7% pre-filtering.

2. **Generalizability across chemotype domains:** All evaluations use either DUD-E targets or CDK7—a kinase. It is unclear how well subgraph mining generalizes to targets without close structural families in PubChem (e.g., GPCRs, ion channels).

3. **Dependence on PubChem annotation quality:** The zero-shot setup relies entirely on PubChem bioactivity annotations. If these annotations are inconsistent (different assay conditions, varying potency cutoffs), the subgraph patterns mined may be noisy.

---

### Overall Assessment

SubDyve presents a creative integration of discriminative subgraph mining with LFDR-based iterative seed refinement for virtual screening in low-label settings, and the empirical results on the PU/CDK7 dataset are genuinely competitive. However, the paper has a fundamental framing problem: labeling its evaluation "zero-shot" when the method uses hundreds of target-specific seed molecules is misleading and creates an unfair comparison against truly zero-shot baselines. The ablation study reveals a counterintuitive result—that the subgraph network alone performs worse than standard RDKit+NP—that goes unexplained and challenges the narrative around the method's key innovation. The 99.7% pre-filtering of the ZINC library introduces selection bias that calls into question whether the PU dataset comparison is fair. The LFDR application rests on distributional assumptions (Gaussian null, two-group mixture) that are not validated in this domain. In its current form, the contribution does not meet ICLR's bar for rigor and scientific honesty in framing. A significant revision correcting the zero-shot framing, properly explaining the subgraph-only failure mode, and clarifying the ZINC filtering's effect on comparisons would substantially strengthen the paper.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents SubDyve, a network-based virtual screening framework designed for low-label regimes where few active compounds are known. It constructs a subgraph-fingerprint similarity network derived from class-discriminative patterns and employs a novel iterative seed refinement mechanism guided by local false discovery rate (LFDR) estimates to control false positives during network propagation. Extensive evaluations on the DUD-E benchmark and a 10-million compound ZINC dataset demonstrate that SubDyve achieves state-of-the-art performance in early enrichment metrics compared to both foundation models and traditional network propagation baselines.

### Strengths
1.  **Addressing Critical Limitations in Network Propagation:** The paper identifies and rigorously addresses two specific weaknesses of existing Network Propagation (NP) methods: the reliance on generic fingerprints that miss subtle substructural cues and the topological bias leading to inflated false positives. The proposed solutions (subgraph fingerprints and LFDR refinement) are well-motivated and directly target these identified failures.
2.  **Robust and Comprehensive Empirical Validation:** The evaluation is extensive, covering both standardized benchmarks (DUD-E with zero-shot/few-shot setup) and a large-scale real-world scenario (CDK7 on 10M ZINC compounds). The inclusion of diverse baselines, including foundation models (MoLFormer, ChemBERTa), structure-based scoring (AutoDock Vina), and other NP variants (Yi et al., 2023), provides a thorough context for the claimed SOTA performance.
3.  **Statistical Rigor in Method Design:** The integration of Local False Discovery Rate (LFDR) control (Efron, 2005) into the seed refinement loop is a novel algorithmic contribution that provides theoretical guarantees on false discovery bounds. Furthermore, the ablation studies (stability across thresholds, seed sizes, and component removal) are detailed and supported by statistical significance testing (e.g., paired t-tests, bootstrapping).

### Weaknesses
1.  **Terminological Ambiguity Regarding "Zero-Shot":** The abstract and Table 1 claim "Zero-Shot Screening on DUD-E Targets," yet the methodology details curating seed molecules from homologous proteins in PubChem. Since the model uses known actives from these homologs to initialize the network, this is technically a few-shot or transfer learning setting rather than strict zero-shot (where no active labels are used), which requires clarification to manage expectations regarding data leakage and difficulty.
2.  **Preprocessing Overhead and Scalability:** While inference is fast, the initial network construction involves supervised subgraph mining (approx. 108 seconds per iteration as noted in Appendix E.5) for each target. For screening across many targets, this preprocessing cost scales linearly per target and is significantly higher than static fingerprint methods (RDKit + NP takes ~9 mins vs. SubDyve's ~17 mins per target in the paper). The trade-off between this added complexity/cost and the accuracy gain could be analyzed more critically regarding large-scale deployment.
3.  **Complexity of the Pipeline:** The method combines three distinct modules (subgraph mining, GNN training, LFDR-guided propagation) which introduces a large number of hyperparameters and tuning steps (e.g., `M` iterations, stratified splits `N`, thresholds, lambda coefficients). Although stability is tested, the engineering complexity is non-trivial compared to single-stage embedding models (e.g., MoLFormer), which could hinder adoption in high-throughput settings.

### Novelty & Significance
*   **Novelty:** The core novelty lies in the integration of LFDR-guided iterative seed refinement into network propagation for virtual screening. While subgraph mining and network propagation exist in isolation, their combination with explicit statistical false discovery control to mitigate topological bias is a distinct methodological contribution.
*   **Significance:** The paper addresses a high-impact problem in computational chemistry (label scarcity in drug discovery). Achieving significant performance improvements over foundation models in these regimes is practically significant, as it reduces experimental costs by prioritizing better candidates. The methodological approach (LFDR for graph expansion) may also generalize to other label-efficient graph tasks.
*   **Clarity:** The core mechanics are described clearly in the main text, though some LaTeX artifacts in the provided text hinder the precision of equation referencing (e.g., formatting in loss functions). The logical flow from problem identification to solution is coherent.
*   **Reproducibility:** The paper provides sufficient detail on the loss functions, network architecture, and experimental setup in the appendices. The inclusion of a GitHub repository link in the text further aids reproducibility, though access is external to the review.

### Suggestions for Improvement
1.  **Clarify Data Usage in Zero-Shot Claim:** Either adjust the terminology to "Few-Shot Transfer" in the abstract/introduction or explicitly justify why using active labels from homologous proteins does not violate the zero-shot assumption (e.g., emphasizing the "target-agnostic" nature of the learning).
2.  **Analyze Cost-Performance Trade-off More Deeply:** Include a discussion or figure showing the accuracy gain (e.g., BEDROC improvement) normalized against the time or computational resources required compared to static fingerprint baselines. This would help readers weigh if the complexity is justified for their specific use case.
3.  **Simplify or Modularize the GNN Component:** To aid adoption, provide an ablation or guidance on whether the GNN is strictly necessary or if a simpler propagation scheme with the LFDR seed update could achieve similar results. Currently, training the GNN adds significant overhead compared to standard NP.
4.  **Explain Subgraph Mining Scalability:** Provide more detail on how the subgraph mining step scales with library size. If the 10M dataset required filtering down to 30k prior to mining (as noted in Section 4.1.2), clarify this bottleneck and how it affects the ability to process unfiltered chemical spaces directly.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Full 10M Evaluation:** The abstract claims screening on a 10-million-compound ZINC dataset, but Section 4.1.2 admits filtering to ~30k compounds before evaluation. Evaluate on the full 10M without subgraph-based pre-filtering to validate the large-scale claim.
2. **Compound Overlap Check:** Report maximum Tanimoto similarity between PubChem seed molecules and DUD-E test actives to rule out data leakage via identical compounds across databases masquerading as zero-shot generalization.
3. **LFDR Null Validation:** Test LFDR calibration on decoy-only graphs to verify if the theoretical FDR bound holds when GNN logits are correlated via graph propagation, which violates independence assumptions.
4. **Cross-Family Testing:** Evaluate on non-kinase DUD-E targets (e.g., GPCRs, Nuclear Receptors) to ensure performance gains aren't specific to kinase structural motifs prevalent in the current selection.
5. **Compute-Normalized Baselines:** Compare against baselines trained with equivalent GPU hours to ensure gains aren't simply due to SubDyve's higher computational budget rather than methodological superiority.

### Deeper Analysis Needed (top 3-5 only)
1. **Graph Dependency Impact:** Analyze the correlation of GNN logits between connected nodes; high correlation violates LFDR independence assumptions, directly undermining the "Controlling False Positive" title claim.
2. **Mining Cost-Benefit:** Quantify performance drop when replacing supervised subgraph mining with random subgraphs to isolate if gains come from mining complexity or just the propagation mechanism.
3. **Iterative Error Accumulation:** Track the precision of added seeds per iteration to determine if false positives compound over refinement steps despite LFDR filtering.
4. **Subgraph Dimensionality Sensitivity:** Analyze performance variance against the number of mined patterns ($d$), as Figure 9 suggests saturation but doesn't reveal instability risks at lower dimensions.
5. **Topology Bias Verification:** Compare hub node degrees in subgraph graphs vs. ECFP graphs to verify if the method actually reduces topological bias or just shifts it to frequent subgraphs.

### Visualizations & Case Studies
1. **Empirical FDR Curve:** Plot observed false discovery rate vs. LFDR threshold to empirically verify if the theoretical bound in Proposition 1 is violated in practice due to graph dependencies.
2. **Seed Trajectory Plot:** Visualize the scaffold diversity of seed sets over iterations to show if refinement expands chemical space or collapses onto a single chemotype.
3. **False Negative Examples:** Display actives removed by LFDR refinement to characterize the sensitivity cost of the proposed specificity control.
4. **Pharmacophore Alignment:** Overlay mined subgraphs onto CDK7 crystal structure binding pockets to verify physical plausibility beyond statistical enrichment.
5. **Runtime Scaling Curve:** Plot preprocessing time vs. library size to expose the true scalability bottleneck of subgraph mining beyond the filtered 30k subset.

### Obvious Next Steps
1. **Clarify Screening Scale:** Explicitly state the 30k filtering step in the abstract or remove the "10-million-compound" claim to avoid misleading readers about scalability.
2. **Revise FDR Theory:** Adjust Proposition 1 to account for graph-induced dependencies or label it as an empirical heuristic rather than a theoretical guarantee.
3. **True Zero-Shot Test:** Evaluate on targets with no homologs in PubChem to test generalization beyond homolog-based ligand transfer.
4. **Release Mining Code:** Provide optimized code for the supervised subgraph mining step, as this is the primary computational bottleneck affecting reproducibility.
5. **Budget-Constrained Comparison:** Re-evaluate baselines with extended training time to ensure SubDyve's gains justify its higher preprocessing complexity.

# Final Consolidated Review
## Summary
SubDyve proposes a network-based virtual screening framework for low-label regimes that combines class-discriminative subgraph mining with LFDR-guided iterative seed refinement. The method constructs a subgraph fingerprint network from mined patterns and propagates activity signals with calibrated seed expansion to control false positives. Evaluations on DUD-E targets and a CDK7 screening task demonstrate strong early enrichment performance compared to foundation models and traditional network propagation baselines.

## Strengths
- **Targeted problem formulation:** The paper identifies two concrete limitations of network propagation for virtual screening—reliance on generic fingerprints that miss activity-relevant substructures, and topological bias causing false positives—and proposes complementary mechanisms to address both (subgraph-aware graph construction and LFDR-based seed refinement).
- **Comprehensive empirical evaluation:** The paper evaluates on ten DUD-E targets with curated homolog-based seeds, plus a large-scale CDK7 task with 10M ZINC compounds (filtered to ~30k). Comparisons span structure-based methods (AutoDock Vina), foundation models (DrugCLIP, MoLFormer, ChemBERTa), pharmacophore matching (PharmacoMatch, CDPKit), and NP baselines with 12 fingerprint variants. Ablation studies examine component contributions, seed set sizes, and hyperparameter sensitivity.
- **Theoretically grounded seed refinement:** The LFDR-based seed expansion provides a statistical FDR bound (Proposition 1), integrating ideas from Efron (2005) into an iterative graph learning pipeline. The calibration analysis in Appendix F.6 demonstrates empirically that LFDR thresholding outperforms naive probability-based thresholding in expected calibration error across targets.

## Weaknesses

- **Misleading "zero-shot" terminology:** The abstract frames the DUD-E evaluation as "zero-shot conditions," yet SubDyve requires curating 174–1,277 seed molecules from PubChem homologs per target (Table 7). This is transfer/few-shot learning, not zero-shot in the conventional sense where baselines like PharmacoMatch, DrugCLIP, and MoLFormer operate. The comparison is asymmetric: SubDyve uses target-specific ligand information while true zero-shot baselines do not. This framing should be corrected to "few-shot transfer" or "homolog-based transfer."

- **Subgraph-only ablation shows performance degradation:** Table 3 reveals a counterintuitive finding: the subgraph fingerprint network alone (without LFDR) achieves BEDROC 63.78 and EF1% 67.22, substantially worse than the rdkit+NP baseline (79.04 and 89.24). This means the core innovation—class-discriminative subgraph fingerprints—degrades performance when used in isolation. The paper claims gains arise from "interaction" between subgraph features and LFDR, but this failure mode is not analyzed. The negative result warrants explanation: do mined subgraphs overfit to seed compounds? Do they create graph structures that amplify topological bias?

- **99.7% candidate pre-filtering in PU experiment:** The 10M ZINC library is reduced to ~30k compounds by retaining only molecules matching at least one discriminative subgraph pattern (Section 4.1.2, Appendix D.2). This structural pre-selection fundamentally biases the candidate pool toward compounds similar to known actives. It is unclear whether baseline methods (GRAB, DrugCLIP, PSICHIC) are evaluated on the same filtered set or the full 10M. If baselines screen the unfiltered library while SubDyve operates on a pre-screened subset, the comparison is not apples-to-apples. This should be clarified, and ideally, a comparison on the same filtered library for all methods should be provided.

- **S₂ serves dual role in training and early stopping:** The held-out set S₂ (approximately 44 compounds: 10% of 30% of 1,468 actives) is used for all three loss terms (BCE, RankNet, Contrastive) during GNN training and also as the criterion for early stopping (Section 3.3.2). While the paper states "early stopping relies solely on intermediate enrichment scores, preventing bias from direct training objectives," this is unconvincing—the GNN parameters are optimized to perform well on S₂, so enrichment on S₂ is not an independent validation signal. The small size of S₂ exacerbates overfitting risk.

- **KIT target shows substantial underperformance:** On the KIT target (Table 1), SubDyve achieves BEDROC 44±3 and EF1% 13.8±2.6, while MoLFormer achieves 66±1 and 36.8±0.9. This represents a 22-point BEDROC gap and 23-point EF1% gap—substantial underperformance that is not discussed. Understanding why the method fails on certain targets would strengthen the paper's analysis of failure modes.

- **LFDR theoretical assumptions not empirically validated:** Algorithm 3 assumes z-scores from GNN logits follow a two-group mixture f(z) = π₀f₀(z) + π₁f₁(z) with Gaussian null density. In severely imbalanced settings (few positives among thousands of candidates), this distributional assumption may not hold. The paper does not provide calibration checks (e.g., Q-Q plots comparing empirical z-score distributions to the fitted mixture model). Without validation, Proposition 1's FDR bound is a theoretical statement whose practical applicability is uncertain.

- **Runtime comparison to AutoDock Vina is not meaningful:** The paper claims SubDyve is "12.3× faster than AutoDock Vina" (1,088s vs. 13,343s). However, AutoDock Vina performs structure-based docking with pose generation and scoring, while SubDyve operates purely on ligand-side features. These are fundamentally different computational tasks—the comparison is misleading and should be removed or reframed.

- **Inconsistent statistical significance reporting:** Table 2 shows statistical significance markers (**) for BEDROC and EF1% but "—" for EF0.5%, EF5%, and EF10% with no explanation. If improvements at certain enrichment thresholds are not statistically significant, this should be stated explicitly rather than left ambiguous.

## Nice-to-Haves
- **Analysis of why subgraph fingerprints degrade performance in isolation:** Understanding whether mined patterns overfit to seeds or create problematic graph topologies would illuminate when and how the method works.
- **Performance on non-kinase targets:** All DUD-E targets evaluated are kinases or kinase-related (EGFR, KIT, SRC, PLK1) or well-studied targets (CDK7). Evaluation on structurally diverse protein families (GPCRs, ion channels, nuclear receptors) would demonstrate broader applicability.
- **Cross-check for compound overlap between PubChem seeds and DUD-E test sets:** Given that seeds are curated from PubChem homologs, reporting maximum Tanimoto similarity between seed molecules and test actives would rule out data leakage from shared compound identities across databases.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Feature redundancy claim:** While GNN input features (propagation weight, NP score, PCA similarity, hybrid rank) capture correlated information, this is common in ensemble designs. The ablation in Appendix F.8 shows each component contributes.
- **Proposition 1 novelty critique:** While the FDR bound is from Efron (2005), its integration into iterative seed refinement for network propagation is the methodological contribution. The proposition is correctly cited.
- **Max pooling choice:** Element-wise max pooling for ensemble aggregation is a standard technique; the criticism is minor.
- **Claim that Proposition 1 is presented as novel:** The paper cites Efron (2005) for the FDR bound. The contribution is applying it to seed refinement, not claiming novelty for the theorem itself.

## Novel Insights
The paper's most interesting finding is that LFDR-guided refinement and subgraph fingerprints are complementary but neither works well alone. The ablation shows LFDR without subgraph features drops BEDROC from 83.44 to 78.68, while subgraph fingerprints without LFDR drop it from 79.04 (baseline) to 63.78. This suggests the subgraph network creates a different graph topology that, when combined with uncertainty-aware seed expansion, enables better propagation—but the topology alone is suboptimal. Understanding this interaction mechanistically (does LFDR correct topological bias created by subgraph graphs? do subgraph features provide better LFDR calibration?) would deepen the contribution.

## Suggestions
- Revise the "zero-shot" terminology throughout the paper to "few-shot transfer" or "homolog-based transfer" to accurately reflect the experimental setup.
- Clarify whether baseline methods in the PU experiment screened the full 10M library or the filtered 30k subset. If baselines operated on different candidate pools, provide results on the same filtered set for fair comparison.
- Add analysis explaining why the KIT target underperforms—is it a limitation of subgraph patterns for certain protein families, or a failure of the homolog-based seed curation?
- Validate LFDR calibration empirically: show that z-scores from the GNN approximately follow the assumed mixture distribution, or discuss when the assumption may fail.
- Remove or reframe the AutoDock Vina runtime comparison, as it conflates structure-based docking with ligand-based screening.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
