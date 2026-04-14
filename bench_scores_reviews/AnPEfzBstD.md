## Summary

This paper presents a large-scale empirical benchmark—over 2,100 runs across four datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS)—comparing 1D, 2D, and 3D molecular representations as feature extractors within Bayesian optimization (BO) loops for materials discovery. The benchmark combines MPNN (2D), Equiformer v2 (3D), and MolFormer (1D LLM) feature extractors with GP and linearized Laplace approximation (LLA) surrogates, evaluating performance across data regimes and transfer-learning settings. The central finding is that 1D/2D representations consistently match or outperform 3D representations in BO, with the pretrained 1D LLM (MolFormer) being the strongest performer on most datasets.

---

## Strengths

- **Scale and statistical robustness of the benchmark:** 35 configurations per dataset, over 2,100 total runs averaged across 15 seeds with reported standard errors, covering four chemically distinct datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS). This scale is uncommon in representation-learning benchmarks and provides a meaningful empirical foundation.

- **Sample-complexity characterization of equivariant models:** Section 5.2 provides direct empirical validation that 3D equivariant models require substantially more training data to match 2D baselines, aligning with and extending the theoretical result of Elesedy & Zaidi (2021) into the sequential decision-making setting. This is a specific, actionable finding for practitioners.

- **Inclusion of both GP and LLA surrogates:** Testing two qualitatively different approaches to uncertainty quantification (kernel-based GP on frozen embeddings vs. Laplace-linearized BNNs) makes the comparison more robust and reveals that the dimension-ordering result is consistent across surrogate families.

- **Anonymous code release:** A reproducible codebase is provided, adhering to open-science standards.

---

## Weaknesses

### Fatal

None that fully invalidate the paper, but the Major weaknesses below collectively undermine the central interpretive claim.

---

### Major

- **Uncontrolled pretraining confound—the paper's central framing is misleading.** MolFormer is a large transformer pretrained on millions of molecules from large-scale chemical databases, whereas MPNN and Equiformer v2 are constrained to ~1.5M parameters and appear to be trained from scratch or only on QM9 (Section 4: "the GNN feature extractors are constrained to similar sizes, with each containing approximately 1.5 million parameters"). The headline conclusion—"1D beats 2D/3D"—is therefore primarily a statement about *large pretrained foundation model beats small task-trained models*, not about representational dimensionality. This confound is never disentangled anywhere in the paper. Without a controlled comparison (e.g., a pretrained 2D/3D GNN at comparable data exposure, or an unpretrained/fine-tuned LLM), the dimensionality framing is unjustified. This is the most significant flaw because it affects the interpretation of essentially every result in the paper.

- **Internal inconsistency between abstract/conclusion and Section 5.1.** The abstract states: "LLM methods consistently outperform," and the conclusion opens with: "Across all datasets examined LLMs consistently outperformed both 2D and 3D models." However, Section 5.1 explicitly states for QM9: "Contrary to all other datasets, LLMs performed worse than 2D and 3D models." This is a factual internal contradiction. QM9 is not a marginal dataset—it is the dataset on which encoders are pretrained. An inconsistency of this kind in the central claim is a serious presentation failure.

- **Task selection is systematically biased against 3D representations, but the title draws a general conclusion.** All target properties—atomization energy (QM7), HOMO-LUMO gap (QM9), and absolute energy (GEOM)—are largely determined by molecular connectivity (2D topology) rather than precise 3D geometry. The paper acknowledges in the conclusion that "future research should focus on… tasks where 3D information might be more important, e.g. protein docking," but still titles the paper "Is 3D A Step Too Far For Optimizing Molecules?" and draws general conclusions. The experimental scope only justifies a conclusion about quantum scalar property optimization on small organic molecules under equilibrium geometry—not molecular optimization broadly. The negative 3D result may simply reflect that the chosen tasks do not require 3D information, not that 3D is unhelpful in principle.

- **Conformer handling for 3D models is entirely unspecified.** GEOM provides multiple conformers per molecule, yet the paper gives no information on whether the lowest-energy conformer, a random conformer, or some ensemble is used in 3D experiments. This is critical: using a suboptimal conformer could entirely account for 3D's poor showing relative to 2D, and readers have no way to assess or reproduce the 3D results without this information.

- **No computational cost measurements despite central efficiency claims.** The abstract and aggregated results discussion both frame the contribution as evaluating the "trade-off between computational cost and predictive accuracy." However, no wall-clock time, FLOPs, memory usage, or conformer-generation time is reported anywhere. The claim that "computational overhead of 3D models often outweighed their predictive performance" (Section 5.1) is stated as a fact but has no empirical support in the paper.

---

### Minor

- **Acquisition function and BO implementation details absent.** Section 2.1 introduces BO abstractly but does not specify which acquisition function is used in experiments, whether evaluation is sequential or batched, or whether there is observation noise. For a benchmarking paper at ICLR, these are reproducibility requirements, not minor details.

- **Transfer-learning section is too narrow to support its conclusions.** Figure 5 covers only QM7 and QM9 (two out of four datasets), and the LLM is excluded from the direct comparison (only transfer learning perspective). The conclusion that "foundation models prove a good tool to leverage in molecular optimization" is drawn from a comparison between 2D and 3D models alone on two datasets, making it speculative.

- **Fig. 2 aggregation methodology is unexplained and potentially misleading.** The paper aggregates all 1D models (including the large pretrained MolFormer) into a single curve. Since MolFormer dominates most individual results, this curve primarily reflects MolFormer's performance rather than the average 1D behavior. The weighting across datasets of different sizes and property types is not described. The dramatic gap between 1D and 2D/3D in Fig. 2 should be decomposed by model type.

- **No statistical significance testing.** Results are reported with standard error over 15 seeds, but no formal tests (e.g., bootstrap, Wilcoxon) are performed. Given overlapping error bars visible in some figures, it is unclear whether reported differences between 2D and 3D are significant.

---

### Tiny

- The LLA covariance notation in Section 2.2 writes $\mathcal{N}(\theta_*, \Sigma_*^{-1})$ where $\Sigma_*^{-1}$ is described as "determined by the inverse Hessian," conflating the Hessian (precision) with its inverse (covariance). This is a minor notational inconsistency with no impact on the experimental results.

---

## Nice-to-Haves

- **Control for pretraining data scale:** Even a brief analysis using a pretrained 2D GNN (e.g., a graph foundation model) or a version of MolFormer fine-tuned from scratch on QM9-scale data would help isolate dimensionality from pretraining effects. This would substantially strengthen the core claim.

- **At least one 3D-critical task:** Including even one conformer-sensitive property (e.g., internal energy at 298K, stereoselectivity, or a conformer-ranking task from GEOM) would make the scope of the negative result much more defensible and informative.

- **Uncertainty calibration analysis:** Calibration plots (reliability diagrams, or rank correlation between predicted uncertainty and error) per representation-surrogate combination would clarify whether 3D's BO underperformance stems from worse point predictions or from poorly calibrated uncertainty—two root causes with very different remedies.

- **Per-dataset decomposition of aggregated Figure 2:** Showing the 1D/2D/3D curves decomposed by model class (e.g., separating MolFormer from fingerprint GP in the 1D category) would make the aggregated result more interpretable and honest.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's complaint about Section 2.3 being encyclopedic rather than benchmark-specific:** This is a style/writing nitpick. The background section is appropriate for a paper with broad audience (chemists and ML practitioners).

- **Critic's complaint about LLA not being justified over ensembles or MC dropout:** The paper uses two standard surrogate families (GP and LLA); this is a reasonable and defensible scope. Demanding coverage of all BNN variants is scope creep.

- **Critic's complaint about the Laplace approximation notation being "inconsistent":** The $\Sigma_*^{-1}$ notation is sloppy but standard shorthand in the Laplace literature. It is a minor notation issue, not a scientific error.

- **Critic's complaint about "why do nobody use them" phrasing being imprecise:** This is a stylistic/rhetorical point about the figure caption, not a scientific issue.

- **Critic's complaint that BO barely beats random search on DRUGS as "potentially troubling":** This is actually reported transparently in the paper ("GP regression and random search performed similarly"), and the paper acknowledges this rather than hiding it. It is not a fabricated result; it is an honest finding about a hard dataset.

- **Demand for theoretical proofs or user studies:** Not standard for an empirical systems/benchmarking paper.

- **Demand for confidence intervals on large-scale benchmarks where single-run evaluation is norm:** Already above norm with 15 seeds; formal statistical testing is nice-to-have, not required.

---

## Novel Insights

The most genuinely novel empirical finding in this paper—beyond recapitulating known supervised-learning results—is the direct measurement of data hunger in equivariant models within the BO loop itself: 3D equivariant models require training sets exceeding ~10,000 molecules to approach 2D MPNN performance, even in the sequential optimization setting where labeled data accumulates gradually. This operationalizes the theoretical prediction of Elesedy & Zaidi (2021) in a realistic discovery context and provides concrete guidance for practitioners: when operating in typical drug/materials discovery library sizes (thousands, not hundreds of thousands), investing in 3D featurization is unlikely to pay off relative to a well-pretrained 2D or 1D model, at least for the class of scalar quantum-chemical properties studied here. However, this insight is currently entangled with the pretraining confound, and its full value cannot be realized until that confound is controlled.

---

## Suggestions

1. **Deconfound pretraining from dimensionality (essential).** Either (a) use a pretrained 2D GNN (e.g., a graph foundation model trained on comparable data to MolFormer) as the 2D representative, or (b) train MolFormer from scratch on QM9-scale data to match the 2D/3D training conditions, or (c) report separate curves for pretrained-MolFormer vs. from-scratch GNNs. Without this, the paper's central framing is not scientifically defensible.

2. **Document conformer handling explicitly.** State in Section 4 which conformer(s) from GEOM are used (lowest-energy, random, ensemble-averaged) and, if possible, include an ablation over conformer choice. This is critical for reproducibility and for interpreting the 3D negative result.

3. **Quantify computational costs.** Add a simple table reporting wall-clock time per BO iteration (conformer generation + forward pass + surrogate update) for each feature extractor. This directly substantiates the paper's core cost-accuracy trade-off claim.

4. **Fix the abstract/conclusion inconsistency on QM9.** Change "LLM methods consistently outperform" to an accurate summary that acknowledges the QM9 exception.

5. **Separate MolFormer from the "1D" aggregate in Figure 2.** Show curves for (a) MolFormer, (b) fingerprint GP/LLA, and (c) SMILES GP separately so readers can see what drives the 1D aggregate.

6. **Scope the title and conclusion appropriately.** The claim "3D is a step too far for optimizing molecules" is only supported for scalar quantum-property optimization on small organics under equilibrium geometry. A scoped title (e.g., "…for quantum property optimization in closed-library BO") would be more accurate and still interesting.

---

**Overall assessment:** The paper addresses a practically relevant question with commendable experimental scale. However, it is currently **moderately weak** in its present form. The pretraining confound is not a subtle concern—it affects the interpretation of the headline result throughout. The internal inconsistency in the central LLM claim, the unspecified conformer handling, and the absence of any cost measurements are additional gaps. The paper would need to resolve these issues—especially the pretraining confound—before its conclusions could be trusted at the level required for an ICLR publication.