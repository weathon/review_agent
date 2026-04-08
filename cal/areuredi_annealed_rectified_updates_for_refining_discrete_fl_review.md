=== CALIBRATION EXAMPLE 21 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately captures the method. The abstract makes strong claims of "convergence to the Pareto front" and "distributional invariance" that are partially undermined later—the paper itself concedes (Section 4) that these guarantees "hold only in the limit of an infinitely long Markov chain," and then immediately introduces a monotonicity constraint that breaks the ergodicity requirement altogether. The abstract omits any mention of this practical deviation from the theoretical framework, which is misleading. The claim of "outperforms both evolutionary and diffusion-based baselines" is supported, but the comparison is qualified by important caveats not surfaced in the abstract.

---

### Introduction & Motivation

The motivation is strong and well-grounded: multi-objective optimization in discrete biological sequence spaces is a genuine unmet challenge. The framing around ReDi is sensible—using a rectified flow as a learned prior for MCMC sampling is a natural building block. The contributions are clearly enumerated. However, the claim in Contribution 2 ("We provide theoretical guarantees that AReUReDi preserves distributional invariance and converges to the Pareto front with full coverage") is an overstatement given the practical algorithm deployed in all experiments (see below). The paper would benefit from a more careful statement like "the unconstrained algorithm admits these guarantees."

---

### Method (Section 3 & Appendix A)

**3.1–3.4 Core Algorithm**

The combination of annealed Tchebycheff scalarization, locally balanced proposals, and MH updates is technically sound as presented. The proof in Appendix A that the Barker-proposal MH kernel leaves π_{η,ω} invariant is standard and correct.

**Critical flaw: The monotonicity constraint (Section 4)**

The single most important issue in the paper is the introduction, at the start of Section 4, of a "monotonicity constraint that accepts only token updates that increase the weighted sum of the current objective scores." This is disclosed almost as a footnote before the experiments begin. The constraint converts the algorithm from a reversible MCMC chain (with invariance and ergodicity) into a greedy hill-climber. **This violates ergodicity and destroys the very guarantees that constitute Contributions 2.** Table 6 demonstrates how essential this constraint is in practice (without it, non-fouling/solubility collapse from ~0.87 to ~0.45). In other words, the theoretically sound version of AReUReDi doesn't work well in practice, and the practically effective version lacks the claimed guarantees. This gap is neither reconciled nor adequately discussed.

**Pareto Point Representability (Appendix A, Theorem 3)**

The proof constructs ω_n = (1/s̃_n(x†)) / Σ_k (1/s̃_k(x†)) but explicitly requires s̃_n(x†) > 0 for all n. For boundary Pareto points where some normalized objective is zero, the paper says "perturb objectives by ε > 0 and take the limit"—but this limit argument is not worked out and may fail if the optimal weight vector lies on the boundary of the simplex (where the coverage guarantee also requires interior ω). The full Pareto front may not be covered by the theory.

**Coverage Guarantee (Appendix A, Theorem 4)**

The proof appeals to "continuity" of x† being optimal in a neighborhood U_{x†} of ω†. However, S_ω(x) = min_n ω_n s̃_n(x) is not smooth in ω; the argmax can jump discontinuously at weight boundaries. The neighborhood argument as written is incomplete.

**Conditional TC non-monotonicity (Table 4)**

The paper reports that conditional TC *increases* after the first round of rectification (10.60 → 12.63) before decreasing. The authors attribute this to distributional shift from using model-generated couplings. This directly contradicts the stated guarantee from ReDi that TC "provably decreases" across iterations (Section 2.2, Eq. for TC_{s|t}). The explanation given is plausible but the relationship between "within-coupling" monotonicity and "across-coupling" monotonicity needs to be stated precisely. As it stands the empirical result is in tension with the theoretical framing.

---

### Experiments & Results

**Evaluation circularity: in-silico-only assessment with the same surrogates used for optimization**

Every reported objective score—hemolysis, non-fouling, solubility, half-life, and affinity—is evaluated using exactly the same XGBoost/neural-network surrogates that guide AReUReDi's sampling. No wet-lab validation, no independent held-out surrogate, no cross-validation of surrogate predictions on generated sequences. This is a significant scientific limitation: the method could be exploiting artifacts in the surrogates rather than discovering genuinely better peptides. Crucially, the same circularity applies to all baselines, so the relative comparisons are internally consistent—but the absolute claim that AReUReDi "designs therapeutic peptides" is unsupported by experiment.

**Weak surrogate quality**

The guidance/evaluation models have mediocre predictive power:
- Hemolysis XGBoost: F1 = **0.58** (barely above chance for binary classification)
- Non-fouling: F1 = 0.71
- Solubility: F1 = 0.68
- Binding affinity: Spearman ρ = 0.64
- Half-life R² = 0.60 (fine-tuned on **only 105 data points**)

A half-life model with 105 training examples that purportedly predicts absolute half-life in hours is particularly concerning. The dramatic half-life improvements reported throughout the paper (e.g., 22-fold increase over PepTune in Table 2, half-lives of 40–100 hours in Table 1) almost certainly reflect optimization of surrogate artifacts rather than true biological half-life.

**Missing standard MOO metric: hypervolume indicator**

The paper never reports the hypervolume indicator (HVI), which is the standard metric for comparing Pareto front quality in multi-objective optimization. All comparisons use averaged individual objective scores across 100 samples. This conflates single-point quality with Pareto front coverage and cannot detect whether one method produces better trade-offs than another (e.g., a method with high average hemolysis and low average non-fouling could have the same individual averages as a balanced method but be completely dominated in the hypervolume sense). The choice to use averaged single-objective scores also makes the comparison against NSGA-III/SPEA2/MOPSO unfair, since these methods are specifically designed to maximize hypervolume/coverage rather than improve per-objective averages.

**Unfair comparison with PepTune (Table 2)**

AReUReDi uses PepReDi (trained on PepNN + BioLip2 + PPIRef) as its generative prior. PepTune is paired with DPLM (a general protein diffusion language model trained on much broader data). These are fundamentally different base models with different data and scales. The comparison conflates the quality of the base model with the quality of the guidance algorithm. A fairer comparison would run PepTune's guidance strategy on PepReDi, or AReUReDi's guidance on DPLM.

**Runtime disparity**

AReUReDi requires 55–195 seconds per binder versus 2.5–37 seconds for baselines (Table 2). That is approximately 5–80× slower. The paper partially addresses this with a time-matched comparison (Table 11), but the wall-clock comparison is constructed in AReUReDi's favor (only 3–4 AReUReDi binders vs. 100 PepTune binders). A fairer comparison would report AReUReDi performance on 100 binders and PepTune on 2,000–4,000 binders within the same wall clock time.

**Rectification ablation (Table 9) shows inconsistent benefit**

For the 5AZ8 target, PepDFM achieves higher non-fouling (0.8867) and solubility (0.8743) than rectified PepReDi³ (0.8732, 0.8605). The claimed benefit of rectification is robust only for half-life, which is the metric most likely to reflect surrogate noise (given its 105-point training set). Rectification's contribution to multi-objective Pareto performance is not convincingly demonstrated.

**No statistical significance testing**

Table 2 reports single mean values per method over 100 binders, with no standard deviations or confidence intervals. Many differences that appear large (e.g., non-fouling 0.5715 for NSGA-III vs. 0.8680 for AReUReDi) are plausibly meaningful, but others (e.g., affinity 7.3240 SPEA2 vs. 5.7130 AReUReDi) suggest AReUReDi is not universally better, and no analysis of variance is provided to support the claims.

---

### Related Work (Section 5)

The related work is reasonably comprehensive but omits multi-objective GFlowNets (Jain et al., 2023 is cited but not discussed as a discrete-space baseline). GFlowNets naturally handle discrete spaces and multi-objective settings and would be a highly relevant comparison. The paper also does not engage with Gruver et al. (2023) "Protein design with guided discrete diffusion" beyond a citation—this is directly related work on guided generation for discrete protein sequences.

---

### Limitations & Broader Impact

The limitations section (embedded in the Discussion) is too brief. Critical limitations not mentioned:
1. Dependence on surrogate quality (especially the 105-point half-life model)
2. The fact that the deployed algorithm (with monotonicity constraint) lacks the theoretical guarantees
3. No experimental (wet lab) validation
4. Scalability: as sequence length grows, the 20×L MCMC steps may be insufficient for mixing

The ethics statement is adequate.

---

### Overall Assessment

AReUReDi addresses a genuine and important problem—multi-objective discrete sequence optimization—and the integration of locally balanced proposals with Tchebycheff scalarization and annealed MCMC is a technically coherent approach. However, the paper has a fundamental disconnect between theory and practice: the monotonicity constraint used in all experiments destroys the invariance and ergodicity properties on which every theoretical guarantee rests, and this is treated as a minor implementation detail rather than a central tension. Compounding this, the evaluation is entirely in-silico using the same surrogates that guide optimization (including a half-life model trained on 105 points), the standard hypervolume metric is absent from all comparisons, and the baseline comparisons do not control for the generative prior. The contribution stands in spirit—AReUReDi meaningfully improves average multi-property scores over evolutionary and diffusion baselines on the chosen metrics—but the theoretical framing overclaims what the practical algorithm delivers, and the empirical evidence is insufficient to establish that the improvements correspond to genuinely better biomolecules rather than surrogate exploitation. A substantial revision reconciling the theory–practice gap, replacing or augmenting the circular evaluation, and adding proper MOO metrics would be needed for this work to meet ICLR's standards.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces AReUReDi, a multi-objective optimization framework that extends rectified discrete flows to generate biomolecular sequences optimized across competing therapeutic properties. By integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates, the method provides formal guarantees of Pareto front convergence while operating natively in categorical sequence spaces. Extensive experiments on wild-type peptides and chemically-modified peptide SMILES demonstrate superior trade-off navigation compared to evolutionary and diffusion-based baselines.

### Strengths
1. **Rigorous Theoretical Foundation:** The paper provides clear, step-by-step proofs of distributional invariance, Pareto convergence as $\eta \to \infty$, and full Pareto front coverage (Appendix A). This formal grounding is notably stronger than heuristic guidance strategies common in discrete generative modeling.
2. **Principled Integration of Guidance into Discrete Flows:** The locally balanced proposal mechanism (Section 3.3) elegantly blends the generative prior with multi-objective guidance while preserving reversibility, avoiding the approximation errors typical of gradient straight-through or Gumbel-softmax tricks in discrete spaces.
3. **Comprehensive and Well-Structured Empirical Evaluation:** The method is tested across two distinct modalities (amino acid sequences and SMILES) and up to five conflicting objectives. Tables 1–2 and Appendix Tables 6–14 systematically validate guidance ablation, weight steering, and baseline comparisons, showing consistent superiority over NSGA-III, MOPSO, and PepTune.
4. **Strong Ablation and Design Validation:** The paper thoroughly isolates key components, demonstrating that rectification improves base model quality (Table 9), annealing outperforms fixed guidance strengths (Table 10), and the learned ReDi prior is critical for high-quality trade-offs (Tables 15–16).

### Weaknesses
1. **Contradiction Between Theory and Practical Implementation:** Section 4 explicitly states that a monotonicity constraint (accepting only token updates that increase the weighted objective sum) is used in all experiments to accelerate convergence (Table 6). This constraint fundamentally breaks detailed balance and invalidates the exact Markov chain invariance guarantees claimed in Section 3 and Appendix A. The paper treats this as a minor practical tweak without discussing the theoretical implications.
2. **Surrogate Model Uncertainty is Ignored:** The optimization relies entirely on pre-trained score models, yet several have moderate predictive performance (e.g., XGBoost classifiers with validation F1 scores of 0.58–0.71, half-life predictor $R^2=0.60$, Appendix E.1–E.3). Optimizing aggressively against noisy or poorly calibrated surrogates risks exploiting model artifacts rather than discovering biologically valid sequences, yet no uncertainty quantification or robustness mechanism is incorporated.
3. **Significant Computational Overhead:** AReUReDi is notably slower than baselines (Table 2 reports 55–195s per binder vs. 2.5–8.5s for classical MOO and 2.5–4.8s for PepTune). While Table 11 attempts a matched wall-clock comparison, the per-sample compute cost remains high, limiting practicality for large-scale virtual screening or longer sequence lengths without additional architectural or sampling optimizations.
4. **Incomplete Diversity/Novelty Analysis for Peptides:** While Table 5 reports SNN and diversity for SMILES generation, the primary wild-type peptide results (Tables 1–2) lack quantitative novelty metrics (e.g., pairwise sequence identity, coverage of chemical space, or hypervolume calculations against the true Pareto front). Without these, it is difficult to assess whether the model discovers genuinely novel therapeutic candidates or converges to narrow local optima.

### Novelty & Significance
**Novelty:** High. The integration of rectified discrete flows with MCMC-based multi-objective guidance represents a meaningful conceptual advance over single-objective discrete diffusion or heuristic evolutionary search. The use of locally balanced proposals to maintain reversibility in categorical spaces is particularly elegant and underexplored.
**Clarity:** Good. The method is logically partitioned, mathematical notation is consistent, and Algorithm 1 provides a clear implementation blueprint. The main text flows well, though the tension between theoretical guarantees and the practical monotonicity constraint needs clearer articulation.
**Reproducibility:** Strong. The paper provides detailed model architectures, training hyperparameters, dataset curation steps, score model specifications, and complete ablation settings across multiple appendices. The authors commit to releasing code and checkpoints, and the use of public datasets further facilitates replication.
**Significance:** High. Multi-property biomolecular design is a critical bottleneck in therapeutic development. By providing a theoretically grounded, sequence-native framework that outperforms strong baselines across diverse targets, the work aligns well with ICLR's emphasis on principled generative modeling with real-world scientific impact.

### Suggestions for Improvement
1. **Reconcile Theory with Practice:** Either remove the monotonicity constraint from the main experimental results to fully align with the theoretical claims, or formally analyze its impact on convergence (e.g., treating it as a biased sampler) and clearly state that guarantees are asymptotic/idealized.
2. **Incorporate Surrogate Uncertainty:** Integrate uncertainty-aware guidance (e.g., expected improvement, ensemble variance, or conformal prediction thresholds) to prevent over-optimization of noisy objectives. Report calibration curves or confidence intervals for the surrogate models to contextualize the property scores.
3. **Expand Baseline Comparisons and Efficiency Analysis:** Compare against additional discrete MOO methods (e.g., GFlowNet-based multi-objective sampling or classifier-free guided discrete diffusion). Provide a more granular Pareto frontier plot (e.g., HV-I or hypervolume vs. FLOPs/time) to better characterize the efficiency-accuracy trade-off.
4. **Quantify Peptide Diversity and Novelty:** Add pairwise sequence identity distributions, training-set Tanimoto/Levenshtein similarities, or hypervolume metrics for the peptide generation experiments. This will substantiate claims of broad exploration and demonstrate clinical relevance beyond aggregate property scores.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablate the monotonicity constraint** — Section 4 states this constraint was used in ALL experiments, yet it violates the MCMC invariance guarantees claimed in Appendix A. Without showing results both with and without this constraint, the theoretical claims are disconnected from empirical validation.

2. **Add Pareto front coverage metrics** — The paper claims "full coverage of the Pareto front" but provides no hypervolume, spread, or coverage metrics comparing AReUReDi to baselines. ICLR reviewers will expect quantitative evidence of Pareto front quality, not just average property scores.

3. **Benchmark against more recent discrete MOO methods** — Comparisons stop at PepTune (2025) and classical evolutionary algorithms. Need comparison with other discrete flow guidance methods (e.g., Nisonoff et al. 2025, Tang et al. 2025a) to establish the method is actually novel and not just incremental.

4. **Validate score model reliability** — The half-life model was trained on only 105 entries (Appendix E.3), and classifiers achieve F1 scores of 0.58-0.71. Without uncertainty quantification or external validation of generated sequences, the optimization results may reflect model artifacts rather than real improvements.

5. **Test on held-out protein targets** — All 8 peptide targets appear to be used for both development and evaluation. Need true held-out targets to demonstrate generalization rather than potential overfitting to the benchmark suite.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify gap between theoretical and practical convergence** — Theorems require η→∞ and infinite chain length, but experiments use finite η (max 20) and 128-256 steps. Analysis showing how close practical settings come to theoretical limits is essential for claim credibility.

2. **Analyze why rectification helps for MOO specifically** — Table 9 shows rectification improves results, but no analysis explains why reduced conditional total correlation translates to better Pareto trade-offs. This mechanistic link is claimed but unexamined.

3. **Report computational efficiency trade-offs systematically** — AReUReDi takes 10-80× longer than baselines (Table 2). Need analysis of whether the improvement justifies the cost, and whether there are settings where the method becomes impractical.

4. **Analyze failure cases** — No discussion of targets or objectives where AReUReDi underperforms. ICLR reviewers expect honest assessment of limitations, not just success stories.

5. **Weight vector sensitivity analysis is insufficient** — Table 13-14 shows only 4 weight settings. Need systematic analysis of how weight selection affects Pareto front coverage and whether the method finds diverse solutions or clusters around similar trade-offs.

### Visualizations & Case Studies
1. **Pareto front plots** — Show actual 2D/3D projections of the Pareto front achieved by AReUReDi vs. baselines. Without this, claims about "superior trade-off navigation" are unverified.

2. **Sequence diversity visualization** — Figure 1 shows structures but doesn't reveal whether generated sequences are diverse or collapsing to similar solutions. t-SNE/UMAP of sequence embeddings would expose mode collapse.

3. **Convergence trajectories** — Plot property scores vs. iteration for multiple random seeds to show consistency. Figure 1E shows one target; need to demonstrate this pattern holds across targets.

4. **Invalid sequence rates over time** — For SMILES generation, show how validity changes during sampling. The paper claims 100% validity but doesn't show rejection rates or how constraints affect sampling dynamics.

### Obvious Next Steps
1. **Release the monotonicity constraint from theoretical claims** — Either remove it from experiments to match theory, or revise theorems to account for it. This inconsistency undermines the core contribution.

2. **Add external validation of generated binders** — At minimum, run AlphaFold3 docking on a held-out set and compare to known binders. In silico scores alone are insufficient for ICLR biomolecular work.

3. **Include negative results** — Show at least one target or objective combination where AReUReDi fails or offers no advantage. This builds credibility and helps reviewers assess true contribution boundaries.

4. **Justify the 5-objective claim more carefully** — Table 1 shows 5 objectives but Table 2 comparisons use different objective sets across targets. Standardize the evaluation to support the "up to five" claim rigorously.

# Final Consolidated Review
## Summary

AReUReDi extends rectified discrete flows (ReDi) to multi-objective sequence optimization by integrating annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates. The method provides theoretical guarantees of convergence to the Pareto front with full coverage, and demonstrates strong empirical performance on wild-type peptide and SMILES sequence design tasks optimizing up to five therapeutic properties simultaneously.

## Strengths

- **Principled theoretical framework:** The paper provides rigorous proofs of distributional invariance (Theorem A.1), Pareto convergence (Theorem A.2), and Pareto point representability (Theorem A.3) in Appendix A. This formal grounding is notably stronger than heuristic guidance strategies common in discrete generative modeling.

- **Elegant integration of guidance with discrete flows:** The locally balanced proposal mechanism (Section 3.3) blends the generative prior with multi-objective guidance while preserving reversibility. Using Barker's balancing function yields automatic acceptance, providing computational efficiency while maintaining theoretical soundness.

- **Comprehensive empirical validation:** The method is tested across two distinct modalities (amino acid sequences and SMILES), eight diverse protein targets (including structured targets, targets without known binders, and intrinsically disordered targets), and up to five conflicting objectives. The ablation studies systematically validate key components: guidance contribution (Tables 7-8), rectification benefit (Table 9), annealing schedule (Table 10), weight vector steering (Tables 13-14), and prior importance (Tables 15-16).

- **Strong baseline comparisons:** AReUReDi consistently outperforms classical MOO algorithms (NSGA-III, SMS-EMOA, SPEA2, MOPSO) and the state-of-the-art diffusion-based PepTune across multiple targets. The matched wall-clock comparison (Table 11) addresses computational overhead concerns by showing that even with limited samples, AReUReDi achieves better trade-offs.

## Weaknesses

- **Fundamental tension between theory and practice:** Section 4 discloses that "to improve sampling efficiency in all reported experiments, we introduce a monotonicity constraint that accepts only token updates that increase the weighted sum of the current objective scores." This constraint—which Table 6 shows is essential for practical performance—violates the detailed balance assumption underlying the MCMC invariance guarantees. The paper acknowledges that guarantees "hold only in the limit of an infinitely long Markov chain" but does not adequately reconcile this with the monotonicity heuristic used in practice. The theoretical contribution (Claim 2) is thus only partially delivered by the deployed algorithm.

- **In-silico-only evaluation with circular surrogate use:** All reported property scores (hemolysis, non-fouling, solubility, half-life, affinity) are evaluated using the same pre-trained models that guide AReUReDi's sampling. While Table 1 reports AlphaFold3 ipTM scores and AutoDock VINA docking scores as external validation for binding affinity, no external validation exists for the other four properties. This creates risk of exploiting surrogate artifacts rather than discovering genuinely improved sequences.

- **Half-life surrogate trained on very limited data:** Appendix E.3 reveals the half-life model was trained on only 105 entries (curated from PEPLife, PepTherDia, and THPdb2), with validation R²=0.60. The dramatic half-life improvements reported throughout the paper (e.g., 22-fold improvement over PepTune in Table 2, half-lives of 40–100 hours) may reflect optimization of this sparse, uncertain surrogate rather than true biological half-life extension.

- **Absence of standard Pareto front metrics:** The paper claims "full coverage of the Pareto front" but never reports hypervolume indicator (HVI), spread, or coverage metrics—the standard quantitative measures for comparing MOO algorithms. Average scores across objectives (Tables 1-2) cannot distinguish between a method producing diverse trade-offs versus one converging to a narrow region.

## Nice-to-Haves

- **Uncertainty-aware guidance:** Incorporating ensemble variance or expected improvement acquisition functions would reduce the risk of over-optimizing noisy surrogate predictions, particularly for the low-data half-life model.

- **Pareto front visualization:** 2D/3D projections comparing AReUReDi's Pareto front against baselines would strengthen claims of superior trade-off navigation more intuitively than aggregate property tables.

- **Additional discrete MOO baselines:** Comparison with multi-objective GFlowNets or other recent discrete-space MOO methods would strengthen the methodological positioning.

## Removed Points

These points are flagged to be removed or substantially weakened—treat them with caution:

- **"Hemolysis classifier F1=0.58 is barely above chance":** This mischaracterizes binary classification performance. Without knowing the class balance, F1=0.58 cannot be assessed as "barely above chance." Removed as it misrepresents classifier quality.

- **"Unfair comparison with PepTune due to different base models":** The paper acknowledges PepTune uses DPLM and addresses this with a matched wall-clock comparison (Table 11). The concern is partially addressed and not a critical flaw.

- **"Rectification shows inconsistent benefit in Table 9":** For AMHR2, rectification improves all five metrics; for 5AZ8, it improves half-life substantially while maintaining competitive performance on other metrics. The benefit is demonstrated though not uniform across targets.

- **"Pareto Point Representability proof fails for boundary points":** The paper uses standard perturbation arguments (ε→0) for boundary cases. This is a valid mathematical technique and not a fundamental proof error.

- **"Missing peptide novelty/diversity metrics":** Table 5 reports validity, uniqueness, diversity, and SNN for SMILES generation, demonstrating the paper does consider these metrics for at least one modality.

## Novel Insights

The locally balanced proposal mechanism represents an underexplored technique for discrete generative guidance. By combining the learned flow prior p_t with a reward-weighted acceptance probability, AReUReDi achieves a principled middle ground between pure ancestral sampling (which ignores guidance) and greedy search (which sacrifices diversity). The mathematical observation that Barker's balancing function g(u) = u/(1+u) yields automatic MH acceptance while maintaining reversibility is elegant and potentially applicable to other discrete generative frameworks beyond flow matching.

## Suggestions

1. **Separate theoretical claims from experimental implementation:** Either report results without the monotonicity constraint to validate theoretical claims, or clearly state in the contribution list that the theoretical guarantees apply to an "idealized" version while the practical implementation uses a convergence-accelerating heuristic that sacrifices formal guarantees.

2. **Add hypervolume comparisons:** Report hypervolume indicator values (or at least Pareto front plots in 2D objective space) to substantiate claims about Pareto front quality and coverage.

3. **Validate key properties independently:** For at least one target, validate a subset of generated peptides using an external predictor or structural docking for properties beyond affinity (e.g., run an independent solubility predictor if available).

4. **Report results with and without monotonicity constraint:** Table 6 shows the constraint's importance, but does not report full multi-objective results for the theoretically sound version. Including both would clarify the practical impact of relaxing the theoretical framework.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
