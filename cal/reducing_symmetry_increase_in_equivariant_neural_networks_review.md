=== CALIBRATION EXAMPLE 40 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Reducing Symmetry Increase in Equivariant Neural Networks" is mildly misleading: the paper's central contribution is a *theoretical characterization* and *predictive framework* for symmetry increase, not a new architecture that reduces it. The reduction happens through feature design guidelines derived from the theory, but no new ENN is proposed. A title like "Characterizing and Controlling Symmetry Increase..." would be more accurate.

The abstract promises three things: (i) a computable infimum, (ii) algorithms and guidelines for feature design, and (iii) a genericity result ("for *most* equivariant maps"). All three are delivered, but the phrase "for *most* equivariant maps" is informal ML language for a technical notion of genericity (Baire category / measure-zero complements in function space) and risks misinterpretation by a general ML audience. This should be clarified on first use.

---

### Introduction & Motivation

The framing is excellent. The three degeneration types (full, axial, half) give the paper a clear organizing structure, and the positioning relative to Joshi et al. (2023), Cen et al. (2024), Smidt et al. (2021), and Kaba & Ravanbakhsh (2023) is honest and well-calibrated.

One concern: the paper claims throughout that the framework applies to "general cases" beyond *k*-fold structures, yet § 4 (Computation of Orbit Types) is essentially specialized to G = SO(3) or O(3). The completeness results (all closed subgroups are covered in Tables E.3/E.4) hold specifically for these groups. For other physically relevant groups (SE(3), E(n), crystallographic space groups, discrete point groups beyond the O(3) classification), the computational infrastructure is not provided. The scope of generality should be stated more precisely in the introduction.

---

### Preliminaries (§ 2)

The setup is standard and well-executed. Curie's principle (Thm. 2.3) is correctly attributed and the distinction between intentional (kernel-induced) vs. unintended symmetry increase is an important conceptual contribution of this paper. Example 2.2 (the *k*-fold structure) is a good running example.

---

### Theoretical Core: Infimum of Symmetry (§ 3)

**Theorem 3.1 (Uniqueness of Minimal Type)** is the cornerstone. The proof in Appendix B relies heavily on Azzi et al. (2023): Lemmas B.1 and B.2 are directly imported, and Proposition B.4 is described as following "a proof strategy similar to Prop. 2.10 of Azzi et al." The genuinely new technical content is the extension of the complexification argument to the real fixed-point subspace V^[H] and the application to the ML feature-design context. The authors should be more explicit about precisely which steps are novel versus which are direct applications of Azzi et al. The current presentation makes it difficult to assess the incremental theoretical novelty of Thm. 3.1 alone.

**Theorem 3.3** (necessary condition with nontrivial kernel) is clean. The p_Y operator is a nice formalization of the unavoidable kernel-induced increase. The remark that the condition in Thm. 3.2/3.3 is necessary but *not* sufficient (as shown by Counterexample D.3 in the appendix) is important but is only mentioned briefly in §3 ("In §5.1, we will see that this condition is in fact not sufficient"). The logical flow from necessary condition → counterexample → genericity result (Thm. 5.2) is spread across several sections and could be summarized in a diagram or table for clarity.

---

### Computation of Orbit Types (§ 4)

**Proposition 4.2** (Michel's criterion becomes sufficient for high-multiplicity representations) has a concise and correct proof. The high-multiplicity condition m(V, Vi) > dim G is mild for most practical architectures (dim G = 3 for SO(3), so r > 3 channels suffices), which is encouraging.

However, the paper does not explicitly address the intermediate regime r = 2 or r = 3. It states "these predictions are identical for the single representation case (r = 1)," but the argument rests on Linehan & Stedman (2001) for r = 1 and on the high-multiplicity theory for r > 3. The gap (r ∈ {2, 3}) is left unaddressed.

**Table 1** (symmetry infimum for D_{kh} ⊂ O(3)) is the most practically useful single result in the paper. The case analysis across six combinations of parity of k and l_0 is thorough and correct based on the detailed calculation in §C.3.

**The design guidelines in §4.2** are qualitative and somewhat vague for the "general task" case: "avoid components where the symmetry infimum indicates a severe compression of the fixed-point subspace" is not quantitatively actionable. How does a practitioner choose l_0 when balancing expressivity against the risk of degeneration? A concrete decision procedure or flowchart would strengthen this section substantially.

---

### Genericity of Isovariant Maps (§ 5)

**Theorem 5.1** (C^∞-density of TFN in smooth equivariant maps) is a genuine strengthening of Dym & Maron (2021), which proved only C^0-density. The extension to C^∞ topology requires showing that the MLP-parameterized radial functions can approximate derivatives, which follows from classical approximation results (Pinkus, 1999). This is a meaningful technical contribution.

**Theorem 5.2** (almost isovariant maps are generic) has several points deserving scrutiny:

1. **Notion of genericity.** The theorem states the existence of an approximating map g ∈ F that satisfies the approximation bound (Eq. 11) *and* is almost isovariant. This is an existence result, not a statement that randomly initialized TFN parameters will be almost isovariant. In practice, gradient-based training may easily land in degenerate configurations. The practical implication is much weaker than the informal claim "for most equivariant maps."

2. **Scope of applicability.** The theorem requires C^∞ approximation capability (Thm. 5.1), which is established only for TFN with smooth activations. Popular architectures like EGNN, NequIP, MACE, or scalarization-based networks (HEGNN is used in the experiments) are not covered. The paper applies HEGNN in §6.3 without verifying that Thm. 5.2 applies to it.

3. **Multiplicity requirement for exact isovariance.** The feature space Y must contain Ỹ^⊕r for r > max_j dim M_j. For molecular systems with N atoms, the data manifold has dimension ~ 3N-6, so this requirement scales with system size. This makes the exact-isovariance guarantee practically unachievable for large molecules, a limitation the paper does not acknowledge.

4. **Proof strategy.** The proof relies on equivariant transversality theorems (§D.3/D.4), citing Bierstone (1977) and Golubitsky & Guillemin (1973). These are deep results. The paper would benefit from a clearer statement of which equivariant transversality theorem is being invoked and what conditions on the strata of the orbit type decomposition are required.

---

### Experiments (§ 6)

**§ 6.1 (Visualization):** Using a randomly initialized single-layer TFN provides a clean, controlled demonstration. Figures 3 and 4 visually confirm full, axial, and half degeneration in the predicted directions. This is a convincing qualitative validation.

**§ 6.2 (Symmetric Graph Discrimination):** This is the cleanest experiment in the paper. The binary gap (>10^-3 vs. <10^-6) is striking and the result is independent of model depth and channel count, exactly as predicted. The heatmap in Fig. 5 directly confirms Table 1's predictions. This is strong empirical validation.

**§ 6.3 (QM9 Polarizability Prediction):** This experiment is less compelling:

- The setup pretrain on all degrees l ≤ 11, then fine-tune with only the degree l = l_0 features. This is an *indirect* validation: it shows that certain degrees perform poorly for highly symmetric molecules, not that following the design guidelines during training leads to better models.

- Fig. 6 shows boxplots of MAE per degree per symmetry group. The statistical evidence is weak for rare symmetry groups (e.g., I_h with 1 sample, D_6h with 1 sample). Predictions for n=1 molecules are meaningless statistically.

- The design guideline is that one should "avoid feature components with full degeneration." But the experiment doesn't compare a model *designed following the guidelines* against one that does not. The experiment is descriptive (here is what goes wrong) rather than prescriptive (here is how the guidelines fix it). No improvement in prediction error is demonstrated by applying the guidelines.

- The choice of isotropic polarizability (α) is reasonable as a scalar property, but the paper does not explain how scalar property prediction relates to the equivariant features' loss of orientational information. For a scalar output, losing orientation in the intermediate representation should in principle be recoverable if enough invariant information is retained. The connection to the theoretical framework needs to be made more explicit.

- No comparison to baselines (e.g., training models with guidelines-compliant vs. guidelines-violating feature sets) is provided. Without this, it is hard to assess the *practical value* of the guidelines.

---

### Limitations & Broader Impact

There is no explicit limitations section. Key unacknowledged limitations include:

1. The paper restricts detailed computation to G = SO(3) or O(3). Extensions to other physically important groups (crystallographic space groups, E(3), SE(3) with translations, discrete point groups beyond O(3)) are neither discussed nor claimed. This significantly limits the immediate applicability outside molecular point cloud settings.

2. The manifold hypothesis (compact smooth G-invariant submanifolds) is used to justify the genericity results. For discrete graph-structured molecular data with varying atom counts, this is a substantial idealization.

3. The high-multiplicity condition and the C^∞ approximation property are architectural constraints that apply cleanly to TFN but not to the other architectures used in experiments (HEGNN).

4. No direct demonstration that *following the guidelines produces better models* is provided. The experimental evidence is explanatory, not design-oriented.

---

### Overall Assessment

This paper makes a genuine and mathematically substantial contribution by providing a unified, computable framework for predicting symmetry increase in equivariant neural networks. The symmetry infimum concept, the Michel-criterion-based algorithms, and the comprehensive O(3)/SO(3) tables (§E.3, E.4) are practically useful advances over the prior literature. The genericity results (Theorems 5.1 and 5.2) are technically interesting, though their practical implications are subtler than the presentation suggests. The main weaknesses are: (1) the extent of theoretical novelty relative to Azzi et al. (2023) is unclear and should be made explicit; (2) Theorem 5.2's scope is limited to TFN-type architectures while the experiments use HEGNN; (3) the QM9 experiment is descriptive and does not demonstrate that the design guidelines produce measurably better models; and (4) the paper's generality claims are overreached—the computation is specific to O(3)/SO(3). These are meaningful limitations, but the core theoretical contribution stands and would be a valuable addition to the ICLR community studying equivariant architectures. The paper is borderline accept; addressing the gap between the genericity claims and their practical scope, and adding a concrete demonstration that the guidelines improve model design, would significantly strengthen the case.

# Neutral Reviewer
## Balanced Review

### Summary
This paper rigorously characterizes the phenomenon of "symmetry increase" in equivariant neural networks (ENNs), where processing symmetric inputs causes outputs to gain unintended symmetries, leading to representational collapse. The authors prove the existence of a unique "symmetry infimum" dictated by the feature space, develop computable algorithms to derive it via orbit type analysis, and establish that under generic conditions, expressive ENNs achieve this bound. Theoretical insights are translated into practical feature design guidelines and validated through synthetic visualization experiments, symmetric graph discrimination tasks, and molecular property prediction on QM9.

### Strengths
1. **Rigorous Theoretical Foundation:** The paper provides a precise mathematical characterization of symmetry increase using group representation theory, introducing the "symmetry infimum" and proving its uniqueness (Thm. 3.1) along with necessary conditions for isovariant maps (Thm. 3.2, 3.3). This formally addresses gaps left by prior empirical observations (Joshi et al., 2023) and partial collapse-to-zero theories (Cen et al., 2024).
2. **Actionable Computational Framework:** The authors develop concrete algorithms (Algo. 1 & 2) to compute orbit types and symmetry infimums for high-multiplicity representations, supplemented by comprehensive lookup tables for all SO(3) and O(3) closed subgroups (§E.3, E.4). This transforms abstract topology into a practical diagnostic toolkit for architecture design.
3. **Strong Empirical Alignment with Theory:** Experiments across three settings consistently validate theoretical predictions. The QM9 feature-masking study (§6.3, §F.3.2) directly demonstrates how full degeneration in specific degrees (e.g., $l=1$ for $T_d$, $l=2$ for certain groups) degrades predictive performance, providing clear, evidence-based support for the proposed design guidelines.

### Weaknesses
1. **Limited End-to-End Validation:** While the QM9 experiment uses feature masking to isolate theoretical effects, it does not demonstrate training a novel ENN from scratch with the recommended feature degrees against strong architectural baselines. The impact on downstream task performance, optimization dynamics, or data efficiency in a standard training pipeline remains untested.
2. **Heavy Reliance on Asymptotic Assumptions:** Key theoretical guarantees (e.g., Prop. 4.2, density theorems in §5) assume high-multiplicity representations or $C^\infty$-dense parameterizations. Practical ENNs often operate with limited channels, finite depth, or piecewise-linear activations. The paper does not quantify how quickly real-world models converge to the theoretical infimum as width/depth scales.
3. **Accessibility and Notation Density:** The manuscript heavily leverages advanced differential topology and stratification theory (e.g., Whitney conditions, jet maps, Hausdorff measure in §D). While mathematically sound, this may limit accessibility for the broader ICLR audience. Key concepts like "almost isovariant relative to $Y$" could benefit from clearer geometric intuition or simplified examples before formal proofs.

### Novelty & Significance
**Novelty** is high: The paper moves beyond heuristic observations of symmetry collapse by establishing a principled, group-theoretic framework with computable bounds and genericity results for equivariant maps. **Clarity** is structurally strong, with a logical flow from theory to algorithms to experiments, though the dense topological machinery may challenge non-specialists. **Reproducibility** meets ICLR's high bar: all theoretical proofs are self-contained in the appendices, experimental configurations/hyperparameters are explicitly detailed (§F), and implementation code is publicly linked with clear instructions. **Significance** is substantial, as the work provides the first actionable, theory-backed guidelines for preventing representational collapse in geometric deep learning, with direct implications for molecular modeling, materials science, and 3D perception where ENNs are increasingly deployed.

### Suggestions for Improvement
1. **Add an End-to-End Architectural Benchmark:** Include an experiment where an ENN is trained from scratch on QM9 or a comparable geometric dataset using explicitly recommended feature degrees (guided by your infimum tables) versus standard defaults. Report final predictive metrics, convergence speed, and parameter efficiency to demonstrate practical utility beyond diagnostic masking.
2. **Bridge Theory and Practical Capacity:** Provide empirical analysis or ablation studies showing how symmetry increase behavior scales with channel multiplicity, network depth, and activation smoothness. Quantify how far typical finite-width ENNs operate from the theoretical infimum, making the high-multiplicity/$C^\infty$ assumptions more tangible for practitioners.
3. **Improve Pedagogical Framing:** Expand the introduction or add a dedicated subsection with a high-level flowchart/pseudocode showing how a practitioner should use the symmetry infimum tables to select $l$-degrees for a target symmetry group. Consider moving the most technical topological derivations (§D) entirely to the appendix to keep the main text focused on group-theoretic consequences and ML implications.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against existing symmetry-breaking methods** (e.g., Frame Averaging, symmetry-perturbed training). Without this, the claim that this framework provides practical solutions over prior work is unsupported.
2. **Test multiple molecular properties on QM9**, not just isotropic polarizability. A single property cannot validate that the guidelines generalize across different task types and symmetry sensitivities.
3. **Quantify the "most equivariant maps" claim empirically**. Measure what fraction of random initializations actually achieve the predicted symmetry infimum—this is central to §5's genericity argument but remains untested.
4. **Ablation: follow vs. ignore the feature design guidelines** on downstream task performance. Without showing performance degradation when guidelines are violated, the practical价值 of the framework is unclear.

### Deeper Analysis Needed (top 3-5 only)
1. **Validate the manifold hypothesis assumption for QM9 molecules**. Section 5's theoretical guarantees depend on data lying on smooth compact submanifolds, but real molecular data may violate this—this undermines the applicability of Theorem 5.2.
2. **Quantify how much performance loss is attributable to symmetry increase vs. other factors**. The QM9 results show correlation but don't isolate symmetry increase as the causal factor behind MAE differences.
3. **Analyze computational complexity of the orbit type algorithm** for practical feature spaces. Without this, practitioners cannot assess whether the guidelines are computationally feasible for real-world architecture design.
4. **Address the gap between high-multiplicity theory (r > 3) and single-channel实践中**. Most experiments use r=1, but key propositions require high multiplicity—this discrepancy needs reconciliation.

### Visualizations & Case Studies
1. **Show trained network embeddings, not just randomly initialized ones**. Figure 3-4 use random weights, which doesn't demonstrate whether learned representations actually preserve or collapse symmetry after training.
2. **Include failure cases where guidelines don't prevent symmetry increase**. Showing only successful cases creates selection bias and doesn't reveal the method's limitations.
3. **Expand QM9 case studies beyond 3 point groups**. With 22 symmetry groups in QM9, analyzing only C2h, C3h, and Td is insufficient to claim comprehensive validation.

### Obvious Next Steps
1. **Provide a practical tool/library implementing the symmetry infimum calculation**. Theoretical algorithms without implementation code for feature design limits real-world adoption and reproducibility of the guidelines.
2. **Demonstrate end-to-end architecture design using the guidelines**. Show a concrete example where following the framework leads to a measurably better model than standard ENN design choices.
3. **Quantify the measure-zero set where almost-isovariance fails**. Theorem 5.2 claims failures occur on measure-zero subsets, but no empirical estimate of this "negligible" portion is provided for realistic data distributions.

# Final Consolidated Review
## Summary

This paper provides a rigorous mathematical characterization of "symmetry increase" in equivariant neural networks (ENNs)—the phenomenon where processing symmetric inputs causes outputs to gain unintended symmetries, leading to representational collapse. The authors prove the existence of a unique "symmetry infimum" determined by the feature space structure, develop computable algorithms to derive it, establish genericity results showing that expressive ENNs achieve this bound under standard assumptions, and provide comprehensive lookup tables for all SO(3) and O(3) closed subgroups. The work is validated through visualization experiments, symmetric graph discrimination tasks, and QM9 molecular property prediction.

## Strengths

- **Rigorous theoretical foundation with precise mathematical statements:** The paper introduces the symmetry infimum concept (Thm. 3.1), proves its uniqueness via orbit-type stratification theory, and provides necessary conditions for isovariant maps (Thm. 3.2, 3.3). The proofs build on established results (Azzi et al., 2023; Bierstone, 1977) while extending them to the ML feature-design context. The counterexample (D.3) showing the necessary condition is not sufficient adds intellectual honesty.

- **Concrete computational framework:** Algorithms 1 and 2 transform abstract representation theory into practical diagnostics, and Tables 9–20 (§E.3, E.4) provide comprehensive symmetry infimum values for all closed subgroups of SO(3) and O(3). The six-case analysis in Table 1 for D_kh subgroups is particularly useful for practitioners.

- **Strong empirical validation in controlled settings:** §6.2 demonstrates precise agreement between theory and experiment across 24 configurations (TFN and HEGNN with varying channels and layers), with a >10^3 separation between distinguishable (>10^-3) and indistinguishable (<10^-6) embeddings. The result is architecture-independent, exactly as predicted.

- **Technical novelty in approximation theory:** Theorem 5.1 establishes C^∞-density of TFN in smooth equivariant maps, strengthening prior C^0-density results (Dym & Maron, 2021). This is a meaningful contribution beyond the paper's primary focus.

## Weaknesses

- **Gap between theoretical assumptions and experimental validation:** Proposition 4.2 requires high multiplicity (r > dim G, meaning r > 3 for SO(3)), but the visualization and discrimination experiments use r = 1. Theorem 5.2's genericity guarantees require C^∞-approximation capability (Thm. 5.1), established for TFN with smooth activations, yet §6.3 uses HEGNN without verifying this property applies. The paper acknowledges that single-representation predictions match high-multiplicity predictions but does not explain *why* this coincidence holds or what happens in the r ∈ {2, 3} regime.

- **Descriptive rather than prescriptive QM9 experiment:** The QM9 study (§6.3) shows correlation between full degeneration in certain degrees and elevated MAE for specific symmetry groups. However, it does not train a model *designed following the guidelines* against one that violates them to demonstrate improved downstream performance. The experiment validates the theoretical predictions but does not establish practical utility of the design guidelines for real model construction.

- **Limited scope despite generality claims:** The theoretical framework is general, but the computable algorithms and lookup tables are specific to SO(3) and O(3). The paper does not discuss extensions to crystallographic space groups, SE(3), or other groups relevant to molecular modeling beyond point clouds. Practitioners working with other groups must derive analogous computations.

- **Manifold hypothesis as unverified assumption:** The genericity results (Thm. 5.2) assume data lies on smooth compact G-invariant submanifolds. Real molecular data with varying atom counts and discrete structures may not satisfy this, and the paper provides no empirical verification or discussion of robustness to this assumption.

## Nice-to-Haves

- Comparison of guideline-following vs. guideline-violating architecture designs on downstream task performance and convergence speed

- Empirical analysis of how symmetry behavior scales with channel multiplicity and network depth, quantifying the gap between finite-width networks and theoretical infimum bounds

- Extension of computational framework to other physically relevant groups (SE(3), crystallographic space groups)

## Removed Points

- **Title criticism:** The claim that the title is misleading because the paper provides "characterization" rather than "reduction" is unfounded. The paper explicitly provides feature design guidelines (§4.2) that *enable* reduction of harmful symmetry increase, making the title accurate.

- **"For most equivariant maps" informality claim:** The paper precisely defines this notion using Hausdorff measure (§5.1) and establishes the result via Theorem 5.2. The mathematical content is rigorous; at most, clearer pedagogical framing could help.

- **Insistence on comparing against existing symmetry-breaking methods (Frame Averaging, etc.):** This paper's contribution is theoretical characterization and predictive framework. The comparison belongs in related work (which the paper adequately covers), but demanding such comparison fundamentally misunderstands the paper's scope.

- **Request for multiple QM9 properties:** Isotropic polarizability (α) is a scalar property where loss of orientational information is directly relevant. Testing multiple properties would strengthen but is not required for validation.

- **Demand for empirical quantification of "measure-zero" failure sets:** Theorem 5.2 is a theoretical existence result; asking for empirical measurement of measure-zero sets misunderstands the mathematical nature of the claim.

## Novel Insights

The paper's core insight is that the symmetry infimum is determined entirely by the *orbit type structure* of the feature space, not by the equivariant map itself. This reveals that practitioners can predict and control symmetry increase through *feature space design* rather than architectural engineering—a counterintuitive separation of concerns. The finding that low-degree features (l_0 < k) cause full degeneration for k-fold symmetric inputs, while higher degrees exhibit partial or no degeneration, provides a precise theoretical explanation for empirical observations in prior work (Joshi et al., 2023; Cen et al., 2024) that were previously incompletely understood.

## Suggestions

- Add an ablation study comparing models trained with guideline-recommended feature degrees versus standard defaults, reporting both final performance and convergence dynamics to demonstrate practical utility

- Include explicit discussion of the r ∈ {2, 3} multiplicity regime or clarify why single-representation predictions coincide with high-multiplicity predictions

- Provide a brief discussion of potential extensions to other groups or explicit acknowledgment of scope limitations in the introduction

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
