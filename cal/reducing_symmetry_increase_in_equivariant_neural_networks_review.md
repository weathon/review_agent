=== CALIBRATION EXAMPLE 72 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Reducing Symmetry Increase in Equivariant Neural Networks" is accurate, though "reduction" is perhaps slightly misleading — the paper's primary contribution is *characterizing* the infimum of unavoidable symmetry increase and *predicting* it, with practical guidelines following as a consequence. The abstract is well-written and the four main claims are largely substantiated by the paper. One concern: the abstract promises "practical guidelines for feature design to prevent harmful symmetry increases," but these guidelines (§4.2) are relatively high-level. A reader might expect more concrete, prescriptive design recommendations than are ultimately delivered.

---

### Introduction & Motivation (§1)

The motivation is compelling and grounded in real prior work. The three-way taxonomy of degenerations — *full*, *axial*, and *half* — is a genuine advance over the prior "collapse-to-zero" (Cen et al., 2024) framework, which covered only full degeneration. The critique of Kaba & Ravanbakhsh (2023) as "relaxing equivariance rather than solving within the equivariant framework" is fair but somewhat underdeveloped. The paper does not engage with the practical trade-offs: relaxing equivariance may be acceptable in many downstream tasks, and a brief acknowledgment of when the authors' approach is preferable (and when it is not) would strengthen the motivation.

The contributions listed in §1 are clear and numbered, but the relationship to Joshi et al. (2023) remains vague — the paper says that their observations "cannot be fully accounted for by prior theories," but does not explicitly show *which* of Joshi et al.'s results are now explained, nor how their empirical findings map onto the three degeneration types.

---

### Preliminaries (§2)

The background is comprehensive and appropriately self-contained. The notation is clean. Example 2.2, which instantiates k-fold symmetry as a concrete running example, is well-chosen and effectively threads through the paper.

**Concern:** The *projection operator* pY is introduced in §3.2 rather than here, though it is central to the entire framework. Motivating it earlier with an example (the Sₙ kernel in molecular encoding) would help readers anticipate why the straightforward notion of isovariance must be refined.

Theorem 2.3 (Curie's Principle) is correctly attributed and well-contextualized. The transition from "symmetry can increase" to "we characterize how much it must increase" is well-motivated.

---

### Infimum of Symmetry (§3)

**Theorem 3.1 (Uniqueness of Minimal Type):** This is the paper's most foundational result. The proof (Appendix B.1) is technically sound and uses a complexification argument via Azzi et al. (2023), Lemmas B.1–B.2, to show the set of minimal-orbit-type points is dense and open. The key novel step is Prop. B.4, whose proof is self-contained and correct.

**Concern:** The proof leans heavily on Azzi et al. (2023) — Lemmas B.1, B.2, and several corollaries are directly invoked. The novel mathematical content of Prop. B.4, while correct, is essentially an adaptation of Prop. 2.10 of Azzi et al. to the fixed-point subspace setting. For ICLR, this level of novelty in the mathematical core should be made more explicit: the authors should clearly articulate *why the restriction to V^[H] requires a new argument*, beyond just summarizing that their strategy is "similar."

**Theorem 3.2:** The necessary condition for isovariant maps is clean, but the paper itself immediately notes (§5.1) that it is not sufficient. The counterexample (Cex. D.3) demonstrating non-existence despite the necessary condition being met is buried in the appendix. For the reader to appreciate the subtlety, this counterexample deserves at least a brief mention or intuition in the main text alongside the theorem.

**§3.2 (Non-trivial kernels):** The p_Y operator and "isovariant relative to Y" definition are well-motivated. The derivation in Eq. (5) — showing pY(Gx) ⊆ Gf(x) — is concise and correct. This subsection is one of the cleanest parts of the paper.

---

### Computation of Orbit Types (§4)

**High-multiplicity condition (Prop. 4.2):** The proof shows that when m(V, Vᵢ) > dim G, Michel's necessary criterion becomes sufficient. For SO(3), dim G = 3, so this requires more than 3 channels per irrep type — a mild condition for most modern ENN architectures. The paper notes this holds "for all finite groups and for feature spaces with a high number of channels," but does not explicitly verify whether popular architectures (TFN, HEGNN, e3nn) meet this condition.

**Table 1 (Symmetry infimum for k-fold structures):** This is the central computational result, and it is surprisingly tractable. The parity-and-range condition on l₀ relative to k is elegant. However, the paper states: "Although derived assuming high multiplicity (r > 3), these predictions are identical for the single representation case (r = 1), see §C.4." This is an important remark — it means the full-multiplicity theory coincides with the r=1 theory as tabulated by Linehan & Stedman (2001). The paper should explain *why* this coincidence occurs (is it because the abelian subgroups Cₖ are the only ones where the Michel vs. Ihrig-Golubitsky gap matters, and those cases leave the infimum unchanged?), rather than merely pointing to a table.

**Algorithms 1 & 2:** The algorithms are sound in principle, but Algorithm 2's termination and correctness depend on the poset structure of orbit types being well-founded, which is guaranteed for compact Lie groups but not stated. More importantly, the computational complexity is not discussed. How many subgroups does one need to enumerate for SO(3) or O(3)? The paper mentions a "top-down strategy" in Algo. 3 (appendix), which is reassuring, but the main text Algorithm 2 as written would enumerate all supergroups — an exponential search in principle.

**§4.2 (Guidelines):** The two guidelines — one for orientation-dependent tasks, one for general tasks — are stated at a high level. The "orientation-dependent" guideline says to select features where pY(H) ∈ O_G(Y). But practitioners need to know: if I am building a model for, say, molecular conformation prediction, how do I enumerate the relevant input symmetry groups H and check this condition in practice? The paper provides all the mathematical machinery but stops short of an end-to-end worked example at the practitioner level.

---

### Density of (Almost) Isovariant Maps (§5)

**Theorem 5.1 (C∞-density of TFN):** This extends the known C⁰-universality (Dym & Maron, 2021) to C∞-density. The proof sketch in §D.2 invokes standard approximation theory for smooth equivariant functions. This is a useful technical lemma but is stated as a theorem; its role is primarily to enable Thm. 5.2.

**Theorem 5.2 (Generic almost-isovariance):** This is the paper's most practically significant theorem. It states that if the feature space satisfies the orbit-type inclusion condition, then for any expressive ENN parameterization, almost every map in the function space is almost isovariant — and that full isovariance on the data manifold M is achievable with sufficient multiplicity r > max{dim M_j}.

**Concern 1:** The bound r > max{dim M_j} is the key prescription. What are the dimensions of the relevant data manifolds in practice? For molecular point cloud data (k = N atoms in 3D, modulo SO(3) × S_N), what is dim M_j? The paper does not provide this analysis, leaving practitioners without a concrete channel count recommendation.

**Concern 2:** The almost-isovariance result is for *generic* maps in the parameterization family — i.e., for a parameter chosen at random. But neural networks are trained by gradient descent with specific objectives that may not yield generic parameters. The paper does not address whether optimization tends to produce (almost) isovariant maps or whether specific training procedures could systematically select non-isovariant ones.

**Concern 3:** The manifold hypothesis (finite union of smooth compact G-submanifolds) is a strong and idealized assumption. Real molecular data has discrete structure (atoms on a grid, not smooth manifolds) and high-dimensional ambient space. The paper acknowledges this briefly in §A.3 but does not assess how robust the conclusions are to deviations from this assumption.

---

### Experiments (§6)

**§6.1 (Visualization):** The visualization of representation spaces is effective and provides strong intuitive support for the theory. The clear separation between full, axial, and half degeneration in Figs. 3 and 4 matches the predictions from Table 1. Using randomly initialized TFN (rather than trained networks) is appropriate here — it isolates the structural property.

**§6.2 (Expressivity on Symmetric Graphs):** The heatmap experiment is well-designed and shows a clean binary pattern with a > 10³ gap between distinguishable and indistinguishable cases. This strongly validates the theory. However:
- The paper reports "the maximum value was selected for each configuration," combining results across different number of channels (1, 4, 16) and layers (1–4). It would be more informative to show whether single-channel, single-layer models already achieve the theoretical degeneration, or whether multiple channels/layers are needed to reveal the effect.
- The experiment was designed to distinguish G₀ from G₁ where G₁ "does not coincide with G₀." How was G₁ sampled? If it was a random rotation, there may be measure-zero pathological cases. The paper should confirm the sampling was uniform and that the result holds robustly across many random seeds.

**§6.3 (QM9 Molecular Property Prediction):** The experiment is interesting as a theory validation, but has a significant confound: the authors *pretrain* a shared HEGNN encoder on all features (l ≤ 11) and then fine-tune with individual degree features. This means the encoder has already seen all degree information; the degree-specific fine-tuning heads may not cleanly isolate the effect of individual feature degrees. Training from scratch with specific feature configurations (even a subset of molecules) would provide cleaner evidence.

Additionally, the claim that "for non-trivial feature components where molecular symmetry increases to O(3), the prediction loss is substantially higher" requires knowing which molecules cause which symmetry increase — determined by their point group. The paper states the point groups were pre-computed using the PointGroup library, but does not report the quantitative agreement between predicted degenerate degrees and the observed performance drops per molecule group in the main text (only in §F.3.2). The main-text Fig. 6, even without the PDF parsing artifacts, shows boxplots per symmetry group rather than per molecule, making it difficult to confirm the claimed correspondence.

---

### Writing & Clarity

The main text is generally well-written and the progression of ideas is logical. §3.2 and the proof sketches are particularly clear. The most significant clarity issue is in §4 where Algorithms 1 and 2 appear to be typeset in parallel columns in the original PDF, causing the extraction to interleave their lines. Even in the original paper, side-by-side algorithm presentation risks confusion for algorithms of different lengths. The text around the algorithms refers only to "Algo. 1" and "Algo. 2" without adequately narrating their structure.

The paper overuses the phrase "it can be shown that" and related deferments to appendices in several places where at least an intuition would help the reader follow the logic.

---

### Limitations & Broader Impact

The paper includes an Ethics Statement that correctly identifies this as a theoretical work with no direct societal harm. However, the paper does not include a dedicated Limitations section, and the acknowledgment of limitations is dispersed. The most significant limitations that should be addressed:

1. **Scope of computational results:** The orbit-type tables cover SO(3) and O(3) subgroups. Many practical applications involve other symmetry groups (e.g., periodic boundary conditions in crystal structure prediction, where the translational symmetry T³ plays a central role). The paper does not discuss how its framework extends to such cases.
2. **Gap between theory and practice:** Thm. 5.2 guarantees generic almost-isovariance for sufficiently expressive parameterizations, but does not address finite-depth or finite-width limitations. Typical ENNs use l ≤ 5 or l ≤ 6, and the theory's guarantee requires specific multiplicity thresholds that are not validated against actual architectures.
3. **Interaction effects:** The guidelines focus on individual feature components, but practical ENNs use direct sums of many irrep types simultaneously. The composition properties in §C.5 provide some analysis, but the interaction between "good" and "bad" components in a direct sum is not fully characterized in the main text.
4. **No negative results for new architectures:** The paper shows that existing HEGNN and TFN exhibit the predicted degenerations. A natural follow-up would be showing that explicitly designed features following the guidelines *avoid* these degenerations on the same tasks. The current experiments only validate the negative (degeneration occurs as predicted) rather than the positive (avoiding degeneration via guidelines improves performance).

---

### Overall Assessment

This paper makes a genuine theoretical contribution to the study of equivariant neural networks by providing the first rigorous, comprehensive characterization of symmetry increase — the phenomenon whereby symmetric inputs cause ENN outputs to become over-symmetric. The central concept of the *symmetry infimum*, its uniqueness (Thm. 3.1), and the computable algorithm for its determination are solid contributions that subsume and unify prior partial results (collapse-to-zero for full degeneration, Joshi et al.'s empirical observations). The theory is technically sound, and the proofs — while heavily leveraging classical results from equivariant topology (Michel, Ihrig-Golubitsky, Azzi et al.) — correctly extend these to the feature-design context. The experiments convincingly validate the theory on synthetic tasks. The primary weaknesses are: (1) the practical prescriptions remain somewhat abstract — the guidelines are correct but lack concrete, worked-out design examples that practitioners can follow; (2) the QM9 experiment is confounded by pretraining and the per-molecule-group analysis is not sufficiently connected to the theory in the main text; (3) the paper does not include positive experimental evidence that following the proposed guidelines actually improves downstream task performance; and (4) several key assumptions (high-multiplicity, smooth data manifold, sufficiently expressive parameterization) are not checked against real architectures or datasets. Despite these issues, the theoretical framework is of genuine value to the equivariant learning community and represents a clear advance over the state of the art. This paper is at or near the ICLR acceptance threshold, conditional on the authors better connecting theory to practice and addressing the experimental confounds.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the phenomenon of "symmetry increase" in Equivariant Neural Networks (ENNs), where processing symmetric inputs leads to an output representation with unintended higher symmetry, degrading expressivity. The authors provide a rigorous mathematical framework characterizing the lower bound of this increase ("symmetry infimum") based on the algebraic structure of the feature space, accompanied by algorithms for computing it and design guidelines to mitigate harmful degenerations. Theoretical claims are supported by visualizations and experiments on symmetric graph tasks and the QM9 molecular dataset.

### Strengths
1.  **Theoretical Rigor:** The paper provides a deep mathematical grounding using orbit types and isotropy subgroups to characterize symmetry increase, moving beyond empirical observation to necessary and sufficient conditions (e.g., Theorems 3.1, 3.2, and 3.3). This connects geometric deep learning with established results in singularity theory and bifurcation theory.
2.  **Computational Framework:** The contribution of computable algorithms (Algorithms 1 and 2) to derive the symmetry infimum is a significant practical advancement, allowing practitioners to predict expressive limitations before training model architectures.
3.  **Empirical Validation:** The experiments on QM9 (Section 6.3) effectively demonstrate the practical relevance of the theory. The results clearly show that feature components leading to full degeneration (according to the theory) result in higher prediction errors, validating the link between theoretical infimums and predictive performance.
4.  **Reproducibility:** The authors commit to reproducibility by providing open-source code and detailed appendices containing proofs and tables for symmetry infima calculations, which facilitates verification of the theoretical claims.

### Weaknesses
1.  **Applicability to Learned Features:** The guidelines in Section 4.2 largely assume manual design of feature spaces (selecting irreducible representations). It is less clear how these principles apply to end-to-end training where the network learns to mix representations, potentially bypassing the "infimum" constraints through non-linearities that the theory assumes are linear or structured in specific ways.
2.  **Measure-Zero Assumptions:** Theorem 5.2 relies on "almost isovariant" maps where symmetry increase occurs only on a set of measure zero. In finite-data deep learning regimes, the "generic" properties assumed might not manifest unless the model capacity is sufficiently large, and the authors could better quantify the capacity requirements for small datasets.
3.  **Scalability of Analysis:** The orbit type computation (Section 4) relies on checking fixed-point space dimensions and subgroup relations. For general Lie groups or high-dimensional input spaces, the complexity of computing these infimums could become prohibitive, yet the paper does not provide a scalability analysis or heuristic approximations for complex cases.

### Novelty & Significance
The paper offers **high novelty** by formalizing an ENN-specific vulnerability (symmetry increase) using precise group-theoretic language ("symmetry infimum") which was previously treated intuitively or in isolation. Its **significance** is substantial for scientific machine learning (chemistry, physics), where ENNs are standard but can fail catastrophically on symmetric data structures (e.g., high-symmetry molecules). By providing a predictive tool for feature space design, it moves beyond "black box" ENN usage toward principled architectural design.

### Suggestions for Improvement
1.  **Discuss End-to-End Learning:** Add a discussion or small experiment on how standard training dynamics (e.g., SGD) might navigate the "almost isovariant" manifold. Does standard regularization help avoid the identified degenerate regions, or are explicit architectural constraints needed?
2.  **Quantify Capacity Requirements:** Expand the empirical analysis in Section 6 to include a study on model capacity (e.g., number of layers or channels) relative to the data manifold dimension in Theorem 5.2. This would clarify when the "measure zero" guarantee becomes practically useful.
3.  **Clarify Computational Complexity:** Include a brief analysis of the time/space complexity of Algorithm 1 and 2, particularly for general subgroup lattices. If the computation is expensive for arbitrary groups, suggest efficient approximations or restrictions to common groups like O(3).
4.  **Refine the "Almost" Terminology:** Since "measure zero" is a theoretical concept, the experimental verification should explicitly acknowledge that "almost isovariant" in finite datasets means "rarely observable." Clarify if the degenerate cases in experiments (e.g., Fig 5) align with the specific "measure zero" sets or if they represent common train-avoidable pitfalls.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison against existing symmetry-breaking methods** — The paper claims to provide "practical guidelines" but doesn't benchmark against Frame Averaging, SE(3)-Transformers, or other approaches that address symmetry issues. Without this, the claim of practical contribution is unsupported.

2. **QM9 tests only one property (polarizability)** — ICLR expects comprehensive empirical validation. Testing a single scalar property cannot support claims about general effectiveness across tasks with different symmetry requirements (e.g., vector/tensor properties, forces).

3. **No ablation showing guidelines directly improve performance** — The QM9 results show correlation between symmetry increase and error, but don't demonstrate that *following the guidelines* causally improves performance versus ignoring them. Add experiments where feature spaces are deliberately chosen per §4.2 vs. randomly.

4. **Synthetic k-fold experiments don't validate the "almost isovariant" claim** — Section 5's core theoretical contribution (generic isovariance under manifold hypothesis) is only validated on constructed k-fold structures, not real data distributions where the manifold assumption may not hold.

5. **No evaluation of computational overhead** — The guidelines suggest high-multiplicity representations (r > 3), but practical ENNs often use few channels for efficiency. Missing analysis of the accuracy-computation tradeoff undermines practical utility claims.

### Deeper Analysis Needed (top 3-5 only)
1. **High-multiplicity assumption (r > 3) doesn't match practical regimes** — Most ENN applications use 1-4 channels due to memory constraints. The theory's sufficiency conditions may not apply where symmetry increase actually matters. Analyze low-multiplicity cases or justify why high multiplicity is realistic.

2. **Manifold hypothesis assumption is unverified for molecular data** — Section 5's genericity results depend on data lying on smooth compact submanifolds. QM9 molecules are discrete graphs with noise. Need analysis of how violations affect the "almost isovariant" guarantee.

3. **Correlation vs. causation in QM9 results** — High error correlates with full degeneration, but this doesn't prove symmetry increase *causes* the error. Alternative explanations (e.g., those symmetries are inherently harder to learn) aren't ruled out. Need controlled experiments isolating symmetry as the variable.

4. **No analysis of when symmetry increase is actually harmful** — The paper assumes symmetry increase degrades expressivity, but for invariant tasks (e.g., energy prediction), some increase may be beneficial. Missing discussion of task-dependent harm undermines the "prevention" framing.

5. **Algorithms assume known input symmetry groups** — Algo. 1-2 require the input isotropy subgroup as input, but in practice this is unknown. Missing discussion of how to estimate or handle unknown symmetries limits practical applicability.

### Visualizations & Case Studies
1. **Show successful symmetry preservation, not just degeneration** — Figs. 3-4 only illustrate failure cases. Add visualizations where following the guidelines *prevents* symmetry increase, demonstrating the method actually works.

2. **Case studies should compare guideline-compliant vs. non-compliant architectures** — §F.3.2 analyzes degenerate cases but doesn't show a molecule where redesigning features per §4.2 improves prediction. Add paired examples showing before/after applying guidelines.

3. **Visualize the "symmetry infimum" concept on real molecules** — The infimum is the paper's core theoretical contribution but is only shown on synthetic k-folds. Visualize computed infimums for actual QM9 molecules from different point groups.

### Obvious Next Steps
1. **Generalize beyond SO(3)/O(3)** — ENNs are used for SE(3), permutation groups, crystal symmetries, etc. The theory's restriction to rotation groups significantly limits its claimed generality. Extend or explicitly bound the scope.

2. **Provide quantitative design rules, not qualitative guidelines** — §4.2 says "select feature components that contain the orbit type" but doesn't specify which degrees/channels for common tasks. Add a prescriptive table mapping task types to recommended feature spaces.

3. **Address how to handle unknown input symmetries in practice** — The framework requires knowing Gx to compute IG(Y, Gx), but real applications don't have this. Propose estimation methods or discuss robustness to symmetry misidentification.

4. **Connect symmetry infimum to generalization bounds** — ICLR values theoretical contributions that connect to learning guarantees. The current theory characterizes expressivity but doesn't link to sample complexity or generalization, weakening the ML contribution.

5. **Release code for the orbit type computation algorithms** — The reproducibility statement mentions code, but Algo. 1-2 for computing symmetry infimum are central contributions. Ensure these are usable tools, not just proofs-of-concept limited to the paper's specific group representations.

# Final Consolidated Review
## Summary

This paper provides a rigorous mathematical characterization of "symmetry increase" in Equivariant Neural Networks (ENNs)—the phenomenon where processing symmetric inputs produces output representations with unintended higher symmetry, degrading expressivity. The central contribution is the concept of the "symmetry infimum," a unique lower bound on the unavoidable symmetry increase determined entirely by the algebraic structure of the feature space. The authors prove its existence and uniqueness (Thm. 3.1), develop computable algorithms to derive it for SO(3)/O(3) subgroups, show that generic equivariant maps achieve this infimum under mild assumptions (Thm. 5.2), and validate the theory with synthetic experiments and QM9 molecular property prediction.

## Strengths

- **Unified theoretical framework**: The paper provides the first comprehensive mathematical treatment of symmetry increase, subsuming prior partial results (the "collapse-to-zero" theory of Cen et al. (2024) for full degeneration, and empirical observations of Joshi et al. (2023)). The three-way classification of degenerations (full, axial, half) directly extends previous work.

- **Rigorous mathematical grounding**: Theorems 3.1 (uniqueness of minimal orbit type), 3.3 (necessary condition for isovariance relative to Y), and 5.2 (generic almost-isovariance) are technically sound. The proofs correctly leverage classical results (Michel's criterion, Ihrig-Golubitsky criterion) and adapt them to the ENN feature design context.

- **Computational contribution**: Algorithms 1 and 2, along with the comprehensive tables in Appendix E for SO(3) and O(3) subgroup infimums, provide practitioners with concrete tools to predict expressive limitations for specific input symmetries and feature configurations.

- **Empirical validation supports theory**: The synthetic k-fold experiments (§6.1-6.2) show clean binary patterns matching Table 1 predictions—cases predicted to degenerate show >10³ smaller embedding differences than distinguishable cases. The QM9 experiment (§6.3) correlates higher prediction error with feature degrees predicted to undergo full degeneration for specific molecular symmetry groups.

## Weaknesses

- **High-multiplicity assumption may not match practical regimes**: The key theoretical result (Prop. 4.2) requires multiplicity r > dim G (r > 3 for SO(3)) for Michel's criterion to be sufficient. While the paper notes this holds for high-channel architectures, many practical ENNs use few channels due to memory constraints. The paper states that r = 1 results coincide with high-multiplicity results (pointing to §C.4), but does not clearly explain why this coincidence occurs—this is essential for practitioners using low-multiplicity models.

- **QM9 experimental design has confounds**: The encoder is pretrained on all features (l ≤ 11) before fine-tuning with degree-specific masks. This means the encoder has already seen all degree information; the degree-specific fine-tuning heads may not cleanly isolate the effect of individual feature degrees. Training from scratch with controlled feature configurations would provide cleaner evidence.

- **Gap between "generic" maps and trained networks**: Theorem 5.2 guarantees almost-isovariance for *generic* parameterizations, but neural networks are trained by gradient descent with specific objectives. The paper does not address whether standard training tends to produce (almost) isovariant maps or could systematically select non-isovariant ones. This limits the practical guarantee.

- **Manifold hypothesis is strong and unverified for molecular data**: Theorem 5.2 assumes data lies on a finite union of smooth compact G-submanifolds (§A.3). Real molecular data (QM9) consists of discrete graphs with experimental noise. The paper acknowledges this briefly but does not analyze robustness to assumption violations.

- **Symmetry infimum computation requires known input symmetry**: Algorithms 1 and 2 require the input isotropy subgroup G_x as input. In practice, this information is often unknown. The paper does not discuss how to estimate symmetries or handle uncertainty in symmetry identification, limiting immediate applicability.

- **Algorithms lack complexity analysis**: The paper provides algorithms but does not discuss computational complexity. For Algorithm 2, enumeration of all supergroups could be exponential in principle. The appendix mentions a "top-down strategy" (Algo. 3), but scalability for large subgroup lattices is not addressed.

- **Scope limited to SO(3)/O(3)**: The theoretical framework applies to compact Lie groups, but computational results are specific to rotation groups. Many applications involve other symmetries (permutation groups, translational symmetries in crystals, SE(3) equivariance). The paper does not discuss extension to these cases.

- **Guidelines remain abstract for practitioners**: §4.2 states guidelines like "select feature components that contain the orbit type (p_Y(H))" for orientation-dependent tasks, but does not provide worked examples showing how to enumerate relevant input symmetry groups H and verify this condition in practice. A practitioner-level end-to-end example is missing.

- **Missing positive experimental evidence**: All experiments validate that degeneration occurs as predicted when using theoretically "bad" feature configurations. The paper does not show that following the guidelines (selecting "good" configurations) causally improves performance on downstream tasks.

## Nice-to-Haves

- **Computational overhead analysis**: Analysis of the time/space complexity for computing orbit types and symmetry infimums, particularly for practical scenarios.

- **End-to-end worked examples**: Concrete walkthrough showing how to apply the guidelines for a specific molecular task (e.g., energy prediction for molecules of known point group), including which feature degrees to select and how to verify the orbit-type inclusion condition.

- **Analysis of when symmetry increase is harmful vs. beneficial**: For invariant tasks (e.g., energy prediction), some symmetry increase toward the task-relevant symmetry is expected. The paper frames all increase as "degeneration" but does not distinguish harmful from benign cases.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Title "reduction" vs. "characterization" mismatch** — The paper provides both characterization AND guidelines for reducing symmetry increase. This is not a substantive criticism.

- **Proof "leans heavily on Azzi et al. (2023)"** — This is standard mathematical practice. The paper correctly attributes results and provides the novel Prop. B.4 for the fixed-point subspace setting. Attribution is appropriate.

- **"PDF parsing artifacts in figures"** — This is an artifact of the review format, not a paper problem.

- **G_1 sampling method in §6.2 "not specified"** — The appendix §F.2.1 explicitly states: "we apply a random rotation to obtain G_1." This is specified.

- **Missing comparison against existing symmetry-breaking methods** — This requests scope creep. The paper's contribution is theoretical characterization and prediction, not a new architecture competing against Frame Averaging or SE(3)-Transformers. The theory is complementary to such methods.

## Novel Insights

The paper's most significant conceptual insight is the precise identification of the "symmetry infimum" as the fundamental limit—recovery of input symmetry is impossible whenever the feature space Y cannot support orbit type (G_x), and the infimum IG(Y, G_x) precisely quantifies the minimal unavoidable increase. This reframes the ENN expressivity question from "does the model preserve symmetry?" to "what is the best achievable symmetry given the feature space?" The classification into full, axial, and half degeneration provides practical vocabulary: full degeneration (symmetry increases to the entire group, all orientation information lost), axial degeneration (symmetry increases to continuous subgroup, partial information preserved), and half degeneration (symmetry increases to discrete subgroup, fine-grained distinctions lost but gross orientation preserved).

## Suggestions

- **Add explicit capacity requirements for Thm. 5.2**: Specify what "sufficiently expressive" means in practice—what depth, width, or l_max are needed for common architectures to achieve the generic almost-isovariance guarantee?

- **Analyze low-multiplicity regimes**: Either provide explicit analysis of when the high-multiplicity coincidence holds for r < 3, or clearly bound the scope to architectures where r > 3 is realistic.

- **Control the QM9 experiment**: Train models from scratch with different feature configurations (including guideline-compliant ones) to isolate the causal effect of feature design on performance.

- **Address unknown input symmetries**: Discuss robustness of the framework to symmetry misidentification, or propose practical methods for estimating input symmetry groups from data.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
