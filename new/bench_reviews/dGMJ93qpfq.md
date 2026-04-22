Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper proposes Patch-Aware Prompting (PAP), a modular framework that incorporates patch-level information across the vision, text, and prediction branches of prompt tuning for CLIP models. It introduces (1) an intra- and inter-view patch consistency loss, (2) view-tailored text prompts conditioned on Voronoi-clustered patch features, and (3) patch-enhanced vision features with inter-view logit regularization. PAP is evaluated on 11 datasets across three transfer settings (base-to-novel, cross-dataset, domain generalization), achieving consistent but modest improvements over PromptSRC and DePT.

## Strengths

- **Well-motivated core idea**: The observation that existing consistency-based prompt tuning methods operate only at the global feature level, overlooking patch-level information, identifies a genuine gap. The paper systematically addresses this gap across all three branches (vision, text, predictions), which is a principled design choice (Section 3.2, Eqs. 5–14).

- **Consistent improvements across all evaluation settings**: PAP improves HM over PromptSRC by +1.08% on base-to-novel (Table 1a), +0.64% on cross-dataset average (Table 2), and +0.31% on domain generalization average (Table 3). Improvements are not sporadic—every single dataset in base-to-novel shows positive HM gain. When combined with DePT, similar improvements are observed (+1.09% HM, Table 1a).

- **Intra- and inter-view patch losses are complementary**: Table 6 shows that intra-view alone (80.06 HM) and inter-view alone (80.03 HM) both underperform the combination (81.05 HM), confirming that both mechanisms address distinct regularization needs.

- **Modular and composable design**: Table 11 demonstrates improvements when applied to CoCoop (+0.85% HM) and CoPrompt (+0.84% HM), confirming PAP is not tailored to a single base method.

- **Comprehensive ablation studies**: Tables 4–12 cover component contributions, individual loss effects, clustering alternatives (Table 8: Voronoi 81.05 vs. KMeans 79.51 vs. EM 79.22), projection/adapter choices (Table 9), and augmentation/crop configurations, providing solid empirical justification for design decisions.

## Weaknesses

### Fatal
None.

### Major

- **Improvements over baselines are modest given the added complexity, and the attribution of gains to patch-level semantics is not fully established**: The consistent ~1% HM improvements come at the cost of 10× more learnable parameters over PromptSRC (0.46M → 4.89M), 2× training time, and a framework with three loss components, a convolution projection block, a text adapter, Voronoi clustering, and view-specific forward passes. The paper does not include a control experiment that adds comparable regularization capacity (second view, adapter, projection head, additional losses) *without* patch-level semantics. Without such a control, it is difficult to determine whether the improvements come from patch-level information specifically or from the increased regularization and model capacity. This concern is partially mitigated by the ablation showing each component contributes (Table 4), but the ablation does not isolate the patch information itself from the architectural additions that carry it.

- **Per-dataset hyperparameter tuning creates tension with the generalization claim**: Section 4 states "we set λ_p, λ_t, λ_l to 1.0, 0.1, 1.0 respectively as default but modify it for individual dataset when required." A paper whose central thesis is improved *generalization* through patch-level information permits per-dataset tuning of its loss weights, yet provides no details about which datasets required modifications, what the modifications were, or how sensitive results are to these choices. This is especially notable when the reported improvements are sub-1.5%—even small amounts of per-dataset tuning could account for a meaningful fraction of the gains. A sensitivity analysis with fixed λ values across all datasets would substantially strengthen the paper.

### Minor

- **The "first integration" claim in the abstract is imprecise**: The abstract states the method is "representing the first integration of such semantics in this context." However, Section 2 acknowledges that Long et al. (2024) "uses clustered patch tokens for text prompts." While Long et al. lacks inter-view consistency and patch integration into predictions (making PAP's scope broader), the blanket "first integration" claim is inaccurate. The claim should be qualified to specify "first integration across all three branches" or similar.

- **Voronoi clustering's superiority over KMeans lacks explanation**: Table 8 shows a notable 1.54% HM gap between Voronoi (81.05) and KMeans (79.51), with most of the difference on novel classes (77.41 vs. 74.97). The paper simply states "Voronoi clustering generates more generalizable clusters" without explaining why. Analyzing cluster properties (balance, spatial coherence, per-class behavior) would strengthen the mechanistic understanding.

- **Inter-view patch matching in Eq. 6 uses purely feature-based nearest neighbor without spatial correspondence**: Since the two views are different crops/augmentations, spatial correspondence is broken. The paper asserts that "using zero-shot outputs to calculate similarity prevents the model from finding an easier learning path" (line 151) but does not empirically validate this (e.g., comparing zero-shot vs. prompted-feature matching) or analyze what fraction of matches are semantically correct.

- **The stop-gradient design in Eqs. 11 and 13 lacks justification or ablation**: Both apply stop-gradient to the anchor view, making it a fixed target. While this design has precedent in SSL (BYOL, SimSiam), the paper does not discuss the theoretical motivation or ablate alternative designs (symmetric updates, reversed stop-gradient). This is a design choice that could affect the method's behavior but is treated as arbitrary.

- **Parameter overhead framing is somewhat misleading**: The paper describes the parameter increase (0.46M → 4.89M for PromptSRC variant) as a "slight increase in learnable parameters" justified by comparison to the total CLIP model. However, comparing to CoCoop (3.53M) and CoPrompt (4.74M) shows PAP is in the same parameter range, and the 10× increase over PromptSRC is specific to PromptSRC's unusually lean design. The framing should be more balanced.

### Trivial
None.

## Nice-to-Haves

- A control experiment matching the parameter/regularization budget without patch semantics would definitively answer the attribution question.
- Sensitivity analysis for per-dataset hyperparameters (fixed λ vs. tuned λ) would clarify the source of improvements.
- Error analysis showing concrete cases where patch-level information corrects novel-class errors that global-feature methods make would strengthen the narrative beyond "slightly higher numbers."
- Patch attention/alignment visualizations for correct novel-class predictions vs. baselines.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Ablation tables are "unreadable" (all ✓ marks)**: The harsh critic claims Tables 4, 5, 6 display ✓ for every column, making ablations unverifiable. This is a **parser artifact**—the original submission uses ✓/✗ marks to indicate active/inactive components, but the text parser stripped the ✗ symbols. The surrounding text explicitly describes each ablation row ("Each component improves performance"), and the numerical values differ across rows, confirming distinct configurations. This is not an author error.

- **"Not yet released / cannot be independently verified" concerns about Long et al. or any cited model**: As per the hard rules, if the paper cites it, it exists.

- **Missing comparison with MaPLe and KgCoOp for modular claim**: Testing with only CoCoop and CoPrompt is sufficient to demonstrate modularity; demanding more base methods is a generic improvement request, not a weakness.

- **Demanding theoretical proofs for an empirical prompt tuning paper**: Not standard in this community; the paper provides adequate empirical justification through ablations.

- **Missing motivating example showing failure cases from lacking patch information**: This would strengthen the paper but is outside the stated scope; the motivation is clearly articulated conceptually.

- **Demanding confidence intervals or multiple runs**: Single-run evaluation is standard for large-scale prompt tuning benchmarks in this field.

## Novel Insights

The inter-view patch consistency mechanism (Eq. 6–7) is an interesting design that differs from standard multi-view SSL by anchoring to *zero-shot* patch features rather than using a momentum encoder. This creates an asymmetric design where the frozen CLIP model acts as a stable teacher at the patch level, which is distinct from global consistency methods that regularize only class-token embeddings. The key empirical finding that Voronoi clustering significantly outperforms KMeans specifically on novel classes (77.41 vs. 74.97) but is comparable on base classes (85.07 vs. 84.36) suggests the clustering method affects generalization behavior differently from base-class fitting—an observation that deserves deeper analysis.

## Suggestions

- Add a "patch vs. capacity" control: run PAP with the same multi-view, adapter, and projection infrastructure but replace patch-level features with global features throughout. This single experiment would cleanly isolate the contribution of patch-level information.
- Report results with fixed λ values across all datasets alongside the current per-dataset-tuned results to quantify the tuning contribution.
- Qualify the "first integration" claim in the abstract to "first integration of patch-level semantics across vision, text, and prediction branches."
- Provide per-dataset λ values in the supplementary to enable reproducibility.

## Calibration Anchors

| Paper | Avg Human Score | Comparison |
|-------|----------------|------------|
| CLIP Data-Free KD (1aF2D2CPHi) | 8.0 (Accept Oral) | Much stronger contribution with 9.33% improvement and novel problem formulation. PAP is clearly below this. |
| Local-Prompt (Ew3VifXaxZ) | 6.0 (Accept Poster) | Also uses patch-level/local info for CLIP adaptation. PAP is comparable in motivation but evaluated on different tasks (generalization vs. OOD detection). Slightly more complex with modest improvements. |
| CoPrompt (wsRXwlwx4w) | 5.75 (Accept Poster) | Very similar domain — consistency-guided prompt tuning for CLIP generalization. PAP adds patch-level information on top of this paradigm with similar-sized improvements. Comparable quality. |
| APPLe (YG01CZDpCq) | 5.5 (Reject) | Adaptive prompt prototypes for base-to-novel generalization with ~3.66% improvement on novel. Rejected despite larger improvements, partly for limited technical novelty. PAP has more components and a clearer structural contribution but even smaller gains and an attribution question. |
| MVMP (j1FLTvgyAh) | 2.5 (Reject) | Multi-component framework combining existing tricks with marginal gains. PAP is clearly above this — more principled design, clearer motivation, and consistent improvements. |
| LCN (wYVP4g8Low) | 3.0 (Reject) | Added complexity for marginal ~1% improvements over baselines, overclaimed novelty. PAP shares some similarity in the complexity-to-gain ratio, but PAP's improvements are more consistent and the design is more principled. |
| Active test-time prompt (pdzHpQbGrn) | 2.5 (Reject) | Marginal improvements, limited novelty. PAP is clearly above this. |

PAP sits in the medium band, most comparable to CoPrompt (5.75, Accept Poster) and Local-Prompt (6.0, Accept Poster). It is below these slightly because: (1) its improvements are modest even by the standards of this space, (2) the attribution of gains to patch information specifically is not fully established, and (3) the per-dataset hyperparameter tuning concern is not addressed. However, it is clearly above the low-band papers in motivation quality, design coherence, and consistency of results.

## Score and Decision

**Originality**: Moderate. The core idea of patch-level information in prompt tuning is natural but has been partially explored by Long et al. (2024). The specific multi-branch design with intra/inter-view consistency is novel in its scope but builds heavily on existing consistency regularization paradigms.

**Importance of research question**: High. Improving generalization of prompt-tuned VLMs is a central and active research question.

**Claims well supported**: Partially. Consistent improvements are shown, but the central claim that *patch-level information* drives the improvements (versus added regularization capacity) is not cleanly isolated.

**Soundness of experiments**: Adequate. Extensive evaluation across 11 datasets and 3 settings, with comprehensive ablations. The per-dataset tuning and missing capacity control are concerns but do not invalidate the results.

**Clarity of writing**: Good. The framework is clearly described with proper mathematical notation. Some imprecision in claims ("first integration," "slight increase") but overall readable.

**Value to community**: Moderate. The framework provides a useful direction (patch-level prompting) that could inspire follow-up work, but the current evidence does not definitively establish the mechanism.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>