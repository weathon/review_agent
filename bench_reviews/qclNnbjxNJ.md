## Summary

This paper addresses interventional causal discovery in the presence of both latent confounders and post-treatment selection—a realistic but previously overlooked challenge in biological and clinical data where samples are filtered after intervention (e.g., quality control in scRNA-seq). The authors propose a new formulation using Augmented DAGs with explicit selection variables, define a finer FI-Markov equivalence class with corresponding F-PAG graphical representation, and develop the F-FCI algorithm to distinguish causal relations from selection-induced spurious dependencies. The core insight is that intervention on intermediate "Type I inducing nodes" can disambiguate causal structures that existing frameworks cannot.

## Strengths

- **Novel problem formulation:** Post-treatment selection is a genuine gap in the literature. The paper correctly distinguishes it from pre-treatment selection (addressed by CDIS) and biological constraints (addressed by GISL). The key observation—that post-treatment selection yields variant marginals with invariant conditionals, mimicking causal patterns—is correct and the proposed solution using additional interventions on intermediate nodes is non-trivial.

- **Theoretical contribution:** The FI-Markov equivalence class and F-PAG representation extend standard interventional equivalence classes in a meaningful way. The characterization of how CI patterns distinguish causation, latent confounding, and selection (Figure 4 and Lemmas 2–4) provides the formal machinery for identification.

- **Clear biological motivation:** The connection to scRNA-seq quality control and clinical per-protocol analysis grounds the work in real applications where post-treatment selection is unavoidable.

## Weaknesses

- **Completeness claim is qualified:** Theorem 4 claims completeness, but the proof acknowledges that identification of key structures (→− and −) requires Type I inducing nodes on relevant paths. When paths contain only Type II nodes, these marks cannot be identified. The limitation section mentions this, but Theorem 4's statement does not reflect this restriction. This matters practically: if Type II-only paths are common, the algorithm provides no advantage over standard methods.

- **Empirical validation is thin:** Only 10 random graphs per configuration with high variance (Table 1 shows ±15-24% standard deviations). For constraint-based methods where run-to-run variability can be substantial, this is insufficient for reliable conclusions. The selection identification accuracy (57-67% at 500 samples, reaching 70-94% only at 2000 samples) is modest and the high variance undermines confidence in the improvements claimed.

- **No ablation on Type I refinement:** Step 2.3 (Type I node refinement) is the algorithm's key novelty for going beyond standard equivalence classes. Without isolating its contribution from Step 2.2 (endpoint-based orientation), we cannot assess whether the proposed method's gains come from the new theoretical machinery or simply from leveraging more intervention targets.

- **Real-world validation is qualitative:** The Norman dataset evaluation confirms a handful of gene regulatory relationships against enrichment databases, but these databases are not comprehensive ground truth. The claims about correctly identifying CDKN1A, CDKN1C, ZNF318, and RREB1 as selection-affected rely on biological plausibility arguments rather than independent validation. No precision/recall metrics are provided against a held-out ground truth.

- **No analysis of Type I node frequency:** The core identifiability claim depends on Type I inducing nodes existing along paths between intervened variables. The paper provides no theoretical or empirical characterization of how often this condition holds in realistic graph structures, making it difficult to assess when the method actually works in practice.

## Nice-to-Haves

- Analysis of robustness to CI test errors would strengthen practical applicability claims.

- Empirical comparison with methods that handle selection bias in other ways (beyond just noting theoretical differences) would better situate the contribution.

- Characterization of intervention target misspecification robustness would enhance real-world applicability.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Definition 2 is under-specified regarding conditioning sets"**: The conditioning sets are explicitly enumerated in Algorithm 1 Step 2.1. While Definition 2 could be more explicit, the algorithm provides the specification.

- **"Type I inducing node definition is circular"**: While Definition 6 references F-PAG, the structural property (incoming arrowhead into a square) refers to the ground truth structure, not solely the learned representation. The apparent circularity is resolvable.

- **"Missing comparison with CDIS/GISL on post-treatment data"**: The paper correctly explains in Appendix E that CDIS handles pre-treatment selection (different invariance patterns) and GISL handles biological constraints—these are fundamentally different problems. Demanding comparison on post-treatment selection is scope creep.

- **"The conditioning throughout on S=1 needs clearer treatment"**: Conditioning on selection is standard in selection bias literature. While additional exposition could help, this does not represent a flaw in the approach.

## Novel Insights

The structural symmetry argument distinguishing causation from selection is genuinely insightful: direct causation yields asymmetric CI patterns (intervening on X₁ changes p(X₂) but not p(X₂|X₁), while intervening on X₂ does not change p(X₁)), whereas symmetric selection yields symmetric patterns (both interventions change marginals). This asymmetry, exploitable via interventions on third variables, provides a principled way to separate genuine effects from selection artifacts—an insight that generalizes beyond the specific algorithm proposed.

## Suggestions

- Add experiments or theoretical analysis characterizing the frequency of Type I vs. Type II inducing nodes in random graphs to quantify when identifiability guarantees actually hold.

- Include an ablation study separating Step 2.2 (CI pattern matching) from Step 2.3 (Type I refinement) to isolate the contribution of the novel refinement procedure.

- Provide quantitative evaluation on real data against known causal relationships (e.g., from experimental validation studies) rather than enrichment database consistency alone.

- Explicitly state the Type I inducing node requirement as a restriction in Theorem 4's statement rather than only in the limitations section.