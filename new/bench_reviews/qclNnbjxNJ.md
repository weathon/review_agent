Now I have enough context for calibration. Let me synthesize the review.

The key calibration points:
- CDIS (Dai et al., 2025), "When Selection Meets Intervention" - scored 8 (Oral) - handles *pre-treatment* selection with interventional data, sound algorithm, good theory
- GISL (Luo et al., 2025), "Gene Regulatory Network Inference" - scored 3-5 (Reject) - similar topic but weak theory, weak experiments, overclaims
- Current paper - same group as CDIS (Kun Zhang's group), extending to *post-treatment* selection

Now let me carefully assess the actual paper against reviewer claims.

Key issues to verify:

1. **Reliance on Type I inducing nodes** - The harsh reviewer claims this is a structural issue. The paper itself acknowledges this in Limitations (Section 6): "The identification of direct causal links and selection structures depends critically on the presence of Type I inducing nodes." The harsh reviewer argues this makes the main identifiability gain conditionally achievable only under strong assumptions. Let me check if this is valid...

Looking at Algorithm 1 Step 2.3, it explicitly requires detecting Type I inducing nodes and intervening on them. The completeness claim in Theorem 4 says "Each type of substructures represented by tail, arrowhead, square, ▲→, and ▲ between a pair of intervened nodes in the corresponding augmented DAG of Ĝp can be identified by different types of CI patterns." But this only applies between *intervened nodes*, and only when Type I nodes exist. The theorem appears sound within its scope but the scope is limited.

2. **Incomplete definition of FI-Markov equivalence** - The harsh reviewer says Definition 2 and the mapping to F-PAG is underspecified. Let me check... Definition 2 says "same d-separation among X[N]\I, and the same CI patterns between ψ and any intervened variable." This seems reasonable but the harsh reviewer raises valid concerns about whether the mapping from augmented DAG properties to F-PAG edge marks is fully established, particularly with truncated Theorem 2. Let me check Theorem 2 in the paper... The theorem statement in the paper is cut off: "Two augmented DAGs... are FI-Markov equivalent for a set of" and then the text continues to conditions about same skeleton, same v-structure, same inducing paths. The full formal statement is not visible. This is a legitimate concern.

3. **Experiments** - The harsh reviewer claims experiments don't validate the core claimed capability. Let me check Section 5... The simulations compare F-FCI against baselines on precision and SHD. Table 1 is mentioned for distinguishing post-treatment selection but not shown. The real-world experiment uses Enrichr validation but doesn't have ground truth for selection structure. The harsh reviewer is correct that the experiments mainly show improvement in standard metrics rather than specifically validating the ability to distinguish causation from selection.

4. **Treatment of S=1** - The harsh reviewer says this is conceptually thin. Looking at the paper, S=1 conditioning is mentioned and Lemma 1 acknowledges it introduces additional dependencies. The paper does address this but not deeply enough.

5. **Overclaiming of completeness** - Theorem 4's completeness is stated without clear qualification about intervention coverage, though Step 2.2 explicitly falls back to ambiguous marks (◦→) when Type I nodes aren't available.

Now let me finalize my assessment. The paper has genuine strengths in identifying an important problem and providing initial theoretical results. However, the theoretical framework is not fully rigorous (incomplete theorem statements, unclear mappings), the completeness claim is overstated relative to the assumptions, and the experiments don't directly validate the key new capability. This paper sits somewhere between the CDIS oral paper (scoring 8) and the GISL reject (scoring 3-5). It's better than GISL because the problem formulation is more rigorous and the theoretical framework is more careful, but it falls short of CDIS's level of completeness.

I'll give it around a 5-6 given the meaningful contribution in problem identification but significant gaps in execution and validation.</think>Now I have enough context from the calibration papers. Let me compile the final review.

## Summary

This paper addresses interventional causal discovery in the presence of both latent confounders and post-treatment selection—a common but overlooked challenge where samples are selectively included after interventions (e.g., quality control in single-cell perturbation studies). The authors demonstrate that post-treatment selection creates spurious dependencies mimicking causal relations under existing interventional frameworks, propose a new formulation with augmented DAGs, define a Fine-grained Interventional (FI) Markov equivalence class and F-PAG graphical representation, and develop the F-FCI algorithm with claimed soundness and completeness guarantees.

## Strengths

- **Important and well-motivated problem.** Post-treatment selection is genuinely overlooked in interventional causal discovery, and the demonstration that it produces invariance patterns (variant p(effect), invariant p(effect|cause)) identical to causal relations under existing frameworks (Figure 1, Section 2.2) is both correct and compelling. The examples from gene expression QC and clinical trial per-protocol analysis (Introduction) are concrete and relevant.

- **Conceptually insightful structural analysis.** The identification of "structural symmetries" (Figure 4) — that selection produces symmetric ψ-X CI patterns while causation produces asymmetric ones, and that further interventions on internal nodes ("Type I inducing nodes") break this symmetry — is a genuine insight that goes beyond simply adding a selection variable to existing models.

- **Constructive algorithmic approach.** The F-FCI algorithm (Algorithm 1) provides a concrete, step-by-step procedure: skeletal recovery from observational data, CI pattern extraction from interventional data, orientation rules, and Type I inducing node detection. The orientation rules in Step 2.2 that map specific CI pattern tuples to specific edge types (→, ↔, ◦→, etc.) are explicit and implementable.

- **Goes beyond standard PAG equivalence.** The F-PAG representation with new edge marks (□ for nodes with mixed tail/arrowhead inducing paths, ▲ for non-direct inducing paths) is a genuine extension of the PAG framework that can represent distinctions that standard PAGs collapse, as illustrated in Figure 5.

## Weaknesses

### Major:

- **The central identifiability gain is conditional on strong, largely implicit assumptions about intervention availability.** The paper's key refinement over standard interventional PAGs — distinguishing Figure 1(a) from (b), and (c) from (d) — requires hard interventions on "Type I inducing nodes" (Step 2.3 of Algorithm 1). The paper acknowledges in Section 6 that "The identification of direct causal links and selection structures depends critically on the presence of Type I inducing nodes," but treats this as a narrow limitation. In reality, this is a design constraint on the experimental setup: one must intervene on specific intermediate nodes that happen to lie on inducing paths. In realistic applications (e.g., gene perturbation studies), we do not know in advance where Type I nodes lie, and intervention targets are typically chosen independently. Without these nodes, Step 2.2 falls back to ambiguous marks (◦→), yielding no gain over standard methods. The paper does not characterize when Type I nodes exist, how often they are accessible for a given intervention set, or provide any fallback analysis. This is not an implementation detail but a fundamental scope limitation on the paper's core claim.

- **The completeness claim (Theorem 4) is overstated relative to the algorithm's actual operation.** Theorem 4 states completeness for all substructure types between intervened nodes, but F-FCI's Step 2.2 explicitly uses ambiguous circle marks (◦→) when Type I inducing nodes are unavailable. The completeness guarantee thus applies only under an unstated condition that sufficient hard interventions on Type I nodes are available. The theorem should be qualified with the precise intervention-design requirements under which it holds. Similarly, Theorem 3's soundness is stated unconditionally but the algorithm's behavior when interventions don't cover internal nodes is to leave edges with circle marks, which is consistent but trivially so — any method is "sound" if it outputs uncertainty.

- **The theoretical framework has incomplete formalization in key places.** Definition 2 (FI-Markov equivalence) defines equivalence in terms of d-separation in augmented DAGs and CI patterns between ψ and intervened variables, but the mapping from these conditions to the specific edge marks in F-PAGs is not fully established within the main text. Theorem 2's full statement is truncated in the paper, and the reader cannot verify the claimed correspondence between FI-equivalence and the tail/arrowhead/square/▲/▲→ marks. Lemmas 2–4 provide conditions for specific edge mark types, but the derivation from augmented DAGs (with L and S latent) to marginal MAG representations under selection conditioning is not fully detailed—the interplay between conditioning on S=1 (which by Lemma 1 adds dependencies) and the standard MAG theory is acknowledged but not rigorously integrated.

- **Experiments do not directly validate the main claimed capability.** The core contribution is distinguishing post-treatment selection from causal relations, yet the experiments report standard precision/SHD metrics against baselines that do not model selection. Improved SHD over methods that ignore a known confounding factor (selection) is expected regardless of whether F-FCI correctly identifies the selection mechanism. There is no controlled experiment where two graphs that are *Markov equivalent under standard interventional frameworks but distinct under FI-equivalence* are compared head-to-head, and Table 1 (referenced for "ability to distinguish post-treatment selection") is not visible in the provided text. The real-world Norman dataset evaluation relies on enrichment analysis (Enrichr) which validates plausibility of recovered causal edges but cannot confirm whether edges identified as "post-treatment selection" are truly selection artifacts vs. genuine causal relations.

### Minor:

- **Relationship to pre-treatment selection (CDIS) is underexplored.** The paper distinguishes post-treatment from pre-treatment selection but does not discuss how F-FCI handles the mixed setting where both types coexist, even though CDIS (Dai et al., 2025) is included as a baseline and addresses pre-treatment selection. Given that real biological studies can have both, this boundary condition matters.

- **Notation density impedes accessibility.** The paper introduces augmented DAGs, intervention indicators, FI-Markov equivalence, F-PAGs, Type I/II inducing nodes, square and triangle edge marks, and multiple CI pattern tuples in rapid succession. This makes the paper hard to follow even for readers familiar with causal graph theory, reducing the probability of adoption and extension.

- **Scalability concerns.** Like all constraint-based methods, F-FCI's Step 2.1 involves CI tests over conditioning sets along paths, which can be exponential. The paper mentions a scalability evaluation (Figure 11 in Appendix) but no complexity analysis is provided in the main text.

## Nice-to-Haves

- Explicit characterization of the minimum intervention set required for FI-equivalence refinement, or at least empirical analysis of how often Type I inducing nodes are available in random graphs.
- Ablation experiment comparing F-FCI to a variant that ignores selection but otherwise uses the same augmented DAG + intervention indicators, to isolate the contribution of selection modeling.
- Per-edge type evaluation (accuracy on ▲ vs. → vs. ↔ edges) rather than aggregate precision/SHD, to directly validate the new identification capability.
- Finite-sample robustness analysis under imperfect CI tests, which is critical since the algorithm requires many CI tests with interventional data.

## Removed Points

- **"Not yet released" / availability concern about code:** The paper references a GitHub repository for F-FCI implementation. Removed per hard rules — if the paper cites it, it exists.
- **Demand for missing related works:** The neutral reviewer suggested comparing with deep learning-based causal discovery methods. This is scope creep — F-FCI is a constraint-based method, and comparing with functionally different paradigms without shared baselines would not be informative. Removed.
- **Formatting/style nitpicks:** Several reviewers pointed out notational density and proof accessibility issues. While the notation is dense, this is inherent to the topic (mixed graphs with selection), and presentation alone is not a substantive weakness.
- **Demand for user studies / interpretability for domain experts:** The neutral reviewer's suggestion that F-PAG needs a "reading guide for domain scientists" is outside the paper's scope. The paper is a theoretical/methodological contribution, not a usability study.
- **Demand for finite-sample statistical analysis:** While important, single-run evaluation with oracle/near-oracle CI tests is standard practice in constraint-based causal discovery theory papers (e.g., FCI, CDIS). This is a nice-to-have, not a core flaw.
- **Claim that the paper doesn't discuss post-treatment selection in real data:** The paper does attempt this (Section 5.2, Norman dataset). The weakness is that the validation is indirect, not that the claim is absent.
- **Harsh reviewer's claim that "completeness is only for intervened node pairs" is a structural flaw:** This is actually an expected and clearly stated scope limitation. Algorithms like JCI-FCI also primarily recover orientations at intervened variables first. The completeness for intervened pairs is correctly stated in Theorem 4. Outside the "Type I" scope issue (which is a legitimate major concern), this specific narrowness is by design.

## Novel Insights

The distinction between pre-treatment and post-treatment selection in interventional settings is genuinely novel and practically significant. The key theoretical observation — that post-treatment selection produces the same invariance pattern as causation (variant p(Y), invariant p(Y|X) when intervening on X), making it invisible to standard interventional discovery — is correctly identified and well-illustrated. The proposed resolution using structural symmetry of ψ-X CI patterns and further hard interventions on internal inducing nodes is creative, though it ties the method's advantage to experimental design rather than purely to data analysis.

## Suggestions

- **Qualify completeness and identifiability claims explicitly.** State Theorem 4 with the precise condition: "Given hard interventions on all Type I inducing nodes along inducing paths between intervened pairs, F-FCI is complete." Discuss what happens when this condition is not met.
- **Add a targeted experiment for the distinguishing capability.** Design a controlled simulation where pairs of DAGs are Markov equivalent under standard interventional equivalence but distinct under FI-equivalence, and show that F-FCI correctly distinguishes them (while baselines cannot). This directly validates the paper's core claim.
- **Provide edge-type-level evaluation.** Report accuracy on each new edge type (▲→, ▲, □, □□) separately, not just aggregate precision/SHD, to show the method actually recovers the distinctions it claims.
- **Add fallback analysis.** Characterize what F-FCI returns when Type I inducing nodes are not intervened on — does it gracefully degrade to standard PAG/FCI output, or does it produce incorrect results? Empirical analysis of this degradation would significantly strengthen practical applicability claims.

## Evaluation

**Originality:** High. Post-treatment selection in interventional causal discovery is genuinely underexplored, and the FI-equivalence framework is novel.

**Importance of research question:** High. The problem is relevant to real experimental settings (gene perturbation, clinical trials).

**Claims well supported:** Partially. The theoretical framework is promising but incompletely formalized (truncated Theorem 2, unresolved mapping from augmented DAG properties to F-PAG marks). The completeness claim overreaches the conditions under which it holds. Experiments validate overall improvement but not the distinguishing capability that is the paper's central claim.

**Soundness of experiments:** Moderate. Simulations are standard but don't isolate the key contribution. Real-world validation is indirect.

**Clarity of writing:** Below average for the field. Heavy notation, many new concepts introduced rapidly, truncated theorem statements.

**Value to research community:** Moderate-to-high potential, but currently limited by the conditional nature of the identifiability result and the incomplete empirical validation of the distinguishing capability.

Compared to the CDIS paper (Dai et al., 2025, same group) — which scored 8 and was accepted as Oral — this paper addresses a harder problem (post-treatment selection adds complexity beyond pre-treatment) but has a less complete theoretical treatment and weaker experimental validation. The CDIS paper benefited from a cleaner problem scope and stronger validation. Compared to GISL (same group, scored 3-5, rejected) — which had vague theorems, small experiments, and overclaimed identifiability — this paper is significantly more rigorous in its theoretical setup but still suffers from overclaimed completeness relative to assumptions and inadequate experimental validation of the key capability.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>