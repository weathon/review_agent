Now I have all the information needed. Let me compose the final review.

## Summary

This paper introduces a zigzag persistence framework from topological data analysis (TDA) to analyze how internal representations of LLMs evolve across layers. Rather than comparing static layer representations, the framework tracks topological features (specifically, $p$-cycles) as they are born and die across the layer sequence, yielding a descriptor called *persistence similarity* ($\mathcal{S}_p$). The authors observe a consistent three-phase profile in average persistence similarity ($\bar{S}_1$) across seven transformer models—increasing, plateauing, and decreasing—and demonstrate layer pruning based on persistence similarity as a practical showcase, achieving results "comparable" to state-of-the-art similarity-based pruning methods.

## Strengths

- **Novel conceptual contribution**: Applying zigzag persistence to LLM internal representations is genuinely creative. The mathematical framework (Equations 1–6) is correctly formulated, and interpreting layer sequences as a dynamic system amenable to zigzag persistence is a natural and previously unexplored idea. The construction of kNN-based filtrations with intersection layers (Equation 2) and effective persistence images (Equation 4) are sensible technical contributions specific to this application.

- **Consistent universal three-phase pattern**: Figure 4 (Right Panel) demonstrates that $\bar{S}_1$ exhibits a three-phase profile (increase, plateau, decrease) peaking at the same relative depth across all seven evaluated models (Llama 2 7B/13B/70B, Llama 3 8B/70B, Mistral 7B, Pythia 6.9B). The increasing phase has a constant rate across models. This is a non-trivial empirical finding that persists across architectures, sizes, and datasets.

- **Robustness across hyperparameters**: Figure 4 (Left Panel) shows that the qualitative three-phase structure is preserved across $k_{\text{NN}} \in \{2, 5, 8, 11, 15\}$, with kNN only scaling the curve rather than altering its shape, lending practical credibility.

- **Asymmetric similarity matrices**: Figure 3 shows that persistence similarity matrices are approximately but not exactly symmetric—this is a genuinely different property from standard pairwise measures like CKA and could reflect meaningful aspects of the directional transformation between layers.

- **Well-written and accessible presentation**: The paper clearly explains zigzag persistence for an audience that may be unfamiliar with TDA, and provides complete algorithmic specifications (Algorithms 1–2).

## Weaknesses

### Fatal
None.

### Major

- **No direct comparison with standard similarity measures across layers undermines the "deeper insights" claim**: The abstract and introduction claim persistence similarity provides "deeper insights" than "traditional similarity measures" (line 15, line 40). However, the paper never directly compares $\bar{S}_1$ profiles against, e.g., CKA, angular distance, or linear regression similarity across layers. The three-phase pattern (increase in early layers, plateau in middle layers, decrease in late layers) closely mirrors what is already known from CKA and cosine similarity studies—that middle layers are most similar to each other while early and late layers diverge. Without such a comparison, it is impossible to assess whether zigzag persistence reveals genuinely new structure beyond what simpler, cheaper-to-compute measures already capture. This gap significantly weakens the central claim of the paper.

- **The pruning application demonstrates "comparable" but not superior performance to simpler methods**: The paper positions layer pruning as a key practical validation, yet Table 1 shows mixed results. On Llama 2 7B MMLU at 10% cut, "other works" achieves 43.95 vs. 37.38 for this work. On Mistral 7B MMLU at 20% cut, "other works" achieves 37.86 vs. 24.26 for this work. Conversely, this work wins on Mistral 7B MMLU at 10% cut (53.17 vs. 38.20). The paper claims "comparable performance," which is accurate, but this actually undermines rather than strengthens the case for the framework: a method that requires computing kNN graphs and zigzag persistence across all layers (~2 hours for 10K points, line 121) achieves no clear advantage over simple angular similarity. If the practical application doesn't benefit from the added complexity, the contribution rests entirely on whether the topological insights are genuinely new—which brings us back to the first major weakness.

### Minor

- **The kNN hyperparameter selection criterion lacks principled grounding**: The paper chooses $k_{\text{NN}}$ to "maximize the total number of cycles" (Section 4.2), which is an arbitrary criterion. While Figure 4 (Left Panel) demonstrates robustness, this shows stability of observations rather than justification for a particular choice. The paper acknowledges this (line 281), so this is a recognized limitation rather than an unaddressed flaw.

- **Pruning high-similarity layers requires clearer justification**: Algorithm 2 prunes layers where $\bar{S}_1$ exceeds a threshold (i.e., layers in the "plateau" phase). The paper claims these layers "retain the most cycles" and are therefore "redundant." But high persistence similarity means the topology at these layers is stable and similar to neighbors—this could alternatively indicate that these layers are where important representations are consolidated, not where redundancy resides. The empirical results work out, but the conceptual reasoning is underdeveloped.

### Trivial
None.

## Nice-to-Haves

- A direct comparison plotting $\bar{S}_1$ profiles alongside CKA / angular distance / cosine similarity profiles across the same model layers would either validate that persistence similarity captures genuinely different information (strengthening the paper) or clarify what the additional information is. This single experiment would substantially strengthen the paper.
- An ablation pruning low-similarity layers instead of high-similarity ones, to establish whether the pruning success comes from the topological insight per se or simply from removing middle layers (which is already known to be viable for LLM pruning).
- Analysis of what persistent 1-cycles correspond to semantically, e.g., whether points in the same cycle share linguistic or semantic properties, to ground the topological findings in model behavior.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that the "captures the entire evolutionary trajectory" framing is misleading**: The paper does address this in Section 3.2, noting that p-cycles are equivalence classes and can change their simplex composition across layers. The claim is about tracking whether a feature persists, not tracking its exact composition — a legitimate use of "trajectory" in the zigzag persistence sense. Overstated in the abstract, but not fundamentally misleading.

- **Harsh Critic's claim about the intersection construction yielding sparse intersections for distant layers**: In principle true, but this is inherent to the zigzag persistence framework and empirically the method produces meaningful results. The paper's framework correctly handles this through the birth/death formalism.

- **Harsh Critic's claim that 1-cycles may be artifacts of kNN construction**: The paper explicitly discusses this (line 177), noting it "might be expected for a kNN-graph based construction, since connections are dense even for low values of kNN." The focus on 1-cycles is justified by their prevalence and the stability analysis in Appendix B.

- **Harsh Critic's concern about Equation 5 notation**: This is a minor notational clarity issue, not a substantive weakness.

- **Strength Finder's claim that persistence similarity "cannot be recovered from pairwise static comparisons alone"**: This claim needs qualification. The $\bar{S}_1$ average profile could potentially be similar to an average of pairwise CKA values—this is precisely the concern raised as a Major weakness above, and cannot be listed as a strength without direct evidence.

## Novel Insights

The most interesting insight is the potential asymmetry in persistence similarity matrices (Figure 3): $\mathcal{S}_p(\ell_1, \ell_2) \neq \mathcal{S}_p(\ell_2, \ell_1)$ by construction, since cycles born before $\ell_1$ and surviving to $\ell_2$ depend on the specific path between them. This directional property—tracking whether features persist through the actual transformation path rather than just comparing endpoint geometry—is the core distinction from standard similarity measures. If this asymmetry turns out to carry meaningful information about the sequential processing in transformers (as the paper hints at), it would validate the zigzag approach beyond what CKA-type measures can offer. Currently, however, the paper doesn't isolate or analyze this asymmetry as a distinct finding, blending it instead with the symmetric $\bar{S}_1$ profile that could be recapitulating known layer-similarity patterns.

## Suggestions

- Add a direct comparison of $\bar{S}_1$ profiles with CKA, angular distance, and/or cosine similarity curves computed across the same model layers. This is the single most impactful experiment to validate the paper's core claim.
- Analyze and report the asymmetry in $\mathcal{S}_p(\ell_1, \ell_2)$ vs. $\mathcal{S}_p(\ell_2, \ell_1)$ as a distinct finding—quantify how much the similarity matrices differ from their transpose and compare this with the asymmetry (or lack thereof) in CKA.
- Add a random layer pruning baseline to Table 1 to contextualize whether both this method and "other works" outperform naïve pruning, or whether all methods are roughly equally effective.

## Evaluation

**Originality**: High. The application of zigzag persistence to LLM representations is novel and conceptually well-motivated. The kNN-based filtration construction and effective persistence images are genuine technical contributions.

**Importance of research question**: Moderate to high. Understanding internal representations of LLMs is important, and topological methods offer a genuinely different lens from standard similarity measures.

**Whether claims are well supported**: Partially. The empirical observations (three-phase pattern, robustness, cross-model universality) are well supported. The claim that persistence similarity provides "deeper insights" beyond traditional methods is not supported, as there is no direct comparison. The pruning claim is supported ("comparable") but undermines the practical value of the framework.

**Soundness of experiments**: Moderate. The experiments are correctly executed and cover multiple models and benchmarks, but the critical comparison with standard similarity measures is missing, and the pruning comparison is limited by its mixed results.

**Clarity of writing**: Good. The paper clearly explains zigzag persistence for a broad audience and provides clean algorithmic descriptions.

**Value to the research community**: Moderate. The framework is a promising conceptual lens that could inspire further work, but without clearer evidence that it outperforms or reveals different information than standard measures, its immediate practical value is limited.

## Score Calibration

- **High anchors**: TopoNets (7.5 avg, Accept Spotlight) — similar topology+DL domain but with clear practical contributions and stronger empirical validation. This paper is clearly below TopoNets.
- **Medium anchors**: Convexity in deep representations (6.0 avg, Reject) — novel analytical framework (convexity) with empirical findings but criticized for lacking practical impact. Very similar profile: novel lens, real observations, limited validation of added value over simpler tools. Layer similarity analysis (vVxeFSR4fU, 6.5 avg, Accept Poster) — also studies layer-wise similarity in transformers with a practical application. Somewhat above this paper because it provides theoretical justification and demonstrates practical utility.
- **Low anchors**: NormWear (3.0 avg, Reject) — claimed improvement but baselines were weak. This paper is clearly above this level; it has a genuine novel framework and non-trivial findings.

This paper is comparable to the convexity paper (6.0, Reject) in profile: novel analytical lens, interesting empirical observations, but incompletely validated against simpler alternatives. It falls between the low 5s (where the descriptor paper tZk3LnvVtK sits at 5.6, Reject) and the 6.0-6.5 range (where layer similarity with practical contributions sits). The key question is: does the conceptual novelty of zigzag persistence for LLMs compensate for the incomplete validation? The answer is borderline—the framework is promising but the paper needs the missing comparison experiment to make a convincing case that TDA adds genuinely new information beyond standard similarity measures.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>