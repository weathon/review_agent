=== CALIBRATION EXAMPLE 28 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Mostly yes. The paper indeed introduces a symmetric divergence variant and a normalized similarity measure for topological representation analysis. The phrase “scalable topological toolkit” is supported only for the lite / rank-based variants; the full SRTD itself is not scalable in any strong sense, so the title slightly overstates the scope unless interpreted as the toolkit as a whole.
- **Does the abstract clearly state the problem, method, and key results?**  
  Yes at a high level. It identifies the core issue with RTD asymmetry and lack of normalization, proposes SRTD / SRTD-lite and NTS, and summarizes empirical claims.
- **Are any claims in the abstract unsupported by the paper?**  
  Some claims are stronger than what the experiments establish. In particular, “more efficient, comprehensive, and interpretable divergence measure” for SRTD is not fully substantiated with broad empirical evidence; the paper mainly shows consistency with existing RTD-family behavior and some optimization performance. Also, the claim that NTS “provides a clearer view of inter-model relationships” is supported only by a specific LLM benchmarking setup, not by broad evidence across tasks or architectures.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes, the motivation is clear: RTD is asymmetric and unnormalized, limiting interpretability and cross-scenario comparison; CKA lacks topological sensitivity. That said, the paper would be stronger if it more carefully justified why asymmetry is actually a defect rather than a useful directional signal in some settings. The introduction treats symmetry as an unquestioned desideratum, which is not always obvious for representation comparison.
- **Are the contributions clearly stated and accurate?**  
  Largely yes. The paper states two main contributions: SRTD/SRTD-lite and NTS. However, the introduction claims SRTD “completes the theoretical framework of RTD” and “matches the top performance of existing RTD-based methods in optimization tasks,” which is too strong unless the comparison is confined strictly to the specific autoencoder setup in Appendix G.
- **Does the introduction over-claim or under-sell?**  
  It over-claims in a few places. Most notably, it suggests NTS “combines the interpretability of CKA with the structural sensitivity of TDA” as if this is established generally, whereas the paper only demonstrates it on selected synthetic, vision, and LLM settings. The introduction also implies that RTD’s asymmetry is theoretically unexplained, but the paper’s own later treatment shows a fairly specific relationship between RTD, Max-RTD, and SRTD; the “theoretical ambiguity” framing could be more precise.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  Partially. The high-level constructions are understandable: SRTD is built from a symmetric auxiliary graph based on min/max distance matrices, and NTS correlates merge orders on MST “core pairs.” However, several critical implementation details remain underspecified:
  - For **SRTD**, Algorithm 3 normalizes by the 0.9 quantile, but the exact dependence on persistent homology computation and the auxiliary matrix construction is not fully spelled out in the main text.  
  - For **SRTD-lite**, the algorithmic description in Appendix C / D is not fully clear enough to reproduce without prior knowledge of RTD-lite.  
  - For **NTS**, the definition of “core pairs” is potentially subtle: using the union of MST edges from the two point clouds is clear algorithmically, but the rationale for why this set is the right basis for a normalized similarity score is not established rigorously.
- **Are key assumptions stated and justified?**  
  Some are stated, but not well justified. The strongest assumption is the one-to-one correspondence between samples in both point clouds; the paper uses it throughout, but does not discuss how fragile the methods are when this assumption is only approximate or unavailable. This is important because many representation-analysis settings are not naturally paired pointwise.
- **Are there logical gaps in the derivation or reasoning?**  
  Yes, several. The most concerning is Theorem 3.3 and its proof in Appendix D.3:
  - The theorem statement appears to relate RTD, Max-RTD, and SRTD through kernel dimensions integrated over filtration radius, but the derivation from the long exact sequence is not fully rigorous in the paper as written.
  - The notation and algebra in the proof are difficult to parse, and the step from Betti-number relations to the claimed integral identity is not convincingly established.
  - The paper also asserts a “complementary phenomenon” between RTD and Max-RTD from empirical observation, then builds SRTD on that intuition. That is plausible, but not enough on its own to justify the new construction.
  
  For NTS, Theorem 4.1 and 4.2 are more straightforward, but there is still a logical gap: Theorem 4.2 relies on the existence of a strictly increasing map relating edge weights when NTS-E = 1. This is plausible under rank-equality assumptions, but the proof should be more careful about ties and about the effect of restricting to the union of MST edges rather than all pairwise edges.
- **Are there edge cases or failure modes not discussed?**  
  Yes:
  - **NTS can be unstable when MST unions are small or when many ties occur**, especially after quantization or low-precision embeddings.
  - **Symmetry of SRTD may hide meaningful directionality** in asymmetric comparisons, such as teacher-student or pre/post-finetuning analyses.
  - **Normalization choices matter**: SRTD uses quantile normalization and NTS-E requires z-score normalization in LLM experiments, which means the methods are not fully scale-free in practice.
  - Both methods depend heavily on MST structure; if the representation geometry is noisy, hub-like, or high-dimensional with distance concentration, the rank order may be brittle.
- **For theoretical claims: are proofs correct and complete?**  
  The proofs are not fully convincing as presented. Theorem 4.1 is reasonably plausible, though still somewhat informal. Theorem 4.2 is also reasonable, but the converse counterexample is only sketched and the proof depends on a specific choice of core edges and rank comparison. Theorem 3.3 and the associated corollaries are the weakest part: the chain-complex / mapping-cone argument is not cleanly presented, and the exact relationship between the auxiliary graphs and the claimed barcode sums is not fully transparent.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Partially. The experiments do address the central claims:
  - synthetic clustering tests for sensitivity to hierarchical structure,
  - UMAP embeddings to test monotonic trends,
  - autoencoder optimization to test utility as a loss,
  - TinyCNN layer analysis to test representation comparison,
  - LLM comparisons to test discrimination and saturation.
  
  However, the experimental suite is still narrow relative to the breadth of the claims. The paper positions NTS as a general representation-similarity tool, but the evidence is concentrated in a few controlled datasets and selected LLM families.
- **Are baselines appropriate and fairly compared?**  
  The chosen baselines are mostly relevant: CKA for normalized similarity, RTD / RTD-lite / Max-RTD for the topological family. But there are some concerns:
  - For the LLM analyses, CKA is the main baseline, but the paper does not compare against other widely used representation similarity metrics that could matter for the claims, such as RSA, SVCCA/PWCCA, or distance correlation in the same setting.
  - The autoencoder section compares within the RTD family, but if the claim is that SRTD is a better divergence for optimization, it would help to compare against stronger non-topological objectives or at least include a simpler baseline loss.
- **Are there missing ablations that would materially change conclusions?**  
  Yes, several:
  - For **NTS**, the paper needs ablations on the choice of core set: union of MST edges versus alternatives, and on using merge-time ranks versus edge-distance ranks in more settings.
  - The effect of **z-score normalization** for NTS in LLMs is only discussed in Appendix K.1, but this seems crucial enough to require a more systematic main-text ablation.
  - For **SRTD**, the role of the 0.9-quantile normalization should be tested more explicitly. Since the paper criticizes RTD for lack of normalization, the exact normalization mechanism for SRTD matters.
  - The optimization experiments would benefit from an ablation separating the effect of symmetry from the effect of the “lite” approximation.
- **Are error bars / statistical significance reported?**  
  Some error bars appear in the autoencoder tables, but the overall reporting is uneven. The layer-similarity and LLM heatmaps are largely qualitative, and the paper does not report statistical significance or confidence intervals for the key LLM claims. For a paper making strong claims about better discrimination and less saturation, quantitative summary metrics and uncertainty estimates would be important.
- **Do the results support the claims made, or are they cherry-picked?**  
  The results generally support the claims qualitatively, but there is some cherry-picking risk:
  - In the LLM section, the paper highlights a case where CKA “fails” and NTS looks more discriminative, but this is based on a selected layer and selected datasets.
  - The paper repeatedly emphasizes cases where NTS outperforms CKA, but does not clearly discuss cases where CKA may be equally good or where NTS is less informative.
  - The statement that NTS-E correctly identifies DeepSeek-R1-Ds as closer to Qwen than CKA does is interesting, but one example is not enough to establish broader superiority.
- **Are datasets and evaluation metrics appropriate?**  
  Mostly yes, but with caveats. The synthetic cluster experiment is appropriate for topology-sensitive methods. The TinyCNN/CIFAR-10 setup is standard for layer similarity. The LLM datasets, TruthfulQA and ToxiGen, are reasonable for eliciting representation differences, but the choice of last-token representations and specific layer selections should be more systematically justified. For NTS, Spearman correlation of ranks is a sensible metric, but because it only captures ordering, it may miss important magnitude differences in representation geometry.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  Yes, mainly the theory sections. The most difficult parts are:
  - the exact definition and construction of the auxiliary graphs for SRTD and Max-RTD,
  - the algebraic topology proof in Appendix D,
  - the relation between min/max matrices and union/intersection filtrations.
  
  The high-level ideas are understandable, but the formal exposition is not always clear enough for a reader to verify the claims independently.
- **Are figures and tables clear and informative?**  
  The figures appear useful in intent, especially the synthetic and LLM heatmaps, but the paper relies heavily on visual interpretation without enough quantitative summaries. Tables 1 and 2 are helpful, though the key takeaways from them are not always clearly tied back to specific claims. Some tables and figure captions also seem to over-interpret the results beyond what the numbers alone establish.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Partially. They acknowledge that NTS is currently analysis-only and non-differentiable, which is good. They also note that SRTD-lite can be dominated by long barcodes. However, several major limitations are not adequately discussed.
- **Are there fundamental limitations they missed?**  
  Yes:
  - Dependence on **paired data / one-to-one correspondence** is a major restriction.
  - Dependence on **MST structure** may make NTS brittle in high-dimensional noisy settings.
  - **Normalization sensitivity** is a practical limitation for both SRTD and NTS.
  - The methods may not generalize well to **unpaired representation comparison** or to settings where representation cardinalities differ.
- **Are there failure modes or negative societal impacts not discussed?**  
  The paper does not discuss societal impact beyond the usual omission, but the more relevant issue is methodological misuse: LLM representation comparisons can be over-interpreted as functional or causal claims. That risk is present here, especially when the paper infers lineage or “credibility” from similarity heatmaps. The broader impact section is minimal, and the paper should caution against reading these measures as direct evidence of model behavior or safety.

### Overall Assessment
This is a thoughtful paper with a potentially meaningful contribution: it extends the RTD family with a symmetric variant, and it proposes a rank-based normalized similarity measure that may be useful for layer and model comparison. The experimental story is plausible and the application to LLM representations is timely, which fits ICLR’s interest in representation analysis methods. That said, the paper’s strongest weaknesses are in the rigor and completeness of the theory, the limited scope of experimental validation relative to its broad claims, and the under-justified design choices behind NTS and the SRTD normalization. On ICLR’s bar, I think the contribution is interesting and likely publishable only if the authors substantially tighten the proofs, clarify the assumptions and limitations, and provide stronger ablations and quantitative evidence that the proposed measures are robust beyond the showcased examples.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes two extensions to representation-topology analysis for neural network representations: SRTD, a symmetric version of RTD with a related lightweight variant, and NTS, a normalized similarity score based on rank correlations of MST-derived merge information. The main claim is that SRTD completes and clarifies the theoretical RTD framework, while NTS provides a more interpretable, scale-invariant alternative to RTD-like divergences and can reveal layer/model relationships in CNNs and LLMs more clearly than CKA in the authors’ experiments.

### Strengths
1. **Addresses real shortcomings of prior RTD-style methods.**  
   The paper clearly identifies two limitations of RTD: asymmetry and lack of normalization/scale comparability. This is a meaningful problem statement because interpretability and cross-setting comparability are indeed important for representation analysis, especially under ICLR standards for practical relevance.

2. **Provides a coherent conceptual toolkit rather than a single metric.**  
   The decomposition into SRTD, SRTD-lite, and NTS is well-motivated: SRTD aims to “complete” the divergence family, while NTS tackles normalized similarity from a different angle. The relationship among RTD, Max-RTD, and SRTD is emphasized throughout, which makes the paper feel more systematic than an ad hoc metric proposal.

3. **Nontrivial theoretical claims, especially for the lite variant.**  
   The paper gives explicit results such as Corollary 3.4 and Corollary 3.5, and the lite-family identity/ordering appears mathematically clean. The proposed connection between MSTs and 0D persistence is leveraged in a way that yields an interpretable rank-based similarity measure.

4. **Experiments span multiple relevant settings.**  
   The paper evaluates on synthetic clustering, UMAP embeddings, autoencoder optimization, CNN layer analysis, and LLM representation comparisons. This breadth is a strength because it probes both controlled and practical settings, which ICLR reviewers often value for representation-analysis papers.

5. **NTS is potentially useful in practice.**  
   The reported claims that NTS is more discriminative than CKA on some LLM comparisons, while also being scalable, are practically appealing. If validated, this could be valuable for analyzing large models where geometry-based metrics saturate.

6. **The paper includes reproducibility-oriented details.**  
   Architecture specifics for TinyCNN, dataset choices, and several experimental parameters are provided in the appendices. That is better than many papers in this area, even though it still falls short of full reproducibility.

### Weaknesses
1. **The theoretical presentation is not fully convincing at ICLR standards.**  
   The paper makes strong formal claims, but several proofs and definitions are difficult to assess as presented. In particular, the SRTD construction and the theorem connecting SRTD, RTD, and Max-RTD are stated in a way that feels under-justified, with heavy reliance on auxiliary graph constructions and mapping-cone arguments that are not fully transparent to the reader. For ICLR, this raises concerns about mathematical rigor and verifiability.

2. **NTS appears somewhat bespoke and not clearly grounded as a universal similarity measure.**  
   NTS is built from the union of MST edges and Spearman correlation of merge times/distances. This is plausible, but the paper does not sufficiently justify why this particular construction is the right notion of “normalized topological similarity,” nor does it compare against other reasonable rank- or graph-based alternatives. The method may be useful, but the conceptual leap from MST rank correlation to a robust topological similarity is not fully established.

3. **Experimental evidence is suggestive but not yet decisive.**  
   The paper reports qualitative improvements over CKA and RTD variants, but many results are described mainly through heatmaps and narrative claims. There is limited evidence of statistical testing, effect sizes, robustness across seeds, or ablations that would demonstrate the gains are consistent and not dataset-specific. For ICLR, stronger quantitative validation would be expected.

4. **Baseline selection is somewhat narrow.**  
   CKA is used prominently, but the comparisons do not seem broad enough for the claims being made. For representation analysis, reviewers would likely expect stronger baseline coverage, especially for layer similarity and model-family comparison tasks. The paper cites why SVCCA is omitted, but the justification for excluding other relevant topological or geometric baselines is limited.

5. **The normalization step for NTS may reduce conceptual cleanliness.**  
   The paper emphasizes scale invariance, but later notes that Z-score normalization is crucial for NTS to work effectively. This makes the “normalized” story less clean than the abstract suggests. If preprocessing is essential, the method’s invariance properties should be more carefully characterized.

6. **Scalability claims are not fully substantiated.**  
   The paper argues that NTS-E is efficient, but complexity statements are somewhat informal and the runtime comparison is only shown on a TinyCNN setting. It is unclear how the method scales for the very large LLM comparisons shown in the paper, and whether the end-to-end cost is practical for broader use.

7. **Interpretability claims are somewhat overstated.**  
   The paper repeatedly claims that NTS “provides a clearer view” and that CKA “fails,” but the evidence shown does not fully justify such strong conclusions. In several cases, the results appear to be dataset- and layer-dependent, and it is not demonstrated that NTS is consistently superior across all relevant settings.

8. **Reproducibility is incomplete.**  
   The reproducibility statement says code will be released upon acceptance, which is not sufficient for an ICLR submission aiming for strong reproducibility. Some parameters are listed, but several crucial details remain unclear: exact implementation choices, preprocessing specifics for each representation type, how tie-breaking is handled, and how the LLM experiments were operationalized.

### Novelty & Significance
**Novelty: moderate.** The core ingredients are largely recombinations of known ideas: persistent homology, MST-based 0D analysis, Spearman rank correlation, and the RTD/Max-RTD framework. The novelty lies in the particular symmetric construction for RTD and the NTS formulation as a normalized rank-based similarity over MST core pairs. This is a meaningful methodological extension, but not a fundamentally new paradigm.

**Significance: moderate.** If validated more thoroughly, the toolkit could be useful for representation analysis, especially in cases where CKA saturates or where topological structure matters. However, the current evidence is not strong enough to support a high-confidence claim of broad impact. Under ICLR standards, this looks like a potentially useful specialized contribution rather than a clear-cut breakthrough.

**Clarity: mixed.** The high-level motivation is clear, but the mathematical exposition is dense and often hard to follow. The paper would benefit from clearer definitions, cleaner theorem statements, and a more direct explanation of how each method should be used in practice.

**Reproducibility: fair but incomplete.** There is more detail than average in the appendices, but the lack of released code and the complexity of the constructions reduce confidence.

### Suggestions for Improvement
1. **Strengthen the theoretical exposition.**  
   Rewrite the definitions and theorems for SRTD and NTS more cleanly, with an explicit statement of assumptions, a more intuitive derivation of the auxiliary constructions, and clearer notation. Consider moving the most technical homological arguments to an appendix while providing a concise main-text explanation.

2. **Add more rigorous experimental validation.**  
   Include confidence intervals, multiple-seed statistics, and ablations showing sensitivity to preprocessing, sample size, layer choice, and tie-breaking. For the LLM experiments, report whether the qualitative patterns are stable across multiple random subsamples.

3. **Broaden the baseline suite.**  
   Compare against additional similarity methods beyond CKA, including at least one rank-based baseline and one topology-aware baseline if feasible. This would help isolate what is genuinely new about NTS.

4. **Clarify the role of normalization in NTS.**  
   Since Z-score normalization is stated to be important, explicitly separate the mathematical invariance properties of NTS from the practical preprocessing required in experiments. If the method is not fully scale invariant in practice, revise the claims accordingly.

5. **Provide a more convincing scalability story.**  
   Report runtime and memory on the same LLM-scale settings used in the similarity analysis, not only on TinyCNN. If the method is intended for large-scale representation analysis, practical feasibility should be demonstrated directly.

6. **Release code and exact experimental pipelines.**  
   For ICLR, code availability is important. The paper should specify data preprocessing, representation extraction, normalization, MST computation details, and all hyperparameters in a reproducible form.

7. **Tone down strong comparative language.**  
   Replace claims like “CKA fails” or “NTS is more trustworthy” with more measured statements unless supported by stronger evidence. A balanced empirical framing would be more credible and more consistent with the evidence shown.

8. **Add intuition for when each method should be used.**  
   A concise decision guide would help readers: when to use SRTD versus SRTD-lite versus NTS, what each metric is sensitive to, and what failure modes to expect. This would improve practical usability and clarity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the strongest existing representation-similarity baselines on the same tasks, not just CKA. For ICLR, the paper’s claims about “more discriminative” similarity need SVCCA/PWCCA, RSA, Geometry Score/IMD where applicable, and at least one modern topological baseline on the LLM and layer-analysis settings.

2. Add permutation / random-label controls for NTS and SRTD on all main benchmarks. Without showing scores collapse under representation shuffling, the claim that these methods detect meaningful structure rather than dataset or scale artifacts is not convincing.

3. Add ablations isolating the exact contributions of the proposed design choices: symmetric vs asymmetric, min/max construction, 0.9-quantile normalization, Z-score normalization, and core-pair selection. The paper currently asserts these are essential, but without ablations it is unclear which part actually drives the reported gains.

4. Add robustness tests across sample size, random subsampling, and representation preprocessing for the LLM experiments. The main claims rely heavily on heatmaps and score separation, but ICLR reviewers will expect stability analyses showing the conclusions do not depend on one layer choice, one dataset, or one sampling seed.

5. Add end-to-end optimization comparisons against RTD-lite and Max-RTD on the autoencoder task with matched compute budgets and multiple seeds. The claim that SRTD “matches top performance” is weak without learning curves, variance, and fair wall-clock-normalized comparisons.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze why NTS-E can require Z-score normalization if it is supposed to be scale-invariant. This is a critical internal inconsistency: the paper claims robustness to scale, but later admits normalization is “crucial,” which undermines the core similarity claim.

2. Quantify whether the “better discrimination” of NTS over CKA is actually meaningful in the LLM setting. The paper shows heatmaps, but it does not report retrieval accuracy, rank correlation with known family lineage, or any calibration metric to support the visual claims.

3. Provide a principled explanation of when NTS-M and NTS-E should be preferred and how often they disagree in practice. The theory states NTS-E is stricter, but the paper does not analyze disagreement rates or failure cases, so the method family’s practical meaning remains unclear.

4. Analyze the sensitivity of all topological scores to the choice of distance metric and feature normalization on representations. Because topological constructions depend directly on pairwise distances, ICLR readers will expect evidence that the conclusions are not an artifact of Euclidean distance or preprocessing.

5. Give a clearer empirical link between the claimed theoretical decomposition of SRTD and the observed experimental behavior. The paper states SRTD is a “shared symmetric component” plus private parts, but does not show that the private parts are small or explain when that decomposition fails.

### Visualizations & Case Studies
1. Add failure-case heatmaps where NTS and CKA disagree, with qualitative interpretation of which one is correct. The current figures are mostly success cases; without counterexamples, it is impossible to tell whether NTS is genuinely better or just differently biased.

2. Show barcode plots for representative success and failure pairs, including borderline cases. For a topology paper, the key question is whether the barcodes expose interpretable structure or are dominated by a few unstable long intervals.

3. Add a layer-by-layer trajectory plot for LLMs showing how NTS, SRTD-lite, and CKA evolve across depth for a fixed model family. This would reveal whether NTS is capturing coherent hierarchy or just producing isolated peaks.

4. Visualize rank-order agreement for NTS-M versus NTS-E on controlled perturbations. A simple plot of where merge-order and edge-order similarities diverge would expose whether the “stricter” variant is actually more informative or merely more sensitive.

### Obvious Next Steps
1. Add a unified benchmark table across all tasks with consistent metrics, seeds, confidence intervals, and runtime. The paper currently mixes qualitative heatmaps, partial tables, and different evaluation protocols, which makes the overall claim hard to trust.

2. Add a differentiable or approximate training objective for NTS, or else narrow the contribution to analysis-only use. If the paper’s toolkit is meant to be practical, it needs a path from the normalized similarity score to optimization, not just a future-work note.

3. Add a careful comparison to RSA and CKA under matched preprocessing and normalization. Since the paper repeatedly positions NTS as a replacement for geometric similarity, it needs a strict apples-to-apples study rather than separate plots on different pipelines.

4. Add a reproducibility appendix with exact model checkpoints, layer extraction details, normalization constants, and hyperparameters for every experiment. ICLR reviewers will not accept claims of broad robustness without enough detail to rerun the main results.

# Final Consolidated Review
## Summary
This paper proposes a small toolkit for topological representation analysis built around RTD. The first part introduces SRTD and SRTD-lite as symmetric variants of RTD/Max-RTD, and the second part introduces NTS, a rank-based normalized similarity score derived from MST core edges, with experiments on synthetic clustering, UMAP embeddings, autoencoders, CNN layers, and LLM representations.

## Strengths
- The paper identifies two real limitations of RTD-style measures — asymmetry and lack of normalization — and proposes targeted fixes rather than a vague rebranding. The SRTD/Max-RTD relationship is a genuine conceptual step beyond plain RTD, and the lite variants are practically relevant.
- The empirical coverage is reasonably broad for a representation-analysis paper: synthetic structure, optimization, CNN layer similarity, and LLM comparisons. In particular, the LLM section is timely and shows that the authors are trying to address a setting where CKA can saturate and become hard to interpret.
- The lite family appears mathematically cleaner than the full homological construction. The exact relation in the lite setting and the MST-based formulation make the lightweight part of the toolkit more compelling than the full SRTD machinery.

## Weaknesses
- The theoretical exposition is much weaker than the claims suggest, especially for SRTD. The mapping-cone / long-exact-sequence derivation is hard to verify as written, and the main theorem connecting RTD, Max-RTD, and SRTD is not presented with enough clarity to be convincing at ICLR standards. This matters because a large part of the paper’s novelty is claimed to be theoretical completion of the RTD framework.
- NTS is not as cleanly “scale-invariant” in practice as the paper’s framing suggests. The method relies on Z-score normalization in the LLM setting, and the paper itself notes this is crucial. That weakens the central normalization story and raises the possibility that the score depends materially on preprocessing choices. This matters because the paper positions NTS as a robust normalized similarity measure.
- The experimental evidence is suggestive but not decisive. Most of the strong claims rest on a small number of heatmaps and selected examples, with limited statistical testing, few robustness checks, and no serious ablation of core design choices such as min/max construction, quantile normalization, core-edge selection, or the NTS-M vs NTS-E choice. This matters because the paper is making broad claims about superiority over CKA and improved interpretability.

## Nice-to-Haves
- A clearer decision guide for when to use SRTD, SRTD-lite, NTS-M, or NTS-E, including their expected failure modes.
- More quantitative summaries for the LLM experiments, such as retrieval/ranking metrics or stability across subsamples, rather than mostly qualitative heatmaps.
- A small ablation on preprocessing, especially quantile normalization for SRTD and Z-scoring for NTS, since these steps appear operationally important.

## Novel Insights
The most interesting idea in the paper is that topological comparison can be split into two distinct tasks: a divergence family that preserves the RTD intuition but symmetrizes it through a min/max construction, and a separate similarity family that abandons barcode lengths in favor of rank agreement over MST-derived “core pairs.” That separation is sensible and may be practically useful, but the paper also reveals a tension: the more “normalized” and scalable NTS becomes, the more it depends on preprocessing and design choices that are not yet theoretically pinned down. In other words, the paper’s best idea is not just symmetry or normalization individually, but the attempt to unify both within one representation-analysis toolkit; the open question is whether the resulting methods are principled enough to be trusted outside the showcased settings.

## Potentially Missed Related Work
- SVCCA / PWCCA — relevant as standard representation similarity baselines that are not fully addressed in the main comparisons.
- RSA — relevant because the paper repeatedly contrasts topological methods with geometry-based similarity, and RSA is a canonical alternative.
- distance correlation / related rank-based similarity measures — relevant as a non-topological baseline for the NTS claims.
- other topology-aware comparison methods beyond RTD/RTD-lite — relevant for the stronger claims of superiority and discriminative power.

## Suggestions
- Tighten the theoretical section substantially: state assumptions explicitly, simplify the main theorem statements, and make the SRTD proof readable without reverse-engineering the auxiliary graph construction.
- Add ablations on quantile normalization, Z-score normalization, core-edge selection, and the NTS-M/NTS-E variants.
- Provide quantitative robustness results for the LLM experiments, including sensitivity to sampled pairs, layer choice, and random seeds.
- Broaden the baseline suite beyond CKA in the similarity experiments, especially for the LLM setting, and include at least one strong non-topological baseline and one additional rank-based baseline.
- If NTS truly requires preprocessing to work well, revise the claims so the method is presented as practically useful rather than fully scale-invariant in the strongest sense.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
