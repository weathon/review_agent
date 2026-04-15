## Summary
This paper applies Bayesian NMF to fMRI responses from ventral, lateral, and dorsal visual pathways to extract sparse components, then introduces Sparse Component Alignment (SCA), an axis-sensitive similarity measure intended to complement RSA and linear encoding. The main empirical claim is that while conventional alignment metrics suggest broadly similar brain–model correspondence across pathways, SCA reveals substantially stronger alignment between standard image-trained DNNs and ventral cortex than with lateral or dorsal cortex.

## Strengths
- **The paper tackles a specific and important inconsistency in prior brain–model alignment results:** namely, how object-trained DNNs can appear to fit ventral, lateral, and dorsal pathways similarly well despite long-standing evidence that these pathways support different functions. This motivation is clearly articulated in Sections 1 and 4 and the experiments are organized around resolving exactly this contradiction.
- **SCA is a genuinely novel and conceptually interesting metric:** unlike RSA and linear encoding, it is explicitly sensitive to the axes of a representation rather than only population geometry or linear readout performance. The rotation simulations in Figure 2 clearly demonstrate the narrow mathematical point the paper wants to make: RSA is insensitive to these rotations while SCA is not.
- **The component analysis yields interpretable stream-specific structure, especially in ventral and lateral cortex:** ventral components recover familiar selectivities (faces, scenes, bodies, food, text), while the lateral-stream decomposition into group interactions, implied motion, hand actions, scenes, and reachspaces is a concrete and potentially useful empirical finding.
- **The paper does more than present a new metric; it links decomposition, alignment, and behavioral structure:** the Meadows behavioral analysis provides at least some external evidence that the NMF-derived connectivity matrices are not arbitrary and preserve behaviorally relevant information.
- **The empirical pattern under SCA is coherent across models and layers:** the ventral advantage is not a one-off result from a single model, and the layerwise AlexNet analysis is consistent with stronger ventral correspondence in later layers.

## Weaknesses

###: Fatal
- **The central claim depends on SCA being a valid measure of “truer” brain–model alignment, but the paper does not validate that strongly enough.**  
  The headline conclusion—standard vision DNNs are substantially more aligned with ventral than lateral/dorsal cortex—appears most clearly only under SCA. The paper shows that SCA is *different* from RSA/encoding and sensitive to axis rotations in synthetic settings, but that is not enough to establish that it is *better* or more faithful on real neural data. In the real-data setting, SCA combines multiple assumptions at once: Bayesian NMF, non-negativity, sparsity, finite rank, consensus across stochastic runs, and then a hard top-component co-assignment rule. The paper does not sufficiently disentangle whether the ventral-specific effect is due to axis sensitivity per se or to this full package of inductive biases. Since the main conclusion hinges on SCA, this is a fundamental evidential weakness.

### Major:
- **The paper overstates what is “hypothesis-free,” especially regarding the three-way pathway dissociation.**  
  The decomposition within each stream is unsupervised, but the three pathways are analyzed *after restricting the data to predefined anatomical masks* (“We extracted the most consistent components ... separately in the dorsal, ventral, and lateral stream”; Fig. 4d shows the masks explicitly). So the claim that the three-way dissociation emerges “free of any a priori ... spatial hypotheses” is too strong. A more accurate claim would be that, *within anatomically defined pathways*, an unsupervised sparse decomposition reveals distinct dominant components.
- **The explanation for why RSA/encoding “miss” the pathway differences is not isolated convincingly enough.**  
  The paper argues that conventional methods fail because of rotational invariance. Figure 2 supports that in simulations involving controlled rotations. But in the real analysis, SCA differs from RSA in many ways besides rotation sensitivity: sparse NMF preprocessing, hard winner-take-all assignment to the top component, and binary connectivity construction. The paper partially acknowledges this by including CMS and noting that “the same dominant components—following a strict 1-1 mapping as used in CMS—show only a modest similarity across all streams,” but this actually underscores that the mechanistic interpretation is not yet pinned down.
- **The strongest mechanistic interpretation—shared axes of tuning between DNNs and ventral neurons—is not fully supported by the presented evidence.**  
  The paper concludes: “We thus conclude that DNNs share similar axes of neural tuning as neurons in the ventral visual stream.” That is too strong relative to the analyses shown. SCA measures similarity between image-level co-assignment patterns after separate decompositions; it is not direct evidence of matched axes or one-to-one component correspondence across brain and model. Indeed, the paper’s more direct component-matching control (CMS) is described as showing only “modest similarity across all streams,” which weakens the stronger tuning-axis claim.
- **The critical alignment comparisons are under-supported statistically in the main text.**  
  The paper reports mean correlations such as ventral SCA \(r=0.187\) versus lateral \(r=0.047\) and dorsal \(r=0.058\), but does not report uncertainty, confidence intervals, or formal tests in the main paper text provided here. Because the entire contribution rests on differences between metrics and across streams, this missing statistical characterization matters. The problem is not that every benchmark needs exhaustive significance testing, but that here the paper is asking readers to accept a new metric based on relatively small absolute differences and a small number of subjects.
- **The data regime likely favors ventral-style semantics more than dorsal/lateral computations, which weakens the interpretation of low lateral/dorsal SCA.**  
  The authors themselves note in Section 4.1 that “the methods we used here do not fully capture the representations and computations of the dorsal and lateral streams, which would require collecting neural responses to a wider variety of stimuli and tasks.” That is an important and appropriate caveat, but it substantially limits the force of the conclusion. Weak dorsal/lateral alignment may reflect a combination of model mismatch and stimulus/task mismatch; the current experiments do not clearly separate these.
- **The hard winner-take-all design of SCA is a substantial modeling choice that is not sufficiently justified or ablated.**  
  Equation 2 defines image similarity entirely by whether two stimuli “maximally load onto the same component.” This throws away the magnitude structure of component responses and all sub-dominant components. A stimulus with nearly tied responses on two components is treated the same as one overwhelmingly dominated by one component. This may be a useful simplification, but the paper does not demonstrate that this discretization is essential or biologically better motivated than softer alternatives.

### Minor
- **The component-interpretation pipeline is less rigorous than the “hypothesis-free” framing suggests.**  
  The decomposition is unsupervised, but the semantic labels are assigned after inspecting response profiles, and then behavioral saliency ratings are used to quantify those post hoc interpretations. This does not invalidate the component analysis, and the paper is transparent enough about examining top images and then correlating saliency ratings, but it means the semantic claims should be read as plausible interpretations rather than fully independent confirmation.
- **The Bayesian NMF superiority argument is somewhat overstated on real data.**  
  The simulations in Section 3.1 are supportive but closely match the assumptions of the method (sparse nonnegative latent factors). Figure 3 shows a sparsity/variance tradeoff, but does not establish that Bayesian NMF yields *truer* or more biologically meaningful real-data components than alternative sparse decompositions. The paper does mention robustness over \(C=10\) to \(30\), which helps, but key robustness analyses remain outside the main text.
- **The dorsal results are visibly weaker and less interpretable than the ventral/lateral findings.**  
  The abstract already hedges this (“some less interpretable components in the dorsal stream”), which is fair. But the overall framing still occasionally reads as if all three streams are characterized with equal clarity; the actual evidence is strongest for ventral, reasonably suggestive for lateral, and much thinner for dorsal.
- **Cross-model/layer selection details are not fully clear in the main text.**  
  For encoding, the feature extraction rule differs by architecture (“ultimate,” “penultimate,” or “best performing attention head”), and the main text does not fully standardize how headline cross-model comparisons are constructed. This is not a fatal comparison issue, but the paper would benefit from a cleaner and more uniform presentation.

### Trivial
- **The claim that negative response magnitudes are biologically inconsistent is too absolute.**  
  As a rationale for non-negativity, this is somewhat overstated for preprocessed fMRI response matrices. The broader case for non-negativity as an interpretability prior is still reasonable.

## Nice-to-Haves
- Test whether the main SCA conclusions hold under a **soft-assignment** version of the connectivity matrix rather than only top-1 component assignment.
- Bring the robustness to **different component counts** into the main paper, since factor granularity could affect both interpretability and SCA.
- Provide **subject-level uncertainty / reliability / noise-ceiling analyses** for the alignment scores.
- Include **more direct visualizations of brain/model image connectivity matrices** and examples where SCA and RSA disagree most.
- If feasible, test models better matched to dorsal/lateral hypotheses (e.g., video- or action-oriented models), though this is more of a strong extension than a strict requirement for the current scoped contribution.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Only 4 subjects are analyzed, therefore the paper is too weak.”**  
  Kept only in weakened form elsewhere. Four NSD subjects is a real limitation for generalization, but NSD is a deep per-subject dataset and the paper does not hide the sample size. On its own, this is not a decisive flaw.
- **Complaints about incomplete release/reproducibility or missing implementation minutiae (e.g., full hyperparameter settings, code release).**  
  Removed per instructions. These are not central scientific weaknesses here.
- **Pure formatting/algorithm pseudocode issues.**  
  The extracted text contains parser artifacts, so I do not treat minor notation/pseudocode inconsistencies as substantive flaws.
- **Claims that the model comparison is unfair because asymmetries favor baselines.**  
  Removed where applicable under the asymmetric-comparison rule.
- **Requests for unrelated additional literature.**  
  Omitted per instruction.

## Novel Insights
The most interesting synthesis is that the paper is strongest not as a definitive overturning of prior alignment results, but as a demonstration that **the notion of alignment itself is underdetermined by the metric**. The work usefully shows that population-geometry alignment and axis-sensitive component alignment can lead to materially different scientific conclusions. That is a genuine contribution. However, the current evidence supports this as an important *methodological warning* more convincingly than it supports the stronger claim that standard image DNNs truly share ventral-like tuning axes while failing to align with dorsal/lateral cortex.

## Suggestions
- **Reframe the main claim more conservatively.** Replace “DNNs share similar axes of neural tuning as neurons in the ventral visual stream” with something like: *“Under an axis-sensitive component-based similarity measure, standard vision DNNs show substantially stronger alignment with ventral than with lateral or dorsal cortex.”*
- **Explicitly narrow the hypothesis-free claim.** State that the decomposition is unsupervised *within anatomically defined streams*, rather than implying discovery of the stream partition itself.
- **Add ablations that isolate what in SCA matters.** In particular: soft vs hard assignment, continuous vs binary connectivity, and comparisons using other decompositions or matched-sparsity controls.
- **Report uncertainty for the key alignment differences** and, if possible, reliability-adjusted ceilings.
- **Strengthen the discussion of scope:** make clearer that the current evidence is strongest for ventral alignment under static natural-image conditions, and weaker claims should be made for dorsal/lateral conclusions.

## Score and Decision
**Novelty:** good. SCA is a real new idea and the decomposition/alignment story is cohesive.  
**Technical soundness:** mixed. The method is interesting and mathematically motivated, but the central interpretation outruns what has been validated.  
**Empirical support:** moderate but not yet sufficient for the strongest claims. The ventral/lateral/dorsal pattern under SCA is intriguing, but key controls and uncertainty estimates are missing from the main case.  
**Significance:** potentially high if the metric is validated more strongly, because it challenges how alignment is measured in neuroscience. In the current form, significance is more methodological/suggestive than definitive.  
**Clarity:** strong overall, aside from some overclaiming.

Relative to comparable ICLR papers on brain–model alignment and representational metrics, this lands above weak submissions with diffuse claims, but below the acceptance bar because the new metric is not yet validated strongly enough to support the paper’s strongest conclusions.

**Score: 5.6**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.6</pineapple>
MY FINAL DECISION: <orange>Reject</orange>