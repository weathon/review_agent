Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper identifies that VLMs disproportionately rely on spurious attributes — attributes co-occurring with categories but not intrinsic to them (e.g., "road" for "vehicle") — and that these attributes harm OOD generalization despite constituting <7% of the attribute pool. The authors propose two complementary methods: **SAP** (Spurious Attribute Probing), which uses MLLMs and CBMs to automatically identify and filter spurious attributes from the language branch, and **SAS** (Spurious Attribute Shielding), a plug-and-play subsidiary task module that creates pseudo-categories from spurious attributes to mitigate their influence on the vision branch across arbitrary PEFT methods. Experiments across 11 datasets and 3 generalization tasks show consistent OOD improvements.

## Strengths

- **Compelling problem identification with quantitative evidence.** Table 1 demonstrates that manually removing spurious attributes constituting <7% of the pool improves new-category accuracy from 65.30%→67.66% (CPL) and 66.07%→67.69% (ArGue), while Figure 1's CBM weight analysis shows 2 of the top-3 attributes influencing predictions are spurious (e.g., "sea"/"lake" for fireboat). This is a real and underexplored phenomenon.

- **Broad and consistent empirical improvements.** Figure 3 shows gains across 11 datasets, 3 generalization tasks, and 11 PEFT baselines. The plug-and-play nature of both SAP and SAS is demonstrated empirically, not just claimed, with average OOD improvements exceeding 2%.

- **Counter-group evaluation provides strong targeted evidence.** Table 2 constructs adversarial test subsets by filtering images that contain spurious attributes, retaining only images where spurious shortcuts are unavailable. SAS improves counter-group accuracy by up to ~6% (CPL on FGVCAircraft: 27.65→32.12) while standard test improvements are only ~1-2%, showing an asymmetric benefit consistent with spurious-attribute-specific mitigation rather than generic regularization.

- **Clean conceptual decomposition.** SAP addresses the language branch (attribute pool purity) while SAS addresses the vision branch (learned spurious features). This separation is well-motivated and allows each method to complement different types of existing approaches.

- **Ablation studies validate key design choices.** Table 4 shows that both too-high γ (missing spurious attributes) and too-low γ (false positives) hurt performance, confirming that identified spurious attributes — not just additional data — drive SAS's gains. The adaptive threshold (γ_c = minimum core attribute weight) outperforms all fixed thresholds (HM 80.38 vs. best fixed 79.81 at γ=0.4).

## Weaknesses

### Fatal
None.

### Major

- **SAS's claimed mechanism cannot be fully distinguished from regularization without a random-pseudo-categories control.** While Table 4 shows that noisy attributes (low γ) hurt performance and Table 2 shows asymmetric counter-group improvements, the most direct control experiment is missing: construct pseudo-categories from non-spurious attributes (e.g., core attributes or random vocabulary words) and apply the same subsidiary task. If similar gains appear, the mechanism is generic regularization, not spurious-attribute-specific shielding. The paper acknowledges this concern ("A natural concern is whether the model's gains are due to the introduction of additional data rather than an increase in robustness," Section 4.2) but addresses it only indirectly via the γ ablation. The counter-group results provide strong circumstantial evidence but cannot fully substitute for this control — this is the single most important missing experiment. That said, the asymmetric counter-group improvement (Table 2: ~6% on counter-group vs. ~1-2% on standard test) is harder to explain via generic regularization, which would be expected to help uniformly.

- **SAP's identification quality is never validated against the manual ground truth.** The paper establishes manual identification of spurious attributes in Section 3.2 as the foundational motivation (Table 1), yet never compares SAP's automated output against this manual ground truth. Reporting precision/recall of SAP relative to the manual labels would directly validate whether SAP identifies the right attributes. Without this, we cannot assess whether SAP works for the right reasons, even though end-to-end improvements suggest it is partially effective. Given that the manual labels already exist (they were used to produce Table 1), this comparison would be straightforward to include.

### Minor

- **The manual identification process in Section 3.2 lacks inter-annotator agreement.** The process — "we randomly sample 5 images from the shots and visualize the heatmap... we determine whether it is a part of the main object, or separate objects in the background" — is inherently subjective (e.g., is "water" spurious for "fireboat"?). No inter-annotator agreement is reported, and the boundary between core and spurious is philosophically ambiguous. However, this process is used only for motivation (Table 1), and the general phenomenon of spurious attributes is well-established in prior work (Singla & Feizi, 2021).

- **No standard deviations reported despite averaging over three runs** (line 234). For improvements of ~1-2% on some tasks, variance information would help assess significance. This is standard practice in the field but would strengthen the empirical claims.

### Trivial
None.

## Nice-to-Haves

- **Interventional analysis of spurious attribute reliance.** Rather than CBM weight analysis (correlational), perturbing or removing spurious attributes in test images (e.g., inpainting out "road" from "vehicle" images) and measuring prediction changes with/without SAS would provide causal evidence for the mechanism.

- **SAP output examples.** Showing the full attribute pool before/after SAP for several categories, with each attribute labeled as core/spurious by both manual annotation and SAP, would make identification quality directly assessable by readers.

- **Error analysis of SAP.** What kinds of attributes does SAP systematically misclassify? Are there context-dependent attributes that are core in some categories but spurious in others?

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "Counter-group evaluation is circular."** The critic claims Table 2's evaluation is circular because it filters test images based on the same spurious attributes SAP identified. This is incorrect — this is standard adversarial evaluation methodology for debiasing methods. Testing on samples where the spurious shortcut is unavailable is exactly what one should do, and the asymmetric improvement (much larger on counter-group than standard test) provides evidence *against* the generic regularization alternative the critic raises in their first concern.

- **Harsh Critic: "SD vs. LAION comparison is missing."** The paper explicitly states in the implementation details (Section 4) that "the comparison between pseudo category construction with synthesized and pre-training data [is] provided in Supp. Mat. B." This content exists in the original submission's appendix.

- **Harsh Critic: "CBM weight analysis is correlational, not causal."** While true in isolation for Figure 1, the paper provides multiple other forms of evidence (counter-group evaluation in Table 2, saliency maps in Figure 5, γ ablation in Table 4) that collectively support the mechanistic claim. The CBM analysis is one piece of a broader evidential picture.

- **Harsh Critic: "ArGue* variant undermines the claim that LLM prompting can address the problem."** The paper presents ArGue* as evidence that simple LLM prompt modifications are insufficient, which *motivates* the need for SAP/SAS. This is not a weakness — it strengthens the motivation.

- **Strength Finder: "Plug-and-play design enables easy adoption"** as a separate strength. This is generic and already captured by the empirical demonstration across 11 baselines.

## Novel Insights

The paper's most insightful contribution is the "black sheep" asymmetry: spurious attributes constitute <7% of the pool but occupy 2 of the top-3 positions in CBM weight rankings. This small-fraction-disproportionate-influence pattern is more subtle than standard spurious correlation settings (where the spurious feature is the majority signal), and suggests that VLMs' attribute reliance may be a distinct phenomenon from the group-shift spurious correlations studied in the broader debiasing literature. The counter-group evaluation's asymmetric improvement pattern (~6% on hard subset vs. ~1-2% on standard) is itself a useful diagnostic that future work on VLM debiasing could adopt.

## Suggestions

- Add a random-pseudo-categories control experiment: construct pseudo-categories from core attributes (or random vocabulary words) using the same SD/LAION pipeline and train with the same subsidiary loss. Compare against SAS to directly test whether the spurious-attribute-specificity claim holds.

- Compare SAP's automated spurious attribute identification against the manual labels from Section 3.2 (which already exist), reporting precision and recall. This is the most direct way to validate SAP.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Two Effects, One Trigger (VLM attribute bias) | /home/wg25r/review_agent/human_reviews/uAFHCZRmXk.md | 8.0 | More rigorous controlled experiments and clearer mechanistic validation; this paper is weaker on mechanism |
| Interpreting CLIP via Text Decomposition | /home/wg25r/review_agent/human_reviews/5Ca9sSzuDp.md | 8.0 | Deeper analysis with cleaner methodology; this paper has broader but shallower evaluation |
| MetaCoCo (few-shot spurious correlation benchmark) | /home/wg25r/review_agent/human_reviews/DiWRG9JTWZ.md | 7.0 | Benchmark contribution; this paper is more methodological but comparable empirical scope |
| Cross-modal Spurious Correlation Mitigation | /home/wg25r/review_agent/human_reviews/BlzBcWYmdB.md | 5.0 | Very similar topic and goal; this paper is clearly stronger with better motivation, much broader experiments, and two complementary methods |
| Unclipping CLIP's Wings (spurious correlations in CLIP) | /home/wg25r/review_agent/human_reviews/DPp5GSohht.md | 4.25 | Similar domain; this paper has more extensive evaluation and a more complete methodological contribution |
| Stabilized EGNN (mechanism vs. regularization confound) | /home/wg25r/review_agent/human_reviews/NeWiiF6KLB.md | 4.67 | Shares the "mechanism unvalidated vs. generic regularization" concern; this paper has stronger counter-evidence (Table 2 counter-group) |
| TCIG (overclaimed, missing baselines) | /home/wg25r/review_agent/human_reviews/RFJGFrMvYj.md | 1.5 | Genuinely weak paper; this paper is far stronger with real phenomenon and extensive experiments |

**Score reasoning:** This paper sits above the medium-scoring VLM spurious correlation papers (4.0-5.0 range) because of its significantly broader experimental coverage (11 datasets, 3 tasks, 11 baselines), better motivation through the "black sheep" phenomenon, and stronger evidence through counter-group evaluation. It sits below the high-scoring analysis papers (7.0-8.0 range) because of the two Major weaknesses: the missing random-pseudo-categories control and the lack of direct SAP validation. The counter-group results provide meaningful evidence against the generic regularization alternative, but the control experiment would make this conclusive. The paper is at the upper end of borderline — it makes a real contribution that the community would benefit from, but the mechanistic claims need stronger validation.

## Score and Decision

**Originality:** The "black sheep" phenomenon identification is novel and well-motivated; SAP and SAS are competent but not groundbreaking individually. The combination is well-designed.

**Importance:** The problem is important — spurious attributes in VLMs harm generalization, and this is increasingly relevant as attribute-based methods proliferate.

**Claims support:** Empirical claims are well-supported by extensive experiments. Mechanistic claims about *why* SAS works are partially supported but need the random-pseudo-categories control.

**Experimental soundness:** Broad and well-structured, with useful ablations and the counter-group evaluation. Missing the most informative control experiment.

**Clarity:** Well-written with clear problem motivation and method description.

**Community value:** The phenomenon identification and the two-method framework (SAP+SAS) will be useful to the PEFT/VLM community.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>