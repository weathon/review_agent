Now I have enough information to write the final review. Let me compile my assessment.

## Summary

The paper develops conformal confidence sets for biomedical image segmentation that provide probabilistic guarantees (rather than FDR-based guarantees) that the true mask is contained within inner and outer bounds. The key practical insight is that distance-transformed logit scores produce dramatically tighter outer confidence sets than raw logits, and Theorem 2.8 provides formal justification for why this works: accurate segmentation (small Hausdorff distance) translates directly into tight confidence bounds under distance transformations—a property untransformed logits lack.

## Strengths

- **Distance transformation as effective score function for outer confidence sets is a genuinely valuable practical insight.** The evidence is compelling: Figure 1 shows DT scores achieve far better separation of inside/outside distributions at the outer threshold than raw logits, and Figure 5 shows DT scores achieve diameter ratios closest to 1 for outer sets across all confidence levels. The paper explicitly demonstrates the failure mode of raw logits (Figure A20 for brain imaging) and shows DT addresses it convincingly.

- **Probabilistic mask containment guarantees are a clinically meaningful criterion.** Unlike prior conformal image segmentation work (Bates et al., 2021; Angelopoulos et al., 2024) that controls expected proportion of false negatives, Theorems 2.1 and 2.2 guarantee the true mask is entirely contained within/contains the confidence set with specified probability. The paper argues this is analogous to FWER vs FDR in multiple testing—a stricter but more appropriate criterion for medical settings (Section 1).

- **Theorem 2.8 provides principled justification for distance-transformed scores.** It establishes that when DT scores are used, segmentation accuracy (small Hausdorff distance ≤ k on sufficient calibration data) guarantees the outer set's Hausdorff distance from the prediction is ≤ k, and from the truth is ≤ 2k. The paper explicitly notes this relationship does not hold for untransformed logits, providing a clear theoretical reason to prefer DT scores.

- **Validated coverage across three diverse biomedical applications with 1000-run Monte Carlo validation.** Figure 4 shows coverage rates at or above nominal levels across all confidence levels for all score transformations on the polyps dataset, with 95% uncertainty bands from 1000 random calibration/test splits. Coverage validation is also reported for brain (Section A.7.4) and teeth (Section A.8.4).

- **Computationally efficient:** ~0.03 seconds (downscaled) or ~2.64 seconds (original) to compute calibration thresholds over 1000 images on a standard laptop (Section 6), making it practical for clinical use.

## Weaknesses

### Fatal
None.

### Major

- **Quantitative evaluation of set sizes is limited to one dataset; brain and teeth applications are evaluated primarily through visual examples.** For a paper whose central claim is "tight" confidence sets, the quantitative efficiency comparison (Figure 5, diameter ratios) is only provided for the polyps dataset. The brain section (Section 4) and teeth section (Section 5) contain one figure each with visual examples but no systematic quantitative comparison of set sizes across transformations, coverage levels, or methods. The paper references Figure A17 (proportion of image covered) but again only for polyps. Key quantities like average area of inner/outer sets or symmetric difference with ground truth are absent for brain and teeth. This makes it impossible to assess whether the "tight" claim generalizes across applications. The paper would be substantially stronger with fewer but more thoroughly evaluated applications.

- **The theoretical contribution is largely a straightforward application of split conformal inference.** Theorems 2.1, 2.2, and 2.6 follow directly from standard split conformal prediction once the nonconformity scores (max of transformed logits inside/outside the mask) are defined. The paper acknowledges this in Remark 2.4, noting inner/outer coverage "can also be viewed as a special case of conformal risk control." The only genuinely novel theoretical result is Theorem 2.8, which links Hausdorff distance to confidence set tightness—but it is conditional on model accuracy, limiting its practical value precisely when uncertainty quantification is most needed (when models are poor). The theoretical framework is clean and correct, but should not be presented as a central contribution.

### Minor

- **The "tight" claim is supported relative to known-weak baselines.** The only efficiency comparisons are against raw logits (which the paper itself shows produce uninformative outer sets) and bounding box methods (which the paper shows are conservative by construction, being a special case of their framework per Corollaries A.6/A.7). While these are the existing mask-level conformal approaches and comparing against them is fair, outperforming known-weak alternatives does not independently establish that the resulting sets are tight in any absolute sense. A comparison to a more competitive mask-level conformal approach would strengthen the efficiency claims.

- **The diameter ratio metric (Figure 5) conflates sets with different shapes.** A thin annular ring and a compact blob could have similar diameter ratios but very different areas relative to the ground truth, limiting what can be concluded from this efficiency measure.

- **Using training data as the learning dataset in the teeth application carries overfitting risk.** Section 5 states "We use the original training data as a learning dataset." The paper acknowledges this risk in Section 2.4 but the concern that transformation selection based on overfit data could systematically favor suboptimal transformations is not analyzed empirically (e.g., by comparing to an independent learning dataset split).

- **The bracket notation [(1-α)(n+1)] in equations (3), (5), and Theorem 2.6 is ambiguous.** It is unclear whether this denotes the floor or ceiling function. The standard conformal prediction literature uses ⌈(1-α)(n+1)⌉, and the choice affects finite-sample coverage. This should be explicitly defined.

### Trivial
None.

## Nice-to-Haves

- Analysis of failure cases: show what happens when the segmentation model misses a polyp entirely or makes a gross error—the method's value is most important in these cases.
- Sensitivity analysis of the learning dataset: how does performance degrade as the learning dataset shrinks, or how does the chosen transformation vary across random splits?
- Analysis of conditional (per-image) coverage rather than marginal coverage—marginal coverage could mask systematic over-coverage on easy images and under-coverage on hard ones, particularly relevant in medical settings.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Missing comparison to pixel-wise conformal prediction with Bonferroni/FWER correction or risk-control approach of Angelopoulos et al. (2024) at mask level."** The paper explicitly argues (Section 1) that Angelopoulos et al. (2024) controls expected false negative rate, which is a fundamentally different (weaker) guarantee than mask containment in probability. Comparing methods with different coverage criteria would be apples-to-oranges. The FDR vs FWER distinction the paper draws is central to its contribution.

- **"Bounding box comparison is against a method already known to be conservative by construction."** The paper actually establishes bounding boxes as a special case of its own framework (Corollaries A.6, A.7) using chessboard distance transformation. This makes the comparison meaningful within the same theoretical framework and shows the advantage of using masks directly rather than boxes.

- **"Coverage estimates in Figure 4 are not independent across runs because images are shared across splits."** The paper acknowledges this doesn't affect the validity of the coverage guarantees. The non-independence only affects the precision of the uncertainty bands, which is a minor statistical refinement.

- **"Sections 4 and 5 are remarkably thin."** While these sections are brief in the main text, they reference detailed appendix sections (A.7 and A.8). The parser strips appendices, but they exist in the original submission. The concern about thin evaluation is valid but has been captured under the Major weakness about quantitative evaluation.

- **"The learning dataset of 298 images is used to both choose transformations AND calibrate thresholds in Section 3.1."** The paper clearly states (line 167) that Section 3.1 uses the learning dataset for both purposes "in order to maximize the amount of important information we can learn from it," and that Section 3.2 uses separate calibration data. The visual evidence in Figures 1-2 is on the learning dataset for illustration; the actual inference uses independent calibration data.

## Novel Insights

The paper makes a useful distinction between FDR-type and FWER-type guarantees for image segmentation uncertainty quantification, drawing an analogy to the multiple testing literature. This framing helps explain why mask containment in probability is a stronger but more appropriate criterion for clinical settings than controlling expected false negative rates. Theorem 2.8's result—that DT scores convert segmentation accuracy into confidence set tightness in a way raw logits do not—is the most insightful contribution, providing a principled explanation for an empirical observation that might otherwise seem like a trick.

## Suggestions

- Add quantitative set-size analysis (area ratios, symmetric differences with ground truth) for all three applications, not just diameter ratios for polyps. This would substantially strengthen the evidence for the "tight" claim.
- Consider condensing from three to two applications with deeper evaluation rather than three with thin evaluation. Polyps and brain show contrasting behavior (logits work for inner sets in polyps but fail for brain), making them the most informative pair.
- Reframe the theoretical contribution honestly: the framework is a clean application of split conformal inference, and the main contribution is the practical insight about distance transformations supported by Theorem 2.8. This would set more appropriate expectations.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** Angelopoulos et al. conformal risk control (33XGfHLtZg, avg 7.0) — genuinely extended the theoretical framework of conformal prediction with broad applicability; this paper's theoretical contribution is more incremental. ValUES framework (yV6fD7LYkF, avg 7.5) — comprehensive evaluation framework with systematic ablations; this paper's evaluation is less thorough.

- **Medium band (4-6):** Conformal bounds on image quality for MRI (8VXWQmNrca, avg 5.0) — similar profile of applying split conformal with limited novelty, was rejected; this paper has a clearer practical insight and better validation. Hierarchical CP for 3D segmentation (sgaMYvGRG5, avg 5.8) — more comprehensive empirical evaluation but similar novelty concerns; this paper is comparable. Class-conditional CP (Dtxc7mlKRg, avg 4.6) — limited evaluation; this paper is stronger.

- **Low band (<3):** Regression CP under bias (v8RDgaEtE2, avg 2.5) — fundamentally flawed theoretical claims and missing coverage probability in experiments; this paper is far superior. SEMF prediction intervals (8GFoOB7XB4, avg 3.0) — weak baselines and unconvincing results; this paper is clearly stronger.

This paper sits solidly in the medium band. It has a genuine practical insight (distance transformations for tight outer confidence sets) with clean but not novel theory. The evaluation is convincing for one application but thin for the other two. Compared to the 5.0-scoring MRI paper, this paper is somewhat stronger due to its clearer insight and more diverse validation; compared to the 5.8-scoring hierarchical CP paper, it is slightly weaker in evaluation breadth. I place it at 5.5.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>