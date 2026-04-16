Good, I've read the full paper. Now let me search for calibration papers in the human review database.Now I have enough to write the comprehensive final review. Let me synthesize everything.

---

## Summary

This paper proposes "Accuracy per Batch" (ApB), a simple single-model proxy for identifying memorized training points, and incorporates it into a revised memorization scoring pipeline called pLOO_improved. The central empirical observations are: (1) memorized points are learned later in training than generalized points; (2) ApB (fraction of training batches a point is correctly classified) correlates strongly with pLOO memorization scores (Pearson < -0.95) across multiple architectures and datasets; (3) restricting pLOO to the top-5,000 lowest-ApB points reduces shadow model requirements by 90% and, in a narrow LOO comparison on 160 points using VGG-6/CIFAR-10, reduces error against the LOO gold standard by 65%.

---

## Strengths

- **Real and important problem**: pLOO-style memorization estimation is genuinely compute-intensive, and reducing that cost matters for the ML privacy community.

- **Hypothesis validation is broad**: The claim that generalized points are learned earlier is empirically validated in Figure 2 across VGG19, ResNet18, MobileNet on CIFAR-10, CIFAR-100, and Tiny ImageNet. This cross-model, cross-dataset consistency is noteworthy.

- **ApB correlation is strikingly consistent**: Pearson scores below -0.95 across all five model-dataset combinations in Figure 3 is a strong empirical result for such a simple metric (4 extra lines of code). This is the paper's most solid contribution.

- **Computational savings are well-documented**: The reduction from 2,000 to ~200 shadow models under the restricted-subset regime is clearly derived and consistent with prior shard-count heuristics.

- **The paper correctly identifies a real flaw in pLOO**: Showing that pLOO overestimates memorization scores relative to the LOO gold standard (RMSE 35.5) is a genuine and underappreciated finding, even if the validation is narrow.

- **The proposed mechanism for pLOO inaccuracy is coherent**: The paper's explanation—pLOO drops 15,000 points per shard while pLOO_improved drops only ~1,500—is intuitive and testable (Section 6).

---

## Weaknesses

### Fatal
*None identified.*

### Major

**W1. The "65% more accurate than pLOO" headline claim rests on a single tiny experiment.**
The LOO comparison (Part 2, Section 5.2–5.3, Figure 5) is conducted on only 160 points, using a VGG-6 model (not one of the three main architectures evaluated throughout the paper), on CIFAR-10 only, and crucially these 160 points were *selected specifically* as those with the largest disagreement between pLOO and pLOO_improved. As the paper states: *"we run it over 150 points that had the largest difference in memorization scores between the original pLOO and pLOO_improved."* This sampling strategy selects points where the two methods already diverge maximally—it is not a representative sample for estimating overall RMSE. The resulting RMSE comparison (35.5 vs. 12.19) cannot be generalized to other architectures or datasets. The paper defends this by asserting: *"pLOO and LOO are model-independent methods... Therefore, we have no reason to believe that our findings will not extend to other models."* This is not evidence; it is an argument from assumption. The paper's main accuracy claim—arguably half its title—is thus supported by a single narrow experiment.

**W2. The ApB proxy is validated primarily against pLOO, the very quantity it is meant to replace.**
Figure 3 shows strong Pearson correlation between ApB and *pLOO-derived* memorization scores—not LOO scores. Since the paper later shows pLOO has RMSE 35.5 against LOO, a strong ApB-to-pLOO correlation does not establish that ApB tracks true memorization. At best it shows ApB replicates pLOO's approximation, including its biases. The narrowness of the LOO comparison means this circularity is never broken for the proxy itself: ApB is never validated against LOO ground truth. This weakens confidence that the top-5,000 ApB-selected points actually contain the truly memorized points (as opposed to the pLOO-estimated memorized points).

**W3. The method does not compute the same thing as pLOO: it scores only a preselected top-k subset.**
The paper frames pLOO_improved as a faster, more accurate *replacement* for pLOO, but it only assigns memorization scores to the top 5,000 lowest-ApB points. Points outside this set never receive scores. The footnote acknowledges this: *"a user might choose more or fewer points based on their ML task and the computational resources."* But no recall analysis is given—how many truly memorized points (per LOO) fall outside the top 5,000 selected by ApB? The efficiency gain is partly achieved by simply running a narrower task, not by accelerating the same task. This distinction matters for downstream applications like membership inference, where missed memorized points are safety-relevant.

**W4. The r=0.5 vs. r=0.7 change conflates two sources of improvement.**
The original pLOO uses r=0.7 (Section 3.3); pLOO_improved uses r=0.5 (Section 5.2). The paper identifies the mechanism of improvement as *dropping fewer points per shard* (Section 6)—but changing r from 0.7 to 0.5 by itself drops *more* points per shard (50% vs. 30%), which should worsen accuracy by the paper's own logic. The accuracy gain thus comes entirely from the reduced search space (5,000 vs. 50,000 points). There is no ablation to separate the effect of the restricted candidate set from the effect of the changed sampling ratio. This means the mechanism attributed to pLOO_improved's accuracy gain is not cleanly isolated.

### Minor

**W5. The scalability motivation is unsubstantiated.**
The abstract claims the method "makes it possible to study memorization in large datasets and real-world models." Yet all experiments are on CIFAR-10/100 and Tiny ImageNet with small models. No experiment on ImageNet-scale data or a transformer architecture is provided. The gap between motivation and demonstrated scope is significant.

**W6. No retrieval-style evaluation of the proxy.**
The paper's practical use of ApB is to *rank* points and select a top-k subset—yet Figure 3 only reports Pearson correlation, which measures linear association across the full distribution and can be dominated by the large bulk of easy (low-memorization, high-ApB) points. Spearman rank correlation or precision-at-k would directly measure whether the proxy correctly recovers the top-memorized points. This omission is significant because false negatives (truly memorized points excluded by ApB) are never quantified.

**W7. Sensitivity to the top-k threshold is unexplored.**
The choice of 5,000 points is stated to be "adequate for our evaluation" (Footnote 1) but no sensitivity analysis is provided. How does the RMSE and recall change as k varies from 1,000 to 10,000? This is a critical hyperparameter for practitioners.

### Trivial

- The "erodes the trust in the original pLOO method" (Section 5.3) is an overclaim given the narrow LOO validation performed.
- The claim that a "point with ~100% memorization score belongs to a sub-population of size one" (Section 3.1) is an ontological overreach from Equation 1 alone, though it is consistent with prior work's intuition.

---

## Nice-to-Haves

- Run the LOO comparison on at least one second architecture (e.g., ResNet18) to strengthen the accuracy claim, even if only on a small sample.
- Add ablation: compare pLOO at r=0.95 (dropping only ~5% of points per shard) against pLOO_improved to isolate whether improvement comes from reduced search space vs. altered sampling ratio.
- Compare ApB against simpler baselines such as final-epoch training loss or prediction confidence; if ApB uniquely outperforms, the design choice is more motivated.
- Report Spearman or precision-at-k alongside Pearson for Figure 3.
- Sensitivity analysis over the top-k threshold (1k, 2.5k, 5k, 10k).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Human Finder Reviewer W4 (no variance / error bars):** The paper repeatedly reports results averaged over 50 runs (Section 4.2, 4.3) for the hypothesis and proxy validation. The concern about missing error bars is partially addressed. Removed as a standalone weakness.
- **Human Finder Reviewer W5 (natural vs. artificial memorization):** The paper explicitly acknowledges and studies *natural* memorization throughout, distinguishing it from artificial memorization (Section 2: *"it has yet to be verified for natural points... part of our contribution is to show that natural points behave in the same manner"*). The concern that ApB might behave differently on artificial memorization is out of scope for this paper's stated goals.
- **Harsh Critic note on Section 3.1 sub-population interpretation:** This is a minor framing issue consistent with Feldman (2020)'s own language. Not a paper-level problem.
- **Harsh Critic note on "strict criterion" for Figure 2 (all 50 models):** The paper uses this criterion to address catastrophic forgetting (Section 4.2), which is a reasonable and explicitly justified methodological choice. Not a weakness.

---

## Novel Insights

The paper's most underappreciated contribution is the *mechanistic diagnosis* of pLOO's inaccuracy: by dropping tens of thousands of points per shard, pLOO creates a distribution shift that artificially inflates memorization scores. Reducing the sampling space to the likely-memorized subset incidentally brings the shard composition closer to the LOO ideal (dropping only ~1,500 vs. 15,000 points). This insight—that the approximate procedure's accuracy is governed by how many points are dropped per shard, not just shard count—is potentially generalizable to other shadow-model-based privacy auditing methods beyond memorization scoring.

---

## Suggestions

1. **Critical priority**: Run even a partial LOO comparison (50–100 points) on one of the main architectures (ResNet18/VGG19) to escape the single-model validation bottleneck. Even a second data point would substantially change the credibility of the generalization claim.

2. **Ablate r**: Run pLOO with r=0.95 on the full dataset to test whether the accuracy gain is purely from fewer dropped points (regardless of search space size). This is the paper's own proposed mechanism and is cheap to test on CIFAR-10.

3. **Recall analysis**: Use the 160-point LOO sample to check how many of the LOO-confirmed memorized points were in the top-5,000 ApB set. Even one number here would address the silent-exclusion concern.

4. **Pearson → Spearman + precision@k**: Report rank correlation and top-k overlap in Figure 3 to demonstrate that ApB correctly *ranks* the most memorized points, not just that it is linearly associated with pLOO scores.

---

## Score and Decision

**Calibration:**

- **lTh7DEJV5W** (Memorization and Orders of Loss, Reject, 3/3/3/8): Most directly comparable. Also proposes a simple training-dynamics proxy (CSL/CSG) validated against pLOO-style scores rather than LOO ground truth. Rejected because theoretical claims are unsound and experimental validation is circular. The paper under review has better empirical validation breadth (multiple architectures/datasets) and does not overclaim theoretical grounding, but shares the validation circularity issue.

- **u9Z6gL5MlL** (Back to Fundamentals, Reject, 3/6/3/6): Comparable in scope—natural memorization, CIFAR-scale experiments, limited evaluation. Rejected partly for insufficient experimental breadth and story clarity issues. The paper under review has cleaner narrative and stronger cross-model validation.

- **9RNfX0ah0K** (LOO Distinguishability, Accept poster, 5/6/8/6): Strong theoretical grounding, validated LOO framework. Substantially stronger than the paper under review.

**Assessment**: The paper is clearly above the lTh7DEJV5W tier (which had false theoretical claims) but below the 9RNfX0ah0K tier (analytical rigor). Its strengths—consistent multi-architecture proxy correlation, clear compute savings, real diagnostic finding on pLOO accuracy—are genuine. But the central "more accurate" claim is validated on 160 points of a model not even used in the main experiments, the proxy is primarily validated against the approximation it replaces, and the paper's stated motivation (enabling large-scale memorization analysis) is never demonstrated empirically.

**Originality**: Moderate — the "late learning = memorization" insight is known for noisy labels; applying it to natural memorization is the contribution. **Importance**: High if claims generalize; uncertain given narrow validation. **Claim support**: Partial — compute savings well-supported; accuracy claim is not. **Experimental soundness**: Below average for the central claim. **Clarity**: Good. **Value to community**: Moderate.

**Score: 4.5 — Reject**. The paper has a promising idea and the ApB proxy is genuinely interesting and useful, but the headline accuracy claim is insufficiently validated, the proxy's validation is circular, and the scalability motivation is unsupported by experiment. The paper would need a broader LOO comparison, a recall analysis of the proxy, and ideally one larger-scale experiment to meet acceptance standards.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>