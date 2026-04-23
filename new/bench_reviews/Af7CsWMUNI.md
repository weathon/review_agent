Now I have thoroughly verified the claims. Let me write the final review.

## Summary

The paper proposes a new ICL paradigm that operates at the representation level using only unlabeled inputs from the test set. The method extracts hidden states from demonstration and test inputs independently, uses an attention-based mechanism to reconstruct test representations using retrieved similar test inputs, and then feeds the reconstructed representations through the LM head for prediction. The paper also provides preliminary analysis identifying two gaps between pretraining and ICL inference: label appearance and weak semantic relevance, and shows that removing labels helps general-domain but hurts specific-domain datasets.

## Strengths

- **Insightful preliminary analysis of label appearance effects (Table 2, Figure 1):** The domain-dependent finding—that removing labels from ICL demonstrations improves general-domain datasets (+9.56 avg for GPT-Neo-2.7B, +7.56 for Llama2-7B) but hurts specific-domain datasets (−21.16, −10.62)—is a genuine and non-obvious observation. The random-label ablation in Figure 1 further confirms that correct input-label mapping matters far more for specific-domain datasets (performance drops of 18–26 points) than general-domain ones (2–9 points).

- **Table 5 provides a controlled comparison supporting representation-level over text-level ICL:** When both approaches use unlabeled texts from the test set (a fair comparison), all five representation-level mapping methods outperform text-level concatenation for both GPT-Neo-2.7B (+7.94 to +10.51 avg improvement) and Llama2-7B (+10.57 to +11.71 avg improvement). This is the paper's strongest piece of evidence for the value of representation-level processing.

- **Training-free and label-free method with practical appeal:** The method requires no gradient updates and no access to labeled training data, making it immediately applicable to new tasks and low-resource scenarios. The approach is conceptually sound in addressing the "weak semantic relevance" gap by processing demonstration inputs independently rather than concatenating them.

## Weaknesses

### Fatal
None.

### Major

- **Unfair comparison with traditional ICL due to test-set access (Tables 4, 6):** The method retrieves unlabeled inputs from the test set (a transductive setting), while all ICL baselines retrieve labeled demonstrations from the training set. The paper's central claim—"our method with no labels outperforms traditional ICL with gold labels"—is confounded by this asymmetry: the method benefits from access to the test distribution (similar inputs from the test set implicitly carry distribution information via neighborhood structure), while ICL baselines benefit from label information. A fair comparison would either give ICL baselines test-set demonstrations with labels, or restrict the method to training-set retrieval. Neither is provided. The paper is transparent about the setup (Table 6 caption explicitly notes "test set" vs "training set"), but the headline framing still presents the comparison as evidence that representation-level ICL outperforms text-level ICL, when the confound of test-set vs. training-set access has not been disentangled.

- **Identical accuracies across fundamentally different models on MRPC and COLA suggest method collapse (Tables 4, 6, 7):** MRPC accuracy is 66.49 and COLA accuracy is 69.13 for ALL four models (GPT-Neo-2.7B, Mistral-7B, Llama2-7B, Llama2-13B) across ALL reported configurations, including all pooling strategies (Table 7). These models differ in architecture, training data, and parameter counts by factors of 2–5×. Producing identical accuracies across all of them is essentially impossible unless the method's output is nearly independent of the model on these datasets. 66.49 on MRPC (a binary task with ~68% majority class) and 69.13 on COLA are suspiciously close to majority-class baselines, suggesting the reconstructed representation may override model-specific predictions and collapse to a trivial predictor for some datasets. The paper does not investigate this at all—a serious omission given that these results constitute 25% of the reported datasets and inflate the reported averages.

- **Best-of-N configuration reporting inflates significance claims (Table 4):** The paper reports "the best improvement over zero-shot" across 5 mapping methods and sequential hyperparameter search (3 pooling × 3 k × 2 τ), yet the p-values (0.047, 0.022, 0.006, 0.046) are computed against a single zero-shot baseline without any multiple-comparison correction. Selecting the best configuration from multiple candidates and then testing significance against one baseline yields inflated p-values; two of the four (0.047, 0.046) would not survive even a simple Bonferroni correction for 5 comparisons (α = 0.01). The improvements are large enough that they likely persist in substance, but the reported significance levels are misleading.

### Minor

- **Missing ablations for the representation reconstruction mechanism:** No ablation isolates the core attention-based reconstruction (Eq. 10–12) from simpler baselines like directly averaging retrieved hidden states without the per-token attention mechanism. This would reveal whether the attention-based reconstruction adds value beyond simple retrieval-based averaging. Similarly, the 0.4/0.6 weighting in Eq. 11 and the normalization in Eq. 12 are critical design choices with zero justification or ablation.

- **Task description confound with zero-shot:** The method uses a task description T (e.g., "Represent the movie review to better determine whether it is positive or negative") in Step 1 for extracting hidden states, but this description is not part of the zero-shot baseline. While Step 3 uses the same label descriptions as zero-shot, the task description provides additional task-specific information to the hidden states that could contribute to the observed improvements over zero-shot.

- **Theoretical motivation partially mismatched with method (Section 2.3):** Equations 3–4 compare concatenation-based self-attention vs. independent-then-cross-attention processing. However, the actual method (Eq. 10) computes attention within each demonstration's token representations using the test input's representations as queries, which is not the same cross-attention mechanism analyzed in Eq. 4. The general motivation (independent processing) is consistent, but the specific mathematical argument doesn't directly support the chosen mechanism.

### Trivial
None.

## Nice-to-Haves

- A comparison with ICL that also retrieves from the test set (with labels) would definitively settle the test-set access confound—the single most important missing experiment.
- A comparison restricting the method to training-set retrieval only would isolate the contribution of test-distribution access.
- t-SNE/PCA visualization of original vs. reconstructed representations to investigate whether the method meaningfully reshapes the representation space or collapses to class-prototypical vectors on MRPC/COLA.
- Ablation of the 0.4/0.6 weights in Eq. 11 to determine whether the method is sensitive to this choice or if normalization makes it irrelevant.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Unfair comparison with ICL—method's advantage stems entirely from test-distribution access" (Harsh Critic Issue 1, extreme form):** The claim that the advantage stems "entirely" from test-set access is too strong. Table 5 shows representation-level outperforms text-level when BOTH use test-set retrieval, suggesting representation-level processing does add value beyond test-set access alone. However, the confound remains valid as a Major weakness for the comparison with traditional ICL baselines.

- **"Best-of-90 configuration" (Harsh Critic Issue 2, numerical claim):** The critic calculates 5×3×3×2=90 configurations, but the implementation details describe sequential search (first fix k, τ to find best pooling; then fix best pooling, τ to search k; then fix best pooling, best k to search τ), not a full grid. The actual number of evaluated configurations is lower (~5×(3+3+2)=40), though the core concern about multiple comparisons remains valid. Downgraded to Major rather than Fatal.

- **"Conflated contributions—improvement cannot be attributed to representation-level mechanism" (Harsh Critic Issue 4, strong form):** While the task description is an additional component not in zero-shot, Table 5 provides a controlled comparison where both representation-level and text-level use the same task description and test-set retrieval, showing representation-level wins. This partially addresses the concern. The concern about missing ablations for simpler retrieval baselines remains valid as Minor.

- **"BM25-ICL reports best of five score functions" (implicit in Harsh Critic):** Section 4.1 states "We report the best result of five score functions" for BM25-ICL. While this also involves best-of-N selection for baselines, it gives baselines the same advantage, so it slightly reduces the asymmetry concern rather than increasing it.

- **Strength Finder's "outperforms traditional ICL with gold labels on average" (Strength 2):** This strength is weakened because the comparison is confounded by the test-set vs. training-set access asymmetry (see Major weakness 1). Retained as partial strength only via Table 5's controlled comparison.

- **Strength Finder's "small models outperform zero-shot of models twice their size" (Strength 3):** This claim is misleading—small model + test-set transductive access vs. large model zero-shot without test-set access is not a fair comparison of model capability. Moved to Removed Points.

## Novel Insights

The identical-accuracy phenomenon across all four models on MRPC (66.49) and COLA (69.13) is a genuinely novel and troubling observation that goes beyond what the reviewers raised. The constancy holds across 12 different table entries (4 models × 3 pooling strategies) for each dataset, and even extends to the text-level baseline for MRPC. This pattern—where the method produces a fixed number regardless of model architecture—strongly suggests that on certain datasets, the reconstructed representation dominates the LM head's output so thoroughly that model-specific knowledge becomes irrelevant. This could mean the method effectively embeds dataset-level statistics (class proportions, prototypical representations) into the reconstruction in a way that bypasses the model entirely. Understanding when and why this collapse occurs would be valuable for the community, but the paper misses this opportunity by not investigating it.

## Suggestions

- Report results for a single fixed configuration (or mean ± std across configurations) rather than best-of-N, with Bonferroni-corrected p-values, to make significance claims credible.
- Investigate the MRPC/COLA identical-accuracy anomaly: compute the majority-class baseline for each dataset and analyze whether the reconstructed representations are collapsing to class-prototypical vectors. Report per-class accuracy to determine if predictions are simply the majority class.
- Add one critical experiment: run traditional ICL with test-set demonstrations (including labels) to isolate the contribution of test-set access from the contribution of the representation-level mechanism.

## Score and Decision

**Calibration anchors:**

- `/home/wg25r/review_agent/human_reviews/3M0GXoUEzP.md` (CrIBo, avg 8.0, Accept spotlight): Self-supervised learning with nearest-neighbor retrieval at test time—similar idea but far more rigorous methodology and evaluation. This paper is significantly weaker.
- `/home/wg25r/review_agent/human_reviews/ohJxgRLlLt.md` (Unsupervised ICL joint inference, avg 5.75, Accept poster): Also uses unlabeled test data for ICL with a transductive approach. More rigorous framework, but similar category of concern. Our paper has more serious issues (identical accuracies, best-of-N).
- `/home/wg25r/review_agent/human_reviews/OcXsdBo6vK.md` (Active ICL, avg 4.0, Reject): Similar transductive test-set access concerns, plus problem-setting clarity issues. Our paper is somewhat stronger in having a clearer method and better preliminary analysis, but weaker in having unacknowledged degenerate results.
- `/home/wg25r/review_agent/human_reviews/5Osk2F3qCn.md` (TADIS, avg 3.75, Reject): Method produces degenerate/identical results across conditions, similar to our identical-accuracy issue. Our paper has a more novel core idea but the degeneracy is similarly unacknowledged.
- `/home/wg25r/review_agent/human_reviews/pdf6MbXnAS.md` (STICL, avg 5.75, Reject): ICL disentanglement method with evaluation concerns on query overlap and compute.

This paper sits between the Active ICL paper (4.0) and the unsupervised ICL joint inference paper (5.75). It has a more interesting core idea than the Active ICL paper and a valuable preliminary analysis, but the identical-accuracy anomaly is a more serious unaddressed concern than what either anchor paper faced. The test-set access confound and best-of-N reporting further weaken the evidence. The paper is above the TADIS level (3.75) because its idea has genuine merit and some controlled comparisons (Table 5) partially support it, but below the 5.75 papers because the evaluation issues directly undermine the central claims.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>