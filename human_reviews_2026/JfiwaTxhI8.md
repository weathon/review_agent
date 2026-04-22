# Epistemic Uncertainty Quantification To Improve Decisions From Black-Box Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 2

## Abstract
Distinguishing epistemic uncertainty (model ignorance) from aleatoric uncertainty (task randomness) is critical for reliable AI systems, yet standard confidence evaluation metrics capture different and incomplete aspects of uncertainty. While AUC and accuracy measure predictive signal, proper scoring rules assess overall uncertainty, and calibration metrics isolate part of the epistemic uncertainty but ignore within-bin heterogeneity of errors, known as grouping loss. We bridge this evaluation gap by introducing asymptotically consistent and sample-efficient estimators of the grouping loss and excess decision risk, providing a fine-grained assessment of epistemic uncertainty that complements existing calibration metrics. Applied to LLM question-answering with inherent aleatoric noise, our estimators reveal substantial grouping loss which decreases with model scale and instruction tuning. Their local nature enables automatic identification of subgroups with systematic over- or under-confidence, supporting interpretable confidence audits. Finally, we leverage these estimates to design LLM cascades that defer high excess decision risk predictions to stronger models, achieving higher accuracy at lower cost than competing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the excess risk, focussing on the grouping loss, of predictors. In particular, it proposes an estimator for these quantities in the context of proper scoring rules. The estimators are shown to have desirable properties such as consistency and unbiasedness, and instantiations with the 0-1 loss are considered. The method is evaluated on a semi-synthetic setting with known ground-truth probabilities and on a range of experiments involving LLMs and real data.

### Strengths
- The paper covers a valuable topic and does a good job of explaining exactly why considering the epistemic uncertainty quantification and, specifically, grouping loss, is important by including illustrative examples.
- The provided solution is reasonable, satisfies desirable properties, and performs well empirically.
- The paper reads well, and has clear notation and exposition for the formal results.
- There is a good range of empirical settings.

### Weaknesses
- The paper focuses on the binary setting, which is, of course, fairly limited.

There seems to be a leftover comment in the Appendix p. 13, l. 697.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces asymptotically consistent and sample-efficient lower-bound estimators for the grouping loss and excess risk, i.e., suboptimality of a prediction, serving as a complement to existing calibration metrics. The authors leverage the proposed estimator to design efficient LLM cascades that defer to stronger models, achieving higher accuracy at a lower cost than competing approaches.

### Strengths
1. The work paper is well structured, balancing the theoretical analysis and experimental validations.

2. Proofs for the related propositions are given in the appendix.

3. The extensive evaluations demonstrated the performance improvements convincingly.

### Weaknesses
1. Would the author be more explicit in indicating what the resources of the epistemic uncertainty are considered in the proposed method?

2. Minor: There is a lack of a REPRODUCIBILITY STATEMENT in the main body.

3. Minor: In the appendix, there is an unnecessary comment in green remaining. line 699.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper propose lower-bound estimators for grouping loss and excess risk, designed to address deficiencies in standard metrics (AUC, accuracy, calibration error) that conflate model confidence without properly distinguishing epistemic error or bias. Experiments cover semi-synthetic, tabular, and LLM question answering tasks, including cost-sensitive settings and cascades.

### Strengths
- The paper is generally well written. Extensive notational and methodological setup supports reproducibility and rigour.

- It advances epistemic uncertainty estimation using consistent, efficient lower-bound estimators,

- It addresses a critical gap for high-stakes AI applications.

### Weaknesses
- It does have dense mathematical sections which might make it harder to digest. Intuitive diagrams could help.

- Computational considerations for large-scale or real-time applications are missing. Scalability and latency constraints need more discussion.

- An ablation on tree depth is missing.

### Questions
- What are the practical limits for inference-time overhead with honest tree partitioning?

- How does the proposed approach adapt to free text tasks?

- How to select tree hyperparameter for new tasks or domains?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose estimators of “grouping loss” and “excess risk”, and apply them to three decision problems: predicting the benefit of LLM finetuning, assessing LLM confidence, and routing queries between multiple LLMs.

### Strengths
Originality: the proposed estimators appear to be novel.

Quality: Figures 2, 4 and 6 communicate the authors’ results well.

Clarity: a good amount of the low-level text is easy to follow.

Significance: query routing is a potentially impactful application.

### Weaknesses
I believe the technical motivation of this work is unsound.

- First, describing an aleatoric-epistemic uncertainty decomposition as “crucial for reliable AI” and using that as the premise of the rest of the paper is not appropriate. The aleatoric-epistemic view has been shown to be incoherent, and subjective model-based uncertainties alone are not a legitimate basis for assessing model reliability (Bickford Smith et al, 2025).
- Second, dismissing standard evaluation metrics—particularly proper scoring rules, which are very well established and have strong theoretical foundations (Savage, 1971)—in favour of new quantities ought to be extremely well argued from a technical perspective, yet it isn’t. Instead the authors offer an assortment of statements about various metrics, then quickly land on “grouping loss” and “excess risk” as the quantities of apparent interest.

Even if we were to set this aside, the practical contribution is hard to get behind.

- In Section 4.2 we see results for predicting the benefit of LLM finetuning. The practical significance of these is unclear. We see a scatter plot of the “excess risk” estimator against “cost reduction”, but the magnitudes of these quantities are hard to interpret from the plot. This is a nonstandard setup that needs explaining better or supplementing with a more practically relatable setup. It also appears a basic baseline method is missing: why not just evaluate the “cost” on the training set, if this is what we care about?
- In Section 4.3 there isn’t a well-defined task or notion of success. Instead the authors just report confidence results (confidence is not a quantity they are proposing) in Figure 1 and “grouping loss” results in Figure 4. What are we to do with these?
- In Section 4.4 the focus is on routing a query to an appropriate LLM, balancing prediction costs against predictive accuracy. The results in Figure 5 appear to be positive, although the choice to stop at $t=0.001$ gives a false impression of how expensive the proposed method can be. This high cost is clear from Figure 6 (right): even with the baseline method in a high-cost configuration ($\lambda=100$), the proposed method is up to ~60 percent more expensive, and the payoff in terms of accuracy is small, at less than ~3 percentage points. At a higher level it seems the extent to which the results are positive here is highly contingent on the dataset having many examples that can be accurately classified by the smaller models.

---

Bickford Smith et al (2025). Rethinking aleatoric and epistemic uncertainty. ICML.

Savage (1971). Elicitation of personal probabilities and expectations. Journal of the American Statistical Association.

### Questions
Can you provide a technical argument for the insufficiency of proper scoring rules? Note that any appeal to the idea of epistemic/reducible uncertainty would have to be rigorously derived, not just stated as important.

Section 4.2:

- Can you clarify the cost matrices you used?
- Can you explain the axis magnitudes in Figure 3?
- Why not just evaluate the “cost” on the training set, if this is what we care about?

Section 4.3: what is the concrete task or notion of success you have in mind?

Section 4.4: can you extend the lines in Figure 5 to include $t=0$?

### Soundness
1

### Presentation
1

### Contribution
1
