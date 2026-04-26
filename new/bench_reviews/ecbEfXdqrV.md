Now let me run calibration searches to anchor my scoring.Now I have enough context to write the consolidated review. Let me compile the final assessment.

---

## Summary

This paper investigates whether the "counterintuitive phenomenon" of likelihood-based detection failure — documented by Nalisnick et al. (2019a) for image OOD detection with normalizing flows — also appears in tabular anomaly detection. The authors propose a formal, domain-agnostic definition of this phenomenon (Definition 3.3) based on relative AUROC ranking against baselines, conduct extensive experiments on 47 tabular and 10 CV/NLP embedding datasets from ADBench against 12 baselines, and find that NF-SLT (a normalizing flow with simple likelihood thresholding) consistently outperforms or ties other methods. They further provide theoretical analysis (Theorem 5.4, Corollary 5.6) linking dimensionality and feature correlation to the rarity of the phenomenon in tabular settings, validated through dimensionality reduction experiments and d Ratio analysis.

---

## Strengths

- **Comprehensive, unbiased empirical evaluation**: All 57 datasets from ADBench (47 tabular + 10 CV/NLP embeddings) are evaluated without cherry-picking, compared against 12 baselines including both shallow (PCA, LOF, OCSVM, etc.) and deep methods (DAGMM, DeepSVDD, ICL, NeuTraLAD, MCM). This is more thorough than typical tabular AD studies and gives real breadth to the empirical findings. NF-SLT achieves the best mean AUROC (0.8575), best average rank (3.43), and lowest fail ratio (0.02) in Table 1.

- **d Ratio analysis as a practical interpretive tool**: Using intrinsic dimensionality (via MLE and TwoNN) relative to ambient dimension to quantify feature correlation is a concrete operationalization. Table 4 demonstrates that image datasets have d Ratios of ~0.002–0.019 while tabular datasets range from 0.389–0.810, and the correlation between low d Ratio and NF-SLT failure on the 25 underperforming datasets is an informative finding that stands independently.

- **Controlled dimensionality reduction experiment (Table 2)**: Applying ICA to high-dimensional image data and measuring AUROC as a function of retained components provides empirical support that dimensionality degrades likelihood-based detection in a controlled, reproducible manner. The trend is consistent with Theorem 5.4.

- **CV/NLP embedding results reconciled with existing literature**: The explanation that embedding representations (d Ratio ~0.018–0.023) are less correlated than raw pixels is consistent with and extends Kirichenko et al. (2020), providing a unified account of why embeddings alleviate the phenomenon.

---

## Weaknesses

### Fatal
None.

### Major

- **The central definitional framework partially conflates "our method works well" with "the phenomenon is absent."** Definition 3.3 operationalizes the counterintuitive phenomenon as *most baselines outperforming the generative model (NF-SLT) by margin γ*. By construction, if NF-SLT ranks at the top of the leaderboard — which is the primary empirical result — Definition 3.3 declares the phenomenon absent. This creates a conceptual circularity: the definition is established, NF-SLT is shown to outperform baselines, and it is concluded "no phenomenon." The original Nalisnick et al. problem was a distributional one — whether anomalies receive *higher average log-likelihood* than normal samples under a model trained on normal data — not a comparative ranking problem. A dataset could exhibit likelihood inversion (anomalies assigned higher log-likelihood than normal samples) while NF-SLT still achieves good AUROC through appropriate threshold setting, and Definition 3.3 would still declare no phenomenon. The paper justifies the definitional shift (§3) by arguing that pure likelihood inversion can stem from dataset difficulty rather than the phenomenon, which is a reasonable point. However, the paper then never directly measures the original likelihood inversion rate across the 47 datasets. The conclusion that "the phenomenon is consistently rare in tabular settings" would be substantially strengthened — and much less circular — if the authors also reported the fraction of datasets where anomalies have *higher mean log-likelihood* than normal samples, independently of the comparative ranking. As it stands, the paper's headline claim is weaker than stated.

### Minor

- **Theorem 5.4's independence assumption is acknowledged but incompletely addressed.** The theorem requires P and Q to be products of independent marginals. The ICA preprocessing in Table 2 is designed specifically to produce approximately independent components, which makes the experiment consistent with the assumption. However, in the naturalistic Table 3 experiment (raw image resize), the paper correctly notes independence is not guaranteed and therefore the theorem does not apply — yet Table 3 is still cited as qualitative evidence for the theory's predictions. For the tabular domain, the d Ratio analysis (§5.2) shows that some tabular datasets with low d Ratio (high correlation) are precisely where NF-SLT fails, suggesting the independence condition is not uniformly met. The paper should more carefully separate claims about "approximately independent tabular data" (where the theorem applies) from "correlated tabular data" (where it doesn't), rather than treating the theorem as a blanket explanation for tabular success.

- **NeuTraLAD's preprocessing exclusion is ad hoc and could affect relative rankings.** Excluding NeuTraLAD from RobustScaler "because a significant performance decrease was observed" is a result, not a principled justification. If NeuTraLAD is scale-sensitive, that is a property of NeuTraLAD relevant to fair comparison. Using different preprocessing for different models creates an inconsistency that is hard to audit without the appendix. Ideally the paper would show NeuTraLAD's AUROC under both preprocessing regimes.

- **Corollary 5.6's moment condition is stated but not empirically verified.** The condition that the n-th absolute central moment of the log-likelihood difference scales as O(d^k) for k < n is stated as an assumption in the corollary. The paper does not check whether this condition holds on any of the actual datasets used, which makes the corollary's claim about AUROC's upper bound remaining theoretical rather than grounded.

### Trivial

- The β and γ thresholds in Definition 3.3 are defined in Appendix B, but the main text gives no indication of their values or any sensitivity analysis. Readers evaluating the claims about 'yeast' and 'imdb' cannot assess the conclusions without these values.

---

## Nice-to-Haves

- Directly reporting the fraction of tabular datasets where the anomaly set has higher mean log-likelihood than the normal test set under NF-SLT (i.e., testing the original Nalisnick et al. criterion) would directly validate or complicate the central claim without requiring any definitional reframing.
- A sensitivity analysis over β and γ in Definition 3.3, showing how the count of "phenomenon-exhibiting" datasets changes, would assess robustness of the main conclusion.
- For the 25 datasets where NF-SLT underperforms (rank ≥ 3), showing that baselines *also* fail on low-d-Ratio datasets (rather than only NF-SLT) would help disentangle "dataset difficulty" from "the counterintuitive phenomenon."
- Log-likelihood distribution overlap plots (normal vs. anomaly) on representative tabular datasets — analogous to the image-domain figures in Nalisnick et al. (2019a) — would directly visualize whether classical likelihood inversion occurs.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's claim about hyperparameter optimization asymmetry**: The paper states "The hyperparameter search space for each model and hyperparameter sensitivity experiment is recorded in Appendix F," implying that all models received grid-search treatment. The criticism assumes unfair tuning without evidence of this from the paper text. Removed.

- **Harsh critic's claim about Definition 3.3 producing a result entirely tautological**: While the circularity concern is real and kept as a major weakness, the harsh critic's framing that the definition was "crafted" post-hoc to guarantee NF-SLT wins is not supported. The paper provides a coherent conceptual motivation for why comparative ranking is a better criterion than raw likelihood inversion, and the CIFAR-10/SVHN example (AUROC 6.4% vs. >90%) serves as external validation. The valid part of the criticism is retained as a major weakness, but the extreme framing is removed.

- **Claim that the ICA dimensionality reduction experiment does not test the theorem in a natural setting**: The harsh critic notes that ICA "directly imposes the independence assumption" of the theorem, which is true, but this is precisely what the paper intends — to validate the theorem in a controlled setting where its assumptions hold. The paper is transparent about this and also includes Table 3 as a less controlled companion. This is not a fair criticism of the methodology.

- **Harsh critic's claim that circular reasoning "structurally undermines" the central claim as fatal**: After reviewing the paper, this concern is real but does not rise to fatal status. The empirical finding that NF-SLT outperforms 12 baselines on 47 datasets is meaningful and robust, even if the definition of "counterintuitive phenomenon" could be more directly tied to likelihood inversion. Retained as Major, not Fatal.

---

## Novel Insights

The most genuinely novel aspect of this paper is the d Ratio analysis, which provides a domain-level diagnostic for when likelihood-based tabular anomaly detection is expected to succeed or fail. The empirical finding that image datasets have d Ratios 2–3 orders of magnitude lower than typical tabular datasets, and that this correlates with NF-SLT's failure pattern within tabular data (Table 4, bottom), is a concrete and actionable insight for practitioners — it suggests that before applying NF-SLT to a new tabular dataset, computing the TwoNN intrinsic dimensionality relative to ambient dimension can serve as a rough predictor of whether likelihood-only detection will be competitive.

---

## Suggestions

1. Add a supplementary analysis directly measuring the fraction of the 47 tabular datasets where anomaly samples achieve higher mean log-likelihood than normal samples under NF-SLT (classical Nalisnick criterion). This would decouple the empirical finding from the definitional reframing and dramatically strengthen the paper's claims.
2. Report the β and γ values for Definition 3.3 in the main text (even briefly) and include a sensitivity table showing how the count of "phenomenon-exhibiting" datasets changes across a grid of thresholds.
3. Add a brief per-dataset breakdown for the 25 NF-SLT underperforming cases: do baselines also fail, or do they succeed? This would separate "hard datasets" from "datasets exhibiting the counterintuitive phenomenon."

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Score | Comparison to paper under review |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/oDGkq0AleM.md` | 3.0 | Low anchor. Tabular density-based AD with no supporting theory and logically unsound motivation. Weaker than this paper's empirical sweep and formal theorem. |
| `/home/wg25r/review_agent/human_reviews/6Z8rZlKpNT.md` | 3.4 | Low anchor. NF-based OOD detection, missing baselines, poorly written, incomplete methods. More clearly deficient than this paper. |
| `/home/wg25r/review_agent/human_reviews/hWF4KWeNgb.md` | 4.25 | Medium-low anchor. NF-based multi-class AD, partially valid method but rejected. Closer in quality to this paper. |
| `/home/wg25r/review_agent/human_reviews/LGafQ1g2D2.md` | 5.2 | Medium anchor. "Can LLMs understand time series anomalies?" — empirical investigation of a phenomenon with comprehensive experiments. Structurally similar to this paper. Accepted poster. |
| `/home/wg25r/review_agent/human_reviews/7VkHffT5X2.md` | 6.75 | Medium-high anchor. AnoLLM, tabular AD with LLMs. Novel method, accepted poster. More methodologically innovative, better-aligned definition of contribution. |
| `/home/wg25r/review_agent/human_reviews/cJs4oE4m9Q.md` | 8.0 | High anchor. Deep anomaly detection with strong theory and rigorous experiments. Much stronger theoretical foundation. |

**Assessment against anchors:** This paper is most similar in structure to the LGafQ1g2D2 "Can LLMs…?" paper (5.2, accepted poster) — it empirically investigates a phenomenon across many datasets with theoretical backing. However, that paper is clearer about what it measures and does not change the definition of the phenomenon mid-paper. This paper's major weakness (the conceptual circularity in Definition 3.3 conflating "our model works" with "phenomenon absent") is a meaningful flaw that would be highlighted in a rebuttal. The breadth of evaluation and the d Ratio analysis are genuine strengths that raise it above the 3.0–3.4 rejected papers. The lack of a direct measurement of the original likelihood inversion phenomenon is a notable gap. Positioning this between hWF4KWeNgb (4.25, rejected) and LGafQ1g2D2 (5.2, accepted), slightly closer to the lower end given the definitional circularity concern, I arrive at **4.5**.

**Originality**: Moderate. The question is natural and well-motivated; the d Ratio operationalization is new; the definitional formalization is useful but imperfect.
**Importance of research question**: Good. Practitioners using normalizing flows for tabular AD benefit from knowing when simple likelihood scoring is safe.
**Claims vs. support**: Mixed. Empirical results strongly support NF-SLT's ranking; the claim that "the counterintuitive phenomenon is rare" is more fragile due to the definitional shift.
**Soundness of experiments**: Generally sound; comprehensive evaluation; some concerns about NeuTraLAD preprocessing and the ICA-specific nature of Table 2.
**Clarity**: Adequate; the paper is readable but the definitional arguments in §3 require careful attention.
**Value to community**: Real but moderate; practitioner-relevant guidance exists, but the theoretical apparatus is somewhat underdeveloped for the claim it carries.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>