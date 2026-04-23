Now I have all the information needed. Let me compose the final review.

## Summary

The paper introduces Learning Embedding Linear Projections (LELP), a knowledge distillation method for few-class and binary classification that creates pseudo-subclasses via PCA of the teacher's final-layer embeddings, avoiding the need to retrain the teacher (unlike Subclass Distillation). The student is trained on expanded S×C pseudo-subclass labels using a unified cross-entropy loss. Experiments on binarized CIFAR and large-scale NLP benchmarks (Amazon Reviews, Sentiment140) show LELP consistently outperforms baselines including Subclass Distillation.

## Strengths

- **LELP eliminates costly teacher retraining while matching or exceeding Subclass Distillation.** Table 2 shows LELP beats Subclass Distillation in 6/8 NLP columns (e.g., Amazon Reviews 5-class: 78.06% vs 76.28%), without requiring the iterative teacher retraining that makes Subclass Distillation "excessively computationally intensive and impractical" (Section 2). This is a clear practical advantage in the era of large language models.

- **Oracle Clustering upper-bound analysis validates the fundamental motivation.** Table 1 shows Oracle Clustering achieves 86.42% vs 72.87% for Vanilla KD on CIFAR-100bin (ResNet-92→ResNet-56), demonstrating large potential gains from pseudo-subclass approaches, lending strong conceptual support to the paper's premise.

- **Systematic comparison showing linear projections outperform other clustering methods.** Table 1 compares six approaches across six teacher-student-dataset combinations. LELP achieves the best performance among practical methods in every setting (e.g., 79.91% on CIFAR-100bin vs next-best 77.03%), while K-means and Agglomerative clustering sometimes underperform Vanilla KD. This validates the core claim that *how* subclasses are created matters critically.

- **Architecture-agnostic effectiveness across diverse teacher-student pairings.** The method works with matching dimensions (ResNet-92→ResNet-56), mismatched dimensions (ResNet-92→MobileNet), and cross-architecture NLP settings (ALBERT-XXL→ALBERT-Base), as demonstrated across Tables 1 and 2.

- **Practical simplicity with minimal invasiveness.** Section 3.3 explicitly contrasts LELP's simplicity with FitNet (needs pretraining), CRD (needs memory buffer), and Subclass Distillation (needs teacher retraining). LELP requires only PCA computation on cached teacher embeddings and expanding the student output layer from C to S×C.

## Weaknesses

### Fatal

None.

### Major

- **Table 2's "Avg. gain" summary rows contain values that are factually inconsistent with the individual entries.** The "Avg. gain over the best baseline" row reports +0.05 for columns 6–8, but computing from individual entries yields 1.78% (78.06−76.28), 0.38% (77.45−77.07), and 1.67% (87.60−85.93) respectively. None of the 8 columns' reported summary values match the actual gain of LELP over the best baseline. The other summary rows ("Avg. gain over non-subclass baseline" and "Avg. gain over Vanilla KD") are similarly inconsistent with individual entries. This is not a rounding issue—the discrepancies are large (e.g., 1.78% vs 0.05%) and affect every column. These errors undermine confidence in the paper's summary of its own results.

- **Figure 1 and Table 2 report substantially different numbers for the same-named datasets without explanation.** For Sentiment140, Figure 1 shows LELP at ~82.5% with teacher at ~81.5%, while Table 2 (column 8, "Sentiment-60 Bin") shows LELP at 87.60% with teacher at 87.29%—a ~5 percentage point discrepancy. For Amazon Reviews, Figure 1 shows teacher accuracy at ~75.5%, while Table 2 shows 77.58%. The paper describes two student architectures for the large NLP datasets (ALBERT-Base and a sentence-T5 MLP, Section 4.1) and different teacher models (ALBERT-XXL and ALBERT-XL), which could explain these differences, but never specifies which setup Figure 1 uses vs Table 2. This makes it impossible to determine whether Figure 1 and Table 2 are even showing the same experiments.

- **The abstract's headline improvement claims do not match Table 2.** The abstract claims "+1.85% over the best baseline" for Amazon Reviews and "+0.88%" for Sentiment140. From Table 2, the actual gains over the best baseline (Subclass Distillation) are 1.78% (78.06−76.28) and 1.67% (87.60−85.93). The abstract numbers likely come from the Figure 1 experimental setup (different student architecture or dataset subset), but the paper presents them as the main claims without clarifying this. A reader relying on the abstract would receive incorrect improvement magnitudes.

### Minor

- **All experiments use α = 0 (pure distillation loss, no ground-truth cross-entropy), which limits the scope of practical conclusions.** The paper justifies this as isolating the effect of the distillation loss and notes it's standard in the semi-supervised setting (Section 4.1). However, in standard supervised KD, α > 0 is almost universally used, and the relative ranking of methods could change when ground-truth labels provide stabilization. Without at least one α > 0 experiment, it remains unclear whether LELP's advantages hold in the most common practical setup.

- **Table 2 contains duplicate column labels ("Am. Reviews Bin" twice, "Am. Reviews" twice) without differentiating what each pair represents.** The paper describes two student architectures (ALBERT-Base and sentence-T5 MLP) and two teacher models for the large NLP datasets, so the duplicate columns likely correspond to these different setups, but this is never stated in the table or its caption.

- **Table 2 uses abbreviations that don't clearly correspond to the baselines named in Section 4.1** (e.g., "CKD" vs "CRD" in the text, "Retained KD" vs "Relational KD", "Feature" vs "FitNet"). This makes it harder for readers to connect results to methods.

## Nice-to-Haves

- Sensitivity analysis for β (subclass tempering) and S (number of projections per class), which the paper mentions are in Appendix C but are important for practical deployment guidance.
- Experiments with at least one α > 0 setting to confirm results generalize to the standard KD setup.
- Analysis of what semantic structure the PCA projections capture in NLP tasks (beyond "it works"), to deepen understanding of why LELP is effective.
- Per-class PCA variance explained plots to justify the choice of S and the linear projection assumption for each dataset.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Unfair comparison with Subclass Distillation drives headline claims"**: The paper acknowledges the comparison asymmetry (Section 4.1, last paragraph) and provides both comparisons in Table 2. Moreover, for Amazon Reviews 5-class, the Subclass Distillation teacher (78.45%) is actually *better* than the standard teacher (77.58%), meaning the asymmetry favors the baseline in that key experiment, and LELP still wins. Per review rules, criticisms about unfair comparisons that favor the baseline should be removed.

- **"Student outperforming teacher suggests underfitting or artifact"**: Student outperforming teacher is a well-documented phenomenon in the KD literature (especially with subclass/Oracle methods) and is also observed with Oracle Clustering in Table 1. The paper doesn't claim this is a "principled advantage of LELP"—it presents it as an empirical finding. This is not a weakness.

- **"No ablations for null-space projection and random rotation in the main paper"**: The paper explicitly states these are in Appendix C (Section 3.1, line 114: "In Appendix C we perform ablations..."). The appendix was stripped from this review, so this criticism cannot be verified.

- **"β and S sensitivity not analyzed in the main paper"**: Same as above—ablations are stated to be in Appendix C. Cannot verify absence.

- **"The Oracle gap suggests LELP captures only a fraction of available subclass information"**: This is not a weakness of LELP; it's an inherent limitation of any practical method compared to an oracle. The paper acknowledges this in Section 5.

- **"Missing experiments for the MLP student architecture in the main paper"**: The duplicate columns in Table 2 likely *are* the MLP student results (columns 5 and 7 seem to show different performance profiles), though this isn't clearly labeled. Regardless, demanding more results is a generic request.

- **"Reproducibility concerns about hyperparameters"**: Per review rules, nitpicks about undisclosed hyperparameters or trivial implementation details should be removed. The paper provides hyperparameters in Appendix H.

- **Formatting/stylistic criticisms** (duplicate descriptions, notation): Per rules, formatting artifacts are removed.

## Novel Insights

The tension between Figure 1 and Table 2 likely reflects a deeper structural issue: the paper evaluates two distinct student architectures (ALBERT-Base and sentence-T5 MLP) but never clearly delineates which results correspond to which setup. This ambiguity is not merely cosmetic—it means the headline claims in the abstract may come from the MLP setup (which leverages a frozen 11B-parameter encoder), while the main table may show ALBERT-Base results. Clarifying this distinction would substantially improve interpretability and might reveal that LELP's gains are architecture-dependent in important ways.

## Suggestions

- **Fix the "Avg. gain" summary rows in Table 2.** These are the most straightforwardly incorrect element and should be corrected to match the individual entries.
- **Label the duplicate columns in Table 2** to clearly indicate which student architecture (ALBERT-Base vs MLP) and which teacher (ALBERT-XXL vs ALBERT-XL) each column uses.
- **Add a brief note in the Figure 1 caption** explaining which experimental setup it shows and how it relates to Table 2.
- **Run at least one experiment with α > 0** (e.g., α = 0.5) on a subset of Table 2's conditions to confirm LELP's advantages persist in the standard KD setup.

## Evaluation

**Originality**: The connection between Neural Collapse and pseudo-subclass creation via PCA is novel. LELP's core idea—projecting teacher embeddings into linear subspaces to create pseudo-subclasses without teacher retraining—is a genuine contribution over Subclass Distillation.

**Importance of research question**: Few-class KD is an important and underexplored problem, especially for NLP applications. The practical motivation of avoiding teacher retraining is timely.

**Claims well supported**: The CIFAR results (Table 1) are clean and convincing. The NLP results (Table 2) show consistent improvements in individual entries, but the incorrect summary rows and inconsistency with Figure 1/abstract undermine confidence in the exact magnitude of gains.

**Soundness of experiments**: The experimental design is comprehensive (9 baselines, 8 NLP conditions, CIFAR analysis). The α = 0 choice is defensible but limits scope. The missing column labels make interpretation difficult.

**Clarity of writing**: The method description is clear and well-organized. The experimental section is hampered by the reporting issues noted above.

**Value to community**: LELP addresses a real practical need (no teacher retraining) with a simple method. If the reporting issues are corrected, this would be a useful contribution.

## Calibration

Anchors used:
- **wPMRwmytZe** (avg 7.6, Oral): Progressive distillation with implicit curriculum—strong theory + clear experiments. LELP is below this due to reporting inconsistencies and lack of theoretical analysis.
- **1aF2D2CPHi** (avg 8.0, Oral): Data-free KD from CLIP—comprehensive experiments, clear presentation. LELP is well below this.
- **IcVSKhVpKu** (avg 5.67, Poster): CKA-based KD with dimension mismatch—comparable practical motivation to LELP, some clarity issues but no factual errors in tables. LELP is slightly below this due to Table 2's incorrect summary rows.
- **0cBttXaOUK** (avg 5.0, Reject): Multi-aspect KD from LLMs—conceptually similar pseudo-class expansion, rejected for unclear methodology. LELP has a clearer method but worse reporting errors.
- **2TOcJivjpt** (avg 3.0, Reject): KD under distribution shift—overclaimed findings, inconsistent results, poor presentation. LELP is above this because its core method is sound and individual Table 2 entries are informative.
- **8TbqoP3Rjg** (avg 2.0, Reject): KD to mitigate model collapse—fundamentally flawed claims. LELP is well above this.

LELP sits between the medium-scoring KD papers (5.0–5.67) and the low-scoring ones (2.0–3.0). Its method is sounder than the rejected papers and its practical motivation is clear, but the factual errors in Table 2's summary rows and the unexplained Figure 1/Table 2 inconsistency are worse than the issues in the 5.0–5.67 range papers. I place it at **5.0**—a borderline paper with a real contribution that needs significant reporting corrections.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>