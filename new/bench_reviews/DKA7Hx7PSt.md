Now I have all the information I need. Let me write the final review.

## Summary

The paper introduces LELP (Learning Embedding Linear Projections), a knowledge distillation method for binary and few-class classification problems. LELP addresses the well-known limitation that KD provides diminishing returns when the number of classes is small by applying PCA to teacher embeddings, projecting onto the null-space of teacher output weights, and splitting each class into pseudo-subclasses via tempered softmax over PCA coordinates. The student is then trained with a unified cross-entropy loss over the expanded pseudo-subclass label space, requiring no teacher retraining. Experiments on binarized CIFAR and large-scale NLP benchmarks (Amazon Reviews, Sentiment140, GLUE tasks) demonstrate consistent improvements over baselines, particularly over Subclass Distillation, while avoiding its expensive teacher retraining requirement.

## Strengths

- **Addresses a genuine and important gap**: The paper identifies that standard KD is ineffective for binary/few-class tasks, particularly in NLP, and this problem is underexplored. The motivation is well-grounded in the Neural Collapse literature (Yang et al., 2023) and Subclass Distillation (Müller et al., 2020). Table 1 and Figure 3 clearly demonstrate that subclass-splitting helps, and the Oracle Clustering upper bound provides compelling evidence that embedding structure is exploitable.

- **Practical efficiency advantage**: LELP achieves performance competitive with or superior to Subclass Distillation without requiring teacher retraining — a significant practical advantage, especially for large teacher models. Section 3 explicitly motivates this with clear desiderata (modality-independent, architecture-agnostic, no teacher retraining), and the method satisfies all three.

- **Strong empirical performance on the target problem**: On the core NLP experiments (Table 2, columns 6–8), LELP shows substantial gains: +1.78% over Subclass Distillation on Amazon Reviews 5-class, +1.67% on Sentiment140, and the student even outperforms the teacher (78.06% vs. 77.58%) on the Amazon Reviews 5-class task — a compelling result.

- **Controlled subclass creation ablation**: Table 1 compares LELP against multiple clustering alternatives (K-means, Agglomerative, t-SNE+K-means), demonstrating that the specific mechanism matters — not just the idea of creating subclasses. LELP consistently outperforms all alternatives, validating the linear projection design.

- **Diverse experimental setups**: The paper covers Vision (ResNet→ResNet, ResNet→MobileNet) and NLP (ALBERT→ALBERT, sentence-T5→MLP) with both binary and 5-class tasks, supporting the modality-independence claim.

## Weaknesses

### Fatal
None.

### Major

- **Table 2 "Avg. gain" rows are mathematically inconsistent with reported accuracy values**. The "Avg. gain over the best baseline" row reports values like +0.02 to +0.05 across all columns, but the actual per-column differences between LELP and the best visible baseline are much larger and sometimes negative (e.g., Column 6: LELP=78.06 vs. Subclass=76.28 yields +1.78%, not +0.05; Column 5: Feature=91.40 > LELP=91.16, making the gain negative; Column 3: Subclass=92.85 > LELP=92.81, also negative). Since each column has a single gain value (not an average), the "Avg." label and the reported numbers do not correspond to the raw data. The full table likely includes additional columns (sentence-T5/MLP architecture) that could affect these computations, but as presented, the summary rows undermine confidence in the paper's own reporting of its results.

- **In column 5 of Table 2 (2nd Am. Reviews Bin), Feature/FitNet (91.40) outperforms LELP (91.16), yet the bold formatting marks Subclass Distillation (90.38) as "the best forming algorithm"** per the paper's stated convention. Since 91.40 > 91.16 > 90.38, neither Subclass (best baseline? not even that — Feature is) nor LELP follows the stated bolding rule. This further raises questions about the table's internal consistency and how "best" is being defined. The paper should clarify whether the bolding means "best baseline excluding LELP" and account for cases where a non-Subclass baseline exceeds both.

### Minor

- **The comparison with Subclass Distillation is partially confounded by differing teacher models**: The paper explicitly acknowledges this (Section 4.1: "the accuracy of the teacher model in Subclass Distillation usually differs from the one used for LELP") and partially addresses it by reporting "Avg. gain over non-subclass baseline." However, the paper's narrative still leans heavily on LELP "typically exceeding" Subclass Distillation (Section 4.3.2: "LELP exceeds or performs as well as Subclass Distillation"), while in at least one case (QGLUE/sst2: 92.81 vs. 92.85), LELP is slightly worse. A more nuanced framing would acknowledge that the comparison is not fully controlled and that LELP sometimes underperforms Subclass Distillation.

- **α=0 throughout experiments**: All results use α=0 (no cross-entropy with ground-truth labels) to isolate distillation effects and reduce hyperparameter variance. While this is a defensible design choice for a controlled comparison, it means the reported numbers do not directly reflect how LELP would perform in the standard supervised KD setting (α>0), which is how the method would be deployed in practice. The paper should at least discuss whether LELP's advantages persist when combined with supervised loss.

- **Hyperparameter sensitivity of β and S is not studied in the main paper**. The tempered softmax temperature β (Section 3.2) and the number of projections per class S are critical hyperparameters specific to LELP. Ablations reportedly appear in Appendix C, but main-text discussion of sensitivity is absent. Given these directly control the pseudo-subclass structure, understanding their impact is important for practical adoption.

- **The method's composition lacks principled justification**: LELP combines null-space projection, PCA, random rotation, and tempered softmax — each justified empirically ("we have found that it helps") rather than theoretically. While this is common in empirical KD work, the null-space projection in particular raises an architectural dependence concern: when D < S+C, the null-space is trivial. The paper notes "we have not run into the problem" (Section 3.1), but this is a limitation that deserves more discussion.

- **LELP's scope limitation is underemphasized relative to the introduction's claims**: Section 5 honestly acknowledges that LELP converges to Vanilla KD as the number of classes grows and that subclasses may not be linearly separable. However, the introduction and abstract frame the contribution more broadly, which could mislead readers about the method's narrow applicability (binary and few-class settings only).

### Trivial

- The "CKD" label in Table 2 likely refers to CRD (Contrastive Representation Distillation), but the abbreviation mismatch is confusing.

## Nice-to-Haves

- Experiments with α>0 would strengthen practical relevance and distinguish LELP's contribution from pure distillation-loss effects.
- A controlled comparison where both LELP and Subclass Distillation use the same teacher model would eliminate the confound and directly test LELP's advantage.
- Visualization of learned projections on NLP data (analogous to Figure 4 for Vision) would reveal whether pseudo-subclasses capture semantically meaningful structure.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Abstract's 1.85% and 0.88% claims mismatch Table 2**: The visible Table 2 columns show gains of 1.78% and 1.67% over Subclass Distillation — close to but not matching the abstract. However, the paper describes experiments with two student architectures (ALBERT-Base and sentence-T5+MLP), and Table 2 in the parsed version only shows ALBERT-Base columns. The abstract numbers likely come from the full table including the sentence-T5/MLP setup, which is not visible due to appendix/main-text truncation. Without access to the complete table, this mismatch cannot be confirmed.

- **Figure 1(a) shows teacher accuracy ~75.5% for Amazon Reviews vs. Table 2's 77.58%**: This discrepancy may stem from reading approximate values from a bar chart, or the figure may represent a different experimental configuration. Given the inherent imprecision of reading bar charts and potential for different dataset splits, this is not a substantive issue.

- **Student-outperforming-teacher is not unique to LELP**: The harsh critic argues this is well-documented and not uniquely attributable to LELP. However, the paper does not claim this phenomenon is unique to LELP — it presents it as supporting evidence. The critic's objection is a strawman.

- **Demand for theoretical grounding of the full method pipeline**: While desirable, this is not standard for empirical KD papers. Several high-scoring calibration papers (e.g., m50eKHCttz at 7.25, 1aF2D2CPHi at 8.0) also lacked theoretical justification and were accepted.

- **Request for LMRD/cola/sst2 results in Table 2**: The paper's Section 4.3.2 mentions "the Large Movie Review dataset... two GLUE datasets: cola and sst2" but Table 2 shows "MRQ, QGLUEval, QGLUEval..." — these likely correspond to those datasets with different naming, or the full table includes more columns.

- **Reproducibility concerns about unavailable models/tools**: Removed per hard rules.

- **Missing comparison with random pseudo-labels control**: This is a nice-to-have but not a fundamental flaw; the Table 1 ablation against different clustering methods partially addresses the question of whether gains come from the specific subclass structure.

## Novel Insights

The Oracle Clustering experiments in Section 4.2 reveal an important finding: the gap between Oracle and LELP on CIFAR-100-bin (86.42 vs. 79.91 for ResNet-56) suggests that linear projections capture only a fraction of the available subclass structure, implying significant room for improvement through richer (nonlinear) projection methods. Conversely, LELP outperforming Oracle on CIFAR-10-bin ResNet→MobileNet indicates that LELP can extract information beyond the original class labels, a surprising result that deserves deeper investigation.

## Suggestions

- Fix the "Avg. gain" summary rows in Table 2 to match the actual per-column data, or clarify precisely what "average" refers to (and over which experimental settings).
- Clarify the bolding convention in tables: if bold means "best baseline excluding LELP," state this explicitly and flag columns where a non-Subclass baseline is actually the best.
- Add even a brief discussion of α>0 results or acknowledge this as a limitation of the current evaluation to guide practitioners.

## Score and Decision

Calibration anchors:
- **High-scoring KD papers**: m50eKHCttz (avg 7.25, Accept spotlight) — general KD transfer with data partitioning, strong large-scale experiments, limited theory. 1aF2D2CPHi (avg 8.0, Accept oral) — data-free KD with strong empirical gains but limited theory.
- **Medium-scoring KD papers**: IcVSKhVpKu (avg 5.67, Accept poster) — CKA-based KD for NLP with practical advantage but limited experimental scope. 9ccZzuix2D (avg 5.33, Reject) — KD for data pruning, simple but effective, but experiment-theory disconnect.
- **Low-scoring KD papers**: p3NVJg6ywM (avg 3.5, Reject) — table data inconsistency flagged explicitly by reviewers. 4NtrMSkvOy (avg 3.0, Reject) — table formatting errors and insufficient experiments.

This paper sits between the medium and high anchors. Like m50eKHCttz, it identifies a real gap, proposes a practical method, and shows strong empirical results. However, the Table 2 data integrity issues (inconsistent "Avg. gain" rows, incorrect bolding in at least one column) place it closer to p3NVJg6ywM on experimental reliability, though the raw accuracy numbers themselves appear plausible and the core method contributions are sound. The paper's genuine contributions — a well-motivated, simple method for an underexplored problem with practical efficiency advantages — are substantial, but the table reporting errors significantly weaken confidence in the experimental validation. This is fixable rather than fatal: the raw numbers can be verified and the summary rows corrected. Relative to the calibration anchors, a paper with this profile (real contribution + significant but fixable presentation/data errors in the main table) warrants a score in the 5-6 range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>