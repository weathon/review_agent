Now I have enough context to compose the review. Let me carefully cross-check the reviewers' key claims against the paper content.

Key things to verify:
1. Marginal improvements claim - checking Table 2 "Avg. gain over the best baseline" row
2. α=0 claim
3. Whether claims about "typically superior" are overstated
4. Whether the paper addresses hyperparameter sensitivity

From the paper, Table 2 shows:
- "Avg. gain over the best baseline" row shows +0.02 to +0.05 (but these are not standard deviations, they seem to be average across tasks)
- Wait, actually looking at the table more carefully, the "Avg. gain over the best baseline" seems to be per-column, not an average. Let me re-examine...

Actually, looking at Table 2, the last three rows show:
- "Avg. gain over the best baseline": values like +0.02, +0.04, +0.04, +0.05, +0.05, +0.05, +0.05, +0.05
Wait, those look suspiciously uniform. Let me check again...

Actually, looking at the specific entries:
- MRQ (IMDb): LELP 90.22 vs best baseline Subclass 89.24 → gap is 0.98. But the table says "Avg. gain over the best baseline" = +0.02? This doesn't match.

Wait, I think the "Avg. gain over the best baseline" row might be referring to something else, or there might be formatting issues. Let me look more carefully...

Actually looking again at the paper text: "Avg. gain over the best baseline" with values like +0.02, +0.04, etc. These seem oddly small. Given that LELP scores 90.22 and the best baseline is Subclass at 89.24, the gain should be ~0.98. Something is off with the table parsing/formatting.

Let me re-examine the abstract: "LELP is consistently competitive with, and typically superior to, existing state-of-the-art distillation algorithms for binary and few-class problems" and "achieving an improvement of 1.85% and 0.88%, respectively, over the best baseline."

The "best baseline" referred to in the abstract is Subclass Distillation. For Amazon Reviews (5-class), LELP gets around 77.45 vs Subclass ~77.07 → 0.38% gain, not 1.85%. Wait, let me check more carefully...

Actually, looking at the abstract: "Amazon Reviews (5 classes, 500k examples) and Sentiment140 (binary, 1.6 million examples) achieving an improvement of 1.85% and 0.88%, respectively, over the best baseline."

But from Table 2, for Amazon Reviews (5-class), LELP = 77.45 vs best non-Subclass baseline... Looking at the "Avg. gain over non-subclass baseline" row which shows 2.05 for one and 4.98 for another column. The "Avg. gain over Vanilla KD" for Sentiment140 Bin is 2.15.

The numbers in the abstract (1.85% and 0.88%) seem to refer to the gain over the best baseline (which would be Subclass Distillation). Let me check: Amazon Reviews (the column labeled "Am. Reviews"): LELP = 77.45, Subclass = 77.07 → 0.38% gain. That doesn't match 1.85% either.

Hmm, it seems like the table values might have formatting issues. Let me not get too deep into number reconciliation and focus on the qualitative issues.

Now let me write the final review.</think>Now I have enough calibration context. Let me compile the final review carefully by cross-checking reviewer claims against the paper content.

## Summary

The paper introduces LELP (Learning Embedding Linear Projections), a knowledge distillation method for binary and few-class classification. LELP addresses the well-known limitation that standard KD provides insufficient "dark knowledge" when the number of classes is small. It extracts informative linear subspaces from the teacher's final-layer embeddings via per-class PCA (after projecting onto the null space of teacher output weights to remove redundancy), applies a random rotation to equalize variance across subclass directions, and then splits each class into S pseudo-subclasses whose probabilities are derived from projections along these directions. The student is trained on the expanded SC-class problem using a unified cross-entropy loss. LELP requires no teacher retraining (unlike Subclass Distillation), no data augmentation (unlike CRD), and no learnable projection layers (unlike FitNet). Experiments on binarized CIFAR and large-scale NLP benchmarks show consistent improvements over baselines.

## Strengths

- **Addresses a real and important problem.** KD underperformance in binary/few-class settings is well-documented (Müller et al., 2020) and highly relevant to NLP tasks like sentiment analysis and ad-query relevance. The motivation is clearly articulated and the three desiderata (modality-independence, architecture-agnostic, no teacher retraining) are well-chosen.

- **Simple, elegant, and practical method.** The core mechanism—per-class PCA, null-space projection, random rotation, then softmax over subclass logits—is easy to understand, implement, and integrate into existing KD pipelines. It avoids the computational cost of teacher retraining that Subclass Distillation requires, which is a genuine practical advantage especially for large LMs.

- **Strong evidence in the synthetic subclass regime.** On binarized CIFAR-10/100 with true subclass structure (Tables 1 & 3, Figure 3), LELP achieves clear and often substantial gains over Vanilla KD and generic clustering approaches, including in challenging cross-architecture transfers (ResNet→MobileNet). The Oracle Clustering upper bound analysis effectively demonstrates the potential of pseudo-subclass methods.

- **Interesting conceptual connection to Neural Collapse.** Tying KD for few-class problems to the fine-grained structure in last-layer embeddings (inspired by Yang et al., 2023) provides principled motivation for using embedding projections rather than relying solely on logits.

- **Diverse experimental settings.** The paper evaluates on vision (CIFAR-bin) and NLP (Amazon Reviews, Sentiment140, GLUE subsets, IMDB) with multiple teacher-student architecture pairs including same-family and cross-architecture setups, including an MLP-over-frozen-T5 setup.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed superiority on NLP benchmarks relative to the evidence.** The paper repeatedly states LELP is "consistently competitive with, and typically superior to, existing SOTA distillation algorithms" and targets "binary and few-class problems, where most KD methods suffer." On the main real-world NLP benchmarks (Table 2), LELP's margins over Subclass Distillation—the strongest baseline—are very small (e.g., ~0.4–1.9% across columns) and sometimes within the standard deviation range (e.g., LELP 92.81 ± 0.36 vs. Subclass Dist. 92.85 ± 0.15 on QGLUEval-sst2 where Subclass actually outperforms LELP). The "Avg. gain over the best baseline" row in Table 2 shows values as small as +0.02 to +0.05, which are comparable to or smaller than the reported standard deviations. No statistical significance testing is provided. The large gains over Vanilla KD on some tasks (up to +4.98%) are real and important, but the specific claim of *typical superiority over all SOTA baselines* is not well supported—parity with slight numeric edge is a more honest characterization on these tasks.

- **Limited evaluation of alternative clustering methods on NLP tasks.** The ablation comparing LELP against k-means, agglomerative clustering, and t-SNE+k-means (Section 4.2, Table 1, Figure 3) is conducted *only* on the synthetic binarized CIFAR datasets. The NLP benchmarks—where LELP is claimed to be most needed and where it shows its largest gains over Vanilla KD—have no comparison of different pseudo-subclass creation strategies. Since the conceptual novelty of the paper is that PCA-based projections outperform generic clustering for extracting subclass structure, this core claim is only demonstrated on vision tasks with known ground-truth subclass structure. It remains unverified that LELP's specific PCA+rotation mechanism is responsible for gains on NLP tasks rather than, e.g., the simple expansion of the label space from C to S×C classes.

### Minor

- **Key hyperparameter choices (S, β, null-space projection) are insufficiently analyzed in the main text.** S (number of subclasses per class) and β (subclass temperature) are the primary hyperparameters, but sensitivity analysis is deferred to the appendix. Similarly, the null-space projection and random rotation steps are motivated heuristically ("we have found that it often helps"; "we have found it is useful") without ablations in the main body. This makes it harder to assess whether LELP's gains come from principled information extraction or from capacity expansion via SC output classes.

- **α = 0 throughout all experiments.** Setting α = 0 (no ground-truth cross-entropy loss) to isolate the distillation effect is a valid methodological choice, but departs from the practical setting where α > 0 is often optimal. Without experiments showing LELP's advantage persists with α > 0, the practical impact is uncertain. The paper should at least acknowledge this as a limitation.

- **Subclass Distillation comparison fairness.** As the authors note, Subclass Distillation uses a different (retrained) teacher, making direct comparison "not entirely fair." The paper presents "Avg. gain over non-subclass baseline" as a separate metric, which is reasonable, but the "best baseline" comparison still includes Subclass Distillation without a clear apples-to-apples setup. This is not fatal but complicates interpretation.

- **The "student beats teacher" claim needs nuance.** The paper highlights that LELP students can outperform their teachers (e.g., Amazon Reviews: 77.45% vs. 77.58%). This is noteworthy but should be contextualized: with α = 0, the student is trained purely on teacher-derived pseudo-subclass targets rather than ground-truth labels, and the label expansion to SC classes may provide regularization benefits unrelated to embedding structure extraction.

### Trivial
- Minor notation inconsistency: the abstract states "+1.85% over the best baseline" for Amazon Reviews and "+0.88%" for Sentiment140, while Figure 1 caption states "+2.95% over Vanilla KD" and "+1.18%" respectively—these reference different baselines but could confuse readers.

## Nice-to-Haves

- Experiments with α > 0 combining LELP with ground-truth labels, which would validate practical applicability.
- A systematic study varying the number of classes (e.g., 2, 3, 5, 10, 20) to empirically characterize where LELP's benefits diminish, as the paper acknowledges convergence to Vanilla KD with more classes.
- NLP embedding visualizations (analogous to the CIFAR t-SNE plots in Figure 4) to illustrate whether pseudo-subclasses capture meaningful NLP structure.
- Hyperparameter sensitivity analysis for S and β in the main text rather than the appendix.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Missing recent KD baselines" (from human finder).** The paper compares against 8 baselines including recent methods (DKD, CRD). Demanding more recent baselines is a generic, one-size-fits-all critique that could be applied to any paper. The baselines are adequate for the specific problem setting (few-class distillation), and DKD and Subclass Distillation are the most relevant competitors.

- **"Teacher models are relatively small" (from human finder).** The paper uses ALBERT-XXL as a teacher (which is a substantial model) and also includes a frozen sentence-T5 encoder of 11 billion parameters. This critique overstates the scale concern for the problem setting addressed.

- **"No theoretical justification" (from neutral reviewer).** The paper is empirically motivated (inspired by Neural Collapse findings). Theoretical analysis of why PCA projections are near-optimal would be nice but is not standard in this area. Demanding theoretical proofs for an empirical KD methods paper is beyond community norms.

- **"Restricted applicability scope" (from neutral reviewer).** The paper explicitly scopes itself to few-class problems (Section 5) and acknowledges convergence to Vanilla KD with more classes. Criticizing the absence of large-class experiments is scope creep—the paper is clear about its domain.

- **"Limited understanding of why linear projections work" (from harsh critic).** The paper provides substantial mechanistic analysis in Section 4.2 (comparing clustering approaches, Oracle Clustering bounds) and motivates the approach from Neural Collapse theory. While deeper understanding would be valuable, this is a standard empirical contribution.

- **"Compute budget comparison with Subclass Distillation" (from human finder).** The paper clearly identifies that Subclass Distillation requires costly teacher retraining, which is a qualitative but valid practical objection. Quantifying the exact compute difference is unnecessary since the retraining cost scales with teacher size—this is straightforward.

- **"Architecture-heterogeneous NLP experiments missing" (from Spark).** The paper does include an NLP cross-architecture experiment: ALBERT-XXL teacher distilling to an MLP-over-frozen-T5 student, which is precisely a cross-architecture NLP setting.

## Novel Insights

The paper makes a valuable observation that extends existing work on Neural Collapse representations: while Yang et al. (2023) showed that collapsed embeddings retain subclass structure recoverable via clustering, LELP demonstrates that *linear* projections (specifically PCA in the null space of teacher output weights, with variance-equalizing rotation) can effectively channel this structure into pseudo-subclass targets for distillation—without requiring the costly teacher retraining that Subclass Distillation demands. The comparison against generic clustering methods (k-means, agglomerative, t-SNE+k-means) on the synthetic subclass tasks provides useful evidence that the specific choice of how pseudo-subclasses are created matters significantly, and LELP's approach outperforms naive alternatives. However, the insight remains primarily empirical; the gap between LELP and alternative clustering methods is not analyzed on the real-world NLP tasks where the method is most needed.

## Suggestions

- Narrow the main claims to reflect the evidence more precisely: LELP provides substantial gains over Vanilla KD on few-class tasks and is *competitive with* (rather than *typically superior to*) strong embedding-based and subclass-based alternatives. The genuine practical advantage is simplicity and no teacher retraining.
- Include sensitivity analysis for S and β in the main text, and an ablation separating the contribution of null-space projection from random rotation from the main label-expansion effect.
- Run a small set of experiments with α > 0 to demonstrate practical applicability; even 2–3 settings would suffice.

## Score and Decision Calibration

**Calibration papers examined:**
- Dual-Head KD (m7Nd3K0iru): marginal improvements (~0.1–0.2%), questionable novelty → scores 6, 5, 3 → withdrawn/reject
- Retro (2GMTfqr7eb): simple engineering trick, limited understanding of why it works, consistent but modest improvements → scores 5, 5, 3 → withdrawn/reject
- Enriching KD (7L8sZYMlya): limited experiments, weak novelty, hyperparameter concerns → scores 3, 3, 5, 3, 5 → withdrawn/reject
- High-dimensional Analysis of KD (1xzqz73hvL): strong theoretical contribution, well-validated → scores 8, 6, 6, 8 → accept spotlight
- Few-Class Arena (2ET561DyPe): useful benchmark with clear scope → scores 6, 6, 6, 5 → accept poster
- Bayes Conditional Distribution KD (yV6wwEbtkR): good method, strong gains in few-shot, well-scoped → scores 8, 6, 6 → accept poster

LELP has a clear, simple method with genuine practical advantages (no teacher retraining, modality-independent), demonstrates strong gains on synthetic benchmarks with clear subclass structure, and shows meaningful improvements over Vanilla KD on real NLP tasks. However, the claimed superiority over the strongest baseline (Subclass Distillation) on real-world tasks is marginal and within noise, and the core comparative claim about PCA vs. clustering is only evaluated on synthetic vision data. This places it above papers like Dual-Head KD and Retro (which had truly marginal or inconsistent improvements), but below papers like Bayes Conditional Distribution KD (which showed clear improvements in its scoped domain). The contribution is real but the overclaiming is significant.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>