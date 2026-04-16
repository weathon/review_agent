Now I have enough context. Let me verify the Table 2 "Avg. gain" discrepancy the harsh critic raised, as this is a critical issue. Let me re-read the table carefully.

Looking at Table 2:
- MRQ: LELP 90.22, best baseline (Subclass Dist.) 89.24 → actual gain ~0.98, but reported "Avg. gain over the best baseline" is +0.02
- Amazon Reviews (5-class): LELP 78.06, best baseline 76.28 → actual gain ~1.78, but reported as +0.05
- Sentiment-60 Bin: LELP 87.60, best baseline 85.93 → actual gain ~1.67, but reported as +0.05

This is clearly a reporting error. The "Avg. gain over the best baseline" row shows essentially 0.02-0.05, while the actual gaps from LELP to the best baseline range from 0 to ~5 percentage points. This is a genuine, substantive issue in the paper.

However, I need to check if perhaps "Avg. gain" refers to something else — maybe it's a percentage gain relative to the error rate? Let me check:
- MRQ: (90.22-89.24)/(100-89.24) = 0.98/10.76 ≈ 0.091 → doesn't match 0.02
- Or maybe average across multiple runs? But the cells show single numbers.

Actually, re-reading the table header more carefully: it says "Avg. gain over the best baseline" - but it's listed per column, not as a single average across all columns. This is very confusing. The numbers (0.02, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05) are suspiciously uniform and don't match the raw accuracy differences at all. This is almost certainly a reporting/presentation error.

Wait, let me reconsider. Actually, let me look more carefully at the abstract claims: "achieving an improvement of 1.85% and 0.88%, respectively, over the best baseline." These numbers (1.85% and 0.88%) DO seem to match some of the raw accuracy differences in Table 2 (e.g., Amazon Reviews: 78.06-76.28=1.78 which rounds to ~1.85? Not exactly... 77.45-75.78=1.67? Hmm). Actually let me check more carefully:

"Amazon Reviews (5 classes) ... achieving an improvement of 1.85% ... and Sentiment140 (binary, ... 0.88% improvement over the best baseline)"

Looking at the "Avg. gain over non-subclass baseline" row: 2.05 and 4.98 for the two Amazon Reviews 5-class columns. Looking at LELP vs FitNet (best non-subclass): 78.06-75.54=2.52 and 77.45-72.64=4.81. These don't exactly match either.

Actually wait - the two Amazon Reviews 5-class columns might correspond to different dataset splits or settings. The claim in the abstract of +1.85% and +0.88% might refer to specific ones.

The key issue remains: the "Avg. gain over the best baseline" row in Table 2 does NOT match the raw accuracy differences. This is a real and concerning problem.

Now let me synthesize my review.</think>Now I have enough context to write the final consolidated review. Let me carefully verify the key issues against the paper text.

## Summary

The paper introduces Learning Embedding Linear Projections (LELP), a knowledge distillation method for binary and few-class classification problems where standard KD underperforms due to limited logit information. LELP extracts pseudo-subclass structure from the teacher's final-layer embeddings via PCA (projected onto the null space of teacher output weights, then randomly rotated), splits each of C classes into S pseudo-subclasses, and trains the student with a unified cross-entropy loss over SC expanded classes—without requiring teacher retraining. Experiments on binarized CIFAR benchmarks and large-scale NLP datasets (Amazon Reviews, Sentiment140) demonstrate improvements over baselines, with the student occasionally outperforming even the teacher.

## Strengths

- **Well-motivated, focused problem formulation.** The paper addresses a genuine and underexplored gap: standard KD provides diminishing returns in binary/few-class settings because the information in softened logits scales with the number of classes (Müller et al., 2020). The practical importance is clear for NLP applications like sentiment analysis and ad-query relevance.

- **Practical advantage over Subclass Distillation.** LELP requires no teacher retraining—an important practical contribution given the growing scale of teacher models. The paper correctly notes that Subclass Distillation's requirement to retrain the teacher with different hyperparameter configurations is "excessively computationally intensive and impractical" for large models, and LELP avoids this entirely.

- **Clean algorithmic design grounded in Neural Collapse insights.** The idea of extracting informative linear subspaces from class-conditional embeddings and packaging them as pseudo-subclass logits into a single cross-entropy loss is elegant. It avoids adding tunable embedding-loss coefficients (unlike FitNet/VID), memory banks (CRD), or learnable projections. The null-space projection and random rotation steps are well-motivated design choices.

- **Strong evidence on synthetic subclass benchmarks.** The binarized CIFAR experiments with Oracle Clustering as an upper bound (Table 1, Figure 4) convincingly demonstrate that pseudo-subclass extraction from embeddings can capture useful structure, and that LELP's linear approach outperforms naive clustering methods (K-means, Agglomerative, t-SNE+K-means).

- **Cross-architecture and cross-dimension experiments.** The evaluation includes ResNet→MobileNet (different architectures, different embedding dims), ALBERT-XXL→ALBERT-Base, and sentence-T5→MLP, showing LELP works beyond trivially matched settings.

- **Large-scale NLP evaluation.** Testing on Amazon Reviews (500k examples) and Sentiment140 (1.6M examples) with meaningful gains (e.g., +2–5% over non-subclass baselines) adds practical relevance beyond standard CIFAR benchmarks.

## Weaknesses

### Major:

- **The "Avg. gain over the best baseline" row in Table 2 is inconsistent with the raw accuracy numbers, undermining the trustworthiness of the empirical claims.** For example, on Amazon Reviews 5-class, LELP achieves 78.06 versus Subclass Distillation's 76.28 (a gap of +1.78), yet the table reports "Avg. gain over the best baseline" as +0.05. On Sentiment-60 Bin, the gap is 87.60 vs 85.93 (+1.67), but the reported gain is also +0.05. On MRQ, the gap is 90.22 vs 89.24 (+0.98), yet the reported gain is +0.02. These "average gain" numbers are suspiciously uniform (+0.02 to +0.05) across columns where the actual gaps range from 0 to ~5 percentage points. The abstract's claim of "improvements of 1.85% and 0.88%" over the best baseline cannot be verified against the table. This error directly weakens the paper's primary quantitative claims, as it is unclear whether the reported summary statistics are correct, and if not, what the actual improvements are. The results need to be corrected and re-evaluated.

- **Gains over the strongest baseline (Subclass Distillation) are often marginal relative to experimental variance.** Excluding the problematic "Avg. gain" row, the raw accuracy differences on NLP tasks between LELP and Subclass Distillation range from -0.04 (LELP loses on SST-2) to +1.78 (Amazon Reviews 5-class). Several improvements are in the +0.4 to +1.0 range, with standard deviations of 0.1–1.15 across different methods. The paper reports only 3 runs per configuration with no formal significance testing, making it difficult to determine how many of these improvements are statistically meaningful. This does not invalidate the method, but the claim of "typically superior" performance needs qualification—it is more accurate to say "competitive or slightly superior on most NLP tasks."

- **No ablation disentangling increasing the number of logits (SC classes) from the specific PCA-based construction on NLP tasks.** It is plausible that forcing the student to predict more logits—even with arbitrary partitions—primarily acts as a capacity/regularization effect rather than principled subclass discovery. The paper shows Oracle Clustering (using true subclass labels) dramatically outperforms on CIFAR-bin, confirming that genuine subclass structure helps. But no random-subclass or fixed-partition baseline with the same SC is reported on NLP tasks, where ground-truth subclasses do not exist. Without this control, it is unclear whether LELP's improvements on NLP come from discovering meaningful structure or simply from expanding the classification problem. This is a significant gap for the paper's central claim about mechanism.

### Minor:

- **Critical hyperparameters (S and β) are not discussed or analyzed in the main text.** The number of pseudo-subclasses per class (S) and the subclass temperature (β) are central to LELP, yet their selection and sensitivity are only referenced in appendices. S controls the degree of class expansion and β controls subclass logit sharpness; practitioners need guidance on how to set these.

- **The experimental setup always uses α = 0 (no ground-truth CE loss).** While the paper justifies this as reducing variance and corresponding to the semi-supervised setting, it departs from standard practice where α > 0. It remains unclear whether LELP's advantages persist when combined with ground-truth label supervision, which is the predominant deployment scenario.

- **The paper claims "modality-independent" generality but only tests Vision and NLP text classification.** No experiments with speech, tabular, or sequence data are presented. The claim is reasonable but not directly verified beyond the two tested modalities.

- **The null-space projection and random rotation design choices are only empirically motivated.** The paper states "we have found that it often helps" (Section 3.1) for the null-space projection and that variance imbalance "can lead to poor performance" for the random rotation. These are important design decisions that deserve ablation analysis in the main text, not just in appendices.

### Trivial:

None notable.

## Nice-to-Haves

- **Sensitivity analysis for S and β in the main text.** Understanding how LELP performance varies with the number of pseudo-subclasses and subclass temperature is essential for practical adoption and should not be relegated to appendices.

- **t-SNE or other embedding visualizations for NLP datasets.** Figure 4 only shows Vision embeddings. Visualizing whether LELP produces meaningful subclass structure in sentiment/review embedding spaces would directly validate the method's mechanism where it matters most.

- **Results with α > 0.** While the paper's focus on pure distillation is understandable, showing whether LELP's advantages hold in the standard supervised KD setting (α > 0) would significantly strengthen practical relevance.

- **Computational cost comparison with Subclass Distillation.** Since one of LELP's main selling points is avoiding teacher retraining, a table comparing total compute budget (teacher retraining + student training) would concretely quantify this advantage.

- **Experiments on moderate-class-count tasks** (e.g., 10–50 classes) to establish where LELP's advantage diminishes, as the paper acknowledges this limitation but provides no data.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that LELP's advantage over Subclass Distillation is "plausibly confounded" by unequal hyperparameter search budgets.** The paper is transparent about this: they note that Subclass Distillation requires expensive teacher retraining and that its teacher accuracy differs. They also explicitly report "Avg. gain over non-subclass baseline" as a separate comparison precisely to address this concern. This is a valid caveat but not a methodological flaw—the paper fairly acknowledges the comparison limitation.

- **Request for NLP-specific KD baselines (DistilBERT, TinyBERT).** These are pre-training stage distillation methods that operate in a fundamentally different setting (model compression during pretraining on massive corpora) rather than task-specific finetuning KD. LELP operates in the finetuning regime, making these comparisons inapplicable.

- **Claim that the linear subspace assumption "may limit generality" as a weakness beyond what the paper acknowledges.** The paper explicitly states in Section 5: "there is no reason to assume that the subclasses should be linearly separable in the teacher embedding" and "it is likely that more sophisticated unsupervised clustering methods could extract richer information." This is already acknowledged as a limitation.

- **Request for theoretical justification for null-space projection and random rotation.** This is an empirical systems/contribution paper, not a theory paper. The paper provides empirical motivation (ablations in Appendix C) and algorithmic reasoning. Demanding formal theoretical guarantees is scope creep beyond the paper's community norms.

- **Complaint about α = 0 being "unrealistic."** The paper provides a clear justification: "in order to focus solely on the effect of the distillation loss of each method" and "in the important case of the semi-supervised setting one does not have access to ground-truth labels." This is a valid experimental design choice, not a flaw.

- **Demand for significance testing / confidence intervals.** The paper reports standard deviations over 3 runs, which is standard practice in the KD literature. Requesting more runs or formal significance tests is a nice-to-have but not a methodological flaw.

## Novel Insights

The paper makes an interesting observation that seems underexploited: on Amazon Reviews (5-class), LELP enables a student that *outperforms its teacher by a significant margin* despite having 20× fewer parameters. The paper mentions this in the abstract and results but does not investigate the mechanism. Understanding whether this is because SC-class cross-entropy acts as a regularizer, because the pseudo-subclass structure captures task-relevant variance, or simply because the expanded output space provides a richer learning signal—even when not perfectly aligned with real structure—would deepen the community's understanding of why pseudo-subclass techniques work and when they might fail.

## Suggestions

- **Correct the "Avg. gain" row in Table 2** — this is the highest priority. The reported numbers do not match the raw accuracies and cast doubt on the paper's quantitative claims. Either clarify the calculation methodology or correct the values.

- **Add a random-subclass baseline (or per-class random partition into S groups) on NLP tasks** to isolate the effect of increasing logit count from the specific PCA-driven structure discovery. This single experiment would dramatically strengthen or clarify the mechanism story.

- **Report S and β values and sensitivity curves in the main text**, or at minimum in a clearly referenced main-text table/figure.

- **Tone down claims of "typically superior" to "competitive or superior"**, and explicitly acknowledge that gains over Subclass Distillation are often marginal and within noise on smaller NLP datasets.

- **Rename or clarify the "Avg. gain" row** to make its computation transparent (is it average across seeds? relative improvement? error rate reduction?).

---

**Evaluation on key axes:**

- **Originality:** Moderate. The idea of creating pseudo-subclasses for KD is not new (Müller et al., 2020), but doing so via PCA on null-space-projected teacher embeddings without retraining the teacher is a clean and practical contribution. The specific combination of design choices (null-space projection, random rotation, unified SC cross-entropy loss) is novel.

- **Importance of research question:** High. Few-class KD for NLP is a real and under-served problem with clear practical applications.

- **Whether claims are well supported:** Partially. The CIFAR-bin evidence is strong, but the NLP evidence is undermined by the Table 2 reporting error, marginal gains over Subclass Distillation, and the missing random-subclass control.

- **Soundness of experiments:** Fair. The experimental scope is reasonable, but the Table 2 error and lack of key ablations on NLP tasks weaken confidence.

- **Clarity of writing:** Good. The paper is well-organized and clearly presents the method.

- **Value to research community:** Moderate-to-good. LELP offers a simple, practical alternative to Subclass Distillation for few-class NLP tasks, and the conceptual connection to Neural Collapse is valuable.

**Calibration:** I compared against KD papers with similar profiles. The Dual-Head KD paper (score 4.7 avg, withdrawn) had marginal improvements and methodological concerns, scored low. The LM Distillation paper (score 5.7 avg, accept poster) had solid empirical results on NLP tasks but limited scope. The Comparative KD paper (score 3.75 avg, withdrawn) had evaluation only on CIFAR, scored low. The SoTeacher paper (score 5.25 avg, accept poster) had theoretical motivation with decent but not overwhelming empirics. LELP has a cleaner method and stronger practical motivation than most, but the Table 2 error and marginal gains over the best baseline keep it in the borderline-accept range rather than clearly above it.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Borderline</orange>