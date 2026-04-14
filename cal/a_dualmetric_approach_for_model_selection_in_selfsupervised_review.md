=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
This paper proposes a dual-metric model selection procedure for self-supervised learning (SSL) in computational pathology, combining task-specific benchmark performance (classification accuracy, segmentation AJI) with task-agnostic representation quality metrics (RankMe, LiDAR, α-ReQ). Nine Vision Transformer encoders are trained on Lung Adenocarcinoma (LUAD) data using DINOv1, and the study demonstrates that optimal downstream performance typically arises well before training converges — a finding with direct practical implications for resource allocation in medical imaging SSL. The selected small-scale, domain-specialized models achieve competitive or superior instance segmentation performance relative to large pan-cancer foundation models.

---

## Strengths

- **Mid-training performance saturation documented in histopathology SSL.** The paper provides consistent, multi-model empirical evidence that SSL training loss continues to decrease while downstream task performance degrades past a certain epoch. This is documented across nine architectures and directly contradicts the prevailing assumption (imported from natural image / NLP practice) that longer SSL training is always better. The finding is practically significant for the community's resource decisions.

- **Differential behavior of rank metrics across task types — a crisp, actionable finding.** The paper clearly shows (Figure 3, Section 5.1) that task-agnostic rank metrics (RankMe, LiDAR, α-ReQ) correlate reasonably with linear-style classification benchmarks but fail as predictors for instance segmentation performance. This is not merely a null result — it identifies a specific failure mode tied to the non-linear nature of segmentation, and it motivates the dual-metric framework in a principled way.

- **Three task-type-specific checkpoint variants are a practically useful design.** Offering classification-best ($e_c^*$), segmentation-best ($e_s^*$), and all-round ($e_a^*$) checkpoints from a single training run addresses a real tension in histopathology practice, where models are expected to support both MIL-based slide classification and instance segmentation. The empirical demonstration that these diverge meaningfully in epoch timing (classification-best occurs substantially later) is concrete.

- **Diverse and well-characterized model zoo.** Nine models varying by backbone size (ViT-S/B), mixture-of-experts capacity (SMoE-4/32/128), and magnification diversity (single vs. multi-scale data) provide substantial variation for evaluating the selection procedure. This breadth strengthens the generalizability of the core finding within the stated scope.

- **Truly held-out evaluation exists.** EGFR classification and LUAD subtyping (Figure 4) were explicitly excluded from the model selection procedure (Section 5.3), providing a genuine generalization test beyond the benchmarks used for selection.

---

## Weaknesses

### Fatal
*(None. The paper's central empirical claim — that mid-training checkpoints outperform final checkpoints in histopathology SSL — is supported across multiple models and tasks.)*

### Major

- **Critical ablation absent: task-agnostic component is unvalidated.** The core claim of the paper is that *combining* task-agnostic and task-specific metrics outperforms task-specific-only selection. However, no ablation compares the proposed dual-metric procedure to a baseline that uses only task-specific benchmark performance across all epochs (i.e., Algorithm 1 with M=0, selecting the epoch that simply maximizes summed normalized benchmark performance). Given the structural design of the algorithm — where Steps 5–6 re-rank candidate epochs using only task-specific metrics — it is entirely possible that the task-agnostic component's main contribution is to prune the candidate set S slightly differently, with negligible impact on the final selected epoch. Without this ablation, the "dual-metric" framing is an untested hypothesis, not a validated contribution.

- **No standard early-stopping baseline.** The paper positions "final checkpoint" as the strawman, but the standard alternative in supervised settings — early stopping based on validation loss or linear probing accuracy monitored during training — is never evaluated. If a practitioner simply stopped training when the loss plateau began, would they arrive at a similarly good checkpoint? The paper needs to show its procedure outperforms this simpler and common practice, not just the naive final-epoch baseline.

- **Small performance differences without statistical significance.** Many cell entries in Table 2 differ by only 0.01–0.02 AJI or F1 units between checkpoint types (e.g., ViT-S 20× PanNuke: 0.46/0.46/0.47; ViT-B MoNuSeg: 0.57/0.57/0.56). Without confidence intervals or significance testing, it is unclear whether these differences are meaningful or within evaluation noise. Figure 4's AUC bar plots similarly lack error bars, making it difficult to assess the practical relevance of the checkpoint selection for the held-out tasks.

- **Single-tissue training scope weakens the generality of training-dynamics claims.** All nine models are trained exclusively on LUAD data. The conclusion that "training for longer is often detrimental to generalization" in histopathology SSL is drawn from a single tissue type, a single SSL method (DINOv1), and one institution's scanner characteristics. Whether this degradation pattern holds for colorectal, breast, or prostate tissue — or for models trained on more diverse staining protocols — is unknown. The paper's scoping in Section 1.3 is appropriate, but the conclusions and abstract should more carefully restrict their generalization claims to LUAD/DINOv1 rather than "histopathology" broadly.

### Minor

- **Structural asymmetry in Algorithm 1 is unanalyzed.** Steps 5–6 re-rank the candidate epoch set S using *only* the sum of task-specific metrics, which means the task-agnostic metrics contribute solely to the composition of S, not to the final ranking. When S is large, this degrades toward a task-specific-only selection. The paper does not analyze how often this asymmetry is consequential (e.g., how frequently do the task-agnostic metrics actually change which epoch is ultimately selected, relative to just maximizing task-specific metrics from the start?). This analysis is directly relevant to validating the dual-metric design.

- **No quantitative correlation analysis between rank metrics and downstream tasks.** Section 5.1 makes visual correlation claims from Figure 3 (scatter plots color-coded by epoch). Adding Pearson/Spearman correlation coefficients between each of RankMe, LiDAR, α-ReQ and each benchmark task across all models would transform a qualitative observation into a quantitative finding, and would help the community understand which rank metrics are worth computing for which task types.

- **Computational cost not quantified.** The paper acknowledges in Section 6 that the selection procedure is expensive because it requires running full benchmark evaluations at every saved checkpoint. However, no concrete estimate is given (e.g., evaluation cost per checkpoint × number of checkpoints, relative to training cost). Practitioners need this information to assess feasibility. The justification for incurring this overhead — rather than using a cheaper proxy — should also be quantified against the performance gain.

### Tiny

- **Min-Max normalization sensitivity.** Min-Max normalization (Steps 1–2) is sensitive to outliers at the extremes of the training trajectory (e.g., very early unstable epochs). No ablation on normalization strategy (e.g., z-score, percentile-based) is provided. This is unlikely to change the main findings but could affect the composition of the candidate set S in edge cases.

- **α-ReQ special treatment buried in footnote.** Footnote 4 describes negating and subtracting 1 from α-ReQ, making it the only metric requiring sign inversion. This ad hoc treatment should be briefly motivated in the main text rather than relegated to a footnote, particularly since the paper cites the known instability of α-ReQ (Section 2.2).

- **Notation inconsistency in Algorithm 1.** The loop variable for MinMaxNormalize is defined as $\forall i \in [1,\ldots,E]$ in Steps 1–2, but the epoch index elsewhere in the algorithm is $e$. This is cosmetically confusing.

---

## Nice-to-Haves

- **Extension to DINOv2 or iBOT.** The authors explicitly scope to DINOv1 for principled reasons (batch-size stability), but a brief check on whether the mid-training saturation pattern holds for a DINOv2 run — even one model, even partially — would substantially increase the community impact of the finding.

- **Joint loss-vs.-performance visualization.** A single figure explicitly overlaying training loss and a representative downstream benchmark metric (e.g., PanNuke AJI) on the same axis across epochs would make the core claim — "loss keeps decreasing while performance degrades" — immediately visually compelling, rather than requiring the reader to compare Figure 1 with Table 2 manually.

- **Analysis of representation degradation mechanism.** The paper documents *that* segmentation performance degrades at late epochs but does not hypothesize *why*. Even a brief analysis (e.g., attention map visualization at early vs. late checkpoints, or feature collapse diagnostics) would strengthen the theoretical grounding of the observation and guide future work on DINOv1 pathology training dynamics.

- **Proxy metric development.** The paper's main practical limitation is that full benchmark evaluation every 5 epochs is expensive. Suggesting or preliminarily evaluating cheaper proxies (e.g., a lightweight linear probe on a small validation subset) would increase the method's practical utility.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Task-agnostic" naming is misleading.** The harsh critic argues the label is inappropriate because benchmark tasks require labeled data. However, the paper clearly defines "task-agnostic" as referring to the representation quality metrics (RankMe, LiDAR, α-ReQ), which are computed from the *unlabeled* pre-training test set (Section 3: "calculated from the test set of the pre-training dataset"). The terminology is internally consistent and well-defined. **Removed.**

- **DINOv1 is "dated."** The paper explicitly justifies this choice in Section 1.3 (training stability at small batch sizes for single-tissue scale) and Section 2.1 (incremental modifications can be introduced later). Criticizing the choice of DINOv1 as a limitation is scope creep given the explicit scoping. **Removed; moved to Nice-to-Have (DINOv2 extension).**

- **MHIST "-" is unexplained.** Table 1 makes clear that single-magnification models (†) train on only 20× or 40× data, while MHIST is a 5× benchmark. The absence of MHIST results for single-mag models is structurally expected. **Removed.**

- **Comparison with foundation models is unfair because of domain specialization.** The paper is explicit (Section 5.2) that the comparison involves "substantially larger datasets" and different scope. The comparison is intentionally included to show that domain-specialized small models + good selection can be competitive with large generalist ones — a practical finding of its own. Moreover, OOD benchmarks (PanNuke, MoNuSeg include non-LUAD tissue) actually disadvantage the LUAD-specialized models, so the asymmetry is not uniformly favorable to the paper's models. **Removed.**

- **"Self-interpreted" phases of development criticism.** The paper itself labels the warm-up / convergence / degradation stages as "self-interpreted" in the Figure 3 caption, which is transparent and honest. **Removed.**

- **Requesting confidence intervals for Table 2 segmentation benchmarks.** Single-run evaluation is standard for PanNuke and MoNuSeg benchmarks in the computational pathology literature; requiring multiple runs would be non-standard for this community. The concern about statistical significance for *held-out* tasks (Figure 4) is retained as a weakness. **Removed from major; concern retained only for Figure 4.**

---

## Novel Insights

The most genuinely novel observation in this work — beyond what is stated in the paper itself — is the **differential timing of classification-best vs. segmentation-best checkpoints**. The paper shows that the epoch maximizing classification benchmarks systematically occurs *later* in training than the epoch maximizing instance segmentation benchmarks. This implies that the SSL training dynamics have qualitatively different effects on linear-separability-relevant representations (which continue improving) versus structural, spatially-grounded representations relevant to segmentation (which peak early). This is not merely a model selection curiosity — it suggests that DINOv1's training objective, as applied to histopathology data, progressively optimizes for globally discriminative features at the expense of fine-grained spatial structure. Whether this reflects a known property of DINOv1 (teacher-student collapse in local features) or something specific to histopathology texture homogeneity is an open and interesting question that the paper identifies but does not pursue.

---

## Suggestions

1. **Run the key ablation:** Implement Algorithm 1 with M=0 (task-specific metrics only, no task-agnostic component) and report which epoch is selected and what performance results. This single ablation either validates the dual-metric contribution or reveals it to be marginal — either outcome is scientifically informative and necessary for the paper's claims to stand.

2. **Add a standard early-stopping baseline:** Monitor validation loss (or a cheap linear probe on a held-out pre-training split) and report the epoch selected by this criterion, alongside its downstream performance, to situate the proposed method against standard practice.

3. **Add quantitative correlation analysis:** Report Spearman correlations between each rank metric and each benchmark task across all nine models, tabulated. This directly supports or refutes the visual claim in Figure 3 and determines which rank metrics are worth including in the algorithm.

4. **Report variance for Figure 4:** For the EGFR task (10 train/test splits), report the standard deviation or a confidence interval over splits rather than pooling all predictions into a single AUC estimate. This is critical for assessing whether the checkpoint-type differences are real or noise.

5. **Quantify the compute cost:** Provide a concrete estimate of the total wall-clock or FLOPs overhead of the benchmark evaluation procedure (e.g., "evaluating all 5 benchmarks every 5 epochs adds X% overhead over base training") so practitioners can assess practical feasibility.

---

**Axis evaluations:**

- **Novelty:** Moderate. The individual components (rank metrics, SSL in pathology, checkpoint selection) are established. The specific combination and the crisp empirical demonstration of mid-training saturation in histopathology are new, but the conceptual leap is limited.
- **Technical soundness:** Weak-to-moderate. The algorithm is a reasonable heuristic, but the absence of the key ablation leaves the core claim that the dual-metric combination outperforms task-specific-only selection unsubstantiated.
- **Empirical support:** Moderate. Good breadth across nine models and multiple benchmarks; undermined by absent ablations, no significance testing on key comparisons, and single-tissue scope.
- **Significance:** Moderate-to-high for the computational pathology community; limited for the broader ML audience typical of ICLR.
- **Clarity:** Good. The paper is clearly written and the scope is honestly stated, though Section 5 interleaves results and interpretation in ways that can obscure what is data-driven versus interpretive.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 1.0]
Average score: 2.5
Binary outcome: Reject
