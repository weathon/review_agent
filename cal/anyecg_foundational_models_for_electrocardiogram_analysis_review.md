=== CALIBRATION EXAMPLE 40 ===

# Final Consolidated Review
Now let me produce the final consolidated review based on my careful reading of the paper.

---

## Summary

AnyECG introduces a family of ECG foundation models (254M–1.7B parameters) built on a two-stage self-supervised pre-training pipeline: (1) an ECG Tokenizer using a VQ-VAE-inspired Rhythm Codebook and Multi-View Synergistic Decoder (morphology, frequency, demography proxy tasks) to produce compact, noise-resilient discrete rhythm codes; (2) a masked token modeling stage over these codes using a novel Cross-Mask Attention mechanism that respects ECG's multi-lead structure. The model is evaluated on anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long ECG recognition, claiming state-of-the-art across all four tasks.

---

## Strengths

- **Multi-View Synergistic Decoder with demographically-aware proxy tasks**: Jointly reconstructing morphology (time-domain), frequency (DWT wavelet coefficients), and patient demographics in the tokenizer pre-training is a genuinely differentiated design for ECG. Most prior ECG SSL methods use only reconstruction or contrastive objectives; coupling demographic recovery to the codebook learning is specific to the medical heterogeneity problem being solved.

- **Cross-Mask Attention (CMA) as a physiologically-motivated inductive bias**: Restricting attention so that each patch can interact only with patches in the same lead or the same temporal position (with positional tolerance to account for conduction delays) is a principled and non-trivial departure from standard full self-attention, directly grounded in the complementary-lead structure of 12-lead ECGs.

- **Hierarchical sliding-window adaptation for ultra-long ECGs**: Extending the architecture to handle ultra-long recordings (30-minute INCART sequences vs. 10-second PTB-XL recordings) is a practically important contribution. The large AnyECG gains in Table 5 (AUROC 0.91 vs. 0.87 for RNN1D) on this genuinely challenging task are among the most convincing results in the paper.

- **Consistent five-seed evaluation with standard deviations**: All reported results are averaged across five random seeds with standard deviations, a rigorous practice that is not universal in ECG ML literature and allows meaningful statistical comparison.

---

## Weaknesses

### Fatal
None that unambiguously invalidate the entire approach, but two major issues—evaluation fairness and apparent result duplication—severely undermine confidence in the reported numbers and must be resolved before the paper's claims can be trusted.

### Major

- **Evaluation fairness: AnyECG pretrained on all evaluation datasets, baselines are not.** The paper states that pretraining uses "all available unlabeled data" from the 7 datasets—the same 7 datasets from which evaluation examples are drawn. No baseline is pretrained on any data. The paper acknowledges ECG-FM's lower performance may be "due to substantial differences between the pre-training data and the downstream task dataset," which implicitly concedes that AnyECG's advantage is partly distributional, not purely architectural. Without a cross-dataset evaluation (pretrain on held-out datasets, test on unseen ones) or at minimum re-training representative baselines on the same pretraining corpus, it is impossible to disentangle architectural advantage from data-distribution advantage. This is the central validity concern for all reported tables.

- **Apparent copy-paste of baseline results from Table 3 into Table 5.** Five baseline rows in Table 5 (Ultra-Long ECG Recognition) have numbers identical to their Table 3 (Arrhythmia Detection) counterparts—including DENS-ECG (0.3202, 0.1514, 0.2669, 0.2866), ContraWR (0.3075, 0.1359, 0.2802, 0.2794), and CNN-Transformer (0.3284, 0.1417, 0.2685, 0.2641)—despite the two tables having different metric columns (Table 3: Accuracy, AUC-PR, Weighted F1, Precision; Table 5: Accuracy, AUC-PR, AUROC, Weighted F1). The AUROC column in Table 5 holds the same value as the Weighted F1 column in Table 3 for these baselines, which is inconsistent with independent measurement. This strongly suggests these rows were copy-pasted rather than independently evaluated on the ultra-long task. Additionally, one row in Table 5 has an entirely blank method name. The authors must confirm whether these baselines were actually run on ultra-long data or defaults were copied.

- **Undisclosed dataset constitutes ~19% of pre-training corpus and appears in evaluation.** Table 1 lists an "Undisclosed Dataset" of 10,000 recordings described only as "Geographically distinct test set." This dataset contributes ~18.8% of the total pre-training corpus and is simultaneously used in evaluation. Independent reviewers cannot verify the data's composition, clinical labels, demographic distribution, or whether there is patient overlap with other included datasets. For a foundation model paper where the pretraining data directly determines downstream performance, this lack of transparency is a material reproducibility failure, not a minor limitation.

- **Notation inconsistency between patch count P and patch size P.** Section 2.1 defines P as the number of patches ("P is the minimal positive integer satisfying P × w ≥ T") and uses w/s for patch size. Section 3.2 then states "we set the patch size P = 300, which corresponds to 1 second of ECG data," conflating P (number of patches) with the patch size in samples. This is not a trivial typo — it prevents unambiguous reconstruction of the tokenization scheme and raises doubt about whether the same symbol refers to the same quantity in the loss functions (Eq. 3, 5, 7).

### Minor

- **Low absolute accuracy in arrhythmia detection (~34%) is unexplained.** AnyECG-XL achieves only 0.345 accuracy and 0.164 AUC-PR on arrhythmia detection, barely above most baselines in absolute terms. The paper never explains the task difficulty: how many classes are there, what is the class distribution, are some arrhythmia types near-absent in the data? Without this context, it is impossible to assess whether 34% represents an informative model or near-random behavior. This is especially important given the paper's claims of clinical utility.

- **Key architectural hyperparameters absent from main paper.** The codebook size K (defined but never instantiated in main text), commitment loss coefficient β (Eq. 7, value not given), and number of DWT decomposition levels Lw (Eq. 5, referenced but not specified) are all left unspecified in the main paper. These are non-trivial hyperparameters for VQ-based models and prevent reproduction of the tokenizer.

- **Ultra-long ECG hierarchical method inadequately described.** Section 3.3 mentions "a sliding window method" in a single sentence but provides no formal description of window size, stride, or aggregation strategy. This is listed as a contribution but lacks the methodological detail needed for reproducibility.

- **Scaling regression is unexplained.** AnyECG-XL (1.7B) achieves worse PSNR (32.43) and MAE (0.0376) than AnyECG-L (32.74, 0.0296) in corrupted lead generation, and scaling from B to XL (7× more parameters) yields <1% improvement in anomaly and arrhythmia detection. The "foundation model family" framing implies beneficial scaling laws, but these results are neither consistent with that claim nor discussed.

- **Patient-level data split not confirmed.** Section 3.2 states an 80/20 training/validation split but does not specify whether splitting is stratified by patient ID. If the same patient's recordings appear in both train and test, within-patient correlations inflate performance estimates.

- **Corrupted lead generation compared only to outdated generative baselines.** CGAN (2014) and WGAN (2018) are the only comparators. ECG-specific lead imputation or reconstruction models are absent, making it difficult to assess whether AnyECG's PSNR/SSIM gains reflect architectural superiority or simply dominating weak baselines.

- **Demography decoder training coverage not specified.** The paper does not state which of the seven datasets provide demographic labels, how many training examples have complete demographic vectors, or how missing demographics are handled in Eq. 6. If only a subset of pretraining data provides demographics, the coverage of challenge (3) is weaker than claimed.

- **Codebook collapse not acknowledged.** VQ-VAE training is well-known to suffer from codebook collapse (most entries go unused). The paper proposes a rhythm codebook as central to its noise-robustness claim but provides no discussion of collapse mitigation strategies or any evidence that the codebook entries are meaningfully utilized.

### Tiny

- "AnyECGs" (plural) appears in one introduction paragraph while the model is called "AnyECG" throughout the rest of the paper — minor inconsistency but conspicuous on first read.
- The wavelet filter w/s notation in Section 2.1 is inconsistent with itself (patch size w vs. stride s used interchangeably in the same paragraph before `w` disappears from subsequent sections).

---

## Nice-to-Haves

- **Low-resource fine-tuning curves**: Plotting downstream performance vs. percentage of labeled data (1%, 5%, 10%, 100%) would provide the most compelling evidence that the foundation model pre-training yields genuine data efficiency.
- **External zero-shot evaluation**: Evaluating on a completely held-out hospital system not represented in the 7 pretraining datasets would provide the cleanest evidence of generalization.
- **Codebook utilization statistics**: Reporting codebook perplexity and per-code usage frequency would substantiate the claim that rhythm codes are semantically diverse.
- **Inference efficiency analysis**: A 1.7B-parameter model warrants a latency/FLOP comparison vs. XResNet1D or other lightweight baselines to help practitioners assess deployment feasibility.
- **t-SNE of codebook token space**: Visualizing whether discrete rhythm codes cluster by clinical diagnosis would add interpretability evidence.
- **Scaling laws plot**: A formal plot of pre-training loss and downstream performance vs. model size and data volume would strengthen the "foundation model family" claim and justify the scaling trajectory.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "300 Hz / Nyquist claim is misleading."** While the Nyquist framing is a simplification, 300 Hz is indeed widely accepted for standard 12-lead ECG diagnosis. The bandpass at 75 Hz is standard per AHA/ESC guidelines. Calling this "misleading" overstates the issue.

- **Harsh critic: CMA is a "severe inductive bias."** While it is true that CMA cannot directly relate cross-lead, cross-time patches in a *single* attention layer, information does propagate across multiple stacked transformer layers. The restriction is a design choice with physiological motivation, not a fundamental failure mode. This weakens but does not eliminate the concern about the absence of an ablation (which remains a minor weakness above).

- **Harsh critic: "Several comparable SSL methods are absent."** The review names ST-MEM, MERL, CLOCS, C-MSCL as missing baselines. This cannot be verified without external sources, and the author may have good reasons for the comparison set used. Per review policy, absent related work is not penalized.

- **Harsh critic: "Unfair that ECG-FM is compared without retraining on the same data."** ECG-FM's lower performance is consistent with domain mismatch, and this comparison is asymmetric *against* AnyECG's claim — it shows the risk of pretraining on different data. The comparison is intentionally conservative toward the baseline and constitutes a valid (if incomplete) experiment.

- **Positive reviewer: Generic "the topic is important" framing** for significance — not included as a standalone strength.

---

## Novel Insights

The combination of a demographically-conditioned VQ codebook with a multi-view proxy-task decoder is a relatively underexplored direction in biomedical SSL. The intuition that demographic information should be *baked into* the tokenizer rather than injected at fine-tuning time is interesting: it forces the discrete rhythm codes to be invariant to demographic confounders rather than merely correlated with them. However, it introduces a risk not discussed in the paper — the demography decoder could encourage the model to encode demographic shortcuts *into* the rhythm codes themselves, potentially making representations less portable across demographically distinct populations rather than more. A systematic demographic-bias audit of the codebook embeddings would be a valuable contribution to the literature beyond this paper's stated scope.

---

## Suggestions

1. **Disentangle pretraining data advantage from architecture**: Run at minimum one ablation where AnyECG is pretrained on held-out datasets and evaluated on unseen ones, or retrain a representative baseline (e.g., ST-Transformer or ECG-FM) on the same pretraining corpus. This is the single most important change for the paper's credibility.

2. **Audit and correct Table 5 baselines**: Explicitly verify whether DENS-ECG, ContraWR, CNN-Transformer, and others were genuinely evaluated on ultra-long ECG data. If these models cannot process long sequences, state so clearly and use a truncation baseline rather than numbers copied from a different table. Also identify the missing method name in row 8 of Table 5.

3. **Address the undisclosed dataset**: Provide detailed summary statistics (demographic distribution, label distribution, recording length distribution) and confirm it does not contain patients also in other datasets. Ideally, report results excluding this dataset so readers can assess its contribution.

4. **Fix notation for P**: Consistently use one symbol for patch size (e.g., s = 300 samples) and another for number of patches (e.g., P), and propagate this correction through all equations and Section 3.2.

5. **Describe arrhythmia detection task**: Add a brief characterization of the INCART arrhythmia detection task — number of classes, class distribution, and a per-class F1 breakdown — so that the 34% accuracy result can be contextualized clinically.

6. **Add key hyperparameters to main text**: Codebook size K, β, Lw, mask ratio, and positional tolerance values should appear either in Section 3.2 or a dedicated hyperparameter table in the main paper.

7. **Formally describe the hierarchical ultra-long approach**: Provide at minimum a paragraph or algorithm box specifying the window size, stride, and token aggregation strategy used for ultra-long ECG recognition.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
