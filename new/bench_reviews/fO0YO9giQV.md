Now I have enough information. Let me write the final review.

Key findings from my analysis:

1. **The "AnyECG" universality claim is unsupported**: The paper's title and motivation promise handling "any real-world ECG data" with varying lead configurations, sampling rates, etc. However, all evaluations use standardized 12-lead, resampled-to-300Hz data. The missing channel padding mechanism is described but never tested. This is a significant gap between claims and evidence.

2. **Data leakage concern**: Section 3.1 states "all available unlabeled data" from the datasets is used during pre-training, and the same datasets are then used for downstream evaluation. For small datasets like INCART (74 recordings) this means the model has seen the test-set signals during pre-training. This is a legitimate concern but it's common for self-supervised foundation models to pretrain on all data and then do downstream splits - this is the standard approach (e.g., LaBraM did the same).

3. **Arrhythmia detection results**: All methods achieve 18-34% accuracy. The paper claims "strong ability" at 34.49% accuracy. While this is indeed low in absolute terms, without knowing the number of classes it's hard to say if it's above chance. This is a valid concern about overclaiming.

4. **Table 5 data duplication**: The first several rows of Table 5 (Ultra-Long ECG Recognition) appear to contain EXACT numbers from Table 3 (Arrhythmia Detection), including DENS-ECG, ContraWR values. This is a serious error that undermines the credibility of the ultra-long ECG results.

5. **Baseline comparisons**: Only one other pre-trained ECG foundation model (ECG-FM) is compared, and it's excluded from half the tasks. The paper cites but does not compare against Zhang et al. 2022, Na et al. 2024, Yun et al. 2024.

6. **Pre-processing eliminates heterogeneity**: Resampling everything to 300Hz and padding to 12 leads transforms all data into a homogeneous format, making the "handling heterogeneity" claim untested.

7. **Codebook size K** is not specified in the main text (mentioned in equation but values not given).

8. **Demography decoder requires patient attributes at inference time** — not discussed.

9. **CMA positional tolerance (mask width)** not specified.

10. **No main-text ablation of CMA vs standard attention**.

Now let me think about how to calibrate my score.

Calibration anchors:
- LaBraM (QzTpTRVtrP): ECG's close cousin (EEG foundation model), avg score 7.33, Accept spotlight. LaBraM also has vector-quantized tokenizer, similar two-stage pre-training. LaBraM was critiqued for handling of heterogeneity but still accepted. AnyECG's architectural design is similar but the evaluation has more issues (data duplication in Table 5, low arrhythmia accuracy, overclaimed universality).
- Big Learning (Wa6Ebn4AYL): avg score 5.25, Reject. Overclaimed universality, missing quantitative evaluation.
- EEGPT (wJ6Bx1IYrQ): avg score 4.0, Withdrawn. Data leakage concerns, similar structure to AnyECG.
- LPTM (KJ1w6MzVZw): avg score 3.8, Reject. Train-test overlap undermining claims.
- ECG report generation (vFfVXSP24J): avg score 5.5, Reject. ECG domain paper with baseline/data concerns.
- Population Transformer (FVuqJt3c4L): avg score 7.5, Accept Oral. Self-supervised neural time series model that handles heterogeneity well.

The most relevant anchor is LaBraM (7.33) since it's the closest paper. But AnyECG has more serious issues:
- Table 5 appears to have duplicated data from Table 3, which is a data integrity concern
- Only one foundation model baseline, and it's excluded from half the experiments
- The arrhythmia task results are very low (though this might be the nature of the task)
- The "Any" in AnyECG is overclaimed when all evaluation is on 12-lead 300Hz data

These issues place AnyECG below LaBraM. The data duplication in Table 5 is particularly concerning. Compared to the LPTM paper (3.8, reject) which had train-test overlap, AnyECG has similar data overlap issues but also has the Table 5 data duplication.

I think this paper falls in the 4-5 range. The architecture is reasonable and has some novelty, but the evaluation has serious integrity and claim-evidence gap issues. The Table 5 duplication is a real problem.

Let me set my score at 4.</think>Now I have a thorough understanding of the paper. Let me write my final review.

## Summary

AnyECG introduces a family of ECG foundation models (254M–1.7B parameters) pre-trained in two stages: first, an ECG Tokenizer with a VQ-based rhythm codebook and a Multi-View Synergistic Decoder (morphology, frequency, demography), then a masked rhythm-code prediction phase with a Cross-Mask Attention mechanism. The model is evaluated on four downstream tasks: anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long ECG recognition, claiming to outperform state-of-the-art methods across all tasks.

## Strengths

- **Principled two-stage architecture**: The ECG Tokenizer with VQ rhythm codebook and Multi-View Synergistic Decoder is a well-motivated design that forces tokenized representations to recover clinically relevant information (morphology, frequency, demography) rather than just reconstructing noise. The three complementary decoder losses jointly optimize for clinically meaningful features (Eqs. 3–8).

- **Cross-Mask Attention (CMA)**: The structured sparse attention that limits each patch to attend only to patches from the same lead or same temporal position across leads (Eq. 1, Fig. 1) encodes a reasonable domain prior about ECG lead structure and conduction delays. This is an architecturally meaningful contribution distinguishing AnyECG from standard transformers.

- **Consistent improvements across model scales**: The paper shows a clear scaling trend from AnyECG-B (254M) to AnyECG-XL (1.7B) on anomaly detection (Table 2), supporting the claim that the architecture benefits from increased capacity.

- **Ultra-long ECG handling**: The sliding-window hierarchical approach for ultra-long ECG recognition (Section 3.3) addresses a practically important clinical use case that prior foundation models like ECG-FM cannot handle (noted in Table 5).

## Weaknesses

### Fatal

- **Table 5 (Ultra-Long ECG Recognition) contains duplicated data from Table 3 (Arrhythmia Detection)**: The first two rows of Table 5 (DENS-ECG: 0.3202/0.1514/0.2669/0.2866 and ContraWR: 0.3075/0.1359/0.2802/0.2794) are *exactly identical* to the corresponding rows in Table 3. Row 238 (blank method name with values 0.2011/0.0941/0.1996/0.2018) matches ST-Transformer's Table 3 values. This strongly suggests that at minimum the non-AnyECG rows of Table 5 were copy-pasted from Table 3 rather than computed from ultra-long ECG data, casting serious doubt on the integrity of the ultra-long ECG recognition results—one of the paper's four claimed contributions. This cannot be dismissed as a formatting artifact; it invalidates that experimental table.

### Major

- **The "AnyECG" universality claim is unsupported by the evaluation**: The paper's title and introduction explicitly emphasize handling heterogeneity—varying sampling rates, missing leads, wearable single-lead data (Section 1). Yet every downstream experiment uses standardized 12-lead data resampled to 300Hz. The missing-channel padding mechanism described in Section 2.1 is never tested. Without evaluating on any non-12-lead or non-300Hz ECG, the core framing of "any real-world ECG data" is an overclaim relative to the evidence provided.

- **Pre-training on downstream test data creates information leakage**: Section 3.1 states that "all available unlabeled data" from the listed datasets is used during pre-training, and these same datasets are then used for supervised downstream evaluation. For small datasets like INCART (74 recordings) and even PTB (516 recordings), the model has directly observed the test-set signals during self-supervised pre-training. Since the rhythm codebook learns signal-level patterns, this gives AnyECG an advantage that non-pretrained baselines lack. The paper does not acknowledge this concern or provide any ablation where the model is pretrained on data excluding downstream test recordings.

- **Only one pre-trained ECG foundation model baseline (ECG-FM), excluded from half the tasks**: ECG-FM is the sole contemporary foundation model compared, but it "could not be applied" to lead generation (Table 4) and cannot handle ultra-long ECG (Table 5, absent). Meanwhile, the paper references but does not compare against Zhang et al. 2022, Na et al. 2024, and Yun et al. 2024 in experiments—methods that are directly comparable and would strengthen or weaken the claimed improvements.

- **Arrhythmia detection results are clinically meaningless yet claimed as "strong"**: Table 3 shows all methods achieving ~18–34% accuracy, with AnyECG-XL at 34.49%. The paper states this "demonstrates its strong ability to handle arrhythmia detection effectively." The number of classes and class distribution are never reported, making it impossible to determine whether 34% is meaningfully above chance. Describing ~34% accuracy as "strong ability" is misleading at best.

### Minor

- **Cross-Mask Attention (CMA) is not ablated in the main text**: CMA is presented as a key architectural contribution, but no ablation in Tables 2–5 compares AnyECG with CMA vs. standard self-attention. The paper references ablations in Appendix 7.4, but the main experiment section offers no evidence that CMA itself contributes to the improvements over vanilla attention.

- **Key hyperparameters are unspecified in the main text**: The codebook size K, the CMA positional tolerance (mask width), and batch sizes/fine-tuning procedures are omitted from Section 3.2, reducing reproducibility of the core claims.

- **Demography decoder requires patient attributes whose availability at inference time is not discussed**: The Demography Decoder (Eq. 6) reconstructs age, weight, sex during pre-training, but it is unclear whether these attributes must be available at inference time or whether this decoder is only used as a regularizer during pre-training.

- **Corrupted lead generation loses on MAE**: In Table 4, AnyECG loses to CGAN on MAE (0.0142 vs. 0.0312), the metric most directly measuring point-wise reconstruction fidelity. The paper attributes this to "prioritizing capturing abstract rhythms over pixel accuracy," but this is a post-hoc justification without supporting evidence.

### Trivial

- The undisclosed dataset (10,000 recordings, ~24% of total data) reduces reproducibility, but this is not uncommon in medical foundation model papers and does not invalidate the results.

## Nice-to-Haves

- Evaluation on genuinely heterogeneous ECG data (single-lead, varying sampling rates, missing leads) would substantiate the "AnyECG" framing and is the single most impactful addition the authors could make.
- Comparison with other recent ECG pre-training methods (Zhang et al. 2022, Na et al. 2024, Yun et al. 2024) would strengthen the baseline comparison significantly.
- Reporting the number of classes, class distribution, and per-class metrics for arrhythmia detection would allow readers to contextualize the 34% accuracy figure.
- Analysis of codebook utilization (what fraction of K codes are active, whether collapse occurs) and visualization of what rhythm codes represent clinically would add depth to the tokenizer claims.

## Removed Points

- **"The evaluation only uses 12-lead 300Hz data"** — This is partially addressed by the paper's preprocessing design (padding missing channels, resampling), which creates a mechanism for handling heterogeneity. However, the mechanism is *never tested*, so the criticism is kept but downgraded to Major (unsupported claim) rather than Fatal (the mechanism exists, it just lacks evidence).

- **"Formatting issues in Table 5"** — The harsh reviewer noted a garbled/empty row, but the real problem is much more serious: the row contains duplicated data from Table 3, which is an integrity concern, not a formatting issue. Promoted to Fatal.

- **"Reproducibility concerns about undisclosed hyperparameters"** — Removed per rule: nitpicks about undocumented hyperparameters are minor reproducibility concerns that don't threaten core claims.

- **"Request for ablation of CMA vs. standard attention in main text"** — The paper does reference appendix ablations (Section 7.4), so this is kept as Minor rather than Major.

## Novel Insights

The most striking finding from cross-referencing the paper is that Table 5 appears to contain data from Table 3 re-used as ultra-long ECG baseline numbers for methods like DENS-ECG and ContraWR. This is not a cosmetic formatting issue—it raises questions about whether those baselines were actually run on the ultra-long ECG task, which undermines one of the four claimed experimental contributions. Separately, the arrhythmia detection task, with all methods at 18–34% accuracy, may simply be too difficult or poorly configured for the chosen dataset/evaluation setup, rather than reflecting model competence. The paper's claim of "strong ability" for arrhythmia detection deserves far more nuance, and contextualizing these numbers against chance-level or number of classes is essential.

## Suggestions

1. **Re-run Table 5 baselines properly**: The duplicated rows must be corrected. Ultra-long ECG recognition is a key contribution, and its baselines must be independently computed.
2. **Add one experiment on non-12-lead ECG**: Even a small-scale test on single-lead or varying-lead data would substantially strengthen the "Any" framing.
3. **Report class counts and chance-level for arrhythmia detection**: This takes minimal effort and would prevent readers from dismissing the task entirely.
4. **Acknowledge and discuss the pre-training data overlap concern**: Either provide an ablation excluding downstream test data from pre-training, or explain why this overlap is standard practice (as in LLMs) and why the self-supervised objective makes it less problematic.
5. **Add comparisons with other recent ECG pre-training methods** cited in Section 4 to strengthen the foundation model comparison beyond ECG-FM alone.

## Score and Decision

**Calibration comparison**:

| Anchor | Score | Relevance |
|--------|-------|-----------|
| LaBraM (QzTpTRVtrP) — EEG foundation model, very similar architecture (VQ tokenizer + two-stage) | 7.33 | Closest methodological match; AnyECG has weaker evaluation integrity |
| Big Learning (Wa6Ebn4AYL) — overclaims universality without evidence | 5.25 | Similar overclaim pattern but less severe data issues |
| LPTM (KJ1w6MzVZw) — train-test overlap undermining claims | 3.8 | Similar data leakage concern |
| EEGPT (wJ6Bx1IYrQ) — EEG foundation model with data overlap concerns | 4.0 | Similar domain and data issues |
| Population Transformer (FVuqJt3c4L) — self-supervised neural signal model | 7.5 | Stronger evaluation and methodology |
| ECG report generation (vFfVXSP24J) — ECG LLM with dataset overlap concerns | 5.5 | Same domain, similar overlap issue |

AnyECG's architecture is reasonable and shares DNA with the well-received LaBraM. However, Table 5's data duplication is a serious integrity issue that invalidates one of four experimental contributions. Combined with the unsupported "AnyECG" universality claim, only one foundation model baseline (excluded from half the tasks), and the arrhythmia detection overclaim, this paper falls well below acceptance-quality work like LaBraM (7.33) and closer to papers with data/evaluation concerns like LPTM (3.8) and EEGPT (4.0). The architecture ideas themselves have merit, but the evaluation is not trustworthy in its current form.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>