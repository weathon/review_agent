Now I have enough context for calibration. Let me synthesize the review.

Key calibration anchors:
- TA-PCLR (ECG FM): rejected, scores 1-6, weak methodology/evaluation
- ST-MEM (ECG SSL): accepted poster, scores 6-8, good but narrow
- FORMED (MedTS FM): rejected, scores 3-6, overclaimed generalization
- Wearable Biosignal FM: accepted poster, scores 5-8, strong empirical work
- bioFAME (Biosignal pretraining): rejected, scores 3-6

Now let me write the consolidated review.</think>Now I have enough calibration context. Let me write the final review.

## Summary
AnyECG proposes a foundation model family for ECG analysis that handles heterogeneous ECG data through a two-stage pre-training pipeline: (1) an ECG Tokenizer with a rhythm codebook and multi-view synergistic decoder (morphology, frequency, demography) that produces noise-resilient discrete tokens, and (2) masked modeling with a cross-mask attention (CMA) mechanism that learns rhythm-event associations. The model is evaluated on four downstream tasks (anomaly detection, arrhythmia detection, corrupted lead generation, ultra-long ECG recognition) and claims to significantly outperform existing methods.

## Strengths
- **Coherent and principled architecture targeting real ECG challenges**: The design systematically addresses four identified challenges (heterogeneity, low SNR, demographic shift, rhythm-event association) with corresponding components (unified patchification, VQ codebook, demography decoder, CMA). The multi-view synergistic decoder and cross-mask attention are physiologically motivated and technically sound.
- **Multi-task empirical evaluation**: The paper evaluates across four diverse downstream tasks including both classification and generation, going beyond single-task ECG papers. The consistent improvements across model scales (B, L, XL) with 5-seed reporting demonstrate engineering rigor.
- **Scalability demonstration**: Three model scales (254M, 500M, 1.7B parameters) show generally positive scaling behavior, supporting the foundation model framing.

## Weaknesses

### Major

- **Core "any ECG" / generalization claim is unsupported by the evaluation design.** The paper's central claim—to handle "any real-world ECG data" and generalize across devices, sampling rates, and demographics—is not actually tested. Section 3.1 states all seven datasets are mixed for both pretraining and downstream tasks (80/20 split), meaning *all test data is in-distribution with respect to pretraining sources*. No leave-one-dataset-out experiment, cross-device evaluation, or held-out demographic group analysis is conducted. The "Undisclosed Dataset" is also included in the pretraining pool despite being described as "geographically distinct." The paper thus demonstrates that a big model pretrained and fine-tuned on a large pooled dataset performs well on the same pool—not that it generalizes to new devices, populations, or recording scenarios. This is a structural misalignment between the foundational claims and the evidence.

- **Unfair baseline comparisons undermine performance claims.** The comparisons are asymmetric in AnyECG's favor across multiple dimensions: (i) AnyECG is pretrained on all downstream data sources, while baselines appear to be trained from scratch—confounding pretraining benefit with architectural innovation; (ii) ECG-FM (the only other pretrained baseline) underperforms, but the paper itself speculates this is because of pretrain-test domain gap, while AnyECG has no such gap, making the comparison uninformative about architectural merits; (iii) in ultra-long ECG (Table 5), baselines receive no hierarchical adaptation ("Adaptation ✗") while AnyECG gets a custom sliding-window method ("Adaptation ✓"), conflating architectural support with pretraining benefit; (iv) with AnyECG-XL at 1.7B parameters, there is no capacity-matched non-pretrained transformer baseline to disentangle scale from architectural innovation. Without these controls, it is impossible to attribute gains to the proposed ECG Tokenizer, CMA, or multi-view decoder versus mere scale and in-distribution pretraining.

- **Task definitions and label provenance are underspecified, making results hard to interpret.** The paper does not define what "anomaly detection" vs. "arrhythmia detection" are in terms of label sets, number of classes, or how SNOMED-CT codes from heterogeneous datasets were harmonized. Arrhythmia detection accuracy of ~34% (Table 3) is extremely low in absolute terms and left unexplained. It is unclear whether this reflects a difficult multi-class problem, noisy merged labels, or a flawed experimental setup. The lack of patient-level splitting information raises potential data leakage concerns (especially with 30-minute INCART recordings). Without task/task-specific definitions, results cannot be compared to prior work or assessed for clinical significance.

### Minor

- **Demography decoder effectiveness is claimed but not evaluated.** The paper claims the demography decoder "improves generalization across diverse populations" and "captures demographic-related pattern shifts," but provides no quantitative analysis of demographic prediction performance, no subgroup analysis, and no comparison between models with/without the demography decoder on cross-demographic generalization.

- **Ultra-long ECG evaluation has inconsistencies.** Table 5 contains duplicated rows from Table 3 (DENS-ECG, ContraWR values are identical), a malformed row with no model name, and implausibly low baseline values. The sliding-window method for ultra-long sequences is mentioned but not specified (window length, stride, aggregation). These issues undermine confidence in this task's results.

- **Missing key architectural details in the main text.** The codebook size K, masking ratio and strategy, commitment loss coefficient β, and positional tolerance values are either absent or deferred entirely to the appendix. These are important for understanding model capacity and reproducibility.

## Nice-to-Haves
- Leave-one-dataset-out evaluation or cross-device/cross-demographic analysis to substantiate the generalization claim.
- Visualization of rhythm codebook entries and t-SNE/UMAP of learned embeddings to demonstrate that codes capture clinically meaningful patterns.
- Comparison against stronger contemporary ECG SSL baselines (e.g., ST-MEM, PCLR, MAEFE) and capacity-matched transformer baselines without pretraining.
- Analysis of codebook utilization and collapse (how many of K codes are actively used).

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Undisclosed dataset undermines reproducibility/trustworthiness"** (from Spark and Harsh Critic): Per instructions, if the paper cites a dataset, it is assumed to exist. The concern about undisclosed data is primarily a reproducibility issue, not a fundamental flaw. However, the more substantive issue—whether including this dataset in pretraining vitiates generalization claims—is retained above.

- **"Missing baselines like CLOCS, PTP, MERL, HeartLang/QRS-Tokenizer"** (from Spark and Human Finder): Per instructions, I should not flag missing related work as I cannot confirm existence or relevance of specific uncited works. The broader concern about baseline quality is addressed above.

- **"Codebook size K and masking ratio are missing from main text"** (partially from Spark): This is a minor reproducibility nitpick per the hard rules—moved to minor. Hyperparameters absent from main text but referenced in appendix are not a fatal flaw.

- **"The novelty of the tokenizer is questionable vs. HeartLang/QRS-Tokenizer"** (from Human Finder): This is a missing related work concern. Per instructions, I should not mention missing related works. The novelty assessment is appropriately left to readers.

- **"Low absolute performance on arrhythmia detection (~34%) undermines clinical utility"** (from Human Finder): While the absolute scores are concerning, this is better framed as a task definition problem (which I keep under Major #3) rather than a standalone weakness about clinical utility per se, since the paper does not make explicit clinical deployment claims beyond the tasks evaluated.

- **"No statistical testing across seeds"** per the soft rules: Single-run + 5-seed std is standard in this area; demanding confidence intervals is a nice-to-have, not a core weakness.

## Novel Insights
The paper's most interesting architectural contribution is the combination of a VQ codebook with physiologically-motivated multi-view reconstruction (morphology + frequency + demography) specifically for ECG, forcing the tokenizer to extract clinically grounded discrete codes rather than generic reconstructions. The cross-mask attention restricting interaction to same-lead and same-position patches is also a sensible inductive bias for multi-lead physiological signals. However, these contributions remain inadequately validated due to the evaluation design issues identified above.

## Suggestions
- Add at least one leave-one-dataset-out experiment to test whether the model actually generalizes to unseen data sources, which is the core claim.
- Define downstream tasks precisely (label mappings, number of classes, class distributions) and clarify whether splitting is patient-level.
- Run a baseline comparison where a comparable-size Transformer is trained supervised on the same pooled data (no tokenizer, no pretraining), with and without the sliding-window adaptation, to isolate the contributions of pretraining and architectural design.
- Provide CMA vs. standard attention ablation results.

## Score and Decision

**Calibration**: Compared to TA-PCLR (ECG FM, rejected, scores 1-6, weak evaluation and overclaimed generalization), AnyECG has a more sophisticated architecture and broader task evaluation. Compared to ST-MEM (ECG SSL, accepted poster, scores 6-8), AnyECG tackles a harder, more ambitious problem but has much weaker evaluation rigor—ST-MEM validates on standard benchmarks with clear task definitions. Compared to FORMED (medical time series FM, rejected, scores 3-6, overclaimed generalization without proper out-of-domain testing), AnyECG has similar structural issues with generalization claims but more technical novelty. The biosignals foundation model paper (scores 5-8, accepted poster) had stronger evaluation design with explicit demographic analysis.

The core issue is that AnyECG's central "generalization across any ECG data" claim is structurally unsupported—all evaluation is in-distribution. This is compounded by unfair baseline comparisons and underspecified task definitions. These are not minor gaps but fundamental misalignments between claims and evidence. The technical contributions (CMA, multi-view decoder, rhythm codebook) are interesting and well-motivated, but without proper evaluation, their true impact cannot be assessed.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>