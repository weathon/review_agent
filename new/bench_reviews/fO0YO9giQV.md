---

## Summary

The paper introduces AnyECG, a foundation model family for ECG analysis that employs a two-stage pre-training pipeline: first, a vector-quantized ECG Tokenizer transforms noisy continuous patches into discrete rhythm codes guided by multi-view proxy tasks (morphology, frequency, and demography reconstruction); second, the full model is pre-trained via masked token prediction on a pooled corpus of seven heterogeneous ECG datasets. The approach is evaluated across four downstream clinical tasks—anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long sequence recognition—where it reports state-of-the-art results over specialized baselines and a prior ECG foundation model.

The paper shares striking methodological overlap with LaBraM (QzTpTRVtrP.md, accepted at score 7.33), which also used VQ tokenization + masked code prediction for self-supervised physiological signal pre-training (EEG vs. ECG), evaluated on multiple downstream tasks with data pooling across datasets. AnyECG's novelty hinges on the domain-specific adaptations (Cross-Mask Attention for multi-lead topology, demography proxy task).

## Strengths

- **Coherent architectural pipeline grounded in physiological structure.** The two-stage design—VQ-based tokenization followed by masked token prediction—is conceptually clean and directly adapted from the tokenization + MAE paradigm in CV/NLP to the ECG domain (Sec 1-2). The Multi-View Synergistic Decoder (Sec 2.2.2) decomposes self-supervision into time-domain morphology, multi-scale DWT frequency, and demographic attribute reconstruction, providing a structured pre-training signal beyond standard MSE or contrastive loss. This is well-motivated for a noisy, heterogeneous modality.

- **Domain-specific Cross-Mask Attention with physiological rationale.** CMA (Eq 1, Sec 2.1) restricts cross-lead token interactions to same-position patches (with positional tolerance), incorporating a useful inductive bias about the spatial-temporal topology of multi-lead ECG recordings. This is more parameter-efficient and structurally appropriate than full cross-lead self-attention, particularly when handling variable lead counts.

- **Strong empirical performance across a wide task spectrum.** Tables 2–5 demonstrate AnyECG outperforming both supervised architectures (DENS-ECG, XResNet1D, ST-Transformer) and the prior ECG-FM foundation model across anomaly detection, arrhythmia detection, lead generation, and ultra-long recognition. The largest variant (AnyECG-XL) consistently achieves the best scores across all metrics. Notably, AnyECG succeeds on the ultra-long recognition task where ECG-FM fails entirely due to sequence length constraints (Table 5). Reported standard deviations across 5 random seeds provide statistical support for the gains.

- **Robust handling of data heterogeneity.** The preprocessing pipeline (Sec 2.1) standardizes recordings from 7 datasets with varying sampling rates (1000Hz to 257Hz), lead counts, and durations into a unified token sequence via resampling to 300Hz, temporal padding, and separate temporal/lead positional encodings. This enables joint pre-training on a diverse corpus and demonstrates the model's capacity to absorb heterogeneous inputs.

## Weaknesses

### Fatal
None.

### Major

- **Pooled-dataset evaluation protocol undermines cross-dataset generalization claims (Sec 3.1, Sec 3.2).** The paper states: *"For various downstream tasks, we mixed all datasets together to minimize biases introduced by individual datasets and to better validate the model's generalizability"* (Sec 3.1). This pooling strategy conflates data scale with architectural novelty. If all 7 source datasets (including PTB-XL with 21,837 recordings) are mixed into a single train/val/test split with an 80/20 ratio (Sec 3.2), the evaluation cannot distinguish true zero/few-shot transfer from in-distribution supervised learning on a massive pooled corpus. Patient-level and device-level leakage between the training and test sets—where recordings from the same patient or device may appear in both partitions—is not addressed. This evaluation design means the headline result that *"AnyECG outperforms SOTA methods across tasks"* (Abstract) cannot be attributed to the representation learning pipeline alone; gains may derive simply from significantly more fine-tuning data than baselines received. The closest analogue, LaBraM (avg 7.33), also pooled datasets but evaluated on clearly separated downstream tasks with standard dataset-held-out splits (TUAB, TUEV, etc.), preserving the ability to claim transfer. In contrast, AnyECG's mixed-split protocol is a structural gap in the evaluation design that cannot be resolved post-hoc.

- **Baseline comparisons are confounded by data exposure and model scale (Sec 3.2, Tables 2-5).** AnyECG models range from 254M to 1.7B parameters (Sec 3.2), and are fine-tuned on the pooled dataset described above. The paper does not clarify whether baselines were (a) retrained on the same pooled dataset, (b) evaluated on their original dataset splits, or (c) matched in parameter count. Given that baselines like DENS-ECG, ContraWR, and ST-Transformer are standard architectures not typically trained at half-billion parameter scales, the comparison is inherently asymmetric. Without controls that isolate the contribution of the proposed tokenizer and CMA from raw data/parameter scaling (e.g., training a standard Transformer of comparable size on the identical pooled dataset without VQ tokenization or CMA), the claim that AnyECG's architectural innovations specifically drive the performance gains remains unproven. The LaBraM paper (avg 7.33) similarly compared against architectures of varied sizes but included explicit ablations on pre-training data scale and masking to attribute gains to its design choices. AnyECG lacks such disentanglement experiments.

- **Demography decoder may introduce spurious classification shortcuts (Sec 2.2.2, Eq 6).** The ECG Tokenizer is explicitly trained to predict patient demographic attributes (age, weight, sex) via a supervised decoder head (Eq 6). While the paper notes this is intended to *"better account for inter-patient variability"* (Sec 2.2.2), demographics are well-known confounders in ECG diagnosis—age and sex strongly correlate with arrhythmia prevalence, QRS morphology, and ST-segment baselines. Training the tokenizer to embed demographic information into every token risks encoding patient identity as a shortcut feature, which downstream classification heads may exploit rather than learning genuine pathological signal patterns. The paper provides no ablation comparing downstream performance with and without the demography prediction head, nor does it analyze whether the learned representations are overly dependent on demographic features for classification. This is a potential validity concern for the medical claims, as classifiers relying on demographic shortcuts would not genuinely represent improved ECG understanding.

- **Hyperparameters and training details under-specified, limiting reproducibility (Sec 2.2.1, Sec 2.2.2, Sec 2.3).** The paper cites several critical details only in the appendix: codebook size $K$, commitment loss weight $\beta$ (Eq 6–7), DWT decomposition levels $L_w$, the exact weighting between the five loss terms in $\mathcal{L}_T$ (Eq 8 states they are summed but provides no coefficients), the masking strategy and ratio for Section 2.3 (random contiguous vs. random patches, mask ratio), and the positional tolerance width for CMA (Sec 2.1). While the paper states ablations appear in the appendix, these are not implementation nitpicks—they define the actual pre-training regimen. The paper reports only the Adam learning rate of 1e-4 and that results are averaged over 5 seeds (Sec 3.2). Key details like training epochs, batch size, scheduler, and the relative loss scaling in Eq 8 are missing from the main text. For a foundation model paper claiming generalizability across heterogeneous clinical data, this level of omission is significant and limits the community's ability to assess training stability or to reproduce the results.

### Minor

- **Evaluation metrics for corrupted lead generation lack clinical grounding (Sec 3.3, Table 4).** The lead generation task uses PSNR, SSIM, and MAE as evaluation metrics. PSNR and SSIM are borrowed from image processing and may not correlate well with clinical diagnostic utility (e.g., a generated signal could look visually similar but fail to preserve QRS morphology or ST-segment deviations required for diagnosis). While the paper acknowledges AnyECG does not achieve the lowest MAE and argues this reflects prioritization of rhythmic over fine-grained detail (Sec 3.3), the choice of metrics themselves warrants discussion. MAE in μV/mV should be contextualized against diagnostic thresholds for clinically meaningful features.

- **Ultra-long ECG Recognition task is inadequately defined (Sec 3.3, Table 5).** The paper describes a "hierarchical modeling approach" with a "sliding window method" (Sec 3.3) but provides no detail on the window size, stride, how predictions are aggregated (voting, temporal pooling, majority vote), or what the clinical objective actually is (segment-level rhythm classification, event detection, or full-recording diagnosis?). The Undisclosed Dataset is listed in Table 1 as "Geographically distinct test set" and uses 10000 recordings, suggesting it could be the ultra-long task dataset, but this mapping is not made explicit. The lack of methodological transparency makes the results in Table 5 difficult to interpret relative to the other tasks.

### Trivial

- **Table 3 baseline naming inconsistency.** The header for Table 3 has "Weighted FI Score" (letter I instead of number 1) which is a minor typo. Similarly Table 5 has "Weighted F1 Score" with the correct formatting. The inconsistency is cosmetic but worth noting.

- **Reference to "Patient Attribute Tokenizer" in Figure 1 caption.** The Figure 1 caption states "The Patient Attribute Tokenizer is pre-trained through proxy tasks" whereas the main text consistently uses "ECG Tokenizer" (Sec 2.1, Sec 2.2). The terminology should be made consistent.

## Nice-to-Haves

1. **Attention pattern visualization for CMA.** Visualizing attention weights across leads and temporal positions would concretely demonstrate that CMA captures physiologically meaningful cross-lead interactions (e.g., systematic P/R/T wave timing offsets across leads) rather than merely restricting to same-position tokens.

2. **Scaling law analysis.** Given the three model sizes (254M/500M/1.7B), reporting performance curves across data/compute budgets would be valuable for the ECG community to understand the data requirements for effective foundation model training, paralleling similar analyses in LaBraM and MOTOR.

3. **Failure case and clinical error analysis.** Displaying false positive/negative examples on downstream tasks would help the community understand whether AnyECG's errors reflect clinically meaningful confusions (e.g., similar arrhythmia morphologies) or artifacts of the tokenization scheme.

## Removed Points

**These points are flagged to be removed, treat them with caution:**

1. *From Harsh Critic: "Zero-padding at sequence ends introduces sharp discontinuities that produce high-frequency artifacts after wavelet denoising."* — This is a presentation/processing speculation that the paper does not claim to innovate on (standard signal processing technique); the harsh critic is raising a signal-processing concern that does not invalidate the model's claims. Weakness moved down.

2. *From Harsh Critic: "Codebook size $K$ and commitment loss weight $\beta$ are absent from the main text."* — While true that these details are sparse, the paper states these are in the appendix. Under the hard rules, appendix-deferred details are considered present in the original submission. However, given that these are central loss-function parameters (not trivial hyperparameters), and are referenced in equation form, I have kept a truncated version as a Major point focused on the more critical issue of loss-term weighting and masking strategy details.

3. *From Harsh Critic: "Cross-Mask Attention contradicts electrophysiology because cardiac conduction causes temporal offsets across leads."* — The paper explicitly addresses this: *"a positional tolerance (mask width) is used to improve the model's robustness, accounting for slight delays in certain leads caused by variations in cardiac signal conduction"* (Sec 2.1). The harsh critic did not acknowledge this stated design choice. However, the specific tolerance width parameter value is not reported (it is deferred to the appendix per hard rules), and the paper does not empirically validate that the tolerance aligns with physiological conduction delays. This concern is partially addressed but the lack of empirical validation remains as part of the broader reproducibility gap.

4. *From Strength Finder: "Noise-Resistant Discrete Tokenization via Multi-View Proxy Tasks"* — Kept but trimmed; the claim that VQ tokenization "effectively filters low-SNR artifacts" is asserted without ablation comparing against continuous representations, so the noise-resistance claim is somewhat over-stated. Included as a supporting strength with caveat.

5. *From Harsh Critic: "Missing ablation of the Demography Decoder."* — This is a valid concern, kept in the main review as it directly relates to the potential for demographic shortcut learning, which would undermine the medical validity of the approach.

## Novel Insights

The paper follows a nearly identical architectural pattern to LaBraM (EEG foundation model, accepted at ICLR with score 7.33): VQ tokenization → masked code prediction → multi-task downstream evaluation. This convergence suggests the VQ + masked modeling paradigm is emerging as the de facto standard for self-supervised foundation models in physiological time series. AnyECG's primary differentiators are its domain-specific attention mechanism (CMA) and the demography proxy task. The CMA provides a genuinely novel inductive bias for multi-lead ECG that is absent in EEG models, but its effectiveness is claimed without visualization or ablation. The demography proxy task is conceptually innovative but introduces the risk of demographic confounding, which the paper acknowledges as a benefit without analyzing the potential harm. The pooled-dataset evaluation protocol is the most critical structural concern — in the EEG domain, LaBraM preserved dataset separation in downstream evaluation, while AnyECG mixes datasets into a single train/test split, fundamentally altering what can be claimed.

## Suggestions

1. **Adopt dataset-held-out evaluation.** At minimum, hold out one or two entire datasets from both pre-training and fine-tuning (e.g., the Undisclosed Dataset or INCART) and evaluate on these unseen sources to substantiate generalization claims.

2. **Include a parameter-matched, architecture-matched baseline.** Train a standard Transformer (no VQ tokenizer, no CMA, full self-attention) on the identical pooled dataset and training budget to isolate whether performance gains come from the architecture or simply from scale and data volume.

3. **Demography decoder ablation.** Report downstream performance when the demography prediction head is ablated or replaced with a random/demography-agnostic signal to verify that classification is not driven by demographic shortcuts.

4. **Clarify baseline training conditions.** Explicitly document whether each baseline in Tables 2–5 was retrained on the pooled dataset or evaluated on its original split, and report the number of trainable parameters for each baseline model.

5. **Add a clinical error metric for lead generation.** Supplement PSNR/SSIM with clinically meaningful metrics (e.g., QRS onset/offset timing error, ST-segment deviation error) to bridge the gap between signal fidelity and diagnostic utility.

## Score and Decision

Calibration was performed against the following anchor papers:

| Anchor Paper | Avg Human Score | Comparison to AnyECG |
|---|---|---|
| **QzTpTRVtrP.md (LaBraM)** | 7.33 (Accept Spotlight) | Most directly comparable: same VQ+MAE paradigm, physiological signal (EEG vs. ECG), multi-dataset pre-training. LaBraM had cleaner downstream evaluation with dataset-held-out splits and provided more thorough ablations. AnyECG has comparable architecture novelty but weaker evaluation design. |
| **NialiwI2V6.md (MOTOR)** | 7.50 (Accept Spotlight) | Foundation model for medical records with 19-task evaluation, external dataset validation, and careful leakage controls. Much stronger empirical validation than AnyECG. |
| **PdaPky8MUn.md** | 8.00 (Accept Oral) | Pre-training demonstration paper with decisive empirical evidence. Sets the high bar for foundation model papers in medical/signal domains. |
| **IRL9wUiwab.md** | 6.00 (Reject) | Time-series foundation model paper that was rejected despite interesting methodology due to insufficient experimental validation. Relevant as a cautionary anchor. |
| **oVCVCo3laS.md (DualTime)** | 5.20 (Reject) | Medical time-series model with good results undermined by unfair baseline comparisons (frozen baselines), limited datasets, and missing ablations. Highly analogous weakness pattern to AnyECG's confounded baselines. |
| **tfTGSm31F7.md (EpilepsyFM)** | 4.00 (Reject) | Medical foundation model rejected for insufficient novelty over prior work and limited empirical support. |
| **85Ik12q2hP.md** | 4.00 (Withdrawn) | Incomplete analysis to support claims, results unsurprising given prior work. |
| **vFfVXSP24J.md** | 5.50 (Reject) | ECG-specific paper with mixed reviews, rejected for insufficient novelty and limited evaluation scope. |

AnyECG sits between the LaBraM/MOTOR cluster (7–8, accepted) and the DualTime/EpilepsyFM cluster (4–5.5, rejected). It has genuine architectural novelty (CMA, multi-view decoder) and strong empirical results across 4 tasks, which aligns it with the accepted papers. However, its pooled-dataset evaluation protocol, confounded baseline comparisons, and under-specified training details are significant methodological gaps that LaBraM avoided. The evaluation protocol concern is the most serious: mixing all datasets into a single split means the "foundation model" claim is not actually tested.

Positioned relative to anchors: AnyECG is weaker than LaBraM (7.33) on evaluation rigor and ablation completeness but stronger than DualTime (5.20) in results magnitude and scope. It aligns roughly with IRL9wUiwab's range (6.0) — an interesting methodology with insufficient validation — though AnyECG's empirical results are stronger. The evaluation design flaw is significant but doesn't entirely invalidate the methodology contributions. A score of **5.0** reflects a borderline paper with real contributions (the tokenizer/CMA designs are worth the community's attention) undermined by serious evaluation and clarity gaps that limit confidence in the claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>