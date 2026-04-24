## Summary

AnyECG proposes a family of ECG foundation models (up to 1.7B parameters) pre-trained on seven diverse ECG datasets via a two-stage pipeline: a vector-quantized ECG tokenizer with a multi-view decoder (morphology, frequency, demography), followed by masked discrete-token prediction with a Cross-Mask Attention transformer. The paper evaluates on anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long ECG recognition, claiming state-of-the-art results and universal generalization across devices and scenarios.

## Strengths

- **Novel two-stage architectural design.** The paper introduces a VQ-based Rhythm Codebook with cosine-similarity quantization (Eq. 2) and a Multi-View Synergistic Decoder that jointly reconstructs morphology, wavelet frequency coefficients, and patient demographics (Eq. 3–8). This is a genuinely new adaptation of BEiT-style masked modeling to multi-lead ECG data.
- **ECG-specific inductive bias via Cross-Mask Attention.** CMA (Eq. 1) restricts attention to same-lead or same-temporal-position patches with positional tolerance for conduction delays, which is a domain-appropriate structural prior unlike generic full self-attention.
- **Scale and engineering ambition.** Compiling seven heterogeneous datasets and training models up to 1.7B parameters represents a notable engineering effort for the cardiology domain. The corrupted lead generation (Table 4) and anomaly detection (Table 2) results appear properly reported and show gains over relevant baselines.

## Weaknesses

### Fatal

- **Table 5 (Ultra-Long ECG Recognition) contains baseline results that are copied from Table 3 with permuted columns, rendering the comparison invalid.**  
  Verifying the numbers directly:  
  - DENS-ECG in Table 5 reports Accuracy 0.3202, AUC-PR 0.1514, AUROC 0.2669, Weighted F1 0.2866. DENS-ECG in Table 3 reports Accuracy 0.3202, AUC-PR 0.1514, **Weighted F1 0.2669, Precision 0.2866**. The last two values are identical but shifted to different columns.  
  - ContraWR in Table 5: AUROC 0.2802 and Weighted F1 0.2794 match ContraWR’s Weighted F1 (0.2802) and Precision (0.2794) from Table 3.  
  - CNN-Transformer in Table 5: AUROC 0.2685 and Weighted F1 0.2641 match CNN-Transformer’s Weighted F1 (0.2685) and Precision (0.2641) from Table 3.  
  - Inception1D in Table 5: Accuracy 0.1823, AUC-PR 0.0832, AUROC 0.1770, Weighted F1 0.1736 match **FFCL** from Table 3 (Accuracy 0.1823, AUC-PR 0.0832, Weighted F1 0.1770, Precision 0.1736).  
  - An unnamed eighth row in Table 5 (between ST-Transformer and AnyECG-B) contains 0.2011, 0.0941, 0.1996, 0.2018, which exactly match ST-Transformer’s values from Table 3.  
  Because the paper’s claim of handling ultra-long ECG signals depends on Table 5, and the baseline rows are either fabricated by copying another table or reported with catastrophic sloppiness, the evidence for this task is not credible and the claim is unsupported.

### Major

- **The evaluation protocol structurally cannot support the central “foundational” and “universal” generalization claims.**  
  Section 3.1 states: “For various downstream tasks, we mixed all datasets together to minimize biases introduced by individual datasets and to better validate the model’s generalizability.” Mixing all pre-training and downstream data into a single pool and splitting 80/20 tests i.i.d. performance on a heterogeneous mixture, not cross-dataset or cross-device generalization. A foundational-model claim requires zero-shot, few-shot, linear-probe, or leave-one-dataset-out evaluation. Furthermore, the baselines are trained from scratch on this exact mixture, while AnyECG is pre-trained on the same pooled data and fine-tuned; this compares supervised learning with and without large-scale pre-training on the identical corpus, rather than isolating generalization to unseen distributions. The “Undisclosed Dataset” is noted as geographically distinct in Table 1, yet the paper explicitly mixes all datasets, making that distinctness irrelevant. This protocol gap undermines the paper’s core contribution.
  
- **The arrhythmia detection task is inadequately specified, making reported metrics uninterpretable.**  
  Table 3 reports accuracies below 35% for all models. The paper does not disclose how many classes are involved, how arrhythmia labels from CPSC, PTB-XL, INCART, and G12EC were harmonized (these datasets use incompatible taxonomies such as SNOMED-CT and SCP), or why all models perform so poorly. Without this information, readers cannot determine whether 34.49% represents meaningful progress or a broken setup, and the “SOTA” claim on this task is vacuous.

### Minor

- **Core mechanistic claims about the Rhythm Codebook are unsubstantiated.**  
  The paper asserts that vector quantization “enhances low-SNR signals into a high-SNR representation” and “effectively mitigates signal noise” (Section 2.2.1). No ablation compares VQ tokens against continuous tokens or a standard VAE bottleneck under controlled noise levels, and no SNR analysis is provided. The corrupted lead generation results (Table 4) show AnyECG underperforming GANs in MAE, which the authors attribute to prioritizing abstract rhythms; without isolating the codebook’s contribution, the central architectural justification remains speculative.
  
- **The Demography Decoder operates on a single patch but predicts patient-level attributes.**  
  The decoder takes a single patch embedding \(h_{j,k}\) (Section 2.2.2) and predicts patient-level age and weight (Eq. 6). A single 1-second ECG patch does not reliably contain such information, so this task either forces the encoder to memorize patient-level dataset metadata or adds unsolvable noise to the tokenizer objective.

- **Notation inconsistency impedes reproducibility.**  
  Section 2.1 defines \(P\) as the number of patches, but Section 3.2 states “patch size \(P = 300\),” contradicting the earlier definition where \(w\) was the patch size.

### Trivial

- The positional tolerance (mask width) for CMA is never specified numerically or ablated in the main text.
- The hierarchical modeling for ultra-long signals is mentioned only briefly (Section 3.3) without sufficient architectural detail to assess its novelty.

## Nice-to-Haves

- Cross-dataset transfer evaluation (e.g., leave-one-dataset-out or zero-shot linear probing) to validate foundational generalization.
- Ablation comparing discrete VQ tokens vs. continuous bottlenecks under varying synthetic noise levels.
- Visualization of codebook token semantics to validate that Rhythm Codes correspond to clinically meaningful patterns rather than arbitrary clusters.
- Label harmonization protocol and per-class metrics for arrhythmia detection.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **“Demography Decoder is theoretically contradictory to mitigating demographic shift”** — The paper presents demographic prediction as enabling personalized/domain-aware modeling, not as producing demographic-invariant features. The mechanism is therefore not inherently contradictory; the concern is better framed as lacking empirical evidence that it improves cross-population generalization.
- **Criticisms about ECG-FM availability** — The paper cites ECG-FM; per review policy, cited entities are assumed to exist.
- **Missing appendix proofs or references** — The parser strips appendices; they may exist in the original submission.
- **Typos, grammar, and formatting nitpicks** — These are parser artifacts, not author errors.
- **Criticisms about missing related works** — No external sources were used to confirm their existence.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Correct or re-run Table 5.** The ultra-long baseline results must be independently computed with a clearly described adaptation protocol (e.g., sliding-window hierarchical pooling), not copied from another task. If the baselines cannot be fairly adapted, the claim should be removed.
2. **Add cross-dataset generalization experiments.** Pre-train on a subset of datasets and evaluate on a held-out dataset (e.g., PTB-XL or the undisclosed set) to substantiate the foundational-model claim.
3. **Disclose the arrhythmia label harmonization protocol** and report the number of classes, class distribution, and per-class metrics to make the task interpretable.

## Score and Decision

**Calibration anchors used:**
- **High:** PaPaGei (avg 6.25, Accept Poster) — an open PPG foundation model with extensive downstream evaluation and no data integrity issues. AnyECG has comparable scale and ambition but falls well below due to Table 5 and evaluation protocol flaws.
- **Medium:** PAD (avg 5.00, Reject) — had minor table annotation reversals and limited theoretical contribution. AnyECG’s table issues are far more severe (entire rows copied across tables with permuted columns), placing it below this anchor.
- **Low:** ju4EwaLeoI (avg 3.25, Withdrawn) — criticized for unfair comparison splits and limited baselines; lf8QQ2KMgv (avg 3.75, Reject) — data leakage and methodological errors undermining integrity. AnyECG’s mixed-dataset protocol creates unfair comparisons analogous to ju4EwaLeoI, while the Table 5 copy-paste error is comparable in severity to the data-leakage concerns in lf8QQ2KMgv. AnyECG retains more genuine architectural novelty than these anchors, so it sits slightly above the worst low anchors but firmly in the low-quality band.

Relative to these anchors, AnyECG’s fatal data-integrity flaw in Table 5 and its major evaluation-protocol shortcomings make it unsuitable for acceptance in its current form. The paper would require a complete re-evaluation with held-out datasets and corrected tables to be reconsidered.

**Score: 3.5**
**Decision: Reject**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>