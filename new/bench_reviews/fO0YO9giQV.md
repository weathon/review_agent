Now let me run calibration searches to anchor the score.Now I have enough information to write the consolidated review. Let me verify one more detail about Table 5 numbers.Now I have sufficient information for a well-calibrated final review.

---

## Summary

AnyECG introduces a family of ECG foundation models (254M–1.7B parameters) pre-trained in two stages: (1) an ECG Tokenizer using vector-quantized rhythm codes learned through a Multi-View Synergistic Decoder (morphology, frequency, demography proxy tasks), and (2) a full model trained via masked token prediction using Cross-Mask Attention (CMA) that constrains patch interactions to within-lead or same-time-position across leads. The model is evaluated on four downstream tasks: anomaly detection, arrhythmia detection, corrupted lead generation, and ultra-long ECG recognition, consistently outperforming task-specific baselines and the prior ECG-FM foundation model.

---

## Strengths

- **Two-stage tokenizer + masked pretraining design (Sections 2.2–2.3):** The first stage establishes a rhythm codebook via VQ and multi-view reconstruction before the second stage learns rhythm-event associations via masked prediction. This decomposition is well-motivated by the four identified ECG challenges and distinguishes the work from standard single-stage pretraining.

- **Cross-Mask Attention (Eq. 1, Section 2.1):** Unlike generic full self-attention, CMA restricts each patch to attend only to patches in the same lead or the same temporal position across leads, with a positional tolerance to account for conduction delays. This inductive bias is principled and ECG-specific, going beyond generic transformer variants in prior ECG models.

- **Multi-View Synergistic Decoder (Section 2.2.2):** The three complementary proxy tasks—time-domain reconstruction (Eq. 3), wavelet-coefficient prediction (Eqs. 4–5), and demographic attribute prediction (Eq. 6)—force the tokenizer to encode clinically relevant information from multiple perspectives, a domain-informed departure from standard VQ-VAE reconstruction.

- **Broad evaluation across four qualitatively distinct downstream tasks (Tables 2–5):** Covering anomaly detection, arrhythmia classification, lead imputation, and ultra-long signal recognition provides broader coverage than most ECG foundation model papers.

- **Demonstrated scalability with consistent model size gains (Tables 2–5):** AnyECG-B → L → XL show monotonic or near-monotonic improvement across tasks, supporting the claim that the architecture scales.

- **Five-seed evaluation with standard deviations:** Appropriate empirical practice for the field.

---

## Weaknesses

### Fatal
*None.* The methodology is coherent and the paper makes a real contribution; the problems below are in the experimental validation, not the core framework.

### Major

- **Pre-training data overlaps with all downstream evaluation data — unfair comparison (Section 3.1, Tables 2–5).** The paper explicitly states AnyECG is pre-trained on "all available unlabeled data" from the same seven datasets used for downstream evaluation. The 80/20 train/test split for fine-tuning is applied *after* pre-training on the full pool, meaning AnyECG has already processed the signal-level distribution — and possibly exact recordings — of the held-out test set in an unsupervised manner. All other baselines (DENS-ECG, ContraWR, XResNet1D, CNN-Transformer, RNN1D, FFCL, InceptionID, ST-Transformer) are trained from scratch only on the 80% labeled split and have no such advantage. The performance gains in Tables 2–5 therefore conflate representation quality with in-domain data exposure; no control experiment (e.g., pre-training on a disjoint pool) is provided to isolate the contribution of the architecture from the data advantage.

- **Ultra-long ECG comparison is structurally asymmetric (Section 3.3, Table 5).** AnyECG is equipped with a purpose-built hierarchical sliding-window adaptation (✓ in the Adaptation column), while every baseline is evaluated without such an adaptation (✗). The headline claim that AnyECG excels for ultra-long ECG cannot be attributed to its pre-trained representations when the primary design difference is an architectural mechanism withheld from all baselines. Additionally, three baselines (DENS-ECG, ContraWR, CNN-Transformer) have digit-for-digit identical accuracy, AUC-PR, and other metrics in Table 5 as in Table 3 (arrhythmia detection), raising a concern that these models were not actually run on the ultra-long task — though it is possible these models simply produced similar outputs when failing on ultra-long input. The paper provides no explanation for this coincidence.

- **Undisclosed pre-training source raises reproducibility and data integrity concerns (Table 1).** One of the seven pre-training datasets is labeled "Undisclosed Dataset (10,000 recordings)" with the ambiguous note "Geographically distinct *test set*." The paper offers no explanation of whether this dataset is used purely for pre-training, as part of downstream evaluation, or both. A foundation model paper cannot claim reproducibility when 10,000 (~19%) of its pre-training recordings come from an entirely opaque source whose provenance, license, and labels are unknown.

- **Outdated generative baselines for corrupted lead generation (Table 4).** The only comparisons for lead imputation are CGAN (2014) and WGAN (2018) — methods that are 6 and 8 years old. The paper acknowledges ECG-FM cannot be applied but makes no attempt to include any recent waveform imputation, diffusion, or flow-based baseline. Claiming "state-of-the-art" for lead generation based solely on outperforming decade-old GANs is not substantiated.

### Minor

- **Low absolute arrhythmia detection performance without explanation (Table 3).** AnyECG-XL achieves 34.5% accuracy and AUC-PR of 0.163 on arrhythmia detection. These are very low absolute numbers. The paper provides no discussion of class distribution, number of target classes, or why the task is inherently difficult. Without this context, readers cannot assess whether a 2 percentage-point gain over the best baseline represents meaningful progress or whether all models — including AnyECG — are essentially near-random on a severely imbalanced fine-grained multi-label problem.

- **Handling of missing demographic labels in Demography Decoder is unspecified (Section 2.2.2).** The Demography Decoder (Eq. 6) requires ground-truth age, weight, and sex for all pre-training samples. Several pre-training datasets (notably INCART with only 74 recordings) are known to have incomplete or absent demographic records. The paper does not state whether samples with missing demographics are skipped, masked, or pseudo-labeled during pre-training, which affects both the reproducibility and interpretation of the demography branch.

- **Codebook utilization not reported.** With a discrete rhythm codebook, codebook collapse (few active codes) is a known failure mode in VQ-VAE-style architectures. The paper does not report per-code assignment frequency, utilization entropy, or the effective number of active codes. This is particularly important for validating the claim that the codebook captures "clinically meaningful local rhythm codes."

### Trivial

- The masking ratio and masking granularity (patch-level vs. lead-level) for the masked pre-training phase are not stated in the main text, though ablations are presumably in the appendix. A brief mention of the default value in the main text would help readability.

---

## Nice-to-Haves

- A pre-training isolation experiment (pre-train on datasets disjoint from all downstream evaluation tasks, then fine-tune) would directly isolate the quality of learned representations from data distribution advantage.
- Applying an equivalent sliding-window or hierarchical aggregation to at least RNN1D and XResNet1D in Table 5 would enable a fairer ultra-long ECG comparison.
- Analysis of which rhythm codebook entries correspond to known morphological features (P-wave, QRS, T-wave, artifact) would validate the "clinically meaningful" claim concretely.
- Statistical significance tests across seeds for primary comparisons (optional in this field, but would strengthen the modest margins in Tables 2–3).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"300 Hz resampling unjustified"** (Harsh Critic): The paper explicitly justifies this choice: "300 Hz is considered sufficient for diagnosing most cardiac conditions based on the Nyquist-Shannon sampling theorem." This is standard clinical ECG practice. The specific concern about epsilon waves in ARVC is a niche edge case outside the paper's stated scope. Removed.

- **"Significant outperformance" wording is vague** (Harsh Critic): The word "significantly" in the abstract is hyperbolic for some tasks (e.g., arrhythmia detection, ~2 pp improvement), but this is a presentation nitpick, not a substantive flaw. Removed as trivial.

- **"Masking ratio undisclosed — method underspecified"** (Harsh Critic): The main text says ablation results are in the appendix. Per rules, appendix sections are stripped from all parsed papers; this concern would be resolved by the full submission. Removed.

- **"Demography Decoder loss (Eq. 6) requires complete demographics for all samples"** promoted to Minor above (partially kept as a valid concern), but the further claim that INCART's lack of demographics specifically invalidates the results is speculative. Severity kept at Minor.

- **Strength: "Consistent outperformance of ECG-FM" as an independent strength**: While factually accurate from the tables, this strength is clouded by the data-overlap flaw (ECG-FM is pre-trained on different data than the evaluation sets, while AnyECG is pre-trained on the same data). Left in Strengths with the caveat that the advantage is partially confounded.

- **"Abstract claim of 'significant outperformance' is never backed by statistical tests"** (Harsh Critic): Five-seed std reporting is the norm in this community; demanding formal significance tests across seeds is not standard for this type of systems paper. Moved to Nice-to-Haves.

---

## Novel Insights

The most genuinely novel observation from this review is the *structural* nature of the two major experimental confounds: (1) The pre-training data overlap is not incidental — it is a design choice to pre-train on "all available" data to maximize representation quality. This conflates the model's representational strength with its in-domain knowledge advantage, making it impossible from the results alone to determine which factor drives the gains. (2) The ultra-long ECG experiment exposes a common but underappreciated pitfall in foundation model papers: a task where the proposed model is *architecturally equipped* while baselines are not should be presented as an architectural contribution rather than a representation learning benchmark. Together, these confounds suggest that future ECG foundation model evaluations should distinguish (a) representation transfer to distribution-shifted tasks, (b) architecture-level task adaptation, and (c) in-domain fine-tuning — and report each separately.

---

## Suggestions

1. Add a "pre-training disjoint" ablation: pre-train AnyECG using only INCART + PTB (not used for the main downstream tasks) and evaluate on CPSC/PTB-XL/G12EC. This cleanly isolates representation quality from distribution exposure.
2. Either disclose the "Undisclosed Dataset" (license permitting) or exclude it from pre-training entirely, and report results with and without it.
3. Provide a clear statement of the temporal ordering of data splits vs. pre-training (i.e., whether test-set recordings were ever seen by the pre-training pipeline).
4. Add at least one post-2020 baseline for corrupted lead generation, or acknowledge explicitly that this is a lower-bound comparison.
5. Report codebook utilization statistics (active code count, assignment entropy) to validate the VQ component.
6. In the ultra-long ECG section, either add the sliding-window adaptation to baseline models or frame the hierarchical adaptation as a separate contribution rather than a benchmark comparison.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg score | Comparison |
|---|---|---|---|
| TA-PCLR (ECG foundation model, contrastive) | `7zJDTnogdG.md` | 3.33 (Reject) | Less comprehensive than AnyECG; single-task focus; but similar data-fairness concerns |
| CuPID (single-lead ECG masked modeling) | `QjrC77Nyu6.md` | 2.50 (Reject/Withdrawn) | Much narrower scope; weaker baselines; simpler architecture |
| PaPaGei (PPG foundation model) | `kYwTmlq6Vn.md` | 6.25 (Accept) | Similar physiological FM framing; stronger because data is fully public, 20 evaluation tasks, cleaner protocol |
| NeuroLM (EEG VQ + autoregressive FM) | `Io9yFt7XH7.md` | 6.25 (Accept) | Most similar architecture (VQ tokenizer, masked/autoregressive pretraining, multi-task) and similar scale (1.7B); less severe data-fairness concerns |
| V-JEPA (latent video SSL) | `WFYbBOEOtv.md` | 4.40 (Reject) | Masked modeling FM with structural comparison issues; provides a medium-low anchor |
| MOTOR (medical EHR FM) | `NialiwI2V6.md` | 7.50 (Accept spotlight) | Strong medical FM with truly clean evaluation protocol; shows what top-tier medical FM papers look like |

**Reasoning:** AnyECG is architecturally similar to NeuroLM (avg 6.25), which was accepted. However, AnyECG has two more serious experimental problems: (1) the pre-training data overlap with all downstream evaluation data, which NeuroLM does not have to the same degree; and (2) an entirely undisclosed pre-training source. These concerns put AnyECG closer to V-JEPA's 4.4 than NeuroLM's 6.25. The methodology is creative and the scope is broad, which keeps it well above the truly weak ECG papers (TA-PCLR at 3.33, CuPID at 2.5). Given the cluster of medium papers at 4.5–5.5 and the specific severity of the data-overlap issue (which goes to the heart of every empirical claim), a score of **4.5** is appropriate.

**Axis-by-axis:**
- *Originality:* Moderate-to-good. Two-stage VQ + masked ECG modeling with CMA is a thoughtful combination, though each individual piece has precedents.
- *Importance of research question:* High. Unified ECG representation learning across heterogeneous real-world settings is clinically important.
- *Whether claims are well supported:* Poor. The central empirical claims are compromised by the data-overlap issue, asymmetric ultra-long comparison, and undisclosed training data.
- *Soundness of experiments:* Poor-to-moderate. Five-seed evaluation is good; but the comparison protocol has structural flaws throughout.
- *Clarity of writing:* Moderate. The methodology is clearly described; limitations are understated.
- *Value to research community:* Moderate. The architectural ideas (CMA, multi-view decoder) are genuinely useful; but the experimental issues limit direct adoption of the results.

**Decision: Reject.** The paper presents a promising architecture, but its three major experimental flaws — pre-training data overlap with all evaluation benchmarks, asymmetric ultra-long ECG evaluation, and an entirely undisclosed training source — collectively prevent the core performance claims from being taken at face value. Acceptance would require either (a) a cleanly disjoint pre-training/evaluation protocol demonstrating that the architecture itself drives the gains, or (b) explicit acknowledgment and quantification of these confounds.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>