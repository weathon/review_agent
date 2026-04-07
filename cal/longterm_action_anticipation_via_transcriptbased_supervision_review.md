=== CALIBRATION EXAMPLE 37 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me produce the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "Long-Term Action Anticipation via Transcript-Based Supervision" is precise and accurately reflects the paper's scope. The abstract is reasonably informative but slightly overpromises: the phrase "very robust and less costly alternative to its fully supervised counterpart" glosses over the significant performance gaps visible in the actual experiments (particularly on 50Salads and EGTEA). The claim that the approach is the "first weakly-supervised" one for LTA is the paper's cornerstone; it holds up under scrutiny, though it requires careful hedging given that Kim et al. (2024) also avoids explicit time annotations in an LTA context. The distinction (dense frame-level anticipation vs. symbolic sequence prediction) is valid but should be sharpened in the abstract itself.

---

### Introduction & Motivation

The motivation is genuine and well-articulated: dense frame-level annotation is expensive, and transcripts are far cheaper to obtain. The five bullet-point contributions are clearly stated. However, several issues arise:

1. **Over-reliance on ATBA as a black box.** The third bullet claims "we propose to temporally align video transcripts to frame-level features," but Section 3.1 later clarifies that this is achieved by directly adopting the ATBA module from Xu & Zheng (2024). This is a significant mismatch between the contribution claim and the technical reality. The novel contribution is *how ATBA is integrated into an LTA framework*, not the alignment mechanism itself. The introduction should be more precise here.

2. **The "first" claim requires sharper qualification.** The comparison against Kim et al. (2024) — which uses a VLM with in-context learning for action sequence prediction without time annotations — is dismissed in one sentence in related work. A direct comparison or a more principled distinction (e.g., "first *dense* LTA model trained with *only* ordered transcript labels, without any temporal grounding of supervision") would strengthen the claim.

3. **The introduction does not discuss the practical availability of transcripts.** Transcripts are assumed to always accompany training videos. In practice, obtaining clean ordered action lists still requires human effort (or automation that may fail). The practical savings over boundary annotation should be empirically estimated.

---

### Methodology

This is the section with the most substantive technical concerns.

**3.1 — Architecture**

**(a) ATBA dependence.** The temporal alignment module is entirely the ATBA module from Xu & Zheng (2024). The paper says it is "adopted" but later (Section 3.2.1) describes the alignment loss as "ATBA-style surrogate" and "L_atba." There is no clear novelty in the alignment component. The novel parts are: integrating ATBA into an LTA pipeline, using pseudo-labels to build a cross-attention mask, and the CRF-based anticipation decoder. The paper would be more credible if it explicitly acknowledged ATBA as a building block rather than framing alignment as an original contribution.

**(b) Cross-attention formulation (Eqs. 1–2).** These equations are technically suspicious. In Eq. 1, the query comes from transcript embeddings (A), keys from video features (X̂), producing an updated A. Then in Eq. 2, the update is injected back into video features X̂ as `X̂ ← X̂ + (M^⊤ ⊙ σ(AWg))A`. This is unusual: the "values" in the residual update are the *original* transcript embeddings A, not the cross-attention output from Eq. 1. Why is the attention output from Eq. 1 unused in Eq. 2? This seems either erroneous or requires explicit justification. The design appears to gate transcript embeddings back into video features without actually using the cross-attended output, which raises doubts about what information flow is actually happening.

**(c) Circular pseudo-label dependency.** The binary mask M in cross-attention is constructed from pseudo-labels Ŷ (from ATBA). These same pseudo-labels supervise the segmentation head. The segmentation head output then feeds back into the ATBA alignment in the next training iteration. This feedback loop between pseudo-labels and cross-attention masking is never analyzed. Does it amplify errors from ATBA, or is it self-correcting? This is a critical stability question that goes unaddressed.

**(d) Inference-time k* estimation.** The problem formulation (Section 3) states that k* (the boundary between observed and future actions) must be "implicitly estimated." However, the inference procedure is not described clearly. How does the model decide which portion of the transcript corresponds to the observed segment at test time? This is operationally important, and the paper is silent on it.

**(e) Duration loss (Eq. 7).** The self-supervised duration prior d̂ is computed from the *full video* during training (counting predicted label frequencies from the segmentation head). At inference, only the *observed portion* is available. The paper does not discuss how d̂ is maintained or updated at test time. If the buffer is populated during training, it becomes a class-level global prior — which is effectively a static temporal prior, not per-instance reasoning. This limits the model's ability to generalize to unusual activity durations and should be discussed.

**(f) CRF scalability.** The CRF in Eq. 5–6 operates over T_pred frames with |C| classes. For Breakfast with T_pred potentially up to ~hundreds of frames and |C| = 48 classes, the partition function computation is expensive. The paper gives no complexity analysis or runtime comparison.

**3.2 — Losses**

The loss ablation (Section 4.3) reports only ≈0.6–0.8 MoC improvement from CTC, which is modest. More importantly, no ablation is reported for γ1, γ2, γ3 (the loss weighting hyperparameters), which is a standard expectation for models with multiple loss terms. The paper refers to "details in supplementary material" for the alignment loss components — this is acceptable but some detail should appear in the main paper.

The three-stage progressive training scheme (video-level classification → segmentation+alignment → end-to-end) is a significant implementation choice with no ablation. Is all three stages necessary? What happens with two-stage or one-stage training?

---

### Experiments & Results

**Comparative results (Table 1).**

The primary comparison table (Table 1) is difficult to parse from the extracted text — TbLTA's own scores are not clearly visible (possibly a parser artifact). Based on the ablation tables (Table 4), TbLTA achieves 28.5 avg on 50Salads, vs. ActFusion's 28.39. The claim of "occasionally superior to fully supervised approaches" refers specifically to Breakfast at 30% observation. This is a cherry-picked condition: Breakfast has strong procedural regularity (breakfast cooking follows a highly predictable sequence), and at higher observation (30%), more context is available to exploit that regularity. The result on 50Salads is not competitive with the best supervised methods (e.g., ActFusion: 28.39 vs. TbLTA: 28.5 — this is essentially tied, but only when averaged; many individual horizon results are lower).

**Weak-supervision baseline gap.** There is only one weakly-supervised baseline: Zhang et al. (2021) (WS-DA), and it reports only a single setting (Obs 30%, 10% anticipation). This is highly insufficient. Several natural baselines are missing:
- A simple frequency-based prior (predict the K most frequent next actions given the observed action, using transcript statistics only).
- A model that uses only transcript ordering without any visual information (to show what the signal from transcripts alone is).
- Other weakly supervised TAS methods (e.g., Xu & Zheng 2024 applied to LTA) adapted as baselines.

Without these, it is impossible to assess how much of TbLTA's performance comes from the visual signal versus pure transcript statistics.

**EGTEA results (Table 2).** The table in the extracted text only shows supervised baselines (Timeception, Anticipatr) but TbLTA's actual numbers appear missing (likely parser artifact). However, the discussion acknowledges that "supervised models retain a clear edge overall." Given this, the EGTEA results seem to show TbLTA underperforming significantly — yet the text focuses on rare-class performance as the positive story. A fair reading is that TbLTA does *not* match supervised methods on EGTEA, which should be presented more prominently rather than relegated to "competitive on rare classes."

**Statistical significance.** All results are reported as point estimates. For Breakfast (4 splits) and 50Salads (5 splits), standard deviations or confidence intervals should be reported. A 1-point difference in MoC across 4 splits could easily fall within the noise.

**Ablation coverage.** Notable missing ablations:
- Effect of the CRF module alone (vs. without CRF).
- Effect of transcript encoding quality (DistilBERT vs. simpler word embeddings vs. stronger models).
- Effect of I3D features (the use of pre-extracted I3D is dated; how much does feature quality matter?).
- Sensitivity to the three-stage training schedule (number of epochs per stage).
- Effect of pseudo-label noise at different training stages.

**Use of I3D features.** All experiments use pre-extracted I3D (2048-d) from Carreira & Zisserman (2018). This is a widely used but relatively dated feature representation. Several recent baselines (especially ActFusion, which uses diffusion models) may also use I3D, but this should be explicitly confirmed. The choice of not using richer representations (e.g., VideoMAE, InternVideo) is never justified — it is likely that TbLTA's transcript-based supervision would benefit substantially from stronger visual features, and this is a missed opportunity.

---

### Writing & Clarity

Several passages impede comprehension beyond typical PDF parsing issues:

1. **Section 3.1, cross-modal attention**: Equations 1 and 2 appear out of order in the text (Eq. 2 is described before Eq. 1 is stated). The description of the mechanism is non-standard and requires re-reading to understand the intended flow.

2. **Section 3.2.2**: The CTC loss formulation (Eq. 4) and the CRF loss (Eqs. 5–6) appear interleaved in an unusual order. The CRF loss (for anticipation) is introduced in the segmentation-oriented subsection, while CTC (a segmentation tool) appears later — this reversal is confusing.

3. **Table 1 captioning**: The caption states "highest accuracy under a probabilistic framework is indicated in gray" but the gray formatting is not distinguishable in a printed or black-and-white version. More importantly, the table seems to mix deterministic and stochastic results without sufficient visual separation.

4. **Section 4.2 discussion**: The text says "TbLTA consistently surpasses prior (semi-) weakly-supervised baselines of Zhang et al. (2021)," but Zhang et al. only report a single number (15.65 on Breakfast, 21.30 on 50Salads). This is a very limited comparison from which "consistent surpassing" cannot be inferred.

---

### Limitations & Broader Impact

The conclusion acknowledges duration estimation as a challenge but is otherwise superficial. Several important limitations are omitted:

1. **Transcript availability assumption.** Transcripts are assumed to always be available at *training* time but not at *test* time. In practice, transcripts require human labor or external tools (e.g., recipe databases). The paper provides no analysis of transcript quality sensitivity — what happens if the transcript has errors or missing actions?

2. **Closed-world action vocabulary.** The model assumes that all actions at inference belong to the vocabulary seen during training (since inference uses learned class tokens E). This severely limits generalization to novel activities.

3. **Distribution shift between training and test.** Since pseudo-labels are generated from the training transcripts, the model is implicitly trained to recognize the distribution of actions in those specific datasets. There is no cross-dataset evaluation.

4. **Computational overhead.** Three-stage training with pseudo-label generation and CRF inference is computationally intensive. No training time comparison is provided.

5. **Societal impacts.** The paper's potential application to human activity surveillance (e.g., kitchen monitoring, workplace monitoring) deserves acknowledgment given the LTA task's relevance to human-machine interaction.

---

### Overall Assessment

TbLTA addresses a genuine and underexplored problem — dense long-term action anticipation under weak (transcript-only) supervision — and the high-level approach is well-motivated. The framework is technically coherent, combining pseudo-label alignment, cross-modal attention, CTC loss, and a CRF-based decoder in a principled architecture. The results on Breakfast are encouraging, and the paper establishes a meaningful first benchmark for this setting.

However, the submission has significant weaknesses that limit its readiness for ICLR acceptance. Most critically: the temporal alignment module — arguably the most important component — is borrowed wholesale from Xu & Zheng (2024) (ATBA), and the actual novelty is understated and blurred by the contribution framing. The cross-attention formulation contains a technical ambiguity that questions whether the described mechanism works as claimed. Experimentally, the comparison against weak-supervision baselines is thin (one baseline, one data point), statistical significance is not reported, key ablations are missing (CRF, training stages, loss weights), and the EGTEA results do not support the competitive narrative. The claim that TbLTA "occasionally surpasses" supervised methods is technically true but applies to specific settings on a dataset with strong procedural regularity — this should not be generalized. For ICLR, where methodological rigor and comprehensive empirical evaluation are expected, the paper in its current form requires substantial additional experiments and clearer technical exposition before it meets the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **TbLTA**, a novel framework for Dense Long-Term Action Anticipation (LTA) trained exclusively using video transcripts as supervision, eliminating the need for costly frame-level annotations. The method employs a weakly-supervised temporal alignment module (ATBA) to generate pseudo-labels, combined with cross-modal attention to ground video features semantically and a CRF-based decoder for coherent prediction. Experimental results on Breakfast, 50Salads, and EGTEA benchmarks demonstrate that transcript-based supervision can achieve performance competitive with, and occasionally superior to, fully supervised baselines.

### Strengths
1.  **Novel Problem Setting and Cost Efficiency:** The proposal of training LTA models exclusively from transcripts (ordered action lists without temporal boundaries) addresses the critical scalability bottleneck of dense annotation. As highlighted in the Introduction and Abstract, this shifts the paradigm from annotation-heavy supervised learning to transcript-only weak supervision, which is significantly more practical for large-scale deployment.
2.  **Strong Performance on Procedural Datasets:** The method establishes a new state-of-the-art for transcript-based LTA. Table 1 (Breakfast) shows TbLTA outperforming fully supervised baselines like FUTR (Gong et al., 2022b) at specific observation horizons (e.g., 34.5% vs 24.15% Avg accuracy at Obs 20%), proving that high-level semantic structure can effectively guide temporal modeling.
3.  **Robust Architectural Design for Weak Supervision:** The integration of a temporal alignment module (ATBA) with CTC loss and a CRF decoder effectively mitigates pseudo-label noise. The ablation study (Section 4.3) provides concrete evidence that removing the CTC supervision degrades performance by ~0.8 points on Breakfast, validating the design choice to marginalize over alignment paths.

### Weaknesses
1.  **Performance Gap on Egocentric Data:** While results on Breakfast and 50Salads are strong, Table 2 shows a notable performance gap on the EGTEA Gaze+ dataset compared to supervised methods (74.10 mAP for TbLTA vs 76.80 mAP for Anticipatr). The method appears more susceptible to the domain shift and higher variability found in egocentric videos compared to third-person view datasets.
2.  **Reliance on Pseudo-Label Quality:** The entire training pipeline hinges on the alignment module's ability to generate accurate pseudo-labels. Although ATBA is referenced, there is insufficient analysis in the paper regarding the sensitivity of performance to alignment errors. If the alignment is imperfect, noise propagation is evident in the "Quality" section (Figures 3a/b), where future duration estimation remains challenging.
3.  **Feature Dependency:** The experiment setup specifies using pre-extracted I3D features rather than end-to-end training (Section 4.1). While common in this subfield, it prevents the model from learning optimal temporal representations from raw data and limits generalizability to settings where pre-computed features are unavailable or mismatched.

### Novelty & Significance
The paper makes a technically sound contribution by bridging the gap between transcript-level semantics and dense action anticipation. While ATBA and CTC are known components in temporal action segmentation, applying them to *dense LTA* without boundary supervision is a novel adaptation supported by the authors' claim of being the "first weakly-supervised approach for LTA... relying solely on video transcripts." The significance is high for the community, as it validates the hypothesis that semantic narrative structure is sufficient for long-horizon prediction, potentially reducing annotation efforts by orders of magnitude. The work meets ICLR's bar for novelty by redefining the supervision paradigm for LTA rather than just improving model architecture.

### Suggestions for Improvement
1.  **Analyze the "Transcript Availability" Assumption:** Clarify the source of the transcripts used in experiments. If they are human-annotated, this is still labor-intensive. If they are ASR-derived, the performance on EGTEA (which often involves complex audio/visual cues) could be heavily influenced by ASR errors. The authors should discuss how robust the alignment module is to noisy transcript text.
2.  **Ablation on Alignment Noise:** Add a control experiment or discussion on how the model performs when the pseudo-labels are intentionally corrupted. This would demonstrate the robustness of the LTA decoder against the inherent uncertainty of weak supervision.
3.  **Address the EGTEA Discrepancy:** Provide a qualitative or quantitative analysis of why the model lags on EGTEA compared to Breakfast. Is it due to the lack of frame labels in long egocentric videos, or the complexity of verb-object interactions? Understanding this limitation will help establish the true boundaries of the proposed supervision method.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Train-Test Distribution Match:** Train the encoder using only observed frames ($X_{obs}$) to match inference conditions, as currently training on the full video ($X_{obs} + X_{pred}$) introduces severe data leakage that invalidates comparisons to supervised baselines.
2. **Pseudo-Label Quality Quantification:** Report the Intersection-over-Union (IoU) between ATBA-generated pseudo-labels and ground truth boundaries to verify that the "weak supervision" is not introducing prohibitive noise.
3. **Transcript Noise Robustness:** Inject errors (missing actions, swapped order) into training transcripts to test real-world applicability, as the current "oracle transcript" assumption undermines the claimed scalability.
4. **Annotation Cost-Benefit Curve:** Plot performance against annotation cost (hours) compared to fully supervised methods to substantiate the claim that this approach is a "less costly alternative."

### Deeper Analysis Needed (top 3-5 only)
1. **Text vs. Structure Disentanglement:** Ablate the text embeddings while keeping the CTC loss to determine if gains come from semantic grounding or merely sequence constraints, as the current setup conflates the two.
2. **Alignment-Anticipation Error Correlation:** Analyze whether anticipation failures correlate directly with temporal alignment errors to prove the modules are causally linked rather than independently learning shortcuts.
3. **Duration Prior Validity:** Provide statistics on intra-class duration variance within datasets to justify the assumption that observed action durations reliably predict future action durations.
4. **ATBA Dependency Analysis:** Evaluate performance when freezing vs. fine-tuning the ATBA module to clarify if the method relies on the specific strengths of that external alignment tool.

### Visualizations & Case Studies
1. **Timeline Alignment Comparison:** Visualize side-by-side timelines of Ground Truth, Pseudo-labels, and Model Predictions to expose systematic boundary smoothing or shifts caused by weak supervision.
2. **Cross-Attention Heatmaps:** Display attention weights between transcript tokens and video frames to verify the model attends to semantically relevant visual evidence rather than just temporal proximity.
3. **Horizon Degradation Curves:** Plot performance decay across observation ratios (10% to 50%) to identify the specific horizon where transcript guidance fails to compensate for lack of visual context.

### Obvious Next Steps
1. **Resolve Training Leakage:** Restrict training inputs to observed frames only ($X_{obs}$) to ensure the reported gains are due to transcript supervision and not access to future visual features during training.
2. **Statistical Significance Testing:** Report standard deviations and significance tests over multiple splits, as current tables show marginal gains over supervised baselines that may not be statistically reliable.
3. **Compute Efficiency Metrics:** Include training time and GPU memory usage comparisons to validate the claim that the method is computationally "less costly" than fully supervised counterparts.

# Final Consolidated Review
## Summary

TbLTA proposes the first weakly-supervised framework for dense long-term action anticipation (LTA), training exclusively from video transcripts—ordered action lists without temporal boundaries—rather than frame-level annotations. The method uses a temporal alignment module (ATBA) to generate frame-level pseudo-labels, combines cross-modal attention to ground video features with transcript semantics, and employs a CRF-based decoder for coherent anticipation. Experiments on Breakfast, 50Salads, and EGTEA show competitive performance with fully supervised methods in certain settings.

## Strengths

- **Novel problem formulation:** The paper addresses a genuinely underexplored setting—dense LTA without frame-level supervision. The claim of being the "first transcript-only LTA method" is substantiated; prior work (Zhang et al., 2021) uses weak labels for only the first anticipated action, while Kim et al. (2024) predicts symbolic sequences without dense frame-level output. This distinction matters for the paper's contribution scope.

- **Principled architecture for weak supervision:** The integration of pseudo-label generation (via ATBA alignment), CTC loss for sequence consistency, and CRF decoder for temporal coherence is technically sound. The ablation in Table 3 confirms that CTC contributes ≈0.6–0.8 MoC improvement, demonstrating that transcript-level supervision provides meaningful signal.

- **Competitive results on procedural datasets:** On Breakfast at 30% observation, TbLTA achieves results comparable to or exceeding fully supervised methods (e.g., outperforming FUTR and ActFusion at certain horizons). This validates the core hypothesis that procedural regularity can be leveraged from transcripts alone.

## Weaknesses

- **Insufficient weak supervision baselines:** The only weakly-supervised comparison is Zhang et al. (2021), which reports a single metric (Obs 30%, anticipating 10%). This provides an extremely limited baseline. The paper lacks a transcript-only frequency baseline, a text-only sequence model, or adaptation of other weakly-supervised TAS methods. Without these, it is unclear how much performance derives from visual learning versus transcript statistics.

- **Clear performance gap on EGTEA:** While Breakfast and 50Salads show competitive results, EGTEA shows consistent underperformance (the paper acknowledges "supervised models retain a clear edge overall"). The narrative of being "competitive on rare classes" does not offset the overall gap. This suggests the method may not generalize well to egocentric videos with higher variability and less procedural structure.

- **Pseudo-label quality and error propagation unanalyzed:** The entire pipeline depends on ATBA-generated pseudo-labels, yet no analysis is provided on pseudo-label accuracy (e.g., IoU with ground truth), error propagation from alignment to anticipation, or sensitivity to alignment noise. Figure 3 shows qualitative results where "degradation in prediction quality for the future interval remains relatively small," but this is not quantified.

- **Inference procedure is underspecified:** Section 3 states that the boundary index k* between observed and future actions must be "implicitly estimated," yet the inference algorithm is never described. How does the model determine the observed transcript portion at test time when only X_obs is available? This operational gap hinders reproducibility.

- **Duration prior lacks per-instance adaptation:** The duration loss (Eq. 7) uses a momentum buffer of class-wise duration statistics computed during training. At inference, this becomes a static prior—not conditioned on the specific video. The paper does not discuss how this affects generalization to videos with unusual temporal patterns, nor how the buffer is initialized or updated.

- **Missing key ablations:** Critical ablations are absent: (1) CRF module contribution isolated from the decoder; (2) three-stage training necessity (video-level → segmentation+alignment → end-to-end); (3) loss weighting hyperparameters (γ₁, γ₂, γ₃); (4) transcript encoding quality (DistilBERT vs. alternatives). These are standard expectations for a multi-component architecture.

## Nice-to-Haves

- **Transcript noise robustness analysis:** Real-world transcripts may have missing or misordered actions. Testing robustness to corrupted transcripts would strengthen practical applicability claims.

- **Computational cost analysis:** No training time or memory comparison is provided despite a complex multi-stage pipeline with pseudo-label generation and CRF inference.

- **Cross-dataset generalization:** All experiments use in-domain training and testing. Cross-dataset transfer would help assess real-world viability.

- **Statistical significance testing:** Results are reported as point estimates across 4–5 splits; standard deviations would clarify whether marginal improvements (e.g., +0.11 on 50Salads average) are meaningful.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Cross-attention formulation error claim:** The harsh critic alleged Eqs. 1–2 contain technical errors. Upon verification: Eq. 1 computes cross-attention A ← softmax(...)X̂W_V, and Eq. 2 injects gated transcript embeddings back into video features via X̂ ← X̂ + M^T ⊙ σ(AW_g). The attended A is explicitly used through σ(AW_g). The formulation is unconventional but not erroneous.

- **Training data leakage claim:** The spark finder claimed training on full video (X_obs + X_pred) constitutes data leakage. This is incorrect for temporal anticipation—training with full video visibility to learn temporal structure, then predicting only future frames at inference, is standard practice in supervised LTA methods (FUTR, ActFusion also do this).

- **I3D features being "dated":** Using pre-extracted I3D features is standard practice in this subfield for fair comparison. Calling it a weakness ignores community norms; all compared baselines also use these features.

- **"First" claim overstatement:** The paper correctly distinguishes dense frame-level anticipation (this work) from symbolic sequence prediction (Kim et al. 2024). The distinction is valid and clearly articulated in Related Work.

- **CRF scalability concern without justification:** The critic raised CRF partition function complexity but provided no runtime evidence or comparison. Without actual measurements, this remains speculative.

## Novel Insights

The paper demonstrates that procedural video structure—encoded purely in transcript ordering—provides sufficient signal for dense frame-level anticipation, challenging the assumption that temporal boundaries are essential for LTA. The cross-attention masking mechanism (using pseudo-labels to restrict transcript-to-video attention) is an interesting design for grounding weak supervision, though its effectiveness relative to simpler alternatives remains unexplored. The most significant insight is empirical: on datasets with strong procedural regularity (Breakfast), transcript supervision can match dense annotation, but this advantage diminishes on less structured egocentric data (EGTEA), suggesting the method's applicability is tied to task structure.

## Suggestions

1. **Add transcript-only baselines:** Include a frequency prior baseline and a text-sequence-only model to isolate the contribution of visual learning.

2. **Quantify pseudo-label quality:** Report IoU between ATBA pseudo-labels and ground truth boundaries, and correlate alignment error with anticipation error.

3. **Clarify inference procedure:** Describe explicitly how k* (observed/future boundary) is determined at test time.

4. **Add CRF ablation:** Isolate CRF contribution by comparing decoder output with and without CRF refinement.

5. **Report standard deviations:** Include variance across splits for all metrics to establish statistical significance of claimed improvements.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
