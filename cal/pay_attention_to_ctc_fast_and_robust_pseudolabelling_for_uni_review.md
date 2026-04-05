=== CALIBRATION EXAMPLE 96 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy & Clarity:** The title accurately captures the core innovation (CTC attention) and benefits (Fast, Robust, Unified). The abstract is excellent, clearly defining the problem (slow AR pseudo-labelling, self-reinforcing errors), the method (CTC-driven teacher forcing, mixed sampling), and the results (2x speedup, OOD robustness, SOTA).
- **Claims vs. Content:** Claims regarding "halving training time" and robustness gains are substantiated by Figure 5 and Sections 5.1–5.3. The claim of SOTA performance is well-backed by Tables 7–9 in the appendix. No unsupported over-claims are evident.

### Introduction & Motivation
- **Motivation & Gap:** The motivation is strong. The authors effectively highlight that while USR enabled unified modeling, its reliance on decoupled autoregressive pseudo-labelling created a computational bottleneck and OOD brittleness. This gap is clearly delineated.
- **Contributions:** The three main contributions (CTC-driven teacher forcing, aligned targets/joint prediction, mixed sampling) are precise and directly address the identified limitations.
- **Over/Under-claiming:** The introduction is balanced. It frames USR 2.0 as a targeted improvement to the pseudo-labelling scheme rather than a fundamentally new architecture, which is accurate and appropriate.

### Method / Approach
- **Clarity & Reproducibility:** The method is described clearly. Equations 3–6, along with Figure 2, provide a complete picture of the CTC-driven and AR modes. Model architectures and hyperparameters are detailed in Appendices A and B, ensuring reproducibility.
- **Assumptions & Justification:** The crucial assumption is that CTC pseudo-labels, though potentially globally incoherent, provide a valid conditioning signal for the decoder during self-training because the teacher and student share the exact same inputs. Section 4.1 ("Global coherence") justifies this well: the student learns the conditional next-token distribution $P(y_u | y_{<u}^{CTC}, x)$, and since inference uses the student's own predictions autoregressively, the global incoherence of teacher targets does not hinder the learned conditional distributions.
- **Logical Gaps/Edge Cases:** A subtle issue arises in **AR Mode** (Eq. 6), where the CTC head is supervised by both CTC PLs and Attention PLs. CTC loss inherently handles variable-length targets via marginalization. However, Attention PLs may contain non-monotonic patterns or repetitions that the autoregressive decoder permits but the CTC monotonic constraint struggles to align efficiently. If the Attention PL is significantly longer or structurally divergent from the CTC alignment path, could this introduce gradient noise in the CTC head? The paper reports good results, but a brief mechanistic discussion on how the CTC loss gracefully handles these potentially non-monotonic AR targets would add theoretical depth.
- **Theoretical Claims:** The theoretical argument regarding matched conditioning during self-training is sound and standard in semi-supervised learning analysis.

### Experiments & Results
- **Testing Claims:** Experiments rigorously test the claims. Efficiency is shown in Figure 5. Robustness to length, noise, and domain shift is detailed in Section 5. Ablations in Section 7 support the design of loss weights and mixed sampling.
- **Baselines:** Baselines (AV-HuBERT, BRAVEn, USR) are state-of-the-art and appropriate. The authors note using official codebases for robustness experiments, ensuring fairness.
- **Ablations:** Comprehensive ablations cover loss weights, collapse operations, and sampling probabilities. One suggestion: Figure 4 shows the effect of AR sampling probability, but the choice of `p=0.5` as the default lacks a strong justification beyond "balanced." The footnote mentions an adaptive schedule performed similarly, but showing this comparison in a table or plot (e.g., in Appendix C.2) would strengthen the claim that simple fixed mixing is sufficient, rather than just stating it.
- **Error Bars / Statistical Significance:** **This is a significant concern for ICLR.** While the authors report mean ± std over three seeds for convergence curves (Figure 8) and the epoch ablation (Table 13), the primary result tables (Tables 1, 2, 3, 4) report single-point WERs without variance. Some improvements over USR are incremental (e.g., 3.2% vs 3.0% in Table 4). Without error bars or significance tests on these key robustness and in-distribution metrics, it is difficult to assess whether the gains are statistically significant or within the noise of training stochasticity.
- **Datasets/Metrics:** Datasets (LRS3, LRS2, WildVSR, LibriSpeech, VoxCeleb2) and metrics (WER) are standard and appropriate for the field.

### Writing & Clarity
- **Clarity:** The writing is clear, technical, and precise. The distinction between the original USR decoupled approach and the USR 2.0 coupled approach is easy to follow.
- **Figures & Tables:** 
    - Figure 2 effectively visualizes the data flow differences.
    - Figure 6 and Table 11 provide excellent evidence for the length generalization claims, visually demonstrating the failure modes of USR that USR 2.0 resolves.
    - Figure 7 is particularly valuable for illustrating the "local vs. global" coherence issue, preempting reviewer questions about PL quality.
    - The parser artifacts in the text dump (e.g., in Figure 1 caption text) are noted and ignored as per instructions.

### Limitations & Broader Impact
- **Key Limitations Acknowledged:** Section D candidly discusses trade-offs: training time is still longer than simple fine-tuning; ASR/AVSR gains may be bottlenecked by greedy decoding quality; and the method is specific to self-training regimes.
- **Missed Limitations:** A deeper limitation worth mentioning is the dependence on the **CTC head's quality** in the base model. Since the entire decoder pseudo-labelling relies on `y_tilde_CTC`, if the CTC branch performs poorly on a specific domain (e.g., heavy accents where co-articulation breaks monotonic assumptions more severely than AR models), USR 2.0 might inherit significant target noise. A brief discussion on failure modes where CTC alignment degrades more than AR predictions would be valuable.
- **Broader Impact:** Section E covers positive impacts (accessibility, forensics) and risks (surveillance, bias). This is sufficient.

### Overall Assessment
This is a strong, well-executed paper that delivers a practical and effective improvement to semi-supervised unified speech recognition. The insight to leverage CTC targets for decoder teacher forcing is simple yet powerful, effectively addressing both computational cost and OOD robustness. The empirical evaluation is thorough, spanning multiple domains, noise levels, and sequence lengths.

**Primary Concerns:**
1.  **Statistical Rigor:** The absence of error bars or significance measures in the main result tables is a weakness given the incremental nature of some improvements over the baseline. ICLR standards typically require reporting variance for all major claims.
2.  **CTC/AR Mismatch Handling:** Clarifying how the CTC loss optimizes for potentially non-monotonic AR targets in the AR mode would strengthen the methodological justification.
3.  **Adaptive Sampling Evidence:** Moving the comparison of fixed vs. adaptive sampling schedules from a footnote to the appendix would better support the methodological choices.

**Conclusion:** The contribution stands solidly despite these points. The paper is clearly written, the method is sound, and the results are compelling. I recommend acceptance, assuming the authors can provide the requested statistical variance analysis and minor clarifications in the rebuttal or camera-ready version.

# Neutral Reviewer
## Balanced Review

### Summary
The authors propose USR 2.0, a semi-supervised framework for unified audiovisual speech recognition that replaces costly autoregressive pseudo-label generation with CTC-driven teacher forcing. By feeding collapsed CTC outputs into the attention decoder, the method produces attention pseudo-labels in a single forward pass, significantly accelerating training while inheriting CTC's robustness to distribution shifts. Coupled with a mixed sampling strategy to mitigate train-test mismatch, the approach halves training time, improves OOD generalization, and achieves state-of-the-art results across ASR, VSR, and AVSR benchmarks.

### Strengths
1. **Addresses a genuine bottleneck with a practical solution:** The paper correctly identifies that autoregressive pseudo-label generation in self-training frameworks is both a computational bottleneck and a source of error compounding under distribution shift. Replacing it with CTC-driven teacher forcing (Sec 4.1) is an elegant, computationally efficient fix that directly targets these pain points.
2. **Strong theoretical justification for a counterintuitive design:** The authors proactively address the concern that CTC-derived decoder outputs lack global sequence coherence. They correctly argue that matched teacher-student conditioning in self-training renders global incoherence irrelevant for knowledge transfer, and support this with clear qualitative examples (Fig 7, Sec C.4) and ablations (Table 4).
3. **Rigorous and comprehensive empirical evaluation:** The evaluation covers low/high-resource settings, multiple modalities, noise robustness (Table 1), length generalization (Fig 3), and OOD dataset transfers (Table 3). The consistent improvements over strong baselines (AV-HuBERT, BRAVEn, original USR) and the scaling experiments to a "Huge" model demonstrate method reliability.
4. **Well-executed mixed sampling mechanism:** The 50/50 alternating schedule between CTC-driven and autoregressive modes (Sec 4.2) is simple yet effective. The ablation on sampling probability (Fig 4, Table 10d) transparently shows the trade-off between ID accuracy, OOD robustness, and training speed, justifying the design choice empirically.
5. **High reproducibility standards:** The paper provides detailed hyperparameters (Table 5, 6), loss weightings, confidence thresholding logic, compute specifications (H200 GPUs, training days), and promises released code. This aligns well with modern ML transparency expectations.

### Weaknesses
1. **Lack of direct throughput measurement:** While the paper claims "~2× faster training," it primarily illustrates this via wall-clock time to reach certain WER thresholds (Fig 5, Table 13) rather than a direct samples-per-second or steps-per-epoch comparison under identical hardware. This makes the exact computational savings slightly ambiguous.
2. **Limited quantitative analysis of pseudo-label quality dynamics:** The paper assumes CTC-driven targets stabilize the learning loop, but does not track the evolution of pseudo-label CER/WER against ground truth (or a strong oracle like Whisper) across training epochs. Showing how PL quality differs between USR and USR 2.0, especially in early training, would strengthen the robustness claims.
3. **Inference-time implications are underexplored:** The authors note that CTC-driven forcing is only for training (Table 12), but do not investigate whether the resulting student decoder learns more stable internal representations that tolerate smaller beam sizes at inference. Quantifying inference-time beam search requirements would highlight additional practical benefits.
4. **Slightly ad-hoc mixed sampling rationale:** While the constant 0.5 probability works well, the appendix (Table 10d) shows linear/cosine schedules perform similarly. The paper does not explore more dynamic strategies (e.g., confidence-based switching or curriculum learning) that could theoretically better adapt to the model's evolving competence, leaving room for future optimization.
5. **Modality-specific robustness trade-offs lack granularity:** Table 3 and Figure 3 report aggregated or task-specific results, but do not explicitly dissect how CTC forcing impacts audio-only vs. video-only pseudo-labels under domain shift. Given the modality imbalance in real-world data, this breakdown would be valuable.

### Novelty & Significance
**Novelty:** The core idea of leveraging CTC outputs to parallelize attention pseudo-labeling is a targeted but meaningful methodological contribution. While joint CTC-attention training and teacher forcing are established concepts, their specific repurposing for *efficient, self-consistent pseudo-label generation in iterative self-training* is novel and well-motivated.
**Clarity:** The paper is exceptionally well-written, with a logical flow from problem identification to method design, extensive ablations, and transparent limitations. Mathematical formulations and algorithmic steps are clearly delineated.
**Reproducibility:** High. Hyperparameters, dataset splits, confidence filtering, and compute details are thoroughly documented. The reliance on the existing USR codebase and standard libraries (ESPnet, SentencePiece) further lowers the barrier to replication.
**Significance:** Strong. The method directly addresses scalability and robustness challenges in semi-supervised multimodal speech recognition, a highly active research area. The ~50% training time reduction, improved OOD generalization, and single-model unification across ASR/VSR/AVSR make it highly relevant for both academic research and practical deployment. It fits ICLR's interest in efficient learning paradigms and robust self-training frameworks.

### Suggestions for Improvement
1. **Add a throughput benchmark table:** Include a direct comparison of training speed (e.g., samples/sec, GPU hours/epoch, or memory footprint per batch) between USR and USR 2.0 on identical hardware to concretely validate the efficiency claims.
2. **Track pseudo-label quality over training:** Plot the WER/CER of teacher-generated pseudo-labels (against an oracle reference) across epochs for both methods. This would empirically demonstrate how CTC-driven forcing stabilizes the learning loop and reduces error compounding.
3. **Evaluate inference efficiency gains:** Report in-distribution and OOD WER under greedy and small-beam (e.g., 5, 10) decoding for both models. If USR 2.0 achieves comparable or better results with smaller beams, this would highlight an additional practical advantage beyond training speed.
4. **Clarify mixed sampling implementation:** Explicitly state whether mode sampling is applied per-batch, per-step, or per-sequence, and explain how gradients are handled if batches contain mixed modes. A brief algorithmic pseudo-code snippet in the main text or appendix would improve reproducibility.
5. **Provide modality-resolved OOD analysis:** Break down OOD robustness results by modality (e.g., audio-noise shift vs. video-motion/lighting shift) to show whether CTC forcing benefits specific branches disproportionately, which would guide future architectural choices.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5)
1. Report teacher pseudo-label WER (or token accuracy) against ground truth for CTC, AR, and CTC-driven targets across ID and OOD splits. Without quantifying target quality, claims about effective knowledge transfer and robustness are unsubstantiated.
2. Provide standard deviations across at least 3 random seeds for all primary benchmark tables (currently only Table 13 shows them). ICLR requires statistical rigor to verify that SOTA and robustness improvements are not artifacts of high variance or cherry-picked seeds.
3. Add an ablation controlling for compute budget (fixed epochs/FLOPs) to isolate the effect of CTC-driven joint prediction from simple convergence acceleration. Without this, it is unclear whether gains stem from the proposed mechanism or just increased training stability.

### Deeper Analysis Needed (top 3-5)
1. Quantify the train-test mismatch (exposure bias) reduction achieved by mixed sampling. Report decoder perplexity or log-probability divergence between AR and CTC-driven conditioned prefixes to prove the sampling strategy actually mitigates the stated mismatch.
2. Demonstrate *how* CTC supervision transfers robustness to the attention decoder rather than acting as a generic regularizer. Analyze decoder gradient magnitudes, attention entropy, or KL divergence between branches to validate the claimed mechanism of monotonic alignment transfer.
3. Characterize the interaction between the 0.8 confidence threshold and CTC-driven targets in distribution-shifted regimes. Show filtering retention rates and WER correlation for retained vs. dropped sequences to verify that the method isn't simply discarding hard OOD samples.

### Visualizations & Case Studies
1. Plot decoder-to-encoder attention heatmaps for base USR vs. USR 2.0 on long/OOD utterances. Direct visual evidence of stabilized, monotonic attention patterns would substantiate the robustness claims better than qualitative text errors.
2. Decouple training speedup into per-step wall-clock time vs. convergence epoch reduction in a clear table/bar chart. Current figures conflate algorithmic efficiency with optimization dynamics, obscuring where the 2× speedup originates.
3. Visualize the distribution of collapsed CTC sequence lengths vs. ground truth lengths across domains. This would expose whether the CTC collapse operation systematically truncates context, potentially masking failure modes on long sequences.

### Obvious Next Steps
1. Explicitly report inference-time latency, memory footprint, and beam-search cost compared to USR. The "fast" claim applies only to training, and ICLR reviewers will expect transparency on whether the method introduces deployment penalties or changes inference complexity.
2. Extend OOD evaluations (Table 3) to larger model scales (Base+/Large). Robustness demonstrated only on the Base model is insufficient to support claims of scalability and real-world generalization.
3. Provide a transparent computational accounting for the Huge model (FLOPs, VRAM peak, total GPU hours). Scaling claims require reproducible resource benchmarks to fairly contrast with fully supervised or self-supervised baselines.

# Final Consolidated Review
## Summary
USR 2.0 introduces a unified semi-supervised framework for audiovisual speech recognition that replaces costly autoregressive pseudo-labelling with CTC-driven teacher forcing. By feeding collapsed CTC outputs as fixed prefixes to the attention decoder, the method generates attention pseudo-labels in a single forward pass, nearly halving training time while transferring CTC's robustness to long sequences, noise, and out-of-distribution shifts. A mixed sampling strategy alternates between CTC-driven and standard autoregressive modes to mitigate train-test mismatch, yielding state-of-the-art results across ASR, VSR, and AVSR benchmarks.

## Strengths
- **Directly targets a well-defined bottleneck in self-training pipelines:** The paper accurately identifies autoregressive pseudo-label generation as both a computational bottleneck and a source of error compounding under distribution shift. Replacing it with CTC-driven parallel target generation (Sec 4.1) is an elegant, low-overhead solution that yields concrete efficiency and robustness gains without requiring architectural overhaul.
- **Sound theoretical justification for a counterintuitive design:** The authors proactively address the concern that CTC-derived decoder outputs may lack global sequence coherence. They correctly argue that matched teacher-student conditioning during iterative self-training renders global incoherence irrelevant, as the student learns stable conditional next-token distributions. This claim is rigorously supported by qualitative analysis (Fig 7, App C.4) and loss weight ablations (Table 4).
- **Comprehensive and well-controlled empirical validation:** Experiments span low/high-resource settings, multiple modalities, noise robustness (Table 1), length generalization (Fig 3), and cross-dataset OOD transfer (Table 3). The inclusion of a compute-controlled ablation (50 vs. 75 epochs in Table 13) effectively isolates algorithmic benefits from simple convergence speed, and detailed hyperparameter/logging specifications ensure high reproducibility.

## Weaknesses
- **Missing statistical variance in primary comparison tables:** While the paper reports mean ± standard deviation over three seeds for convergence curves (Fig 8) and epoch ablations (Table 13), the main benchmark tables (Tables 1–4) present single-point WERs. Given that some improvements over the original USR and other baselines are incremental (e.g., 3.2% vs 3.0% in Table 4), reporting variance for these primary claims is necessary to rule out training stochasticity and meet rigorous ML benchmarking standards.

## Nice-to-Haves
- Tracking teacher pseudo-label quality (e.g., WER or token accuracy against an oracle reference) across training epochs would provide additional empirical insight into how CTC-driven forcing stabilizes the self-training loop and reduces error compounding compared to standard AR pseudo-labelling.
- Explicitly decoupling the reported `~2×` training speedup into per-step wall-clock time versus reduced total epochs would offer finer-grained accounting of efficiency gains, though the current wall-clock-to-convergence metric is already practically meaningful.
- Clarifying in the methodology whether the mixed sampling strategy is applied per-batch or per-iteration, and briefly noting how gradients are handled if modes switch mid-batch, would eliminate minor implementation ambiguities for replication.

## Novel Insights
The paper demonstrates that global sequence coherence—a traditional prerequisite for attention-based decoding—is unnecessary for pseudo-label generation in iterative self-training. What matters is conditional consistency: by using collapsed CTC outputs as fixed prefixes for the decoder, the teacher and student operate under identical conditioning signals. This simple alignment transfers CTC's inherent robustness to distribution shifts directly into the attention decoder's training dynamics, effectively decoupling training-time stability from inference-time autoregressive coherence. The insight broadens the utility of joint CTC-attention paradigms beyond hybrid beam search, positioning CTC as a robust conditioning scaffold for scalable self-training in any sequence-to-sequence setting where temporal order is preserved but frame-level alignment is sparse.

## Suggestions
- Add standard deviation or confidence intervals to Tables 1–4 (or explicitly state that single-seed runs are reported due to compute constraints, with variance provided in an appendix) to quantify the statistical reliability of the primary WER improvements.
- Move the adaptive vs. constant sampling schedule comparison (currently a footnote) into the main text or explicitly reference Appendix C.2 Table 10(d) when justifying the fixed `p=0.5` default, to preempt reviewer questions about curriculum strategies.
- When reporting the Huge model results, briefly note the per-GPU memory peak or FLOPs estimate alongside the `~4 days on 64 GPUs` metric to improve transparency for researchers aiming to replicate scaling experiments.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
