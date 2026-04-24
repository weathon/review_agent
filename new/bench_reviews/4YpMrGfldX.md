## Summary

This paper introduces the Transformer Audio AutoEncoder (TAAE), a predominantly transformer-based speech codec scaled to ~950 M parameters. Its core innovations are a Finite Scalar Quantization (FSQ) bottleneck and a novel post‑hoc residual decomposition that yields RVQ‑like multi‑token flexibility without RVQ training instabilities. Trained on 105 k hours of English 16 kHz speech, TAAE achieves strong objective metrics (e.g., SISDR 3.18, PESQ 2.96, STOI 0.90 at 400 bps) and near‑ground‑truth MUSHRA scores, substantially outperforming evaluated baselines at comparable or higher bitrates.

## Strengths

- **Novel architecture and quantization scheme.** The paper presents the first speech codec whose encoder/decoder are primarily transformers at the ~1 B scale, and devises a mathematically clean post‑hoc residual decomposition for FSQ (Eqs. 3–4, Table 1) that guarantees quantized latents remain inside the training distribution. This directly addresses the multi‑stream complexity and codebook‑utilization problems of RVQ‑based codecs (Section 3.2.1).
- **Strong empirical results on in‑domain speech.** On LibriSpeech test‑clean, TAAE at 400–700 bps exceeds same‑rate and higher‑rate baselines—including 16 kHz models such as SpeechTokenizer and DAC—by large margins in SISDR, PESQ, STOI and MOSNet (Table 2). A MUSHRA subjective test likewise places TAAE close to the ground‑truth reference (Figure 2).
- **Flexible bottleneck design.** The model can operate without quantization post‑training (Table 2, “TAAE + no quant.”), enabling seamless use as a continuous latent autoencoder for diffusion‑based generators without architectural changes.
- **Commitment to release.** The authors pledge to release code and model weights, improving reproducibility and practical impact.

## Weaknesses

### Fatal
None.

### Major
None. The central claim—that scaling transformer autoencoders with an FSQ bottleneck yields state‑of‑the‑art low‑bitrate speech coding on the evaluated data—is supported by the experiments. Confounding factors in baseline comparisons (see below) are real but do not invalidate the claim because (i) the paper also outperforms same‑sampling‑rate baselines by large margins, and (ii) the authors explicitly acknowledge the differences.

### Minor
- **Subjective‑test reporting gaps.** The MUSHRA experiment lacks reported variance, confidence intervals, or significance tests (Section 4.3, Figure 2), which makes it hard to gauge the reliability of the claimed subjective margin. In addition, Figure 2 shows bitrate points (e.g., TAAE at 800 bps, Mimi at 400 bps) that are not described in the methods or Table 2, creating confusion about which exact configurations were evaluated.
- **Non‑causal headline results vs. causal streaming baselines.** Table 2 and Figure 2 present the non‑causal TAAE, while Mimi—a streaming‑focused baseline—is causal. The paper does discuss a causal variant in Appendix A.4 and states that it shows “minimal degradation,” but the main text does not quantify the causal/non‑causal gap. Because the stated target is generative pipelines, this is a presentation issue rather than a methodological flaw, yet readers interested in streaming latency must rely on the appendix to assess the trade‑off.
- **Unablated FSQ contribution.** The paper does not isolate how much of the gain stems from the FSQ bottleneck versus simply scaling the autoencoder (e.g., no RVQ ablation at matched parameter count). While the combined design is valuable, a controlled ablation would strengthen the attribution of improvement to the quantization scheme.

### Trivial
- Minor bitrate inconsistencies between Table 2 and Figure 2.

## Nice-to-Haves
- Move concise summaries of the scaling ablation (Appendix A.2) and causal evaluation (Appendix A.4) into the main paper so readers can directly see the impact of scale and causality.
- Report empirical token entropy (Appendix A.8) in the main text to complement the flat‑index bitrate figures.
- Add spectrographic error visualizations and qualitative out‑of‑domain failure examples to complement the quantitative metrics.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **“Baseline comparisons are incommensurable and invalidate SOTA claims.”** The paper *does* compare against 16 kHz baselines (DAC, SpeechTokenizer, SemantiCodec) that are matched in sampling rate, and it still outperforms them by large margins. The 24 kHz baselines are resampled, but the presence of matched controls means the confound, while real, is overstated as fatal.
- **“The causal variant is relegated to an appendix, exaggerating practical gains.”** The paper explicitly scopes its primary use case to generative pipelines (Section 1) and mentions the causal variant in Sections 3.1 and 4.6. Because the appendix exists in the original submission, criticizing its location is a formatting nitpick, not a scientific flaw.
- **“Ablations deferred to appendices belong in the main paper.”** The original submission contains Appendices A.1–A.9; the parser strips them. It is inappropriate to penalize the authors for material that is present in the original submission.
- **“SemantiCodec inclusion is misleading without caveats.”** The paper explicitly notes baseline differences in Section 4.4 and omits SISDR for SemantiCodec in Table 2 because it is a generative mel‑spectrogram model. The caveat is present.
- **“Continuous latent row is under‑explored because no diffusion experiments are shown.”** The paper only suggests the possibility; it does not claim diffusion experiments.
- **Typo/formatting criticisms.** Any garbled text or line‑break issues are parser artifacts, not author errors.

## Novel Insights

The post‑hoc residual decomposition of FSQ (Eqs. 3–4) is a genuinely new mechanism. It allows a single scalar quantizer trained without any residual structure to be split after training into a hierarchical residual code that stays inside the original training distribution. This insight is independent of the transformer scaling story and could be adopted by convolutional or recurrent codecs as well, offering a practical alternative to RVQ that avoids codebook collapse and multi‑stream causal‑modeling headaches.

## Suggestions

- Clarify in the Figure 2 caption or surrounding text how the 800 bps TAAE and 400 bps Mimi points were obtained (e.g., which residual/token configuration was used).
- Add error bars or statistical annotations to the MUSHRA plot, or report per‑condition standard deviations and significance tests in the text.
- If space permits, include a short main‑text paragraph summarizing the scaling and causal ablations so readers need not rely solely on the appendix.

## Score and Decision

**Calibration anchors used:**
- **High (≥6):** *WavTokenizer* (avg 6.5, Accept poster) — similar neural‑audio‑codec domain; our paper has comparable novelty and broader baseline coverage, though slightly weaker subjective‑test reporting. *Fast Direct* (avg 6.75, Accept poster) — had unfair baseline comparisons (different samplers) but strong novelty and results; our baseline mismatches are less severe and acknowledged.
- **Medium (~5):** *SpeechTokenizer* (avg 5.75, Accept poster) — similar speech‑tokenization topic with training‑data mismatch concerns; our paper has stronger architectural novelty and larger empirical gains, placing it above this anchor.
- **Low (≤4):** *VoiceNoNG* (avg 3.5, Reject) and *DMNet* (avg 3.5, Withdrawn) — these lacked original ideas or failed to demonstrate claimed performance. Our paper is far above them in both contribution and experimental quality.

The paper makes a credible architectural contribution (predominantly transformer codec at ~1B parameters + novel FSQ residual decomposition) and backs it with strong objective metrics and a subjective evaluation that, while under‑reported, shows large margins over baselines. The baseline comparisons have acknowledged confounds, but the presence of same‑rate controls and the magnitude of improvement mean they do not undermine the core claim. The weaknesses are addressable in a rebuttal or revision. Relative to the anchors, the work sits between the solid 6.5‑tier codec papers and the 5.75‑tier borderline acceptances. I therefore score it slightly above the midpoint of that band.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>