Now let me do calibration searches to position the score properly.Now I have sufficient calibration data. Let me write the final review.

---

## Summary
TAAE (Transformer Audio AutoEncoder) introduces a predominantly transformer-based speech codec scaled to ~950M parameters, paired with a modified Finite Scalar Quantization (FSQ) bottleneck. The paper's central contribution is demonstrating state-of-the-art speech quality at extremely low bitrates (400 and 700 bps) through a combination of architectural scaling, a novel post-hoc residual FSQ decomposition, and a two-stage training procedure. The model strongly outperforms published baselines in both objective metrics and a MUSHRA subjective study.

---

## Strengths

- **Novel post-hoc residual FSQ decomposition (Eqs. 3–4, Section 3.2.1):** The mathematical construction showing that any FSQ trained with `L = 2^n + 1` levels can be post-hoc decomposed into hierarchical residual tokens without retraining is a genuinely useful algorithmic contribution. It decouples codec training from downstream token structure requirements and cleanly addresses known RVQ codebook utilization pathologies.

- **Strong objective results, including at each model's competitive operating point (Table 2):** TAAE at 700 bps (PESQ 3.09, STOI 0.92) surpasses Mimi at its full designed bitrate of 1100 bps (PESQ 3.01, STOI 0.90), not only at its truncated configuration. TAAE at 400 bps exceeds DAC at 1000 bps and Encodec at 1500 bps across all reported metrics.

- **Compelling MUSHRA subjective results (Figure 2):** TAAE at 400 bps scores ~85 versus all baselines ≤55 at comparable bitrates, approaching the ground truth of ~90. This margin is large enough to be credible despite the limited listener study size.

- **Quantizer dropout via hybrid training (Section 3.2):** The mix of straight-through estimation, uniform noise quantization (Brendel et al., 2024), and Bernoulli masking of unmodified latents is a clean training-time innovation that enables post-training bitrate flexibility without the codebook utilization issues of RVQ.

- **Causal variant with minimal degradation (Section 4.6):** The streaming variant is described and empirically shown to outperform Mimi in objective metrics despite training on far less data and fewer steps—an important practical result for real-time applications.

- **Honest limitations section (Section 5):** The paper explicitly acknowledges the English-only training data, audiobook domain bias, 16 kHz ceiling, and parameter-count constraints without obfuscation.

---

## Weaknesses

### Fatal
None.

### Major

- **Architecture vs. scale confound is unresolved — the headline claim is overstated.** The paper's title and abstract claim that *scaling a transformer architecture* is the key insight. However, TAAE has ~950M parameters versus Mimi (~87M), DAC (~74M), and Encodec (~300M). The scaling ablation (Appendix A.2) confirms quality improves from 250M to 1B parameters, but no CNN-based codec is trained at equivalent scale. Since the parameter count advantage over most baselines is 10–13×, and since the paper itself notes (Section 3.1) that CNNs offer "strong inductive bias and high parameter efficiency" while transformers provide "enhanced scalability, albeit with reduced parameter efficiency," the architectural conclusion is not separable from the scale conclusion in the current experimental design. The paper may be demonstrating that *a larger codec generally performs better*, not that *transformers are the right architecture for this problem*. This does not invalidate the engineering contribution, but the framing as an architectural finding requires either (a) a CNN baseline at matched parameter count or (b) a more conservative scope claim.

- **Baselines at the critical low-bitrate regime are evaluated at sub-design truncations.** Mimi is designed and trained for 1100 bps with 8 RVQ codebook levels; its 550 bps evaluation uses only 4 levels (half its designed capacity). Residual quantization degrades non-linearly under truncation because early-level residuals are not trained to absorb as much signal energy as would be needed without the later levels. The paper's lowest-bitrate comparison (~400 bps TAAE vs. ~550 bps Mimi/4-level truncation) therefore tests TAAE at a configuration it was designed for against Mimi at a configuration it was not designed for. This is partially mitigated by also showing Mimi at its full 1100 bps operating point—where TAAE at 700 bps still wins—but the headline "SOTA at low bitrates" comparison directly exploits this truncation artifact.

### Minor

- **Domain matching advantage is understated in the main results.** TAAE is trained exclusively on English speech from Librilight and MLS-English (~105k hours of predominantly audiobook recordings). The test set is LibriSpeech test-clean—also audiobook English. Mimi is trained on 7M multilingual hours across diverse domains and SemantiCodec uses a different training regime. The paper addresses this partially in Appendix A.5 (multilingual generalization), but the main results section and discussion should note that the English-only, audiobook-to-audiobook evaluation inherently advantages TAAE. The gain over multilingual baselines may be partly domain-specific.

- **MUSHRA listener study is marginal for the strength of the subjective claim.** 24 crowdsourced participants gathered via public forum links, with no hearing screening, no training trials, and no hidden anchor for detecting unreliable responses (MUSHRRA format without anchor). While the margin (85 vs. ≤55) is large enough to be robust to some listener noise, the "strongly outperforms" language in the paper goes beyond what this protocol can firmly establish for a conference submission claiming SOTA. Enlarging and properly screening the listener pool would strengthen the subjective evidence.

- **The 400 bps operating point uses L=6, which is outside the training set {5, 9, 17}.** The paper argues this is valid because L_min=5 ≤ L=6 ≤ L_max=17, so the quantization error is "within the bounds previously seen" (Section 3.2.1). This is a plausible argument, but the claim is not directly validated with a dedicated ablation comparing L=6 post-hoc quality to the nearest trained configurations (L=5 and L=9). A brief validation would remove lingering uncertainty about the 400 bps results.

- **Cross-sample-rate evaluation for 24 kHz baselines introduces an additional artifact penalty.** Encodec and Mimi (native 24 kHz) are evaluated by upsampling 16 kHz test audio to 24 kHz, encoding/decoding, then downsampling back to 16 kHz (Section 4.4). The upsample step introduces low-pass characteristics and means these models process artificially band-limited input. The paper does not quantify whether this penalizes these baselines differently from native-16 kHz inputs. Since both Encodec and Mimi perform worse than TAAE anyway, this does not change the rankings, but a note on the methodology (or native-16 kHz evaluation for these models) would improve rigor.

### Trivial

- The bitrate calculation uses raw uniform-coding rates (Eq. 5); the paper addresses entropy-coded rates in Appendix A.8 but this is not mentioned in the main results section. A parenthetical note in Section 4.5 or Table 2 would prevent readers from suspecting a discrepancy with baselines that may quote entropy-coded rates.

---

## Nice-to-Haves

- A CNN-based codec variant trained at matched parameter count (~950M, if tractable at smaller scale via e.g. a very wide DAC/Mimi) as an architectural control experiment. Even a 250M CNN vs. 250M TAAE comparison would meaningfully support the transformer architecture claim.
- Spectrogram visualizations of reconstructed speech at 400 bps across TAAE, Mimi (550 bps), and SemantiCodec to characterize the qualitative nature of the artifacts at this compression extreme (hallucination vs. muffling vs. metallic distortion).
- An out-of-domain quantitative evaluation (noisy speech, multi-speaker, music) to concretize the audiobook-domain limitation flagged in Section 5.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"SingleCodec as missing baseline" (Harsh Critic, Issue 2):** SingleCodec operates on mel-spectrograms with a BigVGAN vocoder, not on raw waveforms end-to-end. The paper explicitly acknowledges this architectural difference in Section 2.2: "followed by BigVGAN as a vocoder for waveform reconstruction." Evaluating a vocoder-based system in the same Table as waveform codecs would mix methodological paradigms; TAAE's waveform end-to-end approach is categorically different. Removed as a strawman comparison request.

- **"WavLM circular advantage in objective metrics" (Harsh Critic, Section 3.4):** The claimed circularity requires that WavLM-Large features align with PESQ, STOI, and SI-SDR computations. PESQ and STOI are signal-based algorithms (ITU-T P.862 / STFT overlap), not learned features, and therefore cannot be influenced by WavLM fine-tuning in a circular way. MOSNet could marginally reflect this, but the primary metrics in Table 2 are not susceptible. The criticism overstates the concern.

- **"L=6 is outside training set {5, 9, 17}" (flagged as structural issue):** This is an acknowledged design property—the paper explicitly provides the theoretical justification for why any L ≥ L_min is valid, and L=6 satisfies this. The concern has been retained as a minor weakness (missing direct ablation), but it is not a structural flaw in the paper's reasoning.

- **"Post-hoc FSQ not validated for L=6"**: Retained as minor weakness; not removed, but not elevated to major as the theoretical argument is sound.

- **Requests for more baselines in the MUSHRA study (DAC/Encodec exclusion):** The paper states the exclusion is based on objective performance triage. Since DAC and Encodec are included in Table 2 (objective metrics) and uniformly underperform, their exclusion from the subset selected for the listener study is methodologically reasonable. Not a genuine weakness.

- **Bitrate calculation using uniform coding rates:** The paper addresses entropy-coded rates in Appendix A.8 and shows near-full codebook utilization (making entropy coding yield minimal gains). Retained only as a trivial note. The methodological concern is already answered in the paper.

---

## Novel Insights

The post-hoc residual FSQ decomposition (Eqs. 3–4) is the most substantively novel algorithmic idea in the paper. The mathematical argument that FSQ trained with `L = 2^n + 1`-level scalar quantizers admits a hierarchical residual decomposition—matching the token structure expected by RVQ-based generative models—without any retraining, cleanly solves a practically important compatibility problem. This is not a minor engineering choice but a structural property of the FSQ construction that is worth independent notice. Additionally, the hybrid quantizer dropout combining three modes (straight-through, uniform noise, unmodified) via Bernoulli masking is a practically useful training technique for FSQ that appears independently applicable to other bottleneck designs.

---

## Suggestions

1. **Reframe the title and abstract** to reflect what is actually demonstrated: a large-scale speech codec using a transformer architecture and FSQ achieves SOTA at low bitrates. Remove or substantially qualify the claim that the *architecture* (as opposed to scale + architecture) is the key factor until the ablation evidence supports it.
2. **Add a 250M–500M scale CNN vs. transformer comparison** in the scaling experiments section. Even an informal comparison would allow the architectural contribution to be evaluated independently of scale.
3. **Include a sentence in the main results** noting the English-only training vs. multilingual baselines mismatch, with a pointer to Appendix A.5 for the multilingual generalization evidence.
4. **Add a note on the post-hoc L=6 configuration** in Section 3.2.1 or 4.5, with at minimum a brief empirical check showing that L=6 quality falls between L=5 and L=9 as expected.
5. **Expand the MUSHRA description** to clarify participant screening (if any) and report standard deviations or confidence intervals across participants and samples.

---

## Score and Decision

**Calibration anchors used:**
| Paper | Path | Avg. Score | Comparison |
|---|---|---|---|
| SpeechTokenizer | AF9Q8Vip84.md | 5.75 (Accept) | Same domain; TAAE has stronger results and more algorithmic novelty (FSQ decomposition). TAAE should score above this. |
| GenAu | lidVssyB7G.md | 5.25 (Reject) | 1.25B-parameter audio scaling paper; rejected for "engineering effort over novel architecture." TAAE faces the same criticism but has a more novel algorithmic contribution (FSQ). TAAE scores above. |
| DiTTo-TTS | hQvX9MBowC.md | 6.25 (Accept) | TTS scaling to 790M with limited architectural novelty but strong results; TAAE is comparable in scale/novelty balance, and the speech codec results are arguably more impactful. |
| ETTA | xmgvF0sLIn.md | 6.0 (Reject) | Text-to-audio scaling/design-space paper; rejected partly for limited novelty beyond engineering. TAAE has somewhat more novel contributions (FSQ post-hoc decomposition). |
| MAGNeT | Ny8NiVfi95.md | 7.33 (Accept) | Audio generation with multiple strong contributions and ablations; TAAE lacks the same depth of architectural attribution support. Below this anchor. |
| HarmonyLM | mp8ZgMZ1RG.md | 1.67 (Reject) | Music generation with insufficient novelty; TAAE is substantially above this anchor. |
| Audio overlap-add | JOBokGDcX0.md | 2.50 (Reject) | Rejected for presenting known signal processing as novel; TAAE is well above this. |

**Assessment relative to anchors:** TAAE is clearly above the low-scoring (< 3) papers (HarmonyLM, JOBokGDcX0). It is above the medium-range rejects (GenAu 5.25, ETTA 6.0) due to stronger empirical margins and more novel algorithmic content. It is solidly near DiTTo-TTS (6.25, accepted) and above SpeechTokenizer (5.75, accepted). It falls short of MAGNeT (7.33) because the central architectural vs. scale attribution remains unresolved and the MUSHRA study is marginal.

**Originality:** Good. The FSQ post-hoc decomposition is novel; the transformer-at-scale-for-codec framing is new to the literature.
**Importance of research question:** High. Sub-700 bps speech tokenization is directly relevant to practical generative speech pipelines.
**Support for claims:** Moderate. The compression performance claims are strongly supported; the architectural attribution claim is not.
**Soundness of experiments:** Fair. Compelling results, but the baseline truncation at the key bitrate regime and the domain matching advantage reduce the rigor of the comparative evaluation.
**Clarity of writing:** Good. Well-structured, honest about limitations.
**Value to the research community:** High. Open-release model at a practically important operating point; novel FSQ technique is independently applicable.

**Final score: 6.0 — Weak Accept.** The paper sits between the rejected ETTA (6.0 average, rejected) and the accepted DiTTo-TTS (6.25 average) anchors. The genuine algorithmic contribution of the FSQ post-hoc decomposition and the strong empirical results tip it toward acceptance, but the unresolved architecture-vs.-scale confound and the truncated-baseline comparison methodology are real concerns that the reviewers should push the authors to address in revision.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>