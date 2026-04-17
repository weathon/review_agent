Now I have sufficient context from calibration papers. Let me synthesize the final review.

## Summary

DISTAR introduces a zero-shot TTS framework that operates entirely in discrete RVQ code space, coupling an autoregressive language model (AR LM) draft stage with a masked diffusion model (MDM) refinement stage. The AR LM generates patch-level sketches, and the MDM performs parallel iterative demasking within each patch, yielding blockwise parallelism while modeling intra-frame multi-codebook dependencies. The framework supports classifier-free guidance, RVQ layer pruning at test time for variable bitrate/compute, and stable greedy decoding, eliminating explicit duration predictors and forced alignment.

## Strengths

- **Strong empirical results on standard benchmarks.** DISTAR-medium (0.3B) achieves the lowest WER on both LibriSpeech-PC (1.66%) and SeedTTS test-en (1.32%), outperforming baselines including F5TTS, E2TTS, IndexTTS, and DiTAR. The greedy decoding WER of 1.91% with DISTAR-base (0.15B) is especially notable for robustness.

- **Practical controllability without retraining.** The stochastic RVQ layer truncation during training enables test-time bitrate and compute scaling via layer pruning (Figure 2), a clean engineering contribution. Classifier-free guidance and diverse decoding strategies (Table 3) provide interpretable control over the robustness–diversity trade-off.

- **No duration predictor or forced alignment required.** The fully discrete formulation preserves an [EOS] token for natural termination, simplifying the pipeline relative to continuous-space systems that require explicit duration modeling.

- **Effective RVQ-specific decoding heuristics.** The identification of "tail-first" bias and the proposed layer-wise/position-wise temperature shaping, along with hybrid sampling, represent practically useful insights for non-autoregressive decoding in sequential discrete domains.

## Weaknesses

### Major:

- **No ablation isolating the AR+MDM architecture from pure AR or pure MDM baselines.** The paper's core claim is that "tightly coupling" an AR sketcher with masked diffusion "mitigates classic AR exposure bias" and "effectively models RVQ depth–time dependencies." Yet there is no comparison to a purely autoregressive RVQ LM of matched capacity on the same codec, nor to MDM-only next-span prediction. Without this, it is unclear whether gains come from the AR+MDM coupling, from the RVQ codec, from training heuristics (stochastic layer truncation, embedding initialization), or simply from a well-tuned discrete system. This is the single most important ablation gap in the paper, and its absence leaves the central architectural contribution weakly supported.

- **Inference efficiency claims are unsubstantiated.** The abstract claims "maintaining the inference cost close to its continuous counterpart DiTAR," and Section 4.4 discusses controllability, but no wall-clock latency, real-time factor (RTF), or FLOPs are reported anywhere. DISTAR uses 24 NFE steps per patch vs. DiTAR's 10, with an additional AR outer loop. Without quantitative speed benchmarks, the efficiency claim is unsupported and potentially misleading.

- **Long-form and exposure-bias claims lack experimental validation.** The introduction and abstract emphasize "long-form synthesis with blockwise parallelism" and mitigation of "classic AR exposure bias," yet evaluation uses only standard short-utterance benchmarks (LibriSpeech-PC: ~5.4 hours of short clips; SeedTTS: 1088 short samples). No long-form evaluation (e.g., paragraph-level generation, speaker drift over minutes) tests the very motivation claimed.

### Minor:

- **Speaker similarity falls short of the best baselines.** DISTAR-medium achieves SIM of 0.67 (LibriSpeech) and 0.66 (SeedTTS), whereas E2TTS achieves 0.70 and 0.71 respectively. The gap of 0.03–0.05 in cosine similarity is potentially perceptually relevant, yet the paper claims "SIM on par with the best alternatives," which overstates the result.

- **Subjective evaluation lacks protocol detail.** Table 2 reports CMOS and SMOS but does not specify the number of raters, randomization procedure, or statistical significance tests. A CMOS of +0.22 above human reference (0.00) is unusual and warrants discussion.

- **The novelty of the core AR+MDM coupling is incremental relative to DiTAR and LLaDA.** The paper adapts the DiTAR paradigm (AR draft + next-patch diffusion) by replacing continuous diffusion with LLaDA-style discrete masked diffusion over RVQ tokens. While the adaptation is non-trivial and well-engineered, the primary conceptual move is a direct combination of two existing frameworks, and the paper does not provide new theoretical analysis (e.g., likelihood bounds, convergence properties) beyond what LLaDA already establishes.

- **Multiple decoding heuristics lack individual ablation.** Three tricks—layer-wise temperature, position-wise temperature, and hybrid sampling—are introduced together to address "tail-first bias." Table 3 compares three configurations but does not isolate which heuristic contributes what, making it difficult to assess whether these are principled solutions or masking deeper model pathologies.

- **Diversity claims are qualitative.** The abstract claims "rich output diversity" and "trade-offs between robustness and diversity," but no diversity metric (e.g., distinct utterances, prosody variance, or perceptual diversity ratings) is reported. Table 3 shows SIM differences between greedy and sampling, but diversity is only inferred, not measured.

### Trivial:

- The patch size ablation is deferred to the appendix (Section D is referenced but not visible in the main text), which is unfortunate for such a core hyperparameter but not unusual.

## Nice-to-Haves

- **Wall-clock inference speed benchmarks** comparing DISTAR vs. DiTAR, F5TTS, and E2TTS at matched conditions, including RTF and latency measurements.
- **Long-form evaluation** testing speaker consistency and WER as utterance length increases.
- **An AR-only RVQ baseline** using the same codec and training data to isolate the contribution of the masked diffusion component.
- **Individual ablations** of the three decoding heuristics (layer-wise temp, position-wise temp, hybrid sampling) and the overlapping patch design (S < P vs. S = P).

## Removed Points

These points were flagged for removal; treat with caution:

- **Codec fairness across baselines**: The harsh critic argued that comparing DISTAR against continuous-latent baselines with different codecs is unfair. However, DISTAR's claim is that its end-to-end discrete system is competitive with state-of-the-art continuous systems—which is the relevant comparison. Forcing continuous baselines through the same RVQ codec would not be a meaningful comparison. The codec IS part of the system design, and claiming SOTA among zero-shot TTS systems (regardless of representation) is legitimate.

- **Training data parity across baselines**: Standard practice in TTS is to compare against published baselines with their own training setups. DISTAR uses 50k hours of Emilia-English, and baselines like F5TTS use 100k hours. DISTAR achieving competitive results with less data actually strengthens the claims, not weakens them.

- **Missing related works**: Per instructions, we do not flag missing citations as we cannot verify their existence or relevance.

- **Subjective evaluation reproducibility concerns**: The ± values in Table 2 indicate some form of variance reporting. Demanding specific protocol details (exact rater counts, randomization) is a reproducibility nitpick beyond what is standard in the field.

## Novel Insights

The observation of "tail-first bias" in masked diffusion decoding for sequential data—where later positions within a patch accumulate higher confidence earlier in demasking—is an interesting and potentially generalizable finding. This phenomenon likely arises because later tokens benefit from more bidirectional context within the patch during training, and the proposed temperature shaping (cooling deeper RVQ layers and farther-ahead positions) represents a concrete mitigation. Whether this bias persists across different patch sizes, modalities, or discrete token spaces would be a useful direction for future work.

## Suggestions

1. **Add a pure AR baseline** using the same Qwen2.5-style decoder predicting flattened RVQ tokens on the same codec, with matched parameters and training data. This single experiment would substantively strengthen the paper's central claim.

2. **Report inference speed** (RTF, latency) for DISTAR and comparable baselines. The NFE=24 vs. NFE=10 asymmetry with DiTAR must be addressed quantitatively.

3. **Run a long-form evaluation** (e.g., multi-sentence paragraphs from LibriSpeech, measuring WER and speaker drift over longer durations) to validate the claimed mitigation of AR exposure bias.

4. **Ablate each decoding heuristic individually** to separate their contributions and determine which are essential.

5. **Moderate the SOTA claims** in the abstract and conclusion: the WER improvements are strong, but SIM lags behind E2TTS, and the efficiency claim needs quantitative backing.

## Score and Decision

I calibrated against the following papers:
- **Block Diffusion** (avg 8.0, Accept Oral): Novel semi-autoregressive theory, comprehensive ablations, strong empirical results.
- **SoundStorm** (avg ~5.4, Reject): MaskGIT adapted to RVQ audio tokens, limited baselines, incremental novelty.
- **MaskGCT** (avg ~5.25, Accept Poster): Two-stage masked transformer for TTS, incremental novelty but strong results.
- **F5-TTS** (avg ~5.5, Reject): Engineering improvements over E2-TTS, solid results but limited novelty.
- **HALL-E** (avg ~6.4, Accept Poster): Hierarchical codec for long-form TTS, practical contributions.
- **VALL-E 2** (avg ~5.0, Reject): Very incremental (sampling tricks + grouped codes), strong results but minimal novelty.
- **DiTTo-TTS** (avg ~6.25, Accept Poster): DiT-based TTS, good engineering, reasonable novelty.

DISTAR sits most closely to SoundStorm and MaskGCT in terms of novelty profile—applying an existing paradigm (AR draft + masked diffusion) to a new domain (RVQ speech tokens). The empirical results are genuinely strong (best WERs), and the practical tricks (layer truncation, temperature shaping) are useful. However, the paper has more substantive weaknesses than SoundStorm: (1) the core ablation (AR-only baseline) is missing, leaving the architectural contribution unsubstantiated; (2) the efficiency claim is explicit and unsupported; (3) long-form and diversity claims are made but not tested. Compared to MaskGCT (which also had incremental novelty but was accepted as a poster), DISTAR has stronger quantitative results but similarly weak ablation support for its key claims. I place DISTAR slightly above SoundStorm/VALL-E 2 (which were rejected for incremental novelty and weak methodological grounding) but below MaskGCT and HALL-E (which had more complete evaluation or more novel contributions).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>