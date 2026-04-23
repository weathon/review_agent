Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

DISTAR introduces a zero-shot TTS framework that operates entirely in discrete RVQ code space, coupling an autoregressive language model (for cross-patch drafting) with a masked diffusion transformer (for within-patch infilling). This patch-level AR + discrete diffusion factorization natively models RVQ's joint time-depth structure, eliminates the need for explicit duration predictors via [EOS] tokens, and enables practical features such as greedy decoding and test-time RVQ layer pruning.

## Strengths

- **State-of-the-art robustness (WER) on both benchmarks.** DISTAR-medium achieves 1.66% WER on LibriSpeech-PC and 1.32% WER on Seed-TTS test-en (Table 1), substantially outperforming all baselines including the twice-as-large continuous DiTAR (0.6B, WER 2.39%/1.78%). This is the paper's clearest and most compelling result.

- **Best subjective quality among all compared systems.** DISTAR achieves the highest SMOS (3.31) and CMOS (0.22) on Seed-TTS test-en (Table 2), outperforming F5TTS, E2TTS, and CosyVoice 2, supporting claims of improved naturalness and speaker similarity as perceived by human listeners.

- **Practical advantages from the discrete code space: greedy decoding and test-time bitrate control.** Table 3 shows greedy decoding achieves WER 1.91% (competitive with sampling at 1.99%), and Figure 2 demonstrates that stochastic layer truncation enables RVQ layer pruning at inference with graceful quality degradation—both concrete practical benefits directly enabled by the discrete design.

- **Parameter efficiency outperforming larger models.** DISTAR-medium (0.3B) outperforms DiTAR (0.6B) on WER by a large margin while achieving comparable or better subjective quality, demonstrating the architecture delivers superior results at half the parameter count.

## Weaknesses

### Fatal
None.

### Major

- **The claimed computational efficiency advantage is unsupported and contradicted by reported NFE.** The abstract states DISTAR maintains "inference cost close to its continuous counterpart DiTAR," and the introduction claims "comparable or lower computational cost" (Section 1, contribution bullet 3). However, DISTAR-medium uses NFE=24 while DiTAR uses NFE=10 (Table 1)—2.4× more diffusion steps. DISTAR additionally incurs the AR LM forward pass and aggregator computation per patch. The paper provides zero wall-clock time, RTF, or throughput measurements. Section 4.4 is titled "Inference Efficiency and Controllability" but contains only the RVQ layer pruning figure—no efficiency data at all. With no efficiency evidence and a 2.4× NFE disadvantage, the efficiency claim in the abstract and introduction is an overclaim that misrepresents the paper's contribution. This matters because the efficiency framing is central to the paper's positioning of discrete tokens as a *practical* advantage over continuous methods.

- **Overclaimed breadth of superiority in the abstract and conclusion.** The abstract claims DISTAR "surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency," and the conclusion claims "SOTA robustness, speaker similarity, and naturalness." However, DISTAR does not lead on objective SIM (E2TTS achieves 0.70/0.71 vs. DISTAR's 0.67/0.66 on LibriSpeech/SeedTTS) or UTMOS (IndexTTS and DiTAR lead on some benchmarks). DISTAR's strength is robustness (WER) and subjective quality (SMOS/CMOS), and the claims should reflect this rather than claiming across-the-board superiority. While the subjective SMOS results partially support the similarity claim, the objective SIM results directly contradict it.

### Minor

- **DiTAR baseline numbers are taken from the DiTAR paper (marked ♦) rather than reproduced under identical evaluation conditions.** Differences in codec, ASR model, and embedding extractors can shift WER and SIM by margins comparable to the reported gaps. While using reported numbers is common practice, it introduces comparison noise that the paper does not acknowledge or control for.

- **The "no duration predictor" claim is architecturally true but lacks empirical validation as an advantage.** Section 1 states DISTAR "dispenses with both an explicit duration predictor and forced alignment," using [EOS] tokens for termination instead. With fixed patch size P=8 and stride S=8, each AR step generates a fixed-duration segment. There is no evaluation of whether the model handles variable speaking rates, prosodic timing, or pause insertion flexibly—it may simply reproduce average tempo from training data. A comparison of generated vs. reference durations or a test on deliberately fast/slow prompts would strengthen this claim.

- **Thin ablation section.** Table 3 shows only three decoding strategies on DISTAR-base. Missing ablations that would meaningfully strengthen the paper: (a) effect of NFE (especially DISTAR at NFE=10 to match DiTAR), (b) contribution of the AR LM vs. pure masked diffusion, (c) whether the RVQ-aware sampling heuristics (layer-wise temperature, position-wise temperature, hybrid sampling) actually help relative to vanilla masked diffusion decoding. The patch size ablation is deferred to the appendix.

- **Decoding heuristics introduce three hyperparameters (T_layer=0.8, T_time=0.95, 50% hybrid split) with no sensitivity analysis or principled justification.** The "tail-first bias" is hypothesized but not empirically characterized (e.g., no confidence maps across positions). While Table 3 shows the shaped sampling improves SIM from 0.626 to 0.640, it is unclear how sensitive these gains are to the specific parameter choices.

- **Inconsistent baseline coverage between objective and subjective evaluations.** Table 1 (objective) excludes CosyVoice 2 and FireRedTTS, which appear in Table 2 (subjective), preventing cross-referencing between metrics.

### Trivial
None significant.

## Nice-to-Haves

- Wall-clock latency and RTF measurements at matched compute budgets (especially DISTAR at NFE=10 vs. DiTAR at NFE=10) to substantiate or revise the efficiency claim.
- Inter-patch error propagation analysis: since DISTAR is AR at the patch level, quantifying how quality degrades for later patches in long utterances would be informative.
- Confidence maps across patch positions during iterative decoding to substantiate or refute the "tail-first bias" hypothesis.
- DISTAR at NFE=10 to isolate architecture quality from compute budget.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"RVQ codec fundamentally bounds speaker similarity":** The critic claimed SIM is structurally bounded by the RVQ codec ceiling (0.66/0.70). However, DISTAR-medium actually *exceeds* the codec ceiling on LibriSpeech (SIM=0.67 > 0.66), and E2TTS exceeds it on both benchmarks. The codec resynthesis SIM is not a hard upper bound for the SIM metric, which measures cosine similarity between speaker embeddings of generated audio and the reference prompt—not between generated and reconstructed audio. The codec ceiling argument is factually incorrect as a "fundamental structural limitation."

- **"Eq. 1 mismatch with patch-level architecture":** The critic noted Eq. 1 conditions on c_{<i} (all previous codes) while the model conditions on previous patches. Eq. 1 presents the exact chain-rule factorization; the patch-level model is a practical approximation of this factorization, which is standard and clearly stated in the text ("inference realizes the autoregressive step at the patch level"). This is not a mismatch but a standard presentation approach.

- **"VALL-E 2 and Seed-TTS absent from comparison tables":** This is scope creep—demanding specific baselines beyond what the paper chose to compare. The paper already includes the most relevant contemporaneous baselines (F5TTS, E2TTS, DiTAR, IndexTTS).

- **"WER can be gamed with Whisper-large-v3":** This is a generic concern about the WER metric applicable to any paper using it, not a specific weakness of this paper.

- **"Stochastic layer truncation harms full-model quality":** The critic noted WER increases from 6 to 9 layers in Figure 2. However, the paper explicitly discusses this pattern: "WER changes little and reaches its minimum around six layers. This pattern is consistent with the hypothesis that upper RVQ layers primarily encode acoustic detail rather than linguistic content." The paper does not hide this trade-off.

- **"Missing appendix, proofs":** The parser strips appendices from all papers; these exist in the original submission.

## Novel Insights

The discrete code space creates a distinctive practical profile: DISTAR supports high-quality greedy decoding (a rarity among diffusion-based TTS systems) and test-time bitrate/compute control via RVQ layer pruning. These are genuine and underexplored advantages of the discrete paradigm that go beyond the typical quality-vs-diversity trade-off discussed in continuous systems. However, the efficiency framing is ironic: while discrete tokens enable these practical levers, the architecture as implemented actually requires more diffusion steps (NFE=24) than its continuous counterpart (NFE=10), meaning the practical advantages come at a computational cost the paper does not acknowledge.

## Suggestions

- **Narrow the claims** in the abstract and conclusion to match the evidence: DISTAR achieves SOTA robustness (WER) and strong subjective quality (SMOS/CMOS), with competitive but not leading objective SIM and UTMOS. Drop or qualify the efficiency claim until wall-clock measurements are provided.
- **Report DISTAR at NFE=10** in Table 1. This would directly address whether the WER advantage comes from the architecture or from using 2.4× more diffusion steps.
- **Add wall-clock or RTF measurements** for DISTAR at various NFE values and compare with DiTAR, even if approximate. This is the single most important missing experiment given the paper's efficiency framing.
- **Add a duration analysis** comparing generated utterance durations to reference durations, or at least discuss the implications of fixed-stride generation for prosodic flexibility.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| DiFlow-TTS | /home/wg25r/review_agent/human_reviews_2026/FaGDopTTTC.md | 2.50 | Zero-shot TTS with discrete flow matching, overclaimed, weak baselines. DISTAR is clearly stronger—better architecture, real SOTA WER results, standard benchmarks. |
| CaT-TTS | /home/wg25r/review_agent/human_reviews_2026/VPju7xAxb1.md | 2.00 | LLM-based TTS, limited novelty, unfair comparisons. DISTAR is significantly better. |
| Zero-Shot NAR TTS | /home/wg25r/review_agent/human_reviews_2026/im2a2MHoke.md | 2.50 | NAR TTS, unreliable baselines, missing efficiency metrics. DISTAR has similar efficiency-measurement gaps but much stronger empirical results. |
| Gogo | /home/wg25r/review_agent/human_reviews_2026/JbLmIoWwDC.md | 6.00 | Group-wise codec for TTS, competitive but mixed results, questionable ablations. DISTAR is comparable—both have genuine contributions with incomplete evidence. DISTAR has stronger WER results but more prominent overclaiming. |
| Hierarchical Semantic-Acoustic | /home/wg25r/review_agent/human_reviews_2026/h5KLpGoqzC.md | 5.20 | Semi-discrete TTS, mixed quality results, incomplete evaluation. DISTAR is comparable in profile—real contributions with overclaiming and thin ablations. |
| StableToken | /home/wg25r/review_agent/human_reviews_2026/17DNmdQ9aU.md | 7.50 | Noise-robust tokenizer, strong experiments but overclaimed scope. DISTAR has a similar pattern (strong in one dimension, overclaimed breadth) but with weaker evidence for the breadth claim. |
| AudioX | /home/wg25r/review_agent/human_reviews_2026/qjJWxK3yWo.md | 7.00 | Unified audio generation, strong experiments but overclaimed "anything-to-audio." DISTAR is below this—AudioX has more comprehensive evaluation and the overclaiming is less central to its narrative. |

DISTAR falls between the medium-scoring TTS papers (Gogo at 6.0, Hierarchical at 5.2) and the low-scoring ones (DiFlow-TTS at 2.5). The core architecture is well-motivated, the WER results are genuinely strong, and the practical features (greedy decoding, RVQ pruning) are valuable. However, the efficiency claim is prominently featured and contradicted by the reported NFE numbers with zero supporting evidence, and the breadth-of-superiority claim in the abstract is not supported by the objective metrics. These overclaims are not minor footnotes—they are central to the paper's narrative. The paper is stronger than DiFlow-TTS/CaT-TTS (which had fundamental methodology issues) but weaker than Gogo (which had similar mixed evidence but less aggressive overclaiming). I place it slightly below the Gogo/Hierarchical cluster due to the severity of the unsupported efficiency claim.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>