Now I have enough calibration context. Let me compile the final review.

The key calibration anchors:
- CR-CTC (similar topic: improving CTC for ASR, novel insight + strong experiments): scores 8,8,5,6 → Accept (Poster)
- AV-CPL (pseudo-labeling for AV speech recognition, incremental): scores 3,5,5,3,3 → Rejected
- Multilingual VSR (incremental extension of AV-HuBERT): scores 6,6,8,5,6 → Rejected
- CAV2vec (self-distillation for AV speech, simple method): scores 5,8,8,6 → Accept (Poster)
- SMILE (AVSR, limited novelty): scores 5,5,5,5 → Rejected
- HAINAN (similar concepts: scores 5,8,8,6 → Accept (Poster))

Now let me synthesize the review.

Let me carefully evaluate each of the harsh critic's points against the actual paper content:

**Issue 1: Core mechanism only partially validated for attention branch**
The paper claims "improved decoder robustness" but evaluates end-to-end with joint CTC-attention decoding. The harsh critic argues we don't know if the attention decoder itself is better, or if gains come from stronger CTC influence. Let me check what the paper actually shows:

- Table 4 ablations show that within USR 2.0, removing CTC supervision from the decoder hurts OOD (row where decoder only predicts Att PL: OOD=35.1 vs default CTC-driven mode where decoder predicts both: OOD=24.2)
- But this is within USR 2.0 — not compared against USR's decoder
- Figure 3c shows beam-size sweep: USR 2.0 does better at small beams. But inference uses joint CTC-attention with CTC weight 0.1
- The paper doesn't show attention-only decoding results for USR vs USR 2.0

This is a valid concern. The paper repeatedly frames improvements as "decoder robustness" but doesn't isolate the decoder contribution. However, I should note that USR 2.0 IS a joint CTC-attention system, and the improvements are real. The question is whether the claim about the attention decoder specifically is overclaimed.

**Issue 2: Speedup claim "2×" not quantitatively substantiated**
Let me check. The paper says "approximately 2× faster training" and points to Figure 5. It mentions 50 vs 75 epochs. But there's no detailed breakdown. This is a valid minor point — the claim could be better supported — but Figure 5 does show wall-clock time curves.

**Issue 3: Experimental controls don't fully separate algorithmic gains**
The paper uses the same architecture, same training setup as USR but changes the PL scheme. The main differences are: different number of epochs (50 vs 75) and the PL method. The paper doesn't re-run USR for 50 epochs to match. This is a minor concern.

**Issue 4: OOD evaluation uses Whisper as oracle**
The paper does acknowledge Whisper as oracle. This is a common practice. The critic calls it "structural/methodological tension" but it's quite standard. And importantly, LibriSpeech test-clean is used with true labels.

**Neutral reviewer's weaknesses:**
1. Limited analysis of global incoherence — valid desire
2. Hyperparameter sensitivity of mixed sampling — partially addressed by Figure 4
3. Reliance on Whisper for OOD — same as point 4 above
4. Missing NAT baselines — reasonable but not standard for this comparison

**Spark's weaknesses:**
1. No CTC-only PL baseline — the ablation in Table 4 does test removing attention PLs (decoder predicts only CTC PLs), and the paper argues CTC-only PLs sacrifice modeling capacity
2. OOD only under greedy decoding — valid; the default is beam 40
3. No audio-only baselines — out of scope; the paper's focus is unified models
4. No PL quality evaluation — valid

**Human finder's weaknesses:**
1. Incremental novelty: CTC-driven teacher forcing + scheduled sampling — I think this is actually more novel than the human finder suggests. The key insight that globally incoherent CTC-driven PLs can be used effectively for teacher forcing in self-training is non-trivial.
2. Limited ablation on hyperparameters — Figure 4 does address sampling probability
3. Huge model not apples-to-apples — valid concern
4. Insufficient analysis of failure modes and incoherence — valid
5. Reproducibility concerns — per my instructions, I should remove this

Let me now write the final review.</think>Now I have sufficient context for calibration. Let me carefully verify the key criticisms against the paper and synthesize the final review.

## Summary

The paper proposes USR 2.0, an improvement over the Unified Speech Recognition (USR) framework for training a single model for audio, visual, and audiovisual speech recognition. The core contribution is CTC-driven teacher forcing: instead of generating attention-based pseudo-labels via slow autoregressive decoding, greedily decoded CTC pseudo-labels are fed as forced inputs to the decoder, producing attention targets in a single forward pass. A mixed sampling strategy (50% CTC-driven, 50% AR) mitigates the resulting train-test mismatch. USR 2.0 halves training time, improves robustness to OOD inputs (long utterances, noise, unseen datasets), and achieves state-of-the-art results on LRS3, LRS2, and WildVSR.

## Strengths

- **Clear, well-motivated technical contribution:** The insight that globally incoherent CTC-driven pseudo-labels can effectively train an attention decoder in the self-training setting — because teacher and student share the same conditioning — is elegant and non-trivial. It simultaneously addresses the AR decoding bottleneck and OOD brittleness of decoupled supervision with a single mechanism.

- **Comprehensive and convincing empirical evaluation:** The paper systematically evaluates robustness across three axes — long utterances (Figure 3), noise (Table 1), and OOD datasets (Table 3) — showing consistent and substantial improvements over USR and self-supervised baselines. In-distribution results on LRS3 (Table 2) also show consistent gains, and scaling to a Huge model yields strong SOTA numbers (VSR 17.6%, ASR 0.9%, AVSR 0.8%).

- **Meaningful practical impact:** The ~2× training speedup (Figure 5, fewer epochs 50 vs 75) addresses a real scalability bottleneck in semi-supervised training, and improved greedy decoding performance (Figure 3c) is directly relevant for practical deployment where beam search may be too costly.

- **Thoughtful ablations:** Table 4 carefully dissects the effect of different pseudo-label target assignments for CTC and attention branches, and Figure 4 explores the mixed sampling probability trade-off. These provide genuine insight into why the design works.

## Weaknesses

### Major

- **Decoder robustness claims are conflated with CTC branch contributions:** The paper repeatedly frames improvements as "improved decoder robustness" (e.g., Section 1: "improves robustness to out-of-distribution inputs"; Section 5.1: "integrated CTC and attention supervision mitigates autoregressive drift"). However, the main evaluations use joint CTC–attention decoding with CTC weight 0.1 (Section 4.3), making it impossible to isolate whether the *attention decoder itself* has improved or whether the gains stem from a more robust CTC branch combined with CTC rescoring. There is no experiment with attention-only decoding (disabling CTC at inference) for USR vs USR 2.0, and the ablations in Table 4 are internal to USR 2.0 only. The most compelling evidence — Figure 3c (beam-size sweep under joint decoding) and Table 3 (greedy decoding) — is consistent with either explanation. This does not invalidate the system-level improvements, but the specific claim that the *attention decoder* is more robust is under-supported.

- **CTC-only pseudo-labelling baseline is missing:** The paper argues that "CTC-only supervision... comes at the cost of reduced sequence modelling capacity" (Section 2), referencing prior work. However, it never directly compares USR 2.0 against a CTC-only PL variant under the same training setup. Table 4 does show that removing attention PLs from the decoder hurts in-distribution performance (which is expected), but this comparison is within USR 2.0's coupled regime rather than against a pure CTC-PL baseline that might still achieve some of the robustness gains while being even simpler. Given that the paper's core contribution is about how to generate PLs efficiently, a comparison with CTC-only PLs is the most natural baseline.

### Minor

- **Speedup quantification is imprecise:** The "halves training time" claim is supported by Figure 5 (wall-clock curves for VSR only) and the note that USR 2.0 uses 50 vs 75 epochs. The per-step speedup from CTC-driven teacher forcing is not separately quantified, nor is the total speedup replicated across all modalities. The claim is plausible and likely directionally correct, but the "2×" headline is stronger than the data in the main body.

- **Huge model comparison is not apples-to-apples:** The Huge* model in Table 2 is trained on LRS2+LRS3 (labelled) and English-only VoxCeleb2+AVSpeech (unlabelled) — a different and larger dataset than the other models. While the paper marks this with an asterisk, it makes attribution of SOTA gains to USR 2.0's method vs. data scaling ambiguous for this configuration.

- **Global incoherence of CTC-driven attention PLs is acknowledged but not empirically characterized:** The paper recognizes that CTC-driven teacher forcing can produce globally incoherent attention pseudo-labels (Section 4.1, Appendix C.4) and provides a verbal argument for why this is acceptable. A quantitative analysis (e.g., N-gram statistics, LM perplexity, or PL WER comparing CTC-driven vs. AR attention PLs) would strengthen this claim. Without it, the reader cannot assess how severe this incoherence is in practice.

- **OOD robustness results on WildVSR and AVSpeech rely on Whisper-generated transcripts:** While common in the field and acknowledged by the authors, WildVSR and AVSpeech evaluations use Whisper as an oracle, introducing systematic biases (e.g., in punctuation, disfluency handling) that could differentially advantage certain model architectures. The LibriSpeech test-clean result (25.3→15.4 WER vs. USR) under greedy decoding uses genuine ground truth and is more reliable; it would benefit from being highlighted more prominently.

### Trivial

- **Table 4 has incomplete cells:** Several rows have blank ID/OOD entries, making it harder to fully interpret the ablation. A footnote or explanation would help.

## Nice-to-Haves

- **Attention-only decoding comparison:** Running USR and USR 2.0 with CTC disabled at inference would cleanly isolate decoder-only contributions and significantly strengthen the robustness claims.
- **Pseudo-label quality tracking:** Measuring PL WER against oracle transcriptions over training iterations for both USR and USR 2.0 would directly validate the self-reinforcing error mitigation hypothesis.
- **NAT baseline comparison:** Comparing against a non-autoregressive transformer baseline for pseudo-labelling would contextualize the approach within the broader parallel decoding literature.
- **Failure case analysis:** Identifying conditions where CTC-driven teacher forcing underperforms AR PLs would strengthen trust in the method's generality.

## Removed Points

- **Missing audio-only ASR baselines (wav2vec 2.0, HuBERT):** Suggested by Spark reviewer. This is scope creep — the paper's focus is unified models that handle A/V/AV with shared parameters. Comparing against audio-only self-supervised models would not directly address the claims.

- **Reproducibility concerns about baseline numbers:** Suggested by Human Finder. The paper cites published results from known benchmarks (LRS3, LRS2) and standard methods (AV-HuBERT, BRAVEn). Suggesting baselines may be unfairly re-implemented without evidence is speculative.

- **Incremental novelty claim (CTC + scheduled sampling = existing ideas):** Suggested by Human Finder. While individual components (CTC-attention joint training, scheduled sampling) exist, the specific insight — that CTC-driven teacher forcing produces effective pseudo-labels *despite global incoherence* in the self-training setting — is novel and non-trivial, not a simple combination.

- **Confidence threshold and loss weight ablations:** Suggested by Human Finder and Spark. These are standard hyperparameters in the USR framework carried over directly. Systematic ablation of every hyperparameter is not reasonable scope for this paper.

- **Formatting/style nitpicks:** Per instructions, removed.

## Novel Insights

The key conceptual insight — that globally incoherent pseudo-labels can effectively train an attention decoder in the self-training regime because teacher and student share the same conditioning — is genuinely counterintuitive and well-supported. This challenges the common assumption that pseudo-label quality must be high for knowledge distillation/self-training, and suggests that the *consistency* of the training signal (matching conditioning between teacher and student) matters more than the *global quality* of the labels. This has implications beyond speech recognition for any sequence-to-sequence self-training setting where input-output monotonic alignment holds.

## Suggestions

- Run a small experiment with attention-only inference (e.g., beam search with CTC weight 0) for USR vs. USR 2.0 to isolate the decoder's contribution to robustness.
- Add PL quality metrics (WER of teacher-generated PLs against oracle/Whisper transcripts at multiple training checkpoints) to directly validate the self-reinforcing error hypothesis.
- Provide a per-step time comparison (ms/step) for USR vs. USR 2.0 across modalities to explicitly decompose the 2× speedup into per-step efficiency vs. fewer epochs.

## Score and Decision

**Calibration:** I compared against CR-CTC (8,8,5,6 → Accept Poster), which also proposes a simple but effective modification to CTC-based training with strong empirical results across multiple datasets. The CAV2vec paper (5,8,8,6 → Accept Poster) applies self-distillation to AV speech with incremental novelty. The AV-CPL paper (3,5,5,3,3 → Rejected) proposes semi-supervised PL for AVSR but had much weaker results and novelty. The SMILE paper (5,5,5,5 → Rejected) had limited novelty in AVSR.

USR 2.0 is closer to CR-CTC in profile: a focused, technically sound insight with strong empirical results across multiple settings. However, the decoder robustness claims are somewhat overstated relative to the evidence (the system improvement is clear; the decoder-specific improvement is not isolated), and the missing CTC-only PL baseline is a gap. The paper's novelty is higher than CR-CTC's (the CTC-driven teacher forcing insight is more creative than consistency regularization), but the empirical support for the specific *decoder robustness* narrative is not as tight as it should be.

Overall, this is a strong paper with a real and impactful contribution that would benefit from more targeted decoder-isolating experiments. The core idea is novel and practical, and the results are comprehensive and meaningful.

MY FINAL SCORE: <pineapple>7</pineapple>
MY FINAL DECISION: <orange>Accept</orange>