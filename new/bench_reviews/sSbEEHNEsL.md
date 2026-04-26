Now I have sufficient calibration data. Let me synthesize the review.

**Calibration summary:**
- High anchors (avg ≥6): Zipformer (7.5, oral), Progressive distillation (7.6, oral), InfoBatch (7.0, oral), nGPT (6.5, poster), HAINAN (6.75, poster), UADF (6.6, poster), SpeechTokenizer (5.75, poster)
- Medium anchors: G-Local Attention (3.0, reject), Self-Supervision vs Semi-Supervised (4.0, reject), Strided Transformers (2.33, reject)
- This paper improves upon USR with CTC-driven teacher forcing, achieving OOD robustness, efficiency, and moderate ID improvements. It's similar in scope to HAINAN (extending a predecessor model with new inference modes for efficiency, avg 6.75) and InfoBatch (efficiency + maintained quality, avg 7.0).

The paper has genuine contribution: the CTC-driven teacher forcing idea is clever, the OOD robustness story is well-supported, the efficiency gain is real, and the paper is well-written. The weaknesses are moderate: the 2× claim is slightly stretched (conflates two factors), OOD improvements are largest under greedy decoding, and ID improvements at Base scale are marginal. None of these are fatal.

Let me finalize.

## Summary

USR 2.0 proposes CTC-driven teacher forcing for pseudo-labelling in unified speech recognition, replacing slow autoregressive decoding with a single forward pass through the decoder conditioned on CTC outputs, coupled with mixed sampling to mitigate exposure bias. This simultaneously improves training efficiency (~2× wall-clock), OOD robustness (large gains on long utterances, noisy audio, and cross-dataset evaluations), and moderately improves in-distribution performance, achieving state-of-the-art results on LRS3/LRS2/WildVSR across ASR, VSR, and AVSR with a single model.

## Strengths

1. **CTC-driven teacher forcing is an elegant and well-motivated idea.** The key insight—that globally incoherent pseudo-labels are acceptable because teacher and student share the same CTC-derived conditioning—is carefully argued (Section 4.1, "Global coherence" paragraph). Table 4's ablation confirms this: CTC-driven mode achieves 24.2% OOD WER versus AR mode's 45.1%, a dramatic improvement from removing autoregressive dependency during PL generation.

2. **Strong and consistent OOD robustness improvements.** Multiple evaluations (Table 1: noise robustness; Table 3: cross-dataset; Figure 3a: long utterances under greedy; Figure 3b: long utterances under beam search; Figure 3c: beam size sensitivity) paint a consistent picture. USR 2.0 degrades gracefully where USR's decoder collapses, particularly on long utterances and under domain shift.

3. **Genuine practical training efficiency gain.** Figure 5 shows wall-clock speedups across model scales. The per-step speedup from eliminating AR decoding is concrete and well-demonstrated, and is the more defensible component of the efficiency claim.

4. **Well-designed ablations.** Table 4 systematically varies which PL types supervise each branch, cleanly showing that CTC supervision improves OOD, attention targets improve ID, and both are needed. Figure 4's mixed sampling ablation shows clear trade-offs between ID, OOD, and speed.

## Weaknesses

### Fatal
None.

### Major

- **The "2× faster training" claim conflates per-step speedup with an uncontrolled convergence difference.** Section 6 states the speedup is "driven by two key factors: (i) faster training steps due to CTC-driven teacher forcing; and (ii) faster convergence, requiring fewer epochs (50 vs. 75)." Factor (i) is a genuine methodological contribution. Factor (ii) could result from hyperparameter choices rather than an inherent property of USR 2.0—USR may converge just as fast with a different learning rate schedule or early stopping. Without running USR for 50 epochs with tuned hyperparameters to confirm it underperforms, the 2× multiplier reflects an uncontrolled variable. The wall-clock curves in Figure 5 show a real practical speedup, but the decomposition into two factors is only partially justified. This matters because it inflates a genuine ~per-step speedup into a broader "halves training time" headline.

- **OOD evaluation under greedy decoding somewhat overstates the robustness advantage over USR.** Table 3 reports dramatic OOD numbers (e.g., LibriSpeech ASR: 15.4 vs 25.3) under greedy decoding. The paper itself shows (Figure 3c) that this gap narrows substantially with larger beam sizes at inference time. While greedy decoding is the relevant regime for pseudo-labelling (the paper's focus), readers may interpret these numbers as reflecting deployment-time robustness. The paper would be more honest reporting beam-searched OOD numbers alongside the greedy ones in Table 3, or more prominently noting this caveat at the point of claiming "wide margin" improvements.

### Minor

- **In-distribution improvements over USR are modest at smaller scales.** At Base scale on LRS3, VSR actually regresses (36.0→36.2), and ASR/AVSR improve by only 0.1–0.3 WER. The paper's "SOTA" framing mixes comparisons with fundamentally different paradigms (self-supervised methods like AV-HuBERT that don't use pseudo-labelling). The primary contribution is OOD robustness and efficiency; the in-distribution story is supportive but not the headline.

- **The 0.5 mixed sampling probability is only studied on Base scale.** Figure 4 sweeps this hyperparameter on the Base model. Whether 0.5 remains optimal for Large/Huge models is unknown, which matters for the scalability story. The paper acknowledges an adaptive schedule "performed similarly" but defers details.

### Trivial
None worth noting.

## Nice-to-Haves

- Adding beam-searched OOD results (at least beam 5 or 10) to Table 3 would give a more complete picture of inference-time robustness and prevent potential reader misinterpretation.
- Training USR for exactly 50 epochs as a controlled comparison would cleanly isolate per-step efficiency from convergence effects and strengthen or qualify the 2× claim.
- Showing examples or token-level statistics of CTC-driven attention PLs vs. AR attention PLs would make the "local coherence without global coherence" argument more tangible.

## Removed Points

- **"The abstract claims SOTA surpassing modality-specific baselines, conflating different training paradigms."** — The paper compares against the best available baselines for the same tasks. This is standard practice and the paper is explicit about the comparison.
- **"What fraction of training time does AR pseudo-labelling consume?"** — While useful, this is a nice-to-have profiling detail, not a methodological flaw. The per-step speedup is directly demonstrated in Figure 5.
- **"VoxCeleb2 evaluation uses only ~2,000 samples bucketed into ~200-sample groups, making curves noisy."** — The paper uses an oracle (Whisper) for WER computation on these samples, and the trends are clear across frame buckets. This is a minor statistical concern, not a methodological flaw.
- **"Huge model uses fundamentally different data scale."** — The paper explicitly notes this model trains on LRS2+LRS3 labelled + VoxCeleb2+AVSpeech unlabelled. It is presented as a scaling result, not a direct comparison.
- **"Table 4 partially garbled."** — This is a parser artifact, not an author error.
- **"Missing adaptive sampling schedule details."** — The appendix is referenced (C.2) but stripped by the parser. The authors clearly have this analysis.
- **"No analysis of CTC-driven attention PL quality."** — This would be informative but is not required; downstream performance and the ablation in Table 4 provide indirect but strong evidence.
- **Strength Finder claim: "Approximately 2× training speedup with no accuracy loss"** — The "no accuracy loss" part is slightly overstated given the Base VSR regression; downgraded to note that ID results are mixed at small scale.

## Novel Insights

The paper's most interesting finding is the "global coherence is unnecessary" insight for pseudo-labelling: because teacher and student share the same conditioning, even incoherent CTC-derived sequences enable effective knowledge transfer. This is counterintuitive but well-justified, and the 24.2% vs. 45.1% OOD WER gap in Table 4 makes a compelling case. The interplay between CTC's monotonic alignment (robust but less expressive) and attention's token-level precision (expressive but brittle) is navigated effectively through the dual-PL supervision design, where the decoder simultaneously predicts both aligned targets.

## Suggestions

- Qualify the "2× faster" claim to clearly separate per-step speedup from convergence differences, or add a controlled 50-epoch USR baseline to validate the convergence component.
- Add beam-searched OOD results (even just beam 5 or 10) to Table 3 to give readers a complete picture of inference-time robustness.

## Calibration

**Anchors compared against:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Zipformer | 9WD9KwssyT | 7.5 | Stronger SOTA contribution in ASR with broader empirical sweep; USR 2.0 has comparable engineering quality but narrower scope |
| HAINAN | LrmPGtnros | 6.75 | Similar profile: extending a predecessor architecture with AR/NAR modes for efficiency; USR 2.0 has stronger OOD story |
| InfoBatch | C61sk5LsK6 | 7.0 | Comparable training speedup story with lossless claims; USR 2.0 has similarly real-but-slightly-overclaimed speedup |
| Progressive Distillation | wPMRwmytZe | 7.6 | Stronger theoretical contribution; USR 2.0 is more empirical |
| nnYsWoe1ST (Self-Sup vs Semi-SL) | nnYsWoe1ST | 4.0 | Much weaker paper with confounded comparisons; USR 2.0 is far above this |
| G-Local Attention | LkQoiVp6XG | 3.0 | Fundamentally flawed paper with unfair settings; USR 2.0 is not comparable |

This paper falls between HAINAN (6.75) and InfoBatch (7.0) in terms of contribution profile. It has a cleaner core idea than HAINAN (CTC-driven teacher forcing is more novel than randomly dropping a predictor network) and comparable empirical support, but with a slightly overclaimed speedup. It falls below InfoBatch's "lossless" speedup claim quality (InfoBatch has cleaner controlled experiments) but has a more interesting methodological insight. The OOD robustness story is a strong differentiator that neither comparison paper has.

**Score: 6.5**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>