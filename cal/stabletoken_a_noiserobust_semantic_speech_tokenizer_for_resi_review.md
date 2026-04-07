=== CALIBRATION EXAMPLE 89 ===

# Harsh Critic Review
---

## Section-by-Section Critical Review

### Title & Abstract

The title "StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs" accurately reflects the paper's contribution. The abstract's core claims—a 60%+ relative UED reduction and downstream gains in ASR, SER, and TTS—are all substantiated by Tables 1–3 and Figure 3. The claim that the method introduces "negligible inference overhead" is well-supported by the latency benchmarks in Tables 12–14. One minor concern: the abstract characterizes the two root causes of instability (brittle architecture + distant supervision) with confidence before demonstrating these diagnoses empirically. The ablation in Table 4 does eventually validate this framing, but a reader could reasonably ask whether the gains come from noise-aware data augmentation alone, which the abstract does not caveat.

---

### Introduction & Motivation

The motivation is crisp and genuinely important: existing supervised semantic tokenizers (e.g., CosyVoice2, GLM-4-Voice) are surprisingly fragile even at high SNRs where speech is perceptually unchanged. This is illustrated effectively in Figure 1. The two-flaw diagnosis—single-path architecture + distant supervisory signal—is well-argued. The dismissal of naive remedies (offline ensembles, token-level consistency losses) is appropriately brief but reasonably convincing.

**Concern:** The introduction argues that the ASR training objective is "indifferent to intermediate token stability," which is true but not explored formally. It is not obvious why an ASR-trained quantizer would necessarily be brittle: in principle, if two nearly identical speech signals yield the same transcription, the representations could still be consistent. The paper provides no experiments that measure *why* the ASR loss fails to provide stability (e.g., does training with a longer, richer, lower-bitrate codebook help at all?), relying instead on empirical observation. This is sufficient for a practical paper, but the causal claim is slightly overstated.

---

### Method (Section 2)

**Voting-LFQ Module (§2.2):** The design is clean and technically sound. Branching at the projection layer (not at the encoder) is a key decision: each branch has its own linear map W_i h + b_i, binarized via sign(). During training, aggregation uses a soft average (Eq. 2); during inference, a hard sign is applied to this average (Eq. 3), implementing a bit-wise majority vote. The use of LFQ (Look-up-Free Quantization) instead of standard VQ is well-motivated—it allows the bit-wise voting to be meaningful, since LFQ maps the code space directly onto binary hypercube positions.

**Critical concern—inference-time independence of errors:** The paper's central narrative for robustness is that bit-level errors across branches are *sparse and independently distributed*, so the majority vote can correct them (§2.2, case study in Table 6). **However, at inference time, all n branches receive the same noisy encoder hidden state h_noisy**—their inputs are identical. The branches differ only in their projection matrices W_i. The diversity of errors across branches therefore depends entirely on whether different linear projections of the *same* h_noisy produce independently-erring binarizations. If noise shifts h_noisy in a direction that is aligned with the sign boundaries of most branches, all branches may flip the same bits. This theoretical limitation is not acknowledged. The case study (Table 6) shows convincing empirical recovery, but it is cherry-picked (single example) and cannot establish that independent error patterns are the rule rather than the exception. An analysis of the cross-branch error correlation structure would substantially strengthen this argument.

**Noise-Aware Consensus Training (§2.3):** The setup—majority of branches receive h_clean, minority receive h_noisy—is sensible. The consensus loss (Eq. 4) penalizes each branch's deviation from the global mean p̄_all.

**Concern—target contamination:** p̄_all is computed from *all* branches, including the k noisy ones. So the "clean anchor" is actually a convex combination of clean and noisy representations, not a purely clean signal. The paper justifies this by saying noisy branches are "diluted" by the majority (Appendix B.4), which is true, but the claim that "noisy-minority branches are forced to align with the clean consensus" (§2.3) is somewhat imprecise. If k is close to n/2, the contamination is substantial. A sensitivity analysis showing that the loss still effectively suppresses noisy signals across the full range of valid k values would be welcome. The paper fixes k<n/2 (which ensures majority clean, i.e., k ≤ 2 for N=5) and the empirical results confirm it works, but the theoretical argument could be tightened.

**Concern—encoder is not robustified:** The encoder backbone (Whisper-large-v3) processes the full noisy audio and produces h_noisy, which is then fed into all branches. The training strategy only teaches the quantization stage to be robust, not the encoder itself. This means that severe noise corrupting the encoder output may be beyond the voting mechanism's repair, since all branches will see the same highly-corrupted h_noisy. This is a fundamental architectural limit that should be stated explicitly.

---

### Experimental Setup (Section 3)

The downstream evaluation design is commendable: a controlled isogenic framework where each tokenizer is swapped into an otherwise identical SpeechLLM (Qwen2.5-3B + same fine-tuning recipe) ensures that differences are attributable to tokenizer quality.

**Major concern—training data imbalance:** StableToken is trained on ~183k hours of disclosed open-source data (Table 7) plus undisclosed in-house data, totaling "150k-hour" of processed speech (confusingly, the text says "150k hours" while the table lists more than that). In contrast, the strongest baselines (GLM-4-Voice, CosyVoice2, S³ Tokenizer) are from published systems with no explicit data matching. The paper does not include a baseline that uses the *same* training data as StableToken but with a conventional single-branch LFQ or VQ quantizer. This omission makes it impossible to determine whether the dramatic UED improvements stem from the architectural innovation, the noise-aware training strategy, or simply from training on more/better-curated data. **This is the most critical experimental gap in the paper.** An additional ablation with a single-branch LFQ model trained on the same 150k-hour corpus (i.e., same data, no multi-branch, no consensus loss) is essential.

**Concern—vocabulary size confound:** StableToken uses a vocabulary size of 8192 (13 bits), compared to 4096 (S³ Tokenizer, CosyVoice1) and 16384 (GLM-4-Voice). UED is measured as edit distance between token sequences. A larger vocabulary creates finer granularity, and it is not obvious whether UED is directly comparable across different vocabulary sizes. The paper argues that a larger vocabulary makes the result "even more significant," but the opposite could be argued: a larger vocabulary means quantization boundaries are denser, which might make it easier to have nearby centroids that avoid large UED jumps under noise. Without normalization or a matched-vocabulary ablation, this comparison is ambiguous.

---

### Results (Section 4)

**Token robustness (Table 1):** The improvements are striking—10.17% average UED vs. 26.17% for the next-best supervised tokenizer. The OOD generalization (ESC-10 not seen during training) is particularly compelling, suggesting genuine robustness rather than overfitting to the training noise distribution.

**Reconstruction fidelity (Table 2):** StableToken achieves the best WER on LibriSpeech (3.84/7.99 clean/other) while matching or slightly trailing GLM-4-Voice on MOS for English but beating it on Chinese. The fact that robustness does not trade off against fidelity is an important positive result. However, the flow matching model used for reconstruction evaluation (Table 10) is trained on separate data; differences in this training data could partially explain reconstruction quality differences between tokenizers.

**Downstream ASR (Figure 3, Table 3):** The widening performance gap under increasing noise is a compelling visualization. The CHiME-4 benchmark (real-world multi-microphone noise) is especially convincing since it's genuinely out-of-domain. A ~30% relative WER reduction on CHiME-4 test-real (35.90% vs. 51.08% for GLM-4-Voice) is substantial.

**Downstream SER (Figure 3):** The gains are consistent and follow the same pattern of widening under noise. However, there is no analysis of whether the tokenizer's ability to preserve prosodic/emotional cues specifically is what drives the improvement, or whether the general robustness is sufficient.

**TTS (Table 3):** The WER improvements in TTS are very large (e.g., 4.43 vs. 6.19 for GLM-4-Voice on SEED-TTSEN). The claim that "enhanced token consistency simplifies the learning task" for TTS is plausible but only a hypothesis. An alternative explanation—that the better reconstruction flow matching model (trained on more data) accounts for this—is not ruled out.

**Ablation (Table 4):** The sequential ablation is well-structured. Removing each component sequentially shows that all three (multi-branch, noise-aware training, consensus loss) contribute. But the ablation reuses the *same* training data, so it cleanly isolates the architectural/training contributions *given* that data—which is valuable. The reported validation-set WER numbers (2.03/4.68) are very low compared to the full model in Table 2 (3.84/7.99), suggesting these may reflect a different evaluation protocol (ASR during tokenizer training, not the full reconstruction pipeline). This discrepancy should be clarified.

**Statistical significance:** No confidence intervals, error bars, or significance tests are reported anywhere. For WER and accuracy metrics, this is particularly important given that some differences (e.g., MOS comparisons) are small.

---

### Writing & Clarity

The paper is well-organized and clearly written. Section 2's description of the method is logically structured. The distinction between training-time (soft average) and inference-time (hard vote) behavior in the Voting-LFQ module is explained clearly. One genuine clarity issue: the ablation study (Table 4) evaluates ASR on a "validation set during tokenizer training," which is a very different pipeline from the downstream WER in Table 2 or Table 3. This is confusing because both use "WER%" as the metric label. A clearer label (e.g., "ASR CTC-WER during tokenizer training (validation)") would help readers compare results across tables accurately.

---

### Limitations & Broader Impact

There is **no dedicated limitations section**, which is a significant omission for an ICLR submission. The conclusion is one short paragraph that does not acknowledge any failure modes. Key limitations not discussed:

1. **30-second context window:** The 30-second Whisper window (B.8) limits applicability to streaming/long-form audio. The boundary stability analysis (Table 19) shows no degradation at chunk boundaries, but concatenating chunks without context may degrade discourse-level coherence.

2. **Encoder bottleneck:** Robustification is applied only at the quantization stage. If noise severely corrupts the Whisper encoder's output, the voting mechanism has nothing to work with—all branches see the same damaged h_noisy. The method's effectiveness at very low SNRs (0 dB in ASR) is demonstrated empirically, but the mechanism's limits are not characterized.

3. **Data transparency:** Training on "in-house data" (beyond the listed open-source sets) limits reproducibility despite the published code. The reproducibility statement promises the checkpoint, but the proprietary training data cannot be shared.

4. **Language coverage:** The method is only evaluated on English and Chinese. The encoder (Whisper-large-v3) supports many languages, but whether robustness generalizes to morphologically diverse or tonal languages beyond Chinese is unknown.

5. **Societal implications:** Improving speech robustness in SpeechLLMs has potential dual-use concerns (e.g., surveillance in noisy environments, more effective voice-based manipulation) that are not acknowledged.

---

### Overall Assessment

StableToken addresses a real and underappreciated problem—the fragility of supervised semantic speech tokenizers to acoustic perturbations—with a technically sound and well-motivated solution. The Voting-LFQ module is a principled extension of LFQ to multi-branch consensus, and the Noise-Aware Consensus Training is a clean training paradigm that complements the architecture. The experimental gains, particularly on downstream CHiME-4 ASR and the OOD noise generalization, are substantial and credible. However, the paper has one critical experimental weakness: the absence of a matched-data baseline (same 150k+ hour corpus, single-branch LFQ or VQ, with and without noise augmentation). Without this control, it remains unclear whether the gains arise from the architectural innovation or from training-data advantages, a concern heightened by the fact that baselines are pre-existing publicly-released models trained on much less data. Additionally, the theoretical justification for inference-time voting efficacy rests on an unvalidated assumption of cross-branch error independence—all branches see identical noisy encoder outputs, and their projected representations are not theoretically guaranteed to fail independently. These are significant enough concerns that revision to include the matched-data ablation would substantially strengthen the paper's claims and bring it to a clear accept level. The current state is a borderline case: the approach is novel, the results are impressive, but the experimental foundation is not fully controlled.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the critical lack of noise robustness in semantic speech tokenizers, arguing that their fragility degrades downstream SpeechLLM performance. The authors propose StableToken, featuring a multi-branch Voting-LFQ architecture and a Noise-Aware Consensus Training strategy. Experimental results demonstrate that StableToken achieves state-of-the-art token stability (Unit Edit Distance) while maintaining high reconstruction quality and significantly improving downstream ASR, SER, and TTS tasks under noisy conditions.

### Strengths
1.  **Identification of a Critical Problem:** The paper effectively identifies and validates a specific vulnerability in modern semantic tokenizers (e.g., GLM-4-Voice, CosyVoice) regarding noise-induced token instability. The claim that even imperceptible perturbations break speech-text alignment is well-supported by the reported Unit Edit Distance (UED) metrics, which show drastic degradation in baselines compared to StableToken's stability.
2.  **Effective Architectural Innovation:** The Voting-LFQ mechanism introduces a novel way to handle discrete quantization robustness. By utilizing bit-wise voting across parallel branches, the method achieves error correction at a granular level that single-path VQ cannot. The empirical results (Table 1, as described in text) show a relative UED reduction of over 60% compared to the best baseline (26.17% to 10.17%), which is a substantial improvement.
3.  **Comprehensive Downstream Validation:** Rather than stopping at tokenizer metrics, the authors rigorously evaluate Stability as a foundational component for SpeechLLMs. The inclusion of ASR, SER, and TTS tasks confirms that the architectural changes translate to real-world utility. Specifically, Table 3 shows consistent WER reductions across noise levels in downstream ASR tasks (e.g., 30% relative reduction on CHiME-4).
4.  **Maintained Efficiency:** Despite the multi-branch design, the paper provides evidence (Appendix B.6) that the inference overhead is negligible. The reported FLOPs increase by only 0.01% and memory usage is actually lower than baselines (likely due to the LFQ vs. VQ choice), challenging the assumption that robustness requires higher computational cost.

### Weaknesses
1.  **Lack of Training Cost Analysis:** While inference efficiency is well-documented, the paper does not sufficiently discuss the impact of the multi-branch design on *training* complexity. The Noise-Aware Consensus Training requires generating perturbations and processing $N$ branches for every step. It is unclear if this necessitates a proportional increase in training FLOPs, batch size constraints, or total time compared to standard single-path training, which is a practical concern for scaling.
2.  **Limited Multilingual Diversity in Results:** Although the tokenizer is trained on mixed English/Chinese data (Table 7), the primary robustness evaluations (Table 1, Table 3) focus heavily on English tasks (LibriSpeech, CHiME-4, GLM-4-Voice). The Appendix B.7 provides vocabulary analysis but lacks robustness metrics for non-English ASR/Translation tasks. Given the prevalence of low-resource languages in speech processing, this generalization gap needs to be addressed.
3.  **Ambiguity on Inference Latency:** The paper claims "negligible" latency increase and shows memory savings. However, running 5 parallel projection heads involves higher memory bandwidth consumption. The provided latency data (Table 13) compares StableToken only to GLM-4-Voice. A comparison against simpler single-path tokenizers (like CosyVoice or Whisper VQ) would provide a better baseline for understanding the true hardware cost of the Voting mechanism.
4.  **Visual Presentation Issues:** While I must treat parser artifacts as external issues, the paper's current figures (specifically Figure 3 tables) appear heavily garbled in the source text provided. This suggests that if the tables in the final version are not clean and clearly readable, it will hinder the reviewer and reader's ability to assess the statistical significance of the gains, particularly for lower-bound metrics like MOS.

### Novelty & Significance
The paper demonstrates significant novelty in applying a consensus-based, multi-branch voting mechanism to semantic speech tokenization, adapted from techniques often seen in ensemble learning to the discrete quantization domain. The "Consensus Loss" that operates on continuous pre-quantization vectors to stabilize discrete codes is a methodologically sound contribution to representation learning. The significance is high because tokenizer instability is a fundamental bottleneck for deploying SpeechLLMs in real-world, noisy environments. By solving this at the representation layer, the paper offers a solution that benefits a wide range of downstream modalities rather than just an ASR model specific fix.

### Suggestions for Improvement
1.  **Clarify Training Efficiency:** Provide data or estimates on training time, memory footprint, or FLOPs during the *training* phase. Compare it to the baseline training cost to confirm that the Noise-Aware Consensus strategy does not impose a prohibitive scaling cost on the researcher side.
2.  **Expand Multilingual Robustness:** Include robustness results (e.g., UED, WER) on non-English test sets (e.g., Common Voice non-English or WenetSpeech with noise) to confirm that the stability is language-agnostic and not just an artifact of English phonology.
3.  **Enhance Visual Clearness:** Ensure Figure 3 tables and other key comparison tables are rendered clearly in the production version of the paper to allow readers to easily parse the specific metric gains.
4.  **Ablate on Branch Correlation:** Discuss how the branches maintain independence. Since they share the same encoder hidden states before branching, does noise in the encoder propagate equally to all branches? An analysis of the correlation between branch outputs under noise (beyond the case study in Table 6) would strengthen the theoretical argument for voting efficacy.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Controlled Downstream Ablation:** Replace baseline quantizers with StableToken's Voting-LFQ *within the same encoder architecture* (e.g., all Whisper-based) to isolate the quantizer's contribution from encoder differences.
2. **LLM Training Convergence Curves:** Plot downstream LLM training loss vs. steps to verify the claim that stability reduces the "learning burden," rather than just reporting final task accuracy.
3. **Bit-Error Rate Statistics:** Provide statistical distribution of bit-flip errors across branches to validate the claim that token-level recovery occurs even when a majority of branches fail at the token level.
4. **Clean-Performance Oracle Comparison:** Compare StableToken against a single-path tokenizer trained *only* on clean data to quantify the exact performance trade-off incurred by enforcing noise robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Branch Diversity Metrics:** Calculate correlation or divergence between branch outputs to prove the voters are learning independent perspectives rather than collapsing into identical functions.
2. **UED vs. Semantic Integrity Correlation:** Analyze if lower UED correlates with preserved semantic content, ensuring stability isn't achieved by collapsing diverse inputs into identical trivial tokens.
3. **Gradient Conflict Analysis:** Examine the interaction between the Consensus Loss and ASR Loss to ensure the robustness objective isn't destabilizing the primary semantic learning objective.
4. **Failure Mode Characterization:** Identify specific noise types or SNR levels where the bit-wise voting mechanism fails to correct errors, defining the operational limits of the method.

### Visualizations & Case Studies
1. **Token Sequence Alignment Plot:** Visualize exact token ID sequences for clean vs. noisy inputs (Baseline vs. StableToken) to visually demonstrate the "drastic shifts" vs. stability claim.
2. **Branch Representation t-SNE:** Plot pre-quantization representations from different branches to visualize the consensus formation and separation of clean/noisy views.
3. **Noise Sensitivity Heatmap:** Show UED performance across a grid of noise types and intensities to expose specific vulnerabilities in the voting mechanism.

### Obvious Next Steps
1. **Adversarial Robustness Testing:** Evaluate performance against adversarial audio attacks, which are more severe than random noise and better test true representation stability.
2. **Streaming Latency Breakdown:** Provide a detailed latency profile for the voting mechanism in a real-time streaming scenario, as parallel branches may introduce synchronization overhead.
3. **Scaling to Larger LLM Backbones:** Verify if the robustness gains persist when integrated with larger models (7B+), as small LLMs may not fully expose the tokenizer's instability issues.

# Final Consolidated Review
## Summary

The paper identifies a critical vulnerability in modern supervised semantic speech tokenizers: they produce drastically different token sequences under minor acoustic perturbations, even when speech remains intelligible. The authors propose StableToken, which introduces a multi-branch Voting-LFQ quantizer with bit-wise majority voting and a Noise-Aware Consensus Training strategy. Experiments demonstrate significant improvements in token stability (UED reduced from 26.17% to 10.17%) and downstream SpeechLLM performance across ASR, SER, and TTS tasks under noisy conditions.

## Strengths

- **Problem identification with empirical validation.** Figure 1 and Table 1 convincingly demonstrate that existing tokenizers (GLM-4-Voice, CosyVoice2) produce highly unstable token sequences under perturbations, with UED as high as 38–54% for supervised tokenizers. The two-root-cause diagnosis (single-path architecture, distant supervision) is well-motivated and empirically validated by the ablation in Table 4, which shows each component contributes meaningfully (removing consensus loss increases UED from 10.96% to 17.43%).

- **Novel architectural design with theoretical grounding.** The Voting-LFQ mechanism extends LFQ to multi-branch consensus with bit-wise voting. Branching at the projection layer (not the encoder) keeps computation shared, while bit-wise voting enables error recovery even when token-level majority fails (Table 6 case study shows recovery from 3/5 branches predicting wrong tokens). The design achieves >60% relative UED reduction while maintaining competitive reconstruction quality (Table 2: 3.84%/7.99% WER on LibriSpeech clean/other).

- **Comprehensive downstream validation with controlled setup.** The paper evaluates across three tasks (ASR, SER, TTS) with consistent gains. CHiME-4 results show ~30% relative WER reduction (35.90% vs. 51.08% for GLM-4-Voice on test-real). The isogenic framework (same Qwen2.5-3B backbone, same fine-tuning recipe) isolates tokenizer contribution.

- **Strong out-of-distribution generalization.** Table 1 includes ESC-10 as OOD real-world noise excluded from training. StableToken achieves 10.96% UED vs. 24.47–36.53% for supervised baselines, demonstrating genuine robustness rather than overfitting to training noise profiles.

- **Negligible inference overhead.** Tables 12–14 (Appendix B.6) show FLOPs increase of only 0.01% and memory usage actually lower than GLM-4-Voice (13.5% reduction on GPU), validating the efficiency claims.

## Weaknesses

- **Missing matched-data baseline.** The strongest baselines (GLM-4-Voice, CosyVoice2, S³ Tokenizer) are publicly released models trained on undisclosed datasets, while StableToken is trained on ~183k hours of disclosed data (Table 7) plus in-house data. Although Table 4's ablation controls for architecture within StableToken's training pipeline, there is no comparison to a single-branch LFQ trained on the same corpus. This makes it impossible to fully disentangle architectural innovation from potential training-data or training-duration advantages.

- **Theoretical justification for inference-time error independence is incomplete.** At inference, all N branches receive the identical noisy encoder hidden state h_noisy. The paper's argument that bit-level errors across branches are "sparse and independently distributed" (enabling majority-vote recovery) assumes different linear projections of the same corrupted representation produce independently-erring binarizations. If noise shifts h_noisy in a direction aligned with most branches' sign boundaries, correlated failures could occur. Table 6 shows a single cherry-picked case; statistical analysis of cross-branch error correlation under diverse noise conditions would strengthen this foundational claim.

- **Encoder robustification is out of scope.** The method robustifies only the quantization stage; the Whisper-large-v3 encoder still processes noisy audio directly. If noise severely corrupts encoder output, all branches share the same damaged representation with no mechanism for recovery. The empirical results show effectiveness down to 0dB SNR, but the fundamental limit when encoder degradation overwhelms voting capacity is not characterized.

- **No dedicated limitations section.** The conclusion is brief and does not acknowledge failure modes or scope limitations. Key constraints not discussed: (1) the 30-second Whisper context window limits streaming/long-form applications; (2) proprietary in-house training data limits full reproducibility despite code release; (3) language generalization beyond English/Chinese is untested despite Whisper's multilingual capability.

## Nice-to-Haves

- **Training efficiency analysis.** Inference overhead is well-characterized, but training cost is not analyzed. Noise-Aware Consensus Training requires generating perturbations and processing N=5 branches per forward pass. Training FLOPs/time comparisons would help practitioners assess scalability.

- **Multilingual robustness evaluation.** The tokenizer is trained on English/Chinese but robustness evaluations focus on English benchmarks (LibriSpeech, CHiME-4, SEED-TTS). Evaluation on diverse languages would strengthen generalization claims.

- **Branch diversity analysis.** Analysis of correlation/divergence between branch outputs would validate that voters learn meaningfully independent perspectives rather than collapsing to similar functions under noise.

- **Statistical significance metrics.** No confidence intervals or error bars are reported. While single-run evaluation is common for large-scale speech benchmarks, statistical rigor would strengthen claims where differences are modest (e.g., some MOS comparisons).

## Removed Points

"These points are flagged to be removed, treat them with caution"

- The criticism that the paper doesn't formally explain "why ASR loss fails to provide stability" is an interesting theoretical question but not a flaw in an empirical paper that demonstrates the phenomenon and provides a solution. The paper correctly identifies that ASR loss supervises only the final transcription, not intermediate token stability.

- Visual presentation issues noted by one reviewer are parser artifacts in the provided text, not issues with the actual paper.

- Demands for confidence intervals on large-scale speech benchmarks represent scope creep; single-run reporting is the norm in this field.

## Novel Insights

The consensus-driven architecture reveals an elegant decomposition of the robustness problem: rather than building noise-invariance into a single representation (which requires adversarial training or massive augmentation), StableToken explicitly constructs multiple "views" and aggregates them democratically. The bit-wise voting mechanism is particularly clever—it provides finer-grained error correction than token-level voting while remaining deterministic at inference. The case study in Table 6 demonstrates that recovery is possible even when a majority of branches fail at the token level, provided bit-level errors remain sparse. This suggests a fundamental principle: robust discrete representations can be achieved by combining redundant, diversely-parameterized quantizers, analogous to ensemble methods in continuous domains. The approach sidesteps the gradient instability issues of token-level consistency losses by operating on continuous pre-quantization vectors during training.

## Suggestions

- **Add a matched-data ablation.** Train a single-branch LFQ on the same 150k+ hour corpus (with and without noise augmentation) to isolate architectural contribution from training-data effects. This single experiment would significantly strengthen causal claims.

- **Add cross-branch error correlation analysis.** Quantify the distribution of bit-flip errors across branches under various noise types and intensities. This would empirically validate the independence assumption underlying the voting mechanism.

- **Add a limitations section.** Discuss the 30-second context window constraint, the scope of robustification (quantizer only, not encoder), language coverage limitations, and computational overhead during training.

- **Report statistical significance where feasible.** Include standard deviations or confidence intervals for key metrics, particularly for downstream task comparisons where gains are smaller.

# Actual Human Scores
Individual reviewer scores: [10.0, 6.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
