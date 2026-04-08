=== CALIBRATION EXAMPLE 23 ===

# Final Consolidated Review
## Summary

AnyAvatar proposes an MM-DiT-based framework for audio-driven human animation that introduces three modules: a Character Image Injection Module (CIIM) that adds reference image features along the channel dimension to balance dynamics and consistency, an Audio Emotion Module (AEM) that transfers emotional cues from a reference image via cross-attention, and a Face-Aware Audio Adapter (FAA) that uses latent-space face masks to isolate audio-driven animation per character for multi-character scenarios. The method builds on HunyuanVideo and demonstrates improvements on portrait benchmarks (CelebV-HQ, HDTF) and a newly constructed full-body test set.

## Strengths

- **Effective resolution of the dynamics-consistency trade-off via CIIM.** The three-way ablation (Table 3) comparing token concatenation, token+channel concatenation, and the proposed token+addition injection provides concrete evidence for the design choice. The spatial shift in RoPE positioning (Eq. 6) to prevent copy-paste behavior is a thoughtful, non-obvious detail that shows careful engineering.

- **Practical multi-character audio isolation via FAA.** The latent-space face masking with spatial cross-attention per time step is a clean mechanism that enables sequential multi-character animation in a single forward pass, a capability largely absent from prior work. The ablation in Table 4 demonstrates that mask introduction does not degrade background preservation (SB: 100%, DB: 90%→improved from 87%).

- **Strong portrait animation results.** On established benchmarks (Table 1), AnyAvatar leads on 5 of 6 metrics on CelebV-HQ and 4 of 6 on HDTF, with particularly notable gains in IQA, AES, and FID, validating the core generation quality.

## Weaknesses

- **Overclaimed "multi-character dialogue" capability.** The Abstract promises "multi-character dialogue generation," and the Introduction states the FAA enables "independent audio injection… for multi-character scenarios." However, Appendix A.5 explicitly states the system "is currently not possible to support scenarios where different characters speak different lines simultaneously." Characters can speak different lines *sequentially* (first half audio → character A, second half → character B), but true interactive dialogue with interruptions or overlapping speech is not supported. The term "dialogue" strongly implies back-and-forth interaction, which misrepresents the actual capability. This matters because it is one of the paper's three claimed contributions.

- **No quantitative evaluation of the Audio Emotion Module.** The AEM is presented as a core contribution enabling "fine-grained and accurate emotion style control," yet its evaluation is entirely qualitative (Figure 7a: a side-by-side comparison of text-guided vs. AEM-guided generation). No emotion classification accuracy, facial action unit agreement, or perceptual emotion scoring is provided. Without objective measurement, the claimed improvement in emotion alignment is empirically unsubstantiated.

- **No quantitative evaluation of multi-character scenarios.** The FAA is another core contribution, but Table 2 provides no multi-character-specific metrics—e.g., per-character audio binding accuracy, cross-character audio leakage, or subjective ratings for individual character quality when multiple characters appear simultaneously. The multi-character capability is demonstrated only qualitatively in Figure 5. Since multi-character animation is a distinguishing feature, the lack of dedicated evaluation is a significant gap.

- **Mixed full-body results undermine SOTA claims.** In Table 2, AnyAvatar underperforms on several key metrics: IQA (4.668 vs. WanS2V's 4.812), Sync-D (8.535 vs. 7.957), HKV (0.390 vs. 0.413), FBN (3.88 vs. 4.50), and FVD (650.541 vs. MultiTalk's 613.213). The Abstract's claim of "surpass state-of-the-art methods" and the paper's assertion of "best performance on most evaluation metrics" (technically 6/12) obscures significant losses on distributional realism (FVD) and naturalness. The FVD gap in particular—where AnyAvatar loses to three baselines—suggests the improved synchronization and hand quality may come at the cost of overall video realism, yet this trade-off is never discussed.

- **CIIM ablation relies entirely on uncharacterized subjective ratings.** Table 3 evaluates the three injection mechanisms using subjective dimensions (VQ, MD, IP, LS) with no reported inter-rater agreement, participant count, or statistical significance. The chosen mechanism (c) achieves the best MD (4.127) but sacrifices IP compared to mechanism (b) (4.289 vs. 4.576). This trade-off—lower identity preservation for higher motion diversity—is central to the paper's motivation, yet it is not analyzed or justified; the text simply states mechanism (c) "shows better results."

- **Emotion reference image requirement mischaracterized as "audio-driven."** The AEM requires an emotion reference image as input, making the system Audio + Identity Image + Emotion Image rather than purely audio-driven. The Abstract frames this as aligning "emotions expressed in the audio," but the emotional content actually comes from the reference image. The paper does not analyze sensitivity to reference image choice, failure modes with inappropriate references, or how this additional input burden affects usability. This matters because the framing creates a mismatch between the system's "audio-driven" branding and its actual input requirements.

- **Mask injection slightly degrades distributional quality.** In Table 4, the "w mask" condition has worse FID (74.087) than "w/o mask" (72.124), suggesting the masking mechanism introduces some quality degradation. While background-specific metrics (SB, DB) improve and the authors argue masks do not cause foreground/background distortion, the overall FID regression is not discussed despite being a standard generation quality metric.

## Nice-to-Haves

- Compare AEM against audio-only emotion extraction (e.g., emotion classification from Whisper features) to justify the reference-image design choice over a more streamlined alternative.
- Report inference time and compute costs alongside baselines for fair efficiency comparison, especially given the "real-world applications" framing.
- Add temporal consistency metrics (e.g., frame-to-frame identity drift, motion smoothness over the 50-second long-video claims) beyond qualitative stability observations.
- Release the full-body test set to enable independent verification of the Table 2 comparisons.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Self-constructed test set reproducibility concern** (Harsh Critic #1): The concern that the 250-video full-body test set cannot be independently verified is a reproducibility nitpick; the paper commits to releasing the data pipeline (Section A.2), and demanding its inclusion in the submission is impractical.
- **OmniHuman-1 unfair comparison due to super-resolution** (Harsh Critic #5): The paper itself explicitly acknowledges this issue (Section 4.2: "OmniHuman-1 is not open source and its online service includes super-resolution operations, there is a natural visual advantage in subjective evaluations"), which constitutes reasonable addressal even if imperfect.
- **Tokenizer2 weight copying lacks theoretical justification** (Harsh Critic, Methods): Weight initialization from pretrained tokenizers is standard practice in fine-tuning; the authors report it accelerates convergence, which is sufficient empirical justification.
- **Inference speed as a core weakness invalidating claims** (Harsh Critic, overall): The paper explicitly scopes this as a limitation (Section A.8) and discusses acceleration experiments (4× speedup with Jenga). For offline video production—the primary use case—60 minutes per 10s video, while slow, is not prohibitive. This is a nice-to-have improvement, not a fatal flaw.
- **Missing related works** (general): Per hard rules, not evaluated without external source verification.

## Novel Insights

The paper reveals an interesting asymmetry in how DiT blocks process different conditioning signals: the AEM only works effectively in Double Blocks (not Single Blocks), suggesting that the key-value attention pathway in Double Blocks is critical for mapping abstract emotional cues to concrete facial expressions. This observation, if validated with further analysis, could inform the design of all future emotion-conditioned DiT architectures by pointing to where in the architecture affective conditioning should be injected.

## Suggestions

- Add at least one quantitative metric for emotion alignment (e.g., classify the emotion of generated faces with a pretrained model and measure agreement with the intended emotion) to substantiate the AEM contribution.
- Evaluate multi-character scenarios quantitatively: measure per-character lip-sync accuracy and cross-character audio leakage when multiple characters are present simultaneously.
- Discuss the FVD gap explicitly—analyze whether the FAA masking or CIIM addition mechanism contributes to distributional quality degradation, and whether this is an inherent trade-off.
- Temper the "multi-character dialogue" language to "sequential multi-character animation" or "turn-based multi-character animation" to accurately reflect the system's capability.
- Report statistical details (number of raters, inter-rater agreement) for subjective evaluations in Tables 3 and 2's user study to strengthen the reliability of these results.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
