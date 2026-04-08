=== CALIBRATION EXAMPLE 49 ===

# Final Consolidated Review
## Summary
AnyAvatar proposes a multimodal diffusion transformer (MM-DiT) based framework for audio-driven human animation that introduces three modules: a Character Image Injection Module (CIIM) to balance dynamics and consistency, an Audio Emotion Module (AEM) for emotion-aligned expression generation, and a Face-Aware Audio Adapter (FAA) for multi-character audio-driven animation. Built on the HunyuanVideo backbone, the system demonstrates improvements in visual quality and identity preservation on portrait and full-body benchmarks.

## Strengths
- **CIIM resolves a real dynamics-consistency trade-off**: The paper identifies that conventional padding-frame approaches sacrifice dynamics for consistency, and proposes an injection mechanism (repeat image T times → tokenizer → add to video latent) that is ablation-tested against two alternatives (Table 3), showing meaningful gains in Motion Diversity (3.585 → 4.127) while maintaining strong Video Quality (4.486). This is a concrete, well-motivated architectural contribution.
- **Face-Aware Audio Adapter enables multi-character control**: The FAA uses latent-space face masking combined with spatial cross-attention to localize audio influence per character. This is a practically useful mechanism that addresses a genuine gap in the field, where most methods handle only single-character animation. The qualitative results (Figure 5, 7b) demonstrate the concept works.
- **Comprehensive evaluation scope**: The paper evaluates on both portrait (CelebV-HQ, HDTF) and full-body (self-constructed wild) datasets with multiple metrics (IQA, AES, FID, FVD, Sync-C/D, HKC, HKV, user study), and provides ablations on all three proposed modules.

## Weaknesses

### Major:
- **Overclaiming of multi-character dialogue capability**: The abstract and introduction promise "multi-character dialogue videos" and "independent audio injection for multi-character scenarios." However, Appendix A.5 explicitly states: "they can only speak the same lines according to the audio. However, it is currently not possible to support scenarios where different characters speak different lines simultaneously or where interruptions occur during speech." This means the system cannot handle the most basic form of dialogue—two people speaking different things. The framing of "dialogue" strongly implies turn-taking with independent speech, which the method does not support. This limitation should be prominently disclosed in the main text, not buried in the appendix.
- **AEM requires emotion reference images, contradicting the "audio-driven" framing**: The title and abstract describe the system as "audio-driven," but the AEM requires an explicit emotion reference image ($I_{ref}$) at inference time to control expressions. This means the system is not purely audio-driven—it is audio-and-image-driven for emotion-controllable generation. The paper acknowledges this in Appendix A.8 ("increased complexity for users... inability to reflect dynamic emotional changes"), but this is a core input requirement that should be stated upfront in the abstract and method description. A single reference image also cannot represent dynamic emotional changes within a video, which undermines the claim of achieving "fine-grained and accurate emotion style control."
- **Misleading interpretation of synchronization results**: In Table 1, AnyAvatar scores 4.92 / 5.30 on Sync-C (CelebV-HQ / HDTF) while Sonic scores 5.58 / 5.81. Since Sync-C ↑ indicates better synchronization, AnyAvatar underperforms Sonic on this key metric by a meaningful margin. Yet Section 4.2 states the results "demonstrate that our method achieves the best performance... showcasing its capability in audio synchronization." This selectively omits the one metric where performance is not best. The paper should honestly acknowledge that lip-sync precision does not surpass all baselines, even as overall visual quality improves.

### Minor:
- **FAA masking may suppress co-speech body gestures**: By multiplying the audio cross-attention output by the face mask $g_M$ (Eq. 2), audio influence is strictly confined to the face region. This means natural co-speech behaviors—hand gestures, head tilts, body sway—that correlate with audio cannot be driven by the audio signal. The paper claims "dynamic" animation, but the FAA mechanism inherently produces "talking heads on static bodies" in multi-character scenarios. This tension between the FAA's design goal (isolation) and the paper's broader claim (full-body dynamics) is not discussed.
- **CIIM trade-off not fully acknowledged**: Table 3 shows "Token + Add" (the proposed method) drops Identity Preservation from 4.576 (Token + Channel) to 4.289—a measurable decrease. The text claims it "maintains" consistency, but the data shows a real trade-off. This should be discussed more transparently.
- **Self-constructed full-body test set is not publicly available**: The 250-video wild full-body test set is introduced for the first time in this paper but has no public access or detailed construction methodology, making the Table 2 results difficult to independently verify.
- **Missing full factorial ablation**: The paper ablates each module individually (CIIM in Table 3, AEM in Figure 7a, FAA in Figure 7b, mask injection in Table 4) but never studies how the three modules interact when combined. It is possible that some modules are redundant or that their interactions produce unexpected effects.

### Trivial:
- The long video generation strategy (Algorithm 1) is directly adapted from Sonic (Ji et al., 2024), as acknowledged in the appendix. The novelty of this component is minimal.

## Nice-to-Haves
- Experiments on direct audio-to-emotion mapping (without reference images) to show whether the AEM's image requirement is truly necessary or could be replaced
- Inference time comparisons with baselines, not just the absolute 60-minute figure
- Failure case analysis showing where emotion alignment breaks down or where multi-character audio bleeds across masks
- Statistical significance tests for user study results (though 30 participants is standard for this subfield)
- Visualization of the latent-space face masks to verify FAA isolation works as described

## Removed Points
These points are flagged to be removed, treat them with caution:
- **160-GPU training cost as reproducibility concern**: The paper discloses training resources (160 GPUs with 96GB each). Per the rules, reproducibility concerns about large-scale training infrastructure are not paper flaws—this is standard for large video generation models.
- **Missing related works**: Per rules, we cannot confirm whether specific related works exist and should not flag their absence.
- **Figure 3 ambiguity**: The reviewer claimed inconsistency between Figure 3 and the text regarding Token vs. Add. The appendix (Eq. 5) clarifies the full mechanism as $TokenCat(\{K_1(t_r) + K_2(t_{noise})\}, t_R)$, which is both addition and token concatenation. Table 3 labels the method "Token + Add," confirming this. The main text could be clearer, but this is not a factual error.
- **Formatting artifacts**: Per rules, these are parser issues and not paper problems.
- **Demand for experiments on smaller backbones**: This is scope creep—the paper's contribution is about modules for a specific backbone, not about backbone agnosticity.
- **Demand for generalization to unseen domains**: Generic weakness not specific to this paper's claims.

## Novel Insights
The FAA's design reveals an inherent tension in multi-character audio-driven animation: achieving precise per-character audio isolation (via face masking) comes at the cost of suppressing holistic audio-driven body motion. This suggests that future work may need separate mechanisms for face-localized lip-sync and global body dynamics, rather than a single cross-attention pathway. Additionally, the finding that injecting emotion via reference images (AEM) in the Double Block rather than the Single Block is necessary for effective emotion learning is an actionable architectural insight—indicating that emotion-to-expression mapping requires the richer joint attention of Double Blocks, which could inform future module insertion strategies for similar conditional generation tasks.

## Suggestions
- **Prominently disclose multi-character limitations**: Move the Appendix A.5 admission that characters cannot speak different lines simultaneously into Section 3.2 and the abstract. Reframe "multi-character dialogue" as "multi-character audio-driven animation" or clearly state the current constraint.
- **Reframe "audio-driven" to account for AEM input**: Either describe the system as "audio-and-reference-image-driven" or add a brief experiment showing emotion control directly from audio features without a reference image to clarify what is lost.
- **Honest metric reporting**: Acknowledge that Sync-C underperforms Sonic in Table 1, and discuss why overall quality improvements compensate or do not compensate for this gap.
- **Add a full factorial ablation**: Show CIIM-only, CIIM+AEM, CIIM+FAA, and all-three to demonstrate that all modules contribute additively.
- **Discuss the co-speech gesture limitation of FAA**: At minimum, acknowledge that face-localized masking sacrifices body motion correlation with audio, and propose this as a direction for future work.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
