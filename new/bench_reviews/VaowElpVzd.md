## Summary
This paper introduces **GES-Inter**, the first large-scale concurrent co-speech gesture dataset with mesh-based whole-body representations (~70 hours, 7M+ frames), and **Co³Gesture**, a bilateral cooperative diffusion framework for generating coherent concurrent co-speech gestures for two interacting speakers. The key architectural contributions are a Temporal Interaction Module (TIM) that uses mixed audio to model inter-speaker temporal dependencies, and a mutual attention mechanism for holistic cross-branch interaction. Experiments on GES-Inter show consistent improvements over adapted baselines on FGD, BC, and Diversity metrics, with ablations supporting each design choice.

---

## Claims and Support

| Claim | Supported? | Notes |
|---|---|---|
| GES-Inter is a large-scale dataset with concurrent gestures, mesh annotations, and multi-modal labels | **Yes** | Table 1 clearly differentiates it from existing datasets; scale is backed by reported frame counts, hours, and clip counts |
| GES-Inter is "high-quality" | **Partially** | Pipeline is well-described but no quantitative validation of pose accuracy, diarization error rates, or synchronization error is provided |
| Co³Gesture generates coherent concurrent gestures | **Partially** | FGD/BC/Diversity improve over baselines; user study shows "interaction coherency" advantage but study is very small (15 volunteers, 2 videos/method) |
| TIM effectively models temporal interaction | **Partially** | Ablation confirms it helps (FGD 1.297→0.769, MLP replacement 1.202→0.769); mechanistic interpretation is asserted, not analyzed via M or σ |
| Mutual attention boosts interaction dependencies | **Partially** | Ablation (w/o: 0.924, full: 0.769) supports utility; claimed speaker-order invariance/shared-weight rationale is asserted without empirical support |
| Bilateral branches handle asymmetric conversational dynamics | **Supported** | Strong ablation evidence: FGD 1.669 (single branch) vs 0.769 (bilateral); explanatory claim about *asymmetry* being the cause is not directly isolated |
| Co³Gesture outperforms SOTA | **Conditionally supported** | Numerically true on GES-Inter for all metrics; however, all baselines are adapted outside their native setting (single-person or text-conditioned), so the comparison is informative but not a clean SOTA claim |

---

## Strengths

- **Novel and practically motivated task**: The paper correctly identifies that all prior co-speech gesture work targets single-speaker synthesis, while real conversations are concurrent and interactive. This is a genuine gap with clear applications in avatars, embodied AI, and HCI.
- **Substantial dataset contribution**: GES-Inter fills a real void—Table 1 shows it is the first large-scale dataset combining concurrent gestures, SMPL-X mesh representations, and multi-modal annotations (facial, phoneme, text). At 70 hours and 7M+ frames, it dwarfs the only prior concurrent dataset (TWH16.2 at 17 hours, joint-based only). This alone will enable future research.
- **Well-motivated bilateral design**: The observation that concurrent speakers exhibit asymmetric dynamics (one active, one reactive) is empirically grounded and well-argued in Sec. 3.3. The ablation in Table 4 (FGD 1.669 → 0.769) provides strong numerical support.
- **Comprehensive ablations**: Tables 3–5 systematically ablate TIM, mutual attention, bilateral branches, mixed audio, audio separation, and foot contact loss, with each component showing clear contributions. The TIM vs. MLP comparison (1.202 vs. 0.769 FGD) is particularly informative.
- **Large empirical improvements**: The 24%+ FGD improvement over the best baseline (InterGen) and best-on-all-metrics performance in Table 2 are consistent and not marginal.
- **Audio design is clean and effective**: The combined use of separated audio for per-speaker guidance and mixed audio for interaction cues is an elegant and validated design (Table 4 ablations for both components).

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Evaluation framework is misaligned with the paper's central claim.** The paper's thesis is *coherent concurrent interaction*, yet FGD, BC, and Diversity are single-speaker realism, rhythm alignment, and diversity metrics—none directly quantifies whether the two generated speakers interact coherently with each other. The only direct interaction evidence is the user study, which is too small to carry this claim (see below). The paper itself acknowledges this in the Limitation section ("we will put more effort into designing specific interaction metrics"), but this is the central claim, not a secondary one. Some interaction-specific metric—even a simple cross-speaker motion correlation, gesture activity alternation pattern, or response alignment score—is needed to validate the core contribution.

- **Dataset quality claims are asserted, not demonstrated.** The entire paper (training and evaluation) depends on GES-Inter being reliable. The pipeline uses pseudo-label 3D pose estimation (PyMAF-X), automated speaker diarization (pyannote-audio), ASR (WhisperX), and forced alignment (MFA)—each stage introduces potential errors that compound. The paper mentions "extensive data processing to filter unnatural and jittery poses" (Sec. 3.1) and manual double-checking of audio-identity alignment (Sec. 3.1), but provides no quantitative quality statistics: no pose estimation accuracy on held-out frames, no diarization error rate, no synchronization error distribution, no annotator agreement statistics. The word "high-quality" appears in the abstract and contributions list without quantitative support. If systematic pipeline artifacts exist, they would simultaneously inflate training signal and evaluation scores in ways that are undetectable within the paper's own evaluation framework.

### Minor

- **User study is underpowered for the weight placed on it.** 15 volunteers evaluating 2 videos per method (16 videos total per participant, 6s each) is insufficient to claim statistically robust perceptual advantages, especially for interaction coherency—a subtle quality that requires careful exposure. No statistical significance tests (t-tests, Wilcoxon) or confidence intervals are reported for the user study scores. The study should be interpreted as suggestive, not conclusive.

- **Baseline comparison fairness is limited.** All seven baselines are adapted from either single-person gesture generation or text-to-motion generation—no baseline was natively designed for concurrent audio-conditioned two-speaker generation. The text2motion models (MDM, InterX, InterGen) received the paper's audio encoder as a replacement for their text encoder. While the paper acknowledges this is a new task (Sec. 4.2: "to the best of our knowledge, we are the first"), this context should be stated more prominently rather than presenting results as clean SOTA comparison. The margins reported in Table 2 demonstrate superiority over adapted baselines, not over native competitors.

- **Foot contact loss with T-pose lower body is methodologically opaque.** The paper explicitly states (Sec. 4.2): "Since we only model the upper body joints in experiments, we complete the lower body joints as T pose in forward kinematic function during calculate loss." This is a non-standard use of foot contact loss, and no justification is given for why a T-pose lower body provides meaningful regularization in a seated talk-show setting. The ablation (Table 5) shows it helps (FGD 1.082 → 0.769), but without explanation, it is unclear whether it is acting as intended or providing some other form of regularization.

- **No evaluation on the existing concurrent dataset TWH16.2.** The paper acknowledges TWH16.2 (Table 1) as the only other concurrent gesture dataset. While the paper argues GES-Inter is more suitable (mesh-based, larger, includes facial data), a single experiment on TWH16.2 would demonstrate cross-dataset generalization and that GES-Inter is not simply a specialized evaluation niche.

- **Short generation length limits practical applicability.** All clips are fixed at 90 frames / 15 FPS = 6 seconds. Real conversations are typically much longer. The paper does not discuss how the method would handle longer sequences or sliding-window inference, which is relevant to the practical motivation stated in the introduction.

### Trivial

- The claim of "speaker-order invariance" justifying shared weights in mutual attention (Sec. 3.3: "exchanging the input order of the speaker's audio results in an invariance effect") is asserted without evidence. While the practical benefit is shown in ablations, the theoretical justification is unverified.
- FGD and BC are reported as point estimates without confidence intervals; only Diversity reports them. For completeness, confidence intervals for FGD and BC would be informative.

---

## Nice-to-Haves

- **Add an independent two-speaker baseline**: Run two single-speaker gesture models independently (one per speaker, no interaction) to directly demonstrate that explicit interaction modeling is necessary rather than simply running two parallel models.
- **Analyze the learned σ in TIM**: Visualizing σ over time for interaction-heavy vs. quiet segments (e.g., overlapping speech vs. monologue) would reveal whether TIM learns meaningful turn-taking patterns or acts as a generic blending weight.
- **Interaction-specific metric design**: Cross-speaker motion energy correlation, gesture activity alternation score, or response latency metrics would strengthen the interaction coherence claim considerably.
- **Audio separation quality analysis**: Even a spot-check of diarization error rate (DER) on a sample of videos would significantly strengthen dataset quality claims.
- **Facial expression integration**: GES-Inter includes FLAME parameters; even a preliminary discussion or prototype of joint body+face generation would better leverage the dataset's unique annotations.
- **Failure case analysis**: Discussion of when the method degrades (heavy overlap, rapid turn-taking, separation failure) would provide a more complete picture.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"TIM mechanism is not established"** (Harsh Critic, Critical Issue): The ablation in Table 3 (w/o TIM: FGD 1.297, w/ MLP: FGD 1.202, full: FGD 0.769) provides solid evidence that TIM is not just adding capacity—the MLP replacement is a capacity-matched comparison that performs substantially worse. The mechanistic interpretation is imprecise, but this does not invalidate the contribution. WEAKENED to a trivial note rather than kept as a major issue.

- **"Bilateral branch asymmetry argument is unvalidated"** (Harsh Critic): The ablation (Table 4: FGD 1.669 single branch vs. 0.769 bilateral) provides strong empirical support. While the specific asymmetry mechanism is not isolated, the contribution is demonstrated. The explanatory narrative is a reasonable interpretation, not a false claim.

- **"SOTA comparison is too weak to support the paper's claims"** (Harsh Critic, Spark): The paper explicitly states this is the first work on the task, so no native baselines exist. Adapting existing methods is the correct scientific approach. The comparison is informative and the margins are large enough to be convincing. The "SOTA" framing is mildly loose but not a material error.

- **Reproducibility concerns about hyperparameters, training details**: The paper discloses optimizer, learning rate, batch size, GPU used, training duration, and all architectural hyperparameters (Sec. 4.1). This is adequate for the field. REMOVED per hard rules.

---

## Novel Insights

The most genuinely novel conceptual contribution is the framing of concurrent gesture generation as an *asymmetric* rather than symmetric interaction problem—unlike choreography or action generation where exchanging performers is semantically neutral, conversational gesture dynamics are inherently asymmetric (one speaker drives, the other reacts), and bilateral rather than symmetric-joint generation is the principled response. The TIM design—using mixed audio as an interaction signal to compute soft blending weights between per-speaker and joint motion embeddings—is a clean and generalizable idea for injecting interaction context into any diffusion-based motion model. The combined use of separated audio (speaker fidelity) and mixed audio (interaction context) as distinct conditioning signals is an elegant decomposition that could transfer to other multi-agent generation settings.

---

## Assessment

**Originality**: Good. The task formulation is novel, and the bilateral + TIM design is non-obvious. The architecture is not radically new but is principled and task-appropriate.

**Importance**: High. Concurrent co-speech gesture generation is underexplored and practically relevant. The GES-Inter dataset alone is a meaningful community contribution.

**Claim Support**: Moderate. Quantitative results and ablations are convincing, but the central claim of "interaction coherence" is inadequately measured and the dataset quality is unvalidated quantitatively.

**Experimental Soundness**: Moderate. Ablations are comprehensive, but lack of interaction-specific metrics and underpowered user study are real gaps.

**Clarity**: Good. The paper is well-organized, the method is clearly described, and the limitations are honestly acknowledged.

**Value**: Moderate-high. The dataset and task formulation will be useful to the community regardless of method quality.

---

## Score and Decision

The paper introduces a genuinely new task with a large-scale dataset that fills a real gap, a reasonably principled method, and strong ablation evidence. The limitations—lacking interaction-specific metrics and quantitative dataset validation—are real but partially acknowledged by the authors. The paper does not have fatal methodological errors; the weaknesses are addressable in revision. However, the mismatch between the central claim ("coherent interaction") and the evaluation framework is a substantive gap that reviewers should weigh carefully.

**Score: 5.5**
**Decision: Borderline Accept (Weak Accept)**

The dataset contribution and task formulation are valuable enough to warrant publication with revision, particularly addressing the interaction evaluation gap and dataset quality reporting.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>