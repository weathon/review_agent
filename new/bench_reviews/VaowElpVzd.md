Now I have enough context to write a thorough meta-review. Let me synthesize all the information.

Key points to verify against the paper:

1. **Evaluation metrics don't measure interaction** - TRUE. The paper uses FGD, BC, and Diversity, which are all single-speaker metrics. The paper acknowledges in limitations "we will put more effort into designing specific interaction metrics for better concurrent gesture evaluation."

2. **Baseline configuration unfair** - The paper does specify adaptation details for some baselines but not all. The text says: "For other methods, we modify their final output layer to match the dimensions of our experimental settings." And "Since the above text2motion counterparts are designed without the audio incorporation setting, we adopt the same audio encoder as ours in the models." This is somewhat specified but the exact integration is unclear. However, the harsh reviewer's claim that this "likely unfairly penalizes" baselines needs careful consideration - the paper actually adapts baselines in ways that could be seen as giving them a fair chance (using the same audio encoder). The InterGen comparison is relevant since it's a two-person interaction model.

3. **Dataset quality not validated** - TRUE. No quantitative validation of pose quality.

4. **TIM design loosely justified** - TRUE. The mathematical details of TIM (temporal correlation matrix M, motion encoder) are under-specified. The σ analysis point is valid.

5. **User study too small** - TRUE. 15 participants, 2 clips per method, vague criteria.

Let me also check: the harsh reviewer claims the comparison unfairly penalizes co-speech methods because they don't specify how they're adapted. But actually the paper does say "all the competitors are implemented by official source codes or pre-trained models released by authors" and specific audio encoder details are given for TalkSHOW and DiffSHEG. This is at least partially addressed. The claim that baselines are "likely suboptimal" is speculative - using the same audio encoder could be seen as giving them a fair comparison.

For the "unfair comparison" rule: "REMOVE 'weaknesses' about unfair comparison with other methods if the asymmetry favors the baseline and not the author's method." The potential unfairness here is unclear - it could go either way. The author's method is specifically designed for two-person concurrent gesture generation; adapting single-person methods requires some modification. This is inherent to the new problem setting, not really an unfair comparison. The paper makes reasonable adaptations. I'll keep a weakened version of this concern.

For the calibration:
- CoCoGesture (similar domain, dataset contribution): scores 6, 3, 3 → withdrawn/rejected
- InterDance (similar domain, two-person interaction, dataset+method): scores 5, 5, 6, 6, 6 → rejected
- MMG-VL (multi-person motion generation, interaction): scores 6, 3, 3, 5, 3 → withdrawn/rejected
- TANGO (co-speech gesture, oral): 8, 8, 8, 10 → strong accept
- CyberHost (audio-driven body generation, oral): 8, 10, 6, 6, 8 → strong accept
- Think-Then-React (action-to-reaction, poster): 8, 6, 6, 6 → accept poster
- PianoMotion10M (dataset + baseline, spotlight): 8, 6, 6, 6 → accept spotlight

The Co³Gesture paper has a novel task formulation and substantial dataset contribution. Its main weaknesses are the lack of interaction-specific metrics and some dataset quality concerns. Compared to InterDance (which was rejected with scores 5-6), this paper has a more comprehensive evaluation and a larger dataset. The interaction metric gap is acknowledged as future work.

The paper is making a real contribution with the new task and dataset. The method is reasonable. The evaluation gap (no interaction metrics) is significant but not fatal. I'd position this paper around 5-6 range, similar to InterDance but somewhat better due to more comprehensive evaluations and a more clearly novel task definition.</think>Now I have sufficient context from calibration papers and verified claims. Let me write the consolidated review.

## Summary
The paper introduces the novel task of concurrent co-speech 3D gesture generation for two interacting speakers, contributing both a large-scale dataset (GES-Inter, 7M+ frames, 70 hours) with SMPL-X body meshes and separated speaker audio, and a method (Co³Gesture) built on bilateral cooperative diffusion branches with a Temporal Interaction Module (TIM) and mutual attention mechanism to model asymmetric speaker dynamics.

## Strengths

- **Novel task formulation with practical significance.** Two-speaker concurrent gesture generation from conversational audio is genuinely under-explored and practically important. The paper makes a compelling case that existing methods handle only single-speaker gestures, while real conversations inherently involve mutual interaction dynamics (Sec. 1, Fig. 1).

- **Large-scale dataset contribution.** GES-Inter fills a clear gap: it is the first large-scale co-speech dataset providing concurrent two-speaker upper-body SMPL-X/FLAME annotations with separated audio, phoneme, and text alignments (Table 1). At 70 hours / 7M+ frames, it substantially exceeds the only comparable concurrent-gesture dataset (TWH16.2, 17 hours, joint-based only, no facial data). This is likely to catalyze future research.

- **Well-motivated architectural design.** The bilateral branch structure is justified by the observation that two speakers exhibit asymmetric body dynamics. The ablation in Table 4 confirms this (FGD degrades from 0.769 to 1.669 without bilateral branches), and the audio separation/mixing ablation (Table 4) confirms that both separated and mixed audio contribute meaningfully.

- **Comprehensive ablation studies.** Tables 3–5 provide systematic validation of TIM, mutual attention, bilateral branches, audio separation, mixed audio, and foot contact loss, each showing consistent improvements.

- **Strong quantitative results.** Co³Gesture achieves a 24% FGD improvement over the next-best competitor (InterGen) and leads on all three metrics (Table 2). The user study (Fig. 5) also shows favorable perceptual ratings.

## Weaknesses

### Fatal
None.

### Major

- **No metric directly evaluates cross-speaker interaction quality — the paper's central claim.** The paper's headline contribution is "coherent concurrent co-speech gesture synthesis including two-person interactive movements." Yet all three quantitative metrics (FGD, BC, Diversity) are adapted from single-speaker evaluation. FGD measures distributional distance of unary motion features; BC measures per-speaker beat alignment; Diversity measures average pairwise distance of generated samples. None quantify whether the two generated streams are *interactively coherent* (e.g., whether one speaker gestures while the other listens, whether reaction timing aligns with the partner's emphasis, or whether there are complementary/mirroring gestures). The authors themselves concede this in their limitations: "we will put more effort into designing specific interaction metrics for better concurrent gesture evaluation." This creates a disconnect between the claimed contribution and its empirical validation: the quantitative evidence cannot distinguish "better interaction" from "better single-speaker gesture quality." While the ablations show improvement when TIM and mutual attention are added, the same metrics cannot confirm these components specifically improve *interaction* rather than general motion quality.

- **Pseudo-labeled dataset quality is under-validated despite being a core contribution.** GES-Inter is built using Pymaf-X on in-the-wild talk show video. The paper mentions "extensive data processing to filter the unnatural and jittery poses" (relegated to supplementary) but provides no quantitative characterization of pose quality, filtering thresholds, error rates, or comparison against ground truth. Audio-visual alignment relies on WhisperX + MFA + manual verification, but again no alignment accuracy statistics are reported (Sec. 3.1). For a dataset that is itself a claimed contribution, this absence makes it difficult to assess the reliability of training data and, consequently, the trustworthiness of results trained thereon.

### Minor

- **TIM mathematical details are underspecified.** The temporal correlation matrix $M \in \mathbb{R}^{N \times N}$ between updated current gesture embedding and interactive embedding is not defined (Equation 2). The architecture and output shape of Enc(M) producing σ are not described. Whether σ is per-time-step or a global scalar matters for interpreting whether TIM learns dynamic interaction patterns or a near-constant weight. Without analysis or visualization of learned σ, it is unclear whether TIM functions as claimed or collapses to a fixed interpolation.

- **Baseline adaptation for text-to-motion methods (MDM, InterX, InterGen) is under-specified.** Section 4.2 states "we adopt the same audio encoder as ours in the models" and "modify their final output layer," but does not describe how audio features are integrated (replacing text conditioning? concatenated? at which layers?). Since these methods were originally text-conditioned, the adaptation could be suboptimal. This is partially mitigated by the fact that InterGen itself models two-person interaction (making it a directly relevant comparison), but the gap in adaptation detail reduces confidence in the full Table 2 comparison.

- **Short generation horizon.** Experiments use N=90 frames at 15 FPS (6 seconds). Conversational interactions often span longer turns with multiple turn-taking events. The paper does not address how the framework handles or scales to longer sequences.

- **Only upper body generated; facial expressions excluded.** The paper acknowledges this limitation, but facial expressions are particularly crucial for conversational interaction (e.g., listener backchanneling). The exclusion limits the practical usability for the claimed task.

- **User study methodology is limited.** 15 participants evaluating 2 clips per method on a 0–5 scale, with no reported statistical significance tests, paired comparison protocol, or operationalization of "interaction coherency." This provides only suggestive evidence, especially for the interaction quality claim.

### Trivial
- The foot contact loss is computed by completing lower body as T-pose in forward kinematics (Sec. 4.2). While conceptually unusual, it acts as a reasonable regularizer and the ablation in Table 5 confirms its utility.

## Nice-to-Haves

- **Interaction-specific metric design.** Metrics such as cross-speaker motion correlation, reaction latency distributions, complementary gesture frequency, or mutual information between gesture streams would directly validate the core claim.
- **Visualization/analysis of learned σ** to confirm TIM learns dynamic interaction patterns rather than a constant weight.
- **Per-speaker metric breakdown** (active speaker vs. listening speaker) to validate the asymmetric modeling claim.
- **Evaluation on TWH16.2** or another external concurrent-gesture dataset to demonstrate generality.
- **Analysis of dataset statistics** (speech overlap frequency, speaker activity ratios, gesture synchronization patterns) to help readers understand the data.

## Removed Points

- **Harsh reviewer claim that baseline comparison "likely unfairly penalizes" co-speech methods and is "in serious doubt."** The paper specifies adaptation details for TalkSHOW and DiffSHEG (using their respective audio encoders), states all others use the same audio encoder, and notes output layer modifications. While more detail would be helpful, this is a standard comparison setup for a new task where no direct baseline exists. The asymmetry (Co³Gesture is purpose-built for this task) is inherent to introducing a new problem setting, not an unfair experimental choice. The concern about specification is kept as a minor point.

- **Harsh reviewer claim about dataset construction "implausible" manual verification scale.** The paper states professional inspectors "manually annotate" and "double-check" the separated audio signals. This is a standard practice for dataset construction; questioning its feasibility without evidence is speculative.

- **Harsh reviewer claim that bilateral branch justification ("asymmetric dynamics") is "not empirically demonstrated."** Table 4 shows FGD degrades from 0.769 to 1.669 when bilateral branches are removed, which is an empirical demonstration that single-branch (symmetric) generation is inferior for this task. Whether this is specifically due to "asymmetry" versus capacity is debatable, but the ablation does support the design.

- **Human finder's concern about "limited methodological novelty" of TIM and mutual attention being "built on established techniques."** The combination and task-specific adaptation of these techniques is appropriate novelty for a systems/contribution paper introducing a new task. Individual components being standard does not diminish the overall contribution.

- **Neutral reviewer's suggestion about "comparison fairness with text-to-motion methods."** Already covered above (kept as minor point). The broader concern about fairness is weakened because InterGen is a directly relevant two-person interaction baseline.

## Novel Insights
The most insightful observation across reviews is the fundamental tension between the paper's central claim (coherent, interactive concurrent gesture generation) and the complete absence of any metric that actually measures cross-speaker interaction. This is not merely a missing ablation — it is a structural gap where the claimed contribution cannot be empirically verified. The authors' own acknowledgment in the limitations section ("we will put more effort into designing specific interaction metrics") implicitly concedes this point. Additionally, the pseudo-labeled dataset being a core contribution without quantitative validation of pose/alignment quality leaves a significant gap in establishing the foundation upon which all results depend.

## Suggestions
- Design at least one interaction-specific metric (even a simple cross-speaker motion correlation or reaction delay metric) and report it alongside FGD/BC/Diversity.
- Report quantitative pose quality statistics (e.g., percentage of frames filtered, jitter metrics before/after smoothing) and audio-alignment accuracy to substantiate the dataset contribution.
- Visualize the learned σ values from TIM on representative examples to show whether interaction weighting is dynamic or nearly constant.

## Score and Decision

**Calibration comparison:**
- **InterDance** (similar two-person interaction + dataset, rejected with scores 5/5/6/6/6): Co³Gesture is broadly comparable but has a clearer novel task definition and more comprehensive ablations. InterDance was criticized for weak qualitative results and limited novelty; Co³Gesture's methodological contribution is incremental similarly (standard components combined purposefully), but the task is more novel.
- **CoCoGesture** (similar pseudo-labeled gesture dataset, withdrawn with scores 6/3/3): This paper was dinged heavily for over-smoothed, speech-uncorrelated results. Co³Gesture avoids this perception somewhat but shares the pseudo-label quality concern.
- **Think-Then-React** (action-to-reaction, accepted poster with scores 8/6/6/6): That paper had a novel task, LLM-based method, and good quantitative results but weak qualitative evaluation. Co³Gesture has a less technically novel method but comparable task novelty.
- **PianoMotion10M** (dataset + baseline, spotlight with scores 8/6/6/6): That paper is stronger because it provides thorough dataset statistics and purpose-built evaluation metrics. Co³Gesture is weaker on both counts.

The paper makes a genuine contribution in task definition and dataset, but the central interaction claim is empirically unverifiable with the provided metrics, the dataset quality is under-validated, and the methodological novelty is incremental (bilateral diffusion + cross-attention temporal fusion). These are significant but not fatal issues. The paper would benefit substantially from interaction metrics and dataset validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>