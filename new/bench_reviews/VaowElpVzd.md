Now let me do calibration searches to properly score this paper.Now I have enough information to write the final consolidated review.

---

## Summary

Co³Gesture introduces the novel task of concurrent two-speaker co-speech gesture generation. The paper makes two primary contributions: (1) GES-Inter, a new large-scale dataset with 7M+ frames / 70 hours of concurrent co-speech gestures annotated with SMPL-X, facial expressions, phonemes, and separated audio; and (2) a bilateral cooperative diffusion framework featuring a Temporal Interaction Module (TIM) and mutual attention mechanism, which outperforms adapted SOTA baselines on the GES-Inter benchmark. The task is genuinely new—no prior method addresses concurrent two-speaker audio-conditioned gesture synthesis—making both the dataset and the method real contributions to the field.

---

## Strengths

- **Novel and underexplored task framing**: The paper formalizes concurrent two-speaker co-speech gesture generation as a distinct research problem. No existing method or dataset addressed this setup, making the task introduction and GES-Inter dataset the most immediate community-level contribution.

- **GES-Inter dataset fills a genuine gap (Table 1)**: The dataset provides 70 hours / 7M+ frames of concurrent gestures with SMPL-X mesh, separated audio per speaker, phoneme, text, and facial expression annotations—uniquely combining concurrent gestures *and* mesh-based whole-body representation. TWH16.2, the only other concurrent gesture dataset, covers only 17 hours with joint-only annotations and no separated audio.

- **Bilateral branch design is empirically validated with large margins**: Table 4 shows removing bilateral branches degrades FGD from 0.769 to 1.669 (a 117% increase), strongly supporting the core architectural claim that asymmetric speaker dynamics require separate generation branches.

- **TIM contribution is well-evidenced (Table 3)**: Removing TIM worsens FGD from 0.769 to 1.297, and replacing it with a simple MLP worsens it further to 1.202, providing clear evidence that the learned temporal-interaction dependency weight σ is not merely a fusion artifact.

- **Comprehensive ablation suite (Tables 3–5)**: Each design choice—TIM, mutual attention, bilateral branches, mixed/separated audio, foot contact loss—is tested individually, giving a clear decomposed picture of contribution.

- **User study includes interaction coherency dimension (Figure 5)**: Co³Gesture leads on interaction coherency (~4.4) versus the next best InterGen (~4.1), providing at least qualitative evidence for the core claim.

---

## Weaknesses

### Fatal
None.

### Major

- **No quantitative interaction metric to validate the paper's headline claim**: The central stated contribution is *coherent concurrent* gesture generation—two streams that are temporally and dynamically coherent *with each other*. Every quantitative metric in the paper (FGD, BC, Diversity) measures individual gesture quality per speaker in isolation. A strawman system running two independent single-speaker generators could score identically on all three metrics. The paper itself acknowledges this gap: *"we will put more effort into designing specific interaction metrics for better concurrent gesture evaluation"* (Limitations section). This is a structural mismatch between the paper's central claim and the evidence provided. The user study partially compensates (15 participants rated "Interaction Coherency"), but the small sample (15 participants, 8 methods, 16 clips, no significance testing, no inter-rater agreement) cannot carry the full evidentiary weight of the paper's main claim.

- **Evaluation is conducted exclusively on the authors' self-constructed dataset**: All quantitative comparisons (Tables 2–5) are performed only on GES-Inter. TWH16.2, the only other concurrent gesture dataset, is cited in Table 1 but never used as an evaluation target. Since the model architecture, hyperparameters, and training choices were developed with knowledge of GES-Inter's domain (mostly seated talk-show postures, pyannote-audio separation pipeline), results on a single self-curated benchmark cannot establish generalizability. Even adapting to TWH16.2 representation would provide independent evidence.

### Minor

- **Shared weight justification is in tension with asymmetry motivation**: The paper motivates bilateral branches by the fact that "the motions of two speakers are asymmetric" (Abstract, Introduction). Yet the justification for sharing weights across branches relies on "exchanging the input order of the speaker's audio results in an invariance effect of interactive body dynamics" (Section 3.3). These two design rationales pull in opposite directions. There is no ablation comparing shared vs. separate weights (Table 3 only ablates TIM and mutual attention), leaving this design choice empirically unsupported. Notably, this may be a reasonable design (different audio inputs drive different outputs through shared parameters), but it needs either justification or an ablation.

- **Audio separation quality is uncharacterized**: The BC metric depends entirely on correct speaker-audio pairing, which is mediated by pyannote-audio diarization on in-the-wild talk-show videos. The error rate of this step is never evaluated. If speaker-audio mis-attribution is non-trivial, BC scores are unreliable as absolute numbers, and improvements over baselines in BC could partly reflect better noise tolerance rather than better rhythmic alignment.

- **User study design is weak for the claims it must support**: 15 participants × 8 methods × 2 clips = effectively 3.75 observations per method-level comparison. No pairwise significance tests are reported, no blind validation that participants understand "interaction coherency" in the technical sense, and no information on rating interface or instructions. Given that the interaction coherency dimension is the paper's primary claim, this user study as described is too thin to be the main supporting evidence.

- **Foot contact loss on upper-body-only model has unclear rationale**: Section 4.1 states: *"since we only model the upper body joints in experiments, we complete the lower body joints as T pose in forward kinematic function during calculate loss."* Foot contact loss was designed to penalize floating/jitter in full-body synthesis; its physical meaning when the lower body is statically T-posed is ambiguous. The ablation shows FGD degrades significantly without it (0.769 → 1.082), but without a clear explanation of what this loss is penalizing in this modified setting, the ablation result is difficult to interpret causally.

### Trivial

- **FGD percentage calculation typo**: The paper writes "$(1.102 - 0.769)/1.012 \approx 24\%$" where the numerator should reference InterGen's FGD of 1.012, not 1.102. The stated 24% result is arithmetically correct ($0.243/1.012 \approx 24\%$) but the textual description introduces confusion.

- **BC metric notation is ambiguous**: Section 4.1 lists "Beat Consistent Score (BC) / Beat Alignment Score (BA)" citing two different papers, without specifying which formula is actually computed.

---

## Nice-to-Haves

- **Two-person joint FGD**: Computing FGD over the joint distribution of (speaker A feature, speaker B feature) would directly measure whether the generated joint distribution resembles real concurrent gesture pairs, as opposed to per-person marginals. This is what the paper's central claim requires and appears straightforwardly computable.
- **Ablation: shared vs. separate branch weights**: Would resolve the tension between the invariance justification and the asymmetry motivation.
- **Audio separation quality characterization**: Even a rough diarization error rate on a held-out subset would help readers calibrate BC values.
- **Long-sequence coherency analysis**: The model generates 6-second clips; an analysis of how interaction coherency evolves across clip boundaries (e.g., sliding window inference) would strengthen practical claims.
- **Failure case analysis**: A structured presentation of failure modes would bound the coherency claims more honestly.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Unfair baseline comparison invalidates SOTA claim"** (Harsh Critic Issue 2): Removed as an independent weakness. Since the paper explicitly introduces a new task ("to the best of our knowledge, we are the first to explore the coherent concurrent co-speech gesture generation"), there are *no* prior methods designed for this task. Using adapted single-speaker and text-to-motion baselines is the only available option. This is not a flaw in the paper—it is a natural consequence of pioneering a new problem. The note that "outperforms SOTA" means "outperforms the best adapted alternatives" is worth mentioning once, but it is not a structural weakness.

- **BC tight clustering suggests small absolute differences**: The harsh critic noted BC values cluster at 0.613–0.692. This observation is noted in the quantitative results but does not invalidate the relative improvement. Minor observation, removed as a standalone weakness.

- **Sequence length too short for conversation dynamics**: The 6-second fixed length is the field norm for this type of model (consistent with TalkSHOW, DiffSHEG, etc.) and is not a comparative disadvantage. Moved to nice-to-haves.

- **Missing related works**: Removed per hard rule (cannot verify external existence of additional related works).

---

## Novel Insights

The most genuinely insightful observation across reviews—not explicitly stated in the paper—is the structural disconnect between the proposed task and the evaluation framework. The paper introduces a two-person generation task whose defining property is *inter-speaker coherency*, but the standard metrics (FGD, BC, Diversity) measure per-speaker marginal quality and are blind to cross-speaker dynamics. This is not unique to Co³Gesture: the HOI-Diff paper (Liang et al.) received similar criticism for using metrics that evaluated human or object motion independently rather than the interaction jointly. The field appears to lack standardized interaction evaluation metrics across multiple two-person generation tasks (gesture, HOI, dance). Co³Gesture's acknowledged limitation points to a broader measurement gap that would benefit the community if addressed.

---

## Suggestions

1. **Design a joint interaction metric before the final version**: Even a simple cross-correlation of velocity profiles between the two speakers, or a two-person FGD computed over concatenated (speaker A, speaker B) features, would directly support the headline claim. This is the single most important revision.
2. **Run evaluation on TWH16.2**: Even with adapted metrics (joint-based rather than SMPL-X), a cross-dataset result would establish generalizability.
3. **Expand user study**: Add pairwise comparison tasks (forced-choice) between the top 3 methods, report inter-rater agreement (Krippendorff's α), and increase to ≥30 participants. Provide explicit instructions and examples for the "interaction coherency" criterion.
4. **Clarify foot contact loss semantics**: Add one paragraph explaining what physical property this loss is actually enforcing when the lower body is T-posed—e.g., whether it functions as a posture orientation regularizer.
5. **Add shared vs. separate weight ablation** to Table 3 to resolve the design tension.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | How it compares |
|---|---|---|---|
| CoCoGesture | `/home/wg25r/review_agent/human_reviews/g3kK6YBSZ1.md` | 4.0 (Withdrawn) | Very similar structure: new gesture dataset + new method + evaluation on own dataset; single-speaker task is less novel; similar methodology novelty level; Co³Gesture is slightly stronger due to task novelty |
| HOI-Diff | `/home/wg25r/review_agent/human_reviews/ZYwLfi50GI.md` | 5.25 (Rejected) | Dual-branch diffusion for interactive motion with new dataset; also lacked interaction-specific metrics; tested on TWO datasets vs Co³Gesture's ONE |
| TANGO | `/home/wg25r/review_agent/human_reviews/LbEWwJOufy.md` | 8.5 (Oral) | Co-speech gesture generation with strong evaluation, independent benchmarks, clean cross-modal alignment design; far stronger than Co³Gesture in evaluation rigor |
| f6GMwpxXHG | `/home/wg25r/review_agent/human_reviews/f6GMwpxXHG.md` | 2.2 (Rejected, low) | GAN with new loss, weak experiments; far weaker than Co³Gesture in methodological grounding |

**Reasoning**: Co³Gesture sits clearly above the low-quality anchor (2.2) and above CoCoGesture (4.0) due to more novel task framing and stronger architectural motivation. It is roughly comparable to HOI-Diff (5.25) in overall quality profile—both introduce new interactive-motion datasets + dual-branch diffusion methods, both lack interaction-specific metrics, but HOI-Diff evaluates on two datasets while Co³Gesture evaluates only on one self-constructed set, which slightly drags it below HOI-Diff. It sits well below TANGO (8.5) in evaluation rigor and technical depth.

The center of the relevant anchor cluster is approximately **4.5–5.0**. Given Co³Gesture's stronger task novelty (truly new problem) than CoCoGesture but slightly weaker evaluation breadth than HOI-Diff, I place it at **5.0**.

**Score: 5.0 / Reject**

The paper makes a genuine contribution in introducing the concurrent co-speech gesture generation task and the GES-Inter dataset. However, the mismatch between the headline claim (coherent concurrent generation) and the evaluation framework (per-speaker individual metrics), combined with evaluation exclusively on a self-constructed benchmark, prevents acceptance at this time. The authors are encouraged to revise with a joint interaction metric and cross-dataset evaluation.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>