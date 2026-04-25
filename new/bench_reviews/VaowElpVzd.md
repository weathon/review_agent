Now let me search for calibration anchors.Now I have enough information to write the final consolidated review.

---

## Summary

Co³Gesture introduces the new task of concurrent two-speaker co-speech 3D gesture generation, where both speakers' gestures are synthesized jointly from conversational audio. The paper contributes (1) **GES-Inter**, a 70-hour, 7M-frame dataset of whole-body concurrent co-speech gestures with SMPL-X mesh representation, multi-modal annotations (audio, phoneme, text, face), and (2) a bilateral cooperative diffusion framework with a Temporal Interaction Module (TIM) and mutual attention to model inter-speaker dynamics. Experiments on GES-Inter show FGD/BC/Diversity improvements over adapted single-person gesture and text-to-motion baselines, with ablations validating each design component.

---

## Strengths

- **Novel task + first large-scale concurrent dataset (Table 1):** GES-Inter is the first pseudo-label dataset providing whole-body mesh-based concurrent postures for two speakers at 70 hours/7M frames. The comparison in Table 1 clearly documents the gap it fills relative to existing datasets (TWH16.2 has concurrent gestures but lacks facial, mesh, and sufficient scale; BEAT2/SHOW have richer annotations but no concurrent speaker coverage). This is a genuine infrastructure contribution.

- **Informative ablation studies (Tables 3–5):** The bilateral-branch ablation (FGD: 0.769 → 1.669 without bilateral branches) and audio separation/mixing ablation (BC: 0.692 → 0.633 without audio separation; FGD: 0.769 → 1.227 without mixed audio) test the paper's specific design choices against each other on the same task. These are the paper's most interpretable results, as they isolate Co³Gesture-specific contributions rather than comparing against non-dedicated adapted methods.

- **Explicit acknowledgment of limitations:** The paper honestly states "we will put more effort into designing specific interaction metrics for better concurrent gesture evaluation" and acknowledges that automatic pose extraction may leave bad instances. This transparency is appropriate.

---

## Weaknesses

### Fatal
None.

### Major

- **The central claim is not quantitatively evaluated.** The paper's headline contribution is *coherent concurrent interaction* between two speakers. Yet all three quantitative metrics (FGD, BC, Diversity) measure individual gesture realism, rhythm alignment for a single speaker, and per-speaker diversity, respectively. None measures whether Speaker A's generated gestures are temporally coherent with Speaker B's in the same clip. The user study does include an "interaction coherency" criterion, but with only 15 participants evaluating 2 samples per method (30 total ratings per method), with no inter-rater reliability or statistical significance testing, this is insufficient to support quantitative claims. The authors themselves acknowledge the evaluation gap in the limitations section ("specific interaction metrics"). This is a structural mismatch between the paper's title/claims and its evidence.

- **Fully closed-loop evaluation on a proprietary dataset.** All quantitative results are on GES-Inter, which the authors construct. The FGD metric is computed via an autoencoder trained exclusively on GES-Inter. All baselines are retrained on GES-Inter using the authors' preprocessing choices. There are no results on any external benchmark (e.g., running the single-speaker variant of Co³Gesture on SHOW or BEAT2 to compare its underlying gesture quality to known standards). This means any dataset construction choices that favor Co³Gesture's motion representation are undetectable.

- **Baselines are entirely adapted methods with no prior concurrent-gesture art.** While introducing a new task necessarily means no prior dedicated methods exist, the competitive framing in Table 2 is misleading. All compared methods were purpose-built for other tasks (single-person gesture generation or text-to-motion) and have been adapted post-hoc; InterGen and InterX are text2motion models with the audio encoder swapped in. That a purpose-built method outperforms ad-hoc adaptations of non-dedicated methods is expected and does not establish how strong Co³Gesture is in absolute terms. The paper would benefit from making this framing explicit (exploratory/proof-of-concept) rather than presenting it as a standard competitive evaluation.

### Minor

- **TIM provides only weak inter-speaker coupling through audio.** In Eq. (1), the cross-attention uses the current speaker's audio embedding ($f_{C_a}$) as query against the current speaker's *own* motion embedding ($f_{x_a}$) as K and V. Inter-speaker interaction enters the model only through the mixed audio signal $C_{mix} = C_a + C_b$ used to compute $f_{x_a, C_{mix}}$. The other speaker's *motion* features never directly attend to Speaker A's motion in TIM; interaction is implicit through audio mixing. This is a limitation of the claimed "temporal interaction" modeling that is underacknowledged.

- **FGD improvement claim contains a numerical error.** Section 4.2 states "(1.102 − 0.769)/1.012 ≈ 24%". The value 1.102 does not appear anywhere in Table 2; the second-best FGD is InterGen at 1.012. The correct expression for the claimed ≈24% improvement is (1.012 − 0.769)/1.012 ≈ 24%, which happens to be numerically correct despite the erroneous numerator. This appears to be a transcription error that should be fixed.

### Trivial

- The foot contact loss applied to upper-body-only generation (with lower body completed as T-pose) is non-standard. The paper acknowledges this and ablates it (Table 5 shows it matters), but a brief physical explanation of *why* T-pose lower body regularizes upper body generation would improve clarity.

---

## Nice-to-Haves

- **Develop a quantitative interaction metric** (e.g., pairwise joint correlation, inter-speaker synchrony, or a learned interaction quality classifier trained on the dataset) and apply it before final submission. The limitation acknowledgment is honest but this is the paper's most pressing gap.
- **External evaluation:** Run Co³Gesture (single-speaker variant) on SHOW or BEAT2 to cross-validate the underlying gesture quality against established benchmarks, separating dataset construction effects from model effects.
- **Ablation of separate vs. shared branch weights:** The paper motivates asymmetric dynamics but adopts shared weights for an invariance argument; an ablation confirming shared weights outperform separate weights would validate this design choice.
- **Richer visualization of interaction:** Showing joint trajectories of both speakers over time in the same clip alongside audio energy curves would make inter-speaker coordination patterns visible beyond key-frame snapshots.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Harsh critic W2 (asymmetry vs. shared weights = "internal contradiction"):** The paper reconciles this explicitly in Section 3.3: asymmetry is in behavior (one speaker moves more while the other is quiet), but the *distribution* of each speaker's role is symmetric ("exchanging the input order of the speaker's audio results in an invariance effect of interactive body dynamics"). Sharing weights is a reasonable architectural choice given distributional symmetry; this is not a contradiction. **Removed: paper addresses the concern.**

- **Harsh critic W3 (baseline comparisons "structurally unfair"):** The critic argues that comparing a dedicated system to non-dedicated adapted methods proves nothing. However, the baselines are harder on the adapted methods—the unfairness, if any, favors *the baselines'* original designs on their own tasks, not Co³Gesture. By the hard rule, asymmetry favoring baselines should not be kept as a weakness. The real issue (that this looks misleadingly competitive) is captured in the Major weakness above in a more nuanced framing. **Removed per hard rule; absorbed into Major weakness with softened framing.**

- **Harsh critic (dataset quality validation—65% rejection, pose estimation error):** The paper states professional inspectors manually annotate and double-check audio-posture alignment. Requesting full ground-truth mocap validation of pseudo-label datasets is not standard practice in this field; existing accepted work (TalkSHOW, TED) uses the same paradigm without such validation. **Moved to nice-to-have / weakened.**

---

## Novel Insights

The paper surfaces an underappreciated asymmetry in two-person gesture modeling: the motion statistics of each speaker, conditioned on the interaction, are distributionally symmetric (either speaker can play either role), even though the *instantaneous* behavior during conversation is asymmetric (one moves while the other is relatively still). This motivates shared weights for the bilateral branches while still separately conditioning each branch on its own audio. This "symmetric prior, asymmetric behavior" framing is a genuinely useful design insight, though it would benefit from empirical validation (ablation: shared vs. separate weights). The TIM architecture, which balances individual audio-fidelity against interaction coherency through a learned soft weight $\sigma$, is a clean implementation of this idea—the weakness is that the interaction signal feeding into $\sigma$ is audio-only and lacks direct motion-level coupling between speakers.

---

## Score and Decision

**Evaluation axes:**
- *Originality:* Moderate–good. The two-speaker concurrent setting is genuinely novel; the method adapts known diffusion components (bilateral branches, cross-attention) to a new structure.
- *Importance of research question:* High. Concurrent interaction generation is practically relevant and underexplored.
- *Claims well-supported:* Weak. The primary claim (coherent concurrent interaction) is supported only by a small user study and individual-quality metrics that don't measure interaction.
- *Soundness of experiments:* Below average. Closed-loop evaluation, fully adapted baselines, no external benchmark, no interaction metric.
- *Clarity of writing:* Adequate. Minor arithmetic error in Section 4.2.
- *Value to research community:* Moderate. Dataset is a genuine contribution; method is exploratory.

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| CoCoGesture | g3kK6YBSZ1.md | **4.00** (Withdrawn) | Same group (HKUST), co-speech gesture + large dataset, rejected for weak motion-speech correspondence; Co3Gesture has cleaner ablations and more specific task novelty but shares the closed-loop evaluation problem |
| MMG-VL | B5AN6IRyXc.md | **4.00** (Withdrawn) | Multi-person motion generation + new dataset, rejected; shares the pattern of closed-loop evaluation + adapted baselines |
| Human Motion Diffusion (Prior) | dTpbEdN9kr.md | **6.00** (Poster) | Two-person motion generation with diffusion composition; accepted with rigorous ablations on external benchmarks — Co3Gesture is weaker on evaluation rigor |
| TANGO | LbEWwJOufy.md | **8.50** (Oral) | Co-speech gesture with strong evaluation on standard external benchmarks and strong ablations — significantly above Co3Gesture |

Co3Gesture sits above MMG-VL and CoCoGesture (which were both straightforwardly rejected) because its task novelty is more focused, its ablations are richer, and the dataset itself is a genuine community resource. However, it falls clearly below the accepted poster benchmark (Human Motion Diffusion, 6.0) because: (a) it cannot quantitatively measure its central claim, (b) all evaluation is closed-loop, and (c) the baselines are all architecturally mismatched. Anchoring on the low cluster (4.0 for similar pattern papers) and the medium anchor (6.0 for accepted two-person diffusion paper), with the paper sitting closer to the low cluster due to the evaluation gap, I arrive at **4.5**.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>