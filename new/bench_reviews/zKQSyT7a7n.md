Now I have a good understanding of the paper and the calibration papers. Let me synthesize the final review.

## Summary

The paper introduces Visuo-Tactile World Models (VT-WM), the first multi-task latent world model that fuses exocentric RGB observations (Cosmos encoder) with fingertip tactile sensing (Sparsh-X encoder from Digit 360 sensors on an Allegro Hand) to ground contact dynamics in robot manipulation. A transformer predictor processes concatenated vision and tactile tokens with action-conditioned cross-attention to autoregressively predict next-step latent states. VT-WM is evaluated on imagination quality (object permanence and causal compliance metrics via CoTracker), zero-shot real-robot planning via CEM, and data efficiency against behavioral cloning on a novel plate-insertion task.

## Strengths

1. **Well-motivated and clearly articulated problem:** The paper identifies a concrete failure mode of vision-only world models—hallucinated contact dynamics under occlusion—and proposes tactile sensing as a principled remedy. The argument that touch disambiguates visually aliased contact states (e.g., grasp vs. no-grasp) is compelling and well-illustrated (Fig. 1, Fig. 7).

2. **Rigorous imagination quality evaluation with statistical tests:** The use of CoTracker-based normalized Fréchet distance with paired t-tests across five tasks provides a more principled assessment than typical qualitative rollout comparisons. Statistically significant improvements are shown on several tasks (place fruits, push fruits for object permanence; place fruits, push fruits, wipe cloth for causal compliance).

3. **Real-robot zero-shot transfer results:** Demonstrating that VT-WM plans transfer directly to a physical Franka+Allegro system across five tasks of increasing difficulty is a significant experimental contribution, with the largest gains on multi-step contact-rich tasks (35% on reach & push, 31% on wipe cloth) aligning with the hypothesis.

4. **Complete end-to-end system:** The paper presents a complete pipeline from sensing (Digit 360) through pretrained encoders (Cosmos, Sparsh-X) through a transformer predictor to CEM planning on a real robot—a non-trivial engineering contribution that validates the concept experimentally.

5. **Tactile prediction congruence:** The paper demonstrates that VT-WM generates consistent tactile predictions alongside visual ones (Appendix B, Fig. 13), showing the model genuinely leverages contact information rather than just using tactile as auxiliary signal.

## Weaknesses

### Major:

- **The "object permanence" and "causal compliance" metrics are generic trajectory errors, not direct measurements of the semantic properties claimed.** The "object permanence" metric (normalized Fréchet distance between CoTracker keypoints in ground-truth and model rollouts) simply measures how closely predicted object trajectories match ground truth—it will decrease whenever the model better matches global object motion, regardless of whether the object is correctly represented while occluded or correctly reappears after occlusion. Similarly, "causal compliance" conflates all sources of tracking error (registration noise, background motion, etc.) with violations of Newton's first law. The abstract claims "33% better performance at maintaining object permanence" and "29% better compliance with the laws of motion," but these are over-interpretations of generic rollout accuracy. A more direct evaluation would track object visibility during occlusion phases, detect disappearance/teleportation events explicitly, or measure trajectory error only during occluded intervals. That said, the qualitative evidence (Fig. 5, Fig. 7) does show VT-WM preserving object representations under occlusion and preventing phantom motion, so the underlying phenomenon is likely real—the metrics just don't isolate the claimed properties as precisely as stated.

- **The planning comparison between VT-WM and V-WM may not fairly isolate the contribution of tactile information.** Key details about the V-WM baseline are missing: whether it has the same architecture, parameter count, and training recipe (just omitting tactile tokens), or whether VT-WM effectively receives more information at similar capacity. If VT-WM gets additional tokens and parameters, gains could partially reflect capacity/information advantages rather than contact grounding per se. Additionally, CEM hyperparameters (population size, iterations, sampling distribution) are not reported, and it is unclear whether they were tuned equally for both models. The planning cost function is purely vision-based (ℓ2 in visual latent space), which means success reflects better visual prediction under sampled actions, not necessarily tactile-informed physics reasoning. These ambiguities weaken the causal interpretation that tactile grounding specifically causes the planning improvements.

- **The data-efficiency comparison (VT-WM vs. ACT) confounds the benefit of multi-task pretraining with the benefit of touch.** VT-WM is pre-trained on a diverse suite of contact-rich tasks and then fine-tuned on 20 demonstrations, while ACT is trained from scratch on those same 20 demonstrations. This is effectively a comparison between "large pre-trained model + planning" and "small model trained from scratch"—the structural advantage of pretraining is enormous and not acknowledged. A V-WM with identical pretraining fine-tuned on the same 20 demos would be a much more informative baseline to isolate tactile's contribution to data efficiency. Furthermore, only 9 trials are reported with no confidence intervals, making the 77% vs. 22% difference unreliable.

### Minor:

- **Degradation on "scribble with marker" is unexplained.** VT-WM shows worse causal compliance than V-WM on this task (t = −1.22, p = 0.23), which contradicts the narrative that tactile universally improves physical fidelity. No analysis of this failure mode is provided, which would be valuable for understanding the boundaries of the approach.

- **Open-loop-only planning limits applicability conclusions.** The CEM planner generates a single action sequence executed open-loop. For contact-rich tasks where closed-loop feedback matters most, this leaves a gap between the claimed contact grounding and actual deployment robustness. The paper acknowledges this implicitly (using zero-shot open-loop transfer as the evaluation) but does not discuss it as a limitation.

- **The loss function uses equal weighting for vision and tactile latent reconstruction, but no justification or sensitivity analysis is provided.** These modalities have very different information densities and scales, and the choice of equal ℓ1 weighting could affect how much the model attends to each modality.

### Trivial:

- **CEM and planning hyperparameters deferred to appendix or missing.** While core architectural details are provided, CEM population size, iterations, and noise schedule are not specified in the main text, making the planning results harder to interpret and reproduce.

## Nice-to-Haves

- **Closed-loop replanning experiments** would significantly strengthen the deployment relevance argument. Even periodic replanning every K steps would demonstrate whether VT-WM's contact grounding provides practical advantages under execution errors.
- **Ablation on tactile provided only at initialization vs. throughout rollout** would clarify whether gains come from training-time grounding or test-time disambiguation.
- **A V-WM with identical pretraining fine-tuned on the new task** for the data efficiency comparison, to isolate touch's contribution.
- **Direct disambiguation demonstration:** paired rollouts from visually identical but tactually distinct initial states would directly validate the core motivation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released" / availability concerns about Digit 360, Sparsh-X, Cosmos, or other cited models and datasets.** The paper cites these as existing, and per the rules, we assume they exist and are available.

- **Missing external baselines (DreamerV3, TD-MPC2, etc.).** The comparison is an ablation (VT-WM vs V-WM), which is the appropriate comparison to isolate the contribution of adding tactile to a world model. Comparing to entirely different world model families would not directly test whether touch improves world models—it would conflate architectural differences with modality differences. The V-WM baseline is the right comparison for this paper's core claim.

- **Reproducibility nitpicks about hyperparameters, training details, dataset size.** These are standard for systems-style papers and are addressed in the appendix. The core claims can still be evaluated without every implementation detail.

- **Hardware-specific generality concerns (Digit 360 sensor requirements, applicability to simpler grippers).** This is scope creep. The paper demonstrates the approach on a specific hardware platform; generalizing to other hardware configurations is future work and not a weakness of the current contribution.

- **Missing comparison to other tactile-aware planning approaches.** The paper states in Section 2 that prior tactile dynamics models are task-specific, while VT-WM is multi-task. There are no other multi-task visuo-tactile world models to compare against.

- **Tactile noise/robustness concerns.** Training is on real-world data from Digit 360 sensors, which already includes real sensor noise. This is not a simulation-only paper where noise would be artificially clean.

## Novel Insights

The key insight of this paper—that tactile sensing provides a local contact signal that disambiguates visually aliased states in world models—is both intuitive and well-supported by the qualitative evidence. The failure mode of vision-only world models hallucinating object motion during no-contact periods (e.g., the wiping example in Fig. 7) is compellingly demonstrated. However, the paper's strongest claim—that VT-WM specifically understands "object permanence" and "causal compliance" in a physical sense—is not well-matched to the metrics used. The normalized Fréchet distance measures rollout accuracy, not occlusion-aware object persistence or force-based causation. The paper makes a genuine contribution in demonstrating that tactile grounding improves contact-rich manipulation in world models, but should frame its contributions more precisely as "reduces rollout trajectory error and improves planning success in contact-rich tasks" rather than claiming principled understanding of object permanence and physical laws.

## Suggestions

- Reframe the "object permanence" and "causal compliance" claims more precisely: report the metrics as "trajectory error on objects" and "trajectory error on static objects," and supplement with explicit detection of disappearance/teleportation events to support the stronger claims about physical reasoning.
- Add a V-WM fine-tuned baseline in the data efficiency experiment to isolate touch's contribution from pretraining's contribution.
- Discuss the "scribble with marker" degradation and what it reveals about when tactile sensing may hurt rather than help.
- Report confidence intervals on the planning success rates (particularly given n=5 per task), or increase trial counts.
- Specify CEM hyperparameters in the main text and confirm both models use identical search parameters.

## Score and Decision

**Calibration comparison:**

- **Mani-WM** (world model for manipulation, limited baselines, open-loop planning): Scores 6/5/3, rejected.
- **DINO-WM** (latent world model, zero-shot planning, limited benchmarks): Scores 6/6/6/5, rejected.
- **M3L** (vision+touch for manipulation, simulation-only, limited tasks): Scores 3/5/5, rejected.
- **ViTaS** (vision+tactile for RL, limited tasks/baselines): Scores 5/6/3/8, rejected.
- **GR-1** (video pretraining for manipulation, strong results but limited generalization): Scores 8/6/5/3, accepted poster.
- **FLIP** (planning world model, real robot): Scores 6/6/6/8, accepted poster.

This paper is stronger than Mani-WM, DINO-WM, M3L, and ViTaS because it demonstrates real-robot zero-shot transfer with a well-motivated approach, has sound statistical testing on imagination quality, and addresses a clear problem. However, it is weaker than GR-1 (which shows more extensive generalization) and FLIP (which validates on more benchmarks). The key issues that lower the score are: (1) metric-claim mismatch that overstates what the experiments actually measure, (2) the data efficiency comparison confounds pretraining with tactile, and (3) small trial counts for planning. These are substantive but not fatal—the core contribution of demonstrating tactile grounding for world models is valid and useful.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>