##Summary

This paper introduces Visuo-Tactile World Models (VT-WM), a multi-task latent world model that integrates exocentric vision (Cosmos encoder) and fingertip tactile sensing (Sparsh-X encoder on Digit 360 sensors) to ground contact physics in robot manipulation imagination. The transformer-based predictor fuses visual and tactile tokens via factorized spatio-temporal self-attention and action cross-attention, trained with combined teacher-forcing and autoregressive sampling losses. Experiments on a Franka+Allegro Hand platform demonstrate improved object permanence (~33%) and causal compliance (~29%) in imagined rollouts versus a vision-only baseline, along with up to 35% higher success rates in zero-shot real-robot planning on contact-rich tasks and strong data efficiency (77% vs. 22% BC) on a novel plate-insertion task.

## Strengths

- **Well-motivated multimodal integration for a real gap.** The paper identifies a concrete failure mode of vision-only world models—visual aliasing of contact states—and demonstrates that tactile input specifically disambiguates these cases (Fig. 7: V-WM hallucinates cloth displacement when the hand hovers; VT-WM correctly predicts stasis). This is a targeted, non-obvious contribution rather than a generic "add a modality" approach.

- **Demonstrated real-robot zero-shot transfer.** The planning results go beyond latent-space metrics to actual physical execution. The 31% gain on wipe cloth and 35% gain on reach & push are substantively meaningful because they correspond to qualitatively different behaviors (establishing contact vs. hovering above the object), not marginal improvements.

- **Data efficiency result is practically compelling.** The 77% vs. 22% comparison on plate insertion with only 20 demonstrations shows the multi-task contact priors transfer meaningfully. The failure-mode analysis (VT-WM places beside rack; BC never reaches rack) provides mechanistic insight into why the WM representation helps.

- **Honest reporting of negative results.** The causal compliance evaluation shows VT-WM *degrades* on scribble with marker (t = −1.22, p = 0.23), and the paper reports this without obscuring it. This strengthens trust in the positive claims.

## Weaknesses

### Major:

- **No quantitative evaluation of the tactile prediction channel.** The paper claims to capture "the physics of contact through touch reasoning," yet all quantitative metrics (Fréchet distance, success rates) measure *visual* outcomes. The tactile predictions are shown qualitatively in Appendix B (Figs. 12, 13) but never evaluated quantitatively (e.g., tactile prediction error, contact detection accuracy, slip prediction). A world model that claims to reason about contact should demonstrate that its *tactile* predictions are accurate, not merely that tactile inputs improve visual outputs. Without this, the mechanism—"tactile grounding disambiguates contact states"—is asserted but only indirectly validated.

- **Data efficiency experiment conflates planning vs. BC with tactile vs. vision-only.** Section 4.3 compares VT-WM (world model + CEM planning) against ACT (behavioral cloning). This conflates two variables: the representation paradigm (WM vs. policy) and the modality (visuo-tactile vs. vision-only). A V-WM + CEM baseline in the same low-data setting would isolate whether the advantage comes from tactile grounding or from planning vs. cloning. The paper's own Limitations section partially acknowledges this ("does not fully rule out the possibility that a multi-task BC policy could also exhibit strong data efficiency"), but the comparison as presented is misleading about the *source* of the gain.

- **V-WM and VT-WM parameter/capacity matching is unspecified.** VT-WM concatenates tactile tokens alongside visual tokens, giving the transformer more input information and effectively more capacity per forward pass. The paper does not state whether V-WM was given additional visual tokens or architectural capacity to match, or whether it is strictly the same transformer with fewer input tokens. If V-WM is simply VT-WM minus the tactile tokens, the 33%/29% gains could partly reflect the benefit of additional tokens/attention targets rather than tactile grounding per se. A V-WM with matched token count (e.g., duplicated visual tokens or deeper processing) would isolate the modality effect.

### Minor:

- **Temporal alignment between vision and tactile inputs is unclear.** Section 3.2.2 states tactile input consists of "two frames per Digit 360 sensor, covering the most recent 0.16 seconds," while vision uses a 1.5-second clip. Yet it also states "The model uses a maximum context length of 9 frames for both vision and touch modalities." How 2 frames of tactile at high frequency map to a 9-frame temporal context alongside the 9-frame visual context is not explained. This creates ambiguity about whether tactile history is subsampled, padded, or operates at a different effective framerate within the transformer.

- **Binary gripper action space limits the scope of "contact-rich manipulation" claims.** The Allegro Hand has 16 DOF, but the action space uses only "a binary hand state representing pre-set open/close configurations" (Section 3.2.2). This reduces the hand to a parallel-jaw gripper. While the tested tasks (pushing, wiping, stacking) are genuinely contact-rich in the sense of requiring sustained physical contact, the claims about "dexterous manipulation" and "physics of contact" would be more precise if qualified—contact richness here comes from task geometry, not from finger-level dexterity.

- **CoTracker metric may be unreliable precisely where the model's advantage lies.** The Fréchet distance metric relies on CoTracker to track keypoints through occluded phases (e.g., object in hand). CoTracker can lose track during heavy occlusion and re-acquire afterward, potentially introducing noise that affects both models equally but reduces the metric's sensitivity to the very phenomenon (object permanence under occlusion) that VT-WM is claimed to improve. This does not invalidate the results but means the 33% figure may be an underestimate or overestimate.

- **Negative result on scribble with marker is not discussed.** VT-WM shows *worse* causal compliance than V-WM on this task (t = −1.22). The paper reports the number but offers no analysis of why tactile input might hurt in this case (e.g., marker contact is always present during the task, so tactile provides no disambiguation but adds noise). Understanding when touch *doesn't* help is as important as when it does for evaluating the generality of the approach.

- **Sparsh-X fine-tuning is not ablated.** Appendix A.1 states "fine-tuning the Sparsh-X encoder was beneficial" while Cosmos was kept frozen. The asymmetry is not justified or ablated. If fine-tuning the tactile encoder substantially improves performance (while the visual encoder remains frozen), this could mean the gains come from domain-adapted representations rather than from multimodal fusion per se. An ablation with frozen Sparsh-X would clarify this.

### Trivial:

- The CEM planning algorithm (Algorithm 1) uses 36 particles and 10 iterations, which is standard and adequate for the demonstrated tasks. No issue here.

## Nice-to-Haves

- **Closed-loop replanning experiments.** The open-loop execution is acknowledged as a limitation. Even a simple 2-step replanning demonstration (update context after first chunk) would substantially strengthen the practical relevance argument.

- **Multi-task BC baseline for data efficiency.** A multi-task ACT policy trained on all tasks (including the 20 plate-insertion demos) would clarify whether the 77% vs. 22% gap is due to the world model paradigm or the multi-task transfer.

- **Latent space analysis.** Attention visualizations on tactile tokens during contact-rich vs. free-motion phases, or a probing experiment, would provide mechanistic evidence that the model actually uses tactile information rather than ignoring it.

- **Inference latency quantification.** CEM with 36 particles × 10 iterations of autoregressive rollout is computationally expensive. Reporting wall-clock time for planning would help readers assess deployment feasibility.

- **Generalization to novel objects.** Testing on objects with different sizes, weights, or friction coefficients than those in training would strengthen the multi-task generalization claim, though the paper explicitly scopes this out.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"First multi-task visuo-tactile world model" claim disputed.** The harsh critic questions whether prior tactile servoing/dynamics work (Tian et al., 2019; Sutanto et al., 2019) invalidates the "first" claim. These prior works are task-specific dynamics models, not multi-task latent world models for planning. The distinction is clear in the paper's Related Work section. Removed as factually ungrounded—the paper's claim is specific to the multi-task latent WM + planning setting.

- **Missing comparison to Robopack or other tactile-dynamics baselines.** The balanced reviewer requests comparison to Robopack (Ai et al., 2024). Robopack addresses dense packing with tactile-informed dynamics, which is a different task and architecture. The paper's primary comparison is the ablation (V-WM vs. VT-WM), which is the most informative baseline for isolating the tactile contribution. Removed as unfair comparison demand—different problem setting.

- **Formatting/parser artifact complaints.** Both reviewers mention broken equations and tables from PDF extraction. Per hard rules, formatting nitpicks are removed.

- **Reproducibility concerns about fine-tuning details for Sparsh-X.** The harsh critic wants more details about Sparsh-X fine-tuning. Per hard rules, nitpicks about trivial implementation details are removed. The paper already states the fine-tuning was done and the appendix provides training parameters.

- **Compute infrastructure for inference.** The harsh critic asks about the inference hardware stack. Per soft rules, this is a nice-to-have, not a core flaw. Moved above.

## Novel Insights

The paper reveals an asymmetry in how tactile grounding helps: it provides the largest gains in tasks where *visual aliasing of contact states is the primary failure mode* (pushing, wiping, reach & push), but offers diminishing or even negative returns in tasks where contact is always present or where visual information is already sufficient (scribble with marker, reach button). This suggests that the value of tactile sensing in world models is not uniform but concentrated in the specific regime where contact state is under-determined by vision alone—a finding with practical implications for when tactile hardware investment is justified versus when vision-only approaches suffice.

## Suggestions

- Add a quantitative tactile prediction metric (e.g., L1 loss on predicted vs. ground-tr