Now let me search for calibration anchors.Now I have enough information to write the final meta-review. Let me compose it.

---

## Summary

This paper introduces **trajectory attention**, a novel attention mechanism that performs attention along known pixel trajectories across frames for fine-grained camera motion control in video generation. Rather than replacing temporal attention, the method models trajectory attention as a parallel auxiliary branch that inherits QKV weights from temporal attention and adds its output as a residual. This design cleanly separates the roles of trajectory-consistency enforcement (trajectory attention) and natural motion synthesis (temporal attention). The method is evaluated on camera motion control for single images and videos, and is also applied to first-frame-guided video editing.

---

## Strengths

- **Well-motivated and clean auxiliary branch design (Sec. 3.3, Fig. 3, Table 3):** The distinction between temporal and trajectory attention — one focused on short-range dynamics, the other on long-range trajectory consistency — is supported by concrete attention map visualizations (Figure 2) and a systematic ablation (Table 3) showing monotonic improvement: vanilla (ATE 1.7812) → + Tuning (0.3147) → + Add-on Branch (0.0724) → + Weight Inheriting (0.0396). This 45× ATE reduction validates the design philosophy.

- **Principled training strategy (Sec. 3.4, Fig. 5):** Zero-initialization of the output projector and QKV inheritance from temporal attention are sound design choices that enable smooth training start and fast convergence, supported by the ablation's "weight inheriting" row.

- **Competitive primary-task quantitative results (Table 1, 25-frame group):** At 25 frames, the method outperforms CameraCtrl (RPE-Rot: 0.1939 vs. 0.3480, a ~44% improvement) and NVS_Solver on all four metrics, with a stronger FID (103.5 vs. 115.8 / 108.5). The RPE rotation improvement is substantial.

- **Training efficiency (Sec. 5.1):** ~24 GPU hours on a single A100 with 10k training clips, while generalizing from a 12-frame training regime to 25-frame generation, is a practical advantage.

- **Algorithmic transparency (Algorithms 1–4):** The trajectory sampling and back-projection operations are specified precisely, including occlusion-aware masking. The pipeline is well-described and independently reproducible.

---

## Weaknesses

### Fatal
*None.*

### Major

- **No quantitative evaluation for video editing (Sec. 5.4, Fig. 8):** Video editing (first-frame-guided) is presented as a co-equal third contribution in the abstract and Figure 1. Yet Section 5.4 provides only two qualitative examples (Figure 8) with no FID, CLIP score, SSIM, LPIPS, or temporal consistency metric of any kind. The claim that the method "excels in maintaining content consistency over large spatial and temporal ranges" is unverifiable from two cherry-picked pairs. For a contribution that is elevated to abstract-level status, qualitative-only evaluation is a significant gap.

- **Single baseline for video camera motion control (Table 2, Sec. 5.3):** The video camera control task (Section 5.3) is compared against only one baseline: NVS_Solver. A single baseline is insufficient to establish state-of-the-art for this task. Additionally, the improvement in ATE (0.5112 → 0.3572) is moderate, and the strongest variant in the table ("Ours w. NVS_Solver") is a combination, making it unclear whether standalone trajectory attention alone is competitive against a wider pool of methods.

### Minor

- **Base-model heterogeneity in Table 1 (16-frame group):** The 16-frame comparison pits the proposed method (built on SVD) against Motion2V (built on AnimateDiff). The paper does acknowledge this with an asterisk, but the footnote does not resolve the confound — improvements over Motion2V at 16 frames cannot be attributed solely to trajectory attention when the base generative prior differs. The 25-frame group (SVD-based methods only) is the most interpretable comparison and should be the headline result. The 14-frame comparison with MotionCtrl suffers from the same heterogeneity (different released versions). The paper should either provide a unified same-base-model comparison or more clearly de-emphasize the heterogeneous sub-groups.

- **Ablation "Vanilla" row computed on a filtered subset (Table 3, Sec. 5.5):** The paper explicitly states that "some outputs exhibiting complete noise (we omit such invalid results during evaluation)." Computing ATE/RPE/FID on a non-failed subset makes the Vanilla baseline's numbers incomparable to all other rows (which include all outputs). Even though 1.7812 ATE still looks catastrophically bad, the actual performance with failures included would be worse, and the *magnitude* of the improvement from Vanilla → +Tuning is understated relative to what would be computed on full outputs. This should be noted more prominently.

- **Claim about "broader attention window" rests on a single example (Fig. 2):** Figure 2 is from SVD and shows one attention map. The paper presents this as a general property of trajectory attention, but a single visualization is not convincing evidence. Averaging across multiple layers and examples would strengthen this claim.

### Trivial

- **Section 5.6 is purely qualitative (Fig. 9):** The extension to Open-Sora-Plan (full 3D attention) is presented with four qualitative examples only and no metrics. This reads as a teaser rather than a validated extension. Given the paper's primary scope, this is acceptable, but calling it a validated contribution would be overclaiming.

---

## Nice-to-Haves

- A unified comparison of all image camera control baselines at a single frame count (ideally 25) would make Table 1 cleaner and more interpretable, even if some baselines cannot operate at 25 frames (they could be excluded from that sub-table with explanation).
- An analysis of sensitivity to depth estimation errors (Algorithm 3 requires monocular depth) would characterize practical robustness.
- Failure mode examples (large displacements, heavy occlusion, inaccurate optical flow) would clarify the method's scope.
- At least one quantitative metric for video editing (e.g., CLIP-T, SSIM against reference trajectory propagation) would make the third contribution publishable on its own merits.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

**Harsh Critic: "Separation of base model vs. trajectory attention contribution" as a required experiment.** The 25-frame group in Table 1 already compares trajectory attention (Ours) against CameraCtrl — both built on SVD — which does isolate the contribution of the mechanism. This experiment partially exists and the demand for a fully separate ablation is scope creep.

**Harsh Critic: Long-range consistency metric demand.** ATE and RPE are standard camera trajectory accuracy metrics in the NVS literature. Demanding the authors introduce a novel appearance-consistency metric (e.g., feature drift across trajectory-aligned crops) is beyond the paper's scope. The existing metrics are appropriate for the stated task.

**Harsh Critic: Test set overlap with training data concern.** The paper uses MiraData for training (10k clips) and 230 scene/trajectory combinations for evaluation. The harsh reviewer raises overfitting concerns but provides no evidence of overlap. This is removed as speculative.

**Harsh Critic: Robustness experiment corrupting trajectories.** A reasonable suggestion but outside the core paper scope. Moved to Nice-to-Have territory if anything.

**Harsh Critic: FID as unsuitable metric.** FID is standard in the NVS and video generation literature and is used consistently by all compared methods. Criticizing the choice of FID without a concrete alternative already adopted by the community is not actionable.

**Strength Finder: "Extension to video editing task" as a strength.** Since the video editing task has no quantitative evaluation, it cannot be claimed as a verified strength. Removed from strengths section per the rule that strengths conflicting with verified weaknesses must be removed.

---

## Novel Insights

The paper's most insightful observation is the **functional divergence between temporal and trajectory attention**: temporal attention emerges from large-scale training to prioritize local, adjacent-frame consistency (observable in Figure 2's diagonal concentration), while trajectory attention, given explicit dynamics, naturally attends globally. This divergence motivates the two-branch architecture more rigorously than a pure engineering argument would. The zero-initialization + weight-inheritance training strategy is a concise, principled realization of this insight, avoiding the need for careful hyperparameter tuning to balance the two branches. This pattern — separate mechanisms for implicit and explicit inductive biases in video generation — may generalize beyond camera control to other structured motion priors (object trajectories, skeleton poses).

---

## Calibration

**Anchors reviewed:**
- `/home/wg25r/review_agent/human_reviews/Gx04TnVjee.md` — 3DTrajMaster, avg 6.75, Accept. Multi-entity 3D motion control with gated self-attention; has a novel synthetic dataset (360-Motion), strong multi-baseline comparison, and extensive qualitative+quantitative results. Stronger than the paper under review due to more comprehensive evaluation.
- `/home/wg25r/review_agent/human_reviews/rDRCIvTppL.md` — Boosting Camera Motion Control (CMG), avg 5.50, Reject. Camera motion guidance for DiT; had comprehensive analysis but was perceived as more of a technical report than a novel method. Similar positioning to the paper under review but with weaker fundamental novelty.
- `/home/wg25r/review_agent/human_reviews/CU7QfWJ6nC.md` — FreeTraj, avg 5.50, Reject. Tuning-free trajectory control for video diffusion; competitive approach but limited novelty in components.
- `/home/wg25r/review_agent/human_reviews/n6To2wAOKL.md` — Ctrl-V, avg 4.00, Withdrawn. Motion-controlled video generation; lacked novelty and failed to show clear advantage. Below the paper under review in both novelty and evidence.
- `/home/wg25r/review_agent/human_reviews/FHhj5d2gYe.md` — LIVE (video editing), avg 4.00, Withdrawn. Limited technical novelty, insufficient experimental validation. Below the paper under review.

**Positioning:** The paper is meaningfully above the 4.0 anchors (it has genuine methodological novelty, a solid primary-task ablation, and competitive quantitative results at 25 frames). It is above the 5.5 anchors (CMG, FreeTraj) in novelty of contribution — the auxiliary branch design is architecturally principled, and the ablation is cleaner. However, the missing quantitative video editing evaluation and single-baseline video control comparison prevent it from reaching the 6.75 level of 3DTrajMaster, which had thorough multi-task quantitative validation. The paper's core camera-control contribution would stand on its own; the video editing claim needs quantitative support to be credible.

**Originality:** Medium-high — the auxiliary branch idea adapted from ControlNet is not entirely novel in mechanism, but the specific application to trajectory-based attention and the functional divergence insight are new.

**Importance:** High — fine-grained camera motion control is an active and impactful problem.

**Claims vs. support:** Mostly supported for camera control on images; unsupported for video editing.

**Experimental soundness:** Moderate — primary task solid, secondary task thin.

**Clarity:** Good — algorithms, figures, and ablation are well-presented.

**Value to community:** Good — practical, efficient, and extensible approach.

## Score and Decision

Positioning between the 5.5-rejected and 6.75-accepted anchors, the paper is closer to borderline acceptance. The primary contribution (trajectory attention for image camera control) is technically solid and validated. The video editing gap is real but concerns a secondary extension. On balance, I score this at **5.5** — a marginal submission that requires quantitative video editing evaluation and more baselines for video camera control before it can be confidently accepted.

**MY FINAL SCORE: <pineapple>5.5</pineapple>**
**MY FINAL DECISION: <orange>Reject</orange>**