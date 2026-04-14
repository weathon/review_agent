## Summary

I2VControl-Camera proposes a camera control method for image-to-video generation that replaces sparse extrinsic matrix representations with dense 2D point trajectories derived from 3D point clouds in the camera coordinate system. The method additionally introduces a scalar motion strength parameter to decouple and adjust subject dynamics independently of camera motion. The framework uses an adapter architecture atop MagicVideo-V2 and demonstrates improved camera control precision over retrained MotionCtrl and CameraCtrl baselines on both static (RealEstate10K) and dynamic (movable object) scenes.

---

## Strengths

- **Denser control signal with measurable gains:** The shift from extrinsic matrices / Plücker embeddings to dense 2D point trajectories (T_λ) is well-motivated: trajectories encode per-pixel camera displacement in image space, providing a spatially dense conditioning signal rather than sparse 3×4 matrix entries. This translates into concrete, substantial improvements: RotErr drops from 1.26 (CameraCtrl) to 0.53 and TransErr from 21.60 to 9.72 on RealEstate10K (Table 1), an improvement of roughly 2× and 2.2× respectively — not marginal gains.

- **Principled decoupling of camera and subject motion via a novel theoretical framework:** The decomposition F(p, λ) = R_λ · F(p,0) + t_λ + G(p, λ) separates "rigid linear" (camera) from "higher-order nonlinear" (subject) components. Using the spatial integral of ∂G/∂λ as a motion strength scalar is a theoretically grounded idea that fills a real gap: prior methods either suppress subject dynamics entirely (NVS-based methods) or conflate them with camera motion. The practical demonstration in Table 2 confirms this: Ours-0 achieves the lowest MSC (most static subjects) and Ours-600 achieves the highest (most animated subjects), while maintaining RotErr/TransErr superior to both baselines at all motion strength levels.

- **Iterative static/dynamic segmentation for noisy real-world data:** The RANSAC-style Algorithm 1, which iteratively refits R_λ, t_λ on the static region and removes outlier points, is a practical and sensible solution to the ill-posed problem of separating camera from subject motion without ground-truth segmentation masks. This allows the pipeline to train from arbitrary RGB videos — a non-trivial engineering contribution.

- **Base-model-agnostic adapter design:** The adapter inserts control tokens into the temporal self-attention layers without modifying the base model weights, making the method applicable to different video diffusion backbones. This is a practically significant design choice validated by the fact that retraining MotionCtrl and CameraCtrl within the same framework is feasible.

---

## Weaknesses

### Fatal
None.

### Major

- **No ablation studies — the primary omission of the paper.** The two central contributions — (i) point trajectory vs. extrinsic matrix, and (ii) motion strength module — are never evaluated in isolation. It is unknown how much of the RotErr/TransErr gain comes from the denser trajectory signal vs. better training data vs. the base model vs. the adapter design. Similarly, it is unclear whether the motion strength module affects camera precision or operates purely independently. Without at least a "trajectory only, no motion strength" and "motion strength only, extrinsic matrix" ablation, the relative contribution of each design choice cannot be assessed. This significantly weakens the paper's scientific claims.

- **Theory-practice gap in the core formulation.** Two inconsistencies reduce the rigor of the theoretical framework:
  1. *R_λ is not a rotation matrix.* Eq. (6) defines R_λ ≜ I + J_F(0,λ) − J_F(0,0), which is a generic 3×3 matrix with no orthogonality or det=1 constraint. Yet Eq. (12) subsequently solves for (R, t) that is presumably constrained to SE(3). The paper never explicitly closes this gap or explains how the theoretical R_λ relates to the practical fit.
  2. *Maclaurin locality.* The decomposition in Eq. (2) follows from a Maclaurin expansion around p = 0, which is a local result. The remainder o(p) is only guaranteed small near the origin. For points far from the camera origin — e.g., background in wide-angle shots — the approximation quality is uncharacterized. The paper does not discuss this limitation or provide empirical evidence that the decomposition holds well for real-scene point distributions.
  These issues do not invalidate the practical method (Eq. 12 is a sound practical fitting procedure regardless of the theoretical derivation), but they weaken the claim that the method is grounded in a principled theoretical framework.

- **Proprietary training data and base model limit reproducibility.** The 30K-clip training set is internal with no public release. The base model (MagicVideo-V2 internal version) is also proprietary. The baselines are retrained internally rather than evaluated against published checkpoints. While retraining on the same base model is the right approach for a fair comparison, the lack of any public artifact makes independent replication of the core results impossible. The paper does not report whether the retrained baselines are competitive with their published numbers, making it hard to verify that the retrained implementations are faithful.

- **Limited and unexplained baseline exclusions.** CamCo and Camtrol are cited in Section 2.3 as directly related camera control methods, yet neither appears in the quantitative evaluation. No justification is given. Camtrol in particular (training-free point-cloud rendering approach) has overlapping goals with this work, and its exclusion is conspicuous.

### Minor

- **Algorithm 1 hyperparameters never specified.** The values of ε (tolerable error), α (acceptable ratio), and N_max (maximum iterations) in Algorithm 1 are listed as inputs but never given in any experiment. Without these, the data pipeline cannot be reproduced and it is unclear how sensitive results are to these choices.

- **Motion strength scale is opaque to users.** The paper demonstrates motion strength values of {0, 200, 400, 600} but never explains the units, how these numbers relate to physical motion amplitude, or how a user should select a value for a desired effect. This undermines the "user-friendly" claim since the scalar has no semantic interpretation.

- **Ours-0 FID on movable-object dataset is worse than baselines.** Ours-0 achieves FID = 100.36 vs. MotionCtrl (98.54) and CameraCtrl (99.59) in Table 2. The paper's explanation ("forcing static motion is unnatural") is plausible, but this does represent a quality trade-off worth acknowledging more explicitly — especially since Ours-0 is the mode most analogous to prior methods.

- **Error propagation from external estimators uncharacterized.** The training pipeline depends on UniDepth (metric depth) and SpatialTracker (point tracking), and inference depends on UniDepth. Monocular depth errors directly corrupt 3D point trajectories and thus training signals. The paper does not analyze failure rates, noise levels, or their downstream effects on control quality — for instance, how reflective/transparent surfaces or textureless regions affect the resulting signals.

### Tiny

- **Motion strength characterization is binary.** Only two values (0 and 600) are shown in Figure 6. A plot of RotErr or FID as a continuous function of motion strength would better characterize the control curve and whether degradation is gradual or sharp.

- **Camera precision vs. motion strength trade-off.** Table 2 shows RotErr increasing monotonically from 0.76 (Ours-0) to 1.18 (Ours-600). This modest but consistent degradation — that higher subject dynamics slightly impair camera precision — is mentioned only implicitly and warrants a brief discussion.

---

## Nice-to-Haves

- **Intermediate motion strength values in visualizations.** Showing Figure 6 with 3–4 intermediate values (e.g., 0, 150, 300, 450, 600) would give users and reviewers a better sense of the control curve's linearity and monotonicity.

- **Comparison with true camera ground truth.** Where ground-truth camera poses are available (e.g., RealEstate10K metadata), testing against them instead of SfM-estimated poses from generated frames would remove potential circularity in the error metrics.

- **3D trajectory alignment visualization.** Showing the estimated 3D camera path of generated videos overlaid with the input control trajectory (rather than 2D pixel-level overlays in Figure 5) would provide stronger evidence of true 3D camera control.

- **Optical flow decomposition.** Separating estimated optical flow into camera-induced and subject-induced components would visually validate that m_λ specifically activates subject motion and not global scene drift.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Plücker embeddings add no information over the camera matrix" deserves more proof** *(Harsh Critic)*: While the paper's claim is stated tersely, its practical thrust — that both representations encode camera geometry and neither encodes appearance of the reference image — is defensible. The critic's concern about equivariance is a theoretical nuance that, while valid, is peripheral to the paper's practical contribution. Removed.

- **Contribution 2 (data pipeline) is "just an engineering detail"** *(Harsh Critic)*: The data pipeline that converts arbitrary RGB video into (T_λ, m_λ) training signals is a necessary and non-trivial component enabling training without ground-truth camera annotations. Labeling it as non-conceptual is overly dismissive. Removed.

- **No user study for "user-friendly" claim** *(Harsh Critic)*: User studies are not a standard expectation for an algorithmic camera control paper at this venue. The claim is qualitative; a user study would strengthen it but its absence does not constitute a weakness under community norms. Removed.

- **Video-level temporal metrics missing** *(Harsh Critic)*: While useful, temporal video metrics are not standard in the camera control sub-field, and FID on frames is the established evaluation protocol (used by CameraCtrl). Removed as a weakness; could be a nice-to-have.

- **Retrained baseline comparison is unfair** *(Implicit concern)*: Retraining MotionCtrl and CameraCtrl on the same base model, resolution, and frame count is precisely the correct approach to isolate the contribution of the control signal design. Any asymmetry here is deliberate and favors the baselines (same base model = no base model advantage). Removed.

- **MSC "↓↑" annotation is non-standard** *(Harsh Critic)*: The bidirectional annotation correctly reflects that MSC is used to measure *range* of adjustability — lower Ours-0 vs. higher Ours-600 vs. baselines proves the adjustable control claim. The annotation is intentional, not a metric design flaw. Removed.

- **No negative societal impact section** *(Harsh Critic)*: Formatting/venue-tag concern. Removed per instructions.

- **Long-term temporal consistency beyond 24 frames** *(Spark Finder)*: The model is explicitly designed for 24-frame generation. Evaluating beyond design scope is scope creep. Removed.

---

## Novel Insights

The most interesting technical insight in this paper is the use of the static/dynamic scene decomposition (Ω_S ⊔ Ω_D) to *jointly* solve for camera pose (R_λ, t_λ) and subject motion (m_λ) from raw RGB video without any camera annotation — essentially a RANSAC-style monocular camera reconstruction that bootstraps a training signal from unconstrained internet video. The implicit consequence, visible in Table 2, is that a model trained this way can then be *controlled at inference* via two orthogonal knobs (trajectory and scalar strength) that were never independently annotated during training. This self-supervised factorization of camera vs. subject dynamics through iterative linear fitting is a practically valuable idea whose robustness (w.r.t. depth noise, tracking failures, and scene diversity) remains the key open question the paper leaves unanswered.

---

## Suggestions

1. **Add minimal ablations.** At minimum: (a) replace T_λ with the extrinsic matrix (R_λ, t_λ) while keeping the same adapter and motion strength module, (b) remove the motion strength input while keeping T_λ. These two ablations would directly validate the two core claims and are likely feasible within the existing training infrastructure.

2. **Specify Algorithm 1 hyperparameters.** Report ε, α, N_max used in experiments, and include a brief sensitivity analysis (e.g., how ε affects the fraction of points assigned to Ω_S and the quality of the resulting R_λ, t_λ fit).

3. **Clarify the R_λ theory-practice relationship.** Add one paragraph explicitly noting that R_λ in Eq. (6) is a locally-valid linear approximation, while the practical estimation in Eq. (12) fits a proper SE(3) transformation. Acknowledge that Eq. (12) is the operative definition used throughout and that the theoretical motivation in Eq. (2)–(6) provides intuition rather than a strict guarantee.

4. **Provide motion strength semantics.** Report what physical quantity m_λ = 600 corresponds to (e.g., average pixel displacement per frame in training data), or provide a histogram of m_λ values across the training set so users can contextualize their input.

5. **Add a qualitative comparison with Camtrol** (or explain clearly why it is excluded). Even if quantitative comparison is impractical, a visual comparison on matched examples would address the concern about omitting a directly related method.

---

**Overall assessment:** The paper's practical contribution — denser trajectory control and decoupled motion strength — is genuine and measurably effective. The RotErr/TransErr improvements on static scenes are convincing. However, the complete absence of ablation experiments is a significant methodological gap that prevents attribution of gains to specific design choices, and the theory-practice inconsistencies in the mathematical framing reduce the paper's scientific depth. Reproducibility is substantially limited by proprietary data and base model. As it stands, the paper reads as a competent and well-motivated engineering contribution with promising results, but falls short of the level of rigorous validation expected at ICLR. The path to a stronger paper is clear: ablations, hyperparameter disclosure, and theoretical clarifications.