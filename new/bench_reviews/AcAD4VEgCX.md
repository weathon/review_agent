Now let me search for calibration papers to anchor the score.Now I have enough context to write the consolidated review. Let me synthesize everything.

---

## Summary

I2VControl-Camera proposes a camera control method for image-to-video (I2V) generation that uses dense 2D point trajectories in the camera coordinate system (rather than sparse extrinsic matrices) as the control signal, and introduces a scalar "motion strength" parameter derived from the higher-order residuals of a trajectory decomposition to enable user-adjustable subject dynamics. A practical data pipeline is built from ordinary RGB videos using metric depth estimation (Unidepth) and point tracking (SpatialTracker), and the method uses an adapter architecture on top of MagicVideo-V2. Experiments on RealEstate10K and a movable-object dataset show substantially lower rotation and translation errors than retrained MotionCtrl and CameraCtrl baselines, while the motion strength scalar demonstrably varies scene dynamics.

---

## Strengths

- **Genuine novel contribution: motion strength control.** The ability to adjust the global dynamics of subject motion via a scalar input—decoupled from the camera trajectory—is a real capability absent from MotionCtrl, CameraCtrl, and related prior art. The visual results in Fig. 6 convincingly show that the same camera path produces either a frozen scene (m=0) or vivid subject animation (m=600). The monotonic behavior of MSC across m ∈ {0, 200, 400, 600} in Table 2 provides quantitative corroboration.

- **Strong camera control precision.** The RotErr / TransErr numbers on RealEstate10K (0.53 / 9.72 vs. 1.26 / 21.60 and 2.66 / 12.70 for the two baselines) are large, consistent improvements, corroborated by the qualitative pixel-level alignment shown in Figs. 5 and 7.

- **Practical data pipeline without 3D-annotated data.** The iterative static/dynamic partition algorithm (Alg. 1) combined with off-the-shelf depth and tracking estimators to construct training signals from raw RGB video is a useful engineering contribution that avoids dependence on specialized 3D datasets.

- **Adapter-based design is pragmatic.** Freezing the base model and training only the control adapter makes the method lightweight and potentially composable with future backbone updates.

- **Clear presentation.** The motivation, notation, and method are explained in a structured and accessible way, and the project page provides additional video evidence.

---

## Weaknesses

### Fatal
*None that unambiguously invalidate the core claims.*

### Major

- **The matrix R_λ in Eq. (6) is not a rotation matrix—the theoretical framing is misleading.** Eq. (6) defines R_λ ≜ I + J_F(0,λ) − J_F(0,0), which is a generic 3×3 linear map. The paper denotes it R_λ, implies it is a rigid rotation, and builds the conceptual story that the linear term captures "camera motion" while the residual G captures "subject dynamics." This decomposition is not mathematically justified: the derived matrix need not be orthogonal or have unit determinant, and the residual is just a Taylor remainder—not an intrinsic measure of scene dynamics. Perspective effects, depth variation, and imperfect linearisation over the observed region all contribute to this residual. Crucially, the practical algorithm (Eq. 12, Alg. 1) correctly optimises for a proper rigid transform via L-BFGS, which is independent of the theoretical derivation and is where the actual computation takes place. So the method may still work, but the theoretical narrative in Sec. 3.1 is imprecise and oversells the mathematical justification for the camera/subject decomposition. This should be corrected or clearly qualified.

- **No ablation studies.** The paper introduces three simultaneous changes relative to baselines: (i) point-trajectory conditioning vs. extrinsic/Plücker, (ii) static/dynamic partitioning of training data, and (iii) the motion strength scalar. There is no experiment that isolates any single component. Without ablations, it is impossible to determine whether gains come from the denser conditioning signal, a richer training dataset, or the proposed decomposition. This is the most significant gap for assessing the actual contribution of each design choice.

- **Limited baseline scope.** Only MotionCtrl and CameraCtrl are compared quantitatively. CamCo—explicitly discussed in the related work as an approach that "keeps 3D-consistent well but causes small motion dynamic" and is directly relevant to the paper's claimed benefit of simultaneous precision + dynamics—is absent from the evaluation. Camtrol (training-free) is similarly missing. This limits how strongly the "outperforms previous methods" claim can be interpreted.

### Minor

- **Algorithm 1 hyperparameters not reported.** The tolerable error ε, acceptable ratio α, and maximum iterations N_max are introduced but never specified for the actual experiments. No sensitivity analysis is provided, leaving the reproducibility and robustness of the static/dynamic partitioning unclear.

- **MSC metric conflates camera and subject motion.** MSC uses RAFT optical flow followed by 2D rigid alignment to "remove camera motion," but this alignment is itself approximate in scenes with parallax, articulated motion, or generation artifacts. The metric therefore measures global residual flow, not isolated subject dynamics. The "↓ ↑" annotation in Table 2 reflects this ambiguity: the paper interprets both smaller and larger values as favorable depending on the setting. A more carefully designed subject-motion metric would strengthen the motion-strength claim.

- **Scalar motion strength cannot express spatial variation.** A single global scalar m_λ cannot differentiate objects that should move differently (e.g., a walking foreground figure against a dynamic foliage background). This limits practical expressiveness compared to per-region or per-object controls.

- **Error propagation of the data pipeline not analyzed.** The entire training signal depends on Unidepth and SpatialTracker estimates. Failure modes (reflective surfaces, fast motion, thin structures) are neither discussed nor shown. Since supervision is derived entirely from these estimated signals, this matters for understanding where the method breaks down.

### Trivial

- The claim in the introduction that CameraCtrl's Plücker embedding "does not offer any additional information compared to the camera matrix used in MotionCtrl" is stated but not substantiated; it is a debatable assertion about information content rather than an established fact.

---

## Nice-to-Haves

- A sweep of intermediate motion strength values (e.g., m ∈ {0, 100, 200, 300, 400, 500, 600}) on the same scene would demonstrate whether the control is continuous and well-behaved, rather than just monotonic at a few discrete points.
- Testing the adapter on a second base model (e.g., SVD or AnimateDiff) would substantiate the "model-agnostic" claim, which currently rests on a single base model.
- Intermediate pipeline visualizations (depth maps, tracking results, static/dynamic masks) would help readers assess whether the data construction is producing reasonable signals in practice.
- A failure-case analysis would give an honest picture of the method's limitations (e.g., monocular depth failures, fast object motion, extreme camera angles).

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

1. **"Inference-time depth estimation failure modes undermine the method"** (Harsh Critic, Sec. 3.2): The critic raises valid practical concerns about monocular depth quality at inference, but this is a known limitation of depth-based methods broadly. The paper cites Unidepth specifically and the concern is partially covered under the retained "error propagation" minor weakness. Removed as a standalone structural criticism because it is not unique to this paper's design and does not undermine the core contribution.

2. **"The adapter is not truly model-agnostic"** (Harsh Critic, Sec. 3.4): The adapter inserts into temporal self-attention and assumes tokenized features, which is not universal across all architectures. However, the paper does not claim full universality—it says "independent of the base model structure" in the sense of not requiring changes to the backbone. This is a scope quibble rather than a factual error.

3. **"All other training strategies remain unchanged is too vague for reproducibility"** (Harsh Critic): Removed per the hard rule on trivial implementation details. The paper specifies the base model, GPU count, batch size, step count, and training duration, which is adequate for this type of systems paper.

4. **"FID computed against WebVid is arbitrary"** (Harsh Critic): Valid but minor; FID choice is standard practice for the field and does not invalidate the comparisons. Removed as a standalone concern since it does not materially affect any claim.

5. **"Preview agreement only shows consistency with the conditioning representation"** (Harsh Critic, Sec. 4.2): Technically correct but unavoidable—any camera-control evaluation must condition on some reference. The point is too obvious to constitute a weakness.

6. **Reproducibility concerns about the proprietary 30K training dataset** (Neutral, Spark): This is a real concern for the community, but per the hard rule on large artifacts impractical to include in a submission, the unavailability of the training data is not a paper flaw. The composition of the dataset (diverse camera motion + natural object motion, vs. static-scene-only RealEstate10K) is described at a conceptual level. Removed as a standalone reproducibility criticism; retained as context for interpreting the baseline comparison.

---

## Novel Insights

The most genuinely novel insight in this work is the practical realization that a single global scalar—obtained by aggregating the residual-velocity norm over the image domain—is sufficient to shift a generative video model between "static subject" and "animate subject" behavior under a fixed camera path. This is a surprisingly coarse-grained signal that nonetheless appears to have real leverage over the model's dynamics generation. It suggests that modern video diffusion models may be more sensitive to low-dimensional motion-strength cues than previously appreciated, and that the distribution of training videos along a motion-intensity axis may be a key factor in subject-dynamics control. Whether this scalar is truly capturing subject motion or proxy-capturing training-data distribution effects (e.g., whether m=600 correlates with high-motion training clips) is an open and interesting question that the paper does not fully resolve, but is worth pursuing.

---

## Suggestions

1. **Fix the theoretical framing of R_λ.** Either prove it is a rotation (by adding an orthogonality constraint and appropriate conditions on F) or clearly state it is a generic linear map used for first-order approximation, and reframe the "camera vs. subject" decomposition as a practically motivated heuristic rather than a mathematical theorem.

2. **Add ablations**: at minimum, (a) point trajectory vs. Plücker embedding under identical data/settings; (b) with vs. without motion strength conditioning; (c) with vs. without the static/dynamic partition. These three experiments would substantially clarify which component drives the gains.

3. **Report Algorithm 1 hyperparameters** (ε, α, N_max) and provide at least a brief sensitivity check (e.g., grid over α) to show the partition algorithm is robust.

4. **Include CamCo** in the quantitative evaluation or explicitly justify its exclusion with a clear reasoning (e.g., architectural incompatibility with retraining), since it is the most directly comparable method for the paper's target capability.

---

## Score and Decision

**Calibration anchors:**

| Paper | Decision | Avg Score | Key similarities |
|---|---|---|---|
| CameraCtrl | Accept (Poster) | ~6.5 | Adapter-based camera control for video diffusion, comparable scope |
| VD3D | Accept (Poster) | ~6.2 | Camera control adapter, strong empirical results, limited theoretical novelty |
| ReMoCo | Reject | ~5.0 | Point-trajectory motion control, region-wise dynamics, missing key comparisons |
| MotionFlow | Reject | ~4.0 | Camera + object motion integration, unclear novelty, weaker results |

This paper sits between ReMoCo/MotionFlow (rejected) and CameraCtrl/VD3D (accepted). It is stronger than ReMoCo in empirical results and contribution clarity, and introduces a genuinely novel motion-strength mechanism absent from CameraCtrl/VD3D. However, unlike CameraCtrl (which includes dataset ablations) and VD3D (which includes thorough ablations), this paper has **zero ablation studies**, uses a proprietary training set, and has a theoretical framework with a demonstrable flaw in its core notation. These gaps push it below the acceptance threshold of comparable works in this space.

**Assessment on key axes:**
- *Originality*: Moderate-to-good. The motion-strength idea is genuinely new; the point-trajectory conditioning is incremental over CameraCtrl/MotionCtrl.
- *Importance*: High. Camera control + dynamics in I2V generation is a real practical problem.
- *Claims supported by experiments*: Partially. Camera precision claims are well-supported; motion-strength claims need ablation to be fully credible.
- *Soundness of experiments*: Weak. No ablations; two baselines; MSC is a coarse metric.
- *Clarity*: Good. Well-written, but the theory section misleads through notation.
- *Value to community*: Moderate. The motion-strength idea is worth knowing; the missing ablations reduce trust in the pipeline design.

**Final Score: 5.0** (borderline reject). The paper makes a real contribution with the motion-strength concept and achieves strong camera precision numbers, but the absent ablations prevent attribution of the gains to the proposed design choices, the theoretical foundation overstates its rigor, and the baseline comparison is too narrow for the broad claims made.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>