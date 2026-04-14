## Summary
I2VControl-Camera proposes a camera control method for image-to-video generation that replaces sparse extrinsic-matrix representations with dense 2D point trajectories (lifted from monocular depth) as the primary control signal, and introduces a per-frame scalar "motion strength" to decouple and adjust subject motion amplitude. An adapter-based architecture trained on 30K proprietary video clips is shown to outperform retrained MotionCtrl and CameraCtrl baselines on both static (RealEstate10K) and dynamic-scene benchmarks.

---

## Strengths

- **Dense trajectory control signal with clear empirical grounding.** Replacing a 12-dimensional extrinsic matrix with a spatially dense $(T,2,H,W)$ point trajectory provides substantially stronger geometric grounding per pixel. The improvement in RotErr (2.66 → 0.53 on RealEstate10K) and TransErr (12.70 → 9.72) is substantial—roughly 5× better rotation adherence—and is a believable consequence of the architectural choice, not just a data effect (since both methods are evaluated on the same static scene test set).

- **Practically novel decoupling of camera vs. subject motion.** The explicit separation of the rigid-body (camera) component from the higher-order residual (subject dynamics), formalized via Eqs. 7–9 and operationalized via Algorithm 1, is a coherent design that addresses a genuine pain point ignored by prior work: existing methods either suppress subject motion (CamCo) or conflate it with camera motion.

- **Adapter architecture with base-model agnosticism.** The adapter design (conv layers → token concatenation → adaptive self-attention) genuinely freezes the backbone, making migration to new base models straightforward. This is a practical virtue in a rapidly evolving landscape where base models are replaced frequently.

- **Motion strength adjustability demonstrated qualitatively.** Figure 6 provides a convincing within-paper demonstration that the single scalar does modulate subject motion amplitude (polar bear, astronaut, wolf all transition from static to active as $m_\lambda$ increases from 0 to 600), directly supporting the paper's core claim about the motion strength control axis.

---

## Weaknesses

### Fatal
None that would invalidate the approach outright. The core architectural and control-signal design is sound and the claimed improvements on camera precision metrics are real.

### Major

- **No ablation studies.** This is the most serious experimental shortcoming. The paper introduces three distinct innovations: (1) dense point-trajectory control signal vs. extrinsic matrix, (2) the motion strength scalar, and (3) a richer proprietary training set. There is no experiment that isolates any of these. It is entirely unknown whether replacing the sparse extrinsic matrix with the dense trajectory alone (without motion strength or new data) explains the gains, or whether training on 30K diverse clips would suffice with the original MotionCtrl control signal. For an ICLR paper making architectural claims, ablations are necessary.

- **Training data confound for dynamic scene results.** On the movable-object evaluation (Table 2), baselines are trained on RealEstate10K (nearly static scenes), while the proposed method trains on 30K proprietary clips containing "natural motion." Any model trained on dynamic data would likely score higher on MSC and FID for dynamic scenes. The improvements in Table 2 may be largely due to richer training data rather than the architectural choice. Training baselines on the same 30K clips with their original control signals would disentangle this. (Note: for the static RealEstate10K benchmark in Table 1, this confound is much less acute since the improvements in RotErr/TransErr are evaluated on static scenes where the richer training data matters less.)

- **Mathematical formalism in Sec. 3.1 conflates a local approximation with a global claim.** The Maclaurin expansion in Eqs. 3–7 is valid only locally (as $\mathbf{p} \to 0$), so the $o(\mathbf{p})$ term is an infinitesimal only near the origin. For scene points far from the camera origin the residual $\mathcal{G}(\mathbf{p}, \lambda)$ is not small, and the Taylor-based "proof" of Eq. 2 does not hold globally. What the paper actually needs—and what Eq. 10 states correctly—is the physically intuitive fact that static scene points undergo *exact* rigid-body motion. This statement does not require the Maclaurin apparatus and could replace it. The current derivation introduces unnecessary confusion without adding rigor.

- **Hyperparameters of Algorithm 1 ($\epsilon$, $\alpha$) are never specified.** These parameters control the static/dynamic partition on which the entire data pipeline and control signal depends. Their values are not reported in the paper or appendix (as extracted). No sensitivity analysis is provided. This is a reproducibility concern for one of the paper's claimed contributions (the data pipeline).

### Minor

- **FID reference set too small.** FID is computed over 2,000 reference frames from WebVid, far below the 50K-frame standard. The FID differences in Tables 1–2 (e.g., 164.62 vs. 155.01, a ~6% difference) are likely within estimation noise at this sample size. Claims based on FID ordering should be treated cautiously.

- **Inference-time motion strength has no calibration guidance.** The paper uses values {0, 200, 400, 600} without explaining what these numbers represent physically, what their training distribution looks like, or how a user should choose a value for a desired effect. Does the relationship between the scalar and perceived motion amplitude hold across diverse scenes?

- **$\mathbf{R}_\lambda$ is defined as an arbitrary matrix but named as a rotation.** In Eq. 6, $\mathbf{R}_\lambda \triangleq \mathbf{I} + \mathbf{J}_\mathcal{F}(\mathbf{0},\lambda) - \mathbf{J}_\mathcal{F}(\mathbf{0},0)$ is not constrained to $SO(3)$. Calling it $\mathbf{R}$ while later optimizing a rotation in Eq. 12 creates notational ambiguity. The paper should distinguish the theoretical $\mathbf{R}_\lambda$ from the practical camera rotation.

- **Qualitative comparison is sparse.** Only two comparison examples are shown in Figure 7 (one static bedroom, one outdoor mountain). There are no qualitative comparisons for dynamic scenes alongside the subject-motion experiments, making it hard to assess whether baseline methods at equivalent motion strength produce comparable dynamics.

### Tiny

- The third contribution bullet ("our method outperforms previous methods") states a result, not a contribution. This is a minor rhetorical imprecision.
- The claim in the abstract about "precise pixel-level control" applies strictly to static background elements; the cat/fox/bear subjects are not controlled at the pixel level, only their aggregate energy is.

---

## Nice-to-Haves

- **User study on control intent alignment.** Automated metrics (RotErr, FID) do not capture whether the user's intended camera path is perceived as correct. A human evaluation comparing alignment with intent would meaningfully support the "user-friendly" framing.
- **Analysis of motion strength generalization.** A plot showing average optical flow (MSC) as a function of the input scalar across scenes would validate that the conditioning is monotone and approximately scene-invariant, and reveal whether saturation or discontinuity occurs beyond the tested range.
- **Robustness to depth estimation failure.** A brief evaluation on scenes where UniDepth is expected to struggle (reflective surfaces, textureless backgrounds) would bound practical failure modes.
- **Comparison with CamCo on the static RealEstate10K benchmark.** CamCo is deliberately excluded from the dynamic-scene comparison (justified because it induces small motion), but it is a relevant comparison on the static evaluation where camera precision is the focus.
- **Continuous strength interpolation visualization.** A figure showing a finer grid (e.g., 0–600 in steps of 100) on a fixed seed would confirm smooth, continuous control rather than discrete mode-switching.
- **Cross-base-model test.** Demonstrating the adapter works on a second base model (e.g., SVD) would validate the claimed model-agnostic architecture beyond a single experimental configuration.

---

## Removed Points
*These points were flagged for removal — treat them with caution.*

- **Critique of Plücker embedding as "offering no additional information."** The harsh reviewer calls this "unsubstantiated." The paper's claim is a motivational statement comparing the information content of Plücker coordinates vs. the raw rotation/translation matrix. While mathematically imprecise, it is a reasonable intuitive argument (both parameterize the same extrinsic DOFs without per-pixel grounding from the input image). Removing as a weakness because (a) it is a motivational framing, not a load-bearing proof, and (b) the paper's point trajectory construction addresses the actual problem regardless.

- **Claim that Camtrol should be a baseline.** Camtrol is training-free and operates in a video-to-video manner; it occupies a fundamentally different practical niche (no generative prior, relies on rendered point clouds directly). Excluding it from a training-based adapter comparison is defensible. Flagged as removed since comparing training-based and training-free methods conflates different problem settings.

- **Claim that CamCo should be in Table 2 (dynamic scenes).** The paper explicitly notes that CamCo "causes small motion dynamic" by design (via epipolar attention). Excluding a method from a benchmark precisely because its design philosophy suppresses the very capability being evaluated is not a flaw — if anything it favors the baseline in a controlled comparison. Removed as a criticism.

- **Statistical significance testing for RotErr/TransErr.** RotErr differences of 2.66 → 0.53 (5×) and 2.10 → 0.76 (nearly 3×) are substantive enough that formal confidence intervals are not needed to establish the claim. Requesting confidence intervals here is a non-standard expectation for this field.

- **Demand for theoretical proofs of convergence for Algorithm 1.** Algorithm 1 is a practical RANSAC-like iterative fitting procedure for data pipeline construction. Convergence guarantees for such heuristic segmentation algorithms are not standard expectations in the video generation literature. Moved out of weaknesses.

- **Criticism about the paper's scope excluding individual subject trajectory control.** The paper explicitly states the limitation: "we cannot control the motion of every individual point, so we instead resort to a secondary strategy." Criticizing the absence of per-object trajectory control is scope creep given this stated scoping. Removed as a weakness (acknowledged as nice-to-have direction).

---

## Novel Insights

The most genuinely insightful observation across the three reviews—which the paper itself does not emphasize sufficiently—is the **training data confound as a methodological stress test for the field**. The paper's decision to collect diverse proprietary data (because RealEstate10K is "nearly static") implicitly acknowledges that training data distribution is at least as important as the control signal design for dynamic video generation. This raises a broader point: evaluation protocols that retrain baselines on the same data but with different control signals would be a valuable community standard to establish clean architectural comparisons, and the paper is in a unique position to provide this but does not. A secondary insight from the spark finder is the suggestion to **overlay commanded vs. generated optical flow** — this would directly expose whether the point trajectory is actually followed at the pixel level, or merely correlated with it through shared training distribution, which would meaningfully distinguish the paper's claim of "precise pixel-level control" from a weaker "approximate directional control."

---

## Suggestions

1. **Add ablations**: At minimum (a) train on the same 30K clips using MotionCtrl's original extrinsic-matrix signal to isolate the data effect from the control-signal effect; (b) train with point trajectories but without the motion strength scalar to measure its independent contribution; and (c) test removing Algorithm 1's iterative fitting (e.g., using simple optical flow–based motion masks) to assess the data pipeline's contribution.

2. **Report Algorithm 1 hyperparameters** ($\epsilon$, $\alpha$, $N_\text{max}$) and provide a brief sensitivity analysis (e.g., how the static region fraction changes across a grid of $\epsilon$ values).

3. **Reframe or replace Sec. 3.1.** Either (a) replace the Taylor expansion argument with the simpler and correct statement that static scene points follow exact rigid-body motion (Eq. 10 already says this) and reserve the Maclaurin framing for motivation only, or (b) add a footnote clarifying that the approximation is local and that the global version holds exactly only for $\Omega_S$.

4. **Provide a motion strength calibration figure**: Plot mean MSC as a function of the input scalar $m_\lambda$ averaged over a held-out set of scenes. This validates monotonicity, approximate scene-invariance, and the absence of saturation at the values used in experiments.

5. **Expand qualitative comparison**: Add at least one dynamic-scene comparison in Figure 7 where MotionCtrl and CameraCtrl are also shown under comparable motion strength conditions, to allow readers to assess subject dynamics quality across methods.

---

**Evaluation axes:**
- **Novelty:** Moderate-to-high. The specific formulation of dense projected point trajectories as camera control signals and the explicit scalar motion strength conditioning are meaningful advances over sparse pose-matrix approaches, even if individual components (adapters, depth lifting, RANSAC-like segmentation) are borrowed from existing work.
- **Technical soundness:** Moderate. The operational method is sensible and the algorithm is well-described, but the theoretical formalism (Sec. 3.1) has a genuine logical gap (local-to-global misapplication) and a notational inconsistency. The data pipeline has undisclosed hyperparameters.
- **Empirical support:** Below expectations for ICLR. Substantial improvements on RotErr/TransErr are credible and meaningful. However, the absence of any ablation study, the training data confound for dynamic-scene results, and the small FID reference set collectively leave the attribution of gains unsubstantiated.
- **Significance:** Moderate. Camera control in I2V generation is practically important, and the decoupling of camera from subject motion is a genuine open problem. The adapter approach is deployment-friendly.
- **Clarity:** Good overall. The method sections are readable; the formalism is sometimes overwrought relative to the actual operational algorithm, but Figure 2 and Figure 4 are informative.

MY FINAL SCORE: <pineapple>5.2</pineapple>