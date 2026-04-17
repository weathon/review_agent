Now let me look at calibration papers for scoring.Now let me read the OmniPhysGS paper's weaknesses and the Sync4D paper for more calibration context:Now I have sufficient context to write the final review. Let me compile everything.

---

## Summary

CP4D presents a three-stage compositional framework for physics-aware 4D scene generation. Given a text prompt, it (1) builds separate 3D Gaussian representations of a static background and foreground objects using pre-trained expert models, (2) produces physically grounded motions via heterogeneous physics simulators (MPM/rigid/PBD) refined by SDS-based optimization of material parameters and object displacements, and (3) fuses dynamic foregrounds with the static background via a depth-aware placement and optimization mechanism. Experiments on 17 curated prompts show quantitative improvements over video generators, physics-driven baselines, and text-to-4D methods across VBench, WorldScore, and GPT-4o evaluation.

---

## Strengths

- **Compelling compositional formulation.** Separating the 4D scene into a static 3D environment and physically governed dynamic foregrounds mirrors real-world scene structure. This is a principled, practical design that enables modular editing (shown in Fig. 6) and naturally handles the stylistic coherence problem that naive independent generation creates.

- **Hybrid motion synthesis is a genuinely useful idea.** Using physics simulators (MPM, rigid body, PBD) to anchor physically consistent trajectories and then using SDS from a video diffusion model to correct imprecise material parameters and collision artifacts is a sensible design that addresses known complementary weaknesses of both paradigms. The ablation in Fig. 5 visually validates both components.

- **Strong quantitative results across multiple metrics.** CP4D achieves the best or near-best scores on VBench (motion smoothness 0.998, consistency 0.972), WorldScore (photo consistency 97.42, 3D consistency 95.55), and GPT-4o physical realism (0.694 vs. best baseline 0.624), outperforming both commercial video generators (Sora, Runway, Wan) and dedicated physics-driven methods (PhysGen, PhysGen3D, OmniPhysGS).

- **Depth-aware automated composition is practical.** The frustum-based scale bounding heuristic (Eq. 8) and sequential scale-then-translation optimization (Eq. 9) is a reasonable, largely automatic solution to the coordinate-alignment ambiguity between independently reconstructed 3D assets.

---

## Weaknesses

### Fatal
*(None that fully invalidate the contribution, but the following major issues collectively undermine the strength of claims.)*

### Major

- **Evaluation set of only 17 self-curated examples is insufficient to support broad claims.** The paper explicitly states "We curate a dataset of 17 examples" (Sec. 5.1), and the composition of this set is entirely opaque: no category breakdown, no diversity analysis (rigid vs. deformable vs. fluid, simple vs. complex interactions), and no description of selection criteria. With n=17, the aggregate metrics in Tables 1–2 carry very large variance; a few outlier cases can shift rankings meaningfully. No error bars or per-category statistics are reported. Yet the Abstract and Conclusion repeatedly assert the method "significantly outperforms" and "consistently outperforms" baselines. This is the paper's most severe weakness—it is not a minor scaling request but a fundamental mismatch between claim scope and evidence scale. The entire comparative case rests on results that are statistically fragile.

- **Evaluation metrics do not measure physical correctness, yet physical fidelity is the central claim.** VBench and WorldScore measure perceptual quality, temporal coherence, and 3D consistency—not whether dynamics obey physics. The GPT-4o "physical realism" ratings in Table 2 are subjective per-video LLM judgments from visual appearance alone; they cannot reliably distinguish correct physics from plausible-looking motion, and they heavily conflate physical realism with photorealism. There is no evaluation using physics-specific measures (e.g., energy/momentum checks, trajectory comparison against analytical or simulator ground truth, contact/penetration metrics). The paper cites VideoPhy but does not use it. The method's core "faithful adherence to complex physical dynamics" claim is thus structurally unsupported by the evaluation: the metrics confirm visually good videos, not physically correct dynamics.

- **Ablation is insufficient to isolate the physics contribution.** The ablation in Sec. 5.3 only removes the material-parameter SDS optimization and the relative-position SDS optimization. It does not test: (a) physics simulator–only motion without SDS refinement; (b) SDS–only motion without any simulator; (c) the full pipeline without the compositional background/foreground separation. Without these ablations, the source of CP4D's quantitative gains cannot be attributed to the physics component specifically—improvements may come primarily from the 3D compositional architecture or the strong video prior, not from physics.

- **The SDS-based refinement of material parameters and global displacements creates tension with the "physically faithful" framing.** The paper explicitly notes that "VLMs often lack the numerical accuracy required" and that grid-based solvers produce "perceptually implausible outcomes" (Sec. 4.2). To fix these issues, SDS optimizes Θ (material parameters) and ΔΓ (global per-object displacements) to match what a video diffusion model finds plausible. This is an appearance-driven correction, not a physics-grounded one. The paper acknowledges global displacements are added to "alleviate inaccuracies… during inter-object interactions"—i.e., objects are shifted so the rendered video looks better, not so the collision is physically correct. There is no explicit constraint keeping the optimized parameters or trajectories near physically valid values. The method is more accurately characterized as "physics-initialized, appearance-refined generation" than "faithful physical dynamics," and the paper's language should reflect this.

### Minor

- **Pipeline complexity and error propagation are not analyzed.** CP4D chains at minimum 8 components: GPT-4o (prompt decomposition), Qwen-Image (background generation), Qwen-Image-Edit (composite synthesis), SAM (segmentation), Depth Anything (depth estimation), Trellis (foreground-to-3D), Viewcrafter (background-to-3D), VLMs (material estimation), physics solvers, and SDS optimization. Errors in early stages (bad segmentation, incorrect material inference, depth errors) will propagate. No failure cases are discussed and no robustness analysis is provided, making it unclear how often the pipeline produces usable results.

- **Static background assumption limits scope.** The formulation explicitly assumes backgrounds are immutable while only foregrounds are dynamic. Many physically interesting scenarios require dynamic backgrounds (flowing water, deformable terrain, contact with deformable environments). The paper briefly acknowledges this in Sec. 2.2 but does not quantify or discuss the limitation concretely.

- **Computational cost is not reported.** Given multi-stage inference with large pretrained models, heterogeneous physics solvers, and SDS optimization loops, the runtime and GPU requirements are critical for assessing practical applicability. These are entirely absent from the paper.

### Trivial

- The paper claims to compare against "all competing approaches" in the physics-aware 4D space, but the text-to-4D baseline set is thin (only DreamGaussian4D, an older method). TC4D and 4D-Fy are cited in related work but not compared against.

---

## Nice-to-Haves

- **Scale up the evaluation and show per-category statistics.** Even 50–100 diverse examples with clear category breakdown (rigid, elastic, fluid; single vs. multi-object; simple vs. complex) would make the comparative claims far more credible.

- **Add a physics-specific evaluation component.** Even a small study comparing simulator trajectories before vs. after SDS refinement, or checking energy/momentum conservation on simple controlled scenarios, would add rigor to the "physics-aware" claim.

- **Report novel-view rendering quality.** The claim that CP4D produces truly "explorable" 4D scenes (vs. good-looking single-view videos) would be strengthened by multi-view PSNR/SSIM/LPIPS or 4D consistency metrics across held-out viewpoints.

- **Validate the GPT-4o physical realism scores with a user study.** Even a small-scale human preference study (20 participants) would establish that GPT-4o's "physical realism" ratings correlate with human perception.

- **Report parameter values before and after SDS optimization** (e.g., Young's modulus, density) to show whether SDS refinement moves parameters toward or away from physically reasonable values.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic – "Hybrid motion synthesis is under-specified / effectively opaque" (beyond what is kept above):** The reviewer raises legitimate concerns but also includes reproducibility nitpicks (exact optimization schedules, learning rates, CFL conditions, time step sizes). These cross into implementation detail not standard to disclose in a systems paper of this type. The core concern—how Gaussians are mapped into and out of the simulator, and how solver coupling works for multi-object scenes—is a legitimate weakness retained in the Major section above. The reproducibility-specific sub-points about learning rates and stability conditions are removed.

- **Harsh Critic – "Comparison to baselines mixes fundamentally different capabilities":** The point about 2D video generators being included is valid as a minor concern (3D consistency metrics naturally favor a 3D-native method), but the harsh reviewer frames it as a fundamental methodological flaw. The paper's goal is precisely to produce superior 4D content versus ALL alternatives including video generators; such comparisons are standard in this field (OmniPhysGS does the same). This is not an unfair comparison against the author's method—the asymmetry is intentional. Removed per hard rules.

- **Harsh Critic – Background/foreground decomposition reproducibility (Sec. 4.1):** The sub-point about undisclosed segmentation thresholds and number of views used in reconstruction is a reproducibility nitpick. Removed per hard rules.

- **Spark – "Missing baselines Physics3D and Phys4DGen":** While these are cited, we cannot confirm the specific availability or usability of these methods as baselines without external sources. Per the "do not mention missing related works" rule, this is removed.

- **All reviewers – "Undisclosed hyperparameters for SDS":** Learning rates, number of optimization steps, batch sizes for SDS are trivial implementation details. Paper notes details are in Appendix B/C. Removed as reproducibility nitpick.

---

## Novel Insights

The most underappreciated tension in CP4D—and in the broader physics-aware generation literature—is that appearance-based objectives (SDS, GPT-4o ratings) and physics fidelity are not just complementary but are in structural opposition: a visually plausible motion need not be physically correct, and SDS will always push toward the diffusion prior's preference regardless of physical law. CP4D's design honestly acknowledges this (VLM parameters are inaccurate, grid approximations fail, so we apply visual correction), but then claims "faithful adherence to physical dynamics." The field needs evaluation methodology that can distinguish "looks physically plausible" from "is physically correct"—and CP4D's results, which are entirely the former, are being framed as evidence of the latter. Resolving this distinction is the key open problem for physics-aware 4D generation.

---

## Suggestions

1. **Expand evaluation to ≥50 examples** with clear provenance and category coverage; report per-category breakdowns to identify failure modes.
2. **Add an SDS-free control condition** to Table 1–2 ablation: simulator only, no SDS. This directly tests the physics contribution.
3. **Add at least one physics-grounded metric** (e.g., comparing final positions/trajectories against known-correct simulator output for simple scenes; or an energy conservation check) to ground the "physical fidelity" claim.
4. **Calibrate the GPT-4o evaluator** with a small human study, or replace with a validated physics realism benchmark (e.g., VideoPhy).
5. **Report total runtime per sample** and GPU memory requirements.
6. **Show at least 3 failure cases** and discuss the likely failure mode (e.g., bad segmentation, poor VLM material estimation, depth estimation error).

---

## Score and Decision

**Calibration anchors:**
- *OmniPhysGS* (accepted poster, scores 8,6,6,6,6 avg ~6.4): Most directly comparable—physics-aware 3D generation with SDS-guided material parameter optimization. More principled physics (12 constitutive sub-models), larger evaluation set, but narrower scope (single-object dynamics, no compositional full-scene). OmniPhysGS also received criticism for unconvincing metrics and weird result in collision demos.
- *Physics3D* (withdrawn/rejected, scores 8,5,5,1 avg ~4.75): Similar MPM+SDS architecture. Single-object only. No compositional formulation.
- *CompGS* (withdrawn/rejected, scores 5,5,5,6,6 avg ~5.4): Compositional text-to-3D with Gaussian splatting—no dynamics or physics.
- *Sync4D* (withdrawn/rejected, scores 3,5,5,5 avg ~4.5): Physics-based 4D generation, limited quantitative evaluation.

CP4D is more ambitious than any individual anchor—it combines full-scene compositional generation (à la CompGS), physics-driven simulation (à la Physics3D/OmniPhysGS), and 4D dynamic rendering. Its quantitative numbers are genuinely strong. However, the n=17 evaluation set is markedly smaller than what supported OmniPhysGS's acceptance, the "physical realism" claim is inadequately validated, and the ablation does not isolate the physics contribution. Placed below OmniPhysGS (which had a more principled physics mechanism and broader evaluation) but above Sync4D/Physics3D (which had narrower scope and equally or more severe validation weaknesses). The paper lands at a borderline-reject: the ideas are solid and the system works, but the evidence does not yet support the strength of claims being made.

**Originality:** Moderate-high — compositional + physics-aware 4D generation is a novel combination.
**Importance:** High — 4D scene generation with physical plausibility is an important open problem.
**Claim support:** Weak — overclaimed relative to 17-example evaluation.
**Experimental soundness:** Moderate — metrics are reasonable for visual quality but inadequate for the physics claim; ablation is insufficient.
**Writing clarity:** Good overall; motivation and pipeline description are clear.
**Value to the community:** Positive if claims are calibrated; the compositional framework is practically useful.

**Final Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>