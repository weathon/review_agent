Now let me look at what the paper says about evaluation, particularly regarding 3D consistency and multi-view evaluation:

The paper text says in Section 5.1: "WorldScore (Duan et al., 2025) is employed to measure photo consistency, 3D consistency, and motion smoothness."

But the 3D consistency metric in WorldScore likely requires multi-view rendering. Let me check — actually, looking at Table 1, the paper reports "3D Consist" scores for various methods including ones that don't produce 3D content. For video-only methods like Sora and Runway, how can they have a 3D consistency score? WorldScore is a benchmark designed for world generation that can evaluate videos rendered from generated scenes. But for video-only baselines, the paper must be rendering a single video from a single viewpoint and then evaluating 3D consistency from that, which is different from evaluating true multi-view 3D consistency.

Looking more carefully at Table 1, Sora gets 64.26 3D consistency, and CogVideoX gets 81.90 — video-only methods are getting evaluated on 3D consistency. This suggests the metric is applied to single-view videos and doesn't actually test multi-view consistency. The paper claims "explorable" and "3D consistency" but evaluates everything via single rendered videos.

Let me now synthesize the final review.

## Summary

CP4D proposes a compositional framework for photorealistic 4D scene generation that decomposes scenes into static backgrounds and physically dynamic foregrounds. The method follows a three-stage pipeline: (1) generating stylistically coherent 3D representations of background and foreground via conditioned image editing and pre-trained 3D reconstruction models, (2) producing physically grounded trajectories through a hybrid strategy combining physics simulators (MPM, rigid-body, PBD) with video diffusion model refinement via SDS, and (3) fusing foreground and background through an automated depth-based composition mechanism. The paper reports superior performance over video generation and physics-driven baselines on 17 curated examples across VBench, WorldScore, and GPT-4o evaluation metrics.

## Strengths

- **Well-motivated compositional formulation.** The decomposition of 4D scenes into static backgrounds and physically grounded dynamic foregrounds aligns with real-world scene structure and naturally enables controllable editing (demonstrated in Section 5.4 with zero-shot replacement of objects, backgrounds, and motions). This is a practical and intuitive design choice.

- **Hybrid motion synthesis is a sound technical idea.** Combining physics simulators for coarse, physically constrained trajectories with video diffusion priors for perceptual refinement addresses complementary weaknesses — physics simulators produce physically consistent but visually coarse motions, while video diffusion provides commonsense interaction priors. The ablation in Figure 5 demonstrates that removing either the material parameter optimization or the displacement optimization degrades results, validating both components.

- **Stylistic coherence strategy for 3D generation.** Generating the foreground image conditioned on the background (via F_edit) before segmentation is a simple but effective approach to avoid the "realistic background + cartoon foreground" style mismatch that would arise from independent text-to-3D generation.

- **Automated depth-aware composition.** The frustum-based scale initialization (Eq. 8) and sequential scale-then-position optimization (Eq. 9) provide a principled approach to placing independently generated 3D assets in a shared coordinate space, addressing a real engineering challenge in compositional 3D generation.

## Weaknesses

### Major:

- **Evaluation scale and claim strength mismatch.** The entire evaluation uses only 17 author-curated examples, with no description of selection criteria, diversity coverage, or failure rates. Claims of "significantly outperforming existing methods" and "consistent outperformance" carry strong statistical implications that cannot be supported by a 17-example test set without confidence intervals or variance reporting. The small scale also raises cherry-picking concerns that are not mitigated by showing only two qualitative examples (Fig. 4) and no failure cases.

- **Evaluation protocol misaligned with 4D claims.** The paper repeatedly claims to generate "explorable and interactive 4D scenes" with "3D consistency" (Abstract, Intro, Sec. 5.1), but all evaluation is conducted on single-view rendered videos. WorldScore's "3D consistency" metric is applied uniformly to all methods including 2D video generators (Sora, Runway), meaning it does not actually test multi-view 3D consistency of the generated scene. No experiment demonstrates rendering from novel viewpoints, probing the core advantage of having a 4D volumetric representation over simply producing a video.

- **"Physics-aware" and "faithful adherence to complex physical dynamics" claims are overstated relative to evidence.** No quantitative ground-truth physical validation is provided — no comparison against known physics trajectories, no energy/momentum conservation checks, no calibrated physical accuracy metrics. The physical realism assessment relies entirely on GPT-4o subjective ratings and perceptual metrics. SDS-based refinement (Eqs. 4–5) optimizes for text-conditioned visual realism as judged by a video diffusion model, not for adherence to physical laws; visually pleasing but physically incorrect trajectories (e.g., exaggerated collisions, unrealistic restitution coefficients) would be rewarded by this objective. This is not to say the method doesn't produce physically plausible results — it likely does — but the language of "faithful adherence" goes beyond what perceptual metrics can establish.

- **GPT-4o serves as both pipeline component and evaluator, creating potential circularity.** GPT-4o/Qwen is used for prompt decomposition and physical parameter inference, and then GPT-4o is the sole evaluator for physical realism, photorealism, and semantic alignment (Table 2). While different specific models (GPT-4o for evaluation, Qwen for generation) are used, both are large language models with correlated biases, and the same class of model generating and judging the content undermines the independence of evaluation.

### Minor:

- **Pipeline complexity without failure analysis.** The system chains together 6+ pre-trained models (LLM for decomposition, text-to-image, image editing, segmentation, depth estimation, two 3D reconstruction models, physics solvers, video diffusion). Each can fail independently, and error propagation is not analyzed. The paper does not discuss failure modes or what types of prompts or scenes the system cannot handle (e.g., scenes where the background should itself be dynamic, thin deformable objects, complex fluid-structure interactions).

- **Ablation depth is limited.** The ablation study (Fig. 5) shows only qualitative on/off comparisons for material optimization and displacement optimization. There is no ablation of Stage I (style-coherent generation vs. independent text-to-3D), Stage III (composition mechanism vs. simpler placement), or disentangling the contribution of physics simulation from SDS refinement.

- **Limited baseline comparison for core 4D claims.** DreamGaussian4D is the only true 4D generation baseline, and it is a text-to-4D object method, not a compositional scene generator. The other baselines (Sora, Runway, CogVideoX, Wan) are 2D video generators structurally incapable of producing explorable 3D scenes. Comparing a 4D scene generator primarily against 2D video methods on video quality metrics is informative but does not test the claimed advantage of producing explorable 4D content.

### Trivial:

- The conclusion restates strong claims without acknowledging limitations or qualifications.

## Nice-to-Haves

- **Novel-view evaluation.** Rendering from 3–4 substantially different viewpoints and measuring consistency (PSNR/SSIM/LPIPS) would directly substantiate the "explorable 4D scene" claim.

- **Larger, more diverse evaluation set.** Expanding to 50+ examples covering different material types (rigid, elastic, fluid), interaction patterns (collision, stacking, flowing), and failure cases would significantly strengthen confidence in the results.

- **Human evaluation study.** A small-scale human study comparing physical plausibility across methods would ground the GPT-4o scores and address circularity concerns.

- **Computational cost analysis.** The pipeline involves iterative SDS optimization alongside multiple pre-trained models; reporting wall-clock time and GPU costs would help assess practical applicability.

- **Physics compliance metrics.** Even simple measures like trajectory smoothness, energy conservation in closed systems, or collision accuracy compared to ground-truth simulations would substantiate the physics-awareness claims far more convincingly than perceptual scores.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unreliable VLM-based physical parameter estimation"** (harsh critic, #3 under Section-by-Section): The paper already acknowledges this limitation explicitly (Sec. 4.2: "VLMs are not explicitly trained on physics-oriented datasets, the inferred material parameters...often lack the numerical accuracy required to reflect precise physical behavior") and proposes SDS-based optimization of Θ as a mitigation. While the mitigation is imperfect (SDS optimizes for visual plausibility, not physical accuracy), the paper does address this concern rather than ignoring it. Keeping as minor concern about SDS not guaranteeing physical accuracy, but removing the claim that this is unaddressed.

- **"Closed-source models treated as directly comparable"** (harsh critic): This is a standard comparison practice in the field — Sora, Runway etc. are widely compared against in recent literature. Removing as a standalone weakness.

- **"Reproducibility of proprietary components"** (harsh critic, #1 under Sec. 4.1): This is a nitpick about reproducibility of components that exist and are available (GPT-4o API, Qwen models). Per hard rules, removing this.

- **"Missing related works"** (spark's suggestion to compare with Physics3D and DreamPhysics): Per hard rules, do not mention missing related works as they may not exist or may not be directly comparable.

- **"Formatting/style nitpicks"** (neutral reviewer #5 about sequential optimization being described with just "our experiments reveal"): This is partially valid — the sequential optimization is supported by empirical observation rather than theoretical justification — but the criticism about presentation style is removed.

- **"No confidence intervals or variance"** (harsh critic): For a 17-example evaluation, reporting variance would be appropriate but this is close to a standard practice nitpick in a field where single-run evaluations are common. Moved to Nice-to-Haves.

## Novel Insights

The paper identifies a genuine and underexplored tension in physics-aware generation: physics simulators produce physically grounded but visually coarse dynamics (due to VLM estimation errors and grid-based collision approximations), while video diffusion models produce visually plausible but physically unconstrained motions. The hybrid approach of using physics simulation as initialization and SDS refinement for perceptual correction is a reasonable middle ground, but the paper's own evidence (Fig. 5) suggests the "physically grounded" claim primarily means "visually more plausible than simulation alone," which is a different and weaker claim than faithful physical dynamics. This distinction matters for applications requiring actual physical accuracy (robotics simulation, scientific visualization) versus applications needing only perceived realism (entertainment, content creation).

## Suggestions

- **Soften "physics-aware" claims** to "perceptually more physically plausible" or "physics-informed" throughout the paper, since the evaluation provides no quantitative evidence of physical correctness beyond perceptual metrics.

- **Add a novel-view rendering experiment** (even qualitative) to substantiate the "explorable 4D" claim — this is the single highest-impact addition possible.

- **Expand the evaluation set** to at least 30–50 examples with explicit coverage of different material types and interaction patterns, and report per-scene results or variance.

- **Add failure case analysis** showing where the pipeline breaks down, which would build trust in the method's general applicability.

- **Disentangle simulator vs. SDS contributions** with a granular ablation (simulator-only trajectories alone vs. SDS-only vs. full pipeline).

## Score and Decision

Calibration against similar papers:
- **Trans4D** (compositional text-to-4D, weak results, limited novelty): scores 3, 5, 3, 5 → avg ~4
- **Sync4D** (physics-based 4D with MPM, limited validation): scores 3, 5, 5, 5 → avg ~4.5
- **OmniPhysGS** (physics-aware 3DGS with multi-material, stronger results): scores 8, 6, 6, 6, 6 → avg ~6.4
- **Physics3D** (physical properties via video diffusion, limited novelty per component): scores 8, 5, 5, 1 → avg ~4.75
- **ElastoGen** (4D elastodynamics with physics priors, limited experiments): scores 5, 3, 5 → avg ~4.3
- **CompGS** (compositional 3D with SDS, engineering-heavy pipeline): scores 5, 5, 5, 6, 6 → avg ~5.4

CP4D sits between OmniPhysGS (which has stronger per-component novelty but similar evaluation issues) and Trans4D/Sync4D (which have weaker results). CP4D's pipeline engineering is more comprehensive than most comparators (three-stage with genuine physics integration plus composition), and the results on the curated examples look convincing. However, the small evaluation scale (17 examples), overclaimed "faithful adherence to physical dynamics" without calibrated physics validation, GPT-4o circularity, lack of multi-view evaluation for claimed 4D capability, and limited ablation depth are substantive weaknesses that undermine the core claims. The method is a reasonable system contribution but not a strong scientific one given the gap between claims and evidence.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>