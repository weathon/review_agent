## Summary

CP4D proposes a compositional framework for physics-aware 4D scene generation that decomposes the problem into a static 3D background and physically grounded dynamic foregrounds, processed through a three-stage pipeline: (1) style-coherent 3D representation synthesis using image editing models to ensure foreground–background harmony; (2) hybrid motion synthesis combining heterogeneous physics solvers (MPM, rigid-body, PBD) with SDS-based refinement from video diffusion models; (3) automated scene composition using depth-aware heuristics and optimization. The method supports elastic, rigid, and fluid materials, and enables compositional editing of scene elements.

## Strengths

- **Compositional formulation with style-coherent generation** (Sec. 4.1, Eq. 2): The strategy of using an image editing model to generate a harmonized composite image before separate 3D reconstruction is a simple, practical, and effective solution to the real problem of stylistic mismatch between independently generated backgrounds and foregrounds. This is a genuinely useful design choice.

- **Heterogeneous physics solver coverage** (Sec. 4.2): Unlike prior physics-driven 4D methods that typically handle only one material type (e.g., PhysGen3D's elastic-only simulation causing rigid objects to collapse), CP4D employs MPM for elastic/flexible, rigid-body, and PBD for fluid simulation, enabling genuinely multi-material scene dynamics as shown in the qualitative comparisons (Fig. 4).

- **Hybrid motion synthesis with separate SDS refinements for distinct failure modes** (Sec. 4.2, Eqs. 4–5): The two-problem, two-solution design—SDS optimization of material parameters Θ for VLM estimation inaccuracies, and SDS optimization of displacement variables ΔΓ for collision approximation artifacts—is more principled than applying a single diffusion prior. The ablation (Fig. 5) shows distinct failure modes when each component is removed, validating the necessity of each.

- **Strong quantitative results against physics-driven baselines**: CP4D achieves the best physical realism score (0.694 vs. 0.624 for PhysGen3D in Table 2) and best 3D consistency (95.55 vs. 92.99 for PhysGen3D in Table 1) among physics-driven methods, supporting the claim of improved physical plausibility within this category.

- **Compositional design enables controllable editing** (Sec. 5.4, Fig. 6): The decomposition into independent background, foreground, and motion components naturally supports zero-shot replacement of scene elements while preserving coherence—a practical advantage over monolithic generation.

## Weaknesses

### Fatal

None.

### Major

- **SDS refinement optimizes for visual plausibility, not physical accuracy—the paper overclaims "faithful adherence to complex physical dynamics."** The core mechanism for delivering "physics-awareness" is SDS-based optimization of material parameters Θ (Eq. 4) and displacement variables ΔΓ (Eq. 5). SDS loss measures alignment with a video diffusion model's visual prior, not agreement with physical ground truth. The paper claims the method "faithfully complies with physical laws" and achieves "faithful adherence to complex physical dynamics" (Abstract, Sec. 4.3, Conclusion), but provides zero evidence that SDS-optimized parameters are more *physically* accurate than the VLM-initialized ones—only that they look better (Fig. 5). The displacement variables ΔΓ explicitly override simulator outputs with position adjustments optimized for visual appearance, further disconnecting the result from physical grounding. The paper should acknowledge this distinction: the method produces *visually plausible* motion that is *initialized* by physics, not motion that is *faithful* to physical dynamics. This matters because the paper positions itself as physics-aware in contrast to "physically inconsistent" prior work, yet the refinement mechanism could in principle drive the system away from physical correctness toward visual appeal.

- **Missing contemporary text-to-4D baselines.** TC4D (Bahmani et al., 2024a) and 4D-Fy (Bahmani et al., 2024b) are explicitly cited in the related work as relevant text-to-4D methods, yet neither appears in experiments. The only text-to-4D baseline is DreamGaussian4D (Ren et al., 2023), an early method with significantly weaker performance. Given that the paper claims to "significantly outperform existing methods," the absence of the most directly comparable contemporary baselines is a significant gap. While the comparison against physics-driven methods (PhysGen, PhysGen3D, OmniPhysGS) is appropriate and meaningful for the physics-awareness claim, the broader "outperforming existing methods" claim requires comparison with more recent 4D generation approaches.

- **No evaluation of novel view synthesis despite claiming "explorable" 4D scenes.** The paper repeatedly claims to generate "explorable and interactive 4D scenes" (Abstract, line 9; Contributions, line 27) and "supporting flexible viewpoint changes" (Sec. 4 overview, line 46). The entire point of 4D generation versus 2D video generation is novel view synthesis. Yet all evaluation metrics (VBench, WorldScore, GPT-4o) are computed on videos rendered from fixed camera trajectories—none measure rendering quality from viewpoints different from the reference view. The composition mechanism uses a single monocular depth estimate (Eq. 7) and optimizes against a single reference image (Eq. 9). Whether the composed scene looks coherent from novel viewpoints—the defining capability of 4D over video—is completely untested.

### Minor

- **Small evaluation set (17 examples) with no variance reporting.** Tables 1–2 report only mean scores over 17 hand-curated examples with no standard deviations or confidence intervals. The GPT-4o evaluation (Table 2) is particularly noisy at this sample size (e.g., Physical realism 0.694 vs. 0.670 for Runway). Claims of "significantly outperforming" are not statistically supported. This is somewhat typical for 4D generation papers, but the strong language in the claims should be tempered.

- **Ablation study is qualitative only.** Fig. 5 shows three ablation configurations but reports only visual comparisons, not quantitative VBench/WorldScore numbers for ablated variants. This makes it difficult to assess the magnitude of each component's contribution. Additionally, the most fundamental design choice—compositional decomposition vs. monolithic generation—is not ablated.

- **Limited evidence for controllability claims.** The "strong interactive controllability" claim (Sec. 4.3, line 17) is supported by only two editing examples (Fig. 6), with no evaluation of whether edited scenes remain physically plausible.

### Trivial

- None.

## Nice-to-Haves

- Validation that SDS-optimized trajectories are more physically accurate (not just visually appealing), e.g., by simulating a known scenario (bouncing ball, pendulum) and measuring trajectory error against ground truth.
- Rendering the composed 4D scene from novel viewpoints (30°, 60°, 90° off reference) to directly test the "explorable" claim.
- Quantitative ablations reporting VBench/WorldScore for ablated variants.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Unfair comparison against 2D video models on 3D consistency metrics**: The harsh critic argues that beating Sora/CogVideoX on 3D consistency is "entirely expected and uninformative." While the 3D consistency comparison with 2D models is indeed uninformative, the paper also compares against physics-driven methods (PhysGen, PhysGen3D, OmniPhysGS) where the advantage is meaningful. Including 2D models for breadth is not unfair—the asymmetry actually demonstrates a stronger point. The real issue is the *missing* 4D baselines, which is covered above.

- **Scale initialization formula assumes foreground should fill the view**: The harsh critic criticizes Eq. 8 for computing the maximum scale. But the paper explicitly states this is an *initialization* (line 134: "we initialize S as the maximum feasible scale"), which is then refined by optimization (Eq. 9). Maximum-feasible-scale is a reasonable starting point for optimization. This is not a weakness.

- **VLM/physics accuracy acknowledgment "raises questions" about physics-awareness**: The paper's transparency about VLM estimation inaccuracies and simulator limitations is a strength, not a weakness. Acknowledging known problems and proposing solutions (SDS refinement) is standard practice.

- **Qualitative comparison is "selective"**: The paper shows two qualitative scenarios (Fig. 4) and references more results in Appendix E and F. This is typical for generation papers. Calling it "cherry-picked" without evidence is speculative.

- **Reproducibility concerns about undisclosed hyperparameters**: Per hard rules, these are removed as nitpicks about implementation details.

## Novel Insights

The most insightful observation across the reviews is that CP4D occupies an interesting middle ground: it is *more* physics-grounded than pure diffusion-based 4D methods (because simulation provides the trajectory backbone) but *less* physics-faithful than it claims (because SDS refinement optimizes for visual appeal, not physical accuracy). The honest characterization would be "physics-initialized, visually refined" rather than "physics-aware." This distinction matters because the field is converging on hybrid simulation–diffusion approaches, and being precise about what the diffusion prior actually provides (visual plausibility, not physical correction) is important for future work.

## Suggestions

- Tone down claims from "faithful adherence to complex physical dynamics" to "physics-initialized motion with visual refinement" — this is both more accurate and still a meaningful contribution.
- Add TC4D and 4D-Fy as baselines; even if their code is not easily runnable, reporting numbers from their papers on overlapping metrics would strengthen the comparison.
- Render composed scenes from 2–3 novel viewpoints and report PSNR/SSIM or conduct a user study to validate the "explorable" claim.

## Evaluation Axes

- **Originality**: Moderate. The compositional decomposition and hybrid simulation–SDS approach are well-motivated and practically effective, but individual components (SDS, physics solvers, monocular depth composition) are existing techniques combined in a sensible pipeline.
- **Importance of research question**: High. Physics-aware 4D generation is an important and active research direction with clear applications.
- **Claims well supported**: Partially. The visual quality and physical plausibility claims are supported by Table 2, but the "faithful adherence" claim is overclaimed, and the "explorable" claim is untested.
- **Soundness of experiments**: Moderate. The comparison against physics-driven baselines is sound, but missing contemporary 4D baselines, lack of novel view evaluation, and small sample size limit the conclusiveness.
- **Clarity**: Good. The three-stage pipeline is clearly described, equations are well-formulated, and the motivation for each design choice is explained.
- **Value to community**: Moderate-to-high. The compositional framework and heterogeneous solver support address real limitations of prior work, and the style-coherent generation strategy is a practical contribution others will likely adopt.

## Score and Decision

**Calibration comparison:**

| Anchor Paper | Avg Score | Comparison to CP4D |
|---|---|---|
| MTVCraft (7.33) | High | CP4D is weaker: less technical novelty in core components, less convincing evaluation |
| VoMP (7.00) | High | CP4D is weaker: VoMP has a dedicated training pipeline and larger-scale evaluation |
| ShapeGen4D (5.33) | Medium | Roughly comparable: both have real contributions but significant evaluation gaps |
| PhyCAGE (3.50) | Low | CP4D is stronger: more complete pipeline, better visual results, broader evaluation |
| DiffuPhyGS (2.50) | Low | CP4D is clearly stronger: much better visual quality, more baselines, multi-material support |
| MoCtrl4D (2.50) | Low | CP4D is clearly stronger: more convincing results, broader comparison |

CP4D sits above the low-scoring physics-aware generation papers because it has genuine contributions (heterogeneous solvers, compositional design, style-coherent generation) and reasonable visual results. It sits below the high-scoring papers because of overclaimed physics fidelity, missing 4D baselines, no novel view evaluation, and a primarily qualitative ablation. It is roughly at the ShapeGen4D level but with somewhat more significant evaluation gaps (no novel view test at all, missing 4D baselines). A score of 4.5 reflects a paper with real contributions undermined by incomplete evaluation and overclaimed scope.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>