---
job_id: 732a7a75-3fc3-452e-a306-48a489053890
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 5qd7V5TNGV.pdf
paper: CP4D: Compositional Physics-Aware 4D Scene Generation
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses physics-aware 4D scene generation using diffusion models, differentiable simulators, and compositional 3D representations, which fits squarely within generative models and representation learning for vision.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present. The work is technically non‑trivial, reasonably clearly written, and includes both qualitative and quantitative experiments, with baselines and ablations. While there are weaknesses (see review), none rise to the level of desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The only prompts in the text are for evaluation (GPT‑4o) and for material/force estimation; they are clearly part of the method/appendix and do not attempt to manipulate the review process.

---

# Expected Review Outcome:

## Summary

The paper proposes CP4D, a compositional framework for text‑driven 4D (dynamic 3D) scene generation that explicitly incorporates physics. The method decomposes a text prompt into background and foreground components, generates 3D Gaussian representations for each, simulates foreground dynamics with heterogeneous physical solvers whose parameters are refined via SDS from a video diffusion model, and finally composes foreground and background via depth‑based initialization and optimization of object scale and position. Experiments on 17 curated prompts compare CP4D with physics‑based video/4D methods and text‑/image‑to‑video models, showing better VBench/WorldScore metrics and GPT‑4o judged physical realism, along with ablations and editing demos.

## Strengths

1. **Clear compositional formulation and modular pipeline.**  
   Framing 4D generation as composing a static 3D background with one or more dynamic, physics‑driven foreground objects is conceptually clean and practically useful. **Figure 1** (Page 2) nicely clarifies the three stages and how LLM‑based prompt decomposition, image editing, 3D reconstruction, physics solvers, and SDS refinement connect. This modularity facilitates controllable editing (Section 5.4, **Figure 6**) and re‑use of strong pretrained “experts” for different sub‑tasks.

2. **Integration of physics solvers with video diffusion priors is well‑motivated.**  
   The hybrid motion pipeline (Section 4.2) combines heterogeneous physical solvers (MPM, rigid body, PBD) with SDS‑based refinement over both material parameters and object displacements. The discussion around **Figure 2** makes a concrete and compelling case: grid‑based discretization leads to “phantom” collisions, and optimizing object displacements via SDS corrects these while preserving physically plausible trajectories.

3. **Automated 4D composition mechanism is non‑trivial and mathematically specified.**  
   The composition stage (Section 4.3) uses monocular depth and a depth‑aware scale heuristic to initialize translation and scale, followed by image‑space optimization. **Equation (7)** and **Equation (8)** make explicit how the centroid depth is back‑projected and how the maximal feasible scale is computed from frustum bounds and the foreground’s spatial extent. **Figure 3** gives an intuitive visualization of this heuristic and connects well to Eq. (8). The sequential optimization in Eq. (9) is a reasonable practical fix for the scale/position ambiguity.

4. **Empirical comparison against strong baselines, including proprietary models.**  
   The paper compares CP4D to Sora, Runway, Wan, CogVideoX, and several recent physics‑based generators (PhysGen, PhysGen3D, OmniPhysGS, DreamGaussian4D) using VBench and WorldScore metrics plus GPT‑4o scoring. In **Table 1** (Page 8), CP4D achieves the best or second‑best scores across most columns, especially in WorldScore photo consistency, 3D consistency, and motion smoothness. **Table 2** shows GPT‑4o preferences for CP4D in physical realism, photorealism, and semantic alignment, which aligns with qualitative impressions from **Figure 4**.

5. **Ablation studies connect design choices to observed behavior.**  
   Section 5.3 and Appendix D analyze the necessity of material and position optimization (Figure 5), and of the composition initializations (Figure 14, **Table 3**). For instance, Figure 5 visually shows that disabling material optimization yields overly compliant, unrealistic deformations, while disabling position optimization leads to visible ghost contacts; Table 3 quantifies degradations in VBench/WorldScore and GPT‑4o metrics when various refinement/composition components are removed. This gives at least partial causal evidence that the proposed components matter.

6. **Demonstrated controllability and editing.**  
   The compositional design yields practical editing capabilities: **Figure 6** illustrates zero‑shot background and object swapping while maintaining plausible dynamics and temporal consistency. This is a differentiator relative to monolithic text‑to‑video or text‑to‑4D systems where disentangled editing is usually harder.

7. **Technical appendices show non‑trivial physics implementation effort.**  
   Appendices B and C detail MPM, rigid body, and PBD solvers, including explicit equations like the MPM momentum update (**Eq. (10)**), deformation gradient update (**Eq. (11)**), inertia tensor from Gaussian particles (**Eq. (14)**), and PBD constraints (**Eqs. (19)–(28)**). While much of this is adapted from prior work, it evidences a serious attempt to build a consistent physics stack around 3D Gaussians.

## Weaknesses

1. **Very small, curated evaluation set; limited evidence of generality.**  
   The main experiments use only 17 prompts (Section 5.1). These are hand‑designed, adapted from VideoPhy with explicit structure (Fig. 7 in the appendix), and likely tailored to the types of dynamics CP4D handles well (single object or a small number of objects; simple interactions). This is a major limitation: the strong numbers in **Tables 1–3** may not generalize to more diverse, cluttered, or long‑horizon scenarios. There is no evaluation on standard 4D or physics video benchmarks (e.g., PhysGen3D’s datasets, generic real‑world dynamics datasets), nor stress tests on prompts where LLM‑based material/force inference might fail.

2. **Physics advantages are mostly assessed via perception proxies, not physical metrics.**  
   The central claim is improved physical realism, but the quantitative evidence is almost entirely high‑level metrics (VBench, WorldScore) and GPT‑4o judgments. None directly measure physical law adherence (e.g., conservation of momentum, plausible deformation under a given load, restitution coefficients), nor do they compare simulated trajectories to ground truth physical simulations or real videos. For example, in the MPM analysis (Appendix C.1, **Figure 11**), the discussion is qualitative; no numeric measure is given for how close the optimized parameters get to plausible material ranges. Without such metrics, it is hard to disentangle “looks physically reasonable” from “actually respects physics”.

3. **Heavy reliance on proprietary and black‑box models raises reproducibility and attribution concerns.**  
   The pipeline uses GPT‑4o for prompt decomposition and material/force estimation, Qwen‑Image and Qwen‑Image‑Edit for background and composite images, Depth Anything for depth, SAM for segmentation, Trellis and ViewCrafter for 3D, and a proprietary or unspecified video diffusion model for SDS supervision in Equations (4) and (5). Many of these are large, rapidly changing closed‑source systems. The method’s performance may be brittle to changes in these components, yet the paper does not analyze sensitivity, nor does it carefully ablate the dependence on GPT‑4o’s physics knowledge. This also makes reproduction by the community significantly harder; the provided implementation details (Appendix A.2) are less convincing when core components are opaque.

4. **Ambiguity and under‑specification in the SDS‑based optimization objectives.**  
   While Equations (4) and (5) formally define gradients, several important details are missing or glossed over:

   - It is not specified over which time window or camera trajectories the videos \(V\) and \(V_{\Delta \Gamma}\) are rendered during SDS; this affects what aspects of motion are constrained.
   - The sampling distribution for timesteps \(\zeta\), noise \(\epsilon\), and camera viewpoints is not described, nor is the weighting function \(\omega(\zeta)\) beyond being “a weighting function”. Since the physical parameters \(\Theta = \{\rho, E, \mu\}\) and displacements \(\Delta \Gamma\) can have very different sensitivity to these grad distributions, this omission matters.
   - For the displacement optimization, \(\Delta \Gamma_i\) is introduced as a “global displacement variable”, but the exact parameterization is not clarified: is this a static offset applied to the entire trajectory, or time‑varying, or per‑object but shared across frames? The text around Eq. (5) suggests static, which may be insufficient to correct time‑varying phantom contacts, particularly in complex trajectories.

   These under‑specified aspects make it hard to assess convergence behavior and whether the optimization actually enforces a well‑posed objective rather than just heuristically “nudging” the visuals to match the prompt.

5. **Composition heuristics may fail in non‑centroid, non‑single‑object scenarios, and the limitations are understudied.**  
   The translation initialization in Eq. (7) uses the centroid depth of the foreground mask. This implicitly assumes the foreground is roughly convex and lies in a single depth layer. For articulated or extended objects (a long rod, a person), or multi‑object masks, the centroid may be deeply misleading. The depth‑aware scale heuristic in Eq. (8) enforces that the 3D foreground fits inside the frustum slice at depth \(P^z\), but only based on bounding boxes in \(x\) and \(y\); self‑occlusions or non‑axis‑aligned objects are not considered. **Figure 3** makes the simple spherical case look elegant but does not show failure cases. In the Appendix ablation (**Figure 14(a–b)**), removing position or scale initialization leads to obviously bad compositions, but there is no analysis of borderline cases where the heuristic is wrong but the final optimization in Eq. (9) cannot fully fix it.

6. **Evaluation protocol mixes heterogeneous task setups in ways that complicate fairness.**  
   Some baselines are image‑to‑video (CogVideoX‑I2V, Wan), some are text‑to‑video (Sora, Runway), some are image+physics (PhysGen), and some are 3D‑to‑video (OmniPhysGS, DreamGaussian4D). CP4D has access to both text and specialized 3D reconstruction of the foreground plus a physically explicit simulator, and its evaluation setup (Section A.1) constructs foreground 3D for all methods. However:

   - OmniPhysGS is explicitly 3D‑to‑video and outputs with blank backgrounds; unsurprisingly it scores very poorly on photorealism and semantic alignment (**Tables 1 and 2**), which may say more about the evaluation protocol than the method.
   - For Sora and Runway, it is not fully clear how prompts are constructed or what images are provided; they may operate under less structured input than CP4D, yet are judged by the same GPT‑4o protocol (Fig. 8).
   - The 4D competitors like DreamGaussian4D are not given the same compositional setup; instead they operate more as single‑object 4D generators.

   The paper does not explicitly discuss these mismatches, which weakens claims like “consistently outperforms state‑of‑the‑art baselines” in a strict apples‑to‑apples sense.

7. **Limited discussion of failure modes and scope.**  
   The method is clearly tuned for scenes with a small number of structured foreground objects on relatively simple backgrounds. Yet the main text barely discusses where it fails. The only explicit limitation, in Appendix G, is computational cost. There is no systematic exploration of, for example, complex multi‑body collisions, non‑rigid coupled bodies, occlusions, lighting mismatches between ViewCrafter backgrounds and Trellis foregrounds, or the failure of GPT‑4o to infer correct material/force parameters. A few cherry‑picked qualitative results in Appendix E/F look strong, but without negative examples or quantitative breakdown by scenario type, it is hard to judge robustness.

8. **Missing direct engagement with some very relevant recent work.**  
   The related‑work section is reasonably broad, but some directly related lines of work are not cited or compared:

   - Trans4D (Zeng et al., 2024), which explicitly tackles compositional text‑to‑4D synthesis with realistic transitions and interactions. While focused on transitions, the compositional 4D setting is very close to CP4D’s; this should be discussed in Section 2.1 and compared conceptually.
   - The Phys4D line (e.g., Phys4D: Fine‑Grained Physics‑Consistent 4D Modeling from Video Diffusion), which also uses video diffusion priors to enforce fine‑grained physics consistency in 4D; this is directly relevant to the SDS‑based refinement in Section 4.2.
   - Work on compositional dynamic scene understanding with physics priors (e.g., “Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering”), which, while focused on understanding rather than generation, uses related 4D representations and physics priors; it could help strengthen the motivation and positioning in Sections 1 and 2.2.

   Given that the paper positions itself as a compositional, physics‑aware 4D framework, failing to engage with these strands undermines the novelty and context.

9. **No systematic analysis of computational cost versus baselines.**  
   Appendix A.2 mentions multiple optimization stages (5 epochs for material, 100 for position) and multi‑GPU training for each scene, but there is no wall‑clock comparison against, say, PhysGen3D or OmniPhysGS. For an ICLR audience, knowing that CP4D may take orders of magnitude longer per scene than direct video diffusion generation or purely simulator‑driven approaches is important when judging practicality.

10. **Clarity issues and minor mathematical inconsistencies.**  
    While most equations are standard, there are places where notation drifts or assumptions are implicit:

    - In Eq. (6), \(S\) is a scalar but later in Eq. (8), \(S\) is computed by comparing different directional extents; it is implicitly assumed that isotropic scaling suffices, which is not generally true for elongated objects.
    - In Eq. (10), the symbol \(\Psi\) for strain energy density and its derivative \(\partial \Psi/\partial F\) are introduced without defining the chosen constitutive model in the main text, leaving important modeling choices (e.g., Neo‑Hookean vs. corotated) to the appendix. Since the visual behavior can be highly sensitive to this, at least a brief summary in Section 4.2 would help.
    - In Section C.3, PBD equations (19–27) are written as generic constraints, but the concrete choice of kernel and density estimator is omitted; this may be fine for an appendix, but the main paper implicitly leans on this for “physically plausible” fluids without stating what constitutes physically correct behavior.

    These are not fatal bugs, but they do complicate precise understanding of the method.

Taken together, the paper proposes a promising and thoughtfully engineered pipeline, but the empirical evidence and clarity are not, in my view, strong enough yet for a clear positive recommendation at ICLR.

## Potentially Missing Related Work

1. **Bohan Zeng, Ling Yang, Siyu Li, “Trans4D: Realistic Geometry-Aware Transition for Compositional Text-to-4D Synthesis,” 2024.**  
   - Relevance: Also tackles compositional text‑to‑4D generation with realistic transitions and interactions, which is conceptually very close to CP4D’s compositional 4D scene generation.  
   - Integration: Should be discussed in Section 2.1 as a key compositional 4D baseline, with a comparison of CP4D’s physics‑based motion and composition mechanisms versus Trans4D’s transition modeling; if feasible, it would be a valuable baseline in qualitative/quantitative comparisons.

2. **Xingrui Wang, Wufei Ma, Angtian Wang, “Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering,” 2024.**  
   - Relevance: Uses compositional 4D dynamic scene representations with physics priors, though for understanding/Q&A rather than generation. The representation and physics prior design are relevant to how CP4D structures scenes and motion.  
   - Integration: Could be cited in Section 2.2 when discussing physics‑based modeling and compositionality, and in the Introduction to better position CP4D within broader efforts to combine physics priors and 4D representations.

3. **Haoran Lu, Shang Wu, Jianshu Zhang, “Phys4D: Fine-Grained Physics-Consistent 4D Modeling from Video Diffusion,” 2026.**  
   - Relevance: Directly addresses enforcing fine‑grained physical consistency in 4D modeling using video diffusion models, similar in spirit to CP4D’s SDS‑based refinement in Section 4.2.  
   - Integration: Should be cited in Section 2.2 and compared against the material/position optimization strategy in Section 4.2; the authors could discuss differences in how the diffusion prior is used (e.g., optimizing physical parameters vs. directly optimizing 4D representations).

If some of these works are contemporaneous or unavailable to the authors at submission time, it is still useful to discuss them in a camera‑ready version to better situate CP4D.

## Questions

1. **On SDS objectives and optimization details.**  
   - Over which time horizon and camera path do you render the videos \(V\) and \(V_{\Delta \Gamma}\) used in Equations (4) and (5)? Are they the same as those used at inference time, or a smaller set of canonical views?  
   - How are timesteps \(\zeta\) sampled and how is \(\omega(\zeta)\) chosen? Some ablation or justification of these choices would increase confidence that the optimization is well‑behaved rather than fragile.

2. **On parameterization of displacement \(\Delta \Gamma\).**  
   - Is \(\Delta \Gamma_i\) a single 3D translation per object applied to the entire trajectory, or is it time‑dependent? If static, how does it correct for time‑varying phantom collisions in long interactions? If time‑dependent, what regularization prevents physically implausible jitter? Clarifying this (and possibly adding a small synthetic example) could strengthen the argument around Fig. 2.

3. **On evaluation scope and datasets.**  
   - Could you evaluate on (or at least provide a qualitative subset from) an existing physics or 4D benchmark, even if adaptations are needed, to test generalization beyond your 17 curated prompts? For example, some standard rigid‑body or deformable benchmarks, or a subset of PhysGen3D’s scenes. Evidence here could significantly increase my confidence.

4. **On ablations of GPT‑4o material/force inference.**  
   - How sensitive is performance to the quality of the VLM‑estimated physical parameters? For instance, if you deliberately perturb density or Young’s modulus by ±50% from GPT‑4o’s estimate before SDS refinement, how much do the final results and metrics change? Such an experiment would clarify whether the diffusion prior can robustly correct bad initial guesses.

5. **On runtime and scalability.**  
   - Can you report average wall‑clock times and GPU hours per scene for the full CP4D pipeline, broken down by stages (3D reconstruction, material optimization, displacement optimization, composition refinement)? And ideally compare that to at least one physics‑based baseline like PhysGen3D? This would help assess practicality.

6. **On multi‑object and complex interactions.**  
   - Beyond the two‑orange collision toy example in Fig. 2 and the bottle/cloth examples in Fig. 4, have you tried scenes with more than two interacting objects or with coupled rigid–deformable contact? If so, what failure modes arise? Even a qualitative discussion or a few failure cases in the appendix would be informative.

Addressing these questions with additional experiments or clarifications in the rebuttal could shift my assessment closer to acceptance.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The high‑level method is coherent and the physics solvers are standard, but the SDS optimization and evaluation for physics realism are under‑specified, and the tiny curated dataset limits how strongly the empirical claims are supported.

## Presentation Rating

3: good.  
The paper is generally clear, with helpful figures (especially Figures 1–3, 4, 5, 6) and equations; however, some important implementation details and limitations are pushed to the appendix or omitted, and the related‑work positioning misses a few key references.

## Contribution Rating

2: fair.  
The compositional 4D formulation plus hybrid physics–diffusion refinement is interesting and practically useful, but the conceptual novelty over concurrent work is moderate, and the empirical evidence for “physics‑aware 4D scene generation” is not yet fully convincing.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents a thoughtful and potentially impactful framework that combines compositional 4D representation, heterogeneous physics solvers, and diffusion‑based refinement, with strong qualitative demonstrations and promising quantitative metrics. However, the evaluation is conducted on a very small curated dataset, physics realism is judged mostly via perceptual proxies, key optimization details are under‑specified, and some closely related work is not discussed. With stronger, more systematic experiments and clearer positioning, this work could reach ICLR quality; in its current form I lean slightly toward rejection, while acknowledging its merits.

## Reviewer Confidence

4: confident.  
I am familiar with 3D/4D generative models, SDS‑based optimization, and physics‑based simulators. I carefully checked the core equations and experimental tables, though I did not attempt to re‑implement the method.