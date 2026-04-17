The paper text is in the user message. Let me search there. The paper says:

- Figure 5: "We adopt a compact RDT as the policy network, in which only VFMs are modular."
- Section 3.1: "all baselines and EmbodiedMAE share the same architecture, ensuring fair comparison by isolating the visual representation component."

The phrase "only VFMs are modular" combined with "isolating the visual representation component" strongly suggests that VFMs are frozen, acting as fixed extractors. However, the paper does not explicitly state "the VFM backbone is frozen during policy training." This is an ambiguity worth noting but not necessarily indicating unfairness — it likely means all VFMs are treated equally (all frozen or all fine-tuned).

Now let me also check: does the paper specify that SPA is not trained on DROID data?

From Section 2.1: "SPA (Zhu et al., 2025) employs CrocoV2-Stereo (Weinzaepfel et al., 2023) to estimate depth for approximately 1/15 of the DROID dataset."

This tells us SPA is pre-trained on a subset of DROID with estimated depth, while EmbodiedMAE uses the full DROID with ZED SDK depth. This is relevant to the fairness concern.

Let me verify more about what section A.3 says (referenced for DINOv2-RGBD):
The paper references "Section A.3 for details of this variant" about DINOv2-RGBD. We don't have the appendix, so we cannot fully evaluate this baseline. This is a legitimate concern.

Now, let me synthesize my assessment. Here are the key issues I need to verify:

1. **Frozen vs fine-tuned** — The paper implies all VFMs are modular (frozen) in the policy, but doesn't explicitly state it. However, "ensuring fair comparison by isolating the visual representation component" strongly suggests equal treatment.

2. **Unfair 3D baselines** — DINOv2-RGBD gets a "naïve trainable depth branch" while EmbodiedMAE is jointly pretrained on 3D data. SPA uses estimated depth on 1/15 of DROID while EmbodiedMAE uses ZED depth on full DROID. This is a legitimate concern about confounding data vs architecture.

3. **No statistical significance** — No error bars, no multi-seed results. Legitimate.

4. **Overclaimed semantic understanding** — The re-coloring experiment is qualitative. Legitimate but not fatal.

5. **Limited architectural novelty** — Valid observation but not a strong weakness per se; system contributions are standard in this area.

6. **Point cloud underperformance** — The paper acknowledges this. It's a valid observation but doesn't undermine the core contribution since RGBD works well.

7. **Missing comparison with 3D foundation models** (Point-MAE, etc.) — These are object-level models, not designed for embodied tasks. Not entirely fair to request this.

8. **No DINOv2 trained on DROID-3D baseline** — This is a valid and important point that would isolate data vs architecture contribution.

Let me now calibrate the score. Looking at the reference papers:

- **MCR (Robots Pre-train Robots)**: Scores 3, 8, 8, 8 → Accept (poster). This is very similar — pre-trains on DROID, evaluates with frozen encoders on robot tasks. Key concern was unfair comparison due to domain data advantage. Got a wide score spread.

- **9GKMCecZ7c (Building Generalist Robot Policy from PTMs)**: Scores 3, 3, 3, 3, 5 → Withdrawn/Reject. Limited evaluation, limited novelty, weak baselines.

- **GR-1**: Scores 8, 6, 5, 3 → Accept (poster). Strong real-world results, video pre-training for robot manipulation.

- **RDT-1B**: Scores 6, 8, 6, 8 → Accept (poster). Diffusion foundation model for bimanual manipulation.

- **EmbodiedSAM**: Scores 8, 8, 8, 6, 8, 8 → Accept (Oral). Strong 3D embodied perception, real-time performance.

This paper has:
- **Valuable dataset contribution** (DROID-3D)
- **Comprehensive evaluation** (70 sim + 20 real tasks)
- **Consistent improvements** over baselines
- But **confounds data domain advantage with architecture**, making it unclear how much the MAE architecture itself matters
- **Missing statistical significance**, making "consistent outperform" claims weak
- **Limited architectural novelty** (assembling existing components)
- **Real-world details sparse**

This paper is somewhat similar to MCR in the pre-training-on-DROID + domain advantage concern. MCR got scores of 3,8,8,8 and was accepted as a poster. EmbodiedMAE has a more comprehensive evaluation (70+20 tasks vs 20+3) and a more novel architecture, but has the same domain advantage confound and less statistical rigor. I'd place this in the 5-6 range — solid contribution but with significant methodological concerns that weaken the claims.

Let me draft the final review.</think>## Summary

EmbodiedMAE proposes a unified 3D multi-modal masked autoencoder for robot manipulation that jointly learns representations across RGB, depth, and point cloud modalities. The authors construct DROID-3D by extracting high-quality depth maps and point clouds from the full DROID dataset using the ZED SDK, then pre-train a ViT-Giant multi-modal MAE with stochastic Dirichlet-distributed masking and cross-modal decoding, followed by feature-aligned distillation into smaller variants. Evaluation on 70 simulation tasks (LIBERO + MetaWorld) and 20 real-world tasks (SO100 + xArm) shows improvements over existing VFMs including DINOv2, SigLIP, R3M, VC-1, SPA, and DP3.

## Strengths

- **Valuable dataset contribution (DROID-3D):** Processing the complete 76K trajectories of DROID with ZED SDK temporal fusion and AI-augmented depth extraction—rather than using estimated depth on a subset—is a tangible community resource. The systematic depth quality comparison across BridgeDataV2, RH20T, and DROID (Figure 2) clearly motivates this contribution.

- **Comprehensive evaluation breadth:** Testing across 70 simulation tasks (LIBERO's 4 suites + MetaWorld's 3 difficulty levels) and 20 real-world tasks on two distinct robot platforms (low-cost SO100 and high-performance xArm) provides substantially more empirical coverage than many prior representation-learning papers that remain in simulation or on a single platform.

- **Well-designed stochastic masking strategy:** Using a symmetric Dirichlet distribution to allocate unmasked patches across modalities avoids modality bias while maintaining flexibility for diverse downstream input configurations. The concentration parameter α provides interpretable control over masking diversity.

- **Practical engineering contributions:** The multi-level distillation pipeline (Bottom/Middle/Top alignment), HuggingFace-compatible codebase, and the finding that RGBD outperforms point cloud due to sensor noise are all practically useful for the robotics community.

- **Policy-agnostic validation:** Including ACT alongside the diffusion-based RDT demonstrates that learned representations transfer across policy architectures, not just to one specific head.

## Weaknesses

### Major:

- **Confounding of data advantage with architecture contribution:** EmbodiedMAE is pre-trained on DROID-3D (domain-matched, robot manipulation data with high-quality 3D annotations), while key baselines like DINOv2 and SigLIP are pre-trained on web-scale or ImageNet data (out-of-domain). SPA, the most directly comparable 3D-aware baseline, uses estimated depth on only ~1/15 of DROID. Without a controlled experiment that pre-trains a standard VFM (e.g., DINOv2 MAE or a vanilla single-modality MAE) on the same DROID-3D data, it is impossible to determine whether the observed gains come from the multi-modal MAE architecture or simply from domain-matched pre-training data. The paper acknowledges that "EmbodiedMAE is pre-trained on a large embodied dataset (DROID-3D), while several baselines are generic image VFMs" but does not discuss or control for this fundamental confound. This directly undermines the core claim that the architecture is responsible for "consistent outperformance."

- **Unfair treatment of 3D baselines:** The paper claims that "adding a trainable depth branch for DINOv2... can degrade performance" (Finding 3, Section 3.3) and that DP3 underperforms on real robots. However, the "naïve" DINOv2-RGBD baseline (described only as having a "trainable depth branch" with details relegated to Appendix A.3, which we cannot verify) and DP3 are not pre-trained on DROID-3D or given any domain adaptation, while EmbodiedMAE's RGBD branch is jointly pretrained on exactly this data. The conclusion that "naïve 3D integration degrades performance" may simply reflect that poorly adapted 3D baselines fail—something already known—rather than demonstrating EmbodiedMAE's architectural superiority for 3D fusion. The paper needs to either (a) pre-train DINOv2-RGBD on DROID-3D as a fairer comparison, or (b) explicitly qualify the 3D comparison as comparing pre-training strategies rather than architectures.

- **Missing statistical characterization:** The paper claims "consistently outperforms" and "SOTA performance" across 90 tasks, yet provides no confidence intervals, standard deviations, or multi-seed results. LIBERO results are shown as learning curves without variance bands; MetaWorld reports task-group averages without per-task scores; real-world results are single success rates without number of trials or seeds. Given the well-known volatility of policy training from visual observations, these claims cannot be assessed for statistical significance. This is particularly concerning for comparisons where margins appear small (e.g., EmbodiedMAE-Large vs. DINOv2 on some LIBERO suites) and for the scaling claim where Base vs. Large differences appear marginal.

### Minor:

- **Overstated semantic understanding claims:** Section 3.2 claims that EmbodiedMAE has "implicitly learned object-level semantic segmentation" based solely on a single qualitative re-coloring visualization (Figure 3, column 12). No quantitative evaluation (e.g., zero-shot segmentation metrics, feature clustering analysis, cross-modal retrieval) is provided. The observed behavior could be explained by spatial proximity-based feature sharing in the transformer rather than genuine object-level semantics. The claim should be softened or supported quantitatively.

- **Point cloud underperformance contradicts framing:** The paper frames itself as a "unified 3D multi-modal representation," yet EmbodiedMAE-PC consistently underperforms RGB-only policies, attributed to sensor noise. If the core value proposition is 3D understanding but the 3D-only modality doesn't work well, this should be discussed more prominently as a limitation rather than buried in a finding. The effective contribution is primarily an RGBD model enhanced by depth, not a general 3D multi-modal model.

- **Insufficient real-world experimental detail:** The paper does not report the number of demonstrations, evaluation episodes, or environmental randomization for real-world tasks. Without these details, the 20 real-world tasks cannot be properly interpreted or reproduced.

- **Ablation only at distillation scale:** Due to the prohibitive cost of ViT-Giant training, ablations only cover distillation parameters (masking ratio, feature alignment, β). Key architectural choices at the pre-training stage—Dirichlet masking vs. fixed ratios, cross-attention decoder vs. independent decoders, the choice of 96/60 unmasked patches—are not validated. The distillation ablation showing "performance insensitivity" to masking ratio (Table 4) paradoxically raises questions about whether the stochastic masking design matters at all.

### Trivial:

- Equation (1) contains typographical labels ("Dst", "RdB") in the rendered version, though this appears to be a PDF parsing artifact.

## Nice-to-Haves

- A controlled experiment pre-training DINOv2 (or a vanilla MAE) on DROID-3D RGB-only data, to isolate the contribution of domain-matched data from the multi-modal MAE architecture.
- Quantitative depth estimation or cross-modal retrieval metrics to validate the claimed cross-modal alignment.
- Per-task breakdown tables for LIBERO and MetaWorld, along with multi-seed error bars.
- Explicit statement of whether VFMs are frozen or fine-tuned during policy training (the phrase "only VFMs are modular" implies frozen, but this should be unambiguous).
- Discussion of how DROID-3D's ZED SDK dependency generalizes to other depth sensors and datasets.

## Removed Points

- **"Missing comparison with dedicated 3D foundation models (Point-MAE, Point-BERT)":** These are object-level classification/segmentation models trained on isolated objects (ModelNet40, ShapeNet), not designed for embodied manipulation with sequential observations. Requesting them as baselines for a robot manipulation VFM is scope creep. *Treat with caution.*

- **"ZED SDK dependency limits generalizability of the data contribution":** The paper explicitly states DROID-3D is a supplement to DROID. The ZED SDK is a specific technical choice that produced higher-quality depth than alternatives; this is a feature, not a bug. Criticizing the specific tool used for data processing is a nitpick about engineering methodology, not a scientific limitation. *Treat with caution.*

- **"Limited architectural novelty":** Assembling existing components (MultiMAE masking, DP3 patchifier, DINOv2 ViT, MAE reconstruction) into a system for a new domain is a standard and accepted form of contribution in systems/embodied AI papers. The system-level integration, data contribution, and comprehensive evaluation constitute novelty even without individually novel components. *Treat with caution.*

- **"No weighting between RGB/depth/point cloud losses is mentioned":** The paper states in Section 2.3 that "normalized targets yield better performance" and uses l2-normalization for RGB/depth and group center-normalization for point clouds. This is an explicit normalization strategy. While an ablation on per-modality weights would be informative, the normalization approach is a reasonable default. *Treat with caution.*

- **"The teacher model's training details beyond MAE loss are unclear":** Section 2.4 describes the teacher as pre-trained from scratch on DROID-3D. Since no additional losses are mentioned and the training section (2.5) only references MAE loss and standard hyperparameters, the assumption should be that the teacher trains with MAE loss only. This is not ambiguous. *Treat with caution.*

## Novel Insights

The most interesting empirical finding is that properly integrated RGBD (depth as auxiliary cue) significantly outperforms point cloud input for real-world policy learning, despite the paper's framing around 3D understanding. This suggests that the embodied AI community's current emphasis on point cloud representations (DP3, 3D Diffuser Actor) may be misplaced for practical manipulation—depth's robustness to sensor noise may matter more than point cloud's geometric expressiveness. The distillation ablation also reveals a surprising finding: the top-layer alignment between teacher and student is critical (removing it drops performance from 92.4% to 74.4%), which suggests the student model relies heavily on final-layer representations rather than developing independent multi-scale understanding.

## Suggestions

- Add a critical controlled experiment: pre-train a DINOv2-scale MAE (ViT-Large) on DROID-3D RGB-only data with the same training budget, then evaluate with the same RDT policy. This directly isolates the data domain advantage from the architecture contribution and would either validate or refute the core architectural claim.
- Report multi-seed results (minimum 3 seeds) with mean ± std for all simulation benchmarks, and provide per-task breakdowns. For real-world tasks, report number of evaluation episodes per task.
- Explicitly state whether VFM backbones are frozen during policy training in the main text (not just implied by "modular"), and specify if any baselines received different training treatment.

## Score and Decision

**Calibration comparison:**

- **MCR (Robots Pre-train Robots)** — scores 3/8/8/8, Accept (poster). Similar domain advantage confound (pre-trains on DROID, baselines don't), but had a novel training objective (manipulation-centricity). EmbodiedMAE has broader evaluation but less architectural novelty and the same fairness concern unaddressed.
- **GR-1** — scores 8/6/5/3, Accept (poster). Strong real-world results on CALVIN + real robot, clearer architectural contribution (video generative pre-training). EmbodiedMAE's fairness confound is more severe.
- **9GKMCecZ7c (Building Generalist Robot Policy from PTMs)** — scores 3/3/3/3/5, Reject. Weak evaluation, limited novelty, overclaimed findings. EmbodiedMAE is clearly stronger than this.
- **RDT-1B** — scores 6/8/6/8, Accept (poster). System-level foundation model for bimanual manipulation, comprehensive real-world evaluation, with clearer novelty in the unified action space design.

EmbodiedMAE is a solid systems-level contribution with a valuable dataset, but the core experimental claim—architectural superiority of multi-modal MAE—remains confounded by domain data advantage, and the lack of statistical rigor weakens "consistent outperformance" claims. The paper falls between MCR (which was accepted despite similar concerns) and weaker rejected papers. Given the severity of the data confound and missing statistics, I place it slightly below MCR.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>