=== CALIBRATION EXAMPLE 16 ===

# Final Consolidated Review
##Summary

VIST3A introduces a framework for text-to-3D generation that stitches a pretrained feedforward 3D reconstruction model (e.g., AnySplat, VGGT, MVDUSt3R) onto a pretrained text-to-video VAE as its decoder, then aligns the generative model with this stitched decoder via direct reward finetuning. By identifying the most compatible layer in the 3D model via linear MSE fitting and attaching the downstream portion through a learned stitching layer, the method avoids training a 3D decoder from scratch and achieves state-of-the-art results on T3Bench, SceneBench, and DPG-Bench for text-to-3DGS generation, while also enabling text-to-pointmap generation.

## Strengths

- **The model stitching idea is well-motivated and practically significant for this domain.** The core insight—that independently pretrained video VAEs and 3D reconstruction models share partially linearly-transferable latent representations, enabling lightweight stitching rather than training from scratch—is non-obvious and empirically validated across multiple architecture pairings (Fig. 5, Fig. 10, Table 3, Table 5). This directly addresses a real bottleneck in LDM-based 3D generation.

- **Strong and consistent empirical improvements over prior work.** VIST3A outperforms all baselines on T3Bench, SceneBench, and DPG-Bench across nearly all metrics (Tables 1–2), with particularly large margins on Imaging Quality and Coherence. The human evaluation (Table 4) corroborates these gains, with VIST3A ranked first in >68% of text alignment cases and >87% of visual quality cases.

- **Generalizability across multiple video generators and 3D backbones is convincingly demonstrated.** The paper tests Wan, CogVideoX, SVD, and HunyuanVideo as generators, and AnySplat, VGGT, and MVDUSt3R as 3D models, showing the stitching framework is not architecture-specific (Tables 3, 5, Fig. 10).

- **The integrated vs. sequential ablation (Fig. 8, Appendix D.2) provides meaningful evidence for the unified latent-space design.** Injecting noise into latents and comparing stitched decoding vs. decode-then-reconstruct shows the sequential pipeline amplifies errors even at imperceptible noise levels, justifying the architectural choice beyond just quality metrics.

## Weaknesses

### Major:

- **No ablation comparing stitched decoder vs. training a decoder from scratch.** The paper's central claim is that stitching preserves pretrained 3D knowledge and is superior to training custom decoders. Yet no experiment directly compares the stitched approach against training a decoder of equivalent capacity from scratch on the same data. Without this, it remains unclear whether stitching provides a genuine advantage over simply using a good initialization, or whether the gains come entirely from the reward finetuning and data. This is the single most important missing experiment for validating the core methodological contribution.

- **Computational costs are entirely unreported.** The paper involves stitching and finetuning large video generators (Wan 2.1 T2V Large) with large 3D models (AnySplat, VGGT), plus reward finetuning that requires simulating full denoising trajectories. No training time, GPU memory requirements, FLOPs, or inference latency comparisons with baselines are provided. This is critical for practitioners to assess whether the quality gains justify the computational investment, and for understanding the tradeoff vs. SDS-based or multi-stage methods. The claim of "efficient" generation (implied by the end-to-end design) needs quantitative backing.

- **Text-to-pointmap generation is claimed as a contribution but lacks quantitative evaluation.** The paper introduces text-to-pointmap as a novel capability enabled by choosing VGGT or MVDUSt3R as the 3D backbone, but provides only qualitative results (Fig. 14) with the acknowledgment that "no established benchmarks or baselines exist." While the absence of benchmarks is understandable, the paper could still report standard reconstruction metrics (e.g., on generated scenes where reference geometry is available) or compare against a sequential baseline (generate images → run VGGT). Without any quantitative signal, this claimed contribution cannot be evaluated.

### Minor:

- **Reward weight selection lacks ablation justification.** The quality reward is scaled by 1/16 and the consistency reward by 0.05 (Appendix B.2), but no ablation explores alternative weightings. Table 6 ablates reward components but not their relative scales. Given that the consistency reward alone causes dramatic performance collapse (Imaging Quality drops from 58.23 to 38.67), the balance appears fragile and potentially architecture-dependent—practitioners would benefit from understanding sensitivity to these choices.

- **User study lacks standard methodological details.** Table 4 reports average ranks from 28 participants on 14 samples, but does not specify whether participants were blinded to method identity, how samples were randomized, or report inter-rater agreement (e.g., Fleiss' kappa). Without these details, the human evaluation is less informative than it could be.

- **No failure case analysis.** All qualitative results show successful generations. Given the complexity of the pipeline, honest presentation of failure modes (e.g., geometry collapse, prompt misalignment, artifacts from pose prediction errors) would strengthen the paper and help readers understand practical limitations.

- **The consistency reward depends on predicted camera poses, but pose error sensitivity is unanalyzed.** Eq. 6 computes the consistency reward using rendered views at poses predicted by the stitched 3D model. If the stitched model outputs inaccurate poses (especially in the generative setting where inputs are noisy latents rather than clean images), the consistency reward could become misleading. No sensitivity analysis or discussion of this dependency is provided.

- **The "no labels" framing could be clearer.** The abstract and introduction emphasize that the method requires "no labels" and is "self-supervised," but training uses DL3DV-10K and ScanNet—datasets that contain 3D annotations. The method does not use these 3D annotations (it uses pseudo-targets from the original 3D model), which is the intended meaning, but the phrasing risks misleading readers into thinking no datasets are needed at all. A brief clarification would help.

### Trivial:

- The theoretical justification for MSE as a layer selection criterion (Eq. 4, citing Insulla et al. 2025) provides an upper bound dependent on the Lipschitz constant κ₂, which is unknown and varies across layers. The empirical correlation in Fig. 5 is more compelling than the theoretical argument, which offers limited practical guidance beyond the observation that lower MSE is directionally better.

## Nice-to-Haves

- Error bars or significance tests on the main quantitative results, particularly where improvements are modest (e.g., CLIP scores on SceneBench).
- Layer-wise probing or feature analysis showing what specific geometric capabilities (depth estimation, pose prediction, multi-view correspondence) survive the stitching operation.
- Ablation on stitching layer complexity (3D convolution vs. simpler linear map) to assess whether the current design is necessary.
- Evaluation on more diverse or out-of-distribution prompt types to stress-test generalization.
- Analysis of how reward weights should be adapted when switching to different video/3D backbone pairings.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Equation/table formatting complaints** (Eq. 2 "garbled," Table 2 "incomplete," Fig. 6 "garbled"): These are PDF extraction artifacts, not author errors. Removed per formatting nitpick rule.
- **Missing related works as baselines** (LAN et al. 2025, Li et al. 2025b): Cannot confirm these are appropriate baselines without external verification. Removed per missing related works rule.
- **Code/model/checkpoint availability concerns**: Project page is provided; questioning release status is not appropriate per hard rules.
- **Questions about 2025-2026 citation existence** (Wang et al. 2026, Yang et al. 2025e): Per hard rules, all cited references are assumed to exist.
- **Demand for SDS comparisons at comparable compute budgets**: This asks to show SDS results when given similar compute to VIST3A's one-time training cost, which reverses the fairness asymmetry (SDS's per-scene cost is a known weakness VIST3A avoids). Removed per unfair comparison rule.
- **DPG-Bench table data appearing garbled**: The text clearly states VIST3A scores ">75 often ≈85" on DPG-Bench; the extracted table formatting issue is not an author error.
- **Training data overlap with evaluation benchmarks**: No evidence of overlap is presented; this is speculative.
- **Broader impact / misuse discussion**: Not standard for this venue and paper type.

## Novel Insights

The observation that independently pretrained video VAEs and 3D reconstruction models exhibit partially linearly-transferable representations at early layers—despite being trained on different data, with different objectives, and for different modalities—is genuinely surprising and has implications beyond this paper. It suggests that the low-level visual features encoded by video compression models share structural similarities with those learned by 3D reconstruction models, which may reflect a convergent representation of image statistics rather than task-specific encoding. The finding that CKA, a standard representation similarity metric, fails to identify the optimal stitching layer while MSE succeeds (Fig. 6) hints that linear predictability (affine transferability) and distributional similarity (CKA) capture different aspects of representation alignment—MSE captures what matters for functional stitching, while CKA captures overall distributional overlap. This distinction could inform future work on model composition.

## Suggestions

- **Add a stitched-vs-scratch decoder comparison:** Train a decoder of similar architecture from scratch on the same data (DL3DV + ScanNet) without stitching, keeping all other components identical. This single experiment would decisively validate or refute the core contribution.
- **Report training/inference compute:** At minimum, provide GPU hours for stitching search, stitched VAE finetuning, and reward alignment training, plus per-scene inference time compared to baselines.
- **Include 3–5 failure cases** with analysis of what goes wrong (e.g., pose prediction failures leading to consistency reward noise, out-of-distribution prompts, or scenes requiring non-sequential viewpoints).
- **Clarify the "no labels" claim** by explicitly stating "no 3D ground-truth labels are required" rather than the more ambiguous "no labels."
- **Add a brief sensitivity analysis** on the consistency reward's dependence on pose accuracy—e.g., perturb predicted poses during training and measure the impact on final generation quality.

---

**Assessment by axis:**

- **Novelty:** High. Model stitching between generative video models and discriminative 3D reconstruction models in this specific context is novel, and the finding that their latent spaces are partially linearly compatible is non-obvious. The direct reward finetuning adaptation for 3D alignment is a meaningful technical contribution, though it builds on established techniques (DRTune).

- **Technical soundness:** Moderate-to-good. The methodology is clearly described and the empirical results are strong. However, the missing stitched-vs-scratch comparison is a notable gap in validating the central claim, and the theoretical justification for MSE-based layer selection is weak (upper bound with unknown constants). The reward formulation works in practice but the fragility revealed by the consistency-only ablation is underanalyzed.

- **Empirical support:** Good. Consistent improvements across three benchmarks, multiple architecture pairings, and human evaluation provide strong support. The ablation studies (Table 6, Fig. 8) are informative. However, the text-to-pointmap claim lacks quantitative support, and computational cost data is absent.

- **Significance:** High. If the stitching approach is validated against from-scratch training, this could shift how the community builds 3D generation pipelines—reusing rather than rebuilding 3D decoders. The framework is modular and future-proof as better 3D foundation models emerge.

- **Clarity:** Good. The paper is well-structured with clear problem framing and method description. Some claims ("no labels") could be more precise, and the limitations section is understated relative to the actual constraints (sequential input requirement). Appendix dependency is heavier than ideal but acceptable.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
