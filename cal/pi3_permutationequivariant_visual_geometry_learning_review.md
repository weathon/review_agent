=== CALIBRATION EXAMPLE 41 ===

# Harsh Critic Review
## Section-by-Section Critical Review of π³: Permutation-Equivariant Visual Geometry Learning

---

### Title & Abstract

The title and abstract are accurate and appropriately scoped. The key claim — that removing a fixed reference view via permutation equivariance yields more robust and accurate reconstruction — is specific and testable. The abstract correctly characterizes the approach as predicting "affine-invariant camera poses and scale-invariant local point maps." One notable tension: the abstract says the approach "breaks the reliance on a conventional fixed reference view," yet Appendix A.4 reveals that stable training from scratch still requires an auxiliary reference-view proxy head. This tension between the headline claim and the training reality should be surfaced in the main paper rather than relegated to an appendix.

---

### Introduction & Motivation

The motivation is compelling and concrete. Figure 2 effectively illustrates the reference-view sensitivity problem. The claim that this inductive bias is "unnecessary" is well-argued for the feed-forward setting.

**Concern — priority of contribution:** The paper claims to be "the first to systematically identify and challenge the reliance on a fixed reference view." However, the broader observation that arbitrary canonical frame choices are problematic is well-known in SfM (and the paper cites this). The contribution is more precisely *demonstrating the effect in feed-forward networks and proposing a specific fix*. The first-mover claim should be stated more carefully.

**Concern — performance claims in the intro:** The introduction quotes Sintel ATE improving from VGGT's 0.167 to 0.074. Table 1 confirms this. However, on Co3Dv2 AUC (an in-domain benchmark), VGGT scores 88.59 versus π³'s 88.41 — a reversal of the claimed superiority — which is not mentioned in the introduction. The introduction should be more balanced.

---

### Method

**3.1 Permutation-Equivariant Architecture**

The formalization of permutation equivariance (Equations 1–3) is correct and clean. The practical implementation — removing frame index positional embeddings and reference view tokens — is appropriately described and justified.

**Critical concern — equivariance in practice vs. theory:** The paper claims *strict* permutation equivariance, but Table 6 reports nonzero standard deviations (e.g., DTU mean Acc. std = 0.003; DTU mean Comp. std = 0.006). For a model claimed to be *exactly* equivariant, these should be numerically zero (up to floating-point noise). The paper never discusses why the deviations are not zero, nor verifies whether they arise from floating-point non-associativity in parallel GPU execution or from a genuine architectural violation. This gap between the theoretical claim and empirical evidence deserves explicit discussion.

**3.2 Scale-Invariant Local Geometry**

The single-scale ROE solver approach (Equation 4) is reasonable. The depth-weighted L1 loss is standard. The decision to use a single global scale *s* across all N frames is a strong assumption — for scenes with large depth range variation or significant occlusion, a single scale may be suboptimal. No analysis or ablation of this choice is provided.

The confidence loss (BCE with a threshold-based pseudo-label) is reasonable but threshold ε is not specified in the main paper, only mentioned implicitly. This affects reproducibility.

**3.3 Affine-Invariant Camera Pose**

Supervising on relative poses (Equation 7) to avoid the global frame ambiguity is theoretically sound. The use of the same *s*\* scale from the point map loss to calibrate translations is elegant. However, this coupling means camera pose supervision quality is contingent on point map quality early in training, potentially amplifying instability — which the authors acknowledge as the "cold start" problem (Appendix A.4).

The comment that "real-world camera paths are highly structured" and "lie on a low-dimensional manifold" (Section 3.3) is used to motivate the affine-invariant formulation, but the causal link to the architecture is unclear. The PCA visualization (Figure 4, Appendix A.3) shows π³'s pose predictions are more structured than VGGT's, but this is partly *definitional*: per-camera-frame predictions in a local coordinate system are inherently more constrained than predictions anchored to an arbitrary global frame. This is a circular argument that should not be presented as independent evidence.

**3.4 Model Training**

**Major concern — training initialization from VGGT:** The final model initializes the encoder and alternating attention module from pre-trained VGGT weights (with the encoder frozen). This is a *critical* methodological issue that is buried in Appendix A.2 and not prominently discussed in the main paper. It means:

1. The main experimental comparisons (Tables 1–5) compare π³ (warm-started from VGGT) against VGGT itself (and other methods). The improvement may largely reflect the different fine-tuning objective and scale-invariant/affine-invariant losses rather than the permutation-equivariant architectural design *per se*.

2. Table 8 (Appendix A.4) shows that when both models are trained from scratch under equivalent conditions, π³ *underperforms* VGGT on 7-Scenes (Acc. 0.064 vs. 0.057, Comp. 0.068 vs. 0.046) and NRGBD (0.071 vs. 0.060, 0.047 vs. 0.042). Only with an auxiliary reference-view proxy head does π³ outperform VGGT on ETH3D and NRGBD from scratch. This is a significant finding. It suggests the permutation-equivariant design has an intrinsically harder optimization landscape, and the gains in the main experiments depend substantially on leveraging VGGT's pre-training, not purely on the architectural innovation.

This does not negate the contribution, but the paper's framing obscures this dependency. The main paper should explicitly discuss this tradeoff rather than treating the VGGT initialization as a mere computational efficiency choice.

**Concern — internal dataset:** The training data includes "an internal dynamic scene dataset" not available to the research community. No ablation quantifies its contribution, creating a fairness concern when benchmarking against published methods.

---

### Experiments & Results

**4.1 Camera Pose Estimation (Table 1)**

Results are strong on zero-shot benchmarks (Sintel, RealEstate10K). The comparison is largely fair, with clear notes about which datasets appear in training. However:

- Co3Dv2 AUC: VGGT (88.59) slightly outperforms π³ (88.41), which is inconsistent with the "state-of-the-art across all tasks" claim. This is glossed over in the text ("competitive SOTA results alongside VGGT").
- The paper uses a relatively lax angular threshold of 30° for the main table. Table 9 (appendix) provides tighter thresholds and confirms π³ is generally better, which is good — but it would be more convincing if these appeared in the main paper rather than the appendix.

**4.2 Point Map Estimation (Tables 2–3)**

Results are generally strong, but on 7-Scenes sparse, VGGT outperforms π³ in Accuracy (0.044/0.025 vs. 0.047/0.029) and Completion (0.056/0.033 vs. 0.075/0.049). The text does not address this specific shortfall. For a paper claiming SOTA across the board, failures on specific benchmarks should be explained rather than elided.

**4.3 Depth Estimation (Tables 4–5)**

Video depth results (Table 4) are clearly SOTA, with large improvements on Sintel (0.233 vs. VGGT's 0.299). The speed improvement (57.4 FPS vs. VGGT's 43.2 FPS) is attributable to using 36 instead of 48 alternating attention layers — this is an architecture change, not purely a benefit of permutation equivariance, and should be clarified.

For monocular depth (Table 5), π³ lags MoGe v1 on δ<1.25 accuracy on Sintel (0.614 vs. 0.695) and KITTI (0.971 vs. 0.979), which is expected given π³ is not specialized for monocular depth. The comparison is contextualized appropriately.

**4.4 Robustness Evaluation (Table 6)**

This is the strongest empirical evidence for the core claim. Near-zero standard deviations across DTU and ETH3D, orders of magnitude below VGGT (e.g., DTU Acc. std: 0.003 vs. 0.033), are compelling. The experimental design — using each frame as the "first" frame in turn — is a reasonable operational test.

However, the paper's framing "near-zero" coexists with the theoretical claim of *exact* equivariance. A sentence explicitly acknowledging that residual variance is likely due to floating-point non-associativity (not architectural violation) would be appropriate.

**4.5 Ablation Study (Table 7)**

**Major concern — scope of ablation:** The ablation compares three models: Model 1 (neither scale-invariant nor affine-invariant), Model 2 (scale-invariant but not affine-invariant), and the full model. Models 1 and 2 both use a reference view token and are not permutation-equivariant. This conflates multiple variables:

1. The loss formulation changes (scale-invariant vs. not).
2. The architecture changes (reference token vs. none).
3. The training objective changes (global coordinates vs. relative).

There is no variant that tests permutation equivariance alone without the scale/affine-invariant loss changes, nor one that tests the losses without the equivariant architecture. The ablation cannot isolate which component drives the gains.

Furthermore, in Table 7, gains from Model 1 → Model 2 (adding scale-invariance) on ETH3D are substantial (Acc. 0.229 → 0.197), while the gains from Model 2 → Full Model (adding affine-invariant poses + equivariance) are larger still (0.197 → 0.131). This suggests the affine-invariant pose modeling (and equivariance) contributes significantly — but the clean attribution is muddied by the confounds noted above.

---

### Writing & Clarity

The paper is generally well-written and easy to follow. The description of the auxiliary proxy task (Section A.4) is somewhat difficult to parse — a cleaner description of what "global proxy" means architecturally (cross-attention from the reference frame) would help readers evaluate whether the final model is truly reference-free. The claim in the main body that "our final model remains fully permutation-equivariant" despite being trained with an auxiliary non-equivariant task requires more careful justification.

---

### Limitations & Broader Impact

The limitations section is absent. The paper should discuss:

1. **Training dependency on VGGT initialization** — The model may not work as advertised without this specific pre-training, limiting the design's generalizability.
2. **Dynamic scenes are incompletely evaluated** — Figure 1 prominently features dynamic content, and TUM-dynamics results are provided for pose, but no quantitative evaluation of point map quality in dynamic scenes is provided.
3. **Internal training data** — The use of a proprietary dynamic scene dataset is not acknowledged as a limitation to reproducibility.
4. **Single global scale assumption** — The scale-invariant loss uses one global *s*\* for all N frames. This may fail for sequences with large scale changes or scenes where depth ground truth is noisy across frames.
5. **The "cold start" optimization difficulty** — The fact that training from scratch requires a reference-view auxiliary head means the design is not as architecturally self-contained as claimed.

---

### Overall Assessment

π³ addresses a genuine and well-motivated problem in feed-forward 3D reconstruction — the reference-view bias — and the empirical results on most benchmarks are strong. The near-zero permutation variance (Table 6) is a compelling demonstration of the core property. However, the paper has a significant methodological issue that undermines the strength of its causal claims: the main model is warm-started from VGGT's pre-trained weights, and the from-scratch comparison (Table 8, appendix) shows that the equivariant model actually underperforms VGGT without the auxiliary reference-view proxy task. This means the claimed advantages in Tables 1–5 may largely reflect favorable fine-tuning conditions rather than the permutation-equivariant design itself. The ablation study is insufficient to disentangle the contributions of equivariance versus the scale/affine-invariant loss formulations. For ICLR, the novelty of the idea, the quality of empirical results, and the clean technical presentation are positive factors, but the paper needs a more honest accounting of the VGGT-initialization dependency and a stronger ablation to make its causal claims credible. As currently written, the contribution is real but overstated.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces π³, a feed-forward neural network for visual geometry reconstruction that eliminates the reliance on a fixed reference view by employing a fully permutation-equivariant architecture. By predicting scale-invariant local pointmaps and affine-invariant camera poses relative to each frame, the method avoids reference-view bias, yielding improved robustness to input ordering and state-of-the-art accuracy across camera pose estimation, depth estimation, and point cloud reconstruction tasks. Extensive experiments across multiple benchmarks demonstrate its superior accuracy, efficiency, and consistency compared to reference-anchored baselines like VGGT and DUSt3R.

### Strengths
1. **Well-Motivated Problem Identification & Strong Empirical Validation:** The paper convincingly identifies the fixed reference view as a detrimental inductive bias in feed-forward 3D reconstruction. Figure 2 and Table 6 provide compelling evidence of performance degradation and high variance in prior methods when the reference frame changes, directly motivating the proposed architectural shift.
2. **Elegant & Mathematically Sound Formulation:** The permutation-equivariance property is formally defined and carefully implemented by removing order-dependent components (e.g., frame position embeddings, reference tokens). The use of relative supervision (N×N pairwise constraints) effectively resolves scale and coordinate ambiguities without sacrificing geometric consistency.
3. **Comprehensive Benchmarking & Efficiency Gains:** π³ achieves SOTA or highly competitive results across 4 tasks on 7+ datasets. The near-zero standard deviation in reconstruction metrics across permuted inputs (Table 6) empirically validates the equivariance claim. Furthermore, the model is notably efficient (959M params, 57.4 FPS on KITTI), outperforming VGGT (1.26B, 43.2 FPS) despite a simpler, reference-free design.

### Weaknesses
1. **Training Instability & Proxy Task Dependency:** Appendix A.4 reveals a "cold start" problem: purely relative training from random initialization struggles to converge due to the highly coupled O(N²) constraints. The authors introduce an auxiliary "global proxy" head that re-introduces a reference frame via cross-attention to stabilize optimization. While the final model drops this head, the reliance on a reference-dependent warm-up partially weakens the claim of a purely reference-free learning paradigm and introduces a non-trivial training complexity.
2. **Unaddressed Computational Scaling for Long Sequences:** The camera pose loss evaluates all ordered view pairs, implying O(N²) complexity. While the paper claims support for video sequences, it lacks evaluation on long-range inputs (e.g., >100 frames) and does not discuss memory/compute bottlenecks. This omission makes it difficult to assess practicality compared to scalable baselines like Fast3R, which explicitly targets 1000+ frame inference.
3. **Limited Evaluation on Dynamic Scenes & Real-World Robustness:** Despite highlighting dynamic scene reconstruction in the abstract/introduction and using an internal dynamic dataset for training, all quantitative evaluations are conducted on predominantly static benchmarks (7-Scenes, DTU, ETH3D, etc.). Additionally, failure cases (e.g., transparencies, grid artifacts from pixel shuffling) are noted in the limitations but lack targeted ablation or failure analysis, leaving gaps in understanding the model's operational boundaries.

### Novelty & Significance
The work presents a clear methodological novelty within the fast-moving domain of feed-forward 3D vision. While the backbone (DINOv2), attention blocks, and decoder heads build on established components, synthesizing them into a truly reference-free, permutation-equivariant system addresses a structural limitation shared by the DUSt3R/VGGT lineage. **Clarity** is strong: the mathematical formulation of equivariance and relative supervision is precise, and the experimental narrative is well-structured. **Reproducibility** is largely ensured through public code, detailed hyperparameters, dataset listings, and a clear two-stage training protocol, though the exact implementation of the ROE solver and dynamic dataset details could benefit from minor supplementation. **Significance** is high; the demonstrated variance reduction, speed improvements, and SOTA metrics on standard benchmarks provide tangible value for downstream applications in robotics, AR, and autonomous systems. The paper successfully argues that removing reference bias is both theoretically sound and empirically beneficial, making it a solid contribution to ML-driven geometric learning.

### Suggestions for Improvement
1. **Quantify the Trade-off of the Proxy Training Strategy:** Provide an ablation comparing final performance when training purely from scratch (without the proxy) versus with the proxy. Discuss alternative stabilization techniques that maintain pure permutation equivariance, such as progressive N-sampling, curriculum learning over pair distances, or contrastive pre-training on relative views, and report whether they mitigate the cold-start problem.
2. **Benchmark Long-Sequence Scalability & Complexity:** Evaluate the model on progressively longer video sequences (e.g., 50, 100, 200 frames) to measure memory usage, training time, and inference latency growth. If the O(N²) loss becomes prohibitive, discuss or implement efficient approximations (e.g., sliding windows, k-nearest neighbor pairing, or hierarchical grouping) and compare against sequence-optimized baselines.
3. **Expand Dynamic & Failure Mode Evaluation:** Include quantitative results on a public dynamic benchmark (e.g., TUM-dynamics results are in Table 1 but only for pose; depth/point cloud on dynamic scenes should be added or clarified). Provide a dedicated qualitative section or appendix analyzing common failure cases (transparent surfaces, specular reflections, extreme motion blur, and pixel-shuffle artifacts) to establish clear operational boundaries and guide future work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Fair Efficiency Comparison:** Retrain VGGT with 36 layers to match $\pi_3$'s depth (Appendix A.1), as the current FPS advantage (57.4 vs 43.2) may stem from fewer layers rather than the permutation-equivariant architecture. Without this, the efficiency claim is confounded and unconvincing.
2. **Long-Sequence Drift Evaluation:** Evaluate trajectory error on long odometry sequences (e.g., KITTI) rather than short video clips, as relative pose supervision without global optimization risks accumulating drift that undermines geometry consistency. This is critical to validate the "robust video reconstruction" claim.
3. **Dynamic Object Geometry Metrics:** Provide quantitative metrics specifically on moving object reconstruction (e.g., dynamic IoU or depth error on moving regions), as the current benchmarks (DTU, ETH3D) are predominantly static despite abstract claims of dynamic scene support.
4. **Standalone Training Viability:** Demonstrate competitive performance when training from scratch without VGGT initialization, as Table 8 (Appendix) shows significant degradation otherwise. This is needed to prove the architecture is a superior paradigm rather than a fine-tuning strategy.
5. **Oracle Scale Dependency:** Quantify performance drop when using predicted scale versus oracle alignment for depth evaluation, to determine if the method is practically usable without ground-truth scale supervision.

### Deeper Analysis Needed (top 3-5 only)
1. **Optimization Landscape Difficulty:** Analyze why the permutation-equivariant formulation suffers from the "cold start" problem (Appendix A.4) compared to reference-based methods, as this suggests the proposed inductive bias may be harder to optimize rather than superior.
2. **Mean Accuracy vs. Permutation:** Table 6 shows zero variance (expected by design), but you must analyze if *mean* accuracy fluctuates with different input orders to prove robustness impacts quality, not just stability.
3. **Global Consistency Mechanism:** Explain how local point maps align globally without bundle adjustment or a reference frame, as disjoint local predictions risk forming incoherent scenes over large baselines.
4. **Strict Zero-Shot Generalization:** Provide analysis on domains completely disjoint from the 15 training datasets (e.g., underwater or medical), as current "zero-shot" benchmarks (ScanNet, Co3D) overlap significantly with training distributions.
5. **Memory Scaling with Sequence Length:** Analyze memory usage and performance degradation as input sequence length $N$ increases, since transformer attention scales quadratically and limits "large-scale" applicability.

### Visualizations & Case Studies
1. **Global Point Cloud Stitching:** Visualize the union of local point maps for a full scene to expose misalignments or drift that single-view metrics hide.
2. **Dynamic Object Failure Cases:** Show reconstruction quality on moving subjects specifically, as this is a key claimed advantage over static-only methods.
3. **Drift Over Time Plot:** Plot trajectory error growth over frame index for long videos to visualize stability compared to VGGT.
4. **Attention Map Visualization:** Show how global attention aggregates information without a reference token to verify the model actually learns global context.
5. **Scale Ambiguity Examples:** Visualize cases where the single scale factor $s^*$ fails to align diverse views correctly, exposing limitations of the scale-invariant assumption.

### Obvious Next Steps
1. **Decouple from VGGT Initialization:** Demonstrate competitive performance training solely from scratch to prove the architecture's intrinsic value.
2. **Implement Loop Closure:** Add a mechanism to correct drift in long sequences, as pure feed-forward relative poses cannot guarantee global consistency.
3. **Standardize Model Depth:** Retrain VGGT with 36 layers to ensure the speed comparison is fair and attributable to architecture.
4. **Predict Metric Scale:** Integrate a module to predict absolute scale to remove reliance on oracle alignment for practical deployment.
5. **Dynamic Masking Integration:** Explicitly model dynamic masks to prevent moving objects from corrupting the static geometry estimation.

# Final Consolidated Review
## Summary

π³ introduces a permutation-equivariant architecture for feed-forward visual geometry reconstruction that eliminates the reliance on a fixed reference view. By predicting affine-invariant camera poses and scale-invariant local pointmaps without any reference frame designation, the method achieves robustness to input ordering and demonstrates state-of-the-art or competitive performance across camera pose estimation, depth estimation, and point cloud reconstruction tasks.

## Strengths

- **Clear identification of a structural limitation in prior work:** The paper convincingly demonstrates that existing methods (VGGT, DUSt3R, etc.) suffer from reference-view sensitivity. Figure 2 and the robustness evaluation (Table 6) provide direct evidence that reordering input frames causes significant performance variance in prior methods—VGGT shows DTU accuracy std of 0.033 while π³ achieves 0.003, an order-of-magnitude reduction.

- **Principled architectural design:** The permutation-equivariant formulation is mathematically sound. By removing frame index positional embeddings and reference view tokens, and supervising on relative poses (N×N pairwise constraints), the architecture guarantees output consistency under input permutation—this is validated empirically by near-zero variance across permuted inputs.

- **Strong empirical coverage:** The method achieves SOTA or competitive results across multiple tasks and benchmarks: Sintel pose ATE improves from VGGT's 0.167 to 0.074; video depth Abs Rel improves from 0.299 to 0.233. The evaluation spans 7+ datasets with both in-domain (Co3Dv2, ScanNet) and zero-shot (Sintel, TUM-dynamics) settings.

- **Efficiency improvements:** The model runs at 57.4 FPS with 959M parameters, compared to VGGT's 43.2 FPS and 1.26B parameters, demonstrating that the simpler reference-free design does not come at a computational cost.

## Weaknesses

- **Training dependency on VGGT initialization undermines the architectural claim:** Appendix A.2 reveals that the final model initializes encoder and alternating attention weights from pre-trained VGGT. More critically, Appendix A.4 shows that when training from scratch, π³ *underperforms* VGGT on 7-Scenes (Acc. 0.064 vs. 0.057, Comp. 0.068 vs. 0.046) and NRGBD (0.071 vs. 0.060, 0.047 vs. 0.042). Only with an auxiliary "global proxy" head (which re-introduces a reference frame via cross-attention) does π³ outperform VGGT from scratch. The paper frames permutation-equivariance as a fundamentally superior design, but the empirical advantage in Tables 1–5 depends substantially on warm-starting from a reference-based method. The paper should discuss this dependency more prominently.

- **"Cold start" optimization difficulty reveals a trade-off:** The paper acknowledges (Appendix A.4) that "purely relative training from random initialization struggles to converge due to the highly coupled O(N²) constraints." This suggests the permutation-equivariant formulation creates a harder optimization landscape, not simply a better one. The method needs a reference-dependent auxiliary task to stabilize training, which partially contradicts the claim of a purely reference-free paradigm.

- **Ablation conflates multiple design changes:** Table 7 compares Model 1 (no scale/affine invariance), Model 2 (scale-invariant only), and the full model—but all three also differ in architecture (reference token presence/absence) and supervision objectives. The ablation cannot cleanly attribute gains to permutation-equivariance versus the scale/affine-invariant loss formulation. A cleaner ablation would isolate the equivariant architecture from the loss changes.

- **Limited evaluation on long sequences:** The camera pose loss evaluates all N(N-1) ordered view pairs, creating O(N²) complexity. While the paper claims support for video sequences, all evaluations use relatively short clips (typically 10-frame sequences per the evaluation protocol). No analysis is provided for memory scaling or drift accumulation on longer inputs (50+ frames), limiting assessment of practical applicability for large-scale scenes.

- **Some benchmark results are overstated:** On Co3Dv2 (in-domain), VGGT achieves AUC 88.59 versus π³'s 88.41—a marginal but real reversal. On 7-Scenes sparse, VGGT achieves better Accuracy (0.044/0.025 vs. 0.047/0.029) and Completion (0.056/0.033 vs. 0.075/0.049). The abstract claims "state-of-the-art across a wide range of tasks," which should be qualified to acknowledge these exceptions.

## Nice-to-Haves

- **Fair efficiency comparison:** The FPS improvement (57.4 vs. 43.2) is confounded by architecture depth (36 vs. 48 alternating attention layers). Retraining VGGT with matched depth would clarify whether efficiency gains come from the permutation-equivariant design or simply fewer layers.

- **Dynamic scene quantitative evaluation:** The paper prominently features dynamic scenes in Figure 1 and uses an internal dynamic dataset for training, but all point map evaluations are on static benchmarks (DTU, ETH3D, 7-Scenes, NRGBD). Quantitative metrics on dynamic regions would strengthen claims of dynamic scene capability.

- **Scale prediction for deployment:** The scale-invariant formulation requires alignment during evaluation. A module predicting absolute scale would improve practical deployability without oracle alignment.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Exact equivariance vs. near-zero variance":** The harsh critic claimed that Table 6's nonzero standard deviations (e.g., 0.003) contradict the claim of *exact* permutation equivariance. However, values this small are consistent with floating-point non-associativity in parallel GPU computation and are practically negligible—the ~10x reduction versus VGGT (0.033) validates the claim. This is an overly pedantic criticism.

- **"Threshold ε not specified":** The confidence loss threshold is a minor hyperparameter; the method section describes the BCE formulation clearly enough for practitioners.

- **"Causal link between pose structure and architecture":** The PCA visualization of pose distributions (Figure 4/Appendix A.3) is presented as supporting evidence for the design. While not independently dispositive, it provides a reasonable qualitative signal that the formulation induces more structured outputs.

- **"Internal dataset fairness concern":** The use of an internal dynamic scene dataset is acknowledged. Many competitive works use internal data; without evidence of unusual overlap with test sets, this is not a substantive criticism.

- **"Mean accuracy should be analyzed under permutation":** This is already implicitly shown—Table 6 reports mean values (which remain stable across permutations by design), and the standard deviation is computed over permutations. The mean not changing is inherent to the method's properties.

- **"Global consistency mechanism unclear":** The relative pose formulation (N×N pairwise constraints) provides a well-defined mechanism for global consistency; this is standard practice in SfM-derived supervision.

## Novel Insights

Beyond the paper's own contributions, the reviews surface an important tension that the paper acknowledges but underweights: the permutation-equivariant formulation trades one type of inductive bias (reference frame dependency) for another (optimization landscape complexity). The "cold start" problem requiring an auxiliary proxy task suggests that global coordinate supervision provides useful learning signal early in training—a finding that could inform future work on curriculum strategies for relative pose learning. Additionally, the near-zero permutation variance (Table 6) serves as a strong empirical verification that the architectural modifications achieve their intended property; this type of permutation-robustness validation should become standard practice for multi-view reconstruction papers.

## Suggestions

- Prominently discuss the VGGT initialization dependency and "cold start" trade-off in the main paper, not just the appendix. Quantify the performance gap between scratch and initialized training to set expectations for practitioners.

- Add a controlled ablation isolating the permutation-equivariant architecture from the scale/affine-invariant loss to properly attribute performance gains.

- Evaluate on longer sequences (50+ frames) and report memory/runtime scaling to establish practical limits for video applications.

- Include at least one quantitative dynamic scene benchmark to substantiate the dynamic reconstruction capability claimed in the abstract and introduction.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
