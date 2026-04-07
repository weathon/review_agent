=== CALIBRATION EXAMPLE 80 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract

The title "π³: Permutation-Equivariant Visual Geometry Learning" accurately captures the paper's central thesis. The abstract's claims are mostly well-supported, though one claim deserves scrutiny: it states π³ achieves "state-of-the-art performance on a wide range of tasks, including… monocular/video depth estimation." For monocular depth, Table 5 shows π³ (0.277 AbsRel on Sintel) is essentially tied with MoGe v1 (0.273) and v2 (0.277), and is worse than MoGe on KITTI. Saying SOTA here is a stretch—"competitive with SOTA specialized models" would be more accurate. The abstract also omits the critical dependency on VGGT pre-trained weights, which materially affects how one interprets the contribution.

---

### Introduction & Motivation

The core motivation—that reference view selection is an unnecessary inductive bias—is compelling and clearly articulated. Figure 2 effectively illustrates sensitivity to reference selection in existing methods. However, there are two gaps:

1. **The motivation conflates two distinct problems.** The paper frames reference-view dependence as a fundamental problem, but it is worth distinguishing: (a) the problem of *sensitivity to which view is chosen* (robustness), and (b) the problem of *achievable quality under the best reference* (accuracy). These are separable, and the paper only establishes that π³ reduces (a). Whether (a) directly causes the accuracy improvements in (b) is not rigorously shown—it is asserted, but confounded by architecture changes and training differences.

2. **Figure 2 lacks quantitative anchoring.** The bar chart shows qualitative differences across reference frames, but it is unclear how reference frames are selected (worst-case? sampled randomly?), what the actual numerical values are, and whether the variance shown for prior methods is typical or cherry-picked. A formal robustness table (like Table 6 on permutation) would be more convincing here.

---

### Method

**3.1 Permutation-Equivariant Architecture**

The formal definition in Equations 1–3 is clean and correct. The practical implementation—removing frame-index positional embeddings and reference tokens—is intuitive but technically straightforward. The paper should acknowledge that permutation equivariance is only *approximately* achieved in practice (due to finite-precision arithmetic, batch normalization, and stochastic training), yet claims "true permutation equivariance" in the introduction. The near-zero (but non-zero) standard deviations in Table 6 confirm this is approximate.

**3.2 Scale-Invariant Local Geometry**

Equation 4 solves for a *single global scale factor* s* across all N images simultaneously. This is non-trivial because each local point map Xᵢ is defined in its own camera coordinate system. How does one single s* simultaneously align all N local point maps to their respective ground truths when they are in different coordinate frames? The formulation implicitly assumes that ground-truth point maps {xᵢ,ⱼ} are expressed in each frame's local coordinates as well, which must be the case—but this should be stated explicitly. More importantly, since each local pointmap is in its own camera frame and only related to others through the predicted camera poses, the meaning of a single shared s* warrants fuller explanation. Is this equivalent to constraining the metric scale of all depth maps to share the same scale? This is a subtle but important technical point.

The depth-weighted L1 loss formulation (1/zᵢ,ⱼ weighting) and the ROE solver are both borrowed from MoGe (Wang et al., 2025c). This is fine, but the paper should be clearer that this is direct reuse rather than a novel contribution.

**3.3 Affine-Invariant Camera Pose**

Supervising relative poses (Equation 7) to handle reference-frame ambiguity is a sensible choice, though not novel on its own. The key technical question is how s* from the pointmap loss is used to disambiguate translation scale (Equation 10). This is a reasonable design, but it creates a coupling between pointmap quality and camera pose quality that is not discussed. If the pointmap scale estimate s* is noisy early in training, the translation supervision becomes noisy too.

**Figure 4 / A.3: Pose distribution analysis**

This analysis is interpretively problematic. The paper argues that the low-dimensional structure of predicted poses (Figure 4, 6) is a positive indicator of "capturing the underlying geometric manifold." However, the analysis is conducted only on *predicted* (not ground-truth) poses. Low-dimensional predicted pose distributions could also indicate an underfitting model that collapses predictions to a small manifold, not necessarily a geometrically meaningful one. A proper analysis would compare the intrinsic dimensionality of *predicted* vs. *ground-truth* trajectory distributions on the same test sequences and show they match. Without this, the analysis is suggestive but not rigorous.

**3.4 Training: Cold Start Problem (Critical)**

The most significant weakness of the paper is partially buried in Appendix A.4. The authors admit that training from scratch leads to "suboptimal convergence" due to a "cold start" problem with relative pose supervision. Their solution for scratch training requires introducing an auxiliary head that uses cross-attention with a reference view—precisely the paradigm they critique. While this auxiliary head is only a "proxy task" and the final model is equivariant, the following issues arise:

1. **The main experiments use VGGT initialization.** The encoder and alternating attention layers are loaded from a pre-trained VGGT model, with the encoder kept *frozen* throughout training. This means the representation learned by π³'s backbone is VGGT's representation. The performance gap between π³ and VGGT in the main tables is thus partly—or perhaps largely—attributable to VGGT's pre-training, not π³'s architectural innovation. This is a major confound.

2. **The scratch comparison (Table 8) is not the main result.** Table 8 shows that with a global proxy, π³ outperforms scratch VGGT on ETH3D and NRGBD, but not clearly on 7-Scenes (accuracy: 0.064 vs 0.057 for VGGT). This head-to-head comparison is done at lower resolution (224×224) and without the second fine-tuning stage, so the results from Table 8 do not cleanly validate that the architectural innovation alone drives the improvements claimed in Tables 1–5.

---

### Experiments & Results

**Camera Pose Estimation (Table 1)**

Results are strong, particularly the Sintel improvement (ATE 0.074 vs VGGT's 0.167). However, several comparability issues arise:

- On Co3Dv2 (seen), π³ achieves 88.41 AUC vs VGGT's **88.59 AUC** — π³ is *worse* on this in-domain benchmark. This is not prominently discussed.
- The "seen" vs. "unseen" distinction (flagged in Appendix A.5) should be more prominent in Table 1 itself, not just in the appendix.
- CUT3R (Wang et al., 2025b) is a recurrent model designed for sequential data, while π³ processes all frames at once. These are different operating regimes, and the comparison should acknowledge this.

**Point Map Estimation (Tables 2, 3)**

On 7-Scenes sparse, π³ underperforms VGGT on Accuracy (0.047 vs **0.044** mean, 0.029 vs **0.025** median). The paper's claim of SOTA across "a wide range of tasks" requires more careful qualification. The results are generally strong but not uniformly dominant.

**Robustness Evaluation (Table 6)**

This is the paper's most compelling and clean result. Near-zero standard deviations across both DTU and ETH3D (e.g., ETH3D Acc std: **0.000** vs VGGT's 0.049) provide compelling empirical evidence for permutation equivariance. This is a genuine, measurable contribution.

**Ablation Study (Table 7)**

The ablation is valuable but limited. It demonstrates the contribution of each proposed component. However, it only evaluates on point map metrics, not camera pose metrics. An ablation of pose accuracy (as in Table 1) across Model 1, Model 2, and Full Model would strengthen the case for affine-invariant camera modeling. Additionally, Model 1 and Model 2 still use VGGT initialization, so the baseline is high—the absolute gains are modest on indoor datasets like 7-Scenes.

**Inference Speed (Table 4)**

The 57.4 FPS vs. VGGT's 43.2 FPS improvement is highlighted, but π³ uses only 36 alternating attention layers vs. VGGT's 48—a direct model-size reduction. With 959M vs. 1.26B parameters, this speed improvement may simply reflect fewer layers. The authors should clarify whether this is a genuine architectural efficiency or simply a smaller model.

---

### Writing & Clarity

Section 4.4's introductory sentence ("We then compute the standard deviation of the metrics across these N runs. We then compute the standard deviation of the reconstruction metrics across these N outputs.") is an apparent duplicate that should be edited. More substantively, the key limitation (VGGT initialization dependency) appearing only in Appendix A.4 significantly misleads a casual reader about the method's self-contained novelty.

---

### Limitations & Broader Impact

Appendix A.8 acknowledges transparent objects, lack of fine detail, and grid artifacts. However, several important limitations go unmentioned:

1. **Initialization dependency**: The model cannot be trained from scratch without an auxiliary reference-view head, which is philosophically at odds with the paper's core thesis.
2. **Quadratic relative pose supervision**: L_cam (Equation 8) averages over all O(N²) ordered pairs. This is expensive for large N and may explain why training uses 2–24 images per batch.
3. **No online/streaming capability**: Unlike CUT3R, which supports sequential processing, π³ requires all frames at once. This limits its applicability to streaming or incremental reconstruction scenarios.
4. **Scale consistency across sequences**: The method resolves scale per-sequence, but says nothing about cross-sequence or absolute metric scale recovery.

---

### Overall Assessment

π³ presents a clear, well-motivated contribution: demonstrating that reference-view bias in feed-forward 3D reconstruction is both identifiable and removable. The robustness results (Table 6) are genuinely impressive and novel, the overall benchmark performance is competitive with or superior to the state of the art on most tasks, and the paper is well-organized. However, the contribution is substantially weaker than it appears at first read due to a critical confound: the main model initializes from and *freezes the encoder of* pre-trained VGGT, making it difficult to disentangle architectural gains from inherited representational power. The "cold start" problem (Appendix A.4) reveals that the proposed training objective is unstable without VGGT's prior—effectively requiring the very reference-view mechanism the paper critiques as an auxiliary training crutch. The paper would be significantly stronger if it presented clearly separated results for (a) the architectural contribution alone (on equal footing with VGGT in training), and (b) the full system with VGGT initialization. As it stands, the ICLR bar for novelty is met on the conceptual level and the robustness insight is solid, but the experimental attribution is insufficiently rigorous to fully support the paper's strongest claims. This is a borderline accept that would benefit from revisions addressing the initialization confound and the cold-start limitation more prominently.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces $\pi[3]$, a feed-forward neural network for visual geometry reconstruction that eliminates the reliance on a fixed reference view through a fully permutation-equivariant architecture. By predicting affine-invariant camera poses and scale-invariant local point maps in a purely relative manner, the model achieves state-of-the-art performance on camera pose estimation, depth estimation, and 3D reconstruction tasks while maintaining robustness to input ordering. The work challenges a fundamental inductive bias in the field and demonstrates superior speed (57.4 FPS) compared to existing methods.

### Strengths
1.  **Strong Empirical Performance:** The model establishes new state-of-the-art results on multiple benchmarks. For instance, Table 1 shows significant improvements in camera pose estimation ATE on Sintel (0.074) compared to VGGT (0.167), and Table 4 demonstrates superior video depth estimation accuracy across Sintel, Bonn, and KITTI.
2.  **Robustness to Input Order:** Section 4.4 and Table 6 provide compelling quantitative evidence that $\pi[3]$ is permutation equivariant. The standard deviation of point cloud estimation metrics is near-zero across permutations on DTU and ETH3D (e.g., DTU Acc. std. 0.003 vs. VGGT 0.033), directly validating the core architectural claim.
3.  **Efficiency:** The method offers a favorable accuracy-speed trade-off. Table 4 reports an inference speed of 57.4 FPS on KITTI, outperforming VGGT (43.2 FPS) and others like Aether (6.14 FPS), while maintaining competitive model size (959M parameters).
4.  **Comprehensive Experiments:** The paper evaluates on a wide array of tasks (pose, depth, point map) and datasets (indoor, outdoor, synthetic, real-world), including rigorous ablation studies (Table 7) that isolate the contribution of affine-invariant poses and scale-invariant point maps.

### Weaknesses
1.  **Training Complexity:** Appendix A.4 admits that training the model "from scratch" leads to suboptimal convergence without an auxiliary "global proxy" task, requiring initialization from the pre-trained VGGT model for practical deployment. This contradicts the implication that the method is self-contained or trivially trainable, and relies on external architectural priors.
2.  **Limited Discussion of Trade-offs:** While the paper argues reference views introduce instability, it does not deeply analyze scenarios where a fixed reference might be beneficial (e.g., specific SLAM pipelines requiring incremental global alignment). The benefits are shown empirically, but the theoretical trade-off between reference-free flexibility and reference-based efficiency/convergence is not explored.
3.  **Qualitative Limitations Acknowledged:** Section A.8 admits the method produces "grid-like artifacts" due to the upsampling mechanism and struggles with transparent objects. While honest, the paper lacks qualitative visual comparisons showing these artifacts specifically, focusing mostly on quantitative metrics.
4.  **Baseline Coverage:** While compared against SOTA (VGGT, Fast3R, MoGe), the field is moving rapidly. Some recent works leveraging diffusion or specific global optimization refinements are compared indirectly, but a deeper analysis of how $\pi[3]$ fares against non-feed-forward or iterative optimization hybrids in edge cases could strengthen the claim of "universal" applicability.

### Novelty & Significance
The paper's primary novelty lies in formally identifying and dismantling the "fixed reference view" bias common in modern feed-forward 3D reconstruction (like DUSt3R and VGGT). By enforcing permutation equivariance, it proposes a new paradigm for geometry learning that guarantees consistency regardless of input ordering. This is significant because it addresses a fundamental stability issue in visual geometry, moving the field closer to fully invariant, end-to-end systems. The performance gains on robustness and speed suggest this architectural change is not just theoretical but practically impactful for real-time applications like robotics and AR.

### Suggestions for Improvement
1.  **Clarify Training Strategy:** Explicitly discuss the dependency on VGGT initialization in the main text or conclusion. Explain *why* the self-supervised relative loss is hard to optimize from scratch to inform future researchers.
2.  **Expand Discussion on Reference Views:** Add a discussion on whether the permutation-equivariant approach could be combined with reference-based methods for hybrid scenarios (e.g., initializing a map with a reference then refining).
3.  **Enhance Qualitative Analysis:** Include specific visualizations in the main body (not just tables) highlighting the grid artifacts mentioned in A.8 or cases where $\pi[3]$ fails, to provide a more balanced view of its limitations.
4.  **Theoretical Insight:** Provide more intuition or analysis on *why* removing the reference view reduces the standard deviation specifically (e.g., relating to the conditioning of the transformation space). The current explanation is mostly empirical.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **True Ab Initio Training:** Demonstrate convergence without initializing from VGGT or using reference-based proxy tasks (Appendix A.4 admits failure otherwise), as relying on a reference-dependent teacher undermines the core "bias-free" claim.
2. **Capacity-Controlled Baselines:** Compare against VGGT with matched layer counts and parameters to ensure accuracy and speed gains are not solely due to reduced model size (36 vs. 48 layers, 959M vs. 1.26B params).
3. **Dynamic Object Geometry:** Evaluate point map accuracy specifically on moving objects (e.g., Dynamic Replica) rather than just camera pose on dynamic scenes, as current evidence for dynamic geometry reconstruction is insufficient.

### Deeper Analysis Needed (top 3-5 only)
1. **Inference Scale Strategy:** Clarify how metric scale is recovered at test time without ground-truth alignment, as the current training relies on GT-derived scale factors ($s^*$) that are unavailable during inference.
2. **Initialization Dependency:** Quantify the performance drop when removing VGGT pretrained weights to isolate the specific contribution of the permutation-equivariant architecture versus transferred features.
3. **Symmetry Ambiguity:** Analyze failure cases on symmetrical scenes where permutation equivariance might prevent resolving view-ordering ambiguities that reference-based methods could leverage.

### Visualizations & Case Studies
1. **Worst-Case Reference View:** Visualize VGGT failures with poor reference selection versus $\pi_3$ stability to empirically validate the claimed robustness advantage beyond standard deviation metrics.
2. **Dynamic Object Consistency:** Show temporal consistency of point maps on moving subjects to verify the method handles non-rigid geometry without artifacts.
3. **Long-Sequence Drift:** Visualize global alignment drift in long videos without bundle adjustment to test whether local scale-invariant predictions maintain global consistency.

### Obvious Next Steps
1. **Self-Supervised Scale:** Develop a mechanism to predict absolute scale without ground-truth alignment to ensure practical utility in wild settings.
2. **Independent Initialization:** Train without relying on reference-dependent teacher models to prove the architecture stands on its own merits.
3. **Explicit Motion Heads:** Integrate explicit motion segmentation or dynamic geometry heads to robustly support the claimed dynamic scene capabilities.

# Final Consolidated Review
## Summary

The paper introduces π³, a feed-forward neural network for visual geometry reconstruction that eliminates dependence on a fixed reference view through a fully permutation-equivariant architecture. By predicting affine-invariant camera poses and scale-invariant local point maps without any reference frame designation, the model achieves robustness to input ordering and state-of-the-art or competitive performance on camera pose estimation, depth estimation, and point map reconstruction benchmarks.

## Strengths

- **Empirical robustness to input ordering** — Table 6 provides compelling quantitative evidence: on DTU and ETH3D, π³ achieves near-zero standard deviation across input permutations (e.g., ETH3D Acc std: 0.000 vs VGGT's 0.049), directly validating the permutation-equivariant architecture claim.

- **Strong benchmark performance on key metrics** — The method achieves SOTA on Sintel camera pose estimation (ATE 0.074 vs VGGT's 0.167, a 56% reduction) and strong results across depth estimation and point map reconstruction tasks, demonstrating practical effectiveness.

- **Comprehensive ablation study** — Table 7 isolates contributions of affine-invariant poses and scale-invariant pointmaps across multiple datasets, showing outdoor scenes benefit more from scale-invariant modeling (consistent with prior literature on scale ambiguity).

## Weaknesses

- **Critical dependency on VGGT initialization** — The paper's main results rely on initializing from pre-trained VGGT weights with a frozen encoder (Section 3.4, Appendix A.2). The "cold start" problem disclosed in Appendix A.4 reveals that training from scratch fails without an auxiliary reference-view proxy task—the very paradigm the paper argues against. This significantly undermines the claim that the architecture independently eliminates reference-view bias; practical success depends on inheriting representations learned with a reference-based approach. Table 8 shows scratch training comparisons, but only at lower resolution (224×224) and without the fine-tuning stage, making it difficult to isolate the architectural contribution.

- **SOTA claims require qualification** — On monocular depth (Table 5), π³ achieves 0.277 AbsRel on Sintel vs MoGe's 0.273, and 0.060 on KITTI vs MoGe's 0.049—π³ is worse on both. On Co3Dv2-seen (Table 1), π³ achieves 88.41 AUC vs VGGT's 88.59. The abstract's claim of "state-of-the-art on a wide range of tasks" overstates results that are strong but not uniformly dominant.

- **Speed improvement may reflect smaller model** — π³ uses 36 alternating attention layers vs VGGT's 48, and 959M vs 1.26B parameters. The 57.4 FPS vs 43.2 FPS speedup may simply result from reduced model capacity rather than architectural efficiency. The paper does not control for this confound.

- **Quadratic supervision scaling** — The camera loss L_cam (Equation 8) averages over all O(N²) ordered view pairs, limiting training to 2-24 images per batch. This affects scalability to longer sequences and is not analyzed as a limitation.

- **No online/streaming capability** — Unlike CUT3R, which processes frames sequentially, π³ requires all frames simultaneously, limiting applicability to streaming or incremental reconstruction scenarios. This is a material architectural trade-off not discussed.

- **Pose distribution analysis is incomplete** — Figure 4/Appendix A.3 argues that low-dimensional predicted pose structure indicates geometric insight, but analyzes only predicted (not ground-truth) poses. Low-dimensional predicted distributions could indicate underfitting or collapse to a manifold, not necessarily capturing true trajectory geometry. Comparison to GT trajectory dimensionality is needed.

## Nice-to-Haves

- **Controlled capacity comparison** — A comparison against VGGT with matched layer counts and parameters would clarify whether efficiency gains stem from architecture or reduced capacity.

- **Scale recovery at inference clarification** — Training uses GT-derived scale factors s* that are unavailable during inference. The paper should explicitly address how metric scale is recovered in deployment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Permutation equivariance is approximate due to finite precision"** — Near-zero vs zero standard deviation (Table 6) is a minor numerical precision issue well-understood by practitioners. The practical achievement of ~0.000 std is sufficient for the stated claims.

- **"Figure 2 lacks quantitative anchoring"** — While the figure itself is qualitative, Table 6 provides rigorous quantitative robustness evaluation that supports the same claim. This is a presentation preference, not a substantive concern.

- **"Depth-weighted L1 and ROE solver borrowed from MoGe"** — Proper attribution of standard components is not a weakness; the paper cites Wang et al. (2025c) appropriately.

- **"Equation 4 single scale factor technical concern"** — The concern that a single s* cannot align all N point maps in different coordinate frames misunderstands the formulation: GT point maps {x_i,j} are in each frame's local coordinates, and s* represents the consistent unknown scale across all predictions relative to their GTs. This is technically sound and standard practice in scale-ambiguous depth estimation.

- **"Coupling between pointmap and pose quality during training"** — While the scale factor from pointmap loss is used for translation supervision, this is a design choice with a clear rationale. The coupling concern is speculative without evidence that it harms convergence.

## Novel Insights

The paper's identification of reference-view selection as an "inductive bias" rather than a necessary design choice is conceptually valuable. The robustness results (Table 6) genuinely demonstrate that this bias creates practical brittleness in existing methods—VGGT's accuracy standard deviation of 0.033 on DTU vs π³'s 0.003 across reference frame permutations shows that the bias is not merely theoretical. The connection between reference-free design and permutation equivariance provides a clean architectural mechanism for eliminating this bias, though the practical training reliance on VGGT initialization tempers the strength of this insight.

## Suggestions

- **Conduct controlled experiments** — Train π³ and VGGT from scratch with matched capacity (same layer counts) to isolate architectural contributions from initialization effects. Report these results even if they show lower absolute performance.

- **Clarify the training dependency in the main text** — The cold-start problem and VGGT initialization are disclosed only in the appendix. Prominent discussion in the method section would improve transparency about the method's practical requirements.

- **Address inference-scale recovery** — Explain how metric scale is obtained at deployment time when ground-truth alignment is unavailable.

- **Compare to GT trajectory dimensionality** — When arguing that predicted pose structure indicates geometric insight (Figure 4), show that the intrinsic dimensionality of predictions matches that of ground-truth trajectories on the same sequences to rule out underfitting explanations.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
