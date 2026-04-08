=== CALIBRATION EXAMPLE 33 ===

# Harsh Critic Review
Now I have a thorough picture of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "DENS3R: A Foundation Model for 3D Geometry Prediction" is broadly accurate but overpromises on the "foundation model" framing—the paper is ultimately an incremental extension of the DUSt3R/MASt3R lineage with a normal prediction head, a shared decoder, and position-interpolated RoPE. The abstract claims "joint regression" and "explicit modeling of structural coupling," but the primary mechanism for coupling is feature concatenation (Eq. 9) rather than any deep architectural integration. The abstract also states that a two-stage training framework "progressively builds a pointmap representation that is both generalizable and intrinsically invariant"—the "intrinsic invariance" terminology is introduced here but is not rigorously defined until later (and even then remains vague). Overall, the abstract raises expectations the body only partially meets.

---

### Introduction & Motivation

The motivation for unified geometric prediction is sound and clearly explained. The critique of diffusion-based approaches for deterministic geometric tasks is a reasonable position. However, several claims require scrutiny:

- **Overclaiming on DUSt3R/MASt3R shortcomings**: The introduction states that DUSt3R-family methods "overlook" surface normals. In reality, surface normals can be trivially derived from predicted pointmaps via finite differencing. The authors acknowledge this themselves in Fig. 3, which shows that normals *from* the Stage-1 pointmap are "not accurate enough"—but this is a quantitative accuracy argument, not a fundamental architectural omission.

- **Unclear novelty boundary**: The four claimed contributions (Dens3R model, intrinsic-invariant pointmap, PI-RoPE, multi-view post-processing) overlap significantly with prior work. Position-interpolated RoPE is borrowed directly from the NLP context-window extension literature (Chen et al., 2023), and the multi-view inference relies entirely on the existing MASt3R-SfM pipeline. The authors should more precisely delineate what is new versus adapted.

---

### Method

**Shared Encoder-Decoder (Sec. 3.1)**
The weight-sharing in the decoder reduces parameters by ~113M (from 737M to 624M, Table 4) without loss in prediction quality. This is a clean engineering contribution. However, Table 4 shows *identical* compute cost (1.362 TFlops) with/without sharing, so the only benefit is memory, not speed. The paper should clarify whether there is any accuracy trade-off measured on downstream tasks, not just cost.

**Multi-resolution Input (Sec. 3.1, Eq. 2)**
Applying positional interpolation from the LLM domain to 2D vision transformers is reasonable and practically effective. However, the ablation in Fig. 8a is only qualitative—the paper should provide a *quantitative* comparison showing resolution-accuracy trade-offs with/without PI-RoPE at different resolutions. More importantly, the authors themselves note (Appendix A.7) that PI-RoPE alone is insufficient without the coarse-to-fine training scheme, meaning the contribution is not standalone.

**Two-Stage Training (Sec. 3.2)**
This is the core contribution. Several issues:

1. **"Intrinsic-invariant" is under-formalized**: The concept is introduced by analogy to MoGe's affine-invariant pointmap. The proposed mechanism is simply to concatenate the predicted normal with the pointmap representation (Eq. 9: $P_i^n = P_i \oplus n$). This is very lightweight. The paper claims this forces the model to internalize "intrinsic invariance," but there is no formal proof or even intuitive argument for why concatenating normals specifically achieves invariance beyond improving task-specific prediction quality.

2. **Loss weight selection is unmotivated**: The loss weights $\eta_1=1.0, \eta_2=0.1, \eta_3=0.075$ (Stage 1) and $\lambda_1=1.0, \lambda_2=0.1, \lambda_3=1.0$ (Stage 2) are chosen without any sensitivity analysis or ablation.

3. **Confidence loss removal**: The paper makes a notable claim that confidence-weighted losses cause models to "ignore complex scenarios such as reflective surfaces." This is a significant claim about a widely used design choice in DUSt3R, MASt3R, and VGGT, yet no ablation isolates this specific factor. How much of the improvement on reflective surfaces is due to confidence removal vs. normal supervision?

4. **One-to-many vs. one-to-one supervision change**: The switch from multi-view to single-view supervision in Stage 2 is presented as enabling monocular data use, but this architectural change also fundamentally changes the model's operating mode. The paper should clarify whether the Stage 2 model can still leverage multi-view input during training (if paired data is available) or is restricted to single-view pairs.

---

### Experiments & Results

**Normal Estimation (Table 1)**
Dens3R achieves the best results on all five benchmarks (NYUv2, ScanNet, IBims-1, Sintel, DIODE-outdoor). However, several concerns arise:

- **Potential train/test leakage**: The training set (Table 5) includes ScanNet++ (used as Type B training data), while **ScanNet** is used as a zero-shot test set. ScanNet and ScanNet++ share the same environments and sensors. If there is scene-level overlap between the ScanNet training split and ScanNet++ training data, the comparison is compromised. The paper does not address this.

- **VGGT absent from normal comparison**: VGGT (Wang et al., 2025a) also jointly predicts multiple geometric quantities and presumably can produce normals. The paper compares Dens3R against VGGT for depth (Table 7) and matching (Tables 9/10) but notably omits it from the normal comparison (Table 1). This selective omission is conspicuous.

**Image Matching (Table 2, ZEB Dataset)**
The improvement over MASt3R (64.5% vs 59.9% mean AUC@5°) is solid. However, some per-category results raise questions: on "NIG" (night scenes), Dens3R scores 50.4 vs. MASt3R's 43.7, which is good, but on "WEA" (weather), the gap is smaller (57.4 vs. 52.8).

**Pointmap Evaluation (Sec. 4.2)**
The pointmap comparison is **entirely qualitative** (Fig. 5). For a paper claiming high-quality pointmap prediction as its central contribution, the absence of any quantitative evaluation against MoGe, VGGT, or DUSt3R on standard benchmarks is a significant gap. There are well-established metrics (pointmap accuracy, scale-invariant error, etc.) used in prior work.

**Depth Estimation (Table 7, Appendix)**
Results are mixed. Dens3R achieves the best REL and RMSE on DIODE-outdoor but is **not best** on NYUv2 (MoGe achieves REL=0.035 vs. Dens3R's 0.042). On DIODE-indoor, Dens3R outperforms MoGe but not consistently. The paper does not comment on these inconsistencies.

**Tables 8, 9, 10 are critically incomplete**
The camera pose estimation table (Table 8) shows only DUSt3R's row—the rows for MASt3R, VGGT, and Dens3R are missing entirely. Similarly, Tables 9 (ScanNet-1500) and 10 (MegaDepth-1500) show only header rows with no numerical data. These may be PDF parsing artifacts, but if so, the absent data makes several claimed contributions completely unverifiable from this manuscript alone. The authors must ensure these tables are populated in the final submission.

**Ablation (Table 3)**
The ablation table for normal prediction is partially unreadable due to formatting—the row labels indicating which configuration each row corresponds to are not present in the parsed text. This obscures which combination of components produces each result. Additionally, the ablation only covers normals; there is no ablation of individual components on depth or pointmap accuracy.

---

### Writing & Clarity

The description of the "intrinsic-invariant pointmap" (Sec. 3.2) is vague. The only technical difference from Stage 1 is the addition of normal supervision and the concatenation in Eq. 9. The term "intrinsically invariant" is never given a formal definition, which makes the contribution hard to evaluate. The "one-to-many vs. one-to-one" supervision distinction is mentioned but the implications are not clearly worked out.

---

### Limitations & Broader Impact

The limitation section (Appendix A.8) is honest about thin structures but brief. Missing acknowledgments include:
- Computational cost: 32 H20 GPUs for two-plus weeks of training limits reproducibility.
- Dataset quality classification (A/B/C) and its effect on performance is partially discussed but not rigorously evaluated (e.g., what if all training data were type A?).
- No discussion of failure modes beyond thin structures (e.g., textureless indoor surfaces, extreme lighting).
- The paper claims broad foundation model applicability, including segmentation (Fig. 8c), but the segmentation results are preliminary and not quantitatively evaluated.

---

### Overall Assessment

Dens3R presents a practically useful extension of the DUSt3R/MASt3R family by incorporating surface normal prediction via a two-stage training strategy with a shared encoder-decoder and position-interpolated RoPE. The normal estimation results are strong, and the ZEB matching improvement over MASt3R is genuine. However, the paper has several issues that must be addressed before it can meet ICLR's standards. Most critically: (1) critical tables (8, 9, 10) are missing numerical data, making key claims unverifiable; (2) the absence of quantitative pointmap evaluation is a major gap for a paper whose central contribution is pointmap quality; (3) the "intrinsic-invariant pointmap" concept—the paper's main theoretical contribution—is poorly formalized, amounting to feature concatenation with normal outputs; (4) VGGT is suspiciously absent from normal comparisons despite being directly comparable; and (5) potential ScanNet/ScanNet++ train-test leakage is unaddressed. The contribution is incremental relative to MASt3R and VGGT, and falls short of the "foundation model" framing. In its current form, the paper would need substantial revisions—particularly to its experimental rigor and theoretical clarity—to be competitive at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
Dens3R is a feed-forward visual foundation model that jointly regresses multiple 3D geometric quantities—including depth, surface normals, 3D pointmaps, and dense image-matching features—from unposed and unconstrained images. The method introduces a two-stage training framework that first learns scale-invariant pointmaps and subsequently refines them into intrinsic-invariant representations by injecting explicit normal supervision and enforcing one-to-one viewpoint mapping. Coupled with position-interpolated rotary positional encoding and a weight-shared encoder-decoder backbone, the model supports high-resolution inputs and enables a post-processing pipeline for geometrically consistent multi-view inference.

### Strengths
1. **Strong Empirical Performance Across Diverse Tasks:** The model consistently outperforms recent state-of-the-art methods (DUSt3R, MASt3R, MoGe, VGGT, StableNormal) on standard benchmarks for normal prediction, depth estimation, and two-view image matching. Quantitative gains are evident across indoor and outdoor datasets (e.g., NYUv2, ScanNet, DIODE, ZEB), and qualitative results demonstrate improved handling of reflective surfaces and complex backgrounds (Tables 1, 2, 7; Figures 4, 5, 13).
2. **Practical Engineering Solutions to Known Bottlenecks:** The shared encoder-decoder architecture meaningfully reduces parameter count and GPU memory usage compared to dual-decoder baselines, directly addressing a scaling limitation in prior 3D vision models (Table 4). Additionally, adapting position-interpolated RoPE effectively mitigates the high-resolution degradation commonly observed in transformer-based geometric predictors, enabling stable 2K inference (Figures 6, 21; Section 3.1).
3. **Versatility and Downstream Utility:** By decoupling the heavy backbone training from lightweight task-specific heads, Dens3R demonstrates strong adaptability. The authors successfully extend the frozen backbone to segmentation, camera relocalization (Map-free benchmark), and neural surface reconstruction (NeuS/AutoRecon pipelines), proving its value as a reusable geometric prior for broader applications (Section A.2, Figures 8, 9).

### Weaknesses
1. **Incremental Novelty Relative to the DUSt3R/MoGe Family:** The core methodological contributions are largely engineering adaptations of existing paradigms. The "intrinsic-invariant" training is conceptually equivalent to adding direct normal supervision to resolve scale/shift ambiguity, and the position-interpolated RoPE is directly transplanted from LLM context-window extensions with minimal vision-specific adaptation. The paper would benefit from deeper theoretical justification of why the two-stage decoupling is fundamentally superior to end-to-end multi-task optimization.
2. **High Computational Cost Contradicts "Lightweight" Claims:** While inference is framed as efficient, training requires ~2 weeks on 32 H20 GPUs across a heavily curated, 20+ dataset mixture with complex quality tiers and balancing ratios. This massive upstream compute cost limits reproducibility for academic labs and contrasts with the paper's emphasis on efficiency. The true accessibility of the method remains unclear without open-sourcing the exact data processing pipeline and pre-trained checkpoints.
3. **Incomplete Reproducibility and Empirical Rigor:** Critical training hyperparameters (optimizer type, learning rate schedules, weight decay, batch size, gradient clipping, data augmentation strategies) are omitted. Furthermore, all benchmark results report single point estimates without standard deviations or variance across multiple random seeds. For an ICLR submission, this lack of statistical reporting makes it difficult to assess whether the reported gains over strong baselines are robust or marginal.
4. **Multi-View Inference Relies on Non-Neural Heuristics:** The claimed multi-view consistency is achieved via a classical pipeline (dense matching → triangulation → MASt3R-SfM) rather than a unified neural fusion mechanism. This introduces a dependency on external optimization routines that can fail in low-texture or highly repetitive regions, weakening the end-to-end consistency claim and limiting failure mode analysis in the neural model itself.

### Novelty & Significance
**Novelty:** Moderate. The architectural components and training recipe are sound and well-executed, but they represent iterative engineering improvements over the DUSt3R/MASt3R/MoGe lineage rather than a new paradigm. The reuse of RoPE interpolation from LLMs and the intuitive two-stage normal injection lower the methodological novelty threshold typically expected for ICLR. However, the specific integration and ablation provide a useful, cohesive system design.
**Clarity:** High. The paper is logically structured, with clear motivation, well-labeled figures, and a coherent progression from problem definition to solution and evaluation. The loss formulations and training pipeline are easy to follow, though some mathematical notation suffers from minor OCR artifacts (ignored per instructions).
**Reproducibility:** Moderate. While the model architecture, loss functions, and dataset compositions are described, the absence of key training hyperparameters, optimization schedules, seed variability reporting, and the complexity of the custom dataset curation pipeline pose practical barriers to exact reproduction.
**Significance:** High. Joint, consistent 3D geometry prediction from unposed images is a critical challenge in computer vision. Dens3R delivers a robust, high-quality backbone that meaningfully advances practical applications in robotics, SLAM, AR/VR, and automated reconstruction. If properly open-sourced, it would serve as a highly valuable baseline for the community.

### Suggestions for Improvement
1. **Report Full Reproducibility Details & Statistical Robustness:** Explicitly list the optimizer, learning rate schedule, batch size, gradient accumulation, and data augmentation pipeline. Re-run the core benchmarks (at least NYUv2, ScanNet, ZEB) across 3 different seeds and report mean ± standard deviation to statistically validate the claimed improvements over baselines.
2. **Deepen Ablation Studies & Loss Analysis:** Provide a systematic ablation on the loss weights ($\eta_1, \eta_2, \eta_3$ and $\lambda_1, \lambda_2, \lambda_3$) to show sensitivity. Compare the two-stage training against a single-stage end-to-end multi-task baseline with dynamic loss balancing (e.g., gradient normalization or uncertainty weighting) to rigorously justify the stage decoupling. Additionally, benchmark position-interpolated RoPE against alternative high-resolution encoding strategies (e.g., ALiBi, learned extrapolation, or grid interpolation) in a controlled setting.
3. **Clarify Compute vs. Efficiency Trade-offs & Open-Source Commitment:** Revise the "lightweight" claim to specifically refer to inference-time parameters and FLOPs, and explicitly separate it from the upstream training cost. To align with ICLR's emphasis on accessibility, commit to releasing pre-trained checkpoints, training scripts, and the exact data sampling/filtering pipeline. Acknowledge domain biases introduced by the heavy reliance on synthetic datasets (Type A/B) and discuss potential strategies for mitigating real-world distribution shifts.
4. **Integrate or Analyze the Multi-View Fusion Rigorously:** Either develop a lightweight neural refinement module to replace the classical triangulation/SfM post-processing for stronger end-to-end consistency, or add a dedicated failure-mode analysis of the current pipeline under challenging conditions (motion blur, occlusion, low texture). Quantify the error propagation from the matching head to the final reconstructed point cloud to provide a complete picture of system-level robustness.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Scaling Laws:** Provide performance vs. model size and data size curves; without this, the "Foundation Model" claim is unsupported marketing rather than empirical fact.
2. **Ensemble Baseline:** Compare Dens3R against an ensemble of SOTA specialized models (e.g., Depth Anything V2 + StableNormal); without this, the benefit of "unified" training over inference-time combination is unproven.
3. **Intrinsic Multi-view Consistency Metric:** Quantify cycle-consistency or reprojection error across N views without the MASt3R post-processing pipeline; relying on external SfM success does not prove the model itself learns consistent geometry.
4. **Inference Throughput:** Report FPS and latency on standard hardware alongside parameter counts; claiming "efficiency" requires throughput metrics, not just memory footprint.

### Deeper Analysis Needed (top 3-5 only)
1. **Task Interference Analysis:** Visualize gradient conflicts or loss landscapes between depth, normal, and matching heads; without this, the claim that joint regression avoids mutual interference is speculative.
2. **Normal-to-Pointmap Causal Link:** Isolate the specific contribution of normal supervision to pointmap accuracy via strict ablation (e.g., freeze backbone, add normal head); current results conflate training stages with architectural benefits.
3. **Sim-to-Real Gap:** Analyze performance disparity between synthetic (Type A/B) and real (Type C) data subsets; heavy reliance on synthetic data (Table 5) risks undisclosed domain bias.

### Visualizations & Case Studies
1. **Failure Mode Comparison:** Show side-by-side failures where the unified approach degrades performance compared to specialized SOTA models; this exposes whether "unified" introduces detrimental constraints.
2. **Multi-view Consistency Heatmaps:** Display reprojection error heatmaps across multiple views to visually verify geometric consistency claims beyond qualitative pointcloud renders.
3. **High-Frequency Detail Crops:** Provide 2K/4K resolution crop comparisons against standard RoPE extrapolation; this validates whether position interpolation actually preserves geometry or just smooths artifacts.

### Obvious Next Steps
1. **Scaling Study:** Train variants (S/M/L/XL) to demonstrate predictable performance gains with compute; this is mandatory to justify the "Foundation Model" terminology at ICLR.
2. **End-to-End Downstream Benchmark:** Evaluate on a concrete downstream task (e.g., Visual Localization or SLAM) without task-specific fine-tuning; this proves utility beyond head replacement.
3. **Data Manifest Release:** Publish the exact training dataset composition and licensing; reproducibility is critical for foundation models, and vague "reorganized" datasets undermine trust.

# Final Consolidated Review
## Summary

Dens3R proposes a unified 3D geometric foundation model that jointly predicts depth, surface normals, pointmaps, and dense image-matching features from unposed images. The key contributions include a two-stage training framework (scale-invariant → intrinsic-invariant pointmaps), a shared encoder-decoder backbone with position-interpolated RoPE for high-resolution handling, and explicit normal supervision integrated into the pointmap representation. The method demonstrates state-of-the-art or competitive performance across normal estimation, depth prediction, and image matching benchmarks.

## Strengths

- **Strong empirical performance across diverse benchmarks:** Dens3R achieves the best reported results on all five normal prediction benchmarks (NYUv2, ScanNet, IBims-1, Sintel, DIODE-outdoor) with consistent improvements over DSINE, StableNormal, and GeoWizard (Table 1). The method also achieves state-of-the-art results on the ZEB matching benchmark (64.5% vs. 59.9% mean AUC@5° for MASt3R) and competitive depth estimation results (Table 7).

- **Practical efficiency improvements:** The shared encoder-decoder architecture reduces parameters from 737M to 624M and GPU memory from 4.6GB to 4.1GB (Table 4) without sacrificing prediction quality. The position-interpolated RoPE enables stable inference at 2K resolution, addressing a known failure mode of prior ViT-based geometric predictors (Figures 6, 21).

- **Demonstrated downstream versatility:** The frozen backbone successfully transfers to segmentation (Fig. 8c), camera relocalization (Table 8, Map-free benchmark), and neural surface reconstruction (Fig. 8d) with minimal task-specific fine-tuning, supporting the foundation model framing.

- **Coherent system design for multi-task geometry:** The two-stage training strategy (Stage 1: scale-invariant pointmap; Stage 2: intrinsic-invariant with normal supervision) is well-motivated, and the ablation study (Table 3) demonstrates the contribution of each component to normal prediction accuracy.

## Weaknesses

- **Critical tables are incomplete or missing numerical data:** Tables 8 (camera pose estimation), 9 (ScanNet-1500 matching), and 10 (MegaDepth-1500 matching) are either empty or contain only baseline data, making the corresponding claims unverifiable from the manuscript. This must be corrected for reproducibility.

- **No quantitative evaluation of pointmap quality:** Despite pointmap prediction being a central claim in the title and abstract, the paper provides only qualitative pointmap comparisons (Fig. 5). Standard pointmap metrics (e.g., chamfer distance, scale-invariant error) used in prior work should be reported to substantiate this contribution.

- **VGGT is conspicuously absent from normal prediction comparisons:** VGGT (Wang et al., 2025a) jointly predicts multiple geometric quantities including depth and is directly comparable. It appears in Tables 7, 9, and 10 but is omitted from Table 1 (normal prediction). The paper should include VGGT in the normal benchmark or explain its exclusion.

- **Loss weights and key design choices lack ablation:** The loss weights ($\eta_1=1.0, \eta_2=0.1, \eta_3=0.075$ in Stage 1; $\lambda_1=1.0, \lambda_2=0.1, \lambda_3=1.0$ in Stage 2) are presented without sensitivity analysis. The claim that removing confidence loss prevents the model from "ignoring complex scenarios" (Sec. 3.2) is significant but not isolated in the ablation—improvements could stem from normal supervision alone.

- **Training reproducibility barriers:** The training procedure (32 H20 GPUs for ~2 weeks on a curated 20+ dataset mixture with A/B/C quality tiers) is resource-intensive. Critical training hyperparameters (optimizer, learning rate schedule, batch size, data augmentation) are not specified, and no code or data manifest is currently available.

- **"Intrinsic-invariant" terminology is under-justified:** The core technical mechanism is feature concatenation ($P_i^n = P_i \oplus n$, Eq. 9) and adding normal supervision. The paper claims this yields an "intrinsic-invariant pointmap" but provides no formal definition or theoretical grounding beyond analogy to MoGe's affine-invariant formulation. The practical benefit is clear, but the terminology overreaches the technical contribution.

## Nice-to-Haves

- **Comparison against an ensemble of specialized models:** A baseline combining separate SOTA models (e.g., Depth Anything V2 + StableNormal) would isolate whether joint training provides benefits beyond inference-time composition. This is a reasonable extension but not a core flaw.

- **Standard deviations across multiple random seeds:** Reporting mean ± std on key benchmarks would strengthen statistical confidence in the improvements over strong baselines.

- **Scaling analysis:** A performance vs. model size/data scale study would better justify the "foundation model" framing, which typically implies scaling properties.

- **Inference throughput metrics:** FPS and latency measurements would complement the parameter/memory efficiency claims.

## Removed Points

- **ScanNet/ScanNet++ train-test overlap claim:** The criticism that ScanNet++ training data overlaps with ScanNet test data is speculative. ScanNet and ScanNet++ are distinct datasets with separate scans (Yeshwanth et al., 2023 vs. Dai et al., 2017), and the paper uses them as separate entities. There is no evidence of scene-level overlap presented.

- **Multi-view inference uses classical pipeline as a weakness:** The paper explicitly states that multi-view processing follows the MASt3R-SfM pipeline (Sec. 3.3). This is not hidden or misrepresented—it is a deliberate design choice leveraging established tools rather than a novel neural fusion mechanism. Criticizing the absence of a neural multi-view module is scope creep.

- **"Foundation model" framing as a major weakness:** Whether this work meets the threshold for "foundation model" terminology is debatable but not a technical flaw. The model does demonstrate multi-task capability and downstream transferability. This critique is stylistic rather than substantive.

- **Demand for comprehensive scaling laws:** While scaling analysis would strengthen the paper, it is not a requirement for every architecture paper claiming foundational utility. The method demonstrates clear improvements over prior work across multiple tasks.

## Novel Insights

The most novel insight from this synthesis is the **bidirectional synergy between pointmaps and normals**: the paper demonstrates that Stage-1 pointmaps (which encode multi-view geometry) help the normal head resolve monocular ambiguity, while explicit normal supervision in Stage 2 regularizes and refines the pointmap representation (Fig. 11). This mutual reinforcement is under-emphasized in the paper but represents a genuine conceptual advance—most prior work treats normal and depth estimation as separate tasks or derives normals post-hoc from pointmaps. The "intrinsic-invariant" framing, while over-branded, captures a real insight: normals provide a locally deterministic geometric property (one surface → one normal map) that can anchor scale/shift-ambiguous pointmaps to a more stable reference frame.

## Suggestions

1. **Populate the incomplete tables** (Tables 8, 9, 10) with full numerical data for all compared methods. Without this, key claims about camera pose estimation and matching are unverifiable.

2. **Add quantitative pointmap evaluation** using standard metrics (e.g., chamfer distance, absolute/relative pointmap error) against DUSt3R, MASt3R, and MoGe on at least one standard benchmark. Even a subset would substantiate the central claim.

3. **Include VGGT in normal prediction comparisons** or provide a clear justification for its absence. The current selective inclusion raises concerns about fairness.

4. **Ablate the confidence loss removal independently** from normal supervision to validate the claim that confidence weighting harms reflective surface prediction.

5. **Provide training hyperparameters and data manifest** in a supplementary document or code release. At minimum, specify optimizer, learning rate schedule, batch size, and exact dataset splits used.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
