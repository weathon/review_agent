=== CALIBRATION EXAMPLE 59 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me produce my review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation" accurately captures the method. However, the abstract's claim that the model "consistently outperforms state-of-the-art VFMs" is undermined by the authors' own reported results: in the xArm real-world RGB-only setting, EmbodiedMAE achieves only "comparable performance to SOTA baselines" (Section 3.4 / Figure 8), not superiority. The abstract also bundles dataset construction and model training into a single framing, obscuring whether gains come from better architecture, better pre-training data, or sheer scale of training data. These distinctions matter significantly and are never cleanly teased apart.

---

### Introduction & Motivation

The motivation is clear and well-argued: existing 3D VFMs suffer from a domain gap (outdoor/indoor static data vs. tabletop manipulation), and naïve 3D integration can hurt policy performance. The two root causes identified are data and architecture — a reasonable framing. However, the claim that "many advanced 3D VFM architectures demonstrate unexpectedly poor performance in policy learning, sometimes even underperforming simple MLPs" is attributed to Ze et al. (2024) and Zhu et al. (2024) without any summary of *why* this happens, making the motivation for EmbodiedMAE's specific architectural choices feel somewhat unsupported at this stage.

The contributions are stated clearly. The benchmark contribution is legitimate but somewhat over-stated: LIBERO and MetaWorld are pre-existing benchmarks; the paper's contribution is selecting subsets and configuring evaluation protocols, not constructing new environments.

---

### Method

**Section 2.1 – Data Collection (DROID-3D):**

The motivation for choosing DROID over BridgeDataV2 and RH20T is reasonable, but the depth quality comparison in Figure 2 is entirely qualitative. No objective metric (e.g., RMSE against ground-truth depth, or temporal consistency measured by inter-frame difference on static scenes) is provided. The claim that "AI models lack temporal consistency" compared to ZED SDK is stated assertively without quantification. Since the quality of DROID-3D is a foundational claim for the entire paper, the absence of quantitative depth evaluation is a notable gap.

Additionally, the ZED SDK's limitations are not acknowledged — reflective or transparent objects and challenging lighting conditions are known failure modes for stereo depth estimation and are particularly common in manipulation tasks (shiny tools, glass cups). The paper later (Section 3.4) reports that real-world point cloud policies underperform due to "sensor noise from object reflectivity and lighting variations," which seems in direct tension with the earlier strong claims about ZED SDK quality.

**Section 2.2 – Multi-Modal Encoder:**

Several specific hyperparameters critical for reproducibility are missing from the main paper:
- The Dirichlet concentration parameter **α** is defined conceptually but its chosen value is never stated. This is the key hyperparameter governing the cross-modal masking distribution and should appear in Section 2.5.
- For point cloud processing, the values of N (FPS cluster centers), K (KNN neighbors), and the DP3 encoder configuration are not specified in the main text (they may be in the appendix, but the appendix text was not explicit about this either).

The decision to **remove the [CLS] token** from DINOv2's ViT is mentioned but not justified or ablated. In DINOv2, [CLS] carries global semantic information and is typically the most useful feature for downstream tasks. Its removal may be significant for policy performance, yet no experiment tests this design choice.

The justification for **omitting explicit modality-type embeddings** ("the bias term in each projection layer implicitly encodes modality-specific information") is architecturally unconventional. This is a meaningful design departure from MultiMAE (Bachmann et al., 2022) and requires an ablation to be credible. None is provided.

**Section 2.3 – Decoder:**

The cross-attention decoder design is reasonable. The shared ViT decoder across modalities is a sensible efficiency choice. However, the paper does not compare this to simpler decoder designs (e.g., independent decoders per modality as in MultiMAE, or even a simple MLP head). The claim that the decoder "reduces computational cost by approximately a factor of three" applies to the number of decoder parameters but conflates parameter count with actual runtime/memory savings. A cross-attention decoder operating on a concatenated key/value sequence from all modalities may not actually be faster than three independent decoders.

**Section 2.4 – Model Distillation:**

The multi-layer feature alignment approach is sensible and well-grounded in prior work (Bai et al., 2023). However, the alignment layer mapping rule (aligning at "3/4 of the encoder depth") is misleadingly labeled "Middle" — if the student is a 12-layer model aligning with layer 9, that is the top quartile, not the middle. This naming is confusing and should be clarified.

Critically, **all ablation studies are performed in the distillation phase**, not during Giant model pre-training. The authors acknowledge this ("prohibitive cost"), but this means the following key design choices are empirically unvalidated: the decoder architecture, the α parameter value, the choice to omit modality-type embeddings, and the DINOv2 weight initialization vs. training from scratch.

---

### Experiments & Results

**Unfair comparison due to pre-training data scale and domain specificity:**

This is the most serious concern in the paper. EmbodiedMAE is pre-trained on 76K trajectories (350 hours) of robot manipulation data, while the main comparable baseline, SPA (Zhu et al., 2025), uses approximately 1/15th of DROID. The paper frames this as a quality advantage ("SPA employs CrocoV2-Stereo to estimate depth for approximately 1/15 of the DROID dataset"), but it is simultaneously a scale advantage. DINOv2 is trained on general internet images, not robot data. **There is no experiment that controls for training data domain or scale** — e.g., DINOv2 fine-tuned on the full DROID-3D RGB data, or SPA trained on the full dataset. This makes it impossible to determine whether the gains come from the EmbodiedMAE architecture, the 3D modalities, or simply having 15× more domain-relevant pre-training data than SPA and far more domain-specific data than DINOv2.

**Real-world evaluation:**

- Only **10 trials per task** is statistically very weak. A single-trial flip changes a task's reported success rate by 10%. With 10 tasks per platform, many individual task scores will have overlapping confidence intervals between methods.
- **No standard deviations or confidence intervals** are reported for real-world results in Figure 8, making it impossible to assess significance of the reported differences.
- The xArm RGB-only results are described as "comparable" to baselines, yet the abstract claims consistent outperformance. This inconsistency should be reconciled.

**MetaWorld Table 1:**

The table as extracted shows EmbodiedMAE-RGB average (73.0) matching SPA-RGB (73.0) exactly. This zero-gap result for the primary simulation benchmark is not highlighted or discussed. The paper's "Finding 1" (consistent outperformance) appears to rest largely on LIBERO and the RGBD/PC variants, not RGB-only simulation.

**Baseline for 3D comparisons:**

The "DINOv2-RGBD" naïve baseline (Section A.3, referenced but not fully reproduced in main text) uses a separately trained depth branch fed into an otherwise frozen DINOv2. This is arguably the weakest possible integration of depth information. A fairer baseline would be DINOv2 with a properly fine-tuned depth branch, or a multi-modal contrastive model. The comparison to this strawman may inflate the apparent advantage of EmbodiedMAE's RGBD variant.

**Missing ablations:**

1. Pre-training data ablation: train EmbodiedMAE on a subset of DROID-3D comparable to SPA's training set.
2. Architecture ablation: EmbodiedMAE pre-trained on RGB only vs. DINOv2 fine-tuned on DROID-RGB, to isolate architecture contribution from pre-training data contribution.
3. DINOv2 initialization vs. random initialization for the ViT encoder.
4. Effect of removing the [CLS] token.
5. Modality-type embeddings present vs. absent.

**Scaling behavior (Finding 2):**

The claim of "strong scaling behavior" is based on performance improvements from Small to Giant. However, comparing across four model sizes on the same benchmarks conflates parameter count with representational quality. There is no analysis of scaling efficiency (performance per FLOP), and the differences between Base and Large are described as "similar, with the Large model slightly ahead" — this is not what most would consider compelling scaling evidence.

---

### Related Works

The related works section (Section 4) is remarkably brief — less than half a page — for a paper with this scope. Notably absent:
- **MultiMAE (Bachmann et al., 2022)** is cited in Section 2.2 as inspiration for the stochastic masking strategy, but the section contains no comparative discussion of how EmbodiedMAE departs from or improves upon it.
- **Point-MAE (Pang et al., 2022)** is cited for the point cloud tokenizer but not discussed in the context of 3D pre-training methods.
- The relationship to **SPA** (the most relevant prior work) is discussed primarily in Section 2.1 as a data quality comparison, not as an architectural or methodological comparison.

---

### Limitations & Broader Impact

The limitations paragraph (Section 5) is essentially one sentence: the model lacks language instruction support. This dramatically undersells the actual limitations:

1. **Distribution specificity:** EmbodiedMAE is trained exclusively on DROID data, which uses ZED stereo cameras in specific lab environments. The paper does not test whether representations transfer to robots without stereo cameras, different environments, or very different object classes.
2. **Sensor dependency:** The RGBD setting requires a calibrated depth camera. The point cloud setting requires high-quality 3D sensing. The paper itself reports that PC-based policies underperform in real-world due to sensor noise — this is a significant practical limitation that deserves discussion.
3. **Evaluation scope:** All real-world tasks are tabletop pick-and-place style. The abstract acknowledges this ("particularly in precise tabletop manipulation settings") but positions it as a scope qualifier rather than a limitation.
4. **Low-data regime:** Both real-world platforms use only 20 demonstrations per task. How the model performs with more data — and whether it closes or expands the gap over baselines — is unknown.

---

### Overall Assessment

EmbodiedMAE makes a genuine practical contribution by constructing a large-scale, high-quality 3D robot manipulation dataset (DROID-3D) and demonstrating that domain-specific 3D multi-modal pre-training can consistently improve robot manipulation policies. The evaluation scope — 70 simulation and 20 real-world tasks across two robot platforms — is among the most comprehensive in this subfield. However, the paper's central empirical claims rest on a comparison that is fatally confounded: EmbodiedMAE enjoys both a massive scale advantage (76K vs. ~5K trajectories compared to SPA) and a domain-specificity advantage over general VFMs like DINOv2. Without a controlled experiment disentangling these factors from the architectural contribution, it is unclear whether the 3D multi-modal MAE architecture itself is necessary, or whether simply training DINOv2 on more robot data would yield comparable results. Additional concerns include the absence of quantitative depth quality metrics for DROID-3D, missing key hyperparameters (α, N, K), unjustified architectural choices (no [CLS], no modality-type embeddings) without ablations, statistically weak real-world evaluation (10 trials, no confidence intervals), and a summary mismatch between the abstract's "consistent outperformance" claim and the xArm RGB-only results showing parity with baselines. The paper is a solid engineering contribution and could be competitive for ICLR with a major revision that either (a) controls for pre-training data scale/domain in baselines, or (b) restructures claims to be more accurately scoped to what the experiments actually support.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents EmbodiedMAE, a unified 3D multi-modal representation learning framework for robot manipulation that integrates RGB, depth, and point cloud inputs via a masked autoencoder (MAE) architecture. A key contribution is the creation of DROID-3D, a large-scale dataset containing 76K trajectories with high-quality metric depth and point clouds derived from the original DROID dataset using ZED SDK processing. The authors demonstrate that EmbodiedMAE outperforms state-of-the-art Vision Foundation Models (VFMs) across 70 simulation tasks and 20 real-world manipulation tasks, exhibiting strong scaling behavior and effective cross-modal fusion.

### Strengths
1.  **Contribution of a High-Quality 3D Dataset (DROID-3D):** The creation of DROID-3D addresses a critical bottleneck in embodied AI: the lack of temporally consistent, high-fidelity 3D data. The paper provides evidence (Figure 2 comparison) that existing datasets (BridgeDataV2, RH20T) suffer from noisy or missing depth, and the ZED SDK processing pipeline offers a verifiable improvement in metric depth quality. This is a significant resource contribution to the community.
2.  **Comprehensive and Rigorous Evaluation:** The evaluation extends beyond standard simulation benchmarks. The authors test on diverse platforms (SO100 low-cost vs. xArm high-performance) and benchmarks (LIBERO, MetaWorld). Providing results on real-world robots is crucial for embodied AI claims and significantly strengthens the empirical validity of the representation's utility.
3.  **Methodological Rigor in Multi-modal Fusion:** The masking strategy (stochastic allocation across modalities via Dirichlet distribution) and the teacher-student distillation process (aligning features at Bottom, Middle, and Top layers) are well-justified and supported by ablation studies (Table 4). The paper validates that 3D inputs (depth/point clouds) specifically improve performance in spatially complex tasks rather than acting as noise (Figure 6, Finding 3).

### Weaknesses
1.  **Incremental Architectural Novelty:** While the engineering integration is sound, the core architecture relies heavily on established MAE and Distillation techniques (Oquab et al., 2024; Bai et al., 2023). The novel contributions lie primarily in the data pipeline and application, rather than new foundational mechanisms for representation learning. For an ICLR submission, a stronger theoretical or architectural abstraction might be needed beyond "applying MAE to multi-modal robot data."
2.  **Dependency on High-Quality Pre-processing for Point Clouds:** The paper acknowledges that Point Cloud (PC) performance degrades in real-world scenarios without enhanced preprocessing (Table 9). While this validates the need for DROID-3D-like data, it suggests the model does not gracefully handle noisy, uncurated real-world 3D data "out of the box." The reliance on external depth estimation (CrocoV2-Stereo) for PC enhancement adds a dependency that limits robustness.
3.  **High Pre-training Cost:** The Giant-scale model requires 8xNVIDIA L40 GPUs for 200K steps (Section A.8/A-11, inferred from Table 8). While distillation produces smaller models, the barrier to entry for training the Teacher model is high, potentially limiting reproducibility of the pre-training phase itself compared to lighter foundation models (e.g., DINOv2 training).
4.  **Lack of Language Integration:** The paper explicitly states the model is a "vision backbone" and does not support language instructions (Conclusion). While acknowledged as future work, in the current landscape of Vision-Language-Action (VLA) models (e.g., Octo, OpenVLA), this limits the direct applicability and significance in the broader "Embodied Foundation Model" context.

### Novelty & Significance
**Novelty:** The architectural novelty is moderate, as MAE-based approaches for robotics are recent but established. However, the novel integration of stochastic cross-modal masking specifically tuned for 3D embodied data is a distinct technical contribution. The **DROID-3D dataset** constitutes a higher novelty contribution in terms of community utility.

**Significance:** The significance is high for the robotics community. By successfully training a foundation model that leverages metric 3D depth in robot manipulation, the paper bridges the gap between computer vision (3D understanding) and robotics (policy learning). The empirical evidence that carefully processed 3D data is superior to raw depth or RGB-only inputs reinforces the direction for future research in embodied perception.

### Suggestions for Improvement
1.  **Clarify the Distillation Protocol on Smaller Models:** Since the Giant model's pre-training is expensive, provide more detailed analysis on why the distillation works for the *Giant* vs. *Small* specifically. Does the distillation recover the full representational capacity of the Giant model for smaller variants, or is there an inherent capacity ceiling? The "Scaling Behavior" section should quantify this gap more explicitly.
2.  **Analyze Computational Trade-offs:** Provide a more detailed latency/throughput comparison between EmbodiedMAE and baselines (e.g., DINOv2) during *inference* on the real robot controllers. Table 13 is helpful but focuses on GPU forward pass; controller loop timing (end-to-end policy latency) is more relevant for real-world robots.
3.  **Enhance Generalization Analysis:** Since Point Cloud performance is sensitive to noise, include a simulation experiment where Gaussian noise or outliers are explicitly injected into the DROID-3D data to see if the model learns robustness or just memorizes the clean distribution. This would distinguish between "learning spatial priors" and "overfitting sensor quality."
4.  **Address the Language Gap:** If language is out of scope, frame the contribution specifically as a "3D Visual Foundation for Control" to manage expectations against VLA models. Alternatively, briefly discuss how the learned embeddings could be projected into existing language-action models, demonstrating the potential for this "next step."

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Raw Sensor Noise Robustness:** The Point Cloud modality fails on real-world data without heavy preprocessing (Appendix B.3), undermining the claim of "robust 3D perception." Add an experiment quantifying performance decay as synthetic noise increases to define the operational limits of the representation.
2. **Baseline Data Parity:** SPA is compared against EmbodiedMAE, but SPA was trained on a subset of DROID while EmbodiedMAE uses the full DROID-3D. Re-train SPA on the full DROID-3D dataset to ensure performance gains are due to architecture, not simply data scaling.
3. **Modality Dropout at Inference:** To prove "unified fusion" rather than modality-specific experts, evaluate an RGB-only trained model on Depth-only inputs (and vice versa). Without this, the claim that the model learns cross-modal inferences remains unverified.

### Deeper Analysis Needed (top 3-5 only)
1. **Feature Space Alignment:** Provide t-SNE or PCA visualizations showing that RGB and Depth tokens for the same spatial location cluster together in the embedding space. Without this, the claim of a "unified representation" is unsupported by evidence.
2. **Pretraining vs. Distillation Contribution:** Table 4 suggests feature alignment (distillation) dominates performance over MAE masking (100% masking ratio performs nearly identically to 90%). Ablate the distillation step entirely to prove the MAE architecture adds value beyond knowledge transfer from a larger teacher.
3. **Compute Efficiency Metrics:** The paper claims "computational efficiency" but trains a 1.1B parameter Giant model. Report total FLOPs and training hours compared to training a standard DINOv2 on the same data to substantiate the efficiency claim.

### Visualizations & Case Studies
1. **Cross-Attention Heatmaps:** Visualize the decoder's cross-attention weights to confirm that Depth tokens explicitly attend to RGB tokens during fusion. This is necessary to verify the proposed cross-modal fusion mechanism actually functions as designed.
2. **EmbodiedMAE Failure Cases:** The paper only visualizes baseline failures (Figure 7). Show specific scenarios where EmbodiedMAE fails (e.g., transparent objects, extreme lighting) to establish trust boundaries and honesty about limitations.

### Obvious Next Steps
1. **Zero-Shot Robot Transfer:** Evaluate the model on a completely unseen robot dataset (e.g., BridgeData) without fine-tuning to test true generalization capabilities beyond the DROID domain.
2. **Language Instruction Grounding:** As a foundation model for "Embodied AI," the lack of language integration is a critical gap. Demonstrate instruction-following capabilities or explicitly benchmark against VLAs to justify the "foundation" claim.

# Final Consolidated Review
## Summary
The paper presents EmbodiedMAE, a unified 3D multi-modal masked autoencoder for robot manipulation that processes RGB, depth, and point cloud inputs through stochastic masking and cross-modal fusion. A key contribution is DROID-3D, a large-scale dataset (76K trajectories, 350 hours) constructed by extracting high-quality metric depth and point clouds from the original DROID dataset using ZED SDK temporal fusion. The authors demonstrate strong performance across 70 simulation tasks (LIBERO, MetaWorld) and 20 real-world tasks on two robot platforms (SO100, xArm), showing consistent improvements over vision foundation model baselines.

## Strengths
- **DROID-3D Dataset Contribution:** The creation of a large-scale, high-quality 3D robot manipulation dataset addresses a genuine bottleneck in embodied AI. The ZED SDK processing (temporal fusion, AI-augmented enhancement, metric depth calibration) provides verifiable improvements over AI-estimated depth, which the paper correctly identifies as lacking temporal consistency. This is a valuable resource contribution.
- **Comprehensive Evaluation Scope:** Testing across 90 total tasks (40 LIBERO + 30 MetaWorld + 20 real-world tasks across two distinct robot platforms) provides meaningful breadth. The inclusion of both a low-cost open-source robot (SO100, ~$250) and a high-performance robot (xArm) demonstrates practical applicability across hardware tiers.
- **Cross-Modal Fusion Capabilities:** Figure 3 demonstrates that the model learns genuine cross-modal reasoning—reconstructing RGB from depth (preserving structure, lacking precise color), depth from RGB (learning smoothness priors), and propagating semantic information in re-coloring experiments. This suggests the architecture learns meaningful multi-modal representations rather than modality-specific features.
- **Scaling Behavior:** The paper provides clear evidence of performance improvement from Small → Base → Large → Giant across multiple benchmarks (Figure 6), with the Giant model consistently achieving the best final performance and training efficiency. This is meaningful evidence that the architecture supports scaling.
- **Real-World 3D Performance:** The RGBD variant shows substantial improvements over RGB-only baselines on real-world tasks (Figure 8), confirming that properly integrated 3D information benefits manipulation—a finding with practical significance given prior work showing naïve 3D integration can *hurt* performance.

## Weaknesses
- **Pre-training Data Scale Confound:** The most significant methodological concern is that EmbodiedMAE is pre-trained on the full DROID-3D dataset (76K trajectories), while the primary 3D-aware baseline SPA uses approximately 1/15th of DROID (as the paper notes in Section 2.1). General vision models like DINOv2 are trained on internet-scale data, not robot data. Without experiments controlling for pre-training data scale—e.g., SPA trained on the full DROID-3D, or DINOv2 fine-tuned on DROID-RGB—it is impossible to determine whether performance gains derive from the EmbodiedMAE architecture, the 3D modalities, or simply from having substantially more domain-specific training data. The appendix shows scaling with dataset subsets (Table 11), but this is for EmbodiedMAE specifically, not a comparison against baselines with controlled data.

- **Missing Key Hyperparameters:** The Dirichlet concentration parameter α, which governs the stochastic masking distribution across modalities, is defined conceptually but never specified with a numerical value. Additionally, the point cloud patchifier parameters (N cluster centers, K nearest neighbors for FPS/KNN) are not provided in the main text. These are critical for reproducibility.

- **Unablated Architectural Design Choices:** Two significant departures from prior work lack ablations: (1) removal of the [CLS] token from DINOv2's ViT, despite [CLS] typically carrying the most useful semantic features for downstream tasks; (2) omission of explicit modality-type embeddings, justified only by a brief claim that "bias terms implicitly encode modality information." Neither choice is validated experimentally.

- **Statistical Weakness in Real-World Evaluation:** Real-world experiments use only 10 trials per task with no reported confidence intervals or standard deviations. A single trial flip changes a task's success rate by 10%. The xArm RGB-only results are described as "comparable" to baselines (Section 3.4), while the abstract claims the model "consistently outperforms state-of-the-art VFMs"—an inconsistency that matters more given the low trial count.

- **Point Cloud Fragility in Real-World Deployment:** Appendix B.3 reveals that PC-based policies require "enhanced pre-processing" (radius outlier removal, CrocoV2-Stereo-guided filtering) to achieve competitive performance (77.1% → 82.1% on xArm). The paper correctly identifies sensor noise from object reflectivity as a challenge, but this limitation undermines the claim of robust 3D perception "out of the box." The model appears to rely on high-quality point cloud inputs that may not match uncurated real-world conditions.

## Nice-to-Haves
- **Language Instruction Integration:** The paper explicitly notes the model "does not natively support language instruction as input." Given the current landscape of Vision-Language-Action models, demonstrating how EmbodiedMAE embeddings could interface with language-conditioned policies would strengthen practical applicability.
- **Noise Robustness Experiments:** Adding synthetic noise to point cloud inputs during evaluation would help characterize operational limits and distinguish between learning spatial priors versus overfitting to clean data.
- **Cross-Modal Inference Verification:** Evaluating models trained with all modalities on held-out modality combinations (e.g., depth-only input after RGBD training) would verify that the model learns unified representations rather than modality-specific experts.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"Related works section is too brief"** — The paper adequately covers MAE, embodied AI representations, and 3D robot learning. Length is not a substantive issue.

- **"Benchmark contribution is overstated"** — The paper correctly credits LIBERO and MetaWorld as existing benchmarks; their contribution is the evaluation protocol configuration, not claiming to create new benchmarks.

- **"Naming 'Middle' for 3/4 layer alignment is confusing"** — This is a minor naming nitpick. The method description is clear about which layers are aligned.

- **"Abstract bundles dataset and model contributions"** — Both contributions are legitimate and clearly described. This framing criticism is not substantive.

- **"DINOv2-RGBD is a weak baseline"** — While the naïve depth integration baseline may not be state-of-the-art for 3D fusion, the paper does compare against SPA (which incorporates 3D priors) and DP3 (a dedicated point cloud policy). The baseline criticism is weakened because the paper provides multiple 3D comparisons.

## Novel Insights
The paper demonstrates an important practical finding: depth as an auxiliary cue (RGBD) outperforms point cloud representations in real-world deployment due to sensor noise sensitivity, even though point clouds show promise in simulation. This has implications for practitioners—investing in high-quality depth sensors may be more practical than sophisticated point cloud processing pipelines for embodied AI applications. Additionally, the cross-modal reconstruction visualizations (Figure 3) reveal that the model learns to separate geometric structure from appearance information, suggesting the architecture encodes meaningful factorized representations of the visual world.

## Suggestions
1. **Control for pre-training data in baseline comparisons:** Train SPA (or another 3D-aware VFM) on the full DROID-3D dataset to isolate architectural contributions from data scale effects. This is the single most important change for establishing the core claim.
2. **Add the missing hyperparameter values:** Specify α for the Dirichlet distribution and N/K for point cloud patchification in the main text or appendix with clear numerical values.
3. **Include confidence intervals for real-world results:** Even with 10 trials, bootstrap confidence intervals would improve statistical credibility and allow readers to assess whether reported differences are meaningful.
4. **Ablate architectural choices:** At minimum, run experiments comparing: (a) with vs. without [CLS] token, and (b) with explicit modality-type embeddings vs. relying on bias terms. If these choices don't matter, report that result.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
