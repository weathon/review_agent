=== CALIBRATION EXAMPLE 55 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the paper's core contribution. The abstract succinctly summarizes the problem, proposed solution (DROID-3D dataset and EmbodiedMAE model), key results, and claims. All claims made in the abstract appear to be supported by the content of the paper.

### Introduction & Motivation
The introduction effectively motivates the need for 3D Vision Foundation Models (VFMs) in embodied AI, identifying two key gaps: the domain gap between existing 3D datasets and robot manipulation tasks, and the lack of efficient architectures that effectively integrate 3D information. The contributions are clearly stated at the end of the section. The related work is appropriately cited to establish context.

### Methodology (Section 2)
**2.1 3D Data Collection:** The creation of DROID-3D is well-motivated. The qualitative comparison of depth quality in Figure 2 is compelling. However, a **significant weakness** is the lack of any *quantitative* evaluation of the depth map quality (e.g., comparing against ground truth via standard metrics like RMSE). The claim of "superior and consistent depth quality" is thus only supported visually. The description of point cloud extraction is brief but sufficient.

**2.2 Multi-Modal Encoder:** The stochastic masking strategy using a Dirichlet distribution is a thoughtful design for encouraging cross-modal learning. The use of modality-specific patchifiers (standard for RGB/depth, DP3-style for point clouds) is appropriate. The decision to initialize from DINOv2 weights is pragmatic. A minor point: the claim that omitting explicit modality embeddings is compensated by bias terms is plausible but not experimentally verified; an ablation on this design choice would strengthen the argument.

**2.3 Multi-Modal Decoder:** The decoder design using cross-attention for explicit fusion is clearly described. The shared transformer across modalities is a sensible efficiency choice. The loss function is standard. However, the **decoder architecture details (e.g., number of layers, hidden dimensions) are not specified**, which hinders reproducibility. These details should be included in the appendix or main text.

**2.4 Model Distillation & 2.5 Put All Together:** The distillation procedure is clearly explained, aligning features at multiple network depths. The training hyperparameters are provided in Appendix Table 8, which is good. The extremely high masking ratios (90% during distillation) are notable and interesting.

**Overall Methodological Concerns:**
1.  **Reproducibility Gap:** Missing architectural details for the decoder.
2.  **Data Quality Validation:** Lack of quantitative metrics for DROID-3D depth maps.
3.  **Computational Cost:** The cost of pre-training the ViT-Giant model is not discussed, which is a practical concern for the community.

### Experiments & Results (Section 3)
**3.1 Experimental Setup:** The evaluation framework is comprehensive and fair. Using a compact, fixed RDT policy backbone isolates the impact of the visual representations. The chosen baselines (DINOv2, SPA, etc.) are appropriate and state-of-the-art. The benchmarks (LIBERO, MetaWorld, two real robot platforms) are diverse and relevant.

**3.2 MAE Predictions (RQ1):** The qualitative analysis in Figure 3 is insightful and demonstrates compelling cross-modal reasoning capabilities (e.g., object-level color propagation). However, this is **purely qualitative**. A quantitative analysis of reconstruction fidelity (e.g., PSNR/SSIM for RGB/depth, Chamfer distance for point clouds) would substantially strengthen the claim that the model "effectively leverages available modalities."

**3.3 Overall Comparison (RQ2):** The results are convincing. EmbodiedMAE outperforms strong baselines, shows clear scaling with model size, and demonstrates the unique benefit of its architecture for 3D inputs (unlike the naive DINOv2-RGBD baseline). A **major weakness** is the **absence of measures of statistical significance or variance** (e.g., standard error, confidence intervals) in all reported results (Figures 6, 8; Table 1). For a conference like ICLR, reporting the mean and spread across multiple seeds/trials is essential to assess robustness. The statement "each task is evaluated across 150 trials" (Sec. 3.3) implies multiple runs, but no variance is shown.

**3.4 Real-World Experiments (RQ3):** The real-world results on two distinct platforms strongly support generalization. The discussion of point cloud modality's sensitivity to sensor noise is honest and valuable. The same issue regarding statistical reporting applies here (10 trials per task).

**3.5 Ablation Studies:** The ablations (masking ratio, feature alignment, loss ratio) are adequate and show the design is robust. The test with the ACT policy is a good addition to show generality. The ablation is limited to the distillation phase due to cost, which is understandable but leaves the impact of design choices on the Giant pre-training unexplored.

**Appendix Experiments:** The supplementary analyses are thorough and address important questions: comparison with other 3D VFMs (B.1), encoder design (B.2), data quality impact (B.3), data efficiency (C), comparison with VGGT (D), and latency (E). This greatly strengthens the paper.

### Writing & Clarity
The paper is generally well-written and logically structured. Figures are informative. There are some minor grammatical errors and the parser has introduced formatting artifacts (e.g., broken equation formatting in the text), but these do not impede understanding. The methodology section is dense but manageable.

### Limitations & Broader Impact
The limitation (no native language support) is clearly stated, and a natural future direction is proposed. The ethics statement is standard. The reproducibility statement is good, promising code release. The paper could briefly discuss potential negative societal impacts of advancing robot manipulation capabilities (e.g., job displacement, security), even if to note that such impacts are not immediate.

### Overall Assessment
This paper presents a significant and well-executed contribution: a high-quality 3D embodied dataset (DROID-3D) and a novel multi-modal masked autoencoder (EmbodiedMAE) that demonstrates state-of-the-art performance on a wide range of robot manipulation benchmarks. The core ideas are sound, the experimental evaluation is extensive, and the scaling properties are impressive. The primary weaknesses are the lack of **quantitative depth validation**, **missing decoder architecture details**, and most critically, the **absence of statistical error reporting in all experimental results**. Addressing these issues, particularly the statistical reporting, is crucial for acceptance at a top-tier venue like ICLR. If these concerns are adequately addressed in a revision, the paper's strong empirical results and well-motivated design make it a compelling candidate for publication.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces EmbodiedMAE, a multi-modal masked autoencoder designed to learn unified 3D visual representations (RGB, depth, point clouds) for robot manipulation. A key contribution is the creation of DROID-3D, a large-scale, high-quality 3D augmentation of the existing DROID dataset, produced using ZED SDK processing. The model is evaluated extensively on simulation (LIBERO, MetaWorld) and real-world (SO100, xArm) tasks, demonstrating superior performance and training efficiency compared to various state-of-the-art vision foundation models (VFMs).

### Strengths
1.  **High-Quality Dataset Contribution:** The creation of DROID-3D is a significant, concrete contribution. The paper provides a compelling analysis (Fig. 2) comparing depth quality across datasets and details the substantial engineering effort (500 hours of processing) to produce temporally consistent, metric depth and point clouds for the entire DROID dataset. This addresses a well-known data gap in 3D embodied AI.
2.  **Comprehensive and Rigorous Evaluation:** The evaluation is exceptionally thorough, covering 70 diverse simulation tasks and 20 real-world tasks across two distinct robot platforms (low-cost SO100 and high-performance xArm). This multi-faceted benchmark strongly supports the paper's claims of generalization and practical utility. The inclusion of scaling laws (Fig. 6) and ablation studies (Sec. 3.5, App. B-D) adds scientific depth.
3.  **Strong Empirical Results:** EmbodiedMAE demonstrates clear, quantitative improvements over a well-chosen set of strong baselines (e.g., DINOv2, SPA) in both final performance and sample efficiency. The results convincingly show that the model's multi-modal design effectively leverages 3D information, whereas naive fusion (DINOv2-RGBD) often degrades performance (Sec. 3.3, Finding 3).

### Weaknesses
1.  **Limited Architectural Novelty:** The core architecture—a multi-modal masked autoencoder with a ViT encoder—builds directly upon established frameworks like MultiMAE and DINOv2. While the stochastic masking strategy and distillation process are well-executed, the fundamental architectural innovation is incremental. The paper's primary novelty lies in the application domain (embodied AI), the dataset, and the integrated design choices rather than a breakthrough in representation learning architecture.
2.  **Under-Explored Design Decisions:** Several key design choices lack thorough justification or exploration. For example, the symmetric Dirichlet distribution for stochastic masking uses a concentration parameter `α=1`, but no ablation studies justify this choice over alternatives that might bias towards certain modalities. The choice of three specific layers for feature alignment during distillation is also presented without comparative analysis against other strategies.
3.  **Incomplete Analysis of Modality Trade-offs:** While the paper shows RGBD generally outperforms point-cloud (PC) inputs in real-world settings (attributed to sensor noise), the analysis in Appendix B reveals that enhanced PC preprocessing closes this gap significantly. This suggests the reported inferiority of PC might be partly an engineering/data preprocessing issue rather than a fundamental limitation of the modality, a nuance not highlighted in the main claims.
4.  **Ambition/Scope Mismatch:** The title and abstract position EmbodiedMAE as a "unified 3D multi-modal representation," but the model is evaluated solely as a frozen visual encoder for visuomotor policy learning. Its performance as a general-purpose representation for other embodied tasks (e.g., navigation, 3D scene understanding) or its compatibility with language inputs (as noted in the limitations) remains unvalidated, slightly overstating its current "unified" nature.

### Novelty & Significance
**Novelty:** The work is **moderately novel**. Its main novelty resides in the systematic construction of a high-quality 3D embodied dataset (DROID-3D) and the tailored application of multi-modal masked autoencoding to the specific challenges of robot manipulation, including the stochastic masking strategy across 3D modalities. The architectural components themselves are combinations of existing techniques.
**Significance:** The work is **potentially significant for the robotics community**. DROID-3D is a valuable resource that can catalyze further research in 3D VLA models. EmbodiedMAE provides a strong, reproducible baseline that clearly demonstrates the benefits of dedicated 3D multi-modal pre-training for manipulation, setting a new state-of-the-art on several benchmarks. Its strong scaling properties and real-world validation enhance its practical significance.

### Suggestions for Improvement
1.  **Strengthen Architectural Ablations:** Conduct and report ablation studies on critical hyperparameters of the proposed method, such as the Dirichlet concentration parameter `α`, the number and position of feature alignment layers in distillation, and the impact of different cross-modal fusion mechanisms in the decoder. This would solidify the rationale for the chosen design.
2.  **Refine Claims on Modality Performance:** Integrate the findings from Appendix B more prominently into the main narrative. Acknowledge that with improved sensor processing, the PC modality's performance can rival RGBD, refining the claim that "RGBD yields better performance and is more robust to noise" to a more nuanced discussion about sensor fidelity and preprocessing requirements.
3.  **Explore Language Integration:** As noted in the conclusion, a direct and compelling next step is to integrate language conditioning, leveraging DROID's annotations. A preliminary experiment or a concrete architectural proposal for a vision-language-action variant would greatly increase the impact and align with the "foundation model" aspiration.
4.  **Clarify Comparison Baselines:** In Tables 1 and related results, ensure the notation (e.g., "DINOv2-RGBD") is explicitly defined in the main text or caption. The reader must refer to Appendix A.3 to understand this important baseline's implementation, which should be briefly summarized in the main experimental setup.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the necessity of pre-training on DROID-3D vs. other 3D datasets.** The paper claims DROID-3D is essential due to domain gap, but does not show what happens if you pre-train EmbodiedMAE on existing large-scale 3D datasets (e.g., ScanNet, Objaverse) or on other robot datasets with estimated depth. Without this, it's unclear if the gains come from the *data* or the *architecture*.
2. **Comparison with a simple late-fusion baseline of independent encoders.** The paper compares against naively added depth (DINOv2-RGBD) but not against a late-fusion model where separate RGB, depth, and point-cloud encoders are combined at the policy input. This is needed to validate the benefit of *unified* cross-modal pre-training versus simply using multiple modalities.
3. **Systematic evaluation of each modality's contribution.** The paper shows combined results but lacks an ablation where each modality (RGB, depth, point cloud) is incrementally added during pre-training and evaluated downstream. This would clarify whether the gains are from depth/point cloud or from the multi-modal fusion itself.
4. **Test on out-of-distribution (OOD) manipulation tasks.** The evaluation uses standard benchmarks (LIBERO, MetaWorld) and custom real-world tasks, but does not test on OOD scenes/objects to verify the claim of "generalization capabilities." This is critical for a foundation model.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of cross-modal alignment.** The paper provides qualitative visual predictions (Fig. 3) but no quantitative measure of cross-modal consistency (e.g., depth prediction error from RGB, or vice versa). This is needed to substantiate the claim of "strong multi-modal fusion capabilities."
2. **Analysis of why point clouds underperform in real-world despite good simulation results.** The paper attributes this to sensor noise but does not analyze the representation sensitivity (e.g., via robustness studies with synthetic noise). This gap undermines the understanding of 3D modality trade-offs.
3. **Investigation of what spatial information is actually learned.** The model is claimed to have "spatial perception capabilities," but there is no probing (e.g., on depth estimation, 3D object detection, or spatial relationship tasks) to confirm that the representations encode geometry beyond what RGB provides.

### Visualizations & Case Studies
1. **Visualize attention maps or feature activations for key manipulation steps.** Show where the model focuses (in 3D) during critical phases like grasping or obstacle avoidance. This would reveal whether the model actually uses 3D cues for spatial reasoning or relies on RGB patterns.
2. **Case studies of failures, especially when 3D input is provided.** The paper shows successful rollouts (Fig. 7) but not failures. Analyzing failure cases (e.g., when depth is noisy, or point clouds are sparse) would expose the limitations and boundary conditions of the method.

### Obvious Next Steps
1. **Incorporate language conditioning.** The paper notes that EmbodiedMAE is vision-only, but DROID includes language instructions. A clear next step is to integrate language into the pre-training (e.g., via masked language modeling) to create a vision-language-action foundation model, which is more aligned with current VLA trends.
2. **Benchmark on a wider set of 3D representation tasks.** The evaluation is solely on policy learning. To claim "unified 3D multi-modal representation," the features should be tested on standard 3D understanding tasks (e.g., 3D object classification, segmentation) to show generality beyond robotics.
3. **Study the transferability to non-tabletop domains.** The work is focused on tabletop manipulation. An obvious extension is to test on mobile manipulation or navigation tasks to see if the 3D representations generalize to broader embodied settings.

# Final Consolidated Review
## Summary
This paper introduces EmbodiedMAE, a multi-modal masked autoencoder that learns unified 3D visual representations (RGB, depth, point clouds) for robot manipulation, along with DROID-3D, a high-quality 3D augmentation of the DROID dataset. The model is extensively evaluated on 70 simulation and 20 real-world tasks, demonstrating state-of-the-art performance and effective scaling.

## Strengths
- **High-quality dataset contribution**: DROID-3D provides synchronized, temporally consistent depth maps and point clouds for the entire DROID dataset, addressing a critical data gap for 3D embodied AI. The paper includes a qualitative comparison of depth quality and details the substantial processing effort (500 hours).
- **Strong and comprehensive empirical validation**: EmbodiedMAE consistently outperforms strong baselines (e.g., DINOv2, SPA) across diverse benchmarks (LIBERO, MetaWorld) and two real robot platforms (SO100, xArm), with clear gains in training efficiency and final performance. The model shows beneficial scaling with size and effectively leverages 3D inputs where naive fusion fails.

## Weaknesses
- **Lack of quantitative depth validation**: The claim of "superior and consistent depth quality" for DROID-3D is supported only by visual comparison (Figure 2); no standard quantitative metrics (e.g., RMSE against ground truth) are provided, weakening the dataset's credibility.
- **Incomplete architectural details**: The decoder architecture (e.g., number of layers, hidden dimensions) is not specified, hindering reproducibility. While encoder details are given, the decoder is critical for the training process.
- **Absence of statistical error reporting**: All experimental results (success rates, learning curves) report only point estimates without measures of variance (e.g., standard error, confidence intervals), despite multiple trials per task, making it difficult to assess robustness.
- **Qualitative cross-modal analysis**: The evaluation of multi-modal fusion (Figure 3) is purely qualitative; quantitative metrics (e.g., reconstruction error for depth from RGB) would strengthen the claim of "strong cross-modal alignment."
- **Under-explored design choices**: Key hyperparameters, such as the concentration parameter α for the Dirichlet masking distribution, are set without ablation studies, leaving the rationale for these choices unverified.

## Nice-to-Haves
- Ablation studies comparing pre-training on DROID-3D versus other 3D datasets to isolate the contribution of domain-specific data.
- Comparison with a late-fusion baseline using separate encoders to better highlight the benefits of unified cross-modal pre-training.
- Visualization of attention maps or feature activations to illustrate how 3D spatial information is utilized during manipulation.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add quantitative depth evaluation metrics (e.g., RMSE) for DROID-3D in the appendix or main text.
- Specify the decoder architecture (layers, dimensions) in the methodology section or appendix.
- Include error bars, standard deviations, or confidence intervals in all tables and figures reporting performance metrics.
- Conduct an ablation study on the Dirichlet parameter α and other masking strategies to justify design choices.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
