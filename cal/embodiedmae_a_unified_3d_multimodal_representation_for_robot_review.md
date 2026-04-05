=== CALIBRATION EXAMPLE 59 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title Accuracy**: The title accurately reflects the core contribution: a unified 3D multi-modal representation learned via masked autoencoding, specifically tailored for robot manipulation.
- **Abstract Clarity**: The abstract clearly outlines the problem (domain gap between static 3D datasets and manipulation, lack of scalable 3D architectures), the method (DROID-3D dataset construction + multi-modal MAE with stochastic masking), and the key results (outperforms SOTA VFMs across 90 tasks, exhibits scaling, works on real robots). 
- **Unsupported Claims**: The claim "consistently outperforms state-of-the-art vision foundation models... in both training efficiency and final performance" is broadly supported by Figures 6, 8, and Table 1. However, the phrase "particularly in precise tabletop manipulation settings" slightly over-constrains the dataset used (DROID contains non-tabletop, open-space trajectories), though the evaluation is indeed tabletop-focused. This is a minor semantic point that does not detract from the contribution.

### Introduction & Motivation
- **Motivation & Gap**: The problem is well-motivated. The authors correctly identify that existing 3D VFMs are trained on static/scene-scale data, causing a domain mismatch for close-range manipulation, and that naive 3D fusion (e.g., adding a depth channel) often degrades policy performance. The cited observations from Ze et al. (2024) and Zhu et al. (2024) ground the motivation solidly.
- **Contributions**: Clearly stated and accurate. The dual contribution of a new dataset (DROID-3D) and a pre-training framework (EmbodiedMAE) is appropriately scoped for an ICLR submission.
- **Over/Under-claiming**: The introduction appropriately positions EmbodiedMAE against both vision-centric (DINOv2, SigLIP) and embodied-specific (SPA, R3M, VC-1) baselines. It does not over-claim generality; it explicitly frames the work around manipulation. One minor gap: it does not discuss how EmbodiedMAE compares to recent open-source VLAs (e.g., OpenVLA, π0) that use vision backbones but integrate language. The authors clarify in Section 5 that language is out of scope, which is acceptable, but a brief acknowledgment in the intro would strengthen positioning.

### Method / Approach
- **Clarity & Reproducibility**: The architecture is largely well-described, but there are two notable reproducibility gaps:
  1. **Point Cloud Tokenization (Sec 2.2)**: The authors state they use KNN to group each FPS center with its `K` nearest neighbors to form groups of `K+1` points, yet the value of `K` is never specified anywhere in the main text or appendices. This is a critical hyperparameter for the DP3-style encoder.
  2. **Initialization Contradiction**: Section 2.2 states the ViT structure is chosen to "initialize the ViT directly from DINOv2 pre-trained weights, thereby enhancing its general capabilities." However, Section 2.4 states "we first train a ViT-Giant EmbodiedMAE model from scratch on the DROID-3D dataset." Training from scratch and initializing from DINOv2 are mutually exclusive unless the authors mean DINOv2-style training (i.e., scratch teacher + distillation protocol). This contradiction must be resolved for methodological clarity.
- **Assumptions & Justifications**: The assumption that ZED SDK temporal fusion yields superior depth to AI-estimated depth (Fig 2) is plausible but lacks quantitative validation (e.g., absolute/relative depth error metrics against CrocoV2 or DepthAnythingV2). Qualitative claims about "lack of precision and temporal consistency" need at least a baseline metric comparison given the 500-hour processing cost claimed.
- **Logical Gaps / Edge Cases**: 
  - The Dirichlet distribution concentration parameter `α` controls modality dropout balance, yet its value during training is never stated, nor is it ablated. The choice of `α` directly impacts whether the model learns true cross-modal fusion or collapses to relying on a single modality.
  - Equation 1's notation is slightly inconsistent: the decoder section defines outputs as `g_I(h_I, h)`, but `h` is introduced as the joint encoder representation rather than a decoder state. Additionally, the targets `\hat{I}_2` are referenced without explicit definition in the surrounding text.
- **Theoretical Claims**: None; the paper is empirically driven.

### Experiments & Results
- **Testing Claims**: The experiments directly address the stated RQs. RQ1 (cross-modal fusion) uses controlled masking visualizations (Fig 3). RQ2 (SOTA comparison) is tested on LIBERO and MetaWorld. RQ3 (real-world efficiency) uses SO100 and xArm. The experimental design aligns well with the claims.
- **Baselines**: Appropriate and fairly compared. The inclusion of SPA (implicit 3D) and DINOv2+naive depth branch isolates the value of explicit, co-trained multi-modal fusion. The DINOv2-RGBD baseline (App A.3) is carefully implemented to ensure fair comparison.
- **Missing Ablations**: 
  - `α` (Dirichlet concentration) is missing, as noted above.
  - The impact of *decoder depth/size* relative to the encoder is not discussed, despite the claim that the shared decoder reduces cost by "approximately a factor of three."
- **Statistical Significance / Error Bars**: **This is a significant concern for ICLR standards.** Figures 6, 8, and Table 1 report point estimates without standard deviations or confidence intervals. Robotic control and diffusion policy training are highly stochastic. Reporting only mean success rates over 50 (sim) or 10 (real) trials without variance makes it difficult to assess whether the gains over DINOv2 or SPA are statistically significant or within noise margins. Error bars or statistical tests should be included.
- **Cherry-picking**: Results do not appear cherry-picked; the scaling curves (Fig 6) and real-world trends (Fig 8) show consistent patterns. The observation that Point Cloud inputs underperform RGB in real-world due to sensor noise is transparently discussed and further investigated in Appendix B.3.
- **Datasets/Metrics**: LIBERO, MetaWorld, SO100, and xArm are standard and appropriate. Success rate is the standard metric for policy evaluation.

### Writing & Clarity
- **Clarity Issues**: 
  - The notation in Section 2.3 and Eq. 1 requires tightening. The joint representation `h` is conflated with decoder hidden states. A clear forward pass diagram or explicit variable definitions would resolve this.
  - The "scratch vs. DINOv2 init" contradiction in Sections 2.2 and 2.4 creates confusion about whether the backbone benefits from ImageNet priors.
  - Table 1 is heavily garbled due to parser extraction, but the underlying data structure is understandable from context. I will not penalize this.
- **Figures/Tables**: Fig 3 is highly informative qualitatively. Fig 6 and 8 would benefit immensely from error shading. Table 4's formatting is split across text and a table object in the PDF extraction, but the ablation trends (90% masking is optimal, Bottom alignment is crucial, β=1 is robust) are clear.

### Limitations & Broader Impact
- **Acknowledged Limitations**: Section 5 explicitly notes the lack of native language support and frames future work toward VLA training. This is an honest and appropriate scope limitation.
- **Missed Limitations**: 
  - **Sensor Dependency**: The heavy reliance on high-quality, temporally fused depth (ZED SDK) assumes hardware capabilities not present in many open-source robots or mobile setups. The real-world performance drop for raw PC inputs (despite being a claimed modality) suggests EmbodiedMAE is highly sensitive to input depth quality. The limitations section should explicitly discuss this dependency and the bounds of its robustness to commodity depth sensors (e.g., RealSense D435, stereo cameras on drones).
  - **Domain Shift**: The model is trained and evaluated exclusively on tabletop manipulation. DROID-3D includes more diverse trajectories, but the paper does not discuss or test generalization to locomotion, navigation, or non-prehensile manipulation, which limits claims of being a "unified 3D representation for embodied AI."
- **Broader Impact/Ethics**: The ethics statement is standard and appropriate. No major negative societal impacts are overlooked beyond standard robotics safety concerns.

### Overall Assessment
EmbodiedMAE presents a timely and valuable contribution to embodied AI: a high-quality 3D robot manipulation dataset (DROID-3D) and a carefully designed multi-modal MAE that demonstrably improves policy performance over existing VFMs. The methodological soundness is strong, and the ablation of masking strategies, distillation layers, and real-world hardware validates the practical utility of the approach. However, the paper requires clarification on a direct methodological contradiction (scratch training vs. DINOv2 weight initialization), specification of missing hyperparameters (KNN `K`, Dirichlet `α`), and standardization of statistical reporting (error bars/variance across seeds) to meet ICLR's empirical rigor standards. The point cloud sensitivity in real-world deployment is a critical finding that should be more prominently discussed as a hardware-dependent limitation rather than solely an input modality characteristic. Addressing these points will solidify an already strong submission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces EmbodiedMAE, a unified 3D multi-modal masked autoencoder designed to learn robust spatial representations for robot manipulation, alongside DROID-3D, a large-scale dataset augmented with high-quality, temporally consistent depth and point clouds. By leveraging stochastic cross-modal masking, explicit cross-attention fusion, and feature-aligned knowledge distillation, the model achieves state-of-the-art policy learning performance across 70 simulation tasks and 20 real-world manipulation tasks on two distinct robot platforms, demonstrating strong scaling behavior and effective 3D perception.

### Strengths
1. **Comprehensive and Multi-Faceted Evaluation:** The model is rigorously tested across simulation (LIBERO, MetaWorld) and real-world environments (SO100, xArm), multiple policy backbones (RDT diffusion, ACT), and numerous baseline VFMs. Evidence: Section 3 and Tables 1-3, 9, 11 consistently show EmbodiedMAE outperforming vision-centric, language-contrastive, and embodied-specific models in both final success rates and sample efficiency (e.g., LIBERO learning curves in Fig 6).
2. **High-Quality Dataset Contribution (DROID-3D):** The paper directly addresses a critical bottleneck in embodied AI by processing the full 76K DROID trajectories with ZED SDK temporal fusion, yielding metrically accurate and temporally consistent depth/point clouds. Evidence: Section 2.1 explicitly contrasts this with the noisy, incomplete, or AI-estimated depth in BridgeDataV2, RH20T, and SPA (visualized in Fig 2), filling a clear gap in the literature.
3. **Well-Ablated Training Design and Scaling:** The architectural choices are methodical and empirically justified. Dirichlet-distributed stochastic masking prevents modality bias, cross-attention enables explicit fusion, and multi-depth feature alignment efficiently compresses the Giant model into smaller variants. Evidence: Sections 2.2-2.4 detail the pipeline, while the ablation (Table 4/LIBERO) demonstrates robustness across masking ratios, loss weights ($\beta$), and confirms the critical contribution of bottom-to-top feature alignment.
4. **Transparent Real-World Analysis:** The authors honestly document point cloud sensitivity to real-world sensor noise and provide an actionable preprocessing pipeline that recovers performance. Evidence: Section 3.4 notes PC underperformance, and Appendix B.3 (Table 9) shows enhanced preprocessing boosting PC success on xArm from 77.1% to 82.1%, providing valuable practical insight.

### Weaknesses
1. **Incremental Architectural Novelty:** The core technical components are well-established adaptations rather than fundamental methodological breakthroughs. The ViT backbone omits the CLS token to leverage DINOv2 weights (Sec 2.2), stochastic masking follows MultiMAE (Sec 2.2), and the distillation framework aligns with DINOv2/Bai et al. (Sec 2.4). While expertly integrated for robotics, the methodological novelty is primarily engineering-driven.
2. **Asymmetric Multi-Modal Baselines:** EmbodiedMAE is evaluated natively with RGBD and PC inputs, but most baselines are not fully adapted to these modalities. The DINOv2-RGBD variant uses a simplistic, na¨ıve depth patchifier (App A.3), and strong 3D-aware models like SPA or DP3 aren't given equivalent multi-modal footing, which slightly weakens the isolation of EmbodiedMAE's specific architectural advantages versus data-scale advantages.
3. **Practical Deployment Constraints Understated:** Real-world latency and preprocessing requirements are critical for robotics but are relegated to appendices. Appendix E shows EmbodiedMAE-L in bf16 takes ~32-47ms for batch size 8 (which is tight for 30Hz control loops), and PC performance heavily relies on enhanced preprocessing (App B.3). These constraints should be discussed in the main experimental section to set realistic expectations.
4. **Absence of Language-Conditioned Evaluation:** Despite the rapid shift toward Vision-Language-Action (VLA) models, experiments focus solely on vision-to-imitation pipelines. Without testing instruction-following or language grounding, the paper misses an opportunity to demonstrate whether the unified 3D representations transfer effectively to language-augmented policy architectures.

### Novelty & Significance
**Novelty:** Moderate. The paper does not introduce a fundamentally new self-supervised objective or transformer variant. Instead, it offers a carefully engineered integration of proven techniques (stochastic MAE, cross-modal attention, feature distillation) specifically optimized for 3D robot manipulation. The primary novelty is the cohesive pipeline, the DROID-3D dataset curation, and the demonstration of how unified 3D pre-training can be effectively scaled for downstream control.
**Clarity:** High. The manuscript is logically structured, mathematically precise, and clearly explains the masking strategy, decoder fusion, and distillation protocol. Figures and tables effectively support empirical claims.
**Reproducibility:** Strong. Hyperparameters, training schedules, benchmark configurations (Sec 3.1, App A), and ablation studies are thoroughly documented. The DROID-3D processing pipeline is explicitly detailed, and the authors commit to releasing the codebase, ensuring the work is highly reproducible pending dataset/public code release.
**Significance:** High for ICLR's representation learning and embodied AI tracks. By directly tackling the 3D domain gap in manipulation datasets and demonstrating a scalable, multi-modal VFM that translates to real-world policy gains, the paper addresses a recognized bottleneck. The empirical depth, real-world validation, and dataset release make it highly valuable for the community.

### Suggestions for Improvement
1. **Elevate Real-World PC Insights to Main Text:** Move the analysis of point cloud sensor noise and the enhanced preprocessing pipeline (currently Appendix B.3) into Section 3.4. This will strengthen the paper's practical utility and justify the RGBD vs. PC design choices more transparently for robotics practitioners.
2. **Strengthen Multi-Modal Baseline Fairness:** Add at least one rigorously adapted multi-modal baseline (e.g., integrating a proper depth/PC tokenizer into a competitive 3D-aware model like SPA, rather than a na¨ıve DINOv2 patch adder) to better disentangle whether performance gains stem from DROID-3D's data quality, the MAE architecture, or both.
3. **Explicitly Discuss Latency vs. Control Frequency:** In the experimental discussion, analyze how the reported inference latencies (Appendix E) interact with the 30Hz control loops of SO100/xArm. Clarify whether batching, asynchronous feature extraction, or hardware optimization is necessary for real-time deployment of the Large/Giant variants.
4. **Detail Dataset Release and Licensing:** Explicitly state the planned release timeline, hosting infrastructure, and licensing terms for DROID-3D in Section 2.1 or the conclusion. Given its role as a primary contribution, clear access guarantees will significantly boost the paper's reproducibility and community impact.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Quantify training efficiency with explicit metrics (wall-clock hours, GPU days, or FLOPs) and plot sample-efficiency curves against baselines, otherwise the central claim of superior efficiency is unsupported.
2. Evaluate properly adapted multi-modal versions of all baselines (e.g., RGB-D/PC variants of DINOv2, SPA, and SigLIP) instead of using a naive depth-channel appending baseline, to prove architectural superiority rather than baseline under-engineering.
3. Report mean ± standard deviation across multiple random seeds for all policy fine-tuning results, as robotic control exhibits high variance and ICLR requires statistical significance to validate performance claims.
4. Test out-of-distribution (OOD) generalization using entirely unseen objects, distractor clutter, or novel camera viewpoints during fine-tuning to verify that the 3D representation enables true spatial reasoning instead of memorizing the 20 demonstrations per task.

### Deeper Analysis Needed (top 3-5 only)
1. Provide quantitative feature evaluation (e.g., linear probes for 6D pose estimation, spatial coordinate regression, or center-aligned kernel analysis) to empirically prove the model learns "spatial perception" and "object-level semantics" beyond downstream policy success rates.
2. Systematically quantify the domain gap between clean ZED-SDK pre-training data and noisy real-world sensors by injecting calibrated depth noise during evaluation, clarifying whether the PC modality's failure stems from architectural flaws or data mismatch.
3. Isolate the contribution of the distillation pipeline versus training from scratch for the Base/Large models to determine whether performance scaling is driven by the multi-modal MAE objective or the feature alignment losses.

### Visualizations & Case Studies
1. Project pre-trained patch embeddings using t-SNE/UMAP with spatial or semantic coloring to demonstrate that 3D inputs produce structurally superior feature clusters compared to RGB-only or naive baselines.
2. Show overlaid 3D trajectory plots or joint-angle time-series comparing successful EmbodiedMAE executions against baseline failures to directly quantify improvements in reach precision, grasp alignment, and collision avoidance.
3. Visualize attention maps or activation heatmaps under different stochastic mask ratios to confirm the encoder attends to geometric boundaries and object contact points rather than background textures or lighting cues.

### Obvious Next Steps
1. Release detailed per-task success rates for all 70 simulation and 20 real-world evaluations instead of only suite-level averages, preventing result masking and revealing exactly which manipulation primitives benefit from 3D inputs.
2. Conduct a hyperparameter sensitivity analysis on the distillation loss weight (β) and masking schedule during the downstream fine-tuning phase to prove the representations are robust to task-specific adaptation settings.
3. Benchmark on a completely independent real-world manipulation dataset or protocol (e.g., CALVIN or Franka Kitchen) to validate whether the DROID-3D scale genuinely transfers to broader embodied environments as claimed.

# Final Consolidated Review
## Summary
This paper introduces EmbodiedMAE, a unified 3D multi-modal masked autoencoder pre-trained on DROID-3D, a novel dataset augmented with temporally consistent depth and point clouds extracted via ZED SDK processing. The framework employs stochastic cross-modal masking, explicit decoder fusion, and multi-stage feature distillation to learn robust spatial representations. Evaluated across 70 simulation tasks (LIBERO, MetaWorld) and 20 real-world tasks on two robotic platforms, the model demonstrates consistent improvements in policy learning over existing vision foundation models, with particular gains when leveraging RGB-D inputs.

## Strengths
- **Rigorous multi-environment evaluation pipeline:** The authors systematically validate representations across two simulation benchmarks and two distinct real-world robot platforms (SO100 and xArm), using both diffusion-based (RDT) and sequence-model (ACT) policy backbones. This breadth strongly supports the generalization claims.
- **Targeted dataset contribution addressing a clear domain gap:** DROID-3D directly tackles the lack of large-scale, temporally consistent 3D manipulation data. The explicit processing pipeline using SDK-based temporal fusion yields higher-fidelity depth/point clouds than common AI-estimated or noisy dataset alternatives, providing a tangible resource for the community.
- **Transparent empirical characterization of 3D modality limitations:** The paper honestly documents that point cloud inputs, while effective in simulation, significantly degrade under real-world sensor noise without aggressive preprocessing. The appendix analysis quantifies this failure mode and demonstrates a recovery pipeline, offering practical value beyond the core algorithm.

## Weaknesses
- **Unsubstantiated efficiency claims and absence of statistical rigor:** The abstract claims the model "consistently outperforms... in both training efficiency and final performance," yet no quantitative training efficiency metrics (GPU-days, FLOPs, or explicit convergence sample counts) are provided. Furthermore, all policy success rates across multiple seeds and trials are reported as point estimates without standard deviations or confidence intervals. For stochastic diffusion policy fine-tuning and RL-adjacent benchmarks, this omission violates standard empirical rigor, making it impossible to determine whether observed gains over baselines are statistically significant or stochastic variance.
- **Asymmetric baselines obscure architectural vs. data-scale contributions:** The multi-modal comparisons are unbalanced. While EmbodiedMAE natively fuses modalities via its MAE decoder, baseline adaptations (e.g., DINOv2-RGBD) rely on naive depth-channel appending, and established 3D-aware models (SPA, DP3) are not equipped with equivalent native RGB-D or fused tokenizers. Consequently, it is unclear whether performance gains stem from the proposed cross-modal fusion architecture or simply the scale/cleanliness of the DROID-3D pre-training corpus.
- **Critical reproducibility gaps and methodological ambiguity:** Several essential hyperparameters are entirely missing: the Dirichlet concentration parameter $\alpha$ governing stochastic modal masking allocation is never stated, nor is its sensitivity evaluated. The point cloud tokenizer's KNN neighbor count ($K$) is also omitted. Additionally, Section 2.2 claims architectural compatibility for initializing from DINOv2 weights, while Section 2.4 states the Giant teacher is trained "from scratch." This contradiction creates confusion regarding whether ImageNet priors are leveraged and undermines methodological transparency.

## Nice-to-Haves
- Provide quantitative depth quality metrics (e.g., RMSE, AbsRel) comparing ZED-SDK processed frames against AI-based stereo depth to empirically justify the claimed 500-hour processing cost.
- Explicitly map the reported inference latencies (Appendix E) to the 30Hz control frequencies of SO100/xArm to clarify real-time deployment feasibility for Large/Giant variants.
- Include a dedicated ablation isolating the impact of distillation vs. direct multi-scale MAE pre-training to confirm whether scaling gains are driven by knowledge transfer or the masking objective alone.
- Clearly state the planned release timeline, hosting infrastructure, and licensing terms for DROID-3D in the main text to maximize community impact.

## Novel Insights
The work's most compelling empirical finding is the stark inversion of point cloud modality utility between simulation and physical deployment. While PC inputs consistently boost policy success in noise-free benchmarks, they actively degrade performance on real xArm/SO100 hardware due to reflectivity, lighting artifacts, and sensor drift. This explicitly demonstrates that unified 3D representation learning for robotics cannot treat geometric modalities as interchangeable; rather, foundation models trained on pristine synthetic or SDK-fused geometric priors are highly vulnerable to commodity depth sensor noise unless coupled with domain-aware preprocessing or robustness-oriented self-supervision.

## Potentially Missed Related Work
- None identified. The manuscript comprehensively cites and discusses relevant geometric foundation models (3D-VLA, PonderV2, VGGT, SpatialVLA) and properly contextualizes EmbodiedMAE within current 3D embodied representation literature.

## Suggestions
- Quantify training efficiency explicitly by reporting FLOPs, training wall-clock hours, or sample-efficiency curves alongside the existing convergence plots, or adjust the claims to strictly reflect final performance if such metrics are unavailable.
- Add error bars or standard deviations to all multi-seed policy evaluation figures (LIBERO, MetaWorld, real-world) and clarify the exact seed counts per evaluation tier.
- Specify the missing hyperparameters ($\alpha$, $K$) in the methodology or appendix tables, and resolve the contradiction regarding DINOv2 weight initialization vs. scratch training in Section 2.2/2.4.
- Implement and evaluate at least one properly adapted native multi-modal baseline (e.g., a depth-augmented ViT with learned positional embeddings rather than zero-init patch addition, or a multi-modal extension of SPA) to rigorously isolate the architectural contributions of Cross-Modal MAE fusion from dataset quality effects.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
