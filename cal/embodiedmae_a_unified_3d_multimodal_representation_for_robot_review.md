=== CALIBRATION EXAMPLE 61 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title ("EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation") accurately reflects the core contribution. The abstract clearly states the problem (domain gap, lack of effective 3D architectures), the proposed solutions (DROID-3D dataset, EmbodiedMAE model), and the key results (outperforming SOTA across many tasks). The claims of superior "training efficiency and final performance" are substantiated by the learning curves and success rates in the experiments.

**Introduction & Motivation:** The problem is well-motivated, citing the domain gap between standard 3D datasets (e.g., indoor/outdoor scenes) and robotic manipulation needs, and the architectural challenge of effectively integrating 3D data. The related work is appropriately cited. The three contributions are clearly and concisely stated at the end of the section.

**Methodology (2.1-2.5):**
*   **Data Collection (2.1):** The creation of DROID-3D is a significant contribution. The justification for using ZED SDK over AI-based depth estimation (temporal consistency, quality) is convincing and supported by Figure 2. A minor concern: the term "AI-augmented enhancement" in the ZED SDK is not elaborated upon. While not critical, a brief explanation or citation would enhance reproducibility.
*   **Model Architecture (2.2-2.4):** The multi-modal masked autoencoder design is clearly described. The stochastic masking strategy via a Dirichlet distribution is a sensible adaptation of multi-modal MAE principles. The omission of explicit modality-type embeddings, relying instead on projection layer biases, is noted but not justified; an ablation on this choice would strengthen the design narrative. The distillation process is well-detailed, aligning features at multiple network depths. A logical gap: the paper states the ViT encoder follows DINOv2 but removes the [CLS] token. It is unclear how the pooled representation for the policy is then obtained (e.g., average pooling over token embeddings?). This is a crucial detail for reproducibility.
*   **Putting It Together (2.5):** Training hyperparameters are provided in Appendix A. The claim that aggressive masking (90%) during distillation "significantly decreases training costs without compromising representational quality" is supported by the ablation study in Table 4. The HuggingFace integration is a practical strength for adoption.

**Experiments & Results (3.1-3.5):**
*   **Setup (3.1):** The choice of a compact RDT policy backbone for fair comparison is sound. The baseline selection is comprehensive, covering vision-centric, language-augmented, embodied-specific, and 3D-aware models. The benchmarks (LIBERO, MetaWorld, two real-world platforms) are diverse and appropriate. The number of trials (50 for sim, 10 for real-world) is reasonable, though statistical error bars or confidence intervals on success rates would strengthen the results, especially for the real-world experiments with 10 trials.
*   **MAE Predictions (RQ1, 3.2):** The qualitative analysis in Figure 3 is compelling for demonstrating cross-modal understanding. However, this is purely visual. A **major weakness** is the lack of **quantitative metrics** for reconstruction quality or cross-modal translation accuracy (e.g., PSNR, FID, Chamfer distance). The interesting observations about object-level semantics from the "re-coloring" experiment remain a qualitative claim without supporting quantitative analysis.
*   **Overall Comparison (RQ2, 3.3):** The results in Figure 6 and Table 1 are strong and generally support the claims. EmbodiedMAE outperforms baselines, scales with model size, and benefits from 3D input. A **significant concern** is the **presentation of MetaWorld results in Table 1**. The table structure is garbled in the text (e.g., "Col3", "Col4"), making precise scores for each baseline difficult to parse. The crucial result—that EmbodiedMAE-RGBD (76.2%) outperforms SPA-RGB (73.0%) and DINOv2-RGBD (54.4%)—is discernible but the table's formatting hinders clear interpretation. This appears to be a parser artifact, but it critically impacts the reader's ability to assess the results.
*   **Real-World Experiments (RQ3, 3.4):** Results on two distinct robot platforms greatly enhance the paper's impact. The finding that point clouds underperform in the real world due to sensor noise, while depth-augmented RGB excels, is valuable and practical. The discussion of failure cases in Figure 7 is insightful.
*   **Ablation Studies (3.5):** The ablations on masking ratio, feature alignment, and loss ratio are thorough and informative. The extension to the ACT policy (Tables 2 & 3) effectively demonstrates generalizability beyond the primary RDT policy. A **notable omission** is an ablation on the **importance of the new DROID-3D dataset**. How does pre-training on DROID-3D compare to pre-training on, say, a combination of original DROID (RGB) and a general 3D dataset like ScanNet? This would directly justify the effort of creating DROID-3D versus using existing resources.

**Writing & Clarity:** The paper is generally well-written. The flow from problem to method to experiments is logical. The figures are informative. The primary clarity issue is the mangled Table 1, which is a severe impediment to evaluating a key result. Other tables and figures are clear.

**Limitations & Broader Impact:** The limitation section is brief but hits the key point: the model is a vision backbone without native language support. The discussion of point cloud sensitivity to real-world noise is also a de facto limitation. The broader impact statement is standard and appropriate. A missed discussion point is the **computational cost and environmental impact** of pre-training the ViT-Giant model, which the authors acknowledge as "prohibitive." For ICLR, a brief statement on this would be expected.

**Appendix:** The appendix is extensive and adds valuable detail on task descriptions, hyperparameters, and additional analyses (point cloud encoders, data scaling, comparison with VGGT). The latency analysis in Appendix E is a welcome addition for practical deployment considerations.

### Overall Assessment
This is a strong paper with two substantive contributions: a high-quality 3D robotic dataset (DROID-3D) and a well-designed multi-modal MAE framework (EmbodiedMAE) that effectively leverages it. The experimental validation is extensive, covering simulation and two real-world platforms, and generally shows clear improvements over a sensible set of baselines. The main weaknesses are (1) the lack of **quantitative evaluation for the cross-modal fusion claims** (RQ1), leaving them as interesting but unmeasured observations; (2) the **critically garbled Table 1**, which undermines confidence in a core result; and (3) the **missing ablation on the necessity of the new dataset**. If the authors can provide quantitative metrics for RQ1, fix Table 1, and include a dataset ablation, the paper's contribution would be solid and likely meet ICLR's acceptance bar. The current empirical evidence is promising but requires these clarifications and reinforcements to be fully convincing.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces EmbodiedMAE, a multi-modal masked autoencoder framework designed to learn unified 3D visual representations from RGB, depth, and point cloud data for robot manipulation. A key supporting contribution is the creation of DROID-3D, an enhanced version of the DROID dataset with high-quality, temporally consistent depth maps and point clouds. The model is evaluated extensively on simulation (LIBERO, MetaWorld) and real-world robot tasks, demonstrating superior performance and training efficiency compared to several state-of-the-art vision foundation models.

### Strengths
1. **Comprehensive and Rigorous Evaluation**: The paper provides exceptionally thorough experimentation, validating the model across 70 diverse simulation tasks and 20 real-world tasks on two distinct robot platforms (low-cost SO100 and high-performance xArm). The consistent outperformance of established baselines (e.g., DINOv2, SPA, R3M) in both final performance and training efficiency is compelling evidence of the method's effectiveness.
2. **Valuable Dataset Contribution**: The construction of DROID-3D addresses a clear data gap in embodied AI. The systematic analysis of depth quality in existing datasets and the use of ZED SDK for high-fidelity, metric depth extraction is well-motivated. Releasing this dataset would be a significant resource for the community.
3. **Detailed Analysis and Ablations**: The paper goes beyond simple performance reporting. It includes insightful qualitative analysis of cross-modal predictions (Figure 3), scaling laws with model size, and systematic ablation studies on critical components like masking ratio, feature alignment, and loss weights. The analysis of point cloud modality challenges in real-world settings (Appendix B) is particularly valuable.

### Weaknesses
1. **Limited Novelty in Core Architecture**: The architectural core of EmbodiedMAE is a direct adaptation of established concepts: masked autoencoding (MAE) and multi-modal fusion via a ViT encoder-decoder. While the application to 3D embodied perception is novel, the paper could do more to delineate its specific technical innovations beyond the stochastic masking strategy and distillation approach, which are inspired by prior work (e.g., MultiMAE, DINOv2).
2. **Incomplete Baseline Comparisons**: The chosen baselines, while relevant, omit some recent and highly pertinent works in 3D representation learning for robotics. For instance, a comparison with models like PonderV2 is relegated to an appendix, and there is no discussion of how EmbodiedMAE relates to or differs from other contemporary masked autoencoders for point clouds or recent 3D vision-language-action models.
3. **Ambiguity in Distillation Necessity and Cost**: The paper employs a costly two-stage pre-training (Giant model) and distillation pipeline. While results show it works, the justification for this choice over direct pre-training of smaller models is not deeply explored. The computational cost of pre-training the Giant model is mentioned but not quantified in terms of GPU hours/CO2, which is a relevant consideration for ICLR's focus on efficient and scalable methods.

### Novelty & Significance
**Novelty** is moderate. The primary novelty lies in the *application domain* and the *integrated system*: a tailored multi-modal MAE pre-trained on a newly created, high-quality embodied 3D dataset (DROID-3D). The stochastic masking strategy across modalities and the feature-level distillation scheme are thoughtful adaptations but build directly upon existing ideas (MultiMAE, DINOv2 distillation).

**Significance** is high. The work successfully addresses a recognized pain point—the lack of effective 3D-aware vision foundation models for precise manipulation—and demonstrates clear empirical gains. The release of DROID-3D would lower the barrier to entry for 3D robot learning research. The observed scaling laws and the model's ability to effectively leverage 3D inputs where naive fusion fails are important findings for the field.

### Suggestions for Improvement
1. **Strengthen the Novelty Narrative**: Clearly articulate the specific, novel technical contributions of EmbodiedMAE's architecture beyond the composition of existing components. A detailed comparative table or discussion contrasting the masking, fusion, and distillation mechanisms with the closest prior works (MultiMAE, DINOv2, SPA) would help.
2. **Expand and Mainstream Critical Comparisons**: Integrate the comparison with other 3D VFMs (like PonderV2) and recent embodied 3D models into the main experiments. This would provide a more complete picture of the state-of-the-art and better justify the claim of superior performance.
3. **Provide a Clearer Cost-Benefit Analysis**: Include a section or table detailing the computational cost (GPU hours, energy) for pre-training the Giant model and distilling smaller variants. Discuss the trade-offs: is the performance gain worth the significant extra pre-training compute? Could a simpler, single-stage training of the Large model achieve comparable results with more data?
4. **Deeper Analysis of Failure Modes**: The paper shows successful rollouts and mentions typical baseline failures. A dedicated analysis of *EmbodiedMAE's own failure cases*, especially in challenging real-world tasks, would provide greater insight into its limitations and directions for future work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to 3D-specific robot learning methods.** The paper compares to SPA and DP3 but omits other recent 3D-aware robot models like 3D Diffuser Actor (Ke et al., 2024) or PointVLA (Li et al., 2025a). Without these, the claim of "SOTA performance" for 3D VFMs is incomplete and may overstate the contribution.
2. **Ablation on the importance of the new DROID-3D dataset.** The paper attributes success to the dataset but does not ablate pre-training on DROID-3D vs. other 3D datasets (e.g., ScanNet, Matterport3D) or even training without it. This gap makes it impossible to disentangle the contribution of the dataset from the architecture.
3. **End-to-end training comparison.** The model is pre-trained and then frozen for policy training. A critical experiment is to compare against fine-tuning the visual backbone end-to-end on the downstream tasks, which is standard in robot learning. The claim of superior representation is undermined without showing it outperforms or matches fine-tuning a strong baseline (e.g., DINOv2).
4. **Robustness to varying 3D sensor quality.** The paper notes point clouds underperform in the real world due to sensor noise. They should test robustness by systematically degrading depth/point cloud quality in simulation (e.g., adding noise, dropout) to show where the method breaks and whether the multi-modal fusion actually provides robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what the model actually learns about 3D geometry.** The paper shows visual predictions but lacks quantitative analysis of the learned 3D representations. For instance, they should probe the features with linear probes for depth estimation, surface normal prediction, or 3D object segmentation on held-out data to verify the claimed "spatial perception capabilities."
2. **Explanation for why naive depth addition hurts performance (DINOv2-RGBD).** The paper notes this but provides only a superficial reason. A deeper analysis (e.g., feature space visualization, gradient analysis) is needed to explain why their fusion works and others fail, which is core to their architectural contribution.
3. **Failure mode analysis for real-world tasks.** The paper shows successful rollouts and generic failure cases but does not systematically categorize why EmbodiedMAE fails when it does (e.g., specific object types, lighting conditions, occlusion). This is critical for understanding the method's limitations.
4. **Computational efficiency vs. performance trade-off.** The paper reports latency but does not compare the parameter/compute efficiency of EmbodiedMAE against baselines in a controlled way (e.g., FLOPs vs. success rate). For robotics, this trade-off is crucial, and the claim of "computational efficiency" is unsubstantiated.

### Visualizations & Case Studies
1. **Visualizations of attention maps or feature similarity across modalities.** To convincingly show cross-modal fusion, visualize where the model attends in the depth map when reconstructing an RGB patch, or show feature similarity between modalities for the same scene. Without this, the fusion mechanism remains a black box.
2. **T-SNE plots of representations.** Show that EmbodiedMAE's features cluster by task or object semantics better than baselines, especially when modalities are missing or noisy. This would directly support the claim of learning a "unified" representation.
3. **Case studies of tasks where 3D input is critical vs. not.** Show side-by-side rollouts of RGB-only and RGBD versions on tasks where depth is essential (e.g., precise insertion) and where it isn't, to demonstrate when the 3D gains come from.

### Obvious Next Steps
1. **Integrate language conditioning.** The paper notes this as future work, but for a "unified" representation aimed at VLA models, a language-aware variant (even a simple one) should have been explored as an ablation to show the framework's extensibility.
2. **Test on a broader set of real-world robots and environments.** The evaluation uses two specific robot platforms (SO100, xArm) in tabletop settings. To claim general "embodied AI" relevance, they should include at least one mobile manipulation or non-tabletop task to stress-test the representation.
3. **Release the DROID-3D dataset with clear licensing and documentation.** The paper positions DROID-3D as a key contribution, but without immediate public release and reproducibility details (processing code, download links), the community cannot build upon it, severely limiting the paper's impact.
4. **Compare to a simple early fusion baseline.** A strong baseline is to simply concatenate RGB and depth channels and feed them into a standard ViT (with appropriate positional embeddings). This is a common approach, and its absence makes it hard to judge the necessity of the proposed stochastic masking and cross-modal decoder.

# Final Consolidated Review
## Summary
EmbodiedMAE proposes a multi-modal masked autoencoder framework for learning unified 3D visual representations from RGB, depth, and point cloud data, specifically for robot manipulation. It introduces DROID-3D, an enhanced dataset with high-quality 3D annotations. The model demonstrates superior performance across diverse simulation and real-world tasks compared to existing vision foundation models, with effective scaling and the ability to leverage 3D inputs where naive fusion fails.

## Strengths
- **Extensive and rigorous evaluation**: The paper validates the model on 70 simulation tasks (LIBERO and MetaWorld) and 20 real-world tasks across two distinct robot platforms (low-cost SO100 and high-performance xArm), showing consistent outperformance over multiple strong baselines (e.g., DINOv2, SPA, R3M) in both final success rates and training efficiency.
- **Valuable dataset contribution**: DROID-3D addresses a critical data gap by providing high-quality, temporally consistent depth maps and point clouds for robot manipulation, with thorough justification and processing details using ZED SDK, making it a significant resource for the community.
- **Detailed analysis and ablations**: Includes comprehensive ablation studies on masking ratios, feature alignment, loss weights, and scaling behavior, as well as insightful analysis of modality effectiveness and point cloud challenges in real-world settings (Appendix B).

## Weaknesses
- **Lack of quantitative cross-modal evaluation**: The claims about cross-modal fusion and understanding (RQ1) are supported only by qualitative visualizations (Figure 3). Quantitative metrics for reconstruction quality or cross-modal translation accuracy (e.g., PSNR, Chamfer distance) are absent, leaving the strength of fusion as an unmeasured assertion.
- **Clarity issues in results presentation**: Table 1, reporting key success rates on the MetaWorld benchmark, is garbled in the provided text (e.g., with "Col3", "Col4" placeholders), severely hindering interpretation and verification of these critical results.
- **Missing ablation on dataset necessity**: No experiment isolates the contribution of the DROID-3D dataset. For instance, pre-training on alternative 3D datasets or ablating dataset quality would clarify whether the gains stem from the new data versus architectural choices.
- **Insufficient justification for architectural success**: The paper observes that naive depth integration (e.g., DINOv2-RGBD) degrades performance, but provides no deep analysis (e.g., feature space inspection or ablation on fusion components) to explain why EmbodiedMAE's fusion strategy succeeds. This weakens the rationale for the proposed design.
- **Ambiguity in visual representation pooling**: The encoder removes the [CLS] token, but the method for obtaining a fixed-dimensional visual representation for the policy network is not explicitly described (e.g., average pooling over tokens), affecting reproducibility.
- **Incomplete computational cost discussion**: While training costs are mentioned as "prohibitive," there is no quantification of resources (e.g., GPU hours, carbon emissions) for pre-training the Giant model, which is important for assessing scalability and environmental impact in line with conference expectations.

## Nice-to-Haves
- Comparing end-to-end fine-tuning of visual backbones versus the frozen representation approach used.
- Conducting systematic robustness tests to varying 3D sensor noise or quality in simulation.
- Providing quantitative probes (e.g., linear evaluation for depth estimation or 3D segmentation) to analyze what geometric properties the representation captures.
- Including visualizations like attention maps or T-SNE plots to illustrate cross-modal fusion and feature clustering.
- Expanding comparisons to more 3D-specific robot learning methods (e.g., 3D Diffuser Actor) in the main experiments, though some are discussed in the appendix.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- Criticism about the term "AI-augmented enhancement" in ZED SDK not being elaborated (minor detail that does not affect core claims).
- Subjective critiques about architectural novelty being limited (the contribution is in the integrated system and application to embodied AI, not in fundamentally new components).
- Requests for language integration or broader robot testing (e.g., mobile manipulation), which are outside the stated scope of a vision representation model.
- Demand for confidence intervals on success rates for real-world trials, as single-run evaluation with 10 trials is common in robotics, and the paper already reports averages.

## Novel Insights
The paper demonstrates that a multi-modal masked autoencoder, pre-trained on domain-specific 3D robot data, can learn representations that scale effectively with model size and enable policies to leverage 3D inputs for improved spatial understanding in manipulation. Notably, it reveals a practical trade-off: point cloud modalities, while compact, are sensitive to real-world sensor noise, whereas depth-augmented RGB provides more robust performance gains, offering valuable guidance for modality selection in robotics applications.

## Suggestions
- Add quantitative metrics for the cross-modal predictions in RQ1 (e.g., reconstruction error or cross-modal translation scores) to substantiate the fusion claims.
- Fix Table 1 to clearly present success rates for all baselines and modalities on MetaWorld, ensuring readable formatting.
- Include an ablation study pre-training on subsets of DROID-3D or alternative datasets (e.g., ScanNet) to validate the dataset's unique contribution.
- Provide a brief analysis or hypothesis on why naive depth fusion fails and how EmbodiedMAE's design mitigates this, perhaps through feature visualizations or an ablation on the stochastic masking and cross-modal decoder.
- Clarify in the methodology how the visual representation is pooled or used by the policy network after encoder processing (e.g., specify if average pooling over tokens is employed).
- Quantify the computational cost of pre-training (e.g., GPU hours) in the appendix or limitations section to address environmental impact concerns.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
