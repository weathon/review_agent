=== CALIBRATION EXAMPLE 80 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title "π³: Permutation-Equivariant Visual Geometry Learning" clearly signals the core contribution. The abstract succinctly states the problem (reliance on a fixed reference view), the solution (fully permutation-equivariant architecture predicting affine-invariant poses and scale-invariant pointmaps), and the outcome (SOTA performance, robustness). Claims appear supported by the results later in the paper. One minor point: the abstract mentions "breaking the reliance on a conventional fixed reference view," which is slightly overstated as traditional SfM/MVS also deals with gauge freedoms, but the framing for *feed-forward neural networks* is appropriate.

**Introduction & Motivation:** The introduction effectively frames the problem, identifying the inductive bias of anchoring to a reference view as a limitation in modern feed-forward methods (e.g., VGGT). The motivation is clear: this bias leads to instability and performance variance. The contributions are listed precisely. A potential weakness is that the "first to systematically identify" claim (Contribution 1) might be challenged, as the issue of gauge freedom/global coordinate system ambiguity is fundamental in 3D vision. However, the paper's focus on exposing and solving this *within the specific paradigm of feed-forward neural networks* is novel and well-argued.

**Method / Approach:**
*   **Permutation-Equivariant Architecture:** The formal definition (Eq. 1-3) is clear. The implementation detail of omitting "order-dependent components" is stated, but the description is somewhat high-level. Crucially, the paper states the architecture is "similar to (Wang et al., 2025a)" but removes reference tokens and frame indices. For full reproducibility and to verify true equivariance, a more detailed description of the token handling and attention mechanism across views would be beneficial. The appendix provides more architecture details but still assumes familiarity with VGGT's alternating attention scheme.
*   **Scale-Invariant Local Geometry:** The formulation of the scale alignment (Eq. 4) and loss (Eq. 5) is sound and follows prior work (ROE solver). The use of depth-weighted L1 is justified. The normal and confidence losses are standard.
*   **Affine-Invariant Camera Pose:** The use of relative pose supervision scaled by the consistent factor \(s^*\) is elegant and logically consistent with the scale-invariant pointmaps. The loss terms (Eq. 8-10) are standard. The claim that this leads to a "low-dimensional structure" in the predicted pose distribution (Figure 4, Appendix A.3) is interesting but presented more as an observation than a designed property or a proven theoretical advantage. It's not entirely clear if this structure is a cause or a consequence of the performance.
*   **Training Details & Initialization:** A **significant concern** emerges here (detailed in Appendix A.2 & A.4). The model is **not trained from scratch** on the proposed objective. The encoder and alternating attention layers are initialized from a pre-trained **VGGT** model (which is reference-dependent), and the encoder is frozen. This choice, while pragmatic, complicates the narrative. It raises a critical question: Is the reported performance gain primarily due to the novel permutation-equivariant *decoding* head and loss formulation, or is it heavily reliant on features learned by a reference-dependent pre-trained model? The ablation study (Sec. 4.5, Table 7) compares variants that also use this initialization, so it doesn't isolate this effect. The experiment in Appendix A.4 (training from scratch with a global proxy) is important but presented as a "fair comparison" side note. This initialization strategy should be prominently disclosed and discussed as a limitation in the main text, as it affects the interpretation of the core contribution's standalone efficacy.

**Experiments & Results:**
*   **Scope and Benchmarks:** The evaluation is exceptionally comprehensive across pose, depth, and pointmap reconstruction tasks on numerous datasets, meeting ICLR's high standard for empirical validation.
*   **Performance:** Results are strong, often SOTA or highly competitive. The video depth results (Table 4) are particularly impressive. The monocular depth results (Table 5) are competitive with specialized models like MoGe, which is notable for a generalist model.
*   **Robustness Evaluation (Sec. 4.4):** This is a key experiment validating the central claim. The near-zero standard deviation across permutations (Table 6) is compelling evidence of achieved permutation equivariance and a clear advantage over baselines. The evaluation protocol (cycling the first frame) is reasonable, though testing random permutations (not just cyclic ones) would strengthen the claim further.
*   **Ablation Study (Sec. 4.5):** The study is useful but has limitations. As noted, all ablated models use the same VGGT-initialized backbone. The performance gains from adding scale-invariant pointmaps and affine-invariant poses are clear, especially outdoors. However, the ablation does not isolate the impact of the *architecture's permutation equivariance* from the impact of the *supervision strategy* (relative vs. absolute). Model 1 and Model 2 use a camera token (i.e., are not equivariant) but also use different supervision. A cleaner ablation would be to compare a reference-based model trained with relative pose supervision (using \(s^*\)) against the full π³ model.
*   **Baseline Comparison:** Baselines appear appropriate and state-of-the-art. However, the running time/FPS comparison (Table 4) requires careful interpretation. π³ is faster than VGGT, but it uses a frozen encoder and fewer alternating attention layers (36 vs. 48). The speedup may be attributed more to these design choices than to the permutation-equivariant formulation itself. This should be clarified.

**Writing & Clarity:** The paper is generally well-written. The method section is mathematically clear. Some parts, like the description of the alternating attention and the initialization strategy, require flipping to the appendix for full understanding, which slightly disrupts the flow. Figures are effective.

**Limitations & Broader Impact:** The limitations in Appendix A.8 are appropriate (transparent objects, lack of fine detail, grid artifacts). A crucial limitation missing from the main discussion is the **dependence on a reference-based pre-trained model for initialization and feature extraction**. This should be added. The broader impact statement is minimal but acceptable for this technical work.

### Overall Assessment
This paper presents a well-motivated, conceptually clean idea: eliminating the reference-view bias via permutation equivariance in feed-forward 3D reconstruction. The method is elegant, and the empirical results are extensive and largely state-of-the-art, particularly in robustness to input order. The work is significant and likely of interest to the ICLR community. However, the **heavy reliance on initialization from a pre-trained reference-dependent model (VGGT)** is a substantial caveat that tempers the claim of a purely novel, standalone architecture. The core contribution might be more precisely framed as a novel *decoding and supervision paradigm* that, when grafted onto strong pre-existing features, yields robustness and accuracy benefits. The authors should address this nuance directly, ideally with additional experiments or a more forthright discussion. If this concern is adequately addressed in a revision, the paper represents a solid contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces π³, a permutation-equivariant neural network for multi-view 3D reconstruction that eliminates the traditional dependency on a fixed reference view. The model predicts affine-invariant camera poses and scale-invariant local point maps for each input view, ensuring robustness to input ordering. The method achieves state-of-the-art or competitive results on camera pose estimation, monocular/video depth estimation, and point map reconstruction across numerous benchmarks, while also being fast and lightweight.

### Strengths
1. **Clear and Well-Motivated Problem Formulation**: The paper identifies and systematically challenges the reliance on a fixed reference view as a detrimental inductive bias in prior work (e.g., DUSt3R, VGGT). This is a compelling critique supported by empirical evidence (Figure 2, Table 4.4).
2. **Strong Empirical Performance**: The method establishes new SOTA or highly competitive results across a wide range of tasks (camera pose, depth, point cloud reconstruction) on many standard benchmarks (RealEstate10K, Sintel, KITTI, 7-Scenes, etc.). The improvements are often significant (e.g., video depth Abs Rel on Sintel from 0.299 to 0.233).
3. **Demonstrated Robustness and Permutation Equivariance**: The paper provides compelling evidence of the model's key property: near-zero standard deviation in reconstruction metrics under input permutations (Table 6), a direct and impressive validation of the core claim.
4. **Efficiency**: The model is notably fast (57.4 FPS) and has a manageable parameter count (959M), making it practical for real-world applications compared to slower predecessors.

### Weaknesses
1. **Initialization Dependency and Training Instability**: The model requires initialization from a pre-trained VGGT encoder for stable training. The ablation in Appendix A.4 reveals that training from scratch with the core objectives fails, requiring an auxiliary "global proxy" task. This suggests the permutation-equivariant objective is difficult to optimize directly and may limit the perceived simplicity of the approach.
2. **Incremental Novelty in Architecture**: The core transformer architecture (alternating view-wise/global attention) is largely adopted from VGGT. The primary novelty is the removal of reference-view tokens and positional embeddings, and the reformulation of the supervision. While the conceptual shift is significant, the architectural change is relatively minimal.
3. **Limited Analysis of Failure Cases and Limitations**: The discussion of limitations (Appendix A.8) is brief and generic (transparent objects, lack of fine detail, grid artifacts). A deeper analysis of specific failure modes, especially related to the scale and affine invariance assumptions in complex scenes, would strengthen the paper.
4. **Marginal Gains in Some Metrics**: While many results show clear improvements, some are only marginally better than VGGT (e.g., point map metrics on 7-Scenes in Table 2). The paper could better contextualize when the reference-free approach yields its largest benefits.

### Novelty & Significance
**Novelty**: The paper makes a clear conceptual contribution by identifying and removing the reference-view bias, a common but largely unexamined practice in feed-forward 3D reconstruction. The realization of this via a fully permutation-equivariant architecture predicting per-view, relative geometry is novel in this domain. While permutation equivariance is a known concept, its application to solve the reference-frame problem in 3D vision is innovative.
**Significance**: The work is significant for the field. It demonstrates that a reference-free paradigm is not only viable but can lead to more robust, accurate, and efficient models. The strong empirical results across diverse tasks suggest this could become a new standard approach. The model's robustness to input order is a critical step towards reliable real-world systems.

### Suggestions for Improvement
1. **Address Training Stability More Forthrightly**: The reliance on VGGT initialization and the proxy task for scratch training should be moved from the appendix into the main paper's methodology or discussion. Analyze why the relative pose supervision is so unstable initially and how the proxy task mitigates this. This is crucial for reproducibility and understanding the method's true requirements.
2. **Deeper Ablation on Scale/Affine Invariance**: The ablation study (Sec. 4.5, Table 7) notes that scale invariance helps more outdoors. Provide a deeper analysis or hypothesis for this phenomenon. Is it related to depth range, dataset diversity, or camera motion patterns?
3. **Expand Analysis of Limitations**: Go beyond the list in A.8. Provide qualitative examples of failures (e.g., grid artifacts, transparent objects) and discuss potential avenues for future work to address them. This is expected for a mature ICLR submission.
4. **Clarify the "Cold Start" Problem**: In Appendix A.4, the "cold start" problem with N×N constraints is mentioned. Elaborate on this in the main text. A brief theoretical or intuitive explanation of why this coupling makes optimization harder than a reference-anchored approach would be insightful.
5. **Strengthen the Pose Distribution Analysis**: Figure 4/6 and the accompanying discussion on low-dimensional pose structure are interesting but somewhat peripheral. Better connect this finding to the model's improved performance or robustness to argue for its significance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with an ensemble of reference-view methods across multiple reference choices.** The paper claims instability from reference view selection but does not compare against a strong baseline that runs a reference-based method (e.g., VGGT) multiple times with different reference views and aggregates results (e.g., via averaging or best-view selection). Without this, the advantage of permutation equivariance is not fully quantified.
2. **Evaluation on truly unordered, internet photo collections (e.g., PhotoTourism, Landmarks).** Most benchmarks are ordered sequences (videos) or structured multi-view datasets. To convincingly demonstrate the benefit of permutation equivariance, the method should be tested on unordered sets where the input order is arbitrary and potentially detrimental.
3. **Ablation on the impact of the number of input views (N), especially large N.** The robustness claim should be tested with varying N (e.g., 2, 10, 50, 100+) to see if performance degrades gracefully compared to reference-based methods, which may struggle with many views due to error accumulation.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of failure modes: when does removing the reference view hurt?** The paper only shows improvements, but there may be scenarios where a fixed reference is beneficial (e.g., when one view is of much higher quality or has a more canonical perspective). Identifying and analyzing such cases is critical to understanding the method's limitations.
2. **Quantitative link between low-dimensional pose structure and reconstruction accuracy.** The paper claims the predicted poses lie on a low-dimensional manifold (Figure 4), but does not demonstrate how this directly leads to better reconstruction. A correlation analysis between pose distribution metrics (e.g., eigenvalue concentration) and final accuracy metrics is needed.
3. **Investigation of scale consistency across scenes with large depth variations.** The method uses a single global scale factor per scene. An analysis of scale estimation error on scenes with extreme depth ranges (e.g., indoor vs. outdoor) would reveal robustness issues.

### Visualizations & Case Studies
1. **Side-by-side visualizations of reconstructions from different input permutations.** The paper reports low metric variance, but visual examples of point clouds/camera trajectories from multiple random orderings are necessary to convincingly demonstrate consistency.
2. **Case studies on challenging dynamic scenes with quantitative benchmarks.** The method claims to handle dynamic scenes, but no quantitative results on dedicated dynamic benchmarks (e.g., Dynamic Scene Dataset, TUM RGB-D with dynamic objects) are provided. Visual examples alone are insufficient.

### Obvious Next Steps
1. **Evaluate on large-scale unordered image collections (e.g., MegaDepth, PhotoTourism)** to demonstrate real-world utility beyond ordered sequences.
2. **Integrate into a SLAM/VO pipeline** to show real-time applicability, leveraging the claimed speed advantage (57.4 FPS). This would strengthen the impact claim for robotics/AR.
3. **Ablation on the choice of backbone and training data.** The model uses a frozen DINOv2 encoder initialized from VGGT. An ablation training from scratch or with other backbones (e.g., CLIP) would clarify the source of performance gains.

# Final Consolidated Review
## Summary
This paper introduces π³, a permutation-equivariant feed-forward network for visual geometry reconstruction that eliminates the common reliance on a fixed reference view. The method predicts affine-invariant camera poses and scale-invariant local pointmaps for each input view, ensuring the output is invariant to input ordering. It achieves state-of-the-art or competitive performance on camera pose estimation, monocular/video depth estimation, and dense pointmap reconstruction across a wide range of benchmarks, while also being fast and highly robust to input permutations.

## Strengths
- **Demonstrated Robustness to Input Order:** The paper provides compelling empirical validation of its core claim, showing near-zero standard deviation in reconstruction metrics (e.g., accuracy, completion) across different input permutations (Table 6). This is a direct and significant advantage over prior reference-dependent methods.
- **Strong and Broad Empirical Performance:** The method establishes new state-of-the-art results on key tasks, such as video depth estimation on Sintel (Abs Rel 0.233 vs. 0.299) and camera pose estimation on Sintel (ATE 0.074 vs. 0.167), while remaining competitive or superior across numerous other benchmarks for pose, depth, and point cloud reconstruction (Tables 1, 2, 3, 4, 5).
- **Efficiency:** The model is efficient, achieving 57.4 FPS on KITTI with a 959M parameter count, making it practical for real-time applications and faster than several comparable large models.

## Weaknesses
- **Dependence on Pre-Trained, Reference-Based Features:** The model's encoder and alternating attention layers are initialized from the pre-trained VGGT model (which is reference-dependent) and the encoder is frozen (Appendix A.2). While pragmatic, this complicates the interpretation of the core contribution's standalone efficacy. The auxiliary experiment in Appendix A.4 shows training the proposed objective from scratch is unstable without a proxy task, indicating the permutation-equivariant supervision is difficult to optimize in isolation.
- **Incomplete Attribution of Performance Gains:** The speed advantage over VGGT is partially attributable to architectural differences (36 vs. 48 alternating attention layers) and the frozen encoder, not solely to the permutation-equivariant formulation. This should be clarified to avoid overstating the efficiency benefit of the core idea.

## Nice-to-Haves
- A more detailed analysis of when a reference-free approach might be disadvantageous (e.g., when one view is of significantly higher quality) would provide a more complete understanding of the method's limitations.
- Testing on truly unordered internet photo collections (e.g., from MegaDepth or PhotoTourism) would further strengthen the claim of robustness in unconstrained settings.
- A clearer quantitative link between the observed low-dimensional structure in predicted camera poses (Figure 4) and the improvement in final reconstruction accuracy would solidify this interesting finding.

## Novel Insights
The paper's core insight is that the standard practice of anchoring a reconstruction to a fixed reference view is an unnecessary and detrimental inductive bias for feed-forward neural networks. By formally removing this bias through a permutation-equivariant architecture and relative, per-view supervision, the work demonstrates that reference-free systems are not only viable but lead to superior robustness and accuracy. This represents a meaningful conceptual shift in the design paradigm for learning-based multi-view geometry.

## Suggestions
- Move the discussion of training instability and the use of VGGT initialization from the appendix into the main methodology or limitations section. Provide a clearer analysis of why the relative pose supervision is difficult to optimize from scratch.
- In the efficiency comparison, explicitly acknowledge the contribution of architectural choices (fewer layers, frozen encoder) to the reported FPS, ensuring the speedup is not attributed solely to the permutation-equivariant formulation.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
