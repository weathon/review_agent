# Language-Guided 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Dynamic rendering methods often prioritize photometric fidelity while lacking explicit semantic representations, which constrains their ability to perform semantically guided rendering. To this end, we introduce Language-Guided 4D Gaussian Splatting (L4DGS), a lightweight framework for real-time dynamic scene rendering that integrates natural language into semantically structured 4D Gaussian representations. Central to L4DGS is a Sparse Multi-Scale Attention (SMSA) mechanism that enables fine-grained, language-driven control by emphasizing semantically relevant regions across space and time. To enforce semantic fidelity and spatial coherence, we propose a static regularization that aligns language-guided features with rendered outputs and ensures consistent depth. To further ensure temporal consistency, A dynamic regularization penalizes abnormal variations in semantics and depth over consecutive unit time intervals. L4DGS achieves a 16.1% improvement in PSNR, reduces perceptual error by 58.8%, and increases rendering speed by over 50\%. Experimental results demonstrate the superiority of our approach in both visual quality and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a language-guided regularization loss to enhance semantic consistency and rendering quality in 4D Gaussian Splatting (4DGS).

### Strengths
The idea of integrating language guidance into 4DGS is interesting. The reported metric is high.

### Weaknesses
1. Unclear relevance of physical motion to language guidance

 (line 354 to line 369) "To further ensure physically plausible motion and accurate structural evolution, we extend our formulation with a depth-based regularization that constrains temporal changes in the predicted geometry. We penalize excessive depth fluctuations in Gaussian primitives across consecutive unit time intervals:" The connection between motion regularization and language semantics is not well explained.


2. Inadequate evaluation of “Language-Guided Semantic Consistency”

The semantic consistency claim is mainly demonstrated through optical flow visualizations. This seems inappropriate because optical flow operates at the pixel level and does not directly measure semantic or language-level consistency. More suitable semantic metrics and visualziation should be considered to substantiate this claim.

3. lack of context for Figure 3
The paper states that “Figure 3 further confirms the key advantages of L4DGS. It enables language-driven control for accurate and localized scene editing.” However, Figure 3 is not clearly connected to 4DGS or explained in sufficient context. The figure and its caption should better demonstrate how language guidance enables localized editing within the 4DGS framework.

4. Missing analysis of computational overhead
The paper lacks an analysis of the computational cost introduced by additional regularization terms (e.g., depth-based and CLIP-based losses). It would be helpful to quantify the impact on training speed and memory usage compared to vanilla 4DGS.

### Questions
1. Can the authors provide concrete examples or quantitative results showing “language-driven control for accurate and localized scene editing”? Ideally, please show examples on datasets of 4DGS

2. Training details and runtime feasibility
The paper claims that all experiments were conducted on a single RTX 3090 GPU and cost 20~30 minutes. How was this runtime measured?
Is the CLIP-based language regularization computed online during training for all 20,000 or 14000 iterations?
What image resolution is used as input to the CLIP model during training?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces language-guided 4D Gaussian Splatting (L4DGS), a lightweight framework that integrates natural language guidance into real-time 4D Gaussian splatting for dynamic scene rendering. The paper points out that although the existing 3DGS has improved the rendering efficiency, it still lacks semantic control capabilities. The dynamic scene expansion version also has problems such as motion blur and scene drift. For the above issues, this paper combines the sparse multi-scale Attention (SMSA) used for cross-modal feature fusion with static and dynamic regularization mechanisms to ensure semantic and temporal consistency. It significantly outperforms existing methods in terms of rendering fidelity and computational efficiency, and the qualitative results demonstrate its ability to guide scene editing through language (such as deleting target objects).

### Strengths
1. The current mainstream dynamic rendering methods mostly rely on visual features and lack explicit control at the semantic level, making it impossible to align human language intentions with dynamic rendering results. L4DGS innovatively constructs a language-guided 4DGS framework, deeply integrating the 4DGS representation of natural language understanding and semantic perception.
2. The proposed Sparse multi-scale Attention (SMSA) effectively aligns language and visual patterns and uses the top-k sparse strategy to improve interpretability and efficiency. The dual regularization design (static + dynamic) ensures spatial consistency and temporal consistency, and resolves the long-standing semantic drift and flickering issues in dynamic NeRF systems.
3. Compared with leading baselines such as MixVoxels, K-Planes and Deformable4DGS, there have been substantial improvements in PSNR, LPIPS and rendering speed. The training time is reduced to a few minutes while maintaining real-time rendering quality. Support for prompt semantic operations (" delete car ", "delete person") highlights the potential of interactive applications (for example, VR/AR, robot perception, content creation).

### Weaknesses
1. Although the paper emphasizes "first language-embedded real-time 4D rendering", there have already been many works combining 4DGS with semantics, such as 4-LEGS[1], 4D LangSplat[2], DHO[3]. Is the core difference between L4DGS and these works in "temporal consistency" or "dynamic semantic alignment"? Moreover, the text does not demonstrate how the semantic control of the editing object is achieved. Is there any difference from other semantic embedding methods?
2. Although the fusion of CLIP features and sparse attention is mentioned, there is a lack of quantification or visualization to verify language consistency (such as attention map visualization, semantic distribution similarity). For instance, qualitative alignment images of language prompts and rendering results, CLIP-score or image-Text retrieval accuracy and other metrics.
3. Both static and dynamic regularization have multiple λ hyperparameters, but in this paper, only "learnable hyperparameters" are mentioned, without ablation or stability analysis. Moreover, the sensitivity of SMSA parameters was not analyzed either. The influence of the "top-k value" (such as k taking 10, 20, 50) on performance was not analyzed, nor was the adaptive selection strategy of k explained. It is suggested to add: sensitivity experiments of λ, comparison of top-k values, and robustness verification for scenarios with different motion intensities or semantic complexities.

[1]Fiebelman G, Cohen T, Morgenstern A, et al. 4‐LEGS: 4D Language Embedded Gaussian Splatting[C]//Computer Graphics Forum. 2025: e70085.

[2] Li W, Zhou R, Zhou J, et al. 4d langsplat: 4d language gaussian splatting via multimodal large language models[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 22001-22011.

[3] Yan Z, Liang Y, Cai S, et al. Divide-and-Conquer: Dual-Hierarchical Optimization for Semantic 4D Gaussian Spatting[J]. arXiv preprint arXiv:2503.19332, 2025.

### Questions
1. The comparison with existing semantic rendering methods is insufficient. The innovative positioning needs to be strengthened. What are the essential differences from some methods based on 4DGS combined with semantics? The paper does not clearly define the core advantages of L4DGS in "4D dynamic support", "attention mechanism", and "regularization strategy", only mentioning "L4DGS supports dynamics", and does not quantitatively compare the performance gap between the two in dynamic semantic rendering.
2. How robust is the system to fuzzy prompts or combined prompts (for example, "The red chair near the window")?
3. How do the lamda and top-k hyperparameters in the paper affect the model performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents **L4DGS**, a lightweight real-time dynamic scene rendering framework that integrates natural language into a semantically structured 4D Gaussian representation. The proposed **Sparse Multi-Scale Attention (SMSA)** mechanism emphasizes semantically relevant regions in space and time, enabling fine-grained language-driven control. Furthermore, the combination of static and dynamic regularization effectively resolves temporal and semantic inconsistencies. The experiments are comprehensive and convincingly demonstrate the superiority of the proposed method as well as the effectiveness of each component.

### Strengths
- The paper proposes the first real-time 4D rendering algorithm with embedded language, demonstrating both effectiveness and innovation.
- By combining static and dynamic regularization, it effectively addresses issues such as semantic drift, flickering, and identity instability.

### Weaknesses
- The paper repeatedly mentions memory efficiency; however, the experimental section seems to lack sufficient discussion or quantitative analysis on memory consumption. It would be helpful to include additional experiments or data in this regard.
- The paper mentions CLIP-based features and language guidance but does not specify how textual prompts are used in training and testing. Are prompts fixed, varied per scene, or user-provided at inference? More examples ofprompt-to-rendering alignment would clarify practical usability.

### Questions
- Since SMSA guides L4DGS to focus on semantically important spatial regions, does it lead to any degradation in rendering quality for background areas?
- Could the proposed dynamic regularization cause loss of motion detail in fast-moving scenes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the probable visual and linguistic guidance for dynamic scene rendering. The authors propose the Sparse Multi-Scale Attention to better fuse vison and language features. Then, the novel static and dynamic regularizations are used to provide 3DGS with visual and linguistic guidance. The experiments show an improvement caused by the proposed methods.

### Strengths
1. The visual and linguistic guidance is probably useful to enhance dynamic scene rendering.

2. The experiments might demonstrate the effectiveness of the proposed methods.

### Weaknesses
1. This paper is not well-written. The pipeline is quite confusing. How do you obtain the language description for each scene to generate linguistic guidance? How do you get the rendered feature map F_rendered in Equation.3? What does the \lambda_o in Line 376 mean, and why the hyperparameters can be learnable? Which model do you use for guidance generation, CLIP or others, and could you cite the paper? How do you get GT depth map in Equation 4 for supervision? 

2. There might have several mistakes in the Paper. In Equation 4, how do you calculate the cosine similarity for depth, which is not a vector?

3. The motivation is not clear. Why visual and linguistic guidance is useful for dynamic scene rendering? I hope the authors could provide a deep insight explanation.

4. The experiments lack qualitative comparison. I think the author should visualize more results on the D-NeRF, HyperNeRF, Nerfies, long-sequence and iphone datasets.

5. Lack of comparison with SOTA methods. Deformable-3D-gs[1], SC-GS[2] and Grid4D[3] might have better rendering quality on the D-NeRF dataset.

6. The average PSNR in the last row of Table 2 in the supplementary might be incorrect, which should be 37.00.

7. I am confused about the training time of the proposed methods. As shown in Table 2, how do you realize extremely fast inference to get the visual and linguistic guidance from a large model while reducing the training time to 5min on the D-NeRF dataset? If possible, could the authors provide more details?
[1] Yang et.al. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. CVPR 2024.
[2] Huang et.al. SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes. CVPR 2024
[3] Xu et.al. Grid4D: 4D Decomposed Hash Encoding for High-Fidelity Dynamic Gaussian Splatting. NeurIPS 2024.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
