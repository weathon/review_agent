# Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Pretrained semantics from large vision models have enabled major advances in open-vocabulary robot policies, e.g., in manipulation and navigation.
However, a striking lack of consensus on the performance and effects of fine-tuning these vision encoders remains a significant challenge.
For example, some papers claim that (task-specific) pretrained encoders outperform general-purpose semantic encoders (e.g., DINO) or that fine-tuning vision encoders improves performance, while others claim the exact opposite.
In this work, we seek to address these long-standing divisions through a principled examination of pretrained semantics from vision encoders in robotics.
We hypothesize that the inconsistencies in prior work arise from a fundamental lack of insight into the feature content of these vision encoders.
Hence, we undertake a systematic study of pretrained semantics in distilled fields to uncover their salient components with the goal of identifying a framework that explains previously contradictory claims.
Specifically, we ask: *what do the semantic features of robotics vision encoders contain?* — and consider visual-semantic encoders (like DINO) and geometry-grounded encoders (like MUSt3R/VGGT).
Notably, we find that these encoders attend to different features in their image inputs. While visual-semantic encoders prioritize object/part-level semantic decomposition, geometry-grounded encoders may discard this information to focus on more structural components, such as edges and corners. 
This observation can be described by catastrophic forgetting of core semantic information, which worsens with increased fine-tuning. 
We validate these findings in two major robotics problems: semantic object localization and radiance field inversion, using distilled fields as a testbed. We observe results consistent with the internal contents of the semantic features of these encoders, highlighting the strong explainability afforded by internal probes.  
For semantics-focused radiance field inversion, we propose a novel framework SPINE
using distilled semantics for coarse inversion followed by a fine inversion procedure with photometric-based optimization, without an initial guess, demonstrating its superior performance compared to competitive alternatives.
Further, our results suggest that geometry-grounding could offer potential benefits if catastrophic forgetting is controlled.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper conducts an empirical study to compare the effectiveness of visual-only semantic features (from DINOv2, DINOv3) against visual-geometry semantic features (from VGGT) when distilled into 3D radiance fields (Gaussian Splatting and NeRFs). The investigation is structured around three core questions relevant to robotics applications:
1. Do visual-geometry features contain higher-fidelity spatial content?
2. Does geometry-grounding improve semantic object localization?
3. Can visual-geometry features enable higher-accuracy radiance field inversion (i.e., camera pose estimation)?

To facilitate the third question, the authors propose SPINE, a novel framework for radiance field inversion that does not require an initial pose estimate. The key findings are that while geometry-grounded features (VGGT) do capture finer geometric details, they do not offer an advantage in semantic localization and surprisingly underperform visual-only features (DINOv2) in the task of pose estimation. The authors conclude that visual-only features currently offer greater versatility for downstream tasks in distilled fields.

### Strengths
- The paper addresses a fundamental and highly relevant question: what kind of pretrained features are most effective for 3D scene understanding in the context of radiance fields? Comparing modern visual-only foundation models like DINO with emerging visual-geometry models like VGGT provides valuable insights for the community.
- The study is well-organized, comparing two types of features on two different radiance field representations (NeRF and GS) across three distinct and important downstream tasks.
- Beyond a comparative study, the paper introduces SPINE, a novel method for radiance field inversion that works without an initial camera pose guess.
- The results are somewhat counter-intuitive; one might expect geometry-grounded features to excel at a geometric task like pose estimation. The finding that visual-only features from DINOv2 perform better is surprising and important.

### Weaknesses
- **Overly Broad Claims** from a Single Model: The paper's claims about "visual-geometry features" as a general class are based on experiments with a single model, VGGT. While VGGT is a strong representative, it is possible that its specific pre-training objective is what limits its versatility, rather than the principle of geometry-grounding itself. The conclusions should be carefully worded to reflect that the findings are specific to the models tested, avoiding generalization to all possible geometry-grounding techniques.
- The paper presents SPINE as a method for pose estimation, but its training protocol and generalization capabilities are not clearly described.
- The authors use perceptual metrics (SSIM, PSNR, LPIPS) to evaluate semantic localization accuracy. While the rationale is noted, this is a departure from standard practice for localization/segmentation tasks. These metrics measure the similarity of the relevancy heatmaps, not how accurately the target object is isolated. Reporting results across a range of thresholds with mIoU would make the results more convincing.

### Questions
The paper tackles a very interesting problem with a well-structured set of experiments and introduces a novel pose estimation framework (SPINE). The findings are surprising and thought-provoking. However, the work is held back by a few significant weaknesses: the conclusions about geometry-grounded features are drawn from a **single model**, and the evaluation for semantic localization uses non-standard metrics in my opinion. These issues prevent a confident recommendation for acceptance in its current form. I believe the paper has high potential, and I would be willing to reconsider my rating if the authors can address these points in a revision.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper revisits the role of geometry-grounded vision backbones (e.g., VGGT) versus visual-only ones (e.g., DINOv2/v3) in semantic distillation for radiance fields. It examines three key questions: Do geometry-grounded features contain richer spatial information? Do they improve semantic object localization? And do they enable more accurate radiance field inversion (pose estimation)?
The authors propose SPINE, a novel inversion framework using semantic features to estimate camera poses without initialization, followed by photometric refinement. Surprisingly, results show that geometry-grounded features do not outperform visual-only ones on localization or inversion, though they contain more geometric detail.

### Strengths
1. This is the first systematic analysis comparing visual-only vs. visual-geometry semantic embeddings in radiance fields, a gap overlooked by existing works like LERF, CLIP-NeRF, or DFF.
2. The study spans multiple datasets (LERF, 3D-OVS, Robotics), two radiance field types (NeRF, GS), and multiple metrics (GFF, SSIM, PSNR, LPIPS, SE(3) error).

### Weaknesses
1. The description of SPINE’s inverse model is conceptually clear but lacks comparative or sensitivity analyses: How critical are the semantic embeddings vs. the photometric refinement? How do different backbone dimensions (e.g., CLIP 512 vs 768) affect inversion? What is the runtime/efficiency cost of SPINE relative to baseline pose estimators?

2. The paper concludes that geometry-grounded semantics hurt versatility but offers no concrete explanation beyond “supervised inductive bias.”

3. Semantic localization results are mostly relative (DINO vs VGGT). It would be more informative to include absolute comparisons against: CLIP-only localization (as in LERF), Geometry-only cues (depth or SDF features). 

4. The term "semantic embedding" is misleading in L161. VGGT is not trained with semantic supervision, and I don't believe its intermediate features can be called semantic embedding. 

5. The core idea, comparing geometry-grounded vs. visual-only features in semantic radiance fields, is primarily evaluative.
SPINE, the only algorithmic contribution, is a straightforward combination of: a shallow MLP mapping semantic embeddings to poses, and standard PnP-based refinement. Both steps are well established in prior literature (e.g., iNeRF [Yen-Chen 2021], CatNIPS [Chen 2024], and Splat-NAV [Chen 2025]). SPINE’s novelty lies mostly in not requiring an initial guess, but the authors never prove that it consistently converges without one — the results appear scene-specific and qualitative.
Overall, the paper feels like an empirical case study rather than a fundamentally new framework or theoretical advance.

6. Mathematical sections contain excessive exposition of standard concepts (e.g., SVD, Sobel operator). Section 4 devotes a full paragraph to PCA projection details, which are trivial and distract from the main analysis.

### Questions
1. Could you provide any intuition or analysis for why geometry grounding degrades downstream task performance?
2. Does SPINE generalize across unseen scenes, or is it trained per scene like a NeRF?
3. You conclude that “visual-only features offer greater versatility.” How general is this conclusion? Do you believe this holds for all geometry-grounded models (e.g., MVDream, Depth-DINO, GeoCLR), or only VGGT? What would you recommend for future work — redesigning geometry-grounding losses or simply abandoning geometry-grounded semantics for robotics applications?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether geometry-grounded vision backbones, specifically VGGT, can provide advantages over purely visual backbones such as DINOv2/DINOv3 when distilled into radiance fields for robotics-centric tasks. The authors evaluate three downstream capabilities: semantic content fidelity, open-vocabulary object localization, and radiance-field inversion. The study concludes that geometry-grounded features exhibit sharper geometric structure but do not improve semantic localization performance and even degrade pose inversion accuracy. The paper additionally introduces SPINE, a scene-specific inversion module that leverages semantic cues to recover camera pose without initialization.

The central message is that geometry-enhanced features do not necessarily translate into broader utility in 3D-aware semantic radiance fields, and that purely visual embeddings remain more versatile for downstream tasks.

### Strengths
1. Addresses a timely and relevant question regarding the real-world value of geometry-grounded semantics for 3D robotics perception.

2. Provides systematic comparisons across multiple semantic backbones and tasks.

3. Experimental setup is generally clear and the negative results are informative for the community.

### Weaknesses
1. **Novelty is limited**
   - The core technical pipeline largely reuses existing radiance-field semantic distillation approaches, with the primary modification being the substitution of pretrained feature sources.
   - SPINE follows a standard design (semantic prior + photometric refinement + PnP/RANSAC) and is trained per scene, further reducing novelty in system design.

2. **Scope of geometry-grounded models is insufficient**
   - Only VGGT is evaluated. Contemporary spatially grounded models such as DUSt3R, MASt3R, CroCo, or other geometric transformer variants are omitted.
   - The conclusion that geometry-grounding harms versatility is based on a narrow model sample and may not generalize.

3. **Lack of mechanistic insight**
   - The paper observes performance degradation with geometry-grounded features but does not provide clear hypotheses or analysis explaining why geometry hurts semantic versatility.
   - Without deeper investigation, the conclusions may appear anecdotal.

4. **Incomplete evaluation of scalability and practicality**
   - SPINE is trained per scene, similar to NeRF-style pipelines, raising concerns about scalability in real robotic deployments.
   - No demonstration on larger-scale scenes, dynamic environments, or cross-scene generalization.

5. **Overall maturity not yet sufficient**
   - Although the question is valuable, the current implementation resembles an exploratory empirical study rather than a fully developed methodology.
   - Lack of ablations on distillation design (e.g., shared hashgrids, separate semantic heads, language-only vs geometry-only distillation) limits interpretability.

### Questions
1. Would the observed trend hold for other geometry-grounded models such as DUSt3R, MASt3R, or other spatial transformers?
2. Can the authors provide more in-depth analysis explaining why geometry-grounding compromises semantic versatility? For example, changes in embedding smoothness, gradient stability, or photometric consistency?
3. Is SPINE capable of generalizing across scenes, or can it be made scene-agnostic? If not, how do the authors envision scaling it in real robotic deployments?
4. Could separate encodings rather than shared hash-grids improve geometry-grounded feature distillation?
5. Do any tasks exist where the geometry-grounded semantic fields *do* offer measurable benefits?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors explore the effectiveness of vision-only and visual-geometry features on downstream tasks including edge computation, object localization, etc. 

Additionally, the authors propose a SPINE framework for inverting radiance fields. More precisely, SPINE predicts poses directly from learned image features and then refines the poses by solving a PnP problem.


Experiments on public datasets demonstrate that (1) visual-geometry features contain higher fidelity spatial content than visual-only features; (2) both features give close performance on semantic object localization; (3) visual-only features give higher accuracy radiance field inversion.

### Strengths
1.	Interesting topic. Recently, distilling semantic knowledge from foundation models (e.g. CLIP) into NeRFs or Gaussian Splatting is a hot research topic as also mentioned in the introduction. However, comparisons of the effectiveness of the visual-only features (e.g. DINO v2/v3) and visual-geometry features (e.g. VGGT) on different tasks are still open problems. Therefore, I believe the topic of this paper is interesting and the conclusions obtained from the experiments are useful. 

2.	The paper is well-organized. 

3.	The experiments, although conducted on a limited number of datasets, are relatively convincing.

### Weaknesses
1.	Limited contribution. The main contribution of this paper is the comparison of the effectiveness of visual-only and visual-geometry features on downstream tasks. Although it is interesting and useful, the contribution is not enough as an ICLR paper.

2.	The second contribution of the paper is the SPINE framework for inverting radiance fields. However, the SPINE is essentially a pipeline for end-to-end relocalization task which takes features as input and predicts the 6-DoF camera poses. There are many works in this area [R1, R2, R3]. It would be better to give a discussion of this task.   

[R1] PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization, Kendall  et al., ICCV 2015.

[R2] Learning multi-view camera relocalization with graph neural networks, xue et al., ICCV 2020.

[R3] The NeRFect Match: Exploring NeRF Features for Visual Localization, zhou et al., ECCV 2024.

### Questions
It seems like some details of the method are missing, which causes the low readability, for example:

1.	The details of f_l, f_s.

2.	How to the match before PnP for pose estimation.

### Soundness
2

### Presentation
2

### Contribution
1
