# SurstSplat: Dynamic Surgical Gaussian Reconstruction  with Spatiotemporal Graph Matching

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Reconstructing dynamic 3D models from surgical videos is crucial for advanced medical applications, but faces challenges from limited textures, inconsistent lighting, and complex tissue deformations. We present \method, a framework that enhances dynamic Gaussian reconstruction through spatiotemporal semantic graph matching. By integrating multimodal features from pre-trained 2D foundation models into 3D Gaussian representations, our approach effectively captures tissue deformations and tool interactions. The spatiotemporal graph matching mechanism improves handling of deformable tissues over standard Gaussian methods while enabling real-time semantic segmentation, language-guided editing, and medical visual question answering. Experiments demonstrate that \method~enhances rendering quality in challenging surgical conditions and allows clinical 3D models to leverage pre-trained 2D multimodal foundation models. Our approach improves both rendering quality and computational efficiency, supporting advanced intraoperative applications and advancing robot-assisted surgery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SurstSplat, a dynamic Gaussian Splatting method for endoscopy that augments Gaussians with 2D foundation-model features and adds a spatiotemporal semantic graph-matching regularizer. The unified field supports novel-view rendering, promptable segmentation, language-guided editing, and VQA. Results show small gains over strong dynamic GS baselines on EndoNeRF/EndoVis.

### Strengths
- Solid engineering: a good systems combo of dynamic GS \+ semantic features \+ space–time regularization.  
- Consistent small improvements over strong dynamic baselines on clinical datasets.  
- Useful capability bundle from one representation (novel-view semantics, editing, VQA) with a coherent clinical motivation.

### Weaknesses
- Novelty is significantly overstated. Integrating 2D FM features into 3DGS is established (“Feature 3dgs”, Zhou et al. CVPR 2024; “Langsplat”, Qin et al., CVPR 2024). Similarly 4D gaussians and even integrating 2D features into 4D gaussians is already established (“4d langsplat”, Li et al. CVPR 2025\)  
- Sec. 2.1 and Sec 2.2 are largely background from prior dynamic GS and semantic GS works and should be framed as such.  
- Confounders: the method likely benefits heavily from strong teacher models (SAM-H, CLIP L/14, LLaVA-Med) compared to the SOTA. The paper does not adequately control/ablate for teacher strength, making it hard to attribute improvements to the proposed method rather than to newer/better features.  
- The graph matching does not seem to be making a significant difference as can be seen in table 5\.

### Questions
- Novelty/positioning: Please explicitly acknowledge that feature-augmented 3D/4D GS is prior art and restate what you believe to be your novelty.  
- Teacher-model confounding: How sensitive are your gains to teacher strength? Please provide controlled ablations (e.g., SAM-B vs SAM-H; CLIP base vs large; LLaVA-Med vs alternatives) with identical training to show improvements aren’t primarily due to stronger external models. Also clarify whether any baselines used the same teacher setup to ensure fairness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SurstSplat, one method for dynamic surgical scene reconstruction using 3dgs enhanced with semantic features distilled from 2D foundation models (e.g., SAM, CLIP, LLaVA-Med). A spatiotemporal graph matching mechanism is proposed to enforce feature consistency across time, enabling applications such as semantic segmentation, language-guided scene editing, and visual question answering. Experiments on surgical benchmarks (EndoVis17/18, EndoNeRF) show that SurstSplat achieves strong rendering quality and semantic performance, with claimed real-time inference speed.

### Strengths
Well-motivated integration of foundation model features into 3D Gaussias for semantic understanding.
Temporal graph matching yields improved consistency in dynamic surgical scenes.
Broad evaluation across multiple datasets and tasks, including rendering, segmentation, and VQA.
Real-time performance metrics now included (FPS > 100).
The writing and structure have been improved.

### Weaknesses
1. The proposed spatiotemporal graph lacks mechanisms for handling topological scene changes (e.g., cutting, splitting), which are common in surgical procedures.
1. Language-guided editing is only demonstrated for instrument removal, raising concerns about the generality of this functionality.
1. Segmentation results appear to be benchmarked against SAM-generated outputs rather than manual ground truth, which could bias reported performance.
1. The method’s generalizability to non-surgical scenes remains unproven despite being architecturally generic.
1. The method claims high visual stability and novel capabilities (e.g., temporal consistency, editing), yet provides no video comparisons to prior methods. I really encourage the authors to providing substantial video comparisions to demonstrate the effectiveness of the paper,.

### Questions
1. How does the graph model handle emergence/disappearance of Gaussians during topological events (e.g., new tissue surfaces appearing after cutting)?
2. Can you confirm whether segmentation metrics are computed against human-annotated ground truth or against outputs of SAM?
3. Have you explored language-guided editing for non-tool objects (e.g., highlighting or removing tissue)?
4. Is the use of LLaVA-Med necessary for all tasks, or could certain models (e.g., SAM only) suffice in practice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- Given an endoscopic surgical video (2d+time), submission 406 aims to fit a dynamic Gaussian splatting model to it, such that it can assign a set of a query-able, editable, and promptable set of features to each Gaussian. 
- To do so, it follows established work in feature splatting that uses foundation models such as SAM and CLIP to assign features to each Gaussian and extends these models to the spatiotemporal setting. 
- As its primary new contribution, it also creates a spatiotemporal graph of foundation model features and regularizes this graph in order to gain spatiotemporal consistency. 
- Experimentally, it shows high-level overviews of better results across several downstream tasks such as semantic segmentation, question answering, and more.

### Strengths
- The breadth of the downstream tasks explored in the experiments section is large and is appreciated. 
- The qualitative results are quite nice. The framework appears to be well executed at least based on what is visualized in the main text.
- The opening paragraph of the methods section is the clearest paragraph in the whole paper and conveys the contributions of the paper well. It should be in the introduction.

### Weaknesses
### Claims:

The paper significantly overclaims its technical contributions. While distilling vision (or vision-language) foundation model features into Gaussian Splatting frameworks for editing and querying has been extensively explored in the literature (examples: - [1](https://feature-splatting.github.io/), [2](https://feature-3dgs.github.io/), [3](https://arxiv.org/abs/2405.18424), [4](https://arxiv.org/abs/2312.16084)), the main text of the paper comes off as if it is presenting it as a novel technical angle that has not been explored before. Moreover, the spatiotemporal aspect of the paper is standard dynamic 4D Gaussian Splatting with the aforementioned feature splatting component integrated. 
  
What is *actually* new in the proposed paper is the spatiotemporal graph matching regularization. However, the main text of the paper conveys all of it as novel, which is definitely not the case. All of this could have been avoided with a well written related works section, but the paper chooses to include only an abbreviated few paragraphs in the appendix. 

As such, while the graph matching seems modestly useful going by the ablation in Table 5, it is much more incremental than is let on by the rest of the text. 

### Experiments:
- While the experiments section covers a vast array of downstream tasks, it is indecipherable how any of it was executed. Neither the main text nor the appendix contains any baseline implementation details, any experimental details, any technical details about how the features are used downstream to generate segmentations, or answer language prompts, etc. It comes across as more of a high level overview and needs significantly more depth. Also, how were the baselines tuned for the downstream datasets? Were the hyperparameters swept?
- All of the contributions contained within this paper appear to be generic and not tied to medical imaging in any way. It is then unclear why it is limited to experiments on endoscopic datasets instead of including other medical/surgical applications or even natural video datasets. Including 1--2 more datasets would go a long way in demonstrating that the proposed method is not overfit to the very specific endoscopic application.
- Table 1 (and others): these PSNR/SSIM numbers seem to be inconsistent with numbers reported on the same dataset in the corresponding baseline papers. For example, EndoGaussian reports much higher PSNR on the EndoNeRF dataset than what it achieves here. What is causing this inconsistency?

### Technical presentation:

- The paper's technical writing can be significantly improved as it takes until half way through the paper to find out what is actually the graph referenced throughout the paper up until that point. As a result, the methods reads like a grab-bag of tricks and it is very hard to parse what is actually new or important there.

### Minor:
- Please use `\citep` or `\citet` to have the references show up correctly in the ICLR template.
- The arrows in Figure 2 and the overall layout is extremely confusing. Please edit to improve its clarity.

### Questions
- Please clearly disambiguate what is the core contribution of this work.
- Please address the experimental questions raised above.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present SurstSplat, a framework for dynamic Gaussian reconstruction with spatiotemporal semantic graph matching. The proposed graph matching regularizes semantics across space and time and supports downstream tasks such as semantic
segmentation, language-guided editing, and medical visual question answering.

### Strengths
1. This is an interesting paper that could support downstream tasks such as semantic segmentation, language-guided editing, and medical visual question answering. In a sense, it has great potential to be adopted widely.
2. The Medical VQA analysis is promising.

### Weaknesses
1. I don't see the significant testing of this paper, and I won't be able to tell how significant these results are compare with current SOTA methods without the rigorous testing. I would recommend to perform such testings for improving the soundness of this paper.
2. The figure wrapped inside of the text is really hard to read and I believe it should be separated from the text.
3. Is there any failure cases (figures)? Why they are not being included in the paper for comparison and discussion?

### Questions
Please refer to the Weakness

### Soundness
3

### Presentation
3

### Contribution
3
