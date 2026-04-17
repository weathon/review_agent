# Rebenchmarking Unsupervised Monocular 3D Occupancy Prediction

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Inferring the 3D structure from a single image, particularly in occluded regions, remains a fundamental yet unsolved challenge in vision-centric autonomous driving. Existing unsupervised approaches typically train a neural radiance field and treat the network outputs as occupancy probabilities during evaluation, overlooking the inconsistency between training and evaluation protocols. Moreover, the prevalent use of 2D ground truth fails to reveal the inherent ambiguity in occluded regions caused by insufficient geometric constraints. To address these issues, this paper presents a reformulated benchmark for unsupervised monocular 3D occupancy prediction. We first interpret the variables involved in the volume rendering process and identify the most physically consistent representation of the occupancy probability. Building on these analyses, we improve existing evaluation protocols by aligning the newly identified representation with voxel-wise 3D occupancy ground truth, thereby enabling unsupervised methods to be evaluated in a manner consistent with that of supervised approaches. Additionally, to impose explicit constraints in occluded regions, we introduce an occlusion-aware polarization mechanism that incorporates multi-view visual cues to enhance discrimination between occupied and free spaces in these regions. Extensive experiments demonstrate that our approach not only significantly outperforms existing unsupervised approaches but also matches the performance of supervised ones. Our source code and evaluation protocol will be made available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an interpretable, opacity-based occupancy representation and introduces a coordinate-transformed sampling algorithm to align predictions with 3D voxel annotations. 
Furthermore, it designs an occlusion-aware occupancy polarization mechanism that leverages multi-view color discrepancies to provide supervisory signals for occluded regions. 
Extensive experiments on KITTI-360 validate the proposed benchmark's rationality and demonstrate good performance, showing competitive results against supervised methods.

### Strengths
1. The paper is well-written and easy to follow. The visual demonstration is also good.
2. The performance of the method is good compared with the single-view based methods.
3. The proposed occlusion-aware occupancy polarization is novel which effectively imposes constraints to occluded regions with different colors along the sampling ray.

### Weaknesses
1. Flawed analysis in the method section. Please see the first three questions in the Questions section.
2. Limited comparison. The paper does not compare with multi-view unsupervised occupancy prediction methods which can readily be applied for single-view task, such as SelfOcc.

### Questions
1. In Line 194-195, the paper claims that the rendering contributions of points B and C are theoretically the same. Further explanation is needed here since in rendering process, C is occluded by B and thus contributes less to the final result.
2. In Line 169-170, the paper claims that existing methods  adopt a fixed threshold of 0.5 to binarize each voxel. I also think this is unreasonable, but the paper should provide proper reference work here.
3. In Line 198-200, the paper analyzes the correlation between sampling interval and density value based on the assumption that the rendering contributions of points B in densely sampled region and C in sparsely sampled region should be the same. However, I think the assumption is incorrect, since densely sampled region would have more individual samples and thus the contribution of a single point in such a region does not have to comparable to that of a point in the sparsely sampled region. After all, the rendering process is a integration along the line.
4. What is the advantage of the proposed method compared with SDF-based representation without the need for a handcrafted threshold, and with methods based on 3D representations such as BEV / TPV without the need for post grid sampling?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper revisits and reformulates the evaluation protocols for unsupervised monocular 3D occupancy prediction—an increasingly important task in vision-centric autonomous driving. The authors systematically dissect the relationship between NeRF-based network outputs and voxel-wise occupancy ground truth, arguing that opacity ($\alpha$), rather than density ($\sigma$), yields a more robust and physically meaningful basis for evaluation. They propose a coordinate-transformed sampling algorithm to align opacity predictions with voxel grids, enabling fair benchmarking against supervised methods. Additionally, they introduce an occlusion-aware occupancy polarization loss, leveraging multiview cues to sharpen occupancy predictions in occluded regions. Experimental results on the KITTI-360 dataset show improved alignment between training and evaluation, competitive or superior performance over existing unsupervised and even some supervised baselines, and substantial advances in occlusion reasoning.

### Strengths
1. The paper thoughtfully revisits evaluation protocols and provides a clear, equation-rich justification for why opacity ($\alpha$) is preferable to network density ($\sigma$) for occupancy probability.
2. The methodology is generally well explained, algorithms are clearly laid out, supplemental details/figures are available, and the quantification of evaluation regions enhances reproducibility and interpretability.
3. Robust Experimental Suite: Quantitative and ablation results on KITTI-360 provide strong support for the method’s claims.

### Weaknesses
1. All results are on a single dataset (KITTI-360), and there is no exploration of other driving, indoor, or multi-modal datasets.  This raises questions about generalization and the broader applicability of the proposed evaluation protocol and methods.  Given the benchmark’s ambition for field-wide adoption, a demonstration on at least one additional, differently-distributed dataset would have been highly appropriate.

### Questions
1. How would the coordinate-transformed sampling and polarization loss perform on non-driving (e.g., indoor, or synthetic) datasets?  Are there failure cases or calibration routines required for generalization?
2. How sensitive are the results (especially occlusion region accuracy) to the choices of $\lambda_r, \lambda_p$, and threshold values for opacity?  Are there guidelines for robust tuning across differing scenes?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses critical limitations in existing unsupervised monocular 3D occupancy prediction, primarily the inconsistency between training and evaluation protocols and poor performance in occluded regions. It proposes an opacity-based occupancy probability representation to replace the scale-sensitive network output, resolving magnitude variation issues. A coordinate-transformed occupancy sampling algorithm is proposed to align radial opacity distributions with voxel-wise 3D ground truth. Furthermore, it proposes an occlusion-aware occupancy polarization mechanism using multi-view visual cues to enhance supervision in occluded areas. Experiments on KITTI-360 show the method outperforms unsupervised SOTA and matches supervised methods, while establishing a unified 3D benchmark.

### Strengths
1. This method identifies the inconsistency between point-wise rendering weight outputs and voxel-wise ground truth in existing NeRF-based methods, and uses opacity to resolve this, improving evaluation reliability.
2. The coordinate-transformed sampling effectively bridges the spatial gap between radial opacity and uniform voxel grids, enabling direct comparison between unsupervised and supervised methods.
3. The occlusion-aware polarization mechanism leverages color differences between adjacent points to supplement supervision in occluded regions.

### Weaknesses
1. The benchmark relies solely on the KITTI-360 dataset. No experiments are conducted on other datasets, such as nuScenes, to verify the method’s generalizability to different driving scenarios.
2. Qualitative results for occluded regions lack dedicated quantitative metrics, making it hard to objectively assess the polarization mechanism’s improvement on occlusion reasoning. Additionally, there is no clear explanation as to whether the improved ability of occlusion reasoning contributes to the safety of real-world autonomous driving.
3. Since the paper criticizes KYN for its high computational cost due to visual-language networks, it should provide a quantitative analysis of its own method’s inference efficiency.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Addressing the evaluation inconsistencies and occlusion modeling challenges in unsupervised monocular 3D occupancy prediction, this paper presents three core contributions: 1) Reinterpreting occupancy probability in neural radiance fields (NeRF) by replacing density σ with opacity α to resolve sampling-dependency issues; 2) Designing a coordinate-transformed occupancy sampling algorithm to align the distribution of α with voxel-wise ground truth, unifying the evaluation space for both unsupervised and supervised methods; 3) Proposing an occlusion-aware occupancy polarization mechanism that leverages multi-view color cues to supplement supervision in occluded regions. On the KITTI-360 dataset, the proposed method outperforms SOTA unsupervised methods and achieves intersection over IoU values comparable to or even exceeding some supervised methods. Ablation studies validate the rationality of each module. The source code and evaluation protocol will be made publicly available upon publication.

### Strengths
1. Clarity of Presentation: The paper is well-structured with clear explanations of concepts, methodologies, and experimental results.
2. Significance of Benchmark Construction: Constructing a benchmark for monocular 3D occupancy prediction addresses a critical gap in the field. As unsupervised monocular 3D occupancy prediction is essential for vision-centric autonomous driving, a dedicated benchmark contributes to standardized evaluation and fair comparison of subsequent methods, which is of great practical value.

### Weaknesses
1. **Limited Dataset Evaluation:** The paper only conducts experiments on the KITTI-360 dataset and lacks evaluation on mainstream autonomous driving datasets such as nuScenes. Mainstream datasets like nuScenes cover more complex scenarios, and evaluating only on KITTI-360 fails to demonstrate the generalizability of the proposed benchmark and method. This limits the reliability of the work’s conclusions regarding real-world applicability.
2. **Insignificant Performance Advantages:** As shown in Table 2, the proposed method does not significantly outperform other existing methods (e.g., ViPOcc). This weakens the persuasiveness of the method’s effectiveness, as readers cannot clearly perceive the added value of the proposed innovations compared to prior works.
3. **Lack of Coherent Contributions:** The research problems addressed in the paper lack a coherent main thread. The three proposed contributions are presented as relatively independent modules, failing to clearly articulate the core overarching issue the paper intends to solve. The work is overly engineering-oriented, with insufficient emphasis on the underlying scientific questions and systematic design logic of the benchmark.

### Questions
All questions are detailed in the "Weaknesses" section above.

### Soundness
2

### Presentation
2

### Contribution
2
