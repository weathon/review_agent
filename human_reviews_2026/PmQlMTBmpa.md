# Beyond Visual Reconstruction Quality: Object Perception-aware 3D Gaussian Splatting for Autonomous Driving

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Reconstruction techniques, such as 3D Gaussian Splatting (3DGS), are increasingly used to generate scenarios for autonomous driving system (ADS) research. Existing 3DGS-based approaches for autonomous-driving scenario generation have, through various optimizations, achieved high visual similarity in reconstructed scenes. However, this route is built on a strong assumption: that higher scene similarity directly translates into better preservation of ADS behaviour. Unfortunately, this assumption has not been effectively validated, and ADS behaviour is more closely related to objects within the field of view rather than the global image.
Thus, we focus on the perception module—the entry point of ADS. Preliminary experiments reveal that although current methods can produce reconstructions with high overall similarity, they often fail to ensure that the perception module outputs remain consistent with those obtained from the original images. Such a limitation can significantly harm the applicability of reconstruction in the ADS domain. To address this gap, we propose two complementary solutions: a perception-aligned loss, which directly leverages output differences between reconstructed and ground-truth images during training; and an object zone quality loss, which specifically reinforces training on object locations identified by the perception model on ground-truth images. Experiments demonstrate that both of our methods improve the ability of reconstructed scenes to maintain consistency between the perception module outputs and the ground-truth inputs. We release code at: https://github.com/Shanicky-RenzhiWang/Perception-aware-3DGS

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a gap in 3D Gaussian Splatting for autonomous driving: existing methods optimize global visual similarity, but this does not ensure consistent perception outputs, which are crucial for ADS behavior. It proposes perception-aware reconstruction with two losses: a perception-aligned loss penalizing output inconsistencies, and an object-zone quality loss emphasizing task-relevant regions. Experiments show improved perception stability while maintaining visual fidelity.

### Strengths
This paper presents novel evaluation metrics that specifically assess perception performance, which is meaningful for autonomous driving simulation.

### Weaknesses
1. One weakness is that the results appear to be highly dependent on the chosen perception model. While testing with Faster R-CNN, more baseline methods should be included for comparison.

2. As shown in Table 2, the Perception-Aligned Loss leads to a drop in SSIM for some methods. While the idea of using perception scores is meaningful, the trade-off between visual quality and perception alignment is not clearly analyzed.

3. The Object Zone metric is fairly straightforward and not novel, as similar concepts such as Dynamic Region PSNR have been used in prior works.

### Questions
To clearly demonstrate the effect of each component, a combination of Object Zone Quality Loss and Perception-Aligned Loss integration should be evaluated in Table 3.

### Soundness
3

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
5

### Summary
This paper points out that existing 3D Gaussian Splatting (3DGS) methods for street-scene reconstruction overlook the essential requirement of preserving autonomous driving system (ADS) behavior. The authors argue that optimizing solely for visual reconstruction performance fails to ensure consistency in downstream perception modules. To enhance the reconstruction quality of object instances, the paper introduces two complementary losses: Perception-Aligned Loss and Object-Zone Quality Loss. Experimental results demonstrate the effectiveness of the proposed approach in improving perception consistency without compromising visual fidelity.

### Strengths
1. The paper is well-motivated, making its ideas easy to follow.
2. Simple yet effective methods: perception-aligned and object-zone losses are easy to implement and show consistent gains.
3. Comprehensive evaluation: multiple base 3DGS models, perception backbones, and visual/perceptual metrics are tested.

### Weaknesses
1. The writing and figures need to be refined, especially in the Introduction and Method section. The figures contain text that is too small to read clearly and would benefit from clearer visual design and labeling.
2. The Broader Applicability discussion provides limited new insights. The paper would be stronger if it included a deeper analysis or ablation of how the proposed methods generalize across different tasks or domains.
3. The performance improvement appears somewhat incremental. Evaluating the method on additional and more diverse scenarios could better demonstrate its effectiveness and practical significance.

### Questions
1. Would an adaptive λ scheduling strategy (balancing Lvisual vs Lperc/Lobj) further improve trade-offs?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies that existing 3D Gaussian Splatting (3DGS) methods for autonomous driving focus on visual quality but neglect the generation performance regarding to downstream perception task.. To address this, the authors propose two simple yet effective losses: (1) Perception-aligned loss, aligning reconstruction with detection results, and (2) Object-zone quality loss, emphasizing reconstruction quality on detected object regions. Experiments on Waymo scenes using S3Gaussian, OmniRe, and EMD show improved mAP, mean IoU, and fewer missed detections, while maintaining similar SSIM/PSNR.

### Strengths
Important and Well-Motivated Problem: The paper clearly identifies a fundamental limitation in current 3D reconstruction pipelines for autonomous driving simulation. The proposed concept of perception stability offers a more meaningful and task-relevant objective than traditional visual fidelity metrics.

Simple, Efficient, and Effective Solution: The proposed loss term, $\mathcal{L}_{\text{obj-vis}}$, is intuitive, easy to integrate, and incurs minimal computational overhead, yet consistently yields substantial gains in perception stability and reconstruction quality.

### Weaknesses
+ Model-Specific Limitation:
The framework’s dependence on a single “ground-truth” perception model ($\mathcal{P}$) may restrict the generality of the reconstructed scenes, potentially reducing their usefulness for training or evaluating diverse perception architectures.

+ Limited Perception Scope:
The study defines “perception” solely in terms of 2D object detection, overlooking other essential autonomous driving tasks such as 3D detection, semantic segmentation, and multi-object tracking.

### Questions
As discussed in the weakness:

+ Generalization:
How does $\mathcal{L}_{\text{obj-vis}}$ perform when trained with one detector (e.g., YOLO) but evaluated using a fundamentally different model family (e.g., DETR-based architectures or 3D detectors)?

+ Task Scope:
Have you explored extending the framework to other perception tasks—for example, leveraging semantic segmentation masks to emphasize safety-critical pixels? Additionally, does the proposed approach improve temporal stability or downstream planning performance in autonomous driving systems?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper points out that existing 3D Gaussian Splatting methods focus on global visual fidelity, which does not necessarily preserve the perception outputs of autonomous-driving detectors. To address this, the authors introduce two simple training objectives—a perception-aligned loss and an object-zone quality loss—that encourage consistency between reconstructed and original images at the detector level. Experiments on Waymo scenes show improved detection consistency (mAP/IoU/Miss) over several 3DGS baselines without degrading visual quality.

### Strengths
1. The insight that “high SSIM ≠ stable perception” is novel and meaningful. Bridging 3DGS reconstruction and detector consistency provides a new angle on how generative scene reconstructions might better serve ADS evaluation.

2.  Despite the conceptual simplicity, the proposed objectives yield measurable mAP/IoU gains across multiple 3DGS baselines and even transfer to unseen detector architectures, indicating that the effect is real and not detector-specific.

3. The paper is concise, well-structured, and easy to follow; figures and tables are clear, and the motivation flows logically from observation → method → experiment.

### Weaknesses
1. Limited evidence for “visual-quality ≠ perception.” The claim is mainly supported by SSIM vs mAP differences in one table. Including PSNR, LPIPS, or correlation plots would make the observation statistically more convincing.

2. Running a full YOLO inference at every iteration introduces heavy overhead. While the authors mention this qualitatively, no quantitative efficiency table or training-time comparison is given.

3. The work evaluates only object detection, whereas ADS perception also involves segmentation, tracking, and planning. Without any closed-loop or end-to-end evidence, it remains unclear how perception-stability improvements benefit actual ADS behaviour.

4. The perception-aligned loss is computed after a detector that includes NMS and hard thresholding, which are non-differentiable. The paper does not explain how gradients flow through this step or whether only pre-NMS continuous outputs are used. As written, the formulation is theoretically questionable.

### Questions
Please clarify the gradient pathway in the perception-aligned loss: is NMS bypassed, and how are matched detections determined when counts differ?

To substantiate the “visual ≠ perception” claim, report correlations among SSIM, PSNR, LPIPS, and mAP, or plot them across scenes.

Add an efficiency table (e.g., training time per 1k frames or per epoch) comparing baseline 3DGS, + L_perc, and + L_obj.

Consider a lightweight feature-alignment proxy (e.g., backbone-feature L2/KL) to avoid full detector forward passes.

If possible, include a simple downstream experiment  to show whether perception-stable reconstructions yield tangible ADS gains.

### Soundness
3

### Presentation
3

### Contribution
3
