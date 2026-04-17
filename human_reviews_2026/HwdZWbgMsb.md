# BEEP3D: Box-Supervised End-to-End Pseudo-Mask Generation for 3D Instance Segmentation

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
3D instance segmentation is crucial for understanding complex 3D environments, yet fully supervised methods require dense point-level annotations, resulting in substantial annotation costs and labor overhead.
To mitigate this, box-level annotations have been explored as a weaker but more scalable form of supervision.
However, box annotations inherently introduce ambiguity in overlapping regions, making accurate point-to-instance assignment challenging.
Recent methods address this ambiguity by generating pseudo-masks through training a dedicated pseudo-labeler in an additional training stage. 
However, such two-stage pipelines often increase overall training time and complexity, hinder end-to-end optimization.
To overcome these challenges, we propose BEEP3D—Box-supervised End-to-End Pseudo-mask generation for 3D instance segmentation.
BEEP3D adopts a student-teacher framework, where the teacher model serves as a pseudo-labeler and is updated by the student model via an Exponential Moving Average.
To better guide the teacher model to generate precise pseudo-masks, we introduce an instance center-based query refinement that enhances position query localization and leverages features near instance centers.
Additionally, we design two novel losses—query consistency loss and masked feature consistency loss—to align semantic and geometric signals between predictions and pseudo-masks.
Extensive experiments on ScanNetV2 and S3DIS datasets demonstrate that BEEP3D achieves competitive or superior performance compared to state-of-the-art weakly supervised methods while remaining computationally efficient.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies 3D instance segmentation using only 3D bounding boxes. The authors propose BEEP3D: a single-stage training framework that generates pseudo-masks online under a student-teacher scheme where the teacher is updated via EMA from the student; instance-center–guided positional queries for refinement; and two consistency losses, query consistency and mask-feature consistency, to align representations. The method achieves near fully supervised AP on ScanNetV2 and S3DIS and removes the need for a separately pre-trained pseudo-labeler.

### Strengths
* **End-to-end pseudo-mask generation**

  * Integrates the pseudo-labeler into the training loop in a single stage, avoiding extra pretraining and simplifying the pipeline and T′ cost.
  * Provides a clear comparison with two-stage methods in terms of pipeline and parameter freezing.

* **Closeness to fully supervised upper bound**

  * On ScanNetV2, the “% full” relative to the corresponding fully supervised method reaches about 98%. Importance: strong performance at the cost of weak supervision.
  * Maintains competitiveness across AP, AP50, and AP25 thresholds. Importance: robustness under different IoU requirements.
  * Achieves leading AP on S3DIS Area 5 as well. Importance: cross-dataset effectiveness.
  * Works with multiple backbones (MAFT, SPFormer). Importance: compatibility.

### Weaknesses
* **Insufficient statistical significance and reproducibility information**

  * Does not report variance across multiple runs, confidence intervals, or significance tests.
  * Random seeds and data split/resampling strategies are not specified.
  * No links to code and model weights or a licensing plan are provided.

* **Confirmation bias and error propagation in the pseudo-label loop**

  * The teacher is obtained via EMA of the student; early student errors may be locked into the target ( $\hat{m}_u \cup m_l$ ).
  * Lacks comparisons to “frozen teacher” or “no teacher” alternatives to bound the loop’s benefit.

* **Limited evidence for generalization and noise robustness**

  * Evaluation is limited to ScanNetV2 and S3DIS, which are similar domains.
  * No systematic degradation curves under box noise, offsets, or scale perturbations (related to the Sketchy-3DIS setting).
  * No stratified analysis of cases with heavy occlusion or dense overlap.
  * The ISBNet variant shows “negligible” gains without analysis of causes.
  * Pseudo-mask quality (mACC) is computed only on the training set; no validation-set measure is reported.

### Questions
* **Statistical significance and reproducibility**

  * Will you provide links to the code and model weights, including environment files and training scripts?

* **Confirmation bias and error propagation in the pseudo-label loop**

  * What is the effect of varying the EMA decay {0.90, 0.95, 0.99} and the update frequency on AP and training stability?
  * Can you add and explain ablations with a frozen teacher and with no teacher to quantify the loop’s upper bound and its necessity?

* **Generalization and noise robustness**

  * Can you evaluate on additional domains (e.g., different building styles) and report at least cross-domain validation results?
  * Will you sweep box center offsets, scale perturbations, and size noise, plot degradation curves, and compare to a Sketchy-style perturbation protocol?
  * Can you provide a hard-case analysis with AP and visualizations on subsets that exhibit heavy overlap and strong occlusion?

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
The paper tackles the challenge of 3D instance segmentation under weak supervision, specifically from 3D bounding boxes instead of dense point-level annotations. The authors propose BEEP3D, a student-teacher framework where the teacher generates pseudo-masks that guide the student network in an end-to-end manner. Unlike prior methods that either require pretraining a separate pseudo-labeler or rely on heavy geometric priors, BEEP3D integrates pseudo-mask generation directly into training. It uses instance center–based query refinement instead of employing learnable parameters. Two consistency losses are introduced to align the student and teacher representations. The framework is implemented on MAFT, ISBNet, and  SPFormer models, and tested on the ScanNetV2 and S3DIS benchmarks, where it showed competitive results compared to fully supervised methods. The method also cuts training complexity compared to previous weakly supervised methods.

### Strengths
- The method achieves near-supervised accuracy on two standard datasets.
- The writing is mostly clear, the figures are illustrative, and the ablation studies are well-structured. Visual results show improved segmentation in overlapping regions.
- Unlike BSNet or GaPro, BEEP3D eliminates an extra pseudo-labeling stage, reducing training time without sacrificing accuracy.

### Weaknesses
- The key ideas proposed in this paper (EMA teacher-student updates, query refinement, and consistency-based losses) are well-established concepts adapted from semi-supervised and weakly supervised learning to 3DIS. 
- The authors note that when integrated with ISBNet (a non-transformer-based network), BEEP3D underperforms compared to BSNet on S3DIS. This suggests that the framework’s advantages rely heavily on transformer-based query mechanisms, limiting applicability to other architectures.
- While the paper explicitly positions BEEP3D as transformer-based, it tests only on MAFT and SPFormer. Since there are several other transformer architectures for 3DIS (e.g., Mask3D, OneFormer3D, QueryFormer), broader evaluation across more transformer variants would strengthen claims of generality even within its intended paradigm.

### Questions
- Can the authors analyze pseudo-mask quality over training epochs to substantiate claims about refinement?
- Can the authors evaluate their framework across a wider set of transformer-based 3DIS methods (1 or 2 extra) to strengthen their evaluation on the two chosen datasets?
- There seems to be a minor issue with the citation format throughout the paper.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces BEEP3D, a novel framework for 3D instance segmentation using only 3D bounding box supervision. The core challenge is the ambiguity from overlapping boxes. BEEP3D cleverly solves this with an end-to-end student-teacher framework, where the teacher model acts as a pseudo-labeler and is updated via an EMA of the student. This unifies pseudo-mask generation and segmentor training into a single stage.

### Strengths
1. The end-to-end, single-stage paradigm is a significant advancement over prior multi-stage methods (e.g., GaPro, BSNet). It's elegant, efficient, and avoids reliance on simulated data or complex priors.

2. The instance center-based query refinement is a smart strategy to leverage the most reliable signal from the weak supervision. The dual consistency losses provide robust supervision for the student model.

3. The method achieves state-of-the-art (or highly competitive) performance on ScanNetV2 and S3DIS. Impressively, it closes the gap to fully-supervised methods, achieving 98.1% of the full-supervision AP on the ScanNetV2 validation set.

4. As shown in Table 5, the method eliminates the separate pre-training time ($T'$) for pseudo-label generation, making it efficient and practical.

Clarity: The paper is well-written, and the method is presented clearly.

### Weaknesses
1. Evaluation on ScanNet++: It would be valuable to include results on ScanNet++, which offers more challenging indoor scenes and richer geometric details. This would further demonstrate the generalization ability of the proposed method.


2. Failure Case Discussion: Adding a brief discussion of failure cases (e.g., in extremely cluttered scenes) would provide a more complete picture of the method's limitations and guide future work.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose BEEP3D, an end-to-end framework for weakly supervised 3D instance segmentation using only 3D bounding box annotations. The method employs a student-teacher architecture where the teacher model is updated via EMA of the student model and serves as a pseudo-labeler. To improve pseudo-mask quality in ambiguous overlapping regions, the framework introduces two key techniques: (1) instance center-based query refinement that leverages center coordinates from bounding boxes as strong priors to guide the teacher model, and (2) two consistency losses (query consistency and masked-feature consistency) to ensure alignment between student and teacher models at both query and feature representation levels.

### Strengths
1.	BEEP3D integrates pseudo-label generation and segmentation model training within a unified training loop. This eliminates the need for separate pre-training stages, significantly simplifying the training pipeline and improving overall efficiency.
2.	The method exploits strong geometric priors implicit in box annotations, specifically instance center coordinates. By enforcing the teacher model's position queries to consistently aggregate these center points, it provides robust spatial guidance for the model. 
3.	To ensure effective knowledge transfer from teacher to student, the paper designs two novel consistency losses.

### Weaknesses
1.	Experimentally, on the validation set, the method achieves relatively limited AP improvements. More critically, it underperforms BSNet+MAFT across both AP₅₀ and AP₂₅ metrics. On the test set, the absence of corresponding performance data from competing methods prevents fair comparison. Additionally, comparisons with the latest state-of-the-art methods are missing. 
2.	The method's core innovations, particularly Instance Center-based Query Refinement and Query Consistency Loss, heavily rely on Transformer-specific query mechanisms, making them relatively incremental. It is not a general weakly supervised approach, when applied to non-Transformer architectures, performance even degrades, as shown in Table 2 where Ours + ISBNet underperforms BSNet + ISBNet. 
3.	The Instance Center-based Query Refinement critically depends on accurate instance center points extracted from bounding boxes. The design does not address method robustness when box annotations contain noise (common in weak supervision). Meanwhile, the hard arg max assignment in pseudo-mask generation may lead to rapid performance degradation when teacher model predictions are inaccurate (error accumulation).

### Questions
1.	Could the authors provide error analysis to better understand failure cases and model limitations?
2.	In Table 1 validation set, although BEEP3D's AP (57.3) is slightly higher than BSNet (56.2), it shows comprehensive deterioration in both AP₅₀ and AP₂₅ metrics. Does this indicate that BEEP3D sacrifices instance detection recall to optimize high-IoU segmentation precision? Can the authors explain the underlying causes of this critical metric regression?
3.	Given that student and teacher models share identical network architecture, a more straightforward self-training baseline seems viable: using only a student model where predictions at step N generate pseudo-masks to supervise training at step N+1. Can the authors justify the necessity of adopting the student-teacher framework compared to this more direct self-training loop?

### Soundness
2

### Presentation
2

### Contribution
2
