# OVID: Open-Vocabulary Intrusion Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Various vision intrusion detection models have achieved great success in many scenarios, e.g., autonomous driving, intelligent monitoring, and security, etc. However, their reliance on pre-defined classes limits their applicability in open-world intrusion detection scenarios. To remedy these, we introduce the Open-Vocabulary Intrusion Detection (OVID) project for the first time. Specifically, we first develop a novel dataset, Cityintrusion-OpenV for OVID,  with more diverse intrusion categories and corresponding text prompts. Then, we design a multi-modal, multi-task, and end-to-end open-vocabulary intrusion detection framework named OVIDNet. It achieves open-world intrusion detection via aligning visual features with language embeddings. Further, two simple yet effective strategies are proposed to improve the generalization and performance of this specific task: (1) A Multi-Distributed Noise Mixing strategy is introduced to enhance the location information of unknown and unseen categories. (2) A Dynamic Memory-Gated module is designed to capture the contextual information under complex scenarios. Finally, comprehensive experiments and comparisons are conducted on multiple dominant datasets, e.g., COCO, Cityscape, Foggy-Cityscape, and Cityintrusion-OpenV. Besides, we also evaluate the universal applicability of our model in real scenarios. The results show that our method can outperform other classic and promising methods, and reach strong performance even under task-specific transfer and zero-shot settings, demonstrating its high practicality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces OVID, an “open-vocabulary intrusion detection” setting, a new dataset (Cityintrusion-OpenV) built from Cityscapes with 8 intruder classes plus AoI masks, and a baseline model OVIDNet built on OpenSeeD with two add-ons: Multi-Distributed Noise Mixing (MDNM) for box jitter and a Dynamic Memory-Gated (DMG) module. Results show modest gains over OpenSeeD and prior intrusion systems, plus qualitative demos.

### Strengths
S1. Problem formulation: Open-vocabulary intrusion detection as a two-stage process: first produce text-conditioned boxes and masks, then make an intrusion decision by measuring overlap with the area of interest. This separation makes the approach easy to follow.

S2. Dataset contribution: Cityintrusion-OpenV extends prior intrusion datasets to eight intruder categories and pairs every image with area-of-interest masks and text prompts. It reports about eighteen interactions per image and gives class-wise counts. This supports both open-vocabulary evaluation and the intrusion decision.

S3. Plug-and-play memory module: The Dynamic Memory-Gated block retrieves a small set of scene features from a fixed memory and fuses them with the decoder through a learned gate. It is a lightweight add-on that aims to capture repeated road layouts and common spatial patterns in driving scenes. The module can be dropped into similar transformer decoders without major changes.

S4. Multi-Distributed Noise Mixing perturbs boxes rather than images. It mixes uniform, gaussian, and Laplace jitters with weights and scales the jitter by box area using a log-like schedule. Small boxes get fine perturbations, large boxes get broader jitter. This is well aligned with an intrusion task where a few pixels of overlap can flip the decision.

### Weaknesses
W1. The zero-shot part is mainly domain shift (COCO→Cityscapes). The novel-class aspect is unclear because most classes (person, car, bus, truck, train, motorcycle, bicycle) appear in COCO. Provide a base/novel split where some intruder classes are withheld during training and evaluated only via text prompts (standard OVD protocol). Compare to strong OVD baselines under the same split. (Table 2, Table 10 show no explicit novel-class setup.)

W2. The intrusion decision uses overlap > 20 pixels (fixed) following Sun et al. (Appendix A.1; Sec. 4.2), independent of image scale (800 vs 1200, Table 9 p. 17). This is confusing to me. Report analyses with a normalized overlap (e.g., fraction of box pixels inside AoI) and calibration curves/ROC, and show sensitivity to the threshold t.

W3. The paper argues that OVD+OVS pipelines are costly (Appendix A.5 Table 8 p. 16) but does not report numbers. Please include GroundingDINO(+CLIP) + panoptic/road segmenter (e.g., OpenSeeD/SAM-family) as a task baseline, and a pure OpenSeeD baseline with the exact same training/compute. Also test YOLO-World style detectors with AoI overlap. This is necessary to show OVIDNet improves over obvious modular alternatives. 

W4. Improvements over OpenSeeD are ~+2.19 PQ and +3.43 Acc (Table 2 p. 7), and category-wise results contain zeros (e.g., Rider = 0 in several tables; Fig. 6 p. 9). The practical impact of MDNM/DMG is seems modest; please include statistical variance (≥3 seeds) and report confidence intervals, if you have it.

W5. Because of GPU limits, training uses 15k iters and 800px images (Table 9 p. 17). This deviates from OpenSeeD’s stronger recipe (368,750 iters, 1200px). Comparisons must either match schedules/crops or adjust for capacity (e.g., stronger OpenSeeD backbones). 

W6. Many visuals imply Road is the AoI (Fig. 5 p. 8; Fig. 10 p. 18), but practical systems often include sidewalks/rails or multiple AoIs. Please clarify whether stuff classes beyond Road are supported and evaluate multi-AoI cases.

W7. Why selectively synthetic fog is only evaluated? For open-world deployment, other distribution-shifts such as common corruptions [1] (rain, snow, so on adapted to Cityscapes) and viewpoint change and perspective distortion [2] (Möbius/MPD augmentation). The scope and reasoning should be clearly defined in paper. Such investigation on robustness of AoI–box consistency under viewpoint change and artifacts can provide worthful insights.
[1] - Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." arXiv preprint arXiv:1903.12261 (2019).
[2] - Chhipa, P. C., Chippa, M. S., De, K., Saini, R., Liwicki, M., & Shah, M. (2024, September). Möbius transform for mitigating perspective distortions in representation learning. In European Conference on Computer Vision (pp. 345-363).

### Questions
Besides mentioned weakness, following are few questions for authors - 
Q1. What are the exact novel classes (if any) absent from training in your zero-shot claims?
Q2. How is class text prompted at test time—single label (“bus”) or templates (“a photo of a bus”)?
Q3. Did you try class re-weighting or few-shot prompts to fix Rider?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Open-Vocabulary Intrusion Detection (OVID), which defines a new task, provides a Cityscapes-derived dataset (Cityintrusion-OpenV), and proposes a multi-modal, multi-task framework (OVIDNet) that aligns image features with text prompts to jointly perform detection, segmentation, and intrusion judgment. The method uses pixel overlap between predicted boxes and AoI masks to identify intrusions. Two components, Multi-Distributed Noise Mixing (MDNM) for adaptive perturbations and a Dynamic Memory-Gated (DMG) module for context, are designed to improve zero-shot generalization and open-world robustness. Reported gains over OpenSeeD are +2.19 PQ and +3.43 intrusion accuracy under lighter training settings (800 resolution, 15K iterations). The dataset extends prior intrusion benchmarks from 4 to 8 categories with Y/N labels and text prompts (18.03 cases per image). Qualitative results on ShanghaiTech and UA-DETRAC show promptable deployment without retraining, although quantitative metrics are missing for those examples.

### Strengths
The paper is the first to formally define open-vocabulary intrusion detection and presents a multi-modal framework that combines detection, segmentation, and intrusion judgment. It connects open-vocabulary detection with security applications in an interesting and practical way.

The Cityintrusion-OpenV dataset expands existing intrusion datasets to 8 categories and introduces text prompts with a much higher number of Y/N cases per image (18.03), which is a solid step toward open-world evaluation.

The experimental coverage is broad, including zero-shot transfer (COCO→Cityscapes), task-specific transfer (Cityintrusion-OpenV), fog domain testing, ablations, and qualitative real-world demonstrations.

The ablations for MDNM and DMG demonstrate how different configurations (α, β, γ values, and memory unit sizes) affect performance and reveal consistent, though modest, improvements.

The figures and algorithm descriptions are clear and easy to follow, and the appendix includes sufficient details for reproduction.

The approach shows potential for practical deployment, as it can handle new categories via prompts without retraining and appears somewhat robust under foggy conditions.

### Weaknesses
The baseline comparisons are not entirely fair. The authors trained with a reduced image size (800) and fewer iterations (15K), while OpenSeeD used 1200 and 368,750. Without matching settings or reporting variance over multiple seeds, it is hard to interpret the reported gains.

The ShanghaiTech and UA-DETRAC results are only qualitative, but the paper makes strong claims of universal applicability. Quantitative results would be needed to back that up.

The technical novelty is limited since OVIDNet largely builds on OpenSeeD with an added intrusion decision module. MDNM’s mixture of distributions and logarithmic area scaling lacks theoretical justification, and DMG feels similar to standard attention mechanisms without a clear distinction.

The reported improvements are small (1–3 percent), and in some cases, performance drops, showing that gains are not stable.

The absolute performance remains low (32.79 percent intrusion accuracy), and the Rider class fails completely (0.0 percent) due to confusion with Person and class imbalance. The paper does not address this failure.

Comparisons with other strong open-vocabulary detection systems, such as YOLO-World and Grounding DINO, are missing, which weakens the SOTA claim.

The dataset labeling relies on an automatic 20-pixel overlap rule with limited manual verification. There is no inter-annotator agreement, threshold sensitivity analysis, or bias auditing.

### Questions
Please follow Strengths and Weaknesses. 

Why is only selectively synthetic fog evaluated? For open-world deployment, it would be important to also consider other distribution shifts such as common corruptions [1] (rain, snow, and similar variations adapted to Cityscapes) and viewpoint or perspective changes [2] (for example, Möbius or MPD augmentations). The paper should clearly define the scope and reasoning behind focusing solely on fog. Exploring how AoI–box consistency behaves under viewpoint changes and visual artifacts could provide valuable insights into the model’s robustness.

[1] - Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." arXiv preprint arXiv:1903.12261, 2019. 
[2] - Chhipa, P. C., Chippa, M. S., De, K., Saini, R., Liwicki, M., & Shah, M.  Möbius transform for mitigating perspective distortions in representation learning. In European Conference on Computer Vision, 2024.

What happens if the models are trained at higher resolution (1200) and for longer schedules? Do the reported gains hold?



I am open to revising my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the first Open-Vocabulary Intrusion Detection task, builds the Cityintrusion-OpenV dataset (with 8 intrusion categories), and designs the end-to-end OVIDNet framework. Then they provide OVIDNet, their end-to-end framework that pairs CLIP for text and a tiny Swin Transformer for images, checking overlaps between bounding boxes and AOIs to judge intrusions.

### Strengths
Open-vocab was a gap in intrusion detection, and the Cityintrusion-OpenV dataset provides new annotations for this task.

### Weaknesses
1. The core logic for “intrusion judgment” feels rather vague. As it stands, the method appears to detect any object that appears on the road, rather than accurately identifying what constitutes an intrusion. Although the paper mentions using the overlap between bounding boxes and the AoI to determine intrusions, it never clearly defines what constitutes an intrusion in different scenarios. For instance, are regular vehicles considered intruders? What about pedestrians or bicycles? Are there specific time periods or restricted zones where entering the AoI becomes an intrusion? Without these clarifications, the task essentially boils down to “object detection plus overlap checking,” rather than genuine intrusion detection, which should focus on identifying unauthorized entries. In fact, if the goal is simply to detect everything on the road, this can be easily achieved using an MLLM or by combining a segmentation model with an open-vocabulary detector.
2. Two core improvements lack mechanism proof. For the Multi-Distributed Noise Mixing strategy, the paper only states it mixes three types of noise and adjusts ratios dynamically, but fails to explain why this specific noise combination enhances unknown category location information or how dynamic ratios avoid damaging normal features. For the Dynamic Memory-Gated Module, it only describes the structural process (global average pooling → memory retrieval → feature fusion) without clarifying why the query vector matches memory units effectively.
3. The method performs poorly on basic categories like “Person” and “Rider,” weakening its practicality for common intrusion scenarios.
4. The figures are too small to be visible clearly, hindering verification of visualization results.

### Questions
- When the paper mentions “multi-domain,” is it only normal and foggy weather? Most tests switch between those two, but there’s no word on other domains like rainy or night. 
- Line 168 has a typo—“detention” should be “detection.”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper establishes a new task of dynamic-view Open Vocabulary Intrusion Detection, including a correlated dataset Cityintrusion-OpenV. An end-to-end model, OVIDNet, is designed as a baseline for the proposed new benchmark. Two strategies are proposed to improve the generalization and performance of OVIDet.

### Strengths
1. The proposed dataset extends the future research area of vision intrusion detection.
2. The overall method is easy to follow.
3. Comprehensive experiments and comparisons are conducted to verify the effectiveness of the proposed framework and methods.

### Weaknesses
1. The designed Multi-Distributed Noise Mixing Strategy is an incremental enhancement of the noise generation in OpenSeeD.
2. While the Dynamic Memory-Gated module is designed to capture the contextual information under complex scenarios, no discussion or analysis demonstrates whether this challenge is inherently significant in OVID. It is also unclear how this module functions in the overall model.
3. The open-vacabulairy capability relies on existing OVD and OVS practices. The overall method lacks in-depth insight into tackling the proposed OVID task.
4. While the established challenge is an open-vocabulary problem, the experiments are conducted under zero-shot and task-specific transfer settings (lines 306-307).
Other typo issues, including but not limited to
1. Line 010, using e.g. and etc. in the same sentence.
2. Line 173, "(D_T^d, D_V^s)" . I believe this should be "(D_T^d, D_T^s)"
3. Line 315, "duo to"

### Questions
1. What does "model singularity" in line 088 mean?
2. What does \textbf{B} denote in Table 4 and Table 5?
3. Is it sufficient to solve the OVID problem by adding the intrusion judgment capability to an open-vocabulary detection and segmentation model? In this work, the implementation of intrusion judgment is simple and learning-free, which suggests that simply combining the two should work for the OVID task. The authors should make an in-depth discussion or analysis to reveal the inherent significance of the proposed OVID task.

### Soundness
2

### Presentation
3

### Contribution
2
