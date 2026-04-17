# ROADSIGHT: A novel dataset for real-time intersection detection in aerial scenes under seasonal variation

- Decision: Reject
- Scores: 6, 2, 0, 4

## Abstract
Road intersections serve as critical nodes in transportation networks, and their accurate detection from aerial imagery is helpful for applications such as autonomous navigation, urban planning, and traffic management. However, existing datasets for object detection in aerial views often lack specificity to intersections, diversity in seasonal conditions, and customized annotations for unmanned aerial vehicle (UAV) captured data, which leads to challenges in model robustness and real-time performance on edge devices. This work introduces a novel dataset named ROADSIGHT (Road-Oriented Aerial DataSet for Intersection detection on edGe Hardware plaTform), a UAV-captured dataset of high-resolution RGB images collected in both summer and winter conditions, with expert bounding-box annotations for two classes: roundabouts and intersections (encompassing 3-leg T/Y and 4-leg types). We benchmark state-of-the-art models(YOLOv8, YOLOv11, YOLOv12, and RT-DETR) and identify YOLOv11s as edge suitable. Robustness is validated through cross-validation with stable mAP. This work contributes a focused, seasonally varying benchmark and corresponding baseline results for real-time intersection detection on resource constrained UAV platforms, while clearly acknowledging current limitations in geographic coverage, adverse weather, and class granularity for future extensions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The study introduces ROADSIGHT, a novel dataset specifically designed for real-time intersection detection in unmanned aerial vehicle (UAV) imagery, notably addressing the crucial need for seasonal variation in aerial computer vision benchmarks. The paper establishes robust performance metrics using YOLO architectures and successfully demonstrates the practical viability of deploying an optimized model, YOLOv11s, on resource-constrained edge hardware, achieving real-time performance of 69 FPS on the NVIDIA Jetson Orin Nano. This work addresses limitations in prior datasets, which often lack intersection specificity or environmental diversity.

### Strengths
1. ROADSIGHT uniquely contains high-resolution UAV images captured during both winter and summer, including challenging conditions such as snow coverage and varied lighting angles, which enhance model robustness and environmental adaptability.
2. The research successfully validates the feasibility of real-time deployment on resource-constrained platforms. Specifically, optimizing the chosen YOLOv11s model using TensorRT FP16 quantization on the NVIDIA Jetson Orin Nano achieves an efficient inference time of 14.5 ms, equivalent to 69 FPS, which is necessary for real-time UAV navigation.
3. A systematic and rigorous benchmarking protocol was applied to multiple state-of-the-art YOLO architectures. The authors conducted ten independent evaluation runs, thereby establishing a robust baseline and successfully identifying YOLOv11s as the optimal model, balancing superior accuracy (0.676 mAP50–95) with efficient computational speed (6.668 ms).

### Weaknesses
1. The dataset scale is relatively modest and geographically limited. Containing 969 total images collected only from localized test areas in Finland, the limited geographic diversity may hinder the model’s generalization capacity when applied to regions with significantly different road infrastructures or climates.
2. The granularity of intersection classification is insufficient for detailed analysis. Although the initial collection included roundabouts, 3-leg (T/Y), and 4-leg intersections, the final annotations simplify these categories into only two: “roundabout” and a merged “intersection” class (for T/Y and 4-leg types). This limited classification may restrict the dataset’s value for applications requiring distinctions among specific intersection geometries.

### Questions
1. The authors need to detail the methodology used to derive the optimal loss function weights (0.5 for classification, 7.5 for bounding-box regression). Was this specific ratio determined empirically through multiple trials or based on theoretical considerations?
2. To justify the significant manual data curation effort, the authors should include a comparative benchmark showing the model’s accuracy when trained on the full initial raw image collection versus the final filtered dataset.
3. Although the final dataset uses only two merged classes, an analysis of the model’s performance under a finer-grained classification scheme (differentiating between roundabouts, 3-leg junctions, and 4-leg intersections) is highly recommended. This would validate the dataset’s utility for advanced traffic applications.
4. Although the dataset includes diverse conditions such as snow and overcast skies, the paper lacks quantitative validation of model robustness under more severe low-visibility scenarios, such as heavy fog. The authors should provide dedicated verification results in an appendix to fully substantiate their claims of enhanced environmental adaptability.

### Soundness
3

### Presentation
2

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
The paper introduces ROADSIGHT, a UAV-captured dataset for intersection detection in aerial images. It contains 969 images collected in Finland across summer and winter conditions. The dataset is evaluated using several YOLO variants. The authors also benchmark edge deployment on an NVIDIA Jetson Orin Nano.

### Strengths
1. The focus on real-time intersection detection from UAV images addresses a practical and under-explored problem important for traffic monitoring and autonomous navigation.
2. The paper is mostly clear and well structured, allowing readers to follow the dataset creation and evaluation pipeline without difficulty.
3. The inclusion of quantitative results on a low-power device (Jetson Orin Nano) is useful for practitioners considering real-time applications.

### Weaknesses
### Major issues
1. The text states three classes (roundabout, 3-leg, 4-leg) on line 102 and again on line 155 but then refers to only two classes at line 166 and in table 1. In addition, the labels shown in figure 3 do not appear to match either definition, as the second class seems to correspond to pedestrian crossings rather than intersections. This inconsistency raises concern about annotation reliability and must be clarified.  

2. Example images show substantial variation in bounding-box tightness and one roundabout that is only partially enclosed by its bounding box. The claimed “standardized protocol” (line 168) and second-annotator review are not supported by quantitative agreement metrics. A written annotation guideline or inter-annotator agreement analysis would be essential.  

3. Collapsing three-leg (T/Y) and four-leg intersections into one class loses valuable structural information. The decision should be justified empirically, as these layouts differ geometrically and limit the use cases of the dataset.  

4. The evaluation section reports basic detection metrics but omits core analyses such as:  
   - Ablation of augmentation techniques despite detailed hyperparameter lists.  
   - Analysis by season, even though seasonal variation is the central claim.  
   - Examination of typical failure cases or qualitative error patterns.  
   - Comparison with satellite-based data.  

5. The paper does not state whether the same intersections appear in multiple splits. Since data come from localised areas, spatial overlap could inflate reported performance. Clarification and ideally geographic separation of splits are required.

6. Descriptions such as “comprehensive evaluation” or “outperforming benchmarks on similar aerial tasks” (line 30) are unsupported: the paper does not compare against models trained on existing aerial-intersection datasets nor any other “similar aerial tasks”.

### Moderate issues
7. Table 2 merges both classes. Per-class distributions (bounding-box sizes, aspect ratios, split ratios) would help assess dataset balance.

8. In table 4, YOLOv11s is bolded without explanation. This model is not the best-performing in all metrics but is presented as “optimal.” Normally, bolding indicates the best score in that comparison, so the formatting is misleading. Table 5 follows the same pattern without justification.

9. In figure 4 D, the “all-classes” recall–confidence curve sometimes exceeds per-class curves, implying micro-averaging. The averaging method should be explicitly stated.

10. Data originate solely from Finland with limited weather variation (mostly clear or cloudy). Claims of diversity or “different weather” (line 144) are slightly overstated. The dataset lacks rain, fog, snowing, dusk, or night conditions, which are especially relevant if the dataset is intended for emergency or autonomous navigation scenarios where adverse weather is common.

11. The paper overlooks synthetic aerial and drone datasets that already include road or intersection classes. For instance, Syndrone [1], MidAir [2] and DDOS [3] provide synthetic UAV images for scene understanding and object detection. A discussion comparing ROADSIGHT with such datasets would strengthen the paper’s positioning.


### Minor issues
12. The “ROADSIGHT” backronym feels unnecessary and forced. Simply calling it ROADSIGHT without the backronym is fine.  
13. Excluding blurry or poorly exposed images removes potentially valuable hard negatives. A short discussion would improve transparency.
14. At line 272, the authors state that class imbalance “reflects real-world distributions.” A supporting reference would be appropriate to substantiate this claim.
15. Figure 2 adds limited value.


[1] Rizzoli, Giulia, et al. "Syndrone-multi-modal uav dataset for urban scenarios." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Fonder, Michael, and Marc Van Droogenbroeck. "Mid-air: A multi-modal dataset for extremely low altitude drone flights." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2019.

[3] Kolbeinsson, Benedikt, and Krystian Mikolajczyk. "DDOS: the drone depth and obstacle segmentation dataset." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
1. What was the precise annotation protocol? Were three classes annotated and later merged?
2. Are the train, validation and test sets geographically independent (no repeated intersections)?
3. How does performance vary between seasons?
4. Could the removed blurry or low-quality images serve as hard-negative examples?


---
I hope the authors will use this feedback to refine their dataset and analyses. This work could make a useful contribution to the field if the identified weaknesses are addressed, and I encourage the authors to continue developing and evaluating the dataset further.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a UAV image dataset for detecting road intersections with seasonal variation (winter/summer), annotated as roundabout vs. intersection, and benchmarked with YOLO variants.

### Strengths
- Introduces a dataset that is missing from the overall community.

### Weaknesses
- This paper reads more like a technical report rather than a research paper. 
- The data collection process is quite standard. 
- I would expect to see dataset specific analysis. Type of road segments which are more important based on transportation studies
- The motivation is lacking. Why is a UAV necessary to detect a fixed infrastructure that does not change much. Please elaborate to provide a better sense of 
- For the type of data and number of images the YOLO performance is not surprising. 
- The real-time aspect and edge deployment are not so relevant to the dataset and do not add to the contributions.
- Only YOLO families are reported. For a dataset paper, include at least one non-YOLO baseline (e.g., RT-DETR/DETR-like or a lightweight transformer detector) to avoid family-specific bias and show the dataset’s value across architectures.

### Questions
- How did you check for data leakage? Maybe a better way is to do K-fold cross validation and report the average over the folds?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a dataset, ROADSIGHT, which contains of high-resolution UAV-captured images for detecting road intersections in different seasons. The images include annotations of road intersections, categorized into two types: roundabouts and intersections (including 4-leg and 3-leg intersections). The annotations are evaluated on YOLOv11 for edge-optimized performance, ensuring real-time performance on resource-constrained edge devices. This fills a gap in current datasets that detect intersections themselves, and can improve traffic safety and geospatial applications, preventing frequent traffic accidents at intersections.

### Strengths
- Current UAV datasets lack intersection detection capabilities; this dataset fills a crucial gap by constructing a high-resolution, multi-seasonal UAV image dataset.
- Considering the robustness and real-time performance issues of models on real-world edge devices, a quantized model is used to enable real-time detection on UAVs.
- The experimental design is rigorous and detailed, with comprehensive evaluation perspectives.

### Weaknesses
- The image dataset suffers from homogeneity and exhibits limited detection and processing of complex terrains.
- The paper compares the size differences between the roundabouts and intersections datasets and explains the reasons, but the intersections dataset itself lacks classification information, and its geographical conditions are relatively uniform, raising concerns about dataset diversity.
- The paper focuses on intersection images in summer and winter, but fails to adequately showcase corresponding environmental factors, such as rain and snow conditions.

### Questions
- The dataset primarily contains images of intersections with minimal occlusion, but the reality is that intersections often experience significant dynamic occlusion and disturbance. The question remains whether effective annotation can be achieved under these conditions.
- What is the basis for the intersection classification granularity? Does this granularity adequately encompass all types of intersections and ensure good generalization performance of the subsequent recognition model?
- What is the basis for determining the recognition boundary of intersections in the intersection dataset? How are boundary standards unified for different types of intersections?

### Soundness
3

### Presentation
3

### Contribution
2
