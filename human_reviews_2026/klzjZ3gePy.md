# Reference-based Category Discovery: Unsupervised Object Detection with Category Awareness

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Traditional one-shot detection methods have addressed the closed-set problem in object detection, but the high cost of data annotation remains a critical challenge. General unsupervised methods generate pseudo boxes without category labels, thus failing to achieve category-aware classification. To overcome these limitations, we propose Reference-based Category Discovery (RefCD), an unsupervised detector that enables category-aware detection without any manually annotated labels. It leverages feature similarity between predicted objects and unlabeled reference images. Unlike previous unsupervised methods that lack category guidance and one-shot methods which require labeled data, RefCD introduces a carefully designed feature similarity loss to explicitly guide the learning of potential category-specific features. Additionally, RefCD supports category-agnostic detection without reference images, serving as a unified framework. Comprehensive quantitative and qualitative analysis of category-aware and category-agnostic detection results demonstrates its effectiveness, and RefCD can learn category information in an unsupervised paradigm even without category labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The submission focused on the task of unsupervised object detection, and proposed Reference-based Category Discovery. Specifically, the proposed RefCD achieves category-aware detection without labels. Besides, the authors propose a feature similarity loss to encourage the detector to mine the categor information. The experiments are conducted on two large-scale benchmark datasets, which indicate the effectiveness of the proposed method.

### Strengths
1. The task of unsupervised object detection is interesting and fundmental in the field of computer vision.

2. The idea of building amodal segmentor based on SAM is reasonable and make sense.

3. The performances of proposed method are shown on various benchmark datasets, and outperform baselines by a large margin.

### Weaknesses
1. The description is confuzing in Line 40, what's the relationship between one-shot detection and unsupervised object detection? Do the authors focus on one-shot detection? The task should be consistently and clearly defined.

2. The task motivation and task-wise comparison are unclear. Compared with One-shot Detector, RefCD also uses reference images, but why is it without class label? What's the difference between the reference images in two tasks? More distinctions should be presented, e.g., including detailed training settings and test settings.

3. Why not train the reference encoder?

4. Is there any more recent baseline in single object tracking?

5. Closely related works focused on learning novel classes with category-agnostic proposal and similarity loss could be complemented to enrich the related works. The similar frameworks are effective supports for method design.

   > 1. "Weak-shot semantic segmentation via dual similarity transfer." Advances in Neural Information Processing Systems 35 (2022): 32525-32536.
   >
   > 2. "Weak-shot fine-grained classification via similarity transfer." Advances in Neural Information Processing Systems 34 (2021): 7306-7318.

6. Some typos found in Line 76 and Line 252.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an unsupervised reference-based object detection framework, RefCD, designed to enable open-set object detection without dependence on human annotations. The proposed method first performs category-agnostic detection to localize potential objects in an image, followed by category-aware matching that leverages feature similarity between reference images and target objects to identify instances of specific categories. A central contribution of this work is the introduction of a Feature Similarity Loss (FS Loss), which encourages the model to capture latent category structures among objects during unsupervised training. Extensive experiments on multiple benchmark datasets indicate that RefCD attains competitive performance in unsupervised object detection and achieves results comparable to several supervised approaches, demonstrating its robustness and potential applicability in annotation-free settings.

### Strengths
* This work proposes a concise and well-structured pipeline that integrates self-supervised representations, Hungarian assignment, and similarity-based objectives. Each component is reasonably motivated and supported by empirical evidence, contributing to the coherence and effectiveness of the overall approach.
* The presentation of the study is clear and systematic, with a transparent problem formulation, well-defined objectives, and detailed optimization and implementation procedures. The descriptions of pseudo-box generation and reference encoding are particularly helpful, and the accompanying figures and qualitative analyses enhance the interpretability of the results.
* The experimental evaluation is thorough, covering a range of datasets and task settings, and includes comparisons with both unsupervised and supervised baselines. The analysis offers useful observations regarding loss variants, similarity measures, and Top-K selection, and also includes an examination of challenging cases such as crowded or occluded scenes.
* The work provides meaningful evidence that label-free, open-set, category-aware detection is feasible and, in certain cases, can approach the performance of supervised methods. This finding highlights the potential for flexible category specification without retraining and suggests a possible path toward reducing annotation costs in Open Vocabulary Object Detection.

### Weaknesses
* The model’s performance appears to rely heavily on the semantic richness of the DINOv2 ViT features (Table 6). In domain-specific scenarios where such pretraining is unavailable or less effective, the generalization and stability of RefCD remain uncertain.
* The study employs four sets of reference images for evaluation (Appendix A.7). However, other works may adopt different template sets. Without a publicly shared reference pool and accompanying scripts, direct cross-paper comparison of quantitative results is difficult to ensure.
* Table 11 combines results from fully unsupervised training (using pseudo boxes) and box-only, label-free training on COCO BASE, which in practice constitutes weak supervision. Clarifying this distinction would help prevent potential overstatement of the unsupervised setting.
* The category-aware detection results are influenced by several hyperparameters—thresholds, the sigmoid temperature (set to 10), and the choice of top-K queries (Table 3; Fig. 4). A more systematic analysis would provide deeper insight into the robustness of these design choices.
* While Fig. 5 demonstrates qualitative success in extending RefCD to single-object tracking, quantitative results on standard tracking benchmarks are absent, which limits the strength of this extension claim.

### Questions
* Beyond the configurations reported in Table 7, have the authors conducted a broader sweep over $\alpha \in {1,\dots,4}$ and $\beta \in {3,\dots,7}$? Additionally, how sensitive is the model to the fixed temperature parameter (set to $10$), and how does it interact with the similarity threshold used for category-aware matching?
* Instead of selecting a single strongest positive per reference, have the authors explored treating queries with high pseudo similarity $p_n > \tau$ as soft positives, or directly regressing $\hat{p}_n$ toward $p_n$? It would be helpful to know whether such strategies yield measurable improvements in recall or category coverage.
* Could the authors provide an ablation or visualization illustrating the impact of the Top-$K$ value (Table 3) and discuss robustness to batch size, as well as the calibration of the foreground confidence head that determines the Top-$K$ queries?
* Appendix A.7 indicates that the reference templates include both COCO and ImageNet images. Are the comparisons with SINE and UNICL-SAM conducted on exactly the same reference set? If not, could the authors include a sensitivity analysis to quantify the effect of template variation?
* In Table 11, the “COCO BASE w/o category labels” configuration effectively constitutes box-only (weakly supervised) training. It would be beneficial to clarify this setup in the main text and report its results separately from those of the fully unsupervised setting.
* Figure 4 suggests that adjusting the similarity threshold enables finer-grained subclass separation (e.g., dog breeds). Would employing a multi-threshold or re-calibration mechanism further mitigate subclass confusion or reduce false positives at inference?
* For the crowded, composite, or occluded scenes discussed in Appendix A.4, have the authors considered incorporating mask-level similarity or iterative box-refinement mechanisms to improve localization? If so, any preliminary findings would be valuable to include.

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
This paper addresses the challenge of unsupervised object detection, where the goal is to detect objects without access to annotated category labels. It introduces RefCD (Reference-based Category Discovery), a novel framework inspired by supervised one-shot object detection methods. Similar to one-shot paradigms—which use a reference image to localize instances of the same class—RefCD leverages feature similarity between a reference image and candidate regions to perform category-aware object detection in an unsupervised setting.

### Strengths
The paper proposes a novel perspective on unsupervised object detection by combining reference images with feature-similarity-based matching, opening a new direction in this research area.

Extensive experiments show strong performance, with RefCD outperforming existing methods in unsupervised object detection benchmarks.

### Weaknesses
The paper attempts to differentiate its approach from one-shot object detection. However, the objective of the proposed method is essentially the same as one-shot object detection; the primary distinction lies in the training procedure. While the work present a new problem setting that they argue differs from existing detection paradigms, the main distinction appears to be the replacement of human annotation with unsupervised detection method, such as CutLER. This reduces the strength of the contribution.

Since the reference images are unlabeled, it is unclear how the method distinguishes between different object categories, i.e., how one can determine whether two reference images correspond to different classes. Additionally, if a reference image contains only a partial view of an object (e.g., just the wheels of a car), it is unclear whether the model should detect the full object (the car) or only the referenced part (the wheels). This ambiguity raises questions about how RefCD handles partial-object references and category granularity.

### Questions
In Figure 1, if we rename “deer” as ref1 and “capybaras” as ref2, what is the difference in the detection objective between (a) and (c)?

### Soundness
3

### Presentation
3

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
The paper proposes a method for unsupervised, reference-based detection and category discovery by assuming that similarity in feature space corresponds to implicit category membership, applying a relatively simple architecture and loss formulation to leverage that assumption, and demonstrating competitive empirical results.

### Strengths
The experimental analysis is good and extensive

### Weaknesses
The paper assumes that feature similarity corresponds to implicit category membership, intuitively plausible,  but how does the method cope in cases of high intra-class variation? For example, if in a scene there are red apples and green apples, and the reference annotation is on red apples, can the method reliably detect green apples of the same “apple” category?

Although the motivation is strong and the experiments fairly extensive, the architecture and loss design remain relatively straightforward, which suggests the contribution may lie more in the integration of existing modules rather than in fundamentally new methodology. Can the authors clarify which parts of the method are novel beyond existing unsupervised object-discovery and detection frameworks?

Specifically, there already exists a line of work exploring exemplar-based or reference-guided detection and recognition across various domains—including detection, counting, and segmentation. In the detection community specifically, methods such as  [1-3] have also utilized exemplar or reference inputs to guide detection, though typically in supervised or semi-supervised settings. In parallel, several text-driven approaches, for example  [4-5], conditioning detectors on textual prompts rather than visual exemplars.

[1]Large-Scale Unsupervised Object Discovery,  
[2]OS2D: One-Stage One-Shot Object Detection,  
[3]Siamese DETR 
[4]OWL-ViT: Simple Open-Vocabulary Detection with Vision-Language Models,    
[5]RegionCLIP: Region-based Language-Image Pretraining

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
