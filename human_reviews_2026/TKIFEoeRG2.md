# SNAP: Testing the Effects of Capture Conditions on Fundamental Vision Tasks

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Generalization of deep-learning-based (DL) computer vision algorithms to various image perturbations is hard to establish and remains an active area of research. The majority of past analyses focused on the images already captured, whereas effects of the image formation pipeline and environment are less studied. In this paper, we address this issue by analyzing the impact of capture conditions, such as camera parameters and lighting, on DL model performance on 3 vision tasks---image classification, object detection, and visual question answering (VQA). To this end, we assess capture bias in common vision datasets and create a new dataset, $\textbf{SNAP}$  (for $\textbf{S}$hutter speed, ISO se$\textbf{N}$sitivity, and a$\textbf{P}$erture), consisting of images of objects taken under controlled lighting conditions and with densely sampled camera settings. We then evaluate a large number of DL vision models and show the effects of capture conditions on each selected vision task. Lastly, we conduct an experiment to establish a human baseline for the VQA task. Our results show that computer vision datasets are significantly biased, the models trained on this data do not reach human accuracy even on the well-exposed images, and are susceptible to both major exposure changes and minute variations of camera settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces SNAP, a benchmark of photographs of real physical objects captured under a large grid of camera and lighting settings. SNAP evaluates three tasks—image classification, object detection, and VQA—and reports results for 52 models. The central finding is that camera parameter choices can significantly affect accuracy, highlighting practical risks when deploying vision systems outside controlled conditions.

### Strengths
1. **Dataset scope and design.** SNAP systematically varies sensor parameters, spanning ~700 camera‑parameter combinations, two lighting conditions, and ten object categories. It supports three representative vision tasks: image classification, object detection, and VQA.

2. **Breadth of evaluation.** The study compares 52 models across three tasks, providing a broad view of model robustness under sensor shift.

### Weaknesses
Despite amazing efforts for constructing the dataset and performing exhaustive experiments, my main concerns are unclear novelty compared to previous work and lack of interesting (or new) findings out of the new dataset. 

1. **Unclear novelty relative to existing benchmarks.**
- The paper’s incremental contribution over ImageNet‑ES [1], ImageNet‑ES‑Diverse [2], and the SenseShift6D [3] dataset is not sufficiently articulated. These recent efforts also use physical cameras to vary lighting and sensor parameters. To strengthen the case for SNAP, the manuscript should present a clear, head‑to‑head comparison—qualitative and quantitative—throughout the introduction, related work, methodology (including Fig. 1), and experiments.

- **Qualitative difference.** A likely distinguishing aspect is that, whereas ImageNet‑ES uses displays and ImageNet‑ES‑Diverse uses printed images, SNAP captures real objects. The paper should explicitly motivate why photographing real objects offers additional scientific value beyond screens/prints and verify this experimentally. Please analyze the limitations of ImageNet‑ES and ImageNet‑ES‑Diverse and show why real‑object capture is necessary (e.g., material/geometry effects, specularities, shadows, interreflections) and how those effects manifest in performance gaps.

- **Quantitative difference.** SNAP’s denser grid (~700 vs. 64 settings in ImageNet‑ES) is a key claim. However, it is not yet clear what new insights this added density uniquely enables. The paper should demonstrate concrete findings that only emerge from a dense sampling (e.g., non‑monotonic interactions between ISO and shutter, lighting‑dependent parameter regimes, model‑specific sensitivity profiles).


2. **Limited novelty in findings.** 
- The principal conclusions—(i) substantial performance degradation under sensor shift and (ii) larger models being less affected—largely echo results reported in ImageNet‑ES and ImageNet‑ES‑Diverse. The paper would benefit from emphasizing new insights (e.g., task‑specific vulnerabilities, parameter interactions, or cross‑task correlations) that go beyond prior work.

3. **Lack of solution analysis.**
- The study documents degradation but does not evaluate or discuss remedies. Do authors claim that the models should be improved for poor capture conditions where humans cannot perform well? or camera should be controlled to avoid poor capture conditions? In particular, adaptive sensing (e.g., Lens, ICLR 2025) has been shown to mitigate sensor shift in classification. An analysis on SNAP would be valuable: (i) how camera control affects performance under SNAP, (ii) whether benefits transfer from classification to object detection and VQA, and (iii) limits of such approaches in multi‑task settings.

[1] E. Baek et al., "Unexplored faces of robustness and out-of-distributions: Covariate shifts in environment and sensor domains," CVPR 2024.

[2] E. Baek et al., "Adaptive camera sensor for vision models," ICLR 2025.

[3] Y. Han et al., "SenseShift6D: Multimodal RGB-D benchmarking for robust 6D pose estimation across environment and sesor variations," arxiv, 2025.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents SNAP, a dataset designed to study how camera settings and lighting affect the performance of deep learning models on image classification, object detection, and visual question answering (VQA). The authors first analyze capture bias in 13 major datasets, finding that most images are taken under similar exposure conditions. They then build SNAP with 37k real images captured under controlled lighting and camera parameters, and evaluate 52 models along with human subjects. Results show that models are highly sensitive to even minor changes in exposure, perform worse than humans on poorly lit images, and that capture bias in training data propagates across downstream tasks.

### Strengths
- Thorough empirical scope: 52 models across classification, detection, and VQA.
- Novel dataset (SNAP) enabling controlled study of real capture bias.
- Human comparison adds interpretability to machine performance.
- Comprehensive dataset analysis across 1.3B images provides strong motivation.
- Clear presentation and reproducible methodology.

### Weaknesses
- No dense prediction tasks (e.g., segmentation, depth estimation), which would show whether pixel-level sensitivity differs from categorical tasks.
- Lack of quantitative comparison to synthetic corruptions (e.g., Gaussian noise, brightness jitter, or CSF filters) to contextualize real vs. synthetic robustness.
- The takeaways, while valid, are somewhat predictable (“models fail on extreme exposure, larger models do better”) and lack actionable insight for improving model robustness.
- The dataset, while well-controlled, is limited to 10 object categories and might not generalize beyond tabletop scenes.
- No analysis of whether findings could have been replicated using synthetic exposure shifts on ImageNet images, which would clarify the added value of real capture data.

### Questions
### **Questions for Authors**

1. Why were no dense tasks (e.g., segmentation/depth) included? Would these show similar exposure sensitivity?
2. Could the same conclusions be drawn using synthetic exposure perturbations applied to ImageNet or COCO (e.g., gamma correction, brightness scaling)?
3. Are there any quantitative correlations between model pretraining datasets’ exposure distributions (e.g., LAION EV histogram) and performance on SNAP?
4. Does the dataset allow evaluating data augmentation strategies that explicitly normalize exposure?


The paper is well executed and clearly written, offering valuable insights into how capture conditions affect vision models. The SNAP dataset and large-scale evaluation are solid contributions. However, the scope feels narrow—it focuses primarily on exposure, lacks dense prediction tasks, and omits comparison with synthetic corruptions, which limits its broader impact. Overall, it is a careful and meaningful study, but not particularly novel or exciting. Including a dense task and a more direct discussion of synthetic corruptions would be necessary to raise my score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors study the effects of capture conditions (i.e., camera parameters and lighting) on three common vision tasks (i.e., image classification, object detection, and VQA). To achieve this, the authors first construct a new dataset under controlled lighting conditions and with densely sampled camera settings. The authors then conduct an empirical study by evaluating a large number of vision models on the created dataset and reveal the effects of capture conditions on each selected vision task.

### Strengths
-	The idea of studying the effects of image capture conditions on the generalizability of models is interesting.
-	The authors reveal some interesting observations based on their created dataset. The evaluation is comprehensive, covering a large number of models.

### Weaknesses
-	My major concern is that the created dataset is less practical in real-world scenarios. 1) The dataset only contains 10 categories, most of which are office supplies and small-scale objects. However, there are many other categories in ImageNet and COCO. Using this small set of categories tends to make the evaluation somewhat biased. 2) The scenes are not cluttered, objects appear large against a plain background, and there is little occlusion. Such a toy case makes me concerned about the actual usefulness of the dataset.
-	I am curious what if the models are fine-tuned on the dataset. Will the models still suffer from under-exposed and over-exposed conditions? To do this, the authors can split the dataset into a training set and a test set. Then the authors can finetune the models on the training set and evaluate them on the test set.

### Questions
I am concerned about the questions mentioned above. Given the current status of the paper, I am leaning towards rejection and hope the authors could address my concerns during the rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2
