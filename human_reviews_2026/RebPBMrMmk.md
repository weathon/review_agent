# RobustSpring: Benchmarking Robustness to Image Corruptions for Optical Flow, Scene Flow and Stereo

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Standard benchmarks for optical flow, scene flow, and stereo vision algorithms generally focus on model accuracy rather than robustness to image corruptions like noise or rain. Hence, the resilience of models to such real-world perturbations is largely unquantified. To address this, we present RobustSpring, a comprehensive dataset and benchmark for evaluating robustness to image corruptions for optical flow, scene flow, and stereo models. RobustSpring applies 20 different image corruptions, including noise, blur, color changes, quality degradations, and weather distortions, in a time-, stereo-, and depth-consistent manner to the high-resolution Spring dataset, creating a suite of 20,000 corrupted images that reflect challenging conditions. RobustSpring enables comparisons of model robustness via a new corruption robustness metric. Integration with the Spring benchmark enables public two-axis evaluations of both accuracy and robustness. We benchmark a curated selection of initial models, observing that robustness varies widely by corruption type and experimentally show that evaluations on RobustSpring indicate real-world robustness. RobustSpring is a new computer vision benchmark that treats robustness as a first-class citizen to foster models that combine accuracy with resilience.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a dataset and benchmarking methodology to assess the influence of image corruptions on optical flow estimation, disparity estimation, and scene flow estimation. Following upon similar assessments in the area of image classification, this type of analysis is extended here not only to dense prediction problems but also to correspondence estimation, which necessitates that (some types of) image corruptions are applied in a consistent manner across views. The analysis builds upon the recent Spring benchmark for optical flow estimation, and complements the accuracy evaluation of Spring with a robustness evaluation, which relies on estimating the stability of the prediction w.r.t. the perturbed image and does not rely on any ground truth. An evaluation of several standard models from the literature complements the analysis.

### Strengths
* A benchmark that assesses the robustness of motion and depth estimation methods w.r.t. image corruptions does not exist so far. This paper extends similar efforts in image classification to correspondence estimation methods and provides a novel and very useful framework to analyze the robustness of motion and depth estimation methods in detail.
* The benchmark is competently designed and well executed on top of one of the most prominent recent benchmark datasets for motion and depth estimation (Spring). I find it very sensible that robustness is decoupled from accuracy in the evaluation.
* The proposed benchmark has the potential to become a standard tool for analyzing the robustness of flow and depth estimation methods.
* Some useful insights emerge from the analysis, such as which family of methods perform well under which conditions. This can allow to design new architectures in the future.
* The paper is written quite well overall and is easy to follow. Occasionally, there are some odd phrases (see below), but these do not distract majorly from understanding the paper.

### Weaknesses
Major points:
* The paper does not argue very clearly why the specific set of image corruptions are particularly relevant in the context of motion and depth estimation. A clear justification why these are the prominent corruptions in real-world imagery would strengthen the paper.
* The notion of stereo, depth and time-consistent corruptions is not clearly defined in the main paper and only becomes more apparent from the supplemental material.
* I find the notion of stereo, depth, and time-consistent corruptions to be suggesting more than there actually is to it, and also possibly confusing. What the paper seems to want to say with this is that the same noise strength is applied to all views, but that’s not particularly surprising. What’s more: it is also possibly confusing because the reader may, for example, wonder what a stereo-consistent JPEG compression would be. Is it somehow a compression algorithm that considers two frames at once to adapt the compression? Yet, this is not actually the case: all this means is that the compression parameter is the same for both views, which again is sort of a given. The only place where the consistency is non-trivial seems to be the consistent synthesis of rain and snow, which is done in 3D. Details are largely absent though, also from the supplement.
* I find it surprising that SEA-RAFT (ECCV 2024) has not been included in the analysis. AFAIK, SEA-RAFT is the most competitive member of the RAFT family to date.
* I am somewhat surprised about the finding that the most accurate models are also the most robust (l. 370f). The paper argues in many places that accuracy and robustness do not necessarily go hand in hand, which I find plausible. Given this, I would have expected this surprising finding to be discussed more deeply.
* [This might be a contentious point, but one may wonder whether this paper is a good fit for ICLR. This is a pure computer vision paper without any learning component to it and thus would be a better fit for CVPR/ICCV/ECCV. That said, ICLR does include benchmarks in its call for papers and indeed the proposed benchmark allows to assess the robustness of learned motion and depth estimators. I am not taking this point into account in my rating and will leave the assessment of topical fit to the (S)AC.]

Minor points:
* The metrics in l. 300 should be explained, at least briefly.
* It does not become clear from the main paper why some frames (“Hero” frames) are being excluded. This becomes apparent only from an experiment in the supplement.
* Minor language oddities, e.g. l. 115 (models do not report, but papers do), l. 139 (evidence is not a verb), or l. 188 (what does it mean for all corruptions to be on a single frame?).

### Questions
* It would be good if the authors could discuss why they chose these particular robustness directions. Yes, they have mostly been considered for image classification, but is it obvious that these are the most important corruption dimensions to consider?
* Is the snow and fog synthesis really the only place where the consistency goes beyond using the same parameters for all views?
* Can the authors discuss the finding that accurate models are also robust in light of the desire to decouple the two? Would this analysis perhaps change if we not only normalized the difference in prediction w.r.t. the difference between the perturbed inputs but also w.r.t. the average displacement/disparity? 
* I do not understand completely why the authors used estimated depth etc. to render 3D-consistent artifacts like rain. Put differently, why is leakage of ground truth a worry? In a real scene, rain would be consistent with the true 3D geometry of the scene, so why shouldn’t we use the ground truth depth for synthesizing rain? Sure, this would give some information about the true depth, but so would a capture of a scene with real rain.
* How are the target SSIM values of 0.7, respectively 0.2 chosen?
* What are “unoptimized data shifts”? (l. 113)
* What does the paper mean by “robustness to corruptions is undefined”? (l. 253) If this was really the case, how could this paper assess robustness to corruptions?
* Which norm is assumed in Eq. 1?
* Why does it make sense to omit the denominator from Eq. 2? The denominator in Eq. 1 is not based on the SSIM but rather on the norm of the image difference, and fixed a SSIM does not guarantee a certain norm on the image difference.

### Soundness
3

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
This paper proposes RobustSpring, a new benchmark that extends Spring with various augmentations. The authors also propose novel corruption robustness metrics along with efficient subsampling methods for evaluation. Detailed experiments are provided.

### Strengths
RobustSpring introduces a large, diverse corruption benchmark for optical flow, scene flow, and stereo, enabling consistent robustness evaluation across real-world perturbations. It provides new metrics along with efficient subsampling evaluation, promoting robustness as a core research goal.

### Weaknesses
I describe my concerns in "Questions".

### Questions
I have several questions about the metrics and list them below.

- [Object Corruptions] What does "value difference" mean (Line 453)? Is that something similar to the intensity difference? Since the rain/snow effects are rendered in blender, why does the paper uses intensity difference instead of directly obtaining rain/snow pixels from rendered results?
- [Relative Robustness] What does this metric mean (Figure 5b)?
- How do corruptions affect the model performance? From Table 1 I can only tell whether a model is robust to different corruptions, but do these corruptions always make the predictions harder? If a corruption actually makes the prediction easier (e.g. better EPE), should we include this data point into proposed robustness metrics? 

I'm happy to adjust my score if my concerns are addressed.

### Soundness
2

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
3

### Summary
The paper updates a recently published benchmark for optical flow and scene flow (SPRING , Mehl et al 2023) so that it also includes image corruptions (e,g. blurring the image, simulating bad weather etc.). It also provides a metric for evaluating algorithms and provides an initial benchmark of existing algorithms.

### Strengths
Datasets help advance the field of computer vision and it is important to evaluate algorithms in more challenging settings (especially since the Spring dataset is based on computer graphics and there is always the danger that algorithms will overfit to the particulars of the renderer).

### Weaknesses
I am afraid that the contribution is too limited in my mind for a conference publication. The results beyond the introduction of the dataset are very limited and I did not see any insights that have been obtained so far from using this dataset. The contribution would be better appreciated in a "datasets and benchmarks" track in a major conference but as far as I know, ICLR does not have such a track.

### Questions
Can you formulate any insights you have learned from the current benchmarking?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces RobustSpring, a large-scale benchmark designed to systematically evaluate the robustness of optical flow, scene flow, and stereo models against 20 types of image corruptions, such as blur, noise, compression artifacts, and weather effects. The benchmark extends the Spring dataset by applying corruptions that are consistent in time, stereo, and depth. A new corruption robustness metric based on Lipschitz continuity is proposed to quantify model stability without relying on ground truth. Experiments across 16 models demonstrate large robustness variations across corruption types, and analyses show that robustness partially correlates with accuracy and transfers to real-world data.

### Strengths
(1) Comprehensive Benchmark Design. The paper introduces the first unified benchmark for evaluating robustness across optical flow, scene flow, and stereo tasks, with a carefully designed corruption taxonomy and consistency mechanisms.

(2) Principled Metric and Evaluation Protocol. The use of a ground-truth-free Lipschitz-based robustness metric provides a theoretically sound and practical approach to isolate robustness from accuracy, avoiding common confounds in prior evaluations.

(3) Extensive and Systematic Experimental Validation. The authors benchmark a wide range of models, analyze architecture-specific trends, and validate the transferability of robustness metrics to real-world datasets (e.g., KITTI), reinforcing the benchmark’s practical relevance.

### Weaknesses
(1) Limited Analysis Beyond Benchmarking. While the benchmark is well-motivated, the paper mainly presents benchmark results without deeper analysis of why certain architectures or design choices improve robustness (e.g., effect of global vs. hierarchical feature aggregation).

(2) Single Severity Level per Corruption. Only one corruption severity level is used to reduce computational cost, which limits fine-grained analysis of model sensitivity. A scalability study across multiple severities could provide more insight into robustness trends.

(3) Lack of Ablation or Analysis Experiments. The work could benefit from additional analysis experiments, such as varying the SSIM threshold used for corruption tuning, testing the effect of different subsampling ratios, or studying the sensitivity of the Lipschitz-based robustness metric’s formulation.

(4) Limited Discussion of Generalization and Scope. The paper acknowledges only briefly that the benchmark does not cover the full corruption space. A clearer discussion of future extensions (e.g., dynamic lighting, motion artifacts beyond the current set) or domain-specific customization would strengthen its impact.

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
3
