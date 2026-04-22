# Inlier-Centric Post-Training Quantization for Object Detection Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Object detection is pivotal in computer vision, yet its immense computational demands make deployment slow and power-hungry, motivating quantization. However, task-irrelevant morphologies such as background clutter and sensor noise induce redundant activations (or anomalies). These anomalies expand activation ranges and skew activation distributions toward task-irrelevant responses, complicating bit allocation and weakening the preservation of informative features. Without a clear criterion to distinguish anomalies, suppressing them can inadvertently discard useful information. To address this, we present InlierQ, an inlier-centric post-training quantization approach that separates anomalies from informative inliers. InlierQ computes gradient-aware volume saliency scores, classifies each volume as an inlier or anomaly, and fits a posterior distribution over these scores using the Expectation-Maximization (EM) algorithm. This design suppresses anomalies while preserving informative features. InlierQ is label-free, drop-in, and requires only 64 calibration samples. Experiments on the COCO and nuScenes benchmarks show consistent reductions in quantization error for camera-based (2D and 3D) and LiDAR-based (3D) object detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes InlierQ, a post-training quantization (PTQ) method tailored for object detection models. The approach introduces a saliency-driven inlier/anomaly decomposition to prioritize quantization precision for task-relevant activations while suppressing noisy or outlier features. InlierQ uses a gradient-based volume saliency score, fits an EM-based posterior over the score, and uses this to define inlier sets per layer. Empirical results on COCO and nuScenes benchmarks for 2D and 3D detection show that InlierQ offers modest but consistent reductions in quantization errors and improved accuracy over BRECQ and LiDAR-PTQ, especially at low bit-widths.

### Strengths
1. The central insight—distinguishing inlier from anomaly activations using a saliency score—addresses a notable gap in previous PTQ approaches that treat all activations equally. The probabilistic EM-based classification is straightforward yet effectively leverages gradient information.
2. Experiments span both 2D (COCO) and 3D (nuScenes) detection tasks with multiple modalities (camera and LiDAR), covering several state-of-the-art detection architectures. Results in Table 1 show consistent improvements of up to 2 mAP over BRECQ on challenging low-bit settings.
3. The derivation connecting the Hessian of the custom loss function to the Fisher Information Matrix provides a solid theoretical grounding.

### Weaknesses
1. A key ablation study is missing. I would like to see the performance of the quantization loss directly weighted by the saliency map.
2. In equation 7, the H is the k-th largest heatmap value. I cannot get the meaning of the $k$. The whole proof has no $k$ in it, then how is equation 7 equivalent to the FIM? The effect of $k$ should also be evaluated by ablation study.

### Questions
See weakness 2.

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
2

### Summary
InlierQ introduces an inlier centric PTQ framework that computes gradient aware volume saliency, classifies activations into inliers and outliers with EM, and concentrates quantization on the inlier subspace. It is label free, needs only 64 calibration samples, and consistently cuts quantization error while preserving detection accuracy on COCO and nuScenes for both camera and LiDAR detectors.

### Strengths
1. Conceptually novel with clear theoretical grounding, using an inlier-centric optimization that allocates bit precision to task-relevant activations.
2. Strong engineering practicality, since it is label-free and training-free, and as a plug-in PTQ module it needs only 64 calibration samples.
3. Broad applicability across modalities and architectures, covering camera-based 2D detection, camera-based 3D detection, and LiDAR-based 3D detection.

### Weaknesses
1. Gains are limited at higher bits. Under W8A8 the performance is close to full precision or baseline PTQ, so the advantage is less pronounced.
2. Sensitivity to hyperparameters and unresolved robustness questions. The threshold τ controls the inlier ratio and the final accuracy, which may require retuning across datasets and detection heads.
3. Modest improvement on 2D detection tasks. Ablations indicate that Inlier and Anomaly Sets are less separable in 2D, which reduces the benefit.

### Questions
Please address my concerns in Weaknesses.

### Soundness
3

### Presentation
2

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
The paper proposes InlierQ, a post-training quantization approach that separates activations into inliers and anomalies using a “volume saliency score.” The authors claim that existing quantization methods treat all activations uniformly, thus failing to account for this distinction.  The idea of this paper is to allocate quantization bit capacity to informative activations (inliers) while suppressing noisy or anomalous ones. Experiments on COCO (2D detection) and nuScenes (3D detection) show consistent but moderate performance improvements.

### Strengths
- The method is clearly described with equations and an algorithmic flow (Algorithm 1), making the paper easy to follow and reproduce.
- The results cover both 2D and 3D object detection models (COCO, nuScenes), showing robustness under different modalities and architectures.
- The authors identify that quantization error can be dominated by high-magnitude anomalies or uninformative background activations, which is indeed an important real-world problem for low-bit quantization in detection models. In practice, InlierQ can be applied as a drop-in PTQ refinement module without retraining.

### Weaknesses
- The claim that existing quantization approaches treat all activations uniformly is inaccurate. A substantial body of prior work has explicitly or implicitly modelled activation importance. Although the authors mention outlier-suppression methods such as SmoothQuant, QDrop, and SVDQuant in Related Work, the distinction they claim is that these works only relax amplitudes while their method decomposes activations into inliers and anomalies.

However, many existing PTQ methods already model activation importance and distribution heterogeneity through gradient-weighted or low-rank mechanisms (e.g., BRECQ, Adaround, SVDQuant). Thus, I think the proposed “inlier decomposition” represents a rephrase of known ideas rather than a novel quantization paradigm.

- The EM-based decomposition into “inliers” and “anomalies” in the Method Section is heuristic and lacks theoretical justification. Thus, it does not introduce a new optimization or probabilistic framework beyond existing adaptive scaling methods.

- The improvements (mAP on COCO / nuScenes) are modest and within the range of normal variance. Thus, it is unclear whether the improvement arises from the “inlier” modelling or simply from additional regularization during calibration. No ablation on this factor is performed.

- While some baselines (SmoothQuant, QDrop) are mentioned in related work, they are not included in experiments. The claim of superiority over “outlier suppression” is not directly supported. 

- It is unclear how InlierQ performs under extremely low bitwidth (e.g., 2–3 bits) when compared to adaptive rounding or Hessian-based PTQ.

### Questions
- Is the inlier detection performed per-layer, per-channel, or per-sample?

### Soundness
2

### Presentation
3

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
1. Motivation: Quantization is a crucial technique for model compression. Existing quantization methods treat all activations uniformly, but in object detection tasks, background activations (task-irrelevant activations) are both numerous and can exhibit anomalously high values. This uniform treatment leads to substantial waste of bit precision, especially under low-bit quantization settings.

2. Method: The authors propose Inlier-Centric Quantization (InlierQ). First, they compute volume saliency scores based on gradients and then use a Gaussian Mixture Model (GMM)-based posterior probability to partition the feature space into inlier and outlier regions. Quantization optimization is applied exclusively to the inlier region. To further emphasize task-relevant features, the authors design a top-K heatmap-based loss, which concentrates the Hessian computation on the most discriminative channels.

3. Experiments: Experimental results demonstrate that InlierQ achieves superior performance in both 2D and 3D object detection quantization tasks, particularly under low-bit settings.

### Strengths
1. The paper is presented fairly clearly.

2. The motivation is well-articulated, and the proposed InlierQ method is reasonably designed.

### Weaknesses
1. In Equation (7), is the supervision applied to the top-$K$ entries for each channel of the heatmap? If so, the summation indices over $K$ and $C$ might be reversed in the equation.

2. Equation (12) is described as “explicitly discards anomalous activations and focuses only on the curvature of inlier distributions.” Does this imply that in Equation (6), $\lambda_I = 1$ and $\lambda_O = 0$? If so, by directly discarding background activations and focusing only on high-gradient regions, could the quantization range be overly compressed? Might this lead to an increase in false positive detections in the quantized model? The authors should ideally provide experimental results on precision to support this claim.

3. In the experimental setup, the authors do not specify the backbones used. Referring to experiments in BRECQ and AQD, it is recommended that the method be evaluated on additional detection algorithms (e.g., RetinaNet) and more backbones (e.g., MobileNetV2) to demonstrate its generality.

### Questions
As shown in Weakness.

### Soundness
3

### Presentation
2

### Contribution
2
