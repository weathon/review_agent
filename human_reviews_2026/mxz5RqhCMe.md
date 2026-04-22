# Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
As one of the fundamental intermediate modules in autonomous driving, online high-definition (HD) maps have attracted significant attention  due to their cost-effectiveness and real-time capabilities. Since vehicles always cruise in highly dynamic environments, spatial displacement of onboard sensors inevitably causes shifts in real-time HD mapping results, and such instability poses fundamental challenges for downstream tasks. However, existing online map construction models tend to prioritize improving each frame's mapping accuracy, while the mapping stability has not yet been systematically studied. To fill this gap, this paper presents the first comprehensive benchmark for evaluating the temporal stability of online HD mapping models. We propose a multi-dimensional stability evaluation framework with novel metrics for Presence, Localization, and Shape Stability, integrated into a unified mean Average Stability (mAS) score. Extensive experiments on 42 models and variants show that accuracy (mAP) and stability (mAS) represent largely independent performance dimensions. We further analyze the impact of key model design choices on both criteria, identifying architectural and training factors that contribute to high accuracy, high stability, or both. To encourage broader focus on stability, we will release a public benchmark. Our work highlights the importance of treating temporal stability as a core evaluation criterion alongside accuracy, advancing the development of more reliable autonomous driving systems. The benchmark toolkit, code, and models will be available at https://stablehdmap.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces mAS, a comprehensive metric suite for evaluating the temporal stability of lane elements in online HD mapping. It measures Presence, Localization, and Shape Stability across frames and aggregates them into a unified score. Evaluation across various baseline methods is also proposed.

### Strengths
- The proposed mAS is a well-designed, multi-dimensional metric (Presence, Localization, Shape) that directly targets temporal stability in online HD mapping, addressing a clear gap in prior evaluations.

- The paper presents a thorough, large-scale comparison across diverse baselines, making temporal consistency differences easy to assess and helping the community select appropriate baselines.

- The manuscript is clear and well-structured; metric definitions and the evaluation protocol are easy to follow.

### Weaknesses
- `Reliance on accurate inter-frame alignment:`
The mAS metric depends on precise ego-motion/pose transformations to align frames. Errors from localization, calibration, or time sync can propagate into the stability score, introducing variance unrelated to the model’s intrinsic stability.

- `Vulnerable to “copy–paste” or over-smoothing strategies:`
A model could boost mAS by copying or heavily smoothing predictions across frames, inflating stability without being correct or responsive to scene changes.

- `Coupling with accuracy and interpretability of rankings:`
Because stability can rise even as accuracy drops, comparing models solely on mAS may be misleading. Low-mAP models could appear “stable” but be persistently wrong. This reduces the value of the submission.

- `Sensitivity to implementation choices:`
Polyline resampling density, curve parameterization, and discretization can affect shape and localization stability. Results may shift with frame rate and temporal window length.

### Questions
N/A

### Soundness
2

### Presentation
2

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
This paper addresses temporal stability for HD Mapping, which is a critical and long-overlooked problem. The authors argue that current mainstream evaluation metrics, such as mAP, focus exclusively on single-frame geometric accuracy. To address this gap, the paper introduces the benchmark specifically designed to evaluate the temporal stability of online HD maps. The core contribution is a new, multi-dimensional metric named mAS (mean Average Stability).

### Strengths
1. The task of this paper is meaningful, which systematically address and quantify the critical evaluation blind spot of "temporal stability" in the online HD mapping domain.
2. The proposed mAS metric is well-designed and comprehensive.
3. The paper is written clearly and with a strong logical flow.

### Weaknesses
1. The effectiveness of mAS relies on the GT stability. If the GT annotations themselves are inconsistent between frames (e.g., "jitter" or "flicker" from human labelers, which is common in large datasets), the mAS metric would unfairly penalize a model that produces a stable (and potentially more correct) output.
2. Before stability calculation (Sec 3.3), the framework must perform "geometric alignment" using the vehicle's pose data. If the vehicle's localization data is noisy, it will introduce artificial "instability" during the alignment process, again leading to an incorrect penalty on the perception model's stability.
3. The paper mentions (Sec 3.3) using uniform resampling along the x-axis to compare polylines. This method may becomes highly unstable or fails for vertical line segments (i.e., those running parallel to the vehicle's direction of travel). The authors do not clarify how this common and critical case (e.g., lane lines) is handled, which could lead to biases in the evaluation.
4. The current mAS focuses on geometry and presence. However, another significant failure mode is "semantic flickering" where an element's position and shape are stable, but its classification label jumps between categories (e.g., "lane" and "road boundary"). This is equally detrimental to downstream tasks, but the mAS framework does not appear to capture this.

### Questions
1. Could the authors elaborate on the sensitivity of the mAS metric to GT quality and ego-motion noise? For example, if varying levels of noise are injected into the GT or ego-motion data, how much do the model rankings change? This seems key to building trust in this new benchmark.
2. How are near-vertical polylines (which would run parallel to the x-axis in the BEV-like coordinate system mentioned) handled during the resampling process? Does this strategy risk ignoring the stability evaluation for these critical elements, such as lane lines?
3. Does the current mAS framework (particularly the instance matching stage) account for the stability of classification labels? If an instance is stable in geometry and location but its class label flickers between frames, is this captured by the metric?
4. An interesting finding is that adding temporal fusion (Temp=4) to a non-temporal model like MapTR-GKT significantly degrades stability (mAS from 71.6 to 66.6). The authors speculate this is due to "noise". Could the authors provide a deeper analysis or hypothesis at the feature level? For instance, is it possible that the temporal module struggles to align the spatial features produced by GKT, leading to feature-level aliasing or confusion?

I would be happy to raise the score if the authors provide more analysis.

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
4

### Summary
This paper presents the first benchmark for temporal stability in online HD mapping, introducing the mAS metric with Presence, Localization, and Shape components, and shows through 42 model variants that accuracy (mAP) and stability (mAS) are largely independent dimensions.

### Strengths
1. The problem is well-motivated - existing metrics focus on single-frame accuracy while ignoring temporal consistency that's critical for safe autonomous driving.

2. Testing 42 model variants across different backbones, BEV encoders, and temporal fusion strategies provides solid empirical evidence for the mAP-mAS independence claim.

3. Breaking stability into Presence, Localization, and Shape gives more actionable insights than a single number would.

### Weaknesses
1. Using ground truth as a bridge for cross-frame matching (Algorithm 2) assumes perfect GT consistency, but annotation noise in consecutive frames could bias your measurements - have you considered direct matching with learned features like in SORT [1] or MOT approaches [2]?

2. Your uniform resampling along x-axis only works for roughly vertical polylines; this breaks for curved roads or horizontal boundaries, whereas arc-length parameterization [3] would handle arbitrary shapes more robustly.

3. Computing curvature as average angles between segments (Algorithm 7) is quite sensitive to sampling density - why not use more robust shape metrics like Fréchet distance [4] or proper curvature signatures [5]?

4. The choices of β=15.0 and ω=0.7 seem arbitrary without ablation studies - how sensitive is mAS to these values, and do they generalize to different map ranges or vehicle types?

5. Only using nuScenes limits your conclusions, especially since it has relatively benign weather - what happens on corruption benchmarks like nuScenes-C [6] or RoboDrive [7] where models might fail differently?

6. While your qualitative examples are compelling, there's no quantitative link between mAS and actual planning metrics like collision rate or trajectory smoothness - does higher mAS actually lead to safer planning?

7. Testing M∈{2,3,5,10} is a start, but you're randomly sampling rather than systematically studying how stability degrades with time or identifying worst-case temporal transitions like occlusion events.

8. You don't compare against simpler alternatives like frame-to-frame Chamfer distance [8] or temporal consistency losses - how do we know mAS is actually better than these baselines?

## References

[1] Bewley et al. "Simple online and realtime tracking." ICIP 2016.

[2] Voigtlaender et al. "MOTS: Multi-object tracking and segmentation." CVPR 2019.

[3] Pottmann et al. "Geometry of the squared distance function to curves and surfaces." Visualization and Mathematics III, 2003.

[4] Alt & Godau. "Computing the Fréchet distance between two polygonal curves." IJCGA, 1995.

[5] Mokhtarian & Mackworth. "Curvature-based shape representation for planar curves." IEEE TPAMI, 1992.

[6] Xie et al. "RoboBEV: Towards robust bird's eye view perception under corruptions." arXiv:2304.06719, 2023.

[7] Kong et al. "Robo3D: Towards robust and reliable 3D perception." ICCV 2023.

[8] Achlioptas et al. "Learning representations and generative models for 3d point clouds." ICML 2018.

### Questions
1. Can you provide ablation studies on β, ω, and N to show that mAS rankings are stable across reasonable parameter choices?

2. What happens to your GT-based matching when annotations themselves have temporal jitter - have you measured annotation consistency in nuScenes?

3. Could you add experiments showing mAS correlation with downstream planning metrics to validate that it actually predicts safe behavior?

4. How would your results change on challenging datasets with weather corruptions or sensor degradation?

### Soundness
2

### Presentation
3

### Contribution
3
