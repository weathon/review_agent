# MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 6

## Abstract
Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. 
Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries.
To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions:
1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 
2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions.
3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 
4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically.
As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
We conduct extensive experiments to demonstrate that our proposed method is capable of improving novel view synthesis of the Gaussian-based explicit representation methods about 1 dB PSNR for various tasks. \href{https://mvgs666.github.io/}{\textcolor{magenta}{Codes are available.}}

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This work propose to use more views in each training iteration of 3DGS. The experiments suggest that this simple startegy consistently leads to better quality. In addition, a new Gaussians densification method is proposed which densifies more on region with larger multi-view losses. A multi-resolution training stategy is also proposed to use more views with lower resolution at the beginning of training. These additional training strategies can further improve the quality.

### Strengths
The core message of this paper is clear: using more training views in each iteration can improve quality. The effectiveness of this is evaluated by extensive experiments on various 3DGS variant and various tasks.

### Weaknesses
- The paper writing need many improvement. There are many technical details sections are hard to understand. See Questions section for details.
- The training time and memory consumption may increase significantly by using more views in each iteration. The tradeoff on this aspect is not discussed. The results tables should disclose the additional training cost comparing to the other.
- When comparing to the other method with more training iterations, is all the scheduler hyperparameters scaled accordingly? For instances, `position_lr_max_steps` for lr annealing, `densification_interval, opacity_reset_interval, densify_until_iter` for adaptive Gaussians. When training with the baseline of 8x longer schedule, are the learning rate downscale by 1/8 as well?

### Questions
Sec.3.1:
- How the multi-view are sampled in each iteration? Is it just uniform sample from the training set?

Sec.3.2:
- L249: is confusion.
    - Which of the $k$-th layer is set to 8 when saying "the set $s_k$ as 8"? I can guess it means {1, 2, 4, 8}. Then the correct writing should be S = {s^{k-1} | k=1...4} and s=2.
    - Seems that $c_k', f_k'$ are the scaled principle point and focal length and $c_k, f_k$ are the source one. Then we do not need the layer index for the source principle point and focal length as there are all the same.
- In Figure 2, it seems that different resolution of the images are employed in each training iteration. However, from the main text, it turn out to be that a single resolution is used in each iteration.
- What is the implementation details for the schedule of multi-resolution training? How many iterations are trained in each of the image scale? How is the multi-resolution training and the original training schedule couple together? Do we start to do densification, pruning, and sh degree increment in the coarser resolution training?

Sec.3.3:
- What is the sliding window size in Sec.3.3 for finding the patches with high loss?
- Four rays are casted from the high-loss patches. How the rays from different images can intersect? As rays are infinite thin line in 3D, two rays from different cameras may can hardly intersect. From figure 2, I guess the image patch frustum is casted instead of just rays. Then the following question is why the intersection of the frustrum form cuboid? Isn't it the intersection form 3D polygon?
- The Gaussians in the high-loss region is densified "to a certain amount". How are these Gaussians actually density? Each duplicated by two?

Sec.3.4:
I have read several times but it still hard for me to understand this section.
In original 3DGS, the viewspace gradient is accumulated for hundreds of iterations and at some interval, the Gaussians with gradient above the threshold $\beta$ are densified.
Is it that the only difference in this work is that the threshold is adaptive choiced between $\beta$ and $0.5\beta$ based on camera positions?
$\hat{\beta}$ seems to be a global threshold as in the original 3DGS but from Eq.4, $\hat{\beta}$ is depend on the distance of a pair of cameras. For each Gaussian, do we accumulate the viewspace gradient for each pairs of the cameras instead of just a global viewspace gradient now?


In L513: It is still unclear to me why the performance degrade when there are too much training views in each iteration. How much is too much? Why using too much views is analogous to using a region of views?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces an approach for novel view synthesis using 3D Gaussian Splatting (3DGS) with multi-view regularization, achieving an approximate 1 PSNR improvement over existing methods on various datasets.

### Strengths
Advantages
Novelty: The paper proposes three key strategies to enhance the rendering quality of 3DGS:
A cross-intrinsic guidance scheme that employs a coarse-to-fine training procedure.
A cross-ray densification strategy that densifies Gaussian kernels in regions where rays intersect, improving details in specific views.
A multi-view augmented densification strategy to further optimize Gaussian density based on view discrepancies.

### Weaknesses
Drawbacks
While the figures are clear, the writing quality could be improved for readability.
Lines 93–96 states: “We first propose a multi-view regulated training strategy that can be easily adapted to existing single-view supervised 3DGS frameworks and their variants, optimized for a large variety of tasks, where NVS and geometric precision can be consistently improved.” However, there are no experiments specifically demonstrating improvements in geometric precision.
The overall pipeline is more complex and larger than standard 3DGS. It is unclear if this increase is due to the larger network or the proposed modules that enhance 3DGS.

### Questions
Among the proposed cross-intrinsic guidance, cross-ray densification, and multi-view augmented densification strategies, which component significantly improves the quality? Have ablation studies been conducted to isolate and measure each component’s impact?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed four main modules to improve the 3DGS model, namely, the multi-view regulated training, cross-intrinsic guidance, cross-ray densification and multi-view augmented densification. This paper tried to impose the constraints from multiple views and multiple resolutions simultaneously, densify more points in the central area and densify more points for distinct views.

### Strengths
1. This paper has conducted experiments with different baselines, different benchmarks and different tasks.
2. This paper investigated the impact of multi-view constraints and proposed many strategies to mitigate the overfitting problem in 3DGS in terms of loss function, training strategy and densification strategy.

### Weaknesses
1. I still don't understand the relationship of these four main contributions well, such as , you were inspired by the multi-view constraints and proposed the cross-intrinsic guidance, so what's the relationship of these two contributions? Do they have to work together to see a performance gain?
2. I am still a little confused about the effectiveness of the multi-view regulated training. Considering from the perspective of gradient backpropagation, this multi-view regulated training is similar to the way that losses from multiple iterations are accumulated into one iteration. Are there any more designs, such as these multiple views?
3. How does the training efficiency compare to existing methods? Perhaps using two samples to show the ablation results is not convincing enough, and it would be better to put Table 5 in the main paper.

### Questions
see weaknesses.

### Soundness
3

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
5

### Summary
The paper introduces a series of improvements to the 3D Gaussian Splatting (3DGS) framework, aiming to address challenges in rendering quality and overfitting by incorporating multi-view regulation techniques. The motivation is reasonable and the rendering quality is impressive.

### Strengths
+: Thanks for providing the code. The results are reproducible and good. 
+: The motivation makes sense since an explicit and discrete primitive like gaussian tends to overfit and get stuck in local minima easily. Multi-view constraints would be beneficial to solve these.
+: The method is thoroughly evaluated in various settings.

### Weaknesses
-: The overhead has increased significantly. Although the total number of iterations remains at 30k, each iteration now involves multiple forward passes, causing the overall training time to multiply. Specifically, training a single scene now takes 2–3 hours, compared to 3DGS's 20–30 minutes. This raises concerns about the method’s practicality, as one of Gaussian Splatting’s key advantages is its efficiency in both training and rendering. Additionally, the output PLY file size is several times larger than other methods, reaching over 1GB for many scenes, which may further hinder its usability.

I suggest the author consider two further improvements:

(1 )Attempt to reduce the final count of Gaussians to demonstrate that the improvements come from better Gaussian placement rather than excessive densification.
(2) Test the effect of using the AVG instead of SUM for multi-view loss aggregation.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
3
