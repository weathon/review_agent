# Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Diffusion models have recently emerged as powerful tools for camera simulation, enabling both geometric transformations and realistic optical effects. Among these, image-based bokeh rendering has shown promising results, but diffusion for video bokeh remains unexplored. Existing image-based methods are plagued by temporal flickering and inconsistent blur transitions, while current video editing methods lack explicit control over the focus plane and bokeh intensity. These issues limit their applicability for controllable video bokeh. In this work, we propose a one-step diffusion framework for generating temporally coherent, depth-aware video bokeh rendering. The framework employs a multi-plane image (MPI) representation adapted to the focal plane to condition the video diffusion model, thereby enabling it to exploit strong 3D priors from pretrained backbones. To further enhance temporal stability, depth robustness, and detail preservation, we introduce a progressive training strategy. Experiments on synthetic and real-world benchmarks demonstrate superior temporal coherence, spatial accuracy, and controllability, outperforming prior baselines. This work represents the first dedicated diffusion framework for video bokeh generation, establishing a new baseline for temporally coherent and controllable depth-of-field effects. Project page is available at this website https://vivocameraresearch.github.io/any2bokeh/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors present a video bokeh rendering method based on diffusion model. They propose a one-step diffusion framework, and employ a multi-plane image representation to the focal plane as a condition. Also, they introduce a progressive training strategy for stability. Experiments on synthetic and real-world benchmarks show the performance of their methods.

### Strengths
- The authors propose a one-step diffusion framework for video bokeh rendering, which exhibits an efficiency advantage in inference time.
- In the third stage, fine tune the VAE decoder and introduce texture loss based on image gradients to improve high-frequency texture and edge clarity, which helps ensure the presentation of details of the focused subject.
- Time consistency and video quality indicators are significantly better than the baseline mentioned in the paper.

### Weaknesses
- Comparison with Video Bokeh Methods. The authors only compared their with the image bokeh method, but it needs to be compared with video methods, such as VBR [1].
- Complex scenes. The authors employ a multi-plane image (MPI) representation, and this representation can bring challenges, such as whether to divide an object into two different layers. The authors should discuss this situation.
- The robustness of this method. The authors should compare their full model with degraded depth maps and without degraded depth maps.
- Missing details. TS (token selection) is not defined in formula (4); How M̄=[1,M] aligns a “global token” with 2D masks should be provided; Specify mask resolutions for each block and the exact interpolation strategy; clarify how “near-focus vs. wide-interval” masks are selected per layer.
- Dataset and generalization. Both synthetic training and testing are based on the construction of "planar disparity" (where d is an affine function of x and y). This simplifies geometry but differs significantly from real-world scenarios.

[1] Luo, Yawen, et al. "Video bokeh rendering: Make casual videography cinematic." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

### Questions
- Weighted overlap inference strategy. I wonder if doing weighted overlap only during inference without the same operation during training will affect inference performance.
- The number of image plane N. I want to know whether to use the same N or different N for different scenarios. If different, how is the value of N determined.
- Additional metrics. Add LPIPS for frame-level perceptual quality;  
- Dynamic effects. I suggest the author provide video effects

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a method for adding controllable Bokeh in videos. They fine tune a video diffusion model to accept a video and a corresponding explicit scene geometry conditioning and output a depth-aware Bokeh video in a single diffusion step. The authors propose a multi-stage training strategy that facilitates robustness to noisy input geometry and high temporal bokeh consistency. The authors provide extensive evaluations showing state-of-the-art results for controllable video bokeh, with control of the focus plane and bokeh intensity.

### Strengths
1.	The authors leverage the prior of a video diffusion model in a novel way, to perform temporally consistent, controllable video bokeh. 
2.	The authors showcase good bokeh results. They also provide a supplementary video with their results and comparisons to other methods, which is very important for the qualitative assessment of their claims for temporally consistent bokeh addition.
3.	The authors show extensive quantitative evaluations, emphasizing their lead over other competing methods.
4.	The authors provide several ablations for their training strategy choices.

### Weaknesses
Major:
1.	The authors do not provide limitations for their method. Are there any scenarios where the model fails to generate a good bokeh video? Maybe in videos with fast motion, such as a car race.
Minor:
1.	Figure 1 is not referenced.
2.	SM figures 10,11 – red border not corresponding to zoom-in area.

### Questions
Please see weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel one-step diffusion framework for generating temporally coherent and depth-aware video bokeh effects. The proposed approach enables video refocusing with explicit control over the focal plane and bokeh intensity, addressing the limitations of prior image-based and video-based bokeh rendering methods. The framework leverages a focal-plane-adapted multi-plane image (MPI) representation to guide the diffusion process, ensuring temporal smoothness and accurate depth-dependent blur transitions.

### Strengths
1. **Innovative Approach:**
   The use of an MPI-guided conditioning mechanism in a one-step diffusion framework for video bokeh generation is both original and well-motivated. It effectively bridges the gap between static image refocusing and temporally coherent video refocusing.

2. **Temporal Coherence and Depth Awareness:**
   The focal-plane-adapted MPI representation efficiently balances detail preservation in focused regions and smooth transitions in defocused areas, improving visual consistency across frames.

3. **Comprehensive Experiments:**
   The paper includes thorough quantitative and qualitative comparisons with existing methods, demonstrating consistent improvements in temporal stability, spatial accuracy, and controllability.

### Weaknesses
1. **Dependence on Depth Estimation:**
   The method relies on pre-trained depth estimation models as input. In dynamic or complex scenes, depth errors may propagate into the final bokeh rendering. The paper would be stronger with a sensitivity analysis or ablation showing how depth inaccuracies affect output quality.

2. **Computational Efficiency:**
   While the results are impressive, the paper provides limited discussion on computational cost. Diffusion-based models are typically resource-intensive; more details on runtime, memory consumption, and scalability (e.g., potential for real-time use) would enhance the practical relevance.

3. **Dataset Bias and Generalization:**
   The primary evaluation uses synthetic datasets. Although this allows for controlled comparisons, such datasets may not fully reflect real-world complexities such as fast motion, varying lighting, or occlusions. Additional experiments on diverse real-world datasets would strengthen claims of robustness and generalizability.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
