# DYNAMIC NOVEL VIEW SYNTHESIS FROM UNSYNCHRONIZED VIDEOS USING GLOBAL-LOCAL MOTION CONSISTENCY PRIOR

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Dynamic novel view synthesis (D-NVS) critically depends on hardware-based synchronization. Current approaches that accommodate unsynchronized settings within the widely-used NeRF or GS frameworks often struggle with local minima, particularly in textureless scenes or when multi-view videos exhibit large misalignments. To tackle this issue, we propose a novel 3D global–2D local motion consistency prior, which evaluates the alignment between predicted scene flow projections and pre-computed optical flows across multi-view videos. Our analysis reveals that the motion, produced by the anisotropy of projected global scene flow across different views, is inherently more effective for correcting temporal misalignments compared to the near-isotropic appearance typically leveraged in NeRF or GS. Extensive experiments on public datasets demonstrate the versatility of our loss function across various D-NVS architectures (NeRF and GS), achieving a 50% reduction in synchronization errors and a PSNR improvement of up to 4dB, thereby outperforming existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to solve the challenge of dynamic novel view synthesis (D-NVS) from unsynchronized multi-view videos. To address the problem that existing methods easily fall into local optima in textureless scenes or with large temporal misalignments, the authors propose a "global-local motion consistency prior". The core of this method is a loss function ($\mathcal{L}_{Flow}$) that jointly optimizes the scene representation ($F_t$) and the learnable temporal offsets ($\Delta t_i$) by comparing the 2D projection of the predicted 3D scene flow ($\hat{f_i}$) with pre-computed 2D optical flow ($f_i$). The authors integrate this loss function into several mainstream D-NVS frameworks (NeRF and GS), and experiments show the method significantly reduces synchronization errors and improves reconstruction quality.

### Strengths
The paper presents an interesting observation: using the "anisotropy" of projected motion as a supervisory signal for temporal alignment. Compared to appearance-based photometric consistency, this prior theoretically provides a more robust signal for alignment in textureless regions. Furthermore, the method demonstrates strong generality; it is designed as a "plug-and-play" loss function that can be easily integrated into various mainstream D-NVS frameworks (including K-Planes, 4DGS, and EDGS). The experimental results are convincing, showing consistent and significant performance improvements across multiple baselines, datasets, and degrees of temporal misalignment, effectively enhancing both reconstruction quality and synchronization accuracy.

### Weaknesses
**W1. Dependency on Optical Flow and Limitations of the Masking Strategy:** The method's core supervision signal relies on pre-computed 2D optical flow ($f_i$). This introduces a critical external dependency, as optical flow algorithms themselves are unreliable in scenarios like occlusions, fast motion, or transparent/reflective surfaces. The authors attempt to mitigate this with a reliability mask, but this mask is based only on flow "magnitude" (selecting the top 50%), which is a coarse strategy. It ignores other major failure modes of optical flow, such as occlusion boundaries or moving textureless regions (which may have high magnitude but low accuracy), which can still introduce erroneous supervision signals into the optimization.

**W2. Missing Key Implementation Details:** The paper lacks key implementation details required for reproducibility, especially the specific implementation of the 3D flow operator ($\mathcal{F}_{3D}$) for the different frameworks (4DGS, EDGS, K-Planes). The descriptions are vague (e.g., 4DGS "implicitly realized through splatting" or EDGS "tracking... Gaussian's motions"), lacking clear mathematical formulations and differentiable implementation paths, which makes the work difficult to reproduce.

**W3. Limited Authenticity of Evaluation:** The experimental evaluation has limited authenticity. All "unsynchronized" data was created by artificially applying random offsets to already synchronized public datasets. The paper does not test on any *truly* unsynchronized, real-world multi-camera recordings. Therefore, it is unverified whether the method can generalize to handle real-world asynchrony issues such as rolling shutter, clock drift, and exposure differences.

### Questions
**Q1. Circular Dependency in Optimization:** This method requires jointly optimizing the scene representation ($F_t$) and the temporal offsets ($\Delta t_i$). However, in the early stages of training, both $F_t$ (which depends on $\Delta t_i$) and $\Delta t_i$ (whose supervision $\hat{f_i}$ depends on $F_t$) are likely incorrect. How does the optimization process break this dependency and ensure convergence? What prevents the optimization from collapsing into a local minimum where an incorrect $F_t$ and an incorrect $\Delta t_i$ mutually reinforce each other?

**Q2. Mismatch of Pre-computed Optical Flow:** The paper uses pre-computed optical flow $f_i$ (e.g., calculated between frames $t$ and $t+1$) as the supervision signal. However, because the videos themselves are unsynchronized (e.g., by an offset $\Delta t_i$), this $f_i$ actually represents the motion between global scene times $(t+\Delta t_i)$ and $(t+1+\Delta t_i)$. The model's objective, however, seems to be the scene flow corresponding to global time $t$. How do you handle this fundamental mismatch in the supervision signal $f_i$ caused by $\Delta t_i$ itself?

**Q3. Temporal Baseline and Scale Inconsistency:** The loss function $\mathcal{L}_{Flow}$ compares the pre-computed 2D optical flow $f_i$ and the projected 3D scene flow $\hat{f_i}$. However, $f_i$ is typically a discrete displacement field corresponding to a specific time interval (e.g., $\Delta \tau=1$ frame), while $\hat{f_i}$ (derived from $\mathcal{F}_{3D}$) might represent an instantaneous velocity or a displacement over a different time step. Please clarify how $f_i$ and $\hat{f_i}$ are kept strictly consistent in terms of their time scale and units.

**Q4. Supervision Mismatch under Large Misalignments:** When handling large temporal offsets (e.g., 0-37 frames), the pre-computed optical flow $f_i$ is still a short-term motion calculated between adjacent frames (e.g., $t$ and $t+1$). However, the $\Delta t_i$ the model needs to optimize is a long-term offset spanning many frames. How can this short-term motion supervision signal ($f_i$) effectively guide the model to find a correct, large-scale temporal offset ($\Delta t_i$)?

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
A global-local motion consistency prior and derived loss function is proposed to address novel-view synthesis in dynamic scenes with unsynchronised views. The prior is used to temporally align multi-view videos by explicitly estimating a temporal offset between views in an optimization framework for a sharper novel-view synthesis.

### Strengths
The method is intuitive and simple. Elegant presentation of the method, by abstracting from the instantiation of operator F. Significant quantitative and qualitative gains across all baselines and benchmarks.

### Weaknesses
The writing style is not uniform. The abstract is super clear, while in the introduction some sentences do not follow academic standard: like "to achieve synchronization by removing cables" and "as everyone knows, the most commonly used photometric consistency prior is prone to local minima optimization" (I personally can intuitively follow and guess but I do not know this, since i do not work on time synchronisation for NVS). Indeed, you introduce section 3.2.1 to explain this so it is not known to everyone. Sentence at line 186 needs revision. Line 217 is splitted (residual from the figure 3).

Optical flow is precomputed, it would have been interesting to see the proposed prior applied to an end-to-end trainable architecture including optical flow training. But i understand this is beyond the specific scope of this paper.

### Questions
No questions.

### Soundness
2

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
2

### Summary
This paper proposes a global-local motion consistency prior to address temporal misalignment in unsynchronized dynamic NeRF and GS frameworks. By aligning global scene flow with optical flow, the method reduces synchronization errors by ~50% and improves PSNR by up to 4 dB, outperforming prior approaches.

### Strengths
The paper is well-structured and easy to follow. The proposed method demonstrates strong effectiveness across different D-NVS architectures and achieves state-of-the-art performance.

### Weaknesses
The proposed method offers limited novelty. This work primarily adds two constraints — Flow and Offset Regularization Term — with the latter assigned a very small weight (0.0002), while the weight for Flow is not specified.

In most reconstruction tasks, temporal misalignment is typically addressed through camera synchronization or pre-alignment of captured data. It is recommended to include results under temporally pre-aligned conditions as a baseline to more clearly demonstrate the effectiveness of the proposed approach.

Please specify the weight used for the Flow term and provide an ablation study to evaluate its influence on the model’s performance.

### Questions
Please provide additional evidence to further demonstrate the novelty and effectiveness of the proposed method.

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
4

### Summary
This paper tackles dynamic novel view synthesis without hardware synchronization.

Task configuration
* input: multi-view videos with camera poses
* output: dynamic NeRFs or Gaussians and temporal offsets for synchronization

Solution
* global-local motion consistency prior: consistency between 3D (learnable) and 2D (pre-computed) flows
* Masking out the consistency loss on textureless region

Experiments
* Competitor: Sync-NeRF on 4DGS, EDGS, and Kplanes.
* Improvements on synchronization: The average temporal error reduces about half.
* Improvements on reconstruction: 1~4 dB

### Strengths
Originality:
1. Please see weakness 1

Quality:
1. The experiments thoroughly compare the competitors.

Clarity:
1. Section 3.1 clearly defines the task: reconstructing the scene and finding the temporal offsets of cameras for given a set of videos and their camera poses.
2. Section 3.2 clearly provides the intuition (consistency between 3D and 2D flows) and the method (adding the difference between projected scene flow and optical flows, masking out textureless region).

Significance:
1. The proposed method improves reconstruction (PSNR, SSIM, LPIPS) by large margins.
2. Please see weakness 3.
3. The experiments cover large offset ranges.

### Weaknesses
1. Section 2.2. implies “Optical flow for synchronization is not novel”. The section should explain why this paper is not obvious compared to previous papers. Especially, [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes] (should be cited) already introduces 3D-2D consistency.
2. The paper will be easier to read if the terms are more precisely named and redundant sentences are removed. E.g., L021 we develop~ is redundant with L017.  It is weird to name 3D scene flow and 2D optical flow as global and local motion, respectively. L017 will be easier as follows: To tackle this issue, we impose 3D-2D consistency loss to encourage 3D scene flow to match 2D optical flow.  L095 is redundant with L093.
3. The importance of the components of the proposed method should be explained by the ablation study. Currently, we do not know whether 3D-2D flow consistency is more important than masking. If masking is more important, the paper would better put more emphasis on the masking than 3D-2D flow consistency because 3D-2D flow consistency is not novel as in weakness 1. Furthermore, the problem statement L051 and the proposed method does not match because the limitation in textureless regions is the same for RGB and flow.

### Questions
1. Resolving weakness 1 will improve my rating regarding originality.
2. Resolving weakness 2 will improve my rating regarding clarity.
3. Resolving weakness 3 will improve my rating regarding significance and logical soundness.

### Soundness
2

### Presentation
3

### Contribution
2
