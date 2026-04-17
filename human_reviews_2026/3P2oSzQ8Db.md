# UniVDC: A Zero-Shot Unified Diffusion Framework for Consistent Video Depth Completion

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Recovering metrically consistent and temporally stable depth from dynamic videos remains challenging, particularly when sparse, noisy measurements coexist with structural voids, occlusion reveals, motion drift, and sensor dropouts. Under these conditions, single-frame methods lack temporal correction while existing video depth estimation approaches underutilize explicit sparse geometry, leading to scale drift and flicker. To address this, we introduce UniVDC, the first unified zero-shot spatiotemporal diffusion framework for long-range video depth completion. Our approach centers on multi-source geometric and semantic priors. We combine two geometric inputs: fine-grained relative depth with structural and edge cues from a depth estimator, and coarse metric depth obtained by inverse-distance–weighted interpolation of sparse measurements. Unlike methods that feed RGB frames directly, we extract global semantic features and inject them hierarchically into the diffusion network, yielding compact geometric inputs and scene context robust to frame-level appearance noise. A four-stage training protocol stabilizes prior fusion and calibrates the long-horizon scale. In inference, we introduce bidirectional overlapping sliding-window (BOSW) to reduce scale drift and boundary error accumulation over long sequences and alleviate occlusion in one-directional inference. Experiments show that UniVDC achieves state-of-the-art performance on multiple zero-shot video depth completion benchmarks in terms of completion accuracy, structural consistency, and temporal coherence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a unified video depth completion pipeline, UniVDC, that can handle various sparse depth patterns within a single framework. The pipeline is built upon Stable Video Diffusion and/or DepthCrafter, modifying a pre-trained video diffusion model to accept additional conditional inputs including sparse depth and estimated depth. Video is encoded via a CLIP-based encoder, and the denoising process is performed in the latent space. A four-stage training protocol and a bidirectional overlapping sliding-window (BOSW) inference strategy are proposed, and their effectiveness is demonstrated through ablation studies.

### Strengths
1. This paper is among the first to tackle zero-shot video depth completion, proposing a reasonable pipeline, training protocol, and inference method (BOSW). The model achieves comparable AbsRel scores to existing zero-shot depth completion approaches under the PriorDA evaluation protocol.
2. The effectiveness of the training protocol and inference method is well supported by ablation studies. In particular, the finding that inference strategies significantly influence performance is interesting

### Weaknesses
1. Unclear Method Presentation (Sec. 3)
- The mathematical formulation is inconsistent with the pipeline diagram. Specifically, equation 2 omits z^(d_init) and z^(d_rel) in Figure 2. The definition of x^d are also missing,  making it difficult to clearly follow the training objective.

2. Novelty and Contribution Clarity
- The proposed pipeline and four-stage training scheme closely follow the structure of DepthCrafter. The main modification appears to be replacing the video latent with an estimated depth latent and adding interpolated coarse depth as an additional conditional input. The last three stages of the training protocol also overlap with DepthCrafter’s three-stage training. To better demonstrate the novelty, it would be helpful to include experiments supporting the claim that “excluding raw RGB helps reduce cross-domain bias” (line 187), and additional evidence showing the effect of each stage in the four-stage training protocol beyond the results in Table 4 (line 255).

3. Evaluation method
- The evaluation strictly follows PriorDA, which focuses on extremely sparse depth inputs. However, PriorDA is not yet a peer-reviewed publication and remains available only on arXiv. In my view, its evaluation setting may not fully align with the goal of temporally and metrically consistent video depth completion, which this paper primarily targets. For practical scenarios such as autonomous driving, assuming denser LiDAR beams, as in NuScenes [a1] or Waymo [a2], would provide a more realistic and relevant benchmark.
- It is unclear whether the paper primarily targets metric-scale completion or temporal consistency. If the latter is the main focus, it would be better to compare with relative depth estimation methods using least-squares alignment using GT-depth, not partial observation. Accordingly, I suggest comparing with relative depth estimation works using GT least squares, and comparing with metric-scale depth inpainting methods following their respective evaluation protocols (e.g., Marigold-DC, Omni-DC).

(Since the proposed pipeline already takes sparse depth as input, performing least-squares alignment solely on the sparse depth points may not be sufficient for a fair zero-shot comparison with relative depth estimation methods for evaluating the temporal smoothness.)

[a1] Caesar et al, "nuScenes: A multimodal dataset for autonomous driving", CVPR 2020.

[a2] Sun et al, "Scalability in Perception for Autonomous Driving: Waymo Open Dataset", CVPR 2020.
 
4. Missing Discussion of Limitations
- The paper does not discuss potential limitations such as inference speed, GPU memory consumption, or failure cases. In particular, if the bidirectional overlapping sliding-window (BOSW) inference introduces additional computational latency or memory overhead, a discussion of these trade-offs would make the contribution more transparent.

### Questions
1. Line 151 states that the method is based on Stable Video Diffusion, while Line 351 claims it employs DepthCrafter. Which one is the actual base model?
2. Should the proposed method be interpreted as targeting metric-scale depth completion & inpainting, or rather as temporally consistent relative depth generation?

### Soundness
2

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
4

### Summary
The paper proposes UniVDC, a diffusion-based framework for zero-shot video depth completion. It fuses sparse depth, monocular relative depth, and semantic priors, trains in four stages to improve temporal consistency, and uses a bidirectional sliding-window inference to reduce flicker and drift. It achieves strong results across multiple datasets (KITTI, NYUv2, Sintel, ScanNet, etc.) without task-specific fine-tuning.

### Strengths
- The proposed method is unified and generalizable, which handles multiple degradation types within a single framework.
- The empirical results looks good with clear improvements in AbsRel and TAE across diverse datasets.
- Good ablation and thorough comparison demonstrates the effectiveness of the work.

### Weaknesses
- The model consists of several stages, which might introduce additional inference time costs. The authors also haven't provided runtime and efficient analysis.
- I would like to see more video based results and failure cases of the method.

### Questions
The overall paper looks good to me. I would like the authors deal with my weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the prompting video depth problem, namely, given an observed RGB video and partial depth cues, complete the full consistent video depth. The model is based on a per-frame relative depth model. First, the video is processed by an off-the-shelf per-frame model to initialize the depth. Then, this depth and interpolated prompting depth are encoded with a VAE encoder into latent space. A finetuned SVD denoises the latent to the target depth latent. There is also a standard depth sliding window stretching included during inference.

### Strengths
On a high level, this task is very useful and practically desired to augment sparse and noisy depth sensors.

### Weaknesses
- The consistency of the model output with the input prompt must be evaluated.
- What is the behavior if the prompt is wrong or has errors?
- The performance in the main comparison table does not show clear and significant improvement over the baselines.
- The model really lacks novelty, finetuning SVD on CVD tasks is standard and widely used. The only difference here is that this SVD is also conditioned on a partial prompting depth, which prior prompting depth anything like methods have also studied.

### Questions
- **Primary:** Why is this paper novel? What is the contribution?
- **Secondary:** How will the model behave if the prompt is noisy or wrong? How consistent is the output with the prompt?

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
This paper introduces UniVDC, a novel zero-shot framework designed to generate metrically consistent and temporally coherent dense depth videos from sparse, noisy, or structurally incomplete inputs. At its core, the method is a unified diffusion model that uniquely eschews direct RGB input. Instead, it relies on a fused set of priors: fine-grained relative depth from a SOTA estimator, a coarse-scale metric map from sparse point alignment, and global semantic features from CLIP. To effectively stabilize this complex multi-prior fusion, the authors propose a progressive four-stage training protocol. To specifically address the key challenge of long-range consistency, the paper introduces a Bidirectional Overlapping Sliding-Window (BOSW) inference mechanism, which runs diffusion both forward and backward in time to mitigate scale drift and error accumulation. Experiments are conducted on several standard and dynamic datasets (e.g., KITTI, NYUv2, ScanNetV2). The results demonstrate that UniVDC achieves state-of-the-art temporal consistency (as measured by TAE) while maintaining competitive depth accuracy against other single-frame and video-based methods.

### Strengths
1. The paper focuses the critical and challenging problem of metric-scale video depth estimation, correctly identifying the failures of single-frame methods (flickering, scale drift) which hinder practical applications.
2. The proposed training strategy is well-conceived. The four-stage progressive protocol (learning spatial features, then short-term, then long-term temporal, then fine-tuning) is logical. Furthermore, the data augmentation strategy, which constructs sparse inputs by applying diverse degradation patterns to ground-truth depth, is natural and effective for building a zero-shot model.
3. The BOSW inference strategy is conceptually elegant. By processing the sequence in both directions and blending the results, it intelligently balances the stability of early-frame predictions with the corrective information from later frames. This intuition is strongly supported by the ablation study (Table 4, I vs. G), which shows a substantial improvement in both AbsRel and TAE metrics over a unidirectional approach.
4. The experimental section is thorough and persuasive. The authors provide a comprehensive comparison against SOTA methods from three relevant domains: single-frame estimation, video estimation, and single-frame completion.UniVDC achieves an impressive, and often dominant, lead in temporal consistency (TAE) across all datasets, while remaining highly competitive in single-frame accuracy (AbsRel).

### Weaknesses
1. Although the paper claims this is A Zero-Shot Unified Diffusion Framework, the method in essence functions as a  refinement and temporal alignment pipeline for the output of a single SOTA estimator (Depth Anything v2). This heavy reliance frames the contribution less as a de novo completion solution and more as a post-processing step, potentially limiting the novelty and overselling the unified claim, as its performance is fundamentally anchored to this external prior.
2. The BOSW inference strategy, while effective, comes at a significant practical cost. It requires two full diffusion passes (forward and backward) over the data, which implies at least a 2x increase in inference time compared to a standard unidirectional sliding window, not including the overhead of overlapping and fusion. This limitation is not discussed and could be prohibitive for real-world applications like robotics or autonomous driving.
3. My Major Concern: The most significant weakness is the absence of any video results in the supplementary material or on a project page. For a paper whose core claim is temporal consistency, this is a critical omission. The static frames presented in the paper, while convincing, are susceptible to cherry-picking. The community cannot fully validate the claims of flicker reduction and temporal smoothness without viewing the actual video outputs.

### Questions
1. Can the authors provide a quantitative comparison of inference speed (e.g., in FPS) between the "Naive" (Table 4, E), "Unidirectional" (G), and "Bidir. (BOSW)" (I) strategies? Is the ~12.5% improvement in TAE (0.921 vs. 1.053 on ScanNet) and corresponding accuracy boost considered a worthwhile trade-off for the >2x increase in computational cost?
2. Given the model's reliance on Depth Anything v2 for the $d_{rel}$ prior, how does the framework perform if a different (and potentially weaker) MDE is used ? Can the MDE module be swapped out directly, and what is the performance impact? This would help clarify the robustness and generalizability of the refinement framework.
3. The zero-shot claim is a key part of the paper's positioning. In a practical scenario (e.g., autonomous driving), sparse depth would come from a live sensor like LiDAR. I would strongly reconsider my score if the authors could provide a live demo (e.g., a Hugging Face Space) where reviewers could input their own videos and sparse patterns. This would be a truly compelling demonstration of the model's zero-shot capabilities.

### Soundness
2

### Presentation
2

### Contribution
2
