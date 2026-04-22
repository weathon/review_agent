# Arbitrary Generative Video Interpolation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4

## Abstract
Generative Video Frame Interpolation (VFI), which synthesizes intermediate frames from a given pair of start and end frames, plays a pivotal role in video creation. However, existing generative VFI methods are constrained to producing a fixed number of intermediate frames, which significantly limits the flexibility in adjusting the frame rate or duration of videos during the creation process. In this work, we present \textbf{ArbInterp}, a novel generative VFI framework that enables efficient interpolation at any timestamp and of any length. Specifically, to support interpolation at any timestamp, we propose the Timestamp-aware Rotary Position Embedding (TaRoPE), which modulates positions in temporal RoPE to align generated frames with target normalized timestamps. This design enables fine-grained control over frame timestamps, addressing the inflexibility of fixed-position paradigms in prior work. For any-length interpolation, we decompose long-sequence generation into segment-wise frame synthesis. We further design a novel appearance-motion decoupled conditioning strategy: it leverages prior segment endpoints to enforce appearance consistency and temporal semantics to maintain motion coherence, ensuring seamless spatiotemporal transitions across segments. Experimentally, we build comprehensive benchmarks for multi-scale frame interpolation (2× to 32×) to assess generalizability across arbitrary interpolation factors. Results show that ArbInterp outperforms prior methods across all scenarios with higher fidelity and more seamless spatiotemporal continuity. Video demos are provided on the website: https://mcg-nju.github.io/ArbInterp-Web.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a method for continuous video interpolation that generates intermediate frames at any frame rate. Given a start and end frame, the method interpolates frames at arbitrary timestamps. To achieve this, the authors assign timestamps 0 and 1 to the start and end frames, respectively, normalize intermediate timestamps, and introduce a timestamp-aware RoPE. They further propose appearance–motion decoupled conditioning, enabling smooth and continuous segment-level interpolation. On VBench, the method achieves significantly superior video quality compared to prior approaches.

### Strengths
- **Novel problem setting:** This is the first work, to the best of my knowledge, to utilize generative models for continuous interpolation between two frames.
- **Clear and intuitive writing:** The paper is easy to follow and provides intuitive explanations. In particular, the concise extension from standard RoPE to timestamp-aware RoPE (TaRoPE) is elegant and highly effective. The method demonstrates strong performance, significantly outperforming existing approaches.

### Weaknesses
- While it is novel to apply generative modeling to VFI, methods that use continuous timestamps already exist in the non-generative video frame interpolation literature [1]. The paper does not compare against these methods. Including such comparisons would more clearly position this work relative to deterministic approaches.
- The authors combine timestamp-aware RoPE with a training dataset containing a wide timestamp range (1/2–1/240), which likely contributes significantly to performance. However, it is unclear whether the baselines were trained under the same settings. If baselines were trained only at fixed rates, it becomes difficult to attribute performance gains solely to TaRoPE rather than to broader timestamp coverage. It would be beneficial to train at least one baseline using the same diverse-rate data to clarify this.

[1] Super slomo: High quality estimation of multiple intermediate frames for video interpolation, Huaizu, et al., 2018

### Questions
- Beyond VFI, it would be valuable to verify whether appearance and motion have indeed been properly decoupled. If the decoupling is effective, could one borrow motion from other objects and use it as input for controlled motion transfer?
- Can the model also predict consecutive frames at non-equidistant timestamps? For example, at timestamps such as [0, 0.1, 0.4, 0.5, 0.9, 1]

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper aims to enable frame interpolation of arbitrary timesteps, which was not possible with pre-trained video diffusion model-based methods. The paper first propose TaRoPE, which is a variation of RoPE which normalizes the timestamp between the first and last frames, and enable predictions at continuous timestamps. To cope with high computation demands, the authors propose to make predictions segment-wise, either linearly (segment-by-segment) or hierarchically. In this process, to ensure spatio-temporal consistencies, appearance-motion decoupling conditioning is propoosed. The last frame tokens of a previous segment are used to ensure appearance consistencies between segments, and using a spatio-temporally tuned CLIP and a QFormer, the motion of the previous segment is encoded and this information is injected via cross attention.

### Strengths
1. The paper is easy to follow, with clear problem definition.
2. The proposed framework seems to be well-designed, with strong results.

### Weaknesses
1. Novelty of TaRoPE: the proposed approach of normalizing the timestamps sounds reasonable and seems to be effective. However, I feel hesitant to agree on the novelty of this scheme. To begin with, normalization of timestamps is a commonly used approach in the field of video frame interpolation (VFI) [1-6]. It does differ in that most methods used them for scaling the optical flows, while this paper used it in positional encoding instead of flows, but the idea itself of normalizing intermediate timestamps is not something very new in the field of (traditional) VFI. Second, normalization / use of decimal position coordinates in PE is something already discussed in [9]. For these reasons, I find the novelty of TaRoPE to be quite limited.

2. Evaluations: 1) This paper is missing several important state-of-the-art methods for comparison [11, 12]. 2) The evaluation is mainly conducted on a benchmark built by themselves, and does not evaluate on a common benchmark existing in the field, such as DAVIS and Pexels [10, 11]. Proposing a new useful benchmark is welcomed, but in order to justify the efficacy of one’s methodology, it is also important to show evaluations on a commonly used benchmark to ensure fairness. Regarding evaluations, I do not think the comparison against LDMVFI to be fair.

## On the definition of the task
I would like to suggest the task of the paper to be defined as either "generative inbetweening" or "keyframe interpolation", rather than video frame interpolation (VFI). This is not merely an issue of this paper only, but is an issue of all recent papers accommodating pre-trained video diffusion models for video frame interpolation. Despite using the same term “video frame interpolation”, the task these recent methods address are quite different from traditional VFI [1-8]. This recent VFI methods with pre-trained diffusion models [10-12] assume a more extreme, challenging scenario than the task of traditional VFI. Traditional VFI has been studied in the past decade to solve a scenario where the input video has a fps of {15, 30, 60, 120} fps and aim to increase this to {30, 60, 120, 240, …, 1000} fps [1-8]. However, recent studies based on pre-trained diffusion models, including this work, assume a scenario where the input frame rate is much lower, as low as 1 fps (25 frame gap) [10-12]. One may wonder how much of a big difference this could make, the input frame rate being 1fps vs 30fps. But I argue that this fps difference in fact does make a huge difference, changing the nature of the task. The traditional video frame interpolation task considers interpolation of 15+fps (usually 30fps) input videos. The time gap between frames is extremely small, approximately 0.03 seconds. This is an extremely short time and therefore the diversity of possible intermediate motions is very limited. On the other hand, when it comes to the scenario of using 1 fps input videos, the number of possible motions increase drastically within a 1 second gap. Accordingly, traditional VFI task is considered a “restoration” task, recovering intermediate data points from a set of densely sampled data points. On the other hand, the new extreme-case frame interpolation with pre-trained diffusion models is considered more of a “generation” task, since the given input data points are far more sparse. The two scenarios pursue clearly different directions, so I think it is not fair to compare pre-trained diffusion-based methods with traditional VFI methods in extreme-low-fps scenario. For instance, LDMVFI, one of the baselines in the paper, is trained under different settings with much smaller-size dataset. Theoretically speaking, I think there is a possibility that the experimental results could have been different under the same amount of computational cost and matched train data settings. Therefore, I think there should be a clear distinction between the two tasks and I suggest the term “keyframe interpolation” to be more appropriate for these extreme-case scenarios. “Generative interpolation” does not seem to be a good alternative, since there has been several generative model-based studies also in traditional VFI literature [13, 14]. If the paper were to adhere to the task of VFI, I suggest that the paper to also compare with traditional VFI methods in their 30 fps input scenarios too, for thorough and fair evaluations. I think this suggestion could partially resolve some of the issues I mentioned above, e.g., novelty of timestamp normalization.

[1] Jiang, Huaizu, et al. "Super slomo: High quality estimation of multiple intermediate frames for video interpolation." CVPR 2018

[2] Xu, Xiangyu, et al. "Quadratic video interpolation." NeurIPS 2019

[3] Niklaus, Simon, and Feng Liu. "Softmax splatting for video frame interpolation." CVPR 2020

[4] Sim, H., Oh, J., and Kim, M. "Xvfi: extreme video frame interpolation." ICCV 2021

[5] Kong, Lingtong, et al. "Ifrnet: Intermediate feature refine network for efficient frame interpolation." CVPR 2022.

[6] Huang, Zhewei, et al. "Real-time intermediate flow estimation for video frame interpolation." ECCV 2022

[7] Xue, Tianfan, et al. "Video enhancement with task-oriented flow." IJCV 2019

[8] Choi, Myungsub, et al. "Channel attention is all you need for video frame interpolation." AAAI 2020.

[9] Chen, Shouyuan, et al. "Extending context window of large language models via positional interpolation." arXiv preprint arXiv:2306.15595 (2023)

[10] Wang, Xiaojuan, et al. "Generative inbetweening: Adapting image-to-video models for keyframe interpolation, ICLR 2025

[11] Yang, Serin, Taesung Kwon, and Jong Chul Ye. "Vibidsampler: Enhancing video interpolation using bidirectional diffusion sampler." ICLR 2025.

[12] Zhu, Tianyi, et al. "Generative inbetweening through frame-wise conditions-driven video generation." CVPR 2025.

[13] Danier, Duolikun, Fan Zhang, and David Bull. "Ldmvfi: Video frame interpolation with latent diffusion models." AAAI 2024.

[14] Lew, Jaihyun, et al. "Disentangled motion modeling for video frame interpolation." AAAI 2025

### Questions
- I wonder the extrapolation ability and extension to other tasks of the proposed method. Despite the model being tuned for interpolation task only, it would be very interesting if the proposed method shows its effectiveness beyond interpolation task.

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
3

### Summary
This paper introduces a novel video frame interpolation pipeline that finetunes a pretrained video generation model. Conditioned on a start and end frame, the model synthesizes intermediate frames through a new rotary position encoding (RoPE) design and an appearance-and-motion decoupling mechanism. Experiments validate the efficacy of the proposed approach.

### Strengths
The paper is well written and easy to follow.
The paper presents strong experimental results that surpasses several baselines.

### Weaknesses
1. The claim that previous latent concatenation is impractical (L313-L316) lacks experimental support. To proof this, the authors should report the impact on GPU memory during training and inference, as well as the performance under an equivalent computational budget compared to the current method.

2. Since the proposed Motion Semantic Extractor (MSE) uses CLIP as its backbone, it may inherit CLIP's limitation in capturing fine-grained details. Consequently, the MSE risks failing to accurately model subtle motion information from the previous video chunk.

3. There're severe artifacts in the generated videos on the webpage. For example, the wheel in the moving car example moves unnaturally.

### Questions
Is multi-stage training necessary? Have you experimented with training all the modules end-to-end in a single stage?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ArbInterp, a generative VFI framework that supports synthesizing intermediate frames at arbitrary continuous timestamps and arbitrary sequence lengths. The core idea is to replace discrete temporal indices with normalized timestamps in temporal dimensions via a timestamp-aware Rotary Position Embedding (RoPE). The authors also introduce an appearance–motion decoupled conditioning strategy, where a Motion Semantic Extractor (MSE) helps maintain motion coherence through cross-attention. Experiments show consistent gains over generative VFI baselines across VBench dimensions.

### Strengths
- The method fine-tunes Wan 2.1 (1.3B) with relatively modest compute (50k videos, 8 GPUs), suggesting decent data efficiency for the reported gains.
- Replacing temporal indices with normalized timestamps in RoPE is simple, training-efficient, and directly addresses the fixed-length limitation in generative VFI.
- The figures and supplementary materials are clear and helpful for understanding and reproducing the approach.

### Weaknesses
- Model design constraints. Wan 2.1 uses a causal VAE, so the receptive field of the last latent can depend on previous content. This implies the encoder needs access to the whole clip, which seems at odds with arbitrarily interpolating between two standalone images. The paper should clarify this limitation and the exact inference procedure for two-image interpolation without full context.
- Limited qualitative evaluation. To better assess generalization, please include:
a: Interpolation results given two arbitrary images (without video context).
b: Multiple interpolation samples (different random seeds) for the same start/end frames to assess diversity and stability.
- The fps in the presented interpolations results is very low, weakening the author's claim of arbitrary-length interpolation.
- MultiInterpBench appears to be largely composed of DAVIS data. If so, its novelty as a benchmark is limited. Please clarify the composition and splits beyond reuse of DAVIS.
- Given that VBench automatic metrics may not fully match human perception, it would help to include more challenging and diverse cases beyond standard DAVIS examples (e.g., large motion, occlusions, motion blur, non-rigid motion, out-of-domain content), along with a small-scale user study if feasible.

### Questions
Please check Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
