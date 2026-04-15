# MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6

## Abstract
Recent advances in generative AI have significantly enhanced image and video editing, particularly in the context of text prompt control. State-of-the-art approaches predominantly rely on diffusion models to accomplish these tasks. However, the computational demands of diffusion-based methods are substantial, often necessitating large-scale paired datasets for training, causing them challenges to employ in practical applications. This study addresses this challenge by breaking down the text-based video editing process into two stages. In the first stage, we leverage an existing text-to-image diffusion model to simultaneously edit a select few key frames without any additional fine-tuning. In the second stage, we introduce an efficient model called MaskINT, which is built on non-autoregressive masked generative transformers. MaskINT specializes in frame interpolation between the key frames, benefiting from structural guidance provided by intermediate frames. The training of MaskINT incorporates masked token modeling. Our comprehensive set of experiments illustrates the efficacy and efficiency of MaskINT when compared to other diffusion-based methodologies. This research offers a practical solution for text-based video editing and showcases the potential of non-autoregressive masked generative transformers in this domain.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a two-stage video editing method, which first leverages an off-the-shelf image editing method to edit keyframes, and then performs interpolation between the edited frames. Quantitative and qualitative experiments demonstrate that MaskINT achieves comparable performance with previous methods.

### Strengths
- Efficiency. The proposed method achieves comparable performance with diffusion methods, but it is much faster.

### Weaknesses
- The video editing performance heavily relies on frame interpolation performance. Almost all showed results (in main submission and Supp) are simple motions, such as car translation, rhino translation. The simple motions can be easily interpolated. But for complex motions, it is difficult to perform frame interpolation, and it also suffer occlusions. Actually, in the showed man dancing case, there are obvious artifacts in arms. Also, the proposed method may suffer a lot in case of long-range video editing. Thus, the generalization ability of the proposed method is somehow limited.
- Evaluation. There are only 11 examples in Supp, and it is difficult to judge the performance. Are the results cherry-picked? Could you give more results?

### Questions
- How about the failure cases? 
- How about long videos and complex motions?
- Would it fail if the first-stage kerframe editing fails?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a two-stage video editing framework, using T2I diffusion model to edit the key frames and then interpolating between those frames. During T2I diffusion process, the paper leveraged controlnet to jointly keep the edge consistency. After that, a Masked generative transformer model called MaskINT is introduced to generate middle frames. The results show that the proposed network can accelerate generate videos compared with baseline pipelines while suffering slightly temporal and prompt consistency decrease.

### Strengths
1. The proposed MaskINT leverage masked generative transformer to interpolate between keyframes.
2. The inference speed outperformed the proposed video editing pipelines.
3. MaskINT is trained on unlabeled video datasets using masked token modeling, without needing text-video pairs.

### Weaknesses
1. Although the proposed MaskINT can beat other methods in speed, the method still suffers consistency degradation in both prompt and temporal domain. 
2. Noticeable degradation across key frames and interpolated frames.
3. No related baseline comparison between video interpolation pipeline.

### Questions
1. By increasing the decoding step and keyframes, the method can increase the performance in Tem-Con and Pro-Con. Can the method reach comparable qualitative results in less time by increasing those hyper parameters?
2. The videos in supplementary seem to have heavy moiré patterns. Why does this occur?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new approach to structure-guided editing for videos. The proposed method is composed of two stages. In the first stage, an image-based diffusion model is leveraged, along with the cross-frame attention technique, to jointly edit a small number of key frames. In the second stage, a structure-guided non-autoregressive masked transformers model is developed for the interpolation task, aiming to propagate the information from the (edited) key frames to the intermediate frames. The experiments in the paper demonstrate the proposed method can enable temporally consistent edit propagation results while achieving better efficiency compared to existing diffusion-based approaches.

### Strengths
The video editing results provided in the paper demonstrate that the non-autoregressive masked generative modelling technique, which have mostly been applied to the unconditional generation or text-condition generation so far, can be effectively adapted to the structure-conditioning generation setting.

The experiments in the paper demonstrate that the proposed method can achieve better efficiency compared to existing diffusion-based approaches.

### Weaknesses
While the idea of extending the non-autoregressive masked transformer technique to structure-guided generation is technically sound, the technical contribution on the fundamental side is somewhat limited. Video editing with diffusion model via key-frame edit propagation has been widely explored. The effectiveness of masked generative model in video generation has also been well established. The key contribution of this paper, from my perspective, is in showing that it is possible to incorporate dense structure information into the masked transformer model. There are limited discussions in the paper, however, to provide insights on why such a task is difficult, what are the fundamental challenges in doing that, and why the proposed technique is a good solution for such challenges. 

The discussion on the technical details is somewhat vague. In particular, it seems that the model architecture details were not elaborated.

The provided evaluation is a bit weak: 
+ I feel that the subjective comparison should be made more complete: video results were only provided (in the supplementary material) for the proposed method, not for competing methods. That makes it difficult to assess the temporal quality of the proposed method in comparison with the other methods.
+ The comparison is not entirely fair, competing methods are all zero-shot setup, which never trains a video model. Existing video diffusion works have been shown to be effective for interpolation ([1], [2]), a fair comparison would be to compare with adapted versions of those methods to incorporate structure control signal.
+ It seems that the provided results are all with stylized content, which tends to make it more visually tolerable to temporal inconsistencies. As the main goal of the second-stage model is to perform keyframe propagation, I think one important test that should be done is to apply the model on the reconstructive setting, i.e. perform propagation with the original keyframes instead of edited ones and assess the reconstruction quality of the intermediate frames.

[1] Make-A-Video: Text-to-Video Generation without Text-Video Data. Singer et al., 2022

[2] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. Blattmann et al., 2023

### Questions
Please find my detailed comments in the Weaknesses section. Other than that, there are a couple of questions I’m curious about:
+ Will the proposed technique works for other type of controls such as depth maps or pose map? 
+ How will the method perform in the extrapolation instead of the interpolation setting? Or in the setting where only one key frame is edited?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
