# Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Video editing using diffusion models has achieved remarkable results in generating high-quality edits for videos. However, current methods often rely on large-scale pretraining, limiting flexibility for specific edits. First-frame-guided editing provides control over the first frame, but lacks fine-grained control over the edit's subsequent temporal evolution. To address this, we propose a mask-based LoRA (Low-Rank Adaptation) tuning method that adapts pretrained Image-to-Video models for flexible video editing.
Our key innovation is using a spatiotemporal mask to strategically guide the LoRA fine-tuning process. This teaches the model two distinct skills: first, to interpret the mask as a command to either preserve content from the source video or generate new content in designated regions. Second, for these generated regions, LoRA learns to synthesize either temporally consistent motion inherited from the video or novel appearances guided by user-provided reference frames.
This dual-capability LoRA grants users control over the edit's entire temporal evolution, allowing complex transformations like an object rotating or a flower blooming. Experimental results show our method achieves superior video editing performance compared to baseline methods. The code and video results are available at our project website: https://cjeen.github.io/LoRAEdit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a video editing method that enhances first-frame-guided approaches with more granular temporal control. The core contribution is a mask-aware LoRA fine-tuning strategy for pre-trained Image-to-Video models. This technique uses a spatiotemporal mask to teach the model to selectively preserve background content while generating new content in specified regions. This dual approach allows the model to learn consistent motion from the source video and new appearances from user-provided reference frames, enabling complex edits like a flower blooming into a different color.

### Strengths
- Video results are pretty good (videos demonstrated in the supplementary and the demo in the supplementary showcase).

- Application-wise it's interesting and definitely be useful. However, still, efficiency is a key problem.

- Clearly outperforms previous approaches in terms of both quantitative and qualitative metrics.

### Weaknesses
- The primary limitation is the need of fine-tuning per-video= The method requires 100 training steps to learn motion and potentially another 100 steps to learn new appearance (Sec 4.1). While this is more efficient than full model training, it is still a significant computational step for every single video edit compared to zero-shot or inference-time methods. How do you compare it in terms of computataional efficiency? I think putting a comparison in term of run-time memory give some insights.

- The method relies on a precise "spatiotemporal mask" to separate the edited region from the background (Sec 3.3.1). The paper does not specify how this mask is acquired. Does this require the user to perform manual video segmentation for all frames? If so, this is still a requirement that would make the method impractical for most users. If the mask is generated automatically, its quality would be critical to the final edit. What could be the possible ways to automatically get these masks?

- The full-frame method requires 20GB of GPU VRAM (Sec 4.1), which is inaccessible to many users. The "low-cost training strategy" (Appendix C) is a good alternative. Can you elaborate on this, for example what ways can be done to even make it more efficient?

### Questions
See weaknesses, but my main questions are based on its practical applicability and efficiency.

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
4

### Summary
This paper proposes a controllable first-frame-guided video editing framework that adapts a pretrained image-to-video diffusion model using mask-aware LoRA. The central idea is to repurpose the I2V model’s spatiotemporal mask not just as an inference constraint but also as a learning signal that tells LoRA what to learn. The method (i) performs per-video LoRA tuning so the model learns the input video’s motion pattern while preserving unedited regions via a mask; and (ii) optionally performs a second, brief LoRA pass where edited reference frames at later timestamps teach the model the desired appearance evolution inside masked regions (e.g., a flower gradually becoming a red rose). Experiments on first-frame-guided and reference-guided editing show better results compared with AnyV2V, Go-with-the-Flow, and I2VEdit.

### Strengths
- **Simple and efficient idea**
Using a spatiotemporal mask as an explicit LoRA supervision signal is a clean and reusable concept that doesn’t require architectural changes. This is likely to inspire follow-up work on using conditioning signals to steer what LoRA learns. This strategy addresses two persistent failure modes in first-frame editing: (i) unwanted background drift/leakage and (ii) insufficient control of how the edited object looks as it rotates/deforms or reveals disocclusions.

- **Low engineering overhead**
Per-video LoRA is a common practice. Adding mask-aware conditioning with edited frames is easy to deploy and works across different I2V-based models.

- Qualitative results look compelling on diverse manipulations (object add/replace/style, clothing/hair edits), with visibly better background preservation and temporal coherence than the chosen baselines.

### Weaknesses
- **Incremental Novelty**
Per-video LoRA for motion propagation and mask-conditioned inpainting are known practices in video editing and inpainting. The novelty is mainly in using the mask to steer LoRA’s learning objective across space-time. This is elegant but not a large algorithmic leap.

- **Mask acquisition and robustness are under-specified**
It is not clear how masks are obtained for training and inference. Is it automatically calculated from the first-frame edit difference, and then propagated with optical flow/segmentation? How to handle later edited references?
Robustness to mask errors (boundary misalignment, sparsity, temporal jitter) is not evaluated. Since the method’s key promise is precise, region-specific control, an analysis with sensitivity to mask quality is critical.

The paper tackles a high-impact practical problem (controllable propagation of first-frame edits) with a minimal, effective mechanism. The idea is easy to adopt on top of modern I2V models, and qualitative outcomes are persuasive. While the novelty is incremental, the contribution is useful and timely for creators and researchers. I tend to give a positive score.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses first-frame-guided video editing with diffusion models, proposing a training-time, mask-aware low-rank adaptation of off-the-shelf image-to-video (I2V) models. By inserting learnable LoRA modules into attention layers and conditioning the denoising network on a user-defined spatio-temporal mask, the model is taught (i) to preserve pixels where the mask is 1 and (ii) to synthesise new content where the mask is 0, either by copying motion from the source clip or by adopting appearance from an extra reference image. At inference, only the edited first frame and the same mask are needed to propagate the edit through the whole sequence. Extensive experiments against recent baselines (I2VEdit, AnyV2V, VACE, Kling1.6) show superior or comparable DEQA, CLIP-score and user-study rankings.

### Strengths
- Two-stage mask scheduling (motion learning → appearance learning) cleanly disentangles dynamics from look
- Solid empirical protocol: two tasks, two backbones (Wan2.1 & HunyuanVideo), three metrics plus user study, ablations for mask usage and extra reference frames.
- Equations are minimal and directly show the change of conditioning, easing reproducibility.
- Delivers a lightweight plug-in that practitioners can apply to any I2V model; potential to become a default extension for personalised video creation tools.

### Weaknesses
- I2VEdit already finetunes LoRA per video and uses attention masks to reduce background leakage. The paper must quantify the marginal gain of feeding the mask into the network input instead of using it only at feature level. 
- 20 self-collected clips plus an undisclosed I2VEdit test set is small. No long-form videos, fast motion, or multi-person scenes are reported. 
- Perfect masks are assumed. Evaluate with noisy masks (erosion / dilation, IoU = 0.80–0.90) or segmentation-model predictions.

### Questions
- Provide a direct ablation that keeps everything else identical except mask conditioning is missing.
- Provide frame-count vs metric curves on Benchmark to verify scalability.
- How does performance degrade under automatic segmentation errors
- Provide some failure case

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for controllable video editing by fine-tuning a pre-trained Image-to-Video (I2V) model using a mask-aware LoRA (Low-Rank Adaptation) strategy. The core idea is to use a spatiotemporal mask during LoRA tuning to guide the model in two ways: 1) to preserve or generate content in specific regions, and 2) to learn either motion from the source video or appearance from a reference image. The method is positioned as an improvement over first-frame-guided editing, offering more granular control over the temporal evolution of edits. Experiments show favorable results against several state-of-the-art baselines in both qualitative and quantitative evaluations.

### Strengths
The proposed framework is relatively simple, does not require architectural changes to the base model, and offers a flexible pipeline for various editing tasks (object replacement, addition, style transfer). The inclusion of a low-memory training strategy in the appendix is a practical consideration.

The paper provides a thorough experimental section, including comparisons with both first-frame-guided and reference-guided methods, quantitative metrics, ablation studies, and a user study.

### Weaknesses
Limited Technical Novelty: The core technical components—using I2V models, LoRA fine-tuning, and spatiotemporal masks—are not novel individually. The contribution lies primarily in their specific combination and application to this problem. However, the approach feels like a natural and incremental extension of existing "tune-a-video" and first-frame-guided paradigms. The paper does not sufficiently establish a significant technical leap over the current state-of-the-art. The method's reliance on per-video fine-tuning, while effective, is also a limitation shared with several prior works.

### Questions
No Questions

### Soundness
2

### Presentation
2

### Contribution
2
