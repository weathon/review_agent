# Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Camera-centric understanding and generation are two cornerstones of spatial intelligence, yet they are typically studied in isolation. We present Puffin, a unified camera-centric multimodal model that extends spatial awareness along the camera dimension. Puffin integrates language regression and diffusion-based generation to interpret and create scenes from arbitrary viewpoints. To bridge the modality gap between cameras and vision-language, we introduce a novel paradigm that treats camera as language, enabling thinking with camera. This guides the model to align spatially grounded visual cues with photographic terminology while reasoning across geometric context. Puffin is trained on Puffin-4M, a large-scale dataset of 4 million vision-language-camera triplets. We incorporate both global camera parameters and pixel-wise camera maps, yielding flexible and reliable spatial generation. Experiments demonstrate Puffin’s superior performance over specialized models for camera-centric generation and understanding. With instruction tuning, Puffin generalizes to diverse cross-view tasks such as spatial imagination, world exploration, and photography guidance. We will release the code, models, dataset pipeline, and benchmark to advance multimodal spatial intelligence research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Puffin, a unified multimodal framework that treats the camera as a first-class modality to both (1) understand camera geometry from images and (2) generate images under precise camera control. Key ideas include: “thinking with camera” (reasoning in photographic terms such as Dutch angle / tilt-up) to bridge numeric parameters and language; a geometry-aligned vision encoder; and a continuous camera latent via pixel-wise camera maps to condition a diffusion generator. A new Puffin-4M dataset (4M image–text–camera triplets) and evaluation sets (Puffin-Und, Puffin-Gen) are introduced. Puffin reports strong results on camera understanding across several datasets and large margins over LMM baselines and PreciseCam for camera-controllable generation; ablations suggest the “thinking” and camera-map latent help.

### Strengths
1. Precise, flexible camera control via discrete tokens plus a dense per-pixel camera map.


2. Unified “think-with-camera” design that improves both understanding (pose/FoV) and controllable generation, and the concept itself is novel.


3. Scales with large curated data and cleanly extends to new tasks and parameters.

### Weaknesses
1. Missing details on the construction of Puffin-Und and Puffin-Gen.

2. Training at fixed 512 and use of central crop + resize for non-square inputs degrades understanding on datasets like LaMAR; this is acknowledged but might affect claims of generality.

### Questions
1. Line 207, table A1 in the appendix should not be directly referred to in the main content.

2. How sensitive are results to the exact photographic term thresholds (Table A1)? Any continuous-to-discrete ablation or learned bins?


3. How does Puffin handle fisheye or smartphone ultrawide distortion at test time? Could the camera-map latent be extended with distortion fields?


4. If you remove LLM-generated “thinking” (or replace with noisy/short versions), how quickly do understanding/generation scores degrade? Any attempts at self-consistency or rationale-free training?


5. The abbreviation “FoV” was never introduced.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on an interesting task -- unified camera-centric understanding and generation. There are two major contributions: they collect a 4M high-quality dataset with multiple labels for both mm understanding and generation, while they also train a unified VLM called Puffin for the target task. The proposed model achieved state-of-the-art performance on the camera-centric tasks and enables quite a few interesting applications. Overall, this paper positions “thinking with camera” as a step toward foundation‑level models that natively incorporate 3D geometry in both understanding and generation.

### Strengths
- This paper is well-structured and easy to follow
- The framing of a single model and interface for both camera‑centric understanding and generation, with camera tokens as the key abstraction, is interesting (though it follows the popular way of designing unified models)
- Puffin‑4M appears carefully constructed to supervise geometry/camera attributes across many scenes.
- Superior performance compared with sotam odels, for both tasks.

### Weaknesses
- Missing specialized strong baselines: some comparisons to **strongest specialized 3D models** (e.g., recent camera calibration/pose methods) seem limited in the main text; more head‑to‑head numbers would strengthen claims.

- L408-416: for generation comparisons, it is also useful to show standard generation evaluation metrics, such as FID -- perfect camera control makes no sense if the overall visual quality is poor. 
- Better to show the error bar for the compared models in the major experiments. 
- Details of thinking with camera is missing.

### Questions
- In A.3.3, the caption of the Puffin is generated by Qwen2.5VL will this explain why Qwen-image (reuse qwen2.5vl as encoder) performs better than GPT4o and Nano-banana? How to mitigate the affect of this issue? 
- Table A.3, bottom right cell, should be 0.2 rather than 0.05?
- For thinking with camera, will RL, such as GRPO, help improve the reasoning capability? 
- Following the previous comment, it seems the thinking with camera training data is generated by Qwen2.5VL, how to make sure Qwen is able to provide some high-quality data since it is not trained on this task? Is it possible to measure their quality? 
- Are **camera tokens** robust to out‑of‑distribution intrinsics/extrinsics?
- How does performance scale with data set size and quality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents Puffin, a multimodal model that unified camera-centric generation and understanding. Puffin treats camera parameters such as pitch, yaw and FoC as discrete tokens, similar to language tokens, thereby enabling thinking with cameras. Puffin-4M is introduced which consists of vision-language-camera triplets, constructed by collecting panoramic images followed by perspective crops on different camera angles, and synthetic captions generated by a VLM. The camera understanding takes as input : text and camera discrete tokens, and image tokens from a geometry-aligned vision encoder, while the camera generation module additionally has a learnable connector module and camera maps as additional conditioning. After multi-stage training on Puffin-4M, experiments demonstrate Puffin’s superior performance over specialized models for camera-centric generation and understanding. Futher, with instruction tuning Puffin can be extended to cross-view tasks such as world exploration and spatial imagination.

### Strengths
1. The paper is easy to read and well-motivated; in that it is the first attempt to unify camera generation and understanding.
2. Puffin outperforms existing baselines and methods across multiple benchmarks.
3. The finding that representing and learning camera parameters as discrete tokens is an impactul finding; making them almost analogous with how text tokens are used in today's generative vision systems.
4. The paper is technically dense; all design choices and the reasonings behind them are well documented.
5. I believe the Puffin-4M dataset will be a great contribution to the community.

### Weaknesses
I have a few minor weakeness/comments : 

1. As shown in Table A1, the parameter-to-term mapping is not exhaustive; for example how is a small tilt-up with a clockwise
dutch angle handled?
2. Since, the camera parameter tokenizer is similar to the text tokenizer, is there any ablation that show the effect of different kinds of text encoder? For example, is there a performance delta in using encoder / encoder-decoder / decoder-only models?

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Puffin, a unified multimodal model for camera-centric understanding and generation. Specifically, Puffin formulates camera understanding as an AR text token generation task, and camera-centric generation as a learnable query + conditional diffusion generation task in a meta-query way. Both tasks are unified in an LLM with multiple additional components (a connector and diffusion model for generation, a visual encoder for understanding).  The authors further build a large-scale dataset, Puffin-4M, containing 4M vision-language-camera triplets, to facilitate this new paradigm. Extensive experiments on benchmarks demonstrate Puffin’s strong performance across both understanding and generation tasks.

### Strengths
1. Unified camera-centric understanding and generation framework: It's a novel and meaningful idea to unify camera understanding and controllable generation in one framework.
2. Thinking with Camera: Interpreting camera parameters as text description bridges geometry and language, and further enables reasoning with the camera in understanding and generation tasks.
3. The newly collected 4M high-quality camera-centric vision-language dataset should be very helpful to the community.
4. The proposed model can outperform previous works in both understanding and generation tasks, covering multiple benchmarks. Also, the ablation comprehensively covers different components of the model.

### Weaknesses
1. Parameter comparison over previous works:  A clear analysis of model scale and computational cost is missing. Since Puffin integrates multiple large-scale model components (LLM, diffusion model, vision encoder), comparing the total parameter count and FLOPs with prior understanding and generation baselines (e.g., GeoCalib, PreciseCam, etc) is needed.
2. Data vs. model contribution: It's not quite clear whether Puffin’s performance gains mainly come from the model architecture or the large new dataset (Puffin-4M). Ablation and fair comparison under the same training dataset would be useful.
3. Multi-round conversation capability: While the paper discusses instruction tuning and cross-view reasoning, it is not clear whether the proposed model can generalize well on multi-turn interleaved understanding and generation. For example, first do generation, then understanding in the second round.

### Questions
Please see weaknesses. 

Besides, I have one more question regarding whether understanding and generation can benefit each other. In this paper, the author(s) claim that unifying understanding and generation can help each other (to a significant degree). However, it seems that in other general unified image understanding and generation works, the mutual benefit is not quite clear or not very significant. Can you explain why?

### Soundness
3

### Presentation
4

### Contribution
3
