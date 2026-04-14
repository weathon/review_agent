# Looking Backward: Streaming Video-to-Video Translation with Feature Banks

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6, 6, 6, 6

## Abstract
This paper introduces StreamV2V, a diffusion model that achieves real-time streaming video-to-video (V2V) translation with user prompts. 
  Unlike prior V2V methods using batches to process limited frames, we opt to process frames in a streaming fashion, to support unlimited frames.
  At the heart of StreamV2V lies a backward-looking principle that relates the present to the past. 
  This is realized by maintaining a feature bank, which archives information from past frames.
  For incoming frames, StreamV2V extends self-attention to include banked keys and values, and directly fuses similar past features into the output.
  The feature bank is continually updated by merging stored and new features, making it compact yet informative.
  StreamV2V stands out for its adaptability and efficiency, seamlessly integrating with image diffusion models without fine-tuning.
  It can run 20 FPS on one A100 GPU, being 15$\times$, 46$\times$, 108$\times$, and 158$\times$ faster than FlowVid, CoDeF, Rerender, and TokenFlow, respectively. 
  Quantitative metrics and user studies confirm StreamV2V's exceptional ability to maintain temporal consistency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a study on real-time video-to-video (V2V) translation for streaming input using diffusion models. The authors identify limitations in batch-processing V2V methods, which are constrained by GPU memory and unsuitable for continuous, long-duration videos. In response, they propose StreamV2V, a diffusion model framework that integrates a feature bank, allowing it to maintain frame-to-frame consistency by leveraging stored features from past frames. This architecture enables StreamV2V to operate at 20 FPS on a single A100 GPU, significantly faster than existing V2V methods, and to support streaming inputs of indefinite length. Experimental results confirm that StreamV2V achieves competitive consistency and temporal stability, with favorable feedback from user studies compared to other V2V models. This paper is well-organized, and the methods and results are clearly presented.

### Strengths
1.This paper addresses a critical need in real-time video-to-video (V2V) translation, especially for streaming applications, which is highly relevant to the field. 
2.The paper is well-organized, with clear explanations of both the limitations in existing methods and the motivations behind the proposed backward-looking feature bank approach. The authors’ solution is explained in a logical and accessible manner, supporting reader understanding.

### Weaknesses
1.The model sometimes struggles to maintain temporal consistency when handling videos with significant camera or object movement, which may impact the overall quality.
2.It may have limited capacity for major visual transformations, such as altering complex object characteristics, which may restrict its applicability in highly dynamic editing tasks.

### Questions
I would like to understand the rationale behind first concatenating, then randomly selecting features for matching and updating during the feature bank update process. Were other methods tested in the experiments, such as retaining a specific proportion of past and current features for matching and updating? Would the performance change if most of the randomly selected features came from either past frames or the current frame?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces StreamV2V, a real-time video-to-video translation model that uses a feature bank and backward-looking mechanism to ensure temporal consistency. StreamV2V achieves significantly faster speeds than existing methods while maintaining consistent outputs.

### Strengths
1. The paper proposes a novel approach to streaming video-to-video translation, introducing a feature bank mechanism to address temporal consistency, enabling each frame in the video stream to reference features from past frames in an innovative manner.

2. This method achieves efficient real-time processing without model fine-tuning, outperforming many comparable methods in experimental results.

3. StreamV2V requires no training or fine-tuning and seamlessly integrates with existing image diffusion models, demonstrating the potential for a wide range of video editing tasks.

4. The paper includes extensive quantitative experiments and user studies, validating its performance.

5. This paper is well written and easy to follow. The supplementary materials also include comprehensive experimental details, content, and discussions, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. Does DyMe Bank still have advantages in video quality compared to queue-based banks when dealing with longer videos, more complex video content, or video motion?

### Questions
See Weaknesses.

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
3

### Summary
The paper presents StreamV2V, a diffusion model for real-time, streaming video-to-video (V2V) translation using user prompts. It achieves efficiency by maintaining a continually updated feature bank of past frames, which it references to enhance incoming frames without needing fine-tuning. StreamV2V seamlessly integrates with image diffusion models and operates at 20 FPS on an A100 GPU, making it significantly faster than competing models. User studies and quantitative metrics validate StreamV2V's strong temporal coherence and adaptability.

### Strengths
1. this paper addresses the challenge of real-time video-to-video translation via the streamV2V framework with backward-looking and training-free techniques. Since streamV2V is training-free and fine-tuning-free compared to other video-to-video techniques, the proposed approach is feasible to be deployed to practical applications.

2. the streamV2V framework enhances coherence between adjacent translated video frames through the extended self-attention layer and a proposed feature fusion approach for intermediate features of the current and the past frames. Due to the extended self-attention layer and the feature fusion approach, streamV2V effectively reuses the intermediate features from the past video frames to generate coherent translated video frameworks between adjacent frames.

3. the streamV2V stores projected keys, projected values, and outputs of intermediate features of the past video frames through a feature bank with a dynamic merging mechanism. With the dynamic feature bank, the streamV2V framework trivially reuses historical features to align the outputs of intermediate blocks of the current and the past frames.

### Weaknesses
1. the proposed streamV2V framework fails to perform video-to-video translation for high-quality video generation in some scenarios while a prompt is provided.

2. the proposed streamV2V framework fails to perform video-to-video translation when input video includes rapid motion movements.

3. to test the generalization of the proposed framework, the manuscript misses experimentation comparisons for variations of difusion models.

### Questions
1. how does the proposed framework still outperform other video-to-video translation techniques for variations of diffusion models? Do different models with the proposed framework still outperform other video-to-video translation techniques?

2. what does the running-time comparison look like on CPU Inferences rather than GPU inferences?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper describes a video-to-video translation approach  that generates video frames in a streaming fashion. The method is based on StreamDiffusion for batch processing of frames and uses SDedit for generating each frame. The method needs no training or fine-tuning can be used together with any image diffusion model. Although the results are superior to some recent video translation models, they are inferior to the results of TokenFlow and FlowVid. However the method excels in its inference time.

### Strengths
The paper has the following novel contributions: 1) videos can be translated in a streaming fashion without the need to process long sequences of video frames, 2) The past information of previous frames is stored in a feature bank and used during the generation of current frame without the need to fine-tune the existing image generation model, 3) Extended self-attention and feature fusion techniques are introduced to provide temporal consistency between consecutive frames. The most important contributions are the streaming video generation capability and the ability to use the method as an add-on with any image generation model without any training or fine-tuning.

### Weaknesses
The proposed method is fast and efficient, but it lags behind state-of-the-art  video translation models in CLIP score and subjective evaluations. The level of novelty is marginal: the paper does not propose a new video translation architecture, but merely develops an efficient feature storage, update and fusion strategy for temporal consistency. The overall approach is based on the existing work of StreamDiffusion and SDEdit.

### Questions
The authors claim that the method can be used as an add-on with any image generation model without any training or fine-tuning. This claim needs further elaboration. How is it that the method can be integrated to any translation model without any fine-tuning? How do the authors decide which features to store in the feature bank and which features to use in feature fusion? Could some fine-tuning improve the performance of the overall model? If not, why? If so, why haven't the authors done that so that overall performance could be improved?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a real-time zero-shot method for streaming video-to-video translation based on diffusion models. The key innovation is the adoption and maintenance of a dynamic feature bank which significantly increases the cross-frame consistency in the resultant videos. Although Steam Diffusion is existing work and there are a huge amount of memory updating algorithms in the field of video understanding and visual tracking, this reviewer appreciates the soundness and practical value of the proposed approach.

### Strengths
This work is well-motivated and addresses an important problem. The method proposed is technically sound and practically feasible. The method achieves real-time V2V translation speed with a translation quality on par with the SOTA methods. The experiments, especially the ablation studies, are convincing enough to show the system-level contribution of the work.

### Weaknesses
There are still some artifacts in the result videos. This reviewer wonders whether a training-free method could eventually achieve a satisfactory performance. This will need some discussions. 
Component wise, the innovation of this work is not very high. But as stated above, this reviewer appreciates the system-level contributions of the work. 
In quantitative metrics comparison, the CLIP score is not quite differentiating.

### Questions
Refer to the first question in weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
StreamV2V is an online approach (as opposed to batch / offline) for generating continuous video-to-video translation output without limits to video length and runs at 20 fps on a single A100 GPU.  The core idea behind StreamV2V is a feature bank from past frames that is retrieved and fused to ensure temporal coherence and consistency in frame generation.

### Strengths
- The core idea of the paper makes sense and is clearly presented.
- The method achieves good results, both in terms of computational time (50 ms per frame) and quality (using objective metrics and subjective user studies).
- Limitations and weaknesses against SOTA methods are presented.

### Weaknesses
- Some magic constants are reported without explanation (e.g., threshold of 0.90, max number of frames of 2, update feature bank every 4 frames).  How sensitive is your methods to these parameters?  What are the impact of higher or lower values? 
- The user study is presented without the necessary details.  How many participants? Age group? Gender? Prior experience with generative videos?  The authors should refer to papers reporting user studies to see what is the norm of doing so (e.g., see papers from CHI)
- The choice of SDEdit (which is 3 years old) should be explained -- why not newer methods?

### Questions
Please clarify the points mentioned in weaknesses above.  Particularly:

- How are the magic constants determined and have you studied the impact?  If so, what did you find?

- Please provide the details of the user study.

- Please articulate the design choices, particularly SDEdit.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents StreamV2V, a real-time streaming video-to-video (V2V) translation method leveraging a backward-looking feature bank to maintain temporal consistency across video frames. Unlike traditional batch-processing approaches, StreamV2V processes incoming video frames in a streaming fashion, which allows it to handle an unlimited number of frames without fine-tuning. The method introduces a feature bank that stores intermediate features from past frames, extended self-attention, and direct feature fusion to improve temporal consistency. Furthermore, a dynamic feature bank updating strategy is implemented to reduce redundancy and optimize efficiency. The experimental results show that StreamV2V can achieve real-time processing at 20 FPS on a single A100 GPU, significantly outperforming existing methods like FlowVid, CoDeF, Rerender, and TokenFlow in terms of speed, with competitive consistency performance.

### Strengths
- The motivation behind StreamV2V is clear and addresses an essential limitation in existing video-to-video methods that require batch processing and are constrained by GPU memory.
- The implementation is designed for practical efficiency, with a clear description of feature fusion and dynamic feature bank updates, ensuring minimal memory overhead and ease of replication.
- The method provides flexible control over generating different styles without requiring model fine-tuning, allowing users to adapt the output to various visual aesthetics, which makes it suitable for creative applications.

### Weaknesses
- A significant concern is that using the feature bank to generate higher resolution could lead to a drastic increase in memory usage.
- The DyMe (Dynamic Merging) mechanism lacks detailed pseudocode to illustrate the workflow, which would help in understanding and replicating the process.

### Questions
- What is the size of the feature bank used in the experiments? Does the size of the feature bank affect the performance of StreamV2V?
- Can the proposed method be applied to previous text-to-video approaches to improve the performance of long video generation?

### Soundness
3

### Presentation
3

### Contribution
3
