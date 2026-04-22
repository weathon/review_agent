# Streaming Drag-Oriented Interactive Video Manipulation: Drag Anything, Anytime!

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Achieving streaming, fine-grained control over the outputs of autoregressive video diffusion models remains challenging, making it difficult to ensure that they consistently align with user expectations. To bridge this gap, we propose \textbf{stReaming drag-oriEnted interactiVe vidEo manipuLation (REVEL)}, a new task that enables users to modify generated videos \emph{anytime} on \emph{anything} via fine-grained, interactive drag. Beyond DragVideo and SG-I2V, REVEL unifies drag-style video manipulation as editing and animating video frames with both supporting user-specified translation, deformation, and rotation effects, making drag operations versatile. In resolving REVEL, we observe: \emph{i}) drag-induced perturbations accumulate in latent space, causing severe latent distribution drift that halts the drag process; \emph{ii}) streaming drag is easily disturbed by context frames, thereby yielding visually unnatural outcomes. We thus propose a training-free approach, \textbf{DragStream}, comprising: \emph{i}) an adaptive distribution self-rectification strategy that leverages neighboring frames' statistics to effectively constrain the drift of latent embeddings; \emph{ii}) a spatial-frequency selective optimization mechanism, allowing the model to fully exploit contextual information while mitigating its interference via selectively propagating visual cues along generation. Our method can be seamlessly integrated into existing autoregressive video diffusion models, and extensive experiments firmly demonstrate the effectiveness of our DragStream.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents the REVEL task, designed to enable users to edit the future generation content by conducting drag-style editing on the existing last frame. It unifies the editing and animation tasks under one framework, allowing for user-specified translation, deformation, and rotation effects on general objects during generation. This paper proposes the DragStream framework to solve the REVEL task. DragStream contains ADSR to handle distribution shift and SFSO to mitigate context inference from previous frames.

### Strengths
The strengths of the framework can be summarized as follows:

1. Interesting task proposed: The REVEL task that allows the user to modify anything on anything that unifies drag-editing and animation.
2. Both ADSR and SFSO are well-structured and decently presented methods
3. The experiment and the appendix section provide a relatively comprehensive view of the performance of the DragStream framework

### Weaknesses
Despite the paper presenting an interesting task and relatively comprehensive experiment results. However, the weakness of this paper can not be ignored. The weaknesses are as follows:

1. **Unclear and impractical task definition:** Despite being conceptually novel, I think the REVEL task is not clearly or realistically defined. The task claims to have the effect of modifying the generated videos at any time and for anything. This creates an internal contradiction in the task definition and leads to practical limitations. In particular, editing only future frames can cause severe temporal discontinuities between past and modified content, violating one of the central goals of video generation—temporal consistency. This is my major concern to the paper.
2. **Overclaiming effect:** The paper claims that the REVEL task allows users to edit anything at any time. However, the DragStream framework can not support that. It only supports three ways of editing (namely translation, deformation, and rotation), but is unable to perform broader semantic or structural edits. Moreover, it operates only during the generation process, not on completed videos.
3. **Lack of failure case studies: T**here are no studies of how, why, and under what circumstances the framework would fail. This makes it hard to understand the effect of the framework comprehensively.
4. **No latency analysis:** Since the proposed framework requires the user to edit during video generation, the computation time cost is vital. However, the paper does not report latency, runtime, or GPU cost for editing a single video, which are essential for assessing the practicality of interactive systems.
5. **Complexity in notation:** Although the methodology is detailed, the notation used in the paper is unnecessarily complicated, which hampers readability and accessibility for a broader audience.

### Questions
I have the following questions for the authors of the paper to refine their paper further:

1. I'd like to know if there are any methods the authors can propose to propagate editing into previous frames, not just the frames to be generated.
2. Since the paper only provides three ways of editing, I wonder whether it can support general editing instructions.
3. I wonder how, why, and under what conditions the DragStream framework would fail.
4. What is the average time to process a video?

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
This paper introduces a new task, streaming drag-oriented interactive video manipulation and a corresponding training-free framework called DragStream. The goal is to enable fine-grained, real-time video control via drag-style user interactions, allowing users to modify generated videos anytime on any object. To address two key challenges—latent distribution drift and context interference—the paper proposes two techniques: Adaptive Distribution Self-Rectification (ADSR), which stabilizes latent features using neighboring frame statistics, and Spatial-Frequency Selective Optimization (SFSO), which mitigates contextual noise through selective propagation in spatial and frequency domains.

### Strengths
* The work defines a clear and meaningful new task that formalizes streaming, interactive drag-style manipulation, a setting not explicitly covered by prior works.
* The proposed DragStream method is conceptually elegant and computationally efficient, requiring no finetuning while offering plug-and-play integration with autoregressive diffusion models.
* The paper is well-written, and the figures effectively convey both the problem and the proposed solution.

### Weaknesses
* Evaluation is limited to adapted baselines that are not optimized for streaming settings, which makes it difficult to judge true generalization advantages.
* The framework depends heavily on empirical hyperparameters, e.g., cutoff frequencies, Gaussian filtering spread, without deeper theoretical justification or convergence analysis. And these empirical hyperparameters may harm its generalization ability.
* The paper focuses on short clips and relatively simple object manipulations. However, in real streaming setups, drag edits often accumulate over long sequences. The manuscript does not examine whether distribution drift resurfaces over extended temporal spans or whether contextual filtering introduces temporal lag or smoothing artifacts.
* Although the method is claimed to be “training-free,” the iterative latent-region optimization may still incur noticeable inference overhead. The paper lacks detailed runtime analysis, such as per-frame latency under streaming conditions, GPU memory usage, or responsiveness to real-time drag input.

### Questions
Most of my concerns have been discussed above, while I still have several minor questions as follows,

* Can the authors provide more insight into how ADSR interacts with the diffusion denoising schedule? For example, does applying rectification at different timesteps change the generation stability?

* How does the framework handle object occlusions or re-appearance during dragging? Is there any mechanism to prevent artifacts when an object moves out of view and re-enters the frame?

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
3

### Summary
This paper introduces REVEL (stReaming drag-oriented interactive vidEo manipuLation), a new task for fine-grained, interactive control of streaming video generation. The authors propose DragStream, a novel training-free method that enables users to drag any part at any time by manipulating objects within autoregressive video diffusion models. DragStream addresses two key challenges: it uses an Adaptive Distribution Self-Rectification (ADSR) strategy to prevent the accumulating "latent distribution drift" caused by user edits, and it employs a Spatial-Frequency Selective Optimization (SFSO) mechanism to mitigate "context interference" from previous frames. This allows users to perform complex translation, deformation, and rotation operations on streaming video content.

### Strengths
The method's primary strengths lie in its practical application and thorough empirical validation.

- The ability to perform streaming, drag-oriented control is a highly useful and interesting application for interactive video generation, moving beyond static, one-off edits.

- The visual results, as demonstrated in the supplementary videos, are impressive and clearly show the method's effectiveness in manipulating video content smoothly and coherently across frames.

- The paper is supported by a strong set of quantitative experiments and a detailed ablation study (e.g., Figures 6 & 7) that clearly validates the contribution and necessity of each proposed component (ADSR and SFSO).

### Weaknesses
- I think the main major weakness is the presentation of the paper. The core concepts, which appear to be intuitive, are obscured by an extremely dense and overly complicated notational system. This heavy use on complex notations makes it very difficult for the reader to build intuition and grasp the main contributions without significant, unnecessary effort.

- In the Spatial-Frequency Selective Optimization (SFSO) mechanism, the Switchable Frequency-domain Selection (SFS) strategy (Prop 3) is key. The paper states the cutoff frequency $\omega$ is randomly selected from {0.2, 0.4, 0.6, 1} (an ablation study is also done on fgure 7). Could the authors elaborate on the intuition and provide more explanation?

- For the Adaptive Distribution Self-Rectification (ADSR) strategy (Prop 2), the latent code is rectified using the statistics (${\mu}_{T'}$, ${\sigma}_{T'}$) from preceding neighboring frames. How is the window size for these neighboring frames determined, and how sensitive is the model to this hyperparameter? Is there a trade-off between a larger window (more stable statistics but potentially less relevant) and a smaller one (more relevant but noisier statistics)?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article introduces a novel task, REVEL, and a corresponding solution, DragStream.

REVEL focuses on streaming-based video generation, enabling users to interactively modify any part of the frame at any time through dragging.

DragStream is a training-free approach that consists of two key modules: (1) ADSR, which addresses latent distribution drift, and (2) SFSO, which mitigates the dominance of high-frequency details in the context during the drag process.

Extensive quantitative and qualitative experiments demonstrate the effectiveness of the proposed method.

### Strengths
- This paper is written clearly and is easy to follow.
- It introduces a new task, REVEL, which focuses on streaming-based video generation, allowing users to interactively modify any part of the frame at any time through dragging.
- To address REVEL, the paper proposes a training-free method called DragStream, which consists of two core modules: ADSR and SFSO.
- ADSR addresses the issue of latent distribution drift by normalizing the current latents through statistical analysis of the mean and variance of neighboring latents, thereby adjusting the distribution.
- SFSO tackles the influence of high-frequency details in the context on the dragging process. Specifically, it uses Switchable Frequency-domain Selection (SFS) to control the amount of high-frequency information and Criticality-driven Spatial-domain Selection (CSS) to focus modifications on the handle region, reducing the impact on the global background.
- Quantitative and qualitative ablation experiments demonstrate the effectiveness of both ADSR and SFSO.
- Section D further discusses the differences between streaming/autoregressive models and non-streaming models.

### Weaknesses
- The efficiency of iterative latent region optimization (Sec. C.1) is stated with $I=4$, but I would like to know the runtime or complexity analysis. Since streaming-based generation is often used for real-time applications, can DragStream maintain sufficiently fast performance?
- For long video generation (e.g., exceeding 5 seconds, requiring multiple extensions), errors from autoregressive generation tend to accumulate over time. Is DragStream still effective in such cases where longer durations are involved?
- Does the denoising timestep ($T^′$) have a significant impact on DragStream? Is the choice of ($T^′$) dependent on the specific (distilled) video model?
- What happens if there is a conflict between the text prompt and the drag signal? For instance, if the text prompt directs a car to turn right while the drag signal indicates a turn to the left.
- Although Section D discusses non-streaming scenarios, It may also be worth validating DragStream on bi-directional models such as Wan2.1/2.2. If it proves effective in such cases, the training-free approach would be highly accessible to the community. At the very least, a discussion on the design considerations for such models would be valuable.
- My understanding may be incomplete, but it seems there could be a contradiction between on-the-fly editing and streaming generation. Typically, users might prefer to view the generated results first, then decide to modify a specific frame if it does not meet their expectations. However, in cases where the text prompt is incapable of achieving the desired outcome, users might apply drag operations before seeing the result. Could the authors elaborate more on the user cases of REVEL?

**Minor point**: The terms "ADSR" and "SFSO" could potentially be replaced with more intuitive, semantic meaningful names.

### Questions
I think the logic and contribution of this paper is clear. What I am most curious about are three points: the efficiency of DragStream, its performance in long video generation, and which hyperparameters are robust versus which are sensitive.

### Soundness
4

### Presentation
4

### Contribution
3
