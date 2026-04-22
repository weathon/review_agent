# Generative View Stitching

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Autoregressive video diffusion models are capable of long rollouts that are stable and consistent with history, but they are unable to guide the current generation with conditioning from the future. In camera-guided video generation with a predefined camera trajectory, this limitation leads to collisions with the generated scene, after which autoregression quickly collapses. To address this, we propose Generative View Stitching (GVS), which samples the entire sequence in parallel such that the generated scene is faithful to every part of the predefined camera trajectory. Our main contribution is a sampling algorithm that extends prior work on diffusion stitching for robot planning to video generation. While such stitching methods usually require a specially trained model, GVS is compatible with any off-the-shelf video model trained with Diffusion Forcing, a prevalent sequence diffusion framework that we show already provides the affordances necessary for stitching. We then introduce Omni Guidance, a technique that enhances the temporal consistency in stitching by conditioning on both the past and future, and that enables our proposed loop-closing mechanism for delivering long-range coherence. Overall, GVS achieves camera-guided video generation that is stable, collision-free, frame-to-frame consistent, and closes loops for a variety of predefined camera paths, including Oscar Reutersvärd’s Impossible Staircase.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a method for video stitching with diffusion models, addressing issues of temporal consistency and scene realism. The proposed approach uses the existing Diffusion Forcing technique as a basis for conditioning of generated chunks of the video on neighboring chunks, both in the past and in the future. The proposed Omni Guidance technique also allows explicit loop closing by conditioning video chunks on both temporally and spatially related neighbors. 

The benefits of the method proposed in the paper are demonstrated on a camera-path guided diffusion task, producing videos which are able to realistically follow a pre-defined path, generating in a way that allows future steps to be consistent with the path as well.

### Strengths
The paper is very well written, easy to follow and understand. The motivation is clearly explained and seems warranted, and the proposed solutions are elegant and simple. 

Video results presented by the authors seem consistent and high-quality. This is even more pronounced when comparing to baseline generations.

### Weaknesses
Would like to see more related work on camera-guided video generation - how have the issues in this specific use-case been addressed in previous work? 

In addition, it would be helpful to show comparisons to other methods for video generation which are able to condition on future camera poses. One of the existing baselines is limited by its autoregressive structure, and the other is designed for a specific use-case (panoramas). 

Table 3: Seems like the explicit loop closing technique hurts frame-to-frame consistency in order to improve long-range consistency. Have the authors considered ways to mitigate this loss in performance?

### Questions
1. For cyclic conditioning (section 3.5) - does conditioning the target chunk on spatially related chunks require users to manually find *all* spatially related chunks in the entire trajectory in advance? This seems doable with cameras looking forward in movement trajectories, or outwards in panoramas, but wouldn’t this be infeasible with non-trivial camera angles?
2. Have the authors verified that the provided camera trajectories are compatible with the type of data present in the model training corpus (in particular, the stair trajectories)? If these were designed to match the data, what do results look like for trajectories not directly represented in the dataset?
3. In what format are camera poses represented for conditioning? 
4. Line 460: “… suggests that the effective receptive field of GVS is local, not global.” This seems to be an effect of the “consistency propagation” as described in section 3.5. How many denoising steps are performed to generate each chunk? Have the authors attempted ablations on this hyperparameter?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Generative View Stitching (GVS), a training-free sampling method for camera-guided long-video generation that synthesizes all frames in parallel rather than autoregressively. GVS targets a key failure mode of autoregressive (AR) video diffusion—collisions and collapse when the predefined camera path would “walk through” previously generated content because the model cannot see the future. GVS divides the sequence into overlapping diffusion windows, jointly denoises target chunks with their past and future neighbors using any Diffusion Forcing (DF) video model. Two additions make this work for video: Omni Guidance (a guidance rule that strengthens conditioning on past & future within DF) and a cyclic conditioning mechanism for loop closure (long-range consistency).

### Strengths
1. Address the future-conditioning problem in camera-guided video and adapts diffusion stitching to video without retraining, leveraging DF’s token-wise noise masking. 

2. The Omni Guidance formulation modifies the joint score toward a conditional score using DF’s null-conditioning, and cyclic conditioning explicitly addresses loop closure, both are new in this context.

3. Enables stable, collision-free, long camera-guided rollouts, including looped or topologically tricky paths (e.g., “Impossible Staircase”), using off-the-shelf DF video models, which lowers the barrier to long-video generation without retraining.

### Weaknesses
1. Limited discussion of related work in camera control.
While the paper introduces a strong motivation around camera-guided video generation, it provides little discussion of prior camera control or camera trajectory-conditioned video synthesis methods. Recent works such as VideoComposer [1], CameraCtrl [2] have explored related spatial or pose-guided control mechanisms.

2. Lack of comparison with camera-control-based baselines. No quantitative or qualitative comparisons are made with existing camera-controllable video diffusion methods, making it hard to judge relative controllability and spatial consistency.

3. Missing ablation study on Diffusion Forcing (DF) effectiveness.  An ablation contrasting DF-free and DF-enabled variants would clarify how much DF drives the gains.

4. Insufficient model description and unclear generality across base models.
The paper under-specifies the DF backbone, leaving ambiguity about architecture and conditioning. It is also unclear whether results hold on stronger base models like Wan 2.1 [3].
 
5. Restricted applicability to Diffusion Forcing models.
The proposed method is tightly coupled to the Diffusion Forcing architecture, and cannot be directly applied to other video diffusion architectures.

[1] "VideoComposer: Compositional Video Synthesis with Motion Controllability". 
[2] "CameraCtrl: Enabling Camera Control for Text-to-Video Generation". 
[3] "Wan: Open and Advanced Large-Scale Video Generative Models".

### Questions
1. How does GVS conceptually differ from prior camera-controllable or pose-guided video diffusion methods?

2. Can you provide more details about the DF backbone (architecture, conditioning format, training data)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles a critical failure mode in camera-guided video generation: when using a predefined camera trajectory, autoregressive (AR) models often generate scene content (e.g., a wall) that the camera is later forced to "collide" with, causing the generation to collapse. AR methods are blind to future conditioning. The authors propose Generative View Stitching (GVS), a non-autoregressive sampling algorithm that generates all parts of the video sequence in parallel, making the entire generation faithful to the full camera trajectory. The authors note that video models trained with Diffusion Forcing already possess the necessary affordances for a training-free stitching algorithm. This is achieved by jointly denoising overlapping chunks, where a target chunk is conditioned on its co-evolving, noisy neighbors. The authors propose a novel guidance technique, Omni Guidance, which modifies the sampling step to significantly enhance temporal consistency between stitched chunks. This guidance is crucial for making the stitching process coherent. A Cyclic Conditioning mechanism is used to enable long-range consistency and loop closure. This is achieved by alternating the denoising process between temporal windows (conditioning on immediate neighbors) and spatial windows (conditioning on temporally distant but spatially close frames, like the start and end of a loop).

### Strengths
- The paper identifies a significant and practical limitation of current video generation models. The "collision" problem in predefined camera-guided generation is a major barrier to using these models for cinematography, simulation, or any form of high-level planning.
- The central idea that the Diffusion Forcing training paradigm inherently enables a powerful, training-free inference-time stitching algorithm is reasonable. It repurposes a training-time feature (conditioning on variably-noised context) to solve a new inference-time problem.
- Decent improvements upon baselines can be observed from experiments.

### Weaknesses
I do not have many problems with the technical part of the paper, but I do have concerns about the motivation and selling point of the paper. The paper puts itself as an improvement upon AR video generation models, so that the collapsing issue can be resolved.

However, the approach necessarily breaks causality. In other words, we won't be having the core motivations of benefits of AR video generation anymore:
- Efficiency because of KV cache/smart use of history.
- "Streaming" ability, where we do not have access to future contents.
By breaking causality, of course we won't be running into the collapsing issue when cameras are given, as in other camera-controlled work --- because we have knowledge to the future already. This is a point that seems trivial to me.

Overall, while the method is sound to me, the motivation/story part of the paper seems unconvincing to me. Would be good if the authors can make it clear.

### Questions
N/A

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
3

### Summary
The paper introduces a camera conditioned video generation method via a parallel sampling strategy that circumvents shortcomings in autoregressive sampling, and proposes strategies for improving temporal consistency.

### Strengths
* Presentation is very clear. 
* Several key challenges are identified, including collision, temporal consistency, etc., and are addressed by corresponding techniques proposed in the paper. 
* The framework is lightweight.

### Weaknesses
* Inference costs and their comparisons with baselines on computation complexity are not reported. 
* Metrics that specifically measure temporal consistency could be reported to strenghthen the claim on improvements along this aspect. 
* Ablations are conducted on one hyperparameter, stochasticity $\eta$. Is the method robust to other hyperparameter choices?
* The paper is restricted to pre-defined camera trajectories. Other limitations are also not discussed.

### Questions
* Does the framework support rolling out beyond 120 frames? 
* What's the size of the dataset? Is it 40 generations per trajectory or in total (line 394)?

### Soundness
3

### Presentation
3

### Contribution
2
