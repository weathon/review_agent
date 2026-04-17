# Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
We introduce Genie Envisioner (GE), a unified world foundation platform for robotic manipulation that jointly learns visual representations and action policies within a single video-generative framework. At its core, GE-Base is a large-scale instruction-conditioned video diffusion model that captures the spatial, temporal, and semantic dynamics of real-world robotic interactions in a structured latent space. Building on this foundation, GE-Act employs a lightweight flow-matching decoder to map latent representations into executable action trajectories, enabling precise and generalizable policy inference across diverse embodiments with minimal supervision. Trained on over 1 million manipulation episodes, GE supports both short- and long-horizon tasks, and generalizes across embodiments.  All code, models, and benchmarks will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a joint video+action diffusion model based on transformers. One major difference to prior work is that they incorporate memory in the video diffusion model, where it is conditioned on features of previously denoised clips, in addition to the current frame observation and language instructions. The model is first fine-tuned for video generation. Then the video generation model is frozen and an action denoising module is learned, conditioned in the video DiT features. During inference, video is generated at lower FPS compared to actions for improved computational efficiency, while preserving precise control signal. The model is pre-trained on the large AGIBot dataset of robot demonstrations and fine-tuned for target embodiments and tasks. In the real world experimental evaluation it is shown to significantly outperforms VLAs, such as Pi0. In a more reliable sim evaluation on the recent RoboTwin benchmark the improvements over Pi0 are rather marginal.

### Strengths
The paper is relatively well written and easy to follow. 

The proposed model design is sound. 

The model demonstrates strong results in real world and sim evaluation. 

It also features high computational efficiency for a model based on expensive video diffusion.

A few ablations are provided.

### Weaknesses
The authors completely ignore prior work on joint video and action diffusion. Specifically, VideoPolicy [Liang et al.,  arXiv'25], Video Prediction Policy [Hu et al., ICML'25], UVA [Li et al., RSS'25], Unified world models [Zhu et al., RSS'25], Dreamitate [Liang et al., CoRL'24], to name only a few most relevant ones. The only methodological novelty of the proposed approach with respect to the ones listed above is the introduction of memory, which is never shown to bring any benefits in the paper. Introduction and related work sections require a major revision to properly reflect the contributions. 

Experimental evaluation is also inadequate. Specifically, most of the evaluations are on real robots and hence are not reproducible. The only reproducible sim evaluation is on a small subset of the very recent, RoboTwin benchmark, which is not widely used in the literature, hence lacking strong baselines. To the very least the evaluation should be reported on the entire benchmark. Preferably, the authors should report results on the standard LIBERO and RoboCasa benchmarks instead, using their respective protocols. Priority should be given to RoboCasa, since LIBERO is essentially saturated. 

Ablations are also not very informative as they are performed on a single real world task. The authors should instead ablate the value of large-scale video generation pre-training on a medium-scale sim dataset (in the same way as done in Table 4 in Video Prediction Policy). 

The effect of the memory module - the only notable methodological contribution of this paper, has to be ablated on a reliable sim benchmark as well. 

Runtime has to be compared to other efficient video+action diffusion models, like UVA. 

The GO-1 baseline is used in the experiments but it's never explained what it is.

### Questions
Please rewrite introduction and related work to properly reflect the contributions with respect to the prior work. 

Please report results on an established sim benchmark, preferably on RoboCasa. Compare to sota video+action generation methods on this benchmark(s).

Please re-do ablations on a mid-size sim benchmark as well, and add the ablation of the memory module.

Please compare runtime to other efficient video+action diffusion models, like UVA.

### Soundness
3

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
4

### Summary
This paper proposes the Genie Envisioner (GE), which consists of a world model, GE-Base, and an action policy, GE-Act. GE-Base aims to unify egocentric visual representations through large-scale pretraining, enabling it to forecast multi-view observations in robot manipulation. Building on GE-Base, GE-Act achieves efficient and accurate action policy prediction via asynchronous inference. The authors provide comprehensive implementation details for both GE-Base and GE-Act, and extensive experiments demonstrate the strong performance of GE-Base and GE-Act.

### Strengths
- Well-presented paper that provides comprehensive details and well-illustrated figures, enabling readers to fully understand the task. 
- GE-Base supports multi-view observation, which is quite helpful for robot planning tasks. 
- I like the idea of asynchronous inference, which provides a practical solution for incorporating the world model and policy.
-The video results are impressive, and the extensive experiments prove the effectiveness of GE.

### Weaknesses
- GE-Base and GE-Act are designed for a specific robot configuration (one head camera and two wrist cameras). This weakens the potential of this pipeline to be leveraged in the general robotics community.
- The two-stage VLA training pipeline, especially the specific task fine-tuning in the second stage, does not showcase the generalizability of the proposed policy. Is there a large performance drop when tested on out-of-distribution tasks?
- I am missing a comparison between single-view and multi-view world models. Does predicting multi-view robot videos damage overall performance, considering that GE-Base is built on a single-view video generator? An experiment comparing single-view robot video prediction with multi-view robot video prediction (perhaps both using the head view for comparison) could help. 
- The idea of cross-view attention is rather simple, but it is completely fine if the results turn out well.

### Questions
- Does GE-Base support co-tuning of both multi-view and single-view videos? If so, would the performance be improved?
- World models often exhibit poor performance when generating long-term trajectories. I am curious about how long GE-Base perform on extremely long trajectories.
- Training is limited to AgiBot-World-Beta, a single-platform dataset. Whether GE-Base retains knowledge from the original LTX model when applied to non-robotic scenes.

I would be happy to raise my score if the authors could address my concerns.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Genie Envisioner (GE), a large-scale instruction-conditioned video diffusion model consisting of two components: GE-Base, a foundation video diffusion model trained at scale; and GE-Act, a lightweight flow-matching action decoder operating on the latent representations produced by GE-Base.

### Strengths
1. The paper demonstrates strong engineering efforts, including large-scale data and model training.
2. This experimental performance is strong.

### Weaknesses
1. The macro-level architecture design—an action model built on top of world model representations—was, to my knowledge, first introduced in the GR-1 and GR-2 series. However, these prior works are not cited or discussed, which weakens the contextual positioning of Genie Envisioner within the literature.
2. The paper’s presentation resembles a technical report more than a polished conference paper. Although it comprehensively covers implementation aspects, it fails to emphasize the key innovations, such as: GE-Act’s fully latent-space operation, and asynchronous inference mechanism,
   which are novel techinques and deserve more prominence. This writing style also makes many details not clear enough due to the limitation of the number of main text pages (see questions below).

### Questions
1. In Line 148, what do **$N$** and **$t$** represent in the notation $x_{1:N}^{(t)}$? How is the temporal step $t$ different from the frame index $N$?
2. In Line 158, does GE-Base load pretrained weights from LTX-Video 2B? If so, how are the cross-view attention layers (described in Line 190) initialized?
3. In Line 292, what is the difference between **$B_i^{vis}$** and **$W$** in Line 151?
4. Could the authors provide a more detailed description or pseudo-code for Asynchronous Inference? This appears to be a central design element, yet it is not well explained.
5. In Figure 7, the terms GE-Act Slow and GE-Act Fast are not introduced in the main text. Are they related to asynchronous inference?

If the authors adequately address these questions, I would be willing to raise my score. I also strongly encourage restructuring the paper to highlight its core technical contributions more effectively.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel Foundational World model for robotic manipulation based on Language-guided  Video Generation. The key contribution of the work is a Diffusion-based autoregressive multi-view video chunk generation model that learns to predict egocentric as well as eye-in-hand camera feeds for left and right hands in a bimanual robot setup. The proposed model is trained in 2 stages. The first is for training the model to learn spatially and semantically coherent video generation from sparse frames at sampled at different frequencies to make the generation more robust. The second stage performs a fixed low-frequency fine-tuning to capture sequential information useful for downstream control. The learnt latent space tokens from this video model (GE-Base) are then used for training a decoder for predicting continuous bimanual actions using flow matching. The proposed approach shows promising results both on robotic video prediction compared to other stat-of-the-art video prediction models as well as on subsequent downstream robotic manipulation tasks on a variety of robots in comparison to other state-of-the-art Vision-Language-Action Models.

### Strengths
The multi-stage training especially with different varying frequencies is a clever idea to make the learned video prediction more robust to different execution speeds. The use of a FiLM style injection of the latents into the policy architechture rather than just using the last layer output is quite a nice way to ensure multi-layered information to be used more effectively for the downstream policy learning. The quality of the experiments are high and thorough. The paper is written clearly and understandably making it easy to follow.

### Weaknesses
- I did not fully understand what exactly the multi-view consistency is. Is it just the joint prediction of head, left and right video feeds? Or is in the structure of the attention layers used in the model?
- The approach requires few-shot fine-tuning for any new robotic embodiment. While this is a problem in general with VLA style policy networks, I would imagine that the video generation model, if trained on a variety of robotic embodiments, such as from Open-X-embodiments or similar datasets, in addition to the AgiBot world dataset, could help reduce this need for fine-tuning the GE-Base world model and require only fine-tuning the action decoder.
- Conceptually, the proposal of using video-based diffusion for learning latent spaces for robot manipulation has been explored previously. However, there is no mention of this in the paper, leave alone a comparison.
Wen, Youpeng, et al. "Vidman: Exploiting implicit dynamics from video diffusion model for effective robot manipulation." Advances in Neural Information Processing Systems 37 (2024): 41051-41075.
Xu, Huilin, et al. "Diffusion-Based Imaginative Coordination for Bimanual Manipulation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
- Notations in Sec. 2.1 are inconsistent with the use of bold font. The subscripts and superscripts aren't adequately explained.

### Questions
- Adding a mathematical explanation of how the losses are calculated would help improve the clarity of the paper.
- In Sec. 2.1, what is the difference between the next chunk $x_{1:N}^{(t)}$ and the sparse memory $\hat{x}_{0:t-1}$? 
- In Lines 197-198, what is the difference between $\hat{x_t}$ and $v_{\hat{t}}^{(i)}$? Isn't $v_{\hat{t}}^{(i)}$ a part of $\hat{x_t}$? Shouldn't the model output be $\hat{x}_{t+1}$ if the model predicts the next video chunk? 
- In Figure 5, are the other methods trained/fine-tuned on robotic manipulation data as well? If not then, a fairer comparison is warranted, potentially by comparing against other robot world models as mentioned in the related works.

### Soundness
3

### Presentation
4

### Contribution
3
