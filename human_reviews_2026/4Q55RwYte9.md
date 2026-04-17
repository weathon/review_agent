# Real-Time Motion-Controllable Autoregressive Video Diffusion

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Real-time motion-controllable video generation remains challenging due to the inherent latency of bidirectional diffusion models and the lack of effective autoregressive (AR) approaches. Existing AR video diffusion models are limited to simple control signals or text-to-video generation, and often suffer from quality degradation and motion artifacts in few-step generation. To address these challenges, we propose AR-Drag, the first RL-enhanced few-step AR video diffusion model for real-time image-to-video generation with diverse motion control. We first fine-tune a base I2V model to support basic motion control, then further improve it via reinforcement learning with a trajectory-based reward model. Our design preserves the Markov property through a Self-Rollout mechanism and accelerates training by selectively introducing stochasticity in denoising steps. Extensive experiments demonstrate that AR-Drag achieves high visual fidelity and precise motion alignment, significantly reducing latency compared with state-of-the-art motion-controllable VDMs, while using only 1.3B parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AR-Drag, a trajectory-based real-time video diffusion model. AR-Drag is built on top of WAN-2.1-1.3B, using motion keypoints to define the motion trajectory. To further improve motion quality and latencies, the paper introduces two designs: i) distilling bi-directional attention in the base model into a few-step autoregressive model with self-rollout, similar to self-forcing but applying DMD on the randomly sampled n-th denoised frame; ii) the model generates history frames, leading to a stochastic MDP rollout that optimises GRPO with motion controllability and visual realism rewards. These improvements result in significantly reduced latencies and enhanced motion and generation quality compared to previous methods.

### Strengths
The paper is well-written and has a clean, straightforward design. The modification of self-forcing to introduce selective stochasticity on the sampled demonised step n is a smart design.  The inclusion of the source code is highly appreciated.

### Weaknesses
- The attached website is rushed and incomplete. It lacks examples of generations from the baselines and without visualisation of the conditioning motion trajectories, and it includes wrong input prompts. This makes it difficult for the reviewer to appreciate the generation qualities, especially for a videogen model. These visualisations would be crucial for supporting the experiment.
- Directly optimising test metrics within model training is slightly cheating compared to other baselines. While I understand GRPO is an essential part of the design, I feel like highlighting FID/FVD alone would be more fair. Additionally, having these metrics purely as an ablation for the RL component, as shown in Table 2, where the performance is also very good, seems sufficient.
- The training data collected by this paper is also important. The authors should provide more details about the collected data, including the source of the videos, how they are filtered, and the total data size of the videos. I am wondering whether the authors also consider open-sourcing the collected data?
- Considering the core design of the self-rollout is very similar to self-forcing, I would suggest adding a section on a side-by-side comparison of the implementation levels (like pseudo-code) to highlight the implementation differences. This would provide a clearer understanding of the method design.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

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
AR-Drag is a novel real-time video generation system that addresses the latency limitations of traditional diffusion models by introducing the first reinforcement learning-enhanced autoregressive approach for image-to-video generation with motion control. The method fine-tunes a base model to support motion control and then improves it through reinforcement learning with a trajectory-based reward model, utilizing a Self-Rollout mechanism to preserve the Markov property and selective stochasticity to accelerate training. With only 1.3B parameters, AR-Drag achieves high visual quality and precise motion alignment while significantly reducing generation latency compared to existing state-of-the-art motion-controllable video diffusion models, overcoming the quality degradation and motion artifacts common in few-step generation approaches.

### Strengths
1. This paper introduces the first RL-enhanced autoregressive video diffusion model that enables real-time motion-controllable generation with significantly reduced latency compared to existing bidirectional approaches.
2. This paper proposes a Self-Rollout mechanism that preserves the Markov property while selectively introducing stochasticity in denoising steps, enabling efficient reinforcement learning training for few-step video generation.
3. This paper achieves high visual fidelity and precise motion alignment with only 1.3B parameters, overcoming the quality degradation and motion artifacts that typically plague few-step autoregressive video generation methods.

### Weaknesses
1. Compared to the pretrained model used, the video quality provided in the supplementary materials of this paper shows significant degradation with very obvious artifacts. For example, there are very obvious artifacts in the cat's pants in the first video.
2. The authors only provide very few videos in the supplementary materials, and none of these videos are annotated with motion control.
3. The authors do not discuss the limitations of this method in the paper. Please ask the authors to supplement this in the rebuttal.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a real-time video generator using autoregressive diffusion model, named AR-Drag. To achieve it, they first finetune a I2V model to achieve motion control. Then the model adopts post-training via reinforcement learning with a trajectory-based reward model. It achieves the state-of-the-art performance using only 1.3B parameters.

### Strengths
1. Real-time video generation is a crucial topic in the video generation community. AR-Drag proposes a valid solution to it.
2. The latency analysis shows AR-Drag is close to real-time generation. The quantitative metrics demonstrates the method achieves the SOTA performance in trajectory controlled video generation.
3. The method incorporates a GRPO post-training process that further improves the performance.

### Weaknesses
1. An important baseline is missing. Please compare AR-Drag to MagicMotion.
2. Can you compare AR-Drag with Longlive[2], which is alose a real-time video generation with 1.3B parameters.

### Questions
Please see the weaknesses.

### Soundness
3

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
5

### Summary
This paper presents AR-Drag, a real-time, motion-controllable autoregressive (AR) video diffusion model optimized for I2V generation. It considers two challenges in AR video diffusion: the quality degradation and motion artifacts due to error accumulation and limited support for complex motion control. A two-stage training pipeline is proposed. The first stage is the base model training, including distilling the model into a few-step AR model, coupled with self-rollout, a training strategy that aligns training with inference. The second stage is reinforcement learning enhancement, which use GRPO to optimize the motion fidelity and visual quality via a trajectory-based reward.

### Strengths
1. The model has the real-time inference capability, orders of magnitude faster than bidirectional models, and achieved low FID and FVD score.

2. This work applies GRPO to AR video diffusion, introducing a trajectory-based reward for quality improvement.The

### Weaknesses
1. The novelty is quite limited, as the core components, like distilling, self-rollout, GRPO for video generation, have been individually explored in prior works. The proposed self-rollout only improves from self-forcing with minor modifications. The selective stochasticity adopted by AR-Drag that only chooses one denoising step instead of the whole stochastical trajectory is also a simple improvement.

2. The comparison in Table 1 is not that fair. AR-Drag uses the Wan backbone, which is much better than the backbone adopted in some competitor methods, line DrageNUWA, Tora. For Self-Forcing, it is not specifically designed for I2V (it’s backbone is T2V).

3. I didn’t find the detailed description of the conditioning on the previous frames (i.e. how the AR works). Only a figure is presented.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
