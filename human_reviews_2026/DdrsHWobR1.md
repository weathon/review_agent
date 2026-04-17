# Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Vision-language-action (VLA) models have shown great potential in building generalist robots, but still face a dilemma–misalignment of 2D image forecasting and 3D action prediction. Besides, such a vision-action entangled training manner limits model learning from large-scale, action-free web video data. To address these issues, we propose DeFI, a novel framework that Decouples visual Forward and Inverse dynamics pretraining to exploit respective data sources, wherein video generation and action prediction are disentangled. We introduce the General Forward Dynamics Model (GFDM), pretrained on diverse human and robot videos for future prediction, and the General Inverse Dynamics Model (GIDM), trained via self-supervised learning to infer latent actions from unlabeled video transitions. These models are then integrated into a unified architecture for end-to-end finetuning on downstream tasks. In this manner, GFDM and GIDM first shine separately and then cooperate for mutual benefit. Extensive experiments on CALVIN ABC-D and SimplerEnv demonstrate state-of-the-art performance, with DeFI achieving an average task length of 4.51 for CALVIN, 51.2% success rate on SimplerEnv-Fractal benchmark and 81.3% success rate in real-world deployment, significantly outperforming prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DeFI, a framework that decouples forward and inverse dynamics pretraining for robot policy learning. A Foundation Forward Dynamics Model (FFDM) is pretrained via video diffusion on human + robot videos to model visual dynamics, while a Foundation Inverse Dynamics Model (FIDM) learns latent actions self-supervisedly from video transitions. The two are later coupled and fine-tuned on downstream tasks (CALVIN, SimplerEnv, Franka). DeFI reports higher average task length and success rates than prior VLA baselines.

### Strengths
There is clear conceptual novelty in separating forward / inverse pretraining, and synthesis of diffusion video modeling and latent-action quantization.

The experiments are extensive, with solid ablations showing component effects.

The targeted problem is significant, which is scaling robot learning with action-free human videos.

### Weaknesses
Unfair empirical comparison (major): DeFI is fine-tuned on target datasets (CALVIN, SimplerEnv, Franka), whereas baselines such as OpenVLA and $\pi_0$ appear evaluated as off-the-shelf checkpoints trained on their original dataset, which are zero-shot evaluated on the author's benchmarks against DeFI's fine-tuned checkpoint.

No zero-shot results: Despite “foundation” framing, all evaluations use fine-tuning ≥10 % of labeled data; zero-shot capability is not demonstrated.

### Questions
I highly appreciate the authors for proposing a novel idea in a potentially very impactful area. The paper is well-written and thorough. However, the concerns over the unfair comparison above is a major concern of mine. If the authors can adequately address why the comparison is, or has to be set up this way, that would help me a lot in re-evaluating the decision.

### Soundness
2

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
The authors address the problem of VLA models having a misalignment of 2d image forecasting and 3d action prediction.  The paper proposes DeFI (Decoupled visual Forward and Inverse dynamics) pretraining to disentangle video generation and action prediction.
Two models are introduced: The Foundation Forward Dynamics model (FFDM) is pretrained on human / robot videos for future predictio; and the Foundation Inverse Dynamics model (FIDM) is trained via self-supervised learning to infer latent actions from unlabeled video transitions.  The FFDM and FIDM are then integrated together in an end-to-end finetuning for downstream tasks. Performance on various manipulation benchmarks is presented.

### Strengths
- The disentanglement of forward and inverse dynamics learning enables leveraging distinct data sources.
- Enables pretraining with action-free internet-scale video data. The pretrained FFDM can then be coupled with a FIDM that is trained for actions with different embodiments if necessary.
- For robots with multiple cameras, the FFDM predicts future videos for each view independently.

### Weaknesses
- The claim of new state of art on Calvin only outperforms prior methods by 4.2%.  It will be good to justify if and why this is a significant outperformance.
- The paper doesn't provide sufficient details on how the FFDM and FIDM are interconnected.  Also more details on the action adapter at the output of the FIDM needs to be provided.
- Overall, the key insight in the paper is to break up a typical VLA network into two parts (FFDM, FIDM) with each of these being pretrained separately and then finetuned together.  This is neither a theoretical contribution nor much of a breakthrough in architecture.  In this sense, the novelty of the paper is limited.  Granted, they do claim a (slight) improvement over SOTA.

### Questions
- Unclear why the inverse dynamics only takes o_t and o_{t+n} and ignores all frames inbetween. Is the produced action then only at one time instant?
- It's unclear what outputs of the FFDM serve as inputs to the FIDM.  Are the o_t and o_{t+N} outputs from FFDM that serve as input to FIDM (particularly, o_t is an input to FFDM as well right?).  Fgiure 2(a, b) should be updated to showcase this connection.
- Authors should provide some detail on why increasing avg. length in CALVIN ABC-D benchmark is good (is this indicative of long-horizon tasks?).
- The authors only consider 3D action prediction.  I presume this is 3D end-effector position prediction (rather than joint angle prediction).  The authors should mention why this is the chos3en action to predict.
- Fig. 2(c) shows an action adapter - this is missing in 2(b) - authors should clearly indicate in 2(b) what the action adapter is.

### Soundness
3

### Presentation
3

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
This paper introduces DeFI, a framework where first a forward dynamics model and a latent inverse dynamics model is learned on various video data without actions, and then an action head is fine-tuned to map latent actions to ground truth actions in a specific embodiment. The forward dynamics model is a diffusion model in visual embedding space, the inverse dynamics model is a VQVAE, and the action head is a diffusion policy. A core claim is that pretraining forward and inverse dynamics models separately improves performance over coupled pretraining on actionless videos. Various experiments in sim and real environments show that DeFI generalizes better than previous state-of-the-art under the same downstream finetuning setup.

### Strengths
- The idea of learning forward and inverse dynamics for better generalization on video data is well-established.
- Disentangling forward/inverse dynamics learning and pretraining on large datasets is well-motivated.
- Various experiments in sim and real environments show that DeFI improves downstream policy performance over prior state of the art.
- Ablations validate the necessity of each design component in DeFI.

### Weaknesses
- The main concern that I have is end-to-end finetuning of forward/inverse dynamics models seem to undercut the claim that disentangled learning of forward/inverse dynamics model improves performance. It would be good to see a clarification on what the authors mean by coupled end-to-end finetuning, as well an ablation where only one of forward/inverse/action head is fine-tuned on downstream policy data. See question section below.
- No details on inference during robot experiments. The paper covers training and finetuning in detail in both the main text and the appendix. I might have missed it, but there doesn't seem to be much information on how exactly the forward dynamics/inverse dynamics/action head is then conditioned to complete tasks on the sim and real environments.

### Questions
- In section 3.3, what do the authors mean by finetuning the coupled FFDM and FIDM end-to-end in the title? If the core claim is disentangled forward/inverse dynamics learning, then this seems to undercut the claim; then Appendix A.2 says you freeze the forward dynamics model and only finetune the inverse dynamics model and action head. It would be nice to see an ablation / clarification.
- Could the authors clarify the inference process?
- Franka Play Dataset seems to have a wrong citation in Table 10.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DeFI, a novel framework that decouples visual forward dynamics and inverse dynamics pretraining to better leverage large-scale action-free videos for robot learning. The key innovation lies in separately pretraining two components: (1) a Foundation Forward Dynamics Model (FFDM) via video generation on mixed human/robot videos, and (2) a Foundation Inverse Dynamics Model (FIDM) using self-supervised learning to extract latent actions from video transitions without requiring explicit action labels. These models are then coupled and fine-tuned end-to-end for downstream tasks. The approach achieves state-of-the-art results on CALVIN ABC-D (4.51 average task length), SimplerEnv-Fractal (51.2% success rate), and real-world experiments (81.3% success rate).

### Strengths
- Motivation: The paper clearly articulates the fundamental misalignment between 2D video forecasting and 3D action prediction in current VLA approaches, making a compelling case for the decoupled approach.

- Presentation: The writing is accessible, the motivation is well-articulated, and the figures effectively illustrate both the method and results. The paper flows logically from problem identification to solution.

- Experimental validation: The authors provide extensive experiments across multiple benchmarks (CALVIN, SimplerEnv, real-world Franka) with consistent improvements demonstrated across all settings.

- Ablations: Tables 4-7 systematically validate multiple design choice, from the importance of pretraining to architectural decisions, providing good insights into what drives performance.

- Method: The approach offers an interesting path toward leveraging action-free data for training, with a novel inference mechanism that combines forward and inverse dynamics in a principled way.

### Weaknesses
- Outdated baselines: The comparison baselines don't include the most recent state-of-the-art VLAs, which diminishes the impact of the results. The paper would be significantly strengthened by comparisons against more recent models like Gr00t or π0/π0.5.

- Frozen FFDM limitations: The authors acknowledge that the frozen FFDM causes performance issues on SimplerEnv due to sim-to-real gaps, which appears to be a fundamental limitation of the approach that isn't adequately addressed.

- Limited gains from human videos: Table 5 shows only modest improvements from incorporating human videos (+0.17 on average task length). Given the added complexity of the dual pretraining pipeline, it's unclear whether this marginal gain justifies the approach.

### Questions
- VQ-VAE discretization: Why specifically does VQ-VAE discretization help inverse dynamics learning? Have you experimented with other discretization methods (Gaussian mixture models, simple binning) or continuous latent actions? The paper would benefit from more analysis on why this particular bottleneck design is optimal.

- DINO-based world model: You briefly mention a DINO-based world model in Table 6. Could you elaborate on why this underperforms the pixel-based approach? Intuitively, predicting future DINO latents with regression loss seems appealing - it would align better with the FIDM input space and reduce inference time.

- Scaling behavior: How does performance scale with pretraining data size? Is there a point of diminishing returns? This is particularly important given the main motivation is to leverage large-scale human data.

- Single denoising step: While you show that one denoising step maintains task performance (Table 6), can you provide visual comparisons showing what motion information is preserved versus lost? This would help understand why this aggressive optimization works.

- Failure modes: What are the primary failure cases? Does the model struggle more with forward dynamics prediction or inverse dynamics inference? Qualitatively, where does this approach excel compared to standard VLAs, and where do VLAs maintain advantages (perhaps in reactive behaviors given their faster inference)?

- Domain adaptation: Have you explored partial fine-tuning or adapter layers for FFDM that could address domain shift while preserving the benefits of pretraining? This question is relvant for the ` Frozen FFDM limitations` raised aboce

### Soundness
3

### Presentation
3

### Contribution
3
