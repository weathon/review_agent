# EchoMotion: Unified Human Video and Motion Generation via Dual-Modality Diffusion Transformer

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Video generation models have advanced significantly, yet they still struggle to synthesize complex human movements due to the high degrees of freedom in human articulation. This limitation stems from the intrinsic constraints of pixel-only training objectives, which inherently bias models toward appearance fidelity at the expense of learning underlying kinematic principles. To address this, we introduce EchoMotion, a framework designed to model the joint distribution of appearance and human motion, thereby improving the quality of complex human action video generation. EchoMotion extends the DiT (Diffusion Transformer) framework with a dual-branch architecture that jointly processes tokens concatenated from different modalities.
Furthermore, we propose MVS-RoPE (Motion-Video Syncronized RoPE), which offers unified 3D positional encoding for both video and motion tokens. By providing a synchronized coordinate system for the dual-modal latent sequence, MVS-RoPE establishes an inductive bias that fosters temporal alignment between the two modalities. We also propose a Motion-Video Two-Stage Training Strategy. This strategy enables the model to perform both the joint generation of complex human action videos and their corresponding motion sequences, as well as versatile cross-modal conditional generation tasks. 
To facilitate the training of a model with these capabilities, we construct \textit{HuMoVe}, a large-scale dataset of approximately 80,000 high-quality, human-centric video-motion pairs.
Our findings reveal that explicitly representing human motion is complementary to appearance, significantly boosting the coherence and plausibility of human-centric video generation. Project page at: https://yuxiaoyang23.github.io/EchoMotion-webpage/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Inspired by VideoJam , this work establishes a model for the joint distribution of video and motion. It explicitly denoises parametric motion and performs text-to-video & motion generation. The results demonstrate an improvement in motion smoothness and human evaluation scores compared to the baseline (Wan video ).

### Strengths
Originality:

It designs and establishes a modeling framework for the joint distribution of video and motion.

Quality:

The quality is acceptable.

Clarity:

The paper is well-structured and clearly articulated.

Significance:
1. This work proposes a solution for modeling the joint distribution of video and motion.
2. Community Contribution: The authors commit to open-sourcing their code, which will be a valuable public resource for advancing the field

### Weaknesses
1. Limited quantitative experiments: The paper only compares results with its base model using metrics that are not specialized for human motion. It lacks comparisons with closed-source models like Kling or Veo3 (it doesn't necessarily need to surpass them, but at least show the gap with SoTA models). The evaluation metrics are not focused on human motion.



2. Lack of necessary ablation studies: The effectiveness of the video-to-motion and motion-to-video  capabilities is unknown, as no quantitative results are provided. This is crucial for validating the joint distribution modeling. Furthermore, there is no ablation study for the complex training process .


3. The visual quality demonstrated in the supplementary materials is still subpar. There are instances of impossible human poses, and the characters' hands are very blurry.

### Questions
1. In the supplementary materials, specifically in sample 6 (especially the last frame) and sample 15, some very unnatural or incomprehensible human poses appear. What are the possible reasons for this?

2. As mentioned in the paper, EchoMotion can perform video-to-motion and motion-to-video  tasks. Could you provide quantitative metrics to demonstrate the performance of these tasks? Specifically, for motion-to-video, could you compare it with models like Champ , Animate Anyone, or WanAnimate (since its base model is also Wan video)?



3. The VBench metrics  used in the comparison are not specifically designed for human motion. Would it be possible to compute an FID (Fréchet Inception Distance) on the generated SMPL motion parameters?

4. It is suggested to also include comparisons with closed-source models, such as Kling, Veo3, etc.

5. If you were to use SMPL-X as the motion representation instead of SMPL, would this lead to an improvement in the representation of hands?


6. Could an ablation study be provided for the complex training process described in Section 3.2 ?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to overcome the limitations of motion generation based on pixel-level supervision in previous studies by proposing joint modeling of human appearance and motion. 

The authors propose a DiT-based architecture that processes tokens from two modalities. The SMPL parameters are used to represent human poses, and to emphasize the dual-modality nature, Query, Key, and Value extracted from both the video and motion are concatenated and processed through self-attention. This structure enables attention to consider multiple modalities, which is advantageous for joint modeling. A motion-video synchronized RoPE (MVS-RoPE) which is an encoding method applicable to both modalities, is also proposed. Specifically, a diagonal extension is proposed to prevent interference between motion latents and video latents. 

In addition, the authors propose the HuMoVe dataset, containing over 80,000 video-motion pairs. This dataset includes descriptive textual captions, 3D SMPL motion parameters, and video pairs, making it valuable for multi-modal generative modeling that considers vision, text, and motion jointly. 

The experimental results present various metrics and human evaluations, showing performance improvements over baselines. Furthermore, ablation studies for each module are provided to analyze the effectiveness of the proposed methods.

### Strengths
- The paper proposes the large-scale HuMoVe dataset. Since the dataset includes test captions, videos, and motion parameter pairs, it is highly useful for multi-modal modeling tasks.

- MVS-RoPE that can be jointly applied to visual and motion embeddings is proposed. This encoding technique utilizes diagonal positioning to prevent interference between vision and motion latents, which is a reasonable approach (although more experimental evidence is needed to support this).

- The paper is easy to follow.

### Weaknesses
- The deep network structure is only a simple extension of existing networks. Except for MVS-RoPE, the network mainly uses self-attention on concatenated features for joint modeling, which is quite simple and straightforward. Discussion on whether other components could be improved to better support joint modeling would strengthen the paper.

- The quantitative evaluation relies only on self-evaluation. Even if direct comparison with prior studies is difficult, the paper should include analyses comparing the video and motion decoder performance improved from joint modeling with existing conditional generation methods (e.g., VideoJAM) to show the degree of improvement or equivalence.

- The explanation of how text descriptions were generated for the HuMoVe dataset needs to be clarified. In particular, since the initial data were created using an LLM, the paper should provide more detailed information about the prompts used.

### Questions
- p.2 L66: The authors mention that previous works are limited because, even with a 3D prior, supervision is applied after projecting it into 2D, which constrains accurate 3D (motion) generation. However, since the proposed method is also trained through a video diffusion process, hasn’t it still failed to overcome the problem of losing 3D information?

- p.4 L189: Motion tokens are generated as 51 dimension. What is the specific reason for this number?

- Motion tokens are added diagonally to visual tokens. Since maintaining temporal alignment is sufficient, there seems to be no strict reason for using the diagonal arrangement. Is there experimental evidence supporting the "positional collisions" mentioned in the text?

- Eq. 6: If the reviewer's understanding is correct, the last term must be u_{\theta}(\phi, m_{t}, \phi)

- Quantitative comparison is provided only as self-evaluation. Although direct quantitative comparison with previous studies may be difficult, joint modeling is expected to enhance the performance of both the video and motion decoders. Therefore, a quantitative comparison between the videos and motions generated by the proposed framework and those produced by conditional generation methods (e.g., VideoJAM), given GT as condition, could better highlight the advantage of joint modeling (even if the performance does not surpass that of conditional generation).


- Minor Comments:

-- p.5 L231: i is the motion token -> i is the motion token index ?

-- Fig.3: The distinction between "noisy" and "clean" is described only in text. It would be clearer and easier to understand if visual symbols were added to indicate the presence or absence of noise.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces EchoMotion, a new framework designed to solve a critical problem in video generation: the synthesis of complex and kinematically plausible human motion. The authors argue that existing models, trained on pixel-only objectives, prioritize appearance fidelity at the expense of learning the underlying physical principles of human articulation, leading to anatomical artifacts and unnatural movements. To address this, EchoMotion's core idea is to model the joint distribution of video (appearance) and 3D human motion (kinematics), rather than just the video distribution conditioned on text. MVS-RoPE is proposed as a unified 3D positional encoding for both video and motion tokens and establishes an inductive bias for video-motion temporal alignment. A large-scale dataset HuMoVe with 80,000 video-motion pairs is constructed for training and achieves better human-centric video generation results.

### Strengths
1. The paper clearly identifies a fundamental weakness in current human-centric video generation models for kinematic correctness and proposes to explicitly model the joint distribution of video and motion as a strong inductive bias to enhance the video generation performance;
2. The MVS-RoPE design is clear and well-justified to the non-trivial problem of aligning modalities with different temporal resolutions.
3. The creation of the 80,000-pair HuMoVe dataset is a substantial contribution to the field. The lack of large-scale, high-quality, paired video and 3D motion data has been a major bottleneck.
4. The experiments are thorough and well-designed.

### Weaknesses
1. The paper does not provide a clear description of the specific "open-source datasets, movies, and the internet" used to build the HuMoVe dataset. Furthermore, the extracted motion could be noisy as the ground truth;
2. The framework's reliance on the SMPL model as its parametric motion representation creates an inherent bottleneck for fine-grained realism. SMPL is a whole-body model that offers very limited, or no, supervision for highly articulated and expressive areas like individual hand gestures and facial expressions.
3. Is the strong inductive bias harmful for those physical disabilities or significant bodily variations, such as amputees, as the underlying parametric model does not support this topology.

### Questions
I believe this paper is substantial, demonstrates improved results, and serves as a positive contribution to advancing the field of controllable video generation. Please refer to the weaknesses to further improve this paper.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes EchoMotion that accepts both video and human motion modalities for video generation with a mixed multi-modal in-context learning strategy during the training. This work also introduces a new human-centric video dataset HuMoVe that includes paired video, 3D human motion parameters, and text data.

### Strengths
1. This work proposed a Dual-Modality DiT architecture that accept input and output with different modality.
2. This proposed Motion-Video Synchronized RoPE is an interesting idea to add motion information to the model.
3. This paper proposed a new high-quality dataset for video, human motion and text.

### Weaknesses
1. The novelty of the Dual-Modality DiT and Motion-Video Synchronized RoPE is limited. The notion of multi-modality DiT is not new and the idea of adding motion information is well studied in human mesh and skeleton generation tasks.
2. There are only baseline model results of Wan-1.3B and Wan-5B which are not enough to give accurate evaluation of the proposed architecture. 
3. There is no ablation study to show the effectiveness of each proposed block in the architecture.
4. The model efficiency evaluation can add metrics like average generation fps for a more direct comparison.

### Questions
1. Why is there no video tuning result for Wan-1.3B at Table 1?
2. I saw there is a parameter named FPS in Table 4. Is that the FPS of the input video or something else?
3. Section 4.3 mentions that EchoMotion can operate bi-directionally. Can you provide quantitative results for this part to see the performance comparison with other state-of-the-art models?

### Soundness
3

### Presentation
3

### Contribution
3
