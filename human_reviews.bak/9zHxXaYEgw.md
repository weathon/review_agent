# LEO: Generative Latent Image Animator for Human Video Synthesis

- Decision: Reject
- Scores: 3, 6, 6

## Abstract
Spatio-temporal coherency is a major challenge in synthesizing high quality videos, particularly in synthesizing human videos that contain rich global and local deformations. To resolve this challenge, previous approaches have resorted to different features in the generation process aimed at representing appearance and motion. However, in the absence of strict mechanisms to guarantee such disentanglement, a separation of motion from appearance has remained challenging, resulting in spatial distortions and temporal jittering that break the spatio-temporal coherency. Motivated by this, we here propose LEO, a novel framework for human video synthesis, placing emphasis on spatio-temporal coherency. Our key idea is to represent motion as a sequence of flow maps in the generation process, which inherently isolate motion from appearance. We implement this idea via a flow-based image animator and a Latent Motion Diffusion Model (LMDM). The former bridges a space of motion codes with the space of flow maps, and synthesizes video frames in a warp-and-inpaint manner. LMDM learns to capture motion prior in the training data by synthesizing sequences of motion codes. Extensive quantitative and qualitative analysis suggests that LEO significantly improves coherent synthesis of human videos over previous methods on the datasets TaichiHD, FaceForensics and CelebV-HQ. In addition, the effective disentanglement of appearance and motion in LEO allows for two additional tasks, namely infinite-length human video synthesis, as well as content-preserving video editing.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces a diffusion-based method for video generation. The proposed method leverages a flow-based image animator to learn motion representations thus enabling disentangle motion from appearance. An LDM is designed to learn the motion distribution by providing the starting motion α1 as the condition.

### Strengths
1. This work tries to solve the challenging issue of disentangling motion from appearance. The method is well-motivated and the proposal method is simple to understand.
2. A Linear Motion Condition (LMC) mechanism is designed in cLMDM to condition the generative process with the first motion code α1.
3. Qualitative results show the ability to generate long videos and enable disentanglement of motion and appearance.

### Weaknesses
1. The author only includes pickup methods for comparison, STOA methods are not included for comparison. Recent methods, such as MoStGAN-V, VDM, Video-LDM, VideoFactory, and Make-A-Video, should be included for comparison.

2. The author should include experiments on more challenging datasets, such as MSR-VTT and UCF101.

### Questions
see weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a temporal generative model LEO for synthesizing editable human performance video. The key idea is to represent motions with optical flow maps to disentangle appearances and dynamics. In particular, LEO leverages a latent diffusion model trained for predicting motions in an auto-regressive fashion, and decodes the latent to form flow maps for pixel-space appearance synthesis.

Their quantitative results show obvious improvement over prior work, with qualitative evidence demonstrating better spatio-temporal coherency.

### Strengths
The paper is well-written and easy to follow, and the proposed solution sounds solid. Particularly:
- Novel formulation of diffusion-based generative model for optical flow generations, which enables long-term motion generation
- Explicit disentanglement of the video into appearance (pixel values) and motion (optical flow) that makes LEO better preserve the identity information in the input.
- Auto-regressive motion generation with careful designs that achieve long-term video generation.

The quantitative and qualitative evaluations also show significant improvement over prior arts.

### Weaknesses
While showing promising results, LEO has some limitations, which are also observed in other baselines:
- Geometry ambiguity: without any explicit notion of 3D geometry or semantic features, LEO often flips or morphs the limbs from one side to the other. This is particularly obvious in the TaichiHD videos.
- Temporal coherency: while LEO improves greatly over the other baselines compared in the paper, the appearance can still drift off/morph arbitrarily between frames, especially for videos with occlusion/dis-occlusions or large motions.
- Limitations and failure cases: these aspects are not presented in the papers and supplementary. Proper discussions on what LEO cannot do well can help the readers to better assess the contribution of the work, and also open up possible future directions.

### Questions
Below are the questions I have:
- How does the proposed LEO compare to the approaches like Siarohin et al. 2019; 2021, where the motions are disentangled into region-based descriptors/flow-field?
- What are the limitations of LEO? What are the failure cases? It would be great if the paper could show and discuss these topics.

### Soundness
3 good

### Presentation
3 good

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
The paper proposes a method to generate videos by disentangling the synthesis of appearance and motion. To this end, the authors propose a flow-based image animator and a latent motion diffusion model. In particular, the motion synthesis is conditioned on the starting motion code. This formulation allows the model to generate sequences of infinite length by changing the starting frame for each subsequence. The efficacy of the method is evaluated on multiple datasets of humans in motion.

### Strengths
- The application of synthesizing videos of arbitrary length is relevant and challenging. 
- The main idea is simple and clearly presented.
- The quantitative and qualitative results showcase the efficacy of the proposed model over the baselines on the TaichiHD, FaceForensics and CelebV-HQ datasets.

### Weaknesses
- It would be nice to see some human-specific baselines, especially since the focus of the paper is on humans, e.g., utilizing skeleton/3DMM guidance.
- I believe a comparison (or at least discussion) to video-ldm [1] would be beneficial.
- I am missing a section on the limitations and ethical considerations.

[1] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

### Questions
In general, I am positively inclined however I would suggest that the authors address the issues raised in the "weaknesses" section, especially regarding the human-specific baselines and the limitations/ethical considerations.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
