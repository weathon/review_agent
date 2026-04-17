# Multi-Human Interactive Talking Dataset

- Decision: Reject
- Scores: 4, 4, 0

## Abstract
Existing studies on talking video generation have predominantly focused on single-person monologues or isolated facial animations, limiting their applicability to realistic multi-human interactions. To bridge this gap, we introduce MIT, a large-scale dataset specifically designed for multi-human talking video generation. To this end, we develop an automatic pipeline that collects and annotates multi-person conversational videos. The resulting dataset comprises 12 hours of high-resolution footage, each featuring two to four speakers, with fine-grained annotations of body poses and speech interactions. It captures natural conversational dynamics in multi-speaker scenario, offering a rich resource for studying interactive visual behaviors. To demonstrate the potential of MIT, we furthur propose CovOG, a baseline model for this novel task. It integrates a Multi-Human Pose Encoder (MPE) to handle varying numbers of speakers by aggregating individual pose embeddings, and an Interactive Audio Driver (IAD) to modulate head dynamics based on speaker-specific audio features. Together, these components showcase the feasibility and challenges of generating realistic multi-human talking videos, establishing MIT as a valuable benchmark for future research. The dataset and code will be public available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Multi-human Interactive Talking (MIT) dataset, designed for multi-person talking video generation, and presents CovOG, a baseline model that integrates both pose and audio cues to produce natural and realistic multi-human talking videos. The resulting dataset comprises 12 hours of high-resolution footage, each featuring two to four speakers, with fine-grained annotations of body poses and speech interactions.

### Strengths
1.This work presents the first dataset specifically designed for multi-human talking video generation, addressing the scarcity of multi-human interactive data.
2. It develops an automatic data collection pipeline and construct the first dataset for multi-human talking video generation, featuring annotations of pose and speech interaction.
3. A baseline model is proposed for this task, which supports a flexible number of human speakers and captures the dynamics of speech interactions. We further conduct extensive studies to benchmark our baseline against existing methods and analyze its performance.

### Weaknesses
1. Although this paper is the first to propose a dataset for multi-human talking video generation, several recent works have also addressed this task, including: 
[1] HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters
[2] Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation
[3] Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router

2. Current video generation models typically require a large amount of training data. However, the proposed dataset contains only about 2 hours of video, which is relatively small and may limit its applicability. Moreover, the videos are sourced from only two channels, resulting in limited diversity.
3. Most talking head generation methods employ evaluation metrics such as Sync-C and Sync-D (as introduced in[4]) to measure the synchronization between audio and lip movements. However, this paper does not include any evaluation of audio–lip synchronization, which makes it difficult to quantitatively assess the alignment quality of the generated videos. 
[4]Out of time: automated lip sync in the wild.
4. It is unclear whether the proposed method supports single-person talking head generation. If it does, a comparison with other single-person talking head methods should be provided.
5. As far as I am aware, TalkNet’s scores can be highly unreliable when the speaker’s speech intervals are short. Additionally, the presence of background noise in the video may lead to evaluation errors. I would appreciate it if the authors could elaborate on how their data annotation pipeline addresses these issues.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a multi-human interactive talking video dataset and provides a first benchmark for this complex task. The paper provides a first baseline implementation, CovOG which utilizes pose + audio to generate multi-human videos.

### Strengths
The problem of multi-human interaction is highly relevant and challenging. Providing high-quality datasets for this task is crucial. To proposed baseline is simple yet effective.

### Weaknesses
The authors note that “How to effectively evaluate lip synchronization in such interactive contexts remains an open problem” (L353-354) and instead opt for a user study. While a user study is useful, I would argue for a dataset an accompanying is crucial. My concern here is that the dataset might be released prematurely without a good benchmark that measures how well the audio and video align, i.e. how well the speaker (lip movement, correct person)  and the audio are in agreement. I wonder if something like LIPINC [1] could be utilized for this.

I find utilizing 2D keypoints as the only conditioning (besides the reference image) limiting - For example, I wonder if the authors tried to extract SMPL poses from the videos?

Can the actors elaborate on why they not directly utilize the speaker scores but instead learn a projection in their interactive audio driver? It seems like the goal is to act as a form of Gating (Figure 3 - c) which can be achieved directly from the scores.

Can the authors elaborate why the generated human motion looks so smooth, both for animate anyone and their baseline? This is particularly noticeable in the face region. I wonder if this stems from the stable diffusion VAE? I believe their relatively low resolution (640x384) + the VAE play a role in this. Furthermore, there are visible background artifacts, which probably also stem from the image-based method. I wonder if the authors have tried video auto-encoders, i.e. the one from WAN to obtain more temporally stable results.
 
Relevant related works: [2] + [3]

[1] EXPOSING LIP-SYNCING DEEPFAKES FROM MOUTH INCONSISTENCIES, ICME 2024

[2] Towards social artificial intelligence: Nonverbal social signal prediction in a triadic interaction, CVPR 2019

[3] A Large-Scale Dataset for Multi-Shot Human Speech Video Generation; NeurIPS DBT 2025

### Questions
In Figure 1, according to the description, the  blue boxes represent existing pipelines and the red boxes represent additional pipeline extensions (paper contributions). What is the green box for?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
In this paper the authors proposed a Multi-human Interactive Talking (MIT) dataset, a benchmark for multi-person talking video generation

### Strengths
The community really need a high-quality large-scale Multi-human Interactive Talking dataset. Where gestures, expressions and other dynamic behaviors of the human subject involve in the talking MUST BE TEMPORALLY AND PHYSICALLY CONSISTENT. However, the results shown in the supplementary video lacks all these attributes,

### Weaknesses
The video result shown in the supplementary results are not temporally and physically consistent.  
For examples 
- the number of fingers and their shapes changing across frames. Flat palm frequently morphing into fist suddenly
- Object in the hand are changing across frames, 
- Facial expressions are not consistent across frames

### Questions
This dataset need a thorough user study to validate the curated/generated dataset is visually acceptable or not, physically accurate or not.

### Soundness
2

### Presentation
2

### Contribution
2
