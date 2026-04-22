# Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Audio–video (AV) generation has often relied on complex multi-stage architectures or sequential synthesis of sound and visuals. We introduce \textsc{Ovi}, a unified paradigm for audio–video generation that models the two modalities as a single generative process. By using blockwise cross-modal fusion of twin-DiT modules, \textsc{Ovi} achieves natural synchronization and removes the need for separate pipelines or post hoc alignment. To facilitate fine-grained multimodal fusion modeling, we initialize an audio tower with an architecture identical to that of a strong pretrained video model. Trained from scratch on hundreds of thousands of hours of raw audio, the audio tower learns to generate realistic sound effects, as well as speech that conveys rich speaker identity and emotion. Fusion is obtained by jointly training the identical video and audio towers via blockwise exchange of timing (via scaled-RoPE embeddings) and semantics (through bidirectional cross-attention) on a vast video corpus. Our model enables cinematic storytelling with natural speech and accurate, context-matched sound effects, producing movie-grade video clips.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces OVI, a unified audio-video generation model using symmetric twin DiT backbones. The model employs blockwise cross-modal fusion and scaled-RoPE for temporal alignment. It is trained in stages: first pretraining a foundational audio tower, then jointly finetuning the twin backbones on a large-scale audio-video corpus.

### Strengths
1. The work presents a one-stage, end-to-end framework for joint audio-video generation. This approach successfully enables lip-synchronized speech generation.
2. The authors have developed a comprehensive data processing pipeline for large-scale audio-video data. This pipeline notably includes strict synchronization filtering using SyncNet and a unified captioning strategy.
3. The paper provides cross-modal attention visualizations (Figure 3). These qualitatively demonstrate that the model learns meaningful alignments between modalities.

### Weaknesses
1. The supplementary material lacks comparative results (e.g., generated videos) against other methods.
2. The subjective evaluation is missing an assessment of semantic alignment.
3. The objective evaluation lacks metrics for video quality, text-video alignment, and audio-video alignment/synchronization.
4. Comparisons against several other relevant methods are missing [3] [4].
5. The paper's objective evaluation is incomplete. It misses an objective comparison table for the joint audio-video generation task against baselines [1][2][3][4].

[1] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization 

[2] UniVerse-1: Unified Audio-Video Generation via Stitching of Experts 

[3] A simple but strong baseline for sounding video generation: Effective adaptation of audio and video diffusion models for joint generation. 

[4] MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation

### Questions
1. Will the training dataset be made publicly available?
2. What would happen if the video and audio branches used their own separate prompts?
3. The paper fails to provide ablation studies for its most innovative and critical components, such as the effect of the scaled-RoPE technique.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a joint audio-video generation model.
Starting from a pre-trained video generation model (Wan2.2), the audio backbone is designed with an identical architecture, which simplifies the design of the cross-modal interaction modules.
The model is trained in two stages: the audio backbone is first trained from scratch on speech and sound-effect datasets, followed by joint training of the audio and video branches on audiovisual datasets.
Subjective evaluations demonstrate that the proposed model outperforms existing open-source joint audio-video generation models.

### Strengths
- The proposed model achieves state-of-the-art quality compared to existing open-source joint audio-video generation models.
- The architecture is simple yet effective, yielding improved generation performance.
- A new audiovisual dataset construction pipeline is introduced, producing well-synchronized audio-video pairs with rich captions, which could serve as a valuable contribution to the community.

### Weaknesses
**Lack of methodological novelty**.  
The proposed approach replicates prior work (especially JavisDiT), raising concerns about the method's originality. Specific overlaps include:

- The overall modeling framework and two-stage training strategy closely follow JavisDiT. JavisDiT also employs an identical architecture for the audio backbone with video branch (while it is based on OpenSora rather than Wan2.2) and uses a similar two-step training procedure: audio pre-training followed by joint audiovisual training. 
- The paper explains the limitation of JavisDiT as requiring "learned prior estimator ... in order to achieve synchronization" (Sec 2.4), but it is unclear why this is problematic or how severe this limitation actually is.
- The proposed unified prompt conditioning mechanism differs from JavisDiT, but there is no direct comparison demonstrating its benefit.
- The proposed RoPE configuration appears to be identical to that of MMAudio (see Fig. A5 in the MMAudio paper), which does not solely establish the novelty.

Given these similarities and lack of justification of the proposed design choices, the contributions (2), (3), and (4) in the introduction are not convincingly supported.

**Insufficient experimental evidence**.  
The experiments are limited and do not clearly demonstrate the advantages of the proposed model.

- Details of the subjective evaluation are missing. How many samples were evaluated per participant? Which prompts (or Verse-Bench splits) were used for generation? What questions were employed for each evaluation criterion?
- Objective evaluation is limited to T2A and TTS tasks and focuses only on audio quality. Since the proposed method jointly generates both audio and video, it would be better to include evaluation of video quality (e.g., FVD[1], CLIP score[2], or VBench[3]) and audiovisual alignment (e.g., ImageBind[4] similarity, AV-Align[5], or DeSync[6]).
- Fig.4 indicates that Ovi underperforms Wan2.2 for video generation, while Table 2 shows that Ovi underperforms most existing T2A or TTS models for audio generation. Based solely on these results, it is difficult to identify the strengths of the proposed approach. It would be more convincing to compare Ovi with a sequential baseline (e.g., Wan2.2 for T2V followed by MMAudio-L for V2A, or FishSpeech for TTS) to demonstrate the benefit of joint modeling.

[1] "FVD: A new metric for video generation," ICLRW, 2019  
[2] "Learning transferable visual models from natural language supervision," ICML, 2021  
[3] "VBench: Comprehensive Benchmark Suite for Video Generative Models," CVPR 2024  
[4] "Imagebind: One embedding space to bind them all," CVPR, 2023  
[5] "Diverse and aligned audio-to-video generation via text-to-video model adaptation," AAAI, 2024  
[6] "Synchformer: Efficient synchronization from sparse cues," ICASSP, 2024

### Questions
See weaknesses above, particularly regarding the novelty and experimental setup.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OVI, a method for generating audio-visual content through a single feedforward pass, which avoids the need for post-hoc processing or a multi-step pipeline. The methodology involves two main stages. First, an audio tower is trained from scratch on a large-scale audio-visual (AV) dataset to ensure high-quality audio generation. Second, a twin-tower architecture, which utilizes blockwise bidirectional fusion and scaled Rotary Position Embeddings (RoPE), is trained to achieve strong synchronization and cross-modal alignment.

### Strengths
1. The paper is well-structured, and the methodology is presented with clarity, making it straightforward to understand.

2. The paper's contributions encompass multiple aspects of the problem, including data, architectural design, and training strategies.

### Weaknesses
1. The experimental evaluation appears to be missing key comparisons. The authors highlight cross-modal alignment as a primary strength of OVI; however, the empirical evaluation is predominantly focused on audio quality. The assessment of audio-video synchronization is not sufficiently demonstrated.

### Questions
1. To better substantiate the claims regarding audio-video synchronization, we suggest the authors benchmark OVI against relevant video-to-audio models (e.g., Diff-Foley [1], FoleyCrafter [2]). This would provide a more direct and comprehensive evaluation of the model's synchronization capabilities in comparison to other state-of-the-art approaches.

2. The large-scale video-audio dataset is a significant contribution and is crucial for enabling the reproducibility of the reported results. Could the authors clarify whether they plan to release the dataset or the associated data processing pipeline?

### Soundness
3

### Presentation
3

### Contribution
3
