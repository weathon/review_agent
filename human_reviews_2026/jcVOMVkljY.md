# Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
This study focuses on a challenging yet promising task, Text-to-Sounding-Video (T2SV) generation, which aims to generate a video with synchronized audio from text conditions, meanwhile ensuring both modalities are aligned with text. 
Despite progress in joint audio-video training, two critical challenges still remain unaddressed: (1) a single, shared text caption where the text for video is equal to the text for audio $(T_V = T_A)$ often creates modal interference, confusing the pretrained backbones, and (2) the optimal mechanism for cross-modal feature interaction remains unclear.
To address these challenges, we first propose the Hierarchical Visual-Grounded Captioning (HVGC) framework that generates pairs of disentangled captions, a video caption ($T_V$), and an audio caption ($T_A$), eliminating interference at the conditioning stage.
Based on HVGC, we further introduce BridgeDiT, a novel dual-tower diffusion transformer, which employs a Dual CrossAttention (DCA) mechanism that acts as a robust ``bridge" to enable a symmetric, bidirectional exchange of information, achieving both semantic and temporal synchronization.
Extensive experiments on three benchmark datasets, supported by human evaluations, demonstrate that our method achieves state-of-the-art results on most metrics. Comprehensive ablation studies further validate the effectiveness of our contributions, offering key insights for the future T2SV task. All the codes and checkpoints will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of text-to-sounding video (videos with synchronized audio) generation. The proposed method initializes from a pretrained text-to-video network and a text-to-audio network. The main contributions are twofold:

1. A multistage, hierarchical prompt rewriting approach to generate video caption and audio caption separately instead of sharing one caption for both video and audio generation.
2. A “BridgeDiT” block that connects pretrained video and audio towers with attention layers.

### Strengths
This paper proposes a reasonable approach for text-to-sounding video generation. It is fairly simple, yet achieves strong performance compared with state-of-the-art models. The motivation for separating video and audio captions when pre-trained models are used makes sense.

### Weaknesses
My main concern is that the contributions are weak. 

- Prompt rewriting is not new (has been used in Wan, for example). The proposed method uses multiple stages to rewrite the prompt, which brings additional complexity and is not carefully ablated. Table 3 compares the performance of a few captioning settings but does not ablate on the three stages. For instance, a single-stage method can also generate two types of captions for the video and the audio stream separately with in-context learning, which the proposed method has also used. I agree that using separate captions for video and audio makes sense when the network is initialized from pre-trained models, but this has a limited scope (the situation will be very different with a fully and jointly trained audio-video generation network) and limited contribution.
- If I am understanding correctly, the proposed “Dual Attention Fusion” module is a joint attention (“full-attention fusion” in the paper) with a masked block diagonal in the attention matrix. I don’t see why this is better than full attention, and I don’t see the justification for it in the paper. Can the authors elaborate? Figure 4 provides some training curves for these attention variants, but the curves don’t seem to have fully converged.

Moreover, I have some concerns about the experimental setup.

- Fairness of comparisons. Since the proposed method and some of the baselines (e.g., SSVG) bootstrap from pre-trained networks, the quality of these pre-trained networks plays a large role. This is evidenced by the results in Table 1 – the results (e.g., FVD and FAD) of the decoupled generation baselines are already better than the joint generation baselines. The proposed method uses these better pre-trained networks as backbones and compares with the prior works that use weaker pre-trained backbones. How do the authors ensure that the comparison is fair? 
- Data leakage. From Section 4.1, it seems like the proposed method has been trained and evaluated on AVSync15, VGGSound-SS, and Landscape (though this is not clear; see question). It is known that AVSync15 and VGGSound have cross-contamination in the train/test split (MMAudio Section E). Have the authors considered this?

Wan: Open and Advanced Large-Scale Video Generative Models

### Questions
- Could the authors clarify the composition of the training set?
- Could this paper be compared with CoDi-2 on text-to-sounding video generation?
- Other questions in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new framework for generating captions for sounding videos and proposes a new model architecture for sounding video generation. The captioning framework provides modality-specific captions for both audio and video. It first creates a detailed video caption from a given video clip using VLLM. Then, another LLM extracts sounding events from the video caption and creates a corresponding audio caption. These captions are used to train a new model called BridgeDiT, which is constructed by combining two pretrained DiT models for audio and video through dual cross-attention fusion. The experimental results show that the proposed model outperforms several baselines, including independent generation methods, sequential generation methods, and joint generation methods.

### Strengths
- This work addresses a challenging and important problem: the efficient construction of sounding video generation models. While numerous video and audio generation models have been released, models capable of jointly generating audio and video remain limited. Efficiently constructing such models is a hot topic in this field.
- The design of the proposed model is simple and applicable to any DiT-based models, which are a standard choice for recent audio and video generation tasks.
- The idea of creating dedicated text prompts for audio and video seems generally reasonable, though there are concerns about how to create them, which I will mention in the Weaknesses section.

### Weaknesses
- The empirical findings obtained through this work are somewhat limited, and it is not clear if they are sufficiently general.
  - Since the proposed architecture is widely applicable to DiT-based models, it is highly recommended to examine it with various backbones other than Wan + Stable Audio Open.
  - Several empirical findings are interesting; for example, a few BridgeDiT blocks are enough to construct sounding video generation models, and dual cross-attention is more effective than full-attention or addition. However, it is not clear if these results depend on the choice of backbone models.
- It would be helpful if the authors could show how to ensure that the generated audio caption accurately describes the content in the soundtrack of the source sounding video.
  - Since the proposed captioning framework solely relies on visual content in a sounding video, the generated audio caption is essentially limited to on-screen audible events. This means that any off-screen sounds could degrade the quality of the audio caption, making the training data noisier.
  - The datasets used in the experiments are all carefully filtered, so I think this may not be a significant issue in the experiments. However, it would be highly desirable in practice to have a validation stage in the framework to ensure that the generated audio caption aligns with the actual audio of the sounding video.
- Regarding the proposed captioning framework, it is not clearly described how spatial information is handled.
  - Since this study uses Stable Audio Open as a backbone for audio generation, the proposed model can generate stereo audio.
  - Therefore, spatial information about audible objects in a scene is important for the corresponding caption to properly describe the audio content.
  - However, this aspect is not mentioned in the instruction prompts shown in the appendix.
- The demo page should include results from baselines for qualitative comparison.

### Questions
- Did the authors examine the proposed architecture with other backbones?
- How can we ensure that the generated audio caption accurately describes the content in the soundtrack of the source sounding video?
- How is spatial information handled in the proposed captioning framework?

### Soundness
1

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
This paper proposes a Text-to-Sounding-Video (T2SV) generation framework that integrates two key components: Hierarchical Visual-Grounded Captioning (HVGC) for generating modality-specific captions for both video and audio, and Dual CrossAttention (DCA) fusion mechanism for optimal audiovisual interaction.
Experimental results show that the proposed method achieves consistent improvements over existing baselines, including sequential generation (T -> V -> A or T -> A -> V) and joint generation models.

### Strengths
1. The proposed model is conceptually simple yet effective. DCA modules are inserted into only four layers, which likely achieves strong audiovisual alignment with modest computational cost.
2. HVGC addresses a critical but underexplored issue: how to construct suitable captions for each modality in sounding video generation.
3. Experiments on multiple benchmarks demonstrate clear and consistent gains. The paper includes well-chosen baselines, including sequential and joint generation pipelines, providing fair and meaningful comparisons.

### Weaknesses
1. The experimental setup lacks clarity. It is unclear whether models were trained separately for each dataset or jointly across all datasets. More importantly, it is unclear whether baseline models were trained with the same HVGC captions or standard captions, which directly affects the fairness of the comparison.
2. The analysis of fusion mechanisms is limited to audiovisual alignment. Although the reviewer understands that audiovisual alignment is the most important metric for DCA, video and audio quality, and text alignment are also crucial. Further comparison of DCA with alternative fusion schemes would strengthen the analysis.
3. The user study involves only five participants, which limits the statistical significance of the subjective evaluation.

### Questions
1. Did the authors train separate models for each dataset or a single unified model?
2. Were baseline models trained or fine-tuned on the same dataset using HVGC captions?
3. Have the authors evaluated DCA against other fusion mechanisms beyond audiovisual alignment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This papers explores the text-to-sounding-video generation task, in the context of jointly fine-tuning a pair of pretrained text-to-audio and  text-to-video generation models, with the following contributions:
- Use different text prompts for each tower as opposed to a single prompt in previous works.
- A Hierarchical Visual-Grounded Captioning (HVGC) framework meant to generate pairs of disentangled captions: one for video with emphasis on the visual aspect, one for audio with an emphasis on sound.
- Incorporate a dual cross attention mechanism in the joint model architecture for modality interaction, named BridgeDiT.

### Strengths
I believe that despite its apparent simplicity this paper is significant for the joint audio-video generation task.

Using different prompts for audio and video is clever and dual cross attention is a natural intermodal network design. This paper conducts rigorous ablations on both aspects to demonstrate their benefit for the task.

Moreover the HVGC builds on the intuition that audio captioning is more accurate when being in context of the visual scene.

### Weaknesses
The major weakness of the paper is the lack of novelty.
- Using different prompts for audio and video is clever but common sense.
- The visually grounded audio captioning tasks lack a related works section. 
- Dual cross-attention architectures have been employed in the past for other multimodal tasks, yet this paper cites almost none of them (e.g. `Generative Spoken Dialogue Language Modeling` by Nguyen et al., `Dual-Stream Diffusion Net for Text-to-Video Generation` by Liu et al., `Domain Adaptation via Bidirectional Cross-Attention Transformer` by Wang et al., `Multimodal Transformer for Unaligned Multimodal Language Sequences`, `ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks`, `Pano-AVQA: Grounded Audio-Visual Question Answering on 360◦ Videos`). Moreover the two years old MM-diffusion also employs dual cross attention for the same task.

### Questions
- What does the 1000 value correspond to between equations 5 and 6?
- It is common to employ temporally-aware positional encodings when interacting between video and audio modalities (that usually operate at different frame rates). Does the proposed model employ such design?

### Soundness
3

### Presentation
3

### Contribution
3
