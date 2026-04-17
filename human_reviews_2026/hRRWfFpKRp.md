# JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent AIGC advances have rapidly expanded from text-to-image generation toward high-quality multimodal synthesis across video and audio. Within this context, joint audio-video generation (JAVG) has emerged as a fundamental task that produces synchronized and semantically aligned sound and vision from textual descriptions. However, compared with advanced commercial models such as Veo3, existing open-source methods still suffer from limitations in generation quality, temporal synchrony, and alignment with human preferences. To bridge the gap, this paper presents JavisDiT++, a concise yet powerful framework for efficient and effective JAVG. First, we introduce a modality-specific mixture-of-experts (MS-MoE) design that enables cross-modal interaction efficacy while enhancing single-modal generation quality. Then, we propose a temporal-aligned RoPE (TA-RoPE) strategy to achieve explicit, frame-level synchronization between audio and video tokens. Besides, we develop an audio-video direct preference optimization (AV-DPO) method to align model outputs with human preference across quality, consistency, and synchrony dimensions. Built upon Wan2.1-1.3B-T2V, our model achieves state-of-the-art performance merely with around 1M public training entries, significantly outperforming prior approaches in both qualitative and quantitative evaluations. Comprehensive ablation studies have been conducted to validate the effectiveness of our proposed modules. All the code, model, and dataset are released at https://JavisVerse.github.io/JavisDiT2-page.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a new framework for audio-video joint generation, which features three new techniques. First, a modality-specific mixture-of-experts (MS-MoE) design is adopted, which leads to both cross-modal alignment and single-modality generation quality. Second, a temporal-aligned RoPE (TA-RoPE) strategy is introduced for facilitating frame-level synchronization between audio and video tokens. Third, an audio-video direct preference optimization (AV-DPO) method is applied to align model outputs with human preference across quality, consistency, and synchrony dimensions. Experimental results with a model built on Wan2.1-1.3B demonstrate that the proposed framework outperforms previous approaches both qualitatively and quantitatively. Ablation studies justify the design of the proposed modules.

### Strengths
1. Although there is room for improvement in presentation, the paper itself is written well enough to make readers understand the authors' motivation, the proposed method, and the experimental results.
2. The proposed framework is well-designed. The ablation studies are comprehensive and support the design.
3. The experimental results are good both quantitatively and qualitatively.

### Weaknesses
Although the proposed framework is well-designed and works better than the previous ones (JavisDiT and UniVerse-1), there is one concern/question. Newly introduced FFNs for audio tokens will modulate audio tokens so that the attention layers trained for video tokens can deal with audio tokens as well. This idea is great. However, as for the first Transformer block, audio tokens (the output of the audio encoder) will be fed into the self-attention and cross-attention layers as they are. It is suspected that the self-attention and cross-attention layers in the first Transformer block will not be able to deal with audio tokens very well. So, inserting another FFN between the audio encoder and the first Transformer block and training it in the pretraining phase might help the first self-attention and cross-attention layers process audio tokens.

### Questions
I have a question/concern, which I provided in "Weaknesses". I would appreciate it if the authors could share their thoughts on it. If the authors' response is convincing, I will raise my rating.

### Soundness
3

### Presentation
2

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
This paper presents a new method for efficiently constructing a model capable of jointly generating audio and video. Given a pretrained DiT-based video generation model, the proposed method first adds audio-specific FFNs to each DiT block, and these added FFNs are trained with text-audio pairs for text-to-audio generation. Then, the entire model is fine-tuned in a parameter-efficient manner using text-audio-video triplets for joint audio-video generation. Finally, the model is further fine-tuned by direct preference optimization with a dedicated audio-video preference dataset. Experimental results show that the model obtained through the proposed method achieves superior performance compared to existing models.

### Strengths
- The proposed method introduces a novel approach for efficiently constructing joint audio-video generation models. While many prior works use a separate audio branch in addition to a video branch, the proposed model only adds dedicated FFNs for audio.
- The proposed positional encoding is carefully designed to leverage pretrained video generation models while inducing strong audio-visual alignment via temporal encoding.
- This work explores direct preference optimization for joint audio-visual generation. The preference data is automatically constructed using several reward models. This approach is simple and reasonable, and could serve as a good baseline for future research.

### Weaknesses
- While the idea of applying DPO to joint generation models is interesting, its effect appears to be relatively small compared to the other proposed modifications. It would be beneficial to investigate the potential causes of this result. 
  - For example, if the quality of the preference data is a factor, this could be clarified by a subjective evaluation of a small subset of the preference data. If the amount of preference data is insufficient, showing performance as a function of the amount of preference data used would be helpful.
- Some visualizations could be further improved:
  - Showing spectrograms rather than waveforms would be more informative for illustrating the semantics of the audio content.
  - What does the vertical axis in Figure 7 represent?

### Questions
- Why are several numbers (especially DeSync in sequential generation baselines) missing from Table 1? These are necessary for a comprehensive comparison of joint generation performance.
- What models are used for T2A (or T2V) in the sequential generation baselines?

### Soundness
3

### Presentation
2

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
This paper proposes a joint audio-video generation (JAVG) model built on the pretrained text-to-video model Wan2.1.
To enable synchronized audio and video generation, three mechanisms are introduced: a model design called MS-MOE for joint generation, a new RoPE variant (TA-RoPE) that manages both cross-modal interaction and single-modal structure, and an audio-video direct preference optimization method (AV-DPO) that aligns the model with human preference. 
Experimental results demonstrate that the proposed method outperforms existing baselines.

### Strengths
- The proposed model design (MS-MOE with LoRA finetuning) is simple yet effective, achieving both a high cross-modal alignment and a high single-modal generation quality. It is also computationally efficient, as inference cost per token remains constant while the parameter size increases.
- The proposed TA-RoPE requires minimal engineering efforts. It provides a natural extension of Wan's RoPE to support both inter- and intra-modal interactions of audio and video.
- AV-DPO sounds novel, as applying DPO for aligning JAVG models with human preference is underexplored.

### Weaknesses
The primary concern lies in the **fairness and clarity of the experimental evaluation**.

For Table 1:
- It is unclear which T2A models are used for T2A+A2V and which T2V models are used for T2A+V2A. Also, many results of T2A+A2V and T2V+V2A baselines are missing. 
- The authors re-evaluate JavisDiT, but the reported scores deviate substantially from those in the original paper. For instance, TA-IB = 0.197 (original) vs. 0.151 (this paper), CLIP = 0.325 vs. 0.308, AV-IB = 0.201 vs. 0.197, AVHScore =  0.183 vs. 0.179, and JavisScore = 0.158 vs. 0.154. These metrics consistently decrease, and in some cases, the original scores are higher than those of the proposed method.
- Only the final model (finetuned using AV-DPO) is reported. It would also be important to show results after the Audio-Video SFT stage to isolate the contributions of the model architecture and AV-DPO optimization.

For Fig.2:
- All evaluation metrics appear identical to those used in AV-DPO training, relying on the same reward models. Since only the proposed model is optimized for these metrics, this evaluation is biased and not directly comparable to other baselines.

For Fig.7:
- The definitions of A-LoRA, A-noLoRA, AV-AttnLoRA, and AV-LoRA are missing, making the figure difficult to interpret. A more detailed explanation is needed.


**Lack of justification for the TA-RoPE design.**  
While the reviewer agrees that TA-RoPE is a reasonable engineering extension, the motivation behind the specific positional correspondence between audio and video modalities is not fully explained.
In particular, the mapping of video height and audio time, and video width and audio frequency, may introduce implicit correlations that lack perceptual grounding.
A brief visualization or analysis of the positional-similarity structure would help clarify how the TA-RoPE affects the interaction between audio and video modalities.


**Lack of subjective evaluation.**  
Although the paper claims that AV-DPO improves perceptual alignment with human preference, no user study is conducted to support this claim.
Including even a small-scale user study would make the evaluation more convincing.

### Questions
- Did the authors evaluate the model performance after the Audio-Video SFT stage, prior to AV-DPO finetuning?
- For TA-RoPE, could the authors visualize or analyze the similarity structure between RoPE embeddings used for audio and video?
- For the training data used in AV-DPO, how did the authors sort the videos based on three scores?
- Is AV-DPO applicable to other existing JAVG models?
- Did the authors conduct a user study to compare the proposed method with baselines?

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
This paper proposes a new method for joint audio-video generation. In contrast to most prior works that model the two modalities as two separate streams, this paper proposes to use a mixture-of-experts design to allow frequent communication and joint modeling of both modalities. This paper also proposes to be a temporally aligned rope encoding scheme and a DPO method for preference alignment. The proposed approach achieves state-of-the-art performance.

### Strengths
- The joint modeling of audio and video tokens with separate FFN layers is simple and elegant, although these ideas already exist in prior works (e.g., BAGEL) under different contexts.
- The presented model has a strong empirical performance, setting a new state-of-the-art for the challenging problem of joint audio-video generation.
- The study of different finetuning strategies (e.g., Lora with different ranks) is interesting. It provides additional context and insights to the readers.
- The ablation study on data composition in the appendix is also informative.

### Weaknesses
- The proposed network seems to be significantly larger than the networks used in prior works. Can the authors also include the parameter counts and runtime in Table 1 for a more comprehensive comparison? 
- The gains in Table 4 from using DPO seem to be very limited, except in FVD. How important is DPO? Are there any user studies or qualitative examples that support the use of DPO? 
- There are no details on training data filtering. Section D2 contains a high-level sketch, but it does not mention which algorithms or what thresholds were used.
- The last update date of the linked repo is after the ICLR deadline.

### Questions
- Will the code and model be open-sourced?
- Will the filtered training dataset be released? This would be a significant contribution.

### Soundness
3

### Presentation
3

### Contribution
2
