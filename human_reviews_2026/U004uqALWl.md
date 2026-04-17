# SpeakerVid-5M: A Large-Scale High-Quality Dataset for Audio-Visual Dyadic Interactive Human Generation

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
The rapid development of large-scale models has catalyzed significant breakthroughs in the digital human domain. These advanced methodologies offer high-fidelity solutions for avatar driving and rendering, leading academia to focus on the next major challenge: audio-visual dyadic interactive virtual human. To facilitate research in this emerging area, we present SpeakerVid-5M dataset, the first large-scale, high-quality dataset designed for audio-visual dyadic interactive virtual human generation. Totaling over $8,743$ hours, SpeakerVid-5M contains more than $5.2$ million video clips of human portraits. It covers diverse scales and interaction types, including monadic talking, listening, and dyadic conversations. Crucially, the dataset is structured along two key dimensions: interaction type and data quality. First, it is categorized into four types (dialogue branch, single branch, listening branch and multi-turn branch) based on the interaction scenario. Second, it is stratified into a large-scale pre-training subset and a curated, high-quality subset for Supervised Fine-Tuning (SFT). This dual structure accommodates a wide array of 2D virtual human tasks. In addition, we provide an autoregressive (AR)-based video chat baseline trained on this data, accompanied by a dedicated set of metrics and test data to serve as a benchmark (VidChatBench) for future work. Both the dataset and the corresponding data processing code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents SpeakerVid-5M, a large-scale audio-visual speaker understanding dataset containing 5 million labeled video clips paired with aligned face and speech data. The authors aim to enable multi-modal representation learning for speaker identification, speech-to-face generation, and lip-synchronization tasks. The dataset is collected from online sources (e.g., YouTube) using an automated pipeline that detects talking faces, aligns speech segments, and filters content for language and quality.

### Strengths
1. SpeakerVid-5M is one of the largest open-source datasets in the speaker understanding domain, with diverse speakers, recording conditions, and languages, making it valuable for pretraining multimodal encoders.

2. By integrating speaker verification, lip-reading, and voice–face retrieval within one dataset, it provides a practical testbed for representation transfer and multi-task learning.

3. The use of automatic speech-face alignment and quality filtering (lip-sync thresholding, blur detection) demonstrates thoughtful engineering and scalability.

### Weaknesses
1. The core contribution is the dataset’s size, not a new method, model, or theoretical insight. It extends existing pipelines rather than rethinking them.

2. There is little quantitative characterization of dataset diversity, no breakdown by gender, age, language, recording environment, or cultural representation.

3. Large-scale scraping of online videos raises questions about consent, copyright, and potential bias amplification. The ethical section is superficial and lacks compliance discussion (e.g., GDPR, US fair-use).

### Questions
1. Beyond scale, what distinguishes SpeakerVid-5M from VoxCeleb2 + LRS3? Are there new annotation dimensions (e.g., speaker emotion, interaction type, recording context) that justify the “5M” contribution?

2. Have authors quantified the rate of misaligned or noisy samples? What proportion of clips have accurate lip audio synchronization?

3. Could you report statistics on gender, age, ethnicity, and geographic diversity? How do you mitigate bias from the over-representation of Western/English speakers?

4. Are the improvements in Table 4 solely due to dataset size, or did you modify architecture or hyperparameters? A same-architecture ablation on VoxCeleb vs. SpeakerVid would help isolate the effect.

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
This paper introduces a new dataset and a new benchmark, accompanied by a baseline model.
The dataset, SpeakerVid-5M, is a large-scale, high-quality collection specifically designed for audio-visual dyadic interactive virtual human generation. It contains over 8K hours of human portrait video clips.
The dataset is richly annotated with multi-modal information, including structured textual captions, human skeletal sequences (DWpose), ASR transcripts, and blur scores. It is further organized into four interaction types (dialogue, single, listening, and multi-turn branches) and divided into two tiers: a large-scale pre-training subset and a high-quality Supervised Fine-Tuning (SFT) subset.
The authors also propose an autoregressive (AR)-based video chat baseline model trained on this dataset and introduce VidChatBench, a dedicated benchmark with tailored metrics for evaluating performance on the complex task of audio-visual dyadic generation, which requires both multi-modal understanding and generation capabilities.
The dataset and processing code are planned to be publicly released.

### Strengths
1. The paper addresses a critical and timely problem. The research community's focus is clearly shifting from passive, audio-driven "talking heads" to proactive, interactive digital humans. The authors correctly identify that the single greatest barrier to open-source academic research in this area is the lack of a large-scale, high-quality dataset specifically capturing dyadic audio-visual interactions. This contribution directly unblocks this important future direction.
2. The dataset is a significant contribution, and its value is strongly reinforced by the rigorous and transparent data curation pipeline. The authors provide excellent clarity (particularly in Figure 2) on each methodical step, which builds trust in the data's fidelity. The use of a modern, multi-model suite for filtering and annotation is a major strength. For example, employing DOVER for video quality assessment, SyncNet to ensure tight audio-visual synchronization, and a powerful VLM like Qwen2.5-VL to generate rich, structured annotations (like motion, expression, and pose) goes far beyond simple ASR transcripts and ensures the data is high-quality and multi-faceted.
3. The proposed model architecture makes sense to me. The baseline is built on Qwen2.5-Omni. The 'thinker' to understand the initiator's multimodal input, and a joint Audio-Visual Generator that acts as a 'renderer' to produce the audio and video response. This end-to-end approach is a modern paradigm, and providing this baseline (along with the VidChatBench) gives the community a strong starting point and clearly demonstrates the dataset's immediate utility.

### Weaknesses
1. Limited Data Domain and Generality: The data sources, while high-quality, are heavily skewed towards formal or semi-formal scenarios (interviews, news, seminars, debates). The dataset appears to lack more casual, "in-the-wild" interaction styles, such as personal vlogs, movie/TV drama scenes, or general user-generated content. This "domain bias" might limit the ability of models trained on this data to generalize to the full spectrum of human interaction needed for a truly "general purpose" digital human.

2. Ambiguous Baseline Model Architecture: The description of the autoregressive baseline in Section 5 is too high-level, leaving several critical implementation details unclear. This makes the model difficult to reproduce or fairly assess.

"Diffusion MLP": This component is a black box. What is its architecture? Is it a simple MLP, or a more complex structure like a DiT? How are the time-step and visual token conditions precisely integrated?

"AR Generation Loop": This is the most confusing part. The model is autoregressive, but it also uses a diffusion process to "refine" latents. For the next generation step, what exactly is fed back into the AR model? Is it the initial, raw tokens from the AR generator, or the refined latents after the diffusion/denoising step? .

"Next-chunk prediction": What is the temporal size of a "chunk"? Are the visual tokens continuous or discrete token?

3. Missing SOTA Modular Baseline Comparison: The paper rightly claims no single end-to-end model exists for this task, but it fails to compare against the most obvious and powerful SOTA alternative: a modular pipeline. A proper evaluation would benchmark the proposed model against a baseline that 1) uses an LLM/VLM to generate a text response, 2) uses a TTS model for speech, and 3) uses a state-of-the-art speech-to-video model (like Hallo3, etc.) to synthesize the video. Without this comparison, it's impossible to know if the proposed end-to-end method is actually superior to (or even competitive with) what's already possible by combining existing SOTA components.

### Questions
Modular Baseline Comparison: Could the authors provide results for a strong modular baseline on VidChatBench?  See W3

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SpeakerVid-5M îs a large-scale dyadic talking humans dataset. The dataset contains automatically extracted annotations of 2D kpts, audio, audio-to-text, and scene descriptions. The dataset is accompanied by an extensive video benchmark.

### Strengths
Dyadic human videos are an important mode of human-centric video generation. Having a large dataset with paired ASR, Audio and video is highly valuable. The paper is easy to follow and the authors provide an extensive ethics statement. VidChatBench is a reasonable benchmark suite for the proposed dataset.

### Weaknesses
It seems that the dataset only contains pairs of videos - initiator -> respond. However, to future-prove this, I wonder if the authors could make available extended back-and-forth sequences of their data as well, i.e. where the initiator and responder engage in a back and forth way.

The authors claim that the video resolution is 1080P - however, their sample videos in the supplementary material are crops which are significantly smaller than 1080P. Could the authors clarify if they will release the full resolution? 

I find Figure 1 a bit unclear: what are the multi-modal annotations? I think the figure should make more clear in what the model inputs and outputs are for the generation process. 

It seems some of the example videos contain jump cuts at the end (Body Composition/full_body/3.mp4 ) - I wonder how many videos contain those “”incorrect cuts - could the authors comment if they are planning to post-process the dataset to identify those instances or if they believe those to not be a problem?


Suggested additional citations: TalkCuts [1].


[1] A Large-Scale Dataset for Multi-Shot Human Speech Video Generation; NeurIPS DBT 2025

### Questions
Are the person ids per video or are they consistent across multiple videos, i.e. same speaker, different day?

### Soundness
3

### Presentation
3

### Contribution
3
