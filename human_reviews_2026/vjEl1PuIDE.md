# AVoCaDO: An Audiovisual Video Captioner Driven by Temporal Orchestration

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Audiovisual video captioning aims to generate semantically rich descriptions with temporal alignment between visual and auditory events, thereby benefiting both video understanding and generation. In this paper, we present **AVoCaDO**, a powerful audiovisual video captioner driven by the temporal orchestration between audio and visual modalities. We propose a two-stage post-training pipeline: (1) **AVoCaDO SFT**, which fine-tunes the model on a newly curated dataset of 107K high-quality, temporally-aligned audiovisual captions; and (2) **AVoCaDO GRPO**, which leverages tailored reward functions to further enhance temporal coherence and dialogue accuracy while regularizing caption length and reducing collapse. Experimental results demonstrate that AVoCaDO significantly outperforms existing open-source models across four audiovisual video captioning benchmarks, and also achieves competitive performance on the VDC benchmark under visual-only settings. The model will be made publicly available to facilitate future research in audiovisual video understanding and generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AVoCaDo, an audio-visual video captioner driven by the temporal orchestration between audio and visual modalities. AVoCaDo is built upon the Qwen2.5-Omni, which is enhanced by a two-stage post-training: 107K temporally-aligned audio-visual captions are used for SFT, followed by a reinforcement learning using three reward-based GRPO. Experiments verify the effectiveness of the proposed method.

### Strengths
- This paper involves a well-structured two-stage post-training pipeline including both SFT and RL.
- The proposed 107K high-quality audiovisual caption dataset used in the SFT can be a good resource for relevant topics.
- The paper is well-written, easy to follow. The proposed method is clearly described.

### Weaknesses
- The ablations lack evaluation on the dataset composition. Tab. 4 reports ablations only for reward terms, not for dataset subsets (TikTok-10M vs. FineVideo etc.). It is unclear whether gains stem from data diversity or model tuning.

- During the RL stage, the paper briefly mentions that 2K caption data is randomly sampled and  8 rollouts are generated for GRPO calculation. It would be better to provide ablation studies on the number of video samples and rollouts used in the RL stage.  Moreover, the visualization of the different reward types during training would also be helpful for readers. 
- The literature review on audiovisual captioning could be more comprehensive, such as Fine-grained Audible Video Description (CVPR 2023), LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos (CVPR 2025), etc.
- I appreciate the simple, elegant, and clearly written paper. However, since there are many research papers using SFT and GRPO, I am afraid that the paper does not provide sufficiently new technical insights to the community.

### Questions
- What is the total training time for the SFT and RL? The calculation of the rewards also requires some time.
- How to define the five key points in the checklist for computing Reward_c? Will there be other caption cases beyond the predefined five points?

### Soundness
3

### Presentation
4

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
This paper presents **AVoCaDO**, a model designed to enhance audiovisual video understanding by jointly learning from audio and visual modalities. The authors argue that most existing video captioning methods are overly vision-centric and fail to capture temporal alignment between auditory and visual events. To address this, AVoCaDO introduces a two-stage post-training framework (SFT, GRPO).
Experimental results demonstrate that AVoCaDO achieves state-of-the-art performance among open-source models on audiovisual captioning, QA, and visual-only benchmarks.

### Strengths
The paper is well written and easy to follow. The core idea is clearly explained, with strong logical flow from the motivation to the experimental results. Each component of the proposed approach is described with sufficient clarity, and the reasoning behind design choices is well supported by empirical evidence. The experimental results are clearly presented and convincingly demonstrate the effectiveness of the proposed method across multiple benchmarks.

### Weaknesses
- **Limited transparency of data construction.**
Although the paper claims to curate a 107K high-quality audiovisual dataset, the process relies heavily on closed-source models (Gemini-2.5-Pro, GPT-4.1). This limits the reproducibility and transparency of the dataset and may raise concerns about data bias or accessibility.
- **Dependence on proprietary judge models.**
The evaluation heavily depends on Gemini-2.5-Pro as a judging model, which may introduce bias or instability in QA-based evaluation. A human or open-source baseline judge would make the comparison more robust.
- **Lack of technical contribution in model design.**
While AVoCaDO is built upon Qwen2.5-Omni, the paper does not clearly describe how the audio and video inputs are encoded, temporally synchronized, and fused within the model architecture. As a result, the contribution leans more toward fine-tuning and reward engineering rather than introducing a technically novel modeling component. A clearer explanation of the underlying multimodal representation mechanism would strengthen the technical contribution of the work.

### Questions
Could the authors clarify whether the curated 107K audiovisual caption dataset (SFT and GRPO versions) will be released publicly or partially shared to support reproducibility?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes an audiovisual captioner built on Qwen2.5-Omni with a two-stage post-training pipeline: SFT on 107K curated, temporally aligned AV captions created via a two-stage prompting scheme, and RLHF using checklist rewards combined with a dialogue reward and a length regularizer to curb repetition and excessive verbosity. The paper shows strong results on video-SALMONN-2 testset, UGC-VideoCap, and caption-driven QA transforms of Daily-Omni and WorldSense, plus competitive performance on VDC-Detailed (visual-only). Ablations studies show the effectiveness of proposed rewards and the value SFT data to the model.

### Strengths
1. The main novelty comes from the usage of checklist rewards and dialogue-based reward proposed in RLHF stage, and it proves very effective according to the ablation results (shown in Table 7).

2. The value of the constructed dataset is proven to be very effective across both video captioning and joint audio video QA benchmarks.

3. The paper is very well-written with clear motivation and clear analysis of the results both quantitatively and qualitatively.

### Weaknesses
1. The paper only showcases its strong capability in video-speech related scenarios, but audio could be from also other categories, such as music / general sound. It would be great to incorporate some experiments covering these aspects as well, which will make the contribution even stronger. Some related works in this area for references such as [1]-[3].

2. How does the method perform when the sound / speech is very noisy scenario, does the visual help answering these questions? How is the captioning like in audio-only mode in these benchmarks? It would be more comprehensive if the authors incorprate these results to see the value of vision part only.

3. How does the proposed method generalize to long context audio-visual video captioning and how does different sampling strategies in audio / video affect the final performance.

4. Can such framework work in real-time mode and what are current limitations / trade-offs between latency and accuracy? Maybe some dicussions or future works could be mentioned to showcase the challenges / limitations of existing framework.

[1] Learning to Answer Questions in Dynamic Audio-Visual Scenarios, CVPR 2023.

[2] Tackling data bias in MUSIC-AVQA: Crafting a balanced dataset for unbiased question-answering, WACV 2024.

[3] AVQA: A Dataset for Audio-Visual Question Answering on Videos, ACM MM 2022.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an audiovisual video captioning system that generates temporally-aligned descriptions. The approach employs a two-stage training pipeline: (1) AVoCaDO SFT fine-tunes Qwen2.5-Omni on 107K synthetic audiovisual captions, and (2) AVoCaDO GRPO applies reinforcement learning with three complementary rewards targeting temporal coherence, dialogue accuracy, and caption quality. The paper is well-motivated by a compelling pilot study and demonstrates strong empirical results across multiple benchmarks.

### Strengths
1. The pilot experiment (Figure 1) provides compelling evidence that joint audiovisual captioning with temporal alignment significantly improves understanding.
2. The topic itself is of high value. Temporally-aligned audiovisual captions are important for training next-generation video understanding and generation models. 
3. Well-designed reward functions for AV captioning.
4. The proposed method shows strong improvement over the base model

### Weaknesses
Comparing transcription accuracy against specialized ASR models (e.g., Whisper) would be useful to understand the quality of the model caption. Metrics like Word Error Rate (WER) on the dialogue portions could quantify transcription quality.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
