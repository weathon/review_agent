# AudioX: A Unified Framework for Anything-to-Audio Generation

- Decision: Accept (Poster)
- Scores: 8, 4, 10, 6

## Abstract
Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, and 2) large-scale, high-quality training data. As such, we propose AudioX, a unified framework for anything-to-audio generation that integrates varied multimodal conditions (i.e., text, video, image, and audio signals) in this work. The core design in this framework is a Multimodal Adaptive Fusion module, which enables the effective fusion of diverse multimodal inputs, enhancing cross-modal alignment and improving overall generation quality. To train this unified model, we construct a large-scale, high-quality dataset, IF-caps, comprising over 7 million samples curated through a structured data annotation pipeline. This dataset provides comprehensive supervision for multimodal-conditioned audio generation. We benchmark AudioX against state-of-the-art methods across a wide range of tasks, finding that our model achieves superior performance, especially in text-to-audio and text-to-music generation. These results demonstrate our method is capable of audio generation under multimodal control signals, showing powerful instruction-following potential. We will release the code, model, and dataset.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a unified DiT model for anything-to-audio generation conditioned on text / video / image / audio. The model adds a lightweight Multimodal Adaptive Fusion (MAF), per-modality gates and learnable query sets with cross/self-attention, to calibrate condition embeddings before a DiT generator. To train the model, the authors proposed a large-scale dataset, IF-caps, sourced from existing joint audio-video dataset, by using frontier audio-video LLMs for providing detailed captions for audio / music.

### Strengths
1. Unified model architecture: A single model handling text/video/audio conditions for both sound effects and music is non-trivial, and it seems like the results are promising.

2. Detailed evaluations and benchmarking: Evaluations cover many tasks (T2A/V2A/TV2A; T2M/V2M/TV2M; inpainting; completion; image->audio zero-shot), and the results look solid.

3. Novel evaluation prototype: beyond fidelity and diversity (FD / KL metrics), they propose to evaluate more fine-grained aspects,  temporal/count/order control (incl. AudioTime) and adds a new benchmark (T2A-bench).

4. Detailed user studies: the supplementary material provides detailed user studies and qualitative results over different aspects of the proposed method, covering a wide variety of tasks.

### Weaknesses
1. Lack of reference / comparisons against some important prior works in V2A / V2M, including but not limited to [1] - [5]. Except [3], all the rest are either open-sourced or providing samples for evaluation comparison. And also, [4] provides a new benchmark, which I think should be considered to evaluate.

2. Lack of important metrics regarding temporal alignment evaluation. The main table (Table 1) should also incorporate metrics that address temporal alignment, such as ones "Align Acc" proposed in [6] and "AlignSync" proposed in [7].




[1]. Tell what you hear from what you see-video to audio generation through text, NeurIPS 2024, Xiulong Liu, et.al.

[2]. From vision to audio and beyond: A unified model for audio-visual representation and generation. Kun Su, et.al.

[3]. V2meow: Meowing to the visual beat via video-to-music generation, AAAI 2024, Kun Su, et.al.

[4]. Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation.

[5]. ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing, NeurIPS 2025, Huadai Liu, et.al.

[6]. Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models, NeurIPS 2023, Simian Luo, et.al.

[7]. Audio-Synchronized Visual Animation, ECCV 2024, Shentong Mo, et.al.

### Questions
See weaknesses section.

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
4

### Summary
The paper proposes a unified any-to-audio framework based on diffusion transformers. The main contributions of the paper are: 
1) large-scale dataset. 2) a unified framework for any-to-audio and 3) a proposed benchmark for instruction-followed in audio generation. The dataset is constructed by annotating the audio from the video dataset, while the proposed benchmark is constructed to test the instruction following of the model under controlled settings, and the model is proposed to use a Multimodal adaptive fusion module (MAF), which processes the input conditions before feeding it to the DiT.

### Strengths
- The paper is well written and easy to follow 
- The model weights and proposed dataset and benchmark are promised to be released, which will make a good contribution to the field 
- The direction of having a unified framework that combines multiple conditions is important.
- Extensive experiments are provided on multiple tasks. 
- The proposed benchmark is solid.

### Weaknesses
Dataset contribution:
- The construction pipeline of the dataset is not convincing: how are the videos selected? The authors just mentioned using "existing video datasets". Were the audio-video alignment considered in the selected samples? Also, the authors mentioned that they used Gemini 2.5 to provide general captions, then Qwen2-Audio for augmentation in order to save compute. I am not convinced why Gemini 2.5 cannot provide the complete augmented instruction. The two-stage process is not well-justiied. 
- The dataset quality is not well-tested. I could not find any analysis on the text-audio or audio-video alignment or the dataset in general in the paper or the supplement. Without this, it is hard to understand the dataset contributions or their comparison to existing datasets beyond scale (e.g AVSync15, AutoRecap, or even VGGSounds)

Model contribution:
- While the motivation of having an any-to-audio generation framework is interesting, it is not well justified in the paper experiments. Main experiments support text, audio, and/or video conditioning, which is already supported by existing frameworks (e.g MMAudio), except for the audio in-painting, making the framework not appear as truly any-to-audio framework and just another baseline for T2A, V2A. 
- The architectural contribution is limited to the MAF module to process multiple conditions. However, it is not discussed well. For example, how is the positional encoding handled in this module? How is the temporal alignment between different conditions handled?  How does it compare to existing conditioning mechanism? (e.g MMAudio style). 
- Other components of MAF are also not well justified. The gating is not intuitive and does not seem to bring significant improvement in Table 4

In general, while the paper presents significant experiments at a large scale, most of the author's choices are not well justified, and the relation to existing work (e.g, existing architecture and datasets) is not well discussed.

### Questions
- Is the video and audio passed to Gemini or just the audio? Figure 2 suggest only audio while L169 suggests both.
- Figure 4 is confusion. z_t is not mentioned in the figure suggesting that the combination of the conditions is fed as input for DiT rather than conditioning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper proposes AudioX, a unified "anything-to-audio" generation framework to resolve the lack of unified frameworks for handling diverse input modalities generation.  It also provides the  IF-caps dataset to resolve the scarcity of large-scale, high-quality multimodal training data. Experiments show that AudioX enables high-fidelity audio/music generation from flexible inputs while enhancing instruction-following capabilities.

### Strengths
1. Audiox present a unified framework to support audio/music generation from text, video, image, and audio inputs in a single model, breaking modality/domain constraints of specialist models.
2. The authors build a large-scale Dataset, IF-caps, provides structured, fine-grained multimodal supervision, solving data scarcity for unified training.
3. The authors plans to release code, model, and IF-caps to support reproducibility

### Weaknesses
1. The paper’s experiments and dataset design are focused on short audio clips (mostly 10 seconds), with no discussion or evaluation of long-form audio generation (e.g., clips longer than 30 seconds. For long-form generation, there are many aspects, including temporal coherence, memory and computational efficiency, error acccumulation should be evaluated. Without them, the paper cannot fully demonstrate AudioX’s practical value for real-world applications that require extended audio.
2. The experiment only compares 4 methods (Caption2Audio, Im2Wav, Seeing&Hearing, AudioX) on a single set of metrics (KL, IS, FD, FAD, Align.), using "the same settings as in (Xing et al., 2024)" () but without specifying:
3. The paper does not provide any strategies to reduce computational cost to make it reproducible for small labs.

### Questions
1. Whether the images include "rare modalities" (e.g., underwater scenes, ancient artifacts, abstract art) that are less frequent in training data.
2. Whether the image-to-audio generation relies on human-written prompts, and whether rare objects/scenes were tested.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents AudioX, a unified framework for "anything-to-audio" generation, designed to produce high-quality audio (both sound effects and music) from a flexible combination of multimodal inputs, including text, video, and audio signals. The work makes four primary contributions:
- AudioX Framework: A unified model built on a Diffusion Transformer (DiT) backbone.
- MAF Module: A novel "Multimodal Adaptive Fusion" (MAF) module that adaptively fuses these diverse input modalities to reduce interference and improve conditioning.
- IF-caps Dataset: A new, large-scale (7 million sample) dataset for multimodal audio generation. This dataset is not manually labeled but created via a structured data annotation pipeline that uses a powerful MLLM (Gemini 2.5 Pro) for high-quality initial annotations, which are then augmented at scale by another model (Qwen2-Audio).
- T2A-bench: A new benchmark designed specifically to evaluate fine-grained, instruction-following capabilities in text-to-audio models, such as event counting and temporal ordering.

### Strengths
- Unified Generalist Model: The paper successfully presents a single generalist model, AudioX, that performs competitively across a very wide array of "anything-to-audio" tasks (T2A, T2M, V2A, V2M, inpainting, etc.).
- Superior Controllability: The model's key strength is its SOTA performance on instruction-following benchmarks (T2A-bench and AudioTime), demonstrating a new level of fine-grained control over audio content, such as event ordering and counting .
- Novel Data Pipeline: The IF-caps dataset and its generation pipeline are a significant contribution, providing a practical blueprint for overcoming data scarcity by using LLMs for high-quality annotation and augmentation.
- Strong Ablation Studies: The paper includes excellent ablation studies that validate both the architectural contribution (the MAF module, Table 4) and the data contribution (the IF-caps pipeline, Table 3).

### Weaknesses
- Dependency on Proprietary Models: The entire 7-million-sample IF-caps dataset is bootstrapped from annotations generated by Gemini 2.5 Pro. This makes the data curation process heavily reliant on a closed-source, proprietary, and expensive black-box model, raising significant concerns about reproducibility.
- Potential Evaluation Bias: The new T2A-bench is also evaluated using Gemini 2.5 Pro as the "automated judge". While the paper describes a thoughtful two-step "blind annotation" pipeline to mitigate bias , this is still a form of LLM-as-a-judge. Using the same model family to both generate the training data (via IF-caps) and judge the evaluation (T2A-bench) creates a potential for self-referential bias that is not fully addressed.
- Overclaim on "Image" Modality: The abstract and introduction repeatedly claim the framework is for "text, video, image, and audio". However, "image" is absent from the main model architecture (Fig. 4), training process (Sec 4.2), and main results tables (Table 1, 2). It only appears as a zero-shot task in the appendix (Table A.6). This is an overstatement of the model's core, trained capabilities.

### Questions
- The data generation (IF-caps) and the new benchmark (T2A-bench) both rely heavily on Gemini 2.5 Pro. Can the authors comment on the risk of this self-referential loop, where a model trained on Gemini-annotated data is then shown to be SOTA on a benchmark judged by the same Gemini model?
- The T2A-bench's automated evaluation pipeline relies on Gemini 2.5 Pro's ability to perform blind annotation (SED, counting, etc.) . How was the judge (Gemini) itself validated for accuracy on these tasks? If the judge is inaccurate, the benchmark scores in Table 2 are not reliable.
- Why is "image-to-audio" listed as a core input modality in the abstract when it is not part of the main architecture diagram and only appears as a zero-shot appendix task? Was the model ever trained on image-audio pairs?

### Soundness
3

### Presentation
3

### Contribution
3
