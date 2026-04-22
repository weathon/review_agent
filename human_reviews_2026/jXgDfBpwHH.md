# Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 4, 2

## Abstract
The goal of this work is to enhance balanced multimodal understanding in audio-visual large language models (AV-LLMs) by addressing modality bias without additional training. In current AV-LLMs, audio and video features are typically processed jointly in the decoder. While this strategy facilitates unified multimodal understanding, it may introduce modality bias, where the model tends to over-rely on one modality due to imbalanced training signals. To mitigate this, we propose Fork-Merge Decoding (FMD), a simple yet effective inference-time strategy that requires no additional training or architectural modifications. FMD first performs modality-specific reasoning by processing audio-only and video-only inputs through the early decoder layers (fork), and then merges the resulting hidden states for joint reasoning in the remaining layers (merge). This separation allows each modality to be emphasized in the early stages while encouraging balanced contributions during integration. We validate our method on three representative AV-LLMs—VideoLLaMA2, video-SALMONN, and Qwen2.5-Omni—using three benchmark datasets. Experimental results show consistent gains in audio, video, and audio-visual reasoning tasks, highlighting the effectiveness of inference-time interventions for robust and efficient multimodal understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the fork-merge decoding method to improve audio-visual LLM performance in a training free way. The process first masks audio or visual inputs separately in earlier TFM layers, and then merge them in later layers so that the model is not overlooking any modality when answering questions.

### Strengths
1. The method is training free and trying to solve a critical problem in audio-visual LLMs

2. The proposed method achieved superior performance on a range of benchmarks over 3 different audio-visual LLMs in the experiments. 

3. The paper is very well-written.

### Weaknesses
I mainly have the following two concerns regarding the experimental setup, and require the authors to provide more information:

(1). Including more difficult benchmarks. I would suggest at least adding results on Video-MME [1] and/or AVUT [2] to demonstrate the consistency of improvements and the balance between audio and visual modalities. No need to do for all 3 models but at least 1 model is needed.

(2). More recent models, e.g. video-SALMONN 2+ [3] which achieved the best performance on Video-MME would be very helpful to show that the state-of-the-art models still benefit from this method.

[1] Fu et al. "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis", https://arxiv.org/abs/2405.21075

[2] Yang et al. "Audio-centric Video Understanding Benchmark without Text Shortcut", https://arxiv.org/abs/2503.19951

[3] Tang et al. "video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models", https://arxiv.org/abs/2506.15220

### Questions
1. Have the authors tried soft masking?

2. How much additional computational cost does the merge decoding process introduce?

3. Any insights into how the model improves on hallucination induced by not having balanced modality attention? My experience is that audio-visual LLMs may make up things on a modality that they fail to attend to, so any insights?

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
This paper introduces Fork-Merge Decoding (FMD), a training-free inference strategy designed to mitigate modality bias in Audio-Visual Large Language Models (AV-LLMs). The core problem addressed is the tendency of AV-LLMs to over-rely on one modality (e.g., video) at the expense of another (e.g., audio), leading to imbalanced understanding and errors. The authors validate FMD on three representative AV-LLMs (VideoLLaMA2, video-SALMONN, Qwen2.5-Omni) across three challenging benchmarks (AVQA, MUSIC-AVQA, AVHBench). The results consistently show that FMD improves performance over vanilla decoding, particularly in tasks requiring balanced audio-visual reasoning. However, the performance improvement offered by this method is not particularly significant, and some of its design choices lack robust experimental validation.

### Strengths
- FMD is convenient to deploy; as a training-free, plug-and-play method, it is adaptable to a wide range of applications.
- The comprehensive ablation studies presented in the paper demonstrate the authors' thoroughness in validating their design.
- The proposed method is simple and intuitive, and the manuscript is of high writing quality.

### Weaknesses
Although the paper includes extensive ablation studies, I suggest that the following aspects require further investigation:
- I would like to clarify if the model employs Nucleus Sampling for decoding. If it does, given its stochastic nature, it is crucial to conduct multiple decoding runs and report the results with error bars. This analysis is essential because the performance improvement from FMD is currently marginal, which raises concerns about the stability and statistical significance of the method.
- For the model using CHANNEL-WISE FUSION, there is a lack of detailed experimental analysis regarding the audio and video weights during the merging stage. The majority of the analysis in the paper is based on VideoLLaMA-2, with insufficient corresponding analysis for the CHANNEL-WISE FUSION model.
- More powerful models should also be evaluated, such as the recently released video-SALMONN 2+, Qwen3-Omni.
- More benchmarks should be included, such as Video-MME, WorldSense, AVUT.

### Questions
I agree with the authors that the current choice of $\alpha$ and the fork layer is suboptimal. I would be interested to know if the authors have any thoughts on potential improvements or alternative approaches. This question will not affect my overall assessment of the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the modality bias problem in current audio-visual large language models (AV-LLMs) by proposing Fork-Merge Decoding (FMD)—a training-free, architecture-agnostic inference-time strategy. FMD separates early-stage unimodal reasoning (fork) from later-stage multimodal integration (merge). The method consistently improves performance across three representative AV-LLMs (VideoLLaMA2, video-SALMONN, and Qwen2.5-Omni) and three standard benchmarks, especially on tasks requiring balanced cross-modal reasoning. The authors further validate FMD’s compatibility with both token-wise and channel-wise fusion architectures and support its effectiveness through attention analysis, ablation studies, and qualitative examples.

### Strengths
1. Important and practical problem: Modality bias is a core challenge in modern multimodal LLMs; this work tackles a real-world issue with clear applicability.

2. Simple and efficient method: FMD requires no additional training or architectural changes—only input masking and hidden-state fusion during inference—making it a plug-and-play solution that is easy to deploy.

3. Comprehensive experiments: The evaluation covers diverse AV-LLM architectures and multiple benchmark datasets, demonstrating consistent gains and robustness.

### Weaknesses
1. Heuristic hyperparameter selection:
The fusion weight α is estimated as a fixed value from 100 samples on AVHBench (e.g., α = 0.8 for VideoLLaMA2). Although it generalizes well to other datasets, it lacks theoretical grounding or an adaptive mechanism. Manual calibration per model undermines true plug-and-play usability in real-world settings.
As shown in Figure 6, the optimal Lfork (fork layer depth) varies significantly across tasks (e.g., A→V prefers shallow layers, V→A prefers deeper ones), raising concerns about practical adaptability.

2. Inference speed overhead:
As shown in Table A.1, FMD increases latency by 1.26× compared to vanilla decoding. While modest, this overhead may limit deployment in latency-sensitive applications.

### Questions
1. On the generalizability of α:
How sensitive is performance to the choice of α? Specifically, if we use a suboptimal α (e.g., from a different model or dataset), how much does performance degrade? Could you quantify this drop?

2. On the meaningfulness of masked features:
In the fork phase, one modality (e.g., video tokens) is zeroed out. Yet the resulting hidden states still carry meaningful unimodal signals. Why is this the case? Does the language prompt alone provide enough context for the model to generate coherent intermediate representations—even when one sensory modality is completely masked?

3. On adaptive Lfork selection:
Figure 6 shows that A→V tasks benefit from shallow Lfork, while V→A tasks prefer deeper Lfork. Have you considered a dynamic Lfork selection mechanism—for example, based on the modality bias inferred from the input prompt or early attention patterns? Would such an adaptive strategy further improve performance?

4. On text modality bias and counterfactual inputs:
The paper focuses on audio-visual imbalance, but text is also a modality. Could the model answer correctly even when both audio and video are masked, relying only on the textual prompt? Moreover, when audio and video are counterfactual or inconsistent, does the model tend to output stereotypical or common-sense answers (biased by pretraining) rather than reflecting the actual (possibly rare) content in the video? This would reveal whether text itself introduces another form of bias.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the issue of modality bias in current audio-visual large language models (AV-LLMs), where models tend to over-rely on the visual modality while overlooking the audio. To mitigate this, the authors propose Fork-Merge Decoding (FMD), an inference-time strategy that works for both token-wise fusion and channel-wise fusion AV-LLMs. The decoding process is divided into two stages:
  -  **fork**: the model performs modality-specific reasoning by masking out the other modality
  - **merge**: the resulting hidden states are combined for joint multimodal reasoning

The method is evaluated on three representative AV-LLMs, achieving consistent improvements over vanilla decoding strategies and other inference-time contrastive decoding strategies.

### Strengths
1. **Novel yet simple method:** FMD is easy to implement, making it broadly applicable across different AV-LLMs.
2. **Comprehensive analysis:** detailed quantitative studies, including attention analysis and ablation experiments on fork layers are conducted
3. **Latency:** FMD sets the fork layer to be early layers. This leads to limited additional computation, and may be optimized by engineering (e.g., parallel processing masked audio and video inputs)

### Weaknesses
1. **Dataset/task-specific $\alpha$:** The optimization of $\alpha$ for specific datasets limits the robustness of FMD. For real-world applications, it may be infeasible to retrieve representative samples in advance.
2. **Models and benchmarks for validation:** Although representative AV-LLMs have been evaluated on benchmarks, the robustness still needs to be assessed on more models, such as reasoning models, since the improvement on VideoLLaMA2 and Qwen2.5-Omni is slight

### Questions
1. More base models and benchmarks. Suggestions: model - Video-SALMONN 2; benchmarks - Video-MME, WorldSense, AVUT
2. L235-236: "We do not apply attention-guided fusion in the channel-wise setting, as the hidden states do not disentangle audio and visual embeddings." This is confusing as audio and video features are concatenated along the channel axis. Do you mean audio and visual embeddings are not disentangled along the time axis?
3. I have concerns about 100 samples for calibration. Do they cover main domains? What is the influence of sample numbers on the performance?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed Fork-Merge Decoding (FMD) inference-time strategy: first, audio-only and video-only inputs are processed separately through the early layers of the decoder (the fork phase), and then the hidden states are merged to perform joint reasoning through the remaining layers (the merge phase). This method has been validated on three representative AV-LLMs (VideoLLaMA2, video-SALMONN, and Qwen2.5-Omni) and three benchmark datasets (AVQA, MUSIC-AVQA, and AVHBench), achieving consistent performance improvements in audio, video, and audio-visual joint reasoning tasks.

### Strengths
- The design of FMD avoids additional training or architectural modifications.
- The paper is easy to follow.

### Weaknesses
- Lack of technical novelty: This method does not alter the fundamental nature of modality imbalance. During training, visual information has higher information density, and there are cases where the model can answer correctly by focusing solely on one modality—these factors lead to the model assigning different levels of attention to different modalities. Therefore, even though the model adopts a separate masking strategy, it still fails to address the aforementioned issues and thus cannot effectively resolve the modality imbalance problem.
- The modality masking method is relatively suitable for samples with consistent audio and video. For example, in the cases provided in Figure 2, the audio and video are completely consistent: either the model is required to output descriptions that include both audio and video content (case a), or the audio provides supplementary information for the video (e.g., the video fails to clarify the type of vehicle, but the audio contains the sound of a train, which helps the model make the correct selection, case b). However, this method cannot handle scenarios where audio and video are inconsistent. For instance, when asked about a sound-producing object that does not appear in the video, masking the audio and relying solely on the video is unnecessary.
- There is a feature space gap between audio features and video features. Therefore, directly fusing them in the merge phase is unreasonable, as it fails to account for this gap between the features.
- The method lacks performance analysis on unimodal understanding of audio and video. According to the modality masking strategy proposed by the authors, it seems impossible to achieve performance improvement in the understanding of video-only or audio-only content.
- This training-free approach cannot enhance the model’s inherent performance on fine-grained tasks, such as temporal grounding and spatial grounding. The authors do not conduct relevant evaluations in the paper.
- How does the method perform on unimodal benchmarks for video or audio? Examples include video-only benchmarks like Video-MME, Video-MMMU, MMVU, MVBench, MMBench-Video, LongVideoBench, EgoSchema, PerceptionTest, MLVU, LVBench, TempCompass, and Charades-STA, as well as other benchmarks for audio/music understanding 
- No performance evaluation on unimodal (video-only/audio-only) benchmarks: The article only tests FMD on audio-visual joint benchmarks (AVQA, MUSIC-AVQA, AVHBench) and completely ignores unimodal benchmarks—including video-only benchmarks (e.g., Video-MME, MVBench, MMBench-Video, LongVideoBench) and audio-only benchmarks.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
