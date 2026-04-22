# OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Composed video retrieval presents a complex challenge: retrieving a target video based on a source video and a textual modification instruction. This task demands fine-grained reasoning over multimodal transformations. However, existing benchmarks predominantly focus on vision–text alignment, largely overlooking the rich semantic signals embedded in audio—such as speech, music, and environmental sounds—which are often decisive for comprehensive video understanding. To bridge this gap, we introduce **OmniCVR**, a large-scale benchmark for omni-composed video retrieval that establishes vision, audio, and text as first-class modalities. OmniCVR is constructed via a scalable, automated pipeline integrating content-aware segmentation, omni-modal annotation, and a rigorous dual-validation protocol involving both large language models and human experts. The benchmark comprises vision-centric, audio-centric, and integrated queries, with the latter forming the majority to accurately reflect real-world multimodal complexity. Furthermore, we propose **AudioVLM2Vec**, an audio-aware extension of VLM2Vec. By incorporating explicit audio semantics, AudioVLM2Vec achieves state-of-the-art performance, highlighting fundamental limitations in the audio reasoning capabilities of current multimodal retrieval systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the OmniCVR benchmark for composed video retrieval, encompassing vision, audio, and text modalities. A key novelty of OmniCVR is its pioneering inclusion of the audio modality as a modification target, distinguishing it from previous text-video and composed video retrieval benchmarks. Furthermore, the paper proposes a corresponding AudioVLM2Vec framework, which effectively highlights significant limitations of existing models when evaluated on the newly-constructed OmniCVR benchmark.

### Strengths
1. As a benchmark paper, this work contains the basic elements of a newly constructed benchmark. It covers essential aspects such as the limitations of previous benchmarks, the construction pipeline and its analysis, and the presentation of several baselines evaluated on the new benchmark.

2. The identified problem, i.e., the absence of an audio stream in previous Composed Video Retrieval (CVR) benchmarks, is indeed a prevalent issue in existing literature. This establishes a clear and compelling motivation for the current work.

3. The observed poor performance of several baselines when evaluated on the newly-introduced OmniCVR benchmark effectively highlights the limitations of existing models and, by extension, underscores the necessity and value of benchmarks that incorporate the audio modality.

### Weaknesses
1. My primary concern relates to the modification texts in the current OmniCVR benchmark. As shown in Table 2, the average length of queries is 52.6 words, which seems excessively long compared to previous literature, as illustrated in Figures 1 and 2. The modification texts appear to contain too many details, especially for integrated-type retrieval, which, in turn, makes the task easier to some extent.

2. This observation is further supported by the performance of the newly proposed AudioVLM2Vec framework. By incorporating audio signals into LLMs, AudioVLM2Vec achieves substantial gains of +28.5% and +64.8% in overall and audio-centric retrieval, respectively. These results are obtained without training on the provided training set, correct? The extremely lengthy modification texts make this task much easier, resembling a text-video retrieval scenario where the source videos play a less significant role.

3. I am also confused by the statements regarding long-form videos in Lines 212, 482, and 602. The average length of the source videos is only 11.8 seconds, which can hardly be categorized as long-form. Additionally, why is the real long-video benchmark MLVU introduced in Line 130? It appears that MLVU is not directly related to the composed video retrieval task.

4. Are there any performance comparisons for vision-centric retrieval? The main differences appear to arise from the audio-centric retrieval results. How do previous works perform on the visual-centric category within the OmniCVR benchmark?

### Questions
Please see the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces OmniCVR, a large-scale benchmark for composed video retrieval that treats visual, audio, and textual modalities as equally important. It addresses the oversight of audio in existing benchmarks by including audio-centric and integrated queries, where integrated queries dominate to reflect real-world multimodal scenarios. OmniCVR is constructed via a scalable pipeline involving video segmentation, omni-modal annotation, and dual validation (using LLMs and human experts). Evaluations show that current models struggle with audio-centric tasks, prompting the proposal of AudioVLM2Vec—an extension of VLM2Vec that explicitly incorporates audio semantics. This model achieves state-of-the-art performance, revealing critical gaps in handling audio-dependent queries and underscoring the necessity of audio-aware multimodal retrieval systems.

### Strengths
Originality: This work first define and benchmark "audio-inclusive composed video retrieval." This effectively eliminates the key limitation of prior CVR benchmarks: the neglect of audio as a first-class semantic modality. The proposed AudioVLM2Vec also offers a novel, pragmatic architecture for audio infusion.

Clarity: The paper is easy to follow and understand.

Significance: The paper proposes "audio-inclusive composed video retrieval", which enables direct future research on audio-visual-text reasoning and is an essential tool for developing genuinely holistic video understanding models.

### Weaknesses
1. While the paper's introduction of audio-centric queries is commendable, its treatment of "audio" remains oversimplified. A truly comprehensive audio-centric benchmark must consider a wider array of acoustic, semantic, and para-linguistic dimensions and provide specialized evaluations for each. The benchmark misses the opportunity to define and evaluate based on these essential audio dimensions:

     (a)  Para-linguistic Features in Speech:  (i)Emotion & Tone, (ii) Speaker Characteristics 

     (b) The lexical content in Speech

     (c) Hierarchy of Environmental Sounds: "Environmental Sound" is a vast category that should be broken down into:   (i) Nature Sounds (wind, rain, animals), (ii) Mechanical Sounds (engines, machinery, alarms), (iii) Urban Soundscapes (traffic, crowds, construction), (iv). Foley/Action Sounds (footsteps, glass breaking, doors closing)

     (d) Temporal Dynamics: This involves rhythm, pace, and intensity over time. A query could be: "Find the same scene but where the background music has a slow, steady beat instead of a fast, erratic one."

2. Lack of Specialized Evaluation per fine-grained Dimension: The paper's aggregated "Audio-Centric" results (Table 4) are misleading. A model might excel at modifying speech content but fail completely at modifying musical timbre or emotional tone. This critical diagnostic information is lost.

3. In the same way, Limited Exploration of Audio-Visual Interactions: The benchmark and proposed method treat audio and vision as largely separate streams that are fused late (e.g., by concatenating text descriptions). It does not explicitly model or evaluate the complex, synergistic relationships between audio and vision, such as causal links (a person's mouth movements and their speech). The query text may be that "Take the source video of a man speaking and replace his audio with a different language while also modifying the video to show his lips moving out of sync."

### Questions
1. The paper contains many figures and tables that are not explicitly cited or referenced in the text.

2. Efficiency Benchmarking: Report the inference latency and FLOPs for AudioVLM2Vec compared to other baselines, highlighting the cost of the audio-to-text step.

3. Inadequate Ablation Studies: The paper fails to provide sufficient ablation analysis to disentangle the contributions of different components in AudioVLM2Vec. Critical questions remain unanswered: How much performance gain comes from audio transcription versus the fusion mechanism? How does the model perform on non-speech audio domains?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce OmniCVR, a composed video retrieval benchmark that incorporates audio as a key modality alongside vision and text. The authors argue that the queries included in the dataset are representative of real-world queries, and target audio in addition to vision. Additionally, the authors extend the VLM2Vec retrieval method to simultaneously model audio, which they name AudioVLM2Vec. This approach achieves SoTA performance in multimodal retrieval.

### Strengths
- The paper accurately identifies that audio is an under-utilized modality in video retrieval. We do lack benchmarks that target this modality.

- In addition to dialogue, the dataset also centers non-language audio like music and sound effects, which historically have been even less focused on than dialogue.

- The dataset is large enough to serve as a training set in addition to validation.

- AudioVLM2Vec achieves impressive performance compared to existing methods. Its development is sensible and is described clearly. The chosen comparison methods are representative of the current research space.

- The paper is generally sound and well-written.

- The limitations and future work section is detailed and thoughtful.

### Weaknesses
- The video-text retrieval benchmarks section in the related works segment is quite small. There have been many more video retrieval benchmarks, some of which center audio. Describing the differences between this dataset and others that center audio (VaTeX, MultiVENT 2.0, etc.) would be informative for those evaluating the novelty of the dataset.

- It would be nice to describe the genres and types of content focused on in the dataset in more depth.

### Questions
- What key contributions does this dataset make compared to existing video retrieval benchmarks that center audio? What is the importance of composed video retrieval tasks?

- What genres of video are included in the dataset?

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
This paper introduces OmniCVR, a new benchmark designed to evaluate omni-composed video retrieval, where the system retrieves a target video given a source video and a textual modification instruction that may involve changes in visual content, audio, or both.
The authors describe a large-scale data construction pipeline that combines LLM-assisted multimodal annotation (Qwen2.5-Omni, Gemini 2.5 Pro) with validation across multiple modalities. They also propose AudioVLM2Vec, an embedding model that integrates audio information via textual transcriptions to better handle audio–visual composition. The benchmark aims to capture fine-grained multimodal reasoning, extending beyond traditional text–video retrieval tasks..

### Strengths
- The paper tackles an underexplored and ambitious problem where compositional video retrieval jointly involves visual, auditory, and textual reasoning. By integrating audio-based instructions alongside visual composition, it broadens the conventional scope of video retrieval tasks. This is a valuable and forward-looking addition for multimodal benchmarks.

- The benchmark creation process is well-documented and systematically structured. The use of multiple LLMs for multimodal captioning, filtering, and validation makes the dataset large, diverse, and reproducible. This methodological transparency is a clear strength.

- The implementation details, model configurations, and evaluation metrics are clearly presented. The writing quality is high, and the paper provides good contextualization against prior work .

### Weaknesses
- The task assumes that, for a given source video and a detailed compositional instruction, another video exists in the corpus that satisfies all the described changes. Many examples involve complex spatial, temporal, or auditory transformations that are implausible to find in any real-world dataset. Such a setup feels conceptually closer to conditional video generation than retrieval. The paper does not convincingly justify why retrieval is a meaningful or achievable approach for these highly specific instructions.

- If OmniCVR truly captures omni-compositional reasoning, models trained on it should exhibit strong transfer to conventional video retrieval datasets (e.g., MSR-VTT, VATEX, Dense-WebVid-CoVR). Yet the paper does not test this. The absence of out-of-domain evaluation makes it impossible to assess whether the learned representations generalize beyond the synthetic distribution or simply overfit to the compositional language space introduced by the benchmark. 

- While the benchmark emphasizes audio–visual–text reasoning, the proposed model (AudioVLM2Vec) primarily uses text transcriptions of audio, not raw acoustic embeddings. This simplifies audio reasoning into another text-conditioning signal, limiting the extent to which the benchmark truly evaluates multimodal understanding.

- The paper does not present qualitative examples or human evaluations showing that retrieved videos actually match the intended compositional modifications. Without such validation, it is unclear whether the benchmark’s retrieval results are semantically meaningful.

### Questions
1. Many of the benchmark’s compositional instructions describe modifications that seem infeasible to find in existing datasets (e.g., changing lighting, replacing sound events, or altering temporal composition). Could the authors clarify why they chose to formulate this as a retrieval problem rather than as conditional video generation? Do they believe that realistic corpora can ever contain such compositional matches at scale?

2. If the benchmark truly captures omni-compositional reasoning, models trained on OmniCVR should generalize to more conventional video retrieval datasets such as MSR-VTT, VATEX, or ActivityNet Captions. Have the authors tested such out-of-domain performance? If not, how can they claim that the benchmark teaches general multimodal retrieval rather than overfitting to its own synthetic compositions?

3. Could the authors provide example showing that the retrieved videos actually reflect the compositional modifications specified in the text?

### Soundness
3

### Presentation
3

### Contribution
3
