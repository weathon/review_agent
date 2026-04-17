# WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs

- Decision: Accept (Poster)
- Scores: 4, 6, 10, 6

## Abstract
We introduce WorldSense, the first benchmark to assess the multi-modal video understanding, that simultaneously encompasses visual, audio, and text inputs. In contrast to existing benchmarks, our WorldSense has several features: (i) collaboration of omni-modality, we design the evaluation tasks to feature a strong coupling of audio and video, requiring models to effectively utilize the synergistic perception of omni-modality; (ii) diversity of videos and tasks, WorldSense encompasses a diverse collection of 1,662 audio-visual synchronised videos, systematically categorized into 8 primary domains and 67 fine-grained subcategories to cover the broad scenarios, and 3,172 multi-choice QA pairs across 26 distinct tasks to enable the comprehensive evaluation; (iii) high-quality annotations, all the QA pairs are manually labeled by 80 expert annotators with multiple rounds of correction to ensure quality. Based on our WorldSense, we extensively evaluate various state-of-the-art models. The experimental results indicate that existing models face significant challenges in understanding real-world scenarios (65.1% best accuracy). By analyzing the limitations of current models, we aim to provide valuable insight to guide development of real-world understanding. We hope our WorldSense can provide a platform for evaluating the ability in constructing and understanding coherent contexts from omni-modality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces WorldSense, a new benchmark for evaluating the omnimodal understanding capabilities of multimodal LLMs. The benchmark simultaneously incorporates visual, audio, and text inputs to assess a model's ability to comprehend and reason about real-world scenarios. The paper also presents a detailed evaluation of existing multimodal LLMs on the WorldSense benchmark.

### Strengths
- The WorldSense Benchmark seems well annotated and comprehensively considers visual and auditory information, which is meaningful for evaluating a model's ability to understand video.

- The experiments are sufficient and convincing.

### Weaknesses
- In terms of task design, although WorldSense includes a list of tasks, the paper does not clearly show the special design targeted at the integration of audio and visual information. To me, it seems many tasks either assess audio understanding independently or visual understanding independently, rather than evaluating the understanding of audio and video as a whole.

-  There are already some benchmarks for evaluating audio-visual video understanding, such as AVUT [1] and DailyOmni [2]. The paper did not discuss or compare with them. What are the main differences between WorldSense and these benchmarks?

[1] Yang et al, Audio-centric Video Understanding Benchmark without Text Shortcut, arXiv preprint arXiv:2503.19951

[2] Zhou et al, Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities, arXiv preprint arXiv:2505.17862

### Questions
- About the evaluation settings.  In Line 316, the paper mentions that the models are tested "following the recommended pre-processing procedures". What are the specific test settings for each model (such as the frame rate for video frame extraction, the maximum number of extracted frames, etc.)? Could the different pre-processing settings between models lead to incomparable results?

- The audio-visual LLMs evaluated in the paper seem a bit weak on WorldSense. Could the authors provide an analysis of recent powerful omni models like Qwen3-Omni and Video-SALMONN 2? This would establish a more robust and current baseline, and also help to verify if the significant challenges highlighted by the benchmark are persistent even for the latest generation of models.

### Soundness
3

### Presentation
3

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
This paper proposes WorldSense, a benchmark designed for omni-modal video understanding. Specifically, to answer questions in WorldSense, an MLLM’s response must rely on both video and audio information. Experiments on the proposed WorldSense benchmark reveal the limitations of current MLLMs in omni-modal reasoning.

### Strengths
1. Requiring both video and audio modalities for accurate responses to each question in WorldSense facilitates a more comprehensive evaluation of current MLLMs in omni-modal reasoning.
2. Experimentally, the performance drop of current video-audio MLLMs indicates that the fusion between modalities is ineffective or even detrimental.

### Weaknesses
1. While WorldSense emphasizes real-world omni-modal perception, understanding, and reasoning, the benchmark primarily consists of QA pairs. I believe that interactive question answering would be more practical for real-world scenarios. Moreover, isn’t the term "omni-modal" somewhat overstated, given that the benchmark only includes video, audio, and text modalities?
2. Lack of analysis on why the fusion of open-source audio and video models failed. While I wouldn’t tend to reject the paper for this reason, providing such an analysis would offer the community deeper insights than merely presenting the conclusion.

### Questions
1. #82–83: There’s a typo ,“THe”.
2. #129–130: Pay attention to the spacing between the image title and the main text. It currently looks a bit confusing.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper propose WorldSense, an omnimodal benchmark including video, audio, and text. WorldSense requires joint reasoning over synchronized visual and audio inputs to evaluate the ability of existing MLLMs. It contains 3,172 MC QA pairs, spanning various domains and subcategories. The paper evaluates open-source and proprietary MLLMs on WorldSense, showing that current MLLMs are having trouble in integrating audio and visual data effectively in realistic setting.

### Strengths
1. Solid motivation. The paper focus on omnimodal reasoning, requiring the models to utilize visual and audio inputs together to answer the question. This designed correlations make it distinct from most existing MLLM benchmarks.
2. High-quality human-reviewed annotations. Unlike recent benchmarks, WorldSense is reviewed and revised by human expert, instead of relying solely on LLMs. This is a guarantee for the quality of the benchmark. It is very rare these days.
3. Comprehensive experiments. The paper conducted comprehensive experiments on existing models, including open-source ones and proprietary ones, providing a benchmarking foundation for future research.

### Weaknesses
1. Video caption is not a good representative for "text" modality. This makes the "omni" a little overclaim. Given real-world constraints, audio–video may already suffice for evaluating omnimodality, as seen in emerging “world models.”
2. The question types are restricted in multiple-choice QA. This is common in existing benchmarks, but given the ability of MLLMs, free-form answers can yield deeper insights and align with user-end usage.
3. The benchmark is currently focusing on perception and recognition. Given the lengths of the dataset (141.1s) and the requirement for multiple modalities, it may be possible to curate a subset for higher-level reasoning tasks.

### Questions
1. How does WorldSense handle temporal reasoning—do questions depend on sequential context or only short clips? An ablation study on image / a few frames would elaborate this.
2. This is beyond the scope of this paper. But is it possible to extend WorldSense to an open-ended or generative benchmark? This would make it more aligned with user-end usage.

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
3

### Summary
The paper introduces WorldSense, a new benchmark designed to evaluate how MLLMs understand and reason about real-world video-audio-text inputs. WorldSense integrates both audio and video to test true omni-modal understanding.
The dataset includes 1,662 synchronized videos across 8 domains and 67 categories, with 3,172 multiple-choice QA pairs spanning 26 different tasks. These questions require both audio and visual cues to answer correctly. The authors also evaluate various open-source and proprietary MLLMs, showing that even advanced models like Gemini 2.5 Pro only achieve 65.1% accuracy, exposing significant gaps in real-world multimodal reasoning.

### Strengths
- A novel audio-video benchmark focused on omni-modal understanding.
- Expert curated by annotators and manual QA design for better quality.
- Paper is well-written.

### Weaknesses
- Although, the dataset is diverse, the distribution of question difficulties across categories or cognitive levels is not very clear. Some tasks might be more perception-heavy than reasoning-heavy, which could bias model comparison.

### Questions
- How do you ensure that questions cannot be answered from text transcripts alone?
- Will the benchmark be publicly released, and under what license?

### Soundness
3

### Presentation
3

### Contribution
3
