# JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Understanding videos inherently requires reasoning over both visual and auditory information. To properly evaluate Omni-Large Language Models (Omni-LLMs), which are capable of processing multi-modal information including vision and audio, an effective benchmark must comprehensively cover three key aspects: (1) multi-modal dependency (i.e., questions that cannot be answered using vision or audio alone), (2) diverse audio information types (e.g., speech, sound events), and (3) varying scene spans.
However, existing datasets fall short in one or more of these dimensions, limiting strict and comprehensive evaluation.
To address this gap, we introduce JointAVBench, a novel benchmark with strict audio-video correlation, spanning five cognitive dimensions, four audio information types (speech, sound events, music, vocal traits), and three scene spans (single-, cross-, and full-scene).
Given the high cost of manual annotation, we propose an automated pipeline that leverages state-of-the-art vision-LLMs, audio-LLMs, and general-purpose LLMs to synthesize questions and answers that strictly require joint audio-visual understanding. 
We evaluate leading vision-only, audio-only, and Omni-LLMs on our dataset. Results show that even the best-performing Omni-LLM achieves only 62.6\% average accuracy, outperforming uni-modal baselines but revealing substantial room for improvement, especially in cross-scene reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes yet another audio-visual understanding benchmark called JointAVBench. The benchmark is trying to capture audio-visual correlation in videos, especially focusing on questions that cannot be answered by visual information alone. The authors also mention the idea about multi scene-spans and emphasize that their benchmark exhibits multiple scenes.

### Strengths
1. Paying attention to the role of audio in video understanding is needed in the research community. 

2. They provided webpage for the benchmark and their experiments spans a range of audio-visual LLMs, which might be useful information for the research community.

### Weaknesses
1. Technical Contribution:

I have significant doubt on the technical contribution of this paper/benchmark. Given that it is mainly focused on audio in video, a clear discussion of how important the audio modality in these tasks is needed with supporting evidence in experiments. Please also compare to the following papers:

[1] Yang et al. "Audio-centric Video Understanding Benchmark without Text Shortcut", https://arxiv.org/pdf/2503.19951.
[2] Hong et al. "WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs", https://arxiv.org/abs/2502.04326

which also introduce audio-visual benchmark with audio-centric data. In particular, [1] also includes mainly questions that cannot be answered with visual modality alone, and also uses both speech and audio events, so the first two points are already being done previously unfortunately. 

2. Ambiguity in the definition of __scene__:

Regarding the 3rd contribution, I am unsure why datasets like video-MME are not multi-scene. I don't see a clear difference between the proposed dataset and videos in video-MME. 

3. Careless Writing:

Please correct the question marks on page 3 for those citations.

### Questions
See weaknesses about the definition of scenes

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce a benchmark for evaluating models that jointly process audio and vision. The dataset consists of 2.8k QA pairs organized into fine-grained categories. Unlike existing benchmarks, it explicitly includes diverse audio types—speech, music, vocal traits, and sound events. In addition, the data are annotated along cognitive dimensions (e.g., temporal, spatial, emotional) and by scene type (single, multiple, or full, indicating how much of the video is needed to answer a question). The evaluation reveals that models struggle disproportionately on certain question types, and even the strongest model assessed achieves only 56% accuracy on average, highlighting substantial room for improvement.

### Strengths
- The dataset is automatically generated and then manually inspected, an important step that strengthens its reliability as a benchmark.
- I appreciate the fine-grained categorization of question types, which enables more detailed error analysis of current models.
- The analysis in Table 4, comparing joint modality performance with single-modality performance, is particularly insightful and underscores an important dimension for evaluating “omni-models.”

### Weaknesses
- Although the dataset’s fine-grained design is valuable, the effective size of each subset is necessarily small given the overall scale of 2.8k examples, which limits the strength of conclusions about model performance on specific question categories.
- While the dataset has been manually inspected, reporting human performance from an independent set of annotators would provide a useful reference point for comparison with model capabilities.

### Questions
Could you provide a breakdown of each question type's size in the benchmark? (as broken down in Table 2)

Typos
- line 481: "JointAVBench is" - missing space.

### Soundness
2

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
This paper introduces JointAVBench, a new benchmark designed to evaluate the joint audio-visual reasoning capabilities of Omni-LLMs. JointAVBench is organized according to a comprehensive, multi-dimensional taxonomy. With the proposed three-stage semi-automated generation pipeline, the data acceptance rate during human verification reached 71.8%, significantly improving data collection efficiency. An extensive evaluation of current Omni-LLMs, Video-LLMs, and Audio-LLMs on JointAVBench highlights significant performance gaps and specific areas of weakness.

### Strengths
- The paper is well-written and easy to follow. The proposed semi-automated benchmark generation pipeline is reasonable and and effectively lowers data labeling costs.
- The comparative analysis of various models reveals the limitations of existing models, which in turn reflects the value of the benchmark.

### Weaknesses
The benchmark's exclusive reliance on pre-existing datasets raises two methodological concerns. Firstly, it compromises the integrity of the evaluation, as the test data may be contaminated (i.e., previously seen by models) or decontextualized. Secondly, this approach circumvents the foundational challenge of raw data acquisition and curation. The fidelity of any benchmark is fundamentally contingent on its source data, and a robust construction pipeline should therefore incorporate an automated or semi-automated mechanism for this essential process.

### Questions
Given the submission deadline, the exclusion of some recent models from the evaluation on JointAVBench is understandable. Nevertheless, the paper would be significantly strengthened if the authors could include performance metrics for models like Qwen3-Omni, video-SALMONN-2+, and Gemini-2.5-pro, as this would provide a more comprehensive reflection of the current capabilities of Omni-LLMs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The JointAVBench benchmark proposed a comprehensive evaluation tool that covers 5 cognitive dimensions (temporal, spatial, emotional, plot, and long-form), 4 audio types (speech, sound events, music, and vocal traits), and 3 scene spans (single-scene, cross-scene, and full-scene). It addresses the shortcomings of existing datasets in terms of multimodal dependency, audio diversity, and scene complexity.

### Strengths
- Achieves a 100% Audio-Visual Correlation Ratio (AV Corr. Ratio), where every question requires the integration of both audio and visual information to answer, avoiding the flaw that single-modality can solve questions in existing benchmarks.
- Conducts a comprehensive evaluation of three types of mainstream models (Omni-LLMs, Video-LLMs, Audio-LLMs) and analyzes performance differences across audio types, scene spans, and cognitive dimensions, providing in-depth insights for model optimization.

### Weaknesses
- The benchmark exclusively uses the SF20K short-film dataset (1,046 films after filtering) as its video source, but this choice introduces severe scene bias and data homogeneity: Films are professionally produced, narrative-driven content—they do not reflect real-world audio-visual scenarios (e.g., surveillance footage, live streams, industrial monitoring, daily vlogs) where Omni-LLMs are likely to be applied. 
- The "semi-automated pipeline" (omni-caption generation → QA creation → quality control) relies heavily on LLMs (Qwen2.5-Omni, Qwen2.5-VL) but fails to validate the accuracy and absence of hallucinations in LLM outputs—this undermines the entire benchmark’s credibility.
- Meanwhile, the authors use Qwen2.5-Omni and Qwen2.5-VL, and having these models serve as both annotation models and evaluation models will introduce bias. This causes the answers to be more inclined towards the understanding and responses of these two models, leading to biased evaluation.
- The dataset adopts movies, which focus heavily on verbal dialogue. This results in greater bias in the evaluation of natural sounds and music, making the dataset unsuitable to be used as a benchmark for evaluation.
- No information on annotator qualifications (e.g., whether they have expertise in audio-visual analysis) or inter-annotator agreement (e.g., Cohen’s Kappa coefficient). If two annotators disagree on 30% of QAs, the "accepted" data is subjective and unreliable.
- The manuscript uses "official codebase with default configurations" for open-source models  but "official APIs" for closed-source models . However, it does not disclose API parameters (e.g., max tokens) or whether closed-source models used larger parameter sizes  This makes performance comparisons meaningless—higher accuracy could stem from larger model size, not better joint reasoning.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
