# BLAB: Brutally Long Audio Bench

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models.
To address this gap, we introduce Brutally Long Audio Bench, a challenging long-form audio reasoning benchmark that evaluates audio LMs on localization, duration estimation, emotion and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB, and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, speaker counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding and reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
A  benchmark for large audio language models, with speech-focused tasks including temporal localization  of words and named entities,  emotion, speaker counting and event duration. Due to the incapability of existing models to handle large audio, the evaluation is on two Gemini models, with a shorter version of the dataset used for broader evaluation. Several newer models have not been benchmarked (Qwen 2.5 Omni, Audio Flamingo 2, Audio Flamingo 3, Kimi.)  Moreover, the benchmark does not appear to include music. Nevertheless, it could be a valuable contribution to benchmarking audio models, if details of its release, and the ease of benchmarking (scripts, frameworks) were discussed in detail.

### Strengths
Many practical tasks for LALMs require diarization, event or named entity recognition, and these are tasks at which current models do poorly. A public benchmark focused on these would be valuable.

### Weaknesses
The limited number of models studied.
The lack of a commitment to release the benchmark and associated scripts.
Does not cover music or entertainment audio -- is speech/noise focused.

### Questions
which audio flamingo model did you evaluate?
A comparison of the models' performance on Chime with their performance on BLAB would be interesting to see what value BLAB adds.
What languages were covered?

### Soundness
3

### Presentation
3

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
This paper introduces BLAB, a new benchmark designed to evaluate the reasoning capabilities of audio language models on long-form audio. The authors argue that existing benchmarks, which focus on short clips, fail to assess a critical aspect of real-world audio understanding. BLAB consists of over 833 hours of diverse audio, with an average clip length of 51 minutes. It features 8 challenging tasks across 4 categories: localization, counting, emotion, and duration. The authors evaluate several audio LMs and find that all models "struggle" substantially with these long-context tasks.

### Strengths
- A Novel Benchmark: The paper's primary contribution is the BLAB benchmark itself. It addresses a critical, well-documented gap in the field: the lack of evaluation for audio-grounded reasoning on long-form content (averaging 51 minutes). This moves the community beyond short-clip evaluations to a more realistic and challenging domain.
- Transparent Data Collection Pipeline: The authors detail a rigorous data pipeline, using permissively licensed sources from YouTube. This process is strengthened by a "human-assisted filtering procedure" and a human-AI collaborative annotation framework to ensure data quality.
- Insightful Analysis: The paper provides valuable insights by evaluating state-of-the-art models and identifying key failure modes. These include clear performance degradation as audio duration increases and the presence of the "lost in the middle" problem for audio inputs.

### Weaknesses
- The evaluation of the full, long-form benchmark (BLAB) relies almost entirely on closed-source, proprietary models like Gemini. While this effectively demonstrates the benchmark's difficulty, it limits reproducibility and prevents the research community from conducting a deeper analysis of the models' failure modes. The open-source models were only evaluated on the BLAB-MINI subset, so their long-context audio capabilities remain unevaluated.
- The design of the word localization task seems confounded by model limitations that are unrelated to audio reasoning. The paper notes that the model's poor performance is largely due to its output token limit (8096), which only allows it to generate ~2% of the ground truth timestamps for an average audio file. This makes it an unfair evaluation, as it is testing the model's output length rather than its true localization ability.

### Questions
Please refer to the weaknesses above for the questions.

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
4

### Summary
**Summary:** Audio language models aim to understand all the spoken interactions within a single recording. However, most audio language models can only process short audio segments (under 30 seconds), and they are rarely evaluated on longer, conversational speech segments.

Authors of this work developed a benchmark that consists of very long audio segments, and used it to evaluate some SOTA models' performance, and showed their limitations at this scale and how they struggle differently on diverse tasks due to such much longer audio recordings.

To be specific, the benchmark (BLAB) is 833 hours of conversational speech, with 8 tasks, and evaluates 4 reasoning skills, which are temporal localization, speaker counting, emotion interpretation, and duration estimation. Each audio recording has human-annotated question and answer pairs. A total of 6 audio LMs are evaluated on BLAB.

### Strengths
**Strength:**
The background of the limitations of current audio language models and the motivations are clearly and logically stated and emphasized throughout the paper.

Each category of the reasoning capabilities and the tasks under it are clearly designed and described in detail.

The experimental setup is designed carefully and fairly.

### Weaknesses
**Weakness:**

The description of the tasks contained in BLAB and the 4 reasoning skills can be clearer. For instance, after reading the abstract section, and up to the point 'across eight tasks and evaluates four fundamental reasoning skills', the relationship between the eight tasks and the four reasoning skills can be a little confusing. It would be better if you could describe that the 8 tasks you are evaluating are under 4 categories, just like your caption for figure 1, in the abstract, to avoid this minor confusion.

It is also unclear whether each audio segment has the 8 tasks, or if they have variations.

At the end of the Introduction section, you briefly mentioned the audio LMs' performance on a certain category, such as localization. It looks like the reported metric. I can only assume that all 3 tasks in this category are classification tasks, so the reported results are the average? Since you discussed the performance here, this needs to be better explained. 

After evaluation, it is found that prompts have higher importance than the actual audio contents. If so, I would suggest placing a greater emphasis on various types of prompts, beyond those introduced on page 8, to better understand the limitations of the prompts.

For fairness and consistency, I understand that authors are only focused on audio sources on YouTube. But if the authors have resources and time, I would suggest additional evaluations on more specific tasks, such as speech datasets that are specifically for counting, emotion recognition, etc. There are some datasets that have conversation speech that far exceeds 30 seconds. So the results evaluated on the current audio LMs could be different, then you can at least have some conclusions based on the difference between BLAB and these datasets. Because focusing only on YouTube and English datasets is also a limitation.

### Questions
For emotion recognition, you can take a look at the work below that also aligns with your approach and considerations somehow.

"The MERSA dataset and a transformer-based approach for speech emotion recognition." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes BLAB, a long-form (minutes to hour-scale) "audio LM" benchmark spanning: (i) localization (word, named-entity, advertisement), (ii) counting (speaker number), (iii) duration (file/event), and (iv) emotion (valence/arousal ranking; comparative reasoning). The core datasets are Creative-Commons speech from YouTube; annotation relies heavily on WhisperX forced alignment plus LLM-assisted span extraction; evaluation is mostly exact-match/JSON and frame-F1. Results chiefly report Gemini 2.0 Flash/Pro on long audio; robustness tests replace/overlay content with silence or Gaussian noise; a short-audio subset (BLAB-MINI) is included.

### Strengths
- Truly long speech (≈50–60 min avg per item) where current LAMs struggle, including controlled noise/silence robustness and position-sensitivity tests.
- Clear, reproducible prompt formats and metric definitions per task family.

### Weaknesses
- Doesn't evaluate many recent models like Gemini 2.5, GPT-4o-audio.
- Doesn't evaluate models recent models like Audio Flamingo 3 which claims long audio understanding.
- NE & Ad localization are derived by running text-only LMs on transcripts, then mapping spans back via WhisperX timestamps. This makes many items solvable from text alone and entangles evaluation quality with ASR/FA errors rather than acoustic understanding. 
- Word timestamps over 191 hours are WhisperX with just ~1% corrections, reviewed on a subset by an author is kind of insufficient for long-audio with overlaps; alignment errors propagate directly into and impact ground truth. 
- Despite "audio LM" claims, items come from interviews/podcasts/political talks; background music is incidental. There's almost no coverage of non-speech environmental or musical structure. With the recent advancements of large audio language models I think the benchmark should also focus on including other modalities of audio and not just speech.
- How many items per task were fully human-audited? What is inter annotator agreement?

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
