# MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Recent advances in large language models have laid the foundation for multimodal LLMs (MLLMs), which unify text, speech, and vision within a single framework. As these models are rapidly evolving toward general-purpose instruction following across diverse and complex tasks, a key frontier is evaluating their crosslingual and multimodal capabilities over both short- and long-form inputs.
However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on a single modality at a time, rely on short-form inputs, or lack human annotations--hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first crosslingual human-annotated benchmark based on scientific talks on NLP and beyond. MCIF evaluates instruction following in crosslingual, multimodal settings over different input lengths and spans four macro-tasks: recognition, translation, question answering, and summarization. It covers three core modalities (speech, vision, and text) and four diverse languages (English, German, Italian, and Chinese), fully aligned across all dimensions. This parallel design enables a systematic evaluation of MLLMs' abilities to interpret instructions across languages and effectively integrate multimodal contextual information. Our benchmarking and analysis of 23 models highlight universal challenges across modalities and tasks, indicating substantial room for improvement in future MLLMs development. MCIF is released under CC-BY 4.0 license to promote open research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new benchmark for evaluation of instruction-following capabilities under a comprehensive context of multi-modal, cross-liingual, and varied input length. The benchmark is validated on 23 models, which is good for comparison, and provides guidance on future development.

### Strengths
* This benchmark is indeed covering a wide arange of topics, tasks, and languages. It could be used as a supplement to the existing benchmarks that challenge the IF capabilities of MLLMs.
* The paper is generally well-written, easy to follow with pretty illustrations.
* The detailed descriptions on manual annotations are provided for guaranteeing the quality of the benchmark.

### Weaknesses
* The definition of "scientific talks" might be limited in the ACL context, which might not be that broad enough to include nature science/medicine domains.
* The evaluation metrics might not be comprehensive enough where recent LLM-as-a-Judge-based metrics are under-explored.
* The prompt templates (see Table 4,5,6) might not be varied enough to be practical. The dynamic prompting by user-simulation (LLM-driven) is not considered and there might exists a risk of over-fitting where future MLLMs might overfit the patterns inherent in such fixed, less-varied templates.
* Discussions and analyses are not enlightening enough.

### Questions
* Perhaps modify the descriptions about the scientific talks to be clear, explicit in the introduction.
* The authors are encouraged to extend more metrics as a complement to existing rule-based computing metrics.
* The user-simulation by LLMs is important to an evergreen benchmark which prevents from hacking. The authors are encouraged to provide such prompt augmentation.
* The discussions should contain explicit take-home messages which help the development of future MLLMs. In addition, the case-by-case (especially typical cases in each task) analysis should be provided to be intuitive.
* The authors might consider citation of recent instruction-following studies from the perspective of data recipe:
e.g., Qin, Y., Yang, Y., Guo, P., Li, G., Shao, H., Shi, Y., ... & Sun, X. (2024). Unleashing the power of data tsunami: A comprehensive survey on data assessment and selection for instruction tuning of language models. arXiv preprint arXiv:2408.02085.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents MCIF, a new benchmark to test how well large multimodal models can follow instructions across languages and input types. The dataset comes from real scientific talks with aligned text, audio, and video in four languages: English, German, Italian, and Chinese. It covers 13 tasks grouped into recognition, translation, question answering, and summarization. The authors also test 23 different models including text-only, speech, video, and multimodal ones. The results show that models still struggle a lot when dealing with long inputs, with combining speech and video, and when asked to summarize complex content.

### Strengths
1) The idea is very timely. Multimodal and multilingual instruction following is clearly the next big step for LLMs.

2) The dataset design is thoughtful. Everything is aligned across languages and modalities, which makes comparisons fair and controlled.

3) It is fully human-annotated. The effort to manually create transcripts, translations, and QA pairs really increases reliability.

4) The analysis is broad and systematic. The paper looks at different model types, context lengths, and prompt variations. The findings are detailed and easy to follow.

5) The authors share code, prompts, and data guidelines openly, which helps with reproducibility and future research.

### Weaknesses
1) The dataset size feels small compared to other multimodal benchmarks. About ten hours of content may not capture much variation in topics or speakers.

2) The evaluation relies mostly on automatic metrics. Some human checks or qualitative examples would make the results more convincing.

3) The discussion could go a bit deeper on why models fail.

### Questions
How consistent were the human annotations across different annotators and languages?

### Soundness
3

### Presentation
3

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
This work introduces a new benchmark called MCIF, comprised of four classes of tasks (transcription, translation, question answering, and summarization) with aligned multimodal data (text, speech, and audio) from ACL lectures across four languages (English, German, Italian, and Chinese).

The benchmark uses human experts to provide annotations such as audio transcriptions and translations. Domain experts contribute the Q&A pairs, including unanswerable pairs.

The paper reports performance of 23 different models supporting various subsets of modalities (or all of them) and provides some analysis on prompt adherence and the contribution of individual modalities for those models that support multiple.

### Strengths
* The construction of the benchmark heavily relies on human experts, including domain experts of the material used, for ground truth annotation. It is really great to see such a rigorous collection protocol.
* The proposed task metrics (WER, BertScore, and COMET) are widely used and can be computed without relying on external APIs that may change over time.
* The paper presents results on their benchmark across 23 different models from various model families.

### Weaknesses
* The proposed benchmark is cross-lingual only from English into one of the three alternative languages (German, Italian, Chinese). Input audio and video (e.g. slides) are always English. This may limit the utility of the work for evaluating multi-lingual models’ capabilities in practice.
* While the work is clearly describing their contribution to be a benchmark about scientific talks, the data used seems quite narrow even within that domain; The benchmark exclusively uses recordings from presentations of main conference papers at ACL 2023. The limitation to this split was described as aiming to select videos newer than the knowledge cutoff of models to be evaluated, but this seems perhaps not well motivated as many of the evaluated models in the paper have a cutoff later than that (e.g. Phi 4, Gemma 3, GPT-oss, Gemini 2.5 Flash, for example). Selecting recordings from a single conference from a single year, and only main conference papers seems to significantly and perhaps arguably unnecessarily constrain the diversity of the produced benchmark (topics presented, recording setup and environment, speaker demographic,  etc. are presumably quite homogenous in the resulting work).
* The limited diversity of the benchmark, in my opinion, makes it a bit difficult to draw strong conclusions from the observed results. For example, section 5.1 argues that since no improvements were observed on the summarization task when adding speech or video in addition to the text transcript, this “underscores the limitations in multimodal integration”. However, an alternative interpretation may be that since the target summary is the abstract of the presented work, the content expected there may be particularly well captured in a transcript of the talk, since by nature such abstract would not include anything that may only occur on stage or any specifics of the visual presentation.
* The work focuses on models smaller than 20 billion parameters for its main study. While technical limitations are understandable, evaluating some stronger frontier models besides Gemini 2.5 Flash would be a welcome addition (GPT-5, Gemini 2.5 Pro, for example). Section 5.2 also does not seem to discuss Gemini 2.5 Flash, which is perhaps particularly limiting as the conclusion suggested, that “current MLLMs struggle to effectively integrate speech and video”, which may be particularly true for smaller model sizes, and notably Gemini 2.5 Flash was the best performing audio-video model reported in Table 2.
* The paper discusses instruction following through the lens of prompt adherence by comparing the results of $MCIF_{fix}$ and $MCIF_{mix}$. This seems a somewhat limited view of instruction following since the main signal will be whether the model will treat one (specific) prompt similar to 10 other semantically equivalent paraphrases. For example, assume a model that is very sensitive to the prompt - as long as the particular selected prompt for $M_{fix}$ achieves performance similar to the average of performances of $MCIF_{mix}$, this issue would not be visible in the chosen metrics. Additionally, the prompts themselves are relatively basic one-sentence instructions which may not exercise the instruction following capabilities of modern LLMs to their fullest extend.
* (Note that the first prompt for summarization in German as shown in Table 7 seems semantically slightly different to the other 9; it asks for a summary with at most 200 words instead of approximately 200 words; I am unable to verify all prompts, but minor semantic differences may also explain much of the observed fluctuation in scores?)

### Questions
* In table 1, the context type for the text-inputs is “-“. Does that mean that for the short context setting we still provide the full transcript and only non-text modalities are shortened to the relevant region of interest? If not, why is the performance of text input for short context not reported for translation and QA?
* In the main results in table 2, are all question pairs used for QA regardless of whether the questions can be answered with the provided input modalities? If so, what is the “correct” way to answer questions that are not answerable with the provided input modalities (including questions that are generally unanswerable)?

### Soundness
3

### Presentation
3

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
This paper presents MCIF, the first human-annotated, multimodal, crosslingual instruction-following benchmark built from scientific presentations. It evaluates large multimodal language models (MLLMs) across three modalities (text, speech, and video), four languages (English, German, Italian, and Chinese), and 13 tasks grouped into four macro categories: recognition, translation, question answering (QA), and summarization. The dataset includes short-form and long-form contexts, enabling the study of model performance across input complexity.

### Strengths
1. The paper tackles a critical gap in evaluating instruction-following across multiple languages and modalities - an under-explored but essential part of MLLMs.

2. MCIF is carefully constructed with parallel multimodal and multilingual alignment, allowing systematic and fair cross-dimensional comparisons—something absent in prior benchmarks. Using human-curated data from scientific talks ensures higher linguistic and contextual richness than synthetic or crowdsourced datasets. The academic domain provides realistic, challenging material.

3. It conducts comprehensive evaluation over 23 diverse models (open and closed-source, text-only and multimodal) under both short- and long-form settings provides broad insight into model generalization and limitations. And it also introduces MCIF_fix vs. MCIF_mix is a thoughtful addition—it measures how sensitive models are to prompt variations, reflecting real-world user interaction diversity.

### Weaknesses
1. Scale issue: despite being well-curated, MCIF covers only ~10 hours of content and 21 talks, which may limit statistical robustness and task diversity.

2. Human baseline: it is better to establish a human baseline, making it clear how far current systems are from “human-level” multimodal understanding.

3. Evaluation metrics: Automatic evaluation on WER, COMET, and BERTScore may not well capture multimodal grounding or factual consistency - especially for tasks like question answering or summarization.

### Questions
1. How you control the quality to ensure annotation consistency across languages and modalities, especially for translations and Q&A? 

2. Can you elaborate on whether degradation in long-form tasks is primarily due to model architecture (context window limits) or instruction misalignment? or any other potential reasons?

3. For the variation shown between MCIF_fix and MCIF_mix, do you suggest any strategies to improve real-world prompt robustness?

### Soundness
3

### Presentation
3

### Contribution
3
