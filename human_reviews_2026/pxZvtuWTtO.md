# WildSpeech-Bench: Benchmarking End-to-End SpeechLLMs in the Wild

- Decision: Reject
- Scores: 4, 6, 6, 4, 6

## Abstract
Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we introduce the first comprehensive benchmark designed to systematically evaluate end-to-end speechLLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces WildSpeech-Bench, a new benchmark for evaluating end-to-end (E2E) Speech-to-Speech (S2S) Large Language Models (SpeechLLMs). The authors claim that existing benchmarks are ill-suited for this task, as they are often adapted from text-based benchmarks, evaluate Speech-to-Text (S2T) output, and use unrealistic, clean, synthesized audio queries .

The paper's primary contributions are:

- A collection of 1,100 queries. This set is composed of 1,000 "general" queries sourced from the WildChat dataset (real-world user-ChatGPT text interactions) and 100 "Paralinguistic-Featured" (PF) queries that were manually crafted and recorded by humans to test nuances like pause, stress, and tone . The audio for all queries is augmented with "Human Noise" and "Natural Noise" to simulate real-world conditions .

- The paper proposes an automatic evaluation method where an LLM (GPT-4o-mini) scores the model's transcribed response. To improve accuracy, this LLM-judge is guided by "meticulously hand-crafted checklists" specific to each query, which are designed to assess factual, structural, and semantic criteria .

### Strengths
- The concept of a manually crafted, human-recorded set of queries to test nuanced paralinguistic understanding is new.
- Using per-query, hand-crafted checklists to guide LLM-judges is a clever way to improve reliability and reduce the ambiguity of generic prompts.
- The systematic inclusion of both human and environmental background noise is a good feature, pushing models beyond idealized "clean room" performance .

### Weaknesses
- The benchmark's core claim is evaluating SpeechLLMs "in the wild". This claim is not met for two fundamental reasons:
  - The benchmark is 100% focused on single-turn evaluation. Real-world, "in-the-wild" speech conversations are almost never single-turn. They are interactive and multi-turn. This benchmark completely fails to evaluate core conversational abilities such as context maintenance, coreference resolution, or managing a coherent dialogue over time.
  - 90% of the benchmark (1,000 out of 1,100 queries) is not human-recorded but synthesized using voice-cloned TTS. While diverse profiles are used, this is still synthesized speech, not authentic human speech "in the wild." The only truly "wild" acoustic data is the 100-item PF set.

- The dataset contains only 1,100 queries in total. This scale is insufficient for a robust and generalizable benchmark. The authors' justification (comparing it to MT-Bench's 80 queries) is not persuasive. Evaluating the vast, multi-faceted domain of S2S interaction (including acoustic robustness, paralinguistics, and content quality) requires a much larger sample. This small, fixed set is highly susceptible to "teaching to the test," as the authors themselves acknowledge . Models can easily be fine-tuned to perform well on this specific set of 1,100 queries and their corresponding checklists, rendering the benchmark obsolete quickly.

- The paper's evaluation methodology contains a contradiction.
  - It criticizes prior benchmarks for "focus on evaluating the text output rather than the generated speech".
  - However, its own primary evaluation framework does exactly that. The model's audio response is transcribed by Whisper-large-v3, and this text transcription is then scored by the GPT-4o-mini judge .
  - This is, by definition, a Speech-to-Text (S2T) evaluation, albeit a more complex one using an LLM-judge. The paper fails to explain how an LLM reading text can assess "speech-centric criteria like clarity... and tonal appropriateness". The paper even dedicates an entire appendix (Appendix E) to demonstrating how ASR errors make evaluation unreliable , yet it builds its entire scoring system on top of an ASR system.

### Questions
- How could you claim this benchmark evaluates models "in the wild" when it contains zero multi-turn conversations, which are the defining characteristic of "in-the-wild" interactions?
- The paper's central criticism of other benchmarks is their reliance on evaluating text output. How do you justify that your own primary evaluation method—which transcribes speech to text and has an LLM judge that text —is not guilty of the exact same flaw?
- How does your text-based LLM-judge assess "speech-centric criteria" like "tonal appropriateness"? Are you claiming the ASR transcription captures this, or is this specific criterion simply not evaluated by the LLM-judge?
- Given the small, fixed scale of 1,100 queries and their hand-crafted checklists, what prevents model developers from simply fine-tuning their models on your benchmark data, leading to inflated scores that do not reflect true general-purpose S2S capability?

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
This paper argues that existing evaluation methods for large speech language models often rely on text-based benchmarks, thereby overlooking the unique characteristics and challenges inherent to spoken language—such as prosody, homophones, stuttering, and divergent user expectations. To address this gap, they introduce the first systematic benchmark specifically designed to evaluate speech-to-speech dialogue systems in realistic conversational scenarios. They detail the full pipeline for constructing this benchmark and propose a novel approach that leverages tailored evaluation checklists to enhance the accuracy of automated assessments. Experimental results demonstrate that the automated evaluation correlates well with human judgments, and further conduct ablation studies to validate the effectiveness of our strategy for injecting noise into user queries.

### Strengths
1. The authors present the first comprehensive benchmark designed to systematically evaluate end-to-end SpeechLLMs in practical speech-based conversations, along with an effective pipeline for constructing such benchmarks.
2. The authors propose a query-aware evaluation method that leverages customized evaluation checklists and prompts to improve the accuracy of automatic evaluation, and demonstrate that this approach aligns more closely with human judgments.

### Weaknesses
1. During evaluation, the authors adopted a checklist-based approach to provide the large language model with comprehensive reference information and then asked it to score model responses based on this information. If that’s the case, why not explicitly check each item in the checklist individually and assign scores based on whether each criterion is met? This may yield more precise and reliable evaluation results.

2. The current categorization of scenarios in the dataset is not yet sufficiently clear. Providing more explicit and well-defined scenario categories would help clarify the intended application scope and strengths of this benchmark.

3. In the naive pipeline, it's great to include additional results where the ASR model is replaced with a captioning model for comparison.

### Questions
Please refer to weaknesses.

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
This paper introduces WildSpeech-Bench, a novel benchmark designed for the end-to-end evaluation of Speech Large Language Models (SLLMs) in real-world, conversational scenarios. The authors identify significant gaps in existing benchmarks, such as an over-reliance on text-based queries, a lack of acoustic diversity, and evaluation methods that fail to capture the nuances of spoken language. To address this, WildSpeech-Bench is constructed by curating realistic queries from actual user-ChatGPT conversations (via the WildChat dataset) and generating speech data with diverse speaker attributes (age, gender, timbre) and background noise conditions. Speacially, authors proposed "Paralinguistic-Featured" queries to test a model's ability to handle speech-specific challenges like stuttering, prosodic ambiguity, and homophones. The benchmark also proposes an automatic evaluation protocol that uses query-aware checklists and LLM-based semantic scoring.

### Strengths
The benchmark is built upon a dataset of real user interactions (WildChat), ensuring the queries reflect genuine spoken language use cases and user behavior.

The benchmark systematically incorporates paralinguistic phenomena and acoustic challenges that are largely ignored by text-centric benchmarks, providing a more comprehensive test of model capabilities.

### Weaknesses
1. The benchmark is relatively small (1100 queries, with only 100 for paralinguistic features) and relies heavily on synthesized speech from only two base speakers. 

2. The benchmark focuses exclusively on single-turn interactions, omitting the critical aspect of multi-turn conversations, which is a fundamental element of conversational AI and a key use case for voice assistants.


3. Both the dataset curation and the core evaluation protocol depend heavily on another LLM (GPT-4o-mini). This replaces rigorous scientific validation with automated prompts, raising serious concerns about the reliability, accuracy, and circularity of the benchmark's judgments.

### Questions
Overall, the existing evaluation data is too limited and has many sub-categories, thus slightly data leakage could make the benchmark invalid. Is there a efficient way to generate more data to make the benchmark more solid?

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
The paper introduces WildSpeech-Bench, a benchmark tailored for evaluating end-to-end SpeechLLMs (speech-to-speech models). Unlike prior S2T-oriented benchmarks (e.g., VoiceBench, OpenAudio, SD-Eval), this work constructs a dataset of 1,100 curated speech queries spanning five categories (Information Inquiry, Solution Request, Opinion Exchange, Text Creation, and Paralinguistic-Featured). It employs both synthetic and human-recorded audio, includes multi-tiered noise augmentation, and proposes a query-aware evaluation framework that improves automatic alignment with human judgments. Experimental results compare major proprietary and open-source models (GPT-4o-Audio, Qwen2.5-Omni, GLM-4-Voice, etc.), showing that current systems degrade sharply under noise and paralinguistic variation.

### Strengths
1. The paper propose the evaluation of end-to-end SpeechLLMs, which are rapidly emerging but lack standardized benchmarks.
2. The benchmark construction is well-motivated and systematic, combining real user queries from WildChat, voice cloning, paralinguistic diversity, and noise realism.
3. The use of customized, query-specific rubrics improves correlation with human judgments (Pearson r = 0.86), a clear methodological improvement over prior automatic evaluations.
4. The comparison of multiple leading models and ablation studies (multi-round ASR, noise effects, etc.) are thorough and informative.
The reproducibility and LLM-usage statements are commendable. Figures and tables are well organized.

### Weaknesses
1. The dataset (1.1 k samples) is small relative to modern benchmarks. While quality is prioritized, the paper lacks analysis of statistical coverage or how well this scale generalizes across domains or accents. Expansion beyond English is critical.
2. Despite inclusion of 100 human recordings, most data come from TTS (CosyVoice), which constrains prosodic and emotional realism. Results may overestimate model robustness to human variability.
3. Although the authors mitigate transcription bias via multi-round ASR and query-aware scoring, evaluation still depends on text-based scoring. A truly end-to-end audio–audio metric (e.g., perceptual similarity or content alignment) is absent.
4. Potential circularity in LLM-based evaluation. The benchmark uses GPT-4o-mini as the judge, while GPT-4o-Audio is also a tested model—raising potential bias in favor of OpenAI systems. The paper should quantify or control for evaluator–model overlap.
5. Limited discussion of error analysis and ethical aspects. While a few case studies are shown, the error taxonomy is shallow. No discussion of dataset bias (e.g., cultural, gender, or accent representation) or privacy in using WildChat data.
6. While human scoring is mentioned, the setup (e.g., annotator consistency, inter-rater reliability, audio listening conditions) is relegated to Appendix B and not summarized quantitatively.

### Questions
1. Were the differences in Table 2 tested for significance (e.g., paired t-tests or bootstrap confidence intervals)?
2. Without statistical testing, how confident can we be that GPT-4o-Audio’s reported advantage is not due to noise in small-sample evaluation?
3. Did you analyze whether performance degradation under noise correlates with specific frequency bands or noise types (e.g., human vs. ambient)?
4. Would fine-grained error analysis help isolate which components of end-to-end models (encoder vs. decoder) are most vulnerable?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce WildSpeechBench, a benchmark to evaluate spoken language models (SLMs) on realistic audio. The benchmark is developed by extensively filtering human-AI conversations and synthesizing them into audio with a TTS system. To better reflect real-world audio, acoustic noise is added to each synthetic speech sample. The authors complement the synthetic speech by recording real human voice actors on a subset of the data. Finally, the authors benchmark a variety of SLMs on the data, showing that leading open-source models significantly lag behind proprietary ones.

### Strengths
- The authors develop a pipeline to create test samples with life-like synthetic speech by combing TTS and noise
- The authors complement the synthetic samples with real recorded speech
- The authors show that SOTA SLMs exhibit significantly degraded performance when processing samples with noise, showing their fragility despite strong reported results

### Weaknesses
- The set of benchmarked systems is relatively small, missing notable models like Moshi
- My main concern is the lack of evaluation on the realism of the synthetic noisy audio. All of the results of this paper are dependent on this factor. Since real speech is already used for a subset, why not synthesize another small subset with the real-speech as the speaker prompt? Then you can have a direct comparison of the impact of real speech vs synthetic noisy samples.

### Questions
>  revealing that strong performance on S2T dialogue benchmarks does not
translate to S2S settings

I believe the benchmarked models all support text output. In that case, obtaining S2T versions of the model outputs should be feasible. I think it would be beneficial to have a comparison between the S2T results and S2S results to rigorously validate this claim

### Soundness
3

### Presentation
3

### Contribution
3
