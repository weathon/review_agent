# EverydayMMQA: A Multimodal and Multilingual Framework for Culturally Grounded VQA

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they often fail when queries require culturally grounded, everyday knowledge, particularly in low-resource and underrepresented languages. To bridge this gap, we introduce **Everyday Multimodal and Multilingual QA (EverydayMMQA)**, a framework for creating large-scale, culturally grounded datasets for spoken and visual question answering (SVQA). Using this framework, we developed **OASIS**, a multimodal dataset integrating speech, images, and text. With over ∼0.92M images and 14.8M QA pairs, OASIS contains 3.7M spoken questions, enabling four unique input combinations: speech-only, text-only, speech+image, and text+image. Focused on English and Arabic varieties across 18 countries, the dataset content is curated to reflect diverse, real-world situations. OASIS tests models on tasks beyond object recognition that involve pragmatic, commonsense, and culturally aware reasoning. We benchmarked four closed-source models, three open-source models, and one fine-tuned model. EverydayMMQA and OASIS together provide both a benchmark and a training dataset for building multimodal LLMs capable of handling a comprehensive set of everyday tasks within cultural contexts. The framework and dataset will be made publicly available to the community.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new benchmark and framework for generating multilingual and culturally relevant multimodal QA. The resulting dataset, OASIS, covers 18 countries and multiple modalities. The authors also evaluate several existing models on this benchmark.

### Strengths
The motivation is strong and addresses an important gap in multimodal and multilingual QA research. The resulting dataset is large-scale and has the potential to be highly beneficial for the research community.

### Weaknesses
I’m concerned about the direction of this data. If the aim is to have a multimodal QA mainly focusing on visual, then we should expect the model to get ~0 performance if the question is only given in text. i.e., there is no image so you can’t really answer the question (as mentioned in l36 and the example in Figure 1)

But, we see non-zero text-only performance, which means some questions are solvable without the image anyway, hence making this data not necessarily VQA. If the image is supposed to be an additional modality that can be utilized but not mandatory for answering, then this should be stated more explicitly without contradicting the examples (Figure 1). This should also be explicitly designed both in QA generation and human validation.

The implication on the improved result for T+I modality perhaps is not because of the “language burden” as stated. The boost may simply be because some questions require the image to be answered (again, as in Figure 1).

Another concern is on the LLM-generated data for benchmarking. The data is validated somehow (i.e., Likert scale), but it is not detailed enough, especially for machine-generated data. The Likert scale is not clear on what it measures (question difficulty? question quality? what is the definition of “quality”?). Providing the annotation guideline, including the exact Likert scale used, would be useful. Moreover, clarity is needed because of multiple aspects and caveats of AI-generated questions, e.g:
 - is the question correct? do we have a score on this?
 - is the question culturally relevant at all?
 - is the generated language correct? especially in dialectal Arabic and low-resource languages, we need confirmation that LLMs can indeed write the question naturally.
 - is the QA according to the expectation, i.e., requires the image to be answered (see previous point).

While the human agreement is high, it would not be very informative if the definition of what is being measured is unclear.

### Questions
- Missing important details, e.g., l118 → how many were filtered or removed? I assume it is not removed since you want to keep the size of 310; in that case, how many were replaced?
- l130 → “which results in approximately 10 to 86 queries per topic.” The variance is extremely high; more discussion or details would be helpful (why is this the case, which dialect is affected, and is this due to LLM weakness on the task or something else?). Subsequently, how trustworthy is this process given such high variance?

Missing citation:
- In related works, datasets were discussed but not cited: Maya and PALO.
- Some other multilingual efforts for multimodal, cultural VQA: CVQA, SeaVQA.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes EveryDayMMQA, a framework for building culturally grounded multimodal question-answering datasets, which consists of (i) image collection/filtering (ii) metadata and QA generation, (iii) speech generation/recording (iv) translation and (v) quality assurance. The authors use the proposed framework (EveryDayMMQA) to build OASIS, which is a large multimodal dataset that integrates speech, images and text, focused on english and Arabic, covering 18 countries. The proposed dataset supports four inputs: text, speech, text+image and speech+image, covering different type of questions, Open-ended, Multiple choice and True/False. With the proposed dataset the authors perform a variety of experiments with open and closed source models, additionally they finetune one model and show that this achieves a higher accuracy on cultural knowledge, specially when the question is image-grounded.

### Strengths
Paper Strenghts:
1. The paper extends the common modalities of image-text in cultural-related resources for VQA, and add speech, targeting Arabic dialects alongside English. 
2. With the proposed pipeline the authors are able to build Oasis, a large scale multimodal dataset providing a testbed for multimodal and culturally diverse evaluation. EveryDayMMQA helps for gathering culturally related data, which is usually difficult, specially for low-resource languages.
3. Broad evaluations with open and closed-source models.

### Weaknesses
Paper Weaknesses:
1. Section 2 would benefit for a brief general description of the method before describing it in detail, otherwise is just a bit complicated at first to understand why you are doing each part. On the other hand, when each part is described like in "Topic Generation and Filtering" or "Query Generation and Filtering", please refer where the reader can see those things that you are describing, like in what part of the paper the reader can see those topics or queries, this makes it easier to understand.
2. In Section 2 "Query Generation and Filtering" the authors say that they use "GPT-4o to assign a cultural-relevancy score, which reflects the cultural fit". Page 19 shows the prompt used for this, which contains: "Ensure that the topics reflect the cultural, historical or modern significance of the specified location". It's well know that VLMs lack cultural knowledge even the biggest closed source models from the GPT family, multiple works benchmark them in cultural-related VQA, for example [1] CVQA, so using them to give a cultural-relevancy score seems to not be really appropriate.
3. Section 2.2 describes the process to get images via country-localized search using Google Custom Search, in my personal experience getting images related to culture for low-resource cultures is hard, which can be the case of Arabic countries, which is why some works get images directly from annotators like in [1]. Perhaps there were difficulties related to this?
4. More examples of dataset questions would benefit the paper.
5. Section 2.6 describes translation from English into MSA, it is also known [1] that using translation fails to capture cultural nuances inherent in different languages, was this considered ?.  

[1] https://arxiv.org/pdf/2406.05967

### Questions
Please refer the weaknesses for questions and doubts. Overall, I would like to see the authors responses in using VLLMs/LLMs to do a dataset related to culture, when is well known that these models still lack cultural knowledge. Sorry if I misunderstood something, currently I'm giving borderline reject, but after rebuttal I will revise my decision.

### Soundness
2

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
5

### Summary
The paper proposes "EverydayMMQA," a framework for generating large-scale, multimodal, multilingual question-answering datasets. Using this framework, the authors create "OASIS," a dataset with approximately 0.92 million images and 14.8 million QA pairs focused on English and several Arabic varieties, covering 18 Arab countries. A main feature of OASIS is the inclusion of spoken questions (both synthetic and human-recorded), enabling evaluation across four input modalities (speech-only, text-only, speech+image, text+image). The authors benchmark several closed and open-source models on OASIS and conduct a fine-tuning experiment. Their main findings suggest that visual grounding is the most critical factor for improving performance and that fine-tuning on their dataset enhances a model's cultural awareness.

### Strengths
- The paper's goal is certainly well-motivated. The lack of culturally diverse, spoken-language multimodal benchmarks is a clear gap.

- The development of the EverydayMMQA framework is a valuable contribution on its own. By modularizing the process into distinct stages (e.g., query generation, image retrieval, QA generation, quality control), the authors provide a scalable and reusable methodology that other researchers can adopt for creating similar resources for different cultural or linguistic contexts.

- The inclusion of spoken questions, both synthetic and human-recorded, is a major differentiator from most existing VQA datasets. This enables a much more comprehensive evaluation of "omni" models that are designed to handle multiple input modalities simultaneously.

- In experiments, authors test models across four different input combinations, which allows for a clear analysis of the impact of each modality. The conclusion that visual grounding acts as an "equalizer" that mitigates the performance loss from speech input and ASR errors is a particularly strong and interesting result. The fine-tuning experiment also effectively demonstrates the utility of the dataset for improving model performance.

### Weaknesses
- The framwork of the data construction appears to be a description of a specific, multi-stage data collection pipeline that relies heavily on a series of LLM API calls. It is not clear what makes this a generalizable or reusable "framework" beyond the specific implementation for this paper. The contribution seems to be the dataset itself, and framing the pipeline as a separate, novel framework feels like an overstatement.

- A major concern is the almost complete reliance on automated generation for all critical components of the dataset. The cultural topics, search queries, image descriptions, and the QA pairs themselves are all generated by large language models. The paper argues it is addressing the "Western-centric bias" of existing resources, but it does so by using the very same large, predominantly Western-trained models to generate its "culturally grounded" content. It is highly questionable whether a model like GPT-4 can genuinely produce authentic cultural nuances for 18 different Arab countries. This methodology risks creating a large dataset of plausible-sounding but ultimately synthetic and stereotyped content, rather than a truly culturally grounded resource. The "quality check" also relies on LLMs, which compounds this issue.

- The vast majority of the spoken data (~20,000 hours) is synthetic, generated by a TTS model. The paper’s own analysis (Appendix A.3) shows that this synthetic speech is much "cleaner" and easier for an ASR model to process than the small amount of human-recorded speech. This means the benchmark is not truly evaluating performance on real-world "spoken" QA, but rather on "TTS-based" QA. The conclusions drawn from the speech modality are therefore of limited applicability to realistic, noisy human-computer interaction scenarios.

- The main conclusions from the experiments are not particularly surprising. It is expected that providing an image for a visual question (the T+I condition) would drastically improve performance over text-alone (the T condition). This finding, while true, is largely confirmatory. More concerning is the fine-tuning experiment. The authors fine-tune a model on a subset of the automatically generated training data and show that its performance improves on the automatically generated test set. This result is at high risk of being circular. It may primarily demonstrate that the model has learned to replicate the specific patterns, style, and potential artifacts of the data generator, rather than acquiring genuine cultural or multimodal reasoning abilities.

- Furthermore, this paper does not cite two very relevant papers, authors must explain why:

Romero, David, et al. "CVQA: culturally-diverse multilingual visual question answering benchmark." Proceedings of the 38th International Conference on Neural Information Processing Systems. 2024.

Vayani, Ashmal, et al. "All languages matter: Evaluating lmms on culturally diverse 100 languages." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
N/A

### Soundness
4

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
The paper presents EverydayMMQA, a systematic and scalable framework for building multimodal, multilingual, and culturally grounded QA datasets. OASIS, the resulting resource, integrates 0.92M images, 14.8M QA pairs, and 3.7M spoken questions, spanning English, Modern Standard Arabic, Egyptian, and Levantine Arabic. Four unique input modes are supported: text-only, speech-only, text+image, and speech+image. The dataset is annotated for cultural nuance, with QA pairs grouped by knowledge vs. commonsense reasoning, semantic domain, and cognitive focus. Robust benchmarking of state-of-the-art closed (e.g., GPT-4.1, Gemini 2.5 Pro) and open (e.g., Qwen2.5, Phi-4) models demonstrates significant gains.

### Strengths
1. EverydayMMQA stands out through its tri-modal design (text, image, and speech) and explicit attention to diverse cultural contexts across the Arab world, including multiple dialects.  
2. The framework involves LLM-driven topic/query/image filtering, human-in-the-loop validation, manual and LLM-based annotation, and multi-layered quality control.  
3. Finetuning on a smaller subset of OASIS shows improvement in model performance, indicating its usability as a training dataset.

### Weaknesses
1. The authors rely on GPT-4o for assigning cultural relevance scores; however, they do not study how effective it is at this task. This overreliance on GPT-4o's capability in a non-English setting could be a problem.  
2. The authors tested the finetuned model on the same data source (OASIS). The paper would benefit from an understanding of the model's generalizability, for instance, if a model trained on OASIS were tested on different benchmarks.  
3. The analysis in Section 4 is good at dissecting performance by modality (T vs. S vs. T+I) and language (En vs. MSA). However, it stops short of a deeper qualitative analysis of the cultural dimension. Even with visual grounding (T+I), models are not perfect. The paper would benefit if it included examples of culturally nuanced questions that the best models still get wrong.

### Questions
1. Line 72 states there are 16M QAs, which contradicts the 14.8M QA pairs mentioned in the abstract. Please clarify this discrepancy.  
2. Given the paper's focus on dialectal diversity, are there plans to expand the spoken dataset to include Arabic dialects?  
3. Could you provide a qualitative error analysis for the best-performing models (e.g., GPT-4.1 with T+I)? What specific types of cultural, pragmatic, or commonsense reasoning do they still fail at, even with visual grounding?  
4. Can you please clarify the dataset size? Is the 14.8M figure the total number of text QA pairs (i.e., 3.7M unique QA sets $\times$ 4 language varieties)?

### Soundness
3

### Presentation
3

### Contribution
3
