# HealMQA: A Healthcare Multimodal Question Answering Dataset for Benchmarking Large Language Models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Consumers increasingly rely on digital platforms to seek healthcare advice, yet existing medical question answering (QA) resources are primarily unimodal or professional-facing, limiting their relevance to real-world users. To address this gap, we introduce Consumer Healthcare Multimodal Question Answering (CHMQA), a novel task for generating clinically valid free-text answers to multimodal consumer health queries. To benchmark this task, we present HealMQA, an expert-annotated multimodal QA dataset, consisting of 1,022 real-world consumer questions paired with medically validated images and expert-written answers spanning 17 healthcare topics. We evaluate eight state-of-the-art large language models (LLMs) under zero-shot and few-shot prompting methods, complemented by automated metrics and human evaluation by licensed medical professionals. Our experiments reveal that while LLMs achieve strong performance in accuracy, clarity, and safety, they continue to struggle with effectively integrating the visual modality. These findings highlight both the promise and limitations of current systems, and position HealMQA as a foundation for advancing medically accurate and consumer-centered multimodal healthcare QA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CHMQA, a visual question answering task that focuses on free-text answering of consumer health questions. To study this setting, the authors curate HealMQA, a benchmark dataset built from real user questions that annotators rephrase to reference an image, pair with a relevant medical image retrieved from authoritative websites, and answer in clinician-written sentences across 17 topics. The paper benchmarks eight multimodal LLMs under zero- and few-shot prompting using BERTScore, SacreBLEU, ROUGE, and a small human study by two physicians. Results suggest decent textual quality and safety but weaker image-text integration among tested systems.

### Strengths
1. The problem itself is interesting, targeting an emerging, high-impact use case: consumer health queries that often mix vague text with a clarifying photo. 
2. The dataset design is mostly sound with clear annotation steps, yielding examples that resemble what end users might ask multimodal assistants. 
3. The human evaluation, while small, adds a clinically meaningful lens that automated n-gram metrics cannot provide. 
4. The paper is well-written and easy to follow.

### Weaknesses
1. I am concerned about the pairing between the image and the real scene behind the user's description in the proposed dataset. The image is not user-provided, but retrospectively retrieved to match a textual visual term. This creates **two layers of distributional shift**: (1) the real scene may contain key features that are not described by the users due to their non-professionalism. (2) the image retrieved via Google search may not reflect the user’s description. Such weak alignment makes the HealMQA dataset less suitable for benchmark curation which poses higher standard for validated data than training data curation. 

2. Is there an ablation study to quantify the potential image-query misalignment above? For example, start the evaluation with real user image-text pairs and then replace the paired images with ones retrieved from Google and see how the evaluation results will change.

3. Inter-annotator “agreement” via BERTScore on a small pilot is an unusual proxy that conflates style and semantic overlap and can reward superficially similar wording. And I am not sure if 0.7257 should really be seen as “high level similarity and hence, agreement”. This is a potentially large gap, particularly in the medical scenario. A more feasible solution is to use clinically oriented rubric agreement (e.g., structured checklists for differential, red-flag triage, etc) and inter-rater reliability on those rubrics (κ), not just embedding similarity. The current claim of high agreement feels weakly supported.

4. Is there any data leakage detection for the benchmark dataset? Note that all images and texts are directly from the Internet and they might already be consumed by existing MLLMs trained on web-crawled data, making the evaluation results less valuable.

5. Evaluation under-emphasizes visual grounding. The automatic metrics are text-to-text overlap and do not verify whether an answer actually uses image evidence correctly. Although the human study notes “multimodal consistency” as the weakest dimension, again, the experiment is too small (20 items, 2 raters) to draw confident conclusions. 

6. I am also concerned about licensing and provenance. “Open access” or preference for .edu/.gov domains is not a license. It is unclear whether images are redistributed, and whether usage complies with original site terms and patient privacy constraints. 

7. Some typos. For example, in Line 200, “With HealMQA, we the creation...”. In Line 261, “addresses the user’s.”

### Questions
Will this benchmark dataset be open-source?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HealMQA, a clinical expert annotated multimodal question answering dataset designed for consumer healthcare queries. It consists of more than 1k real healthcare questions from online sources along with images retrieved from Google and expert written answers across several healthcare domains, primarily focusing on skin conditions. The authors define a new task, Consumer Healthcare Multimodal Question Answering - CHMQA, to evaluate how AI systems handle multimodal queries in a generalized setting. They benchmark eight LLMs, including GPT-5 and open-source models like Llama and Mistral, under zero- and few-shot settings, using automated metrics (BERTScore, ROUGE, BLEU) and human evaluation by medical professionals. Results show that LLMs produce generally accurate and clear answers but often fail to integrate visual information effectively.

### Strengths
- The paper introduces the CHMQA task, a novel and consumer focused multimodal question answering setup that addresses the lack of datasets combining real world consumer facing health queries with visual inputs, bridging the gap between professional medical QA systems and everyday user needs.
- It includes manual annotation and validation by licensed medical experts, providing clinically grounded answers and a structured evaluation of LLM outputs across key medical criteria, though the scale of expert involvement remains relatively limited.

### Weaknesses
- The fundamental problem with this benchmark is that there is no relationship between the image and the user query. The quality of the benchmark would simply depend on the quality of the images retrieved by the annotators. Obtaining exact images that accurately matches the user description is quite challenging and is not a scalable approach. This would also lead to unwanted biases in the benchmark where the same image from google could be used for multiple similar user queries. 
- The evaluations are not comprehensive enough, Table 2 is missing many state of the art open and close sourced models like – Gemini 2.5 Pro [1], Gemini 2.5 Flash and medical models like Google’s MedGemma[2], Huatuogpt[3], Bimedix2[4] and LLaVA-Med.
- Additionally models like Gemini 2.5 Pro and Gemini 2.5 Flash have muti-image support which helps them take advantage of the few-shot setting. Currently, as mentioned in A.3 models like Llama and mistral are evaluated by giving only the text input which does not incentivize them to use multimodal signals in the few shot setting.
- When evaluating model responses in an open-ended context, the most effective current approach is to employ an LLM as a judge. Alternative methods, such as embedding-based metrics or BERTScore, even those adapted for clinical domains, fail to fully capture the nuanced intricacies of natural language.

[1] *Comanici, Gheorghe, et al. "Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities." arXiv preprint arXiv:2507.06261 (2025)*

[2] *Sellergren, Andrew, et al. "Medgemma technical report." arXiv preprint arXiv:2507.05201 (2025).*

[3] *Chen, Junying, et al. "Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale." arXiv preprint arXiv:2406.19280 (2024).*

[4] *Mullappilly, Sahal Shaji, et al. "Bimedix2: Bio-medical expert lmm for diverse medical modalities." arXiv preprint arXiv:2412.07769 (2024).*

[5] *Li, Chunyuan, et al. "Llava-med: Training a large language-and-vision assistant for biomedicine in one day." Advances in Neural Information Processing Systems 36 (2023): 28541-28564.*

### Questions
Please address the above weaknesses. 
- What instructions were given to the annotators for the benchmark creation, please include detailed instructions in appendix to ensure transparency and reproducibility.
- Line 106: “We provide a detailed human analysis of LLM” use correct article “a”.
- Line 202: “With HealMQA, we the creation of a widely useful” please correct the grammar.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces HealMQA, a benchmark dataset for consumer-healthcare multimodal question answering (CHMQA), featuring around 1k real world user questions paired with medically validated images and expert-written answers. The dataset is positioned as bridging the gap between unimodal medical QA datasets and VQA style tasks. The authors benchmark eight large language models under zero/few‐shot settings.

### Strengths
- The topic is timely and socially relevant given the growing use of foundation models for healthcare advice. 

- The dataset is carefully annotated by medical professionals and covers diverse consumer health topics, enhancing realism and safety awareness. 

- The paper is well written, provides a clear task definition and methodology, and complements automatic metrics with human evaluation.

### Weaknesses
- The technical novelty and dataset scale are limited. HealMQA contains only about 1 k examples, which is small compared to recent multimodal medical benchmarks.

- The authors do not sufficiently compare or position HealMQA relative to large and diverse multimodal datasets that already address similar challenges at greater scale and modality coverage. For example:
-- EHRXQA (Bae et al., NeurIPS 2023): integrates textual EHR data and chest X-ray images for multi-modal question answering on structured clinical data.
-- MedTrinity-25M (Xie et al., arXiv 2024): provides 25 M multimodal medical pairs across 10 modalities with multi-granular annotations.
-- WorldMedQA-V (Matos et al., Findings NAACL 2025): a multilingual, multimodal benchmark for medical reasoning and LMM evaluation across diverse clinical domains.

- HealMQA lacks comparison with these stronger baselines and does not clarify its unique contribution beyond the “consumer-health” focus.

- The image acquisition process is insufficiently documented, raising reproducibility and bias concerns. 

- The evaluation focuses on generic language metrics that poorly reflect multimodal comprehension, with limited analysis of grounding, hallucination, or safety violations.

- Finally, the dataset’s scale and diversity are inadequate for benchmarking large multimodal models, and the paper does not provide clear evidence of multimodal benefit (i.e., how much images improve text-only performance).

### Questions
Add ablations showing how multimodal input (image + text) improves over text-only baselines.

Address copyright, licensing, and bias issues in image sourcing.

Provide a detailed ethical statement covering dataset release, privacy, and intended-use limitations.

### Soundness
4

### Presentation
3

### Contribution
1
