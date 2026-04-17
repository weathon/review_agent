# MILA (MULTILINGUAL INDIC LANGUAGE ARCHIVE): A DATASET FOR EQUITABLE MULTILINGUAL LLMS

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large Language Models (LLMs) are structurally biased toward high-resource languages like English due to corpus skew, a problem particularly severe for Indic languages. To address this deficit, we introduce ***MILA***, the largest expert-curated Indic corpus to date, comprising ***7.5 trillion tokens*** across ***16 scheduled Indic languages*** and English. MILA is constructed via a multi-stage data engineering pipeline that integrates large-scale ***web acquisition***, script-sensitive ***OCR*** for under-digitized Indic writing systems, LLM-assisted post-correction for ***Translation*** fidelity, and ***targeted data distillation*** through the ***Indic-Persona Hub***. The pipeline further incorporates ***synthetic augmentation and rewriting***, followed by stringent ***quality, toxicity, language, and deduplication filtering***, and culminates in ***human-in-the-loop linguistic*** and cultural validation with comprehensive ***PII redaction*** and ensuring ***downstream task and benchmark-based decontamination***. This pipeline yields a distributionally stable, contamination-controlled, high-fidelity pretraining substrate. Alongside, we release ***Indic-MMLU***, a translated and verified adaptation of MMLU into 16 Indian languages, offering the first large-scale Indic multilingual benchmark for assessing LLMs and their extent of cross-lingual knowledge transfer. We further propose a ***Parity-based fairness Metric*** capturing cross-lingual performance asymmetries relative to English. Comprehensive experiments including controlled ablations of translation quality, OCR incorporation, synthetic SFT generation, and continual pretraining demonstrates that models trained on MILA achieve substantial gains on Indic-MMLU and materially narrow cross-lingual disparities. Collectively, MILA, Indic-MMLU, and the associated validation protocols establish a scalable foundation for equitable multilingual modeling in the Indic context. All resources are released anonymously for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors create MILA, the largest Indic multilingual dataset. They combine existing multilingual datasets and construct OCR pipelines specific to Indic scripts. The authors create MILA, the largest Indic multilingual dataset to tackle the major imbalance in LLM training data for Indic languages. The dataset is built using a multi-step process that includes combining existing datasets, OCR pipelines specific for Indic scripts (including comparison with existing VLMs, and creating  synthetic OCR benchmark), LLM-assisted translation correction (with human evaluation), synthetic data generation, data distillation, and expert linguistic validation. This ensures both high quality and cultural authenticity. In addition to the dataset, the authors release Indic-MMLU, a multilingual benchmark that translates and verifies the popular MMLU test into 16 Indian languages. Through extensive experiments, models trained on MILA show better results on Indic-language tasks, reducing the performance gap with English.

### Strengths
- The authors have released the dataset publicly which will be really beneficial for the community.
- The authors built a very thorough pipeline for performing OCR for Indic scripts. The kind of problems they discussed are very relevant to what researchers face when scanning documents. Their comparison with existing VLMs, and creating synthetic OCR benchmark further validates their approach.
- There are not many low-level details about the translation pipeline but the high level idea sounds good to create translation based data.
- Their instruction tuned model performs well compared to off-the-shelf models and with preprocessing and postprocessing it gets even better.
- By embedding diverse Indian identities and reasoning patterns, it goes beyond translation, ensuring language models are culturally relevant and capable of nuanced task performance.
- The results of parity metric and Indic MMLU look decent when compared to pretrained models.

### Weaknesses
- A huge chunk of the paper resides in the appendix. I’d request authors to bring more things from appendix to main paper otherwise readers will find it difficult to understand what’s going on. For instance:
  - In table 3, authors don’t clarify what do different models mean?
  - In section 3.5, what is the teacher model for distillation, what’s the final model, how many parameters it has etc.
  - Creation of Indic MMLU: In the abstract the authors mention we create IndicMMLU and then every mention of Indic MMLU in the paper cites MMLU paper which is super confusing. Later, I saw the pipeline of Indic MMLU is in the appendix which makes things super confusing.
  - In Section 3.4 translation pipeline: You had 3 language evals per language? Which LLM you used for inference. Do you have inter annotator agreement scores?
  - Details about Indic PersonaHub also lie in the appendix and I couldn’t completely understand what’s happening at a lower level.
  - To sum up, there is a lot going on in the paper and all of it is good but I feel the main paper gives just high level ideas and leaves implementation/low level details in appendix. The authors need to do a better job at presentation. 
- In Figure 4, what is low and high and which model did you train, Mistral or NLLB? I don’t see these models having comparable parameters so I doubt if it’s a fair comparison
- The formatting of the paper is quite weird in some places. The content beside table 1 is just independent and has no relation with preceding or succeeding paragraphs. In Line 158, after “Paullada et al…” comes “To tackle these” on Line 162 which looks confusing while reading and breaks the flow. Same problem after line 269, after “low-resource Indic..” comes “Evaluation including..” on Line 291.

### Questions
- In figure 1, please increase the font size of legends as they are blurred.
- The authors have cited a lot of multilingual corpora, however, I see they have skipped [1]. It contains 8T tokens covering 193 languages.
- Please change formatting of tables 3 and 4 to that of Table 5 
- Typos
  - Line 197- use \citet instead of citing Mendu et al. twice
  - Line 257, 431- I can see et al.(2025) citation
  - Line 330- x billion tokens?
  - Line 348- Table ??
  - Line 445, 446- use \cite and \citet properly

[1] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT) (Burchell et al., ACL 2025)

### Soundness
3

### Presentation
1

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides a new pre-training corpus and downstream benchmark datasets focusing on Indic languages. The dataset was curated from different sources including OCR of public domain books and expert translation of English MMLU dataset. paper then shows that performance on the MMLU benchmark is significantly improved for models pretrained on the developed corpus.

### Strengths
1. The corpus includes data from public domain books that was extracted using OCR. The books come from diverse domain that are culturally and practically significant for Indic language speakers.
2. Personally identifiable information was removed.
3. A synthetic persona based data generation focusing on Indic context was used to enhance the usefulness of this corpus for modern language model training.

### Weaknesses
1. The OCR pipeline depends on VLMs that were not specifically trained for Indic OCR. The risk of poor OCR quality has not been studied extensively. There should be human evaluations to determine the quality of the OCR'd texts. This is important as this component of the dataset (the OCRed books) are a major novelty of this work.
2. There is a translation component in the pipeline. The quality of those translations needs to be verified by human annotators. The paper mentions human evaluation but do not provide any human evaluation metrics such as number of annotators or inter-rater agreement. Existing translation models are known to be of poor quality for low resource languages. 
3. The synthetic rewriting portion of the pipeline is poorly described in the main paper. It should be described in the main paper in a concise form.
4. There are no statistics reported for how many tokens came from translation and synthetic rewriting.

### Questions
I see that there are some presentation inconsistencies between the main paper and the appendix. The appendix follows a more clear presentation format. Please follow that for the main paper. For example the prsentation of tables in the main paper should be similar to how tables are presented in the appendix.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MILA, a large-scale curated dataset of 7.5T tokens covering 16 Indic languages, built through web crawling, OCR of scanned books, translation pipelines, and synthetic persona-driven data generation. The paper also introduces Indic-MMLU, a multilingual benchmark, and shows that continual pretraining on MILA improves both absolute performance and fairness (parity) across Indic languages.

### Strengths
- The dataset covers multiple Indic languages, including several with limited existing digital resources, making it a valuable resource for model pretraining.

- The continual pretraining experiments demonstrate consistent performance improvements across languages relative to baseline checkpoints.

- The work employs a rigorous data cleaning and processing pipeline, including filtering, normalization, and OCR post-correction.

### Weaknesses
- The overall structure of the paper makes it difficult to clearly understand the pipeline and its distinct components. Methods and results are interleaved, which obscures the main contributions.

- There are no ablations isolating the impact of individual steps in the pipeline, making it difficult to determine where the performance gains originate. This limits the scientific insight of the work.

- While some artifacts (e.g., prompts) are provided, the pipeline remains hard to reproduce due to missing procedural and implementation details.

- The evaluation setup could be stronger. The baselines are limited, and the results do not comprehensively demonstrate improvements over established multilingual models.

- The experimental design is under-specified. For example, Table 2 presents results, but the paper does not describe the experiments beyond mentioning the dataset used.

- There is noticeable repetition in the writing (e.g., lines 178–179 restate lines 114–116), suggesting the paper would benefit from further editing for clarity.

### Questions
1. In line 118, is the intention to refer to Devanagari as a script rather than as a language?

2. In line 199, could you specify which PiiModifier implementation is used (e.g., version number or repository link) to support reproducibility?

3. In Table 2, what criteria distinguish “conventional” from “curated” data in your comparison?

4. In line 244, which dense model is being referenced, and how is “dense” being defined in this context?

5. For Table 4, since the goal is to show improvements from pre-/post-processing, can you also report results for the models in Table 4a where post-processing improved performance?

6. Which table is being referred to in line 348?

7. In Section 5, could you report additional evaluations beyond MMLU to demonstrate broader applicability?

8. In lines 445–447, could you clarify where the referenced tables/figures are located in the appendix? I couldn't find it.
\
Also, the comparison between older multilingual models (e.g., mT5, BLOOM) and monolingual models (e.g., LLaMA-2, Gemma) against Qwen-3 is unfair; could the paper justify or adjust the comparison to account for differences in training recency and model scale?

### Soundness
2

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
4

### Summary
This paper introduces MILA, the largest curated Indic multilingual dataset covering 16 of India’s 22 official languages and totaling 7.5 trillion tokens. With an attempt to reduce existing gaps in model performance for Indic languages, the authors present a multi-stage pipeline for data curation and validation, integrating large-scale crawling, OCR tailored to Indic scripts, synthetic augmentation, and rigorous filtering. They also release Indic-MMLU, a translation of MMLU into 16 Indian languages, and conduct continual pretraining on the new dataset, demonstrating some improvements in downstream performance.

### Strengths
- This dataset is a valuable contribution, providing a replicable pipeline for curating the largest Indic multilingual dataset across 16 official Indic languages.  
- The data curation process is thorough, involving multiple stages designed to preserve both data quality and overall utility, making the dataset useful for future research and model development.

### Weaknesses
- Since this dataset was developed for pretraining, it would have been helpful to include a discussion on potential data contamination from existing test sets on Hugging Face or other web sources. How was data contamination controlled during collection? There’s a chance that some test data could have been included during scraping. Given the relatively low MMLU scores, this may not have occurred, but it would still be valuable to see an explicit discussion of how this risk was mitigated.

- Could you expand more on any data provenance strategies used to ensure that potentially copyrighted materials were not included in the collected dataset? In the past, certain datasets have had to be taken down due to the inclusion of copyrighted or improperly licensed content.
- In the current draft of the paper, it is difficult to tell what drives the marginal improvements in model performance. The gains look somewhat positive, but it's not even clear how statistically significant they are. Even more so, one would have really expected to see more substantial improvements given the size and quality of the new data. If your newly trained model only improves on the Indic MMLU, it’s not obvious that the benefits justify the significant compute and curation effort. So your model may be better on more culturally grounded benchmarks. Is there a reason why you mostly benchmark Indic MMLU ? Aren’t there other Indic datasets that could be added to your evaluation suite?
- Why are there no comparisons to frontier models? It is honestly not my expectation that your new models should beat these models, but such comparisons can provide valuable context, helping us understand where your work stands relative to strong existing baselines and its potential practical impact.
- There are limited analyses or explanations on the correlations between the proportion of data for a given language in your datasets and the performance improvements observed after continual pretraining. One might expect that a language with more resources, like Hindi, would show substantial gains compared to the very low-resourced ones. However, the results suggest their performance is still fairly similar to the very low-resource languages.

### Questions
- In Line 183, you claim to employ in-house model-based quality classifiers (fastText) for all languages in your data at the document level. Could you clarify what constitutes “high quality” for Indian languages, especially for a pretraining dataset like yours? In previous work, people have quantified quality based on formality, nonsensical, or malformed text. 
- What size of the final dataset was translated? Could you provide more detailed stats on different data sources? 
- Did you adapt the Qwen tokenizer before continuing to pretrain ? Since Indic languages are mostly non-Latin script, do you have thoughts on how this might have contributed to the poor performance of your models even after adaptation?
- If compute wasn’t a limiting factor, an ablation that would have been really good to see is studying the impact of the different data sources on your downstream performance. Obviously, this would be compute-intensive and is not a weakness of your current paper. It could just provide insights into aspects of the data curation process that could be scaled further.

### Soundness
2

### Presentation
2

### Contribution
2
