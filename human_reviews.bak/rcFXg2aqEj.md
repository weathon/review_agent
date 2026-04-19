# LMDX: Language Model-based Document Information Extraction and Localization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 6

## Abstract
Large Language Models (LLM) have revolutionized Natural Language Processing (NLP), improving state-of-the-art on many existing tasks and exhibiting emergent capabilities. However, LLMs have not yet been successfully applied on semi-structured document information extraction, which is at the core of many document processing workflows and consists of extracting key entities from a visually rich document (VRD) given a predefined target schema. The main obstacles to LLM adoption in that task have been the absence of layout encoding within LLMs, critical for a high quality extraction, and the lack of a grounding mechanism ensuring the answer is not hallucinated. In this paper, we introduce Language Model-based Document Information Extraction and Localization (LMDX), a methodology to adapt arbitrary LLMs for document information extraction. LMDX can do extraction of singular, repeated, and hierarchical entities, both with and without training data, while providing grounding guarantees and localizing the entities within the document. In particular, we apply LMDX to the PaLM 2-S LLM and evaluate it on VRDU and CORD benchmarks, setting a new state-of-the-art and showing how LMDX enables the creation of high quality, data-efficient parsers.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a method to extract information from visually rich documents with the information and their position. The method extracts information from the OCR'ed document by prompting the PaLM2-S LLM to perform completion tasks. With the LLM with entity extraction training, the model can achieve strong performance even with no training data, comparable to or better than a few baselines. With a few-shot setting, the method shows a high performance with a large margin compared to existing methods.

### Strengths
- This paper proposes a novel method to document information extraction from visually rich documents using LLMs. 
- The absolute performance is greatly higher than existing methods, and some of the proposed enhancements are shown effective through the ablation study. 
- The paper is well-written with the details of algorithms, schemas, sample outputs, etc.

### Weaknesses
- The model's performance highly depends on the off-the-shelf OCR and PaLM2-S large language model, but they are unavailable, so the results are not reproducible. Also, there is no detailed explanation or evaluation of these modules. 
- The authors mention the support of the hierarchical entity and entity localization, but their effect is not directly evaluated since there is no evaluation without them.

### Questions
- Is there any performance assessment of the OCR model? 
- How does the model differ from baselines in, e.g., the number of parameters and runtime? Although the authors remark on using open-source LLMs as future work, how is it difficult to run the model with publicly accessible OCR and existing LLMs? 
- Did the authors try other prompts, schema, or target formats during the development? How are the current settings chosen?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
When applying LLM to Visually Rich Document (VRD) understanding, many methods use the two staged approaches: first execute the text recognition/serialization step and then execute the parsing step. However, lots of methods suffer from the need for large training data or are not able to predict hierarchical entities or hallucinations in domains other than text-only data. These problems are due to the absence of layout encoding within LLMs and the absence of a grounding mechanism ensuring the answer is not hallucinated. To overcome these challenges, the authors propose the five staged frameworks: OCR - chunking - prompt generation - LLM inference - decoding. The suggested framework is experimented with PaLM 2-S and compared to several publicly available baseline models on Visually Rich Document Understanding (VRDU) and Consolidated Receipt Dataset (CORD), resulting in a bigger performance margin than baseline methods.

### Strengths
- Suggest reasonable methods to tackle the challenges of visual document understanding.
- Provide rich information to reproduce experiments

### Weaknesses
* Though the suggested method seems agnostic to specific LLM, the authors experimented only with PaLM 2-s. To verify the superiority of the suggested framework, additional experiments using LLMs other than PaLM are needed (I think the additional experiment would enhance the presentation of the robustness of the proposed method).

### Questions
* In document representation, when generating prompts, how well do coordinate tokens work? Line-level segments with 2 coordinates are enough for various VRD data?
* Schema representation is important in the perspective of getting information in VRD. However, it would be vulnerable to hallucination. Does LLM properly parse JSON format?
* When doing Top-K sampling, we can choose Top-K sampling for individual N chunks and then merge, or do Top-K sampling for entire N chunks. I guess the latter method is better for the semantic integration quality (the similar reason that authors used the entire predicted tree value from a single LLM completion for hierarchical entities), but the authors used the former method. Is there a reason? I think the comparison may be interesting.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes LMDX — a mechanism for information extraction from documents leveraging off-the-shelf Optical Character Recognition service and LLM prompt engineering approach with PALM LLM for processing the extracted information.

### Strengths
Strengths:
* The paper shows potential of using LLMs for information extraction from documents
* Ablation studies are interesting and show the value of fine-tuning the PALM LLM for document information extraction

### Weaknesses
Weaknesses:

* This paper proposes a mechanism for information extraction from documents leveraging off-the-shelf Optical Character Recognition service and complicated LLM prompt engineering approach for processing the extracted information. The main underlying assumption driving the complexity of the prompt engineering approach is limited context length of LLMs. However, models like Claude 2 are capable of working with 100K token context windows. Additionally, methods like RoPE scaling and other context length expansion approaches allow to increase the context size for other LLMs including open-source models. As there are effective ways to address the context length limitation, the presented prompt engineering approach is a somewhat incremental engineering contribution, especially given its complexity. The fine-tuned model, however, is of interest.
* While the proposed approach outperforms other baselines on VRDU and CORD benchmarks, the performance advantage clearly comes from using a powerful LLM. It would be important to compare this method to OCR+long-context LLMs such as Claude 2.
* Another reasonable baseline with the potential to achieve high performance on these benchmarks is a multi-modal vision-text LLM, for example GPT-4. It has potential to work out of the box, without requiring fine-tuning, and significantly outperform other baselines.
* Code is not provided.
* Many unexplained abbreviations: e.g., IOB, NER. Readers would benefit from expanding these abbreviations the first time they are used.

### Questions
Questions:
* In-context learning is likely to significantly improve performance on this task, have you tried any experiments with in-context demonstrations?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
