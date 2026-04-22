# Handwritten Text Recognition Adaptation for Low-Resource Languages: A Case Study on Historical Latin Manuscripts

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Handwritten Text Recognition (HTR) remains a challenging task in document digitization, particularly for historical manuscripts written in low-resource languages such as Latin. In this paper, we focus on recognizing Latin texts from 16th–18th century manuscripts, which exhibit a wide range of handwriting styles. To address this, we propose AdapterTrOCR, a modular extension of the TrOCR model that incorporates two adapter modules: one for historical language adaptation and another for handwriting style adaptation. This architecture enables a robust transition from a generic English HTR model to one specialized in historical Latin. Given the limited availability of annotated data, we also explore Handwritten Text Generation (HTG) as a data augmentation strategy. Our results show the effectiveness of modular adaptation and synthetic data in improving HTR performance, achieving reductions in character error rate (CER) by 13.33% to 35.65% and word error rate (WER) by 8.56% to 27.72%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents AdapterTrOCR, a novel model for Handwritten Text Recognition (HTR) specifically designed for historical Latin manuscripts, a challenging task due to the low-resource nature of the language and the variability of handwriting styles. The proposed model extends the TrOCR architecture by incorporating two distinct adapter modules: one for adapting to the historical Latin language and another for adapting to specific handwriting styles. To address the scarcity of annotated data, the authors also introduce DiffLine, a Handwritten Text Generation (HTG) technique for data augmentation. The experimental results demonstrate the effectiveness of this approach, showing significant reductions in Character Error Rate (CER) and Word Error Rate (WER).

### Strengths
1. The proposed AdapterTrOCR model is innovative. The modular design, which separates historical language adaptation from handwriting style adaptation, is a logical and effective way to handle the complexities of historical manuscripts.

2. The paper tackles the common problem of data scarcity in low-resource scenarios by proposing a new data augmentation method, DiffLine. This is a valuable contribution that could be beneficial for other HTR tasks as well.

3. The authors provide a comprehensive evaluation of their model. The reported reductions in CER by 13.33% and WER by 8.56% are substantial.

4. The paper is well-organized and clearly written. The methodology is explained in detail

### Weaknesses
1. The paper does not include a comparison with existing Large Language Models (LLMs), which have demonstrated strong performance on various text recognition and understanding tasks. Evaluating AdapterTrOCR against state-of-the-art multimodal LLMs could provide a more complete picture of its capabilities.

2. The study focuses exclusively on historical Latin manuscripts. While the results are promising, the paper would be stronger if it included experiments on other low-resource languages or different types of historical documents to demonstrate the generalizability of the proposed method.

3. The historical language adaptation relies on a proxy language (Dutch). The choice of this proxy language could impact the model's performance. The paper could benefit from a discussion on how sensitive the model is to the choice of the proxy language.

4. The proposed solution involves a multi-step pipeline that includes training adapters, generating synthetic data with a diffusion model, and fine-tuning the final model. This complexity might pose a challenge for researchers and practitioners who are not experts in all of these areas.

### Questions
Your work builds a complex, specialized model. Based on my personal experience, however, current large vision–language models (e.g., GPT-4o and Gemini-2.5-Pro) already demonstrate remarkable performance on complex or ambiguous image–text recognition tasks in few-shot and even zero-shot settings. What are the advantages of your proposed method?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses handwritten text recognition (HTR) for low-resource languages, focusing on historical Latin. The authors propose two main components: (1) TrOCR adaptation through task and language decomposition, and (2) synthetic handwritten data generation. For adaptation, TrOCR is first trained on historical Dutch to capture general HTR ability, and two LoRA adapters (one for Latin and one for Dutch) are then used to extract Latin-specific language knowledge via parameter differencing. To mitigate data scarcity, the authors generate synthetic handwriting using DiffusionPen with dual classifier-free guidance and MobileNetV2 for style encoding. Overall, the paper is well-structured and experimentally thorough, with clear motivation and creative integration of modern HTR and diffusion techniques. However, the core contributions are largely incremental, offering limited performance gains and uncertain justification for the proposed adaptation strategy.

### Strengths
+ The paper presents a novel decomposition of HTR into task and language adaptations, using historical Dutch as a proxy for historical-domain adaptation and leveraging cross-language differencing (Latin vs. Dutch) to isolate linguistic knowledge.

+ The experimental setup is comprehensive and well-structured, with evaluations conducted across multiple models (TrOCR, PyLaia, DiffusionPen, and WordStylist)  for both HTR and HTG tasks.

+ The paper provides comprehensive experimental results such as ablation studies, HTG effectiveness, and HTR performance.

+ The work demonstrates strong integration and adaptation of modern HTR and diffusion techniques, showing practical creativity in combining them for low-resource historical languages.

### Weaknesses
- Limited performance improvement. The reported gains for both HTR and HTG are modest and, in some cases, inconsistent. For example, in Table 1 for yes GT, yes Synth 1st scenario, DiffLine achieves a negligible 0.01% CER improvement over base TrOCR, while the accuracy of the proposed method decreases by 0.67% compared to base TrOCR.

- Questionable effectiveness of the language adaptation strategy. Theoretically, the language adaptation strategy is interesting and may help but there is no significant experimental evidence. Training a LoRA adapter on historical Dutch and then applying a differential LoRA derived from Latin-Dutch subtraction may not meaningfully differ from directly training a Latin LoRA adapter, raising doubts about the necessity of this decomposition. For example: TrOCR + Dutch + (Latin - Dutch) = TrOCR + Latin

- Incremental novelty. Many core components rely on adapting existing architectures (e.g., TrOCR, DiffusionPen) via LoRA fine-tuning or different existing loss functions. While the integration is thoughtful, the methodological contribution is limited with no significant performance gains.

- Minimal advantage over baseline models. The base TrOCR achieves comparable performance, improving CER from 23.32% to 20.22% after adaptation, suggesting limited practical impact.

- Limited practicality of synthetic data generation. Although the paper emphasizes its low-resource applicability, it acknowledges that HTG quality scales with dataset size, which undermines its practicality for truly low-resource languages.

- Concerns about generalization. The authors note that including the Bullinger dataset does not yield better results due to differences in layout and style. This raises questions about the robustness and generality of the proposed adaptation pipeline across varied handwriting sources.

- While the paper demonstrates solid experimental effort, the core contributions are incremental and the reported gains minimal. The proposed adaptation strategy does not convincingly justify its added complexity or conceptual novelty.

### Questions
1. What specific challenges in historical Latin make it distinct from modern English for HTR, beyond style variation? Since both use the same alphabet, could this explain why the base TrOCR performs competitively?

2. Why is style adaptation implemented in the decoder of TrOCR? Wouldn’t the encoder, or a combined adaptation strategy, better capture handwriting features?

3. What motivated the choice of generating 2,000 synthetic line images? Was this empirically optimized, or constrained by computational cost?

4. In Table 3, the style adapter appears to contribute most of the improvement, while the language adapter provides minimal gains, even when combined. Could the authors elaborate on this asymmetry?

### Soundness
3

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
4

### Summary
The authors proposed a modular and parameter-efficient approach to adapt TrOCR, a OCR system that is trained on modern English to historical Latin. The idea is to use LoRA adapters on TrOCR decoder to separate the language ability from Dutch to Latin CLM and add a style adapter that can be per-document. This is very interesting for low-resource, writer-specific manuscripts. After that a diffusion-based generator for handwritten text introduces dual classifier-free guidance for text and style. At the end MobileNetV2 style encoder trained with InfoNCE and a Sinkhorn regularizer is used. 

I liked the approach of dividing the “historical HTR task ability” and “language ability,” which could also be applied in other tasks. However, I miss some robustness checks to attribute improvements to the proposed modules rather than to stronger decoding or to overlap between training and testing styles.

### Strengths
The idea of decomposing adaptation into historical HTR “task ability”, cross-lingual “language ability,” and manuscript-specific style via additive LoRA composition is very interesting, especially for the handwriting recognition context. 

The proposed approach improves CER/WER over traditional and strong baselines, which is promising. The proposed ablations show that the style adapter provides the largest lift, with language/task adapters adding further gains. The diffusion module clearly outperforms other generators when GT is ample.

The results show that the approach is practically relevant, and the modular PEFT approach is reproducible and budget-friendly.

### Weaknesses
Not sure I understand correctly but the historical Latin adapter is trained with CLM and the final model is “subsequently fine-tuned”. In this case, I don't see the control with a frozen TrOCR plus shallow-fusion Latin LM or LM-KL reranking. Without those, improvements may partially reflect stronger decoding priors rather than adapter composition.

It seems that the train/test splits are within the same manuscript/writer. Because the style adapters are tuned using 20 “randomly selected images” from a specific writer to choose sT and sS. Considering that they could come from the full manuscript pool, the hyperparameter search may peek at test-style statistics. I think the use of leave-one-manuscript-out evaluation is needed.

The authors suggest all adapters are trained on the decoder “as some tasks are language-only.” For the historical HTR adapter, which raises a modality mismatch, if only the decoder is adapted, improvements might come from a better language model rather than better visual grounding.

I think applying YOLO filtering and a 70% confidence cutoff only for AdapterTrOCR in the ViTLP comparison complicates fairness because differences in missed/false lines can dominate CER/WER.

The “single author per manuscript” assumption is fragile for historical collections. If violated, per-manuscript style adapters may entangle writer and content, reducing portability to mixed-hand manuscripts.

### Questions
Since adapters are composed additively, have the authors tested whether the order of adapter application (style -> language -> task vs. the reverse) affects convergence or performance? 

What are the total training times and GPU hours for AdapterTrOCR and DiffLine? How do they compare with full fine-tuning and prior PEFT baselines?

Could the adapter composition strategy generalize to other low-resource scripts (e.g., Old French, Medieval Spanish)? Have preliminary tests been done?

### Soundness
3

### Presentation
3

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
This article proposes a method for adapting the TrOCR model to historical documents in a language with limited resources. The proposed method is based on two independently conducted adaptations: one to historical writing and one to language. An evaluation is conducted by adapting a TrOCR model trained for modern English to documents in historical Latin.

### Strengths
The article puts forward the interesting hypothesis that handwriting recognition models can be adapted independently of document style (historical) and language. If such an approach were validated, it would enable models to be adapted to a wider variety of documents and languages, taking advantage of combinations for which little training data is available.

### Weaknesses
While the hypothesis is interesting, the experimental protocol and datasets used do not allow it to be validated:

- on the dataset used, in scenario 2, TrOCR already performs quite well: 4.47% CER. The improvement with the proposed adaptation is therefore small, even if significant. In scenario 1, the model remains underperforming even after adaptation (23% CER to 20% CER).
- the generality of the method has not been demonstrated, as the experiments were conducted on a single historical dataset (VOC) and a single language (Latin). Many other historical datasets exist, so other combinations could be tested.
- the proposed method is not compared to a standard alternative, which would consist of adapting the TrOCR model to the historical dataset and then adapting it to the target language. 

The use of synthetic data somewhat obscures the message of the article, which should focus on its initial hypothesis: model adaptation.

The method could be tested on larger models, for which LoRA was originally developed, such as Qwen3.

The details of the LoRA parameters are not provided.

### Questions
L040 : "written in Latin and date from the 16th to 18th centuries" : It is unusual to find manuscripts from this period written in Latin. You should describe the documents.

L044 : "HTR is usually implemented as a two-step process" :  more and more approaches are based on full-page models (DAN, vLLM).

L62 : "Since the simple fine-tuning is not sufficient" : explain why and show results.

L349 : accuracy is not defined : how is it computed, is it at character level ?

### Soundness
1

### Presentation
2

### Contribution
1
