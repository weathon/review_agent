# Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages

- Avg Score: 5.50
- Decision: Accept (spotlight)
- Scores: 3, 8, 6, 5

## Abstract
Recently there has been a significant surge in multimodal learning in terms of both image-to-text and text-to-image generation. However, the success is typically limited to English, leaving other languages largely behind. Building a competitive counterpart in other languages is highly challenging due to the low-resource nature of non-English multimodal data (i.e., lack of large-scale, high-quality image-text data). In this work, we propose MPM, an effective training paradigm for training large multimodal models in low-resource languages. MPM demonstrates that Multilingual language models can Pivot zero-shot Multimodal learning across languages. Specifically, based on a strong multilingual large language model, multimodal models pretrained on English-only image-text data can well generalize to other languages in a (quasi)-zero-shot manner, even surpassing models trained on image-text data in native languages. Taking Chinese as a practice of MPM, we build large multimodal models VisCPM in image-to-text and text-to-image generation, which achieve state-of-the-art (open-source) performance in Chinese. To facilitate future research, we open-source codes and model weights at https://github.com/OpenBMB/VisCPM.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces MpM, a framework to train multilingual multimodal models for image-to-text and text-to-image generation (mostly English+Chinese). For image-to-text, the model (VisCPM-Chat) is typically pretrained on English image captions by freezing the LM, and then fine-tuned on instruction data translated into the target language(s). The model shows competitive performance when evaluated on machine-translated versions of LLaVA Test Set and UniMM-Bench as evaluated by an off-the-shelf model (GPT-4). For text-to-image, both the text model and the image decoder are frozen, and cross-attention layers are trained using English data. The resulting model (VisCPM-Paint) achieves a lower FID score than previous models on 30K samples from COCO Val obtained through machine translation.

### Strengths
1. The proposed framework, MpM, corroborates and extends the line of work on multilingual transfer through frozen language models, showing it to be more effective than previous models on the chosen tasks according to the evaluation measures adopted by the authors.
2. The plot in Figure 2 is interesting, showing how most of the Wukong text data might be poorly aligned with the corresponding images.
3. Human evaluation on Chinese Drawbench shows how the proposed model is often preferred to the baselines.
4. It is interesting to see how much English data helps during fine-tuning in the ablation study, perhaps because it bridges the gap between the style and language axes.

### Weaknesses
1. The paper claims to “generalize to other languages in a zero-shot manner.” This is not true, and very misleading. In fact, the chat model had to be fine-tuned on target language data, otherwise it would generate text in English. Calling the behavior “quasi-zero-shot” is also (i) different from the literature, and especially (ii) misleading: the model cannot generate text in the target language without supervision. Such claims, while valid for the text-to-image model, should be removed.
2. The paper claims to propose “an effective training paradigm for training large multimodal models in low-resource languages.” However, none of the languages tested were low-resource. There is a significant difference in the representations of the LM model between high- and low-resource languages, as well as in the quality of translations that an MT system can provide for those languages. None of these things are true for Chinese. Any claims about performance and applicability to low-resource languages need to be removed as they are never verified.
3. The paper also claims to be multilingual. Yet, most of the experiments are done in a bilingual setup (English+Chinese) or in a few, high-resource European languages. These languages share the same script, similar topology. Claims about multilinguality should be supported by a better selection of languages that increase diversity in scripts, topology, and resource availability.
4. My other major concerns are related to the experimental setup.
- (4a) The model is evaluated on unusual benchmarks for multilingual multimodal modeling. These datasets are obtained through machine translation of the corresponding English dataset. The authors claim to fix minor errors (only for Chinese, and not even for all the other 6 European languages), yet do not provide any indication of the quality of the resulting test data. 
- (4b) The evaluation scores are based on GPT-4, which might not understand languages beyond English well. This is an hypothesis that might or might not hold true, but the fact that such concern exists, I believe it is enough to undermine the evaluation setup used by the authors.
- (4c) There exists multilingual multimodal benchmarks. For instance, IGLUE [1] tests for 4 multimodal tasks in 20 languages, with data *manually* collected by native speakers. MaRVL [2] and XM3600 [3] even include images sourced from the countries where a language is spoken. These benchmarks are adopted in the literature and completely missing (even as part of the related work) in this paper.
5. There is a lack of discussion of relevant related work. Models like {m,x}UNITER [2], CCLM [4] and TD-MML [5] are missing. The latter in particular shows how machine translation data can be very helpful during pretraining if filtering out poor translations, which provides a contrasting point to the one made by the authors in this paper. A discussion of these findings and how they differ is important to guide the community.

---

[1] Bugliarello et al. IGLUE: A Benchmark for Transfer Learning across Modalities, Tasks, and Languages. ICML’22.

[2] Liu et al. Visually Grounded Reasoning across Languages and Cultures. EMNLP’21.

[3] Thapliyal et al. A Massively Multilingual Multimodal Evaluation Dataset. EMNLP’22.

[4] Zeng et al. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training. ACL’23.

[5] Qiu et al. Multilingual Multimodal Learning with Machine Translated Text. EMNLP’22.

### Questions
1. The low CLIP scores in Wukong might be related to misunderstanding of Chinese captions from CLIP. You say that you perform manual inspection, can you elaborate more on your findings and their relation to CLIP scores?
2.  It would be useful for the reader to have more information about the baselines, so that one can easily understand what changes among models when comparing their results.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Text-to-image and image-to-text generation efforts have primarily been focussed on English only due to lack of large-scale, high quality data. This paper proposes MPM, an effective training paradigm for training large multimodal models in low-resource languages. They show that this technique enables competitive zero-shot performance on Chinese language as compared to models trained on the language data. This leverages the Bilingual Dual-coding Theory which states that visual semantics are largely language agnostic. MPM divides the non-English multimodal learning into two consecutive stages: multilingual alignment and multimodal alignment. The former focuses on building a multilingual model by using a pretrained multilingual large language model, while the latter culminates in a multimodal model spanning multiple languages on top of the multilingual model.

### Strengths
- This paper shows a simple yet effective technique of using a mulitlingual LLM as a pivot to transfer multilingual image-text alignment across different languages. 
- The model is trained on English data only but still manages to perform better on Chinese tasks than models trained on Chinese data. This is highly encouraging in making the state-of-the-art techniques on image-to-text and text-to-image generation available in other languages.

### Weaknesses
None.

### Questions
None.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies a simple idea to use multilingual language models as the pivot to connect different languages for multimodal applications. The idea is based on the assumption that visual semantics are language agnostic, thus aligning a visual modal to a pre-trained (and sometimes frozen) multlilingual language model could support multilingual applications even without aligned multimodal data.

### Strengths
1. This paper is an engineering style paper, with great qualitative examples. The idea itself is simple and intuitive (but however the execution and writing is a bit complicated, see the next section). 
2. This paper promised to open-source the code and model weights. This is going to have a positive impact on the community thanks to the open sourced model weights.

### Weaknesses
1. The contribution of the MPM training paradigm is unclear. Taking the image-to-text generation as an example, in the multimodal pre-training step the proposal is to freeze multilingual LLM and only tune the visual module; in the instruction tuning step the proposal is to finetune everything on the datasets from pivot language and the target language. The multimodal pre-training step is quite straightforward as it aligns a trainable vision model to a pre-trained and frozen LLM; the instruction tuning step is a standard finetuning setup. If the contribution of this paper is to use a pre-trained multilingual LLM to support multilingual downstream tasks, then much more in-depth analysis should be done to justify the point. Examples include understanding the MPM design choices, how freezing LLM affects results, how translated pairs affect instruction tuning results, how many translated pairs are enough and what the trend looks like. 
2. Most of the tables present absolute comparisons, showing how great the proposed method is. However that might be less interesting from a reader's perspective without proper baselines and ablation tests as discussed in the above section (due to different dataset mixtures and tuning tricks used). It's high recommended to take a few tables, and perform multiple in-depth analysis and comparisons with controlled study, to make sure there are enough take aways and confidence in the current MPM design choices.

### Questions
See the above sections.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a method for training a multilingual and multimodal model without needing to rely on large amounts of data in all of the target languages. Instead, the method relies on multilingual and multimodal pivoting, using a high-resource language to provide the target distributions for the lower-resource language. The method is realized in an image-text and text-to-image model named VisCPM-Chat/Paint, respectively. The model is trained by aligning the low-resource Chinese data against the high-resource English data using the CPM-Bee bilingual Zho/Eng language model; the method is also shown to generalize to LLaMA.

### Strengths
S1: Conceptually simple approach to learning a multilingual and multimodal model. Makes good use of the vast amount of English language resources.

S2: Extensive experiments with relevant benchmark datasets.

S3: The approach is shown to generalize to more than one language model (LLaMA and CPM-Bee)

### Weaknesses
W1: There are some inconsistencies in the argumentation used throughout the paper. For example, in Section 4.1, the authors argue that translating a large-scale pretraining dataset will consume substantial computing resources, which is a fair argument. However, in Section 4.2, the authors describe that they train a version of their model called VisCPM-Chat+ using 1
36 million translated examples from LAION-COCO. The authors could improve their argumentation if they clarify whether it
is or is not challenging to create such translated examples. Bear in mind that other researchers have already translated
pretraining datasets, e.g. Thapliyal+ EMNLP 2022.

W2: The motivation for this paper is to create multimodal models for low-resourced languages via the proposed pivoting method but the main languages used in the experiments are Chinese and English, neither of which could ever be described as low-resource languages. The remaining languages used for the experiments in Section 5.1.3 are also hardly low-resource: German, French, Spanish, Italian, and Portuguese. Using these languages and claiming that the method applies to low-resource settings affects the credibility of the conclusions that one can draw for actual low-resource languages. See, for example, Joshi+ ACL 2020 for a discussion on low-resource languages.

Joshi, Pratik, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. "The State and Fate of Linguistic Diversity and Inclusion in the NLP World." In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 6282-6293. 2020.

Thapliyal, A. V., Tuset, J. P., Chen, X., & Soricut, R. (2022, December). Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (pp. 715-729).

### Questions
1. Why do you think Chinese can be considered a low-resource language? See W2 for more details on my concerns.
2. In Section 5.1, why is it methodologically sound to use GPT-4 to evaluate model output?
3. In Figure 2, why is a CLIPscore threshold of 0.18 used to define high-quality? How was this determined?
4. In Section 4.2, what was the dataset of 100M image-text pairs used to align with the frozen LLM for 180K steps?
5. Did you also use M2M-100 to translate the 136M examples in the COCO-LAION dataset?
6. Which version of M2M-100 did you use for translation?
7. In Section 5.2.2, you describe an FID of 9.5 or 9.9 "comparable" to Stable Diffusion FID is 8.6. Is this a reasonable claim?
8. How does your model perform on a larger set of languages in the multilingual multimodal benchmark IGLUE by Bugliarello+ ICML 2022?

Bugliarello, E., Liu, F., Pfeiffer, J., Reddy, S., Elliott, D., Ponti, E. M., & Vulić, I. (2022, June). IGLUE: A benchmark for transfer learning across modalities, tasks, and languages. In International Conference on Machine Learning (pp. 2370-2392). PMLR.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
