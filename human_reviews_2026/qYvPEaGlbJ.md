# OOD Learner via In-Context Learning

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Out-of-distribution (OOD) detectors are built on classification models to identify test samples that do not belong to any of their training classes. For classifiers based on pretrained vision-language models (VLMs), recent methods construct OOD detectors using text and few shot in-distribution (ID) images. In this work, we introduce a versatile framework for few-shot OOD detection through in-context learning (ICL). Instead of building an OOD detector for specific ID datasets, we propose a universal OOD Learner that can adapt to arbitrary ID datasets using few-shot texts and images as context, without the need for fine-tuning. Our method is implemented as an attention-based module and pretrained on pseudo-class data curated from large-scale text-image pairs. Experimental results demonstrate that our framework achieves state-of-the-art performance and efficiency in few-shot OOD detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
### Background
- This paper focuses on the few-shot OOD detection task. 
- Prior works usually use the few-shot ID samples to refine text prompts. The authors argue that prompt tuning has several limitations.

### Method
- Therefore, the authors propose a universal OOD learner.
    - The OOD learner takes text and images from an ID class as input and outputs the class representation.
    - It is an attention-based module and trained with contrastive loss.
- The authors apply the ANNOY algorithm to process a large-scale dataset and pretrain the OOD learner.

### Results
- The proposed OOD learner achieves SoTA performance and efficiency in few-shot OOD detection.

### Strengths
- The paper is well-written with clear figures and tables.
- The method is clear and easy to understand.
- The experiments are comprehensive.

### Weaknesses
### Major: Do we really need large-scale training for OOD detection?
- (Gain vs cost) As I recall, MCM and NegLabel do not need extra training or ID examples, and they already get impressive accuracy. Although the proposed method outperforms MCM and NegLabel, the improvement is relatively modest (90.11 vs 88.66 in Table 1, and no comparison with NegLabel on ImageNet-1K). Considering the OOD learner needs to be trained on about 28M text-image pairs, I'm not sure if the performance gain is worth the computational cost.
- (Scaling) The scaling-up property of the OOD learner is not clear. As shown in Table 3, when the number of samples increases from 281K to 28M (100 times), the AUROC goes from 92.42 to 92.93. The gain is very small and maybe not statistically significant.
- (Test-time adaptation) Besides, recent test-time adaptation methods such as AdaND and AdaNeg show improved performance in OOD detection. How is the OOD learner compared with these methods? Could we implement AdaNeg in the few-shot OOD detection setting?


### Minor
- How does the method compare with NegLabel on ImageNet-1K? 
- (Line 166) "without relyingper-dataset fine-tuning" missing space.

### Questions
Please see the Weaknesses section.

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
4

### Summary
Existing OOD detection frameworks typically follow the classic experimental setting, which requires training or fine-tuning on in-distribution (ID) datasets. This work departs from that paradigm by introducing a novel OOD detection framework capable of adapting to arbitrary ID datasets using only a few ID samples, without any additional training or fine-tuning. To achieve this, the authors propose an attention module that jointly learns class representations from both text and corresponding images. Moreover, to efficiently construct one-to-many class–image training sets, the framework employs ANNOY to retrieve the most similar images for each class with reasonable computational cost. In terms of performance, the proposed method achieves state-of-the-art AUROC results.

### Strengths
- The paper is well-written and easy to follow.
- The proposed architecture and data curation pipeline for pre-training are sensible. 
- The experiments are extensive.

### Weaknesses
Weaknesses:

- I understand that the OOD learner is inspired by the in-context learning phenomenon observed in large language models. However, the authors should better motivate and clarify why the proposed architecture, in particular, the attention module,  is expected to work and provide justification for the specific design choices made.
-  The OOD evaluation does not fully follow the standard OOD benchmarking protocol, see https://zjysteven.github.io/OpenOOD/. 
   - Specifically, the results on OpenImage-O are missing when ImageNet-1k is used as the ID dataset.
-  Experiment-wise, the authors should also report FPR95 values to provide a comprehensive evaluation of the proposed framework.
 

Minor weaknesses:
- While the paper includes extensive experiments on far-OOD detection benchmarks, it would be valuable to also include results on near-OOD benchmarks (\eg, ImageNet-1k as ID dataset). This would help determine whether the proposed design maintains strong performance under near-OOD conditions.
-   Some related works are missing. For CLIP-based zero-shot OOD detection, GL-MCM [1] and TAG [2] do not require OOD labels and fine-tuning. 

 [1] GL-MCM: Global and Local Maximum Concept Matching for Zero-Shot Out-of-Distribution Detection, IJCV, 2023.

 [2] TAG: Text Prompt Augmentation for Zero-Shot Out-of-Distribution Detection, ECCV, 2024.

### Questions
See weaknesses.

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
This work focuses on few-shot out-of-distribution (OOD) detection and proposes a general In-Context Learning (ICL) framework, OOD Learner. To address the limitations of CLIP-based prompt-tuning methods—which require per–in-distribution (ID) dataset fine-tuning and exhibit limited generalization—the authors adopt a text–image dual-branch attention module. Using a small number of text–image pairs as in-context examples, the method adapts to arbitrary ID datasets without fine-tuning. During pretraining, the approach leverages large-scale text–image corpora (e.g., LAION-400M) and uses ANNOY to construct a massive set of pseudo-classes, thereby injecting generic OOD detection capability. Experiments on multiple ID datasets claim state-of-the-art (SOTA) performance and efficiency for OOD detection, with flexible adaptation to varying numbers of shots and to single-class scenarios, demonstrating strong generalization and practical utility.

### Strengths
1. By applying ANNOY to LAION-400M and related large-scale corpora, the method constructs roughly 30 million pseudo-classes, offering semantic coverage far beyond ImageNet-1K (1K classes).

2. Compared with prompt-tuning paradigms, the framework accommodates a variable number of shots without modifying the model, and it supports diverse ID settings including single-class and multi-class scenarios.

### Weaknesses
1. Pseudo-class construction relies on CLIP feature retrieval, which introduces potential mismatch risk.

2. Under the setting where ImageNet-1K serves as the ID dataset, the OOD detection performance does not show significant improvement.

3. The OOD Learner requires pretraining, leading to substantial computational cost; relative to some zero-shot methods, the performance gains are comparatively limited.

### Questions
1. There already exist numerous test-time adaptation (TTA) methods for OOD detection with strong results. Given the SOTA claim, a unified head-to-head comparison with representative methods such as AdaNeg \[1], OODD \[2], and AdaND \[3] is recommended.

2. Only AUROC is reported; please also include FPR95 and AUPR, which are more sensitive for open-set evaluation.

3. Appendix E.2 states that the attention complexity scales as $O(L^2)$ with the number of input samples L, which precludes large-scale data augmentation as in CoOp. When the number of shots k is large, a quantitative cost–benefit analysis is advised to assess the practical trade-off.

4. The current setup primarily uses CLIP-L/14 features(Table 2); please also evaluate RN50x16 and ViT-H/G to examine whether marginal gains diminish when the base model already provides strong separability.

\[1]AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models.

\[2]OODD: test-time out-of-distribution detection with dynamic dictionary.

\[3]Noisy Test-Time Adaptation in Vision-Language Models.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The work proposes an OOD detection approach based on attention module that claims to be inspired from the in-context learning using in GPTs. Unlike the prompt tuning approaches, the proposed method claims that they do not need fine-tuning which is a bit confusing in this interpretation and the presentation in the paper. The experiments are presented to compare the empirical results with the existing methods.

### Strengths
The paper introduces a nice and timely problem statement indicating that “prompt tuning lacks flexibility to new information, such as additional shots of images or new ID classes.”. This is indeed a drawback of many existing OOD detection baselines though they generally perform better than those derived from pre-trained models.

### Weaknesses
Weaknesses:

1.	The main issue is the ambiguity of the claim that “no fine-tuning” is needed. The authors are correct that no finetuning is needed on the encoders of the CLIP model during the text time. But, the attention module parameters are updated based on few-shot examples to learn the “in-context” or ID data. This is very similar to the prompt tuning methods such as CoOp, LoCoOp, SCT etc where the prompt vectors are learned using few-shot examples keeping the encoders frozen. Could you justify the reason why you still see your framework as “in-context” and compare with GPT-3?

2.	In addition, the methods also needs hyperparameter tuning and selection for $\lamda$ which is also against the claim of “in-context” learning and no fine-tuning.

3.	In the result table 1, more recent baselines like LoCoOp and SCT are not presented and FPR (False Positive Ratio) is also not reported. It is well-established in recent works that LoCoOp and SCT outperforms the vanilla CoOp by large margins. Table 2 also does not show the advantages of the approach. The results imply the proposed approach is not competitive enough with prompt-tuning based approaches.

### Questions
Please see Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
