# Compositional Generalization in Multimodal Foundation Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
The rise of large-scale multimodal models has paved the pathway for groundbreaking advances in generative modelling and reasoning, unlocking transformative applications in a variety of complex tasks. However, a pressing question that remains is their genuine capability for stronger forms of generalization, which has been largely underexplored in the multimodal setting. Our study aims to address this by examining sequential compositional generalization using CompAct (Compositional Activities), a carefully constructed, perceptually grounded dataset set within a rich backdrop of egocentric kitchen activity videos. Each instance in our dataset is represented with a combination of raw video footage, naturally occurring sound, and crowd-sourced step-by-step descriptions. More importantly, our setup ensures that the individual concepts are consistently distributed across training and evaluation sets, while their compositions are novel in the evaluation set. We conduct a comprehensive assessment of several unimodal and multimodal models. Our findings reveal that bi-modal and tri-modal models exhibit a clear edge over their text-only counterparts. This highlights the importance of multimodality while charting a trajectory for future model development in this domain.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a study around the proficiency of multimodal foundation models in comprehending sequential activities. The authors curate the COMPACT dataset, which is a stratified subset of the Epic Kitchens dataset. They use this dataset to validate the performance of various models such as ImageBind, MERLOT Reserve, as well as Llama2.

### Strengths
The paper is clear and well-written. The study of multimodal models on natural video datasets is quite timely.  
The idea of curating a subset of an existing dataset, rather than creating one from scratch, is a sound idea, especially since there are many similar datasets out there.

### Weaknesses
The paper is essentially an evaluation of well-defined tasks using an existing dataset and pre-trained models. 
Out of the three tasks, the problem of action classification is only slightly different from next-utterance noun prediction and verb prediction (i.e., action is a combination of noun + verb, and the other two are predicted separately.) 
Various pre-trained models are run in a zero-shot manner to predict the next entity in the sequence. While the evaluation results could be of interest to someone who is looking to build these models, the paper main content offers little beyond this evaluation.

### Questions
The result show that multimodality provides minimal advantage in the next noun and verb predictions tasks. However, the BLEU scores for the action classification are significantly improved. This is counterintuitive, since the actions are essentially nouns + verbs combinations. Therefore, if these individual units are incorrect, the overall BLEU score cannot be significantly better?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper aims to understand the capabilities of multimodal foundation models in terms of compositional generalization. To enable this study, the authors carefully constructed a dataset, CompAct (Compositional Activities), by reusing multimodal data and annotations from an existing benchmark, EpicKitchens-100, while defining new tasks and forming new training, validation, and testing splits. 

Each data instance is an instructional video, containing video, audio, and step-by-step descriptions, where each step has a verb and a noun. CompAct is formed such that atomic concepts (verbs or nouns) are consistently distributed across the training and evaluation sets, while compositions of these atomic concepts are novel in the evaluation set. 

CompAct allows for the diagnosis of the compositional generalization capabilities of both unimodal and multimodal models. The authors conduct an assessment of several unimodal and multimodal models, and their findings highlight the limited capabilities of prior foundation models for compositional generalization, as well as the importance of multi-modality over single-modality for certain challenging tasks.

### Strengths
1. The data distributions between training and evaluation in the CompAct benchmark are carefully controlled, allowing for the diagnosis of models' compositional generalization capabilities, which could be useful to the research community.

2. The authors present experimental results for approximately ten different unimodal or multimodal models. Some of the results are intriguing; for example, the language-only method outperforms the multimodal method in noun classification. However, for verb classification or next utterance prediction, the multimodal methods demonstrate superior performance.

3. The proposed method for curating train/eval splits to diagnose compositional generalization appears to be applicable to many other existing video datasets.

### Weaknesses
The authors have overlooked several works and benchmarks that are highly similar (see Questions below). Compared to these existing works, the contribution of this paper does not seem to be very significant. Additionally, the conclusions drawn from the experiments (e.g., recognition that compositional generalization is an area requiring improvement or that multi-modality could be more important than single-modality for certain challenging tasks) lack depth and insight.

### Questions
1. The CrossTask [1] dataset and its associated paper focus on an extremely similar study and settings. How does CompAct differ from the CrossTask benchmark? What unique contributions does your work make compared to the CrossTask paper?

2. The GAIN [2] benchmark is also a similar testing ground to the proposed CompAct in terms of evaluating models’ compositional generalizability and robustness under distribution shift. Unlike CompAct, whose atomic concepts are verbs or nouns and compositions are different verb-noun combinations, the atomic concepts in GAIN are steps, and the compositions are multi-step tasks. The authors should acknowledge these similar benchmarks and research efforts and clearly describe how this work advances the field.

3. Would experimental results on CompAct be translatable to these other similar benchmarks like GAIN, CrossTask, etc.? It would be interesting to find out.

4. Why are Maximum Compound Divergence and the Chernoff coefficient good measures for curating a dataset that requires compositional generalization, as opposed to other possible alternatives?

5. At the beginning of Section 4.1, it is mentioned that the first baseline is a text-only model to account for unexpected biases in CompAct. Why does a text-only model account for this?

6. There are many other instructional video datasets. Why was EpicKitchens-100 chosen?

7. For Noun Classification, there is the MROH baseline, which stands for Most Recent Object Heuristic. Why are there no results for the Most Recent Verb Heuristic in the Verb Classification task?

8. Why is the keyframe selection method different for ImageBind?


[1] Zhukov, Dimitri, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. "Cross-task weakly supervised learning from instructional videos." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3537-3545. 2019.

[2] Li, Junlong, Guangyi Chen, Yansong Tang, Jinan Bao, Kun Zhang, Jie Zhou, and Jiwen Lu. "GAIN: On the Generalization of Instructional Action Understanding." In The Eleventh International Conference on Learning Representations. 2022.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the compositional generalization ability of multimodal approaches including baselines that are trained from scratch and existing large-scale pre-trained models. To do that, the introduces a new dataset called COMPACT, which is curated from the EK-100 dataset by ensuring that the individual concepts (verbs and nouns) exist across training and evaluation sets, while their compositions are novel in the evaluation set. The paper proposes two tasks for evaluation: (1) next utterance prediction: predicting the descriptions of the event in next video clip and (2) atom classification, predicting only the verb/noun involved in the event in the next video clip. The paper benchmarks several neural network models on the proposed tasks, including (train-from-scratch) text-only (unimodal) and multimodal models with different combinations of modalities as well as several large scale pretrained models using prompting techniques. Results show that all multimodal models surpass the text-only baseline.

### Strengths
The detailed strengths are as follows:
1. This paper is interesting because it is trying to understand the compositional generalization capabilities of foundation models. This is a crucial skill for intelligent agents and yet there are limited work and benchmarks proposed to investigate the question. Paper in this topic should be encouraged.
1. It investigates the important topic of compositional generalization capabilities in foundational models. This is a crucial skill for intelligent agents and yet there are limited research and benchmarks in this domain. Studies like this should be encouraged.
  - However, the paper appears to have limitations in addressing this issue for large-scale pre-trained foundational models. See weaknesses for details.
2. To answer this question, the paper presents a carefully curated novel dataset from real-world videos which could be much useful for future studies.
3. The paper also designs a set of multimodal models use different combinations of modalities (including unimodal) and different ways of fusing the multi-modal information. This investigation provides valuable insight on how multi-modality inputs could influence the performance of models' compositional generalization ability.

### Weaknesses
1. The paper does not sufficiently investigate the compositional generalization ability of **foundation** models. Addressing this is challenging due to the potential distributional discrepancies between training and testing splits during their pretraining, as noted in the paper. Consequently, emphasizing "foundation models" in the title may be somewhat overstated.
   - Could incorporating domain-specific fine-tuning offer additional insights?

2. The dataset's domain-specific nature results in text descriptions that lack diversity. As a result, unlike foundation LLM, language models trained on these specific tests might be prone to overfitting and lack reasoning skills. On the other hand, other modalities, such as the vision input processed by a pretrained ResNet model, inherently resist overfitting, potentially leading to enhanced generalization. Thus, the conclusion that multi-modality contributes to improvements and that visual features consistently enhance results could potentially be invalid.

### Questions
Please see weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies the compositionality of vision language model. Specifically, it studied EPIC Kitchens-100 dataset and tailor the dataset with Maximum Compund Divergence heuristic for compositional generalization analysis.  The evaluations are performed on a number of methods, like VL, AL, OL, AVL, OAL, yet they all underperform the most recent object heuristic. The paper does not present novel algorithm or dataset, but provide an analysis for established ones.

### Strengths
The paper focus on studying the composition of foundation models on many variants, on Epic kitchen dataset that is tailored for composition evaluation.

### Weaknesses
The paper draws a conclusion that multimodal helps composition, yet from Table 1, the trend is not very clear.

### Questions
None.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
