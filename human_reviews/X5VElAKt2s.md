# LoRA ensembles for large language model fine-tuning

- Decision: Reject
- Scores: 5, 5, 3

## Abstract
Finetuned LLMs often exhibit poor uncertainty quantification, manifesting as overconfidence, poor calibration, and unreliable prediction results on test data or out-of-distribution samples. One approach commonly used in vision for alleviating this issue is a deep ensemble, which constructs an ensemble by training the same model multiple times using different random initializations. However, there is
a huge challenge to ensembling LLMs: the most effective LLMs are very, very large. Keeping a single LLM in memory is already challenging enough: keeping an ensemble of e.g. 5 LLMs in memory is impossible in many settings. To address these issues, we propose an ensemble approach using Low-Rank Adapters (LoRA), a parameter-efficient fine-tuning technique. Critically, these low-rank adapters represent a very small number of parameters, orders of magnitude less than the underlying pre-trained model. Thus, it is possible to construct large ensembles of LoRA adapters with almost the same computational overhead as using the original model. We find that LoRA ensembles, applied on its own or on top of pre-existing regularization techniques, gives consistent improvements in predictive accuracy and uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces LoRA ensembling to improve LLMs' performance and uncertainty calibration. The core idea is to train multiple LoRA adapters and ensemble them to get more accurate and calibrated predictions, as typically shown in deep ensembling literature. Experiments results verify that LoRA ensembling improves over baseline approaches.

### Strengths
- The method is built on top of well-known results that model ensembles can lead to more accurate and calibrated predictions.
- LoRA ensemble alleviates the need to finetune and update the entire model which is computationally prohibitive.
- Experiment results show that LoRA ensemble does lead to more accurate results as well as reduced calibration error.

### Weaknesses
- It is rather straightforward to consider LoRA finetuning for ensembling. The technical novelty of the proposed method is a bit limited.
- There are several relevant and stronger baselines not considered in the experiments, including calibration for in-context learning [1], and self-consistency [2], both of which shows decent improvements on prediction accuracy.
- The current set of experiments considered is limited to multiple choice questions (predicting only a single token). While the method is indeed compatible with generative tasks, there are no such tasks considered in the experiments. The results would be more convincing with generative tasks since LLMs are commonly used for complex tasks.
- LoRA ensemble requires finetuning datasets. However, LLMs are commonly used in zero or few-shot ways in many scenarios. How does the method perform if only a few finetuning data is available?

[1] Calibrate Before Use: Improving Few-Shot Performance of Language Models. Zhao et al. 2021.

[2] Self-Consistency Improves Chain of Thought Reasoning in Language Models. Wang et al. 2022.

### Questions
- The presentation of experiment results are scattered. It would be easier to compare different methods if they are all listed in the same table/figure. Currently, comparisons to different baselines are in separate tables/figures, making it a bit hard to read.
- In Figure 3, 4, 5, and 6, why ECE increases as number of epoch increases? In this case, isn't the method making calibration worse by LoRA finetuning?
- In many cases, finetuning on specific dataset can compromise the model's performance on other tasks. Does LoRA ensemble suffer the same problem?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This conference paper discusses issues of Large Language Models (LLMs) in overconfidence and uncertainty in their predictions. To mitigate these issues, the paper proposes a new approach called LoRA ensembles, which leverages low-rank adapters and random initialization to create diverse model components. This method addresses the limitations of traditional ensemble approaches, such as excessive storage requirements and a lack of diversity in fine-tuned LLMs. The authors demonstrate the effectiveness of LoRA ensembles in improving accuracy and calibration for various reasoning tasks compared to alternative fine-tuning methods and introduce the concept of regularized LoRA ensembles to further enhance performance and address potential correlations between components. This research enables the scaling of ensemble methods to LLMs with billions of parameters, offering a promising solution for enhancing the reliability of LLM predictions in safety-critical applications like medical diagnosis, finance, and decision-making processes.

### Strengths
(1) The paper is well-written and easy to understand.

(2) This paper analyses potentials of LoRA ensemble with various techniques, such as regulizers, Dropout, weight decay. 

(3) The ablation study of LoRA with randomness is interesting.

### Weaknesses
(1) One major concern is the idea is very naive and straightforward. The performance improvement of deep ensemble is already well-known  to the community, and it is in no way surprising that we can combine LoRA with ensemble to improve the performance, uncertainty, etc.

(2) Another concern is that no computational costs is reported in this paper. I understand the inference costs of LoRA ensemble is much lower than traditional finetuning ensemble, but it is good to demonstrate this. 

(3) The title in the paper doesn't exactly match the title in the openreview system.

(4) Figure 7 is not easy to understand. It is better to put all variants in one figure or use tables to compare the performance of various combination of methods.

### Questions
Please see the above weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper argued ensembling LLMs is computationally challenging due to the sheer size of such models. In light of this, the authors proposed a solution to create an ensemble of models with Low-Rank Adapters (LoRA), referred to as LoRA ensembles. These ensembles are much smaller in terms of parameters and can be efficiently constructed on top of underlying pre-trained models. The results demonstrate that LoRA ensembles, when applied independently or in combination with other regularization techniques, offer improved predictive accuracy and achieve better uncertainty quantification.

### Strengths
- The introduction of LoRA for ensembling is a unique approach, particularly for large models like LLMs. This could be a  useful exploration of how to ensemble such massive models.
- LoRA ensembles, whether used independently or in conjunction with other techniques, demonstrate enhancements in both prediction accuracy and the quantification of uncertainty.
- The observation that regularization may benefit calibration over just the improvement from ensembling can hold practical value.

### Weaknesses
- The paper lacks a comprehensive survey of existing ensemble methods, and it does not adequately discuss or compare with related works such as [1,2,3,4,6] in the literature.
- The focus of the paper is only on prediction ensembles, which neglects the important weight ensemble methods [1,3,4,6]. The paper argues that maintaining an ensemble of, for instance, 5 LLMs in memory can be challenging in certain scenarios. However, it's worth noting that weight ensembles require the maintenance of just one model. Recent papers adopt online ensemble methods [3] that continuously average weight parameters.
- The concept of ensembling adapters in LLMs has been previously explored in [5], yet this prior work is neither discussed nor compared in the paper.
- The method's evaluation is restricted to small datasets, and its scalability remains unverified. Furthermore, the absence of actual ensemble baselines is notable. For example, [1,3] employ ensemble techniques while training the model only once, which is highly relevant to the task addressed in this paper.

A minor issue is the presence of a discrepancy between the title displayed on OpenReview and the actual title in the paper.

[1] Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity. ICLR 2021

[2] Training Independent Subnetworks for Robust Prediction. ICLR 2020

[3] SWAD: Domain Generalization by Seeking Flat Minima. NeurIPS 2021

[4] DNA: Domain generalization with diversified neural averaging. ICML 2022

[5] AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models. EACL 2023

[6] Averaging Weights Leads to Wider Optima and Better Generalization. UAI 2018

### Questions
Please refer to the weaknesses mentioned above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
