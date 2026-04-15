# Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models

- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
One fascinating aspect of pre-trained vision-language models (VLMs) learning under language supervision is their impressive zero-shot generalization capability.
However, this ability is hindered by distribution shifts between the training and testing data.
Previous test time adaptation (TTA) methods for VLMs in zero-shot classification rely on minimizing the entropy of model outputs, tending to be stuck in incorrect model predictions.
In this work, we propose TTA with feedback to rectify the model output and prevent the model from becoming blindly confident.
Specifically, a CLIP model is adopted as the reward model during TTA and provides feedback for the VLM.
Given a single test sample,
the VLM is forced to maximize the CLIP reward between the input and sampled results from the VLM output distribution.
The proposed \textit{reinforcement learning with CLIP feedback~(RLCF)} framework is highly flexible and universal.
Beyond the classification task, with task-specific sampling strategies and a proper reward baseline choice, RLCF can be easily extended to not only discrimination tasks like retrieval but also generalization tasks like image captioning,
improving the zero-shot generalization capacity of VLMs.
According to the characteristics of these VL tasks, we build different fully TTA pipelines with RLCF to improve the zero-shot generalization ability of various VLMs.
Extensive experiments along with promising
empirical results demonstrate the effectiveness of RLCF.
The code is available at https://github.com/mzhaoshuai/RLCF.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper solves test-time adaptation of Vision-Language-Models to improve the zero-shot generalization performances. Unlike previous works that rely on entropy of model outputs, the authors propose to leverage reward from CLIP model as feedback to adapt models. The proposed method based on reinforcement learning with CLIP feedback is evaluated on three tasks, including zero-shot image classification, zero-shot text-image retrieval, zero-shot and cross-domain image captioning, showing improved performances.

### Strengths
+ The paper solves an important task of test-time adaptation of vision-language models. VLMs have played a critical role in many CV and NLP tasks.  How to improve their zero-shot generalization ability, especially in the challenging task of test-time adaptation, is a worthwhile research problem.
+ The authors propose to leverage CLIP feedback for adapting models. This seems to be novel and is interesting to me. As many vision-language learning methods are built upon CLIP, the intergration of CLIP reward is natural.
+ The proposed RLCF method universally applies to different tasks, like zero-shot classification, text-image retrieval, image captioning. Hence, the idea may inspire research in many other VL tasks.
+ The method is simple yet effective. The experiments and analyses are extensive in the paper. For each task, RLCF is compared with a few  baseline methods, showing significant improvements.
+ The paper is very easy to follow with a clear motivation and method descriptions. Visualizations (e.g. fig.1) also look nice.

### Weaknesses
- The method relies on good quality of CLIP feedback. This may restrict its applications in tasks other than images with generic objects. For example, CLIP shows less satisfactory accuracies on fine-grained datasets like FGVC Aircraft, EuroSAT, CUB, etc. CLIPScore may be not informative in the fine-grained classification.
- The authors use CLIP-ViT-L and a few other variants for calculating reward. The commonly used backbone in VL learning is CLIP-ViT-B. It is understandable that CLIP-ViT-L leads to better text-image alignment. But discussions on CLIPScore with CLIP-ViT-B would be helpful to understand how robust RLCF is to unreliable rewards.

### Questions
- In Eq.(4), why need to clip CLIP score to be non-negative? What 'it encourages all model behaviors' means? In Eq.(5), any difference if not subtracting the expectation term?
- The authors mention RLCF-S adopts weighted reward sum and RLCF-S-M adds a momentum buffer. Could the authors explain more about this? Also as mentioned in the weakness, more comments on using less reliable CLIP like CLIP-RN50, CLIP-B would be appreciated.

### Soundness
3 good

### Presentation
4 excellent

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
This paper studies the test time adaptation methods for vision-language models. The authors propose a framework, called reinforcement learning with CLIP feedback  (RLCF).  Specifically, given a test sample, the VLM is optimized to maximize the CLIP reward, which is provided by a CLIP model.  RLCF can be applied to image classification, text-to-image/image-to-text retrieval, and image captioning.  Extensive experiments demonstrate the effectiveness of RLCF.

### Strengths
- This paper is well-organized and easy to follow.

- The proposed RLCF framework is universal and applicable across various VL tasks.

- The experiments on three tasks show the superior effectiveness of RLCF, compared to TPT, KD, and Pseudo-label.

### Weaknesses
- The proposed RLCF utilizes the CLIP model to provide a reward score, which is very similar to a recent work [1]. The two tweaks are sampling strategies and adding a baseline to the reward function. It weakens the novelty of the proposed method. Besides, the comparison between the proposed RLCH and [1] is missing.

- The authors state that compared to pseudo-label and KD, the feedback mechanism combines the merits of both the student and teacher. However, the authors only investigate the original version of KD proposed in 2015. Moreover, I recommend that the authors evaluate the ensemble of the student and teacher model, which can directly combine their merits.

- For the experiments on image captioning, I suggest that the authors add the results of  CLIPCap and CapDec with the CLIP-ViT-L/14 architecture (the teacher model) for a more comprehensive comparison.

- Could the authors provide more implementation details of TPT+KD?

- Typo: kD->KD on page 8


[1] Cho, Jaemin, et al. Fine-grained Image Captioning with CLIP Reward. In Findings of NAACL, 2022.

### Questions
- More advanced KD and the ensemble of the student and teacher model.
- The results of  CLIPCap and  CapDec with the CLIP-ViT-L/14 architecture.
- More implementation details of TPT+KD.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This article suggests a Reinforcement Learning strategy using a CLIP-based model suitable for implementation during test time in the zero-shot learning framework. The suggested concept was tested on three distinct tasks within the zero-shot framework: image classification, image retrieval, and image captioning.

### Strengths
1- The paper introduces a novel reinforcement learning-based reward function to enhance the efficacy of the CLIP-based approach, which can be employed during test time for a new dataset.

2- The suggested concept is applied to three distinct tasks: image classification, test to image retrieval, and image captioning, all of which are experimented within the zero-shot framework.

3-  The experiments were conducted on the Imagenet dataset for all three tasks - image classification, image retrieval, and image captioning, demonstrating superior performance compared to recent state-of-the-art methodologies.

### Weaknesses
1- The proposed method is employed within the zero-shot learning framework for all three tasks: image classification, image retrieval, and image captioning. It would indeed be intriguing to explore whether it can also be applied within a few-shot learning framework?

2-  Figure-2 illustrates the application of a data augmentation strategy for image classification. Is this same strategy for augmentation also utilized for textual data?

3-  Figure-6 demonstrates the visual outcomes of step-1 and step-4, depicting the progression from the worst to the best. Including results from all steps, such as step-1, step-2, step-3, and step-4, would offer a more comprehensive understanding of how the transformation from the worst results to the best unfolds.

### Questions
Please see the questions raised in the weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
