# AutoM3L: Automated Multimodal Machine Learning with Large Language Model

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Automated Machine Learning (AutoML) stands as a promising solution for automating machine learning (ML) training pipelines to reduce manual costs. However, most current AutoML frameworks are confined to unimodal scenarios and exhibit limitations when extended to challenging and complex multimodal settings. Recent advances show that large language models (LLMs) have exceptional abilities in reasoning, interaction, and code generation, which shows promise in automating the ML pipelines. Innovatively, we propose AutoM3L, an Automated Multimodal Machine Learning framework, where LLMs act as controllers to automate training pipeline assembling. Specifically, AutoM3L offers automation and interactivity by first comprehending data modalities and then automatically selecting appropriate models to construct training pipelines in alignment with user requirements. Furthermore, it streamlines user engagement and removes the need for intensive manual feature engineering and hyperparameter optimization. At each stage, users can customize the pipelines through directives, which are the capabilities lacking in previous rule-based AutoML approaches. We conduct quantitative evaluations on four multimodal datasets spanning classification, regression, and retrieval, which yields that AutoM3L can achieve competitive or even better performance than traditional rule-based AutoML methods. We show the user friendliness and usability of AutoM3L in the user study. Code is available at:
https://anonymous.4open.science/r/anonymization_code

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a framework for applying AutoML in a multimodal setup using Large Language Models. The system comprises several stages: 1) modality inference, 2) automated feature engineering, 3) model selection, 4) pipeline assembly and 5) hyperparameter optimization. The authors divide the experiment section in 2 parts: 1) quantitative evaluation, 2) user study

### Strengths
- The problem is important as it is very common to find different use cases where many modalities are available for the prediction. Moreover, there are not many tools that aim to solve this problem, directly, so far.

### Weaknesses
- The paper is very hard to follow, with many different acronyms, stages, and components. At some points, it gives the impression to be a technical report of a very complex software, rather than a scientific paper introducing a novel method.
- Lack of strong benchmarking: the authors compare with only AutoGluon (one method) in four datasets. Although I understand that there are not many tools, the authors should include more datasets, and demonstrate that the tool also performs relatively well in uni-modal cases. Moreover, a valid baseline would be to aggregate the predictions of models that are obtained after optimizing per mode type.
- The authors do not report standard deviation to assess the significance of the results. In most of the experiments, the improvement is very small.

### Questions
- Could the authors elaborate on the time, hardware, and/or price needed for the execution? From my perspective, using an LLM for AutoML seems still very impractical, as it demands a lot of hardware, which many final users can probably not afford.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors study using LLMs for multimodal AutoML. Specifically, the authors propose AutoM3L, which can automate ML for multimodal data using natural language instructions, covering automated pipeline construction, automated feature engineering, automated hyper-parameter optimization, etc. Experimental results showcase the usage of the proposed method over AutoGluon baselines.

### Strengths
(1) Exploring the potential of LLMs for multimodal AutoML is an interesting unexplored direction.  
(2) The proposed method (or system) can leverage natural languages in the pipeline, enhancing user-friendly.  
(3) The authors have conducted user studies for the proposed method.   
(4) The authors have provided the source codes for reproduction and showcases.

### Weaknesses
(1) This paper neglects neural architecture search (NAS), which is one of the most important components in AutoML, if not the single most important one, especially in the deep learning era. There exist many multimodal NAS methods, which should be compared or added into the proposed system. Actually, I find such negligence kind of surprising, considering that NAS has received more attention than other AutoML techniques nowadays.  
(2) Experiments are somewhat weak considering essentially only AutoGluon is compared. Though other methods may focus on a certain aspect of multimodal AutoML, e.g., HPO, the authors need to properly compare with them.  
(3) Since the proposed AutoM3L is more like a library/system than a technical method, I would suggest adding more documentation, tutorials, etc., to help users get familiar with the system.  
(4) Though LLMs have been constantly improving in their abilities to follow instructions, I wonder how the uncertainty and fragileness in LLMs may potentially have on the system. This is especially important if the proposed system is applied in real production scenarios.

### Questions
See Weaknesses above

### Soundness
3 good

### Presentation
3 good

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
In the paper "AutoM3L: Automated Multimodal Machine Learning with Large Language Model", the authors present an AutoML approach based on large language models to tackle multi-modal learning tasks. In their study, they compare their approach to AutoGluon, a state-of-the-art AutoML tool that is also able to tackle multi-modal datasets, achieving competitive performance. Furthermore, a user study is conducted to compare AutoM3L to AutoGluon in terms of the time required for learning the handling of the framework, the accuracy of user actions, the usability of the framework, and the user workload.

### Strengths
- A novel paradigm for designing complex AutoML tools based on LLMs
- Competitive performance to AutoGluon across different types of tasks that exhibit multi-modality
- User study to test the AutoML tools with respect to their usability

### Weaknesses
- Tiny scope of datasets and it appears that only a single train test split has been used for the evaluation
- No significance test is applied to the evaluation results with respect to the performances and standard deviations for repetitions are missing.
- Only single runs of the AutoML tools are considered. However, AutoML tools are known to be quite noisy, so repeated runs would be required to tell how stable the performances are.
- The participants are not fully described in terms of their priming regarding the tools etc and detailed background. In particular, no previous experiences with LLMs or other AutoML tools are mentioned.
- Ablation studies regarding the effect of the different modules are lacking.
- Limitations should be elaborated more, in particular, what are the pitfalls of AutoM3L and how to deal with biases contained in LLMs? E.g., gender or racial biases? To what extent is a corresponding bias even endangering the usage of AutoM3L?

### Questions
- How stable are the performances obtained by the AutoML tools?
- What is the background of the study participants? To what extent did they already touch on LLMs and AutoML or HPO tools beforehand? To what extent are they already capable of handling multi-model data on their own?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper targets to devise an univeral AutoML framework for multimodal tasks, which has been rarely explored. In specific, this paper combines the powerful reasoning ability to their framework. Firstly, the design MI-LLM to identify the data type and AFE-LLM to facilitate the feature engineering. Then an MS-LLM is devised to select the suitable encoder for each modaliyu. Finally, PA-LLM and HPO-LLM generates corresponding excutable codes and optimal hyper-parameters for training model. The experiments of comparison with AutoGluon show the proposed method can outperform the competing baseline.

### Strengths
+S1: This paper has explored how to combine the LLMs with AutoML framework at an early stage.
+S2: The authors provide many details of implementation for their framework, which can ease the reproduction of the work.
+S3: The paper is well-writen, which is easy to understand.

### Weaknesses
-W1: Though the motivation to combine the LLMs is clear, no technical difficulty is seen for combining LLMs with AutoML. It seems only a simple application of LLMs to AutoML, which may degrade the contributions of this paper.
-W2: This paper only introduce few related works, but lack of sufficient relevant work collection. The authors claim that AutoGluon is the only work for automl multi-modal, but I find several other related works [1][2][3].
only one baseline is compared. In my view, you can compare with the variants of some existing approach.
-W3: Some designs in the proposed framwork seems abundant. For example, is it necessary to design the modality inference module? In general, the data format is pre-defined and given by the dataset.
-W4: Some errors exist in the paper. For example, in figure 2(a), the text in outputs_1 should be "state" but not "stage"?
-W5: Lack of related baselines, which is relevant to the weakness W2. Also, I find some baselines in AutoGluon compared, such as H2O AutoML. In my view, these baselines also should be included in the experiments.

[1] Jin, H., Chollet, F., Song, Q., & Hu, X. (2023). Autokeras: An automl library for deep learning. Journal of Machine Learning Research, 24(6), 1-6.
[2] Sun, P., Zhang, W., Wang, H., Li, S., & Li, X. (2021). Deep RGB-D saliency detection with depth-sensitive attention and automatic multi-modal fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 1407-1417).
[3] Erickson, N., Shi, X., Sharpnack, J., & Smola, A. (2022, August). Multimodal automl for image, text and tabular data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 4786-4787).

### Questions
Q1: Does the AFE-LLM only can handle the tabular features, instead of multi-modal features? If it is, the idea is much similar to [4]. Besides, it seems that you conduct such feature engineering for each sample in dataset. I think it is extremely time-consuming, which may conflict the intuition of AutoML.
Q2: Besides, there seems no specific multi-modal information is utilized in the proposed method. Only text path or image path are adopted. If it is, all other single-modal AutoML framework may be adpated to such task. 
Q3: Please also respond the questions mentioned in weakness.

[4] Borisov, V., Sessler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2022, September). Language Models are Realistic Tabular Data Generators. In The Eleventh International Conference on Learning Representations.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
