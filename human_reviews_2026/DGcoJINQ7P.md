# EgoBrain: Synergizing Minds and Eyes For Human Action Understanding

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The integration of brain-computer interfaces (BCIs), in particular electroencephalography (EEG), with artificial intelligence (AI) has shown tremendous promise in decoding human cognition and behavior from neural signals. In particular, the rise of multimodal AI models have brought new possibilities that have never been imagined before. Here, we present \data --the world's first large-scale, temporally aligned multimodal dataset that synchronizes first-person (egocentric) vision and EEG of human brain over extended periods of time, establishing a new paradigm for human-centered behavior analysis. This dataset comprises 61 hours of synchronized 32-channel EEG recordings and first-person video from 40 participants engaged in 29 categories of daily activities. We then developed a muiltimodal learning framework to fuse EEG and vision for action understanding, validated across both cross-subject and cross-environment challenges, achieving an action recognition accuracy of 66.70\%. EgoBrain paves the way toward a unified framework for multimodal and egocentric brain–computer interfaces, bridging neural signals and first-person perception. Our dataset and code are publicly available at: https://huggingface.co/datasets/ut-vision/EgoBrain and https://github.com/ut-vision/EgoBrain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, a new egocentric dataset is proposed which includes the modality of EEG recordings. Unique to this dataset, participants wore an EEG device whilst recording the videos so that their intentions could be better modelled and improve performance. The dataset includes ~61 hours of recording across 4 major classes which are split into 10 verbs which are split into 29 actions. An extension of the Time Interval Machine which instead of audio uses the EEG modality is presented as a baseline and results show that whilst visual only gives a strong baseline, the inclusion of EEG recordings is complementary, providing a boost in overall performance.

### Strengths
* The dataset represents the first dataset of its kind combining both egocentric visual data and EEG recordings, thus opening up a new research area.
* The dataset includes a good variety of actions across the ~61 hours.
* The results showcase the benefit of the EEG modality when combined with the vision modality.
* The paper is generally well-written and easy to follow, there are very few issues with spelling/grammar/presentation.

### Weaknesses
During the description of the dataset, it is not clear within the main paper what the difference is between the verbs Play(I), Play(II), and Play(III) are. This becomes evident within the appendix that it is based on device, but should be included earlier.

The 3 level hierarchy of high-level class, verb, and action is presented but it is unclear whether this hierarchy/taxonomy is utilised in some way during training or whether this was considered at all.

The benchmarking of the dataset could be improved in a few different ways:
* Firstly, there is no random performance given, the gap between the visual modality and the EEG modality is quite large, with the latter reaching only 21% and 10% on the dataset. It would be good to include these numbers to get a sense of the lower bound and how far the EEG only performance is.
* Secondly, whilst the confusion matrices are provided for the verb classification, these are across two 'clusters' of verbs, and do not show all the misclassifications (for example the operate row only adds up to 0.94). Including full confusion matrices for both actions and verbs (potentially in the appendix) would be interesting to look at to check the biases of the model/data and/or providing per class accuracy metrics would be interesting to see how the performance differs. Given that the dataset is imbalanced in terms of length (though seems uniform from a class frequency perspective) it would be good to know if this impacts model training.
* Finally, on the cross-subject only results, the method is already achieving 90% and 80% on the verb/action classification respectively, showcasing that this split is almost saturated already, some discussion regarding this and the more challenging cross-subject and cross-scene setting would be interesting to see.

### Questions
1. What are the differences between the different play labels, i.e. play(I), play(II), play(III), is this referring to computer, physical, mobile? Why were they split up this way, to make sure that the distribution wasn't too long-tailed/dominated by a majority class? (This becomes clear after looking at the appendix but isn't mentioned in the main paper)
2. For the Cross-Subject&Cross-Scene split, it is not explicitly mentioned, but I assume the 6 new sessions in the test set have different subjects to the 28 in training?
3. It would be good to know the random performance of the new dataset for Table 1 to get a sense of the lower bound. The Brain-only results are very low in comparison to the visual only and visual + brain for example and it would be interesting to know the relative performance of this model compared to random.
4. From the dataset construction, there is a 3 stage hierarchy, including the high level classes, the verbs, and the actions themselves, was this considered at all for the method/dataset splits?
5. Why was the verb confusion matrix split into two, just for space reasons? And why were the confusion matrices of the actions excluded?
6. Given the imbalanced nature of the dataset, was an investigation conducted into whether the models are biased towards certain classes would be interesting to see, i.e. looking into per-class accuracy metrics or similar? The confusion matrices shows this in some fashion for verbs, but doesn't include the entire matrix and doesn't include actions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces EgoBrain, the first large-scale, temporally aligned multimodal dataset combining egocentric video and EEG signals for real-world human action understanding. It includes 61 hours of synchronized EEG and first-person video from 40 participants performing 29 categories of daily activities. The authors also propose a Brain-TIM model (Brain-Time Interval Machine), a multimodal transformer-based framework that fuses EEG and vision representations via time-interval MLPs and modality-specific embeddings to capture temporal and cross-modal dependencies. Experiments demonstrate that multimodal fusion improves action recognition performance over unimodal baselines, particularly in cross-subject and cross-scene generalization.

### Strengths
1.	First-of-its-kind dataset integrating real-world egocentric vision with EEG; extensive and ethically curated.
2.	Clear methodological design with well-justified architecture choices (temporal embeddings, modality-aware tokens).
3.	Insightful qualitative results showing when EEG signals complement vision (e.g., occlusion or intent disambiguation).

### Weaknesses
1.	The synchronization precision is stated as <1s jitter. This is a relatively large jitter for fast-changing neural signals and short actions. This level of jitter could potentially limit the precise time-locking necessary for analyzing rapid neural correlates of action initiation or error. A discussion on how the Brain-TIM model's windowing strategy mitigates the impact of this 1s jitter is needed.

2.	The model’s novelty is limited, Brain-TIM primarily applies existing time-embedding concepts to a new modality pair. Moreover, the framework relies solely on two pre-trained models for feature extraction and classification, which restricts the scope of analysis. How would other models or architectures perform on this dataset? Additional comparative experiments are essential to validate the dataset’s generality and utility.

3.	The reported accuracy improvements (≈1–3%) may be statistically marginal, raising questions about their practical significance. The authors should provide statistical validation (e.g., p-values or confidence intervals) to confirm whether these gains are significant rather than due to random variation.

4.	The ablation study (Table 2) demonstrates the contribution of the temporal/modality embeddings, but it doesn't directly test the core hypothesis of fusion effectiveness by removing one entire modality from the multimodal setting. Specifically, in the "Visual & Brain" section, the base case uses both the Visual and Brain encoders. It would be insightful to see an ablation where the fusion mechanism is active, but one modality's tokens are zeroed out or entirely removed to isolate the Transformer's fusion contribution, separate from the unimodal encoders' raw output strength.

5.	The Cross-Subject & Cross-Scene task is expected to be more challenging than the Cross-Subject-Only task; however, the “Brain Only” model surprisingly achieves higher action-classification accuracy in this setting. In contrast, VideoMAE experiences a large accuracy drop, while LaBraM shows only a minor decrease. The paper should clarify why EEG-based performance remains stable (or improves) across scenes, despite EEG being more sensitive to noise, and why VideoMAE—supposedly a stronger zero-shot model—shows a significant decline.

6.	It is unclear whether the dataset includes cross-session experiments, i.e., whether the same subjects were recorded again after a time interval performing the same tasks. Including or clarifying this aspect would be valuable for evaluating intra-subject consistency and longitudinal generalization.

### Questions
Please refer to the Weaknesses section for detailed questions and suggestions to the authors.

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
This paper introduces a multi-modal dataset for Action Classification, including synchronized egocentric videos and 32-channel encephalography recordings. Additionally, this paper proposes a baseline method that achieves an overall accuracy of 80.16% in action classification, utilizing both data modalities. This paper empirically demonstrates that the information extracted from encephalography may be complementary to visual data.

### Strengths
This review evaluates the paper's quality based on the following criteria: task relevance, related work, technical novelty, technical correctness, experimental validation, writing and presentation, and reproducibility. Each aspect is discussed and highlighted as a strength or a weakness in the sections below.
-    **Dataset Contribution and Reproducibility:** This paper contributes to the community a dataset of synchronized egocentric videos and 32-channel encephalography recordings. However, it is not explicitly indicated whether the source code will be released, and it is not included as part of the submission.
-    **Writing and presentation:** Overall, this paper is easy to read and well-written.

### Weaknesses
-    **Relevance of the task and Experimental Validation:** Even though Action Classification from egocentric videos and encephalography recordings may be a relevant problem for the ICLR community. The motivation behind including this novel data type modality is not well stated in the paper's introduction. This paper already reports high performance for the proposed task, so it may probably saturate fast. Considering these results, what are the reasons to keep the data acquisition as simple as possible to not make the task harder?
-    **Technical Correctness and Related Work:** This paper overclaims about contributing a “large-scale” dataset when its size is not comparable to current state-of-the-art benchmarks for human action recognition from egocentric vision data. Moreover, it states that

### Questions
1.	Will the source code and pretrained models be released to support reproducibility? If so, what is the reason for not including them in the supplementary material?
2.	What is the motivation for including encephalography data?
3.	Given the already high reported accuracy, how does the paper address concerns about task saturation?
4.	Why was the decision made to keep data acquisition simple, and how does this impact the task's difficulty and generalizability?
5.	On what basis is the dataset described as “large-scale,” and how does its size compare to existing benchmarks?
6.	The paper states that prior datasets only use visual data, but most include multiple modalities. Can this claim be clarified?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces EgoBrain, a large-scale, multimodal dataset of 61 hours of synchronized egocentric video and 32-channel EEG signals. This data was collected from 40 participants performing 29 different daily activities such as work, play, learn, and consume within a controlled laboratory setting. The authors also propose a model, Brain-TIM which fuses the visual and EEG signals data using a temporal-aware embedding mechanism and modality-specific encodings for action understanding. The paper evaluates this model on action and verb classification tasks for cross-subject and cross-subject & cross-scene settings. The results indicate that the multimodal model of using both vision and EEG modality achieves a 66.70% accuracy, representing a 3.30% absolute improvement over a visual-only baseline in the cross-scene setting.

### Strengths
1. The paper contributes a new dataset, EgoBrain which has synchronized video and EEG signals which can be valuable for computer vision research.  
2. The paper shows that that EEG signals can be a useful modality for tasks such as action recognition when the visual modality is occluded.
3. The paper shows analysis on cross-subject and cross-subject & cross-scene analysis which is a challenging benchmark to evaluate the model generalization.

### Weaknesses
1. The architecture method of Brain-TIM seems to be incremental when compared to TIM [1]. The architecture presented in the paper of modality-specific encoders, embedding layers, Time-Interval MLP, and a Transformer encoder seems to be a direct application of the existing TIM framework to a new pair of modalities. Can the authors clarify the differences between TIM and Brain-TIM? Is Brain-TIM just an extension of TIM to multiple modalities?
2. While the idea of using EEG signal to understand egocentric actions is well-motivated, can the authors discuss the application of their proposed approach to real-world scenarios where collecting EEG signals can be hard and require specific hardware?

[1]. Chalk, Jacob, et al. "Tim: A time interval machine for audio-visual action recognition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
Since the dataset has been collected in a lab setup with isolation chamber with limited limb movement, can the authors discuss the robustness of the EEG signal? How much noise can the EEG signal have and still be able to give better action recognition results than just using the visual modality?

### Soundness
3

### Presentation
3

### Contribution
3
