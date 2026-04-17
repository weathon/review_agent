# ReAugment: Learning to Augment Few-Shot Time Series with Model Zoo Guidance

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Time series forecasting, particularly in few-shot learning scenarios, is challenging due to the limited availability of high-quality training data. To address this, we present a pilot study on using reinforcement learning (RL) for time series data augmentation. Our method, ReAugment, tackles three critical questions: which parts of the training set should be augmented, how the augmentation should be performed, and what advantages RL brings to the process. Specifically, our approach maintains a forecasting model zoo, and by measuring prediction diversity across the models, we identify samples with higher probabilities for overfitting and use them as the anchor points for augmentation. Leveraging RL, our method adaptively transforms the overfit-prone samples into new data that not only enhances training set diversity but also directs the augmented data to target regions where the forecasting models are prone to overfitting. We validate the effectiveness of ReAugment across a wide range of base models, showing its advantages in both standard time series forecasting and few-shot learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposed ReAugment, a novel data augmentation method driven by reinforcement learning.  ReAugment is different from the existing data augmentation methods and uses the RL mechanism for effective augmentation. ReAugment significantly boosts forecasting performance while maintaining minimal computational overhead by leveraging a learnable policy to transform the overfit-prone samples.

### Strengths
1. The viewpoints presented in this work are quite interesting and may bring new insights to the field.

2. The analysis of the existing works in this work is very accurate.

### Weaknesses
1. The efficiency issue in the training phase of reinforcement learning should be discussed more. The discussion of Table 6 in the manuscript is overly simplistic. It is suggested that more experiments be conducted to demonstrate the computational cost of ReAugment.

2. The insight of using reinforcement learning for data augmentation is interesting. However, the two objectives of Line 256 are mutually exclusive. Can more discussions and experimental explorations be conducted on these two objectives?

3. As shown in Fig 1, the performance improvement of ReAugment is insignificant in the Exchange. The reasons of this phenomenon need to be analyzed in detail.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a RL based method to data augmentation for time-series forecasting tasks. The method includes the use of cross-validation errors for identifying potentially data samples prone to overfitting, use this samples to anchor VAE-based augmentation, and use this VAE as an actor in RL to derive the augmentation policy guided by the goal to increase diversity while remaining close to the original samples. Experiments were performed on five public time series datasets for few-shot as well as standard training settings, in comparison to several alternative data augmentation methods.

### Strengths
- The presented idea is overall novel and interesting, especially to learn augmentation policy as a RL problem.

- The use of cross validation (under the name of model zoo) to identify overfitting-prone samples has merits. 

- The analysis showing the impact of the percentage of data augmented (Fig 4a) is insightful, especially the observation that augmenting all data is less effective.

### Weaknesses
- The design of the RL formulation is not well justified in terms of the underlying MDP and the justification for the choices of the state and reward functions. It is not clear what are the states, or why the use of masked time series data and timestamps are sufficient as state variables. The definition of Var in Equation 3 is also very unclear and, more importantly, it is curious that the reward function does not seem to care about the forecasting model performance on the augmented data (but only the diversity and fidelity of the augmented samples). It seems that it is possible that such an augmentation policy could lead to the augmentation samples that will deteriorate the forecasting function, as the augmentation is not guided by this important goal.

- The use of the VAE as the actor is also not well rationalized — why is the prior network chose as the policy network to be optimized in RL (instead of any other components, such as the decoder that is actually generating the augmented samples).


- Statistics from multiple runs should be included for all baselines and experiments, not just for the presented method, especially in datasets where the margins of improvements are small.

### Questions
- The design choices for the MDP underlying the RL formulation need to be clearly motivated, described, and experimentally verified (e.g., the choice of which part of the VAE is considered as the policy network, what states to use, etc).

- Adding statistics from multiple runs for statistical significance of the presented method.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the limited availability of high-quality training data. An RL-based method is proposed that 1) overfit-prone data are identified as augmentation anchors, 2) a tailored VMAE for sequential data produces sample-aware augmentations, and 3) an RL approach is used to train the augmentation network guided by a reward function derived from a forecasting model zoo. Experiments have been conducted to prove the effectiveness of the proposed method compared with various base models.

### Strengths
- The problem of limited high-quality training data is critical, and the idea of identifying the overfit-prone samples is important.
- The description of the methodology is well-structured.
- The experiments showed the improvement of the proposed method.

### Weaknesses
- More details of the methodology need to be disclosed.
- The experimental results and the proposed new metrics need to be justified.

### Questions
- Section 3.1: What’s the motivation for two subsets of the training set? How would the choices of the subsets affect the performance?
- Section 3.2: The authors state that the overfit-prone samples are more likely to negatively affect the training quality of forecasting models. Is this statement universal across other datasets?
- The improvement of the proposed method in Table 1 and 2 is not significant. Could the authors provide visualizations to justify the metric numbers? Also, it would be more consistent to have the variance of all metrics available in tables.
- Could the authors elaborate on the new metrics proposed in Section 5.2? It is not clear what is expected for this metric and the meaning of negative values in Table 3.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper deals with the problem of time-series forecasting and proposes a method for augmenting training time-series examples. More specifically, the proposed method first trains a variational masked autoencoder (VMAE) with the original training examples and then chooses promising training examples that can be a source of augmentation. The proposed example selection introduces a training scheme inspired by a policy gradient method, and its reward is defined by variance of predictions obtained from a group of time-series estimators. In this framework, the distribution parameter predictor of the VMAE can be regarded as a "policy network" in the context of reinforcement learning.

### Strengths
S1. The problem dealt with in this paper is significant. Optimizing data augmentation schemes is one of the significant components for obtaining sufficient prediction performances in low-resource regimes.

S2. The strategy of the proposed method is reasonable and technically sound. Reinforcement learning is one of the standard approaches for automatically optimizing data augmentation schemes. The proposed method incorporates the idea of REINFORCE, one of the standard policy gradient methods, into this optimization. Also, focusing on prediction variances as a measure for "goodness" in data augmentation is reasonable.

S3. This paper is well-written and easy to follow.

### Weaknesses
W1. Novelty of the proposed method against several previous methods should be justified literally and hopefully its effectiveness should have been demonstrated by experimental comparisons. One of the key ideas of the proposed method is to choose promising training examples useful for data augmentation, which have already been explored in the following papers:

  - [Yang+ ICCV2025 https://arxiv.org/abs/2506.21037] (Note: This paper was officially published AFTER the ICLR2026 deadline, which indicates that this paper should not be a source of loss of novelty.)
  - [Huang+ IEEE TGRS 2024 https://ieeexplore.ieee.org/document/10714383]

Although I understand that details of the proposed method is different from those of the above previous methods, novelty of the proposed method and its effectiveness should have been justified.

W2. I could not understand the reason why only the proposed method has standard deviation in the experimental results. Every augmenter has randomness, which indicates that experimental results can be variable according to the randomness. Presenting the deviation is significant for understanding statistical significance for the presented experimental results.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
2
