# Semi-Supervised End-To-End Contrastive Learning For Time Series Classification

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3, 6

## Abstract
Time series classification is a critical task in various domains, such as finance, healthcare, and sensor data analysis. Unsupervised contrastive learning has garnered significant interest in learning effective representations from time series data with limited labels. The prevalent approach in existing contrastive learning methods consists of two separate stages: pre-training the encoder on unlabeled datasets and fine-tuning the well-trained model on a small-scale labeled dataset. However, such two-stage approaches suffer from several shortcomings, such as the inability of unsupervised pre-training contrastive loss to directly affect downstream fine-tuning classifiers, and the lack of exploiting the classification loss which is guided by valuable ground truth. In this paper, we propose an end-to-end model called SLOTS (Semi-supervised Learning fOr Time clasSification). SLOTS receives semi-labeled datasets, comprising a large number of unlabeled samples and a small proportion of labeled samples, and maps them to an embedding space through an encoder. We calculate not only the unsupervised contrastive loss but also measure the supervised contrastive loss on the samples with ground truth. The learned embeddings are fed into a classifier, and the classification loss is calculated using the available true labels. The unsupervised, supervised contrastive losses, and classification loss are jointly used to optimize the encoder and classifier. We evaluate SLOTS by comparing it with ten state-of-the-art methods across five datasets. On an EEG-based emotion recognition task using the DEAP dataset with only 10% labeled data, SLOTS significantly outperforms two-stage baselines, achieving up to a 16.10% higher F1 score (compared to TS-TCC) and a 38.49% higher absolute accuracy (compared to TS2Vec) when the labeling ratio increases to 100%. SLOTS also attains the best performance on four diverse datasets with an average 3.55% margin in F1. In various evaluation setups, including leave-trials-out and leave-subjects-out, SLOTS consistently achieves top performance. The results demonstrate that SLOTS is a simple yet effective framework. When compared to the two-stage framework, our end-to-end SLOTS utilizes the same input data, consumes a similar computational cost, but delivers significantly improved performance. Crucially, our end-to-end framework is model-agnostic, allowing it to be seamlessly integrated with any existing self-supervised contrastive model in order to enhance its performance. We release code and datasets at https://anonymous.4open.science/r/SLOTS-242E.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a semi-supervised contrastive learning approach for time series classification tasks. The main contribution of the model is the proposed hybrid loss, which is a weighted sum of a self-supervised contrastive loss, a supervised contrastive loss, and a classification loss. Experiments on multiple time series classification datasets suggest that the proposed method is superior to prior two-stage contrastive learning methods.

### Strengths
1. The paper is easy to understand.
2. The methods are sound and the experimental designs are reasonable.
3. Strong empirical results compared to prior two-stage contrastive learning methods.

### Weaknesses
1. Technical contribution is marginal. Both self-supervised and supervised contrastive losses (Equations 4-5) exist in the literature and have been well studies. The hybrid loss was simply a weighted sum of the three losses. The backbone encoder and temporal masking augmentation was from the literature too. 
2. No details about data split and hyperparameter selection.
3. Comparisons to baselines are not fair. Default hyperparameters were used for baselines. However, some baselines such as SimCLR was introduced in computer vision domain, and thus using default hyperparameters on time series data would result in suboptimal performance for the baselines. If hyperparameters for SLOTs were tuned, then hyperparameters for baselines should be tuned on the same data too.
4. Citation format is wrong, which reduces the readability of the paper.

### Questions
1. In SimCLR, a large number of negative pairs is needed. How many negative pairs were used in the self-supervised contrastive loss for SLOTS?
2. In ablation studies (Table 3), it would be better to see individual effects of supervised contrastive loss and classification loss.
3. How was data split done? How were hyperparameters selected?
4. If hyperparameters for SLOTS were optimized, please optimize hyperparameters for baselines for fair comparison.
5. How were baseline end-to-end models trained? Please clarify in "Baselines" section. Same question applies to two-stage version of SLOTs.
6. Please correct the citation format.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a new method for semi-supervised learning of time-series. The method is based on contrastive learning for both supervised and unsupervised loss terms. The method uses DEAP, SEED, HAR, EPILEPCY, and P19 to evaluate the method.

### Strengths
The paper has several strengths: 

- The paper targets an important and often neglected area in time-series representation learning. 

- The paper is well-written and easy to follow. 

- 5 datasets are used to evaluate its performance across different fields.

- Strong results are obtained.

### Weaknesses
The paper has several weaknesses:

1- The most important weakness is that the notion of using contrastive loss for semi-supervised leaning is not new. The approach has been used in a variety of semi-supervised literature. Examples include "Class-aware contrastive semi-supervised learning" (already cited in the paper), "CoMatch: Semi-supervised Learning with Contrastive Graph Regularization" (not cited), "Contrastive Regularization for Semi-Supervised Learning" (not cited), and others. In fact, the ablation study shows that almost all the contribution is coming from one of the losses (Ls) and the other isn't doing much. Examples of the general idea being used before is "Supervision Accelerates Pre-training in Contrastive Semi-Supervised Learning of Visual Representations".

2- One of the emphases of the paper seems to be the "end-to-end" aspect of the work. First, this is not unique to this paper, many other semi-supervised methods take a similar approach. Second, why is being end-to-end so important? The claim is that it is more efficient, but there are no studies to back this up. Moreover, while this may impact training, it doesn't seem to impact inference (which is what we really care about).

3- There are key papers that the paper does not compare its results to. Examples include MixMatch, FixMatch, AdaMatch, and others, which have been proposed in other areas (not necessarily for time-series). And then specifically for time-series, there are methods that are missing, e.g., "PARSE: Pairwise Alignment of Representations in Semi-Supervised EEG Learning for Emotion Recognition". All these missing comparisons make it hard to understand how good the method is comparison to others.

### Questions
Please see my comments under weaknesses.

### Soundness
2 fair

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
The paper describes a semi-supervised learning algorithm for time-series classification. Unlike the mainstream two-stage methods, the proposed approach is to train the model in a single training stage with both supervised and unsupervised loss functions.  The proposed solution outperforms 5+ baselines on five different medial domain time series classification datasets.

### Strengths
- The proposed method borrows insights from other semi-supervised and unsupervised learning paper from computer vision domain. Applies the same principles for the time series classification tasks. The proposed method is technically sound and well-described. 
- The experiments are performed 5+ baselines with 5 medical domain time series classification (public) datasets. The paper also creates additional baselines by adapting the baselines for one-stage semi-supervised training. 
- The proposed method outperforms all baselines on all datasets with a significant margin.

### Weaknesses
- The concept of one-stage semi-supervised learning is not novel in the field.  
- The evaluations are limited to medical domain datasets. It is not clear how this method would perform for other domains. 
- The authors suggest that the proposed approach is best suited for optimizing classification performance for target datasets, however, the learned representations would not necessarily transfer for addressing out of domain classification tasks. 
- Robustness. There are missing details such as how the batching is performed, details of how the weights are determined in the loss function etc. It is not clear if the method requires rigorous hyper-parameter tuning or works out of the box with default parameters.
- The paper is missing a pseudo-labeling semi-supervised learning baseline. They are widely studied and commonly practiced in the applied settings due to the simplicity and ease of use.

### Questions
There are number of unknowns about the robustness / generalizability of the proposed method. 
- Q1: How would the proposed method perform beyond medical datasets? 
- Q2: How are the weights in the loss function determined? Do we have to determine new weights every time the labeled data ratio or dataset size changes? How about training schedule and learning rates? Any sensitivity? 
- Q3: How robust is the proposed method against label noise?
- Q4: There are not sufficient details about how the batches are created and what happens to the supervised loss terms when the label is missing for the current training example during training. 
- Q5: How does model capacity impact the training steps? Any known limitations in terms of underfitting or overfitting for certain scale of datasets and/or model sizes?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper addresses two main limitations of existing two-stage contrastive learning methods in time series classification, i.e., the disconnect between unsupervised pre-training and downstream fine-tuning tasks, and the failure to leverage the full potential of ground truth-guided classification loss. 

To tackle these problems, the paper introduces SLOTS, an end-to-end semi-supervised learning model for time series classification that effectively utilizes both labeled and unlabeled data. SLOTS simultaneously calculates unsupervised and supervised contrastive losses along with classification loss, optimizing both the encoder and classifier in a single process. 

The authors show that their approach simplifies the learning framework besides improving the classification performance.

### Strengths
The paper is clear and the flow of writing is good.

### Weaknesses
I have major concerns regarding the methodology, summarized as follows.

- The novelty of this work is quite limited. The paper just adds unsupervised and supervised contrastive losses along with classification loss in an end-to-end way. Nothing is different about the methodology, the loss function, or the augmentations.
- Table 3 shows that the unsupervised contrastive loss is ineffective. Its removal has barely affected the performance. This means that supervised losses (as expected) are the most important ones, which deteriorates the significance of the end-to-end training that mainly targeted including the unsupervised contrastive loss.
- The authors have completely ignored semi-supervised time series representation learning methods in the literature review and did not use any of them as baselines. Despite having the “++” variants of self-supervised methods, this is just an added cross-entropy loss, and might not be the best way to get a good performance. Other methods proposed for semi-supervised time series can have better approaches to handling the data in a way that improves the performance, and therefore, should be included. Examples are:
[1] "Self-supervised learning for semi-supervised time series classification."*PAKDD*, 2020.
[2] "Self-supervised contrastive representation learning for semi-supervised time-series classification." *IEEE TPAMI* 2023.
[3]"Deep Semi-supervised Learning for Time-Series Classification." *Deep Learning Applications*, 2022.
[4] "Selfmatch: Robust semisupervised time-series classification with self-distillation", International Journal of Intelligent Systems 2022.
[5] "Semi-supervised time series classification by temporal relation prediction", ICASSP 2021.
- The hybrid loss is a weighted sum of the three losses. How did you calculate (or assign) these weights?
- The performance gap in Table 1 is significantly high and not well-justified, especially with the smaller fractions of data. Also, I find this an issue with some of the results. Specifically, I noticed that SLOTS is basically the same as SimCLR++, but with a supervised contrastive loss (Ls). However, The accuracy of SimCLR++ in Table 1 is 0.5862, while SLOTS (w/o Ls) in Table 3 is 0.6103.
- In the experiments, the settings of the “two-stage SLOTS” are not clear. Do you use the supervised contrastive loss in fine-tuning or not?
- In the ablation study, I expected to see the performance without both contrastive losses, i.e., only cross-entropy loss, to see how the addition of contrastive learning was useful.
- Since you attempt to achieve a fair comparison in your experiments, I noticed that most of the baselines are using 1D convolutions, while you use a 2D convolution. This can a minor issue, but it can affect the comparison.
- What does “Full model” mean in Table 3 in SLOTS (two-stage)?

### Questions
I expect the authors to answer the above weakness points.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new end-to-end semi-supervised method for contrastive models where the model is fed both unlabeled and labeled data with 3 different losses that take into account where its sample comes from and optimize both unsupervised, supervised, and classification objectives.

### Strengths
- Strong results in a wide array of tasks
- Intuitive explanation of the method

### Weaknesses
- There are no semi-supervised baselines

### Questions
- Can the authors comment on the lack of semi-supervised baselines? All baselines seem to be contrastive models repurposed to perform semi-supervised learning within the proposed framework.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
