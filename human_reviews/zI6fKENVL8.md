# Federated Learning with Differential Privacy for End-to-End Speech Recognition

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). 
Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. 
However, we are not aware of prior work on applying DP to FL for ASR.
In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. 
First, we extend the existing research on FL for ASR by exploring different aspects of recent *large end-to-end transformer models*: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. 
With a *practical* number of central aggregations we are able to train **FL models** that are **nearly optimal** even with heterogeneous data, a seed model from another domain, or no pre-trained seed model.
Second, we apply DP to FL for ASR, which is non-trivial since DP noise severely affects model training, especially for large transformer models, due to highly imbalanced gradients in the attention block. 
We counteract the adverse effect of DP noise by reviving per-layer clipping and explaining why its effect is more apparent in our case than in the prior work.
Remarkably, we achieve user-level ($7.2$, $10^{-9}$)-**DP** (resp. ($4.5$, $10^{-9}$)-**DP**) with a 1.3\% (resp. 4.6\%) absolute drop in the word error rate for extrapolation to high (resp. low) population scale for **FL with DP in ASR**.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper trains ASR models with user-level differential privacy in a federated learning setting. The authors use two main techniques to improve the model performance. The first is using a pre-trained model (seed model) as initialization. The second is applying layer-wise clipping when running DP-SGD. The largest dataset in this paper consists of data from ~40K users, which is not enough to derive a meaningful privacy bound. The authors show the privacy guarantee could be acceptable if the number of users is hypothetically scaled to ~6.9M.

### Strengths
This is the first paper studying DP training of ASR models in the FL setting. The findings presented in this study offer valuable insights and underscore the challenges in this area.

### Weaknesses
1. In Table 2, the authors hypothetically scale the number of users to ~6.9M. This is hard to achieve even in a central setting, and will be even harder considering the implementation challenges in the FL setting. Therefore, it is not clear whether the findings have practical meaning.

2. The seed model in Table 2 is only pre-trained on LS-100, whose size is more than 10 times smaller than the fine-tuning dataset. Why use such a small pre-training dataset? Recent work in this area shows that private learning significantly benefits from better pre-trained models, e.g., see [1] and the references therein. The authors may want to explore pre-training the model on increasingly large datasets, e.g., from LS-100 to LS-960, and see whether better pre-trained models help private training of ASR models.

3. He et al., 2023 [2] also use pre-layer clipping, although their focus is NLP tasks. They also observe the difference in gradient norms of different transformer layers. However, they show that per-layer clipping only achieves comparable or slightly better results compared to global clipping. It would be interesting to the community if the authors could explain why per-layer clipping is much more important in the training of ASR models.


References:

[1]: Why Is Public Pretraining Necessary for Private Model Training? https://arxiv.org/abs/2302.09483

[2]: Exploring the Limits of Differentially Private Deep Learning with Group-wise Clipping. https://arxiv.org/abs/2212.01539

### Questions
Please see `Weaknesses`

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Primary contribution is the application of FL to a heterogeneous automatic speech recognition task, secondary is the application of DP to this FL setting.

### Strengths
This work is very well-written, the idea is justified and the motivation is formulated very clearly. There are a multitude of results in the main body and the appendix, so clearly many hours were put into this work (as well as polishing the manuscript, I could not find any obvious typos etc.)

The results are very promising and show a large improvement over the considered baseline.

### Weaknesses
However, my main concern is the lack of new scientific conclusions. It seems to me that this work is an iterative improvement over the existing works. From the way I interpret the conclusions and results, authors took FL (already well-studied, including the ASR context) and DP (also relatively well-studied, including in conjunction with FL), merged them together and showed that it can work well on some public datasets.

As large as this increase in performance is (and more on that later), I do not see any methodological novelty here. I would like authors to point me to some concrete steps they undertook compared to prior works in the field, but so far I was not able to find any.

Therefore, while I really enjoyed reading the paper, it is really unclear to me how this work differs from a (very thorough) benchmark on public data using FL+DP?

### Questions
It is not clear to me how user-level DP is defined in this work. What is a user with respect to data (e.g. is user - collection of all records with the same ‘author’ or something entirely different?) In fact I could not fully grasp how the number of users go from 50k users to 1M for N?
Additionally, how do you split the data such that it is realistic i.e. so that a user does not have data which is entirely random/disjoint?

Cross-device FL: is it sync/async? What are the implications? 

Data splitting: is there an overlap for LS-960? Since the datasets are combined, how do you ensure that the clients are not accounted from twice from privacy budget perspective
How do you ensure a representative data distribution across train-val-test? It was not covered in the dataset description, I am uncertain if your client distribution at test is necessarily similar to the one at test time

Comment on norm layers: these are indeed harder in FL, but I did not find any discussion on this (maybe worth adding how the choice of norm layers affects FL performance in the appendix?)

Seed models improve results: this is expected, the domain shift result is, however, interesting, but also not unusual (if by seed I have correctly interpreted pre-trained models, often on different datasets of the same modality).

The choice of delta-epsilon seems arbitrary: in general i would expect delta to be 1/dataset size, which is not the case for any of the settings, how does this affect accounting? 
How exactly is the final epsilon computed (especially considering the per-layer clipping)? 

Increasing epsilon is impractical: I would argue that epsilon of 7 is already barely practical given that DP is a multiplicative guarantee.
difficulty of training reasonable models with DP because the impact of noise scales with the model size - I would like the authors to clarify this statement: to me it seems that there is a correlation between some larger models being more prone to performance degradation under DP, but this is correlated with their design rather than their size (e.g. inception net vs resnet)?


Overall, I think there is a lot of work in this manuscript, but I would like the authors to clarify my concerns regarding the scientific contributions as well as some questions on the data distribution/user definition. For now I am really on the fence about recommending acceptance.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors focus on (i) providing an empirical analysis of FL performance on E2E ASR using a large model trained with the Connectionist Temporal Classification (CTC) loss; (ii) analyzing the impact of key FL parameters, data heterogeneity, and seed models on FL performance in ASR; (iii) formulating a practical benchmark and establishing the first baselines for FL with DP for ASR; and (iv) reviving the per-layer clipping for efficient training of large transformer models even in the presence of DP noise.

### Strengths
The quality of this paper is good, but the originality, clarity and significance indeed need to be improved. 
1.    Originality. This paper proposes to apply DP-federated learning to the ASR task, which has not been done before. However, DP for federated learning has already been done long ago, and this work only tests the algorithm on the data instead of improving the frame. Maybe there is a small originality but it’s not enough for ICLR. 
2.     Quality. This article conducted a substantial number of experiments to investigate the impact of seed models, data heterogeneity, and differential privacy on the performance of federated learning. And the content and structure of the paper are relatively complete.
3.	Clarify. The content and structure of the article are relatively comprehensive. However, the experiment logic may not be very clear. For example, from the algorithm pseudocode, it appears that this article only combines DP (Differential Privacy) and FedAvg (Federated Averaging). Yet, later in the text, there is an experimental analysis of Scaffold and FedProx, showing that FedProx yields the best results. So, what is the significance of exploring data heterogeneity? The article could benefit from providing a more explicit and coherent explanation of the rationale for the experimental choices and their implications for understanding data heterogeneity.
4.	Significance. Well this work is the first to combine federated learning with differential privacy in the ASR (Automatic Speech Recognition) field, which can be considered pioneering. The choice of this application direction is valid, but it appears that the level of innovation is relatively limited, which might diminish its overall importance.

### Weaknesses
The level of innovation in this paper is insufficient, not absent but lacking. The combination of federated learning and differential privacy is something that has been explored before, as mentioned in your paper. I suggest further research to discover if there is a federated learning algorithm specifically suited for ASR. ICLR is a highly prestigious conference, and you need to enhance the innovation of your paper in terms of algorithms.
The paper lacks innovation in terms of algorithms.
The logical relationships between experiments are not clear, and there is a weak connection between the models involved in the experiments and the models proposed in paper.
The epsilon in the experiment is too large, around 10^8.

### Questions
It seems that all experimental results in this paper have not been averaged over multiple runs, which introduces some randomness. 
Table 2 has many ‘-‘. Why not delete the lines that are intractable?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
