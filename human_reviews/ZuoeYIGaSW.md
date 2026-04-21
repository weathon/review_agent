# $\texttt{PREMIER-TACO}$ is a Few-Shot Policy Learner: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5

## Abstract
We introduce $\texttt{Premier-TACO}$, a novel multitask feature representation learning methodology aiming to enhance the efficiency of few-shot policy learning in sequential decision-making tasks. $\texttt{Premier-TACO}$ pretrains a general feature representation using a small subset of relevant multitask offline datasets, capturing essential environmental dynamics. This representation can then be fine-tuned to specific tasks with few expert demonstrations.
Building upon the recent temporal action contrastive learning (TACO) objective, which obtains the state of art performance in visual control tasks, $\texttt{Premier-TACO}$ additionally employs a simple yet effective negative example sampling strategy. This key modification ensures computational efficiency and scalability for large-scale multitask offline pretraining.
Experimental results from both Deepmind Control Suite and MetaWorld domains underscore the effectiveness of $\texttt{Premier-TACO}$ for pretraining visual representation, facilitating efficient few-shot imitation learning of unseen tasks.
On the DeepMind Control Suite, $\texttt{Premier-TACO}$ achieves an average improvement of 101\% in comparison to a carefully implemented Learn-from-scratch baseline, and a 24\% improvement compared with the most effective baseline pretraining method. 
Similarly, on MetaWorld, $\texttt{Premier-TACO}$ obtains an average advancement of 74\% against Learn-from-scratch and a 40\% increase in comparison to the best baseline pretraining method.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper aims to learn representations for sequential decision-making tasks. Based on temporal action contrastive learning (TACO), the authors adopt a negative sampling strategy to improve the representation especially in multitask contexts. Employing a shallow ConvNet, the authors benchmark their method on Deepmind Control Suite and MetaWorld.

### Strengths
1. The paper is well-motivated with careful discussions on the challenges and criteria for learning decision-making representations, as well as the shortage of baseline TACO.
2. The introduced method, Premier-TACO, seems easy to implement.
3. The results indicate the effectiveness of the proposed method.

### Weaknesses
1. The introduced Premier-TACO shows incremental contribution over the baseline TACO. Specifically, the only difference is the contrastive loss adopted by TACO and the triplet loss adopted by Premier-TACO. The technique of negative sampling is quite common in the area of metric learning and widely adopted in applications other than robotics, e.g., face recognition. Moreover, the empirical comparison with the baseline TACO is very limited in this paper. The effectiveness of simply adding negative sampling is questionable.

2. The authors repeated several times the infeasibility of adopting visual foundation models (such as those trained on ImageNet or Ego4D) in sequential decision-making tasks. However, it is ungrounded. The authors should evaluate these models (e.g., CLIP, DINOv2, EgoVLP, etc.) on the same benchmark for comparison.

3. The model adopted in the paper is quite small. The pretraining is only performed on synthetic data and also small-scale. The practicality of the method needs further study.

### Questions
Why TACO in Fig.9 achieves the same results with different batch sizes?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes Premier-TACO, a few-shot policy learner for sequential decision-making tasks. This method is build upon the existing work, i.e., temporal action contrastive learning (TACO) objective, and employs a negative example sampling strategy, which is beneficial for large-scale multitask offline pretraining. Experiments on Deepmind Control Suite and MetaWorld show superior performance.

### Strengths
1. The performance on both seen and unseen tasks are superior than other methods.
2. The proposed one negative sample selection is reasonable since it is harder when selecting the negative sample among a window slot than selecting from a batch as in TACO.

### Weaknesses
1. It would be better to add an ablation of using the negative sample strategy in TACO on sequential decision-making tasks. It is important to show the effectiveness of the proposed negative sample selection strategy.
2. Premier-TACO uses additional negative samples selected from a temporal window. Compared with TACO, is the batch size doubled? If it is true, what is the result when decreasing the batch size of Premier-TACO to 1/2$N$ compared with TACO with $N$.
3. Does the selection number influence the performance? How about select more than one samples as negatives? 
4. Similarly, the negative sample is selected randomly from $W$ window. How about select the hardest one or easiest one from $W$ window?
5. Is Premier-TACO model-free? If yes, can it be applied on other model structures used in the previous methods, e.g., SPR?

### Questions
Please refer to the Weakness Section.

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes Premier-TACO, a multitask feature representation learning method, aiming to enhance the efficiency of few-shot policy learning in sequential decision-making tasks. Premier-TACO pretrains a general feature representation using s small subset of multitask offline datasets and then fine-tunes the network to specific tasks with a few experts. Additionally, Premier-TACO employs a negative example sampling strategy on contrastive learning objectives. Experimental results show that Premier-TACO can outperform the state-of-the-art on DeepMind Control Suite and MetaWorld.

### Strengths
1.	The paper is well-written and easy to follow.
2.	The proposed method Premier-TACO can simultaneously achieve versatility, efficiency, robustness, and compatibility. 
3.	Empirical results demonstrate that Premier-TACO can achieve SOTA results on several benchmarks.

### Weaknesses
1.	The novelty is somewhat limited. The proposed method is built on the temporal action constrastive learning (TACO) objective [1]. The overall framework is similar to [1]. The authors additionally employ a negative example sampling strategy. But the negative sampling has been widely used in constrastive learning [2][3]. Considering the above factors, I think that the innovation of the method is limited
2.	The detailed experimental comparisons and discussions with TACO are missed. 
3.	In the ablation study, in order to show the effectiveness of the proposed negative example sampling strategy, the authors should compare Premier-TACO with a baseline without using a negative example sampling strategy. The related experimental results should be added.

[1] TACO: Temporal latent action-driven contrastive loss for visual reinforcement learning, NeurIPS 2023.

[2] Robust Contrastive Learning Using Negative Samples with Diminished Semantics, NeurIPS 2021.

[3] Hard Negative Sampling Strategies for Contrastive Representation Learning, arxiv 2022.

### Questions
See Weakness for detail.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
