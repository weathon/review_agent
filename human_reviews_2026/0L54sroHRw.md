# Diminishing Non-important Gradients for Training Dynamic Early-Exiting Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Early-exiting is an effective mechanism to improve computation efficiency. By adding classifiers to intermediate layers of deep learning networks, early exiting networks can terminate the inference early for easy samples, thus reducing the average inference time. Gradient conflicts between different classifiers are a key challenge in training early-exiting networks. However, current state-of-the-art methods focus solely on the trade-off between gradients, without evaluating whether these gradients are actually necessary. To mitigate this issue, we propose a novel adaptive damping training strategy that adaptively diminishes non-important gradients during the training process based on data samples and classifiers. By adding a damping neuron to the last fully connected layer of each classifier and using our proposed damping loss, our approach effectively reduces gradients that are unlikely to be beneficial. Moreover, we propose power-sqrt loss to concentrate the gradients of damping neurons on classifiers that exhibit relatively better training performance. Experiments on CIFAR and ImageNet demonstrate our proposed method gains significant accuracy improvement for all classifiers with negligible computation increases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission studies the training of multi-exit dynamic networks. It reformulates a K-class classification problem to a (K+1)-class one by adding a separate neuron at each linear classifier. It is claimed that this additional class can prevent the useless/harmful gradients for those samples that achieve a high softmax score. Experiments with a classic multi-exit model, MSDNet, on CIFAR and ImageNet validate that the proposed method improves the performance of multi-exit models.

### Strengths
1. The motivation is reasonable and clear;

2. Experiments in section 3.2 (Fig. 3) successfully support the authors' claim that the gradients of those samples with high softmax scores might be useless or harmful.

### Weaknesses
1. The logic of the proposed method is confusing and not explained clearly.
    - I understand that the second item (damping loss) in Eq. (2) can dominate when a sample achieves a high softmax score (the first item is low). However, until Eq. (2) I can get that the intrinsic operation is setting the additional class also as a "ground-truth" label. This intrinsic logic is not clearly stated in the introduction, Figs. 1 & 2.
    - The relationship between damping loss and the proposed power-sqrt loss is confusing. Eq. (3) is the extension of Eq. (2). But I did not fully understand the motivation and essence of Eq. (4). What does the following gradient equation imply? 
    - In summary, the underlined logics of the authors' proposed losses and equations are not stated with clear words. 

2. I also have some concerns about the experiments.
    - In Tab. 2, The MSDNet baseline's performance is different with the reported results in WPN [1], while the performance of WPN [1] is the same with the reported results in [1]. This raises some concerns about fair comparison.
    - The RANet baseline's performance (Tab. 6) also differs with the reported on in [1] (Tab. 2).
    - The two curves of MSDNet and Meta (WPN?) are close in Figure 4. This is also not aligned with the results in [1] (Fig. 7). 
    - I'm also curious about the effectiveness on latest architectures, such as Dyn-Perceiver [2].

3. An important baseline is missed [3]. It also aims at the calibration of gradients in multi-exit models.

[1] Learning to Weight Samples for Dynamic Early-exiting Networks. ECCV, 2022

[2] Dynamic Perceiver for Efficient Visual Recognition. ICCV, 2023

[3] proved techniques for training adaptive deep network. ICCV, 2021.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
To improve the inference efficiency, early-exiting networks have been proposed, which attach multiple intermediate classifiers and terminate the inference once a intermediate classifier meets a predefined confidence threshold. However, in current early-exiting networks training process, unnecessary gradients may be used to update the model parameters, leading to performance degradation. To resolve this issue, this work introduces a damping neuron into the layer before the classifier softmax, which is used to dynamically reduces unnecessary gradients during training. In addition, a new power-sqrt loss is proposed to improve the efficiency of the algorithm. Empirical results demonstrate that the proposed method outperforms other tested baselines on both CIFAR and ImageNet datasets.

### Strengths
1. This paper has considered an important problem of early-exiting networks. The ideas of damping neuron and damping loss are new. The proposed architecture can be achieved by simply adding a neuron for each classifier, leading to an easy and efficient implementation.

2. The empirical results are good. The proposed method outperforms other tested baselines in most cases. Besides, this work has conducted a comprehensive ablation study to show the effectiveness of each new design.

### Weaknesses
1. The introduction and related work sections of this paper are poorly organized, which makes the overall motivation of the study unclear. Firstly, I don't understand why the paper keeps highlighting the "gradient conflict" issue which seems not related to this work. More importantly, it seems that the original motivation of the proposed method is that "unimportant" gradients may cause the early-exiting model to overfit during training. However, overfitting has been a well-explored issue, and numerous methods have been proposed to address it. This paper does not provide any discussion or comparison of the existing approaches. Also, in the experiments, the proposed method is demonstrated to outperform other tested baselines. However, is the observed performance improvement truly a result of mitigating overfitting? If yes, a comparison between the training loss and validation loss should be shown to verify this. Otherwise, I recommend the authors to reconsider the motivation of the proposed method.

2. The writing of this paper needs to be improved. There are many confusing or inaccurate claims that make the paper hard to follow. For example,

* Throughout this paper, the "unnecessary gradients" are not clearly defined. Does it refer to the gradients that hurt the model performance or the gradients that have a relatively small impact on the model performance? Due to the lack of a formal definition, the correctness and rationality of the theoretical analyses are hard to assess.

* In line 067, it's not clear why a higher gradient of the damping neuron can effectively prevent unnecessary gradients.

* In line 199, it's confusing why "maintaining a higher loss provides additional capacity for other classifiers". The results in Figure 3 do not seem to demonstrate this observation.

* The cross-entropy loss in (2) is wrong.

* The relationship between Proposition 1 and the claims above is not clear. Also, the definition of $g$ is confusing. What does the "gradient from our damping loss" refer to? 

* All the theoretical conclusions and the propositions 1&2 are described based on the loss function (2). However, solely using the damping trick designed in (2) cannot achieve desirable results (e.g., as shown in Table 1). The properties of the better-performed loss functions in (4) and line 347 are not comprehensively discussed.

### Questions
1. Can the results in propositions 1&2 be generalized to the dynamic training scenario in Section 3.5?

2. Can the proposed method be generalized to the more popular LLMs or other transformer-based structures?

### Soundness
2

### Presentation
1

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
This manuscript proposes a novel adaptive damping training strategy to address unnecessary gradients in dynamic early-exiting networks. Unlike existing state-of-the-art methods that only balance gradients without evaluating their necessity, the proposed approach adds a damping neuron to the last fully connected layer of each classifier and uses a damping loss to reduce non-beneficial gradients, while introducing a power-sqrt loss to concentrate damping neuron gradients on better-performing classifiers. Extensive experiments on CIFAR-10/100 and ImageNet datasets using MSDNet and RANet backbones show significant accuracy improvements across all classifiers with negligible computational increases, outperforming existing methods.

### Strengths
1. this paper focus a important problem —— improving the performance of the promising early exiting dynamic networks.
2. the experiments conducted on both cifar and imagenet, which are competitive benckmarks.
3. the sampling wise dynamic training is interesting, which eliminate the meta-learning scheme in previous works.

### Weaknesses
1.  logic of the methods unclear: the core idea of this paper is add an extra neuro in the final layer. However, is the neuro a activated value, or an extra channel in the Linear layer is unclean.
2. although the overall method look new to me, how the finding in sec 3.2 motivate the authors to design the following method is unclear. 
3. the bugeted inference results are missing, make the reviewers can only see the results on each exit.

### Questions
1. how the proposed method reflect the finding in Sec 3.2 is unclear. In Sec 3.2, it seems the loss of over 95% confident samples should be dropped. How the extra neuro in Sec 3.3 suppress the over confident neuro's gradient?
2. How the extra neruo is supervised? It should be clarified explictly in the method section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new training strategy for early-exit networks through adaptive gradient damping. The method introduces a damping neuron into each early exit classifier’s fully connected layer and a corresponding damping loss that selectively suppresses gradients deemed non-beneficial based on softmax confidence. If the softmax value is high, the gradient for that classifier is dampened more. Additionally, a power-sqrt loss is designed to distribute damping across early exit classifiers in proportion to their relative performance. Experiments on CIFAR-10, CIFAR-100, and ImageNet using MSDNet and RANet show improvements over baselines and meta-learning approaches such as WPN, with minimal computational overhead.

### Strengths
The idea of adaptively diminishing “non-important” gradients is conceptually interesting and differentiates this work from prior gradient conflict–resolution approaches.

The proposed approach is complementary to existing multi-task and early-exit training strategies (e.g., linear scalarization, meta-learning).

Results across three datasets and two architectures (MSDNet, RANet) show small but consistent accuracy gains.

The method introduces minimal computational cost (a single neuron per early exit classifier).

### Weaknesses
I am not convinced that the main results in Tables 1, 4, and 6 are truly “notable,” “strong,” or “excellent,” as claimed by the authors. The improvements over the baseline are small, and Table 7 shows relatively large standard deviations. This raises the question of whether the differences between the baseline and the proposed method are statistically significant.

The empirical observation that high-softmax-value samples can yield harmful gradients is backed by an experimental result (Fig. 3). However, more details are needed for this experiment, e.g. which dataset? how many classes? is the accuracy in Fig 3 test set accuracy? How about loss?

Reported accuracy gains are modest, especially given the additional loss design complexity. 

The mathematical derivation of the “power-sqrt loss” is somewhat contrived and lacks intuitive grounding or theoretical justification.

It remains unclear how the damping neuron behaves dynamically and why the square-root aggregation is the optimal design choice.

Comparisons are restricted mainly to WPN (2022) and DFS (2024). More recent adaptive early-exit or uncertainty-aware training methods should be included for a fair benchmark. Examples: 
* Ilhan et al. Adaptive deep neural network inference optimization with EEnet. WACV2024
* Bajpati, Hanawal. BEEM: Boosting performance of early exit DNNs using multi-exit classifiers as experts. ICLR2025
* Lin et al. E3: Early exiting with explainable ai for real-time and accurate dnn inference in edge-cloud systems 
* Xu et al. Early exit via class means for efficient supervised and unsupervised learning. IJCNN2022

While several ablations are presented, they do not isolate contributions cleanly. For example, the separate effects of damping loss vs. power-sqrt vs. dynamic training are not statistically validated.

The term "trade-off between gradients" is not clear. The paper would benefit from a clear (preferably mathematical) definition. Related to this, proposed power-sqrt does a global adjustment which could be seen as making a trade-off among gradients. 

Some equations and figures are difficult to follow (e.g., Fig. 2b caption), and notation is occasionally inconsistent or verbose. A numerical toy example with two or three classes would be much easier to understand the intuition behind how dampening works. 

Most papers are cited using the \citet command. they should have used \citep.

There is paragraph on overfitting in Related Work. I am not sure why this body of work is necessary here. 

The text has several grammatical and stylistic issues that obscure key ideas and would benefit from careful editing. Typo at L484: early-existing.

### Questions
- Can you clarify the details about Fig 3? 
- Are improvements in tables 1,4 and 6 statistically significant? 
- How does your accuracy & computational savings compare to those of other early-exit methods that I mention above?

### Soundness
2

### Presentation
2

### Contribution
2
