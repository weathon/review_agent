# Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning

- Decision: Accept (poster)
- Scores: 6, 3, 3, 8, 6

## Abstract
Methods for carefully selecting or generating a small set of training data to learn from, i.e., data pruning, coreset selection, and dataset distillation, have been shown to be effective in reducing the ever-increasing cost of training neural networks. Behind this success are rigorously designed, yet expensive, strategies for identifying the most informative training examples out of large datasets. In this work, we revisit these methods to understand if the additional computational costs associated with such strategies are justified from the perspective of time-to-accuracy, which has become a critical efficiency measure of deep neural network training over large datasets. Surprisingly, we find that many of the recently proposed methods underperform what we call Repeated Sampling of Random Subsets (RSRS or RS2), a powerful yet overlooked extension of the standard random baseline that learns from repeatedly sampled data throughout training instead of a fixed random subset. We test RS2 against thirty-two state-of-the-art data pruning and distillation methods across four datasets including ImageNet. Our results demonstrate that RS2 significantly reduces time-to-accuracy, particularly in practical regimes where accuracy, but not runtime, is similar to that of training on full dataset. For example, when training ResNet-18 on ImageNet, with 10\% of the dataset each epoch RS2 reaches an accuracy of 66\% versus 69\% when training with the full dataset. The best competing method achieves only 55\% while training 1.6$\times$ slower than RS2. Beyond the above meta-study, we discuss the theoretical properties of RS2 such as its convergence rate and generalization error. Our primary goal is to highlight that future works that aim to minimize total training cost by using subset selection, need to consider 1) the total computation cost (including preparing the subset) and 2) should aim to outperform a simple extension of random sampling (i.e., RS2).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper revisits a rather simple subset selection strategy for efficient deep learning. The authors claim that repeated sampling of random subsets (RS2), that is, only randomly sample subsets at each round, can be a powerful baseline strategy. RS2 is competitive against most of the sampling strategies as well as dataset distillation methods developed previously. Besides, there is no additional computation cost extracting the coreset using RS2, so RS2 reaches the best acceleration under the same sample budget.

### Strengths
1.	Selecting the coreset for efficient deep learning is important for machine learning practices. The paper may be valuable to the community trying to address this problem. To the best of my knowledge, this is the first paper to formally discuss the repeated random sampling strategy.

2.	The paper is clearly written, the authors do a good job in presenting their intuitions, and the analysis is convincing. 

3.	Extensive experiments are conducted to show the effectiveness of RS2.

### Weaknesses
1.	The paper considers only the low data regime (<30% data). RS2 performs well in this regime as it will not reshape the underlying data distribution. Actually, I think whether RS2 can outperform other strategies depends on the subset size and the property of the original dataset itself. In the data-abundance regime,  I believe selecting "harder" samples benefits model training. More discussion on this will greatly strengthen the paper. 

2.	In the theoretical analysis, only RS2 without replacement is considered. I wonder if the result changes for RS2 with replacement.

### Questions
Please see the weakness part above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper revisits the utility of random selection when it comes to speeding up the training. The paper considers two types of random sampling, random sampling with and without replacement. The paper surprisingly shows that these two samplings can easily outperform the popular data subset selection baselines (adaptively or static) when it comes to comparing the time to achieve the same accuracy. Moreover, the authors show that for many training budgets, it outperforms famous baselines, demonstrating the efficacy of uniform random  selection.

### Strengths
The paper conducts experiments across several baselines, including active learning and dataset distillation. In addition, the paper provides an analysis of the convergence of the RS2 algorithm, which I am not sure is how novel is it, in terms of proof technique, but it's a good contribution to have in any data subset selection paper. Lastly, I think robustness and LLM pretraining results are also interesting making the overall comparisons spanning different modalities and scales.

### Weaknesses
- RS2 w/o replacement is the same as training on a reduced number of epochs. For the cases where RS2 achieves the same accuracy in significantly less amount of time, I think the main issue was not tuning the epoch/learning rate hyperparameter of the full dataset baseline. Therefore making the RS2 w/o replacement results less exciting. 

- Can authors provide a plot #unique examples seen throughout training? It could be the case that certain baselines are not exploring the full dataset, due to possibly inadequate hyperparameter search on them. 

- I am very confused about Table 1, where we allow the model to update the subset after every round. I request authors provide a clear description for every baseline for both cases, where the distribution is allowed to change over time. The notion of RC and RS has to be made clear for each of the baselines. 

- Why is RC's performance extremely worse in certain baselines such as GraNd? In general, why is there a strong dip in the performance of all the baselines, when switched to RC? 

- For the baselines that sample set based on submodular function, what does it mean to have a distribution? How are authors defining the distribution over each set of size "k"? If that is the case, how are they sampling? If not, what is the heuristic?

- I think it is not correct to compare active learning baselines to subset selection schemes that look at  - (1) all the dataset, but the reduced number of iterations, and (2) assume labels for all the data points, since AL does not assume labels. 

- Each of these baseline papers has comparisons against random, on the other hand here the authors break the baselines by mere random selection. What is the reason for this discrepancy?  

- Can authors please point to the hyperparameter tuning for each of the baselines? Submodular functions often work well if tuned properly, therefore it is important to see if enough hyperparameter tuning was done to make sure the function is good. 

- For the submodular methods if the corresponding greedy gain was used to sample sets (distribution defined using gains), it should be noted that if the function saturates, then greedy gains do not provide any useful information (yet another reason to provide hyperparameter search grids). 


I am willing to raise scores upon satisfactory responses to my questions.

### Questions
Can authors also add a comparison to more recent versions of CRAIG such as CREST [1]? 

[1] Towards Sustainable Learning: Coresets for Data-efficient Deep Learning (ICML'23)

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper compares approaches to reducing the training time by curating a smaller representative dataset like data pruning, coreset selection, and dataset distillation to a simple random sampling-based approach termed Repeated Sampling of Random Subsets (RS2).  Results show that prior adaptive approaches fail to beat RS2 on four image datasets in both final accuracy and time-to-accuracy when accounting for overhead associated with active selection.  Owing to the properties of random
sampling, RS2 comes with convergence and generalization guarantees.  The authors highlight the importance of evaluating approaches based on time-to-accuracy and the need for more complex approaches to beat a simple baseline like RS2.

### Strengths
- The authors point out important considerations missing in prior work on speeding up training with adaptive dataset subset selection.  First, whether there is a need to restrict data to a fixed subset in the first place if similar accuracy can be achieved by training with a compressed learning rate schedule on fewer epochs.  Second, the importance of including overhead associated with data selection when evaluating training compute efficiency of an approach.

### Weaknesses
- As I understand, RS2 without replacement is effectively the same as training on the full dataset with the learning rate schedule compressed into fewer epochs.  RS2 with replacement is a slight variant to that but still highly resembles standard training with shuffling between epochs just with a condensed training window. This is not discussed anywhere but brings into question the whole exposition of proposing RS2 as a sampling method.  An even simpler baseline is training as usual on the full dataset with the condensed schedule.  I would wager such a baseline would yield similar performance as RS2.
- The paper is light on experimental details for how prior methods are evaluated in particular for case where samples are reselected based on latest model weights.  I am surprised the results in Table 1 right are worse than that for Table 1 left when selecting a new subset with updated model weights can expand the number of samples seen during training.  The poor performance of AL approaches like Entropy and Margin also contradicts the experimental results from Park et al., 2022 where AL outperformance other subset selection methods.

### Questions
- What model weights are used for computing the static subsets of approaches like Entropy, Margin, Least Confidence, etc in Table 1 left?
- How often are importance scores recomputed for adaptive methods in Table 1 right?
- How does random with fixed subset perform on the tasks studied?
- How does training on full dataset with the same LR schedule and training window as that used for RS2 perform?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a novel approach to improve time-to-accuracy of deep learning model training by using a fraction of the full dataset in each training epoch. 

The paper discussed the limitations of two commonly used methods in this domain, (1) Data Pruning: Selecting the most informative examples to train more efficiently. (2) Dataset Distillation: Creating synthetic examples that represent the larger dataset to train quickly.

The paper proposes Repeated Sampling of Random Subsets (RS2), which simplifies the process by randomly selecting different data subsets for each training epoch, promoting broader learning and efficiency. RS2 has been shown to outperform State-of-the-Art methods, achieving near-full dataset accuracy with significantly reduced training time on various datasets, including large scale image benchmarks like ImageNet. 

It is interesting to note that the paper achieves very close performance to models trained on complete datasets with just 10\% of the datapoints for large scale benchmarks like ImageNet.

### Strengths
1. The paper presents a simple but novel approach to achieve significant reductions in time-to-accuracy while training on a fraction of the full dataset per epoch of model training.
2. The paper also presents detailed theoretical properties that support the faster convergence of the model as compared to existing approaches in the domain.
3. The paper demonstrates results on four image datasets including large scale image benchmarks like ImageNet wherein it achieves State-of-the-Art (SoTA) performance (accuracy) with just 10 \% of the data samples in the complete dataset.
4. The paper also demonstrates SoTA performance on auxiliary tasks like data distillation, noisy label classification and pretraining of large language models.

### Weaknesses
1. Although the experimental results are exemplary (primary contributor to my decision), the method RS2 itself is an incremental update over random sampling. The paper must call out the clear difference with SoTA methods (please refer to questions for more details).
2. All experiments demonstrated in the paper adopt canonical benchmarks which are well curated, while lacking experiments on datasets (eg: MedMNIST (Yang et al., 2021), CUBS-2011 (Wah et al., 2011)) with large intra-class variance and class-imbalance wherein data pruning might underperform. 
3. The paper does not show any relation between the theoretical properties of RS2 (convergence rate and bounds on generalization error) and the conducted experiments.

### Questions
1. The subset selection strategy of RS2 without replacement is unclear in section 3. A suggestion would be to replace the textual description in this section with Algorithm 2 in section D of the appendix.
2. The variables $n$ and $N$ are used interchangeably in section 4.
3. The term ‘selection ratio’ and ‘pruning ratio’ has been used interchangeably and should be fixed in the paper.
4. As mentioned in the ‘weaknesses’ of the paper, experiments on real-world class-imbalanced settings ((eg: MedMNIST (Yang et al., 2021), CUBS-2011 (Wah et al., 2011))) would be an effective demonstration of the application of RS2.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work empirically investigated a strong baseline called Repeated Sampling of Random Subsets (RSRS, or RS2), in the context of dataset pruning/distillation. The authors found that this sampling scheme, which has been overlooked by the literature, served as a very strong baseline in terms of many metrics, such as end model accuracy and *time-to-accuracy*.

The authors did intensive experiments that compare RS2 with up to 24 existing dataset pruning/distillation methods and observed the superiority of RS2 under all of the above metrics. The authors called for attention from the community on this strong but overlooked baseline.

### Strengths
+ The highlight of an overlooked baseline in the context of dataset pruning/distillation.
+ Intensive experiments over so many baselines. This provides a very good benchmark and starting point for the following works, which I find really appreciable.

### Weaknesses
- The RS2 without replacement is exactly the same as reducing the number of training epochs but with tuned learning rate scheduling. The new term is not helping to make the concept clear but more confusing. This also means that the theoretical analysis in Section 4 did not make actual contributions over previous work.

- In my opinion, a type of dataset pruning methods, which generate static subsets before real training starts, are up to a slightly different point from RS2. While we all know that the more data used for training the better, these methods try to find a coreset that is essential for good generalization. Therefore, it is important that the pruned data are not seen during the training process later (thus static subset). This reduces the storage cost which is not possible with RS2 because RS2 still requires access to the full training set.

- RS2 with replacement has been adopted in a few more works in the context of efficient training like [Ref-1, Ref-2].

------------------

Refs:

[Ref-1] Wang, Yue, et al. "E2-train: Training state-of-the-art cnns with over 80% energy savings." Advances in Neural Information Processing Systems 32 (2019).

[Ref-2] Wu, Yawen, et al. "Enabling on-device cnn training by self-supervised instance filtering and error map pruning." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 39.11 (2020): 3445-3457.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
