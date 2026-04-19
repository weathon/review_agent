# Blind Coreset Selection: Efficient Pruning for Unlabeled Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
Deep learning methods rely on massive data, resulting in substantial costs for storage, annotation, and model training.
Coreset selection aims to select a representative subset of the data to train models with lower cost while ideally performing on par with the full data training.
State-of-the-art coreset selection methods use carefully-designed criteria to quantify the importance of each data example using ground truth labels and dataset-specific training, then select examples whose scores lie in a certain range to construct a coreset.
These methods work well in their respective settings, however, they cannot consider candidate data that are initially unlabeled.
This limits the application of these methods, especially so considering that the majority of real-world data are unlabeled.
To that end, this paper explores the problem of coreset selection for unlabeled data.
We first motivate and formalize the problem of unlabeled coreset selection, which reduces annotation requirements to enable greater scale relative to label-based coreset selection.
We then develop an unlabeled coreset selection method, Blind Coreset Selection (BlindCS), that jointly considers overall data coverage on a distribution as well as the relative importance of each example based on redundancy.
Notably, BlindCS does not use any model- or dataset-specific training, which increases coreset generalization and reduces computation relative to training-based coreset selection.
We evaluate BlindCS on four datasets and confirm the advance over several state-of-the-art methods that use labels and training, leading to a strong baseline for future research in unlabeled coreset selection.
Notably, the BlindCS coreset for ImageNet achieves a higher accuracy than previous label-based coresets at a 90\% prune rate, while removing annotation requirements for 1.15 million images.
We will make our code publicly available with the final paper.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper describes a technique for coreset selection from unlabeled data. The proposed method consists of three main steps. First, feature embeddings of the unlabeled data are extracted using different pre-trained models. In the second step, the feature distribution of unlabeled data is triangulated, and dimensionality reduction is applied to lower the feature dimensions. Finally, the third step involves selecting samples by eliminating redundant features after dimensionality reduction. The experimental comparisons are presented on four datasets.

### Strengths
1. The topic of how to solve the coreset selection problem for unlabeled data is both interesting and novel.
2. The presentation is clear and easy to understand.

### Weaknesses
1. In Table 1, it is questionable that BlindCS are not trained on the data. To be clear, BlindCS is not trained on the target dataset, but instead uses already trained pre-trained models (possibly by training on larger datasets), which is another way of using the training data. In addition, you need to emphasize that other methods do not use pre-trained models.
2. The experimental setup is unreasonable. BlindCS introduces the pre-trained models of resnet18 and CLIP, leading to an unfair comparison with other methods. The other coreset selection methods compared do not use pretrained models by default. For example,  assuming that the resnet18 pre-trained model can be used, other methods can be fine-tuned on resnet18 directly using the selected coreset.  However, BlindCS must use additional pre-trained models to extract features (e.g. resnet18, CLIP, etc.). Finally, The results of BlindCS has a great relationship with the pre-trained models, and the method has certain defects.
3. Formula (9) selects the nearest sample to the center of each subspace to form the coreset, so how to ensure the coverage rate of the selected sample under different pruning rates.
4. The ablation experiment is in doubt. There is concern about the choice of the parameter ${m}$, and ablation experiments show that the method is sensitive to the parameter ${m}$. Since BlindCS does not introduce labels, the selection can only be made on different dimensions after dimension reduction, but not on different classes. So how to ensure that the coreset does not have long tail problems.

### Questions
1. See the weakness section. Maybe elaborate more on your opinions about point 1-4.
2. In particular, BlindCS introduces pre-trained models, while previous baselines do not use pre-trained models. If pre-trained models can be used, existing coreset selection methods can still solve the unlabeled data problem, such as K-Center greedy.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a new method for unsupervised core-set selection by taking the coverage and redundancy of the embedding space.

### Strengths
1. The motivation of the paper is valid. Selecting a small subset of data for labeling can reduce costs, while many existing methods make an unrealistic assumption that training data is already labeled.
2. The proposed method is simple and easy to implement, which does not involve model training.
3. Experiments on several datasets show the advantage of the proposed method against previous methods.

### Weaknesses
1. The method proposed in this paper is highly heuristic, making it difficult to verify the optimality of core set selection. For example, the sampling strategy in Equation (3) and the feature random selection strategy in Equation (4) lack clear underlying principles, and their effectiveness has not been thoroughly analyzed in the experiments. Additionally, the principles and advantages of Equation (7) are not discussed. Therefore, the method in this paper employs a series of heuristic strategies to improve the efficiency of sample selection, but it does not provide an in-depth analysis of the superiority of the approach itself, nor does it offer theoretical performance guarantees.

2. Self-supervised learning baselines for core-set selection should be included in the experiments. As the primary goal of the paper is to reduce labeling costs, self-supervised learning offers a fair baseline for comparison since it eliminates the need for labeled data.

3. On the ImageNet dataset, the improvement of BlindCS over Random is minimal. Furthermore, for experiments involving randomly selected samples, the process should be repeated multiple times to report the average values and variance.

4. From Table 5, it is evident that Triangular and Gaussian sampling methods perform comparably. Additional results are needed to demonstrate the advantages and justify the effectiveness of Triangular sampling.

5. Hyperparameter analysis for $\alpha$ and $\beta$ is missing and should be included in the experiments to provide further insights.

6. In Equation (1), the expression $\frac{1 - n}{N}$ should be corrected to $1 - \frac{n}{N}$.

7. The paper uses $s$ to denote both the importance score and random sample, which may lead to confusion.

8.  Section 4 is too brief and can be integrated into Section 3 or Section 5 to improve the paper’s structure and flow.

### Questions
1. It is interesting to see if the proposed method can generalize to more pre-trained models in addition to ResNet18 and CLIP-L/14.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper studies the problem of coreset selection for unlabeled data. The paper develops an unlabeled coreset selection method, Blind Coreset Selection (BlindCS), that jointly considers overall data coverage on a distribution as well as the relative importance of each example based on redundancy. The experiments demonstrate the performance of the proposed method.

### Strengths
1. The idea of the article is reasonable and the presentation of the problem is fine.

2. The paper proposes a new method for solving the problem of coreset selection for unlabeled data, which achieves comparable results to the supervised methods.

### Weaknesses
1. In the proposed method, to obtain the embedding space, already trained models (ResNet18 and CLIP in the paper) are required, and the overhead of training these models should be taken into account to compare with the supervised methods. In addition, the inputs to these models may actually contain more information such as textual information, which may lead to unfair evaluation.

2. The advantage of Triangular distributions may actually be due to a priori, such as modeling of the long tails, which is actually valid for some specific real-world scenarios. Although Triangular distributions are beneficial in covering the long tail of the distribution, actually change the distribution relative to the original distribution, as in the two examples shown in Fig. 2, therefore more detailed analysis and discussion are needed to demonstrate the advantages of such sampling.

3. The construction of a coreset generally depends on the particular downstream tasks. The coreset constructed in the paper is not closely linked to the downstream tasks, and lacks theoretical explanation, which weaken the reliability of the proposed method.

Some typos: line 229: ample->sample. 
Eq. 10: s+min(s)->s-min(s)?

### Questions
In line 169, the prune rate (1-n)/N can be negative, right?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposed a pruning-based unlabeled data learning for the Coreset algorithm.

### Strengths
The proposal is interesting as it shows a new direction for unlabeled data learning for the Coreset algorithm.

### Weaknesses
The major weakness of the paper is that the paper lacks experiments.  The experimental section is very shallow. as only CIFER 10, 100 and imagenet are used. There exist many datasets in the literature that should be used, such as MSCOCO, VOC etc. More experiments on varying segmentation tasks, NLP, retrieval, tasks etc should be conducted to prove the proposal's effectiveness.

A comparison with many related works in the literature is also missing.

### Questions
What is the effectiveness of the algorithm on other task rather than classification.

### Soundness
2

### Presentation
2

### Contribution
2
