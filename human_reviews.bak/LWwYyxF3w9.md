# Training-Free Generalization on Heterogeneous Tabular Data via Meta-Representation

- Decision: Reject
- Scores: 5, 1, 3, 6

## Abstract
Tabular data is prevalent across various machine learning domains. Yet, the inherent heterogeneities in attribute and class spaces across different tabular datasets hinder the effective sharing of knowledge, limiting a tabular model to benefit from other datasets.
In this paper, we propose Tabular data Pre-Training via Meta-representation (TabPTM), which allows one tabular model pre-training on a set of heterogeneous datasets. Then, this pre-trained model can be directly applied to unseen datasets that have diverse attributes and classes *without additional training*. Specifically, TabPTM represents an instance through its distance to a fixed number of prototypes, thereby standardizing heterogeneous tabular datasets. A deep neural network is then trained to associate these meta-representations with dataset-specific classification confidences, endowing TabPTM with the ability of *training-free generalization*. Experiments validate that TabPTM achieves promising performance in new datasets, even under few-shot scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel approach to enable training-free generalization for tabular datasets.

The core idea is something like:

For any given dataset, the input label data (x, y) is restructured into a new format: (distance to prototypes of class c, likelihood of the label of class c). This uniform data representation allows different datasets to be organized in a consistent manner. Thus, a model trained on this standardized format can effectively generalize across various tabular datasets.

On unseen datasets, the proposed model achieves superior performances and saves training time.

### Strengths
- It is exciting to witness a tabular learning model deliver remarkable performance without the necessity for fine-tuning.

- On the whole, the model is sound and the meta-representation extraction is novel.

### Weaknesses
The dataset used in this study is somewhat limited. Although I have confidence in the model's ability to generalize effectively to new datasets by representing data points in terms of their similarity to prototypes, there are concerns about its adaptability to other datasets. The reason is straightforward: the issue of feature heterogeneity remains partly unresolved, as there may not be a latent state that aligns to all decision boundaries with similarity measure. The top-K operation on prototypes helps to mitigate this problem to some extent but doesn't completely address it. For example, any transformations applied to the features within the tables, like replacing a feature 'x' with '1/x' (such transformation is reversible, and the information has not been altered or lost), might make the hidden states fail. Therefore, I suspect that this model might only exhibit robustness on certain datasets, and it would be beneficial for the authors to clarify this point and clarify about when the model is effective and when it is not. Otherwise, this paper may potentially lead to misinterpretations in subsequent research.

It might be also advisable to consider using benchmark datasets like those in Taptap (https://arxiv.org/pdf/2305.09696.pdf) or Grinsztajn's work (https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Supplemental-Datasets_and_Benchmarks.pdf) to enhance the credibility of your finding. Perhaps incorporating the average rank and rank standard deviation (like XTab) can provide deeper insights into this research.

### Questions
See equations (3) and (4), is each instance in $D_{y=c}$ considered a different prototype?

There exist multiple versions of DANet, each differing in the number of layers (e.g., DANet-8, DANet-27). You should specify the layer configuration in this paper.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the difficult problem of multi-datasets pre-training/fine-tuning/few-shots-learning where the main challenge is the lack of a common and coherent attributes representation between the different datasets.
In order to solve this problem, in a classification setting, the authors propose :

1- to extract a representative instance x_c of each class of each dataset

2- to represent each instance x of a dataset by the increasing list of distances to the K nearest class-representative instances of this dataset

3- to use this kind of "spectral K-nn" representation as a universal feature replacement

Based on this class-aware representation, they train a generic model to predict the class for any dataset.
The experiments then show that this approach is able to predict correctly the class "without further training" on a fresh dataset.

This paper could be entitled "using k-nearest class-representative neighbours to encode multi-table data"

### Strengths
- The paper is clear
- According to the experiments the method seems to work relatively well

### Weaknesses
I recommand rejection because :

- The "related work section" does not cover the huge literature on k-nearest neighbors and their multiple metric-based variants. It would require at least half a page to position this work amongst these methods
- The authors claim that their algorithm is "training-free". This may appear as a philosophical question, but to my opinion selecting class-representative items in a dataset is a -- strong and costly -- form of training, just as selecting the support vectors for a SVM is a form a training.
- To my opinion this proposal is merely shifting the "not-that-hard" classification problem to the "not-that-easy" metric-learning and representative-choice problems.

### Questions
If we are to tain a k-nn or a metric-based classification algorithm for each new dataset, wouldn't be the class-probability outputs of more robust classifiers like XGBoost or SVM be better and cheaper "universal representations" ?

### Soundness
1 poor

### Presentation
2 fair

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
The paper introduces tabular data pre-training through meta-representation, enabling pre-training on a collection of heterogeneous tabular datasets. This model can be directly applied to previously unseen datasets with diverse attributes and classes without the need for additional training. Specifically, the model computes the distance between a record for which we want to have a representation and the other records within the same class. Then the model selects the K smallest distance values, where the resulting vector of length K forms the meta-representation $\phi_c(x_i)$. For classification tasks, one can train a mapping between the representation vectors and the classification scores using a combination of MLP and Transformer. The authors demonstrate the effectiveness of the proposed model using 22 real-world datasets.

### Strengths
The proposed approach does not requires additional training for unseen datasets.

### Weaknesses
1. The main idea of the approach is a distance-based measurement between data records. However, it may seem somewhat trivial because the Lp norm-based metric (it appears that the authors used L1 or L2 norm) forms an Lp space, and this may be insufficient to represent tabular data, which exhibits complex distributions. 


2. Two heterogeneous tables can have different label distributions, even if feature columns have similar distributions between the two tables. Due to this inconsistent characteristic, fitting one table's distribution to that of another table may not work well.


3. I believe you should clarify the claim, whether the proposed model is for extracting representations or classifying data labels. If the model includes both calculating the representation $\phi$ and training $\textbf{T}_{\theta}$, then the model should be a classifier, not just a representation method. This is because only with the proposed representation $\phi_c$, one cannot directly perform other tasks.


4. The experimental results are insufficient to demonstrate the superiority of the proposed method.

### Questions
As I understand, all the classes in all datasets are learnt as they are each individual class. Then how do you classify unseen data where for which classes are also unknown?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to classify tabular data by learning a shared model above a meta-representation. Samples are represented as distances to prototypes (K the closest samples of particular classes in terms of weighted p-norm distance). The proposed approach TabPTM displays training free generalization to unseen dataset in both few shot and full size regimes and performs comparably to a set of classical and deep-learning baselines.

### Strengths
- The paper is well written
- The goal of learning a shared model for a set of heterogeneous tabular problems is hard and intriguing
- The proposed approach is simple and seems to outperform simple baselines and perform on par with more advanced DL methods, in a training free transfer setting

### Weaknesses
- Regression tasks which are plentiful in practice in tabular problems are not supported by the method.
- Description of the tuning and evaluation protocols are not sufficient for reproducibility. Please provide a more detailed info on how you tuned and evaluated the baselines.
- Some important baselines are missing. Proposed method uses augmentations applicable to other models, but MLP, Transformers and DANets are evaluated without augmentations (which might help in both low-shot and small data regimes)
- TabPFNs do support datasets with around 10k samples (and TabPFN performance was shown to improve with more samples in the original paper). I believe TabPFN should be added to the comparison on larger datasets.
- Number of datasets for which the model is evaluated is rather small by today's standards in tabular DL, especially given that the method generalizes without training. It would be great if you could also add results on the benchmark from `[1]`

**References**
- `[1]` Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. "Why do tree-based models still outperform deep learning on typical tabular data?." Advances in Neural Information Processing Systems 35 (2022): 507-520.

### Questions
- What are the problem with supporting regression with a similar approach?
- Could you describe the tuning and evaluation protocols for the baselines in more detail?

Minor remarks:
- ",why the former one" -- while? (Meta-representation in the few-shot scenario subsection)

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
