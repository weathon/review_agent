# Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Out-of-distribution generalization in neural networks is often hampered by spurious correlations. A common strategy is to mitigate this by removing spurious concepts from the neural network representation of the data. Existing concept-removal methods tend to be overzealous by inadvertently eliminating features associated with the main task of the model, thereby harming model performance. We propose an iterative algorithm that separates spurious from main-task concepts by jointly identifying two low-dimensional orthogonal subspaces in the neural network representation. We evaluate the algorithm on benchmark datasets for computer vision (Waterbirds, CelebA) and natural language processing (MultiNLI), and show that it outperforms existing concept removal methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper focus out-of-distribution generalization in neural networks, and specifically spurious correlations. The methodology considered in the paper is to remove spurious concepts from the neural network's representation of data. To mitigate the weakness of the pervious methods along this line, which is the removal of important features, the authors propose an algorithm that identifies orthogonal subspaces in the neural network representation, separating spurious concepts from the main task. Experiments are done on a toy dataset as well as three benchmark data commonly used in research of spurious correlations.

### Strengths
(S1) The targeted problem of spurious correlation and OOD generalization is important.

(S2) The authors dedicated detailed discussions on their proposed method both in main paper and appendix.

### Weaknesses
(W1) The organization and clarity of this paper needs improvement to enhance its readability. For example, section 2 seems to be a discussion on the problem setting, but it's not clear what exactly is the data distribution model that the paper focus on upon reading the section. The assumptions are not properly and formally stated. A more structured discussion, with clear definitions and assumptions, would make it easier to understand the focus of this paper.

(W2) I would greatly appreciate more discussions on the experiment results. 
1. It seems that ERM outperforms many of the baselines in both the toy data and real data. Given that the baselines are intended to enhance the model's performance on the worst-group accuracy, it would be beneficial if the authors could shed light on why ERM seems to have a superior performance than the presented baselines. 
2. The selection of datasets—Waterbirds, CelebA, and MultiNLI—are frequently used to study spurious correlations. However, there exists a significant body of research [1-4] that specifically addresses these datasets and has demonstrated success in improving the worst-group accuracy. I understand that they are of different approach than this paper, but a proper discussion is needed on this line of methods and why they are not included in comparison. 
3. There are some methods in the previous line of work that are mentioned in Appendix E, but I'm confused which method exactly is implemented as the baseline. By group weighted (GW) ERM, the authors cited papers on last layer retraining (DFR) as well as paper discussing group/class-balanced baselines that are not related to last layer retraining. These methods share similar concepts but have different methodologies and performances. I would appreciate a clearer explanation/citation of the exact compared baseline method.
4. In appendix E, it seems that the proposed method is not doing better than the baseline on the real datasets, but significantly outperforms it on the toy data. Is there a reasoning on why the toy data distribution specifically suits the proposed method, and would we be able to find similar distributions in real data?
5. The considered datasets typically have a very large value of $\rho$. For example, Waterbirds have $\rho=0.95$. The experiments of this paper stops at $\rho=0.9$, which does not seem like a conventional setting. It would be better to show improvement on the original dataset as well. 

[1] "Just train twice: Improving group robustness without training group information." International Conference on Machine Learning. PMLR, 2021.

[2] "Environment inference for invariant learning." International Conference on Machine Learning. PMLR, 2021.

[3] "Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations." International Conference on Machine Learning. PMLR, 2022.

[4] "Robust Learning with Progressive Data Expansion Against Spurious Correlation." Advances in neural information processing systems. 2023.

**Edit**

Thank the authors for the rebuttal. While some of my concerns are addressed, I still hold concerns toward the experiment setting of this paper. Specifically, the ratio of spurious correlation is not standard across all datasets, not just a problem with Waterbirds. When considering spurious correlations, people (and many previous baselines) mostly focus on the setting where correlation is strong and therefore ERM fails significantly. The benchmark datasets therefore all hold a very large value of $\rho$, for example 0.97 with CelebA. These full datasets are not considered in this paper. Meanwhile, it is observable that JSE has a notable decrease in performance on the full dataset of Waterbird, leaving a gap between the baseline GroupDRO. Therefore, I maintain my score.

### Questions
See in weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies OOD generalization of neural network by removing spurious concepts with the proposed joint subspace estimation. The proposed method tries to separate spurious and main-task concepts in the embedding space, by jointly learning two low-dimensional orthogonal subspaces. It is an interesting work.

### Strengths
1.The motivation is clear. Separating and removing spurious from main-task concepts can prevent the model from using the spurious concept for main-task classification.
2.The organization and writing are well. 
3.Extensive experiments are conducted to demonstrate its effectiveness.

### Weaknesses
1. unclear description about the orthogonality assumption. How to guarantee this assumption at any scene? Why linear subspaces are orthogonal? Why linear?
2.The novelty of the proposed jointly subspace estimation is limited and unclear.
3.Why logistic regression can separate spurious from main-task concept?

### Questions
see weaknesses

### Soundness
2 fair

### Presentation
2 fair

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
This paper addressed an out-of-distribution generalization problem under learning from a training dataset with spurious concepts by introducing Joint Subspace Estimation (JSE). Specifically, the authors assume that the main-task and spurious subspaces are orthogonal in the embedding space, and they proposed an algorithm for estimating these two subspaces simultaneously. In experiments, the authors showed that JSE outperforms the other concept removal methods (INLP, RLACE, and ADV) on the benchmark datasets. (modified Waterbirds, CelebA, and MultiNLI)

### Strengths
- The paper is well-written and easy to follow. 
- The motivation for joint consideration of main and spurious subspace is sound and well reflected in the algorithm.
- The proposed JSE outperforms the concept removal baseline methods on the benchmark datasets.

### Weaknesses
- One of my concerns is regarding the experiment settings and baselines. The benchmark datasets used in this paper (Waterbirds, CelebA, and MultiNLI) are also used to evaluate debiased learning algorithms (GroupDRO, DFR, JTT, etc.). Since the settings are the same, JSE should be compared with these debiased learning methods. Alternatively, another option can be made by designing and showing experiment settings that differentiate concept removal methods from debiased learning algorithms.  
- The suggested algorithm contains double for loops, which looks costly. 
- The author leveraged PCA to reduce the computational cost, but it would bring out the information loss. 
- Assuming the known $y_{sp}$ is not practical. Recently published debiased learning algorithms do not require spurious labels.

### Questions
- Is it possible to train and classify another waterbird dataset in which waterbirds are on the sand background, and landbirds are on the forest background (without $y_{sp}$ for both training and validation dataset) if we leverage the subspace information from the original waterbirds dataset? 
- Does the orthogonality assumption always hold for every dataset and trained model? 
- Could you compare the computational cost of JSE with INLP?
- Why $V_{sp}^\perp$ is used to train a last layer instead of $V_{mt}$? Is there an ablation study?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a method for removing spurious correlations in the latent representation by estimating the two orthogonal subspaces --- one associated with the spurious concept and the other with the main-task concept. The proposed method, Joint Subspace Estimation, use statistical test to identify directions in the embedding space associated with the shortcut and main task. The method is evaluated on the Waterbird, CelebA and MultiNLI dataset.

### Strengths
* The paper tries to address an important problem in ML --- detecting and mitigation spurious correlations. The paper is well-written and easy to follow. 

* The idea of estimating orthogonal space is technically sound and novel.

* Results on the CelebA and Waterbird dataset shows that the method is able to disentangle spurious concept from the main concept. Visualization in ** Fig 6. ** confirms and validates this.

* The authors also evaluate an NLP dataset to demonstrate the method can work across different domains.

### Weaknesses
* The method depends on the availability of group labels (i.e., main task and spurious concept label), which is usually unavailable during training time. 

* The method assumes the pixel corresponding to the main concept and the spurious concept doesn't overlap. This may not always hold true --- for e.g., if the main concept is the shape and the spurious concept is colour, the pixels can overlap. 

* Definition of spurious concept in ** eqn 1** is not correct. 
> label $y_mt$ and the spurious features $x_sp$ are independent

They are independent but correlated in the training data (spurious **correlations**)

> while the conditional and marginal distributions are same

I think this is incorrect. If both the conditional and marginal distributions are the same, the joint distribution will be the same too.

* Another major is the use of pre-trained Resnet50. Since it is trained on a large ImageNet data, it can extract features related to both main and spurious concepts. 

* The method is benchmarked against enough baselines. Baselines should also include methods that do not use linear subspace projections/estimations, such as [1,2]. 




[1] Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations
[2] Just Train Twice: Improving Group Robustness without Training Group Information

### Questions
* How do you define main concept and spurious concept? For e.g., in CelebA, gender classification is a much harder problem than blond vs. non-blond hair.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
