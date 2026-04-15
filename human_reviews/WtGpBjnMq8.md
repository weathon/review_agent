# Propensity-driven Uncertainty Learning for Sample Exploration in Source-Free Active Domain Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 5, 3

## Abstract
Source-free active domain adaptation (SFADA) addresses the challenge of adapting a pre-trained model to new domains without access to source data while minimizing the need for target domain annotations. This scenario is particularly relevant in real-world applications where data privacy, storage limitations, or labeling costs are significant concerns. Key challenges in SFADA include selecting the most informative samples from the target domain for labeling, effectively leveraging both labeled and unlabeled target data, and adapting the model without relying on source domain information. Additionally, existing methods often struggle with noisy or outlier samples and may require impractical progressive labeling during training. To effectively select more informative samples without frequently requesting human annotations, we propose the Propensity-driven Uncertainty Learning (ProULearn) framework. ProULearn utilizes a novel homogeneity propensity estimation mechanism combined with correlation index calculation to evaluate feature-level relationships. This approach enables the identification of representative and challenging samples while avoiding noisy outliers. Additionally, we develop a central correlation loss to refine pseudo-labels and create compact class distributions during adaptation. In this way, ProULearn effectively bridges the domain gap and maximizes adaptation performance. The principles of informative sample selection underlying ProULearn have broad implications beyond SFADA, offering benefits across various deep learning tasks where identifying key data points or features is crucial. Extensive experiments on four benchmark datasets demonstrate that ProULearn outperforms state-of-the-art methods in domain adaptation scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses source-free active domain adaptation. The main idea is to select most informative samples based on homogeneity propensity score and correlation index entropy. Then a training procedure is devised to refine model's adaptation with central correlation loss and other losses. Experiments are conducted on four SFADA benchmarks, showing better performances than previous SFDA and Active DA works.

### Strengths
- This paper studies active domain adaptation under the source-free setting, which has its practical meaning considering the increasing concerns of model privacy in domain adaptation. Although there are a handful of ADA works, SFADA is still under-exploited.

- Different from previous ADA works, this work only requests human annotation for once, which is more practical.

- The idea of using homogeneity propensity score to select samples, as far as I know, is novel to this area. HPE is shown to work better than traditional unsupervised clustering methods (Table.4). HPE does not rely on distance calculations or density estimates, thus is robuster.

- Equipped with central correlation loss, the whole framework ProULearn obtains better accuracies on four benchmarks. The improvements are significant.

- Overall, the paper is well-written, with clear figure illustrations and algorithm procedures.

### Weaknesses
- While HPE is interesting, the novelty of correlation index entropy and central correlation loss is relatively small as using entropy to select samples and enforcing clustering structures have been widely validated. The authors have not shown whether correlation index works better than other commonly used distance metrics like L2 distance. 

- The experiments are conducted with only 5% labeling budget. In many previous ADA works, 10% labeling budget is also commonly tested. On some small datesets like Office-31, 5% labeling budget implies a small absolute number of samples, which might not be informative.

- In Table 4, the authors compare HPE with two other clsutering methods, showing the superiority of HPE. It is unclear if ProULearn+other AL criterion will work well enough.

### Questions
- I am curious about the cost of building $g$ separation trees. In order to build one tree, it needs to iterate over the entire dataset at every node. Will it become time-consuming on large dataset like VisDA? What data structure is used to process and store the trees? Have the authors utilized some accelaration strategies in building the trees?

- From the text, the split value is chosen uniformly between minimum/maximum values of the selected feature. Why not using some entropy criterion to decide split value? From my understanding, a random split value can be inefficient, which may lead to deeper trees and require a larger $g$ to reduce noises. Will it work if we simply use medium value as the split value?

- In Eq.(6), I am confused about the function of $h(x_i)$ here as it does not rely on $c$, thus does not change $arg max$ result. In Ln 320, it says 'samples with higher homogeneity scores are given more confidence in the pseudo-labeling process'. Could the authors clarify how it is reflected in Eq.(6)?

- When selecting samples based on $U_i$, how to guarantee the diversity (i.e., avoid selecting duplicated pairs that are very close to each other)?

- In Eq.(5), what is $x_t$? Seems to be the target samples, but a set annotation might be more proper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The key challenge of SFADA is selecting the most informative samples from the target domain for labeling and effectively utilizing target domain data to adjust the model. This paper proposes the ProULearn framework, which is capable of identifying representative and challenging samples (i.e., informative samples) for labeling, and develops a center-related loss to refine the pseudo-labels of the remaining samples. This allows the model to focus on the most informative examples in the target domain and generate accurate pseudo-labels. The core contribution is the combination of homogeneity tendency estimation and correlation index calculation, which allows for the selection of samples in the target domain that have the maximum information and lower model prediction confidence.

### Strengths
Overall, this could represent a significant algorithmic contribution. The paper presents a novel and effective method for selecting informative samples, which, if applicable to real-world scenarios of domain adaptation, could alleviate considerable labeling costs, making it quite significant. Additionally, the experimental results of this paper demonstrate excellent sample clustering and compact class distributions, which effectively aid in generating accurate pseudo-labels.

### Weaknesses
The HPE mechanism requires building a large number of separating trees, and calculating the homogeneity scores and relevance indices for each sample might be overly complex.

### Questions
I would like to know whether the selection of K and g requires extensive tuning across different datasets. HPE is one of the core mechanisms of the paper, and I hope it can be described in more detail, making it easier for readers to understand how to construct the separating trees and why this is useful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the source free active domain adaptation (SFADA) setting. The authors claim that existing methods generally select noisy or outlier samples for annotation, which may influence the prediction of neighbor samples. This paper proposes to adopt homogeneity propensity estimation (HPE) mechanism to assess whether the samples are grouped. Besides, a correlation index that is similar to the Pearson Correlation Coefficient is proposed to evaluate the correlation between two samples. The correlation index is applied to compute entropy and a central correlation loss. Experiments show improvements over existing methods.

### Strengths
1. The paper is well written and easy to follow. 
2. The idea of utilizing trees to assess samples grouping condition is interesting.

### Weaknesses
1. The experiments are insufficient. Table 3 only includes 7 tasks on MiniDomainNet while there are 12 tasks in total. The analysis lacks crucial details about the HPE mechanism, including the actual tree node information and its role in sample selection. Additionally, the impact of varying annotation budgets (e.g., 3%, 10%) is not explored.

2. The novelty is somewhat limited. Except for utilizing HPE, the pseudo labeling process, the information maximization and weighted cross-entropy loss are existing techniques [1]. It seems the major contribution of this paper is only an active selection criterion, rather than an SFADA algorithm.

3. The Correlation index needs clearer explanation. Does “k-th feature of sample x” mean the value of the k-th channel in the extracted feature of sample x? What is d in Eq.2? The notation k are repeatedly used through Eq.2 and 3.

[1] Jian Liang, et al. “Do we really need to access the source data?”

### Questions
The authors mention “computing the correlation matrix between all pairs of samples.” Could this process introduce high computation or storage burden during training?

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
In this paper, the authors present ProULearn, a Propensity-driven Uncertainty Learning framework for source-free active domain adaptation (SFADA). This approach tackles key challenges in SFADA with innovative techniques for sample selection and model adaptation, including a homogeneity propensity estimation mechanism and correlation relationship learning. ProULearn identifies informative samples while reducing the impact of noisy outliers and locates these samples before training begins, eliminating the need for human labels during adaptation. This approach, combined with a training strategy that results in more compact and discriminative class distributions in the target domain, enhances the practicality of domain adaptation.

### Strengths
1. The paper is well written and clear.

2. Extensive experiments done in order to validate the claim.

### Weaknesses
1. The main contribution of the paper, as I understand it, is the efficient selection of samples for labeling, similar to standard active learning approaches. I do not see any component of this method specifically tailored for SFADA; instead, it appears to be a generic sampling technique applicable to any active learning-based adaptation. In this regard, I find limited novelty in both the problem statement and the proposed method.

2. Given that sampling is a key strength of this method, more comparisons are necessary beyond DBSCAN or K-means. For instance, how would the results differ if sampling were based solely on entropy? I believe there are additional sampling techniques in the literature that could also be tested.

3. The experiments are insufficient. The authors should compare their approach with the most recent ECCV work for SFADA, [A], which seems to achieve better results than those reported in this paper.

[A] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence

### Questions
1. In the case of the MHPL method (Active Source-Free Domain Adaptation), the reported number in the original paper is higher than what is reported here. For example, the Office-Home results in the MHPL paper are 79.1, while this paper reports 78.4 under the same experimental settings. What accounts for this discrepancy? I understand that results may sometimes differ when reproducing them from the original code, but if that is the case, it should be explicitly mentioned in the paper.

### Soundness
3

### Presentation
3

### Contribution
2
