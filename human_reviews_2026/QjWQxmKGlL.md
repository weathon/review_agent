# Towards Distribution-Aware Active Learning for Data-Efficient Neural Architecture Predictor

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
As the neural predictor (NP) provides a fast evaluation for neural architectures, it is highly sought after in neural architecture search (NAS). However, the high computational cost involved in generating training data results in its scarcity, which in turn limits the accuracy of the NP. Active learning (AL) has the potential to address this issue by prioritizing the most informative samples, yet existing methods struggle with selection bias when faced with imbalanced data distributions, often prioritizing diversity over representativeness. In this paper, we redefine the sample selection mechanism in AL and propose a Distribution-aware Active Learning framework for Neural Predictor (called DARE). The goal is to select samples that not only ensure diversity but also exhibit a high degree of generalizability, making them more representative of the underlying data distribution. Our approach first extracts architecture representations via a graph-based encoder enhanced with a consistency-driven objective. Then, a two-stage selection strategy identifies both globally diverse and locally reliable samples through progressive representation learning and refinement. For non-uniform data distributions, we further introduce an adaptive mechanism that anchors sampling to key regions with high similarity density, avoiding performance degradation caused by outliers. Extensive experiments have shown that the proposed distribution-aware active learning strategy samples a higher-quality training dataset for NPs, allowing the neural architecture predictor to achieve state-of-the-art results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces an active learning framework designed to tackle the challenge of limited training data for neural predictors in Neural Architecture Search. The core methodology focuses on strategically choosing which unlabeled data should be annotated throughout the training process.

### Strengths
1.The problem tackled in this paper holds practical importance. A key challenge faced by neural predictors today is the substantial cost of obtaining training data. This paper seeks to mitigate this problem by identifying and selecting unlabeled data.

2.The paper is well-organized and clearly articulated, enhanced by a flowchart that facilitates understanding, making it simpler for readers to follow the author's ideas and objectives.

3.The experiments presented in the paper are thorough, carried out across several datasets, and effectively demonstrate the efficacy of the proposed method. Furthermore, a theoretical analysis is included to strengthen the credibility of the method's performance.

### Weaknesses
1.The proposed method lacks significant novelty, as it primarily introduces a relatively straightforward data sampling approach. To convincingly demonstrate its effectiveness, the authors should compare their method with existing state-of-the-art active learning techniques. At the same time, this sampling strategy seems quite straightforward and easy to conceive. The authors need to further explain where the innovation lies.

2.The theoretical analysis provided in the paper is insufficient. Specifically, the analysis only offers a rough estimation of data requirements, which is not enough to rigorously support the claim that a higher cosine similarity necessarily leads to a smaller required data size. At the same time, in the proof in Appendix D, I couldn't find a clear justification for choosing the value of t in this way; it seems that other values could also be feasible.

3.The experimental evaluation is somewhat limited, as the neural network sizes tested are relatively small. This diminishes the practical relevance of the work, especially considering the current trend toward larger model architectures. At the same time, the authors need to conduct experiments in a larger search space. The current experiments only on NASBench are far from sufficient.

### Questions
Please refer to the Weaknesses part. If these issues are addressed, I will consider revising the score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors tackle the data inefficiency in neural predictor training by combining active learning with distribution-aware sampling. A two-stage max–min strategy ensures that selected architectures are both diverse and reflective of the true data distribution, while an adaptive mechanism handles nonuniform datasets. The approach uses a consistency-based training objective that stabilizes representation learning, enabling the predictor to achieve strong generalization with very limited labeled data.

### Strengths
The paper addresses a highly important and timely problem in Neural Architecture Search (NAS), which plays a critical role in the efficiency of deep learning systems. It introduces a novel and wellmotivated approach that combines several existing ideas in a coherent and innovative manner. While the individual components are known, the proposed integration and formulation are original.

### Weaknesses
*
Theoretical Analysis*
The analysis applies Hoeffding’s inequality under an i.i.d. sampling assumption; however, the proposed sampler is an adaptive active-learning procedure in which each selection depends on prior choices. This adaptivity violates the independence requirement underlying the stated bound, and the resulting limitation is not addressed.
Furthermore, the substitution of the tolerance term ( t ) in a Hoeffding-style bound with an empirical “diversity” score—defined as the average ( (1 - \cos(X_i, X_j)) )—is asserted without proof. Higher pairwise diversity does not, in general, guarantee that the empirical mean of a selected subset is closer to the population mean. Consequently, the claim that “higher diversity ⇒ smaller required sample size” is not substantiated
under the stated assumptions.

*Minor*
There is also a type/notation inconsistency in ((X_i)). In the bound, (X_i) is implicitly a bounded scalar random variable to which Hoeffding applies, whereas elsewhere (X_i) is treated as a highdimensional embedding used to define pairwise cosine dissimilarity. Absent an explicit scalar functional of (X_i) and corresponding boundedness conditions, the subsequent reasoning is not mathematically consistent as written.

*Experimental evaluation*
The empirical results are encouraging but not yet sufficiently compelling. The baselines highlighted in the main tables (e.g., classical neural predictors and earlier NAS acquisition strategies) are dated compared to contemporary alternatives. The reported improvements—such as modest gains in Kendall’s ( \tau ), slightly higher final architecture accuracy, and comparable performance with fewer labeled samples—are relatively small and plausibly fall within run-to-run variability. Uncertainty quantification is incomplete: statistical significance tests, confidence intervals, and seed-to-seed variance are not comprehensively reported across methods and budgets. Given that label efficiency is central to the claimed contribution, the absence of thorough variability analysis weakens the empirical support. In its current form, the experimental section substantiates the general direction of the work but does not yet demonstrate robust superiority over modern alternatives.

*Reproducibility and transparency*
For independent verification, the paper must report all details necessary to reproduce the claimed gains. In particular, it should fully specify the predictor architecture and training setup — including the exact GCN design, optimizer configuration, data augmentation and consistency-training procedure, and training schedule — as well as the acquisition strategy, covering all sampling hyperparameters, the precise clustering configuration, and the pseudo-labeling procedure.

The evaluation part — which is central to the paper’s contribution — is insufficiently described. Critical details are missing, such as the dataset splits, the exact procedures used to compute Kendall’s ( \tau ), MSE, and ranking quality metrics. Without these components, the claimed improvements cannot be independently validated. Moreover, the paper should disclose essential experimental logistics, including random seeds, number of runs, and hardware/software versions used.

### Questions
Q1) The theoretical section, in its current form, makes overreaching claims. It cites Hoeffding-style guarantees under assumptions (i.i.d. bounded scalars) that are inconsistent with the adaptive two-stage selection involving pseudo-label retraining, and it informally substitutes “pairwise cosine diversity” into a generalization-style bound without proof. This part should either be reframed as an intuitive explanation (rather than a formal theorem) or strengthened with rigorous justification and clearly stated assumptions.


Q2) The empirical evidence, while promising to some degree, requires stronger baselines highlighted in the main text, clearer reporting of statistical significance and variability, and a more comprehensive disclosure of experimental details to enable full reproducibility of the claimed gains.


Please refer to these questions in the rebuttal and how you will specifically improve the paper in these respects (as reviewers we are not interested in some private education but in improvements of the manuscripts only).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DARE, a distribution-aware active learning framework for training neural predictors used in Neural Architecture Search (NAS). Since labeling architectures (training each candidate to get accuracy) is expensive, DARE aims to efficiently choose which architectures to label. It introduces a GCN predictor to obtain architecture embeddings along with a two-stage max-min sampler and key-point guided adaptive sampling strategy for non-uniform distributions. The authors compare on NASBench101, 201, and DARTS and show that they get SoTA performance.

### Strengths
- The problem of identifying NP training data is overlooked, and AL for this task seems to be a good approach
- The evaluations are thorough, the paper is easy to read

### Weaknesses
- I don't get the usefulness of theorem 4.1 in the context of the paper.  It doesn't utilize the optimization at hand, or the sampling process; nor justifies the two-stage approximates an optimal subset under some acquisition function. It just seems to be an extremely loose bound, with no justification on $[a_i,b_i]$  -- I am not sure if you can bound these embedding norms
- Equation 3 confuses me. In appendix C, the authors write that the GCN is exposed to isomorphic representations of architectures. I would expect them to bring these representations closer; However eq 3 is minimizing the similarity, pushing them away. 
- The paper also states that data augmentation here is used to increase the amount of data. But since it is restricted to isomorphic reorderings, it only solves inductive biases instead of increase data to search the sparse neural space. I don't see the improvement? Can the authors justify this, and run without the closeness term in the loss as ablation? Also, how many permutations do you consider per architecture?
- The paper also lacks comparison to standard AL baselines in the main and is deferred to appendix. Considering the paper essentially proposes an AL method over NAS search space, comparison with AL in the main is more important than other neural predictor baselines.
- L427: how do you process NB101?
- The method also depends on a lot of hyperparameters, with some values not mentioned, which makes it difficult to reproduce. 
- How long is the training time / complexity? Since a lot of operations are O(N^2) 
- Does intersection of the unlabeled samples the best choice? I think re-ranking them to choose would be better suited?
- Minor: typo in L150

Overall, I think the idea of solving the NP training problem via AL is good. But in the current state, I don't find the theory concrete enough. I lean towards weak reject, but I would be looking forward to hear back from the authors regarding the concerns.

### Questions
Mentioned along with Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
