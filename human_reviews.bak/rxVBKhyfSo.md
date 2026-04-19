# Selective Mixup Fine-Tuning for Optimizing Non-Decomposable Objectives

- Decision: Accept (spotlight)
- Scores: 6, 8, 8, 6

## Abstract
The rise in internet usage has led to the generation of massive amounts of data, resulting in the adoption of various supervised and semi-supervised machine learning algorithms, which can effectively utilize the colossal amount of data to train models. However, before deploying these models in the real world, these must be strictly evaluated on performance measures like worst-case recall and satisfy constraints such as fairness. We find that current state-of-the-art empirical techniques offer sub-optimal performance on these practical, non-decomposable performance objectives. On the other hand, the theoretical techniques necessitate training a new model from scratch for each performance objective. To bridge the gap, we propose SelMix, a selective mixup-based inexpensive fine-tuning technique for pre-trained models, to optimize for the desired objective. The core idea of our framework is to determine a sampling distribution to perform a mixup of features between samples from particular classes such that it optimizes the given objective.  We comprehensively evaluate our technique against the existing empirical and theoretically principled methods on standard benchmark datasets for imbalanced classification. We find that proposed SelMix fine-tuning significantly improves the performance for various practical non-decomposable objectives across benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose SelMix which utilizes a pre-trained model for representations and optimizes it for improving the desired non-decomposable objective through fine-tuning. They developed a selective sampling distribution on class samples to selectively mix up, optimizing the given non-decomposable objective. The distribution is also updated periodically based on feedback from the validation set. The efficacy of SelMis is evaluated across a wide range of linear or non-linear objectives under both supervised and semi-supervised settings.

### Strengths
1. The key novelty lies in the idea of selective mixup. The selective procedure ensures that at each timestep the the objective is optimized.

2. The theoretical analyses present in the paper are sufficient. They provide the convergence analysis and show the validity of the sampling procedure. 

3. The experimental results are comprehensive, which consist of strong empirical results under different supervision settings and different distributions.

### Weaknesses
1. The authors use the first-order Taylor expansion on Eq. 5. I am curious about the performance if the second-order terms are involved, i.e., calculating the third term of Eq. 5. Also, the first-order gradient can be also approximated by f(x+ h) + f(x-h) -f(x) / 2h. Would it give better performance than the current one-sided approximation?

2. Can authors provide some ablation studies on the hyper-parameters, such as the imbalanced ratio and the number of samples for feedback?

### Questions
Please see weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel approach called Selective Mixup (SelMix), which leverages an optimized sampling distribution of elements to mix based on a specific objective. This sampling distribution is designed to consider the gain in the objective function achieved by mixing centroids of class samples. Throughout the training process, the sampling distribution is dynamically updated to adapt to changes in class centroids. The paper also includes a rigorous theoretical analysis, establishing the approach's convergence rate and the validity of the sampling distribution. To validate its effectiveness, the method is empirically evaluated across various long-tailed benchmark datasets in both supervised and semi-supervised scenarios, by fine-tuning pretrained models using SelMix.

### Strengths
- **Theoretical Analysis and Convergence Rate:** The paper's presents a rigorous theoretical analysis, particularly the establishment of a convergence rate. This analysis adds credibility and reliability to the proposed SelMix.
- **Strong Experimental Results:** The paper showcases strong experimental results compared to other state-of-the-art approaches. The empirical evaluation is performed across different long-tailed benchmark datasets in both supervised and semi-supervised settings, and for different objectives.
- **Efficiency of the proposed approach:** As discussed in Appendix L, the paper is much more computationally efficient than previous methods, since it relies on fine-tuning a pretrained model.

### Weaknesses
- **On a Fair Comparison with Other Methods:** The paper uses FixMatch + Logits Adjusted (LA) loss as a baseline, while other approaches are evaluated using the vanilla FixMatch. This discrepancy in the comparison may not provide a fair assessment of the true impact of SelMix. As can be seen in Table H.2 and H.1, FixMatch + LA shows already significant improvements over vanilla FixMatch. This makes it challenging to evaluate whether SelMix's improvements reported in the benchmarks are due to the approach or its combination with a more advanced baseline. A fair comparison would involve evaluating SelMix with vanilla FixMatch to better understand its relative performance, or comparing to other state-of-the-art methods when using FixMatch + LA.
- **Lack of Discussion about other related "Selective Mixup":** Several recent paper have discussed using *selected pairs* of examples to mix, between specific classes or different domains [1,2,3,4,5]. Even though the approach presented is applied in a different context and quite novel compared to these papers, I think a discussion about the relation to these methods is important since the paper is currently part of a popular topic. 

[1]: Yao H. et al., C-mixup: Improving generalization in regression. In NeurIPS 2022  
[2]: Yao H. et al., Improving out-of-distribution robustness via selective augmentation. In ICML 2022  
[3]: Hwang I. et al., Selecmix: Debiased learning by contradicting-pair sampling. In NeurIPS 2022  
[4]: Palakkadavath R. et al., Improving domain generalization with interpolation robustness. In NeurIPS 2022 Workshop on Distribution Shifts: Connecting Methods and Applications  
[5]: Teney D. et al., Selective mixup helps with distribution shifts, but not (only) because of mixup. In arXiv:2305.16817, 2023.

### Questions
- What is the *prior distribution* on labels $\pi_i$ ? How do you define this in practice ?
- What is the *unconstrained confusion matrix* $\tilde{C}$ ? It is never clearly defined in the paper, but is important to define the gain.
- I would like to see at least some comparison of SelMix using "vanilla" FixMatch to fairly compare with state of the art.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
For applications with critical consequences, accuracy is not a suitable performance metric, and other metrics such as recall h-mean and worst-case recall should be used. However, such metrics are non-decomposable, which means that they cannot be expressed as a simple average of a function of label and prediction pairs calculated for each sample. Prior techniques to optimize non-decomposable objectives such as CSST lead to sub-optimal representations. Other methods to improve the performance on long-tailed class-imbalanced datasets such as DASO, ABC, and CoSSL perform suboptimally for the non-decomposable objectives. This paper proposes SelMix -- a technique that utilizes a pre-trained model for representations and optimizes it for improving the desired non-decomposable objective through fine-tuning.

### Strengths
In the experimental results for matched label distributions shown in Table 2, the proposed method shows shows superior performance compared to existing methods; DARP, CReST, CReST+, ABC, CoSSL, DASO, and CSST for all metrics; mean recall, min recall, H-mean, G-mean, and min coverage. The results for unknown label distributions in Figure 3 also shows that the proposed method outperforms existing methods on all metrics. The same is true for large datasets such as ImageNet-1k LT, shown in Table 4.

### Weaknesses
The fact that the proposed method requires only fine-tuning of a pre-trained model should result in a huge advantage in training time compared to existing methods, but this is not highlighted in the main results section of the paper. The only mention of computational requirements is in Appendix L, where only the time to calculate the Gain is shown. It would be interesting to see a more comprehensive comparison of the training time of the proposed method vs. all the existing methods.

### Questions
The description of SelMix in Algorithm 1 is a bit confusing. Algorithm 1 says the classifier h is updated, but in Section 4.1 it says the parameter W is updated. I’m assuming the former is a consequence of the latter. If this is the case, wouldn’t it be better to replace h with W in Algorithm 1? Obfuscating the actual operation with the function SGD-Update() also makes it difficult to see the exact algorithm.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to fine-tune a pre-trained model using an additional objective, especially a non-decomposable objective. In particular, the authors propose a mixup-based technique that can determine a sampling distribution over classes for performing objective-oriented mix-up. Empirical results demonstrate the efficacy of the proposed method.

### Strengths
1. The proposed method, SelMix, is simple and reasonable, with a theoretical guarantee.  
2. Sufficient empirical results are provided.

### Weaknesses
1. Literature review is not sufficient, and some sentences are over-claimed. Fine-tuning a pretrained network for non-decomposable objectives and even non-differential objectives has been extensively developed recently. Please refer to a recent paper [1] and the references therein for more details. Particularly, the objective derived $\mathbb{E}[G]=\sum_{i, j} G_{i, j} \mathcal{P}_{M i x}(i, \jmath)$ shares the same motivation as the natural evolution strategies [2] adopted in [1].
2. There are many grammar errors, needed to double check. For example, "existing frameworks theoretical frameworks", "Semi-Supervised Learning is are algorithms", $f: \mathbb{R}^{m \times n} \rightarrow n$, $\rho_l=N_1 / N_K, \rho_l=M_1 / M_K$.  
3. Some theorems are not formal and have overly strong assumptions. For instance, $z_k$ in Theorem 4.1 appears suddenly without intuitive explanation. There is no detailed explanation in the proof in the Appendix. “a reasonable directional vector for optimization” is not professional.   

[1] Li, Jing, et al. "Earning Extra Performance from Restrictive Feedbacks." IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).  
[2] Wierstra, Daan, et al. "Natural evolution strategies." The Journal of Machine Learning Research 15.1 (2014): 949-980.

### Questions
1.	What is the difference between "Non-Decomposable Objective" and "Non-Differentiable Objective"?
2.	The first-order Taylor approximation is adopted in this paper for calculating the Gain matrix, which cannot guarantee convergence for complex objectives, although it is simple and efficient. 
3.	It seems we can fitting a surrogate model to approximate the Non-Decomposable Objective, which can then be used for gradient-based model fine-tuning. It would be interesting to discuss the advantages of the first-order Taylor approximation compared to fitting a surrogate model.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
