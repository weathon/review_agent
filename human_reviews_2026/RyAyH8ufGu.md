# Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Unsupervised domain adaptation (UDA) is a statistical learning problem when the distribution of training (source) data is different from that of test (target) data. In this setting, one has access to labeled data only from the source domain and unlabeled data from the target domain. The central objective is to leverage the source data and the unlabeled target data to build models that generalize to the target domain. Despite its potential, existing UDA approaches often struggle in practice, particularly in scenarios where the target domain offers only limited unlabeled data or spurious correlations dominate the source domain. To address these challenges, we propose a novel distributionally robust learning framework that models uncertainty in both the covariate distribution and the conditional label distribution. Our approach is motivated by the multi-source domain adaptation setting but is also directly applicable to the single-source scenario, making it versatile in practice. We develop an efficient learning algorithm that can be seamlessly integrated with existing UDA methods. Extensive experiments under various distribution shift scenarios show that our method consistently outperforms strong baselines, especially when target data are extremely scarce.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel method for unsupervised domain adaption based on the framework of distributional robust optimization. The key idea is to represent the conditiona distribution $P(Y|X)$ as the mixture of empiricial conditional distribution $\hat{P}^{(k)}(Y|X)$ from multiple sources and allows the pertubation of target input distribution over move inside a small Wasserstein ball. This method is plug-and-play, can be effectively integrated with existing UDA methods. Experiments on digits (MNIST/SVHN/USPS) and spurious-correlation suites (Waterbirds, CelebA, Colored-MNIST) show consistent gains and strong performance when target data are very limited.

### Strengths
1. The paper is quite well-written. 

2. The proposed  method is quite intuitive and easy to understand, with tractable surrogate and a relatively simple algorithm.

3. The algorithm integrates seamlessly with existing UDA frameworks since its neat design that the input of the algorithm is the feature mapping $z$. 

4. The empirical effectivness is validated on real-world datasets with non-trivial improvements.

### Weaknesses
Currently, the largest potential issue is the lack of discussion of the grouping strategy. Specifically, it is not clear for a single soruce distribution, how do we know the number of pseudo-sources $K$? The choice of $K$ should directly influence the performance given the main idea that the target conditional distribution is the mixture of empirical conditional distribution. It will be appreicated to provide a more detailed analysis regarding the choice of $K$ (e.g., heuristics, validation criteria, stability checks), plus a short sensitivity or ablation study.

I’m not an expert in UDA, so I’m unsure whether the current baselines reflect the latest methods. A quick check search leads me to two recent papers [1], [2]. Could you comment on their relevance and, if appropriate, explain why they weren’t included (e.g., different setting, data requirements, or incompatibility)? I’m completely open to your explanation.

Another potential issue the optimization objective is surrogate-based, i.e., we are always apporacing a suboptimal result. In this case, an analysis regarding the tightness of the gap will be appreciated. However, I totally understand if there have not been any since this is not the contribution of the paper. I will not change my assessment of the paper regardless of the absent of such analysis.

[1] Partial Identifiability for Domain Adaptation. 

[2] Subspace identification for multi-source domain adaptation.

### Questions
Most questions have been proposed in the Weakness section.

A minor question:

During the analysis of impact of radius $(\epsilon_1, \epsilon_2)$, I do not quite understand the content from lines 1004-1007. The aurthors argue that $\epsilon_2$ will play a critical role when the target data is quite scarce. However, according to the Figure 5(b), the $\epsilon_2$ provides nearly no influence for a fixed $\epsilon_1$. Can the authors elaborate more on their arugments, or am I missing anything?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a distributionally robust optimization (DRO) framework for unsupervised domain adaptation (UDA), particularly under multi-source and target-data-scarce settings. The method defines an ambiguity set over both the covariate (input) distribution and the conditional label distribution, allowing robustness against (i) uncertainty in target inputs and (ii) uncertainty in which source conditional distributions to rely on. The framework is applicable to both multi-source and single-source UDA, using pseudo-sources generated via sub-sampling. A tractable minimax algorithm is derived (Eqs. (3)–(7); Algorithm 1), optimizing over feature perturbations, mixture weights, and classifier parameters. Experiments on digit datasets (MNIST, SVHN, USPS) and spurious-correlation benchmarks (Waterbirds, CelebA, CMNIST) demonstrate consistent performance gains over UDA and robust-learning baselines (Tables 1–2).

### Strengths
1.	The paper identifies two overlooked issues in UDA: scarce unlabeled target data and spurious source correlations.
2.	This dual modeling is theoretically elegant and practically relevant for multi-source robustness.
3.	The paper is well-written and easy to read.

### Weaknesses
1.	D₁ (Wasserstein-∞) and D₂ (Euclidean) are chosen “for computational tractability” (Sec. 3.3) but without theoretical or empirical justification.
2.	Hyperparameters ϵ₁, ϵ₂ are selected via a small labeled validation set (Sec. 4.1), partially violating the unsupervised setting.
3.	No ablation compares using only conditional-mixing vs. only covariate-perturbation.
4.	The relationship between pseudo-source construction (Sec. 3.1) and mixture weights β is not fully explained; readers may confuse stochastic sub-sampling with real domain partitioning.
5.	Absence of gradient-stability discussion under joint optimization may raise reproducibility concerns.
6.	Compared to DRO literature [a, b], the paper’s theoretical contribution is weak.

[a] Learning models with uniform performance via DRO.
[b] Distributionally robust stochastic optimization with Wasserstein distance.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

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
This paper proposes a novel distributionally robust optimization (DRO) framework for multi-source unsupervised domain adaptation (MS-UDA). The method models both input marginal distribution shifts and label conditional distribution shifts by constructing a mixture-based ambiguity set that combines conditional distributions from multiple source domains with adaptive weighting, while allowing controlled perturbations of the target input distribution. The authors further design a minimax optimization algorithm that alternately updates feature perturbations, mixture weights, and classifier parameters. Experimentally, results on several benchmark datasets (MNIST/SVHN/USPS, Waterbirds, CelebA, CMNIST) demonstrate that this approach achieves significantly better classification performance than mainstream UDA methods such as DANN, CDAN, and STAR, particularly in scenarios with scarce target data or spurious correlations.

### Strengths
1. The paper proposes a unified Distributionally Robust Optimization (DRO) framework that is applicable to both multi-source and single-source Unsupervised Domain Adaptation (UDA), offering flexibility across different scenarios.

2. Unlike traditional approaches that typically model either input uncertainty or label distribution uncertainty in isolation, this method simultaneously accounts for both, providing a more comprehensive solution.

3. The proposed algorithm is highly tractable, ensuring that it can be seamlessly integrated with existing UDA frameworks, making it easy to adopt and implement in real-world applications.

4. The experimental results demonstrate the method's robust performance, especially in tasks where target data is scarce or where spurious correlations are prevalent, outperforming several strong baselines.

### Weaknesses
1. A reliance on labeled target data for model selection moves the problem into a "partially supervised" or "few-shot" adaptation setting. While the main training uses unlabeled target data, the crucial choice of hyperparameters $\epsilon_1$ and $\epsilon_2$ is supervised. 

2. The appendix shows heatmaps of performance vs. $(\epsilon_1, \epsilon_2)$, which is good. However, the main paper should discuss how sensitive the method is. Is there a broad range of "good" hyperparameters, or does the performance collapse without precise tuning? This context is crucial.

3. Although multiple standard datasets are used for testing, there is a lack of evaluation in broader domains (such as NLP or time-series data) to demonstrate the generalizability of the method.

4. The conclusion section does not sufficiently discuss the limitations of the method and potential future directions for improvement. Adding these aspects would help present a more comprehensive view of the research depth.

### Questions
1. How does the reliance on labeled target data for hyperparameter selection (specifically for $\epsilon_1$ and $\epsilon_2$) affect the generalization of the method in unsupervised domain adaptation? Could this be considered as a shift towards a "partially supervised" or "few-shot" adaptation setting?

2. How sensitive is the proposed method to these hyperparameters?

3. How does the proposed method perform on data from other modalities?

4. How critical is the choice of base classifier (ERM vs. CDAN/STAR) for the final performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper applies Distributionally Robust Optimization (DRO) to solve Unsupervised Domain Adaptation (UDA). It constructs ambiguity samples from the target data whose pseudolabel is the linear combination of the label from multiple source domain classifiers. The ambiguous sample is generated using a technique similar to adversarial training. The proposed method can also be viewed as a regularizer.

### Strengths
1. Novel extension of DRO to UDA
2. Impressive results on target accuracy

### Weaknesses
1. The paper attempts to apply Distributionally Robust Optimization (DRO) to the problem of Unsupervised Domain Adaptation (UDA). The idea of applying DRO to UDA appears to be novel although some of the technical contributions are from pre-existing DRO work. The proposed approach is not technically convincing due to tedious notation and workflow. It was quite difficult and frustrating to follow the logic in explanation in the paper. 
2. Target accuracy is the only result made available. This does not give enough insight into the workings of the model.
3. What is the intuition behind Eq. 6. Whichever subset (domain) k has the least predictive power (largest loss) it gets the largest \beta_k. Why is this reasonable? Will this not result in negative transfer? 
4. In Eq. 5, How is Euclidean projection onto set A equivalent to minimizing the Wasserstein distance D1. What do you mean by Euclidean projection? How is the \epsilon_1 constraint satisfied?

### Questions
In Eq 3. What is variable for the Expectation under \hat{P}^{tg}_X. Is it z’ or z(X)? Likewise, in the expectation In lines 775-777 what is the difference between z(x) and z(X). How does it change to z’ in lines 781-782. There is inadequate explanation.

### Soundness
3

### Presentation
2

### Contribution
3
