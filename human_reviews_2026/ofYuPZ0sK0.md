# Optimizing Data Augmentation through Bayesian Model Selection

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Data Augmentation (DA) has become an essential tool to improve robustness and generalization of modern machine learning. 
However, when deciding on DA strategies it is critical to choose parameters carefully, and this can be a daunting task which is traditionally left to trial-and-error or expensive optimization based on validation performance.
In this paper, we counter these limitations by proposing a novel framework for optimizing DA. 
In particular, we take a probabilistic view of DA, which leads to the interpretation of augmentation parameters as model (hyper)-parameters, and the optimization of the marginal likelihood with respect to these parameters as a Bayesian model selection problem. 
Due to its intractability, we derive a tractable ELBO, which allows us to optimize augmentation parameters jointly with model parameters. We provide extensive theoretical results on variational approximation quality, generalization guarantees, invariance properties, and connections to empirical Bayes.
Through experiments on computer vision and NLP tasks, we show that our approach improves calibration and yields robust performance over fixed or no augmentation. 
Our work provides a rigorous foundation for optimizing DA through Bayesian principles with significant potential for robust machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a Bayesian model selection framework for online optimization of the data augmentation (DA) hyperparameters, provides a tractable ELBO for variational approximation of the marginalized likelihood, performs extensive analysis of the theoretical characteristics of the augmented model, uncertainty quantification, calibration, decision boundary smoothness and generalization. The theoretical results are backed by several experiments with synthetic and image datasets optimizing the DA parameters of simple and also state of the art baselines in DA and improving the prediction and calibration with almost no additional computational cost.

### Strengths
The paper is well written and easy to follow. The authors have laid out the background and  the motivation and provide good structured explanation of the proposed method and its theoretical characteristics. Several experiments are provided to compare the Bayesian DA's performance to other baselines and highlight the theoretical results.

The authors use DA hyperparameters as latent mode parameters and use a Bayesian framework with ELBO optimization to efficiently find the best DA parameters according to the data invariance and model sensitivity. In the theoretical analysis, Bayesian DA method is compared to naïve DA in terms of the PAC-Bayes generalization bounds, regularization and smoothening effect of Bayesian DA and improvement in calibration of the predictions. 

The analysis also sheds light on the connections of DA with information bottleneck which I find particularly novel.

### Weaknesses
It might be easier to follow the paper if a brief explanation of the theoretical results of the Theorems are provided before the theorems, as there are multiple theorems and moving from one technical result to the next requires some context in between. 

Minor comment: 
Typo: line 107 wrt -> with respect to
Typo: line 119 nav̈e -> naive
Typo: line 229 extra s

### Questions
Would the proposed method also work for when transformations are not continuous as in natural language e.g. based on word substitutions and syntactic transformations? 

Can the authors provide the intuition of THM4.5? In the the naïve DA the variable $gamma$ is integrated out in $R(\theta)$ with a uniform empirical distribution as opposed to the latent distribution $p(\gamma) = \int p(\gamma|\phi)p(\phi|\theta)d\phi$ or its variational approximation $p(\gamma) = \int p(\gamma|\phi)q(\phi)d\phi$?

### Soundness
4

### Presentation
3

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
This paper proposes the OPTIMA framework, which optimizes data augmentation parameters by using Bayesian methods. The core idea is to treat DA parameters as latent variables and introduce variational inference methods to simultaneously optimize both model parameters and DA parameters. An extensive theoretical analysis is conducted, including generalization guarantees and invariance properties. Experiments demonstrate that OPTIMA enhances confidence and robustness in regression and classification tasks.

### Strengths
-The paper proposes a Bayesian framework for the joint optimization of data augmentation parameters and model parameters.

-The theoretical analysis is comprehensive, including approximation bounds, generalization guarantees, and invariance derivations.

### Weaknesses
-The paper primarily focuses on computer vision tasks (such as image classification) and does not extend to other modalities (such as natural language or time-series data), which limits its generalizability claims.

-The compared algorithms are outdated.

-The impact of the dimensionality of data augmentation parameters on the proposed method is not discussed.

-Some theorems rely on mathematical assumptions that are difficult to apply in practice.

### Questions
Q1: How does OPTIMA perform in non-computer-vision domains, such as time series prediction or natural language processing?

Q2: How high a dimension of data augmentation parameters can the proposed method handle?

Q3: The comparison algorithms selected in this paper were mainly published around 2020. Considering the rapid development in the field of automatic data augmentation, many more representative methods have emerged in recent years. It is recommended that the authors supplement the comparison experiments with cutting-edge methods proposed in the last year or two (e.g., 2022-2024). This will enable a more comprehensive and fair evaluation of the proposed methods' advancement and effectiveness, thereby significantly enhancing the persuasiveness of this paper's contributions.

Q4: What are the innovations compared to the traditional Bayesian selection optimization algorithms based on variational inference?

Q5: The proposed method may get stuck in local optima. Have convergence guarantees or initialization strategies been considered?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose OPTIMA, a Bayesian framework that optimizes data augmentation parameters by maximizing the marginal likelihood instead of relying on validation-based hperparameter tuning. 

They validate their approach with theoretical analysis based on information-theoretic principles, to show that the method is probabilisticly sound. 

They evaluated on ImageNet, ImageNet-C, and CIFAR-10 and shows that OPTIMA achieves better calibration and competitive accuracy.

### Strengths
The authors address the interesting, useful task of data augmentation as a technique that helps improve model performance across different downstream tasks. 

The authors provide solid theoretical foundations for their proposed method, OPTIMA, highlighting the types of invariances it promotes and the uncertainty quantification it enables. 

They also present experiments on standard datasets such as ImageNet, showing that OPTIMA slightly outperforms methods that do not use data augmentation

### Weaknesses
Methods like bilevel optimization [1] also aim to optimize data augmentation functions, and it would be nice to show how this paper differentiates OPTIMA from these related approaches in terms of how good they perform augmentation.

The field of image classification is already quite saturated; it would be more compelling to see results on harder tasks such as segmentation or object detection.

While data augmentation optimization is valuable, it doesn’t provide fundamentally new information, unlike larger supervised models (e.g., transformers) that may implicitly learn augmentation effects by training on a large corpus in a semi-supervised way.

The reported improvements on CIFAR-10 and ImageNet appear insignifcant, and given how saturated these benchmarks are, it’s unclear whether the gains are statistically meaningful.

The paper lacks standard deviations or error bars so it is hard to know if any results are actually significant.

The implementation details are unclear, how were the hyperparameters selected/tuned without a validation set (like in CIFAR10)?

[1] Mounsaveng et al., Learning Data Augmentation With Online Bilevel Optimization for Image Classification, WACV 2021.

### Questions
How does OPTIMA fundamentally differ from existing bilevel optimization approaches for data augmentation, such as Mounsaveng et al. (WACV 2021) [1]?

Can the proposed framework be extended to more complex tasks like segmentation or detection, and what challenges would come up in doing so?

How were  the hyperparameters chosen on datasets without a validation set (e.g., CIFAR-10)?

[1] Mounsaveng et al., Learning Data Augmentation With Online Bilevel Optimization for Image Classification, WACV 2021.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a Bayesian optimization strategy to optimize data augmentation parameters; while the marginal likelihood is intractable, the work proposes a tractable ELBO which variational approximation to the joint $q(\theta, \phi) = q(\theta)q(\phi)$ and introduces regularization terms. The authors include several analyses of this framework, namely: (1) a characterization of the quality of the variational inference approximation in Section 4.1, (2) a PAC-Bayes generalization bound in Section 4.2, (3) a second-order bound on the expected difference in the model’s output under certain transformations (which are perturbations added to  with mean 0) in Section 4.3, (4) a comparison of naive DA compared to the true marginalization (not the variational inference approximation) in terms of covariance of the true posteriors being smaller under naive DA by a factor of K (the number of transformations applied to each data point) in Section 4.4, (5) interpret the solutions which maximize the augmented ELBO as empirical Bayes solutions for $p(\theta | D)$ and $p(\phi | D)$ in Section 4.5, and (6) show a reduction in posterior distribution entropy under the true marginalization (not the variational inference approximation) compared to having no augmentation in Section 4.6. The authors also employ their method on synthetic data, ImagetNet, ImageNet-C, and CIFAR-10 to demonstrate their theoretical findings.

Note: I have reviewed an earlier version of this manuscript, and see that the Appendix F.3 has now been moved to the main body to compare the computational efficiency this work to Bayesian optimization. There also appears to have been effort to better relate the experiments to theoretical results, namely top of page 9. Some other changes which had been promised but not applied are:
- Relation to Chen et al. 2020 (and maybe other works, like Chatzipanzis et. al, 2021) in Section 2. This may fit under probabilistic perspectives of DA.
- Some mathematical precision comments by a couple reviewers, e.g., simply defining terms or noting the overloading of notation would suffice. I would argue that these imprecisions should be clarified explicitly in the manuscript or addressed.
    - For example, the proof of Theorem 4.5 needs K (the number of augmentations/replications) to be large enough that the naive empirical risk is close to the true one. I still think this is a bit imprecise, but the minimal changes that the authors can do is state “for K large enough” in the statement of Theorem 4.5.

### Strengths
- The theoretical results are extensive and address multiple aspects of this framework. The work appears to give practical insight on what relevant quantities and choice of distribution over data augmentation parameters affect various aspects of model performance within this framework of data augmentation via Bayesian optimization. The corollaries written throughout the text after theorems give insight and practical advice, and seem to cover many different lens through which to understand how this framework (alternating between the variational inference approximation and the true posterior under this framework) compares to naive DA and having no augmentation.

### Weaknesses
- The noted mathematical imprecision comments from previous reviews.
- Some experimental results can be confusing to some readers. In Section 5.1, Figure 2, the “test loss” figure is unclear and does not appear to clearly show what the text is saying. Also, perhaps a reference to Appendix F.2 or including it in the main body would be appropriate and explain why ResNet is getting less than 80% accuracy in Table 2.

### Questions
- Why do the authors choose to report their results for Cutmix, etc. reported in Table 2 and have not done the same comparisons, for instance, the setup in Appendix F.2 which uses Gaussian Translation. I think it would confuse some readers why ResNet is achieving under 80% accuracy on these datasets.

### Soundness
2

### Presentation
3

### Contribution
3
