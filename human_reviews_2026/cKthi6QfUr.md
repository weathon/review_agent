# A Statistical Theory of Overfitting for Imbalanced Classification

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Classification with imbalanced data is a common challenge in machine learning, where minority classes form only a small fraction of the training samples. Classical theory, relying on large-sample asymptotics and finite-sample corrections, is often ineffective in high dimensions, leaving many overfitting phenomena unexplained. In this paper, we develop a statistical theory for high-dimensional imbalanced linear classification, showing that dimensionality induces truncation or skewing effects on the logit distribution, which we characterize via a variational problem. For linearly separable Gaussian mixtures, logits follow $\\mathsf{N}(0,1)$ on the test set but converge to $\\max\\{\\kappa,\\mathsf{N}(0,1)\\}$ on the training set---a pervasive phenomenon we confirm on tabular, image, and text data. 
This phenomenon explains why the minority class is more severely affected by overfitting. We further show that margin rebalancing mitigates minority accuracy drop and provide theoretical insights into calibration and uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses minority-class overfitting and poor confidence calibration in high-dimensional, imbalanced classification, where traditional theories fail to explain the mechanisms underlying overfitting. It uses a two-component Gaussian mixture model, focusing on logistic regression and SVM. A margin rebalancing method with a hyperparameter to adjust the constraints on the minority class margin is proposed. Theoretically, it analyzes the proportional asymptotic regime (the sample-dimension ratio converges to a finite value), exploring convergence of the parameter and logit distributions. Empirically, it is verified using CIFAR-10, IMDb, and IFNB data. The paper concludes that the high-dimensional training empirical logit distribution is truncated Gaussian (worse for minorities), and the test logit is standard Gaussian; optimal margin rebalancing eliminates majority bias, aligns class errors, and improves calibration. Contributions: establish an overfitting statistical theory, clarify the truncation mechanism, derive optimal hyperparameters, verify across datasets, and conjecture multiclass/non-isotropic extensions.

### Strengths
1. This paper establishes a dedicated statistical theory for high-dimensional imbalanced classification, clarifying logit distribution truncation as the core of overfitting and deriving optimal margin rebalancing, filling key theoretical gaps in the field.

2. Its theoretical conclusions are supported by 2-Gaussian Mixture Model simulations and experiments on datasets like CIFAR-10 and IMDb, showing strong consistency between theoretical predictions and empirical results.

3. The paper carries out rigorous proportional asymptotic analysis with solid proofs for parameter convergence, and its clear structure (theory→experiments→extensions) effectively improves the readability of complex theoretical content.

### Weaknesses
1. The weaknesses are mainly in the strong model assumptions that restrict the applicability of this paper. First, the paper heavily relies on the two-component Gaussian Mixture Model, which may not reflect real-world data complexities (e.g., non-Gaussian distributions or heavy noise). It lacks analysis of how to adapt its theory to such cases, and testing on datasets with more realistic distributions would enhance its applicability.

2. While conjecturing multiclass extension, the paper provides no detailed theoretical derivation (e.g., optimal τ for multiclass) or dedicated experiments (e.g., full 10-class CIFAR-10). This limits use for multiclass tasks; supplementary multiclass analyses would address this gap.

3. The theoretical analysis assumes homogeneous unit covariance for the 2-GMM. Still, it does not explore how heterogeneous covariance (e.g., different covariances for minority/majority classes) affects logit truncation and error bounds, which are common in real-world data.

4. While the paper explores phase transitions in the high-imbalance regime (Section 3.2, Theorem D.8) and defines signal strength via $|\mu\|_2^2 \propto d^b$ (parameter$b>0$), it only specifies the valid range of $\tau$ (e.g., $\tau \gg d^{(a-b-c)/2}$ for medium signal) but not how $\tau^{\text{opt}}$ adjusts with $b$. Its high-imbalance experiment (Fig.5) fixes $b=0.3$ and only varies $a$. This gap limits guidance for tuning $\tau^{\text{opt}}$ in real-world tasks with different signal intensities.

5. The key novel theoretical contribution of this paper compared to existing works is not clear. What is the challenge in deriving the theoretical results?

6. There are some existing works that establish results on biased decision boundaries in imbalanced classification, e.g., [1], which share similar results to the paper, even from different perspectives. These works should be mentioned in the paper.

[1] Hu et al., Learning Imbalanced Data with Beneficial Label Noise, ICLM, 2025.

### Questions
Please see Weakness.

### Soundness
2

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
3

### Summary
This paper develops a statistical theory for imbalanced data classification in high-dimensional settings. The authors first introduce the logit distribution to characterize overfitting in imbalanced data, and then analyze its impact, particularly on the minority class. They further extend their analysis to examine the consequences of overfitting on model calibration.

### Strengths
1. Quantifying the effects of overfitting in imbalanced data is an important and meaningful problem.
2. The study is comprehensive, and the theoretical analysis is solid.

### Weaknesses
1. The paper is not well written or well organized. The motivation behind some of the definitions and problem settings is not clearly explained. See questions below for details.

### Questions
1. The motivation for proposing the logit distribution in Definitions 2.1 and 2.2 is unclear. It is also not entirely clear what empirical phenomenon Section 2 is intended to highlight or support.
2. The purpose of introducing the problem in Definition 2.3 is somewhat unclear. It would be helpful to discuss whether similar formulations have been studied in previous literature and how this setting differs.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper provides formal results for linear classifiers (logistic regression and SVM) in the asymptotic regime where the size of the training set $n$ scales to infinity proportionally to their number of parameters $d$ ($n/d \to \delta$). The task is binary classification, the input data are assumed to be generated by a 2 component Gaussian Mixture Model, with components centered in $\pm {\boldsymbol \mu}$ and identity covariance, and the labels coincide with the component membership. In particular:
- Theorem 2.1 focuses on the SVM, and shows the existence of a critical threshold $\delta_c$ such that for $\delta<\delta_c$ the training set is linearly separable. In this case, the errors on the majority/minority class are given in terms of the sufficient statistics of the model (inner product between model’s weights and ${\boldsymbol \mu}$, bias and margin), obtained asymptotically as the solution of a variational problem. The training and testing logit distributions are shown to tend, respectively, to a rectified Gaussian and to a Gaussian. The truncation of the training distribution is more severe for the minority class, leading to overfitting.
- Proposition 3.1 considers the strategy of margin rebalancing in the proportional asymptotic regime, finding the optimal value of the parameter rebalancing the margin, in order to attain minimal balanced error.
- Theorem 3.2 considers the “high imbalance” asymptotics, defined by a vanishing (like $d^{-a}$) imbalance ratio, separation between the components of the mixture scaling like $d^b$, size of training set  scaling like $d^{c+1}$. There are 3 phases, according to the relative values of the scaling exponents: one where both minority and majority errors are small regardless the margin rebalancing parameter, one where this parameter has to scale with a certain power of $d$ in order to avoid O(1) error on the minority class, one where the linear classifier behaves as random guess.
- Section 4 pushes the analysis in the proportional regime to prove monotonicity of common miscalibration metrics with the imbalance ratio.

### Strengths
The sharp asymptotic results provide an interesting interpretation of overfitting in terms of skewness of the training logit distribution on the minority class. The sharp prediction on the optimal value of the margin rebalancing parameter is useful in practice. The high imbalance regime presents a rich phase diagram, with clear implications for the margin rebalancing strategy. Theoretical claims are supported by numerical experiments (Fig. 1, 4, 5), also on real data (Fig 2 and 3).

### Weaknesses
The theoretical analysis is limited to the very narrow setting of linear binary classification and data with isotropic covariances. Connections to some of the relevant literature in the field of high-dimensional statistics are not discussed. No clue or narrative on how the theorems are proven is given in the main text, making hard to assess their validity in a 2-weeks review (the appendix is 80 pages long). See Questions below for more detail.

### Questions
- I think the paper should mention in the title or in the abstract that it is about linear classifiers, to better represent its content.
- Can the authors discuss the connections between their work and recent analysis in similar setting? I am thinking at [1] (which considers Gaussian mixtures with generic covariances), [2] (which relaxes the assumption on Gaussian data proving universality results on data distribution), [3] (which uses an heuristic theory to make sharp asymptotic predictions on popular test metrics under imbalance), [4], [5] … Even if an extensive literature review is out of scope here, some of these works have to be mentioned to address the claims of novelty of the paper, as they all derive results in the proportional asymptotics the paper considers.
- If I understand correctly, the main novel point of Theorem 2.1 is the proof of the asymptotic distribution of training/test logits. In [4], Eq. (9) (test logit) and (53) (train logit), and references from early literature therein, similar results are reported. Can the authors comment whether the statements are comparable?
- Can the authors report at least the main ideas of the proofs of their theorems in the main text? In its current form, the main text is substantially not self-contained, which is an issue given reviewers are not required (and mostly unable due to time constraints) to read the appendices. However, the illustration of the results in comparison with numerical simulations (Fig. 1,4,5) is commendable.

[1] Loureiro et al. Learning Gaussian Mixtures with Generalised Linear Models: Precise Asymptotics in High-dimensions, NeurIPS 2021

[2] Dandi et al. Universality laws for Gaussian mixtures in generalized linear models, NeurIPS 2023

[3] Loffredo et al. Restoring balance: principled under/oversampling of data for optimal classification, 

[4] Mannelli et al. Bias-inducing geometries: An exactly solvable data model with fairness implications, Phys. Rev. E 112, 025304

[5] Pezzicoli et al. Class Imbalance in Anomaly Detection: Learning from an Exactly Solvable Model, AIStats 2025

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
2

### Summary
The paper presents a statistical theory for why overfitting effects the minority class significantly more in high-dimensional class imbalance problems by observing the behaviour of the logits. The authors define empirical logit distribution (the distribution of logits for the training set) and test logit distribution (distribution of logits for the test set) and characterise overfitting using these measures. The paper also applies this to the field of calibration, and an extension to multiclass.

### Strengths
- The paper provides a theoretical understanding for the behaviour of the distribution of logits at training and test time, which is an interesting application of an approach often used in calibration.
- The theorems presented seem consistent with the general consensus among the imbalanced classification field.
- Results are presented on a variety of modalities (tabular, image and text)
- Experiments are performed in a good number of scenarios, various imbalance proportions $\pi$, optimising classifiers on the data itself and using features from a pretrained network.

### Weaknesses
- The paper is difficult to follow in places, there are a lot of concepts to grasp in section 2.
- Gordon's theorem should be cited at it first mention on page 2
- [1] presents a method for adjusting decision boundaries based on the uncertainty from drawing a random sample from each class, and could be discussed in section 3. I believe this method would help with the truncation of the TLD, and it could be worth discussing how concentration inequalities relate to the TLD and ELD. 
- Most of the theoreitcal results around calibration and prediction probabilities are in the appendix, which makes the section seem incomplete in the main paper. Section 5 is similar, in that most of the details are in the appendix. The paper could benefit from moving one of these sections to the appendix completely, and expanding on the other.

[1] Clifford, Matt, et al. "Learning Confidence Bounds for Classification with Imbalanced Data." ECAI 2024 (European Conference on Artificial Intelligence) (2024).

### Questions
- Does the truncation of logits still occur in datasets where the means for the positive class and negative class are far apart? i.e. the decision boundary will be placed arbitrarily between both classes. In practice, overfitting still occurs as most classifiers will favour placing the decision boundary close to the minority class, but there will be a set of decision boundaries which give the same training error. How can the TLD and ELD differ, and how does the theoretical approach in this paper deal with this?
- How do post-hoc methods such as rebalancing the margin address the calibration of the classifier in this scenario?

### Soundness
3

### Presentation
2

### Contribution
3
