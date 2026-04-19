# Plugin estimators for selective classification with out-of-distribution detection

- Decision: Accept (poster)
- Scores: 6, 8, 8

## Abstract
Real-world classifiers can benefit from the option of abstaining from predicting on samples where they have low confidence. Such abstention is particularly useful on samples which are close to the learned decision boundary, or which are outliers with respect to the training sample. These settings have been the subject of extensive but disjoint study in the selective classification (SC) and out-of-distribution (OOD) detection literature. Recent work on selective classification with OOD detection (SCOD) has argued for the unified study of these problems; however, the formal underpinnings of this problem are still nascent, and existing techniques are heuristic in nature. In this paper, we propose new plugin estimators for SCOD that are theoretically grounded, effective, and generalise existing approaches from the SC and OOD detection literature. In the course of our analysis, we formally explicate how naïve use of existing SC and OOD detection baselines may be inadequate for SCOD. We empirically demonstrate that our approaches yields competitive SC and OOD detection trade-offs compared to common baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors explore a novel setting, selective classification with OOD detection (SCOD), which combines two common settings in machine learning, selective classification (SC) and out-of-distribution (OOD) detection. This investigation hold promise, as recent studies have highlighted the propensity for challenging in-distribution (ID) samples to be mistakenly classified as OOD. The authors provide a theoretical foundation for understanding SCOD, and they introduce a method that adeptly integrates existing SC and OOD detection techniques. The experimental results affirm that the proposed method performs admirably in SCOD settings.

### Strengths
1. The integration of SC and OOD in the setting is intriguing, as it more closely resembles real-world scenarios.
2. This paper is well-presented, providing a detailed certification process.
3. The proposed method demonstrates superior results in both settings.

### Weaknesses
1. How to choose hyperparameters is crucial for this combined method, and it would be beneficial if the authors could provide more detail.
2. Evaluating performance on datasets that are exclusively OOD or SC is meaningful, as the proportion of each may vary across different scenarios.
3. The performance in the absence of OOD samples is not outstanding, with some results even falling below those of the baselines.

### Questions
1. How to determine the PI_in* through inspection of logged data?

### Soundness
2 fair

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
This work proposes a new view of the framework for selective classification and out-of-distribution (SCOD) detection into a unified classifier, rejector tuple problem. Their framework allows for plugging in any existing OOD similarity score to perform rejection under different scenarios of data access, e.g., only in-distribution or a mixture of unlabeled IND and OOD data. They also showcase a statistical formulation for the problem and derive the Bayes optimal solution from it. They provide an empirical evaluation of the popular image classification benchmarks.

### Strengths
An extensive theoretical formulation of the SCOD problem, with derivations for the optimal classifier rejector pair and an alternative surrogate loss.

The experimental suite is large, with experiments on both CIFAR and ImageNet datasets, comparing to previous SCOD methods (SIRC) and traditional OOD detection metrics, but not selective classification methods.

### Weaknesses
* No new insights are given on better estimating the prediction confidence or the probability ratio between in and out distributions.

* Since no guarantees can be drawn for either $s_{ood}$ or $s_{sc}$, (8) or lemma 4.1 does not inherit any further guarantees. 

* Loss-based results are only shown for the CIFAR benchmark, not ImageNet.

* The AUC-RC is compared against OOD methods but not against selective classification/misclassification detection methods such as [1], [2], [3], etc. Adding them to the benchmark would further strengthen the experiments.

**Notation**: sometimes the notation is a little bit confusing. $L$ is the number of classes, the loss function, and the Lagrangian. Also, $[\cdot]$ is a set, but {${\cdot}$} is also a set. 

**Typo**: the second term of the surrogate loss in Algorithm 1 differs from the one in (10) (sampling over (x,y) instead of x). Notation might become heavy, but maybe introducing the marginal could be helpful. In algorithm 1, line 5 is not a probability if $\hat{s}$ maps into $\mathbb{R}$.

References:

[1] SelectiveNet. Geifman, Y. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option." ICML 2019. /abs/1901.09192.\
[2] ConfidNet. Corbière et al. "Addressing Failure Prediction by Learning Model Confidence." NeurIPS 2019. /abs/1910.04851.\
[3] Doctor. Granes et. al. "DOCTOR: A Simple Method for Detecting Misclassification Errors." NeurIPS 2021. /abs/2106.02395.

### Questions
1. Could the authors show the steps to go from (3) to (4) to make $c_{in}$ and $c_{out}$ appear as in 4.3?
2. Could the excessive loss bound in Lemma 4.1 be rewritten by considering the estimation error of $P_{in}(y|x)$, $\pi^*_{in}$, and $\hat{s}_{ood}$?
3. The constraint considered in (3) takes into account the rejection rate on the test distribution. Why not consider the rejection rate only on the in-distribution like (1) and (2)?
4. Assumption (A2) in page 6 states that $P_{out}(x) = 0$ for $x$ in $S_{in}^{\*}$. How to guarantee this in practice? I.e., how to build $S_{in}^{\*}$? The proposed strategy in the footnote considers $(x,y)$ and not simply $x$. The way I see it is that it is impossible to obtain a strict $S^*_{in}$ without full knowledge on $P_{out}$.
5.  For CIFAR, the proposed SCOD learning in algorithm 1 does not seem to yield better results than training only on the CE loss and using heuristic scores to perform SCOD. Could the authors elaborate on potential limitations on why this is the case?
6. How does the inlier rejection option perform on this task compared to the proposed method and existing OOD detectors? Please check the references cited in the Weakness section.

### Soundness
3 good

### Presentation
3 good

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
The paper addresses the problem of selective classification with OOD detection (SCOD), where the goal is to learn a classifier, rejector pair that can abstain on “hard” in-distribution samples as well as OOD samples. The unification of the selective classification (SC) and OOD detection areas has recently been explored, but its formal underpinnings have not been fully developed. The paper presents a formal statistical analysis of SCOD and derives the Bayes-optimal solution for the classifier, rejector pair. This solution generalizes the Bayes-optimal solutions for the SC and OOD detection problems considered separately. Based on this solution, they propose plug-in estimators for optimally combining confidence scores for SC and density-ratio scores for OOD detection. This provides a principled way of combining existing scoring methods from the SC and OOD detection areas to address the SCOD problem. They address two settings, the first one being a black-box setting with only in-distribution training data, and the second one being a loss-based setting where one additionally has access to an unlabeled mixture of in-distribution and OOD data.

### Strengths
- Strong theoretical work that unifies the seemingly disparate literatures of selective classification and OOD detection. The proposed statistical formulation and Bayes optimal solution for the classifier and rejector is a general result that can guide the design of selective classifiers that can reject on both uncertain in-distribution (mis-classified) inputs and OOD inputs. Although the areas of selective classification and OOD detection have been independently studied well, a combined analysis of the problems in a principled setting has been lacking and somewhat heuristic. This paper addresses the gap. 

- The proposed plug-in estimators specify how to optimally combine existing confidence scores from the selective classification literature and density-ratio scores from the OOD detection literature. Therefore, it allows researchers in these areas to leverage, in a principled way, existing scoring methods for selective classification and OOD detection. The novelty of the paper does not lie in new approaches to estimate the confidence scores or the density-ratio scores, but rather in how to optimally combine them for the SCOD problem.

- They also propose a loss-based approach for learning the classifier and rejector by leveraging an unlabeled mixture of in-distribution and OOD data “in the wild” (similar to Katz-Samuels et al., 2022). 

- Overall, the paper was interesting and insightful to read. There is a lot of discussion and results in the appendices which could be useful to researchers in this area.

### Weaknesses
1. The experiments mainly focus on semantic OOD (or far OOD) inputs, where there is no intersection in the label space of the in-distribution and OOD. It is also important to consider covariate-shifted OOD, e.g. which are caused due to common corruptions, noise weather changes etc. Some results on the covariate-shifted OOD data would strengthen the paper.   

2. The method requires a few constants or hyper-parameters to be set. For instance, the cost of false negatives $c_{fn}$, the maximum rejection rate $b_{rej}$, the proportion of inlier and OOD data in the unlabeled set $S_{mix}$, the choice of training OOD data $P^{tr}_{out}$. The results in the main paper are for specific choices of these parameters (understably due to the page limit), but there is not much discussion or takeaways on how these parameters affect the performance. Some discussion on this would be useful. 

3. Minor: there is lack of clarity in some parts of the paper, which could be improved. Please see the Questions section. 

4. The code has not been made available and some implementation details are missing.

### Questions
### 1. Conditional probability?
In the formulation of selective classification, would it not be better to use the conditional probability of misclassification given the input is accepted, i.e. $P_{in}(y \neq h(x) \~|\~ r(x) = 0)$? Also, under the subsection `Evaluation metrics` on page 8, the joint risk is divided by the total number of accepted inputs, which seems to be consistent with the conditional probability.

### 2. On the Lagrangian
It is not clear to me how to arrive at the Lagrangian in Eqn (4) from the objective (3). Based on my simplification of the Lagrangian from the SCOD objective (3), I am getting a slightly different form than that in Eqn (4). Specifically, I get $c\_{in} = \lambda \pi^{\star}\_{in}$ and $c\_{out} = c\_{fn} - \lambda (1 - \pi^{\star}\_{in})$. Referring to Section 4.3, it seems like $c_{in}$ and $c_{out}$ specified here may have been swapped? 

Also, it seems to me the multiplier of the first term in Eqn (4) is $(1 \~-\~ c\_{out} \~-\~ ((1 - \pi^{\star}\_{in}) / \pi^{\star}\_{in}) \\, c\_{in})$, rather than $1 - c\_{in} - c\_{out}$. 

It is certainly possible I missed/messed something, but would appreciate some clarification. 

### 3. Need for strictly-inlier dataset
It seems to me that the strictly-inlier dataset $S^{\star}\_{in}$ is only needed for estimating the mixture proportion $\hat{\pi}\_{mix}$. It is not clear why the labeled inlier dataset $S_{in}$ (with the labels discarded) cannot be used in place of $S^{\star}\_{in}$ for estimating $\hat{\pi}_{mix}$. Any theoretical reason for this?

### 4. Estimation of $\pi^{\star}\_{in}$
Please clarify if $\pi^{\star}\_{in}$ used in the formulation for SCOD Eqn (3) can be estimated using $\hat{\pi}_{mix}$? Does it have to be estimated at test time as mentioned in Footnote 3 on page 7? From my understanding, the mixture dataset $S\_{mix}$ is collected “in the wild” during deployment, which should give a good idea of $\pi^{\star}\_{in}$ as well. 

### Suggestions for the Theorems/Lemmas
- Would help to restate the Lemma/Theorem statements in the appendix.
- Some comments/takeaways on Lemma 4.1 and Lemma 4.2 would be useful.
- In Lemma 3.1, I believe it should be: Let $(h^\star, r^\star)$ denote any minimiser of (3) (not (2)).
- In the proof of Lemma 3.1 (Appendix A), it seems like the reject class $\perp$ is allowed to be part of the classifier $h(x)$ output. However, the classifier is originally defined as $h : \mathcal{X} \mapsto [L]$. Please clarify this point. 
- Some pointers could be added to the proofs. For example, Eqn (13) on page 18 follows from Pinsker’s inequality, which is worth mentioning.
- Lemma 4.2: it should be $s^\star$ not $r^\star$.
- Lemma 4.2: it would be useful to clarify that $p_{\perp}(x)$ is an approximation for $P^\star(z = 1 \~|\~ x)$. The $\perp$ symbol is commonly used for rejection, whereas here it corresponds to the probability of accepting. 
- Lemma 4.2: for introducing coupling implicitly it should be $s(x) = u^T \Phi(x)$ (no dependency on $y’$ here).
- Typo in the first line of the Proof of Lemma 4.2 on page 17: it should be $s^\star(x) = \log(P^\star(z=1 | x) / P^\star(z=-1 | x))$.

### Rejector definition
Might be better to define the rejectors $r^\star(x)$ and $r_{BB}(x)$ in Eqn (5) and Eqn (8) directly using the indicator function $\mathbb{1}[\cdot]$.

### On the Algorithm
- Line 3 of Algorithm 1: should it be $\hat{f} : \mathcal{X} \mapsto \mathrm{R}^L$? That is, $\hat{f}$ predicts a logit for each of the $L$ classes.
- Line 7 of Algorithm 1: Can directly specify the final classifier $\hat{h}(x) = \arg\max_{y} \hat{f}_y(x)$ since it does not depend on Eqn (8). Can also specify that $s\_{sc}(x) = \max\_{y} \hat{f}_y(x)$ in Eqn (8).

### Other points
1. Under `Baselines` on page 8: reference for energy-based scorer should be (Liu et al., 2020b) and (Hendrycks & Gimpel, 2017) should also be cited for MSP.

2. In Table 2, why do methods such as MSP, MaxLogit, and Energy (which do not use OOD training data) have different performance under the two settings: $P^{tr}\_{out}=$ Random300K and $P^{tr}\_{out}=$ OpenImages? 

3. From Table 4, it seems that the performance of `Plug-in BB [L_1]` is consistently worse than `SIRC [L_1]`, despite the former (proposed) method being more principled. Please explain this discrepancy. 

4. I think it would be useful to provide results of this method using the Deep NN method (Sun et al., 2022), especially on the ImageNet dataset, since it seems like the grad-norm scorers are not providing good estimates of the density ratio.  

5. It might be worth citing the following paper which characterizes the Bayes-optimal detector for mis-classification detection.  
Doctor: A simple method for detecting misclassification errors, https://proceedings.neurips.cc/paper_files/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
