# Understand Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics

- Decision: Reject
- Scores: 5, 6, 6, 5, 5

## Abstract
Similar to surprising performance in the standard deep learning, deep nets trained by adversarial training also generalize well for unseen clean data (natural data). However, despite adversarial training can achieve low robust training error, there exists a significant robust generalization gap. We call this phenomenon the Clean Generalization and Robust Overfitting (CGRO). In this work, we study the CGRO phenomenon in adversarial training from two views: representation complexity and training dynamics. Specifically, we consider a binary classification setting with $N$ separated training data points. First, we prove that, based on the assumption that we assume there is $\operatorname{poly}(D)$-size clean classifier (where $D$ is the data dimension), ReLU net with only $O(N D)$ extra parameters is able to leverages robust memorization to achieve the CGRO, while robust classifier still requires exponential representation complexity in worst case. Next, we focus on a structured-data case to analyze training dynamics, where we train a two-layer convolutional network with $O(N D)$ width against adversarial perturbation. We then show that a three-stage phase transition occurs during learning process and the network provably converges to robust memorization regime, which thereby results in the CGRO. Besides, we also empirically verify our theoretical analysis by experiments in real-image recognition datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper investigates the CGRO phenomenon observed in adversarial training (AT)—specifically, that a model trained by AT achieves good clean generalization (CG) while simultaneously encountering robust overfitting (RO). The paper offers a theoretical explanation for this phenomenon from two perspectives: the model’s representation complexity and the training dynamics of AT.

From the representation complexity perspective, the paper considers a binary classification problem where a classifier that achieves only CG exists. A CGRO classifier can then be constructed based on this "CG-only" classifier, requiring slightly more parameters, which implies that achieving CGRO requires a mildly higher representation complexity.

Regarding AT’s training dynamics, the paper considers a structured data setting where data consists of one patch with a true feature and other patches with spurious features. Under additional assumptions, the model trained by AT is shown to partially learn true features while memorizing spurious features, suggesting that such a model is a CGRO classifier.

### Strengths
The paper explores an interesting phenomenon arising in AT and offers two theoretical explanations. The paper is well-written, with clear descriptions of the problem setup, definitions, assumptions, and proof sketches that effectively convey the intuition behind the theory.

### Weaknesses
There are several technical issues that I would like to discuss:

- **Lemma 4.5**:  the lemma constructs a classifer $f _ {\cal S}$ based on $f_{\rm clean}$ and claim that $f_{\cal S}$ is a CGRO classifier. However,  although by construction $f_{\cal S}$ memorizes the training data ${\cal S}$, this does not necessary imply that $f_{\cal S}$ will exhibit robust overfitting. The generalization performance of $f_{\cal S}$ depends on the choice of ${\cal S}$. If ${\cal S}$ nicely covers the support of the data distribution ${\cal D}$, that is given ${\cal S}$ we have $X\in \cup_ {i=1} ^{N}{\mathbb B} _ {p}(X_{i}, \delta)$ with high probability, then $f_{\cal S}$,  despite memorizing ${\cal S}$, is still able to achieve a good robust testing accuracy. In this case,  $f_{\cal S}$ is not CGRO.  Could you specify conditions on ${\cal D}$ under which $f _ {\cal S}$ would exhibit robust overfitting with high probability over the randomness of drawing ${\cal S}$ from ${\cal D}$? Could you also provide theoretical analysis towards how the sample size $N$ as well as $\delta$ and $p$ affect ${\cal L} _ {\cal D}^{p,\delta}(f _ {\cal S})$ ?
- **Section 5.1, line 325**:  the definition of ${\mathbb A} _ {p}(X, \delta)$ is unclear. It seems that for any  $X'\in{\mathbb A} _ {p}(X, \delta)$ other than $X$, we have $X'[{\rm signal}(X)]=q w^{*}$ for some $q\in{\mathbb R}$ and $X'[j]=0$ for $j\ne {\rm signal}(X)$ . Is this correct? The construction of ${\mathbb A} _ {p}(X, \delta)$ is important for understanding the results in Theorem 5.9. I hope the authors could clarify the definition of ${\mathbb A} _ {p}(X, \delta)$ . 
-  **Theorem 5.9** : Theorem 5.9 attributes CGRO to the model trained by AT simultaneously learning the true feature and memorizing the spurious feature. Under the settings in Section 5, is it possible to show that a model trained via standard (non-adversarial) training would only learn the true feature and ignore the spurious feature? Establishing this distinction is important, as it would suggest that CGRO arises from the nature of AT itself rather than from the artificial construction of the data model.



I recommend the authors address these concerns. I'm willing to engage in a discussion with the authors and am flexible about adjusting my score.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper analyses the phenomenon of clean generalization and robust overfitting (CGRO) from two perspectives: representation complexity and training dynamics. They first show that achieving a robust classifier on test distribution requires many more parameters than achieving a classifier with CGRO. Then, they further investigate the optimization dynamics of adversarial training and show that the weights of the classifier will finally move to the CGRO regime. Their theory about training dynamics is based on a specific data structure: Patch Data Distribution. Some experimental results are provided to validate the CGRO phenomenon and the three-stage dynamics.

### Strengths
This paper provides two theoretical perspectives to understand the CGRO phenomenon. The theory is solid in general although it needs some assumptions of the data structure. The writing is clear and they provide detailed descriptions of the notions, setup, and related background.

### Weaknesses
* The model representation perspective shows that the models need more parameters to achieve robust generalization than CGRO. But how do we validate that the number of parameters of the deep networks such as ResNet is in this CGRO range?
* The data structure is not very realistic. I think a more realistic assumption would be the data lies in a low-dimensional subspace or manifold. The assumption that only a pixel-space patch has useful signals would be too tough in my opinion.
* Although the authors validate the three stages of training dynamics in the synthetic structure data, they don't validate it on real data. Could you provide some experimental results to show it?

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the Clean Generalization and Robust Overfitting (CGRO) phenomenon in adversarial training from two theoretical perspectives: representation complexity and training dynamics. Using ReLU networks, they show that CGRO classifiers require only poly(D)+ND parameters while robust classifiers need exponential complexity. They also analyze training dynamics using a two-layer CNN on structured data, showing how the network converges to a state of partial true feature learning and exact spurious feature memorization.

### Strengths
The analysis is particularly noteworthy for its comprehensive approach - it not only establishes complexity bounds showing that CGRO classifiers require only poly(D)+ND parameters while robust classifiers need exponential complexity, but also provides insights into how this phenomenon emerges during training through a clear three-stage analysis of the training dynamics. The theoretical framework is well-supported by experimental validation on both real and synthetic datasets, demonstrating practical relevance. Most importantly, the paper's findings help explain a commonly observed phenomenon in adversarial training, offering valuable insights into why models often achieve CGRO in practice. The technical depth combined with practical significance makes this work a meaningful contribution to understanding adversarial robustness.

### Weaknesses
I have two major concern regarding the representation complexity and training dynamics, respectively.

**Representation Complexity:**

The central result at Line 255 states:

Clean Classifier (poly(D)) ≲ CGRO Classifier(poly(D)+ND) ≪ Robust Classifier (Ω(exp(D)))

However, the poly(D) complexity of clean classifiers is simply assumed in Assumption 4.3 rather than proven. This is problematic because if clean classification also requires exp(D) parameters, the entire complexity hierarchy becomes meaningless. The authors need to either:

1) Prove that clean classifiers can indeed be approximated by neural networks with poly(D) parameters, even though it is trivial.

2) Cite existing results that establish this fact, even though this is a well-known classical result.

I assume this is only a **clarity** issue, because the authors think this is too trivial or too classical and not worth discussing? 

**Training Dynamics:**

While the analysis clearly shows that training leads to partial true feature learning and exact spurious feature memorization, there is a gap between this result and CGRO that needs to be addressed. Specifically, the paper should analyze:

1) For an unseen samples, what is the probability that the model will correctly classify it? Need analysis of the condition $α^q U + V ≥ 0$ for a new sample.

2) For adversarial examples generated from this unseen sample, what is the probability of misclassification? Need analysis of the condition $α^q U + V < 0$ for adversarial example.

These probability analyses would help complete the connection between the training dynamics results and the CGRO phenomenon.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies the robust overfitting phenomenon in adversarial training, where despite the model's generalization on clean data the robust accuracy differs significantly between training and test samples. The paper aims to show the existence of classifiers with clean generalization and robust overfitting (CGRO) and section 4 includes Theorem 4.4 with a polynomial upper-bound for CGRO classifiers and Theorem 4.7 on the exponentially growing required number of ReLU network parameters to achieve robust generalization. Next, the paper studies a structured-data setting and analyzes the training dynamics of a convolutional network, where the network is shown to converge to robust memorization. A few numerical results on MNIST and CIFAR10 are presented in Section 6.

### Strengths
1- The paper studies a relevant research problem on robust overfitting in adversarial training. The paper presents several theoretical results on the phenomenon and the possibility of robust overfitting and clean generalization in adversarial learning.

### Weaknesses
1- The paper's presentation of the theoretical results can be significantly improved. One confusion I have about the theory discussion in the paper is the order-related notations and their meaning in the results. Beginning in Definition 3.4, the authors clarify in Remark 3.5 that the notations are with respect to data dimension $D$. However, I could not understand how the authors treat the data dimension $D$ as a parameter that they can change and perform asymptotic analysis for. Isn't it the case that the data dimension $D$ is fixed and the asymptotic analysis should focus on the model complexity (number of parameters) or the sample size for asymptotic analysis? 

Continuing about my confusion with the asymptotic analysis on the data dimension, let me refer to Theorem 4.4 where the authors make Assumptions 4.1 -4.3. Looking at these assumptions, they seem quite dependent on the dimension of $X$. For example, if one changes the input dimension, should not the constants $R$ (Assumption 4.1) and $\delta$ (Assumption 4.2) be updated as the dimension $D$ changes? Also, in the statement of Theorem 4.4, how can the dimension $D$ change while the theorem sims to fix the input space $mathcal{X}$? If so, then the function $f_{CGRO}$ will change with dimension $D$ and it is not a fixed function to perform CGRP analysis for.

As I discussed above, the authors' asymptotic analysis in terms of the input dimension seems quite confusing, because it changes the input $X$, the classifier architecture $f$, and also affects the assumptions altogether. I may have missed some details about the authors' analysis and will look forward to discussing the analysis in the discussion period.

2- The numerical results seem rather insufficient. I understand that this is a theory work, but the results in Figure 2 and Table 1 does not support the authors' claim. First of all, the input dimension does not change when the authors vary the model size, therefore the experiments do not analyze the asymptotic scenario when $D$ grows. Also, the improvement in robust test accuracy seems to follow from the improvement in clean test accuracy as the model keeps growing in capacity and hence can achieve higher test accuracy on both clean and perturbed data.

3- Minor comment: In my opinion, copying a figure from another paper (Figure 1 in the introduction) is inappropriate in a top-level machine learning paper. It would have been better if the authors' replicated the results on a different dataset or architecture and while giving credit to the cited paper, they presented their own computed version of the figure in the submission.

### Questions
1- I explained my confusions with the authors' asymptotic analysis in data dimension $D$. Can the authors clarify on the questions I raised on the analysis?

2- Can the authors provide more numerical results where the clean test accuracy remains stable while adversarial test accuracy improves where the model complexity grows?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper investigates the phenomenon of clean generalization and robust overfitting (CGRO) within the context of adversarial training. Initially, the authors demonstrate that, from the standpoint of model expressive power, achieving a CGRO classifier is significantly easier compared to robust generalization, which, in the worst-case scenario, necessitates exponentially large model capacity. Subsequently, the paper examines the training dynamics of adversarial training by considering a patch data distribution, employing a three-stage analysis technique to explore the feature learning process involved in adversarial training. Finally, numerical experiments are conducted to validate the theoretical insights presented in the study.

### Strengths
1. The paper is well-structured and easy to follow.

2. The three-stage analysis of the training dynamics of adversarial training, based on the assumption of a patch data distribution, is both intriguing and innovative.

3. The theoretical findings are well-supported by the numerical experiments, demonstrating a strong correlation between the two.

### Weaknesses
1. The paper falls short in clarifying how its results contribute to understanding why CGRO occurs during adversarial training. The paper raises this broad question in the introduction but lacks a detailed explanation of how its findings can offer insights into addressing it.

2. From the perspective of representation complexity to illustrate the CGRO phenomenon, the main theorem relies on Assumption 4.3, which posits the existence of a clean ReLU network classifier of polynomial size. However, this assertion pertains to a factual claim and cannot be directly treated as an assumption. Instead, it would be more appropriate to formulate it as a theorem accompanied by a rigorous mathematical proof. This would strengthen the foundation of the theoretical results presented in the paper.

3. From the perspective of training dynamics to elucidate the CGRO phenomenon, the analysis depends on the assumption that the perturbation radius 𝛿 is only marginally smaller than the signal norm 𝛼. The theoretical results are derived specifically under this condition for the perturbation radius. However, in practical scenarios, the perturbation radius can be chosen arbitrarily. The paper does not adequately address whether the same theoretical results can be achieved in this more generalized context, which leaves an important gap in understanding the robustness of the findings. Further exploration of this aspect would enhance the comprehensiveness of the study.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
2
