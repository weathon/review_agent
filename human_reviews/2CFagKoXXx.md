# High Dimensional Causal Inference with Variational Backdoor Adjustment

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
Backdoor adjustment is a technique in causal inference for estimating interventional quantities from purely observational data. In medical settings, backdoor adjustment can be used to control for confounding and isolate the effectiveness of a treatment. However, high dimensional treatments and confounders pose a series of potential pitfalls: tractability, identifiability, optimization. In this work, we take a generative modeling approach to backdoor adjustment for high dimensional treatments and confounders. We cast backdoor adjustment as an optimization problem in variational inference without reliance on proxy variables and hidden confounders. Empirically, our method is able to estimate interventional likelihood in a variety of high dimensional settings, including semi-synthetic X-ray medical data. To the best of our knowledge, this is the first application of backdoor adjustment in which all the relevant variables are high dimensional.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper estimates causal effects in the backdoor setting with high dimensional confounders and treatments. The authors propose using variational inference to efficiently estimate the causal effect with better approximation. The proposed approach considers the confounder as an observed variable. It also includes the optimization of three distributional components named encoder, decoder, and prior that are trained separately using observational data. Finally, the encoder is fine-tuned to get a better estimate of the causal effect/interventional distribution.

### Strengths
The derivations of this paper seem technically correct and intuitive. The authors discuss the issues of directly applying the backdoor adjustment formula and joint training. This should help the reader to understand the importance of this problem. The authors provided almost every detail about the experiments which will definitely help reproducing the results easily. The experiment section of this paper is quite rigorous. It contains one synthetic and two semi-synthetic experiments with high dimensional variables. The experiments help readers to understand the algorithm performance for high-dimensional data.

### Weaknesses
Here, I provide the weaknesses of this paper and some concerns.

[Section 2]
* The authors should discuss existing neural causal models [1][2][3] that can sample from identifiable interventional distributions. These methods are suitable for training on high-dimensional data and can sample from interventional distributions.

[Section 3.2]
* The authors mentioned, “sampled high dimensional Z will almost never give high probability to p(y | x, z) for a chosen Y and X.” I would request the authors to explain how this leads to a high variance issue.
* It should be more explicitly mentioned that the authors are using a causal sufficient system i.e., all variables [X, Z, Y] are observed.

[Section 3.3]
* The authors mentioned, “the joint training in equation 10 is not guaranteed to converge”. The authors should explain the reasons behind this in more detail. 

[Section 4.2]
* Figure 4 a,b, and c should be explained in more detail. It is a little unclear what figures 4 a, and b imply.
* The image qualities of the MNIST and X-ray experiment might be improved by using better neural architectures. If the motivation of this paper includes applying causality in vision with high dimensional images, the proposed method should produce the same quality images as the papers that claim to do intervention without utilizing causal approaches.

Major concerns:
* The authors utilize the backdoor adjustment to estimate the P(Y|do(X)) interventional likelihood. They propose their method for causal graphs that satisfy the backdoor criterion and consider all variables are high dimensional. To my knowledge there exist different neural causal model-based approaches that solve more generic version of this problem.  For example, for a causal sufficient system, i.e., when all variables are observed (same as the authors’ assumption), [1,2] can sample from P(Y|do(X)) for any causal graph. This includes the backdoor graph discussed in this paper as well. For example: The MNIST experiment results should be able to be reproduced with [1,2]. Moreover, [3] relaxes the causal sufficient assumption and can sample from any identifiable P(Y|do(X)) even if there exist unobserved variables in the causal graph. For example, in the front door causal graph: X->Z->Y, X<-U->Y, U is unobserved. [3] can sample from P(Y|do(X)) even if all X, Z, and Y are high-dimensional. These existing works make the novelty of this paper questionable. I would request the authors to distinguish the novelty of this paper compared to the mentioned works.



[1] Murat Kocaoglu, et al. Causalgan: Learning causal implicit generative models with adversarial training. 

[2] Nick Pawlowski, et al. Deep structural causal models for tractable counterfactual inference. 

[3] Kevin Xia et al The causal-neural connection: Expressiveness, learnability, and inference.

### Questions
[Section 3.2]
* How does equation 2,3,4 work for more complex graphs? For example, when there exist multiple variables in the backdoor adjustment set and multiple mediators between X and Y.

[Section 3.3]
* The authors suggested that the trained likelihood-based model components can be plugged as suggested by Equation 4. Can the authors discuss briefly how these components can be combined?
* Why will the separate optimization not lead to the tightest lower bound?
* The authors mentioned that the joint training from the beginning is not guaranteed to converge. What is the guarantee that performing fine-tuning after separate training will always converge for arbitrary complex SCM? ie., the finetuned encoder will better fit with prior p(z) and p(y|x,z) and will obtain a better lower bound in all cases.
* Can the author explain how the distribution over Z becomes untethered from the data?
* What is the role of fine-tuning in this paper besides improving the image quality?

[Section 4]
* What did the authors mean by “binary MNIST images”?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a variational formulation of the interventional likelihood (Eq. 4) under the backdoor adjustment. The variational lower bound in Eq. 4 depends on three conditional distributions, each of which is initially trained separately using an MLE. After that, the proposal distribution (or the encoder $q(.|x, y)$) is further trained (or fine-tuned in the terminology of the paper), to further optimize the lower bound.

### Strengths
The authors show an ELBO for the interventional log-likelihood $p(y|do(x))$  under a measured confounder set $Z$. The authors demonstrate a way to optimize the ELBO using a two-stage optimization procedure: in the first stage, the three models are trained separately using the MLE. In the second stage, the proposal distribution's parameters are optimized to increase the ELBO while holding the other two models fixed.
The experiments show that sampling from $q(z|x, y)$ leads to lower variance than sampling from $p(z)$ directly.

### Weaknesses
There are weaknesses in both the theoretical contributions as well as the experimental setup.

## Weaknesses in the theoretical results.

### Re ELBO in Eq. 4:

The main concern with the lower bound in Eq. 4 is when equality will hold, i.e., under what conditions is the ELBO equal to the interventional distribution in Eq. 2? In the subsequent paragraph, the authors say that "penalty incurred by encoder will be its KL divergence with ... $p(z)$". 
The is incorrect: the (Jensen) gap between Eq. 2 and Eq. 4 is *not* equal to KL(q(.|x, y) || p(z)). Did the authors mean something else when they use the word "penalty" here?

The gap is different from the standard variational Bayes formulation (where z is latent) where it is KL between q(z) and p(z|X). In this backdoor case, as the authors say, since $Z$ is observed, the evidence (or Eq. 2) itself depends on Z. So it is not clear to me what $q$ would have to be if equality holds in Eq. 4. Can equality ever hold to begin with?

As a related point, is the estimator obtained by solving Eq. 9 even consistent? Assuming that you had access to infinite data and assuming you could optimize the objective perfectly, will the estimator converge to the true interventional distribution (this should only be possible if the equality can hold in Eq. 4)?


### Re Proposition 1:
Prop. 1 is trying to say how jointly optimizing the objective fails. In the proof, the authors use the fact that if $E[p(y|x)] > E[p(y|do(x))]$, the model converges to p(y|x). I do not understand what $E[p(y|x)] > E[p(y|do(x))]$ has to do with the model converging to $p(y|x)$. The model should converge to whatever maximizes the ELBO (which may or may not be p(y|x)). So even if $E[p(y|x)] > E[p(y|do(x))]$, isn't it possible that the maximum of the objective is obtained somewhere else?


### How do you estimate the ATE?:

In causal inference, we usually care about estimating the ATE or conditional ATE (CATE) using the backdoor adjustment. How exactly do you estimate E[Y|do(x)] from the ELBO in Eq. 4? I can see how, for a given value of $x, y$, you can compute the ELBO in Eq. 4, but how do you compute the expectation E[Y|do(x)]?


### Re motivation:

The authors motivate this work by saying that in case of high-dimensional Z and X, directly sampling Z's leads to high-variance. However, for maximizing the ELBO, the authors need to train models of p(z) and p(z|x, y), both of which are difficult tasks in high-dimensional settings. 
One drawback is that difficulties in estimating p(z) for high-dimensional Z will itself lead to difficulties in optimizing the ELBO (and the errors should propagate)? This seems to not fix the issue of high-dimensional confounders. Can the authors comment on this?

By contrast, the standard doubly-robust AIPW estimators require modeling E[Y|z, x] and P(X|Z). In case of high-dimensional $Z$ and low-dimensional $X$ (e.g., binary treatments), why would the variational approach be better? It is not even clear (and neither do the experiments provide evidence for this) that the variational approach suggested by the authors works better than just estimating the ATE using \hat{E}[Y|z, x], i.e., just training a model to predict Y from (x, z). 
 
## Weaknesses in the experimental results.


The main concern with the experimental results in the lack of any baselines. The approaches should at least be compared to IPW, AIPW, and standard regression estimators (like modeling E[Y|x, z] using a ML predictor).

The applies to the MNIST and chest X-ray experiments as well. For the MNIST, if you directly trained a model to predict the concatened output image Y from (z, x), does the variational model do better than that? Without such baselines, it is difficult to understand how well your proposed model is doing.

Moreover, other works, e.g., Shi et. al., have also used neural networks for estimating interventional distributions in the backdoor case. The authors should also compare to that.

### Questions
I have addressed the questions in the weaknesses section.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the challenge of understanding causal relationships in high-dimensional datasets. The paper introduces "Variational Backdoor Adjustment (VBA)," a framework consists of three distributional components: encoder, decoder, and prior. While the terminology may resemble that of Variational Autoencoders (VAEs), the underlying assumptions are different. These components are optimized separately to ensure correct backdoor adjustment over the observed distributions. The paper provides empirical evidence of the effectiveness of VBA in computing backdoor adjustments. It uses synthetic and semi-synthetic datasets, including high-dimensional linear Gaussian data and image datasets, demonstrating improved interventional density estimation through their optimization technique.

### Strengths
- The introduction of Variational Backdoor Adjustment (VBA) presents an innovative solution for handling high-dimensional datasets, offering a fresh perspective on causal inference.

- The paper provides empirical evidence of the effectiveness of VBA in computing backdoor adjustments, including synthetic and real-world data scenarios. This empirical support strengthens the paper's credibility.

- The paper appears well-structured and clear in presenting the methodology and its empirical validation, making it accessible to a broad audience.

### Weaknesses
1/ There is no comparison with any baseline making it hard to justify for the perforance of the proposed method. I understand that the proposed method is for high dimensional data. But it should be comparable with some baselines. Is it possible to compare with some popular methods such as BART, X-learner, R-learner, CFRnet, TARnet, etc?


2/ It is unclear on how do you calculate causal effect after training the model. Do you use Eq. (4) as an approximation for log p(y | do(x))?

### Questions
Please see section Weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
