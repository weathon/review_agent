# Slicing Mutual Information Generalization Bounds for Neural Networks

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
The ability of machine learning (ML) algorithms to generalize to unseen data has been studied through the lens of information theory, by bounding the generalization error with the input-output mutual information (MI), i.e. the MI between the training data and the learned hypothesis. These bounds have limited empirical use for modern ML applications (e.g., deep learning) since the evaluation of MI is difficult in high-dimensional settings. Motivated by recent reports of significant low-loss compressibility of neural networks, we study the generalization capacity of algorithms that slice the parameter space, i.e. train on a random lower-dimensional subspace. We derive information-theoretic bounds on generalization error in this regime and discuss an intriguing connection to the $k$-Sliced Mutual Information, an alternative measure of statistical dependence that scales well with dimension. We also propose a rate-distortion framework that allows generalization bounds to be obtained if the weights are simply close to the random subspace, and we propose a training procedure that exploits this flexibility. The computational and statistical benefits of our approach allow us to empirically estimate the input-output information of these neural networks and compute their information-theoretic generalization bounds, a task which was previously out of reach.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper establishes information-theoretic generalization bounds for learning algorithms trained on random subspaces. By assuming that the D-dimensional model weights w can be projected into a d-dimensional subspace w’ by w = Theta w’, the authors present tightened versions of generalization bounds in (Xu et al, 2017) and (Bu et al, 2019) by replacing w with w’. The authors further connect these results with sliced mutual information under some simple learning scenarios, and present rate-distortion bounds by incorporating the Lipschitz assumption.

### Strengths
1. The presented bounds effectively lower the dimensionality of random variables used in the key mutual information terms, improving the computational tractability of these bounds.
2. The established bounds successfully reflect the trend of the true generalization gap, and are shown to be tighter than previous competitors.

### Weaknesses
1. The target learning scenarios seems too restrictive for modern deep-learning models. The subspace mechanism severely restricts the available number of parameters in the neural network, making them hardly applicable in modern learning tasks. The Lipschitz condition is also hardly satisfied in conventional network architectures. As seen in Figure 7, the training accuracy cannot reach 80% on CIFAR10, even for a binary classification task.

2. Although the dimensionality is reduced in the presented bounds, this improvement hardly solves the tractability problem. The key mutual information I(W’;S) is still computationally intractable for greater d values, e.g. d > 20. For empirical analysis, the authors adopt a neural network-based mutual information estimator (MINE), whose accuracy lacks theoretical guarantees and is thus questionable for high-dimensional variables. The viability of applying MINE for theoretical analysis is doubtful.

3. The derivation of Theorem 3.1 and 3.2 seems trivial for me: given a fixed Theta, the trainable set of parameters then becomes W’ instead of W. Then Theorem 3.1 and 3.2 directly follows by Xu and Bu’s results by replacing I(W;S) with I(W’;S), or replacing I(W;Z_i) with I(W’;Z_i). The current contribution of the paper may not be sufficient to be published in ICLR.

### Questions
Please refer to weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors build upon "mutual information generalization bound" framework, and extend it by studying a setting with:
* parameters living in low-rank subspace, or close to it
* weight quantization
* Lipschitz losses, bounded weights  
  
They derive new bounds showing that, if the weight are optimized over a low-dimensional manifold, tighter bounds can be obtained than the one of literature. They also propose a new "rate distortion" bound that relies on the Lipschitz constant of the loss.  

Moreover, authors characterize the behavior of their bounds in the following cases:
* Gaussian mean estimation
* Linear regression  
  
Showing that their method effectively extends the work of Xu & Raginsky (2017), as well as Bu et al. (2019).

Finally, authors measure empirically their bound on high dimensional problem, e.g. training of a deep neural network on Mnist and Cifar-10. Since the bounds rely on mutual information, they use mutual information estimators like MINE to evaluate the bound.

### Strengths
### Originality

The work is original, improving upon previous work to take into account the dimension of the parameter manifold, and the importance of quantization.  

### Quality

Both toy models, and deep neural network training is studied. Insight can be useful for practionners and theoreticians. Experiments are convincing. 

### Clarity

The paper is clear overall, but not without defaults (see comments below). The literature review is very accessible. The authors do a good job at making the theorem A.2 more accessible with theorems 3.1 and 3.2.

### Significance

Mutual information bounds are useful, and can be evaluated empirically with MIN estimator. This work gives an additional motivation to study and optimize low-rank neural networks. Code is given.

### Weaknesses
### Clarity

There is a lack of clarity or details at times (see **Questions**).  

### Bias-variance tradeoff

>  On the other hand, decreasing d may increase the training error, implying a tradeoff between generalization error and training error when selecting d.  
  
and
  
>  The choice
of d is also important and can be tuned to balance the MI term with the distortion required (how
small λ needs to be) to achieve low training error.

Most of the bounds derived by the author can be understood as a bias-variance tradeoff, with somes terms that decrease with $n$ (the *variance*), and some other that do not decrease with $n$ (the *bias*). Thereore, it is a bit upsetting to not see a clear discussion of this. For example, in the rate-distortions bounds (theorems 3.3 and 3.4) the left term does not depend on $n$. What are the practical implications of this?

Maybe it is worth adding a discussion on this topic, and to underline the terms of your bounds with `\underbrace`.  

### Quantization

Theoretical results show a dependency of the bound on the number of levels in quantization. However, in none of the experiment the number of levels of quantization is kept constant in experiment of Fig 3. It would be interesting to measure the generation bounds when the number of level vary, compared to the (empirical) generalization error.

### Motivation behind the orthogonal projector

The motivation behind introducing $\Theta$ is unclear to me. It *seems* that sampling an element from the Grassmannian $\mathcal{M}$ would be enough: https://en.wikipedia.org/wiki/Grassmannian It seems that it is not crucial that $w=\Theta w'$ for the theoretical results to hold. What *seems* to matter is that $w$ is effectively low-rank, i.e. live on a low-dimensional manifold $\mathcal{M}$ which is sampled independantly from the data, such that $\mathbb{P}(\mathcal{M}|Z)=\mathbb{P}(\mathcal{M})$ so that expectation can be taken over $\mathbb{P}(\mathcal{M})$ in the bound. The fact that $\Theta$ is orthogonal does not seem to be necessary as long as $w$ is sampled from $\mathcal{M}$.

Is there a motivation behind using orthogonal matrices $\Theta$ ? Getting rid of it would allow to get rid of the `scipy.linalg.orth` in the experiments.  

### Experimental setup of sec 4.2 doesn't match theorems

The theorems assume that the train set $S_n$ is sampled from $\mu$. However, as written in appendix:

> To make generalization harder, we hide a class during training. Specifically, for MNIST, the training dataset contains digits {0, 1, 8} and the test dataset contains digits {0, 1, 8, 2}; for CIFAR-10, the training dataset contains classes {automobile, cat, deer}, while the test dataset contains classes {automobile, cat, deer, truck}.  
  
This makes the test set Out Of Distribution (OOD) compared to the test. It is not Empirical Risk Minimization (ERM) paradigm anymore, but rather evaluation in face of  distribution shift. Therefore, the generalization error reported cannot be compliant with the bound. For example, the marginal probabilities of the $y=1$ class falls from $\frac{1}{4}$ to $\frac{1}{5}$ for Mnist, and from $\frac{1}{3}$ to $\frac{1}{4}$ for Cifar-10. 

I am not convinced about the relevance of this setting. Sticking with ERM paradigm would be better.  

### Typo: arguments switched

* In page 1 the loss is defined as $\ell:Z\times W\rightarrow\mathbb{R}$, but everywhere else it is used as $\ell(w,z_i)$.

### Questions
### Bounded binary cross entropy

You write (p9):
> The role of $\epsilon>0$ is to make $\ell$ bounded thus meet the conditions of Theorems 3.3 and 3.4. 

I failed to be convinced. If no constraint is put on $f(w,X)$ then $f(w,X)+\epsilon$ can take any value in $(-\infty,+\infty)$ range. Unless $f$ is lower bounded. When reading the appendix (p21) that seems to be the case since $w_i$, $b_i$ and $x$ are bounded. Unfortunately, the bound on $f$ is not given, and the link with boundedness of $f$ is not explicitly drawn. Can you clarify?

### Unbounded MI

>  for instance if $W ′$ is a deterministic function of $S_n$ given $\Theta$ then $I_{\Theta}(W ′; S_n) = +\infty$, 

Since Mutual Information is bounded by entropy of variables involved: $I_{\Theta}(W ′; S_n)\leq H(S_n)$, do you mean that $H(S_n)$ is unbounbed?

### Appendix

Can you clarify how you went from step (39) to step (40) in appendix, p15 ? Idem for steps (42) to (43).

### Conclusion

I would be happy to raise my score upon satisfying answer to the questions and remarks.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents generalization bounds for a learning algorithm, utilizing an information-theoretic measure of dependence between the output of the learning algorithm and the training dataset. The primary motivation behind this work is to address the statistical challenges associated with estimating generic information-theoretic generalization bounds.

The paper introduces the concept of "slicing" the network's parameters and provides bounds for learning algorithms that focus on updating only a specific "slice" of parameters. The primary advantage of this approach lies in the ease of estimating mutual information-based bounds.

### Strengths
This paper attacks an important problem regarding obtaining an information-theoretic generalization bound that is easy to estimate. The idea based on the slicing seems very interesting and it gives rise to a connection to the sliced mutual information.

### Weaknesses
My main concern with the paper is that the main message of the paper is not clear. I think the idea of projection of parameters is interesting and it is intuitive that we can have a smaller generalization error. My understanding of the main message of this paper is that: slicing is interesting since we can have a better estimator of the information-theoretic generalization bounds. However, I think that only obtaining numerical values may not the only goal of the generalization theory.  I appreciate if the authors provide more discussion regarding the main message of this paper.

Also, regarding the idea of compressibility, it is not clear why a "data-independent" projection matrix will be able to find the best subspace.

### Questions
1- Motivation of this paper: Could you please provide a more detailed discussion on the motivation behind your paper, addressing the points raised in the weaknesses section?

2- Intrinsic Dimension: Throughout your paper, there is a recurring mention of the intrinsic dimension. Could you clarify what you mean by "intrinsic dimension"?

3- Comparison between your work and "functional" or "evaluated" MI bounds: In the existing literature, there are generalization bounds that do not suffer from the curse of dimensionality. It would be valuable to understand how your proposed bounds in this paper compare to the results presented in the following papers:

-- Harutyunyan, Hrayr, et al. "Information-theoretic generalization bounds for black-box learning algorithms." Advances in Neural Information Processing Systems 34 (2021): 24670-24682.

-- Haghifam, Mahdi, et al. "Understanding generalization via leave-one-out conditional mutual information." 2022 IEEE International Symposium on Information Theory (ISIT). IEEE, 2022.

4- Different Compression Methods: Your paper discusses modifying learning algorithms to improve information-theoretic generalization bounds. This approach reminds me of adding Gaussian noise to the output of a learning algorithm and trading off information with distortion, which has been explored in other works, such as:

-- Neyshabur, Behnam, Srinadh Bhojanapalli, and Nathan Srebro. "A pac-bayesian approach to spectrally-normalized margin bounds for neural networks." arXiv preprint arXiv:1707.09564 (2017).

-- Dziugaite, Gintare Karolina, and Daniel M. Roy. "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data." arXiv preprint arXiv:1703.11008 (2017).

Could you outline the main differences between your approach and the approach based on adding Gaussian noise?

5- PostHoc explanation of low dimensional compressibility: This paper is motivated by the observation that many neural networks are highly compressible. However, it seems that the method based on random projection might not capture low-dimensional compressibility adequately since the subspace might be data-dependent. Additionally, one limitation of your paper is that it's not clear how to choose the dimension for projection (d). Could you provide insights or potential solutions to address these concerns?

6- Connections to Pruning Methods:  Can you explain how your findings are connected to pruning techniques used in neural networks?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an information-theoretic generalization bound for algorithms trained on random subspaces. This approach is driven by the challenge of estimating the input-output mutual information (MI) generalization bound for modern neural networks, which is hindered by the high dimensionality of the parameters. Furthermore, the paper introduces a training procedure based on a rate-distortion framework that allows computation of the MI bound using less restrictive weights, all the while preserving performance.

The paper makes the following contributions: (1) The authors proposed a novel information-theoretic generalization bound for algorithms trained on random subspaces (Section 3.1). (2) The authors showed the connection between this bound and  k-SMI in two settings: Gaussian mean estimation and linear regression (Section 3.2). (3) The authors showed empirically that their bounds are tighter compared to the bounds in Bu et al. (2019) (Section 4.1). (4) The authors proposed a training algorithm that allows computation of the information-theoretic generalization bound on neural networks with less restricted weights. This is done by introducing a regularization method that encourages the weights to be close to the random subspace. They showed that this approach improves both the generalization bound and test performance (Section 4.2).

### Strengths
The paper explores tightening information-theoretic generalization bounds while introducing a technique to evaluate this bound in high dimensional settings. This is indeed an interesting and important topic in deep learning. The proposed training procedures based on the rate-distortion framework is also novel and interesting.

### Weaknesses
The sliced information-theoretic bound, while interesting, may be computationally expensive especially in high dimensions. The bound requires training multiple models for multiple projection matrices. The authors resolved this by using quantization that avoids the estimation of MI, but no comparison was made on how loose this bound is compared to the sliced MI bound. Overall, the bounds also become increasingly loose as the dimension increases. Also, more evaluation is needed for Section 4.2 to demonstrate the effectiveness of this approach.

### Questions
1. In the footnote 3, the authors mentioned that their bound can be shown theoretically to be tighter for the two settings (mean estimation and linear regression) using data processing inequality. Can this detailed discussion be included and made clearer in the main text? Also, can it be shown theoretically that the bound is tighter in a more general setting (e.g., with neural networks)?

1. Is it possible to conduct empirical evaluation of the linear regression generalization bound?

1. Is it possible to empirically evaluate how varying the number of projection matrices and number of samples of $(W’,Z_i)$ affects the bound?

1. For Section 4.1, the authors evaluate the bound using the quantized weights and bypass the estimation of MI by considering the upper bound on $I^{\Theta}(W’;S_n)$. Is it possible to conduct a similar experiment to evaluate how loose this upper bound is compared to the bound in Theorem 3.1?

1. For Section 4.2, is it possible to vary the generalization error by changing the number of samples (similar to Figure 2), and evaluate the rate-distortion bound accordingly? 

1. The writing and details provided in the paper could be improved to make the paper more readable and useful. For example: the paper contains a multitude of bounds but does not provide a unifying perspective. It would be good to have more discussion on the bounds and when to apply each one.

1. Overall, the bounds become increasingly loose as the dimension increases. This limits the value and applicability of the bounds in general. Is it possible to tighten the bounds and say something about converse bounds?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
