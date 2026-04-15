# Untrained Networks' Class Bias: A Theoretical Investigation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 3

## Abstract
The initial state of neural networks plays a central role in conditioning the subsequent training dynamics.
In the context of classification problems, we provide a theoretical analysis demonstrating that the structure of a neural network can condition the model to assign all predictions to the same class, even before the beginning of training, and in the absence of explicit biases. We show that the presence of this phenomenon, which we call "Initial Guessing Bias" (IGB), depends on architectural choices such as activation functions, max-pooling layers, and network depth.
Our analysis of IGB has practical consequences, in that it guides architecture selection and initialization. We also highlight theoretical consequences, such as the breakdown of node-permutation symmetry, the violation of self-averaging, the validity of some mean-field approximations, and the non-trivial differences arising with depth.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the distribution of untrained neural network outputs in a binary classification task. The paper considers Gaussian random inputs and Gaussian random weights in the neural network. By defining a phenomenon named ”Initial Guessing Bias” (IGB), the paper shows that while linear activation introduces no IGB, adding ReLU and Maxpooling will result in IGB. Moreover, IGB is intensified by increasing the number of layers in the neural network.

### Strengths
The notion of IGB introduced in this paper is new. Based on this notion, the paper has several interesting results:

1. The paper shows that, while linear activation gives output distribution centered at one-half (balanced label output), using the ReLU activation actually shifts the distribution away from one-half, even when the width approaches infinity.

2. The paper also shows that the randomness in the weight initialization dominates the randomness in the dataset when the depth of the network grows to infinity (as in Eq. 15).

### Weaknesses
1. The writing of the paper needs to be greatly improved:

    a). Since this is a theoretical paper, it is astonishing to me that there are no theorems stated in the paper. All the theoretical results in the paper are discussed verbally instead of rigorously in formal theorems. It is not clear under what hypothesis these results hold, and whether some equations are formal results or intuitive deductions. There is no mathematical formulation of the neural network considered in the paper so I am not sure what is the exact setting of the paper’s result.

    b). Terms used in the paper need to be described precisely. For instance, in the second paragraph on Page 5, the paper writes $\mathbb{P}_{\mathcal{X}}(O^{(c)}|\mathcal{W})$ is asymptotically a Gaussian whose center $\langle O^{(c)}\rangle$...”. I believe that in this case, the paper means that $O^{(c)}$ is a random variable following a Gaussian distribution, not that the probability is a random variable following a Gaussian distribution. 

    c). Another example is Eq. 9, where I am not sure what is $\mathcal{N}(y; \Delta_{\mu}(f_0), 2\sigma^2_{\infty})$, and I am not sure what the author means to integrate a distribution $\mathcal{N}(y; \Delta_{\mu}(f_0), 2\sigma^2_{\infty})$. Is the author somehow using $\mathcal{N}(y; \Delta_{\mu}(f_0), 2\sigma^2_{\infty})$ as the pdf function?

    d). Notations of the paper need to be introduced rigorously and mathematically. For instance, the only description I found about $f_c(\mathcal{W})$ is that it denotes ”the fraction of points classified as class $c$”. However, it is not clear whether this ”fraction” refers to the average over a finite number of data points or the expectation. The introduction of notations should be gathered somewhere instead of randomly popping up throughout the paper.

    e). In terms of formatting, the paper needs to add spacing between paragraphs for better readability.

2. It is not clear what is the significance of the topic (IGB) studied in this paper. The author tried to explain why it is important but provided no concrete examples and no previous works that directly consider the impact of IGB. How IGB is related to the following training process is also not described in the paper (other than an experiment in the appendix that I will raise concerns about below).

3. The assumption of applying max-pooling on random Gaussian input is weird since max-pooling is usually used to deal with images. In particular, if the input vector has no internal structure that allows dimensionality reduction tricks like max-pooling, it would not be so surprising that max-pooling will cause bad results such as IGB.

4. In Figure 11 the paper shows that CIFAR-10 has IGB on ResNet with ReLU and max-pooling. In Figure 15 the paper shows that with IGB the training process cannot achieve a recall higher than 0.7. Are these two results contradicting the well-known case that ResNet with max-pooling and ReLU works well (although not the best) on CIFAR-10? See e.g.: https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min

### Questions
Please see "Weaknesses" above.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
**Update after rebuttal:** I think the authors have made improvements to the paper that warrant an increase in my rating - I still do not think that the work is fully ready for ICLR, but I personally move away from a Reject to a Marginally Below. More detailed comments in my response to the authors' rebuttal.

The paper investigates the architectural factors which lead randomly initialized neural networks to have classification biases. A network is defined to have ‘initial guessing bias’ if the untrained network, across random initializations, favors one class over another in a binary classification task (with straightforward extension to the multi-class classification setting). The paper shows theoretically and empirically, how IGB arises in MLPs with common activation functions on synthetic data, and (in the appendix) on real-world benchmark datasets and architectures such as CNNs, ResNets, and VisionTransformers. The main takeaway is that the choice of activation function and use of MaxPooling cause IGB; deeper networks and use of real-world data amplify existing IGB of an architecture. Based on preliminary experiments in the appendix, the effects of IGB on training dynamics seem to be marginal and manifest mainly in the early training phase.

### Strengths
**Originality, Clarity:** To the best of my knowledge the phenomenon of IGB has not been investigated in detail before. The paper provides a definition of IGB and, somewhat counterintuitively, shows that commonly used neural network architectures and weight-initialization schemes lead to IGB. The (main) paper focuses on the theory behind IGB, and an empirical evaluation on synthetic data, which are both original. The paper presents the theoretical derivations in detail and with clarity, and the results look sound and correct to me. The appendix adds a lot of detail (to the point where it could perhaps use a bit of trimming). 

**Pros:**
 * Formal Definition of IGB via the fraction of data that gets assigned to one class across random initializations (without training).
 * Theoretical analysis of the conditions for IGB to occur (mainly the choice of activation function and whether to use max-pooling or not).
 * Empirical verification of the theory on synthetic data and MLPs - the results match the theory very well.
 * (In the appendix) Analysis of other commonly used architectures on standard benchmark datasets.

**Quality and Significance:**
The significance of the results is currently unclear. While the paper leaves no doubt about IGB as a phenomenon, it does not sufficiently address the question of whether the phenomenon has any substantial effect in practice. The appendix shows some simple results, suggesting that training dynamics for networks with IGB differ very early in training and with low learning rates. But since virtually all commonly used architectures suffer from IGB, and there is no indication that training with IGB leads to lower final performance or significantly slower convergence, the impact of the results currently seems rather limited. I would rate the quality of the paper as good, except for the main motivation of the criterion of IGB: it is unclear why having no IGB should be a desirable criterion for neural networks. A derivation from first principles, and/or a theoretical justification in terms of how IGB affects training dynamics (particularly final accuracy and convergence speed) would significantly strengthen the paper. See more details under Weaknesses.

### Weaknesses
Disclaimer: I have reviewed a previous version of the manuscript for another conference. While I am happy to see that the paper has been improved, some of the main concerns in terms of significance and quality have not been addressed.

**Cons:**
* Significance: show that IGB actually has a significant effect on training dynamics beyond the very initial phase. The first sentence of the abstract (and other parts of the paper) build the expectation that IGB will have a significant effect on training dynamics; but no such results are shown in the main paper. The effects on training dynamics are not shown until page 46 in the appendix, and the results shown suggest a rather marginal effect on very early training dynamics, but nothing to worry about in terms of final classifier performance or overall convergence speed. Confirming these intuitions would be a result in itself, and would add to the paper. Some theoretical results on how IGB impacts training dynamics would potentially lead to a very strong publication. Without a thorough empirical investigation of training dynamics (which is promised to be upcoming in another publication) I am not fully convinced that ICLR is the right venue, but would suggest a more specialized venue or workshop.
* Quality: The paper assumes that absence of IGB is naturally a desirable property of good neural architectures. This claim (though somewhat intuitive) is never substantiated or derived from first principles. I believe that the claim can be motivated and formalized in general terms from first principles via a maximum entropy argument, which would increase the soundness of the paper. Formally the argument is to find an architecture that maximizes $H(Y|W,X)$ where $y$ are network outputs, $x$ are network inputs and $w$ are network weights. Maximum entropy $H(Y|W,X)$ is achieved when $p(y|w,x)=1/N_c$, meaning uniform over all $N_c$ classes for each weight configuration (drawn from the initialization distribution) and each input. As a corollary, if the condition just stated is satisfied in classification, $1/N_c$ of the inputs are assigned to class $C$ (in expectation) across all weight configurations, meaning that the architecture has no IGB following Definition 3.1 in the paper. Note, however, that the max. ent. criterion strictly requires uniform output probabilities for all inputs and all weights; which is stricter than requiring uniform assignment of datapoints to each class (at least theoretically, the network could be very “sure” that half of the datapoints belong to class 0 and the other half belongs to class 1, but with different sets of datapoints for each weight-initialization; the max. ent. criterion requires class probability $1/2$ for all datapoints). Many of the derivations in the paper would look quite similar if IGB were formulated in terms of requiring max. entropy (at least in the discrete case, and probably also the Gaussian case should go through; in the general continuous case differential entropy when analyzing deterministic reversible maps diverges). Additionally, since $H(Y|W,X) \leq H(Y|X)$; and we maximize $H(Y|W,X)$ we have $H(Y|W, X)=H(Y|X)$ and $I(Y;W|X)=H(Y|X)-H(Y|W,X) = 0$. Meaning the channel capacity between weights and outputs for the initialized networks is zero; which could be an alternative motivating first principle for the maximum entropy criterion: the weights carry no information of any order (even beyond class-assignment information) about the outputs. To me, the maximum entropy principle is more general and better motivated from first principles compared to the IGB criterion in the paper. I am very happy to hear arguments in favor of sticking to requiring uniform class assignments instead of uniform output probabilities.
* Impact and claims: (this is a bit of a repetition of my first weakness, the emphasis here is to place training dynamics results into the main paper) I want to encourage the authors to show the impact on training dynamics in the main paper and perhaps move some of the derivations to the appendix. I understand that the goal now is to write a theory paper, but I think that the ICLR audience would greatly appreciate an analysis of the impact on training dynamics in the main paper. I would even want to encourage the authors to analyze final classifier performance and training convergence speed - even if the presence or absence of IGB does not affect the latter two, that would be an interesting result in itself and I would be very curious to see it reported.

**Final Verdict:**
I think the phenomenon of IGB is interesting and the main idea in the paper is original and results are surprising. However, I think the current paper can be significantly improved. I am leaning towards suggesting another revision of the manuscript - unfortunately ICLR only allows a rating of 3 (reject) or 5 (marginally below). On a free scale I would give the paper a 4, but since it is not possible I have assigned a 3 for now. I am curious to hear the authors response and other reviews, and am happy to change my score accordingly (and I do believe there's a good chance that my issues can be mostly addressed in the rebuttal).

### Questions
No further questions. Suggestions stated in Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the phenomenon of ‘Initial Guessing Bias’ (IGB), which means that the initial prediction of a NN is usually biased toward one single class rather than averaged across all classes, when the task is classification. The authors show that this is empirically true, and focuses on the theory behind it. In particular, they prove that for a 2-layer or a deep NN, the phenomenon provably happens even if the network is infinitely wide and the dataset is infinite. They argue that this phenomenon has some influence on the initial phase of training as well.

### Strengths
- They provide very detailed analysis of several settings, including 2-layer linear network, relu network, relu with max pooling, and deeper versions. 
- The fact that max pooling exacerbates the IGB phenomenon is very interesting.

### Weaknesses
- My main complaint about this work is that the problem doesn’t seem interesting. I don’t see why the initial bias matters, and the authors also fail to provide convincing evidence that this bias leads to meaningful damage to the trained result. In some sense, I believe this bias can be quickly corrected during the course of training.
- There could be more empirical evidence for why this bias matters and how it evolves during training. Right now the authors mainly focus on the theory.
- There’s no outline of the proof, making it hard to tell if the underlying analysis is trivial or not. It would be good if the authors can provide some sketch on the most interesting parts of the analysis.

### Questions
- In figure 1, when there’s no IGB, shouldn’t it be the case that all f_0 is centered around 0.5? It’s unclear to me why there’s still a distribution.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studied the initial state of neural network, which is an interesting question. They showed that the initial structure of neural network can condition the model to assign all predictions to the same class, which they called 'initial guessing bias (IGB)'. They also showed how architectural choices affect IGB.

However, I think there exist serious technique problems in this paper. I highly recommend the author to give more convincing proofs and more clear statements of their results.

### Strengths
The author proposed a very interesting phenomenon, IGB, in the initialization of neural network. Their results are insightful.

### Weaknesses
The authors' technical arguments and proofs are quite ambiguous, especially considering this is a theoretical paper.

1. On the use of limit symbols. For example, in equation 7&8, is $\mathcal{W}$ a deterministic, infinite array? If it is, under which conditions (of $\mathcal{W}$) equation 7&8 holds? Besides, I notice that in the appendix C, the authors say that the dimension of input variable $d$ must increase with $N_1$ in a larger order (equation 69&70) to obtain equation 7&8. I think it's necessary to write such an important asymptotic condition above the limit arrow in equation 7&8.
2. I think that the authors' arguments of the independence are very non-rigorous. In the Remark 1 in Appendix C, the authors try to argue $\{h_i^{(1)}\}_{i=1}^{N_1}$ are independent. This independence is indeed directly used later as the core of their analysis. However, their arguments about this independence are highly intuitive rather than a rigorous proof. How can "w.h.p." pairwise orthogonality (equation 64-68) implies asymptotic independence? Their arguments about the asymptotic behavior of $N_1$ and $d$ are also very causal. Does equation 69 insure that these random vectors are orthogonal? I think the authors should use some rigorous mathematical tools (such as random matrix theory) to give a convincing proof before using these arguments.
3. Besides, from their theoretical results I still don't understand where the phenomenon in Figure 1 arises. Even if the output is a somehow "wide" Gaussian, I don't know how it leads to Figure 1. 
4. Indeed, I have conducted a very toy experiment on a one-dimensional, two-layer, fixed-width relu MLP, and obtain similar figures like Figure 1. The only difference is my MLP includes also bias as its parameter. Therefore, the IGB could not be an asymptotic phenomenon and I hope to get some reasonable analysis.

### Questions
Please see weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
