# Benign Overfitting in Two-Layer ReLU Convolutional Neural Networks for XOR Data

- Decision: Reject
- Scores: 6, 6, 5, 6, 5

## Abstract
Modern deep learning models are usually highly over-parameterized so that they can overfit the training data. Surprisingly, such overfitting neural networks can usually still achieve high  prediction accuracy. To study this ``benign overfitting'' phenomenon, a line of recent works has theoretically studied the learning of linear models and two-layer neural networks. However, most of these analyses are still limited to the very simple learning problems where the Bayes-optimal classifier is linear. In this work, we investigate a class of XOR-type classification tasks with label-flipping noises. We show that, under a certain condition on the sample complexity and signal-to-noise ratio, an over-parameterized ReLU CNN trained by gradient descent can achieve near Bayes-optimal accuracy. Moreover, we also establish a matching lower bound result showing that when the previous condition is not satisfied, the prediction accuracy of the obtained CNN is an absolute constant away from the Bayes-optimal rate. Our result demonstrates that CNNs have a remarkable capacity to efficiently learn XOR problems, even in the presence of highly correlated features.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors study the learning of a 2-layer CNN on a XOR-type dataset, and evidence under certain conditions the onset of benign overfitting when the number of samples is large enough. Complementarily, they provide a related lower bound establishing that below this order of magnitude for the number of samples, the test error is at least a constant away from the Bayes-optimal test error. The main novelty of the work is its data distribution, for which the Bayes-optimal classifier is not given by a linear method.

### Strengths
This works extends previous studies of benign overfitting to a data distribution which is not amenable to being learnt by a linear classifier, and thus better justifies the use of neural network learners. The paper is clear and well organised, and sufficient discussion of related works is provided. The experiment section at the end is a good illustration of the theory. I have not read the proof, and cannot judge of its technical novelty. Overall, I feel that this is a solid paper, which makes an interesting addition to the settings where benign overfitting occurs, and am in overall favour of its acceptance.

### Weaknesses
Minor:
- It would be good to cite Hu \& Lu, 2022 and Xiao et al., 2022 alongside Misiakiewicz (2022); Xiao & Pennington (2022) for completeness when discussing multiple descents in kernel ridge regression in the related works.
- More discussions would be helpful regarding the linearity of the Bayes-optimal classifier in previous studies. Was this assumption in some way instrumental in the derivation of these results? Is the phenomenology different in the present work, or is the point mainly to consider a more complex data distribution, and exhibit another setting where benign overfitting occurs? Currently, I feel this point is insufficiently motivated.

### Questions
- In condition 3.1, is a max or min missing in the $\Omega$ notation?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work studies benign overfitting of Convolutional neural networks trained on non-linear data. The analysis focuses on a generalization of the XOR problem, where the data is centered around 4 possible vectors, giving the same label to vectors in opposite direction. This problem is non-linear in the sense that no linear classifier can approximate the target. In this setting, the authors analyze the conditions under which benign overfitting occurs. They show that benign overfitting depends on the noise level and the angle between the vectors, analogously to similar results on benign overfitting on linear data.

### Strengths
The study of benign overfitting in the context of non-linear data is important, and the paper seems to make novel contributions to this field of study. Most works studying benign overfitting so far focused on linear data, which arguably doesn't capture the full complexity of neural networks trained on real data.

### Weaknesses
1. The paper studies two-layer CNNs, which is a valid and interesting model to study. However, to my understanding the analysis is for networks convolving over only 2 input patches, which seems very restrictive. In this case, it is not clear what is the purpose of studying CNNs instead of MLPs. The authors should explain why they made this choice of architecture? Do similar results hold for standard MLPs, and if so, why do they choose to focus on 2-patch CNNs? Can the results be extended to CNNs with larger number of patches?

2. Some details about the choice of input distribution are not clear. Specifically, in the choice the randomly generated patch, why is the Gaussian covariance matrix chosen to be dependent on the "true" XOR vectors? Can you justify why such choice of random vectors is natural? Would a similar result hold for standard Normal distribution (or other more natural distributions)?

3. In stating the conditions for the theoretical analysis (Condition 3.1), it is better to separate conditions on the learning problem (which are essentially assumptions about the "world"), from choices of learning parameters (learning rate, initialization) and sample complexity. The way the conditions are stated, it seems that e.g. the dimension should grow with the sample complexity. It makes more sense to first assume a problem is given with some fixed dimension, and then adapt the sample size to the problem (in which case, does this mean that the sample size needs to be upper bounded?).

4. The authors state the necessary and sufficient conditions for benign overfitting to be $n ||\mu||_2^4=\Omega(\sigma_p^4d)$. Can you interpret this results? In what regimes of parameters does this hold? Is this always compatible with Condition 3.1?

Minor:
- First sentence in the introduction: highly parameterized => over-parameterized
- "To solve the chaos caused by the high correlation in the training process" - not sure "chaos" is the right term..
- Page 5: "These results also ensure that" - I think you mean conditions

### Questions
See above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies benign overfitting on neural networks, using gradient descent to train XOR-type data on a two-layer ReLU CNN. The authors establish the conditions for when benign overfitting and harmful overfitting occurs in this setting.

### Strengths
Compared to previous work, this paper goes beyond the scope of simple learning problems where the Bayes-optimal classifier is linear.

### Weaknesses
1: Compared to the results in the Classic XOR regime (Section 3.1), the results in Section 3.2 are of greater interest. However, this part of the results relies on strong assumptions. Firstly, the first point of Condition 3.3 is a stronger high-dimensional assumption compared with other benign overfitting papers such as [1]. Then, considering Condition 3.3, we can derive the conditions in the second point of Theorem 3.4 as $\|\| \mu \|\|_2^2 \geq \frac{m^5 n^2 \sigma_p^2}{(1-\cos \theta)^2}$ which is much stronger than [1].

2: I do not agree with the author's claim that they have discovered a sharp phase transition between benign overfitting and harmful overfitting, as in the case of Theorem 3.2, the transition between the second and third points may not necessarily be a phase transition but rather a continuous change.

### Questions
Could the author discuss the relationship and differences between this work and [2]? And is there any connection between this work and the Grokking?

[1] Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural network classifiers trained by gradient descent for noisy linear data. In Conference on Learning Theory, pages 2668–2703. PMLR, 2022.

[2] Zhiwei Xu, Yutong Wang, Spencer Frei, Gal Vardi, and Wei Hu. Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data.  arxiv 2310.02541.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the benign overfitting phenomenon of two-layer ReLU CNNs in an XOR problem and provides the generalization (test) error of the learned CNNs via gradient descent.

### Strengths
This paper demonstrates that CNNs can efficiently learn XOR-type data. Most importantly, this paper provides the test error bound for the learned model in the regime where the features in XOR-type data are highly correlated.

### Weaknesses
1. It is unclear how the test error changes as the number of samples, and CNN width change.

2. As the paper studies the benign overfitting of neural networks, a figure of test accuracy versus the neural network width $m$ should be included.

### Questions
Do we have any requirements on the number difference between two different data points? I am not clear about the high-level understanding of Lemma 4.1. If the data points are highly unbalanced, how do you obtain a result like Lemma 4.1?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers learning an XOR-type problem with an overparameterized two-layer neural convolutional ReLU network using the batch loss, where the input label can flip with a constant probability. This paper proves the convergence w.r.t. the empirical loss, and shows that there exists a certain condition on the sample complexity and signal-to-noise ratio which distinguishes whether near Bayes-optimal test error is achieved or the prediction accuracy is a constant away from the Bayes-optimal rate.

### Strengths
### The problem is well-motivated from both benign overfitting and feature learning literature.

I understand that proving benign overfitting for neural networks involves several difficulties due to nonlinearity of activation functions. On the other hand although recent works have shown the possibility of learning nonlinear features especially XOR data via gradient-based training, I am not aware of a result that the network can memorize label flipping noise of the training data as well. Therefore, this paper is a naturally motivated by both lines of researches.

### Solid technical contribution under reasonable assumptions and worth sharing to the theory community.

I need to note that the data distribution is not a standard XOR setting considered in pervious feature learning analyses ([Ji & Telgarsky (2020)](https://arxiv.org/abs/1909.12292); [Barak et al. (2022)](https://arxiv.org/abs/2207.08799); [Telgarsky (2023)](https://arxiv.org/pdf/2208.02789.pdf)), and requires larger signal-to-noise ratio (signal $\|\mu\|$ or noise $\sigma_p^{-1}$ should be large.). Except for that points, several technical assumptions are rather acceptable and the main result still has a solid contributions to the theory community as the first nonlinear Bayes optimal classifier.

### A sharp transition from sub-optimal classifier to the near Bayes-optimal classifier is interesting.

This paper derives the threshold where benign overfitting happens, analogous to the previous linear classifiers.

### Weaknesses
### Large signal-to-noise ratio

In p.5, the authors claim that
> However, we note that our data model with $\|\mu\|_2 = \sigma_p = \Theta(1)$ would represent the same signal strength and noise level as in the 2-sparse parity problem, and in this case, our result implies that two-layer ReLU CNNs can achieve near Bayes-optimal test error with $n=\omega(d)$, which is better than the necessary condition $n=\omega(d^2)$ for kernel methods.

However, as far as I understand, this explanation is wrong; by letting $\|\mu\|_2 = \sigma_p = \Theta(1)$, Condition 3.1 implies $d=\Omega(n^2)$ is required, while benign overfitting (in Theorem 3.2) requires $n =\Omega(d)$. To solve this issue, either $\|\mu\|_2$ or $\sigma_p$ should be larger than $O(1)$, in the asymptotic limit when $d$ and $n$ diverge. In other words, the signal-to-noise ratio should be larger than the standard analysis on the XOR data ([Ji & Telgarsky (2020)](https://arxiv.org/abs/1909.12292); [Barak et al. (2022)](https://arxiv.org/abs/2207.08799); [Telgarsky (2023)](https://arxiv.org/pdf/2208.02789.pdf)), although [Ba et al. (2023)](https://openreview.net/forum?id=HlIAoCHDWW) considers the anisotropic input data in the regression problem.

### Dependency on the initialization scale.

Suppose that we take $d=\Theta(n^2)$ and $\sigma_p=\Theta(1)$ in Condition 3.1. Then, the initialization scale is $d^{-0.75}$, which is relatively small. This means that the output scale of the network at initialization is $o(1)$.

### Motivation for considering the small angle case

I am basically satisfied with the first result (Theorem 3.2), but I do not see necessity to extend the problem into more general case, where two axes of the input XOR distribution is much closer.

### Questions
- I want to know how the small initialization (as I pointed out) is required in the analysis.

- In the proof sketch, the authors introduced "loss derivative comparison" technique. If the authors could provide more explanations about the motivation of such technique, I would appreciate it.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
