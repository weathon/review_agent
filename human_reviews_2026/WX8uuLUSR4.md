# Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization

- Decision: Reject
- Scores: 6, 6, 4, 0, 2

## Abstract
A widely believed explanation for the remarkable generalization capacities of overparameterized neural networks is that the optimization algorithms used for training induce an implicit bias towards benign solutions. To grasp this theoretically, recent works examine gradient descent and its variants in simplified training settings, often assuming vanishing learning rates. These studies reveal various forms of implicit regularization, such as norm minimizing parameters in regression and max-margin solutions in classification. Concurrent findings show that moderate to large learning rates exceeding standard stability thresholds lead to faster, albeit oscillatory, convergence in the so-called Edge-of-Stability regime, and induce an implicit bias towards minima of low sharpness (norm of training loss Hessian).
 
In this work, we argue that a comprehensive understanding of the generalization performance of gradient descent requires analyzing the interaction between these various forms of implicit regularization. We empirically demonstrate that the learning rate balances between low parameter norm and low sharpness of the trained model. We furthermore prove for diagonal linear networks trained on a simple regression task that neither implicit bias alone minimizes the generalization error. These findings demonstrate that focusing on a single implicit bias is insufficient to explain good generalization, and they motivate a broader view of implicit regularization that captures the dynamic trade-off between norm and sharpness induced by non-negligible learning rates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work investigates both theoretically and empirically two competing implicit biases present in deep neural networks. They show that there is a trade-off between norm regularization and sharpness regularization. The empirical part shows that the learning rate controls the trade-off and for good performance a middle ground is required. Theoretically a toy example with one data point is analyzed to illustrate the empirics further.

### Strengths
The framing of the norm and sharpness trade-off furthers the discussion on finding the correct implicit bias trying to explain generalization performance. The paper clearly visualizes the results, making the different biases present easy to understand. Similarly, the toy example explains the competing implicit biases of norm and sharpness minimization. Empirically various factors have been investigated showing a general trend across settings.

### Weaknesses
The toy example is simplistic, which is also a strength I believe, although for instance in [1] and [2] the effects of SGD noise and weight decay are investigated with more data samples. Based on these two works, would this suggest that the L1 bias is preferred at all times? 
Note that although a continuous model (SDE) is chosen in [1] they do investigate the role of larger learning as well, besides label noise.

The sharpness measure chosen is not scale invariant as can be seen in Fig 58 and 59 and as mentioned in the related work. To isolate the effect of sharpness and scale it could be better to use other sharpness measures as in [3]. What is the meaning of the sharpness measure when it directly will correlate with the learning rate? 
Furthermore, other regularization effects in modern architectures are underexplored such as weight decay and batch norm.
In general, the experiments are of small scale, the results could be strengthened by considering large scale experiments and other data modalities.

Recent works on different aspects of implicit regularization are omitted such as [2, 4, 5, 6] that cover diagonal linear networks.


[1] Pesme, Scott et al. “Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity.” Neural Information Processing Systems (2021).

[2] Jacobs, Tom et al. “Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?” ArXiv abs/2504.12883 (2025): n. pag.

[3] Andriushchenko, Maksym et al. “A modern look at the relationship between sharpness and generalization.” International Conference on Machine Learning (2023).

[4] Wang, Shuyang and Diego Klabjan. “A Mirror Descent Perspective of Smoothed Sign Descent.” Conference on Uncertainty in Artificial Intelligence (2024).

[5] Papazov, Hristo et al. “Leveraging Continuous Time to Understand Momentum When Training Diagonal Linear Networks.” International Conference on Artificial Intelligence and Statistics (2024).

[6] Andriushchenko, Maksym and Nicolas Flammarion. “Towards Understanding Sharpness-Aware Minimization.” ArXiv abs/2206.06232 (2022): n. pag.

The novel framing outweighs the limited scope of the experimental and theoretical results, I recommend marginal accept.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper investigates the implicit bias of gradient descent under varying step sizes. Through a series of experiments, the authors observed that as the step size increases, the form of the implicit bias transitions from norm regularization to sharpness regularization. In general, the optimal test performance is achieved not at the smallest or largest considered step sizes, but at an intermediate value. The authors thus argued that generalization cannot be explained by a single form of bias. Theoretically, they considered a shallow diagonal linear model with a single data point and showed that the minimum-norm and minimum-sharpness solutions differ, and that neither achieves the lowest generalization error.

### Strengths
The key message of this paper is that a single form of implicit regularization is not sufficient to explain the generalization. Indeed, much of the existing work on implicit bias has focused on a specific regime of step size (and other hyperparameters), where the bias can be characterized in a relatively clean form. What is less appreciated is how the implicit bias depends on the step size, and which form of the bias is more representative of practical settings. To my knowledge, this aspect has not been much highlighted in the literature, and I therefore find the paper valuable. 

In addition, the paper has several other strengths: 
* The presentation is very clear and the investigation is well motivated. 
* The experiments are well organized and cover a wide range of settings. 
* The phenomena related to the critical step size $\eta_c$ and the phase transition in Figure 1 are particularly interesting.

### Weaknesses
The theory part of this paper appears relatively weak. As stated in the abstract and Contribution, the goal of the theoretical analyses is to support the main claim of the paper that "*the generalization of neural networks can not be explained by a single implicit bias*". However, I do not think this is achieved by the presented theory, for the following reasons: 

* Results in Section 3 only identified the min-norm and min-sharpness minimizers and showed that they generally differ. This result alone does not support the main claim, as no generalization is concerned. Results concerning generalization, which I believe should be the main part of the theory, are deferred to Appendix E. 

* I find Appendix E also not convincing, as its analysis is confined to a single data point. In the context of *generalization*, I believe a one-sample setting is qualitatively different from realistic settings, and observations made in the former are uninformative for the latter. Specifically, let $w_1$ be the empirical risk minimizer selected by a fixed implicit bias (e.g., norm minimization) and $w_2$ be the empirical risk minimizer that generalizes optimally. The authors aim to argue that $w_1$ and $w_2$ generally differ, hence that generalization can not be explained by a single bias. However, when the sample size is one, the discrepancy between $w_1$ and $w_2$ seems trivial. With a single sample, the empirical risk $L$ is a poor approximation of the population risk $\tilde{L}$. Note $w_1$ solely depends on $L$ whereas $w_2$ strongly depends on $\tilde{L}$. Due to the weak relation between $L$ and $\tilde{L}$, there is no a priori reason to expect that $w_1$​ and $w_2$​ would coincide or even be close. In other words, I think the failure of an implicit bias in a one-sample case merely comes from a lack of data, and does not indicate that this bias is not the "right" explanation for generalization in practice. 



**Initial recommendation**

Despite the above points, I find the main message and the experiments valuable to the community. Therefore, my initial recommendation is borderline acceptance. Nevertheless, I believe revision is needed for acceptance, either to improve the theoretical analyses or to position the work more clearly as an empirical study. In the latter case, I think the paper could still make a valid contribution if the authors are able to provide (even high-level) explanations for the phenomena observed in Figure 1 (see questions below).

### Questions
I find the presented phenomena, e.g., in Figure 1, very interesting. I understand the following questions may be beyond the paper’s current scope and I do not expect the authors to address each of them in detail. 

* Could the authors comment on the form of the critical step size $\eta_c$? Why is $\eta_c$ related to the maximal sharpness of the GF solutions? This paper [1] might be relevant. 

* The final norm and final sharpness simultaneously exhibit a sharp transition at $\eta=\eta_c$. Could the authors provide intuition for the emergence of such a sharp transition? 

* Could the authors explain the trade off between the final norm and the final sharpness? Why should we expect a decrease in sharpness corresponds to an increase in the norm? This is in contrast with the scalar factorization case, $L(x,y)=(x^\top y - a)^2$, where the squared norm of a minimizer is equal to the sharpness (see, e.g., Theorem F.2., [2]).  

[1] Kreisler, Itai, et al. "Gradient descent monotonically decreases the sharpness of gradient flow solutions in scalar networks and beyond." International Conference on Machine Learning. PMLR, 2023.

[2] Yuqing Wang, Minshuo Chen, Tuo Zhao, and Molei Tao. Large Learning Rate Tames Homogeneity: Convergence and Balancing Effect. ICLR 2022

### Soundness
2

### Presentation
4

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
The paper makes the case that depending on learning rate, GD exhibits different forms of bias (small $\ell_1$-norm for learning rates below a critical threshold value and decreasing sharpness as the learning rate increases beyond the critical threshold). A connection to generalization is drawn with the conclusion that a tradeoff between both biases may be best in some settings. The study is mainly based on extensive experiments, but a very simple model ("shallow diagonal linear network with shared weights") is analyzed theoretically to show that the two types of bias can in fact be very different from each other.

### Strengths
The experiments are relatively comprehensive, covering a range of different settings. The phenomena can be observed consistently throughout (although more cleanly in some experiments than others). The paper is also well presented and easy to read.

### Weaknesses
The paper mostly describes an observed phenomenon and provides some general intuition behind it. However, it was not entirely clear to me what I can do with the result. Is the main message that the learning rate is an important parameter to influence the type of bias? In itself, that does not seem particularly surprising. It has been observed before that GD can behave quite differently from GF. Or it could be the insight that, for generalization, it can be useful to avoid the extreme of either bias and aim for something in between. Maybe it would be interesting to define a new parameterized bias that captures sharpness and $\ell_1$-norm at the extremes and then investigate theoretically how generalization can depend on that parameter?

### Questions
The paper does a very good job of highlighting the different bias behaviors of GD depending on the learning rate, but it could do a better job of pointing out either concrete conclusions/guidance for practitioners and/or open problems that if addressed successfully in future work would have significant impact.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper argues that gradient descent’s generalization can’t be explained by a single implicit bias. Instead, the learning rate steers a trade‑off between two biases—small parameter norm and low sharpness (as in Edge‑of‑Stability)—and the best test performance often occurs at intermediate learning rates that balance the two.

### Strengths
The idea of linking norm and sharpness regularization is cool but it is not established in the paper.

### Weaknesses
The theoretical results are strictly weaker of an already published (more than 1 year ago) and **not cited** paper: https://arxiv.org/pdf/2502.20531. 

In diagonal linear networks there is no progressive sharpening, when you pick learning rate > 2/\lambda_{\max} of the hessian of the solution at initialization you converge to an oscillatory regime around the min-norm solution, this does not impact the generalization/error of the solution but only the norm of the way the solution is parameterized. It is thus misleading to claim anything about generalization when looking at diagonal linear networks.

Moreover, (as even Cohen et al. (2021), the first paper on EoS, observes) MNIST **does not** show progressive sharpening which is necessary to see the Edge of Stability behavior. In particular, in MNIST the problem is approximately linearly separable. Thus not only no sharpening happens, the opposite! The dynamics will get stabler always exponentially fast (see, e.g., https://arxiv.org/pdf/2506.02336).

These are only two of the points that heavily undermine the soundness of any of the claims.

On top of this multiple references are missing, as a blatant example, there is a full line of research existent on the fact that flatness and sharpness do not correlate generally, it started in 2017 with https://arxiv.org/abs/1703.04933, see e.g., https://proceedings.neurips.cc/paper_files/paper/2023/file/0354767c6386386be17cabe4fc59711b-Paper-Conference.pdf.

On top of it, this is all but a **systematic** experimental analysis! Cifar10 and MNIST is not a systematic experimental analysis in the age of the large models. Even if you are writing a paper about cifar10 and MNIST the argument about generalization is not strong enough for the claims.

I find the contributions are poor and it completely lacks of soundness. Even presentation is too long for no reason and too generic, even in the abstract.

### Questions
Can you please comment on the weaknesses?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper measures the sharpness and $\ell_1$ norms of neural networks trained by gradient descent with varying learning rates and observes a tradeoff between the $\ell_1$ norm and the sharpness. They investigate the effect of this tradeoff on generalization and theoretically analyze a diagonal linear network to gain insights into this tradeoff.

### Strengths
- The empirical results are verified in a variety of settings with many additional plots in the Appendix. This includes multiple architectures, datasets, and loss functions. In addition, the empirical setup is extensively detailed in the appendix, which provides strong evidence for the claimed tradeoff between $\ell_1$ norm and sharpness.
- The paper proposes multiple theoretical analyses for understanding the tradeoff between $\ell_1$ norm including an analysis of the diagonal linear net, and generalization analyses under different norms (Appendix E)

### Weaknesses
- I believe that while the result is correctly stated in Appendix B, the paper mischaracterizes the results of Woodworth et al. throughout the rest of the paper. Their result is that under the model $f_w(x) = w^{\odot 2} \cdot x$, gradient flow with small initialization will converge to the parameter $w$ with minimal $\ell_2$ norm. If one defines the implicit classifier $\beta = w^{\odot 2}$ then this is equivalent to finding $\beta$ of minimal  norm as $\|\beta\|_1 = \|w\|_2^2$. **Woodworth et al. does not claim the weights $w$ will have minimal $\ell_1$ norm.**
- As far as I am aware, there is no mechanism for $\ell_1$ regularization in general neural networks (outside of the linear nets with small initialization), and for linear networks this implicit bias is usually stated for the implicit end-to-end linear classifier, not for the weights themselves. This calls into question the empirical results on realistic neural networks which appear to measure the $\ell_1$ norm of the parameters themselves.
- For the diagonal linear network, the paper only argues that the minimal $\ell_1$ norm solution is inconsistent with the edge of stability at large learning rates and does not actually characterize the solutions that gradient descent converges to.

### Questions
- In Figure 2, is the quantity on the $y$ axis the $L^1$ norm of the parameters $\|\theta\|_1$? If so, what evidence is there for $\ell_1$ weight norm as an implicit bias of gradient descent?
- Because the dynamics of gradient descent are highly oscillatory at the edge of stability, it seems possible that the increase in the $\ell_1$ norm at the edge of stability could be due to the oscillations, and not necessarily because gradient descent visits regions of higher $\ell_1$ norm. How does the $\ell_1$ norm change if you average the weights over a few steps before computing the norm? Or decay the learning rate and then compute the norm before it reaches the edge of stability at the new learning rate?

### Soundness
1

### Presentation
2

### Contribution
1
