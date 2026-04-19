# Flatness-aware Adversarial Attack

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 8

## Abstract
The transferability of adversarial examples can be exploited to launch black-box attacks. However, adversarial ones often present poor transferability. To alleviate this issue, by observing that the diversity of inputs can boost transferability, input regularization based methods are proposed, which craft adversarial ones by combining several transformed inputs. We reveal that input regularization based methods make resultant adversarial ones biased towards flat extreme regions. Inspired by this, we propose an attack called FAA which explicitly adds a flatness-aware regularization term in the optimization target to promote the resultant adversarial ones towards flat extreme regions. The flatness-aware regularization term involves gradients of samples around the resultant adversarial ones but optimizing gradients requires the evaluation of Hessian matrix in high-dimension spaces which generally is intractable. To address the problem, we derive an approximate solution to circumvent the construction of Hessian matrix, thereby making FAA practical and cheap. Extensive experiments show the transferability of adversarial ones crafted by FAA can be considerably boosted compared with state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this work, the authors propose flatness-aware adversarial attack (FAA) to improve the adversarial transferability. In particular, FAA adopts a regularizer for the gradient to make the generated adversarial example located in a flat local optimum. To avoid the Hessian matrix calculation, they utilize Taylor expansion to approximate the Hessian matrix. Experiments on ImageNet dataset show the effectiveness of FAA.

### Strengths
1. The paper is well-written and easy to follow.

2. The authors have conducted several experiments to validate the effectiveness of the proposed method.

### Weaknesses
1. The launch experiment is not solid enough. In my opinion, the experiments can only conclude that transferable adversarial examples might be in flatter local optima. It is a necessary but not sufficient condition that adversarial examples in flatter local optima are more transferable.

2. It is not a new method that connects flat regions and input regularization methods [1]. Eq. (4) is similar to the Eq. (3) in [1], making the motivation rather limited. The main difference is how to approximate the Hessian matrix. Tylor expansion is not a novel way for such an approximation.

3. I am curious why did the author adopt the number of iterations $T=20$, since existing works mainly adopt $T=10$. Increasing the number of iterations might result in overfitting, which degrades the baselines' performance.

4. It is expected to see more momentum-based baselines, such as [1], [2].

5. Why does Table 2 miss the baseline RAP? It should be a significant baseline since it is the first paper that locates the adversarial examples in flat local optima for better transferability.


[1] Ge et al. Boosting Adversarial Transferability by Achieving Flat Local Maxima. arXiv Preprint arXiv: 2306.05225, 2023.

[2] Zhang et al. Improving the Transferability of Adversarial Samples by Path-Augmented Method. CVPR 2023.

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an approach to improve the transferability of adversarial attacks using the properties of input signals: the more flat region on the loss curve they occupy, the more transferrable they are.

### Strengths
A paper seems to provide a very good method because:
- it provides almost 100% transferability for both normal (Table 1) and secured (Table 2) models (!) for untargeted attacks with a huge margin over other methods
- it beats other methods for targeted attacks (Table 3) - again with a significant margin
- it provides a solid reasoning based on observations, and the theory behind it
- and even the theory was adopted towards the fast computation cycle (to get rid of Hessian matrix computation) 
- it was tested in real CV applications

### Weaknesses
Although the paper provide a lot of insights, there are still some (I hope minor and improvable) drawbacks:
- Page 4, the ref. to Mean Value Theorem: it'd be better to refer to some classic mathematical results (I guess they are discovered hundreds of years ago, not just 15)
- Page 16, Table 5: It's not clear what "Approximation Error" exactly means: is it the overall Hessian-based additive term, or Term1, or Term2 (Equation 8)?
- Page 19, Appendix E.8: "As shown in Figure 9, FAA produces flatter adversarial examples than FAA" -> "As shown in Figure 9, FAA produces flatter adversarial examples than RAP"?
- But my main concern is the (based on my assessment) theory-related inference on Pages 15 and 15 (Appendix C). Let me briefly provide it so the errors could be corrected:

First of all, I'm not sure that the Eq. (9) is correct. It uses a Taylor Expansion of a complex function (which in turn relies on the derivative of a complex function), so I think the correct way to do it: $f(g(x))=f(g(a))+\nabla g(a)\nabla f(g(a))(x-a)$, but in the Eq. (9) the multiplicative term $\nabla F$ is omitted, but it is not always equal to 1, right?

Page 15: "we use p(x) ≤ p(x + δ)" which is incorrect, and should be $p(x)\geq p(x+\delta)$

Page 15: "Notice that the flatness-aware item punishes the norm of gradients of samples around x + δ. Therefore, this induces ||∇logF (x + δ)||2 and ||$\nabla^{2}$logF (x + δ)|| to be 0." The implication is incorrect, the small first derivative doesn't say anything about the amplitude of the second derivate (example: $\sin x^2$)

Page 15: at the end of Appendix C, when doing all the approximations, nothing has been said about $\nabla \log p(x+\delta)$ which also needs to be close to zero, right?

### Questions
Q1: I would be interested in $l_0$/patch-based optimization and transferability, because it is the most applicable techniques for the real-world adversarial attacks, and the authors' thought on extensibility of their framework to this case (taking into account the differentiability problem).

Q2: Why we need the Appendix B (minmax problem complexity)? It seems like a very obvious thing and doesn't provide any insight at all

Q3: I am very interested whether the theoretic part (see my notes above) can be corrected and improved as now it seems to have a lot of mistakes.

### Soundness
3 good

### Presentation
4 excellent

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
The paper presents an adversarial attack algorithm that leverages flat loss regions to generate transferable adversarial examples. Current methods use input regularization to generate better transferable adversarial examples. In this work, the authors instead derive a flatness-aware regularization term. Further, they propose a Hessian approximation for the gradients. The approach is shown to outperform existing transfer attacks.

### Strengths
1. The approach is well-motivated and is intuitive. The overall presentation is good, and the paper is well written.
2. While similar approaches have been applied to defenses, flatness aware attacks are an interesting application.
3. The paper also shows effective results for real world image classifiers as well as a variety of robust and non-robust models.
4. The attack is significantly successful even when the proxy is not adversarially trained.

### Weaknesses
It might be useful to evaluate the approach for flatness-aware adversarial defenses like TRADES (Zhang et al, ICML 2019), SAM-AT [1], and ATEnt [2]. These methods leverage sharpness aware losses to introduce flat loss landscapes with respect to the input, and could possibly behave differently.

[1] Wei, Zeming, Jingyu Zhu, and Yihao Zhang. "On the Relation between Sharpness-Aware Minimization and Adversarial Robustness.", ADvML Workshop at ICML 2023.
[2] Jagatap, Gauri, et al. "Adversarially robust learning via entropic regularization." Frontiers in artificial intelligence 4 (2022): 780843.

### Questions
See weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
