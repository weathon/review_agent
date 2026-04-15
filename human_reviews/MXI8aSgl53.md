# NAG-GS: Semi-Implicit, Accelerated and Robust Stochastic Optimizer

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Classical machine learning models such as deep neural networks are usually trained by using Stochastic Gradient Descent-based (SGD) algorithms.  The classical SGD can be interpreted as a discretization of the stochastic gradient flow. In this paper we propose a novel, robust and accelerated stochastic optimizer that relies on two key elements: (1) an accelerated Nesterov-like Stochastic Differential Equation (SDE) and (2) its semi-implicit Gauss-Seidel type discretization. The convergence and stability of the obtained method, referred to as NAG-GS, are first studied extensively in the case of the minimization of a quadratic function. This analysis allows us to come up with an optimal learning rate in terms of the convergence rate while ensuring the stability of NAG-GS. This is achieved by the careful analysis of the spectral radius of the iteration matrix and the covariance matrix at stationarity with respect to all hyperparameters of our method. Further, we show that NAG-GS is competitive with state-of-the-art methods such as momentum SGD with weight decay and AdamW for the training of machine learning models such as the logistic regression model, the residual networks models on standard computer vision datasets, Transformers in the frame of the GLUE benchmark and the recent Vision Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel, robust and accelerated stochastic optimizer based on combining Nesterov's AGD and semi-implicit Gauss-Seidel method to obtain an iterative scheme for the optimization. Convergence of this algorithm in the quadratic case is proved and numerical experiments are demonstrated for logistic regression model, ResNet-20, VGG-11 and Transformers.

### Strengths
proposed a novel stochastic optimizer by combining Nesterov AGD with Gauss-Seidel semi-implicit method

### Weaknesses
(1) it seems the contribution is relatively moderate by just combining Nesterov with Gauss-Seidel
(2) some of the writings are sloppy. Example: page 4, line 2, two "either" appeared

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a stochastic method through applying implicit Gauss-seidel splitting method on the continuous-time ODE mimicing the trajectory of Nesterov’s accelerated method. They prove the convergence of their algorithm for the special case of strongly-convex quadratic functions in Theorem 1. They extended their algorithm to non-convex functions heuristically based on the step-size they found for the special case of strongly convex quadratic functions. Several experiments were conducted to evaluate the performance of their method.

### Strengths
1- Connecting theoretical findings with practical applications and implementations.

2- One step toward practical implementations through considering stochastic extension of prior art in [Luo & Chen (2021)].

3- Text is smooth and easy to read.

### Weaknesses
1- Related work is very imited. The idea of analyzing accelerated methods through their continuous time perspective has been around for quite some time (since [Su, et. al (2014)] or even before that by [Alvarez & Attouch (2001)] and most of the related works mentioned deal with intrepretations of deterministic methods. It makes more sense to focus on works that see stochastic accelerated methods through the lens of ODEs since these are more related to the proposed research.

2- The theoretical analysis is bounded to quadratic case. This is not mentioned accurately in the contribution. Specifically the second main contribution is: **We analyze the properties of the proposed method both theoretically and empirically;**. 

3- The introduction is not really an introduction of this work. By just reading the introduction, it is not possible to get an accurate idea of what differences this work has with any other work in "stochastic optimization algorithms".

4- "The Preliminaries" is a mixture of background, related work and notations. Here, more organization might improve readability. 

5- Gaus-Seidel splitting used here is not a novel idea in discretizing ODEs for acceleration as it was previously discussed in [Luo & Chen (2021)].


References

Alvarez, F., Attouch, H. An Inertial Proximal Method for Maximal Monotone Operators via Discretization of a Nonlinear Oscillator with Damping. Set-Valued Analysis 9, 3–11 (2001).

### Questions
1- What is $\mathcal R(\lambda)$ under (8)? Does it extract the real part of $\lambda$?

2- Have you tried the predictor-corrector method in [Luo & Chen (2021)] and extend it to the stochastic case (like the way you did for the semi-implicit GS)?

3- [Shi, et al. (2019)] showed that semi-implicit Euler discretization of a “high-resolution ODE” exactly recovers the NAG algorithm. This is not the case for other ODEs like the one in  [Su, et. al (2014)] or [Luo & Chen (2021)], unless some corrections are made. Do you think applying semi-implicit GS on high-resolution ODEs (combined with stochastic gradients) can lead to an even better algorithm?

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel stochastic optimizer called NAG-GS, which combines elements of Nesterov-like Stochastic Differential Equation (SDE) acceleration and semi-implicit Gauss-Seidel type discretization. The method's convergence and stability are extensively analyzed, particularly in the context of minimizing quadratic functions. The authors determine an optimal learning rate that balances convergence speed and stability by considering various hyperparameters. NAG-GS is shown to be competitive with other state-of-the-art methods like momentum SGD with weight decay and AdamW when applied to various machine learning models, including logistic regression, residual networks, Transformers, and Vision Transformers across different benchmark datasets.

### Strengths
- The NAG-GS is derived from an accelerated Stochastic Differential Equation (SDE) using its semi-implicit Gauss-Seidel type discretization, which is interesting.

- The convergence analysis for the quadratic case is comprehensive.

### Weaknesses
- The discussion of NAG-GS with other similar methods is insufficient. For example, Is NAG-GS faster than Polyak's momentum method for solving quadratic objectives? How does it compare with other variants of NAG, such as Triple momentum method [1, 2], and ITEM [3]? It is not clear what is the key benefit of NAG-GS in its original setting.

- The improvement in the neural network experiments seems marginal. No deviation statistics is provided in the empirical results.

- A minor point: As one of the key features of NAG-GS, the derived optimal learning rate should be mentioned and discussed in the main text.   

=================== After Rebuttal ======================

Thanks the authors for their revision and detailed feedback. The contribution is much clearer and I have increased the score to 5. In my humble opinion, the contribution (theoretical and empirical) is still not sufficient for acceptance as pointed out by other reviewers.

=====================================================


[1] Van Scoy, B., Freeman, R. A., & Lynch, K. M. (2017). The fastest known globally convergent first-order method for minimizing strongly convex functions. IEEE Control Systems Letters, 2(1), 49-54.


[2] Zhou, K., So, A. M. C., & Cheng, J. (2020). Boosting first-order methods by shifting objective: new schemes with faster worst-case rates. Advances in Neural Information Processing Systems, 33, 15405-15416.


[3] Taylor, A., Drori, Y. An optimal gradient method for smooth strongly convex minimization. Math. Program. 199, 557–594 (2023).

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new accelerated optimization algorithm using gauss-siedel discretization of a randomization version of the recent spectral lifting technique of Luo and Chuo 2021

### Strengths
The paper is largely well written with minor polishing still required. The method seems sound from a few empirical and numerical experiments conducted by the authors, and achieves competitive performance.

### Weaknesses
The main weakness is that the convergence analysis is only for the quadratic case, and the convergence itself as stated in theorem 1 is a weak statement with only asymptotic convergence. I did not go through the entire proof, but i can understand that the analysis for general f (replacing Ax with grad f  for the method) is non-trivial, since the "lifted" spectrum needs to be bounded effectively. It seems one also requires prior knowledge of \mu for the algorithm which is very limiting, and the convergence analysis also is only valid for strongly convex cases.  

However, the empirical results are quite strong which warrants that the community should know about this paper.

There are some typos etc that need to be fixed.

### Questions
Algorithm 1 requires prior knowledge of \mu. How did you set up the algorithm practically for non-convex losses ?

Mostly minor writing stuff: 

Background: Please explain the definition of A-stable. If you have gone as far to explain the discretization process itself, adding A-stable for reading not familiar with it would benefit from it.

What is \mathcal{R}(\lambda) below (8) ? 

“However, this requires to either solve a linear system either.” --> “However, this requires to either solve a linear system or.”  

Please mention what is the baseline SGD-MW before using it in table 1. Is there a reason Adam or a variant of it was considered as a baseline for Resnet20 ?

What is the accelerated gradient descent baseline in Figure 1 ?

“But still, it is expected that an explicit scheme closer to the implicit Euler method will have good stability with a larger step size than the one offered by a forward Euler method. “ – why ?

Can the authors provide some future work directions ? convergence analysis ?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
