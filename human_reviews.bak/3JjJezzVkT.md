# The Marginal Value of Momentum for Small Learning Rate SGD

- Decision: Accept (poster)
- Scores: 5, 6, 6, 5

## Abstract
Momentum is known to accelerate the convergence of gradient descent in strongly convex settings without stochastic gradient noise. In stochastic optimization, such as training neural networks, folklore suggests that momentum may help deep learning optimization by reducing the variance of the stochastic gradient update, but previous theoretical analyses do not find momentum to offer any provable acceleration. Theoretical results in this paper clarify the role of momentum in stochastic settings where the learning rate is small and gradient noise is the dominant source of instability, suggesting that SGD with and without momentum behave similarly in the short and long time horizons. Experiments show that momentum indeed has limited benefits for both optimization and generalization in practical training regimes where the optimal learning rate is not very large, including small- to medium-batch training from scratch on ImageNet and fine-tuning language models on downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the role of momentum in SGD training. When the learning rate is small but the gradient noise is not small, the paper shows that the trajectories of SGD and SGDM are close. Therefore, the momentum component in SGDM adds only a small effect on top of SGD. This paper also conducts experiments to verify that the optimization and generalization of SGDM and SGD are close when the learning rate is small and the mini-batch size is not large (such that the gradient noise is not small).

### Strengths
Theoretically, momentum provably improves the (minimax) convergence rate of GD for (strongly convex) smooth optimization. Practically, momentum is often used with SGD (as GD is expensive in practical machine learning optimization). In the presence of gradient noise, will momentum still offer benefits? This is the central question studied in this work. 

First, this work presents a theory, showing that when the learning rate tends to zero, the trajectories of SGD and SGDM are close in two regimes. The first regime is when the gradient noise is large (with $O(1/\eta)$ variance) but the total running distance is constant. The second regime allows the total running steps to be large but requires an overparameterized model where the global solutions form a manifold. 

Second, this work conducts experiments to verify that for both training and test, SGDM and SGD are close when the learning rate is small and when the batch size is not too large. 

The writing is clear overall. Some places require clarifications. See more in the next section. 

I did not check the proof.

### Weaknesses
1. I feel the introduction/motivation could use some polishment. Specifically, the benefit of momentum is for stabilizing GD with a large learning rate ($>2/L$, in terms of improving the minimax convergence rate for (strongly convex) smooth optimization) as discussed in the first paragraph of the introduction. However, I did not recall a theoretical result that claims acceleration of momentum for GD with a learning rate. 

One of the main contributions of this paper is to show that momentum does not change SGD trajectory much when the learning rate is sufficiently small. Therefore, momentum is not very helpful. 

I am not sure how surprising this is, since the benefit of momentum is only claimed for GD with a very large learning rate. 

Also, the condition that $1-\\beta\^{(n)}_k = \\Theta ((\\eta^{(n)})\^{\\alpha} )$ (in definition 3.1) for $\alpha \in [0,1)$ (in Theorem 3.5) basically says that the history gradient will be forgotten fast (in a nearly exponential rate) when SGDM runs $\\Theta(1/\\eta)$ steps. So it's not very surprising that SGDM will be close to SGD. 


2. In the last paragraph of Section 3, the paper briefly mentions a phenomenon for $\\alpha>1$. However, the theorem only covers $\\alpha <1$. Could the author elaborate on what happens to SGDM when $\\alpha>1$?


3. In the paragraph after Lemma 2.4, $\\sigma$ is set to be $1/\sqrt{\eta}$ to ensure a non-trivial gradient noise scaling. However, later in Theorem 3.5, it allows that $\\sigma \le \eta^{-1/2}$, and in Theorem 4.5, it allows $\\alpha = \\Theta(1)$. These seem to be different from the prior discussions. Could the authors clarify the scaling of the noise?  In particular, can Theorem 3.5 be applied to GD and GDM (i.e., $\\sigma = 0$)?


4. This paper mentions two versions of SGDM, that is, eq (1) used in practice and eq (3) used in theory. I appreciate the usage of $\\gamma$ and $\\eta$ to differentiate the learning rate for those two versions, which helps highlight the difference in the effective optimization length. 

In the ImageNet experiment, the learning rate for eq (1) is manually rescaled when compared to SGD (to align with the theory setup in eq (3)), which is good. However, for training ResNet-32 on CIFAR-10, the paper writes that "We first grid search to find the best learning
rate for the standard SGDM ($\\ell=1$), and then we perform SGD and SGDM with that same learning rate for different levels of $\\ell$". I am wondering if the learning rate for SGD needs a rescaling here.

### Questions
See above.

### Soundness
2 fair

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
This paper aims to investigate the advantages of incorporating momentum in the Stochastic Gradient Descent with Momentum (SGDM) method for mitigating the variance associated with stochastic gradients. To mitigate oscillations along the high-curvature direction, the authors make the assumption that the learning rate is small. Within this context, they demonstrate that, subject to specific assumptions, Stochastic Gradient Descent (SGD) and SGDM yield identical training error results. Additionally, similar to previous studies, they construct a Stochastic Differential Equation (SDE) for the continuous version of SGDM/SGD and, through an analysis of the SDE's limiting behavior, reveal that both SGD and SGDM exhibit a similar implicit bias.

### Strengths
Prior research has noted that the advantage of incorporating momentum in the training of neural networks is somewhat limited, and there hasn't been a definitive theoretical outcome regarding the effectiveness of momentum in the stochastic setting. Therefore, demonstrating that momentum does not significantly reduce variance or improve generalization can lead to savings in computational resources and memory usage during deep neural network training. A noteworthy technical innovation in this work is the introduction of a novel Stochastic Differential Equation (SDE) to represent the diminishing learning rate in Stochastic Gradient Descent with Momentum (SGDM) and Stochastic Gradient Descent (SGD).

### Weaknesses
The assumptions that the paper considers to show that SGDM approximate SGD is too restrictive. It would be great if they can justify these too restrictive assumptions.

### Questions
In Theorem 3.5, the paper demonstrates that the trajectory of Stochastic Gradient Descent with Momentum (SGDM) approximates that of Stochastic Gradient Descent (SGD) with a specific learning rate. While it may be somewhat expected that these two methods, both being first-order optimization techniques, would exhibit some similarity, the importance of this result lies in its ability to establish a specific condition under which SGDM behaves similarly to SGD.

However, as you rightly suggest, a more generalized result would be valuable. If the paper could establish conditions where, for certain small learning rates, SGDM and SGD consistently approximate each other, it would offer broader insights into the relationship between these methods and provide more guidance on when to expect their similarities. This would enhance our understanding of the interplay between learning rates and optimization algorithms, potentially leading to more informed choices in practical applications.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work compares SGDM and SGD in the setting of small learning rate and gradient noise is large enough to produce instability. Two main results are presented, both indicating that the training trajectory of SDGM is close to that of SGD. The first result states that within $O(1/\eta)$ steps of training, the trajectories of SGD and SGDM approximate each other with distance $O(\sqrt{\eta/(1-\beta)})$. The second result states that within $O(1/\eta^2)$ steps of training, the trajectories of both SGD and SGDM move slowly after reaching a manifold where the local minimizer is located.

### Strengths
1. The main results, i.e., Theorem 3.5 and Theorem 4.5, are strong in terms of both implications and proof techniques.
2. The results are presented in an orderly manner.
3. The result that the training trajectories of SGDM and SGD are similar is intriguing.

### Weaknesses
1. Although the implications of Theorem 3.5 is clear, the statement of the theorem is a bit confusing. Specifically, the averaged learning schedule $\bar\eta_k$ is introduced, and it does not appear again until the appendix.
2. It would be interesting to see other concrete types of hyperparameter schedule apart from the constant ones, both theoretically and empirically.
3. It would also be interesting to see experiments demonstrating the main theoretical contribution of this paper, i.e., the trajectories rather than the loss / accuracy stay close.

### Questions
1. Do the two regimes, i.e. $t=O(1/\eta)$ and $t=O(1/\eta^2)$ have any connections? Can we understand from the results provided in this work what's going on when $t=\Omega(1/\eta)$ but $t=o(1/\eta^2)$?
2. Could the authors clarify why a **class** of hyperparameter schedule $\{\eta_k^{(n)}, \beta_k^{(n)}\}$, indexed by $n$, is considered in both main theorems? It seems more straightforward to me to consider only one choice of hyperparameter schedule.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors have proved theoretically the marginal benefit of momentum in training when the learning rate is small and the noise is dominant. The theory has been verified with some experiments.

### Strengths
1. The paper has demonstrated the limited benefit of momentum theoretically in some regimes, which helps suggest when not to use momentum.
2. The article has illustrated the idea with warmup examples, which makes it easier to digest.

### Weaknesses
1. The analysis is based on the fact that the momentum effect is dominated by other factors when the learning rate is negligible. This is somewhat intuitive and needs to dive deeper into what the special role of momentum is. 

2. There needs to be clear examples of what scenarios the theory fits in. For example, with what class of loss and neural network does the theory fit in?
 
3. The paper seems rushed in polishing. There are some links of reference in the paper that need to be added, for example, in line 2 on page 8.

### Questions
1. Could you show some examples of what kind of networks, loss..etc the theory fits in.
2. It will be nice to make more clear comparison of when with and without momentum.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
