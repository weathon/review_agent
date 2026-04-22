# Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Online-to-non-convex conversion (O2NC) is an online updates learning framework for producing Goldstein $(\delta,\epsilon)$-stationary points of non-smooth non-convex functions with optimal oracle complexity $\mathcal{O}(\delta^{-1} \epsilon^{-3})$. Subject to auxiliary \emph{random interpolation or scaling}, O2NC recapitulates the stochastic gradient descent with momentum (SGDM) algorithm popularly used for training neural networks. Randomization, however, introduces deviations from practical SGDM. So a natural question arises: Can we derandomize O2NC to achieve the same optimal guarantees while resembling SGDM? On the negative side, the general answer is \emph{no} due to the impossibility results of~\citet{jordan23deterministic}, showing that no dimension-free rate can be achieved by deterministic algorithms. On the positive side, as the primary contribution of the present work, we show that O2NC can be naturally derandomized for \emph{weakly convex} functions. Remarkably, our deterministic algorithm converges at an optimal rate as long as the weak convexity parameter is no larger than $\mathcal{O}(\delta^{-1}\epsilon^{-1})$. In other words, the stronger stationarity is expected, the higher non-convexity can be tolerated by our optimizer. Additionally, we develop a periodically restarted variant of our method to enable more progressive updates when the iterates are far from stationarity. The resulting algorithm, which can be viewed as a momentum-restarted variant of SGDM, has been empirically demonstrated to be effective and efficient for training ResNet and ViT models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper extends the Online-to-Non-Convex Conversion (O2NC) framework by introducing an online learning method that efficiently finds approximate stationary points for non-smooth, non-convex functions with optimal oracle complexity. It shows that when random interpolation or scaling is applied, O2NC becomes equivalent to the stochastic gradient descent with momentum algorithm commonly used in deep learning, providing a theoretical explanation for its effectiveness. Although it is generally impossible to remove randomness without losing optimality, the paper proves that for weakly convex functions, a deterministic version of O2NC can still achieve the same optimal rate when the weak convexity parameter is not too large. The authors also propose a momentum-restarted variant that improves progress when the algorithm is far from stationarity and demonstrate through experiments on ResNet and Vision Transformer networks that this version is both efficient and effective in practice.

### Strengths
The paper establishes that O2NC achieves an oracle complexity of O(deltaˆ(-1) epsilonˆ(-3)) for finding Goldstein (delta-stationary points) of non-smooth, non-convex functions. This result improves the state-of-the art complexity of this problem (e.g., compared to Cutkosky, 2019).

Another clear contribution is that theoretical analysis clearly characterizes the boundary between when randomness is necessary and when it is not. Using the impossibility results of Jordan et al. (2023), the paper explains why fully deterministic algorithms cannot achieve dimension-free rates in general non-convex settings. However, it provides a positive theoretical result showing that when the weak convexity parameter is smaller than O(deltaˆ(-1) epsilonˆ(-3)), randomness can be eliminated without losing optimality. 

The experiments show that the proposed D-O2NC method improves upon the baseline SGDM algorithm. When applied to training deep neural networks on CIFAR-10 using ResNet-101 and ViT models, D-O2NC achieves faster convergence and slightly higher test accuracy, demonstrating the practical benefit of its periodic momentum restart mechanism.

### Weaknesses
The results are only for weakly convex functions, which is theoretically fine. But, are there many functions that are non-convex but weakly convex? In particular, if I understand correctly, the problem considered in experiments does not have this property, or at least the paper does not weakly convex, so I dont know, but my feeling is that all the interesting problems are convex. Thus it would be useful if there was better comparison with the state of the art for convex setup as well.

To my understanding, although the analysis is novel, the algorithm itself closely resembles existing momentum-based methods such as SGDM or restarted gradient methods. The contribution lies mainly in the theoretical interpretation rather than in a new optimization technique.

The experimental results are relatively weak and provide only limited empirical support for the proposed method. The evaluation relies on single runs without statistical analysis, making it difficult to assess the consistency or significance of the reported improvements. Moreover, there is little exploration of hyperparameters, such as restart frequency or learning rate, and no ablation study to clarify how these choices affect performance. The experiments also lack comparisons with other widely used optimizers beyond SGDM, which limits the context for evaluating the method’s advantages. 

Moreover, the experimental section feels somewhat disconnected from the theoretical contributions of the paper, as it does not explicitly demonstrate or test the theoretical claims, such as optimal complexity or the benefits of weak convexity, in practice.

### Questions
I feel like there is some lack between the theory and practice. How would you for example use the theory to run the algorithm, e.g., hyper parameters. 

Is it possible to determine or approximate in practice the step-size and other hyperparameter choices required by the theoretical analysis, and if so, how sensitive is the algorithm’s performance to deviations from these theoretically prescribed values?”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the Online-to-nonconvex conversion (O2NC) under weakly convex scenario. While the original O2NC has general update form $w_n = w_{n-1} + s_n\Delta_n$, where $\Delta_n$ denotes the update vector and $s_n$ is a random scaling or interpolation, this paper proposes derandomized O2NC (D-O2NC) with deterministic update form $w_n = w_{n-1} + \Delta_n$. The core difference is that O2NC relies on the identity $\mathbb{E}\_{s_n } [R(w_n) - R(w\_{n-1})] = \mathbb{E}[ \langle \nabla R(w_n), \Delta_n\rangle]$ from random scaling, while D-O2NC relies on the weak convexity identity $R(w_n) - R(w_{n-1}) \le \langle \nabla R(w\_n), \Delta\_n\rangle + \frac{\rho}{2}\\|\Delta_n\\|^2$, where $\rho$ denotes the weak convexity constant. Consequently, upon substituting online learners with proper regret bound on the quadratic loss $f_n(\Delta) = \langle g\_n, \Delta\rangle + \frac{\rho}{2}\\|\Delta\\|^2$, the resulting D-O2NC optimizer achieves good convergence guarantee. In particular, this paper considers two such online learners, clipped OGD and unclipped OGD with periodic restarting, both achieving sub-linear regret on the aforementioned quadratic losses and recovering variants of the popular SGDM optimizer upon substituting into D-O2NC. Finally, this paper also provides empirical experiments on cifar10 with resnets and ViT models and shows that D-O2NC outperforms standard SGDM.

### Strengths
- The weakly convex scenario is rarely studied in the line of O2NC research, and this paper provides new results along this line. These results are novel, and they help to better understand the performance of O2NC under different settings.
- In the convergence rate, I find it interesting that the weak convexity constant $\rho$ is only in the non-dominant terms. In other words, the convergence rate of derandomized O2NC matches that of O2NC as long as weak convexity is bounded by $O(\delta^{-1}\epsilon^{-1})$, even though there is no random scaling.
- Compared to original O2NC, derandomized O2NC better aligns with the practical implementation of popular optimizers such as SGDM, in which the random scaling is typically absent. 
- Empirical results on cifar-10 with resnets and ViT demonstrates better practical performance of D-O2NC compared to vanilla SGDM.
- Overall, I find this paper well-written. The presentation of the main results and corresponding discussions are clear and easy to follow.

### Weaknesses
The discussion of the lower bound on the convergence rate of Goldstein stationary point of a weakly convex loss is missing. Although $O(\delta^{-1}\epsilon^{-3})$ is the minimax optimal rate for the general non-convex losses, it's unclear whether it remains tight for weakly convex functions. In fact, in the discussion section about the connection to Moreau envelope (e.g. Appendix D), it seems to me that the previously achieved rates on Moreau envelope translates to better rates on Goldstein stationary point. Could the authors confirm if I understand correctly, and provide some insights on the lower bound?

### Questions
- Regarding the definition of regulated Goldstein stationary, my understanding is that this definition makes it more convenient for the convergence analysis, but does not include more optimizer algorithms (unlike the relaxed Goldstein stationary definition which relaxes the deterministic radius $\delta$ to a stochastic variance bound). This is further convinced by the equivalence relation between standard Goldstein stationary and regulated version (Lemma 1). Am I understanding correctly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides an algorithm for finding Goldstein stationary points of weakly convex functions using the "online-to-non-convex" framework.
The resulting algorithm does not require randomization, which is required for previous analyses that do not require weak convexity.

A few experimental results are provided using cifar10.

### Strengths
Removing the randomization is intuitively a desirable behavior, and the present result provides a more general way to obtaining convergence rates without randomization. In particular, it strictly generalizes prior results finding stationary points of smooth or second-order smooth objectives.

### Weaknesses
The most obvious weakness is that the results for finding stationary points of the Moreau envelope are suboptimal.

### Questions
For the experiments, what is meant by learning rate and momentum parameters? How are they translated to the new algorithms? Is the learning rate the learning rate of the inner online learner or something else?

The resulting algorithms look rather similar to SGD with momentum. Indeed, without the clipping value, option 1 appears to literally be SGD with momentum. It looks like the algorithm of Cutkosky & Zhang does not require clipping through use of an extra regularization - would that be applicable here? In such a case, the analysis of Mai & Johanansson would also apply to this algorithm and so both kinds of convergence would be achievable.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a derandomized version of the Online-to-Non-Convex Conversion (O2NC) framework for stochastic weakly convex optimization. Two options are given, one is clipped SGDM, and the other is restarted SGDM. The derandomization part is novel but the the weakly convexity condition seems to be strong for this problem class.

### Strengths
1. This paper proposes a deterministic framework that achieves the optimal complexity, showing that this is doable for weakly convex functions even not doable for general non-convex non-smooth functions.
2. Weaker convexity is allowable when the desired error is small.

### Weaknesses
1.  It's not clear under the weakly convex condition, what's the technical challenges addressed in this work. 
2. Lack of examples of weakly convex functions and related numerical experiments. It would be very helpful if there are some simple function constructions to understand these bounds.

### Questions
1. Is weak convexity necessary for achieving optimal rates with deterministic algorithms? 
2. Can you elaborate the hyperparameter tuning of the experiment section? Is it possible that SGDM converges slower due to suboptimal hyper-parameters?

### Soundness
4

### Presentation
3

### Contribution
3
