# Closed-form Last Layer Optimization

- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
Neural networks are typically optimized with variants of stochastic gradient descent. Under a squared loss, however, the optimal solution to the linear last layer weights is known in closed-form. We propose to leverage this during optimization, treating the last layer as a function of the backbone parameters, and optimizing solely for these parameters. We show this is equivalent to alternating between gradient descent steps on the backbone and closed-form updates on the last layer. We adapt the method for the  setting of stochastic gradient descent, by trading off the loss on the current batch against the accumulated information from previous batches. Further, we prove that, in the neural tangent kernel regime, convergence of this method to an optimal solution is guaranteed. Finally, we demonstrate the effectiveness of our approach compared with standard SGD on a squared loss in several supervised tasks -- both regression and classification -- including Fourier Neural Operators and Instrumental Variable Regression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes training with the closed-form last-layer solution (under squared loss): at each step, recompute the final linear weights exactly from current features and update only the backbone by gradient descent. By the envelope theorem, this avoids back-propagating through the matrix inverse and is equivalent to alternating a backbone GD step with an exact least-squares solve for the last layer. For minibatch SGD, the authors add a proximal regularizer that pulls the new last-layer solution toward the previous one, preventing per-batch overfitting; the resulting loop alternates (i) a backbone GD step and (ii) a closed-form ridge solve with proximal coupling, admitting an approximate Kalman filter / RLS interpretation. A backbone-first update order works best empirically.

Theoretically, optimizing the induced implicit loss $L^{*}(\theta)$ is non-convex with possible spurious critical points, but in the infinite-width NTK regime (with a positive-definite kernel and sufficient feature rank) gradient descent converges to a global optimum. Empirically, on regression (FNO/PDEs and DFIV) the method converges faster and attains lower MSE than $\ell\_{2}$-SGD, and in DFIV removes the need for a costly final refit. On CIFAR-10/100, closed-form $\ell\_2$ training consistently beats $\ell\_2$-SGD and can match or exceed cross-entropy on CIFAR-100, though on ImageNet cross-entropy remains stronger. Overall: a simple, practical algorithm with supporting theory and broad empirical gains, especially for squared-loss training.

### Strengths
- Novel Optimization Approach: The idea of enforcing the last layer to be optimal at each step is a fresh departure from standard end-to-end SGD. It exploits a known closed-form solution in an innovative way to simplify the optimization problem, essentially performing exact minimization over last-layer weights in each iteration. This two-timescale strategy has appeared in theory, but this work is the first to turn it into a practical training method integrated with SGD.

 - Theoretical Insight: The paper offers non-trivial theoretical contributions. It analyzes the implicit loss landscape when the last layer is always optimal, proving that while this landscape is generally non-convex with potentially bad critical points, gradient descent in the infinite-width NTK limit will avoid those and find a global minimizer. This convergence theorem (Theorem 4) under positive-definite kernel assumptions is a reassuring theoretical justification for the method’s efficacy. Moreover, the use of the envelope theorem to show that backpropagation through the closed-form solution is unnecessary (Theorem 1 and 2) is a nice theoretical simplification that saves computation.

 - Practical Stochastic Algorithm: The paper identifies and tackles the key challenge of mini-batch training (last-layer overfitting to each batch) by introducing a proximal regularization to the closed-form update. This is a simple and effective fix that keeps the last layer update stable and coupled to the backbone’s progress, unlike a naive moving average approach which could decouple and diverge. The resulting algorithm (Algorithm 1) is easy to implement and can plug into existing training pipelines, as it essentially alternates a usual backbone SGD step with solving a small linear system for the last layer.

 - Empirical Performance: Across regression tasks, closed-form last-layer training yields lower MSE and faster convergence than vanilla SGD, with especially strong gains in small-batch settings (the proximal update stabilizes stochasticity). On DFIV, it outperforms small-batch baselines and nearly matches costly two-stage refitting—eliminating that step. In classification, squared-loss with closed-form updates consistently beats $\ell_2$-SGD on CIFAR-10/100, and on CIFAR-100 it even modestly outperforms cross-entropy. Overall, results show both optimization speedups and better generalization in many cases.

### Weaknesses
- Restriction to Squared Loss: A notable limitation is that the method inherently relies on the squared loss to obtain a closed-form solution for the last layer. This means it cannot be directly applied to tasks where cross-entropy or other non-quadratic loss functions are standard. In classification, using a squared error surrogate is somewhat non-standard and requires a heuristic at prediction time (taking an argmax of outputs since they don’t form a probability distribution). While the authors show this can work reasonably, it forgoes the probabilistic interpretation and other benefits of the softmax cross-entropy framework. The approach’s success on CIFAR-100 vs. cross-entropy is intriguing, but on larger-scale tasks like ImageNet it underperforms cross-entropy training, indicating that the benefits of the closed-form update may not overcome the loss function mismatch when the number of classes is very large or the problem is more complex. This limits the method’s applicability in scenarios where using the true cross-entropy loss (or others like hinge, etc.) is essential.

 - Theoretical Gaps for Finite Networks: The convergence guarantee provided (Theorem 4) is in the infinite-width NTK regime, which is a strong assumption that may not hold in practice. For finite networks, the loss $L^(\theta)$ can have non-global critical points, such as trivial feature representations that zero out gradients. The paper does not prove that gradient-based training will avoid these bad critical points in general finite settings. While standard deep networks typically don’t get stuck in completely uninformative representations, it’s theoretically possible that this new training objective could introduce different failure modes. The authors do not report observing such issues in practice (and indeed their method found good solutions in experiments), but a formal understanding for finite width is lacking. This leaves a slight gap in the theoretical guarantees while in contrast, conventional end-to-end training has well-understood critical point structures in overparametrized settings (e.g. no bad local minima under certain assumptions), it’s less clear for the $L^(\theta)$ objective.

 - Computational Overhead: Another concern is the extra computation required for the closed-form updates. Each iteration involves solving a d×d linear system (or inverting a matrix of size equal to the feature dimension) to compute $W^*(\theta)$. In the experiments, the feature dimension (d) was modest (e.g. 256 or 512), and this overhead was manageable, but for very high-dimensional features or extremely large output layers, this step could become a bottleneck. The paper does not report runtime comparisons or discuss strategies to mitigate this cost. It’s possible to use efficient linear algebra or warm-start techniques (given successive updates are on similar data), but such optimizations are not explored. Therefore, the scalability of the approach to very large networks or datasets (where d or the number of classes is in the thousands) remains a bit unclear.

 - Hyperparameter Sensitivity: The introduction of the proximal regularization coefficient $\lambda$ (and the interaction with the ridge parameter β, if any) adds an extra hyperparameter that needs tuning. The paper does include an ablation showing how performance varies with $\lambda$ and with the ridge coefficient. It appears that the method’s performance can be sensitive to the choice of $\lambda$, especially for smaller batch sizes where this term is critical – too large $\lambda$ might slow adaptation of the last layer, too small might reintroduce instability. The need to tune $\lambda$ (and potentially β and learning rate jointly) makes the method a bit more complex to use than standard SGD (which typically only needs a learning rate schedule and perhaps a weight decay). While this is not a major flaw, it means practitioners must budget some effort for hyperparameter search to fully realize the benefits of the approach.

 - Combining with Other Optimizers: The study found that using Adam (an adaptive optimizer) for the backbone parameters degraded performance relative to SGD. This suggests the closed-form last layer strategy might not be trivially compatible with all optimizer styles. The authors hypothesize that Adam’s internal state (momentum of gradients) might conflict with the idea of immediately resetting the last layer to optimal each time. Similarly, one might wonder if adding momentum to the backbone SGD could interfere with the two-timescale dynamics. The paper focuses on basic SGD; the limited exploration of optimizer variants is a minor weakness, as it leaves unclear whether the method can reap benefits from momentum or adaptive learning rates (which are often important in state-of-the-art training regimes). It may be that the closed-form update provides sufficient acceleration that such techniques are less important, but some discussion or experiments on this would strengthen the work.

### Questions
1. The closed-form last layer update requires solving a linear system at every iteration. Have the authors considered more efficient or scalable ways to implement this? For instance, could one exploit the approximate Kalman filter interpretation to update $W$ incrementally (reusing the previous inverse or using Sherman-Morrison updates) instead of recomputing from scratch each time? Any discussion on how the method scales with increasing feature dimension or number of classes would be valuable – e.g. is the matrix inversion step ever a bottleneck in practice, and how might one mitigate this for very large models?

2. The analysis shows that $L(\theta)$ can have spurious critical points (e.g. the feature map producing zero outputs is stationary). In practice, did the authors ever observe training getting “stuck” in a bad state, or does random initialization and SGD dynamics reliably avoid those trivial solutions? It would be helpful if the authors could elaborate on why, despite the non-convexity of $L(\theta)$, the method seemed to find good solutions (especially in finite-width networks not covered by NTK theory). Are there any conditions or initialization strategies needed to ensure convergence to a good optimum when using this closed-form approach?

3. The introduction of the proximal regularization coefficient $\lambda$ raises the question of how to choose it. The paper provides an ablation, but could the authors offer more guidance on this? For example, should $\lambda$ be scaled with the batch size or learning rate in some way? Is it essentially acting as an “effective batch size” or memory factor for the last layer updates? Additionally, the method still includes the ridge regularizer β in principle – in the experiments, was β set to zero (relying only on $\lambda$), or did the authors keep a small β as well? Clarifying the role of β versus $\lambda$ in practice (and whether one can simply set β=0 and treat $\lambda$ as a replacement) would help practitioners understand how to configure the training objective.

4. Since the closed-form strategy is tied to the squared loss, have the authors considered approximating or extending it to other losses? For instance, is there an analogue for cross-entropy (perhaps using a softmax pseudo-inverse or a one-step Newton update for the last layer)? The results on CIFAR-100 are intriguing in that the squared-loss with closed-form updates actually outperformed cross-entropy SGD. In contrast, on ImageNet the cross-entropy still had the edge. What do the authors believe explains this difference? Is it the larger number of classes, the maturity of cross-entropy tuning, or something about the loss landscapes? Any insight here could point to how one might combine the benefits of closed-form updates with the cross-entropy loss (or whether that is a promising direction at all). It would be interesting to know if a hybrid approach was attempted, for example, training with the closed-form $\ell_2$ method and then fine-tuning or calibrating with cross-entropy, and how that performed.

5. The finding that Adam performed worse than SGD in this framework is thought-provoking. Could the authors shed more light on why an adaptive optimizer or momentum might interfere with the closed-form last layer updates? Does the closed-form update essentially act like a large adaptive step for the last layer, making additional momentum unnecessary or even harmful for the backbone? It would be useful to know if the authors tried variants like adding momentum to the backbone SGD, or if they have recommendations on optimizer choice. Understanding this could help users avoid combinations that degrade performance, and it might reveal interesting interactions between fast two-timescale updates and optimizer internal dynamics.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper exploits the fact that the square loss of a neural network is typically quadratic as a function of the last layer weights, therefore optimal values of the weights can be expressed in closed form (as a function of the other parameters).
They show that optimization of the other parameters does not require differentiating through the closed form solution, and provide a few fixes to the problem of overfitting small batches by the solution. 
Some experiments on PDE and image classification show that the proposed method outperforms SGD on square loss, but it does not outperform standard cross entropy in most cases.

### Strengths
- The paper is extremely clear and straightforward.

- The method is very easy to implement.

### Weaknesses
- The idea is not quite new, other papers had used it even if some details are different.

- Theoretical contributions are weak. Theorem 1 and 2 are trivial. I’m sure you can find them in many previous papers, perhaps in slightly different contexts. They are not really “Theorems”, they are just “chain rule.”

- The section about NTK and Theorem 3 and 4 are also trivial. It’s obvious that a non-convex loss may have stationary points that are not global minimisers, and that is resolved by the NTK. This is all very well known. 

- Experimental results are underwhelming. Although the comparison with SGD on square loss is somewhat fair from the scientific point of view, it’s very weak, and it's not clear whether this method will be ever be practical at all. Cross-entropy wins most of the time, and it’s not clear what would happen when the method is compared with and/or extended to other optimisers that outperform SGD by large amounts. 

Minor:

- I disagree with the statement: “Without the regularization to previous last layer solutions, our method is analogous to putting a large learning rate on the last layer.” If you put a large learning rate on W then optimization of W destabilize, even if loss is quadratic. Instead, your method is equivalent to a Newton step, that converges to the minimum of a quadratic loss in one step.

- I believe Eq.(21) has a typo in the LHS second term, it should be a first derivative. 

- In the non-stochastic setting, it should be possible to show that Eq.(9) is necessarily better than standard gradient descent. That would have been a nice result (although still quite unsurprising).

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces new optimization methods that treat the last layer weights as an explicit function of the backbone parameters and optimize this function with respect to only the backbone parameters. The authors also extend this framework to a stochastic gradient descent (mini-batch) version. They prove the convergence guarantee under the Neural Tangent Kernel (NTK) regime. Experimental results demonstrate that the proposed methods perform better compared to baseline SGD across a variety of tasks.

### Strengths
- Rigorous derivation of closed-form last-layer optimization methods.
- The extension to the SGD mini-batch setting with proximal loss is an interesting derivation.
- Empirical results demonstrate better efficiency and accuracy compared to standard SGD across multiple tasks.
- Theoretical analysis under the Neural Tangent Kernel (NTK) regime provides solid convergence guarantees in the infinite-width limit.

### Weaknesses
- Typos and formatting issues detract slightly from readability and presentation quality.
- The paper lacks a clear explanation of how the NTK regime theoretical analysis directly supports convergence guarantees of the two optimization methods introduced above.

Typos:
- In line 267, equation (18), the author should include the sum of all square losses
- Line 289-290: “the initial function neural network function $\phi$...”
- Line 293-294, “The following result shows that if we make the slightly stronger assumption $\textbf{that}$ the NGPK is positive definite...”

Figures and references format:
- Page 12 and page 21

### Questions
- In Figure 1, why does the curve for $W^\ast(\theta)$  appear to be multi-valued for some values of $\theta$, while later it is treated as a function of $\theta$? Could the authors clarify this discrepancy?
- In equation (13), should the matrices $X$ and $Y$ correspond to mini-batch subsets $\mathcal{B}_t$ instead of the full datasets?
- In the experiments, is the "$l_2$ c.f. ridge ($\beta$)" method implemented with mini-batches, or is it full-batch as suggested by the equation (9)?
- How does the NTK regime theoretical analysis relate to the convergence theory of the two optimization methods introduced before?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes an optimization method that leverages the closed-form solution for the linear last layer under a squared loss. The method treats the last layer as a function of the backbone parameters and optimizes only the backbone, which is shown to be equivalent to alternating gradient steps on the backbone with closed-form updates on the last layer.

**Problem formulation**

The paper considers a model $f(x;W,\theta)=W\phi_{\theta}(x)$, where $\phi_{\theta}$ is the neural network backbone and $W$ is the last linear layer. Since the optimal $W^{\star}(\theta)$ for a fixed $\theta$ is known in closed-form via ridge regression, the authors reformulate the problem to optimize the backbone parameters $\theta$ by minimizing the loss $\mathcal{L}^{star}(\theta):=\mathcal{L}(W^{\star}(\theta),\theta)$.

**Main results. Provide one or two sentence summary**

The key theoretical result is that optimizing this reformulated loss does not require backpropagation through the complex closed-form solution; by the envelope theorem, the gradient $\nabla_{\theta}\mathcal{L}^{*}(\theta)$ is simply $\nabla_{\theta}\mathcal{L}(W^{\star},\theta)$. This property is extended to a practical stochastic (proximal) version of the loss, and the method is proven to converge to a global minimum in the NTK regime.

**Technical approach**

To adapt the method for stochastic gradient descent and prevent overfitting the last layer to minibatches, the authors introduce a proximal loss that regularizes the batch solution against the previous last layer estimate $W_t$. The practical algorithm (Algorithm 1) first updates the backbone parameters $\theta_t$ via a standard gradient step (using $W_{t-1}$), and then computes the new last layer $W_t$ using the closed-form solution of this proximal loss based on the current batch and updated backbone $\theta_t$.

**Experiment**

The proposed proximal method is shown to outperform standard SGD on a squared loss across regression (Fourier Neural Operators, DFIV) and classification (CIFAR, ImageNet) tasks. The approach is particularly effective and stable across all batch sizes, unlike a naive closed-form ridge solution which performs poorly on small batches.

### Strengths
The paper develops a practical and stable proximal-based algorithm that effectively leverages the closed-form last layer solution to accelerate training in stochastic, small-batch settings where naive closed-form updates would otherwise fail. Both theory and experiments are solid

### Weaknesses
.

### Questions
.

### Soundness
3

### Presentation
3

### Contribution
3
