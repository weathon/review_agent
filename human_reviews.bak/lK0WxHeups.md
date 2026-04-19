# Iteration and Stochastic First-order Oracle Complexities of Stochastic Gradient Descent using Constant and Decaying Learning Rates

- Decision: Reject
- Scores: 3, 3, 3, 1

## Abstract
The performance of stochastic gradient descent (SGD), which is the simplest first-order optimizer for training deep neural networks, depends on not only the learning rate but also the batch size. They both affect the number of iterations and the stochastic first-order oracle (SFO) complexity needed for training. In particular, the previous numerical results indicated that, for SGD using a constant learning rate, the number of iterations needed for training decreases when the batch size increases, and the SFO complexity needed for training is minimized at a critical batch size and increases once the batch size exceeds that size. This paper studies the relationship between batch size and the iteration and the SFO complexities needed for nonconvex optimization in deep learning with SGD using constant/decay learning rates. We show that SGD using a step-decay learning rate and a small batch size reduces the SFO complexity to find a local minimizer of a loss function. We also provide numerical comparisons of SGD with the existing first-order optimizers and show the usefulness of SGD using a step-decay learning rate and a small batch size.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores the impact of batch size on the iteration and gradient oracle complexities of the stochastic gradient descent (SGD) algorithm. The objective of the study is to examine how different batch sizes affect the performance of SGD. The paper is written in a reader-friendly manner, making it easily understandable.  In Tables 1 and 2, the authors present a summary of the iteration and gradient oracle complexities of the SGD method using various commonly used step sizes. By presenting this information, the authors offer valuable insights into the behavior of the algorithm. Furthermore, the authors conduct numerical experiments to compare the effectiveness of the step-decay strategy with other optimization algorithms. Through these experiments, they demonstrate the superior performance of step-decay in optimizing the objective function. This finding suggests that step-decay can be a preferable choice when implementing optimization algorithms. However, the contributions of this paper are not sufficient and some of the statements are wrong.

### Strengths
This paper provides a thorough investigation into the relationship between batch size and the complexities of the SGD algorithm. The authors present their findings in a clear and concise manner, making them accessible to readers.

### Weaknesses
The paper is not well ready yet and contributions are trivial. As I checked, the analysis of SGD is quite simple and there is no technical challenge in the analysis. Besides, the main statements on step-decay are wrong. Please see the reasons below.

To calculate the iteration and gradient oracle complexity of the step-decay method, it is crucial to consider the impact of the lower bound values, denoted as $\underline{\alpha}$ and $T$ (representing the length of each stage for step-decay). Unfortunately, the authors of the paper have overlooked this important aspect, which is an incorrect approach. The reason why considering the lower bound values is essential lies in the relationship between $\underline{\alpha}$, $T$, and the total number of iterations, denoted as $K$. Specifically, we have the inequality $\underline{\alpha} \leq \alpha \eta^{p-1}$, where $p = K/T$. This inequality implies that the lower bound value $\underline{\alpha}$ should be taken into account when determining the iteration and gradient oracle complexities of the step-decay method. Ignoring this relationship can lead to flawed conclusions and inaccurate assessments of the algorithm's performance. Therefore, the related complexities results on step-decay are wrong. 

Other weaknesses or typos:
1. In the abstract, the authors made a claim that "SGD using a step-decay learning rate and a small batch size reduces the SFO (Stochastic First-Order) complexity to find a local minimizer of a loss function." However, upon reviewing the paper, it becomes apparent that the study primarily focuses on demonstrating the convergence of SGD to a stationary point rather than specifically proving convergence to a local minimizer.

### Questions
See the weakness above

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the stochastic first-order oracle complexity (SFO, defined by this paper) of SGD with diminishing and constant learning rates. It shows that SGD using a step-decay learning rate and a small batch size achieves the best performance in terms of SFO complexity. Numerical experiments are provided.

### Strengths
1. The paper is well-written and I enjoy reading it.

2. Diminishing learning rate is commonly adopted in deep learning, and it is thus important to study it.

### Weaknesses
1. The stochastic gradient is generated in a different way from practice, and I think this should be highlighted. Specifically, in each iteration, this paper assumes the stochastic gradient is chosen as an ensemble as individual gradients sampled with replacement from some distribution. However, in deep learning practice, the individual gradient is sampled without-replacement and this leads to a gap between practice and the presented theory.

2. I do not think SFO is a reasonable measure. In practice, different individual gradients are calculated parallelly and the corresponding time does not accumulate across samples.

3. The definition of K_{\epsilon} and N_{epsilon} seems to be weird, since the learning rate does not appear in any side of the equation.

4. I wonder what is the novelty of Theorem 3.1. Is not it a very basic analysis of SGD?

5. I find the result of Decay 4 problematic. Specifically, in Theorem 3.2, isn't $\underline{\alpha}$ itself depends on $K(b)$? How can $K(b)$ be further calculated by $\underline{\alpha}$? That being said, when $T$ is independent of $\epsilon$, T is in the same order as $\varepsilon$ as P. Therefore, $\underline{\alpha}$ depends exponentially on $K$. Applying this to Theorem 3.2, it indicates $K(b)$ is also exponentially dependent over $\varepsilon$ and contradicts Theorem 3.4.

### Questions
1. On page 3, Is $N_{\epsilon}$ just $bK_{\epsilon}$?If yes, why not use the simpler one?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the complexity of computing stationary points with SGD 
using a variety of step-size schedules. The authors derive convergence rates 
with an explicit dependence on the batch-size for SGD with a constant step-size
as well as polynomial decay and step-decay step-size schedules. These rates 
are then converted into iteration and oracle complexities and studied as a 
function of the mini-batch size. 
The authors prove that the parameterized complexities are convex functions
and use this to derive optimal batch-sizes for different schedules.
The submission concludes with experiments comparing different schedules on 
CIFAR-10 and CIFAR-100.

### Strengths
The main strength of this paper is its novel approach to hyperparameter
tuning for SGD. While it is typical to tune (at least in theory) the step-size 
parameter to minimize the oracle complexity, maintaining an explicit dependence
on the mini-batch size in the convergence rate and using this to understand
the trade-offs between iteration and oracle complexity is an interesting idea.
In addition to this, the paper has the following strengths:

- The authors provide a simple and clear analysis for SGD which covers
    SGD with a fixed step-size, polynomial decay, and step-decay
    schedules. 

- Although the optimal batch-sizes for fixed step-sizes and polynomial decay schedules
    depend on unknown parameters of the problem, understanding the optimal values
    may allow for new heuristics for selecting the batch-size in practice.

- The experiments, although simple, generally reflect the theory and show that
    tuning the batch-size can lead to improvements in optimization given a fixed
    budget of gradient evaluations.

### Weaknesses
This paper has several significant weaknesses that should be addressed before
publication. In particular,

- The convergence rate given for SGD with the step-decay schedule is misleading
    and leads to incorrect iteration and oracle complexities for this method.

- The paper does not address the fact that $b \leq n$ must be maintained, where
    $n$ is the number of functions in the finite sum. As a result, the optimal
    batch-sizes which the authors derive may not be attainable depending on the 
    desired precision of the solution. For example, as $\epsilon \rightarrow 0$,
    $b \rightarrow \infty$ for a fixed step-size.

- The manuscript is unnecessarily "mathy" and many equations could be 
    omitted while maintaining the same results. See, for example, Equations 1 and 2.
    While this makes the writing seem superficially impressive, it is difficult to
    read and detracts from the flow of the paper. 

- None of the experiments serve to verify the theoretical derivations.
    I would liked to see at least one synthetic experiment for which the problem
    constants are known (e.g. a simple quadratic) and $b^*$ can be computed.
    Plots similar to that in Figures 1/2 could then show that $b^*$ does, in fact,
    obtain the optimal oracle complexity as claimed.

Given these issues, I cannot recommend accepting the submission at this time.
However, I am willing to increase my score if they address these issues. At
a very minimum, I feel the problem with the complexity of the step-size schedule
must be resolved. See "Questions" below for more details.

### Questions
- "SGD using a decaying learning rate...": some additional comment on the type
    of learning rate decay is needed here. SGD with step-size $\alpha_k = 1 / \log(k)$ 
    does not converge as $O(1/\sqrt{K})$ despite $\alpha_k \rightarrow 0$.

- First math display in Section 1.3.2: this bound should mentioned somewhere that 
    $b > n$ isn't possible and $b = n$ reduces to full-batch gradient descent.
    As a result, it is not always feasible to select the batch-size to minimize
    the oracle complexity. 

- "Accordingly, small batch sizes are appropriate for a decaying learning rate or a 
    step-decay learning rate": why is this true? You have said that SFO complexity
    has no positive stationary point, but that doesn't imply it is increasing
    in $b$ or that a small batch-size minimize the complexity over the positive
    integers. Can you please address this fact?

- Equation (2) and Table 2: It is somewhat confusing to switch from measuring 
    convergence using the squared gradient norm in Table 1 to convergence of 
    just the gradient norm in Table 2 and Equation (2).

- Theorem 3.1: I am concerned by the presentation of the convergence rates for
    SGD with step-decay. Firstly, $\underline{\alpha}$ does not appear to be
    defined anywhere. From the proof in the appendix, it seems
    $\underline{\alpha} = \alpha_{K-1} = \alpha \eta^{K/T-1}$.  This quantity
    depends on $K$ --- it is exponentially decreasing every $K/T$ iterations
    --- so that it is incorrect to write it as a constant factor.  Similarly,
    $D_3$ depends on $T$, which may or may not have a relationship with $K$
    depending on algorithm parameters. 

    Only be carefully optimizing over $T$ can a final rate of convergence be
    obtained. Wang et al. [1] set $T = K / \log_{\eta}(K)$ to obtain a final
    convergence rate of $O(\log(T)/\sqrt{T})$.  In contrast, treating
    $\underline{\alpha}$ as a constant leads to an deceptive presentation of
    the convergence rate in Table 1. Moreover, I am fairly certain the
    complexity of $O(1/\epsilon^2)$ for computing an $\epsilon$ stationary
    point in Table 2 is incorrect and violates lower bounds due to Drori and
    Shamir [2]. 

- Theorem 3.4: In addition to the issue with the complexity of step-decay raised
     previously, this theorem assumes that $b \leq n$ can be chosen arbitrarily
     large in order to obtain the desired complexity. For example, SGD with a 
     constant step-size requires $b \geq 2 C_2 \epsilon$, which diverges to 
     infinity as $\epsilon \rightarrow 0$. But this is not sensible because
     $n$ is assumed to be a fixed, finite number of training examples.
     If this is not the cause, then the authors must specify somewhere that they
     assume a setting where $n$ can be taken arbitrarily large. 

### References

[1] Wang, Xiaoyu, Sindri Magnússon, and Mikael Johansson. "On the convergence
of step decay step-size for stochastic optimization." Advances in Neural
Information Processing Systems 34 (2021): 14226-14238.

[2] Drori, Yoel, and Ohad Shamir. "The complexity of finding stationary points
with stochastic gradient descent." International Conference on Machine
Learning. PMLR, 2020.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This manuscript studied the effects of batch size and learning rate for nonconvex smooth optimization. The authors established iteration complexity and SFO (Stochastic First Order Oracle) complexity of the problem.

Despite the result is interesting, most of the results are known (or straightforward extension) in the literature.

### Strengths
The paper is well-written.

### Weaknesses
The paper's theoretical results are known in the literature (e.g., [r1], [r2]). The hardness result [r3] says that: whatever batch size the SGD algorithm can choose, the SFO cannot be better than $O(1/\epsilon^4)$.

[r1] Ghadimi, Saeed, and Guanghui Lan. "Stochastic first-and zeroth-order methods for nonconvex stochastic programming." SIAM Journal on Optimization 23, no. 4 (2013): 2341-2368.

[r2] Ghadimi, S., Lan, G., & Zhang, H. (2016). Mini-batch stochastic approximation methods for nonconvex stochastic composite optimization. Mathematical Programming, 155(1-2), 267-305.

[r3] Arjevani, Yossi, Yair Carmon, John C. Duchi, Dylan J. Foster, Nathan Srebro, and Blake Woodworth. "Lower bounds for non-convex stochastic optimization." Mathematical Programming 199, no. 1-2 (2023): 165-214.

### Questions
Can you describe how your approach is better than the references I gave above (e.g., [r1, r2])?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
