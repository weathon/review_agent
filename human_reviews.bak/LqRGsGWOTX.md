# Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis

- Decision: Accept (spotlight)
- Scores: 6, 6, 6, 8, 8

## Abstract
Bilevel optimization is an important formulation for many machine learning problems, such as meta-learning and hyperparameter optimization. Current bilevel optimization algorithms assume that the gradient of the upper-level function is Lipschitz (i.e., the upper-level function has a bounded smoothness parameter). However, recent studies reveal that certain neural networks such as recurrent neural networks (RNNs) and long-short-term memory networks (LSTMs) exhibit potential unbounded smoothness, rendering conventional bilevel optimization algorithms unsuitable for these neural networks. In this paper, we design a new bilevel optimization algorithm, namely BO-REP, to address this challenge. This algorithm updates the upper-level variable using normalized momentum and incorporates two novel techniques for updating the lower-level variable: \textit{initialization refinement} and \textit{periodic updates}. Specifically, once the upper-level variable is initialized, a subroutine is invoked to obtain a refined estimate of the corresponding optimal lower-level variable, and the lower-level variable is updated only after every specific period instead of each iteration. When the upper-level problem is nonconvex and unbounded smooth, and the lower-level problem is strongly convex, we prove that our algorithm requires $\widetilde{O}(1/\epsilon^4)$ \footnote{Here $\widetilde{O}(\cdot)$ compresses logarithmic factors of $1/\epsilon$ and $1/\delta$, where $\delta\in(0,1)$ denotes the failure probability.} iterations to find an $\epsilon$-stationary point in the stochastic setting, where each iteration involves calling a stochastic gradient or Hessian-vector product oracle. Notably, this result matches the state-of-the-art complexity results under the bounded smoothness setting and without mean-squared smoothness of the stochastic gradient, up to logarithmic factors. Our proof relies on novel technical lemmas for the periodically updated lower-level variable, which are of independent interest. Our experiments on hyper-representation learning, hyperparameter optimization, and data hyper-cleaning for text classification tasks demonstrate the effectiveness of our proposed algorithm. The code is available at [https://github.com/MingruiLiu-ML-Lab/Bilevel-Optimization-under-Unbounded-Smoothness](https://github.com/MingruiLiu-ML-Lab/Bilevel-Optimization-under-Unbounded-Smoothness).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the bilevel optimization under unbounded smoothness and proposes a new algorithm based on the SOBA method. This paper gives the convergence analysis under the unbounded smoothness and the experimental results demonstrate the superiority of the proposed method.

### Strengths
1. Under the unbounded smoothness, this paper obtains a similar convergence rate as that in other papers.
2. The experimental results demonstrate the superiority of the proposed method.

### Weaknesses
1. Too many loops in Algorithm 1,2 which may lead to time-consuming.
2. Some symbols are confusing. Both Algorithms 1, and 2 use K. In lemmas, what is $K_0$?
3. Too many hyperparameters in algorithms. More experimental results of different hyperparameters are needed.

### Questions
1. How will the iteration numbers affect the convergence performance? Is the inner loop necessary? Because this method seems to be a momentum method based on SOBA, why author use an inner loop? why not just update $y$ for a single step?
2. Why only use the SOBA in [1]. SABA also proposed in [1]. I think it should be compared.
3. How to ensure the problem considered in experiments is unbounded smooth? Because the applications considered in this paper are usually considered in other papers with bound smoothness assumption.


[1] Dagréou M, Ablin P, Vaiter S, et al. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms[J]. Advances in Neural Information Processing Systems, 2022, 35: 26698-26710.

### Soundness
3 good

### Presentation
2 fair

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
This paper investigates a bilevel stochastic optimization problem in which the gradient of the upper-level objective function is potentially unbounded, and the lower-level objective function is strongly convex. 
The authors introduce a new algorithm, which integrates the normalized momentum technique for updating the upper-level variable and employs initialization refinement as well as periodic updates techniques for updating the lower-level variable. The authors also offer a theoretical complexity analysis. To substantiate the efficacy of the proposed algorithm, the paper includes a set of numerical experiments.

### Strengths
1. The paper is clear, well organized, and easy to follow.

2. The incorporation of the normalized momentum update for the upper-level variable, as well as the utilization of initialization refinement and periodic updates for the lower-level variable within the proposed algorithm weaken the bounded smoothness requirement of the upper-level objective function.

3. Simulations have shown the empirical advantage of the proposed algorithm.

### Weaknesses
1. The theoretical complexity result of the proposed algorithm, Theorem 1, lacks persuasiveness. Specifically, the choice of the iteration number $N$ in Algorithm 2 for lower-level variable updates is left unspecified. Furthermore, the selection of the parameter $I$ appears improper, particularly in comparison to the selection of the parameter $K$. This peculiarity arises because, according to Theorem 1, $I = O(1/\epsilon^2)$, while $K = O(1/\epsilon)$. Consequently, for sufficiently small values of $\epsilon$, $I$ is significantly larger than $K$. However, both the implementation of the proposed algorithm and the theoretical analysis, as exemplified by Lemma 2, necessitate $I$ to be smaller than $K$.

2. Despite the authors' assertion that they do not require the boundedness of the norm of the gradient of the upper-level objective function, $ || \nabla_y f(x, y) ||$, as stipulated in Assumption 2 (i), it is worth noting that they still demand the boundedness of $|| \nabla_y f(x, y^*(x)) ||$. The rationale behind the relative ease of achieving this condition in practical scenarios, as opposed to the former requirement, remains unclear. Moreover, it remains unaddressed whether the problems examined in the numerical experiments exhibit unbounded smoothness yet satisfy Assumption 2 (i).

3. The selection of the parameters $I$ and $N$ in the implementation of the proposed algorithm in the numerical experiments deviates from the parameters required by the theoretical analysis. Specifically, the theoretical analysis mandates that both $I$ and $N$ should be large as $\epsilon$ becomes small. However, in the context of the numerical experiments, the authors have set the values for $I$ and $N$ as merely 2 or 3, which contradicts the theoretical analysis.

### Questions
1. The proposed algorithm incorporates an initialization refinement procedure for the lower-level variable. I am curious about whether the authors have taken into account the computational cost associated with this initialization refinement procedure when presenting the numerical results in Figure 1.

2. While the proposed algorithm exhibits advantages over its competitors in terms of the number of epochs required, how about its performance concerning computation time?

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes bilevel algorithm, BO-REQ to address unbounded smoothness challenge. Specifically, the authors introduce the normalized momentum and initialization refinement techniques, and proposed algorithm achieves the convergence rate of $\mathcal{\widetilde O}(1/\epsilon^4)$. The experiments across various settings also demonstrate the effectiveness of the proposed algorithm.

### Strengths
The proposed algorithm outperforms other bilevel baselines in experiments across various contexts.

### Weaknesses
1. This paper relaxes the bounded assumptions but is not based unbounded assumption as stated in the abstract. The example pf RNN and LSTMs are misleading unless the authors are able to prove that these networks satisfy the Assumption 1.

2. The second stated contribution in introduction section is not valid. As the author also mentions in the footnote 3, the SOTA complexity results under bounded smoothness is $\mathcal{\widetilde O}(1/\epsilon^3)$. The proposed method does not achieve the SOTA complexity rate, even after considering the logarithmic factors.

3. Momentum technique has been incorporated in the bilevel optimization area before. For example, MRBO [1] achieves the $\mathcal{\widetilde O}(1/\epsilon^3)$ after momentum introduction. The authors are expected to improve the convergence rate of proposed algorithm.

4. The authors are suggested to show the experimental results w.r.t. time rather than only epochs. The proposed algorithm involves three loops per epoch, which might introduce extra computational cost. It will be more fair to compare proposed algorithm with other single-loop or double-loop based algorithms in terms of running time.

[1] Provably Faster Algorithms for Bilevel Optimization

### Questions
Check the weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates bi-level stochastic optimization problems in which the upper-level function could have an unbounded smoothness parameter. To solve the above problem, the authors proposed to use $(L_{x,0}, L_{x,1}, L_{y,0}, L_{y,1})$-smoothness assumption for the upper-level function, which is a generalization of the relaxed smoothness assumption for single-level optimization. Then they designed a new algorithm named BO-REP for solving bilevel optimization problems with unbounded smooth upper-level functions. BO-REP used normalized momentum technique for updating the upper-level variable and initialization refinement and periodic update tecniques for lower-level variables. The experiments demonstrated the effectiveness of the proposed algorithm.

### Strengths
### Originality
The problem setting of this paper is novel. To the best of my knowledge, this is the first work in the literature of bilevel optimization that takes account into the unbounded smoothness assumption for the upper-level functions. 

The proposed method is novel. To solve the above problem, the authors proposed two novel mechanisms for updating the lower-level variables.

### Quality and Clarity

Overall, this paper is well written, although it is not clear enough in some places.

### Significance

This article explores a new problem and designs new algorithms to solve it. I think this will inspire researchers to thin about bilevel optimization problems from a new perspective and design more efficient algorithms.

### Weaknesses
The main drawback of the proposed algorithm is that the structure is complex, and each subroutine contains additional hyperparameters that need to be tuned.

### Questions
1. In initilalization refinement step, what's the purpose to invoke a multi-epoch algorithm to obtain an accurate estimate of lower-level variables and why periodic update doesn't use such a multi-epoch strategy?

2. The initialization refinement and periodic updates for are carefully designed for obtaining an accurate estimate of the lower-level variables at each iteration. Under current unbounded smoothness assumptions, is that possible to design algorithms with inexact estimate of the lower-level variables? Since the periodic update for the lower-level variables make the proposed algorithm a nested loop one, is there any potential strategies for us to design a single-loop algorithm for solving the unbounded smoothness bilevel optimization problems? 

3. In the experiments, how is the ball radius R chosen?

Minor issues:
1) caption of table: "$\mathcal{C}_L^{a, a}$ denotes a-times differentiability with Lipschitz k-th order derivatives." I think it should be $\mathcal{C}_L^{a, k}$ here.
2) In the pseudo-code for Algorithm 2, the outer loop with the iteration variable k spanning from 0 to K-1 is repeatedly defined, which may lead to ambiguity or confusion for the readers. To enhance clarity, I'd recommend passing the current iteration k as an input to the UpdateLower subroutine (which is a part of the BO-REP algorithm) and subsequently removing the loop from the algorithm.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 5

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a novel algorithm for bilevel optimization called BO-REP. This method achieves $\mathcal{O}(1/\epsilon^4)$ convergence rates to find $\epsilon$-stationary point in the stochastic non-convex setup assuming only a relaxed smoothness assumption. Moreover, this result matches the state-of-the-art complexity results under the more restrictive types of smoothness up to logarithmic factors. The key two technical ingredients - initialization refinement and periodic updates - are easily implementable. Finally, authors demonstrate the practical efficiency of the proposed approach on hyper-representation learning, hyperparameter optimization and data hyper-clearning for text classification.

### Strengths
- The paper is very well-written and easy to follow. Moreover, it clearly demonstrate all the novelties of the proposed approach.
- The complexity result matches the existing state-of-the-art methods with less restrictive smoothness assumption, that is critical for presented applications on RNNs.
- The experimental validation shows the superiority of the proposed method over other methods for bilevel optimization.

### Weaknesses
- Lack of lower bounds does not allow to understand optimality of the proposed method under given assumption;
- Proposed guarantees holds only for finding an $\epsilon$-stationary point in expectation.

### Questions
- Is it possible to relax the strong convexity assumption of the lower function to log-concavity? 
- What are challenges to provide high-probability guarantees for $\epsilon$-stationary point for the given algorithm?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
