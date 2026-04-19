# STIMULUS: Achieving Fast Convergence and Low Sample Complexity in Stochastic Multi-Objective Learning

- Decision: Reject
- Scores: 3, 5, 8, 5, 3

## Abstract
Recently, multi-objective optimization (MOO) problems have received increasing attention due to their wide range of applications in various fields, such as machine learning (ML), operations research, and many engineering applications. However, MOO algorithm design remains in its infancy and many existing MOO methods suffer from unsatisfactory convergence performance. To address this challenge, in this paper, we propose an algorithm called STIMULUS (**ST**ochastic path-**I**ntegrated **MUL**ti-graident rec**U**rsive e**S**timator), a new and robust approach for solving MOO problems. Different from the traditional methods, STIMULUS introduces a simple yet powerful recursive framework for updating stochastic gradient estimates. This methodology improves convergence performance by reducing the variance in multi-gradient estimation, leading to more stable convergence paths. In addition, we introduce an enhanced version of STIMULUS, termed STIMULUS-M, which incorporates the momentum term to further expedite convergence. One of the key contributions of this paper is the theoretical analysis for both STIMULUS and STIMULUS-M, where we establish an $\mathcal{O}(\frac{1}{T})$ convergence rate for both methods, which implies a state-of-the-art sample complexity of $O\left(n+\sqrt{n}\epsilon^{-1}\right)$ under non-convexity settings. In the case where the objectives are strongly convex, we further establish a linear convergence rate of $\mathcal{O}(e^{-\mu T})$ of the proposed methods, which suggests an even stronger $\mathcal{O}\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ sample complexity. Moreover, to further alleviate the periodic full gradient evaluation requirement in STIMULUS and STIMULUS-M, we further propose enhanced versions with adaptive batching called STIMULUS$^+$/STIMULUS-M$^+$ and provide their theoretical analysis. Our extensive experimental results verify the efficacy of our proposed algorithms and their superiority over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The focus of the paper is designing multi-objective optimization (MOO) algorithms with faster convergence rates compared to existing SOTA methods (and matching deterministic MOO counterpart) , for non-convex and strongly convex settings. The paper leverage variance reduction techniques to achieve the aforementioned faster convergence rates, which were not reported previously in MOO literature. The authors also provide empirical results, comparing the proposed method with prior MOO baselines, and show improved empirical performance as well.

### Strengths
* The proposed idea of incorporating variance reduction methods to improve convergence rate in MOO setting seems promising.
* The authors provide some theory (which is unclear, as described in next section) and experiments to validate the proposed method.

### Weaknesses
* The definition of Pareto optimality and Pareto stationarity does not seem to align with the metrics used in the convergence results. For example, while the authors claim the convergence to a Pareto stationary point by STIMULUS due to the result obtained in Theorem 1, it is unclear why the merit function used in this result can measure the Pareto stationarity of iterates.

* Due to the problem mentioned above, it is unclear whether the comparison for theoretical results provided in Table 1 is a fair one.

* In proof of Lemma 1, the authors use Lemma 1 of Feng et al. (2018), yet it is hard to see how result in Feng et al. (2018) can be used here, since the problem setting in Feng et al. (2018) is single objective optimization.

* The choice of stepsize in Theorems is unclear. For example, how does one go from equation (13) to (14) (in proof provided in appendix) by the choice of step size $\eta \leq 1/2$ ? 

Minor comments:

* $|\mathcal{A}|$ in equation (2) is not defined before using.
* Using index $s$ in equation (3) seems not necessary.

### Questions
* Can the authors explain the relationship between the merit functions used in Theorems 1-6, and the definitions of Pareto stationarity/optimality?

* Can the authors elaborate on why the inequality (9) (in proof of Lemma 1) hold, and how it relate to Lemma 1 in Fang et al. (2018) ?

### Soundness
2 fair

### Presentation
3 good

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
This paper proposes to use variance reduction techniques to improve the sample complexity of stochastic multi-objective learning in finite-sum problems. It achieves the state-of-the-art sample complexity, matching the one with full-batch gradient descent.
Experiments on some benchmark datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper studies MOO in the finite-sum problem, which has not been extensively considered in MOO literature before as far as I know.

2. This paper proposes a variance-reduced algorithm that improves the state-of-the-art sample complexity of existing algorithms for multi-objective finite-sum problems.

### Weaknesses
1. The comparison with existing algorithms in Table 1 may not be fair because they are not focused on the same settings. The setting analyzed in this paper is the finite-sum setting which is more restrictive.

2. The benefit of the proposed method over linear scalarization in MOO is unclear. This is because linear scalarization can also achieve convergence to Pareto stationary points. By applying variance reduction techniques such as SVRG to linear scalarization, it can achieve a similar convergence rate to Pareto stationary points as this paper.
Therefore, only providing convergence to Pareto stationary points is not enough to show the benefit of the proposed method over the simplest linear scalarization.
More discussion should be provided.



3. Quantitative results are too limited to understand the practical performance of the proposed method.
Also, in addition to performance of each task, a widely used measure is $\Delta m $\% (e.g. in MOCO paper) to show the overall performance on all tasks. 


### Minor

1. Some notations or definitions are not clear. See **Questions-2**.

2. In Section 2 - 2) overview of MOO algorithm, it is inaccurate to say that "recent work such as (Fernando et al., 2022) uses bi-level formulation to mitigate bias". In fact, (Fernando et al., 2022) uses momentum-based methods to mitigate bias, and apply to bi-level optimization problems.


3. Typos

- Below Definition 2: "non-convex MOO probolems" -> "non-convex MOO problems"

- Below Theorem 1: "sample compleixty" -> "sample complexity"

### Questions
1. What is the benefit of the proposed algorithm compared to applying variance-reduced algorithms such as SVRG to linear scalarization in MOO? In other words, applying such algorithms can also achieve similar sample complexity or convergence rate to Pareto stationary points. Therefore, the benefit of using the proposed stochastic variant of MGD is unclear.

2. Some notations are not defined clearly. See below.

- In Definition 3, Theorem 2 and 4, what is $i$ in $\lambda_i^s$? Shouldn't it be $\lambda_t^s$?

- What is $\xi$ in Eq.(6)? In Eq.(6), are you missing a sum of all samples $\xi \in \mathcal{N}_ s$?

- In Definition 4, what is "incremental first-order oracle (IFO)"? I know it is a widely used concept in finite-sum problems, but it is better to provide a formal definition or at least some references for completeness.
In addition, it could benefit to introduce finite-sum problems and IFO earlier to provide some context for readers.

- In Algorithm 1, line 5, it says "compute $\mathbf{u}_ t^s$ as in Eq.(4)", but Eq.(4) computes $\lambda_ t^s$, is this a typo?



3. Why only non-convex and strongly-convex cases are analyzed? What is the rate for convex cases? Are there any additional challenges to analyzing convex cases? It would be better to provide some discussion on this aspect.


4. Below Table 1, it mentions $\mathbf{x}^*$ is the Pareto-optimal point. However, there can be multiple Pareto-optimal points with different function values. This will result in the term $||\mathbf{x}_ 0 - \mathbf{x}_ *||$ not well defined in Theorem 2. Could you elaborate more on this? 

5. The measure $\sum_{s\in [S]} \lambda_t^s [f_s(x_t) - f_s(x_*)]$ has some issues because it can be negative. See more discussions in (Liu & Vincente 2021). You need to make additional assumptions to make this a valid convergence metric.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper gives a systematic study on variance-reduction-aided gradient-based algorithms for multi-objective optimization. A new variance reduction multi-gradient estimator is proposed by combining periodic full multi-gradients and recursive correction with batch gradients, followed by a momentum-based variant. The adaptive-batching technique is further introduced to eschew the need of computing full gradients. Theoretical analysis on convergence rate and sample complexity are provided for all the proposed algorithms, showing superiority over previous stochastic multi-gradient algorithms. Experiments on three datasets verify the theoretical claims in this work.

### Strengths
1. This paper conducts a systematic study on the VR-aided multi-gradient method. Various versions of VR-based algorithms are proposed and supported by theoretical analysis, which may inspire future research in this field.

2. This paper is technical sound. The convergence analysis is comprehensive and non-trivial.

3. This paper is well-written in general and easy to follow.

### Weaknesses
1. The presentation of adaptive-batching versions is a bit ambiguous. I am not sure whether the adaptive batch is applied to the $q$-periodic full gradient or to each step. Adding more background knowledge on adaptive batch technique or a diagram for STIMULUS$^+$ would be helpful. In addition, it is unclear how to decide the batch size in experiments.

2. Besides SMGD and MOCO, CR-MOGM (Zhou et al., 2022b) should also be considered in experiments as a SOTA method.

### Questions
My main concerns are given in the weaknesses part.

### Soundness
4 excellent

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers multi-objective learning problems based on gradient methods. The paper introduce a novel stochastic gradient methods with variance-reduction to minimize multi-objective learning problems. The algorithm is a variant of the spider algorithm in Fang et al 2018 from single-objective learning to multi-objective learning. The algorithm first builds a common descent direction based on stochastic gradients, using the recursive gradient estimates to reduce variance. The paper further improves the efficiency by introducing the momentum scheme and the adaptive batching. Theoretical convergence and sample complexity are present for both nonconvex and strongly convex problems, under a smoothness assumption on loss functions. Experimental results are also presented to verify the efficiency of the proposed algorithm.

### Strengths
The paper introduce several stochastic algorithms for multi-objective optimization problems, which are more challenging than the single-objective problems. The paper The algorithms have better convergence rates and sample complexity than the existing results. The paper is clearly written and the main results are clearly presented.

### Weaknesses
As far as I see, the theoretical analysis seems to be problematic. For example, Theorem 1 gives convergence rates on $\frac{1}{T}\sum_{t=0}^{T-1}\|d_t\|^2$. However, the terms $d_t$ are just common descent directions built based on stochastic gradients (which is similar to the stochastic gradient in SGD). According to Definition 3 and the paragraph above, the quantity to our interest is $d=\lambda^\top\nabla F(\mathbf{x})$. note that $F(\mathbf{x})$ are the true objective functions, instead of the stochastic functions randomly sampled in the optimization process. Therefore, Theorem 1 does not give convergence rates on the $\epsilon$-stationarity, and the convergence in terms of $\|d_t\|^2$ does not show the real behavior of the algorithm. Furthermore, as far as I see from the proof of Theorem 1, one can get convergence rates of $\|d_t\|^2$ if only $q=|\mathcal{A}|$, even if $q$ is very small. In this case, one can choose very $q$ to derive the same convergence rates for $\|d_t\|^2$, but with much less sample complexity.

Definition 3 implicitly assumes that all $f_s$ should have the same minimizer $x_*$, which is a very strong assumption. In multi-objective optimization, it is very unlikely that we have the same minimizer for all tasks. Then, the convergence rates for strongly convex problems are restrictive.

### Questions
Can we derive convergence rates in terms of $d_t=\lambda^\top \nabla F(\mathbf{x}_t)$? Indeed, the convergence of $\lambda^\top \nabla F(\mathbf{x}_t)$ reflects the convergence behavior of the algorithm.

Can we relax the assumption in Definition 3 by letting the $t$-th task have a minimizer $\mathbf{x}_*^t$, i.e., each task has its own minimizer?

In Corollary 2, if $\epsilon>\mu$, then $\log (\mu/\epsilon)<0$. In this case, it seems that the result would no longer hold?

Minor issues:

- Eq (2): there is a missing summation over $\mathcal{A}$
- Eq (4): there is a missing constraint on the nonnegativity of $\lambda$
- Line 4 of Algorithm 1: Eq (4) does not give formula to compute $u_t^s$

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes  STIMULUS, which can achieve lower sample complexities than existing algorithms.

### Strengths
This paper proposes  STIMULUS, which can achieve lower sample complexities than existing algorithms.

### Weaknesses
There are many typos in this paper. Some proofs of this paper are unclear. 
1. Eq. (23) sums both sides of Eq. (22) weighted with  $\lambda_t^s$  from $s\in S$. But why $\frac{1}{2\delta} \|\nabla f_s(x_t) - u_t^s\|^2$ is not weighted with $\lambda_t^s$?
2. Why does it hold that $\|\nabla f_s(x_t) - u_t^s\|^2 = \sum_{...} \|x_{i+1} - x_i\|^2 + \| \nabla f_s(x_{(n_t−1)q}) − u^s_{(n_t−1)q}\|^2  $ in Eq.(23)
3. In the Definition 3, why should $\mathbb{E}  [\sum_{s} \lambda_i^s (f_s(x_t) - f_s(x_*))]$  be non-positive? This is not pointed out and proved in this paper. If this value is not non-positive, it is less than $\epsilon$ is not meaningful. Furthermore, what is the meaning of $i$ in the notation $\lambda_i^s$.
4. In the Line-7, it should be ``gradient'' other than ``graident''.
5. This paper consider the case that $f_s(x)$ are of the finite sum form. However, detailed description of finite sum form is lacked.

### Questions
No

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
