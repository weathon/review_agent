# iFedDR: Auto-Tuning Local Computation with Inexact Douglas-Rachford Splitting in Federated Learning

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Federated learning usually requires specifying the amount of local computation needed a priori. In this work, we instead propose a systematic scheme to automatically adjust and potentially reduce the local computations while preserving convergence guarantees. We focus on proximal-based methods, where we demonstrate that the proximal operator can be evaluated inexactly up to a relative error, rather than relying on a predefined sequence of vanishing errors. Our proposed method, iFedDR, is based on a novel error-corrected version of inexact Douglas-Rachford splitting. It mitigates the need for hyperparameter tuning the number of client steps, by triggering refinement on-demand. We derive iFedDR as an instance of a much more general construction, which allows us to handle minimax problem, and which is interesting in its own right. Several numerical experiments are carried out demonstrating the favorable convergence properties of iFedDR.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes iFedDR, which is a horizontal federated learning algorithm based on inexact Douglas-Rachford splitting. The proposed algorithm features adaptive stepsizes and an extragradient correction step. The algorithm includes an online-checkable relative error condition, which enables the server to dynamically request more accurate solutions to the local subproblems from some clients in order to guarantee convergence.

### Strengths
The paper is well-written with a clear flow. The theoretical results are solid and the proofs are easy to follow.

### Weaknesses
1. Lack of discussion of other methods for solving the primal-dual problem (2.6) (or the primal (2.4a) and the dual (2.5)). It would be helpful to include one or two sentences explaining why Douglas-Rachford splitting is chosen over other existing methods. 

2. Some definition/interpretation of important variables are missing, e.g., the interpretation of $s$ in DRS and the definition of $m$ in Theorem 3.1. Please read through the main manuscript and make sure that it is self-contained.

3. The limitations of the proposed method was not discussed. A potential limitation of the proposed algorithm is its applicability in realistic federated learning tasks with deep neural network models, in which many of the assumptions (e.g., convexity) needed in the convergence analysis are violated and thus it is not clear to me if the proposed algorithm could still outperform the standard ones, e.g., FedAvg, FedSVRG, etc. I suggest the authors to add a paragraph discussing the limitations of the proposed method. 

4. The numerical experiments focus on simple logistic regression and linear models. They are good starting points but are structurally very different from the high dimensional nonconvex problems in training deep networks. I suggest to authors to test the method on, for example, training full ResNet18 on CIFAR10 in the federated scenario, and to report the training efficiency and accuracy against standard methods. Without more comprehensive tests, it is difficult to assess the potential impact of the proposed method.

### Questions
1. The authors mention that the proposed algorithm is also applicable to constrained problems but did not discuss much further in this direction. Could the authors comment on what types of constraints (inequality, equality, etc...) can the algorithm handle? Is it simply reformulating constrained problems to the form of (2.1)? How does the number of constraints affect the dimensionality of the reformulated problem? The application to constrained problems may warrant a standalone paragraph for description and discussion, as well as perhaps some experiments to demonstrate the results.

2. How does the algorithm performs when the client loads are imbalanced (clients with different compute power, different amount of data, etc)?

3. Is the proposed method extendable to allow for asynchronous updates?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes iFedDR, which can be thought of as an inexact variant of Dougals-Rachford splitting, equipped with error-correction step via extragradient computation.

### Strengths
- FedAvg is widely still widely used in practice, but its convergence issue in the heterogeneous setting was known, which was “fixed” by methods that employ splitting methods such as FedSplit and FedDR. The proposed method, iFedDR, extends FedDR such that inexact proximal step can be employed, which is a non-trivial and important extension.
- Since the proposed method can be more generally applied to monotone inclusion problems, the provided theory applies to minimax problems.
- The paper is generally well-written, although it is a little confusing to follow different reductions (original problem to inclusion problem, iPPA to iFedDR.)

### Weaknesses
- It would be great if the authors can comment on the computational aspect of the proposed algorithm. In particular, I’m a bit confused about the authors’ assertion of “computationally negligible error correction”; do you mean that $\zeta$ can be ignored in Step I.3 of Algorithm 1? In that case, does the theoretical result still apply?
- Algorithm 1 involves several matrix inversion per iteration and per client (3.3). I understand the manuscript is of theoretical nature, but some comments on the computational aspect would be helpful as the manuscript is cast as an FL algorithm.

### Questions
- If I understand correctly, automatically adjusting the number of location computations needed is a “trade-off” in the sense that instead the step size $\gamma$ needs to be tuned, which is used for both the client update (3.3 in Algorithm 1) and the server update (I.2 in Algorithm 1). Is there a good way to choose this $\gamma$?
- In line of above comment, could the authors comment on other federated learning works that use adaptive step size, such as [2, 3]?
- Based on Figure 1 & Remark 3.2 (v), large value of $\gamma$ leads to less number of communication steps, but increases the number of local iterations needed. Could this be alleviated by decoupling the client step size and the server step size? This would change the classical DR splitting though, but still curious.
- Remark D.3 mentions $\zeta_k$ can be ignored. Is this at least empirically tested? I believe an answer to this question is important for the authors to assert “computationally negligible” error condition, even neglecting all the matrix inversions needed in the algorithm.

[1] Mukherjee et al. (2024) Locally Adaptive Federated Learning

[2] Kim et al., (2024) Adaptive Federated Learning with Auto-Tuned Clients

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a way to automatically adjust the local computation while preserving the convergence guarantees. In particular, it focuses on a new algorithm, iFedDR, which is an inexact version of the Douglas-Rachford approach (FedDR), which is proposed as a correct way to employ the proximal operator in the FL setting. In particular, iFedDR is an extension of previous work that incorporates an adaptive stepsize and an extragradient correction step to allow for the relaxed condition of relative inexactness. The paper focuses on providing convergence guarantees of iFedDR and tests the performance of the new approach in experiments.

### Strengths
The paper is well-written, and the main contributions are clear. To the best of my knowledge, this is one of the first papers to aim to automate the number of local updates needed when clients use proximal updates. 

I also like the fact that the analysis was done from the monotone inclusion setup that allows direct extensions of the ideas to min-max problems and m-player games. The convergence guarantees are proven for a much more general method, iPPPA, which might be interesting in its own right.

The statement of the theorems is as expected, and from a pass on the proofs in the appendix, they look correct. I did not carefully check all the details but all steps are presented clearly and make the read of them easy.

### Weaknesses
I believe the paper would be improved if a table with a comparison with closely related works is presented. How the final convergence guarantees of the paper are related to the existing results? Can the main theorems capture previous convergence analysis as a special case?

I find the split of sections a bit confusing. There is Theorem 3.1 in section 3, but then later, there is a full section 5 devoted to Convergence analysis. Can the authors comment on that? Why not simply have all theorems related to convergence guarantees in one section?
In my opinion, Theorem 5.1 could be the only theorem in the main paper. The theorem 3.1 is only a corollary that can be easily not mentioned anywhere in section 3 (related to the design of the method) without affecting the flow of the paper. 

On plots: The figures in the paper look like screenshots rather than a higher-quality PDF. This makes the labels and names on the axis hard to read (they are blurred).

### Questions
See the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
