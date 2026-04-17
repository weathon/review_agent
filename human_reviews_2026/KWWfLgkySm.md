# Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
In the context of over-parameterization, there is a line of work demonstrating that randomly initialized (stochastic) gradient descent (GD) converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. However, the convergence rate of GD for training two-layer neural networks exhibits poor dependence on the sample size and the Gram matrix, leading to a slow training process. In this paper, we show that for training two-layer $\text{ReLU}^3$ Physics-Informed Neural Networks (PINNs), the learning rate can be improved from the smallest eigenvalue of the limiting Gram matrix to the reciprocal of the largest eigenvalue, implying that GD actually enjoys a faster convergence rate. Despite such improvements, the convergence rate is still tied to the least eigenvalue of the Gram matrix, leading to slow convergence. We then develop the positive definiteness of Gram matrices with general smooth activation functions and provide the convergence analysis of natural gradient descent (NGD) in training two-layer PINNs, demonstrating that the maximal learning rate can be $\mathcal{O}(1)$ and at this rate, the convergence rate is independent of the Gram matrix. In particular, for smooth activation functions, the convergence rate of NGD is quadratic. Numerical experiments are conducted to verify our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors provide a proof for the convergence rate of a certain type of neural network that depends on the norm of the Hessian, which seems to be a new result (or an extension of a known result to a broader class of problems). The paper is clear and well-written, and overall seems like a good contribution to the field of theoretical guarantees for neural network training.

### Strengths
The paper is clear, of a high-technical quality and of a fair significance. It is an extension of a result known for regression problem to more general PINNs (although still restricted to two-layer networks).

### Weaknesses
A few things come to mind:
1) The type of network it is applied to is quite restrictive (two-layer ReLU PINNs). While I understand that this allows you to construct the proof, could there not be a way to extend the result to more general architectures? This would make the result and impact much stronger.
2) It would be good to numerically validate the main result of the paper, where you take a simple problem where you can calculate $1/|H^{\infty}|_2$ exactly and verify that the convergence rate matches it.
3) A natural question that comes to mind is what happens when you have deeper networks. It would be interesting to at least numerically study this question, by replicating your experiments with three and four-layer networks perhaps.

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
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript provides training guarantees for physics-informed neural networks in an overparametrized setting with NTK-scale parametrization, both for first-order gradient descent as well as a second-order natural gradient descent. For gradient descent, the allowed step size of $O(1/\lambda_\max)$ is different compared to prior works with $O(\lambda_\min)$ step size on optimization in PINNs. For natural gradient descent, a convergence rate independent of the condition number of the problem is derived. Additionally, computational experiments comparing first-order to second-order optimizers in PINNs are provided.

### Strengths
+ Overall, the paper is well written and easy to navigate and the main contributions are clearly described. 
+ The paper addresses a timely and important question of the convergence of optimization physics-informed models. 
+ The guarantees go beyond the use of first-order methods and treat natural gradient, which seems to be the arguably most efficient optimizer in PINNs. 
+ The theoretical analysis uses timely methods from supervised learning and rigorously transfers them to PINNs.

### Weaknesses
+ Discussion of related works: 
    + Use of *our NGD*: At multiple places in the manuscript, *our NGD* is used to refer to the natural gradient method that is being analyzed. Also, in Remark 4.2 it is claimed that the method studied is different from previously proposed natural gradient or Gauss-Newton methods for PINNs. The iteration of energy natural gradient (Müller and Zeinhofer, 2023), which reduces to a Gauss-Newton iteration for PINNs, is given by $$ w(k+1)=w(k)-\eta (J(k)^T J(k))^+J(k)^T \binom{s(k)}{h(k)}, $$ where $A^+$ denotes an arbitrary pseudo-inverse of $A$. By choosing the Moore-Penrose inverse and considering an overparametrized setting in which $J(k)J(k)^T$ has full rank, this specializes to $$w(k+1)=w(k)-\eta J(k)^T (J(k)J(k)^T)^{-1} \binom{s(k)}{h(k)},$$ which agrees with the iteration studied in the manuscript. Hence, from my understanding, the manuscript is studying the convergence properties of the optimizer proposed by Müller and Zeinhofer (2023), however, this is not mentioned at any place. Rather, the impression is conveyed that the manuscript proposed a new variant of a natural gradient method, see Remark 4.2 as well as line 408 and Table 1. This 
   + There are certain works on optimization in PINNs missing in the discussion of related works in Subsection 1.2., e.g., 
      + AN OPERATOR PRECONDITIONING PERSPECTIVE ON TRAINING IN PHYSICS-INFORMED MACHINE LEARNING, Tim De Ryck, Florent Bonnet, Siddhartha Mishra, Emmanuel de Bézenac; in particular, Theorem 2.3 shows that for the linearization problem with preconditioning achieves a converge rate given by the condition number and that the natural gradient preconditioning achieves an optimal condition number of 1. As such, this seems to be a linearized (hence easier) version of the main result in the manuscript. 
      + Convergence of Stochastic Gradient Methods for Wide Two-Layer Physics-Informed Neural Networks, BANGTI JIN AND LONGJUN WU, 2025; provide an exponential convergence rate of SGD under overparametrization
      + Non-Asymptotic Analysis of Projected Gradient Descent for Physics-Informed Neural Networks, JONAS NIESEN, AND JOHANNES MÜLLER, 2025; provide a sublinear convergence guarantee for arbitrary sampled (S)GD without assumption on the network size 
+ Overparametrization assumption: The manuscript is making the global assumption of overparametrization. Where I understand that this allows the use of an established machinery, I do not believe that this is the setting that PINNs are used in. In particular, note that in PINNs, the data points are synthetically sampled integration points of the computational domain rather than data points like in supervised learning. As such, the problem in PINNs is rather an optimization than a statistical problem. In practice, new data points are sampled continuously throughout optimization. Hence, it is not clear how practically relevant the setting of overparametrization is. However, this assumption is not uncommon and the only work I am aware not making an assumption on the size of the PINN is by Niessen and Müller (2025). 
+ Difference of PINNs to supervised learning: In the introduction, it is mentioned that the reason why NGD is not used in supervised learning is due to its high computational cost. However, I believe that there another, arguably more important reason: The loss function of PINNs has a much worse conditionining due to the appearence of the PDE operator. It is stated in line 251 that the conditioning can be really bad for complex PDEs. From my understanding, this can also be the case for simple PDEs. 
+ Experiments: The experiments compare natural gradient to SGD, Adam and L-BFGS. However, this comparison has been made at several places, in particular, by Müller and Zeinhofer (2023). I think, it would be much more informative to not repeat this comparison, but to see, how the empirically observed convergence rates relate to the theoretical guarnatees. Further, in relation to the overparametrization assumption, the influence on the network size and optimizer on the generalization error is not studied empirically.

### Questions
+ Can you elaborate why you regard your $O(1/\lambda_\max)$ as an improvement over $O(\lambda_\min)$? 
+ Can you elaborate on the relation of the natural gradient defined in Section 4 to previously proposed methods like energy natural gradient? In case of equivalence, can you adapt your presentation accordingly, see also my discussion above. 
+ Can you comment why you make the overparametrization assumption? Also, a discussion of the overparametrization assumption inside the manuscript seems reasonable. 
+ Can you comment on the different condition numbers of supervised learning and PINNs? Can you elaborate why you only expect a bad condition number for complex PDEs (see line 251)? This could be added in the manuscript as a motivation for second order and natural gradient methods. 
+ Can you validate the theoretical convergence guarantees within the computational experiments? In particular: 
   + Compare the loss curves to the linear convergence guarantees from Theorem 3.7 and 4.7. 
   + Compare the generalization error for different network size and optimizers. As the convergence guarantees are merely given with respect to the loss function, it is informative to contrast it with the relative L^2 error. 
   + Improve readability of the plot. 

**Minor comments:**
1. In the abstract, it is stated that, *However, the learning rate of GD for training two-layer neural networks exhibits poor dependence on the sample size and the Gram matrix*. Note that the term learning rate usually refers to the step size in machine learning, not the convergence rate of the learning process. 
2. It is stated in several places that the learning rate can be O(1). However, it is shown that the convergence rate is O(1). 
3. In the limitations section on scalability of natural gradient methods in these scenarios, the following directly relevant works are not referenced: 
+ Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks, Dangel et al. 2024
+ Improving Energy Natural Gradient Descent through Woodbury, Momentum, and Randomization, Dangel et al. 2025

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies optimization dynamics for over-parameterized two-layer PINNs trained on a class of second-order linear PDEs in the NTK regime. 

First, it refines NTK-style analyses for gradient descent (GD) on PINNs by: 

- deriving a new residual recursion that improves the admissible stepsize from $\mathcal{O}(\lambda_0)$ to $\mathcal{O}(1/\|H_\infty\|_2)$

- weakening the network-width requirements via concentration for sub-Weibull variables, yielding linear convergence with rate $(1-\eta\lambda_0/2)$ (Theorem~3.7). 


Second, it establishes that full-batch Natural Gradient Descent (NGD) achieves convergence with step-size independent of the NTK kernel which could be very ill-conditioned as sample size increases.

Experiments on Poisson, Heat, and Helmholtz equations (1D--10D) compare NGD to SGD, Adam, and L-BFGS and illustrate faster loss decay and lower relative $L^2$ error.

### Strengths
**Significance.** 
PINNs are widely used yet optimization pathologies are common; providing conditions under which GD can use practical step sizes and NGD can enjoy $\mathcal{O}(1)$ step sizes---and even quadratic behavior for smooth activations---can influence both theory and practice in scientific machine learning. I highly appreciate this work. 

**Originality.**
The work extends NTK-based optimization theory from regression to PINNs with PDE with improved rates for GD compared to prior work (Gao et. al. 2023) and provides the first analysis of NGD for PINNs that provably shows its effectiveness. 


**Quality.**
The technical development is careful: improved GD bounds come from a sharper recursion (Lemma B.1) and stability of $H(k)$ via sub-Weibull concentration; NGD uses Jacobian stability tailored to individual neurons (Lemma 4.6) rather than global matrix stability. 
Theorems~3.7 and 4.7 and Corollaries 4.9--4.10 are stated with explicit dependences on d, $\lambda_0$, and $m$.

**Clarity.** 
The paper is well organized, with explicit assumptions, side-by-side remarks comparing to prior work, and worked proofs in appendices. 
Figures and tables (p. 9--17) clearly support the claims.

### Weaknesses
**Comparison with existing results**: Although the paper compares the results in terms on conditions on the number of neurons, sample size, etc. It particular, the authors should explain why the approach in Zhang et al 2019 for natural gradient for regression is not applicable here as well. In particular, how the dependence on the derivatives makes such approach inapplicable. 

**Experimental evaluation**: Some information is missing. For instance, the authors should clarify what they mean by the L^2 error, is it the error on the training samples (colocation points), or do they consider test samples. In case, they evaluate on training samples, that's fine since the results are for the training, but it would be also interesting to evaluate on a test samples to get a better sense of generalization.


**Minor**: The abstract contains notations (lambda_0 and H^{\infty}) that are not defined. That might give a bad first impression to the reader, please fix this.

### Questions
- Line 206: "In contrast, on one hand, our conclusion is independent of n1 and n2". I am not sure I understand this, isn't the statement dependent  through m (logarithmic dependence).

- In remark 3.8 it is said that the trace H^{\infty} is independent of the sample size. However H^{\infty}  is a matrix of size n time n, so a priori its trace depends on the sample size, unless it has a particular structure. Could the authors provide more evidence for why would such quantity be independent of sample size. 
- In line 262, it is said: "we fix the output weight a and update the hidden weights via NGD." Why fixing the last layer and only update the hidden weights? This seems unusual? 

- In line 320 on the approach of Zhang et al. 2029, it is said that: "However, this approach is not applicable
to PINNs, because the loss function involves derivatives." Can the authors elaborate on why this is not applicable here?


- In line 1823 of the proof of lemma 4.6, there is a reference to equation 46, but I don't see how this equation leads to a control on the probability of the weights being larger than M. Perhaps a typo in cross referencing? 

- In line 1865, did the authors mean lemma D.1 instead of D.4? Actually, in the statement of lemma D.1, the paper introduces two objects L_n and L_n* but uses only L_n*. This is confusing especially since the expression of L_n* simplifies a lot when expressing it directly. I suggest the authors directly define L_n* explicitly.  


- In line 1011 of the proof of lemma 3.3, I don't understand why the second term was discarded, it decays faster in terms of m, but not in terms of log(1/delta) (in the limit when delta goes to 0).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper theoretically analyzes the global convergence of natural gradient descent (NGD) for training overparameterized physics-informed neural networks. The motivation is strong since NGD enjoys faster convergence, as shown in experiments, compared with SGD and Adam. Compared with several previous relevant works, the paper adopts advanced concentration inequalities and a loss decomposition method to further improve the bound.

### Strengths
(1) The presentation is clear and easy to follow.

(2) Experiments show the superior performance of NGD over L-BFGS and Adam, which motivates the study in this paper on the convergence of NGD.

(3) The theory is correct and informative.

### Weaknesses
(1) Can you extend the work to stochastic versions of algorithms? I think there are some works extending GD for PINNs to SGD. Can you similarly do the extension? I think the concern for NGD is its size dependence on data samples, which is unfavorable. Therefore, the convergence (in expectation) for stochastic algorithms may be more attractive.

(2) A previous work (The Challenges of the Nonlinear Regime for Physics-Informed Neural Networks; NeurIPS 2024) shows that the Newton’s method enjoys the fast convergence as in your theorem, where the convergence rate is independent of the least eigenvalues of the Gram matrix. I think the algorithm is quite similar to yours, with (JJ^T)^{-1}. Can you comment and discuss on this paper? What are the advantages of your algorithm and analysis, compared to this paper?

### Questions
Do you think NGD is the next optimizer for PINNs? Can you discuss your insights?

### Soundness
3

### Presentation
4

### Contribution
3
