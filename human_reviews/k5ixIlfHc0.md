# Error Bounds for Deep Learning-based Uncertainty Propagation in SDEs

- Decision: Reject
- Scores: 6, 5, 6, 3

## Abstract
Stochastic differential equations are commonly used to describe the evolution of stochastic processes. The uncertainty of such processes is best represented by the probability density function (PDF), whose evolution is governed by the Fokker-Planck partial differential equation (FP-PDE). However, it is generally infeasible to solve the FP-PDE in closed form. In this work, we show that physics-informed neural networks (PINNs) can be trained to approximate the solution PDF using existing methods. The main contribution is the analysis of the approximation error: we develop a theory to construct an arbitrary tight error bound with PINNs. In addition, we derive a practical error bound that can be efficiently constructed with existing training methods. Finally, we explain that this error-bound theory generalizes to approximate solutions of other linear PDEs. Several numerical experiments are conducted to demonstrate and validate the proposed methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Paper Summary:

Physics-informed neural networks (PINNs) can be used to approximate the solution probability density function (PDF) for stochastic differential equations (SDEs). While existing methodologies can achieve high levels of approximation accuracy, the absence of knowledge regarding the true solution renders it impossible to assess the quality of the approximated solution definitively. The authors’ main contribution is the establishment of a general framework for deriving an arbitrarily tight error bound that quantifies the discrepancy between the true solution and its approximation, utilizing two additional PINNs. Furthermore, this paper presents a more practical error bound that relies solely on one additional PINN, albeit at the expense of losing the property of arbitrary tightness.
The methodology for the more practical bound is shown on a set of systems for which a true solution can be derived.
The paper presents a novel method for estimating the accuracy of an approximated PDF. Although the approach does not yield an exact metric for assessing whether any approximated PDF is well-trained, it offers a way that can, though it does not necessarily guarantee, confirm the well-trained status of an approximated PDF. This bound allows users to assess and decide whether the approximation is acceptable based on their specific criteria.

Review Summary: 

Overall, I really love the idea and think the novelty is "good". However, the presentation needs to improved and and the points flagged as "Major" below should be adressed in order to to be able to recommend acceptance.


EDIT: Raised to 6 after author response.

### Strengths
The authors present an (to the best of my knowledge) novel approach that leverages the linearity of the FP-PDE operator to enable a derivation of an error bound, thereby enhancing the reliability of solution PDF approximations.

The well-structures theoretical section provides a clear and sound proof that is thoroughly derived.

### Weaknesses
Major:

The authors position their method as a general framework applicable to the n-dimensional case. However, the experiments conducted primarily concentrate on the one-dimensional scenario, with the inclusion of a single two-dimensional case. 

While it has been demonstrated, that an arbitrary tight second-order temporal error bound can be achieved, this is contingent upon specific conditions being met for α1 and α2. However, the paper only provides a method to verify α1, but not addresses α2, rendering the second-order temporal bound impractical for use. 

Page 10, Figure 3g illustrates the fragility of the method with respect to the fulfillment of the necessary conditions (16a), highlighting that its effectiveness is highly sensitive to random initialization. This suggests that the method’s practicability may be constrained by the high variance of α1-values.


Minor:

Page 6, Remark 4 refers to equation 17, however equation 15 describes the relation ship between γ and α.

Page 10, Figure 3: Not all y-axes are defined.

### Questions
Major:

Can you demonstrate your method on a high-dimensional example, such as a 10-dimensional Ornstein-Uhlenbeck?

Since that the true solution is known for the experiments conducted, it would be beneficial to compare the second-order temporal error bound with the first-order temporal error bound presented in Table 2 on page 8. Given the paper’s assertion that arbitrary tightness can be achieved with the second-order temporal error bound, it would be interesting to observe this in the experimental results. Additionally, analyzing the corresponding α2-values will provide insights into whether the necessary conditions (16b) are met.

Including a table (or described as additive in Table 2, page 8) that presents the variance of α1-values would provide valuable insights into whether the method can be reliably used to estimate the accuracy of the approximated PDF.

Minor:

Page 4, Definition 1: The paper claims, that the PINN that approximates ei is governed by the recursive PDE D[ei(x,t)]+D[êi−1(x,t)]=0. However, I could not directly verify this assertion from the preceding equations. Instead, my educated guess would yield a different outcome D[ei(x,t)]+Σk=1i−1D[êk(x,t)]=−D[^p(x,t)]. It would be valuable for unfamiliar readers if the authors would clarify the origin of the abbreviated formulation.

"1D Nonlinear SDE (GPU, seed0)." It would be beneficial to specify whether the results for "1D Nonlinear SDE" represent average values over multiple seeds. Furthermore, clarification is needed on whether any seeds were excluded due to α1-values failing to satisfy the necessary condition (16a) at the end of training, which would render the first-order temporal error bound uninterpretable. (Page 10, Figure 3g)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses error bounds for deep learning-based uncertainty propagation in stochastic differential equations by leveraging physics-informed neural networks (PINNs) to approximate probability density functions governed by the Fokker-Planck equations. The authors propose a method to construct error bounds for these approximations. Specifically, they introduce both a recursive error approximation method and a practical single-layer error bound, demonstrating the efficacy of these bounds in safety-critical systems. Experiments validate the theoretical contributions, showing the method’s effectiveness in bounding errors for various SDEs.

### Strengths
1. The paper focuses on a practical aspect of uncertainty quantification in deep learning applications for SDEs, which is valuable for many domains. The proposed recursive error function approach fills a gap by offering a way to bound approximation errors tightly.
2. The authors provide rigorous proofs, including convergence criteria and bounding error functionals, which are clearly laid out and generalized to linear PDEs. Theorems on the sufficiency of using only two PINNs for error bounding and proofs of tightness for error bounds add theoretical robustness to the claims.
3. The experimental results are broad, covering multiple SDEs and boundary conditions, which show the practical relevance of the proposed bounds. The results validate that the first-order temporal error bound covers approximation errors while remaining relatively tight, particularly in terms of normalized error gap.

### Weaknesses
1. The recursive approach, while theoretically sound, might pose practical challenges in computational efficiency and model training times for higher-dimensional SDEs. Although the authors suggest that two PINNs suffice in most cases, a deeper discussion of the computational overhead in high dimension cases would be necessary.
2. The practicality of checking conditions such as those outlined for $\alpha_1$ and $\alpha_2$ during training remains unclear. While conditions are theoretically justified, their feasibility in large-scale implementations with multiple variables is not discussed in depth.

### Questions
1. Could you provide further insights into the computational costs associated with the recursive error approximation, especially in higher dimensions?
2. Are there specific use cases where the single-layer error bound would fail to capture significant error characteristics, or where it might be insufficient compared to the recursive bound?
3. Could you elaborate on the practicality of verifying conditions for \alpha_1\ and \alpha_2 during training, particularly in large-scale applications?
4. There are many recent works on using PINNs to approximate probability distributions governed by the Fokker-Planck equations. In particular, the committor function. For example, Solving high dimensional committor problems by (1) Neural Networks (https://arxiv.org/abs/1802.10275) (2) Neural Network based-Symbolic Regression (https://arxiv.org/abs/2306.12268) (3) PINN + adaptive sampling for solving committors (https://arxiv.org/abs/2404.06206). A discussion of related work would enhance the impact of the work.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors consider an approach for bounding the errors for physics informed neural networks (PINNs) approximating the solution to the Fokker-Planck equation. The Fokker-Planck equation satisfies a linear operator and establishes a connection between an underlying stochastic differential equation (SDE) and the parameters of the operator. PINNs consider applying a boundary condition with a differential operator through automatic differentiation to impose the solution of a PDE. The authors consider a new PDE that operators on the space of errors and analyze the behavior of this PDE. They break this PDE into a set of $n$ subsolutions whose sum is equal to the true error of the PINN approximate to the PDE. The proof techniques only rely on the underling PDE being linear and apply to other linear PDEs. Finally, the authors benchmark on some numerical experiments. They consider experiments on how the proposed bounds compare to the true empirical errors.

### Strengths
Methods for estimating errors of PDEs are important for making the neural network based PDE solvers applicable in real world scenarios. 

The proof techniques are intuitive and can be applied to other settings with other linear PDEs.

The bounds also appear to be fairly tight from the numerical simulations.

### Weaknesses
The bounds are somewhat cumbersome to compute in practice due to the necessity of fitting two functions.

In the case of fitting a single function, there is an decrease in the tightness of the bound, but it becomes more feasible to compute. 

It's unclear what the feasibility of training the error networks is.

### Questions
How does the method perform in higher dimensional cases? For example, there's a connection between the flows given by the Fokker Planck equation and generative modeling. However, the support is over typically very high dimensional spaces in the generative modeling cases. How is the empirical performance in such cases? PINNs sometimes have trouble in high dimensional cases, so I'm curious about the application here. 

The loss for the error network training appears to be much more volatile than the original PDE network error, is there a reason why this is the case? It seems to be more difficult to train than the original PDE approximation. 

Are the $C_{PDE}, C_{embedding}, C_{quadrature}$ constants easy to compute? How does the different quadrature rule affect the constant, for example? Additional information on the interpretation or finding these would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper studies developing data-driven error bounds for stochastic differential equation. The solution builds on the idea of constructing a recursion stack of excess errors made by a Physics-Informed Neural Network (PINN) on the solution of the Fokker-Plack equation. The trick that makes these error bounds feasible is that the error made on the Fokker-Planck equation prediction depends only on the residue of the PINN. In addition to the theoretical implications of this idea, the paper also shows numerical results on five rather well-known SDE systems that demonstrate the tightness of the built error bound.

### Strengths
- The paper touches a clearly novel problem and addresses it with an original solution. Particularly, the idea of building a recursion of excess losses is sensible and attractive. There exist similar approaches in other applications of learning theory. See for example a paper appearing at the upcoming NeurIPS:
https://arxiv.org/abs/2405.14681
- The formal analysis is rigorous and technically correct.
- The numerical analysis is well-conducted and well-communicated.

### Weaknesses
- Despite the title, I am having hard time locating how exactly the proposed approach helps bounding the propagation of error in SDE systems. I assume what is meant by propagation is propagation through time. The approach appears to develop bounds for standalone time points, ignoring how these errors warp when passed through the SDE kernel and accummulate.
- Another limitation of the approach is that it ignores the impact of the estimation error, which is a fundamental issue in quite many system identification problems especially under stochasticity. In the absence of a link to estimation error, it becomes a bit far-fetched to assess the real-world significance of the developed solution.
- The final weakness is the limited scope of the presented experiments. With due respect to the originality and rigor of the presented theoretical material as admitted above, I still think that the chosen set of SDEs is rather trivial. Specifically, their input dimensionality is too small and the level of difficulty is too limited to assess the significance of the presented solution. Three of the five SDE systems have analytical solutions, where the presented methods would not bring any value-added. I would start from dynamics such as the stochastic Lorenz attractor for which numerical solvers fail to deliver accurate results even in the deterministic variant. Only in such cases I can make sense of the true importance of developing rigorous recursive error bounds. I do expect that the presented material will also apply reasonably well to such cases but unfortunately those results are missing in the current version of the work.

### Questions
- I couldn’t locate the governing physics in Eq 4 mentioned in the description of Eq 6. The guidance from the authors is highly appreciated.
- Why do we need to set weights to $\mathcal{L}_0$ and $\mathcal{L}_r$? Is this not assigning arbitrarily relative importance to system initialization and system evolution? Should these weights not come from the theory itself if it is a complete theory? Is such weight association a common practice in the literature or a design choice made by this particular solution? How do the authors choose these weights in the experiments?

----
POST-REBUTTAL: The interaction with the reviewers made me conclude that the weaknesses I pointed out are indeed fundamental. I reflect this to my score by moving one step down from the initial borderline reject.

### Soundness
3

### Presentation
3

### Contribution
2
