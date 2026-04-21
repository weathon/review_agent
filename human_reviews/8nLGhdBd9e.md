# Score-based Neural Ordinary Differential Equations for Computing Mean Field Control Problems

- Avg Score: 4.25
- Decision: Reject
- Scores: 6, 5, 5, 1

## Abstract
Classical neural ordinary differential equations (ODEs) are powerful tools for approximating the log-density functions in high-dimensional spaces along trajectories, where neural networks parameterize the velocity fields. This paper proposes a system of neural differential equations representing first- and second-order score functions along trajectories based on deep neural networks. We reformulate the mean field control (MFC) problem with individual noises into an unconstrained optimization problem framed by the proposed neural ODE system. Additionally, we introduce a novel regularization term to enforce characteristics of viscous Hamilton--Jacobi--Bellman (HJB) equations to be satisfied based on the evolution of the second-order score function. Examples include regularized Wasserstein proximal operators (RWPOs), probability flow matching of Fokker--Planck (FP) equations, and linear quadratic (LQ) MFC problems, which demonstrate the effectiveness and accuracy of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work deals with extending the (now) mainstream score-based generative modeling techniques to the case of mean-field control problems. The authors construct high-order normalizing flows (i.e. neural ODE systems for the score) and reformulate the
mean field control (MFC) problem with individual noises into an unconstrained optimization problem framed by the proposed neural ODE system. They estimate the first- and second-order score functions using a deep neural network function.

### Strengths
1. The connection of score-based modeling with mean-field control problems is novel and noteworthy.
2. The theory is rigorous and the paper is largely written in a self-contained manner. 
3. Many cases of regularization/structure are shown including the HJB regularizer, Wasserstein proximal operators, etc.

### Weaknesses
1) Clarity of writing could be improved. It is at times hard to follow.
2) This is addressed in the Questions section, but I am not sure/convinced about the efficacy of higher-order ODEs in comparison to methods like Flow Matching or Stochastic Interpolants (in general). It would be nice to see some numerical comparisons (if those can be cast for MFC problems as well)  -- I know there is a theoretical section on flow matching but I would like to see tradeoffs gained/lost with this augmentation.

### Questions
1) Is the function L in the MFC objective typically assumed to be strongly convex? 
2) How does this compare with existing, much faster generative models/techniques such as Flow Matching (and variants)? I am aware that those are highly efficient without necessarily being higher-order models.
3) Is there work for control problems beyond LQ problems (which are well-studied)?
4) You mentioned " It is probably more important to train the neural network properly." for the error, can you elaborate a bit more on what is meant by this point?

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
2

### Summary
In this paper, a method for modelling the trajectory of score functions by using neural ordinary differential equations is proposed. Also, its application to solving mean field control problems is shown, along with a regularization term based on the Hamilton-Jacobi-Bellman equation.

### Strengths
I am not familiar with mean field control problems, but I believe that the method proposed in this paper is a novel approach as far as understood from the survey and other papers cited in this paper. Numerical experiments have confirmed that the proposed method in fact solves mean field control problems accurately and that the regularization term based on the HJB equation is effective.

### Weaknesses
Numerical experiments have been performed only with the proposed method, and no comparison with other methods is shown. Therefore, I am not sure that the proposed method is in fact sperior to existing methods. If there are existing methods that can be applied to the problems used in the experiments, the proposed method should be compared with such methods.

In addition, some theoretical results are presented, but most of them are related to the derivation of equations and so on. No theoretical support for the proposed method, such as the generalization error analysis, is presented.

### Questions
What are some other methods that could be used in this type of problem setting? For example, can neural networks used for modeling differential equations in this paper be replaced by, e.g., Gaussian process regression? Can you show experimentally that the proposed method is indeed superior when compared to such methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a neural ODE system, along with its discretization, for computing first- and second-order score functions. As an application, the authors reformulate second-order mean field control (MFC) problems with individual noises into an unconstrained optimization problem using the proposed neural ODE system, and additionally derive a novel regularization term based on viscous Hamilton-Jacobi-Bellman (HJB) equations. In numerical experiments, including regularized Wasserstein proximal operators, probability flow matching, and linear quadratic MFC problems, the authors demonstrate the accuracy of the proposed method and the benefits of the HJB regularization term.

### Strengths
1. The paper is generally clear and well written.
2. The proposed method appears novel and may be impactful (but ultimately a lack of detailed comparisons with the existing literature makes it hard to judge, see below).
3. The novel HJB regularization term appears to significantly improve the results.

### Weaknesses
1. While the paper is generally well written, there are a couple of issues with the presentation. Firstly, the significance of the paper is not clear to me; what, exactly, is the contribution? Please include a paragraph explicitly stating the main contribution of this paper and how it advances the state-of-the-art.

2. More generally, comparisons with the existing literature are not sufficient. There are two types of work that might be relevant here: methods for computing score functions and methods for solving (second-order) MFC problems. The authors list a large number of methods under related work, but none with sufficient detail to understand exactly how they relate to the proposed method. Please include a more detailed discussion of a handful of key works, so the reader can understand how the proposed method compares to the state-of-the-art in terms of approach.

3. Having identified key related works, please include empirical comparisons with the most relevant methods. so the reader can understand how the proposed method compares to the state-of-the-art in terms of performance.

4. In the second paragraph of the introduction, the proposed method is motivated as follows: 

    “While score functions provide powerful tools for modeling stochastic trajectories, their computations are often inefficient, especially in high-dimensional spaces. Classical methods, such as kernel density estimation (KDE) (Chen, 2017), tend to perform poorly in such settings due to the curse of dimensionality (Terrell & Scott, 1992).” 

    However, the experiments are not very high-dimensional (max. 10). As a result, and also due to the lack of empirical comparisons with other methods, it is not clear whether the proposed method actually solves the problem that was identified at the beginning of the paper.

5. Despite computational efficiency being a motivation, the paper does not include timings of the proposed method for any of the experiments. The closest we get is a discussion of the asymptotic complexity for computing the first-order score function. Please include a table or figure showing empirical runtime measurements for the proposed method across different problem dimensions, comparing these to baseline approaches on the same problems.

### Questions
1. What do you consider to be the main contribution / advance of this method? If someone is only interested in first-order score functions, say, is this method still useful?
2. Are there applications other than second-order MFC where the second-order score function is useful?
3. What are the classical methods for computing (second-order) score functions? Did you compare with any of these?
4. What are the state-of-the-art machine learning methods for estimating (second-order) score functions? Did you compare with any of these?
5. In a number of places, the authors briefly mention the applicability of the method to generative modeling. Could you elaborate on how the proposed method would be useful for generative modeling? What advantage would this offer over existing approaches? 
6. Have you tested the proposed method on higher-dimensional systems?
7. Have you timed the proposed method? How does it depend on the dimension of the system? How does it compare to related methods?

While I generally find the paper quite good, I have sufficient doubts about the contribution and the comparisons with related work that I can't recommend it for acceptance right now. However, I would be happy to increase my scores if these doubts are adequately resolved.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The paper gives a system of ODE equations to solve transport equations for a density $\rho(t,x)$ based on the knowledge of the score $s(t,x) = \nabla \log \rho(t,x)$ and the Hessian $H(t,x) = \nabla \nabla \log \rho(t,x)$. These equations are proposed as a way to solve mean field control problems after learning a drift variationally.

### Strengths
The technical presentation of the material is clear.

### Weaknesses
The paper basically contains no new material:

1. Eqs. (3) are well-known and have appeared in many places, including 

https://arxiv.org/abs/2206.00860

https://arxiv.org/abs/2210.04296

https://arxiv.org/abs/2206.04642

All these equations can basically derived by the method of characteristics, assuming that the score $s(t,x) = \nabla \log \rho(t,x)$ and the Hessian $H(t,x) = \nabla \nabla \log \rho(t,x)$ are known. The authors should explained better what is the added values of these equations for the numerical scheme they propose. 

2. The material in Sec. 3 is also standard: Eqs. (7) - (10) are the well-known forward Euler discretization of Eq. (3), and Eq. (11) is just their roll-out version. In addition, the neural architecture proposed in Eq. (5) is just a one-hider layer neural network, which is more more simple that standard approximation use for the score (that use UNet, or DiT, etc). Why use such a simplistic architecture?  This seems limitative as well as unnecessary. The authors should explain better what justified this choice.

3. The algorithm proposed does not solve one of the main (and also well-known) issues with neural ODE, namely that they are not simulation free and as a result are to scale to large problem: in particular, differentiating through Eq. (11) is required in Algorithm 1 and costly. In particular, the authors should explain better why they believe that their algorithm may be scalable to high dimensional problems. 

4. The material in Sec. 4 is again standard. In particular Prop. 3 is a well-known set of forward-backward PDE to solve MFC, and Core. 1 is an immediate generalization of this result for a specific choice of Lagrangian and terminal cost. What is the added value of including these results in main text?

5. The numerical examples are too simple to be convincing. In particular, the example in Sec. 5.1 can be factorized over the dimension (which is why it is analytically solvable), which makes it not very challenging computationally.  Similarly, the problem treated in Sec. 5.2 involves a linear OU process, also factorizable. Including these results as test-cases illustration is okay, but the method should also be tested on more complex examples, like the ones found in the cited papers by Lipman et al. 2022 and Boffi & Vanden-Eijnden 2023.

### Questions
Can the authors address the points raised in the **Weaknesses** above?

### Soundness
1

### Presentation
2

### Contribution
1
