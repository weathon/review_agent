# Gradient-Normalized Smoothness for Optimization with Approximate Hessians

- Decision: Accept (Poster)
- Scores: 4, 10, 6, 6

## Abstract
In this work, we develop new optimization algorithms that use approximate second-order information combined with the gradient regularization technique to achieve fast global convergence rates for both convex and non-convex objectives. The key innovation of our analysis is a novel notion called Gradient-Normalized Smoothness, which characterizes the maximum radius of a ball around the current point that yields a good relative approximation of the gradient field. Our theory establishes a natural intrinsic connection between Hessian approximation and the linearization of the gradient. Importantly, Gradient-Normalized Smoothness does not depend on the specific problem class of the objective functions, while effectively translating local information about the gradient field and Hessian approximation into the global behavior of the method. This new concept equips approximate second-order algorithms with universal global convergence guarantees, recovering state-of-the-art rates for functions with Hölder-continuous Hessians and third derivatives, quasi-self-concordant functions, as well as smooth classes in first-order optimization. These rates are achieved automatically and extend to broader classes, such as generalized self-concordant functions. We demonstrate direct applications of our results for global linear rates in logistic regression and softmax problems with approximate Hessians, as well as in non-convex optimization using Fisher and Gauss-Newton approximations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper present a very impressive general theory for gradient regularized Newton methods. The key observation is characterizing the amount of regularization $\gamma(x)$ through an implicit definition. What is also noteworthy is the many instantiations including Holder smooth Hessians, quasi-Self-Concordant,  $(L_0,L_1)$--smoothness, all of which provide a lower bound on this implicitly defined $\gamma(x),$ which is needed to execute the algorithm. The main issues I see are regarding computing $\gamma(x)$ and generally the fit within ICLR.

### Strengths
This is elegant work in deterministic optimization for Newton type methods. Both unifying previous theories, extending to in-exact Hessians, and developing many examples.

### Weaknesses
This seems to be mostly a follow up work of Doikov2023 and Doikov 2024a, which allows for inexact Hessians, and unifies some of these previous results. It's a dense literature, and difficult from reading this paper to know exactly what is new and what is not. Here I think the authors should write a more extensive background/related work section in the appendix, which makes it clear which parts are taken from which papers, and which parts are entirely new. The main paper itself does not make this very clear, and I had to open some of the prior work to get a sense of what had been done.

Another weakness is generally the fit with ICLR, its reviewing process, and to a lesser degree the audience.  Your paper and theory look very solid. But I regret to say, having it reviewed by ICLR is not a good fit. First the topic, which is full batch optimization. Even convex learning problems, such as GLMs, are only difficult to solve when there is so much data we need stochastic gradients. Your theory depends heavily on being in the deterministic setting. Furthermore, your result rely on the proofs, and your paper is (including proofs+ appendix), 48 pages long.  The turn around time of ICLR does not permit reviewers to go over your proofs in detail. I for one have only sampled your proofs. You would be much better served by submitting this substantial piece of work to an optimization journal (SIOPT, Math programming ...etc) where you would get a proper review.


Not really clear/transparent on limitations of Newton's method in machine learning. You introduction frames it as if Newton methods  are extensively used in ML. But the issues around dealing with stochastic estimates and non-convexity, makes most Newton type approach inapplicable.

### Questions
*Questions*

1. Implicitly defined $\gamma(x)$ and lower bounds. 
In general, $\gamma(x,g)$ is implicitly defined, and needs to be computed or bounded to be used. The paper is clear about this, and gives many useful lower bounds for certain problem classes. But still seem some awkward issues around this. For instance, you give a type of adaptive line search procedure in Algorithm 3, and bound the number of oracle calls to $\mathcal{O}(K)$ where $K$ is the number of iterations of Algorithm K. But here you rely on the $\gamma_k$ computed by Algorithm 3 will satisfy the lowers bounds you give for $\gamma(x_k, F'(x_k)).$ Here I think you need a formal statement about the complexity of this adaptive line search, you can use it to guarantee that Algorithm 3 will have a meaningful complexity. Another clear issue is with your main result for convex functions in Theorem 2, which relies on the exact computation $\gamma_k = \gamma(x_k).$ Here I assume it is hard to characterize the complexity of solving this equation, and of course, exact solutions are probably inaccessible. 




2. Lines 069-065: You claim that you will present a linear convergence result for when using inexact Hessians, and no strong convexity type assumptions. But where exactly is this result? The only linear convergence I see in the main paper is the instantiation of Theorem 2 for the class of Quasi-Self-Concordant functions, which relies on using the exact Hessian. Could you clarify this statement and/or correct it?


*Minor* 

1. A small mistake you repeat is to write "...method equation 1" where all prepositions have been dropped. It should be "... the method in equation 1".

2. At a high level, it seems your gradient normalized smoothness does to Newton, what Directional smoothness, has done for gradient descent methods. That is, you identity an implicitly defined stepwise, that always guarantees descent, and generalizing other notations of smoothness including $(L\_0, L\_1)$-smooothness. This is exactly what the claims of the paper [] are, but for gradient based methods. So this in no way diminishes your contributions, it just seems it could be an interesting connection.

3. The assumption in eq (10) should be formally stated as such. It's a bit confusing how it is now stated. It would be much clearer if you have

Assumption 1.  There exists $0 \leq \alpha\_i \leq 1$ and $M_{1-\alpha\_i} \geq 0$ for $i=1, ..., d$ such that Eq (10) holds.

The same things goes for its counterpart eq (40) in the appendix.

4. Corollary 1. You "Under assumptions of Theorem 1 we can bound ..." But don't you mean, if equation (8) holds and Assumption 1 (above) hold? This is exactly one of the reasons I'm a bit confused about (10) being an assumption or not.

5. There are a few type-Os, quite a few missing prepositions, and similar minor mistakes throughout. You should take a pass with LLM or grammar check to fix theses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper introduces a new assumption on Hessian inexactness and gradient-normalized smoothness. Under the gradient-normalized smoothness assumption, they provide the best-known rates of the Gradient-Regularized Newton method for (a) bounded Hessian, (b) Lipschitz Hessian, (c) third-order Lipschitz smoothness, (d) quasi self-concordant functions, and new rates for generalized self-concordant functions. Additionally, they introduce Gradient-Regularized Newton with Approximate Hessians and prove new convergence rates.

### Strengths
The paper is very well-written, accurate, and clear. The results are correct, new, and interesting for the community. It is an outstanding paper. I list several strengths of the paper:

1. The paper proposes new concepts of smoothness that generalize and easily connect many well-known and modern classes of functions. For this class of gradient-normalized smooth functions, they provide examples of functions satisfying it. For classical function classes, they provide a lower bound on the smoothness constant $\gamma(x)$.

2. The method with proposed step-sizes is universal and adapts to the problem class, achieving the best-known rates with second-order information. Moreover, the same method works with only first-order oracles by using inexact Hessians, and the paper provides new rates in this setting.

3. The experiments show the effectiveness of the proposed step-sizes.

### Weaknesses
I couldn’t find any major weakness in the paper. Minor:
1. While the numerical experiments show the effectiveness of the Gradient-Regularized Newton method with the proposed stepsizes, it would be beneficial to compare it with other adaptive or universal methods, both with exact and inexact Hessians.

### Questions
No questions.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Gradient Normalized Smoothness as a unifying device to analyze approximate Newton methods. The framework couples objective smoothness with Hessian approximation quality and shows when an approximate Hessian can match exact Newton rates. The analysis recovers state of the art rates across several function classes and provides applications to logistic regression with Fisher and to softmax problems. The experiments are small but consistent with the theory.

### Strengths
1. The paper introduces a powerful new theoretical framework for developing optimization algorithms that leverage approximate second-order information while guaranteeing fast global convergence for both convex and non-convex objectives. The central innovation is the Gradient-Normalized Smoothness (GNS), a novel, universal notion that locally characterizes the maximum radius of a ball around the current point where the gradient field is well-approximated. This concept provides a unified mechanism to connect errors stemming from Hessian inexactness and Taylor's approximation, effectively translating local properties into the method's global performance without relying on predefined problem class specific parameters.

2. Algorithm 1 uses a simple gradient regularization structure where the step-size ($\gamma_k$) is chosen based on the GNS quantity, allowing it to adapt automatically to the objective's characteristics and the degree of the Hessian approximation. The theory successfully handles inexact Hessians $H_k$ provided they satisfy a condition related to the gradient norm (Equation 2). When the Hessian is exact, the framework recovers state-of-the-art global convergence rates for various function classes, including those with Hölder-continuous derivatives and quasi-Self-Concordant functions. Notably, the paper demonstrates that if the objective's smoothness degree ($\alpha$) dominates the approximation degree ($\beta$), the method with inexact Hessians achieves the same global rate as the full Newton method.

3. The practical effectiveness is validated using specific approximate Hessians popular in machine learning, such as Fisher and Gauss-Newton matrices, which satisfy the derived bounds and yield new global convergence results. Furthermore, experimental results suggest that Algorithm 1 using an inexact Hessian approximation (when aligned with the theory) is more numerically stable than the Exact Newton method, which can fail to converge on non-convex objectives due to ill-conditioning issues.

### Weaknesses
1. The notion of Gradient-Normalized Smoothness is defined mathematically but not well-motivated in terms of optimization geometry or curvature behavior. The crucial element appears to be the specific gradient-based normalization ((1/γ) ||g||* ||h||). The paper should elaborate on why this specific normalization is the key that unlocks the unified analysis. It is also unclear how this notion of smoothness compares to established notions like relative or anisotropic smoothness. 

3. The theoretical framework fundamentally relies on the existence of γ(x_k), which is defined via a maximization condition but cannot be computed in practice. While the paper acknowledges this limitation and proposes an adaptive search procedure (Algorithm 3 in Appendix D) for implementation, the authors should be more upfront in the main text about this fact.

4. The structural assumption (equation 10) is key to connecting the local definition of GND to the global rates, but its introduction in section 4 feels abrupt and unexpected. It would help if it was alluded to and motivated in earlier sections.

### Questions
1. Could you clarify what makes the gradient-based normalization introduced in Definition 1(i.e., the term (1/γ) ||g||* ||h||) conceptually novel or more powerful than the traditional smoothness bounds used in analyses of cubically regularized Newton or trust-region methods? In particular, what analytical challenges does this specific normalization address or simplify that previous formulations do not?

2. Could you provide more intuition for why the harmonic mean structure is considered the “right” formulation? Does it emerge naturally from combining different smoothness assumptions, or is it primarily introduced as a flexible, unifying model chosen to encompass all the examples presented in the paper?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the concept of Gradient-Normalized Smoothness (GNS) as a new local characterization of the gradient field and Hessian approximation in optimization, particularly for the design and convergence analysis of second-order methods with approximate Hessians. Building on this notion, the authors propose a global convergence framework for both convex and non-convex optimization, unifying the treatment of smoothness and approximate second-order information under a single analytic lens. The work claims to recover state-of-the-art complexities for a wide range of problem classes—including Hölder-smooth, self-concordant, and generalized smoothness settings—and demonstrates the practical advantages and competitive empirical performance of the approach in numerous experiments, particularly with logistic regression, softmax, and non-convex losses.

### Strengths
**1. Unified Theoretical Framework**: The introduction and analysis of Gradient-Normalized Smoothness offers a conceptually appealing unification of smoothness and Hessian approximation error, allowing the same framework to recover or extend the best known rates for several classes of optimization problems.

**2. Strong Experimental and Visual Evidence.**: The paper provides thorough experimental validation with clear visualizations, which is easy to follow, and I like the layout of the paper (e.g., highlighting of some important examples and definitions).

### Weaknesses
I am not an expert in optimization; therefore, from my perspective, the paper does not exhibit any specific weaknesses. I will finalize my rating after reviewing other reviewers’ comments and the authors’ responses.

### Questions
**Q1.** The main challenge in second-order optimization lies in the high cost of computing the Hessian matrix. However, in practice, the Hessian–vector product can be computed efficiently without forming the full matrix by simply applying automatic differentiation twice. Could the authors briefly explain why such techniques are not suitable or sufficient for the tasks considered in this paper?

### Soundness
3

### Presentation
3

### Contribution
3
