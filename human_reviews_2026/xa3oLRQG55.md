# Solvable Gradient Flow Through Diagram Expansions

- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
We propose a general diagram-based approach to analyze scaling regimes and obtain explicit analytic solutions for gradient descent evolution in large learning problems.
We propose a general diagram-based approach to analyze scaling regimes and obtain explicit analytic solutions for gradient descent evolution in large learning problems.
We focus on a class of problems in which an identity tensor is learned by gradient descent starting from a sum of rank-one tensors with random normal weights.
A central element of our approach is to expand the loss evolution in a formal power series over time. The coefficients of this expansion can be described in terms of suitable diagrams akin to Feynman diagrams. 
Depending on the scaling of the initial weight magnitude and the number of parameters, we find several extreme learning regimes, such as NTK, mean-field, under-parameterized learning, and free evolution. 
These regimes include lazy training as well as strong feature learning. We identify these regimes with extreme points and sides of a hyperparameter polygon. 
We then show that in some of these regimes, the loss power series satisfies a formal partial differential equation. For certain scenarios, this equation is first order and can be solved by the method of characteristics, producing explicit loss evolution formulas that agree very well with experiment. 
We give a series of specific examples where this methodology is fully implemented.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a diagrammatic expansion for analyzing gradient flow learning of order ν identity tensor, from a sum of rank one factors with Gaussian initialization, that yields a formal time power series for the expected loss, whose coefficients are computed by diagram pairings and contractions. This allows to classify scaling regimes, such as NTK, mean field, and no learning, by a hyperparameter polygon, that organizes how the target size p, the model size H, and the initial \sigma, scale. The series are turned into generating functions, that satisfy first order PDEs, which are solved by characteristics, producing closed form learning curves, such as an explicit loss decay in free evolution and solvable expressions in under and over parameterized cases.  These predictions are compared with certain numerical gradient descent experiments and match closely. Summary: the overall contributions are:  (i) a diagrammatic calculus for gradient flow dynamics with explicit coefficient counting, (ii) the hyperparameter polygon taxonomy linking canonical limits, such as NTK and mean field, and (iii) closed form solutions for loss dynamics that align with experiments.

### Strengths
(i) Analytic framework that combines a formal loss time series with a diagrammatic calculus to get an explicit dynamics of the expected loss for the identity tensor factorization task. In some regimes, the power series can be extended to a multivariate generating function, that obeys a first order PDE, which can be solved in certain cases. (ii) A hyperparameter polygon, that allows for a classification of large scale limits that recover known regimes, such as the NTK and mean field. (iii) Numerical experiments align well with the theory across regimes.

### Weaknesses
(i) A narrow task: the theory is demonstrated on learning the identity tensor, (ii) The analysis requires large p,H, while finite width corrections are not characterized,(iii) The theory is for gradient flow, and the numerics use Euler discretization to mimic it, which differs from practical SGD, (iv) The method is not rigorous, the PDEs for the generating functions are derived from formal series, while convergence and error bounds are not proven.

### Questions
Suggestions for improvement of the paper: (i) Make the formal method more rigorous by addining the remainder, when going from the formal series to PDE in one regime, which would elevate the method from formal to controlled, (ii) Quantify the effect of dropping non Pareto optimal terms, by providing bounds, or empirical checks, for the size of discarded terms vs. kept terms across polygon regions, (iii) Add an estimate of the leading finite size corrections to the large p and H formulas, e.g. via next to leading order contractions, and compare to numerics. This will narrow the gap between asymptotics and practice.  (iv) Bridge gradient flow and the discrete training by a discretization error analysis.
(v)  Add, using theory or numerics, an example of a simple non identity structured target.

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
2

### Summary
The authors consider a tensorial generalization of the matrix factorization problem in linear deep learning. 
In this case, they are able to compute the expected value of the loss during the learning evolution as modeled by gradient flow using Feynman diagram expansion techniques.
Using this expansion, they classify the learning regime of the dynamics (lazy, rich, etc.) depending on the variance used to initialize the model parameters as well as the values of other parameters playing the role of model complexity and target size.

### Strengths
* The author are able to provide an explicit formula for the expected value expansion for the loss under gradient flow
* The classification of previously identified learning regimes (lazy, rich, etc.) depending on the model hyper-parameters is very satisfying
* The use of Feynman expansions in this way, and the general approach is novel to deep learning and mathematically interesting

### Weaknesses
* The highly mathematical technicality of the paper made it hard for me to assess its correctness. (Part of me thinks this paper may be more appropriate for an applied mathematical journal submission.) The exposition is in general could be improved by focusing in the main paper on a simpler case (for instance matrix factorization), while deferring the more general case of tensor factorization to the appendix. 

* The possibility of the Feynman expansion technique seems to rely heavily on the non-linearity of the network. It would help if there were some comments on how to extend the method beyond this linear toy model. Is this approach a dead-end beyond linear networks, or can it be applied to more useful settings? Is it possible to comment on this briefly?

* The learning-regime classification depends on the assumption of gradient flow. However, in practice, optimization is done in discrete steps whose size influences the learning regime (higher learning rates have been show to bias the model trajectories to flatter, more generalizable solutions for example). Can the author comment on the possibility of extending their method to practical discrete optimization?

### Questions
See weakness section.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors analyze from an asymptotic perspective the gradient flow training dynamics of a tensor approximation problem. Specifically, they consider the problem of approximating in the least squares sense a diagonal identity tensor of order \nu and dimension p by a tensor of rank at most $H$, and focus on the gradient flow dynamics with i.i.d. random Gaussian initialization of variance $\sigma$. 
By exploiting the polynomial nature the loss, they express its time derivatives of any order as polynomials and use so-called "diagrams" (that take the form of certain multi-graphs) to provide "formal expressions" of the involved polynomials using a so-called "diagram merging" operations. They then invoke Wick's theorem and the diagram viewpoint to decompose the expectation of these polynomials (with respect to the Gaussian initialization) into terms expressed as $p^q H^n \sigma^{2\ell}$. Invoking diagram arguments that I was not able to follow, they conduct a Pareto-optimal analysis to identify which exponents $q,n,\ell$ can dominate the other, depending on the relative scaling of $p$, $H$ and $\sigma$. This Pareto-optimal viewpoint is then exploited to identify scaling regimes, which in turn allow to express an asymptotic partial differential equation leading to explicit "asymptotic solutions" in certain regimes.

### Strengths
It is very welcome to analyze the gradient flow dynamics of large-scale learning problems, and the idea that this can be achieved with certain polynomial loss functions is very attractive. The exploitation of the fact that the time derivatives are also polynomials and that their dominant terms can be identified is also very appealing and interesting.
If the results are correct, they do provide some interesting insight on the behavior expected in different scaling regimes (relating the problem dimension $p$, its rank $H$, the initialization scale $\sigma$, and the learning rate $1/T$).

### Weaknesses
The paper heavily relies on concepts, terminology and viewpoints that seems to require a (statistical) physics background, making it very hard to access for a general reader with a standard machine learning or mathematics background.
The use of so-called "diagrams", which are certainly not well-known to the average ICLR audience and are very tersely described, makes it particularly difficult to follow the reasoning. I was not able to check whether the results are correct, due to the heavy use of such informal arguments, combined with "formal expansions" (where actual convergence analysis of series would also seem necessary) and  "non-rigorous" aspects of the proofs in particular when deriving asymptotic PDEs.  These are the main reasons for my grade on "soundness" and "presentation".

While I reckon that the general approach (cf my summary above) *seems* sensible (hence my grade 3 on "contribution"),  it would need to be much more clearly and convincingly conveyed by postponing as much as possible the "diagram" viewpoint and making it much more explicit via polynomials and lemmas where appropriate. 

While the title and beginning of the abstract suggest general results on gradient flows for large learning problems, the setting consider in the paper is very specific, focused on very particular tensor optimisation problems with identity target tensor. Despite claims below (3), this seems very far from being representative of general learning problems.
 
The paper lacks references to the vast literature on the considered tensor optimization problem, which is known under various names such as Canonical Polyadic Decomposition or PARAFAC decomposition, and is known to lead in general to a number of topological difficulties (for example, a minimizer of the loss does not necessarily exist - see e.g. De Silva & Lim "Tensor Rank And The Ill-Posedness Of The Best Low-Rank Approximation Problem"). There is a significant gap between the specific model considered here (in particular, using the identity tensor as a target) and the evoked richness of the "class of problems"  below (3).

The learning rate $1/T$ appearing the gradient flow equation (1) seems superfluous at first sight, since the solution to (1) with $T=1$ yields the solution for general T via a simple time rescaling. It would be helpful to mention this fact while explaining that  $T$ will nevertheless be useful to highlight how certain "time scales" appear in the asymptotic analysis.

### Questions
Can the approach be expressed without reliance on "diagrams" but simply on polynomials, using a notion of "polynomial merging" in (7) ?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors use novel analytical techniques to study the dynamics of a gradient flow algorithm on a specific learning task, which can be understood as learning the identity tensor using a sum of rank one components. They show that the loss for this
task, when the tensor is order 3, is similar to the population loss obtained for a modular arithmetic task in the Fourier basis, thus arguing that it captures real learning phenomena.
Using a diagrammatic approach that mixes tensor networks and Feynman diagrams, they are able to derive closed form expressions for the time evolution of the loss function averaged over the initialisation in some scaling regimes. Furthermore, they systematically
catalog different learning phases as a function of the scalings between the number of model parameters, the dimensionality of the target to be learned and the variance of the initialisation of the weights.

### Strengths
The analytical approach used in this paper is, as far as I can tell, very original and unique and it makes the model analytically solvable in some cases.
The expansion in time, and the subsequent mix of Feynman diagrams and tensor networks is a hybrid approach which combines techniques that come from different fields of computer science and physics. I particularly like that with this single approach, various scaling regimes for the hyper parameters of the model emerge naturally by looking at the dominating diagrams, establishing connections to well known topics such as NTK, Lazy Regime and Feature Learning, and derive natural scalings for the initialisation variance and learning rate.
This is a nice alternative to Dynamical Field Theory for the calculation of the dynamics of gradient descent, and it would be interesting to see if it can be applied for other learning tasks.

### Weaknesses
The main weakness is the relevance of the model studied. 
The connection to modular arithmetic for $\nu=3$ is interesting, however the link seems vague and very specific. Furthermore, the main interest of such modular arithmetic tasks is
typically the phenomenon of Grokking, which is not studied in the paper. 
It would have been nice to see the time evolution of the learning phases for $\nu=3$, and perhaps hope to observe the Grokking phenomenon analytically. 
The time evolution is only obtained for the $\nu=2$ case, which I can only connect to two layer linear networks, but with the unusual property that no dataset is present in the learning. Even disregarding this, no particular insights on the dynamics are derived. Ultimately it seems that the main feat of this paper is to derive the various scaling regimes, NTK, Mean Field etc..., with a single approach. I
wonder however if this technique can be used to study other learning models, and perhaps obtain more intuition in the dynamics of learning.
However, the relevance of the studied model seems limited, and no new interesting phenomena is explained using the analytical
solution. 
The importance of the paper seems to be more technical than anything else.

### Questions
For the presentation:
It would be nice if the notation were more explicitly stated, and quantities listed as vectors, matrices tensors, etc... 
For example in the first page we read ${\bf u}=\\{u\\}$, what does this mean? 
It seems like a pretty ambiguous notation. 
Also on page 2, it is stated that the case $\nu=2$ is equivalent to learning the identity matrix with matrices $UU^T$. What are the dimensions of the matrix $U$? Obviously this can all be derived by the reader, but if you explicitly give the dimensions of the matrix and state how this is connected to a deep linear network it makes it easier to read.
The literature review in the appendix is very long, and touches many topics which are only mildly related to the paper. I would suggest shortening it, and maybe inserting some of the most important references in the main text.
For the relevance for the paper:
I wonder if the case $\nu=2$, for which most of the results are derived, can also be connected to other known learning tasks, as in the case $\nu=3$. This would give more relevance to the results of this paper. The symmetric case for example seems similar to an a linear autoencoder, can the authors comment on this?
Can you find some interesting phenomena in your solved trajectories which can be maybe observed in more realistic learning tasks? 
A numerics section would give the paper more concreteness, and show that this diagrammatic method is really explaining some phenomena.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the gradient flow dynamics of learning a generic tensor under squared loss, aiming to provide insights on the gradient descent dynamics in large learning problems. To be specific, the authors consider a standard gradient flow with learning rate $1/T$: $$\frac{du}{dt} = - \frac{1}{T} \partial_u L(u),$$ where the weights of the model $u$ is a tensor. The target is assumed to be the identity tensor, and the loss $L$ is the standard quandratic loss. The main contributions of this paper are as follows: (i) The authors use a technique, called "diagram calculus", to computed the expected loss function at time $t$, in order to analyze the gradient flow dynamics. The exact formula derived turns out to be connected to the Feynman diagram; (ii) The contribution of each diagrammatic term in the expansion scales with the problem's parameters. By identifying the corresponding hyperparameter polygon, the paper identifies several scalings of the learning regime, e.g., NTK, mean-field and free evolutions; (iii) The paper also explores several special cases, where the gradient flow has a formal solution gives by power series, or an explicit solution given by the method of characteristics.

### Strengths
The use of "diagram calculus" to study gradient flow dynamics is interesting and novel, especially it is able to provide explicit, analytical expressions for $\mathbb{E} [L(t)]$ the expected loss of gradient flow at time $t$. It is also interesting to see that different learning regimes correspond to different "hyperparameter polygons" in the diagram.

### Weaknesses
I have a few concerns regarding the rigor and generality about the main results of this paper: (i) Lack of rigor. The main theorem (Theorem 1) is a mathematically rigourous derivation of power series expansion of $\mathbb{E} [L(t)]$. However, the proof of Theorem 2 is only a formal power series derivation rather than a rigorous proof. (ii) Limited generality. The entire theory, although elegent, is only developed for learning a specific tensor model with quadratic loss. It would be better to discuss more examples from deep learning for which the abstract theory from this paper can be applied to investigate the learning behavior. Otherwise, it might be clear why it is interesting to develop this "diagram calculus"-based method for studying gradient flow dynamics.

### Questions
Below are some specific comments and questions for the authors:
(i) On the bottom of page 6, partial differential equation is abbreviated as PDF. This also occurs in several other places in the paper, please correct.
(ii) Some appendix numbers are not properly referenced, e.g., in the statement of Theorem 1.
(iii) The main results are stated for expected loss, namely $\mathbb{E} [L(t)]$. Since the randomness here comes from the initialization, which is Gaussian, I was wondering whether there is a high-probability statement for $L(t)$? For example, some Gaussian concentration inequalities might help.

### Soundness
3

### Presentation
3

### Contribution
2
