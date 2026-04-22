# Learning the Inverse Temperature of Ising Models under Hard Constraints using One Sample

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
We consider the problem of estimating the inverse temperature parameter $\beta$ of an $n$-dimensional truncated Ising model using a single sample. Given a graph $G = (V,E)$ with $n$ vertices, a truncated Ising model is a probability distribution over the $n$-dimensional hypercube {-1,1}$^n$ where each configuration $\mathbf{\sigma}$ is constrained to lie in a truncation set $S \subseteq $ {-1,1}$^n$ and has probability $\Pr(\mathbf{\sigma}) \propto \exp(\beta\mathbf{\sigma}^\top A_G \mathbf{\sigma})$ with $A_G$ being the adjacency matrix of $G$. We adopt the recent setting of [Galanis et al. SODA'24], where the truncation set $S$ can be expressed as the set of satisfying assignments of a $k$-CNF formula. Given a single sample $\mathbf{\sigma}$ from a truncated Ising model, with inverse parameter $\beta^\*$, underlying graph $G$ of bounded degree $\Delta$ and $S$ being expressed as the set of satisfying assignments of a $k$-CNF formula, we design in nearly $\mathcal{O}(n)$ time an estimator $\hat{\beta}$ that is $\mathcal{O}(\Delta^3/\sqrt{n})$-consistent with the true parameter $\beta^\*$ for $k \gtrsim \log(d^2 k)\Delta^3.$

Our estimator is based on the maximization of the pseudolikelihood, a notion that has received extensive analysis for various probabilistic models without [Chatterjee, Annals of Statistics '07] or with truncation [Galanis et al. SODA '24]. Our approach generalizes recent techniques from [Daskalakis et al. STOC '19, Galanis et al. SODA '24], to confront the more challenging setting of the truncated Ising model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers the problem of estimating the inverse temperature parameter of a Ising model constrained to the satisfying assignments of a $k$-SAT formula, from 1 sample. 

The graph and formula are given, and the graph is assumed to have bounded degree $\Delta$, and for the SAT formula the number of clauses is assumed large relative to the degree of the graph and the formula, $\Omega(\Delta^3\log(dk+1))$. The main result is an efficient (nearly linear-time) algorithm using the maximum pseudolikelihood estimator, which learns $\beta$ within $O(\Delta^3/\sqrt n)$ with high probability. There is an exponential dependence on the size $B$ of $\beta$.

(Note that the setting of "learning under 1 sample" encomposses the setting of learning under multiple ($N$) samples, by simply defining a graph and constraints that are disjoint copies of the original, giving the natural $1/\sqrt{N}$ scaling in error.)

### Strengths
The problem of learning graphical models is an important problem. This paper follows a line of work on learning the Ising model, where a natural generalization is to consider a model truncated by hard constraints. SAT formulas provide a natural family of constraints. The pseudolikelihood estimator is a widely used estimator for its computation efficiency, and this paper gives guarantees for it in a constrained setting.

The work is technically novel, as learning a model under constraints requires new techniques compared to learning without constraints. To bound error in the pseudo-likelihood estimator it suffices to (1) upper bound the first derivative of the log pseudo-likelihood using the technique of exchangeable pairs and (2) lower-bounding the Hessian. The latter is the main challenge, and is done by (a) constructing a linear-sized independent set such that conditioned on the variables outside, the variables inside become a product distribution (ignoring the constraints), and such that the restricted constraints still have large-enough clause sizes, ensured using the Lovasz Local Lemma; this product structure means there are a large number of flippable variables which (b) each contribute to the Hessian by at least a constant, lower bounded by lower-bounding the squared magnetizations of the neighbors of each $i$. 

The proof sketch does a good job of conveying the main ideas of the proof at a high level.

### Weaknesses
I find this to be an interesting learning theory paper, but my main concern is whether it is appropriate for the general machine learning audience at NeurIPS. A theoretical CS conference may be a better fit. In particular, the fact the problem considered is solely that of learning the temperature, rather than the entire model, makes the problem quite niche. 

The title is somewhat misleading, as the only parameter that is being learned is the temperature. I suggest "Learning the temperature of an Ising model under hard constraints".

The paper only deals with the case of bounded-degree graphs, and so does not apply to mean-field type models.

### Questions
Does the theory work in the case where different weights are allowed on the edges? Needing $A_{ij}\in \{\pm \frac 1\Delta\}$ is quite limiting. It would be much more general to allow all $|A_{ij}|\le \frac 1\Delta$. 

It's necessary for Theorem 2 to exclude the empty graph (where the model is the same under every temperature). Is it sufficient to exclude the empty graph or is there a missing condition?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
At a high-level this can be seen as a paper that combines learning ising models from one sample with learnign with truncations. More specifically, the paper studies one-sample estimation of the inverse temperature $\beta$ for a truncated Ising models where the state space is the satisfying set $S$ of a bounded-degree $k$-SAT formula. The estimator is the standard maximum pseudolikelihood (MPLE), optimized by projected gradient descent. The analysis shows the MPLE gradient concentrates at $\beta^*$ (via exchangeable pairs) and that the normalized log-pseudolikelihood is strongly convex w.h.p. The latter is the challenging part, and requires proving the existence of a linear number of flippable coordinates using an independent-set/LLL argument, under large enough $k$.

### Strengths
- The paper presents a novel setting, that seems somewhat motivated.
- The analysis requires new techniques to prove strong convexity, which involves circumventing a few challenges.
- The paper is generally well-written, and easy to follow. It also does a good job of covering related work.

### Weaknesses
- The paper could do with a bit more motivation for the setting, and in particular $k$-SAT, since it seems more of a purely theoretical pursuit.
- The clause-size requirement $k \gtrsim e^{2B}\Delta^3(\log d+\log k)$ is quite large. In bounded-degree regimes, such $k$ pushes the truncation toward an LLL-like easy regime where $|S|/2^n = 1-o(1)$. The constraint can become somewhat vacuous, making the setting not clear.
- The assumptions and dependencies seem a bit loose or proof technique specific. For example, Assumption $A_{ij}\in{\pm 1/\Delta}$ and $\Delta=o(n^{1/6})$ feels stronger than necessary.

### Questions
- My main question is around the choice of $k$ being justified. As a follow-up, which of their techniques would extend to more interesting settings?
- It would be helpful if the authors describe which dependencies are necessary and tight, and which are loose. Especially $\Delta$.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the problem of learning the inverse temperature of a truncated Ising model from one data sample. The Ising model is constrained to lie within a specific set (referred to as that satisfying the k-SAT formula). The authors provide various practical scenarios to motivate this setting. The theoretical results hinge on the analysis of the maximum pseudolikelihood estimator and demonstrate its consistency.

### Strengths
Theoretical arguments are well-stated (although there are some key definitions are missing; see Weaknesses section). The theoretical analysis is based on establishing guarantees on the first order and second order gradients of the pseudolikelihood estimator relative to the inverse temperature function. The novelty is (seemingly) claimed in terms of establishing the bound on the second order derivative in Lemma 3.3.

### Weaknesses
1. The key definition of $k$-SAT formula or how it leads to a hard constraint on the Ising model is not defined with sufficient rigor. Therefore, I could not fully understand why this particular setting was of interest.
2. The novelty within the theoretical results should be emphasized better- some results seem to be simple extensions of previous works, while others are novel. The distinction between the two should be elucidated.
3. While I appreciate the theoretical flavor of this work, some effort into making the theoretical results interpretable for practitioners would have improved this paper.

### Questions
1. What is meant by ${\cal O}(\Delta^3 n \log n)$-Algorithm?
2. The theoretical bound in Theorem 2 becomes tighter with increasing $n$--suggesting that the estimator is likely to perform better when estimating a larger Ising model with a single sample. Could you elaborate on this aspect? Shouldn't the estimation problem become 'harder' with increasing 'n'?

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
4

### Summary
The paper considers a truncated Ising model, where an Ising model on a graph G of size $n$ with inverse temperature $\beta$ is sampled and samples are accepted only based on satisfiable assignments for a $k$-SAT formula. The main question is can you learn inverse temperature parameter $\beta$ from a single sample from this truncated model. 

Authors show that one can learn the inverse temperature parameter up to an error of $\Delta^3/\sqrt{n}$ when the number of clauses in the k-SAT formula exceeds $\tilde{O}({\Delta^3})$ and the inverse temperature is in a bounded interval around $0$. Authors main key ideas are by doing a MLE estimate for $\beta$ on the pseudo likelihood which is the product of likelihood of Glauber flip of every variable conditioned on all others.  1) First, the error in the MLE estimate is bounded by ratio of the magnitude of the first derivative and the minimum second derivative of the log pseudo likelihood. So authors proceed to show that the numerator is small and lower bound the second derivative.  2) Concentration due to exchangeable pairs take care of the concentration of the first derivative which has to be close to zero for the stationary point with a deviation of $\sqrt{n}$  and the lower bound on the second derivative is at least n/\Delta^3. 
3) Key issue is that second derivative's lower bound is non trivial only if there are linear number of flippable variables - variable coordinates that can be flipped from the sample without violating $k-SAT$ constraints. Clever trick is that authors show that there is a large enough independent set in the graph such that it encapsulates linear number of variables and conditioning on the rest, the probability of variables in the independent set factorizes. Then it is easy to show that some variable from the independent set will satisfy the clause in which another fixed variable is in with constant probability. This ensures there are enough flipped variables which contributes then to the lower bound on the second derivative.

### Strengths
Paper considers a very non trivial Ising model with a truncation on the set of satisfiable boolean assignments in a K-SAT formula. Then authors use the Glauber flip based pseudo likelihood to do the MLE estimation of inverse temperature beta from a single sample. 

Techniques are quite novel compared to classical results on single sample learning of un-truncated Ising models and other recent works. specifically the independence set finding algorithm that truncates the k-SAT formula with some guarantees used a clean and elegant version of Lovasz Local Lemma. The trick of creating a conditional product distribution and arguing about flippability of variables that lower bound the second derivative is also pretty neat.

### Weaknesses
1) My main concern is that Section 3.3.1 starts with arguments ensuring enough variables are flippable. However, there is only a small two line comment of how it is related to second derivative lower bounding. So the reader is forced to look in the supplement as to why this issue is prominent. lemma 3.7 in the supplement shows why enough of such flippable variables are important. It is important to include that argument (in a sketch form) at the beginning of Section 3.3.1 

2) Where does the constraint (or what steps in the proof contribute): $\Delta \sim o(n^{1/6}) $ comes from precisely ? 

3) After an independent set is chosen, then the rest of the argument about flippability splits into two parts - one is flippability of variables within the independent set and outside. It seems like the first one directly follows from prior work ? So the main novelty is in the construction of the independent set and the argument for flippability of variables outside it ?

### Questions
I have asked all my questions in the weakness section.

### Soundness
4

### Presentation
3

### Contribution
4
