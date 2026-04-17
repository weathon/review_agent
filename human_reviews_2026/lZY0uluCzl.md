# Interactive Learning of Single-Index Models via Stochastic Gradient Descent

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Stochastic gradient descent (SGD) is a cornerstone algorithm for high-dimensional optimization, renowned for its empirical successes. Recent theoretical advances have provided a deep understanding of how SGD enables feature learning in high-dimensional nonlinear models, most notably the *single-index model* with i.i.d. data. In this work, we study the sequential learning problem for single-index models, also known as generalized linear bandits or ridge bandits, where SGD is a simple and natural solution, yet its learning dynamics remain largely unexplored. We show that, similar to the optimal interactive learner, SGD undergoes a distinct "burn-in" phase before entering the "learning" phase in this setting. Moreover, with an appropriately chosen learning rate schedule, a single SGD procedure simultaneously achieves near-optimal (or best-known) sample complexity and regret guarantees across both phases, for a broad class of link functions. Our results demonstrate that SGD remains highly competitive for learning single-index models under adaptive data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a variation of Single Index Models where the features are 'interactive' meaning they depend on the current state of the parameter vector in a specific way and hence the data is not iid. The required sample complexity to learn the unknown parameter vector under SGD is studied. This paper proves sample complexity bounds on estimating the unknown parameter vector to arbitrary precision for a class of link functions which according to other recent works is tight up to poly-logarithmic factors. Additionally for the same set up the authors prove regret bounds. 

I think the results shown in this paper are interesting and begin to contribute to filling in the picture of SGD and Single Index Models with different features, however, I think the results are somewhat marginal as there seem to be other very closely related questions regarding sample complexity of these models which are not quite flushed out here. Please see the sections on weaknesses and my questions below. If my concerns can be addressed and I can be convinced that the story is more complete than I currently suspect it to be I would be happy to increase my scores. Currently I think this is a good start to a paper but needs a bit more flushing out to be complete.

### Strengths
The paper proves novel results for understanding the training dynamics and sample complexity of SGD for Single Index Models with interactive features. The results are novel and demonstrate the value of SGD in this setting. The paper does a good job at explaining their results and the outlines of the proofs which are often otherwise hidden in the appendix.

### Weaknesses
The assumptions feel rather restrictive, particularly monotonicity of the link function. There is discussion of the 'necessity' of the monotonicity assumption, however there is not nearly enough detail to convince me that the assumption is necessary. If the assumption is truly necessary for the results it would be nice to have a rigorous negative result such as a counter example demonstrating that a lack of monotonicity will indeed break the conclusion of the proof.

The relationship of the information exponent in the Gaussian iid feature case is discussed and how the concept does not directly apply to the non-interactive setting. However the information exponent in general relates to the taylor expansion of the population loss and specifically for single index models with iid gaussian data this concept reduces to checking the Hermite coefficients of the link function. It is unsurprising that when the data is not gaussian the Hermite coefficients are no longer important. What would be nice is to consider the original notion of information exponent applied specifically to this problem. For example it is not clear to me that the assumptions (monotonicity and either bounded derivative from below or convexity) do not somehow still just imply information exponent 1 but under a more appropriate definition.

### Questions
Do you suspect that there is a similar notion of information exponent for single index models with 'interactive features'? With Gaussian data the sample complexity is of course strongly dependant on the information exponent and can vary widely. Here, only one case is presented: i.e. you can solve the problem with quadratic complexity for the given class of functions. Do you suspect that it is possible to solve the problem for a more general class of link functions given larger sample complexity? Or that outside of the given class you cannot solve the problem? Simulations may be informative as well to answer these questions.

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
4

### Summary
In this paper, the author study the time/sample complexity of learning single-index models when the learner is allowed 
to query a specific, instead of random, point. They show that a simple SGD-type algorithm can achieve the best-known 
bounds on this problem, under the assumption that (1) the target function is monotone and has a nonzero derivative 
at $0$, or (2) the target function is convex. 
At each step, the algorithm queries the reward/target value at a perturbed version of the current weight and update the 
weight using the (spherical) SGD. The size of the perturbation controls the exploration-exploitation trade-off: 
In the burn-in stage ($1/\sqrt{d}$ to constant correlation) and the exploration stage (constant correlation to $1 - o(1)$
correlation), larger perturbations are used, and in the final exploitation stage (minimizing the regret in the $1 - o(1)$
correlation regime), a small perturbation is used.

### Strengths
* Overall, this is a well-written paper and is easy-to-follow. In addition, it is short (19 pages), which is a nice and 
  rare thing for a theory paper to have.
* It is somewhat surprising that how being able to choose the position of the query greatly simplifies the analysis and 
  improves the bounds (when the label noise is large). They choose the next query position $a_t$ to be a weighted average 
  of the current weight $\theta_t$ and a noise that is *orthogonal* to the current weight. This makes the 
  $f( \theta_t \cdot a_t )$ part deterministic. Together with the monotonicity/convexity assumption, the analysis 
  becomes much cleaner than the usual analysis.

### Weaknesses
This is a neat paper that does everything the authors claim to achieve, so I do not think there is any major weakness,
though one could complain that the setting is too easy. Nevertheless, the following are a few complaints I have. 

* As someone who is more familiar with IE/Gaussian single-index models, I found the $\tilde{O}(d^2)$ bounds really 
  confusing until I realized that the scaling is different, as the non-interactive bounds are $\tilde{O}(d)$. It 
  might be better to point this out early on, instead of putting the discussion in the Related Work section and 
  Section 5.
* The comparison with the information exponent results is not entirely fair, as the discrepancy comes mainly from the 
  label noises $\epsilon_t$. Without the label noise (i.e., we have access to $f(\theta^*, a)$), the IE bounds are 
  invariant under rescaling. It seems that being able to choosing the query point does not lead to improvements in 
  the no-label-noise setting. Moreover, if the link function is $f(x)=x^{2q}$ for some positive integer $q$ (the convex 
  setting), the IE is $2$, so the IE bound is $\tilde{O}(d)$, while Theorem 2(2) depends on $1/f'(1/\sqrt{d})$, which 
  can be large when $q$ is large.

### Questions
* See the 2nd point of the weakness section. In particular, how do your bounds depend on the size of the label noise?
  Can they recover/improve over the usual IE bounds when there are no label noises or the label noise is small?

### Soundness
3

### Presentation
4

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
This paper considers interactive SGD for single-index bandit problems where the reward is a potentially non-linear function of the dot-product between the current and optimal actions. The paper analyses both a burn-in phase to get a constant dot-product, and a learning phase which starts from a warm-start initialization and achieves a $\tilde{O}(d^2/\varepsilon)$ sample-complexity and $\tilde{O}(d\sqrt{T})$ regret.

### Strengths
While there is now a rich literature for learning single-index models in the supervised setting, I believe the literature on online/bandit settings is more sparse. Therefore, the problem that this paper wants to tackle is novel and of importance to the community. Also, the paper is explicit about its assumptions and it is generally easy to read and follow.

### Weaknesses
My main concerns are the following:
* In the pure exploration case $\sigma_t = 1$, this algorithm is the same as one-pass SGD studied by Ben Arous et al., 2021. However, the sample complexity seems to be worse. Due to the monotonicity of $f$ (hence information exponent 1), the sample complexity of (pure exploration) SGD would scale linearly with $d$ (at least in the noiseless setting, but I believe it should be able to tolerate $O(1)$ i.i.d. noise as well). However, the bound of Theorem 1 (SGD with warm-start initialization) scales with $O(d^2)$ and that of Corollary 1 can be as large as $O(d^p)$ for $f(x) = x^p$. It might be plausible that to get sub-linear regret, one should ultimately settle for a worse sample complexity, but if that's the case it should be better highlighted.

* The argument of Section 5 only shows that $f$ needs to be monotone around $m \approx 1$, and one can drive $m$ towards $1$ by pure exploration. This accommodates higher order Hermite polynomials, e.g. $H\_2,H\_6,...$. It would be interesting to know the effect of high information exponent in such cases.

### Questions
* I think it would be very useful if the schedule of $\sigma_t$ could be presented more explicitly for regret minimization in Corollary 1.

### Soundness
2

### Presentation
3

### Contribution
2
