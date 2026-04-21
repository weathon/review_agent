# Revisiting Convergence: A Study on Shuffling-Type Gradient Methods

- Avg Score: 5.75
- Decision: Reject
- Scores: 5, 5, 5, 8

## Abstract
Shuffling-type gradient methods are favored in practice for their simplicity and rapid empirical performance. Despite extensive development of convergence guarantees under various assumptions in recent years, most require the Lipschitz smoothness condition, which is often not met in common machine learning models. We highlight this issue with specific counterexamples. To address this gap, we revisit the convergence rates of shuffling-type gradient methods without assuming Lipschitz smoothness. Using our stepsize strategy, the shuffling-type gradient algorithm not only converges under weaker assumptions but also match the current best-known convergence rates, thereby broadening its applicability. We prove the convergence rates for nonconvex, strongly convex, and non-strongly convex cases, each under both random reshuffling and arbitrary shuffling schemes, under a general bounded variance condition. Numerical experiments further validate the performance of our shuffling-type gradient algorithm, underscoring its practical efficacy.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper examines the convergence rate of shuffling-type gradient methods without assuming Lipschitz smoothness, achieving results that match the current best-known convergence rates. The theoretical analysis covers non-convex, strongly convex, and non-strongly convex cases under both random reshuffling and arbitrary shuffling schemes.

### Strengths
1. The theoretical analysis is rigorous and considers multiple cases for different function properties.
2. The authors discuss the limitations of their work and suggest directions for future research.

### Weaknesses
My main concern is the experimental section, which feels too limited and simple to fully support the theoretical findings.
1. The data used in the experiments is relatively simple. It would be valuable to see if this method remains effective on more complex datasets, such as image datasets or applications with large language models.
2. Additionally, since the theoretical analysis includes both random reshuffling and arbitrary shuffling, it would strengthen the paper to show results for both methods compared to the baseline SGD.
3. Similarly, since the analysis considers three different cases (non-convex, strongly convex, and non-strongly convex), conducting experiments separately under each case would add depth to the findings.

### Questions
See the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper considers shuffling-type gradient descent under more general smoothness assumption -- $L_0, L_1$-smoothness.

### Strengths
Results match the standard Lipschitz smoothness rates

### Weaknesses
Variance assumptions, which are stronger than the most standard bounded variance assumption

### Questions
- Definition 2.2 seems missing subscript for $x$ in the right hand side. It is unclear what authors mean. It could be either symmetric (x is any of  $x_1$ or $x_2$) either non-symmetric $L_0, L_1$-smoothness (maximization over $x_1$ $x_2$ interval), see [1]. Or something different? Symmetric case is much easier to analyze. I tried to get it through the proofs, and it was very strange to see, that everywhere (e.g. Lemma A.2, Lemma A.4 ) authors used $\|\|\nabla F(\omega)\|\| \le G$, and then said (line 672) that "From definition 2.2, we can use Lipschitz smoothness". The standard smoothness AFAIU. So the question: Can the $G$ be huge? Is it just hidden to the $\cO$? and  is the analisys mainly like for the standard smoothness?
It seems to me that it actually is, because in lines 615, 704, 890, 1000, etc authors use standard smoothness inequalities. 
I simply can bound  $|\|\nabla f(x) \|\| = |\|\nabla f(x) - \nabla f(x^*) \|\| \ \le (L_0 + L_1|\|\nabla f(x^*) \|\|)R = RL_0$ which could be a G.
and then my effective smoothness is $L= L_0 + RL_0L_1$.
In the non-symmetric we have extra exponents as a multipliers, according to [1].
Is this what authors effectively did? Of course the one can recover the same rate as for standard smoothness. The problem is that the constant will be huge.

- what is $r := G/L$ in Theorem 4.5, Theorem 4.6, Lemma A.1, Lemma A.2.  I couldn't find where authors referr to $r$. What is it for? The results of the mentioned theorems and lemmas do not depend on $r$.
Then $r$ is appearing in Lemma A.4 and in bounds two subsequent trajectory points difference norm. Which mainly coincides  with my above bound on the gradient (if we plug in the step).

I briefly check the proofs and it seems they are to adapting my above bound on the gradient norm to the stochastic case (which is where the weaker variance assumption is used -- no expectation, and the difference between the full gradient and its stochastic counterpart is estimated).

However, the correct approach is to allow the step size to increase as the gradient norm approaches zero. E.g [1] suggests clipping with gradient norm in the denominator -- when the norm is large - the stepsize is small and vice-versa.

If I was mistaken, I would be glad to investigate the proofs more carefully if authors argued that I was wrong. My understating is that gradient norm is just trivially bounded and the contribution is poor.

References:
[1] Z. Chen et al. 2023 Generalized-Smooth Nonconvex Optimization is As Efficient As Smooth Nonconvex Optimization

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the shuffling method under the generalized smooth assumption, which was proposed recently to fit many modern machine learning tasks. The authors proved that, under properly picked parameters, the shuffling method provably converges under the weak smoothness condition for both nonconvex/strongly convex/convex objectives. Numerical experiments are also conducted to support the theory.

### Strengths
I appreciate the paper tries to tackle more realistic problems (i.e., functions with non-uniform smoothness) and studies the shuffling algorithm, an arguably more common scheme in practice.

### Weaknesses
**Major points**.

1. All convergence results are proved in a high-probability manner. However, the dependence on the margin $\delta$ is in the order of $\mathrm{poly}(1/\delta)$, which makes the results weak. I also suggest the authors explicitly give the dependence on $\delta$ in the final sample complexity.

2. Some descriptions of the existing works contain wrong facts. **(addressed)**

   - Lines 90-91, the bounded variance assumption is **not** required to improve the rate $\mathcal{O}(1/T^2)$ to $\mathcal{O}(1/(nT^2))$ in Nguyen et al. (2021). Instead, Nguyen et al. (2021) can handle **unbounded** variance.

   - Lines 92-93, results in both Mishchenko et al. (2020) and Nguyen et al. (2021) hold under **unbounded** variance condition. The current description is not correct.

3. The conditions on noises, i.e., Assumptions 4.3, 4.4, and 4.7, are strong compared with the existing literature, which significantly reduces the impact of the work. I will elaborate more on this point in the following.


4. Nonconvex part. **(addressed)**

   - Random reshuffling scheme.

     - In this case, previous works only consider Assumption 4.7, or even a weaker version, i.e., the non-uniformly bounded variance, to obtain the $\mathcal{O}(\sqrt{n}/\epsilon^{3})$ sample complexity.

     - However, to recover the same rate, this work requires much stronger Assumptions 4.3 and 4.4 in Theorems 4.5 and 4.6, respectively. Hence, the results in this paper are not directly comparable to prior literature.

     - When only Assumption 4.7 holds, Corollary 4.8 has extra dependence on $n$ as indicated by the authors.

     - In addition, I am not sure why Corollary 4.8 is a corollary and cannot find its proof. Did I miss anything?

   -  Arbitrary shuffling scheme.

      - Again, the authors require stronger Assumption 4.3 to make their sample complexity as good as the previous results. However, the latter can hold under non-uniformly smoothness, e.g., see Nguyen et al. (2021). As such, the claim in Lines 63-64 is misleading.

   - Moreover, imposing three assumptions on noises may confuse the reader. Especially, the proofs under Assumptions 4.3 and 4.4 are similar as claimed in Lines 860-863. I didn't see a necessary reason for the authors to do so.

4. Strongly convex and convex parts. 

   - Assumption 4.3 is strong and not assumed in previous best-known results, making the contributions weak. **(addressed)**

   - As far as I know, the previous best results don't need any assumption on the noises for convex problems (Assumption 4.14). Hence, whichever condition among Assumptions 4.3, 4.4, and 4.7 is used, the result is not an improvement in my opinion.

5. The writing can be improved. Some theorems give a threshold on the stepsize (e.g., Theorem 4.5) but others give an exact choice (e.g., Theorem 4.12). Can the author present a unified statement? **(addressed)**

**Minor points**. **(addressed)**

1. Line 290, the second $T=\mathcal{O}(\frac{1}{\epsilon^3})$ should be $\mathcal{O}(\frac{n}{\epsilon^3})$.

2. Line 310, $Delta_1$ should be $\Delta_1$.

### Questions
See **Weaknesses**.

### Soundness
2

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
2

### Summary
This paper revisits the case of random shuffling-type stochastic gradient methods for finite sum minimization problems. More precisely, the paper considers objectives without  the traditional structural assumption of Lipschitz smoothness. In doing so, the authors focus on non-convex, strongly convex and convex objectives which satisfy as a smoothness "surrogate" the notion of $\mathcal{l}-$ smoothness.
To that end, the authors provide respective convergence rates for each respective case.

### Strengths
The paper is extremely well written and enjoyable to read. Moreover, as far I checked the math seems correct and sound. Without being an expert myself on the respective literature, I find the respective very interesting and challenging from a theoretical standpoint.

### Weaknesses
My main concerns consider two main factors:

First, the notion of $\mathcal{l}-$ smoothness introduces additional parameters to be tuned as it becomes apparent from the definitions of the step-sizes in the main theorems. It would be could to include some discussion of how difficult are these to be evaluated both  in real life scenarios and in theory. More precisely, do the authors believe that the respective toolbox from the adaptive  algorithm like in [1] can be incorporated?

Secondly, the proposed step-size policies seem to rely on a prior knowledge on the iteration horizon $T$. Do the authors believe that an any time convergence rate guarantee can be achieved? 


[1] Adaptive Stochastic Variance Reduction for Non-convex Finite-Sum Minimization, Neurips 2022.

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
