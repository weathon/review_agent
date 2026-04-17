# Mean Estimation from Coarse Data: Characterizations and Efficient Algorithms

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Coarse data arise when learners observe only partial information about samples; namely, a set containing the sample rather than its exact value. This occurs naturally through measurement rounding, sensor limitations, and lag in economic systems. We study Gaussian mean estimation from coarse data, where each true sample $x$ is drawn from a $d$-dimensional Gaussian distribution with identity covariance, but is revealed only through the set of a partition containing $x$. When the coarse samples, roughly speaking, have ``low'' information, the mean cannot be uniquely recovered from observed samples (i.e., the problem is not *identifiable*). Recent work by Fotakis et al. (2021) established that *sample*-efficient mean estimation is possible when the unknown mean is *identifiable* and the partition consists of only *convex* sets. Moreover, they showed that without convexity, mean estimation becomes NP-hard. However, two fundamental questions remained open: 
1. When is the mean identifiable under convex partitions?
2. Is *computationally* efficient estimation possible under identifiability and convex partitions?

This work resolves both questions. We provide a geometric characterization of when a convex partition is identifiable, showing it depends on whether the convex sets form ``slabs'' in a direction. Second, we give the first polynomial-time algorithm for finding $\varepsilon$-accurate estimates of the Gaussian mean given coarse samples from an unknown convex partition, matching the optimal $\widetilde{O}(d/\varepsilon^2)$ sample complexity. Our results have direct applications to robust machine learning, particularly robustness to observation rounding. As a concrete example, we derive a sample- and computationally- efficient algorithm for linear regression with market friction, a canonical problem in using ML in economics, where exact prices are unobserved and one only sees a range containing the price (Rosett, 1959).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies Gaussian mean estimation when observations are only available through coarse sets rather than exact samples. 
It resolves two fundamental open questions from Fotakis et al. (2021) by (i) characterizing precisely when convex partitions are identifiable and (ii) providing the first polynomial-time algorithm for mean estimation under such identifiability. 
The characterization shows that non-identifiability occurs if and only if the partition consists of parallel slabs in some direction. 
The proposed algorithm performs stochastic gradient descent on a convex log-likelihood objective, with carefully controlled gradient moments and local strong convexity, achieving optimal $\tilde O(d/\omega^2)$ sample complexity. 
The paper also illustrates the approach on linear regression with market friction, demonstrating both the conceptual generality and computational feasibility of learning from coarse data.

### Strengths
**Strengths:**

– The contributions are very clearly explained and well connected to prior work.

– The introduction is well written, motivating the problem both from statistical and practical perspectives.

– The problem studied is theoretically interesting, addressing a fundamental question at the intersection of statistics and learning theory.

– The characterization of identifiability (Theorem 3.1) is elegant and provides geometric intuition.

– The proposed algorithm is explicit and computationally efficient.

– The paper is well organized and easy to follow despite the technical depth.

### Weaknesses
**Weaknesses:  **

- The assumption to only consider Gaussian distributions is restrictive; it would strengthen the paper to discuss how the techniques might extend to broader exponential families, or to outline obstacles or required modifications.  

- At several places the paper is lacking mathematical rigor. Most crucially, it is not clear what a  'probability distribution of a set' should be; please give a rigorous, measure-theoretic definition of \(\mathcal N_{\mathcal P}(\mu,I)\) as the pushforward of \(N(\mu,I)\) under the partition map and specify its sigma-algebra. As written, Definition 1 is hard to parse, and once \(\mathcal N_{\mathcal P}(\mu,I)\) is unclear, the meaning of KL divergence and TV distance between such objects is also opaque; moreover, the text sometimes treats distribution and  density as synonyms, which is mathematically incorrect.  

- There seems to be a close relation to the literature on imprecise probability and set-valued observations (credal sets, lower and upper previsions), which is not discussed; adding a short paragraph contrasting your coarse partition with imprecise-probability frameworks would help position the work.  

- The guarantees rely on the \(\varepsilon\)-information preservation, but the paper offers little guidance on verifying or estimating (\varepsilon\) in practice for a given partition beyond examples; a diagnostic or sufficient conditions (beyond non-slab or boundedness) would improve applicability.

### Questions
**Questions:**  

- You write 'making estimation information-theoretically impossible.' What is the precise mathematical meaning of this statement? Please clarify in terms of identifiability or distinguishability of distributions.  

- How can the definition of identifiability on page 2 be verified or practically tested for a given partition? Are there diagnostic criteria or examples that illustrate this?  

- In Figure 1, it is not immediately clear why panel (a) represents a non-identifiable partition while (b) and (c) are identifiable. Could you provide more intuition or explanation for this already when the figure first appears? As it stands, the reader can only fully understand this after reading Theorem 3.1.  

- Theorem 3.2 appears to be stronger than stated: you not only guarantee the existence of a polynomial-time algorithm, but in the proof you actually construct it explicitly. I suggest making this more prominent in the statement and the discussion.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the problem of mean estimation from coarse data. Concretely, the goal is to learn the location parameter, $\mu \in \mathbb{R}^d$, of a Normal distribution $\mathcal{N} (\mu, I)$ given access to i.i.d coarse samples from the distribution. Here, for a partitioning of $\mathbb{R}^d$, $\mathcal{P}$, one observes not a draw $X \thicksim \mathcal{N} (\mu, I)$ but rather the unique partition $P \in \mathcal{P}$ containing $X$. Prior work established that sample-efficient estimation is possible when $\mu$ is identifiable from coarse observations and the partitioning $\mathcal{P}$ consists only of convex sets. However, this prior work does not characterize \emph{when} $\mu$ is identifiable and if it is, when it can be estimated \emph{computationally} efficiently. The main contribution of this paper is to provide such a characterization. 

To see why identification is not always possible, consider the special case where $d = 2$, and $\mathcal{P} = \{P_i\}_{i \in \mathbb{N}}$ with $P_i \coloneqq \{(x, y): i < y_i \leq i + 1\}$. It is clear that $\mu_1$ (the first coordinate) is not identifiable as any translation along the $x$ coordinate induces the same distribution over partitions. The main result of the paper establishes that these are the only scenarios where identification is not possible. That is, identification is impossible when the partition consists of a set of cylinders along a particular axis, with identification impossible along the axis. Furthermore, they show that a simple gradient descent-based algorithm on the log-likelihood of our observations enables computationally efficient estimation with a mild strengthening of this condition. 

Technically, the main result follows from analyzing the (strict) convexity of the (log) likelihood function. When the partitions consist only of convex sets, the likelihood function can be easily shown to be convex. However, identification requires strict convexity. To do this, the starting point is an explicit characterization of the Hessian as the difference between $I$ and a weighted sum of conditional covariance matrices, which, from classical results in convex geometry, are dominated by $I$. To further extend these results to strict convexity, the authors carefully analyze when each of these conditional covariance matrices has variance $1$ along a particular direction. This, in turn, relies on more recent results in convex geometry.

Overall, this paper makes substantial progress on an interesting and well-motivated problem. In particular, the identifiability guarantees are natural and surprising. The proof, despite largely relying on classical techniques and results, is also quite non-trivial. I believe this paper would make an excellent contribution to the conference.

### Strengths
See main review

### Weaknesses
See main review

### Questions
See main review

### Soundness
4

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
This paper addresses the fundamental problem of Gaussian mean estimation when the observations are coarse data. The key theoretical contributions are resolving two fundamental questions related to identifiability and computational efficiency.
- Theorem 3.1: Intuitively, non-identifiability occurs if the partition is translation invariant in some direction. A convex partition $\mathcal{P}$ is not information preserving if and only if there is a unit vector $v \in \mathbb{R}^d$ such that almost every set $P \in \mathcal{P}$ is a slab in direction $v$.
- Theorem 3.2 provides a polynomial-time algorithm for finding $\epsilon$-accurate estimates of $\hat{\mu}$ for any convex, $\alpha$-information preserving partition $\mathcal{P}$.

The key assumption enabling the tractability and efficiency of Theorem 3.2 is that the partition $\mathcal{P}$ is convex and information-preserving.

### Strengths
The paper provides novel, complete solutions to the fundamental question in Gaussian mean estimation from coarse data. The proposed algorithm has optimal sample complexity. 
- It establishes that a convex partition is non-identifiable if and only if almost every set is a slab in some common direction.
- It shows that identifiability is a property of the partition structure itself, as it does not depend on the true mean $\mu^{*}$.
- The algorithm achieves the optimal sample complexity of $\tilde{O}(d/\epsilon^2)$.
- $\mathcal{Z}(\mu)$ is proven to be convex because the individual component functions, $\mathcal{Z}_P(\mu) = -\log(\mathcal{N}(\mu, I; P))$, are convex for any convex set $P$ (Proposition B.2). 
- The $\alpha$-information preservation assumption is used to prove that $\mathcal{Z}(\mu)$ satisfies a local growth condition around $\mu^{*}$ (Lemma B.3). 
- This effectively means the function is locally strongly convex near the true mean $\mu^{*}$. 
- This condition guarantees that finding an approximate minimizer $\hat{\mu}$ in function value ($\mathcal{Z}(\hat{\mu}) \approx$ $\mathcal{Z} (\mu^{*})$) ensures $\hat{\mu}$ is close to $\mu^{*}$ in $L_2$ distance.
- The natural gradient expression involves expectations over samples $x$ whose observed sets $P$ can be unbounded, leading to an unbounded variance in the gradient.
- They resolves this by introducing an idealized R-Local Partition $\mathcal{P}(R)$ (Definition 5).
- The final step is to remove the temporary R-Local Partition assumption and prove the theorem for a general convex partition $\mathcal{P}$.The argument sets the bounding radius $R$ large enough ($R = D + O(\log(md/\delta))$) so that the true, unobserved samples $x_i$ fall within $B_{\infty}(0, R)$ with high probability.
- The total sample complexity is derived by plugging the calculated local growth parameter ($\eta$) and the bounded second moment ($G^2$) into the PSGD convergence analysis (Theorem C.1)

### Weaknesses
- The polynomial time complexity relies on the existence of an efficient sampling oracle for a truncated Gaussian over the observed convex set $P$ (or $P \cap B_{\infty}(0, R)$). While this is achievable in $poly(d)$ time for polytopes using MCMC (Hit-and-Run) methods, these methods typically have high polynomial dependency and are usually slow in practice.
- The paper specifies the requirement for an efficient sampling oracle (Assumption 1) that outputs an unbiased sample $y \sim \mathcal{N}(\mu, I, P \cap B_{\infty}(0, R))$. This is an oracle for sampling from a truncated Gaussian distribution over a convex set. While the paper explicitly references the Hit-And-Run MCMC in Theorem D.1 as the general tool for implementing log-concave sampling over convex bodies (Appendix D), Langevin Monte Carlo (LMC), specifically Projected Langevin Monte Carlo (PLMC), is a highly relevant alternative method that could potentially be used. State-of-the-art PLMC methods often have better theoretical dependence on the dimension, such as $\tilde{O}(d)$ under certain conditions.
- The paper could be clearer on whether the entire partition $\mathcal{P}$ must have a manageable, unified representation, or just the observed set $P_i$. If the partition is dense (e.g., infinitely many sets), managing the required infrastructure for all $P \in \mathcal{P}$ could be impossible.

### Questions
See "weakness"

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of estimating mean $\mu$ of a Gaussian distribution $\mathcal{N}(\mu, I)$ with identity covariance from coarse data, where the learner only observes which cell of a (potentially unknown) partition each sample falls into. The authors resolve two open questions from prior work (Fotakis et al., 2021):

 (i) Identifiability Characterization: For convex partitions, they provide a geometric characterization of when an instance is identifiable, i.e., the mean can be uniquely determined from observed samples. They show that an instance is non-identifiable if and only if almost all partition cells are parallel slabs in the same direction.

(ii) Efficient Algorithm:  They provide the first polynomial-time estimator for this problem under the assumptions that a bound $D$ on $\|| \mu \||$ is known and the partition is $\alpha$-information preserving (a stronger, quantitative version of identifiability assumption).

### Strengths
The coarse-data model is well motivated, and captures realistic phenomena like rounding, sensor quantization, and economic market friction. 

The paper makes theoretical contributions by resolving two questions definitively: the identifiability characterization is clean and geometrically intuitive; while the estimator is both sample and computationally efficient, with time complexity polynomial in dimension $d$ and inverse desired accuracy $1/\epsilon$, and the bit/facet complexity of the descriptions of the observed partition cells (i.e., coarse samples).

The main text is well-written overall and easy to follow, and Section 4 provides a helpful overview of the intuition and core ideas behind the technical proofs. 

The appendices appear thorough and the proof strategies combine tools from several areas creatively. For example, the authors use variance reduction and Prekopa-Leindler inequality for the identifiability characterization; as well as local partition reduction trick in bounding the gradient moments for the algorithmic result.

### Weaknesses
The estimator's sample complexity scales as $\widetilde{O}\left( \frac{d}{ \alpha^4 \epsilon^2} + \frac{dD^2}{\alpha^4} \right)$, where $D$ is known bound on $\|| \mu \||$. This is strictly worse than Fotakis et al.’s sample complexity of $\widetilde{O}\left(\frac{d }{\alpha^2 \epsilon^2}\right)$, which is independent of $D$. Also, I think claiming the sample complexity in the abstract as $\widetilde{O}\left(\frac{d }{\epsilon^2}\right)$ seems a little misleading, since this holds only in the constant regimes of $\alpha$ and $D$.



Minor: While each lemma and theorem is clear when read in isolation, I find the overall logical flow in the appendices (particularly from Appendix C onward) is quite dense and technically heavy. This is understandable given the paper’s theoretical depth and the secondary nature of these sections compared to Appendices A and B. But maybe adding a few sentences of contextual reminders or even a brief proof roadmap noting how the later technical lemmas connect back to the main results would help improve readability and navigation


Very minor (and less relevant since this is a theory-focused paper): The algorithm’s polynomial-time guarantee depends on access to an efficient sampling oracle. While the authors note in Appendix D that this can be implemented using MCMC schemes such as Hit-and-Run given a separation oracle, the practical runtime of these samplers is a concern. While this yields polynomial-time complexity, the high-degree polynomial dependencies (e.g., $d^{4.5}$) can be prohibitive for large or even moderate dimensions, limiting real-world applicability.

### Questions
If I understand correctly, the $\alpha^{-4}$ dependence in the sample complexity appears to stem from two $\alpha^{-2}$ factors: one from finding an $O(\alpha^2 \epsilon^2)$-optimal point in function value, and another from the $\Omega(\alpha^2)$ local strong convexity parameter. Do you think this dependence is fundamental for computationally efficient algorithms, or could it potentially be improved to $\alpha^{-2}$ (matching the sample complexity of Fotakis et al.) through a sharper analysis or an alternative scheme? A discussion on whether this reflects a intrinsic statistical–computational trade-off or an artifact of the current proof technique could be helpful.


I believe there is a minor typo in Remark 3. after Theorem 3.1. The sentence should likely be:
'The "almost every" quantifier in Theorem 3.1 is important: some convex partitions P contain non-slabs but are non-identifiable because the *non-slabs* have zero total volume.'

### Soundness
3

### Presentation
3

### Contribution
3
