# A Stochastic Approach to the Subset Selection Problem via Mirror Descent

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
The subset selection problem is fundamental in machine learning and other fields of computer science.
We introduce a stochastic formulation for the minimum cost subset selection problem in a black box setting, in which only the subset metric value is available.
Subsequently, we can handle two-stage schemes, with an outer subset-selection component and an inner subset cost evaluation component. We propose formulating the subset selection problem in a stochastic manner by choosing subsets at random from a distribution whose parameters are learned. Two stochastic formulations are proposed.
The first explicitly restricts the subset's cardinality, and the second yields the desired cardinality in expectation.
The distribution is parameterized by a decision variable, which we optimize using Stochastic Mirror Descent.
Our choice of distributions yields constructive closed-form unbiased stochastic gradient formulas and convergence guarantees, including a rate with favorable dependency on the problem parameters.
Empirical evaluation of selecting a subset of layers in transfer learning complements our theoretical findings and demonstrates the potential benefits of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel stochastic optimization algorithm to the *subset selection" problem. The objective of this problem is to minimize a loss on the subsets of an ambient set, and this combinatorial problem is NP-hard.  This paper proposed two stochastic approximations where the optimization variable instead becomes a probabilistic distribution over the subsets.

Compare to previous works in stochastic subset selection problem, this paper only makes little assumption over the loss function, where it is only defined over the discrete subsets and thus not differentiable over some continuous space. 

To approach this setting, the authors defined a stochastic gradient, where the derivative is taken over the parameters determining the probability distributions over the subsets. They further show that this stochastic gradient is 1) unbiased, 2) bounded, and 3) relatively convex (an extended notion of strong convexity). This properties allows the author to apply a stochastic mirror descent algorithm and utilize existing analysis in the literature [1] to prove convergence of their algorithm.

Finally, the authors provided a practical application of their method for the transfer learning problem and provided experimental results.

[1] Zhang S, He N. On the convergence rate of stochastic mirror descent for nonsmooth nonconvex optimization. arXiv preprint arXiv:1806.04781. 2018.

### Strengths
Overall, this paper is well-written and offers a neat solution to a difficult problem. While there are not any mind-blowing mathematical techniques and the prior literature [1] does quite a bit of heavy lifting, the proposed method has sufficient contribution to warrant a publication.

The biggest strength of this paper is the writing. The authors does a good job at guiding the reader through the various steps of their method and their claims are backed-up by technical results. There are no "fluff" in the paper in the sense that all results contribute meaningfully to the overall picture.

I checked most of the proofs and they all seem to be correct up to constants or typos. I did not check Lemma 7.1 or 7.2, but I presume some variation of those claims must be true. The other proofs all look in good shape (in a first pass), and the techniques generally follow established literature so I don't expect there is any hidden "surprises".

My overall assessment is that paper is not ground-breaking but does the job. It solves a relevant problem with a well-constructed method. While the lack of mathematical novelty stops it from being a great paper, it should solidly pass the threshold for acceptance.

### Weaknesses
While the overall quality of the paper is good, there are a few places where the author's intentions could be better clarified,

- The formulation of the mirror descent step in line 4 of Algorithm 1 is non-standard and I suggest the author to add a few lines of math to show how one can arrive at this line from a more standard form of mirror descent.

- Statements of Theorem 5.1 and 5.2 are rather difficult to parse. Initially, it wasn't immediately clear to me if smaller optimality gap $\tau$ implies smaller $c^*$. The author should some discussion to guide the readers on how to better interpret the equation on on 291, and a plot could be nice here. Also, in the proof, line 802 suffers a similar issue and more detailed steps should be provided.

- Section 6 claimed the stochastic gradient is inspired by the REINFORCE RL algorithm. But I cannot see this connection at all. I think it would be better if the authors just directly offer the intuition through a few lines of math instead of wrapping the logic around a gray box.

- The experimental setup with transfer learning may be unfamiliar for many readers and a more gentle introduction to this problem is necessary. I personally had to Google this because the paragraph at the bottom of page 8 is on the shorter side. Also, I am still confused by the role of HPO. The authors should more explicitly state the objectives of these experiments and describe the terminologies for a broader audience.

- The numerical results in Table 1 does not show a significant advantage for the author's approach. But I don't view this as essential to the overall quality of the paper.

Minor errors/typos:
- in Theorem 7.3, should be $(u-l)^2$.
- line 281, should be relaxation in $(P_k^c)$.
- line 673: should be Lemma B.1.1
- line 837, repeated $\in$ symbol

### Questions
- Regarding the stochastic relaxation from $(P)$ to $(P_k)$ and $(P_B)$, do you know the optimality gap of this relaxation? If this has already been studied in the literature, it would nice to have a brief discussion over those results.
- Just to make sure, would Lemma B.1 imply $L^* = L^*_k = L^*_B$? Also, I don't think this lemma is necessary for the proof of Theorem 5.1/5.2?
- What is Appendix E for?

### Soundness
3

### Presentation
3

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
The authors address the subset selection problem, a fundamental task in machine learning and computer science, by introducing a stochastic formulation for selecting minimum-cost subsets in a black-box setting. Here, only the metric value for a subset is accessible, limiting the information available for the selection process. The authors propose a two-stage framework where subset selection is decoupled into an outer selection component and an inner cost evaluation component. The paper parameterizes the subset distribution using a decision variable, optimized via Stochastic Mirror Descent (SMD). The proposed distributions enable constructive, closed-form, unbiased stochastic gradient formulas with favorable convergence guarantees. The approach is empirically validated on a subset selection task involving layer selection in transfer learning, showcasing both theoretical and practical benefits.

### Strengths
1. This paper presents an approach to the subset selection problem by proposing a stochastic formulation optimized using Stochastic Mirror Descent (SMD). Traditional subset selection techniques often involve deterministic methods, such as greedy algorithms or combinatorial optimization, which lack the flexibility and exploration capabilities needed for high-dimensional or complex subset spaces.

2. They derive closed-form, unbiased stochastic gradients for the subset distribution parameters, which makes the method practical and computationally efficient, especially compared to approaches requiring approximate gradients.

3. Overall, the paper is well-structured and clear, with a logical progression from the problem formulation to the proposed approach, theoretical analysis, and empirical evaluation. 

4. The significance of this work is substantial, both theoretically and practically. The subset selection problem is fundamental in machine learning, impacting areas like feature selection, model compression, neural architecture search, and sensor placement. By proposing a stochastic approach, this paper introduces a new direction for subset selection that emphasizes flexibility, exploration, and computational efficiency, which is especially relevant in high-dimensional or complex problem spaces.

### Weaknesses
1. While the layer selection experiment in transfer learning provides an interesting and relevant use case, the empirical evaluation is limited to a single task. Subset selection has a wide range of applications across machine learning, including feature selection, model pruning, sensor placement, and combinatorial optimization. Evaluating the proposed method on a broader range of tasks (as stated above) would strengthen the paper and better demonstrate the method’s generalizability.

2. There are various existing subset selection methods, both deterministic (e.g., greedy algorithms, combinatorial optimization) and stochastic (e.g., Monte Carlo methods, stochastic optimization techniques). The paper lacks a comparison with these methods, which would provide readers with more context on the advantages and disadvantages of the proposed approach and help highlight its unique contributions.

3. The paper does not clearly address how the assumptions of smoothness and favorable structure in the subset metric or cost function impact the theoretical results’ robustness in settings with irregular cost functions. Extending the analysis to cover more realistic conditions or discussing the limitations of these assumptions would provide a clearer understanding of the method’s applicability in diverse practical scenarios.

4. The choice of specific distributions for sampling subsets is not well-justified in the paper. While the authors mention that the distribution parameters are optimized, they do not explain why these particular distributions were chosen or how they might perform compared to other candidate distributions. This choice is critical, as different distributions could have substantial effects on the subset sampling’s efficiency and coverage. A clearer explanation of the rationale behind the selected distributions or a comparison with alternative options would strengthen the argument and clarify the trade-offs involved in the design.

5. Despite the use of stochastic methods, the proposed approach may still involve significant computational costs, particularly when repeatedly sampling and evaluating subsets over many iterations. A more thorough discussion on the scalability, computational complexity, and potential approaches to mitigate these challenges would make the paper more practically relevant for large-scale applications.

### Questions
1. What are the specific properties of the chosen distributions that make them suitable for subset selection? Are these distributions known to have properties that benefit the subset selection problem? 

2. How does the approach handle non-smooth or noisy subset metrics? Would there be significant degradation in performance, or could the method be adapted to handle irregular cost functions?

3. Can you discuss the relative benefits of each stochastic formulation (fixed vs. expected cardinality)? Could the authors discuss specific scenarios where one formulation might be preferable over the other?

4. What potential approximations could make this approach more feasible for large-scale problems? Have the authors considered any approximation techniques, such as subsampling or early stopping, that could reduce the computational cost? Discussing potential modifications or approximations would provide insights into how the method could be adapted for resource-constrained environments.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the subset selection problem. A two stochastic formulations of the 
problem are proposed, as well as a computable approximations, and solved via mirror descent, and 
a $1/\sqrt{T}$ rate of convergence to a stationary point is proven. The methods are tested 
empirically in a transfer learning task.

### Strengths
- It is rather surprising to me that the approach is seemingly able to 
get around making assumptions on the underlying structure of the losses that 
that one would see in the combinatorial bandits literature --- ie, that 
the losses take the form $\langle\ell_t, w\rangle= \sum_i \ell_{ti}\mathbb{I}(w=1)$, whereas in this 
work it is possible to have losses defined in such a way that every possible subset maps to independent values. 

- I really like the idea of second approach, in which the condition that $\\|w\\|_1\le k$ is relaxed 
to $\\|w\\|_1\le k$ *on average*, in exchange for a more readily computable update

- The paper is generally well-written and well explained

### Weaknesses
- Lines 33/34: we make *no assumptions* regarding the loss function other than having lower and upper bounds
    - This is not true; the results rely on relative weak convexity of $f$ wrt $\mu$, and moreover any assumptions made about $\mu$ 
    will imply certain restrictions on the curvature of $f$ as well. For instance, the stationarity measure is related to 
    the more traditional one for $\mu$ with Lipschitz gradients, and hence the results are not necessarily 
    meaningful unless $f$ is RWC wrt some smooth $\mu$, which implicitly limits what $f$ can be
    - The approach critically relies on the approximations to the stochastic problems, and choosing a valid approximation constant 
    seems to require prior knowledge of $\max_C \ell(C)$ and $\min_C \ell(C)$. So we must not only assume that $\ell$ is bounded, but 
    that the bounds are *known* to the user. This is arguably a very strong assumption to make.

- The experiments feature an interesting *application* of the subset selection problem, but given that this paper is about a new approach to a rather fundamental problem, it would have been more illuminating to see results in more standard subset selection problems such as Vertex Cover and comparisons against existing methods. As it stands, the experiments currently in the paper do not shed any light on how well the new approach works for subset selection, because it's not clear to me how important subset selection really is for transfer learning.

- Theorem 7.3 seems unnecessary --- it appears to just be a straight-forward application of Azuma-Hoeffding. 

- There is a long history of using a smoothing of the loss function to relax convexity assumptions; perhaps there should be more discussion on this front in the related work

### Minor Details
- Line 199: This is a very atypical way to write the mirror descent update; is there
  a reason the update is not written in terms of the Bregman divergence?

- It looks like you might be using \Cref in some of your appendix names, you might want to consider using something more based on the standard \ref call, like "Proofs of Section~ \ ref{sec:gradientestimators}", in these places. This will still put an in-text reference with hyperlink, but now the index in the pdf table of contents will correctly say "Proofs of Section 7".

### Questions
- As mentioned above, the approach seems to work even when the loss is defined in such a way that 
every subset maps to a distinct and unrelated loss. This suggests that the problem setting should be significantly harder 
than e.g. the combinatorial bandits setting, where we just need to find all the "best" indices. In fact, this difficulty seems to be reflected directly in the upperbound: the numerator is a factor of $n^2$ larger than what is attainable in the combinatorial bandits setting ($\sqrt{n}$). Do you expect that this is the optimal dependence?

### Soundness
3

### Presentation
3

### Contribution
3
