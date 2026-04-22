# Mini-batch Submodular Maximization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
We present the first mini-batch algorithm for maximizing a non-negative monotone decomposable submodular function, $F=\sum_{i=1}^N f^i$, under a set of constraints. Our experiments demonstrate that a straightforward uniform mini-batch sampling approach significantly outperforms existing state-of-the-art sparsifier methods, requiring only a fraction of their running time. However, explaining this improvement via worst-case analysis is impossible.

Instead, we employ smoothed analysis to provide a theoretical justification for our empirical findings. Under mild assumptions, we show uniform sampling is superior to weighted sampling for both the mini-batch and sparsifier approaches. We further verify empirically that these assumptions hold across various datasets. Unlike weighted sampling, uniform sampling is simple to implement and several orders of magnitude faster, making it ideal for handling massive real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the problem of submodular maximization with a monotone decomposable objective function, i.e., $f(S)=\sum_{i=1}^Nf^i(S)$. The authors propose an algorithm based on uniform mini-batch sampling, which empirically outperforms existing sparsifier-based methods. To explain this advantage, they conduct a smoothed analysis demonstrating that uniform sampling is theoretically superior to weighted sampling in both the mini-batch and sparsifier settings.

### Strengths
1. The paper addresses a well-motivated problem with broad practical relevance and numerous applications.
2. The theoretical analysis of the proposed submodular maximization algorithm is rigorous yet presented in a clear and accessible manner.

### Weaknesses
1. A major concern lies in the significance of the proposed approach. The large-scale optimization of decomposable submodular functions can be viewed as a special case of stochastic submodular maximization, or more generally, submodular maximization with i.i.d. bandit feedback, where uniform sampling corresponds to drawing samples from the stochastic distribution. From this perspective, the authors should compare their results against existing works in this broader setting (e.g., [1–3]). Moreover, the use of uniform sampling to approximate the objective function is typically considered a baseline method in the literature, which limits the novelty of this contribution. As a result, the overall advancement provided by the paper appears marginal.

2. The presentation of the smoothed analysis is unclear, making it difficult to follow. For instance, in the description of the smoothing model, the authors state that $f^i(e*)$  is a random variable, but the source of randomness is not explicitly explained. It remains ambiguous whether the randomness arises from the function $f^i$ itself or from the sampled element $e^*$?

[1] Singla, Adish, Sebastian Tschiatschek, and Andreas Krause. "Noisy submodular maximization via adaptive sampling with applications to crowdsourced image collection summarization."

[2] Wenjing Chen, Shuo Xing, and Victoria G Crawford. A threshold greedy algorithm for noisy submodular maximization. arXiv preprint arXiv:2312.00155, 2023.

[3] Karimi, M., Lucic, M., Hassani, H., & Krause, A. Stochastic submodular maximization: The case of coverage functions. Advances in Neural Information Processing Systems, 30.

### Questions
1. Could you provide a comparison between your algorithm and the one proposed in [1]? It appears that your method may be a straightforward adaptation of the algorithm in [1], obtained by replacing the TopK selection step with taking the same number of samples of the marginal gain for each element in the universe. 
2. Could you elaborate on the specific technical contributions of this work? The problem of submodular maximization under noise has been extensively studied in prior literature, where analyses typically focus on algorithms with additive or multiplicative approximation errors and employ concentration inequalities to estimate the objective value. It would be helpful if you could clarify how your analysis or techniques differ from these existing approaches.

[1] Singla, Adish, Sebastian Tschiatschek, and Andreas Krause. "Noisy submodular maximization via adaptive sampling with applications to crowdsourced image collection summarization."

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a highly efficient mini-batch greedy algorithm for maximizing large-scale decomposable submodular functions ($F=\sum_{i=1}^{N}f^{i}$), where evaluating the full function is prohibitively expensive. To theoretically justify the empirical superiority of a simple uniform sampling strategy over complex weighted methods, the authors move beyond worst-case analysis and employ a novel smoothed analysis framework. This framework allows them to prove that their uniform mini-batch algorithm achieves near-optimal approximation guarantees for both cardinality and p-system constraints, with improved query complexity. Specifically, for a cardinality constraint, the algorithm achieves a $(1-1/e-\epsilon)$-approximation using only $\tilde{O}(\frac{k^{2}n}{\epsilon^{2}\phi})$ oracle queries. Extensive experiments on five real-world datasets substantiate these findings, demonstrating that the proposed method is orders of magnitude faster than the state-of-the-art while providing superior or comparable solution quality. The paper also empirically validates that the core assumptions of its theoretical model hold in these practical settings.

### Strengths
The paper is well-written and easy to follow. A major strength lies in its comprehensiveness, as the authors rigorously validate their approach across three distinct settings: cardinality constraints, p-systems, and curvature. Moreover, the emphasis on mini-batching represents a timely and meaningful contribution to the machine learning community, addressing scalability in submodular optimization.

### Weaknesses
The primary limitation stems from an overly strong assumption. Specifically, the authors assume that for any individual element $e$, each function $f_i$ satisfies both $f_i(e) \in [0,1]$ and an expected value $E[f_i(e)] \ge \phi$. These strict upper and lower bounds on single-element contributions overly simplify the problem, making it plausible that a simple uniform sampling strategy could already achieve a comparable approximation of the overall objective function $F$ using standard concentration results such as the Chernoff bound.

### Questions
1.	If we remove the normalization condition while keeping all other conditions unchanged, does the conclusion still hold?
2.	Alternatively, can you prove that without these assumptions, it is impossible to optimize using fewer than ( Nnk ) queries in terms of order? 
3.	It seems that the paper still contains some minor errors — for example, in proof of lemma 3.4, the Chernoff bound expressions are missing the $\exp()$ wrapper.
4.	The authors did not mention that their assumptions may oversimplify the theoretical analysis of the algorithm.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A decomposable submodual rmaximization, the sub-modular function $F$ is the sum of $N$ other sub-modular functions. The algorithms that make oracle calls to $F$ need to make $N$ oracle calls to the smaller/simpler submodular functions, thus the run-time has undesirable dependency on $N$.  To address this, prior work proposed sparsifier approach, that choses a subset of the smaller functions (and scales them) with certain probabilities. However the calucation of these probabilities takes $O(Nn)$ oracles, and this is treated as preprocessing step, in prior works.

The authors observe that uniform sampling probabilities work very well in practice and attempt to come with a theoretical explanation (though in the worst-case, uniform sampling is bad).  They resort to smoothed analysis, pioneered by Spielman and Tang, to address this.

### Strengths
1. As far as I know, this is the first instance of applying smoothed analysis for this problem.
2. It is known that many heuristic algorithms work well in practice, though their approximation guarantees are bad in the worst case. Thus, there is a need to go beyond the worst-case analysis, and this work is a step in that direction.
3. Generally, when the exact oracles are replaced with approximate oracles that have a multiplicative error, the proofs go through, with multiplicative errors creeping into the approximation factors. This work relies on using additive-approximate oracles, where the additive approximation errors are based on the optimal values. They show that this type of approximation still yields good guarantees, which in my view is clever.
4. Once the framework is set, the proofs are not hard to follow, which is a strength in my view.

### Weaknesses
My biggest concern is the definition of the smoothing model. Where is the randomness coming from? The functions $f^i$ are all deterministic. So what does it mean to say $f^(e^*)$ is a random variable and so what does the Expectation of this reandom variable mean? Where is the underlying distribution? If I were to infer, my guess is that each $f^i$ is chosen with a certain probability, and that's where the randomness is coming from, but I am not sure. This must be addressed.

I am willing to increase my score once I understand this.

### Questions
1. Please see the weakness.
2. Definition of Smoothing model: Can you formally define "each RV depends on at most d others". 
3. Similarly, what does bounded dependency mean in Theorem 3.2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies a uniform sampling algorithm for maximizing a nonnegative monotone *decomposable* subomdular function
$F = \sum_{i=1}^N f^i$, where each $f^i$ is a nonnegative monotone submodular function.
Specifically, it proposes a uniform sampling method to sparsify $F$ when $N$ is large by
sampling a subset of function $f^i$ of size $N' << N$.
They offer a smoothed analysis inspired by [Spielman--Teng, JACM 2004] to go beyond worst-case inputs.
Much of this work is focused on removing a $O(N n)$-time bottleneck step in the
sparsification scheme of [Kenneth--Krauthgamer, ICALP 2024] that computes the
values $p_i = \max_{e \in E, F(e)\ne 0} f^i(e) / F(e)$.
Finally, the authors give an empirical study across a wide range of datasets.

### Strengths
- Builds on sparsification framework for decomposable submodular functions in [Rafiey--Yoshida, AAAI 2022].
- Identifies a hard/pathological case where only a single $f^i$ takes nonzero
  values, and proposes smoothed analysis to bridge this gap (beyond worst-case analysis)
- Diverse set of experiments (e.g., several different interesting datasets, tasks, and objective functions).

### Weaknesses
- The writing is not as crisp as it could be: there are many technical ideas discussed that distract from the main contribution (e.g., the "Approximate oracles" paragraph in Section 2.1 doesn't add to the message and distracts the reader). The paper could benefit from presenting fewer ideas and making things more streamlined, without compromising the main message.
- The theoretical contribution is novel but limited (Section 3)

### Questions
**Questions**
- [049] What are practical examples of $N$ being extremely large? What are
  reasonable assumptions for $f^i$ to all be different? If they are the same
  function (e.g., functions for two students with the same preference), then
  this is the same as increasing the coefficient of $f^i$ and reducing $N$.

**Typos/suggestions**
- [120] Suggestion: The paper could benefit from presenting a more compelling
  running example than lunch menu optimization.
- [528] Typo: Update the Kenneth--Krauthgamer reference to ICALP 2024.

### Soundness
3

### Presentation
2

### Contribution
2
