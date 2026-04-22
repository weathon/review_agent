# How to Square Tensor Networks and Circuits Without Squaring Them

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Squared tensor networks (TNs) and their extension as computational graphs---squared circuits---have been used as expressive distribution estimators, yet supporting closed-form marginalization. However, the squaring operation introduces additional complexity when computing the partition function or marginalizing variables, which hinders their applicability in ML. To solve this issue, canonical forms of TNs are parameterized via unitary matrices to simplify the computation of marginals. However, these canonical forms do not apply to circuits, as they can represent factorizations that do not directly map to a known TN. Inspired by the ideas of orthogonality in canonical forms and determinism in circuits enabling tractable maximization, we show how to parameterize squared circuits to overcome their marginalization overhead. Our parameterizations unlock efficient marginalization even in factorizations different from TNs, but encoded as circuits, whose structure would otherwise make marginalization computationally hard. Finally, our experiments on distribution estimation show how our proposed conditions in squared circuits come with no expressiveness loss, while enabling more efficient learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel framework for efficiently performing marginalization in squared tensor networks (TNs) and squared circuits, which are models for distribution estimation. While squared TNs and circuits offer strong expressive power and closed-form marginalization, the squaring operation typically increases computational complexity, making inference expensive. The authors address this challenge by proposing a new parameterization strategy based on unitary matrices and deterministic structure, inspired by canonical forms in tensor networks and the tractability of maximization in deterministic circuits. Unlike traditional canonical forms, which are limited to tensor network structures, this approach extends to general circuit representations, including those that do not correspond to standard TN factorizations.

### Strengths
- The paper makes an original contribution by bridging two distinct paradigms in probabilistic modeling—TNs and circuit-based representations—through a unified, structure-aware parameterization strategy. While prior work has explored canonical forms for TNs, this paper extends such ideas to arbitrary circuits, including those that do not correspond to any known TN factorization.
- The key insight, parameterizing squared circuits using unitary matrices and determinism to enable closed-form marginalization, is both elegant and novel. It draws inspiration from canonical forms in tensor networks and the tractability of maximization in deterministic circuits.
- The paper demonstrates strong theoretical rigor. The main result, a sufficient condition under which squared circuits admit closed-form marginalization via unitary parameterization, is formally stated and proved.
- The experimental evaluation is well-designed and methodologically sound. The authors test on standard distribution estimation benchmarks. The proposed parameterized circuits achieve competitive or superior log-likelihoods compared to unconstrained models.
- Marginalization is faster, and training converges more rapidly due to better conditioning of the optimization landscape.

### Weaknesses
- The paper claims that its method enables efficient marginalization in non-TN factorizations, but the empirical comparison is limited to a few baselines. There is no direct comparison with other tractable circuit families.
- The paper proves that unitary parameterization leads to closed-form marginals, but the practical utility depends on whether such parameterizations can be optimized effectively. Unitary matrices are highly constrained, and optimization over this manifold is known to be challenging.
- The paper does not discuss gradient flow on the Stiefel manifold or the risk of vanishing gradients in deep unitary circuits.

### Questions
Please refer to the section on weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents novel results for squared Probabilistic Circuits that can represent expressive distributions. In particular, inspired by the applications of determinism in tractable probabilistic models, the proposed work introduces a new concept called orthogonality as a relaxed version of determinism in squared PCs. For a PC that is orthogonal inference they show that inference is tractable (partition function is computed in linear time).  They develop an approach to build orthogonal circuits. However, since such a circuit requires decomposability it is quite restrictive. Instead, they define conditions to construct circuits (called unitary circuits) over multiple layers which does not require decomposability. They develop a marginal inference approach for unitary circuits and prove the inference complexity for such circuits. The experiments show the efficiency of unitary circuits and their expressive power.

### Strengths
Strengths
+ Provides several novel previously unknown results in the context of probabilistic circuits, i.e. a new class of circuits that is expressive and tractable
+ The connection between determinism to orthogonality seems to be unique and perhaps will offer other research directions
+ Guarantees on inference complexity with unitary circuits being in normalized form
+ Paper is high on rigor with proofs for all the key results

### Weaknesses
- While the paper makes a strong contribution in probabilistic circuits with the introduction of unitary circuit learning and inference, it does not show why unitary circuits are better than existing tractable probabilistic models. The baseline comparison is with variants of squared PCs but perhaps the benefits of squared PCs over other approaches is not as clear. Maybe this is an empirical aspect that seems missing in the paper. 
- The choice of experiments and benchmarks was not so clear (MNIST and FashionMNIST). More generally, from the comments in lines 459-463 it seems like this approach is hard to scale for other general problems? This may be a limiting factor for broader use.

In general, the paper is strong in theory and presents novel theoretical results in the context of squared PCs. The empirical evaluation seems a bit weaker.

### Questions
Are squared PCs generalizable across different types of problems? Is the limiting factor learning them at larger scales?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work builds on previous work that investigates the relation between Tensor Networks and Squared Probabilistic Circuits, two computational frameworks developed in different communities. The Matrix-Product State representation of Tensor networks have been previously shown to be equivalent to structured and decomposable circuits with negative weights. Both Tensor networks and Circuits with negative weights can be constrained in order to satisfy probability distributions by squaring and renormalization. As previously noted, Squared Probabilistic Circuits are more general than MPS Tensor Networks, which makes the connection interesting. In this work, the authors investigate the orthogonality property often used by MPS to decrease the overhead of renormalization and squaring from quadratic to linear. That property is lacking from the Probabilistic Circuits literature. Since a direct adaptation of orthogonality leads to squared Probabilistic Circuits with a relatively simple structure, the authors propose a relaxation of the property, called Z-orthogonality (inspired by the concept of X-determinism in PCs). They show how the property can be exploited to speed up marginal computations while increasing flexibility of the model. Experiments in simple tasks (MNNIST and FASHION MNIST) show that their approach can effectively learn good representations and improve scalability.

### Strengths
- Interesting discussion, connecting topics studied in different communities
- Well-written
- Promising approach to learning high-dimensional probability distributions with tractable marginalization

### Weaknesses
- Empirical analysis is very preliminary
- Very long text (including appendices), difficult to revise given time constraints of a conference

### Questions
All in all, I think this is a solid work, although I didn't have the time to check all the material in the appendices. 

The analysis of RQ2 states that "squared unitary PCs gracefully scale, matching the performance of their baseline counterparts." I could not understand from the plots in Figure 4 how one can reach that conclusion.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes new structural and parameterization conditions for squared tensor networks and squared probabilistic circuits that enable linear-time marginalization without losing expressiveness. Prior squared circuits incur quadratic complexity for computing marginals or partition functions, limiting scalability. The authors introduce orthogonality and unitarity constraints on circuits—generalizing canonical forms in tensor networks—to achieve efficient normalization and marginalization even for architectures that do not map to classical tensor-network forms. Experiments demonstrate that these unitary squared circuits are faster and more memory-efficient, while matching or exceeding the performance of unconstrained models.

### Strengths
1. Very well-written paper

2. Novel connection between tensor networks and probabilistic circuits. The paper conceptually unifies canonical forms in tensor networks with determinism in probabilistic circuits and introduces orthogonality as a more general tool for tractable inference .

3. Theoretical contributions with clear practical implications. Orthogonality and unitarity are shown to guarantee O(|c|) marginalization in squared circuits instead of O(|c|²), and results extend to non-structured-decomposable circuits.

### Weaknesses
1. Experiments cover MNIST-style tabular/image datasets; evaluation on more complex tasks (e.g., high-dimensional continuous density estimation, conditional queries, or sampling quality metrics) would strengthen the real-world significance.

2. The core intuition behind orthogonality vs determinism and its practical implications could be communicated more clearly for a broader audience.

3. Orthogonality/unitary constraints often complicate optimization; although addressed here, more ablation on optimizer sensitivity would be valuable.

### Questions
Please see the weakness above.

### Soundness
3

### Presentation
4

### Contribution
3
