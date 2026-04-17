# Towards High-Order Mean Flow Generative Models: Feasibility, Expressivity, and Provably Efficient Criteria

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Generative modelling has seen significant advances through simulation-free paradigms such as Flow Matching, and in particular, the MeanFlow framework, which replaces instantaneous velocity fields with average velocities to enable efficient single-step sampling. In this work, we introduce a theoretical study on Second-Order MeanFlow, a novel extension that incorporates average acceleration fields into the MeanFlow objective. We first establish the feasibility of our approach by proving that the average acceleration satisfies a generalized consistency condition analogous to first-order MeanFlow, thereby supporting stable, one-step sampling and tractable loss functions. We then characterize its expressivity via circuit complexity analysis, showing that under mild assumptions, the Second-Order MeanFlow sampling process can be implemented by uniform threshold circuits within the $\mathsf{TC}^0$ class. Finally, we derive provably efficient criteria for scalable implementation by leveraging fast approximate attention computations: we prove that attention operations within the Second-Order MeanFlow architecture can be approximated to within $1/\mathrm{poly}(n)$ error in time $n^{2+o(1)}$. Together, these results lay the theoretical foundation for high-order flow matching models that combine rich dynamics with practical sampling efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces Second-Order MeanFlow, a theoretical extension of the MeanFlow framework that incorporates average acceleration fields alongside average velocities for generative modeling. The authors provide a rigorous mathematical treatment demonstrating that the proposed formulation satisfies a generalized consistency condition enabling stable one-step sampling, possesses expressivity within the TC⁰ circuit complexity class, and admits an efficient implementation using approximate attention with provable error bounds and nearly quadratic time complexity. By grounding these results in detailed proofs of feasibility, expressivity, and computational efficiency, the work establishes a strong theoretical foundation for exploring higher-order dynamics in simulation-free generative models and points toward potential future applications in fast, expressive generative modeling.

### Strengths
1. The paper provides formal definitions, clear assumptions, and detailed proofs for all major results. Each theorem (on feasibility, expressivity, and efficiency) is logically consistent and well supported by prior work in flow matching and circuit complexity.

2. Extending MeanFlow to second-order dynamics is a meaningful and nontrivial contribution. By modeling average accelerations, SOMF potentially captures richer temporal dependencies in generative flows, offering a principled direction toward high-order simulation-free models.

3. The circuit complexity analysis situates SOMF within the TC⁰ class, establishing theoretical efficiency guarantees and connecting generative modeling to computational complexity.

### Weaknesses
1. The paper remains entirely theoretical, and while the authors acknowledge this, the absence of even small-scale experiments or numerical simulations limits the understanding of how SOMF behaves in practice. Demonstrating a toy example or a simple empirical comparison would significantly enhance credibility.

2. The motivation for introducing acceleration terms is described formally but not intuitively. The paper could benefit from geometric or dynamical explanations illustrating how incorporating second-order information improves expressivity or sampling quality.

### Questions
1. While the theoretical results are strong, the real-world benefits of Second-Order MeanFlow remain unclear. How might this framework impact model expressivity and model performance in practice?

2. Although the proofs are rigorous, the paper’s density might limit accessibility to a broader audience. The authors could consider including intuitive figures (e.g., geometric interpretation of first- vs second-order flows) or summary tables that condense key theorems and assumptions, helping readers grasp the main ideas without diving into all formal details.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper “Towards High-Order Mean Flow Generative Models: Feasibility, Expressivity, and Provably Efficient Criteria” presents a theoretical analysis of Second-Order MeanFlow, an extension of the MeanFlow framework for generative modeling. Whereas MeanFlow replaces instantaneous velocity fields with average velocities to allow efficient single-step sampling, Second-Order MeanFlow further introduces average acceleration fields to model higher-order dynamics. The authors show that this formulation satisfies a generalized consistency condition enabling stable one-step inference, prove that the model’s expressivity lies within the TC⁰ circuit complexity class, and demonstrate provable efficiency through approximate attention computations with O(n^{2+o(1)}) time complexity and 1/poly(n) approximation error. The study provides theoretical support for the feasibility and tractability of high-order extensions to MeanFlow models.

### Strengths
- The paper is original in that it bridges the fields of generative modeling and circuit complexity. While several works on transformers leverage circuit complexity techniques, these haven’t been applied in the context of generative modeling.

- In particular, the finding that MeanFlows and high order MeanFlows are in TC0, which is the same class to which transformers without CoT belong, is very interesting, and may prompt researchers to think of CoT-like approaches for flows.

### Weaknesses
- The main weakness of the paper is that it does not include experiments testing the performance of the second-order MeanFlow. This may not be perceived as a weakness in a theoretical computer science venue, but ICLR is a general machine learning conference and the empirical performance of proposed algorithm must be shown. In that sense, this work is incomplete, as we do not know whether second-order MeanFlow is only a theoretical construction, or whether it has applications in practice.

- Small comment: The meaning of ViT (Vision Transformers) needs to be clarified in the first appearance of the expression.

- Section B in the Appendix is illuminating and explains the purpose of the paper, which is not clear from the main text: the reason for higher order MeanFlow and for proving circuit complexity are only briefly discussed in the introduction, but not in detail. The explanation in Section B should appear in the main text, and arguably in the first part of the paper. In particular the practical implications are too important to the story of the paper to be left in the appendix.

- The writing style is very dry. The main text is essentially a list of definitions and results. The authors should present a subset of their results in the main text, and defer the other ones to the appendix, and move the content of Section B to the main text.

### Questions
- Is there a reason the authors decided to prove circuit complexity results on MeanFlow and higher order MeanFlow instead of on standard flow matching or diffusion models? What I mean is, it looks like this paper has two topics: high order MeanFlow, and circuit complexity techniques for generative modeling. Is there a reason this is one paper instead of two?

- Would the authors be able to include experimental results on higher order MeanFlow? I would be willing to update my score if they provide compelling experiments.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper extends the MeanFlow framework from first-order (average velocity) to second-order (average acceleration). It demonstrates that the new average-acceleration field satisfies a consistency condition, enabling single-step or few-step sampling. It shows that the resulting samplers, when implemented with a constant-depth ViT and a fixed number of Euler steps, lie in the circuit class TC0. It demonstrates an almost-quadratic-time implementation by replacing exact attention with a fast approximate version, while bounding the induced error.

### Strengths
1. Clear formalization. The manuscript is meticulous in its definitions, maintaining consistency with prior Flow-Matching literature.
2. The assumptions are explicitly stated.

### Weaknesses
1. The main technical ideas are not very novel. Most of the argument builds upon earlier work on the circuit complexity of Transformer or VAR models, such as [1] and [2], and the extension to MeanFlow seems relatively straightforward.

2. There are no empirical or simulated results to illustrate whether the proposed approximations actually improve runtime.

3. The paper relies on many assumptions which may not hold in reality or have little practical guidance. The TC^0 result relies on (i) O(1) Euler steps, (ii) O(1) ViT layers, and (iii) poly-bounded weight magnitudes. In practice, state-of-the-art generative ViTs employ dozens of layers, adaptive samplers, and numerous engineering tricks, so the theorem sheds limited light on real-world deployments. It becomes more questionable when the paper fails to provide any empirical evidence.

4. The scope of the analysis is rather narrow, focusing on the MeanFlow backbone.

5. The connection between the circuit-theoretic results and practical model design remains abstract. It would be better if the authors discuss how the results of the paper could lead to practical guidance on architectural design and training.

References
[1] Chen, Bo, et al. "Circuit Complexity Bounds for RoPE-based Transformer Architecture." arXiv preprint arXiv:2411.07602 (2024).
[2] Li, Xiaoyu, et al. "Theoretical constraints on the expressive power of rope-based tensor attention transformers." CoRR (2024).

### Questions
1. I feel that the current analysis is quite ad-hoc. What is the most general architecture that the same analysis could extend to? 
2. I am curious if the authors could design some toy experiments for empirical evidence.

### Soundness
2

### Presentation
1

### Contribution
2
