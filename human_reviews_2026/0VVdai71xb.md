# Mechanistic Independence: A Principle for Identifiable Disentangled Representations

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 8

## Abstract
*Disentangled representations* seek to recover latent factors of variation underlying observed data, yet their *identifiability* is still not fully understood. We introduce a unified framework in which disentanglement is achieved through *mechanistic independence*, which characterizes latent factors by how they act on observed variables rather than by their latent distribution. This perspective is invariant to changes of the latent density, even when such changes induce statistical dependencies among factors. Within this framework, we propose several related independence criteria -- ranging from support-based and sparsity-based to higher-order conditions -- and show that each yields identifiability of latent subspaces, even under nonlinear, non-invertible mixing. We further establish a hierarchy among these criteria and provide a graph-theoretic characterization of latent factors as connected components. Together, these results clarify the conditions under which disentangled representations can be identified without relying on statistical assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes mechanistic independence, which enables identifiable disentangled representations without strong statistical assumptions.

### Strengths
1. As a theoretical paper, the presentation is rigorous.
2. The theorem proofs appear solid.
3. The authors seem familar with recent identifiability literature and compare their results with prior work across the types D, M, S, H.

### Weaknesses
1. The abstract and main text repeatedly state that the paper’s identifiability results allow a non-invertible g, and they give the practical example of the “responsibility problem” (lines 160–161) to justify relaxing the global invertibility assumption. From my reading, permitting non-invertibility appears to be one of the paper’s contributions/novelties. However, all of the identifiability results actually rely on local diffeomorphism, which is stronger than local invertibility. Therefore, I think it is necessary to justify that this relaxation is non-trivial. By “trivial,” I mean the situation where one advertises that the framework does not require Condition A, but actually assumes Condition A holds in all the relevant region in the space (i.e. only not hold on a set of measure 0). It would be nice if you can add a discussion clarifying the gap between global invertibility and local diffeomorphism. For example, could you analyze a practical case like the “responsibility problem” that fails to be globally invertible on the space of interest but still satisfies the local-diffeomorphism condition?

2. In the theoretical framwork of this paper, S is the souce factors space, Z is the representation/target factor space, X is the observational space. But before def 1 (line 120-123), you only introduce S and X. Notation Z and term 'target factor' first appear in def 1 without any explanation. Maybe you can introduce Z and target factor before def 1 like what you have done with S and X. I think this may make the paper easier to read.



3. The theorem numbering in the appendix does not match the numbering in the main text.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper discusses a new set of principles for the identifiability problem of disentangled representations. Distinguished from existing works that usually assume statistical independence, this paper proposes mechanistic independence instead, to achieve identifiability even when the mixing function is non-invertible. Various related mechanistic independence conditions are proposed, most of which advance some branches of existing literatures.

### Strengths
1.	This paper proposes a novel perspective for latent variable identification, i.e., consider disentanglement directly as the goal instead of traditional slot/block-wise equivalence. This is intuitive since global invertibility is not required here.
2.	This paper gives a thorough discussion on the proposed mechanistic independence, providing several options for the identifiability principle. Most options have close relation to related works from different branches of identifiability research, and are unified nicely into one framework.

I am particularly interested in the novel results that global invertibility of mixing function is not required in this paper. Since almost all works on identifiability rely on this assumption, this would be an important advancement if its implication is properly discussed. See my question 1 for discussion.

### Weaknesses
1.	Almost no experiment is provided in this paper, which is my major concern for this paper. I understand this is mainly a theoretical paper. However, at least experiments on synthetic data should be provided (Fig. 1 is not enough), to validate each of your main theorems. Without experimental results, it is hard to understand the application scenario of different mechanistic independence principles, and reliability of the theorems are also weakened.
2.	Lack of explanation for the applicability of proposed assumptions in real applications. It is nice to have so much different independence principles as choices, but all principles lead to one common learning method (Fig. 1). I think it is infeasible to select one principle by validating each assumption in practice, and lacking customized methods significantly blurs the boundary among different principles.

I do not insist that identifiability works *must* include experimental results. But for this paper, I think it is helpful and necessary to demonstrate the superiority given that large part of results are upgrades of existing works.

### Questions
1.	This work does not rely on global invertibility of mixing function, how much is this condition be relaxed? I find that local diffeomorphism is still need. Together with path-connectivity, can global invertibility be derived? If not, please provide examples, and discuss how possible is it for such cases to happen in real applications.
2.	How to determine L (the number of blocks in Z) in practice? Setting $L=1$ clearly leads to trivial results.
3.	In Line 86, how to understand the symbol $D_{ij}^n$? Should “id” be 0 instead?

### Soundness
3

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
3

### Summary
This paper introduces a unified framework for achieving identifiable disentangled representations through "mechanistic independence", a principle that characterizes latent factors by how they act on observed variables (via the generator) rather than by their statistical distribution. The authors propose several related independence criteria (Type D, M, S, and H_n) and prove that each yields identifiability of latent subspaces under nonlinear, non-invertible mixing. They establish a hierarchy among these criteria and provide a graph-theoretic characterization of latent factors.

### Strengths
The novel theoretical perspective shifting from statistical to mechanistic independence is interesting and practically motivated (e.g. by compositionality and transfer learning). This paper is an interesting step connecting disentanglement with discovering the mechanistic structure of data-generating processes.

### Weaknesses
The main weakness of the paper is that the theoretical ideas are not especially intuitive. For example, the paper could better discuss when the various independence assumptions are likely to hold in practice. It might help to include more empirical results, not so much because experiments are required to validate the theory, but because experiments might give intuition about settings where the theory is actually relevant and which assumptions hold in which experiment settings.

As a more minor note, the main text would benefit from a short description of the experimental setup (without reading Appendix C, I didn't understand what Figure 1 shows at all).

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper proposes a framework for identifiability of disentangled representation learning based on mechanistic independence. The authors start from properties of the generator from latent variables to observations, rather than from the distribution of latent variables. A family of independence criteria is discussed: Type D (disjoint support via Hadamard orthogonality of Jacobian columns), Type M (mutual non-inclusion of supports in a fixed basis), Type S (a strict sparsity-gap condition on Jacobian representations), and Type H (vanishing cross *n*-th-order derivatives plus *n*-th-order separability). The paper proves identifiability from the local to the global level and provides a graph-theoretic view where factors correspond to connected components. But the assumptions are not tested in experiments or used to guide new methods. Overall, this paper offers a different view of identifiability of disentangled representation learning.

### Strengths
- The proof of identifiability is detailed and step-by-step, going further.The proof looks right and kind of makes sense.
- The whole paper is connected by mechanistic independence, which is an interesting view for identifiability.
- Graph-based characterization of factors as connected components aids intuition and potential diagnostics.
- The paper considers a more general generating process than previous works.

### Weaknesses
- Several key assumptions are strong, hard to verify, or basis-dependent: Type M depends on a fixed canonical basis; Type S requires the existence and uniqueness of a sparsest product-splitting basis.
- The assumptions are not tested in well-controlled experiments or real-world cases, and the theory doesn't contribute to methods. There is little evidence on real datasets or complex architectures.
- The limitations of mechanistic independence are not specifically discussed.

### Questions
- How can these independence criteria be deleted or used in cases such as a synthetic dataset?
- Can mechanistic independence be linked with current theories? Like in intervations: the second derivative for different components $i, j$ is zero like Type M somehow.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the challenge of identifiability in disentangled representation learning, particularly for nonlinear generative models, by introducing a unified framework based on mechanistic independence. This principle defines latent factors not by their statistical distribution but by how they act on the observation manifold through the generator function. 
The authors propose a hierarchy of mechanistic independence criteria (and corresponding irreducibility notions) ranging from
Type D (Disjointness), Type M (Mutual Non-inclusion), Type S (Sparsity Gap), to Type H_n (Higher-Order Separability).
For each criterion, the paper provides identifiability theory. Furthermore, this framework successfully generalizes and unifies several existing mechanistic constraints based or sparsity based identifiability results.

### Strengths
Clarity: The paper is very well-written.

Relevance: The paper is very relevant to the field and the contribution is significant. The framework represents a fundamental conceptual advance, providing a robust, distribution-agnostic foundation for identifying latent factors. 

Generality: The framework is broad, covering multi-dimensional factors, partial disentanglement, and non-invertible generators. The Theorem 1 result, extending local to global disentanglement under mild topological assumptions, looks like a strong tool.

Unification: By establishing identifiability for a range of mechanistic constraints under nonlinear, non-invertible mixing and statistically dependent latent factors, the paper generalizes and expands several very recent, disparate results.

### Weaknesses
The experimental setting looks limited (and not clear to me).  Plus identifying robust surrogate losses for the relaxed criteria, specifically Type M (mutual non-inclusion) and Type S (sparsity gap), remains an open problem. However, given the solid theoretical contribution, these weaknesses are not major.

### Questions
Practical Losses for Type M/S: Can the authors propose initial or speculative ideas for a robust surrogate loss that is more reliably minimized in practice?

Independence Hierarchy: Could the authors include in the figure what are the precise conditions under which one criterion is strictly weaker or stronger than another?

Experimental setting: It is not very clear to me that which types of independence are considered in the experiments? By following the setup of Brady et al. (2023) are you saying that Type D is considered? Have you considered any preliminary empirical validation for the other types like $H_n$?

### Soundness
4

### Presentation
3

### Contribution
4
