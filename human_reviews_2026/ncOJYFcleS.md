# Achieving Approximate Symmetry Is Exponentially Easier than Exact Symmetry

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 8

## Abstract
Enforcing exact symmetry in machine learning models often yields significant gains in scientific applications, serving as a powerful inductive bias. However, recent work suggests that relying on approximate symmetry can offer greater flexibility and robustness. Despite promising empirical evidence, there has been little theoretical understanding, and in particular, a direct comparison between exact and approximate symmetry is missing from the literature. In this paper, we initiate this study by asking:
What is the cost of enforcing exact versus approximate symmetry?
To address this question, we introduce averaging complexity, a framework
for quantifying the cost of enforcing symmetry via averaging. Our main result is an exponential separation: under standard conditions, achieving exact symmetry requires linear averaging complexity, whereas approximate symmetry can be attained with only logarithmic averaging complexity.
To the best of our knowledge, this provides the first theoretical separation of these two cases, formally justifying why approximate symmetry may be preferable in practice. Beyond this, our tools and techniques may be of independent interest for the broader study of symmetries in machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Summary

This paper investigates the theoretical trade-off between enforcing exact versus approximate symmetry in machine learning models. While exact symmetry is known to act as a strong inductive bias and often improves performance in scientific domains, recent empirical evidence suggests that approximate symmetry can yield greater flexibility and robustness. However, a formal understanding of the cost difference between these two paradigms has been missing.
To address this gap, the authors introduce “averaging complexity,” a new framework that quantifies the computational cost of imposing symmetry through averaging operations. Their main theoretical result establishes an exponential separation: under standard conditions, exact symmetry requires linear averaging complexity, while approximate symmetry can be achieved with only logarithmic complexity.
This finding provides the first formal justification for the practical advantages of approximate symmetry, showing that small relaxations of exact invariance can substantially reduce computational cost without sacrificing desirable structural properties. Beyond this specific result, the proposed analytical tools and complexity framework are likely to be of independent theoretical interest for the broader study of symmetries and inductive biases in machine learning.

### Strengths
The paper’s core strength is its clean, theory-driven comparison of exact vs. approximate symmetry, a question that is both timely and genuinely interesting given how often practitioners trade strict equivariance for flexibility. The authors introduce a model-agnostic notion of “averaging complexity” that crisply quantifies the cost of enforcing symmetry via action queries, and they prove a striking exponential separation: exact symmetry needs linear complexity in |G|, whereas approximate symmetry can be achieved with logarithmic complexity. This result not only formalizes widespread empirical intuition—that approximate symmetry is easier and often more robust in practice—but also provides actionable design insight for symmetry-aware learning (e.g., symmetry discovery or semi-supervised settings where small violations are acceptable). Technically, the framework is conceptually simple yet broadly applicable, with proofs that leverage representation theory in a transparent way and yield constructive bounds (e.g., the degree K controlling when exact symmetry becomes linearly expensive). The work thus bridges a real gap between practice and theory, offering a principled rationale for when and why approximate symmetry should be preferred, and supplying tools likely useful beyond the specific results (e.g., for analyzing other symmetry-enforcing pipelines such as group averaging, canonicalization, or data augmentation).

### Weaknesses
The paper tackles an interesting and timely question—contrasting exact vs. approximate symmetry—but its technical development feels overly elementary relative to the ambition of the claim. Most arguments rely on standard representation-theoretic decompositions, Fourier analysis on finite groups, and matrix concentration, yielding a clean but fairly direct O(log |G| / ε) upper bound and a linear lower bound via a Vandermonde-style character argument. As a result, the “exponential separation’’—while appealing—rests on a simplified averaging model (x-independent linear post-processing with action queries) and finite groups, leaving unclear how far the conclusions extend to practically important settings (compact Lie groups such as SO(3)/E(3), continuous domains, learned filters, or non pointwise mechanisms). The lower bound for exact symmetry hinges on tensor-power feature lifts and a bound on K via the number of distinct character values; this can be loose, and the paper does not establish matching lower bounds for approximate symmetry or tight constants (or necessity) in the O(\log |G|/\varepsilon) rate. Moreover, the framework abstracts away data- or state-dependent averaging (e.g., frame averaging with x-dependent weights), model-dependent implementations (equivariant layers, steerable kernels), and optimization constraints, so the gap between complexity-in-principle and trainable procedures remains wide. From a learning-theoretic perspective, the results are existential and asymptotic: there are no generalization/error-rate guarantees, no minimax or sample-complexity characterizations tied to the symmetry deficit, and no robustness analysis to noise or misspecification. In short, while the problem is compelling, the paper would benefit from strengthening the main theorems (e.g., tighter—ideally optimal—bounds and explicit constants, extensions beyond finite groups, lower bounds for approximate symmetry, or constructive/algorithmic realizations), thereby elevating the contribution beyond what can be achieved with relatively standard tools.

### Questions
1.	Lower Bounds and Tightness: The paper establishes an upper bound of  O(\log |G| / \varepsilon)  for approximate symmetry.
Can the authors characterize whether this rate is tight? In particular, are there matching lower bounds on averaging complexity that depend on the function class F?
2.	Extension to Continuous or Lie Groups: The current framework assumes finite groups. How would the results change for compact or continuous groups (e.g., SO(3), SE(3)) where integration replaces summation? Would the averaging complexity still exhibit a logarithmic versus linear separation under appropriate discretization or measure-theoretic assumptions?
3.	Generalization and Robustness: Since approximate symmetry is shown to be easier to enforce, can this framework be extended to analyze generalization error or robustness to distributional shifts?
For instance, can one derive bounds on sample complexity or Rademacher complexity as a function of the averaging complexity or \varepsilon?
4.	Practical Implications for Optimization: In practice, symmetry enforcement interacts with gradient-based optimization.
How does probabilistic averaging (with only a few sampled group elements) affect gradient variance, bias, and convergence stability during training?
5.	Function Space Dependence: The analysis is primarily framed for continuous functions in L^2(X). Could the authors extend their results to other function spaces—such as Sobolev, Barron, or RKHS spaces—and derive approximation rates depending on smoothness or polynomial degree?
6.	Alternative Norms and Metrics: The results rely on L^2-based definitions of approximate symmetry.
Would the same scaling laws hold under supremum norms or other distances relevant to robust learning or adversarial settings?
7.	Adaptive or Data-Dependent Averaging: If the averaging scheme were made adaptive or data-dependent (e.g., learned during training), could one further reduce the \log |G| scaling? Are there theoretical barriers preventing such improvements?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study group averaging procedures for achieving group equivariance. Given a finite group G, a group averaging scheme is a function $\omega: G\to \R$. The group averaging scheme induces a linear averaging operator $[\mathbb{E}_\omega(f)](x) = \sum_{g\in G}\omega(g) f(g^{-1}x)$. The 'trivial choice' of $\omega(g)=1/\abs{G}$ for all $g\in G$ yields an operator for which $\mathbb{E}_\omega f$ always is invariant to the group action. 

The authors define the size of $\omega$ as the number of non-zero function values $\omega(g)$. They then pose the question: How large must $\omega$ be in order to achieve perfect symmetrization for a given function class $\mathcal{F}$, and how does this compare to approximate symmetry? By approximate symmetrization, they thereby mean schemes that reduce the equivariance error by a factor $\varepsilon$, as measured in $L^2$-norm.

The main results are twofold: First, they show that as long as the function class $\mathcal{F}$ as a representation of the group contains every irrep of the group, the only scheme that achieves perfect symmetry is the trivial choice -- in particular, the size of $\omega$ necessarily is equal to $\vert{G}\vert$. As a contrast, one can use random constructions to construct $\omega$ of size $\log (\vert G \vert)/\varepsilon$

### Strengths
The results of the paper are (mostly, see below for details) correct. The connections between Fourier analysis for finite groups and sampling complexity for achieving invariance through averaging that the authors have discovered are interesting, and they do potentially open up for more research.

### Weaknesses
1. The authors make quite far-reaching claims about their results: They state that their results explain why data rarely is exactly equivariant, and why models exploit approximate symmetries more easily. One has to ask if this is really what they show? What they say is that the only way to achieve exact symmetry through averaging, which is by far not the only way of doing so, is to take an uniform average over the whole group, and that this will need many samples given the function class is, from a representation theory perspective, is complicated enough. The existence of steerable neural networks shows that enforcing symmetry is very possible, and implementing them is arguably not fundamentally harder than implementing any other neural network.

2. Assumption 9 is a lot stronger than the authors make it out to be. For one, it implies assumption 8: If there is a non-trivial group element $g$ for which $gx=x$ for all $x$, there surely cannot exist a function $f$ (in or not in $\mathcal{F}$) with $f(gx)\neq f(x)$. In fact, by essentially the same argument, it implies that each element $x$ in the domain has a non-trivial stabiliser $\{ h \, \vert \, hx=x\}$. This rules out e.g. the canonical action of the permutation group on $\mathbb{R}^d$. Since the authors ultimately only need that the action of the group on $\mathcal{F}$ is faithful, I am unsure why they make this assumption.

3. Definition 10 is as far as I am concerned not the standard definition of a tensor power of function spaces: The elements of the $\mathcal{F} \otimes \mathcal{F}$, for instance, are rather functions on the space $\mathcal{X}\times \mathcal{X}$, with $(f\otimes g)(x,y) = f(x)g(y)$. I think that this is not just a question of conventions -- note that in order for the representation theoretic machinery to work (in particular that the characters of representation on $\mathcal{F}^{\otimes k}$ become $k$:th powers of the characters of the representation on $\mathcal{F}$), the canonical inner product $\langle x \otimes y, z \otimes w\rangle = \langle x,z \rangle \langle y,w \rangle$ needs to be used. This is not what happens when we're forming polynomials from function -- the scalar product is still the $L^2(\mathcal{X})$-scalar product! 

4. There are some mistakes in the reference list. The title of [Bourgain, Gamburd] is missing an $\mathrm{SL}_2(\mathbb{F}_p)$,  [Huang, Levie,Villar] and [Tahmasebi,Jegelka;2023] do not have journals. Furthermore, I can't find the paper "A differentiable metric for discovering groups and unitary representations" by Dongsung Huh, ICLR 2025. Has the paper changed name at some point?

5. Some parts of the text in the appendix do not flow very well. There are e.g. parts of the texts that are repeated verbatim in several proofs.

### Questions
1. Can the authors provide some more arguments why $\mathrm{AC^{ex}}$, $\mathrm{AC^{wk}}$ and $\mathrm{AC^{st}}$ is a good way of measuring the hardness of enforcing symmetries? 

2. Is there a specific reason why the authors do not just assume that the action of $G$ on $\mathcal{F}$ is faithful directly?

3. Can the apparent problem that the definition of the tensor product doesn't seem to be correct be fixed?

4. Is the result that $\mathrm{K}$ can be chosen as $\vert \lbrace \ \mathrm{Tr}(\rho(g)) \ \vert \ g \in G \rbrace \vert$ known or new?

5. In the proof of Theorem 13, can the authors point to the concrete matrix concentration inequality they apply in row 1453?

### Soundness
3

### Presentation
2

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
This paper investigates the theoretical complexity of enforcing exact versus approximate symmetry in machine learning models. The authors introduce a novel framework called "averaging complexity" to quantify the cost of symmetry enforcement through action queries. Their main contribution is proving an exponential separation: exact symmetry requires linear averaging complexity in the group size, while approximate symmetry needs only logarithmic complexity. The paper uses representation theory tools to establish these bounds, providing the first theoretical justification for why approximate symmetry is often preferred in practice.

### Strengths
Originality and Theoretical Contribution

The paper addresses a significant gap in the literature by providing the first direct theoretical comparison between exact and approximate symmetry enforcement. The exponential separation result (linear versus logarithmic complexity) is novel and formally justifies intuitions from empirical work. The averaging complexity framework itself is a creative abstraction that could enable future theoretical analyses in geometric machine learning.

Mathematical Rigor and Technical Quality

The proofs leverage sophisticated representation theory, including character theory, Fourier analysis on groups, and tensor product decompositions. The constructive proof of Theorem 11 provides explicit bounds on the threshold K where exact symmetry requires full group averaging. The probabilistic construction in Theorem 13 using matrix concentration inequalities is elegant and yields tight logarithmic bounds.

Practical Relevance

The work connects to important practical considerations in geometric machine learning, including symmetry discovery, robustness to distributional shift, and semi-supervised learning. The theoretical insights help explain why approximate symmetry methods have been successful across various applications from medical imaging to molecular modeling.

### Weaknesses
Limited Scope to Finite Groups

The entire framework is restricted to finite groups, which significantly limits applicability. Many important symmetries in machine learning involve continuous groups like SO(3) for rotations or SE(3) for molecular systems. The authors acknowledge this limitation but provide no path forward. The technical machinery (character theory, Fourier analysis on finite groups) does not obviously extend to compact Lie groups.

While averaging complexity is mathematically elegant, its relationship to practical computational cost or sample complexity is unclear. Action queries are a theoretical construct, but real systems face different bottlenecks (gradient computation, memory, data requirements). The paper would benefit from discussing how averaging complexity relates to these practical concerns.

The paper is purely theoretical with no experiments whatsoever. Even simple synthetic experiments demonstrating the exponential separation on toy problems would strengthen the work. Without empirical validation, it is difficult to assess whether the constants hidden in the big-O notation matter in practice, or whether the assumptions (faithful action, orbit separability) hold for realistic function classes.

Assumptions 8 and 9 (faithful group action and orbit separability) are needed for Theorem 11 but not for Theorem 13. The paper does not thoroughly discuss when these assumptions hold in practice or provide counterexamples when they fail. For tensor powers reaching degree K, the dimension of the function class grows exponentially, which may be impractical for large groups.

The paper does not compare averaging complexity to other complexity measures in learning theory such as VC dimension, Rademacher complexity, or sample complexity. Understanding these relationships would better situate the contribution within the broader theoretical landscape.

### Questions
Can the authors provide any insight into whether similar results might hold for compact Lie groups? What are the fundamental technical barriers? Would one need to replace action queries with a different notion of complexity?

How should practitioners use these results? Does the logarithmic bound for approximate symmetry suggest specific algorithmic approaches? Can the probabilistic construction in Theorem 13 be turned into a practical sampling strategy?

The authors mention constants differ by at most a factor of four between weak and strong approximate symmetry. What are the actual constants? Are they small enough that the logarithmic bound is meaningful for realistic group sizes?

Are the bounds tight? Is there a matching lower bound showing that logarithmic complexity is necessary for approximate symmetry? Can the authors provide examples where the trivial K ≤ |G| bound is achieved versus cases where K is much smaller?

Can the authors provide concrete examples of important function classes or group actions where Assumptions 8 and 9 fail? What happens to the results in these cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents the first formal theory comparing the enforcement difficulty of exact and approximate symmetry in machine learning models. By introducing the notion of averaging complexity, the authors prove an exponential separation: exact symmetry requires linear complexity in the group size, while approximate symmetry needs only logarithmic complexity. Using representation theory and Fourier analysis, the work explains why approximate equivariance is both more flexible and computationally efficient, offering a principled foundation for its empirical success in geometric deep learning.

### Strengths
The paper’s main theorem establishes an exponential separation between exact and approximate symmetry, supported by rigorous analysis using representation theory, Fourier analysis, and probabilistic tools. The writing is clear and well-structured, effectively connecting the theoretical insights to practical implications for generalization and robustness in symmetry-based learning.

### Weaknesses
1. The paper lacks empirical or numerical evidence to support its theoretical claims. Including simple synthetic experiments, such as verifying the predicted scaling laws of averaging complexity in finite groups, would make the results more concrete and help readers connect the abstract theory to observed model behavior.
2. While the proofs are rigorous, the intuition behind key representation-theoretic arguments, such as the role of tensor powers and character separability in explaining the linear versus logarithmic complexity gap, could be better articulated. Providing illustrative examples of other groups, such as cyclic or dihedral groups, would further improve clarity and highlight the generality of the framework.

### Questions
1. The paper focuses on finite groups throughout the analysis. How challenging would it be to extend the notion of averaging complexity to continuous groups, such as $\mathrm{SO}(3)$ or $\mathrm{SE}(3)$? Do the authors expect a similar exponential separation between exact and approximate symmetry to hold in that case, or are there fundamental obstacles?

2. The exponential separation result relies on representation-theoretic properties and tensor-power constructions. Could the authors provide additional intuition about the role of the number of distinct character values $K = \\{\mathrm{Tr}(\rho(g)) : g \in G\\}$? In particular, how should we interpret this parameter in concrete machine learning models?

3. How does averaging complexity relate to empirical training complexity in neural networks that are approximately equivariant? For example, can this framework provide any guidance on the number of samples or architectural constraints required to achieve a desired level of approximate symmetry in practice?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper examines the question of the efficiency of enforcing exact equivariant compared to approximate invariance from a theoretical point of view. A framework is developed to precisely characterize the notions of efficiency and approximate invariance in this context. The authors show that, under general assumptions, enforcing exact invariance is exponentially harder than enforcing approximate invariance.

### Strengths
- The paper tackles important and timely questions in geometric deep learning, namely what are the benefits and cost of exact equivariance vs approximate equivariance.
- The results provided are nontrivial and sound as far as I could check. The formalism could probe useful to prove other results.
- The paper is well written and clear

### Weaknesses
- The assumption of the function class being a finite-dimensional vector space seems quite strong and potentially restrictive. Can the authors comment on that in the paper?
- Consider including experimental verification of the claim of logarithmic scaling of averaging complexity for approximate invariance (could be in the appendix)
- The definition of symmetry for a function coincidences with what is more commonly referred to as invariance, I suggest using that term instead for clarity
- It seems like the analysis is restricted to invariance and not equivariance. However, this is not clear from the start of the paper. Could the authors clarify that?

### Questions
- I feel like an additional step that could make the story more complete is a discussion of the potential generalization benefits of approximate invariance. I don't know if such results already exist, maybe a connection to Elesedy and Zaidi 2021 could be drawn.

### Soundness
3

### Presentation
4

### Contribution
3
