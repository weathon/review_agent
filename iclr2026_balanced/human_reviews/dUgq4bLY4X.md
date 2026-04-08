## Human Reviewer 1

### Summary
The paper studies symmetry increase in equivariant neural networks. When symmetric inputs are passed through a G-equivariant map, they can acquire more symmetry in their outputs than what the input possesses, which should degrade expressivity. There has been some work on the issue but a unified formulation and treatment has been lacking thus far. The paper proposes (a) a symmetry infimum determined by the feature space, (b) a computable procedure via orbit types to predict that infimum, and (c) under manifold-type regularity and sufficient approximation capacity, it is shown that most equivariant maps achieve this infimum (i.e. can take care of unwanted increases).

The paper motivates the problem with degradation with k-fold symmetric structures: rotated copies can collapse in feature space despite not being equivalent physically. Several empirical works (Joshi et al. 2023) and some theory works (Cen et al. 2024) are cited, and then the problem is framed via orbit types and Curie's principle (which in this case would amount to saying that equivariant maps can not decrease symmetry). To summarize the main ideas and their development. If the input orbit type $G_X$ does not occur in $Y$ (not in $O_{G(Y)}$, then the equality $G_X = G_{f(x)}$ is impossible. This means that strict increase (degeneration) is forced. These are illustrated by three flavours (full, axial, half), qualitatively shown in figure 2. Then a partial order on orbit types is defined by subgroup inclusion (a standard move), then in a fixed point space $X^H$ the minimal orbit type exists is unique (this is stated in theorem 3.1). The symmetry infimum $I_{G(X,H)}$ is that minimal orbit type. The symmetry increase is said to be "unexpected" if $G_{f(x)}$ exceeds the infimum. Isovariance is $G_X = G_{f(x)}$, and the necessary condition for it is $O_{G(X)} \subseteq O_{G(Y)}$. Equivalently, $I_{G(Y,H) = (H)} $ for all $(H)$ in $O_{G(X)}$. If the kernel rho_Y is not equal to the identity, every isotropy in $Y$ contains the kernel, so some increase is forced. Then a projection operator $p_{Y(H)}$ (closure under the kernel) is introduced and the relative isovariance defined. The necessary condition becomes $I_{G(Y,H)} = (p_Y(H))$. Using decomposition of representations (and for high multiplicity using Michel's criterion), a union or min of infima type behaviour is obtained for direct sums (corollary 4.3). Later the details are provided for k-fold geometry (D_kh symmetry), the minimal supergroups among candidates identifies and summarized in table 1. The analysis then discusses three degenerations that matches the visualization figures. Other results at the end work out the " most equivariant maps achieve this infimum" bit.

### Strengths
I think the core contribution is original and provides a crisp, representation-aware lower bound that predicts degeneration modes and unifies disparate observations (collapse to zero is seen as a special full case). 

Main theorems look good and provide useful machinery and results, which formalize quite a few intuitions that had been floating around in the literature. 

The paper is also generally clear, and fairly easy to follow. The k-fold example and three degeneration regimes are intuitive.

The main message of the paper can also lead to actionable design guidance: choose feature spaces whose orbit-type sets include the required (p_Y(H)); increase multiplicity to guarantee realizability; expect almost-isovariance generically.

### Weaknesses
There is a heavy focus on SO(3)/O(3) and dihedral cases, it would be good to add a discussion for other groups (like SE(3), E(2)), products groups. 

The synthetic experiments are fine and align with the theory, but see some suggestions in 'questions.' Despite the theoretical nature of the paper, I think it would be strengthened if the authors include some more experimental setups.

### Questions
There seem to be a lot of typos in the paper that should be correct. For example, there are several just in the abstract ("enpowerred", "scienctific", "indepth"). 

Can the authors add pseudo-code for the orbit-type test and infimum selection in the main text, and a small worked example? This should make the theory easy to follow. 

It would be good to have ablations for r, for different architecture families and depth. It would also be good to include one or two real, even if simple, domain-based sanity checks (perhaps from molecular or crystal contexts).

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper presents a rigorous theoretical framework for analyzing symmetry increase in equivariant neural networks, when outputs become more symmetric than inputs, reducing expressivity. It introduces the notion of a symmetry infimum, a computable lower bound determined by the feature space, and provides an algorithm for computing it. Experiments on synthetic symmetric structures validate the theoretical predictions. However, the work remains highly abstract and focuses mainly on characterizing rather than reducing symmetry increase, offering limited practical guidance for model design.

### Strengths
The paper addresses an underexplored but conceptually important issue in equivariant learning, the tendency of models to exhibit unwanted symmetry increase. It introduces an original and rigorous theoretical framework based on the notion of a symmetry infimum, supported by solid mathematical analysis and consistent experimental validation. Although abstract, it represents a meaningful theoretical advance in understanding the limitations of equivariant neural networks.

### Weaknesses
While the paper offers a rigorous theoretical framework for characterizing symmetry increase, it lacks a clear and consolidated discussion of how to mitigate it in practice. The ideas for reduction, such as adjusting representation degree or multiplicity, are scattered across sections and never formulated as concrete design guidelines, making the work hard to apply in real settings. The paper also feels rather abstract and narrowly focused; the topic is niche within equivariant learning, and the presentation does little to connect theory to practice. Moreover, the title’s promise of “reducing” symmetry increase is not fully realized, as the paper mainly analyzes rather than mitigates the phenomenon.

### Questions
The title and framing suggest that the paper provides methods for reducing symmetry increase, yet most of the work focuses on characterizing it. Could the authors clarify what concrete design principles or architectural adjustments actually help reduce symmetry increase in practice?

Could the authors provide more intuition or visual explanation of the symmetry infimum concept to make the theoretical framework more accessible to practitioners? What about simpler examples? (Consider a 2D shape such as an equilateral triangle. The triangle itself has 3-fold rotational symmetry, it looks the same every 120° rotation. Now, imagine an equivariant network that processes this triangle and produces a feature representation...)

Beyond synthetic examples, do the authors foresee domains (e.g., molecular modeling, materials science) where symmetry increase significantly impacts performance and where their theory could guide practical model improvements?

The phenomenon of symmetry increase seems loosely analogous to aliasing in the Nyquist–Shannon theorem, where insufficient sampling along the translation group causes distinct frequencies to become indistinguishable. In the present context, a restricted feature space (e.g., limited degrees of irreducible representations) similarly prevents the network from distinguishing between different orientations, leading to a kind of symmetry aliasing over the rotation group. Could the authors comment on whether this analogy is meaningful, and whether the symmetry infimum can be seen as a group-theoretic counterpart to the Nyquist limit for translational symmetry?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper investigates why equivariant models can unintentionally increase symmetry in their outputs, causing distinct inputs to become indistinguishable. It introduces a practical notion of a symmetry lower bound determined solely by the model feature space: if the feature space cannot realize the input symmetry, the output is guaranteed to gain extra symmetry. Building on this idea, the authors provide a computable test that predicts when symmetry increase will happen and how severe it will be. They also show that, under the manifold hypothesis, generic equivariant models can avoid unnecessary symmetry increase. Experiments on controlled setups closely match the predictions.

### Strengths
- The paper is well structured and uses clear notation.
- Although I did not check the proofs in depth, the results appear sound and internally consistent.
- The paper positions itself as a relevant contribution to the literature on symmetry-breaking inputs in equivariant machine learning. It formalizes the notion of a symmetry infimum, providing a simple, checkable target for assessing symmetry increase.
- The authors provide all the necessary machinery to derive this object.

### Weaknesses
- Despite the paper being well organized, the exposition is overly compact and hard to follow, as strong backgrounds in differential geometry and Lie groups are uncommon in the ML community. While the paper need not be self-contained, a more explicit notation guide in the appendix would help. In the eventual camera-ready, the extra page could be used to introduce key concepts in plain language with brief examples.
- The (almost)-isovariance results rely on the manifold hypothesis and smooth parameterizations. Non-smooth and distributional regimes, common in practice, are not addressed. Consider clarifying the scope of the claims and discussing whether any results extend beyond the smooth setting.

### Questions
1. Can you discuss the practical feasibility of using higher-dimensional irreducible representations in the output representation, or of increasing their multiplicities? Does this lead to computational or memory bottlenecks during training or inference?
2. Is it possible to adapt Theorem 5.2 to a tubular-neighborhood hypothesis, namely that the data distribution is supported in a tubular neighborhood of, rather than exactly in, a smooth low-dimensional submanifold?

### Soundness
4

### Presentation
2

### Contribution
4

### Rating
8

### Confidence
3