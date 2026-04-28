Now I have enough information. Let me write the final review.

## Summary
This paper proposes a novel identifiability theory for latent variables in nonlinear causal models using only single-domain data. The key contribution is the "minimality" condition, which prevents shared latent variables from absorbing private information. The paper shows that any SCM can be reduced to an equivalent Powerset Bipartite Graph (PBG) structure, and proves identifiability under three assumptions: invertibility, independence, and minimality. Experiments on synthetic data validate that satisfying all three conditions yields near-perfect identification.

## Strengths
- **Novel minimality condition for single-domain identifiability**: Unlike prior work requiring multi-domain data, auxiliary variables, or restrictive functional forms, this paper achieves identifiability with single-domain observations by introducing the minimality assumption (Assumptions 1.iii and 2.iii). Theorem 1 and Theorem 2 formally establish identification up to invertible transformation under these conditions.

- **SCM Reduction provides a canonical form for analysis**: The paper defines a constructive reduction procedure (Section 4) that transforms arbitrary SCMs into Powerset Bipartite Graphs, simplifying identifiability analysis. Figure 1 illustrates this transformation clearly, and the reduction is shown to preserve equivalence in terms of observed variable distributions.

- **Experimental validation demonstrates necessity of all assumptions**: Table 1 (Section 6.2) systematically shows that violating any assumption degrades performance: when minimality is satisfied (correct dimension $d_z=5$) along with invertibility and independence (AE+CLUB), $R^2$ reaches 97.8%, but drops to 82.7% or 66.0% when minimality is violated ($d_z=7$). This validates the theoretical claim that all three conditions are necessary.

- **Transparent about limitations**: Section 7 explicitly acknowledges that "the succeeded algorithms in our experiments still need pre-known knowledge of the intrinsic dimension of latent variables" and identifies this as requiring "future works" for practical algorithm design. This honesty about the theory's current scope is appropriate for a theoretical contribution.

## Weaknesses

### Fatal
None

### Major
- **Limited practical guidance for dimension estimation**: While the paper transparently acknowledges in Section 7 that current algorithms require "pre-known knowledge of the intrinsic dimension," the theory itself does not provide mechanisms to estimate this dimension from data. Corollary 5.1 shows that setting $\text{dim}(\tilde{\mathbf{z}}) = \text{IDim}(\mathbf{z})$ satisfies minimality, but in unsupervised settings the ground-truth intrinsic dimension is unknown. The experiments (Section 6.2) confirm this limitation: performance degrades significantly when $d_z$ is incorrect (7 vs 5). Without a method to estimate intrinsic dimension or verify minimality without ground truth, the theory describes *when* identification is possible given an oracle, rather than *how* to achieve it in practice. This limits the immediate applicability of the contribution, though the theoretical framework remains valuable.

- **The SCM Reduction assumes knowledge of observed descendant sets**: The reduction procedure (Section 4) clusters exogenous variables by their "observed descendant sets" $OD(u_i) = \{o_i \in O | u_i \in An(o_i)\}$. To perform this clustering in practice, one must know which latent variables affect which observed variables—i.e., the causal structure must be known. While Section 2 states "Our work do not aim to identify the structure of underlying SCM," this limitation significantly narrows the scope of problems the theory addresses. The abstract and introduction frame the contribution as addressing identifiability in "nonlinear causal models" generally, but the theory applies to settings where the causal graph (or at least the ancestor-descendant relationships) is already known. This is a substantially easier problem than discovering latent variables and their causal structure from scratch.

### Minor
- **Experiments only validate theory under correct specification**: The experiments in Section 6 verify that the theorems hold when assumptions are manually enforced, but do not explore realistic failure modes. There is no evaluation of: (1) what happens when dimension is estimated via standard methods (elbow, information criteria) rather than set to ground truth; (2) robustness to graph structure mismatch; or (3) comparison to non-identifiable baselines (e.g., standard VAEs) to quantify the gain from the PBG/minimality assumptions. While theoretical papers need not have extensive experiments, some exploration of assumption violations would strengthen the claim that the theory "provides guidance for the design of identification algorithms" (Abstract).

- **Additive structure in Fusion dataset may simplify the problem**: The Fusion dataset description (Section 6.1) uses $\mathbf{v}_1 = f_1(\mathbf{s}_1) + f_2(\mathbf{z})$, which is additive mixing. However, the theory assumes general nonlinear invertible mixing. Additive structure is a restrictive assumption that may make the identification problem easier than the general case claimed. The paper should clarify whether the theory applies to additive models specifically or whether the additive construction is merely a convenient way to generate invertible transformations for experiments.

### Trivial
- **Notation inconsistency in Section 6**: Table 1 uses $d_z$ for basis model experiments but $d_s$ for general PBG-SCM experiments (Figure 4), which could confuse readers about whether these refer to the same concept (latent dimension).

## Nice-to-Haves
- A discussion of potential approaches to estimate intrinsic dimension from data (e.g., using the minimality principle itself by minimizing dimension subject to reconstruction constraints) would help bridge the gap between theory and practice.

- Comparing against standard nonlinear ICA methods or VAEs on the same synthetic data would help quantify the practical benefit of the PBG/minimality assumptions over methods that don't assume them.

- Clarifying whether the additive structure in the Fusion dataset is merely a convenient experimental choice or reflects a limitation of the theory would strengthen the experimental section.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that "minimality assumption requires ground-truth knowledge, rendering the theory circular"**: This is overstated. The paper explicitly acknowledges in Section 7 that current algorithms require pre-known intrinsic dimension, and Corollary 5.1 shows this is one way to satisfy minimality. However, the minimality condition itself is defined without reference to ground truth (Assumption 1.iii: "there does not exist a model... such that $z' \prec z$"). The paper does not claim minimality can only be enforced via known dimension; it shows known dimension is *sufficient*, not *necessary*. The limitation is acknowledged transparently, not hidden.

- **Harsh Critic's claim that the paper "overclaims" by not highlighting the causal structure requirement in Abstract/Introduction**: The paper explicitly states in Section 2: "Our work do not aim to identify the structure of underlying SCM." This is a clear scope statement. The SCM Reduction is a theoretical construct showing equivalence classes, not a claim that practitioners must know the full graph. The critic conflates the theoretical reduction (which shows any SCM can be transformed to an equivalent PBG-SCM) with a practical requirement. The abstract's claim of "single-domain data" is accurate—the theory does not require multi-domain data, which is the key contribution.

- **Harsh Critic's characterization of experiments as "tautological"**: While the experiments do validate the theory under correct specification (standard for theoretical papers), calling them "tautological" is unfair. The experiments systematically test each assumption via ablation (Table 1 shows what happens when minimality or independence is violated), which provides empirical support for the theoretical claims. The critic's demand for experiments on dimension estimation or structure learning is reasonable as future work, but the current experiments are not worthless—they confirm the theory works as predicted.

- **Strength Finder's claim that minimality is "first proposed in this paper" and "overlooked by existing literatures"**: While the minimality condition is novel in this specific formulation, related concepts (sparsity, compression, parsimony) appear in causal abstraction and identifiability literature. The paper's framing is reasonable but should not overstate novelty.

## Novel Insights
The paper's key insight—that minimality of shared latent variables is necessary and sufficient for identifiability in single-domain settings—is genuinely novel. Prior identifiability theories for nonlinear causal models typically require multi-domain data, auxiliary variables, or restrictive functional forms (additive, compositional). This paper shows that a simple information-theoretic principle (shared latents should not contain more information than necessary) can replace these stronger requirements. The connection between intrinsic dimension and minimality (Corollary 5.1) provides a practical pathway: if dimension can be estimated, minimality is automatically satisfied. However, the paper does not fully resolve how to estimate dimension without ground truth, leaving this as an important open problem.

## Suggestions
1. **Add dimension sensitivity analysis**: Include experiments where latent dimension is estimated via standard methods (elbow method, information criteria, or recent intrinsic dimension estimators) rather than set to ground truth. This would show whether the theory holds under realistic model selection and provide practical guidance.

2. **Clarify the scope of the SCM Reduction**: In the Introduction or Section 4, explicitly state that the reduction requires knowledge of observed descendant sets (ancestor-descendant relationships), and discuss whether this is a practical limitation or whether such information can be obtained from domain knowledge or preliminary analysis.

3. **Discuss verifiability of minimality**: Add a theoretical discussion or bound on how one might verify the minimality condition without ground truth, or explicitly prove conditions under which it is impossible. This would clarify the theory's limits and guide future work.

4. **Compare to non-identifiable baselines**: Include standard VAEs or nonlinear ICA methods as baselines on the same synthetic data to quantify the performance gain from the PBG/minimality assumptions.

5. **Clarify the Fusion dataset construction**: Explain whether the additive structure ($\mathbf{v}_1 = f_1(\mathbf{s}_1) + f_2(\mathbf{z})$) is merely a convenient way to generate invertible transformations or whether it reflects a limitation of the theory.

## Score and Decision

**Calibration anchors consulted:**
- **ta8BKRa1bl** (Score 6.0): Theoretical identifiability paper with clear assumptions, synthetic experiments, and explicit limitations. Reviewer noted assumptions are "strong" but appreciated transparency. This paper is comparable in theoretical depth and assumption clarity.
- **0VVdai71xb** (Score 5.6): Theoretical identifiability framework criticized for "almost no experiment." This paper has more experimental validation.
- **qLbTww6vv2** (Score 4.0): Identifiability paper rejected because assumptions (known intervention targets, topological order) were hidden/implicit. This paper is MORE transparent about limitations.
- **CwYxl2l2J6** (Score 5.33): Nonlinear causal reduction framework with theoretical proofs and synthetic experiments. Comparable contribution level.
- **Wa3cfE3Iay** (Score 6.0): Identifiability theory with clear assumptions and near-identifiability definitions. Similar theoretical rigor.
- **ObXB7KJn0B** (Score 6.67): Theoretical paper with explicit limitations section. Shows that transparent acknowledgment of assumptions is valued.

**Positioning:**
This paper is stronger than qLbTww6vv2 (4.0) because assumptions are transparent, not hidden. It has more experimental validation than 0VVdai71xb (5.6). It is comparable to ta8BKRa1bl (6.0) and Wa3cfE3Iay (6.0) in theoretical clarity and assumption transparency. The main limitation—requiring known intrinsic dimension—is explicitly acknowledged in Section 7, which aligns with how high-scoring theoretical papers handle limitations (ObXB7KJn0B, ta8BKRa1bl).

The paper makes a genuine theoretical contribution with the minimality condition, provides experimental validation, and is honest about limitations. Based on calibration, this deserves a score of **6.0**.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>