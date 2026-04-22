# Diverse Dictionary Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
Given only observational data $X = g(Z)$, where both the latent variables $Z$ and the generating process $g$ are unknown, recovering $Z$ is ill-posed without additional assumptions. Existing methods often assume linearity or rely on auxiliary supervision and functional constraints. However, such assumptions are rarely verifiable in practice, and most theoretical guarantees break down under even mild violations, leaving uncertainty about how to reliably understand the hidden world. To make identifiability *actionable* in the real-world scenarios, we take a complementary view: in the general settings where full identifiability is unattainable, *what can still be recovered with guarantees*, and *what biases could be universally adopted*? We introduce the problem of *diverse dictionary learning* to formalize this view. Specifically, we show that intersections, complements, and symmetric differences of latent variables linked to arbitrary observations, along with the latent-to-observed dependency structure, are still identifiable up to appropriate indeterminacies even without strong assumptions. These set-theoretic results can be composed using set algebra to construct structured and essential views of the hidden world, such as *genus-differentia* definitions. When sufficient structural diversity is present, they further imply full identifiability of all latent variables. Notably, all identifiability benefits follow from a simple inductive bias during estimation that can be readily integrated into most models. We validate the theory and demonstrate the benefits of the bias on both synthetic and real-world data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the problem of diverse dictionary learning to make identifiability actionable in the real-world scenario. In particular, they study: in the general settings where full identifiability is unattainable, what can still be recovered with guarantees, and what biases could be universally adopted?

Section 3 introduces the main technical results.
The current presentation is hard to see it's impact in real problems. 
There is a paragraph started with "Why does it matter in the real world?" By reading that paragraph, it's still not clear how the research problem in this paper relevant to any objective that matters in reality, such as improving the performance of a predictive model or dimension reduction. 

The current simulations don't demonstrate the potential usefulness of the work. The synthetic data part focuses on Generalized Identifiability and Element Identifiability. Without knowing the possible impact of quantifying them, it is hard to see the potential value. 

The section 4.2 is written in the way for readers who are familiar with the references. It does not provide sufficient explanation of the experimental results and their potential impact in practice. 

Overall, it is hard to see the potential impact of this work. A better justification is needed.

### Strengths
The paper is free of grammatical errors.

### Weaknesses
See in the Summary; the later part.

### Questions
See in the Summary; especially in the later part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies nonlinear dictionary learning under a general generative model $X = g(Z)$, where both the latent variables $Z$ and the generator $g$ are unknown. It introduces a set-theoretic notion of identifiability, focusing on recoverability of set operations: intersections, complements, and symmetric differences. Under assumptions of positive density and dependency sparsity, the authors show that these structural dependencies are identifiable, and that element-wise identifiability follows when the system exhibits “sufficient diversity.”

### Strengths
1) The paper provides a practical and general inductive bias that improves disentanglement without model-specific changes, May be useful for interpretability and modular representation learning.
2) Strong theoretical results: generalized and structural identifiability (Theorems 1–3) under mild assumptions.
3) The “sufficient diversity” condition unifies and relaxes previous sparsity-based identifiability assumptions.
4) Introduces a novel set-theoretic identifiability framework that generalizes traditional recovery notions through operations on supports of $D_z g$.

### Weaknesses
Below I list a few weaknesses of the paper: 

1. The “sufficient nonlinearity” and independance conditions on $D_z g$ are elegant but may be hard to verify or hold only asymptotically. Some discussion of finite-sample behavior or diagnostics would help.

2. Theoretical results assume $\ell_0$ regularization on $|D_z \hat g|_0$, but experiments use an $\ell_1$ proxy. A clearer justification or sensitivity analysis for this relaxation would strengthen the bridge between theory and practice.

3. Penalizing full Jacobians can be expensive for large decoders or diffusion models. Runtime or memory comparisons against latent-sparsity or Hessian-based baselines are missing.

4. Theoretical claims focus on structural identifiability, but evaluations (e.g., DCI, FactorVAE) measure disentanglement indirectly. A metric targeting recovery of the Jacobian’s support would better reflect the paper’s main claims.

5. Noise and model misspecification (e.g., non-additive noise, partial non-invertibility) are briefly mentioned but not deeply explored; results may not generalize to such cases.

**On another note, I am not sure how good of  a fit is this theoretical endeavor suitable for ICLR**

### Questions
Do you have any results or insights on how accurately the support of $D_z g$ can be recovered with finite samples or under noise?
Can the diversity condition be checked empirically?
Would it be possible to report direct recovery metrics?

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
2

### Summary
The authors introduce a new, weaker form of identifiability based on "set-theoretic indeterminacy". The main theoretical result is the Set-Theoretic Guarantees (Theorem 1): that even when full recovery is impossible, structural relationships between latent variables are identifiable. Specifically, the intersection ($I_K \cap I_V$, shared factors), complement ($I_K \setminus I_V$, exclusive factors), and symmetric difference ($I_K \Delta I_V$, unique factors) of latent variable sets are shown to be disentangled.

These identifiability guarantees (Theorems 1 & 2) are achieved by introducing a "simple inductive bias" during estimation: a sparsity regularization on the dependency structure (i.e., the Jacobian $D_{\hat{Z}}\hat{g}$). This is a practical regularization rather than a strict assumption about the data-generating process itself.

The authors show that their set-theoretic guarantees naturally extend to full, element-wise identifiability (the traditional goal) if a "Sufficient Diversity" condition (Assumption 2) is met. This condition, which is shown to be a weaker and more general structural requirement than in prior work, essentially ensures the dependency structure is rich enough for each latent variable to be isolated in its own "atomic region" of the set-theoretic Venn diagram.

Finally, the authors conduct experiments on synthetic and real-world image datasets to validate their theory and demonstrate the practical benefits of the proposed dependency sparsity bias.

### Strengths
- The research area of expanding traditional dictionary learning inductive priors to learn true latents of the data is interesting and a valuable area of research. 
- The paper appears theoretically sounds. The authors provide rigorous definitions for all new theoretical concepts and include detailed proofs in the appendix. I took a brief look at the proofs and did not find any issues, although I was not able to follow everything and there is a good chance I could have missed an issue. 
- The writing is very clean and easy to understand (although the math is dense) with associated well made figures backing up the intuition with toy examples which really helped my understanding of the paper.

### Weaknesses
- The paper discusses SAEs in the introduction but never again throughout the paper. I was expecting some experiments or discussion about whether Diverse Dictionary Learning could be used in-place of traditional SAE losses. This shouldn't be difficult to do as there are many small models + SAE libraries available and would be really interesting to compare. Indeed, SAEs assume the linear representation hypothesis, but there is some decent evidence that the LRH is at least somewhat true in many scenarios - how does the proposed method do when examining model internals?
- The main theory relies heavily on Assumption 1 (Sufficient Nonlinearity). I checked the referenced papers in Lines 303 and found that some of them contained similar statements but it was hard to determine if they were identical. I think the paper would be much stronger if the authors provided a bit more intuition as to why Assumption 1 is reasonable to make. Even better, if there were some small experiments to show that this generally holds empirically, that would strengthen the paper a lot. However, if this really is a very standard assumption, then this evidence is not wholly necessary, but I would appreciate the authors highlighting exactly which equations in which papers make the same assumption so I can more readily compare the two. (For example, I am guessing Assumption 1 in Lachapelle et al., 2022 makes the same argument?)

### Questions
- The conclusion ends suggesting to explore identifiability in foundation models. How would one practically implement and optimize this dependency sparsity bias in a large-scale transformer as the full Jacobian of a transformer's activations is computationally intractable. 
- How does the proposed diverse dictionary learning compare to traditional SAE methods for *interpreting* model architectures rather than solely looking at traditional datasets? Would the result be identical to that of Farnik et al. 2025 (Jacobian Sparse Autoencoders)?
- I am not sure I fully understand how the terminology and figures would map onto a real world example. In a complex domain like images, what would these "atomic regions" practically represent? For example, if $I_1$ = "dog" latents and $I_2$ = "cat" latents, could the intersection $I_1 \cap I_2$ be the latent for "furry," and the complement $I_1 \setminus I_2$ the latent for "barks"? How does this set-based view compare to traditional feature representations?

Generally, I do not see many issues with the paper, and if my confusions are addressed during the rebuttal I would be happy to raise my score accordingly!

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies a set theoretic notion of dictionary learning, where the properties of the collection of atoms and the model are defined at the full set theoretic level of abstraction, and the generative model admits a Jacobian. The authors define a series of strenghtenings of identifiability with respect to how accurate to “ground truth” the estimated values are. They then study the effect of sparsity. Finally, they define a diversity condition that allows for element-wise recovery. They evaluate their theoretical results empirically.

### Strengths
The main strength of the paper is in abstracting sufficient conditions for recovery of various consistent dictionaries from examples. It is powerful to reason about these properties in a general class of generative models using only set theoretic terms.

### Weaknesses
Some of the theoretical claims are imprecise. In particular, I am really puzzled by the interplay between Definition 6 and Theorem 1. Is Theorem 1 a result about methods that satisfy the regularization condition? Or should I read it as a result about the nature of identifiable models, i.e., they are ones that have higher sparsity than all others that are observationally equivalent? This requires clarification. Also, I am a bit confused about Theorem 2: if all the conditions of Theorem 1 are met, you get both generalized and structure identifiability? There are no additional conditions?

Related to all of this: in section 3.3, what is the quantifier over $\hat{z}, \hat{g}$? I thought it was the estimated value. I don’t see where it shows up in Assumption 1 until the last part, and in Theorem 1, the hat versions are specified but somehow you are invoking Defn 6, in which the quantifier is “any” theta? 

On the necessity side: Are there any necessity conditions for set-theoretic indeterminacy or generic identifiability?

Finally, it would be helpful in your exposition to map your abstracted properties to properties in general linear dictionary learning, non-linear dictionary learning, and combinatorial dictionary learning. (e.g., diversity = RIP = well-structuredness).

I can consider increasing my evaluation of the paper if the theoretical claims are made more precise.

### Questions
In addition to the questions above:

Maybe I’m missing something, but why is Theorem 1, condition (ii) called “regularization”? To me regularization is algorithmic, whereas this is structural.

Can you please clarify why your experiments are about structural properties of dictionary learning, rather than the specific methods you have implemented?

### Soundness
2

### Presentation
2

### Contribution
3
