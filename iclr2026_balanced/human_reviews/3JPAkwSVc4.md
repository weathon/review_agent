## Human Reviewer 1

### Summary
This paper establishes a formal mathematical framework for diffusion modeling on quotient spaces, with applications to molecular structure generation under SE(3) symmetry. The key idea is to construct diffusion processes directly on the quotient space M/G, then lift them back to the total space M for practical implementation using horizontal projections. The framework reduces learning difficulty by removing redundant components corresponding to group actions while guaranteeing correct sampling from the target distribution.

### Strengths
1. The paper provides a comprehensive mathematical framework grounded in Riemannian geometry and stochastic calculus. Theorems 1-4 give explicit characterizations of the projected diffusion process and its horizontal lift.
2. The formulation of diffusion models via horizontal lifting on quotient spaces is innovative. It generalizes previous work on equivariant diffusion and provides a unifying geometric perspective that can be applied to various symmetry groups.
3. Unlike heuristic alignment strategies, the proposed method guarantees correct sampling from the target distribution while reducing learning difficulty by removing redundant degrees of freedom.

### Weaknesses
1. The paper doesn't discuss the computational cost of computing the horizontal projection operator $P_x$ and mean curvature vector $h$ at each step. For the SE(3) case, these involve matrix inversions and cross products - what is the actual time comparison?
2.  The heavy reliance on differential geometry, Riemannian manifolds, Lie groups, and stochastic calculus makes the paper inaccessible to much of the machine learning community. While mathematical rigor is valuable, the presentation could benefit from more intuitive explanations alongside formal derivations.
3.  While consistent, the improvements over baselines are relatively small in some cases, which raises questions about practical significance.

### Questions
1. How does the method perform on other symmetry groups beyond SE(3)?
2. How does the computational cost of projection scale? Are there approximations or efficient implementations for large-scale problems?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper investigates the often overlooked issue of additional training difficulty due to symmetries in diffusion models. The paper proposes considering the projected diffusion dynamics to the quotient space of interest in order to remove the original symmetries and simplify the learning problem. The paper provides a mathematical analysis of this new diffusion process and provide an efficient approach to training and sampling within this quotient space. This approach is then validated empirically in conformation generation for small molecules and protein backbone generation demonstrating improved results.

### Strengths
* I think the paper is very well done. It answers an important question (that I've also been considering) for diffusion models applied to common data modalities in scientific applications.

* The mathematical derivation of quotient space diffusion is handled nicely and the resulting training and sampling algorithms are also nicely implemented in the ambient space to sidestep the complex form of the quotient space.

* Very nice summary of previous approaches of handling the additional difficulty of learning symmetries in Section 3.4, as well as an explanation as to why there are issues with sampling time mismatches. This is a useful exposition as this has been less emphasised in the literature.

* The empirical evaluation is well done with definite benefits reported for the relevant and practically important examples of conformation generation for small molecules and protein backbone generation.

### Weaknesses
* It is perhaps less clear how SMC/reweighing-based approaches for reward tilting could applied to this framework as I am not totally sure that transition probabilities computed from equation 9 represent the correct transition probabilities that you would want from the actual quotient space diffusion.

### Questions
* Are there ever numerical instabilities from the projection operator or the horizontal lift of mean curvature due to matrix inverse for near collinear systems of points?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper addresses the critical problem of intrinsic system symmetries (e.g., $SE(3)$ symmetry in molecular generation) that complicates generative modeling. Through a theoretically rigorous approach, it establishes a formal framework for diffusion models on a general quotient space $\mathcal{M}/\mathcal{G}$. The method involves projecting the standard SDE onto this quotient space and subsequently deriving a new "horizontal lift" SDE in the original space, which is constrained to movements that are purely horizontal to the group action. This design effectively reduces the learning space for the neural network, thereby lowering the learning difficulty. Critically, this new SDE guarantees consistency between the training objective and the inference process, ensuring sampling compatibility, which prior heuristic alignment methods lack. The paper provides a detailed and persuasive implementation, with exhaustive comparisons against these heuristic approaches, demonstrating excellent performance on key benchmarks: small molecule structure generation (on GEOM-QM9 and GEOM-DRUGS) and protein backbone design.

### Strengths
1. **Significance and Theoretical Contribution**: This paper addresses a highly valuable and prevalent problem in generative modeling: the inconsistency between training objectives and sampling processes caused by intrinsic system symmetries. The work is exceptionally timely, as similar issues are being reported in various domains, such as periodic crystal generation [1, 2]. The authors are encouraged to cite similar works to further strengthen significance. The paper provides a powerful and unifying theoretical lens to explain and, more importantly, solve these inconsistencies.

2. **Elegant and Sound Framework**: The proposed framework is mathematically elegant and theoretically sound. The derivation—projecting the standard diffusion SDE onto the quotient space $\mathcal{M}/\mathcal{G}$ and then deriving its corresponding "horizontal lift" SDE back in the original space—is a principled and clean solution.

3. **Excellent Analysis of Prior and Heuristic Methods**: A major strength is the paper's insightful analysis of prior work within its new framework. The discussion (and Table 1) that masterfully categorizes and explains the shortcomings of previous methods (like data augmentation or inference-only alignment) is excellent. The critique of heuristic alignment strategies (GeoDiff, AF3) is particularly brilliant, clearly articulating how they introduce high variance or suffer from sampling incompatibility.

4. **Clarity and Intuition**: The paper is well-written. Figures 1 and 2, in particular, are exceptionally clear and provide excellent intuition for the complex mathematical concepts involved, making the core idea highly accessible.

5. **Strong Empirical Validation**: The authors provide thorough ablation studies and experiments on challenging, standard benchmarks (GEOM-QM9, GEOM-DRUGS, and protein design). The empirical results largely demonstrate the superiority of this principled approach over existing methods.

[1]"Equivariant diffusion for crystal structure prediction." ICML,2024

[2]"Kinetic Langevin Diffusion for Crystalline Materials Generation."ICML,2025

### Weaknesses
1. **Missing Discussion on GEOM-DRUGS Results**: In Table 3, the performance on the GEOM-DRUGS dataset, while strong, appears to be below that of the MCF baseline. This is a noteworthy result, but the paper does not provide any discussion or analysis for this specific comparison. A brief discussion of this would strengthen the experimental section.

2. **Missed Visualization Opportunity**: While Figures 1 and 2 are excellent for conceptual understanding, the paper would be significantly improved by a new figure that visualizes the dynamics of the proposed horizontal diffusion process itself. For example, a qualitative visualization of the distribution's evolution on a simple manifold (e.g., $\mathcal{M}/SO(2)$) would be highly illustrative and provide a powerful complement to the static diagrams.

3. **Minor Formatting Suggestion**: In Table 3, the results for "GEOM-DRUGS" are a key contribution. Consider using bold formatting or out-standing rows to make it more clearly.

### Questions
1. **Guarantee of Recovering the Full G-Invariant Distribution**: The paper's primary focus is to ensure the SDE matches the quotient space distribution and maintains sampling consistency. This is a clear success. However, a question remains about learning the full G-invariant target distribution $p(x)$. For a finite dataset (which is not perfectly G-invariant), does the horizontal lift SDE guarantee that it learns the correct probabilities for all equivalent points $g \cdot x$? For example, if the dataset contains only two samples $x_1, x_2$, the true target distribution should be non-zero and have the same probability on the entire orbit $ \\{g \cdot x_1, g \cdot x_2 | g \in \mathcal{G} \\} $. Does the proposed method recover this full manifold, or does it primarily learn the quotient-space projection well? A deeper discussion on this geometric probability coverage would be valuable.

2. **Comparison to Other Lie Group Diffusion Methods**: Could the authors comment on how this quotient-space framework compares, in terms of theoretical advantages or disadvantages, to other recent work on diffusion for symmetric data? For instance, methods that operate directly on Lie groups (e.g., "Trivialized momentum facilitates diffusion generative modeling on Lie groups," ICLR 2025, "Flow matching on general geometries."ICLR 2024) also aim to simplify the learning process.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper addresses the problem of learning diffusion models for distributions with specific group symmetries. The proposed framework learns the score function on the quotient space with respect to the group, avoiding the need to model components corresponding to group actions. They start by projecting the standard forward SDE onto the quotient space using a Riemannian Ito's lemma. They then observe that the vector field on the quotient space admits an orthonormal decomposition into horizontal and vertical components, where only the horizontal component is relevant as the vertical component corresponds to infinitesimal group actions. Given the non-trivial geometry of the quotient space, they re-define a process in the original space that retains only the horizontal movement, called horizontal lift. Experiments focus on molecular structure generation, which is characterized by the SE(3) symmetry. Results demonstrate that training with this approach outperforms two alternative heuristics for handling symmetries.

### Strengths
The paper tackles the problem of handling symmetries in a novel and principled manner in contrast to previous approaches that rely on heuristics to deal with the symmetry problem. It is the first work I have seen that attempts to constrain the diffusion process within the quotient space.The authors also provide intuitive explanations of the main differences between their proposed method and existing approaches, supported by Table 1 and Section 3.4. Additionally, they include a comprehensive background on Riemannian geometry and stochastic calculus in the appendix to help readers follow the proofs of the theoretical claims presented in the main text. Experimental results demonstrate that applying their framework to both molecular and protein data improves the performance of the model compared to the use of conventional heuristic methods for handling symmetries.

### Weaknesses
I believe that, in principle, the approach proposed in the paper can be applied to any diffusion model by keeping the backbone architecture and changing the training, if I am not wrong. Therefore, it would have been interesting to see whether the improvements observed for the selected model also hold across different architectures. In relation to this, the results table could have been more informative if, for each baseline, the specific heuristics used to handle symmetries were explicitly stated. 

Additionally, while the theoretical advantages of the proposed model are clearly illustrated in Table 1, I would have expected these advantages to be examined empirically as well, rather than being evaluated only based on final performance metrics. For example, I would expect a similar convergence behaviour between the proposed approach and the AF3 heuristic, since they are both removing the unnecessary DOFs and also lowering the variance. With respect to GeoDiff,I would expect the proposed method to achieve faster convergence during training, as it removes the equivalent DOFs. 

Finally, the proposed framework rely on a horizontal projection operator that involves an inverse of a matrix that depends on the input dimensions, i.e. number of points in the cloud, or atom in the molecule. It would have been useful to include a discussion on whether this operation could become a bottleneck when scaling the method to larger molecules, and whether the matrix inversion might introduce numerical instability.

### Questions
- Just to clarify my understanding, could the first paragraph of Section 3.4 be related to [1], where the authors highlight a mismatch between using a translation-invariant score network and a target score that is not translation-invariant? In other words, given $x_t$ and $g\cdot x_t$, which are equivalent, the network is not able to distinguish between them while the corresponding target scores differ, leading to high variance during training?
- By avoiding the modeling of these unnecessary degrees of freedom using the proposed method, do you observe that this leads to better results when sampling with fewer steps compared to models trained using the other heuristics?

- line 961 'up' instead of 'bp'

[1] "Equivariant Diffusion for Crystal Structure Prediction" Lin et al, ICML 2024

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3