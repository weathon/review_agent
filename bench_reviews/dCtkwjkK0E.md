## Summary
This paper introduces an active learning framework for flow matching models, specifically targeting applications with expensive continuous labels (e.g., shape design). It proposes a theoretical analysis using piecewise-linear neural networks to explain how data affects model diversity and accuracy. Based on this, the authors derive two novel, competing query strategies: one to maximize diversity (QD) and one to maximize accuracy (QA). A weighted hybrid strategy balances this trade-off. Experiments on synthetic and real-world aerodynamic shape datasets demonstrate the strategies' effectiveness over active learning methods designed for discriminative models.

## Strengths
- **Novel Problem Formulation:** The paper rigorously addresses the underexplored problem of "active learning *for* generative models" (specifically flow matching), moving beyond the common paradigm of using generative models *for* active learning. This identifies and fills a clear gap.
- **Theoretically-Motivated Strategies:** The core query strategies (QD, QA) are directly derived from a formal, analytical framework that connects dataset composition to the diversity-accuracy trade-off in generation. This provides a principled foundation rare in active learning work.
- **Substantial and Relevant Empirical Validation:** Experiments are conducted on multiple non-trivial, real-world shape design tasks (airfoil, flying wing, starship) where label acquisition via numerical simulation is costly. Results consistently show the proposed strategies outperform adapted discriminative baselines on their respective targets, and the hybrid strategy provides tunable control.

## Weaknesses
- **Strong and Unverified Core Assumption:** The entire theoretical framework rests on the assumption that the flow matching model's neural network behaves as a piecewise-linear function. While motivated by citations on network condensation, this assumption is not empirically validated for the trained models in the paper. The generality of the theoretical claims and their applicability to standard flow matching architectures is therefore uncertain.
- **Limited Domain Demonstration and Baselines:** All real-world experiments are confined to the specific domain of aerodynamic shape design with low-dimensional continuous labels (1D to 4D). The paper's claims are framed generally, but efficacy on other domains (e.g., image generation) or with higher-dimensional conditions remains unshown. Furthermore, a simple but critical baseline—ongoing random sampling across active learning rounds—is omitted, making it harder to gauge the absolute improvement offered by the proposed strategies.
- **Insufficient Detail for Reproducing QD:** The diversity strategy QD (Eq. 4) combines three terms with weighting coefficients (α, β, γ) and uses a ∆_entropy_ term based on label clustering. The paper does not specify how these coefficients are set, how clusters are formed, or what distance thresholds are used. This lack of detail hinders reproducibility.

## Nice-to-Haves
- A discussion or simple experiment on the computational complexity and scalability of the distance calculations in data and label space for large unlabeled pools.
- Exploration of how the accuracy of the RBF network used for label prediction impacts the query strategies' performance.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The proof of Lemma 1 is notationally dense and somewhat difficult to follow." *(This is a subjective comment on presentation, not a substantive flaw in the paper's contribution.)*
- **Weakness:** "Training for 4,000,000 steps is computationally intensive... not discussed." *(The computational cost of training the final generative model is orthogonal to the active learning contribution, which focuses on reducing labeling cost. The query strategies themselves are model-agnostic and efficient.)*
- **Weakness:** "The paper does not compare against recent, sophisticated active learning methods..." *(The reviewer does not cite specific existing methods for active learning *for* generative models, making this an unverifiable demand. The paper clearly compares to relevant, adapted discriminative baselines.)*
- **Suggestion:** "Extend the theoretical analysis to more general network architectures..." *(This demands work beyond the paper's stated scope and contribution. The paper's analysis is explicitly built on the piecewise-linear framework.)*
- **Criticism:** "The description of the entropy term in QD is insufficiently detailed..." *(This point is valid and has been incorporated into the "Weaknesses" section as a reproducibility issue.)*

## Novel Insights
The paper's core novel insight is the data-centric explanation of the diversity-accuracy trade-off in conditional flow matching models. Through the piecewise-linear analysis, it demonstrates that data points sharing the same label primarily contribute to the *diversity* of generated samples for that condition, while data points with distinct labels improve the model's *accuracy* by reducing interpolation error across the condition space. This insight directly motivates two fundamentally conflicting query objectives (QD and QA), providing a principled perspective on a well-known challenge in generative modeling.

## Suggestions
- Provide an empirical validation of the piecewise-linear assumption, for instance by visualizing whether generated samples for intermediate conditions (not in the training set) approximate linear interpolations of nearby training data in the synthetic task.
- Include an ongoing random sampling baseline in the experiments to clearly establish the added value of the proposed query strategies.
- In the experiment section or an appendix, specify the values or tuning procedure for the coefficients (α, β, γ) in QD and provide details on the label clustering process (e.g., distance threshold) to ensure reproducibility. Releasing code would strongly support this.