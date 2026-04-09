## Summary
This paper presents the first theoretical analysis of the training dynamics and ICL generalization of one-layer Mamba models on binary classification tasks with additive outliers in prompts. The key result is that Mamba's nonlinear gating mechanism enables it to tolerate outlier fractions approaching 1 in context examples, while linear Transformers fail when outliers exceed 1/2. The paper characterizes the mechanism: the linear attention layer selects context examples sharing the query's relevant pattern, while the gating layer suppresses outlier-containing examples and induces a recency bias that emphasizes nearby clean examples.

## Strengths
- **First theoretical treatment of Mamba's ICL training dynamics with outliers.** The paper handles the nonlinearity of Mamba's gating mechanism—a significant technical challenge that prior Transformer-focused ICL theory (Zhang et al., 2023; Li et al., 2024a) does not address—by dividing training into two phases (Lemmas 4–5) and characterizing gradient updates along relevant, irrelevant, and outlier pattern directions. This is a genuine methodological contribution.
- **Principled isolation of the gating effect.** The comparison with linear Transformers is methodologically clean: setting $G_{i,l+1}(\mathbf{w}) = 1$ reduces Mamba to a linear Transformer, making the gating mechanism the only architectural difference. This allows the paper to rigorously attribute the robustness gap specifically to nonlinear gating rather than other confounding factors.
- **Mechanistic interpretability supported by theory and experiments.** Corollaries 1 and 2 provide concrete characterizations—attention concentrates on same-pattern examples, gating suppresses outliers ($G \lesssim \text{poly}(M_1)^{-1}$) and decays exponentially with index distance—which are directly validated in Figures 3–4 and Table 1. This goes beyond black-box bounds to explain *why* the architecture works.

## Weaknesses

### Major:
- **The robustness advantage over Transformers is proven only against linear attention, but softmax attention shows comparable robustness empirically.** Appendix B.1 (Table 3) shows that a softmax Transformer achieves 99.28% accuracy in the CQ setting where Mamba drops to 82.73%, and maintains >99% accuracy for $\alpha \leq 0.7$ (Table 4). The paper's framing in the Abstract and Introduction ("Mamba...achieving comparable performance across a wide range of language tasks" and comparison with "Transformer-based models") risks leading readers to believe the robustness advantage applies to standard Transformers. Remark 6 acknowledges this but is insufficiently prominent. The practical significance of the Mamba advantage is substantially reduced by this finding, as standard LLMs use softmax attention, not linear attention.

- **The recency bias that enables outlier suppression creates a structural vulnerability to outlier position.** Corollary 2(ii) and Eq. (18) show that gating values decay exponentially with distance from the query. This is the mechanism by which outliers are suppressed—but it also means that when outliers are placed closest to the query (CQ setting), clean examples are pushed far away and their gating values decay, causing Mamba's accuracy to drop to 82.73% (Table 1). The linear Transformer, lacking this decay, maintains 93.96% in the same setting. The paper's main claims emphasize robustness to outlier *fraction* without adequately foregrounding this positional vulnerability, which is a direct and important consequence of the same mechanism.

- **Generalization to unseen outliers requires them to be positive linear combinations of training outliers (Theorem 2, Condition (a)).** This restricts the "distribution-shifted" outlier robustness to a specific subspace spanned by training outlier patterns. If a test-time outlier has a component orthogonal to all training outliers (which is the more practically relevant adversarial setting), the theoretical guarantees do not apply. The paper should more clearly articulate this limitation and discuss whether the gating mechanism still provides partial protection in such cases.

### Minor:
- **The one-layer, binary classification setting limits direct practical implications.** While this is standard for theoretical ICL analysis (Zhang et al., 2023; Li et al., 2024a; Li et al., 2025b), the gap to practical multi-layer Mamba models handling natural language remains substantial. The 3-layer experiments in Section 4.2 partially address this but do not establish whether the theoretical phase-transition dynamics (Lemmas 4–5) persist at depth.

- **The SST-2 validation (Appendix B.2) provides only weak support for the orthogonal pattern assumption.** Table 6 shows that classification with top-10 PCA components is close to full-dimension accuracy, but PCA orthogonality does not imply that semantic patterns are orthogonal or sparse in the manner required by the theoretical framework (Eq. 6). The "James Bond" outlier experiment (Table 7) is more convincing as a proof of concept but is still limited to a single dataset and outlier type.

### Trivial:
- The abstract's claim that Mamba achieves "comparable performance across a wide range of language tasks" is a general statement about Mamba from prior work and not a contribution of this paper. It does not mislead about this paper's specific contributions but could be more precisely scoped.

## Nice-to-Haves
- Include the softmax Transformer comparison (Table 3) in the main text rather than relegating it to the appendix, since it substantially qualifies the main claims and is essential for readers to assess practical significance.
- Provide a formal or semi-formal analysis of the CQ failure mode—e.g., a lower bound or tighter characterization of when the recency bias becomes a liability—as the current treatment is purely empirical.
- Evaluate on a standard text-based ICL benchmark where outliers take the form of semantically corrupted instructions or mislabeled examples, rather than only additive feature noise.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Missing related works."** Hard rule: cannot confirm existence of uncited works.
- **Weakness: "No computational complexity analysis."** The paper does provide iteration counts (e.g., $T_M = \Theta(\eta^{-1}(1-p_a)^{-1}\beta^{-2}M_1)$) and batch size requirements; claiming no complexity analysis exists is factually incorrect.
- **Weakness: "Missing comparison with data augmentation/ensemble/robust optimization baselines."** This is outside the scope of a theoretical analysis paper; the paper's contribution is provable guarantees, not practical robustness methods.
- **Weakness: "Unclear practical applicability of complex conditions."** This is a generic criticism that applies to nearly all theoretical ML papers with sufficient conditions; the conditions are explicitly stated and interpretable (e.g., outlier magnitude must be in a specific range, context length must exceed a threshold).
- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** Hard rule: remove nitpicks about reproducibility of implementation details.
- **Weakness: Formatting/style issues in equations.** Hard rule: remove formatting nitpicks.

## Novel Insights
The paper reveals a fundamental design trade-off in gated SSMs: the same exponential decay mechanism that provides robustness to outlier *fraction* creates a positional vulnerability when corrupted tokens appear near the query. This is not merely an empirical observation—it is a direct structural consequence of the gating formulation (Corollary 2, Eq. 18). This suggests that robustness in SSMs comes from a specific spatial prior about where noise appears in the context, rather than uniform noise tolerance. The practical implication is that prompt engineering for Mamba-based ICL must consider not just *how many* examples are corrupted, but *where* they are positioned—a constraint that softmax attention does not impose to the same degree.

## Suggestions
- Move the softmax Transformer comparison from Appendix B.1 to the main text and revise framing to say "Mamba outperforms *linear* Transformers" rather than "Transformers" in the Abstract and Introduction, or add a clear caveat about the scope of the theoretical comparison.
- Add a "Limitations" paragraph explicitly discussing the CQ positional vulnerability as a structural consequence of the gating mechanism, not merely an empirical observation.
- In Theorem 2, add a brief discussion of what happens when Condition (a) is violated—does the gating mechanism still provide partial suppression, or does the guarantee collapse entirely? Even informal reasoning here would strengthen the practical relevance.