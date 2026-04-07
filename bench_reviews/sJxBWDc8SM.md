## Summary
This paper conducts a large-scale empirical investigation into the learnability of modern recurrent models (State-Space Models like Mamba) compared to Transformers on fundamental synthetic tasks: multi-query associative recall (MQAR) and copying. Its core finding is that while SSMs can achieve expressive parity, they suffer from critical optimization instability, succeeding only within an extremely narrow learning rate window. The work also reveals divergent scaling behaviors (SSMs favor width, Transformers depth) and provides architectural ablations linking success to components like 1D convolutions.

## Strengths
- **Substantial and rigorous empirical foundation:** The conclusion that SSM performance is highly sensitive to hyperparameter tuning is backed by an extensive and methodical experimental campaign (over 3,000 runs). The paper convincingly demonstrates that prior performance gaps on MQAR (Arora et al., 2023) were likely confounded by insufficient learning rate grids (Figures 1, 2).
- **Actionable insights on scaling and architecture:** The identification of opposing scaling strategies—width for SSMs, depth for Transformers—is a clear, practical finding (Figures 3, 4). Furthermore, the ablation studies pinpoint the 1D convolution as a critical architectural component enabling single-layer models to solve MQAR, providing a concrete mechanistic link (Table 2).
- **Shifts the discourse from expressivity to learnability:** The paper successfully argues that a key differentiator between these architectures is not just what they can represent, but how reliably they can be trained. This reframes the community's comparison criteria and highlights optimization stability as a first-class challenge for SSM research.

## Weaknesses
- **Conclusions are drawn solely from synthetic benchmarks.** While MQAR and copying are well-motivated proxies for in-context learning, the paper's central claim—that optimization instability is a fundamental challenge for SSMs—remains untested on downstream tasks like language modeling. The discussion acknowledges this, but it limits the immediate practical significance of the findings.
- **Incomplete mechanistic explanation for the observed instability.** The paper empirically documents the narrow learning rate window but does not provide direct evidence (e.g., gradient norm analysis) for the proposed hypothesis that vanishing gradients in the S6 recurrence are the root cause. The link to prior theoretical work (Trockman et al., 2024) is noted but not substantiated with new measurements.

## Nice-to-Haves
- A small-scale validation on a real language modeling dataset (e.g., WikiText) would strengthen the claim that the observed optimization brittleness is a practical concern beyond synthetic settings.
- A more detailed analysis of how the effective learning rate window scales with model width, depth, and sequence length would provide clearer guidance for tuning.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness (Harsh Critic):** "Missing detail: The text states 'Attention always solves the task.' It should explicitly note that this refers to the 2-layer configuration..." *Removed because the paper consistently specifies the layer count in each section's context (e.g., Section 3 is explicitly about 2-layer models).*
- **Weakness (Harsh Critic):** "Clarity Issue: Figure 4 is conceptually important but potentially confusing..." *Weakened to a suggestion; the figure's message is interpretable with careful reading.*
- **Weakness (Spark Finder):** "Missing Experiments: Validation on a realistic language modeling task..." *Weakened to a nice-to-have; the paper's stated scope is a controlled investigation using established synthetic benchmarks to study fundamental learnability. Demanding real-world validation is scope creep.*
- **Weakness (Spark Finder):** "Ablation on optimizer hyperparameters beyond learning rate..." *Removed; a comprehensive sweep of all optimizer hyperparameters is not a standard requirement for a paper focused on identifying a core instability phenomenon.*
- **Weakness (Spark Finder):** "Experiments on longer sequence lengths relevant to SSMs' intended use..." *Weakened to a nice-to-have; the chosen sequence lengths (up to 512) are standard for the MQAR benchmark and sufficient to demonstrate the instability phenomenon.*

## Novel Insights
The paper's primary novel insight is the systematic shift of focus from theoretical expressivity to practical learnability as the crucial differentiator between Transformers and SSMs. Beyond this framing, it provides several specific new observations: the extreme learning rate sensitivity of SSMs contrasts sharply with Transformer robustness; single-layer Transformers exhibit a loss bump reminiscent of induction head formation yet fail to solve the task, while single-layer Mamba shows a similar bump but succeeds; and the performance of single-layer models on MQAR is critically dependent on the presence of a 1D convolution, a unifying architectural insight.

## Suggestions
- Clarify the caption and description of Figure 4 to more directly convey the intended message: that performance is dictated by *how* parameters are allocated (width vs. depth), not just by the total parameter count.
- Temper the language in Section 6 regarding the induction head interpretation (e.g., "suggests an attempt" rather than "indicates") to better reflect the speculative nature of this mechanistic claim without direct attention pattern analysis.