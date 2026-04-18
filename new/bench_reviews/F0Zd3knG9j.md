## Summary

This paper introduces a hierarchical filtering procedure for tree-based generative models of discrete sequences, where a parameter $k$ controls the depth of hierarchical correlations. Using this controlled setting, the authors study how vanilla encoder-only transformers learn on root classification and masked language modeling (MLM) tasks. They show that transformers match Belief Propagation (BP) accuracy and marginals across filtering levels, exhibit sequential discovery of longer-range correlations during training, and display attention maps and probing results that are consistent with a hierarchical bottom-up inference computation. An existence proof shows BP can in principle be embedded in the architecture with $\ell$ layers.

## Strengths

- **Elegant filtering framework.** The $k$-filtering mechanism is a genuinely useful methodological contribution that enables fine-grained control over correlation structure, allowing clean in-distribution and out-of-distribution comparisons (e.g., training on $k_{\text{train}}$ and testing on $k_{\text{test}}$). This goes beyond what prior work on the Random Hierarchy Model offered.

- **Strong multi-faceted empirical evidence.** The paper doesn't rely on a single metric. It demonstrates alignment between transformers and BP through: (a) accuracy matching across filtering levels (Figs. 3, 5); (b) calibrated probability distributions matching BP marginals including on out-of-sample filtering variations (Fig. 1b); (c) sequential learning dynamics progressively aligning with BP$_k$ for decreasing $k$ (Fig. 1c-d); (d) structured attention maps reflecting hierarchy (Fig. 6); and (e) ancestor probing showing layer-wise hierarchical information (Fig. 7). This convergent evidence is substantial.

- **Non-trivial calibration result.** The finding that transformers trained with hard (one-hot) labels on $k>0$ data nonetheless produce probability distributions calibrated to BP marginals—despite never being explicitly trained on marginals—is a strong and surprising result. It goes beyond accuracy to suggest genuine understanding of the posterior structure.

- **Clear and pedagogical writing.** The paper is well-structured, the model is clearly specified, and the experimental narrative flows logically from data design to dynamics to mechanistic analysis.

## Weaknesses

### Fatal
None.

### Major

- **The claim that transformers "implement" or have "equivalence in computation to" BP overreaches the evidence.** The paper's central narrative is that transformers approximately implement the BP algorithm. The evidence—output-level matching of accuracies and marginals, attention map patterns, and ancestor probing—is consistent with BP implementation but does not decisively distinguish it from other algorithms that produce the same posterior. A large-capacity network can match the input–output mapping of BP without using message-passing internally. The paper's own existence proof (Sec. 4, Appendix E) is explicitly acknowledged as an idealized construction ("this does not represent an exact explanation of the trained transformer computation") that relies on specific hand-designed embeddings, disentangled representations, and attention patterns not enforced during training. The probing evidence shows ancestor information is *encoded* at appropriate layers, but as widely discussed in the probing literature, encoding does not establish *use* in a specific algorithm. No causal/interventional experiments (e.g., ablating specific attention pathways and checking whether failures match BP predictions) are provided. The paper would be substantially stronger with either (a) moderated language ("consistent with" rather than "implements") or (b) interventional evidence that directly links specific model components to specific BP computations.

- **Experiments rely on a single grammar realization and a single depth ($\ell=4$).** All main results use one randomly sampled transition tensor with $q=4$ and $\ell=4$ (16-token sequences, 4 transformer layers). The staircase learning dynamics, attention map patterns, and calibration results need to be demonstrated as robust across multiple grammars and deeper trees. While the paper references Appendix D.2 for experiments on other grammars, the main body provides no quantitative evidence—no variance across seeds, no multiple grammar instances, and no exploration of how phenomena scale with $\ell$. With only 4 hierarchical levels, the "sequential discovery" narrative rests on very few observable stages, making it easy to over-interpret smooth improvements as discrete transitions.

### Minor

- **The non-overlapping (deterministic parent reconstruction) assumption is a significant simplification that is under-discussed.** The paper acknowledges this makes $k=0$ inference deterministic but does not explore what happens when the grammar is ambiguous. This is the regime where BP and alternative inference strategies diverge most, and where the "implements BP" claim would face its strongest test. This is a natural direction for future work but should be more explicitly flagged as a limitation on generality claims.

- **Quantitative calibration metrics are sparse in the main text.** Claims of "close match" and "near equivalence" between transformer and BP marginals (Secs. 3.2-3.3) are supported primarily by scatter plots and KL divergence curves without reporting numerical values (mean, variance, across seeds). For a paper whose central claim hinges on approximate algorithmic equivalence, more precise quantification would strengthen the argument.

- **Averaged attention maps may obscure per-input variability.** Figure 6 averages attention over $10^4$ inputs. It would be valuable to see whether the hierarchical structure is consistent across individual inputs or whether the average masks significant variation. This is a minor point because the overall pattern is still informative.

### Trivial
None.

## Nice-to-Haves

- **Interventional/causal experiments.** Ablating specific attention patterns (e.g., zeroing attention within blocks of size $2^{\ell-k}$) and verifying that failure modes follow BP predictions would provide much stronger evidence for the BP implementation claim.

- **Experiments with ambiguous grammars or variable tree topologies** would test the robustness and generality of the findings toward more realistic settings like probabilistic CFGs.

- **Direct comparison of learned representations to BP messages** (e.g., correlating hidden states with BP message vectors) would bridge the gap between the existence proof and the actual computation.

- **Comparison to alternative architectures** (e.g., MLPs, RNNs) on the same tasks would clarify whether the observed phenomena are transformer-specific or emerge in any sufficiently expressive model.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"No multi-head attention experiments" (from Spark):** The paper explicitly uses single-head attention (stated in Sec. 4), which is a deliberate and reasonable design choice for mechanistic interpretability. Adding multi-head configurations would add complexity but is not a core flaw of a study designed to understand single-head behavior.

- **"No baseline comparisons (MLPs, RNNs)" (from Spark):** The paper's contribution is specifically about understanding *how* transformers process structured data, not about comparing architectures. The BP oracle already serves as the relevant baseline (the optimal algorithm). Demanding other architectures is scope creep beyond the paper's stated goal.

- **"The number of layers $n_L$ is always set equal to $\ell$" (from Spark):** The paper explicitly references Appendix D.1 for experiments with different $n_L$ values and discusses the design choice in Sec. 3.1. While the main results use $n_L = \ell$, this is acknowledged and partially addressed.

- **"Missing related work" (from Neutral Reviewer):** Removed per instructions—no external sources to confirm existence of specific missing references.

- **"Reproducibility concerns (undisclosed hyperparameters, etc.)" (from Harsh Critic's Section-by-Section notes):** Removed per instructions—nitpicks about reproducibility details are not substantive weaknesses for evaluation.

- **"Formatting/style issues" (from Harsh Critic on notation BP$_k$):** This is a minor notation consideration, not a substantive weakness.

## Novel Insights

The hierarchical filtering parameter $k$ as a diagnostic tool for dissecting *what* correlations transformers learn and *when* they learn them is a genuinely useful methodological contribution. The "staircase" learning dynamics—where a model trained on $k=0$ data sequentially aligns with BP$_k$'s predictions for decreasing $k$ during training—provides unusually clean evidence about the temporal order of feature acquisition in transformers, reinforcing and extending similar findings in other settings (Rende et al., 2024; Cagnetta & Wyart, 2024). The out-of-sample calibration result (transformers match mismatched BP$_k$ predictions when tested on data with different $k$) is particularly compelling because it cannot be explained by simple memorization of the training distribution.

## Suggestions

- Moderate the strongest claims: change "evidence of an equivalence in computation to the exact inference algorithm" to "evidence consistent with the transformer approximating the Bayes-optimal posterior, with patterns aligned to but not confirmed as BP," or supplement with interventional evidence that directly tests the BP hypothesis.

- Include variance across random seeds and at least one additional grammar realization in the main text; add quantitative error bars to the KL divergence curves and calibration metrics.

- Include at least one experiment with $\ell \geq 6$ to demonstrate that the sequential learning and layer-hierarchy correspondence persist beyond the very shallow setting of 4 levels.

- Report numerical values for key calibration metrics (mean KL divergence at convergence, Spearman correlation, etc.) in the main text rather than relying solely on visual comparison.

## Score and Decision

**Calibration context:** I compared against papers studying transformers on synthetic structured/hierarchical data with interpretability claims. "How LMs Learn CFGs" (scores 6,6,5,3) was rejected despite showing clear mechanistic evidence, with reviewers highlighting the gap between "representing" and "using" structural information. "Interpretability Illusions" (scores 3,6,6,8,5) was rejected for showing that simplified models can be misleading about OOD behavior. "Transformers Struggle to Learn to Search" (scores 8,8,6,5) was accepted with stronger causal/mechanistic analysis. "JoMA" (scores 5,6,6,6) was accepted as poster with theoretical analysis but limited practical implications. "Understanding Addition in Transformers" (scores 8,3,8,3) was accepted as poster despite polarizing reviews, with genuine mechanistic insights on a simple task.

This paper has genuine and interesting contributions—the filtering framework, the calibration results, and the learning dynamics analysis are all solid. However, the central mechanistic claim (BP implementation) overreaches the evidence, which is indirect (output matching, attention patterns, probing). The scale limitation ($\ell=4$, $q=4$, single grammar) further limits confidence in the generality of the findings. The paper sits in a similar space to "How LMs Learn CFGs" but with a cleaner experimental design; it has a stronger framework but similar limitations in bridging from "consistent with" to "implements."

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>