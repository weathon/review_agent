## Summary

This paper introduces a hierarchical filtering procedure for tree-based generative models of discrete sequences, where a parameter $k$ controls the depth of hierarchical correlations, enabling controlled experiments on how encoder-only transformers learn structured data. The authors demonstrate that transformers trained on root classification and MLM tasks match Belief Propagation (BP) performance not only in accuracy but in full probabilistic calibration, both in-sample and out-of-sample. They show that hierarchical correlations are learned sequentially during training (aligning with BP$_k$ oracles of decreasing $k$), that attention maps reflect the tree structure, and that probing reveals ancestor information distributed sequentially across encoder layers. An existence proof shows BP can in principle be embedded in a single-head transformer with $\ell$ layers.

## Strengths

- **Elegant and controllable experimental framework.** The filtering parameter $k$ is a genuine methodological contribution, allowing causal investigation of what the transformer has learned at each correlation depth. The ability to train on one $k$ and test on another (Figs. 3–5) provides unusually clean experimental leverage.

- **Strong calibration evidence.** The finding that transformer logits, after softmax, closely match BP marginal distributions—even on uniformly sampled out-of-distribution inputs (Fig. 1b, bottom)—goes well beyond accuracy matching and is a meaningful result. In the $k > 0$ case, the network is never trained with soft labels, yet produces calibrated posteriors, which is a non-trivial observation supporting computational similarity to BP.

- **Compelling learning dynamics.** The sequential alignment of transformer predictions with BP$_k$ oracles of decreasing $k$ during training (Figs. 1c–d, 4, 5), and the resulting staircase behavior, is a clean and interpretable finding about how transformers progressively incorporate longer-range structure.

- **Probing analysis provides complementary mechanistic evidence.** The fact that ancestor information at depth $k$ is best recovered from transformer layer $k$ (Fig. 7), with a clear degradation pattern, is directly consistent with hierarchical computation distributed across layers. The controls in Appendix D.7 (position subsampling, relative accuracy comparisons) address some potential confounds.

- **The BP existence proof in Appendix E** is a useful theoretical contribution, establishing that the architecture has sufficient capacity to implement BP in a natural way, which legitimizes the mechanistic hypothesis.

## Weaknesses

### Major

- **Over-claiming about mechanistic equivalence to BP.** The paper repeatedly states that transformers implement or approximate "the exact inference algorithm" (Abstract, Sec. 3.2, Sec. 4 title). The actual evidence—matching output distributions, attention patterns, and decodable ancestor information—is **consistent with** BP but does not uniquely identify it. For a low-dimensional function mapping $2^\ell = 16$ symbols onto $q = 4$ probabilities, many distinct algorithms produce identical marginals. As the paper itself acknowledges, the BP embedding (Appendix E) uses specialized disentangled embeddings in $d = q(q+2)+\ell$ dimensions and $\mathcal{O}(q^2)$ memory slots per token—architectural choices not enforced in training. No causal intervention experiments (e.g., patching activations or ablating specific attention patterns to disrupt putative message-passing) are performed to test whether the model *relies* on BP-like computation. The paper's language should consistently describe the BP hypothesis as well-supported but not definitively established, rather than treating it as confirmed.

- **Probing reveals representation, not causal mechanism.** The ancestor-probing experiments show that information about level-$k$ ancestors is *decodable* from layer-$k$ representations. However, decodability does not establish that the model *uses* this information in a BP-specific way—it could be a byproduct of any sufficiently expressive model that aggregates context incrementally. As acknowledged in related interpretability literature, representing features does not imply using them in a specific algorithm. The relative-comparison control (different layers decode different depths) is suggestive but insufficient to rule out, e.g., a model that simply accumulates all context by depth without performing message-passing operations on it. This weakness is partially addressable and does not invalidate the probing results, but it limits how much mechanistic weight they can bear.

- **Limited experimental scope.** Nearly all results use $\ell = 4$ (sequence length 16), $q = 4$, and a single random grammar realization. The non-overlapping (deterministic parent recovery) condition at $k = 0$ makes root classification trivially solvable, and even for $k > 0$ the inference problem is small-dimensional. Appendix D.2 claims qualitative robustness across grammars, but no quantitative analysis or visualization of attention/probing patterns for alternative grammars is presented. The central claims about "how transformers learn structured data" extend far beyond this narrow setting, and the paper does not systematically vary $q$, $\ell$, grammar ambiguity, or architecture (multi-head attention, decoder-only) to test generalizability.

### Minor

- **The "sequential discovery of hierarchy" narrative, while appealing, has plausible alternative explanations.** The D$_{KL}$ alignment with BP$_k$ oracles of decreasing $k$ could reflect generic SGD dynamics—learning simpler, higher-SNR features first—rather than a mechanistically specific discovery of hierarchical levels. The paper cites related work on this (Refinetti et al., 2023; Rende et al., 2024) but does not articulate how those theoretical frameworks explain the specific alignment pattern. This does not undermine the empirical observation, which is interesting, but the interpretive claim about mechanism could be more cautious.

- **$n_L = \ell$ is assumed throughout the main text.** The matching of transformer depth to tree depth is central to the layer-by-layer interpretation, but the consequences of mismatch ($n_L < \ell$ or $n_L > \ell$) are relegated to Appendix D.1. Understanding what breaks and what persists under mismatch would significantly clarify whether the structural correspondence is inherent or coincidental.

## Nice-to-Haves

- Causal intervention experiments (activation patching, attention pattern ablation) targeting specific BP-like message-passing operations, which would be the single most important test of the mechanistic hypothesis.
- Experiments on grammars with ambiguous (overlapping) transition tensors, which would test whether the findings extend to the more realistic setting where parent recovery is probabilistic rather than deterministic.
- Experiments with multi-head attention and decoder-only architectures to assess whether the findings are transformer-specific or architecture-dependent.
- Individual-sequence attention maps rather than averages, to confirm that the observed block structure is consistent across inputs and not an averaging artifact.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *Insufficient reproducibility (undisclosed hyperparameters, random seeds, etc.)*: The paper provides a reproducibility statement and specifies key details; this is a standard experimental paper, not a systems paper. Demanding complete training logs is impractical and outside community norms.

- *Missing comparison to alternative architectures (MLPs, RNNs)*: While interesting, the paper's stated scope is understanding transformers specifically on this data model, not benchmarking across architectures. This would strengthen the paper but is not a core flaw within the stated scope.

- *Title claims about "how" when only analyzing after learning*: The training dynamics analysis (Figs. 4, 5, 1c–d) does address learning over time, making this criticism only partially valid. The D$_{KL}$ alignment curves during training represent genuine evidence about the learning process, not just post-hoc analysis.

- *Probes may overfit*: The paper explicitly addresses this concern in Appendix D.7 by showing that relative comparisons across layers are fair (same training data, same probe architecture) and that subsampling positions yields consistent results. While not a fully rigorous causal test, the design is reasonable by community standards.

- *Formatting/presentation issues*: These are style nitpicks and not substantive weaknesses.

## Novel Insights

The hierarchical filtering parameter $k$ and the resulting train-on-$k$-test-on-$k'$$ experimental paradigm is a genuinely novel contribution to the mechanistic interpretability toolkit. The most insightful finding is not merely that transformers match BP accuracy, but that they *calibrate* their output distributions to match BP marginals—even on distributions they were never trained on, and even with hard-label training that provides no direct gradient signal toward soft probabilistic targets. This calibration specificity (particularly the mismatched case where BP$_{k_{\text{train}}}$ is not the optimal oracle yet is still matched) is unusually diagnostic of algorithmic similarity and goes beyond what prior interpretability work on synthetic grammars has demonstrated.

## Suggestions

- **Moderate the mechanistic claims**: Replace language like "implements exact inference" and "equivalence in computation" with "is consistent with BP-like computation" or "approximates BP at the level of output distributions, with internal representations suggestive of hierarchical message-passing." The calibration and mismatched-filtering results are strong evidence, but they do not uniquely identify the algorithm.

- **Add a quantitative metric for attention–tree alignment**: Measure, e.g., the fraction of attention mass falling on true tree-relevant token pairs vs. random, or compute mutual information between attention weights and tree distance. This would transform the visual argument of Fig. 6 into something falsifiable.

- **Show results for at least one alternative grammar realization in the main text**: Even a single additional grammar with its attention maps and probing curves would substantially address the generalizability concern without requiring a full sweep.

## Score and Decision

I calibrated against the following papers from the review finder:
- **J6qrIjTzoM** (Interpretability of LMs on CFGs): scores 6/8/3/8, average ~6.25, Reject — similar mechanistic overclaim issues on synthetic data, but less clean experimental framework than the current paper.
- **qnbLGV9oFL** (How LMs Learn CFGs): scores 6/6/5/3, average ~5, Withdrawn/Reject — weaker evidence base than current paper.
- **0GzqVqCKns** (Probing Latent Hierarchical Structure via Diffusion): scores 6/8/6/6, average ~6.5, Accept Poster — similar simplified model setting, interesting but limited-scope findings.
- **rUC7tHecSQ** (Stacked Attention Heads mechanism): scores 5/6/8, average ~6.3, Accept Poster — reverse-engineered attention mechanism with strong assumptions, comparable in depth.
- **aN4Jf6Cx69** (Mechanistic basis of abrupt learning): scores 8/8/10/10, average ~9, Accept Oral — much stronger theoretical grounding and broader scope than current paper.

The current paper has a distinctly better experimental framework than J6qrIjTzoM and qnbLGV9oFL, with the filtering paradigm providing real causal leverage. Its calibration results are genuinely novel and well-demonstrated. However, it shares the mechanistic overclaim problem seen in weaker CFG papers, and its scope is narrowly limited to small fixed-tree models. Relative to 0GzqVqCKns and rUC7tHecSQ (accepted posters with scores ~6–7), this paper has comparable strengths (novel framework, interesting empirical findings) and comparable weaknesses (limited scope, suggestive but not conclusive mechanistic claims). The overclaiming is more severe than in those papers, but the experimental methodology is stronger.

I place this paper slightly above the median of the comparable accepted posters.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>