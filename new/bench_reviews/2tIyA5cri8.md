## Summary
This paper investigates whether temporal difference (TD) learning emerges in LLMs when solving reinforcement learning problems purely in-context. The authors train Sparse Autoencoders (SAEs) on Llama 3 70B's residual stream across three distinct tasks (Two-Step, Grid World, Graph Prediction), identify SAE latents correlating with TD errors and Q-values, and perform zero-ablation interventions showing that deactivating these latents degrades both behavioral performance and downstream representations. The work establishes a methodology for mechanistic analysis of in-context RL via SAEs and provides evidence that TD-like computations emerge as a byproduct of next-token prediction.

## Strengths
- **Multi-task convergent evidence across diverse MDP structures.** The paper demonstrates TD-correlated features across three substantively different tasks: a simple binary-choice RL task, a 5×5 grid navigation task with four actions, and a reward-free graph prediction task requiring successor representation learning. This empirical breadth is stronger than single-task mechanistic studies. The SR vs. transition matrix comparison (Section 5, Figures 5D-E) is particularly compelling—max correlation with SR TD errors (r=0.60) substantially exceeds transition-matrix surprise signals (r=0.18), suggesting the model tracks TD structure rather than mere transition frequency.

- **Multiple lines of convergent causal evidence.** Beyond identifying correlations, the paper employs three distinct intervention strategies: (1) zero-ablation of the highest-TD-correlated latent degrades performance (Section 3, Figure 2E), (2) zero-ablation of the lowest-correlated latent in the same block produces essentially no performance drop (Figure 2F), and (3) negative scaling by −10 produces graded reductions in downstream Q-value and TD-error correlations (Figures 2G-H, 4E-F). This triangulation goes beyond purely correlational SAE interpretability.

- **Control analyses that rule out simpler explanations.** The random-reward control in Grid World (Figure 4A)—same action sequences with shuffled rewards—demonstrates that Llama's predictions depend on reward structure, not just action-sequence pattern matching. The Two-Step task also includes reward-disruption and transition-disruption variations (Figure 3) where the SAE feature qualitatively tracks the corresponding TD error changes.

- **Clear methodological pipeline for future work.** The paper's systematic approach—prompting, residual-stream recording, SAE training, feature identification, and targeted intervention—is clearly documented (Figure 1D) and directly reusable for studying other in-context learning phenomena with SAEs.

## Weaknesses

### Fatal
None

### Major
- **Causal claims slightly outpace the evidence.** The abstract and conclusion state that the paper "verif[ies]" causal involvement of TD latents, but the zero-ablation methodology provides suggestive rather than definitive causal proof. The identified TD features could be partially entangled with adjacent representations in the SAE decomposition, and zeroing a single latent in an imperfectly disentangled autoencoder may degrade performance through collateral representational damage rather than targeted removal of the TD computation. While the "lowest-correlation" control (Figure 2F) and the negative-scaling experiment (Figure 2G-H) partially address this, a truly matched control—ablation of a direction with identical activation sparsity, magnitude, and reconstruction error—would be needed to isolate causal TD computation from generic representational disruption. This is a known limitation in the SAE literature, but the paper's language about "verifying" causal involvement should be tempered.

### Minor
- **Behavioral model comparisons lack statistical testing and alternative baselines.** The paper claims Llama's behavior is "best described by a Q-learning algorithm" (Section 3, lines 124–125) based on NLL differences between a Q-learning model (2729), myopic model (2864), and repetition model (5745). While the repetition model's failure is dramatic and the direction is informative, the 135-NLL gap between Q-learning and myopic has no reported significance test. Additionally, no comparison against standard in-context pattern-matching baselines (e.g., n-gram models, simple Bayesian state-updaters) is provided, leaving open the possibility that a simpler mechanism could explain the observed behavior. A bootstrapped NLL comparison or information-theoretic model selection criterion (e.g., BIC) would strengthen this foundational premise.

- **The Grid World multi-feature ablation lacks a matched multi-feature control.** In Section 4, four TD latents across four blocks are ablated simultaneously (lines 161–162) to elicit stronger effects. The control condition ablates four lowest-correlated latents, but since the four TD latents are selected to each have r ≥ 0.75 with TD error, they jointly carry more predictive information and likely higher combined activation energy than four random low-correlation features. The performance gap could reflect the aggregate statistical importance of the ablated latents rather than their specific TD content. This echoes the single-latent control issue but is amplified for multi-feature ablation.

### Trivial
- **The abstract's language ("verifies that these representations are indeed causally involved") is stronger than warranted.** A softer phrasing such as "provides evidence consistent with" or "suggests causal involvement" would better reflect the methodological limitations.

## Nice-to-Haves
- Report reconstruction fidelity metrics (e.g., residual stream MSE or cosine similarity) before and after lesioning across tasks to quantify the collateral impact of SAE zero-ablation.
- Perform path-patching analysis from reward/state input tokens to the candidate TD latents to map the information flow more precisely.
- Test whether the identified TD-correlated features transfer to out-of-distribution MDPs held out during SAE training, which would distinguish genuine TD mechanisms from task-overfitting artifacts.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **The critic's claim that "correlation with externally derived TD errors is inherently circular"** — The paper goes beyond pure correlation: it uses random-reward controls (Figure 4A), SR vs. transition matrix comparisons (Figures 5D-E), negative scaling with graded downstream effects (Figures 2G-H), and causal ablation. The critic treats all correlational proxy evidence as equally uninformative, ignoring the convergence across controls. The concern is a general epistemic limitation of all mechanistic interpretability using correlational proxies, not a paper-specific flaw.

- **The claim that the Grid World lesion targets block 64 while "SR correlations peak around blocks 30–40"** — This appears to misread the figures. The paper (line 192) explicitly states SR representations "build gradually and peak late in the model," and Figure 5E shows TD error correlations are highest in blocks 50–70, making block 64 the appropriate intervention target. This is a reviewer error.

- **The fixed γ (discount factor) without sensitivity analysis** — A standard choice in RL literature; testing sensitivity to γ would not meaningfully change the paper's core claims about whether TD-like representations exist.

- **SAE regularization β = 1e-05 presented without ablation** — This is a standard k-sparse / L1 regularization hyperparameter in the SAE literature (cf. Templeton et al., 2024; Gao et al., 2024). Ablating it would be a hyperparameter study, not a core methodological requirement.

- **"Missing related work" and "missing appendix" complaints** — Per the review rules, these should not be counted as weaknesses.

- **General concerns about the existence/release of cited models (Llama 3 70B, SAE frameworks)** — Per the hard rules, these are removed since the paper cites them.

## Novel insights
One genuinely novel observation synthesized from this paper and its calibration peers is that **SAE-based zero-ablation controls may systematically under-detect entanglement-related confounds when the ablated feature is highly active and the control feature is near-zero**. In this paper's case, the TD latent likely has substantial activation variance (since it correlates with a temporally dynamic prediction error signal), while the "lowest-correlation" control may be a near-dead feature with negligible contribution to the reconstruction. This asymmetry means the control's minimal effect (Figure 2F) could reflect lack of feature importance rather than lack of confounding, and the TD lesion's effect could partly reflect removing an active reconstruction component rather than specifically disrupting a TD computation. Future SAE intervention protocols should report per-feature activation norms and consider norm-matched random directions as controls alongside low-correlation features.

## Suggestions
1. Add activation statistics (mean activation, sparsity fraction, L2 norm) for both the TD latent and the lowest-correlation control for each task, and include a third control condition: zero-ablation of a random SAE direction in the same block, matched for the TD latent's activation variance and L2 norm.
2. Report a bootstrap or paired-test significance value for the NLL gap between Q-learning and myopic behavioral models (2729 vs. 2864).
3. Temper the abstract's causal language from "verifies... causally involved" to "provides convergent evidence for causal involvement" or similar.

---

## Calibration and Score

I compared this paper against several anchors from the human-review corpus:

- **High-scoring (8–10):** I4e82CIDxv (8,8,8,8, Oral) — SAE-based causal feature circuits for editing language models. Similar methodology (SAE zero-ablation, steering experiments) with stronger circuit analysis but a narrower scope. aN4Jf6Cx69 (8,8,10,10, Oral) — mechanistic basis of emergent ICL in simplified models, exceptionally clean synthetic experiments.

- **Borderline (5–6):** Pa1vr1Prww (6,5,6,3, Reject) — SAE circuits for ICL in Gemma-2B, criticized for causal claims outpacing evidence. This paper is cleaner than this anchor with more tasks and better structured controls.

- **Low-scoring (1–3):** Wxl0JMgDoU (3,3,3,1, Reject) — SAE interventions on a chess model, criticized for weak causal evidence, unclear presentation, narrow scope. YW79lAHBUF (3,3,6,3, Withdrawn) — LLMs as in-context RL learners but the RL signal is derived from ground-truth labels.

This paper sits above the low-scoring anchors (clearer writing, more tasks, better controls, genuinely novel TD-in-LLM finding) and slightly below the top-scoring Oral anchors (control methodology less rigorous than the feature-circuit papers, causal language slightly too strong). The novelty of discovering TD representations in a pretrained LLM across three tasks provides meaningful value to the community. The methodological concerns are genuine but not fatal or major—they are addressable and reflect known limitations in the broader SAE interpretability literature rather than paper-specific errors.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>