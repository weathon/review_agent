Now let me search for calibration papers to properly score this work.Now I have enough information to write the consolidated review. Let me synthesize everything.

---

## Summary

This paper proposes Two-Agent Nash Policy Optimization (TANPO), a self-play RLHF framework formulated as a two-player zero-sum game with KL-regularized objectives. The max-player minimizes a DPO-style loss on collected preference data, while the min-player minimizes a DPO-style loss augmented with an exploration bonus that encourages coverage of the max-player's distribution. The paper also presents Single-Agent Diversity-driven Policy Optimization (SADPO), a single-agent approximation that selects highest- and lowest-likelihood responses under the reference policy from K candidates. Theoretical guarantees of sublinear regret are provided under a Two-player Generalized Eluder Coefficient (TGEC) framework, and empirical results on Zephyr-7B-SFT with UltraFeedback show improvements over online DPO, Hybrid GSHF, and SELM baselines.

---

## Strengths

- **Novel two-player framing with a clear practical intuition.** The asymmetric objectives for max- and min-players (Eqs. 14–15) are non-trivial to derive and yield a concrete mechanistic hypothesis: max-player stays close to the reference policy, min-player actively diversifies. Equation-level derivation from the theoretical framework (Eq. 8 to Eq. 11/13) is provided with reference to Appendix C.

- **Formal regret guarantee under general function approximation.** Theorem 1 establishes sublinear regret under TGEC—a meaningful generalization of the Generalized Eluder Coefficient to two-player settings—and also specializes to the linear case (Corollary 1), making the theory both general and concrete.

- **Empirically competitive results across multiple benchmarks.** TANPO and SADPO outperform online DPO, Hybrid GSHF, and SELM on AlpacaEval 2.0 (LC win rate ~28% vs ~24%), MT-Bench, and academic benchmarks. The max-player vs. online DPO comparison (identical optimization objective, different training data from the two-agent process) is a useful ablation showing the value of diverse data generation.

- **Interesting overfitting-mitigation finding.** The continued-improvement curve over 6 iterations across two epochs (Figure 4) is a compelling practical observation, suggesting genuine data-efficiency gains from the diversity mechanism.

- **SADPO's practical simplicity.** The rejection-sampling heuristic requiring only K=4 samples from a single policy is directly implementable on top of standard DPO tooling without maintaining a full second model.

---

## Weaknesses

### Fatal
*None that independently invalidate all contributions.*

### Major

- **Theory-to-practice gap: the equivalence step is not self-contained in the main paper.** The core claim that the theoretical algorithm (Section 3.2) is "equivalent" to TANPO (Algorithm 1) rests on a max-min interchange that requires "Assumption 4 in Appendix C." The main text (Section 4.1) states only that "if the reward function class R satisfies Assumption 4, the minimax theorem applies," and then cites the appendix. Crucially, Section 5 qualifies Theorem 1's application to TANPO as holding "provided the reward function class R meets Assumption 4." Without any characterization of Assumption 4 in the main text, readers cannot assess whether this equivalence is substantive or vacuous for the LLM neural-policy setting. The paper's flagship claim—"provably efficient and practical"—hinges on this gap being closed, but the mechanism is hidden. This is a structural presentation issue that undermines confidence in the theoretical contribution as stated.

- **SADPO approximation is not formally justified and uses a different distribution than TANPO.** In TANPO's min-player objective (Eq. 15), the exploration bonus is $\mathbb{E}_{a \sim \pi^{t+1}}[\log \mu(a|x)]$—the expectation is under the *current max-player policy* $\pi^{t+1}$, which evolves over training. In SADPO (Eq. 16), the exploration bonus becomes $\mathbb{E}_{a \sim \pi_{\text{ref}}}[\log \pi(a|x)]$—the expectation is under the *static reference policy* $\pi_{\text{ref}}$. Similarly, SADPO selects responses based on $\pi_{\text{ref}}$ probability rather than $\pi^t$ probability. This is not a small implementation simplification; it changes the exploration objective categorically. The paper offers no formal bound on approximation error nor empirical evidence that SADPO's behavior tracks TANPO's as training progresses.

- **Limited experimental scope and no quantification of uncertainty.** All experiments use a single base model (Zephyr-7B-SFT) and a single dataset (UltraFeedback). No confidence intervals, standard errors, or repeated-run statistics are reported, despite some margins being small (e.g., SADPO 28.43% vs. TANPO 27.66% LC win rate on AlpacaEval; MT-Bench 7.33 vs. 7.47). Without uncertainty quantification, claimed superiority over baselines cannot be assessed rigorously.

### Minor

- **The overfitting claim is not comparatively established.** Figure 4 shows TANPO continues improving over 6 iterations (2 epochs). However, no analogous curves are presented for online DPO, SELM, or Hybrid GSHF under the same two-epoch same-data protocol. The claim that TANPO "mitigates overfitting" relative to baselines is therefore not supported—the paper only shows that TANPO doesn't degrade over its own training, not that it degrades less than alternatives.

- **Selective reporting in Table 1.** The main table reports TANPO based only on the min-player's performance (the better of the two), with full results deferred to an appendix table. Since TANPO is a two-agent method, the selective presentation creates an asymmetry. The authors note this choice in the text, but a combined or min-player + max-player aggregate metric in the main table would be more representative.

- **PairRM circularity.** PairRM is used both as the online AI feedback provider during training and as the judge in the pairwise win-rate evaluation (Figure 3). Evaluation using an independent judge (e.g., GPT-4-Turbo, which is used only in Figure 4) for all pairwise comparisons would reduce the risk of inflated results for methods trained with PairRM feedback.

- **Diversity metric conflated with the selection heuristic.** Figure 1's diversity measure—absolute difference in length-normalized $\log \pi_{\text{ref}}$ between the two responses—is structurally aligned with TANPO's own selection logic (min-player pushes away from $\pi_{\text{ref}}$). This is not independent evidence of semantic or informational diversity. Showing n-gram diversity, embedding-space spread, or reward model uncertainty would provide stronger validation.

### Trivial

- The regret performance metric ($V(\pi^t, \hat{\mu}) := \min_\mu V(\pi^t, \mu)$) is never empirically measured; all evaluation is downstream benchmarks. This is consistent with the literature, but noting the theoretical/empirical metric gap would be transparent.

---

## Nice-to-Haves

- **Test on an additional base model or dataset.** Even a brief experiment on a second base model (e.g., LLaMA-3-8B) or dataset would significantly strengthen generalizability claims.

- **Qualitative analysis of min-player outputs.** Showing concrete response examples from both players at early and late iterations would help readers understand whether min-player diversity is semantically meaningful (different reasoning paths, styles) or low-quality (high-entropy noise with low $\pi_{\text{ref}}$ mass).

- **Compute budget comparison.** TANPO trains two 7B models; SADPO samples K=4 per prompt. A brief wall-clock or GPU-hour comparison with baselines would help practitioners assess cost-benefit tradeoffs.

- **Ablation on exploration bonus.** Removing the exploration term from the min-player's objective (Eq. 15) would more directly test whether the bonus—not merely the two-agent data generation—drives improvements.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they were raised by reviewers but do not hold up against the paper's content.*

- **Harsh critic, Section 6.1 data reuse confusion.** The critic flagged inconsistency between the 3-iteration description in Section 6.1 and the 6-iteration Figure 4 description. Reading the paper, Section 6.1 describes the main experiment (3 iterations over 3 data portions), and Figure 4 explicitly describes a *second* experiment extending this with 3 more iterations on the same dataset. There is no inconsistency; this is clearly explained as a deliberate second-epoch analysis.

- **Harsh critic: "reporting only the min-player biases presentation as if unjustified."** The paper explicitly states "we report the results of TANPO based on the performance of the min-player" and provides full results (both players) in Table 2 in the appendix. This is a presentation choice, not suppression of data.

- **Spark reviewer: "missing SPIN comparison."** Requesting comparison with specific concurrent self-play works is outside scope to enforce; the paper already compares against Online DPO, Hybrid GSHF, and SELM, which are the most directly relevant online RLHF baselines.

- **Spark reviewer: "circularity in SADPO Eq. 16 expectation under π_ref vs. π^t."** This overlaps with the genuine SADPO weakness already documented above and does not need separate treatment as a reproducibility/implementation complaint.

- **Generic request for hyperparameter sensitivity analysis.** This is a nice-to-have for any method paper; the absence of sensitivity analysis for α, η, K does not uniquely undermine the core contribution.

---

## Novel Insights

The paper's most genuinely novel observation is the algebraic demonstration that, within the DPO reparametrization framework, the max-player's exploration objective cancels out, leaving only the MLE (DPO) loss, while the min-player retains a non-trivial exploration bonus targeting coverage of the max-player's distribution. This asymmetry provides a principled justification for why self-play with differentiated objectives should produce more diverse training data than single-agent online DPO—and the observation that max-player improvements over online DPO can be attributed purely to data quality (since their objectives are identical) is a clean ablation design that reviewers underappreciated. The TGEC extension to two-player settings is also a meaningful theoretical contribution, generalizing GEC from single-agent online RL to self-play RLHF.

---

## Suggestions

1. **Summarize Assumption 4 in the main text** (even in one or two sentences) so readers can evaluate whether the theoretical framework applies to the practical implementation without digging into the appendix. This single change would substantially strengthen the paper's headline contribution.

2. **Address the SADPO approximation gap explicitly**: note that π_ref ≈ π^t in early iterations and discuss whether this approximation degrades as training progresses, or provide an empirical comparison between TANPO and SADPO dynamics over iterations.

3. **Add analogous overfitting curves for at least one baseline** (online DPO) under the same two-epoch same-data protocol to substantiate the "mitigates overfitting" claim.

4. **Report uncertainty estimates** (or at minimum, describe the number of runs and explain why single-run evaluation is standard practice in this setting).

---

## Score and Decision

**Calibration:**

- **INPO** (Pujt3ADZgI.md): Accept Oral, scores 6,6,6,6. Two-player Nash RLHF with theory + experiments; simpler algorithm, cleaner derivation, stronger empirical results (42.6% LC win rate vs. 27-28% here, though different base model). No theory-practice gap concern.
- **COPO** (cfKZ5VrhXt.md): Accept Spotlight, scores 6,8,8. Similar theory+practice structure for online exploration in RLHF; cleaner exploration mechanism (UCB-count), comparably narrow experimental scope.
- **MPO** (PDnEDS244P.md): Accept Poster, scores 6,6,8,6. Last-iterate convergence self-play; comparable scope.
- **COMAL** (XuYd9IK7X4.md): Reject, scores 3,3,6. Two-player zero-sum for alignment; rejected due to weak theory understanding, unconvincing experiments, and minimal modifications to existing methods—clearly worse execution than TANPO/SADPO.

**Positioning:** The paper is meaningfully above COMAL (genuine derivation, competitive experiments, novel two-player framework with non-trivial exploration bonus). It is below INPO and COPO in theory-practice coherence (the equivalence derivation requires an unstated appendix assumption) and experimental scope (single model/dataset, no variance). The diversity mechanism is interesting but not rigorously validated. The SADPO approximation is weaker than claimed.

**Assessment:** The paper is in weak-accept territory. It makes a real contribution in formalizing and implementing asymmetric-objective self-play RLHF with theoretical backing, and the empirical results are competitive. However, the theory-practice gap (Assumption 4 not in main text), the unjustified SADPO approximation, and the narrow experimental scope prevent strong acceptance. Against the calibration set, the paper is closer to the 5.0–5.5 range—above COMAL (3) but below INPO/COPO (6–8).

**Originality:** Moderate-to-good. The asymmetric two-agent DPO framework is novel, though the individual components (DPO, Nash equilibrium RLHF, self-play, exploration bonuses) are all established.  
**Importance:** Moderate. Provably efficient self-play RLHF with practical implementations is an active and important problem.  
**Claims vs. evidence:** Partially supported. The "provably efficient and practical" headline is overstated given the appendix dependency for equivalence.  
**Soundness of experiments:** Fair. Results are competitive but narrow and lack uncertainty quantification.  
**Clarity:** Moderate. Main derivations are presented but the key Assumption 4 is hidden in the appendix.  
**Value to community:** Moderate-positive. The framework and SADPO heuristic are implementable and the theoretical analysis is a genuine step forward.

**Final Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>