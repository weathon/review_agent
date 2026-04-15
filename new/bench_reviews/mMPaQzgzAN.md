Now let me search for calibration anchors in the human reviews.Now I have enough information to write the consolidated review.

---

## Summary

JumpReLU SAEs replace the standard ReLU activation in vanilla sparse autoencoders with a JumpReLU function (zeroing pre-activations below a learnable per-feature threshold θ), trained with a direct L0 sparsity penalty via straight-through estimators (STEs). The paper's key theoretical insight is that training with STEs is mathematically equivalent to estimating the gradient of the expected loss via kernel density estimation (KDE), providing principled grounding for what would otherwise be a heuristic trick. On Gemma 2 9B activations across 9 site/layer combinations, JumpReLU SAEs consistently outperform Gated SAEs and match or slightly exceed TopK SAEs on the sparsity–fidelity Pareto frontier, while manual and automated interpretability studies show no sacrifice in feature interpretability.

---

## Strengths

- **Elegant theoretical grounding for STEs via KDE (Section 4, Eqs. 10–12)**: The derivation that the batch-mean pseudo-gradient under STE training exactly recovers a KDE estimator of the true expected-loss gradient is the paper's most distinctive contribution. Unlike prior STE work (e.g., Yin et al., 2019), the authors explicitly close the loop to KDE, connecting a practical training trick to a well-studied estimation framework. This is a specific insight most SAE papers lack.

- **Comprehensive empirical evaluation across 9 site/layer combinations**: Results are reported for residual stream, MLP output, and attention output activations at layers 9, 20, and 31 of Gemma 2 9B—both in the main paper and appendices. The breadth is substantially larger than comparable SAE architecture papers (e.g., Switch SAEs evaluated only on GPT-2 Small layer 8).

- **Automated interpretability analysis is methodologically rigorous**: The logistic regression with confidence intervals (OR=1.27, 95% CI: 1.23–1.31 for JumpReLU vs Gated; OR=0.98, 95% CI: 0.95–1.01 vs TopK) and explicit controls for site/layer confounds represent a level of statistical care rare in this literature. The conclusion that JumpReLU features are as interpretable as TopK features is well-supported.

- **Direct L0 training sidesteps L1 shrinkage without auxiliary losses**: Unlike Gated SAEs (which require auxiliary terms and resampling) or TopK SAEs (which require partial sorting), JumpReLU SAEs use a single forward/backward pass with an elementwise activation, achieving simplicity advantages that are architecturally well-argued.

- **Honest self-assessment**: The authors call their contribution "a mild improvement over existing SAE training methodologies" (Section 7) and explicitly enumerate limitations including single-model evaluation, unresolved high-frequency features, and hyperparameter sensitivity. This transparency is commendable and makes the claims credible.

---

## Weaknesses

### Fatal
None.

### Major

- **No variance or significance analysis for the JumpReLU vs TopK gap** — Fig. 2 shows JumpReLU and TopK Pareto curves that are very close in many regimes. The paper provides no error bars, seed variation, or statistical tests. The claim "often slightly better than TopK" is therefore unsubstantiated; the gap could be within noise. This is especially problematic given that the paper's main practical recommendation over TopK rests on this marginal advantage plus the architectural simplicity argument.

- **Primary evaluation restricted to a single model family** — Gemma 2 9B is the sole main-paper evaluation target; Pythia 2.8B appears only in an appendix. Both are relatively small models. The paper acknowledges this explicitly, but it remains a genuine limitation: whether the JumpReLU advantage over TopK and Gated SAEs generalizes to different architectures, scales, or training distributions is not established. Given that SAEs are most impactful on frontier-scale models, this limits the strength of the generality claim.

### Minor

- **High-frequency feature problem is acknowledged but unresolved**: JumpReLU (and TopK) produce more features active on >10% of tokens than Gated SAEs (Fig. 4), and prior work confirms these are less interpretable. The paper flags this as future work, but the proportion among high-sparsity, high-fidelity SAEs is non-trivial. Crucially, the interpretability studies (manual and automated) sample features uniformly at random, so high-frequency features—the most likely problematic ones—are effectively excluded from interpretability evaluation. The claim that JumpReLU doesn't sacrifice interpretability is therefore not tested for the subset of features most likely to be problematic.

- **Manual interpretability study uses only in-group raters without inter-rater agreement statistics**: All five raters are authors or members of the same research group (Footnote 9), and no Fleiss' κ or similar agreement metric is reported. The automated study partially mitigates this, but the manual study's conclusions should be interpreted with caution.

- **Efficiency claims lack quantitative backing**: The paper argues JumpReLU is more efficient than Gated or TopK SAEs based on architectural reasoning (single pass, elementwise activation). No wall-clock time, tokens/sec, or peak memory measurements are provided. The STE backward pass machinery and threshold optimization could affect training dynamics in ways not captured by the architectural argument alone.

- **Bandwidth ε selection is ad hoc**: ε=0.001 is found by sweep rather than any principled method. The paper acknowledges "we suspect there are more principled ways to determine this parameter" (Footnote 5), but provides no sensitivity analysis across ε values or analysis of how ε should scale with batch size. Given that the KDE interpretation assigns ε the role of a bandwidth, the rich KDE bandwidth selection literature (cross-validation, plug-in methods) is directly applicable but not engaged.

- **Disentanglement evaluation is a single narrow task**: The sport-editing task (baseball→basketball for 50 athletes) is the sole disentanglement evaluation. The striking failure of Gated SAEs on this task (Fig. 5b) is unexplained and could reflect task-specific idiosyncrasies rather than a general limitation of Gated SAEs. One additional task from a different domain would substantially strengthen this evaluation.

### Trivial

- **Ablation study is appendix-only**: The ablation demonstrating both JumpReLU activation and L0 penalty are necessary (Appendix H.2) is a core mechanistic claim. Surfacing at least a summary table in the main paper would strengthen the causal argument.

---

## Nice-to-Haves

- Add confidence intervals / error bars to the sparsity-fidelity Pareto curves (multiple seeds) to rigorously establish whether JumpReLU's edge over TopK is statistically meaningful.
- Report wall-clock training time to target fidelity, tokens/sec, and peak memory under matched settings to back up the efficiency claim empirically.
- Implement automatic KDE bandwidth selection (e.g., Silverman's rule or cross-validation) as a drop-in replacement for the swept ε, and evaluate whether it achieves similar or better results.
- Include at least one external rater pool for the manual interpretability study, and report Fleiss' κ.
- Add a downstream evaluation (e.g., circuit-finding or steering accuracy) to bridge the gap between the proxy sparsity-fidelity metric and the interpretability tasks the introduction motivates.
- Evaluate on at least one additional model family (e.g., LLaMA or Mistral class) to support the generalization claim.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic: "State-of-the-art" framing is too strong** — The paper qualifies "state-of-the-art" with "on Gemma 2 9B activations" in the abstract. This is appropriately scoped and not an overclaim.

- **Harsh Critic: STE claim is only "partially principled"** — The paper derives the expected-loss gradient in Eq. (10) and shows in Appendix C that STE training recovers exactly the KDE estimator of that gradient (Eq. 12). The claim of "principled" is well-established relative to what prior STE work showed; the bandwidth-dependence caveat is explicitly acknowledged. This criticism would apply to any KDE-based gradient estimator and is not a paper-specific flaw.

- **Neutral Reviewer: Architecture vs. loss function contributions are insufficiently disentangled** — The paper notes in Section 6 that weight-shared Gated SAEs are mathematically equivalent to JumpReLU SAEs, and explicitly provides ablation evidence in Appendix H.2 that both the activation function and the L0 loss are necessary. The ablation adequately addresses this concern.

- **Human Finder: Feature absorption and polysemy not evaluated** — The paper explicitly scopes its evaluation to sparsity-fidelity, feature frequency, interpretability, and a disentanglement task. Feature absorption is an active open research question; criticizing its absence here is scope creep.

- **Human Finder (and Spark): Requesting larger model evaluation (frontier scale)** — While desirable, evaluating at 70B+ scale is impractical for a research submission and not a standard expectation in this community. Pythia 2.8B + Gemma 2 9B is a reasonable scope. This is better placed as a Nice-to-Have.

---

## Novel Insights

The most genuinely novel insight is the KDE reinterpretation of STE training: by treating the batch of pre-activations as samples from the feature activation distribution and the kernel function as a density estimator centered on the threshold, the paper unifies two superficially unrelated ideas (STE heuristics and KDE theory) into a coherent framework. This insight is not specific to SAEs—it applies to any piecewise-constant function trained with STEs—and may have broader applicability in quantization-aware training and discrete optimization. The observation that the bandwidth ε directly governs the bias-variance tradeoff of this gradient estimator gives practitioners principled intuition for tuning a parameter that was previously selected by trial and error.

---

## Suggestions

1. **Report error bars on Pareto curves**: Run JumpReLU and TopK SAEs with at least 3 random seeds at representative (L0, delta-LM-loss) operating points, and report variance. This directly tests whether the JumpReLU vs TopK advantage is significant or within noise.
2. **Surface the ablation in the main paper**: A small table comparing ReLU+L1, ReLU+L0, JumpReLU+L1, and JumpReLU+L0 at a representative operating point would make the mechanistic argument self-contained.
3. **Directly evaluate high-frequency features' interpretability**: Oversample high-frequency features in the interpretability study to directly test whether JumpReLU's excess of such features represents an interpretability cost.
4. **Provide training efficiency benchmarks**: Report tokens/sec and time-to-target-fidelity for all three architectures on matched hardware.
5. **Engage with KDE bandwidth selection literature**: At minimum, test Silverman's rule as a data-driven alternative to swept ε, and report how sensitive results are to ε across an order of magnitude.

---

## Score and Decision

**Calibration:**

- *TopK SAE paper (tcsZt9ZNKD, Oral, avg ~8.2)*: Much larger contribution — scaling to GPT-4 scale (16M latents), clean scaling laws, multiple new evaluation metrics, dead feature solutions at frontier scale. JumpReLU is narrower in scope and more incremental.
- *Switch SAE paper (k2ZVAzVeMP, Poster, avg ~7)*: Comparable type of contribution (new SAE architecture with Pareto improvement), but Switch evaluated only on GPT-2 Small layer 8 vs JumpReLU's 9 site/layer combinations on Gemma 2 9B. JumpReLU's theoretical grounding is stronger, empirical scope is broader, and the writing is more honest.
- *Decomposing Dark Matter paper (5IZfo98rqr, Reject, avg ~3.5)*: Rejected partly for thin results and single-context evaluation. JumpReLU has stronger empirical coverage and a cleaner theoretical contribution.
- *STE papers (3j72egd8q1, Reject, avg 5.25; 5pFV1FxG9d, Reject, avg 4.25)*: These STE-focused papers were rejected for narrow contributions and sensitivity issues. JumpReLU's theoretical use of STEs is supporting motivation for a practical architecture improvement, not the sole contribution.

**Assessment:** JumpReLU sits comfortably above the Switch SAE paper in quality (stronger theory, broader evaluation, more honest framing) and meaningfully below the TopK Oral (which made a bigger splash with scaling laws and frontier-model training). The primary weaknesses—no variance analysis for the TopK comparison, single model family—are real but do not invalidate the contribution. The paper is well-scoped, technically sound, and makes a genuine (if incremental) advance.

- Novelty: **Moderate-good** — JumpReLU + L0 combination is novel for SAEs; STE-KDE connection is genuinely insightful
- Technical soundness: **Good** — derivations appear correct; STE formulation is clean
- Empirical support: **Good within stated scope** — 9 site/layer combos, multiple metrics, but one primary model family and no variance analysis
- Significance: **Moderate** — useful for the mechanistic interpretability community; STE-KDE insight may generalize
- Clarity: **Good** — well-organized, honest about limitations, reproducibility supported by pseudocode

**Final Score: 7.0 — Accept (Poster)**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>