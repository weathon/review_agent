## Summary
This paper proposes a Cost-Sensitive Loss (CSL) function for long-tailed classification that dynamically adjusts per-class weights based on two signals: a "semantic scale" (average magnitude of cached feature vectors) and per-class entropy (measuring feature-learning difficulty). A scalar adjustment term loosely inspired by reinforcement learning compares epoch-over-epoch performance to modulate the overall loss. The framework is evaluated on CIFAR-10/100-LT, ImageNet-LT, and Tiny ImageNet against re-weighting and module-improvement baselines.

---

## Strengths

- **Entropy as a proxy for feature-learning difficulty:** Rather than assigning class weights purely from sample counts, the paper's use of per-class predictive entropy (H_i) to distinguish Easy-To-Learn (ETL) vs. Difficult-To-Learn (DTL) classes is a specific and differentiated contribution. Weighting by both feature magnitude and entropy — rather than one or the other — is not a standard combination in the CSL literature, and Figure 2 illustrates that γ_i values do evolve meaningfully over training, differentiating e.g. class 91 ("bicycle") from head classes.

- **Dynamic γ_i design with empirical motivation:** Figures 1 and 2 provide a concrete, class-specific visualization of how semantic scale and gamma values change over epochs, supporting the qualitative claim that the weighting scheme shifts focus adaptively. The specific example of class 91 plateauing in feature storage and being subsequently de-prioritized illustrates the intended behavior in a testable way.

- **Competitive ImageNet-LT result vs. established CSL baselines:** On ImageNet-LT (Table 3), the method achieves 49.3% vs. the strongest CSL baseline (Weighted Softmax, 49.1%), suggesting the approach is at least competitive on a real-world large-scale benchmark.

---

## Weaknesses

- **Critical formula inconsistency for γ_i — reproducibility-breaking.** Two incompatible definitions of the core parameter γ_i appear in the paper. The narrative text (Section 3, after Figure 3) gives:
  > γ_i = S_i / [(1+ε)(H_i · max(S_i))]
  
  Algorithm 1 (Step 6) gives:
  > γ_i ← S_i / [(1+ε − α + max(S) · H_i)]
  
  The denominator structure differs (product vs. sum), a mysterious parameter α appears in the algorithm but is never defined anywhere in the text, and max(S_i) vs. max(S) are used inconsistently. A reader cannot implement this method from the paper as written. This is a blocking reproducibility issue.

- **The "reinforcement learning" framing is a material misrepresentation.** The abstract, introduction, and contributions all claim to "leverage reinforcement learning." The actual mechanism is: compare loss (or accuracy) at epoch t to epoch t−1, then add a scalar constant ("reward value k") to the loss if performance improved. There is no MDP formulation, no state/action/reward space, no policy, and no Bellman update. This is a curriculum scheduling heuristic, and calling it RL will mislead readers and is unlikely to survive ICLR scrutiny. The paper should accurately describe this as epoch-wise dynamic curriculum adjustment.

- **The reinforcement term has no concrete formula.** The reward value 'k' and the "reinforcement_term" are mentioned qualitatively throughout the paper but never appear in any equation or algorithm step. No update rule is provided, and the algorithm (Step 8) writes "reinforcement" as an opaque constant. Without knowing how this term is computed and updated, the method cannot be replicated.

- **Implausibly large and unexplained 13-point gap on CIFAR-100 (p=200).** Table 2 reports CSL-Ours at 49.13% vs. the next best competitor (Focal+CB) at 35.62% — a gap of over 13 percentage points. This would be a landmark result on its own. No analysis, ablation, or error analysis is offered to explain or validate this gap. Without such justification, this result raises serious credibility concerns.

- **No ablation study.** The proposed method has at least three novel components: entropy-weighted γ_i, N_pred,i-based re-weighting, and the reinforcement term. No experiment isolates the contribution of any single component. Given the complexity of the loss formulation and the suspicious CIFAR-100 gap, it is impossible to know which component (if any) is driving performance gains.

- **No many/medium/few-shot accuracy breakdown.** The standard evaluation protocol in long-tail learning reports accuracy broken down by head, medium, and tail classes. Its complete absence makes it impossible to verify whether gains come from improved minority-class generalization (the stated goal) or from redistribution of errors across the distribution. A method that merely hurts head classes to lift overall accuracy would look identical in these tables.

- **Loss function design lacks justification.** The CSL denominator ∑_k(z_k − e_i)² + ε, where z_k are predicted logits/probabilities and e_i is a class label, is unusual. The paper's stated rationale — "to make it differentiable" (Section 3) — is incorrect, since CE is already differentiable. The actual effect of this denominator (which approaches 0 as the model improves on class i) is to make the CSL term blow up when predictions are accurate. While ε prevents division by zero, no analysis of the gradient behavior or the magnitude dynamics of this term is provided.

- **N_pred,i scope ambiguity.** The text (Section 2) states N_pred,i is computed "during validation," but Algorithm 1 places all computations inside the mini-batch training loop (Steps 13–24). It is unclear whether N_pred,i is accumulated over mini-batches, computed per mini-batch, or computed from a held-out validation pass. This ambiguity affects the method's description and its reproducibility.

- **Figure vs. text epoch count discrepancy.** The text (Section 2) states Figures 1 and 2 show "the first 80 epochs," but both figures display x-axes from 0 to 20. This is a direct factual inconsistency in the paper.

- **Weak baselines for a 2025 ICLR paper.** The comparison set is almost entirely from 2019–2021 (CE, Focal, CB Loss, LDAM, LDAM-DRW, IB). Logit Adjustment (Menon et al., 2021), a strong theoretically-grounded CSL baseline, is absent from all tables, as are other well-cited 2021–2023 CSL methods. For an ICLR 2025 submission, this is a substantive gap that makes the claimed "state-of-the-art" conclusion difficult to assess.

---

## Nice-to-Haves

- Report training time and peak memory overhead relative to CE and LDAM, since feature caching per class across epochs could be non-trivial at ImageNet scale.
- Compare with more recent CSL methods (2022–2024) to better position the contribution.
- Provide per-class or per-group t-SNE visualizations to verify that the loss actually improves minority-class feature separability, not just shifts predictions.
- Report results with multiple random seeds for small-gain settings (e.g., Tiny ImageNet, CIFAR-10 p=100) to give a sense of variance.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Introduction repeats the abstract verbatim"** — This is a writing style/formatting nitpick. Removed per scope rules.
- **"Awkward phrasing (motley of imbalances)"** — Pure style nitpick.
- **"Code link is broken"** — The references section provides an explicit anonymized GitHub URL (https://github.com/iclr-sub/csl). The "[iclr](#)" anchor in the body text appears to be a formatting conversion artifact, not an absence of a code release.
- **Figure captions appearing three times in parsed text** — Artifact of PDF-to-text conversion, not a flaw in the paper itself.
- **"Missing baselines cannot be reproduced"** — The harsh critic notes that LDAM, LDAM-DRW, etc. results can be reproduced at any imbalance ratio. While true, the absence of certain entries in tables is noted with "(−)" and the paper attributes this to prior work not reporting those conditions. This is a real limitation in comparisons but the failure to reproduce them is attributable to experimental scope, not intentional unfairness.
- **Claims about references not existing** — No such claims by reviewers were accepted. All referenced baselines are assumed to exist.

---

## Novel Insights

The most genuinely novel observation — if correctly implemented — is the use of per-class predictive entropy *in combination with* feature vector magnitude (semantic scale) to compute adaptive class weights. This goes beyond existing static re-weighting approaches by differentiating ETL vs. DTL classes regardless of sample count: a tail class that is easy to learn (low entropy, e.g., "airplane") should receive less weight relief than a tail class that is hard to learn (high entropy, e.g., "dog"), since the model may already have acquired sufficient discriminative features for it despite sample scarcity. Whether the implementation actually achieves this is undermined by the formula inconsistencies noted above.

---

## Suggestions

1. **Resolve the γ_i formula inconsistency immediately.** Unify the text formula and Algorithm 1, explicitly define α, and provide a single, unambiguous pseudocode that a reader can implement.
2. **Rename and reformulate the "reinforcement" component.** Provide a concrete equation defining the reinforcement_term as a function of epoch-over-epoch accuracy/loss delta. Rename it to "epoch-wise curriculum adjustment" or similar.
3. **Add a per-component ablation study.** Run four conditions: (a) CE only, (b) CE + entropy-weighted γ only, (c) CE + N_pred weighting only, (d) full CSL. This is essential for a paper making component-level claims.
4. **Investigate and explain the p=200 CIFAR-100 result.** Provide a breakdown by head/medium/tail classes, and include class-level predictions to make the 13-point gap credible.
5. **Add many/medium/few accuracy splits** to all main tables, consistent with standard practice in the field.
6. **Clarify the scope and computation of N_pred,i** (validation epoch-level vs. mini-batch training loop) with explicit pseudocode.

---

**Evaluation summary:**
- *Novelty:* Moderate — the entropy + semantic-scale combination is a specific and interesting idea, but the implementation and framing contain substantial misrepresentations.
- *Technical soundness:* Weak — two incompatible formulas for the key parameter, an undefined reinforcement term, and unjustified loss denominator collectively prevent reliable replication.
- *Empirical support:* Weak — no ablations, no per-split analysis, no multi-seed runs, and a suspicious 13-point gap that is left entirely unexplained.
- *Significance:* Limited until core reproducibility and credibility issues are resolved.
- *Clarity:* Poor — figure/text epoch inconsistency, formula discrepancies, and qualitative-only description of the key dynamic term impair comprehension.

MY FINAL SCORE: <pineapple>3.2</pineapple>