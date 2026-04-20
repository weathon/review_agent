Now let me do calibration searches to understand the scoring landscape.Now I have enough information to write the final review.

---

## Summary

This paper introduces Quantum Parameter Adaptation (QPA), a method that combines a parameterized quantum circuit (PQC) with a classical MLP mapping model to generate parameters for parameter-efficient fine-tuning (PEFT) methods—specifically LoRA, DoRA, Prefix-Tuning, and Feed-Forward Adapters—for LLMs. The central claim is that the high-dimensional Hilbert space of the PQC provides an efficient representation advantage, enabling polylogarithmic parameter scaling. Experiments on GPT-2 and Gemma-2 (2B) show that QPA reduces trainable parameters to 52.06% and 16.84% of standard LoRA, respectively, while reporting marginal perplexity improvements in text generation.

---

## Strengths

- **Scale of quantum parameter generation (Section 1; Table 2):** QPA targets Gemma-2's lm_head (0.52B parameters), approximately 1785× larger than the previous largest quantum parameter generation study (0.28M). This is a meaningful engineering achievement within the quantum parameter generation framework.

- **Batched parameter generation scheme (Section 3.2, Eq. 8):** The chunked generation design reduces required qubits from ⌈log₂ m⌉ to ⌈log₂(⌈m/n_mlp⌉)⌉, concretely cutting memory for quantum state storage by a factor of n_mlp. This is a principled practical contribution to the QML toolbox.

- **Practical qubit footprint (Figure 4a):** All tested QPA configurations require only 4–11 qubits, comfortably within near-term quantum hardware capabilities, making the approach credible for future real-hardware deployment.

- **Inference-time quantum independence (Section 1, Figure 1c):** By restricting quantum computation to training only, QPA neatly avoids both the data-encoding challenge and the inference-stage quantum hardware dependency that plague conventional QML approaches.

---

## Weaknesses

### Fatal

None that outright invalidate the paper's logical framework, but the combination of Major issues below severely undermine the empirical core.

### Major

- **Suspicious and internally inconsistent perplexity metric (Table 2, Figures 2–3).** The paper reports testing perplexity of ~1.595 for LoRA-GPT-2 and ~1.418 for LoRA-Gemma-2 on WikiText-2. Standard token-level perplexity for GPT-2 on WikiText-2 is approximately 18–30 before fine-tuning; fine-tuning the lm_head cannot plausibly yield values in the 1.4–1.6 range on the test set without a metric redefinition, which the paper does not provide. Separately, Table 2 shows DoRA with 4.36% parameters achieving perplexity 5.003 versus LoRA with 0.52% parameters achieving 1.595—a result that defies expectation, since DoRA at higher rank should be more expressive, not 3× worse. These internal contradictions raise serious doubts about whether the reported "perplexity" is standard token-level perplexity, and if not, what it measures. Because every headline result (the claimed 0.75% and 0.07% improvements) rests on this metric, the empirical core of the paper cannot be reliably interpreted in its current state.

- **No classical hypernetwork ablation—the central quantum claim is untested (Sections 3.1–3.2, 4).** The full QPA system consists of a PQC (small parameter count) feeding into a classical MLP mapping model $\tilde{G}_\mathbf{b}$ with hidden dimensions [32, 64, 128, 128, 64, 32, n_mlp]. For large n_mlp (e.g., n_mlp = 65,536), the final MLP layer alone has ~2M parameters—far more than the PQC parameters θ. The paper never compares QPA against a matched classical baseline—e.g., a small classical MLP or a learned embedding of the same total parameter count replacing the PQC—feeding into the same mapping model. The hypothesis stated in Section 4 that "the high-dimensional Hilbert space enables efficient representation for adaptation" is entirely untested. Any observed benefit could be attributable entirely to the mapping model $\tilde{G}_\mathbf{b}$ alone.

- **Single final-layer experimental setup is unrepresentative of real PEFT (Section 4).** The paper freezes all transformer layers and fine-tunes only the lm_head ("we simplify the PEFT setup by freezing all layers of Gemma-2 and GPT-2, and fine-tuning only the final linear layer"). This is not how LoRA, DoRA, PT, or FFA are used in practice—standard usage distributes updates across all attention and MLP layers. The parameter efficiency comparison is therefore between two parameterization schemes for one projection layer, not a demonstration of QPA's viability as a practical PEFT method for LLMs. No downstream classification or question-answering task is evaluated; WikiText-2 language modeling perplexity (even if the metric is correctly defined) is a weak proxy for PEFT's actual utility.

### Minor

- **Gemma-2 gains require L >> 8, but main experiments use L=8 (Figure 4d).** Figure 4(d) shows that for Gemma-2, QPA with L=8 performs comparably to LoRA baseline, and only clearly outperforms LoRA when L > 64. The headline Table 2 result (0.07% perplexity improvement at 16.84% of parameters) is achieved by sweeping n_mlp at L=8; but the ablation suggests this benefit is configuration-specific and not robustly demonstrated across L settings. The claimed advantage is therefore fragile.

- **Polylogarithmic parameter claim conflates PQC and total system (Section 3.1).** The claim that QPA achieves polylogarithmic parameter scaling applies to θ (the PQC parameters), not to the total trainable system |θ| + |b|. The MLP mapping model's parameter count scales with n_mlp, which can dominate for large chunk sizes. The paper partially obscures this distinction when making efficiency claims.

### Trivial

- The paper applies QPA to only one dataset (WikiText-2) and one task type (text generation). This is very narrow.

---

## Nice-to-Haves

- Replace the PQC with a classical MLP or learned embedding of equal parameter count (keeping the mapping model identical) to test whether the quantum component provides meaningful benefit beyond a classical hypernetwork.
- Apply QPA-LoRA across all attention layers of GPT-2 or Gemma-2 and evaluate on a standard downstream task (e.g., GLUE or commonsense QA), as is standard for PEFT evaluations.
- Clarify the perplexity computation in detail (tokenization, stride, aggregation method) and explain why values fall in the range 1.4–5.5 rather than the 10–30 range expected for these models on WikiText-2 test sets.
- Provide wall-clock training time comparisons, since quantum circuit simulation at each forward pass may make QPA slower than LoRA even if the parameter count is smaller.
- Move at least one result from the noise model analysis (Appendix G) into the main text, since the quantum-centric supercomputing motivation requires some demonstration under realistic noise.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Strength Finder Strength 1 (Parameter reduction percentages):** Kept in condensed form in the review, but the specific improvements (0.75%, 0.07%) are contingent on the perplexity metric being valid, which is disputed by the Major weakness above. The raw numbers are cited without endorsing their interpretation.

- **Harsh Critic Section on polylogarithmic scaling (Section 3.1):** Partially kept as a minor weakness. The harshest version ("the polylogarithmic claim applies only to |θ| and the paper conflates these throughout") is weakened—Section 3.1 does explicitly discuss the mapping model separately, and the conflation is more of a presentation imprecision than a methodological error.

- **Harsh Critic "Baseline configuration not reported":** Removed as a reproducibility nitpick about undisclosed hyperparameters; the paper states full hyperparameters are in Appendix C.

- **Harsh Critic "Convergence/trainability deferred":** Converted to the nice-to-have tier. Deferring theoretical analysis to future work is common in empirical systems papers and does not invalidate the experimental claims in principle.

---

## Novel Insights

The reviewers surface one genuinely important structural tension: QPA is architecturally a classical hypernetwork where the quantum circuit acts as a context encoder, and the potentially dominant component (the MLP mapping model) is entirely classical. The Hilbert space compression argument is elegant in theory, but without isolating the PQC's contribution from the mapping model's, the system cannot be distinguished from a standard small-MLP hypernetwork generating PEFT parameters. This is not a critique that the reviewers invented—it follows directly from the architecture described in Table 1 and Section 3.2. Resolving it (or confirming that the quantum component is dominant) would substantially change the paper's significance.

---

## Suggestions

1. **Priority:** Provide a clear specification of the perplexity metric, with reproduction of a standard baseline (e.g., unmodified GPT-2 on WikiText-2 test set) to calibrate numbers, and explain the DoRA anomaly in Table 2.
2. **Priority:** Add a matched classical hypernetwork ablation (same MLP architecture, same parameter count, classical context vector replacing PQC output) to test whether the quantum circuit contributes beyond the mapping model.
3. Broaden experiments to multi-layer fine-tuning on at least one downstream classification or QA task.
4. Report perplexity at both L=8 and L=64+ for both models to make the depth effect transparent in Table 2.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| dgR6i4TSng (Quantum-PEFT) | Quantum-inspired PEFT for LLMs | 6,6,6,6 | Accept–Poster |
| dfEuojp0rX (VQLS-QSVM) | Quantum ML on toy data | 1,5,3,3 | Reject |
| 6cDEcJsE1Y (VQC for ML) | Quantum circuits for ML | 3,3,3 | Withdrawn/Reject |
| l3oE5vBjDs (NEAT-PEFT) | PEFT with questionable experiments | 3,6,5,6,5 | Reject |
| bYsieh8LE2 (ULoRA) | LoRA extension with weak validation | 3,5,3 | Withdrawn/Reject |

The closest anchor is **Quantum-PEFT** (6,6,6,6), which is a closely analogous paper applying quantum/quantum-inspired unitary parameterization to PEFT. That paper evaluated on real multi-layer benchmarks (GLUE, vision), used standard metrics without ambiguity, and demonstrated results across multiple architectures. The paper under review falls meaningfully below that bar on all three dimensions: the perplexity metric appears non-standard (or possibly incorrect), experiments cover only the lm_head in a single-layer artificial setup, and there is no ablation validating the quantum contribution. The perplexity inconsistency alone (1.4–1.6 when the standard baseline is 18–30 for GPT-2 on WikiText-2 test) calls all quantitative results into question and is more severe than any weakness in Quantum-PEFT.

Weak quantum ML papers (1–3) had more fundamental problems (trivial datasets, no quantum advantage even in theory). This paper has a genuine novel idea and real engineering at scale, placing it above those. But the experimental validation issues are substantive enough to require revision before the contributions can be reliably assessed.

**Final score: 3.5** — below the Quantum-PEFT anchor (6) due to the metric reliability issue and artificially constrained experimental setup, and above the clearly weak QML papers (1–3) because the concept and scale contribution are real.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>