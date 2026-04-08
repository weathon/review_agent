=== CALIBRATION EXAMPLE 18 ===

# Final Consolidated Review
## Summary

LLEOT addresses an overlooked privacy risk in Offsite Tuning (OT): existing emulators retain substantial inference capability, enabling data owners to misuse the model's intellectual property. The paper proposes Loss Landscape Elevation (LLE), which enforces a fixed loss margin between emulator and original model, theoretically guaranteeing both degraded emulator inference (via perplexity amplification) and preserved gradient alignment for prompt transfer. Combined with Collaborative Prompt Knowledge Distillation (CPKD), the framework enables soft prompts trained on privacy-protected emulators to transfer effectively to the original model, validated across three LLMs and four QA benchmarks.

## Strengths

- **Precise identification of the capability privacy gap in Offsite Tuning.** The paper makes a concrete, well-motivated distinction between parameter privacy (already addressed by OT) and capability privacy (overlooked), supported by Figure 1(c) showing that standard OT emulators achieve 70–89% of the original model's zero-shot accuracy. The proposed CPL metric (Eq. 1) provides a quantifiable measure for this previously unmeasured risk.

- **Principled theoretical connection between loss elevation, perplexity amplification, and gradient preservation.** Theorem 1 establishes that enforcing a constant loss margin $H$ simultaneously yields $PPL_E = e^H \cdot PPL_M$ (degrading inference) and $\nabla_P L_E = \nabla_P L_M$ (preserving optimization dynamics). This is a clean mathematical insight that directly motivates the method design, not a post-hoc justification.

- **Strong empirical privacy-utility trade-off.** LLEOT consistently achieves lower CPL than OT and CRaSh across all settings (Table 1), often reaching CPL values near or below the random initialization baseline, while matching or exceeding baseline accuracy. For instance, on Qwen2 DR=0.5, LLEOT achieves 34.20% accuracy vs. OT's 27.20%, with CPL of 46.52% vs. 70.02%.

- **CPKD is a sensible adaptation of knowledge distillation for continuous prompt spaces.** Traditional KD aligns models only at discrete token inputs; the Proxy Prompt Distillation Loss (Eq. 3) samples from a continuous distribution to cover the soft prompt representation space. Ablation in Table 3 confirms that removing $L_{PPD}$ causes the largest accuracy drop among the three loss terms, validating its necessity.

## Weaknesses

- **Theoretical guarantee rests on an idealization that is only approximately achievable in practice.** Theorem 1 derives gradient alignment ($\nabla_P L_E = \nabla_P L_M$) from the exact equality $L_E(P; x) = L_M(P; x) + H$ (Eq. 6). However, Eq. 7 reveals this is an optimization objective minimized over sampled proxy prompts and data batches, not an exact identity. Since the emulator is a lower-capacity subnetwork (constructed via LayerDrop), it cannot perfectly match the full model's loss landscape shifted by a constant across the entire continuous prompt space. There will be residual approximation error $\epsilon(P, x)$ such that $L_E \approx L_M + H + \epsilon$, and if $\nabla_P \epsilon \neq 0$, the gradient alignment guarantee degrades. The paper provides no bounds on this residual error nor discusses how layer removal affects the *shape* of the loss landscape beyond scalar loss values. This gap between the exact-equality assumption and the approximate-optimization reality is the most significant theoretical concern.

- **No empirical verification of gradient alignment.** The core claim—that LLE preserves gradient consistency between emulator and original model—is never directly measured. A natural and important experiment would be to compute cosine similarity between $\nabla_P L_E$ and $\nabla_P L_M$ across training steps or over the prompt parameter space. Without this, the theoretical guarantee in Theorem 1 remains empirically unvalidated, and it is unclear how closely the approximation in Eq. 7 approaches the ideal in Eq. 6. Figure 3 provides a loss landscape visualization but does not quantitatively assess gradient alignment.

- **Restriction to soft prompt adapters limits practical applicability.** The paper explicitly scopes to soft prompts (Section 4: "we specifically focus on the implementation where adapters are soft prompts, due to their computational efficiency"), and Theorem 1's gradient preservation is with respect to prompt parameters $P$. Modern LLM fine-tuning overwhelmingly uses LoRA or similar parameter-efficient methods. Whether LLE's gradient preservation extends to LoRA parameters—which involve low-rank matrix updates rather than direct input embedding optimization—is a non-trivial open question that the paper does not address, despite claiming the method is "applicable to various types of adapters" (Section 1).

- **CPL metric is unstable when the original model's zero-shot accuracy is low.** In Table 1, OT achieves CPL of 220.95% on WebQs for Qwen2, where the original model's zero-shot accuracy is only 1.82%. Because CPL = $S_{zs}(E) / S_{zs}(M)$, small absolute differences in $S_{zs}(E)$ produce massive CPL swings when the denominator is near zero. This makes CPL unreliable as a comparative metric in low-accuracy regimes and complicates interpretation of privacy results on such benchmarks. The paper does not discuss this instability or propose any normalization.

- **CPL captures only zero-shot inference leakage, leaving other attack vectors untested.** The metric measures the emulator's zero-shot task accuracy relative to the original model. A more sophisticated adversary could fine-tune the emulator on downstream data, use it as a teacher for knowledge distillation, or extract training data via prompt injection. Since CPL only reflects zero-shot performance, it may underestimate actual capability leakage under adaptive attacks. The paper's privacy claims are thus narrower than they appear: LLEOT protects against zero-shot capability extraction, not against all forms of capability recovery.

- **Limited evaluation scope: only QA tasks on models ≤3B parameters.** All four benchmarks are multiple-choice or short-answer QA, and the largest model tested is Llama-3.2-3B. The practical motivation for protecting model IP is strongest for large proprietary models (70B+ parameters). Whether LLE can be reliably trained on emulators derived from much larger models—and whether gradient alignment is preserved at that scale—is unaddressed. Additionally, soft prompt transfer properties may differ substantially in open-ended generation settings where the loss landscape structure is more complex.

## Nice-to-Haves

- Extend LLEOT to LoRA adapters and empirically test whether gradient preservation holds for low-rank parameter updates, which would greatly broaden practical relevance.
- Evaluate on at least one generation task (e.g., summarization, dialogue) and one larger model (7B+) to test generalizability beyond multiple-choice QA on small models.
- Conduct adaptive attack experiments (e.g., fine-tuning the emulator, using it as a distillation teacher) to test whether CPL's zero-shot privacy guarantee extends to more sophisticated threat models.
- Analyze the requirements and privacy implications of the elevation dataset $X_e$: how much data is needed, whether it must be domain-matched, and whether it creates its own leakage channel.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Theorem 1 proof contradiction (from Spark Finder):** The claim that "$\nabla_P H = 0$ is wrong because H is enforced by optimizing emulator parameters" is factually incorrect. $H$ is a constant scalar hyperparameter (the loss margin), not a function of $P$. The optimization in Eq. 7 adjusts the emulator's weights $\Theta_E$ to satisfy the margin constraint; once the emulator is constructed, for any given input and prompt, $H$ is just a constant offset, so $\nabla_P H = 0$ is correct.

- **Perceived simplicity of LLE as a weakness (from Positive Reviewer):** A simple mechanism correctly and insightfully applied is not a weakness. Many impactful contributions rest on straightforward ideas whose value lies in recognizing *where* and *why* to apply them.

- **Missing related works on differential privacy or loss augmentation:** Per rules, I do not have external sources to confirm the existence of specific related works that should have been cited.

- **Reproducibility concerns about hyperparameter choices (e.g., $\sigma=20$ for CPKD):** These are disclosed in the paper and appendix. The ablation in Table 3 validates the necessity of each component.

- **Formatting and clarity nitpicks:** Per rules, these are removed.

## Novel Insights

The paper reveals an underappreciated tension in offsite tuning: the very knowledge distillation that makes emulators useful for adapter transfer also makes them dangerous as capability proxies. LLE resolves this by exploiting a structural property of the loss landscape—adding a constant shifts the loss value (degrading inference) without altering gradients (preserving optimization). However, this creates an interesting paradox visible in the results: LLEOT emulators can achieve CPL = 0.00% (zero zero-shot capability, worse than random initialization) while still producing useful gradient signals for prompt optimization. This suggests that the gradient information needed for prompt tuning is orthogonal to the token-level predictions that constitute "inference capability," and that a model can be a completely useless generator while remaining an effective optimization guide. This decoupling between generative capability and gradient utility is a non-obvious property that merits deeper investigation, particularly regarding whether it holds for adapter types beyond soft prompts.

## Suggestions

- **Add empirical gradient alignment measurements.** Compute cosine similarity between $\nabla_P L_E$ and $\nabla_P L_M$ across training steps and over a grid of prompt values. This directly tests whether the approximation in Eq. 7 sufficiently achieves the ideal in Eq. 6, and would either validate or qualify the Theorem 1 guarantee.

- **Qualify the theoretical claims.** In Section 4.2, explicitly acknowledge that Eq. 6 is an idealization and Eq. 7 is its practical approximation. Discuss conditions under which the approximation error $\epsilon(P, x)$ is small enough for gradient alignment to hold in practice, or provide empirical evidence that the residual is negligible.

- **Address CPL metric instability.** Either restrict CPL reporting to benchmarks where the original model has non-trivial zero-shot accuracy, propose a normalization (e.g., $S_{zs}(E) - S_{zs}(M)$ as an absolute difference alongside the ratio), or add a discussion of why CPL values are interpretable despite denominator sensitivity.

- **Discuss the scope of privacy protection explicitly.** State clearly that LLEOT protects against zero-shot capability extraction but does not constitute a formal privacy guarantee (e.g., differential privacy) against adaptive adversaries who may fine-tune or distill from the emulator. This would set appropriate expectations and frame future work.

---

**Axis Evaluation:**

- **Novelty:** Moderate. The LLE concept (constant loss margin) is simple, but the insight to apply it for decoupling capability privacy from gradient utility in offsite tuning is novel and well-motivated. CPKD is a reasonable but incremental contribution.

- **Technical soundness:** The theoretical guarantee relies on an idealization (exact equality in Eq. 6) that is only approximately achieved in practice, and this approximation error is neither bounded nor empirically characterized. The proof of Theorem 1 is correct *given* the idealization, but the gap between theory and practice is the paper's most significant technical concern.

- **Empirical support:** Good within its scope—three models, four datasets, strong baselines, and informative ablations. However, the scope is narrow (only QA, only ≤3B models, only soft prompts), and the core theoretical claim (gradient alignment) lacks direct empirical verification.

- **Significance:** The problem framing (capability privacy) is significant and timely for proprietary LLM deployment. The proposed solution is practical and shows strong results. However, the restriction to soft prompts and small models limits immediate practical impact.

- **Clarity:** The paper is well-organized and clearly written. The progression from problem formulation through methodology to experiments is logical. The loss landscape visualization (Figure 3) could be more informative with quantitative annotations.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
