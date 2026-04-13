=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

This paper investigates localized machine unlearning, proposing a new localization strategy that identifies critical parameters using channel-wise weighted gradients (inspired by memorization literature) and pairing it with a reset-and-finetune algorithm called DEL (Deletion by Example Localization). The method outperforms prior localized and full-parameter unlearning methods on CIFAR-10 and SVHN across various forget set distributions and parameter budgets, while modifying only 30% of parameters.

## Strengths

- **Systematic ablation of localization design choices:** Table 1 provides a clean $2 \times 2$ factorial analysis of granularity (parameter vs. channel) and criticality criterion (gradient vs. weighted gradient), demonstrating that channel-level granularity with weighted gradients yields the best trade-offs. This is a principled way to isolate what makes localization effective.

- **Strong empirical performance across settings:** DEL consistently outperforms baselines on both IID and non-IID forget sets, across two datasets (CIFAR-10 and SVHN) and two architectures (ResNet-18 and ViT). The method achieves near-oracle performance on forget accuracy and MIA efficacy while modifying only 30% of parameters.

- **Effective control experiments:** The random-masking ablation (Table 3) convincingly demonstrates that parameter selection matters, not just parameter count. The analysis of forget-set-specific vs. data-agnostic criticality criteria (Table 4) provides insight into when tailoring to the forget set matters most.

- **Practical efficiency improvements over CritMem:** By replacing the iterative, example-by-example gradient computation of CritMem with a batched, one-shot approach, DEL achieves similar or better localization quality with substantially lower computational overhead.

## Weaknesses

- **Limited experimental scope:** Experiments are confined to CIFAR-10 and SVHN with 10% forget sets. The paper does not evaluate on larger-scale datasets (e.g., CIFAR-100, ImageNet), larger models, or smaller forget set sizes (e.g., ≤1% of training data, which is more representative of real-world deletion requests). This limits confidence in scalability claims.

- **Unablated hyperparameter $h$:** The neuron criticality score uses "the average of the top-$h$ scores" of constituent parameters (Section 5.2), but the value of $h$ is never specified, discussed, or ablated in the main text. This is a notable omission for a key design choice.

- **No runtime benchmarks:** The paper claims efficiency improvements over CritMem and comparisons to SalLoc, but provides no wall-clock time measurements for the localization step. Since all three methods require backward passes over the forget set, the actual efficiency gain from channel-wise aggregation versus parameter-wise sorting is unclear.

- **Missing SCRUB baseline:** SCRUB (Kurmanji et al., 2024) is mentioned in related work as a strong recent unlearning method that builds on NegGrad+ via distillation, but does not appear in the experimental comparisons. Its absence is a gap given its prominence in recent literature.

- **Statistical significance unclear for key comparisons:** Several important comparisons have overlapping uncertainty intervals (e.g., DEL $\Delta_{\text{forget}} = 0.43 \pm 1.06$ vs. SalLoc-RL $\Delta_{\text{forget}} = -2.8 \pm 1.45$ in Table 2 for Non-IID). Without formal significance tests, claims of superiority should be more cautious.

- **CritMem comparison at unequal budgets:** CritMem is evaluated at $\alpha=16\%$ because its algorithm terminates when predictions flip, while DEL uses $\alpha=30\%$. This asymmetry makes direct comparison difficult—CritMem is constrained to a budget it was not designed to work well at, rather than evaluated fairly at its optimum.

## Nice-to-Haves

- **Theoretical justification for weighted gradients:** The paper attributes the success of $|\theta \cdot g|$ over $|g|$ to "smoothing" and "regularization," but this remains heuristic. A deeper understanding (e.g., linking to influence functions or Fisher information) would strengthen the contribution.

- **Formal privacy guarantees:** The paper does not discuss whether DEL provides any differential privacy or divergence guarantees—important for GDPR-style applications.

- **Sequential unlearning:** The paper considers only single-batch unlearning; sequential or continual unlearning with multiple forget requests is unaddressed.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The abstract does not explicitly flag 2–3% lower test accuracy than oracle":** The abstract accurately summarizes contributions. Margin differences from oracle are standard in approximate unlearning; explicitly highlighting them in the abstract is unnecessary.

- **"Definition 2.1 is exact unlearning; gap with approximate methods should be acknowledged":** The paper correctly uses retrain-from-scratch as an aspirational benchmark and discusses approximate methods throughout. This is standard practice in the field.

- **"MIA evaluation is weak; stronger attacks should be used":** The paper acknowledges that more rigorous evaluations exist but are expensive. Using standard MIA efficacy from Fan et al. (2023) is reasonable for an empirical methods paper.

- **"The four contributions are repetitive":** The contributions are clearly distinct: (1) investigation of localization strategies, (2) improved localization strategy, (3) DEL algorithm, and (4) empirical results. This is appropriately structured.

- **"No mechanistic explanation for why Shallowest is strong":** This is a valid observation for future work, not a weakness of the current paper. The authors correctly identify it as an interesting finding.

- **"Δ sign convention unclear":** The paper defines Δ as Oracle − Algorithm, and lower absolute values are better. This is explained in Table 2 caption.

- **"Negative Δ_test values (test accuracy above oracle) are unexplained":** This occurs when the original model has higher test accuracy than the oracle (which is retrained on a smaller retain set). The phenomenon is natural and not problematic.

- **"Scalability to larger models and memory constraints":** The method operates on gradients; this is a valid concern but speculative without evidence. Memory overhead exists for all gradient-based unlearning methods.

## Novel Insights

The paper's key insight—that memorization localization insights transfer to unlearning when adapted properly—is valuable. Directly applying memorization hypotheses (e.g., "deepest layers memorize") fails for unlearning, but the granularity and criticality criteria from CritMem, combined with batched non-iterative computation, yield a practical improvement. The finding that forget-set-specific localization matters more for non-IID forget sets (Table 4) provides guidance for practitioners: when the forget set distribution matches the training distribution, simpler criteria may suffice.

## Suggestions

- **Ablate $h$ explicitly:** Report the value used and test sensitivity across a reasonable range (e.g., $h \in \{1, 5, 10, 25\}$).

- **Add wall-clock time comparison:** Report localization runtime for SalLoc, CritMem, and the proposed method on the same hardware.

- **Include smaller forget set experiments:** Test 1% and 5% forget sets to demonstrate real-world applicability.

- **Add SCRUB baseline:** Include this strong recent method in comparisons, or explain why it was excluded.

- **Clarify budget fairness:** If CritMem cannot operate at higher budgets due to algorithm design, either acknowledge this as a limitation or extend CritMem fairly (e.g., by running multiple iterations).

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 6.0, 5.0]
Average score: 5.0
Binary outcome: Reject
