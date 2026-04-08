=== CALIBRATION EXAMPLE 32 ===

# Final Consolidated Review
## Summary

This paper proposes a data-centric approach to accelerating speculative decoding (SD) draft model training. The authors identify that tokens with flatter (more uniform) target model distributions contribute disproportionately more to reducing the L₁ distance that governs SD acceptance rates. They formalize this insight through a theoretical analysis under Gaussian assumptions, propose a "flatness" metric (cosine similarity to the uniform distribution) as a practical proxy, and develop Sample-level-flatness-based Dataset Distillation (SFDD), which filters training data to retain high-flatness samples. Experiments on EAGLE-2 with LLaMA3-8B-Instruct show that SFDD achieves over 2× training speedup at 50% data retention while keeping inference speedup within 4% of the full-data baseline.

## Strengths

- **Novel insight connecting distribution flatness to SD training value.** The observation that tokens with flatter target distributions yield larger per-step reductions in the L₁ distance governing acceptance rates is genuinely novel and specific to the SD setting. This reframes data importance in SD training—not just which loss to use, but which tokens carry the most useful signal. The theoretical analysis (Section 3.2, Appendix A) provides formal grounding for this insight under the Gaussian assumption, and the empirical validation (Figure 2) confirms the trend on real LLM outputs.

- **Practical efficiency gains with a simple, target-model-only method.** SFDD requires only a single offline forward pass of the target model to compute flatness scores, then filters data by threshold. At 50% retention, it achieves 2× training speedup (28,787s vs. 58,227s, including selection overhead) with an average inference speedup of 2.41× vs. 2.49× for full data (Table 1, Figure 4). The method plugs into existing training pipelines without architecture changes and the selection overhead is only 3.85% of total training time (Appendix D).

- **Consistent improvement over a comprehensive set of data selection baselines.** Table 1 compares against entropy, top-1 probability, margin, energy score, and perplexity—SFDD outperforms all of them across all five evaluation tasks. The ablation across retain ratios (Table 2, Table 3) demonstrates robustness from 5% to 70%, including extreme data reduction regimes where SFDD still substantially outperforms random filtering.

## Weaknesses

- **Theory-practice gap from the Gaussian assumption.** The core theoretical analysis (Section 3.2) derives the key insight under Gaussian distributions, but LLM output distributions are discrete probability vectors over vocabularies of 32K–128K tokens. The bridge to practice relies on cosine similarity to uniform, justified by showing it correlates with Gaussian variance (Figure 1b, Appendix B). However, Appendix F.3 shows that when the same analysis is applied to Exponential and Half-normal families, "no simple, consistent monotone trend emerges," and the authors acknowledge these families fail to capture "argmax mismatch." This raises the question: what specific properties of real LLM distributions make the Gaussian-derived insight hold? The paper provides empirical validation (Figure 2) but does not characterize how much predictive power the theoretical flatness metric retains on actual LLM distributions versus being an empirical heuristic that happens to work.

- **Why flatness outperforms entropy is not fully explained.** Both flatness (cosine similarity to uniform) and entropy measure "distance from uniform" and exhibit similar training dynamics (compare Figures 2 and 5 in Appendix F.2). Yet Table 1 consistently shows flatness outperforming entropy (e.g., 2.41× vs. 2.20× average speedup). Figure 2d shows that flatness-based filtering removes more saturated tokens than entropy-based filtering, but the paper does not provide a theoretical explanation for why cosine similarity is a better proxy for L₁-relevant headroom than entropy. Since this distinction is central to claiming a new metric rather than repackaging an existing one, the lack of deeper explanation weakens the contribution's novelty claim.

- **No evaluation of downstream generation quality.** All reported metrics—speedup and average acceptance length—measure inference efficiency and draft-target alignment, not generation quality. A draft model trained on filtered data could subtly shift its distribution in ways that affect output quality on downstream tasks (e.g., ROUGE on CNN/DM, accuracy on GSM8K) without being captured by acceptance rate alone. This is particularly relevant because SFDD removes low-flatness (high-confidence) tokens, which could reduce exposure to easy but important patterns.

- **Limited model and framework diversity.** All main experiments use LLaMA3-8B-Instruct within the EAGLE-2 framework. Appendix G.1 adds Vicuna-7B-v1.3 and GSM8K training data, but both still operate within EAGLE. No experiments test larger models (e.g., 34B, 70B) or alternative SD frameworks (e.g., Medusa, layer-skip approaches). Whether the flatness distribution of tokens remains usefully non-uniform across different model scales and architectures—and whether the relative benefit of SFDD persists—is not established.

## Nice-to-Haves

- Statistical significance tests (e.g., standard deviations across multiple seeds) for the main results tables. Appendix F.8 provides two repeated runs for two datasets at two retention ratios, suggesting results are stable, but systematic multi-seed reporting would increase confidence.

- Analysis of the types of samples that get filtered (e.g., domain distribution, linguistic properties) to assess whether SFDD removes genuinely redundant data or potentially useful diversity.

- Experiments with larger target models (e.g., 70B) and alternative SD frameworks to demonstrate broader applicability.

- Discussion of how SFDD's overhead scales to very large pretraining corpora (billions of tokens), where a full forward pass of the target model may be costly.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Weakness: "Missing limitations section."** This is a formatting/style nitpick. The paper discusses limitations throughout (Gaussian assumption in Appendix F.3, token-level filtering infeasibility in Appendix F.6, overhead in Appendix D). A dedicated section would be nice but is not a substantive flaw.

- **Weakness: "Comparison to recent SD-specific data selection baselines."** Per hard rules, I cannot confirm the existence of such baselines and should not flag missing related works.

- **Weakness: "Compute resources not fully specified (number of GPUs, seeds)."** Per hard rules, reproducibility nitpicks about undisclosed implementation details are removed. The paper specifies H800 GPUs, hyperparameters, and provides code.

- **Weakness: "10 samples for Figure 2 analysis is too limited."** This analysis is qualitative/illustrative, showing token-level training dynamics. The main quantitative results use the full dataset. The 10-sample analysis is not used to draw quantitative conclusions about general performance.

- **Weakness: "SFDD at 70% retention sometimes exceeds No Filter (Table 2)."** The authors acknowledge and explain this (filtering can remove noisy data). This is not a weakness of the method—it suggests data filtering can occasionally improve beyond the full dataset, which is a known phenomenon in data selection literature.

- **Weakness: "Different prompt length distributions across datasets confound speedup comparisons."** The paper addresses this in Appendix F.1, explaining that speedup depends on prefill cost as well as acceptance length. This is inherent to SD evaluation, not a flaw in the method.

## Novel Insights

The most interesting finding emerging from the review synthesis is the consistent empirical advantage of flatness (cosine similarity to uniform) over entropy as a data selection criterion for SD, despite both metrics being theoretically related to distribution uncertainty. This gap is shown quantitatively in Figure 2d (flatness removes more saturated tokens) but lacks a principled explanation. Given that SD's objective is L₁-norm minimization rather than KL-divergence minimization, cosine similarity—being an L₂-based measure—may align more naturally with L₁ geometry than entropy (a KL-derived measure). This suggests a deeper connection between the choice of importance metric and the geometry of the training objective that the paper does not fully exploit but could be a fruitful direction for future work.

## Suggestions

- Provide a deeper analysis of why cosine similarity to uniform outperforms entropy as a selection metric. Consider whether the L₂ geometry of cosine similarity naturally aligns with L₁-relevant headroom in a way that the KL-derived entropy does not—this could strengthen the theoretical contribution.

- Report at least one downstream quality metric (e.g., ROUGE-L on CNN/DM, accuracy on GSM8K) for the filtered vs. full-data draft models, to verify that generation quality is preserved alongside acceptance rate.

- Add a brief characterization of what content is being filtered (e.g., what fraction of code vs. conversation, average sequence length of retained vs. discarded samples), so readers can assess whether SFDD preserves task diversity.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
