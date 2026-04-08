=== CALIBRATION EXAMPLE 44 ===

# Final Consolidated Review
## Summary

The paper introduces a data-centric approach to accelerate speculative decoding (SD) draft model training. The key insight is that tokens yielding flatter (more uniform) target model distributions are disproportionately valuable for reducing the L1 discrepancy that governs SD acceptance rate. Based on this, the authors propose "flatness" (cosine similarity to uniform) as a token-level importance metric, aggregate it to the sample level, and develop SFDD (Sample-level-flatness-based Dataset Distillation), which filters training data via a single offline target-model forward pass. Experiments on EAGLE-2 with LLaMA3-8B-Instruct show that using 50% of data via SFDD achieves ~2× training speedup with inference speedup within 4% of the full-dataset baseline.

## Strengths

- **Novel data-centric framing for SD training efficiency.** Rather than modifying the loss function (as prior work on L1-based distillation does), the paper identifies *which data* matters most for acceptance-rate improvement, establishing a new axis of optimization for SD. This is a genuine conceptual contribution that the SD literature has largely overlooked.

- **Theoretically motivated metric that is offline-computable.** The flatness metric depends solely on the target model's output distribution and requires no warm-up of the draft model or tracking of its training dynamics. This is a practical advantage over importance metrics that require gradient computation or draft model snapshots, and the paper correctly identifies this as a key design property.

- **Concrete practical efficiency gains with simple pipeline.** The full pipeline—compute flatness scores, rank, filter, train—is straightforward to implement. The reported training time reduction from 58,227s to 28,787s at 50% retention (Section 5.4) is a meaningful resource saving, and the method consistently outperforms alternative data-selection metrics across five diverse downstream tasks and multiple retention ratios.

## Weaknesses

### Major:

- **No evaluation of generation quality.** The paper exclusively reports speedup and average acceptance length but never verifies that the outputs produced by the SFDD-trained draft model maintain the same quality as those from the full-data model. Faster inference is only valuable if output correctness/fidelity is preserved. Even simple metrics like GSM8K accuracy, MT-Bench scores, or ROUGE on CNN/DM would address this. Without any quality verification, it is impossible to confirm that the ~4% speedup degradation is the only cost of data reduction—the draft model might also produce systematically different outputs that affect downstream task performance.

- **Gaussian assumption does not generalize consistently, and the discrete bridge is not rigorous.** The core theoretical insight (flatter → more valuable) is derived under parametric Gaussian distributions (Section 3.2). However, Appendix F.3 explicitly shows that the same monotone relationship does **not** hold for Exponential or Half-Normal families—"no simple, consistent monotone trend emerges." The paper argues these families are less representative because they are single-parameter and mode-locked, but this defense itself reveals that the insight is not a general property of all distributions—it depends on the location-scale structure of Gaussians. The bridge to discrete categorical distributions (Appendix B) relies on an asymptotic argument (cosine similarity ∝ σ^{1/2}) that requires L ≫ σ, which may not hold for low-flatness tokens where the distribution is concentrated. The paper provides empirical correlation (Figure 2) but no rigorous discrete justification, leaving the theoretical foundation on uncertain ground.

### Minor:

- **Limited model and framework diversity in primary experiments.** The main results use a single target model (LLaMA3-8B-Instruct), single dataset (ShareGPT), and single SD framework (EAGLE-2). Appendix G.1 adds Vicuna-7B, which is encouraging but still limited to small models. Whether flatness-based selection generalizes to larger target models (e.g., 70B), other SD architectures (Medusa, DistillSpec), or different data distributions is unknown. The contribution claim of "a new paradigm" would be stronger with evidence across at least one additional framework.

- **Flatness vs. entropy distinction is not sharply articulated.** Appendix F.2 acknowledges that entropy-based curves are "remarkably similar" to flatness curves, and both metrics fundamentally measure distance from uniformity. Figure 2d shows a positive gap, but the paper does not provide a clear theoretical or mechanistic explanation for *why* cosine similarity outperforms entropy. Given that entropy is more standard and interpretable, the practical advantage of the proposed metric is not conclusively established.

- **Token-level validation relies on only 10 samples.** The key empirical claim that high-flatness tokens exhibit larger ΔL1 changes (Figure 2) is demonstrated on "10 randomly selected samples." This is insufficient to support the general claim about token-level dynamics across the full training corpus. A larger-scale or statistical analysis would substantially strengthen this result.

- **No characterization of filtered data.** The paper does not analyze what types of samples or tokens are removed by SFDD. Without understanding whether the method systematically removes certain domains, token types, or difficulty levels, it is difficult to assess potential distributional biases or failure modes (e.g., removing rare-but-important domain-specific tokens that happen to have peaked distributions).

### Trivial:

- **Key limitations are presented primarily in appendices.** The Gaussian assumption's failure for other distribution families (Appendix F.3) and the token-level filtering infeasibility (Appendix F.6) are important caveats that readers may miss when they appear only in supplementary material.

## Nice-to-Haves

- Statistical significance testing (e.g., multiple random seeds with confidence intervals) for the speedup and acceptance-length differences between SFDD and baselines.
- Ablation of the flatness metric itself—comparing cosine similarity to other potential flatness proxies (e.g., entropy-based selection with different binning, variance-based approximations for discrete distributions) to confirm that the specific choice of cosine similarity matters.
- Dynamic or adaptive selection strategies that re-score flatness during training, since the draft model's alignment with the target changes over time.
- Analysis of whether high-flatness tokens cluster in specific sequence positions or contexts, which could reveal confounding factors.

## Removed Points

These points were flagged for removal and should be treated with caution:

- **Abstract quantitative claims accuracy** — The 2× speedup (2.02× at 50% retention, Section 5.4) and "within 4%" (2.41× vs. 2.49× in Table 1) are supported by the paper's data. This is not a weakness.
- **Formatting/style nitpicks** — Figure reference parsing artifacts, equation formatting, and contribution overlap are presentation issues, not substantive weaknesses.
- **Missing failure mode in abstract** — Demanding limitations be previewed in the abstract is a stylistic preference, not a substantive flaw.
- **ShareGPT version/dataset accessibility** — The paper cites the dataset; per hard rules, availability of cited datasets is not questioned.
- **Hyperparameter disclosure** — The paper provides training hyperparameters in Table 6; requesting additional implementation details is a reproducibility nitpick.
- **Confidence intervals for large-scale benchmarks** — Single-run evaluation is standard practice in this community; demanding statistical testing is a nice-to-have, not a core weakness.
- **Temperature sensitivity** — Already addressed in Appendix C with a full set of temperature=0 results.
- **Compute cost in FLOPs** — Wall-clock time including selection overhead is reported; FLOPs accounting is a nice-to-have.
- **Negative societal impact discussion** — Scope creep for a training efficiency paper; the ethics statement is adequate.
- **Comparison with SD-specific data selection baselines** — No established SD-specific data selection methods exist in the literature; comparing against generic importance metrics is the appropriate baseline.
- **Baseline fairness/tuning** — The baselines (entropy, top-1 probability, margin, energy score, PPL) are standard metrics with clear selection criteria (high entropy, low top-1, etc.); there is no evidence they were unfairly disadvantaged.

## Novel Insights

The paper's central insight—that tokens with flatter target distributions contribute disproportionately to acceptance-rate improvement in speculative decoding—is genuinely novel and well-motivated. The reframing from "which loss function" to "which data" for SD training efficiency is a meaningful paradigm shift. However, the insight that "uncertain tokens matter more" echoes broader active learning intuitions; what makes this contribution distinctive is the *SD-specific* theoretical derivation connecting distributional flatness to L1-norm reduction (and hence acceptance rate), rather than generic model uncertainty. The finding that cosine similarity to uniform marginally outperforms entropy for this specific task—despite their conceptual similarity—suggests that the geometry of the probability vector (rather than just its information content) plays a role in SD-specific training dynamics, though the paper does not fully explain this advantage.

## Suggestions

- **Add generation quality evaluation.** Report at least one quality metric (e.g., GSM8K accuracy, MT-Bench scores) for the draft model trained with SFDD vs. full data to confirm output fidelity is preserved.
- **Expand token-level validation beyond 10 samples.** Scale up Figure 2's analysis to a larger subset or use aggregate statistics over the full dataset to substantiate the token-level dynamics claim.
- **Move the Gaussian assumption limitation to the main text.** The caveat in Appendix F.3 is important for readers to assess the theory's scope and should appear in Section 3.2 or as a dedicated discussion paragraph.
- **Provide a mechanistic explanation for the flatness-over-entropy advantage.** Even a brief theoretical argument or empirical analysis (e.g., on distributions where cosine similarity and entropy diverge) would strengthen the claim that flatness is the right metric, not just any uncertainty measure.
- **Test on at least one additional SD framework** (e.g., Medusa or DistillSpec) to support the generality of the approach beyond EAGLE-2.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
