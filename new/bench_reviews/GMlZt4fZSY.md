Now I have all the information needed. Let me write the final consolidated review.

## Summary

MobileLLM-R1 introduces a data-centric framework for training sub-billion-parameter reasoning models. Through leave-one-out analysis, influence-based cross-capability data mixing, and iterative data-model co-evolution during mid-training, the authors demonstrate that ~2T tokens of curated data (trained for 4.2T tokens with resampling) suffice to produce reasoning models that outperform prior fully open-source baselines (OLMo, SmolLM) and match or surpass Qwen3-0.6B on multiple benchmarks. All models, code, data sources, and mixing ratios are publicly released.

## Strengths

- **Iso-SFT comparison (Table 2) provides controlled evidence that pre-training/mid-training data quality drives downstream reasoning gains.** MobileLLM-R1-950M achieves 57.8 MATH under identical reasoning SFT, outperforming the larger OLMo-2-1.48B (53.0) and SmolLM2-1.7B (41.4). This directly isolates the contribution of the data curation pipeline from post-training advantages.

- **Full open-source release with complete training recipe** — models (HuggingFace), code (GitHub), data sources, and mixing ratios are all disclosed (Abstract), enabling reproducibility in a space where competitive models like Qwen3 use proprietary 36T-token corpora.

- **Cross-domain LOO finding is genuinely novel.** Section 2.1.2 and the discussion around Figure 3 show that StarCoder (code data) benefits math more than OpenWebMath (math data) benefits code — a reversal of the commonly held view from Lewkowycz et al. (2022). This is a concrete, non-obvious insight enabled by the LOO methodology.

- **Post-training ablation (Table 1) yields actionable practical insights.** The staged approach (Tulu-3 first, then reasoning data) outperforms joint training (57.8 vs 56.2 MATH), and scientific reasoning data transfers strongly to math (60.0 MATH for M+S vs 57.4 for M alone). These are useful findings for practitioners training small reasoning models.

- **Influence-based data mixing outperforms uniform sampling (Figure 4).** The influence-weighted mixture consistently outperforms uniform sampling across Code, Math, and Knowledge benchmarks, providing evidence that the methodology works.

## Weaknesses

### Fatal
None.

### Major

- **The "11.7% of Qwen3's tokens" efficiency claim is misleading due to the 950M vs. 600M parameter-count mismatch.** MobileLLM-R1-950M has ~58% more parameters than Qwen3-0.6B. Per scaling laws, larger models require fewer training tokens to reach equivalent performance at the compute-optimal frontier. The paper repeatedly frames this as a token-efficiency result (Abstract, Section 1, Conclusion) without controlling for model size. While the data quality contribution is real (the iso-SFT comparison in Table 2 supports this), the headline "11.7% of tokens" figure conflates two factors — data curation quality AND larger model capacity. An iso-parameter comparison (either a 600M MobileLLM variant vs. Qwen3-0.6B, or MobileLLM-R1-950M vs. Qwen3-1.5B) would properly isolate the data quality effect. Without this, the token-efficiency framing overclaims.

### Minor

- **The "benchmark-free" claim is partially misleading.** The method does not access actual benchmark test sets, which is legitimate. However, the capability-probing datasets (DP_C, DP_M, DP_K) are explicitly curated using hierarchical rejection sampling with domain-specific prompts targeting "code, math, general knowledge" (Section 2.1.1) and are derived from the training corpora themselves. They are functionally benchmark-aligned proxies. Calling the approach "benchmark-free" (Abstract, Section 2.2, Conclusion) when it uses capability-specific curated evaluation sets overclaims; "benchmark-agnostic" or "benchmark-unexposed" would be more precise.

- **No comparison of influence-based mixing against simpler heuristic alternatives.** Figure 4 shows the influence mixture outperforms uniform sampling, but no comparison against simple heuristics (e.g., upweighting math/code data by a fixed factor, or using FineWeb-EDU quality scores directly for weighting) is provided. Without this, it is unclear whether the expensive influence computation is necessary or whether any principled upweighting would suffice.

- **The theoretical interpretation of mid-training "convergence" is imprecise.** Section 3 states that influence scores concentrating around zero indicates "near-complete utilization of the informative content in the dataset." An alternative interpretation is that diminishing influence simply reflects the model becoming less sensitive to individual samples (consistent with memorization/overfitting), not necessarily that information has been "exhausted." While the empirical results in Figure 6 support the method's effectiveness (compressed data outperforms original data), the theoretical justification should be more carefully stated.

- **The iso-SFT comparison (Table 2) does not include Qwen3 models.** This is the most important missing comparison for the paper's central claim. Fine-tuning Qwen3-0.6B under identical reasoning SFT would directly test whether the pre-training/mid-training advantage holds against the primary competitor under controlled conditions.

### Trivial
None.

## Nice-to-Haves

- Variance/confidence intervals across multiple runs would strengthen result reliability, though single-run evaluation is standard for large-scale LLM training due to computational cost.
- An ablation disentangling the contribution of data quality from data repetition (2T unique tokens trained for 4.2T total) would clarify whether data quality or repetition drives performance.
- Qualitative examples of samples rejected vs. retained during mid-training compression would help assess whether the method preserves diversity or reinforces what the model already knows.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Tulu-3 + Code alone causes catastrophic forgetting" (Harsh Critic Point 5):** The Table 1 row "Tulu-3, C" shows Stage 2 ablation where ONLY Code data is used. Of course a model trained only on code will perform poorly at math — that is the expected and informative result of the ablation, not evidence of "brittle capabilities" or catastrophic forgetting.

- **"Introduction mischaracterizes its own positioning" (Harsh Critic Point 6):** The paper explicitly states "While the first assumption has already been challenged by recent sub-billion-parameter reasoning models such as Qwen3-0.6B and DeepSeek distilled variants, the second remains largely unquestioned." The paper clearly scopes its contribution to the second assumption (data scaling), not the first (model size). The critic misread the introduction.

- **"No variance/confidence intervals reported" (Harsh Critic Point 4):** While true, this is standard practice for large-scale LLM training papers where multiple full training runs are prohibitively expensive. This is a nice-to-have, not a substantive weakness.

- **"4.2T tokens obscures repetition" (Harsh Critic Point 3):** The paper is transparent about this — it explicitly states both "~2T tokens of high-quality data are sufficient" and "pre-training with 4.2T tokens on the dataset resampled from these ~2T tokens" (Abstract). Both numbers are provided. The 11.7% comparison uses 4.2T total tokens, and Qwen3's repetition rate is also unknown. This is a minor framing issue, not evidence of obfuscation.

- **"LOO findings confirm common wisdom" (Harsh Critic Point on Section 2.1):** While the finding that FineWeb-Edu is broadly useful ("glue") is somewhat expected, the StarCoder→math transfer reversal of Lewkowycz et al. (2022) is genuinely novel. The LOO methodology also provides quantitative evidence beyond common wisdom.

- **Strength Finder's "exceptional token efficiency achieving competitive reasoning with 11.7% of Qwen3's training data"**: This strength conflicts with the verified Major weakness about the parameter-count mismatch. The 11.7% figure is misleading as stated. This strength is removed because the weakness takes precedence.

## Novel Insights

The reversal of the conventional code-math transfer direction — that code data (StarCoder) benefits math more than math data (OpenWebMath) benefits code — is a genuinely novel finding that contradicts Lewkowycz et al. (2022). Additionally, Figure 7's observation that math-focused mid-training subsequently improves coding perplexity (HumanEval) suggests a sequential transfer pathway that could be strategically exploited in training schedules: math first, then code.

## Suggestions

- Reframe the token-efficiency claim to account for the parameter count difference. Either compare at iso-parameter (600M vs 600M, or 950M vs Qwen3-1.5B if available), or explicitly acknowledge the confound and argue that the 950M model is still far below Qwen3's compute-optimal token budget (Chinchilla-optimal would be ~19T for 950M), making the 4.2T result remarkable even accounting for size.
- Replace "benchmark-free" with "benchmark-unexposed" or "benchmark-agnostic" throughout the paper to more precisely describe the approach.
- Add a simple heuristic baseline (e.g., 2× upweighting of code/math data) to Figure 4's comparison to validate that the influence machinery provides value beyond simple heuristics.

## Score and Decision

**Calibration anchors compared:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| OLMoE (Oral) | 8.67 | Stronger: more novel architecture, fully open, MoE insights. MobileLLM-R1 has less architectural novelty but comparable open-source commitment. |
| MetaMath (Spotlight) | 8.0 | Stronger: more novel data augmentation, cleaner claims. MobileLLM-R1 has broader scope but overclaimed efficiency. |
| Compute-Optimal Sampling (Poster) | 7.0 | Roughly comparable spirit; that paper had confounded comparisons but was clear about limitations. MobileLLM-R1 has stronger empirical scope. |
| Textbooks/phi-1 (Reject) | 6.0 | Similar spirit (small model + curated data). phi-1 had contamination concerns and weaker methodology. MobileLLM-R1 is better: more principled, more transparent. |
| ProX (Reject) | 6.0 | Similar data curation focus. ProX had scalability concerns. MobileLLM-R1 has stronger results but similar overclaiming concern. |
| Small-to-Large Generalization (Poster) | 5.25 | Weaker results, narrower scope. MobileLLM-R1 is clearly above this. |
| Paramanu-Ganita (Reject) | 2.33 | Much worse: extreme parameter mismatch (208M vs 7B+), overclaimed efficiency, contamination concerns. MobileLLM-R1 is far above this. |

MobileLLM-R1 is clearly above the low-scoring papers (Paramanu-Ganita at 2.33, ELR-Diffusion at 2.50) — the parameter mismatch is much less extreme and the methodology is far more principled. It is above medium-reject papers like phi-1 and ProX due to stronger empirical results and better methodology. It falls below the high-scoring papers (OLMoE, MetaMath) due to the overclaimed efficiency metric and less architectural novelty. The central token-efficiency claim is weakened by the parameter-count confound, but the paper makes genuine contributions: the iso-SFT comparison, the open-source release, and the cross-domain LOO analysis.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>