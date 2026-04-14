## Summary

TiC-LM introduces a large-scale benchmark for continual pretraining of language models built on 114 monthly Common Crawl dumps spanning May 2013 to July 2024 (2.9T tokens). The paper proposes a suite of temporally-stratified evaluations (TiC-CC, TiC-WIKI, TiC-StackExchange, TiC-CODEDOCS), establishes multiple continual learning baselines, and finds that data replay most effectively combats forgetting (reducing backward-transfer regret by 60% over non-replay methods), while showing domain-specific tradeoffs: rapidly-evolving domains like StackOverflow are harmed by old-data replay, whereas stable domains like NumPy documentation benefit from it. A key practical result is that scaled replay with an AR learning rate schedule can match a series of retrained oracles at 62% less compute.

---

## Strengths

- **Unprecedented benchmark scale with genuine temporal causality**: At 2.9T tokens across 114 timesteps, TiC-CC is more than 100× larger than the nearest prior time-continual LM benchmark (TemporalWiki at 23B tokens). Critically, the entire data pipeline enforces strict temporal causality (no future data used for filtering or deduplication), creating a realistic simulation of a practitioner's actual data access conditions—something previous work did not achieve at this scale.

- **Domain-specific evolution rate findings are mechanistically informative**: The paper's empirical observation that replay *hurts* StackOverflow but *helps* NumPy—and its mechanistic explanation grounded in library age (NumPy 1995 vs. PyTorch 2016) and the corresponding heatmap analysis of when knowledge appears in CC dumps—is a genuinely non-obvious finding that goes well beyond the aggregate metrics typical of benchmark papers.

- **Practical efficiency case study with concrete compute numbers**: Section 6.3 carefully defines a "series of oracles" baseline that is actually deployable (i.e., retrained every ~2 years), quantifies its cost at 1.16T tokens, and shows that Replay(α=1/2)+AR at 440B tokens matches or surpasses it across TiC-CC and TiC-WIKI evaluations. This is a directly actionable result for practitioners, not just a comparison against an idealized and unattainable baseline.

- **TiC-WIKI-Diff peak-performance lag finding**: The observation that model performance on a given Wikipedia month often peaks *years after* that month is seen—even without replay—is a scientifically important result that reveals delayed alignment between CC crawls and Wikipedia edit coverage, with clear implications for how temporal LM evaluations should be interpreted.

- **Principled hyperparameter selection**: The use of only the first 10 of 113 continual timesteps for tuning (rather than the full sequence) follows realistic conventions and avoids inflating method performance through look-ahead tuning.

---

## Weaknesses

- **All continual experiments conducted at 3B parameters only, with no scaling analysis**: This is the most significant limitation. The key empirical claims—optimal learning rate is 30× smaller than initialization, α=1/t scales poorly beyond 100 timesteps, EWC's relative behavior versus replay—are all scale-dependent in the broader continual and pretraining literature. The paper presents findings as general principles for continual LLM pretraining while providing no evidence they hold at 7B or beyond, which is precisely the scale where practitioners need guidance. The paper does show 7B/9B model *obsolescence* (Figure 2) but conducts no continual learning runs at that scale.

- **Reported standard deviations are 0.000 throughout Table 2, rendering uncertainty analysis vacuous**: Three runs of Cyclic Cosine produce standard deviations that all round to 0.000 at three decimal places. This means the bolding criterion ("within one standard deviation of best") is effectively equivalent to being exactly best, and no confidence can be placed in comparisons between methods that differ by small amounts in Table 2. The paper should report raw variance or use a more meaningful uncertainty estimate.

- **Front-loading of token budget (50% on May-2013 initialization) is not ablated for dynamic evaluations**: The paper addresses this concern for static CORE evaluations (Section 6.2, by training an oracle from the May-2013 init on an equal mix of the remaining 113 months, reaching 48.9), finding the initialization is more at fault. However, no analogous ablation is provided for the dynamic evaluations (TiC-CC perplexity, TiC-WIKI), where the heavily 2013-biased starting point could systematically favor replay of old data and distort comparisons between methods. This design choice is significant: it creates a model that begins the continual phase already biased toward 11-year-old web content.

- **ppl_answer metric is not length-normalized (Eq. 2)**: The answer perplexity computes `(1/|Q|) * Σ_q exp(-log P(a_q|c_q))`, which exponentiates the total log-probability of an answer rather than its per-token log-probability. This means answers of systematically different lengths across StackExchange categories or time periods will produce perplexity differences that reflect length differences rather than knowledge differences. The effect is not quantified or discussed.

- **Static CORE evaluations show near-indistinguishable method performance**: All methods fall within a 0.7 CORE accuracy range (48.5–49.2) versus an oracle at ~50.6. While the paper correctly explains why (initialization bias), it means the static evaluation cannot differentiate between continual methods at this scale, leaving the benchmark's guidance for practitioners relying almost entirely on perplexity-based metrics.

- **Replay ratio inconsistency across tables without justification**: Table 2 uses α=1/t and α=1/2 for replay; Table 3 uses α=1/4 and α=1/2. The switch from α=1/t to α=1/4 for domain evaluations is unexplained and suggests post-hoc selection. A note explaining why α=1/t was not used in Table 3 is needed.

---

## Nice-to-Haves

- A continual run at 7B scale (even a single trajectory) would substantially strengthen the generalizability claims.
- IO/storage costs of replay should be discussed: replaying terabytes of historical data from TiC-CC incurs non-trivial storage and data-loading costs that are not reflected in the FLOPs-based 62% efficiency claim. The practical efficiency argument would be more complete with this analysis.
- An ablation on deduplication strategy (e.g., comparing within-month vs. earliest-occurrence global dedup) would quantify the tradeoff the authors deliberately chose to leave as a method design question.
- A sensitivity analysis on EWC/LwF regularization strength would help distinguish whether these methods inherently underperform or were simply under-tuned at scale.
- PEFT-based methods (LoRA, adapters) are not standard in continual pretraining of base models at this scale, but a brief empirical comparison or discussion of their applicability would help practitioners considering these lighter-weight alternatives.
- Explicit data release and reproducibility details (e.g., HuggingFace indices, reconstruction scripts) for the 2.9T token stream should be provided so the benchmark can be used without re-processing all 114 CC dumps.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"100× larger" comparison mixes units (Harsh Critic)**: The 100× claim explicitly compares against TemporalWiki's 23B tokens (Table 1), making the token-to-token comparison valid. The mixing of units affects other baselines in Table 1 but not the specific 100× headline claim.
- **DCLM Oracle quality mismatch due to classifier filter omission (Harsh Critic)**: The paper explicitly explains this decision (the classifier was trained on all months, so using it would violate temporal causality). This is the methodologically principled choice given the paper's goals.
- **EWC compute unfairness (Harsh Critic)**: The paper explicitly states "we do not try to adjust the token counts to account for this given that our re-implementations may not be optimally efficient." This is an acknowledged limitation and does not invalidate the finding.
- **No discussion of GDPR right-to-be-forgotten (Harsh Critic)**: The paper already briefly notes data deletion as a downside of replay. Demanding a full treatment of GDPR is outside the paper's scope.
- **Abstract's 60% regret reduction is not vs. the Oracle (Harsh Critic)**: The abstract correctly says "60% compared to other optimizer and loss-based interventions"—this is accurate per Table 2 (0.023 vs. 0.058 for EWC on TiC-CC).
- **No multilingual evaluation (Harsh Critic)**: Entirely outside the paper's stated scope.
- **Efficiency analysis "uses different oracle" (Harsh Critic)**: The paper clearly motivates the Series of Oracles as a more practically meaningful comparison; the change in oracle definition is deliberate and explained.
- **Copyright/licensing discussion absence (Harsh Critic)**: Common Crawl benchmarks are standard in the community; this is not a specific weakness of this paper.

---

## Novel Insights

The most genuinely novel insight synthesized from the reviews and the paper itself is the mechanistic connection between a *domain's temporal evolution rate* and the *optimal replay strategy*, grounded in the specific discovery that CC crawl lag relative to Wikipedia edits causes TiC-WIKI performance peaks to appear years after training on a given month—even without replay. This finding has implications beyond continual learning benchmarking: it suggests that temporal perplexity evaluations built on CC data may systematically underestimate how quickly models learn recent factual knowledge, because CC's coverage of recent Wikipedia edits is delayed. Combined with the NumPy/PyTorch asymmetry (where library release date relative to the training window determines whether replay helps or hurts), the paper provides a framework for predicting optimal update strategies from measurable properties of each domain.

---

## Suggestions

- **Report per-token normalized perplexity for ppl_answer**, or at minimum provide a length distribution analysis across time periods for TiC-StackExchange to validate that answer lengths do not confound comparisons.
- **Add one 7B continual run** (even a single Cyclic Cosine or Replay trajectory) and explicitly compare its forgetting/plasticity tradeoff to the 3B results. This would make the empirical claims much more broadly applicable.
- **Address the replay ratio inconsistency** between Tables 2 and 3 with a clear explanation, or add a supplementary table showing α=1/t results for the domain evaluations as well.
- **Report raw standard deviations** without rounding to three decimal places, or alternatively report confidence intervals using bootstrapping over evaluation months rather than over training runs—this would provide meaningful uncertainty estimates.
- **Quantify the cross-month training/evaluation overlap** in the held-out CC sets to establish that the lack of global deduplication does not introduce substantial contamination in the evaluation splits.
- **Discuss storage and IO costs** of replay-based methods in Section 6.3 alongside the FLOPs analysis to give practitioners a complete picture of the 62% efficiency claim.

---

**Novelty**: High — the benchmark fills a genuine and substantial gap in the continual LLM pretraining literature, and the domain-specific evolution findings are non-obvious.

**Technical soundness**: Moderate — the evaluation design is principled and the regret formulation is carefully constructed, but the 0.000 standard deviations and the ppl_answer normalization issue are real deficiencies.

**Empirical support**: Moderate — the 3B-only scale is a meaningful constraint on the scope of the claims; the domain-specific findings are well-supported within that constraint but their generalizability is unverified.

**Significance**: High — provides a standardized, large-scale infrastructure that the community was clearly missing, with immediately actionable findings about replay strategies.

**Clarity**: High — the paper is dense but well-organized; the regret formulation and the efficiency analysis are explained carefully, though the ppl_answer definition and the ID metric for misaligned T_t/T_e deserve clearer exposition.

MY FINAL SCORE: <pineapple>6.7</pineapple>