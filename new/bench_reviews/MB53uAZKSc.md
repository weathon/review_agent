Now I have enough information to write the final review. Let me synthesize everything.

## Summary

TiC-LM introduces a benchmark for continual pretraining of language models, centered on TiC-CommonCrawl (TiC-CC), a time-stratified dataset of 2.9T tokens spanning 114 monthly Common Crawl dumps from 2013–2024, supplemented by domain-specific evaluations from Wikipedia, StackExchange, and code documentation. The paper quantifies model staleness (showing up to 45% noun-perplexity degradation), evaluates several continual learning baselines (optimizer-based, replay, and regularization methods), and demonstrates that optimal strategies are domain-dependent—replay prevents forgetting on slowly-evolving domains but harms performance on rapidly-evolving ones.

## Strengths

- **Large-scale, multi-domain benchmark with careful temporal causality.** TiC-CC spans 2.9T tokens across 114 months, far exceeding prior benchmarks (Table 1; the largest prior, TemporalWiki, had 23B tokens and only 4 timesteps). The paper explicitly avoids cross-month deduplication and classifier-based filtering that uses future data (Section 3), preserving temporal validity—a methodologically sound design choice clearly explained.

- **Well-designed regret-based evaluation metric.** The metric R_{i,j} = E_{i,j} − E_j* (Section 6) subtracts Oracle performance rather than own-timestep performance, avoiding the pathological case where poor in-distribution performance makes backward/forward metrics misleadingly good. This is a genuine methodological improvement over prior work.

- **Domain-dependent findings are the paper's most valuable insight.** Tables 3 and Figure 5 show that replay benefits TiC-STACK-MATH and TiC-CODEDOCS-NumPy (slower-evolving domains) but hurts TiC-STACKOVERFLOW and TiC-CODEDOCS-PyTorch (faster-evolving domains). This nuanced finding transcends the simple "replay prevents forgetting" narrative and is robust regardless of the specific budget allocation.

- **Practical efficiency analysis.** Table 4 demonstrates that Replay (α_t=1/2) + AR at 440B tokens becomes competitive with a series of biennially retrained Oracles requiring 1.16T tokens, establishing a 62% compute saving while providing monthly rather than biennial updates. This is a practically relevant finding for practitioners.

- **Corrects prior findings about replay ratios at scale.** The paper finds α_t=1/t scales poorly beyond 100 timesteps (Section 6.1), unlike TiC-CLIP's finding with ~10 rounds where different replay ratios behaved similarly—useful guidance for practitioners working at larger scales.

## Weaknesses

### Fatal
None.

### Major

- **Front-loaded token budget (50% on month 1) biases method-level conclusions, with no sensitivity analysis.** Section 6 states that half the 220B token budget (110B) is allocated to the first month (May-2013), leaving only ~1B tokens per subsequent month across 113 timesteps. This makes the initialization disproportionately strong on old data, creating a regime where forgetting is the dominant failure mode, which naturally inflates replay's apparent effectiveness. The abstract's headline claim of "60% regret reduction" from replay is a direct consequence of this allocation. The paper acknowledges the initialization bias in Section 6.2 ("the initialization trained on May-2013 already achieves 48.5, the same as the final checkpoint of Cyclic Cosine") and runs an Oracle variant starting from the same initialization (achieving only 48.9). However, it does not run the obvious control: an alternative budget allocation (e.g., distributing the 220B more evenly across months, or initializing on 3 years of data) to test whether the method rankings survive. Without this, the central method-level claim—that replay is most effective for combating forgetting—is conditional on a consequential design choice that favors methods preventing forgetting.

### Minor

- **The "100× larger" claim in the abstract is slightly misleading about the scale of the experimental contribution.** The abstract states "our TiC-CC training data is more than 100× larger compared with prior continual learning benchmarks" and Section 1 repeats this claim citing 2.9T tokens. However, all training experiments use only 220B tokens (Section 3: "We use smaller subset of 220B tokens from a single global shard"). The actual training-scale advantage over TemporalWiki (23B tokens) is ~10×, not 100×. While the 100× claim is technically correct for the *dataset* size (Table 1 shows 2.9T), the phrasing in the abstract ("our TiC-CC training data") could lead readers to believe the experiments operate at 100× prior scale. The paper should clarify that the dataset is 2.9T but the experimental protocol uses a 220B subset.

- **All experiments use a single model scale (3B parameters) with no scale-dependent analysis.** A benchmark paper's primary value lies in producing reliable guidance for future research. All method comparisons are at 3B parameters, and it is unknown whether the observed method rankings, forgetting dynamics, or domain-specific trade-offs change with scale. The most practically relevant use case for continual pretraining involves models ≥70B parameters. Even a single comparison at a different scale (e.g., 1B) would establish whether findings are scale-stable.

### Trivial

- **Standard deviations of exactly 0.000 for Cyclic Cosine across all 9 TiC-CC metrics.** Table 2 reports (0.000) for all nine standard deviations from three runs. While large evaluation sets can produce small variances, reporting exactly zero to three decimal places is unusual. More decimal places or verification would increase confidence.

## Nice-to-Haves

- An alternative token budget allocation experiment (e.g., initializing on 3 years of data or distributing 220B more evenly) would substantially strengthen the paper by testing the robustness of method rankings to this key design choice.
- A comparison at a different model scale (e.g., 1B) would establish whether the findings generalize.
- Analysis connecting TiC-CC held-out loss to downstream task performance would help validate the benchmark's primary metric as a predictor of practically relevant performance.
- Including parameter-efficient methods (e.g., adapter merging per timestep) would broaden the methodological coverage, though these are outside the paper's three-category framework.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "45% larger noun-perplexity" claim is imprecise.** The paper clearly states this is relative to the best-month perplexity (Figure 2 caption: "Ppl./Ppl. of Best Month"). The abstract says "45% larger noun-perplexity on 2024 Wikipedia articles compared to pre-2023 articles," which is consistent with Figure 2 showing the ratio reaching ~1.45 for DCLM on 2024 data. The characterization is accurate.

- **Harsh Critic: Oracle comparison potentially unfair due to data quality differences.** The paper explicitly states in Section 3 that it does not use the classifier-based filter from DCLM-Baseline to maintain causality. The Oracle is described as training on "all data (114 months)" from the TiC-CC pipeline, so the data quality should be the same. There is no evidence the Oracle uses higher-quality data.

- **Harsh Critic: Negative regrets in Table 4 mean the Oracle baseline might be wrong.** The paper clearly explains (Section 6.3) that each Oracle in the series is token-matched with continual checkpoints at their cutoff date, meaning each Oracle has a restricted budget. Negative regrets (i.e., continual methods beating restricted Oracles) is a transparent finding, not a problem with the baseline. The paper discusses this appropriately.

- **Harsh Critic: EWC vs Cyclic Cosine differences on TiC-WIKI-Diff are tiny and "overstated."** The paper says EWC is the "best method for adapting to new knowledge in TiC-WIKI-Diff" and provides a nuanced explanation for why this differs from TiC-CC results. While the absolute differences are small (Backward: 0.030 vs 0.033), this is technically correct and the paper provides plausible mechanistic explanations.

- **Harsh Critic: Missing parameter-efficient methods (adapters).** This is outside the paper's stated scope, which explicitly considers three categories: optimizer-based, data replay, and regularization (Section 5). Adding adapter-based methods would broaden coverage but is not a gap in the stated methodology.

- **Harsh Critic: No data availability or licensing discussion.** Common Crawl is publicly available and the paper describes the processing pipeline. While explicit release plans for evaluation data and scripts would strengthen the paper, this is a nice-to-have for a benchmark submission rather than a substantive weakness.

- **Harsh Critic: The second research question is answered only negatively.** The paper's honest answer that closing the gap to Oracle "proves to be an open and challenging problem" is appropriate for a benchmark paper. The paper does quantify the gap (Table 3, Table 4) and shows methods that partially close it.

- **Strength Finder: "Demonstrates Replay is most effective for combating forgetting, with quantified improvement" as a core strength.** While the 60% regret reduction on TiC-CC Backward is real (Table 2), this conclusion is conditional on the front-loaded budget allocation (verified Major weakness above). Keeping the quantified result but noting the caveat, and downgrading this from a "core strength" to a conditional finding.

- **Strength Finder: "Shows continual training can be 62% cheaper than periodic retraining" as a core strength.** This finding is valid and practical (Table 4), but it specifically applies to replay-based methods with the chosen budget allocation and at 3B scale. The 62% claim is well-supported for this specific setup.

## Novel Insights

The paper's most insightful finding—that the optimal continual learning strategy is domain-dependent, with replay benefiting slowly-evolving domains (NumPy docs, Stack-Math) but harming rapidly-evolving ones (PyTorch docs, StackOverflow)—transcends the specific experimental setup. This suggests that practical continual pretraining systems may need domain-adaptive replay strategies rather than one-size-fits-all approaches. The paper also usefully identifies that initialization bias (toward the earliest data) is a larger bottleneck than the continual training phase itself for static evaluations, which has practical implications: practitioners should consider initializing from more temporally diverse data rather than focusing solely on the continual training method.

## Suggestions

- Run at least one experiment with a more balanced token budget (e.g., allocate only 20% to initialization and distribute the rest evenly, or initialize on 3 years of accumulated data) to test whether the "replay is most effective" finding survives alternative allocations. This single experiment would address the paper's most significant weakness.
- When reporting the "100× larger" claim, explicitly clarify that this refers to the dataset size (2.9T tokens) while the current experimental protocol uses a 220B subset, to prevent reader confusion.
- Report standard deviations with more precision (at least 4 decimal places) to distinguish truly zero variance from rounding artifacts.

## Score and Decision

**Calibration anchors:**
- High: bRLed9prWC (7.33, future language modeling from temporal data—novel task formalization with clean experiments); 07yvxWDSla (8.0, synthetic continued pretraining—strong methodology and clear contribution); dOAkHmsjRX (7.50, budgeted online CL—fair benchmarking with novel method)
- Medium: IGuLzOXTB9 (5.25, Daily Oracle—similar topic but limited novelty, no method comparisons, trivial conclusions); AqN23oqraW (6.75, KoLA benchmark—broader scope but less focused); mz8owj4DXu (6.5, SLM continual learning—novel method but limited benchmark contribution)
- Low: zE4mL85zgg (2.20, CLAD—no ablations, missing baselines); MuXF0UZsoW (2.00, Icon Recognition—no comparison at all); WRxCuhTMB2 (1.67, Uncertainty Evaluation—fundamentally unclear contribution)

TiC-LM is substantially stronger than the rejected benchmark papers: compared to Daily Oracle (5.25), TiC-LM offers a much larger dataset, actual method comparisons (not just observation of degradation), multiple evaluation domains, and practical efficiency findings. However, compared to the high-scoring papers (7+), TiC-LM's method-level conclusions are more conditional due to the front-loaded budget and single-scale experiments. The benchmark itself is a solid contribution, and the domain-dependent findings are valuable and likely robust, but the headline method comparison results ("replay reduces regret by 60%") are less well-supported than they could be. The paper sits above the borderline but below the strongly accepted papers, at approximately 6.0.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>