Now I have a good calibration set. Let me now synthesize the final review.

Key calibration papers:
- GLoRA (LoRA variant): scores 6, 5, 6, 5 → Reject
- FoundTS (TS foundation model benchmark): scores 5, 5, 3, 3 → Withdrawn/Reject
- In-context Fine-tuning for TS Foundation Models: scores 8, 6, 5, 3, 6 → Withdrawn
- Intrinsic-ControlNet: 3, 5, 5, 5 → Withdrawn/Reject
- PaCA (PEFT method with novelty): 6, 6, 6, 6 → Accept Poster
- Enhancing Foundation Models for TS Forecasting via Wavelet Tokenization: 5, 6, 8, 3 → Reject
- AutoML for Sustainable DL: 6, 6, 5, 5, 5 → Reject

Now I understand the landscape. The current paper combines two existing methods (LoRA + LDS) without algorithmic novelty, and critically lacks baseline comparisons for its search strategy. Let me now write the final review.

Now let me carefully verify each claimed weakness against the actual paper text:

**Critical Issue 1 (LDS not compared to baselines)**: VALID. The paper introduces LDS as a key contribution but never compares it to random search, grid search, or Bayesian optimization. Only 10 trials from a 6,075+ size space are explored.

**Critical Issue 2 (Average 5.21% improvement is weak/partially contradicted)**: VALID. Table 3 shows negative results on FRED-MD (-7.82%) and slight degradation on ETT hourly and ETT 15min. The average is driven heavily by Exchange Rate.

**Critical Issue 3 (No fixed LoRA baseline)**: VALID but needs nuance. The paper only compares autotuned LoRA vs zero-shot and full fine-tuning. A fixed LoRA config (using the default y^0) would be a natural baseline to isolate the HPO contribution.

**Critical Issue 4 (Full fine-tuning comparison potentially unfair)**: VALID but weakened. The paper states "We also perform full fine-tuning of the Chronos mini model described in Ansari et al. (2024)" - this suggests using the default approach from the Chronos paper, not a tuned one. The comparison conflates HPO effort with method choice.

**Critical Issue 5 ("First paper" claim)**: PARTIALLY VALID. The claim "first paper to explore the potential of autotuning time series transformer models" is indeed broad, but the paper does scope it to LoRA+LDS specifically. However, the claim is still overbroad - AutoML for TS transformers has been explored in various forms.

**Algorithm scoring bug**: The SCORE procedure (line 26) says `if score > best_score then update` — but MASE is a lower-is-better metric. This appears to be a bug in the algorithm. However, this might be intentional if "score" is negated or represents improvement. Let me check more carefully...

Looking at Algorithm 1, line 5 mentions "Evaluation metric MAE" and then the score function computes MASE. Line 26 says "if score > best_score" which would be wrong for error metrics. This is indeed a potential bug but could also be an error metric being negated. The paper isn't clear.

**Spark's point about only one model size**: VALID. Only Mini (20M) is autotuned.

**Human Finder point about data contamination**: The paper actually discusses this - they note that Traffic, Weather, and Electricity overlap with pre-training data (Section 5, first paragraph). So they're aware of this, but the "in-domain vs out-of-domain" analysis could be more rigorous.

Let me also verify the search space size:
5 (alpha) × 3 (dropout) × 5 (rank) × 3 (bias) × 3 (lr) × 3 (batch_size) × 3 (grad_accum)
= 5 × 3 × 5 × 3 × 3 × 3 × 3 = 6,075

So 10 trials covers 0.16% of the space. The neutral reviewer says 12,150 — let me recount. Discrepancy values of 4 and 8. With max discrepancy (8), that means all 7 hyperparameters can change from default, which allows any configuration. But the actual number of evaluated configurations is 10 trials, which is indeed very small.

Now let me write the final review.</think>Now I have sufficient calibration. Let me write the final review.

## Summary

This paper proposes an autotuning framework for pre-trained time series transformers (specifically Chronos T5 Mini) that combines Low Rank Adaptation (LoRA) with Limited Discrepancy Search (LDS) for hyperparameter optimization. The approach searches a discrete space of LoRA hyperparameters (rank, alpha, dropout, bias) and training hyperparameters (learning rate, batch size, gradient accumulation) starting from a default configuration, using LDS to incrementally explore configurations by increasing discrepancy from the default. Experiments on 10 Monash benchmark datasets show the autotuned model achieves an average 5.21% MASE improvement over zero-shot and can outperform full fine-tuning on out-of-domain datasets.

## Strengths

- **Practical and timely problem framing**: Adapting pre-trained time series transformers to new domains with limited resources using PEFT is a relevant and practical concern, and automating LoRA hyperparameter selection is a reasonable goal.

- **Interesting finding that tuned small models can rival larger zero-shot models**: Table 4 shows the autotuned Mini (20M) model matching or exceeding zero-shot Chronos Small (46M) on 6/10 datasets and even beating zero-shot Large (710M) on 3 datasets — this has genuine practical implications for deployment cost.

- **Out-of-domain improvements are notable**: On datasets like Exchange Rate (20.59% MASE improvement) and Australian Electricity (13.89%), the gains over zero-shot are substantial and practically meaningful.

- **Clear experimental setup**: The paper uses well-established benchmarks (Monash archive), clearly reports the search space (Table 2), and includes standard deviations across 5 runs.

## Weaknesses

### Fatal
None.

### Major

- **LDS is never compared to any alternative HPO method**: The paper's algorithmic contribution is specifically "the adoption of LDS for exploring the LoRA hyper-parameter search space in autotuning to minimize computational overhead" (§1). Yet no comparison against random search, grid search, Bayesian optimization, or any other HPO strategy is provided. With only 10 trials from a 6,075-configuration search space (0.16% coverage), there is no evidence that LDS contributes anything beyond the trivial benefit of trying multiple configurations. The "minimize computational overhead" claim (Abstract, §1) is unsubstantiated without wall-clock time comparisons or search efficiency analysis. This undermines the core algorithmic contribution — the paper demonstrates that "some LoRA finetuning often helps," which is already well-established, rather than demonstrating that *LDS specifically* provides value.

- **No fixed-LoRA baseline isolating the autotuning contribution**: The paper compares autotuned LoRA against zero-shot and full fine-tuning, but never against a single fixed LoRA configuration (i.e., the default $y^0$ mentioned in Algorithm 1). Without this baseline, it is impossible to determine whether the observed gains come from the autotuning procedure itself or simply from applying LoRA with any reasonable configuration. Given that "autotuning" is the paper's stated contribution, this omission is significant.

- **Full fine-tuning comparison is potentially confounded by unequal tuning effort**: The autotuned approach searches across 10 configurations (including learning rate, batch size, etc.), while it is unclear whether full fine-tuning receives comparable hyperparameter optimization. If full fine-tuning uses a single default configuration, the comparison conflates "benefit of tuning" with "benefit of LoRA vs. full fine-tuning." The paper does not specify the hyperparameters used for full fine-tuning. This directly weakens one of the headline claims — that the method "outperforms full fine-tuning specifically for out-of-domain datasets."

### Minor

- **Overstated "average 5.21% MASE improvement"**: The average is unweighted across 10 datasets and driven heavily by Exchange Rate (+20.59%). Three datasets show degradation (FRED-MD: −7.82%, ETT Hourly: −0.13%, ETT 15min: −0.56%). No statistical significance tests are provided, and several results (e.g., Weather: 0.818 vs. 0.821; ETT Hourly: 0.783 vs. 0.796) have overlapping standard deviations. The abstract language of "efficiently identifies the optimal configuration" overstates the mixed, dataset-dependent evidence.

- **"First paper" novelty claim is overbroad**: The claim "this is the first paper to explore the potential of autotuning time series transformer models" (§1) is too broad. Hyperparameter optimization for neural time series models (including transformer-based ones) is a well-explored area in AutoML. The novelty should be more precisely scoped (e.g., "first to apply LDS-guided LoRA configuration search to Chronos-type TS transformers") and appropriately hedged.

- **Only one model size is autotuned**: All autotuning experiments use Chronos T5 Mini (20M). Larger models are only evaluated zero-shot. The claim that the approach "can be easily extended to other time series foundation models" (§6) is untested.

- **Potential scoring bug in Algorithm 1**: Line 26 states `if score > best_score`, but MASE is a lower-is-better metric. This seems like a bug (or the score is implicitly negated), which should be clarified.

- **"Highly transferable" claim is unsupported**: The Introduction claims the approach helps make "informed design choices involving LoRA tunable hyper-parameters with strong performance-cost trade-offs that are highly transferable across different target domains," but no cross-domain transfer experiment is conducted (e.g., tuning on Dataset A and applying the selected configuration to Dataset B).

### Trivial

- The paper uses both "LoRA" and "LoRa" (inconsistent capitalization in Tables 2–4 and running text).

## Nice-to-Haves

- Comparison against random search and Bayesian optimization (e.g., Optuna/TPE) with the same 10-trial budget to establish whether LDS is actually a superior search strategy.
- Ablation study varying the number of trials (5, 10, 20, 50) and discrepancy values (δ = 4 vs. 8) to assess sensitivity and cost-performance trade-offs.
- Reporting the default LoRA configuration's performance as a fixed baseline to isolate the value of the HPO procedure.
- Computational cost analysis (wall-clock time, GPU hours, trainable parameter counts) to substantiate "efficiency" claims.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Reproducibility / undisclosed hyperparameters for full fine-tuning**: The harsh critic and human finder note that full fine-tuning hyperparameters are underspecified. While the details could be better reported, this is a minor reproducibility nitpick rather than a methodological flaw; the paper does reference using the approach described in the Chronos paper.

- **Missing comparison with other PEFT methods (adapters, prefix tuning, etc.)**: The paper explicitly scopes its contribution to LoRA-based autotuning, not to comparing PEFT methods. Requesting comparison with other PEFT techniques is scope creep beyond the paper's stated focus.

- **Data contamination / in-domain vs. out-of-domain distinction**: The paper actually addresses this explicitly in §5, noting that "the pre-trained Chronos T5 model has seen datasets from the aforementioned domains during the pre-training phase." The discussion of why full fine-tuning works better for in-domain datasets is present and reasonable.

- **Missing uncertainty quantification / statistical tests**: While significance tests would strengthen the paper, single-run or few-run evaluation is common in the time series forecasting literature, and the paper does report std over 5 runs. This is a nice-to-have, not a major flaw.

- **Experiments on only one model size (Mini)**: While limiting, this is a scope choice. The paper clearly states it uses Mini to "utilize minimal computational resources for demonstrating the applicability of our approach" (§4). This is minor, not major.

## Novel Insights

The most practically interesting finding — which the paper undersells by focusing on the LDS contribution — is that a 20M-parameter model with even basic LoRA tuning can match or beat a 710M-parameter model used zero-shot on several out-of-domain datasets (Table 4). This "small tuned model beats large untuned model" finding for time series is notable and has clear deployment implications. Conversely, the finding that autotuning degrades performance on some datasets (FRED-MD, ETT variants) suggests that LoRA tuning is not universally beneficial and that dataset characteristics (e.g., series length, domain overlap with pre-training) mediate its effectiveness — this deserves deeper analysis than the paper provides.

## Suggestions

- Add a comparison of LDS vs. random search (same 10-trial budget) as a minimal baseline to validate or invalidate the LDS contribution. This is the single most important experiment to add.
- Report the performance of the default LoRA configuration (the $y^0$ in Algorithm 1) as a fixed baseline to isolate the marginal value of the search procedure.
- Either provide full fine-tuning with comparable hyperparameter tuning, or explicitly caveat the "outperforms full fine-tuning" claim by noting that full fine-tuning used a single configuration while autotune searched over 10.
- Reduce the scope of novelty claims (remove or soften "first paper" and "highly transferable") to match the actual evidence.

## Score and Decision

Calibration comparison:
- **PaCA** (novel PEFT method with theoretical analysis + strong empirical evaluation across model sizes): score ~6.0 → Accept Poster
- **GLoRA** (LoRA generalization with structure search, missing some practical comparisons): scores 5-6 → Reject
- **FoundTS** (TS benchmark, limited novelty, overclaim): scores 3-5 → Reject/Withdrawn
- **Enhancing Foundation Models for TS Forecasting** (wavelet tokenization, mixed novelty/evaluation): scores 3-8, avg ~5.5 → Reject
- **AutoML for Sustainable DL** (AutoML + HPO without key comparisons): scores 5-6 → Reject

This paper's core issue is the same as the AutoML for Sustainable DL paper: it proposes an HPO/search method without comparing against the most basic alternative (random search). The empirical finding that LoRA-tuned small models can be competitive with larger zero-shot models is interesting but not novel enough to carry the paper on its own. The LDS contribution is the primary algorithmic novelty claim, and it is entirely unvalidated. Without this validation, the paper reduces to "applying LoRA with some hyperparameter search to time series transformers works okay," which is a useful engineering result but not a research contribution at the level of top venues. The paper is below the GLoRA and PaCA level (both of which offered genuine algorithmic novelty) and closer to the FoundTS/Wavelet-tokenization level, but with an even weaker methodological contribution due to the missing baselines.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>