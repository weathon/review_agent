## Summary
This paper proposes an autotuning framework for adapting pretrained time-series transformers, combining LoRA-based parameter-efficient fine-tuning with Limited Discrepancy Search (LDS) to select LoRA hyperparameters under a small trial budget. Experiments on 10 Monash/Chronos benchmark datasets using Chronos-T5 Mini show that tuned LoRA often improves over zero-shot inference and is competitive with full fine-tuning on several datasets, with especially strong gains on Exchange Rate, Australian Electricity, and M5.

## Strengths
- **Practical problem setting with clear motivation.** The paper targets a real deployment bottleneck: adapting pretrained time-series foundation models to new domains without full fine-tuning. This is a worthwhile and practically relevant problem.
- **Reasonable empirical coverage across datasets.** The evaluation spans 10 benchmark datasets from multiple domains (transport, weather, energy, finance, economics, retail), which is stronger than a single-dataset demonstration and gives some sense of robustness.
- **Some genuinely promising results.** Table 3 shows notable gains over zero-shot on several datasets, including Australian Electricity (0.965 → 0.831), Exchange Rate (2.054 → 1.631), and M5 (0.942 → 0.925). The tuned Mini model also beats zero-shot larger Chronos variants on some datasets in Table 4, which is practically interesting for low-resource users.
- **Variance is at least partially reported.** The paper reports mean ± std over 5 runs in Table 3, which is better than single-run reporting and allows some assessment of stability.
- **Clarity of the high-level idea.** The basic framing—start from a default LoRA configuration, explore nearby configurations with LDS, and select using validation performance—is easy to understand.

## Weaknesses
###: Fatal
- **The central methodological claim about LDS-based autotuning is not validated.**  
  The paper’s main contribution is not merely “LoRA helps,” but specifically that **LoRA + LDS autotuning** is an effective and efficient search strategy. However, the experiments do not compare LDS against random search, grid search, Bayesian optimization/TPE, or even a fixed/default LoRA configuration. As a result, the paper cannot attribute gains to LDS rather than to LoRA tuning in general. The evidence currently supports only the weaker claim that **some LoRA tuning can help Chronos on some datasets**.

### Major:
- **Claims of efficiency and “strong performance-cost trade-offs” are unsupported by the presented evidence.**  
  The abstract claims “strong performance-cost trade-offs,” and Section 1 says LDS is adopted “to minimize computational overhead,” but the paper provides no cost analysis: no wall-clock time, memory, trainable parameter counts, search cost, or comparison of total compute against full fine-tuning or other HPO methods. In fact, running 10 LoRA trials may or may not be cheaper than one well-tuned full fine-tuning run; the paper does not quantify this.
- **The paper overstates superiority over full fine-tuning and out-of-domain robustness.**  
  Table 3 is mixed, not decisive. Full fine-tuning is better on Traffic, Weather, ETT (Hourly), and NN5; zero-shot is best on FRED-MD and ETT (15 min.); autotune is best on only 4/10 datasets. The paper repeatedly claims that it “outperforms full fine-tuning specifically for out-of-domain datasets,” but “out-of-domain” is only argued informally in Section 5 rather than rigorously defined or operationalized. The empirical pattern is promising on some datasets, but the conclusion is stronger than the evidence.
- **Algorithm 1 is internally inconsistent enough to weaken confidence in the methodological core.**  
  Since the search procedure is the paper’s novelty, the pseudocode matters. Yet Algorithm 1 has several inconsistencies with the text: the input names the evaluation metric as **MAE**, while the paper reports **MASE**; the SCORE procedure appears to train using \(y^*\) rather than the candidate configuration \(y\); and the update condition uses `score > best_score` even though the text states the goal is the **lowest MASE**. These may be pseudocode mistakes rather than implementation bugs, but because the contribution is the search algorithm itself, this imprecision is a substantive issue.
- **The search-budget story is underexplained and does not establish efficiency.**  
  The paper states that only 10 LDS-selected trials are run over a 7-hyperparameter discrete space (Table 2), while varying maximum discrepancy between 4 and 8. But it never reports which discrepancy setting produced the final results, how many candidate configurations LDS enumerated, how the 10 configurations were selected from the search tree, or whether 10 trials are sufficient across datasets. Without budget-performance curves, search traces, or comparisons to alternative strategies under the same budget, the “efficient autotuning” claim remains unsubstantiated.

### Minor
- **The results are promising but not consistently strong, and several differences are small relative to reported variance.**  
  Some improvements are large, but some are negligible or negative: e.g., ETT (Hourly) 0.795 → 0.796, FRED-MD 0.473 → 0.510, ETT (15 min.) 0.709 → 0.713. Given the reported standard deviations, several comparisons do not look decisive. This does not invalidate the paper, but it does weaken the stronger narrative of broad superiority.
- **The paper does not isolate whether gains come from LoRA itself, the hyperparameter search, or specific hyperparameters.**  
  There is no ablation of default LoRA vs tuned LoRA, nor any analysis of which hyperparameters matter most. That makes the contribution less informative for practitioners.
- **Generality beyond Chronos Mini is only weakly supported.**  
  The actual autotuning experiments are conducted only on Chronos-T5 Mini; larger Chronos sizes are used only for zero-shot comparison. This is acceptable as a scoped study, but it weakens claims about autotuning “time series transformers” broadly.
- **Some interpretive claims in the Results section are speculative.**  
  For example, the explanation that full fine-tuning wins on certain datasets because Chronos “has seen datasets from the aforementioned domains during the pre-training phase” is plausible, but not demonstrated by any controlled analysis in this paper.

### Trivial
- **Terminology around MASE is sloppy in one place.**  
  Section 4 refers to “mean absolute squared error (MASE),” which is incorrect terminology. This is minor on its own, but together with the pseudocode inconsistencies it contributes to ambiguity about the optimization objective.
- **Some figures add limited value beyond the tables.**  
  Figures 3–6 are mostly descriptive summaries of Table 3/4 and qualitative examples; they do not add analysis of why LDS helps or fails.

## Nice-to-Haves
- Add a **10-trial random search baseline** and, if feasible, a simple Bayesian/TPE baseline under the same budget.
- Include **default LoRA** and perhaps a hand-tuned fixed LoRA configuration to isolate the benefit of search.
- Report **compute and efficiency metrics**: training time, search time, number of trainable parameters, and memory.
- Clarify the **LDS implementation details**: discrepancy schedule, stopping rule, and how exactly the 10 trials are drawn.
- Provide a small **hyperparameter importance analysis** and failure-case analysis on datasets where tuning hurts.
- Test autotuning on at least one larger Chronos variant to support broader generalization claims.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper is not novel because it combines existing tools.”**  
  This is too blunt as stated. While the paper is largely an application/integration paper rather than a deep algorithmic advance, combining LoRA with LDS for this time-series adaptation setting is still a legitimate contribution. The real issue is not “no novelty at all,” but rather that the paper does not adequately validate that the LDS component materially helps.
- **Complaints about missing unrelated PEFT baselines (adapters, prefix-tuning, prompt-tuning).**  
  The paper scopes itself to LoRA-based autotuning; demanding a full PEFT comparison is somewhat outside scope. Such comparisons would strengthen the work, but their absence is not a core flaw.
- **Pure reproducibility nitpicks about omitted implementation details or code release.**  
  These are not central enough to include as main weaknesses under the stated review rules.
- **Any criticism questioning the existence/release status of cited models or resources.**  
  Removed per instruction.

## Novel Insights
The most important synthesis is that this paper is **closer to a promising empirical observation than a validated autotuning method**. The data do suggest that lightweight LoRA adaptation of Chronos Mini can be practically valuable, especially on some target datasets where zero-shot performance is weak. But the evidence does not yet distinguish between three different claims: (i) LoRA helps, (ii) tuning LoRA helps, and (iii) **LDS is a particularly effective way to tune LoRA**. Right now, the experiments mainly support (i)/(ii), while the paper is written as if it has established (iii). Narrowing the claims to what is actually demonstrated would substantially improve the paper’s credibility.

## Suggestions
- Add a **same-budget search baseline** (at minimum random search with 10 trials) to directly test whether LDS is providing value.
- Fix and rewrite **Algorithm 1** so it correctly matches the implemented objective and search procedure.
- Report **compute cost** and parameter-efficiency metrics to support the efficiency claims.
- Tone down claims of superiority over full fine-tuning and “out-of-domain” robustness unless backed by a more rigorous definition and analysis.
- Add an ablation comparing **default LoRA vs tuned LoRA vs tuned LoRA with LDS**.
- Clarify the validation/test protocol explicitly to show that configuration selection is based only on validation performance.

## Score and Decision
**Originality:** Moderate-low. The paper combines known ingredients in a sensible way, but the algorithmic advance is limited and insufficiently isolated.  
**Importance of research question:** Moderate to high. Efficient adaptation of time-series foundation models is practically important.  
**Whether claims are well supported:** Weak to moderate. Several of the strongest claims—especially about LDS effectiveness and efficiency—are not established by the experiments.  
**Soundness of experiments:** Moderate at best. The dataset coverage is decent, but the lack of HPO baselines and cost analysis leaves the central contribution under-evaluated.  
**Clarity of writing:** Moderate. The high-level story is clear, but the pseudocode and some terminology are inconsistent.  
**Value to the research community:** Moderate if reframed; in its current form, limited by overclaiming relative to evidence.

For calibration, I compared this paper to:
- **D9liZ0D8z8 (WaveToken, Reject, scores 5/6/8/3):** that paper had broader experimentation and ablations but mixed evidence vs claims; the present paper is somewhat weaker because its core methodological claim is less well isolated.
- **xW4J2QlqRx (ContextFormer, Reject, scores 5/5/5/5):** similar in being practically motivated with reasonable experiments but lacking the most decisive comparisons for the core claim; this paper feels in that same reject-but-not-hopeless band.
- **TqYjhJrp9m (Zero-shot forecasting of chaotic systems, Accept, scores 8/6/6):** this accepted paper offered a much more decisive empirical story and clearer contribution-validation alignment. The present submission falls clearly below that bar.
- **GDjwSBZy6l (ROLoRA, Reject, scores 5/6/5/5):** similar pattern of promising LoRA-related idea with incomplete evidence around efficiency/scalability; the current paper is in a comparable range, perhaps slightly weaker due to the absence of search baselines despite centering the search method.

Overall, this is **not a fundamentally broken paper**, and there is a real practical signal in the results. But because the paper’s central claim is about **LDS-based autotuning**, and that claim is not actually validated, I land on a **reject in the lower-middle range rather than a borderline accept**.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>