## Summary

AUTO-RT proposes a reinforcement learning framework for automatic jailbreak strategy exploration that decomposes attack generation into a trainable strategy generation model and a frozen rephrasing model. Two key techniques address sparse-reward challenges: Dynamic Strategy Pruning (DSP), which terminates redundant exploration branches early via constraint checks, and Progressive Reward Tracking (PRT), which uses intentionally safety-weakened "downgrade models" to densify reward signals, with a First Inverse Rate (FIR) metric guiding downgrade model selection. Experiments across 16 white-box and 2 black-box LLMs demonstrate improvements over RL baselines in attack success rate, strategy diversity, and defense generalization.

## Strengths

- **Hierarchical strategy–rephrasing decomposition (Section 2.2):** Separating high-level strategy generation ($AM_g$) from low-level query instantiation ($AM_r$) is a meaningful architectural contribution that enables strategies to generalize across toxic intents rather than overfitting to specific prompts. This is evidenced by the consistent SeD improvements across nearly all models in Table 1.

- **Progressive Reward Tracking with downgrade models (Section 2.3.3):** Using safety-weakened intermediate models to densify sparse binary reward signals is a creative and practically effective solution. The shaped reward $R_s \in \{0, 1, 2\}$ provides graded feedback that vanilla RL lacks. The empirical validation in Figure 4—showing that FIR-guided selection consistently identifies productive downgrade models across six model families—lends credibility to the approach.

- **Multi-dimensional evaluation beyond ASR:** The paper evaluates effectiveness (ASR_tst), semantic diversity (SeD), and defense generalization diversity (DeD), providing a more complete picture of red-teaming capability than single-metric studies. The DeD metric, despite its limitations (see weaknesses), captures a practically important dimension—sustained attack capability under defense updates.

- **Broad experimental coverage:** Testing across 16 white-box models from 6 families, 2 black-box models, and 3 commercial APIs (Appendix G) provides substantial breadth, and the ablation in Table 2 cleanly isolates DSP and PRT contributions across all models.

## Weaknesses

### Major:

- **Inconsistent framing relative to AutoDAN comparison:** Table 3 reveals that on the aggregate ASR metric, AutoDAN achieves 55.23% while AUTO-RT achieves 38.38%—a substantial gap. The abstract claims AUTO-RT "significantly improves success rates (by up to 16.63%)," but this figure appears to be the average improvement over the RL baseline only (which can be verified by computing the average per-model improvement over RL from Table 1: ≈16.63 pp). The abstract does not specify this is versus RL rather than versus all existing methods. Meanwhile, Table 1 excludes AutoDAN entirely, and Section 3.3.3 describes AUTO-RT's ASR as merely "high" despite being 17 pp below AutoDAN. This selective presentation undermines the core effectiveness claim. The paper's genuine strength is in diversity and defense generalization (DeD: 38.19 vs. AutoDAN's 17.88), but the framing obscures the ASR tradeoff.

- **Missing comparison with widely-used adaptive red-teaming methods:** PAIR and TAP are discussed in Related Work as representative adaptive methods using textual feedback, but neither appears in any comparison table. Given their prominence in the red-teaming literature and their claimed advantages over template-based approaches, their exclusion is a significant gap for a paper asserting superiority over "existing methods." Without this comparison, it is unclear whether AUTO-RT's strategy-level exploration offers advantages over iterative prompt-refinement approaches.

- **Unquantified computational cost of downgrade model construction:** PRT requires creating a spectrum of downgrade models ($TM'_1, \ldots, TM'_n$) for each target model. For 16 white-box models, this implies fine-tuning many model instances (the paper uses 6 downgrade levels per target, per Figure 4). The total GPU cost of this setup phase is never reported, making it impossible to assess the efficiency claims ("accelerates discovery") against methods like PAIR or TAP that require no model fine-tuning. The 8×A100 cost quoted in Section 3.1 covers only AM_g optimization, not the downgrade model pipeline.

### Minor:

- **DeD metric's defense construction is underspecified:** Section 3.1 defines Defense Generalization Diversity as evaluating ASR after "constructing defenses based on the successful attacks," but does not specify what defense mechanism is used (adversarial fine-tuning? input filtering? safety training on successful attacks?). Different defense mechanisms would yield qualitatively different DeD scores, making this metric hard to interpret or reproduce without a standardized protocol.

- **ASR_tst computed on top-100 strategies biases toward best-case performance:** Equation 6 evaluates only the top-100 strategies by training-set ASR, which measures peak rather than average policy quality. A method that occasionally discovers a highly effective strategy but produces mostly ineffective ones would score well. This is not necessarily wrong, but it should be acknowledged as measuring upper-bound rather than expected performance.

- **Non-potential-based reward shaping risks policy divergence (Section 2.3.3):** The authors acknowledge that PRT "does not follow the potential-based function structure" of Ng et al. (1999), meaning the shaped reward could in principle change the optimal policy. The empirical results suggest this is not a practical issue, but no convergence analysis or stability study is provided. For models where the downgrade model's safety distribution deviates significantly from the target's, the shaped reward could actively mislead optimization.

- **R2D2 and Mistral counterexamples downplayed:** On R2D2, Few-Shot achieves 27.18% ASR vs. AUTO-RT's 12.45%; on Mistral 7B, Imitate Learning achieves 54.88% vs. AUTO-RT's 52.65%. The paper mentions R2D2's robustness but does not analyze what makes AUTO-RT less effective on these models, limiting understanding of the method's boundary conditions.

### Trivial:

- **FIR definition could be clearer:** The "inverse element" terminology (Section 2.3.3) is somewhat opaque. A simpler explanation—FIR identifies the degradation level where safety collapse becomes erratic rather than monotonic—would improve accessibility without loss of precision.

## Nice-to-Haves

- Ablation isolating FIR-guided downgrade selection vs. random or fixed downgrade model selection, to directly validate FIR's contribution beyond PRT's general mechanism.
- Per-category ASR breakdown on HarmBench (e.g., violence, misinformation, cybercrime) to reveal whether AUTO-RT's gains are uniform or concentrated in specific vulnerability types.
- Computational cost breakdown (GPU hours for downgrade model creation vs. RL training) to substantiate efficiency claims.
- Comparison with PAIR or TAP on the same benchmark, even if only on a subset of models.
- Analysis of strategy novelty: clustering generated strategies against known jailbreak templates to verify that AUTO-RT discovers genuinely new attack patterns rather than rephrasing known ones.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Equation formatting issues** (from Harsh Critic) — These are OCR/parser artifacts, not paper problems. Removed per formatting nitpick rule.
- **Weakness: Missing related works (specific unnamed methods)** (from Spark Finder) — Cannot confirm existence of uncited works. Removed per hard rule on missing related works.
- **Weakness: Exact hyperparameters for downgrade model fine-tuning** (from Harsh Critic, reproduced by others) — This is a reproducibility nitpick about implementation details impractical to include in a submission. Removed per hard rule.
- **Weakness: Ethical concern about reproducing harmful outputs in case studies** (from Review 2) — Standard practice in red-teaming papers; demonstrating effectiveness requires showing actual outputs. This is not a paper-specific weakness.
- **Weakness: "Circular evaluation" using LLM judges** (from Harsh Critic's transferred points) — Using LlamaGuard for both reward and evaluation is standard in this field; the paper partially addresses this in Appendix C.1 with an alternative classifier showing stable results. The concern is generic to the entire field rather than specific to this paper.
- **Weakness: Demand for confidence intervals / statistical tests** (from Spark Finder) — Large-scale RL benchmarks in this community typically report single-run results; demanding statistical testing is not the field's standard. Moved to nice-to-have territory.

## Novel Insights

The hierarchical strategy–rephrasing decomposition reveals an interesting asymmetry: strategies that are individually mediocre can become highly effective when composed with intent-specific rephrasing, which explains why AUTO-RT's diversity advantage (SeD) translates more reliably into defense generalization (DeD) than into raw ASR. This suggests that the red-teaming community's focus on single-attack success rates may be measuring the wrong thing—what matters for practical vulnerability assessment is the breadth of the attack surface (how many distinct strategies work), not the depth of any single attack. The FIR metric's identification of a "sharp transition" in model safety degradation also hints at a phase-transition-like phenomenon in safety alignment that deserves further theoretical study.

## Suggestions

- **Reframe the abstract and claims to be precise about what AUTO-RT improves over:** State explicitly that the 16.63% average improvement is over the RL baseline, acknowledge AutoDAN's higher raw ASR, and foreground the diversity/defense-generalization advantages as the primary contribution.
- **Add a PAIR or TAP comparison** on at least 4–6 models, even as a supplementary result, to situate AUTO-RT against the most relevant adaptive baselines.
- **Report total computational cost** including downgrade model construction, so readers can assess the practical efficiency tradeoff.
- **Specify the defense mechanism used in DeD evaluation** (even a one-line description in Appendix) so the metric is reproducible.
- **Analyze the R2D2 and Mistral counterexamples** to identify boundary conditions where strategy-level exploration is less effective than direct prompt-level methods.