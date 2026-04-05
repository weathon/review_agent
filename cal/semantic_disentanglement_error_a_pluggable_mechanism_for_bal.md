=== CALIBRATION EXAMPLE 9 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Only partially. The title foregrounds “semantic disentanglement error” and “balanced contrastive time-series representation,” which matches the proposed diagnostic plus weighting idea. However, the paper’s actual method is more specific: it starts with an SDE diagnostic, then a CoST-based asymmetric weighting mechanism. The title does not clearly signal that the main positive result is a **reweighting strategy built on CoST**, not a general plug-in mechanism for any contrastive model.
- **Does the abstract clearly state the problem, method, and key results?**  
  The problem and high-level idea are clear. The abstract states semantic imbalance in contrastive time-series learning, defines SDE, and says it is integrated into adaptive weighting. That said, it is vague about what exactly is novel beyond CoST adaptation, and it does not state the key experimental setting or how strong the gains are.
- **Are any claims in the abstract unsupported by the paper?**  
  Yes, at least in terms of evidential strength. The abstract claims “consistent gains in forecasting accuracy and representational robustness, especially under semantic skew conditions,” but the paper’s experimental section is thin and does not convincingly establish robustness under controlled skew beyond a synthetic probe. There is also a mismatch between the abstract’s “plug-gable mechanism” and the implementation, which appears tailored to CoST’s seasonal/trend decomposition.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  The motivation is plausible: weak periodic signals can be overshadowed by dominant trend components in time-series representations. This is a real issue for ICLR-level work. However, the gap in prior work is not crisply separated from existing decomposition-based forecasting and contrastive time-series methods. The introduction claims three limitations of current contrastive approaches, but these are presented more as assertions than as a carefully argued gap grounded in prior evidence.
- **Are the contributions clearly stated and accurate?**  
  The contribution list is not fully accurate relative to the body. The paper claims to “diagnose the semantic imbalance problem in TS2Vec and related contrastive methods,” provide “systematic ablations,” and propose a “semantic rebalancing framework.” But the method section and experiments are mostly centered on **TS2Vec diagnostics** and **CoST + APW**. The “systematic ablations” are not actually shown in a convincing, comprehensive way.
- **Does the introduction over-claim or under-sell?**  
  It over-claims in several places. For example, it suggests the paper identifies “three fundamental limitations” of current SSL time-series learning, but only one of these is really demonstrated with any direct evidence, and even that evidence is limited. The introductory discussion of “embedding collapse” and “time-domain-only objectives” is conceptually reasonable, but the paper does not fully substantiate that these are distinct causal factors in the way implied.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  Not fully. The method has the right high-level pieces, but key implementation details are missing or ambiguous. The paper first defines SDE in Section 3.2 as  
  \[
  \mathrm{SDE}_{a,b} = 1 - \cos\big(v(a+b)-v(b),\,v(a)\big),
  \]
  then later changes the operational definition in Section 4.4.3 to an asymmetry factor based on \(\mathrm{SDE}_{b,a}-\mathrm{SDE}_{a,b}\). It is not clear how these quantities are computed in practice during training, whether they are detached, minibatch-estimated, or precomputed.
- **Are key assumptions stated and justified?**  
  The key assumption is that a time series can be meaningfully decomposed into “trend” and “periodic” components, and that these correspond to recoverable semantic subspaces in embedding space. That assumption is reasonable for some benchmark datasets, but the paper does not justify when it holds or when it fails. This matters because the method depends on classical decomposition and may not extend to non-seasonal domains.
- **Are there logical gaps in the derivation or reasoning?**  
  Yes. The biggest gap is the use of **SDE as both diagnostic and optimization signal**. In Section 4.3, the paper explicitly reports that direct SDE regularization does not help and hypothesizes that it does not provide constructive gradients. Yet the final method still uses SDE-derived quantities to drive loss reweighting. It is not explained why the diagnostic is unsuitable as a direct regularizer but suitable as a control signal for weighting. Also, the relationship between “recoverability” and the proposed cosine-based metric is asserted, not derived.
  
  Another gap is the composite-embedding construction in Section 4.4.2. The method defines \(v(a+b)=g_\phi([v(a)\|v(b)])\), but then the loss in 4.4.3 reweights CoST’s seasonal and trend contrastive losses using SDE computed from this composite representation. The causal loop is not clearly specified: is the MLP used only for measurement, or is it part of the learned model optimized end-to-end to improve SDE?
- **Are there edge cases or failure modes not discussed?**  
  Several important ones:
  - If the decomposition into trend and seasonality is poor, SDE may be meaningless or misleading.
  - If both components are weak or entangled, the asymmetry factor may be unstable.
  - The method seems tailored to additive decompositions; it is unclear how it generalizes to multiplicative or regime-switching signals.
  - The dynamic weighting could amplify noisy estimates of SDE and introduce training instability, especially early in training.
- **For theoretical claims: are proofs correct and complete?**  
  There are no real proofs. The paper introduces a metric and a weighting heuristic, but the theoretical justification remains informal. For an ICLR submission, that is acceptable only if the empirical support is very strong; here it is not.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Only partially. The synthetic SDE experiment in Table 1 does test the claim that semantic imbalance can be observed as component amplitude ratios vary, but it largely confirms an intuitive construction rather than validating the representation-learning method. The forecasting results are meant to support the main claim, but the experimental story is incomplete.
- **Are baselines appropriate and fairly compared?**  
  The baseline set is too narrow for the claims made. Only TS2Vec, TNC, and CoST are included, even though the introduction and related work invoke a broader landscape of forecasting and representation learning methods. For an ICLR paper, the absence of stronger and more recent forecasting baselines, especially decomposition- and frequency-based models, weakens the claims substantially. Also, the paper does not clearly distinguish whether the comparison is against self-supervised encoders or forecasting architectures trained in the same downstream protocol.
- **Are there missing ablations that would materially change conclusions?**  
  Yes, several:
  - No ablation separating the effect of **frequency-aware decomposition** from the effect of **asymmetric reweighting** in the final CoST+APW method.
  - No ablation on the **fusion MLP** in Section 4.4.2.
  - No sensitivity analysis for the hyperparameters \(\gamma\) and \(\gamma'\).
  - No analysis of how performance changes when the decomposition is replaced with simpler filters or when SDE is estimated differently.
  - No evidence that the proposed weighting is better than simpler alternatives such as fixed imbalance correction, uncertainty weighting, or inverse-loss scaling.
- **Are error bars / statistical significance reported?**  
  No. This is a serious weakness. The paper reports single-point MSE/MAE values without variance across seeds. Given the modest magnitude of many differences, lack of statistical reporting makes it hard to judge whether improvements are reliable.
- **Do the results support the claims made, or are they cherry-picked?**  
  The results are suggestive but not convincing enough for ICLR. Table 1 shows the intended synthetic asymmetry trend, but the real-task results are hard to evaluate because the paper’s main result table is incomplete in the text extract and, even if intact, the narrative does not provide enough context about significance, average rank, or consistent improvement across all horizons. The claim of “consistent gains” is too strong without statistical support and broader baseline coverage.
- **Are datasets and evaluation metrics appropriate?**  
  The datasets are standard and appropriate for time-series forecasting. However, the paper’s core claim is about **representation balance**, and forecasting MSE/MAE alone are indirect proxies. The SDE metric is closer to the claimed mechanism, but it is custom and not validated against downstream semantic fidelity in any rigorous way. The paper would benefit from a clearer evaluation of whether improved SDE actually correlates with more faithful component recovery or robustness under controlled semantic skew.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  Yes, primarily the method section. The paper switches terminology between “Semantic Disentanglement Error” and “Semantic Separability Error,” and later “Semantic Decomposition Error,” which makes it hard to tell whether these are distinct concepts or the same metric under different names. More importantly, the exact training pipeline is not fully clear: what is measured, what is optimized, when SDE is computed, and how the MLP interacts with CoST.
  
  The explanation of the causal relationship among decomposition, SDE, and asymmetric weighting is also confusing. Section 4.3 says direct regularization fails, while Section 4.4 implies that SDE can still drive a successful adaptive objective. The narrative needs a clearer mechanistic explanation.
- **Are figures and tables clear and informative?**  
  The figure description is helpful at a high level, but the paper relies heavily on the figure to explain the architecture, and the textual description is not sufficiently precise on its own. Table 1 is useful as a diagnostic probe. Table 2 is much less informative because it is presented as a narrow comparison without enough context or statistical variation. The final results table should be central, but as presented in the text it does not give a sufficiently transparent experimental picture.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Only partially. The conclusion briefly mentions future work on using low-pass filtering without explicit frequency-domain encoders and probing the fusion MLP, but this is framed more as future directions than limitations.
- **Are there fundamental limitations they missed?**  
  Yes. The method is strongly dependent on a particular decomposition into trend and seasonality. That limits applicability to domains where such a decomposition is not natural. Also, SDE is defined through cosine similarity between differences of learned embeddings, so it can be sensitive to scale, training stage, and embedding geometry.
- **Are there failure modes or negative societal impacts not discussed?**  
  No broader societal impact discussion is present. That is not necessarily a major issue for this paper, but the authors do not discuss possible negative failure modes such as misleading confidence in decomposed signals, poor generalization to nonstationary or safety-critical data, or the risk of overfitting to benchmark-specific periodic structure.

### Overall Assessment
This paper raises a genuinely interesting and relevant problem for time-series SSL: dominant components can suppress weaker but important semantics. The SDE diagnostic is a plausible and potentially useful way to quantify imbalance, and the idea of using it to inform asymmetric loss weighting is sensible. However, the paper falls short of ICLR’s stronger bar in several ways: the method description is not yet reproducible enough, the theoretical justification is thin, the experimental evaluation is too limited and under-reported, and the main empirical claims are not supported with sufficient breadth or statistical evidence. As it stands, the contribution is promising but not yet convincing enough for acceptance without substantial strengthening of the experimental validation and a clearer, more rigorous method exposition.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a plug-in mechanism for balancing semantic components in contrastive time-series representation learning. The core idea is a Semantic Disentanglement/Separability Error (SDE) that measures whether one component (e.g., trend or seasonality) can be recovered from a composite embedding, and then uses the resulting asymmetry to reweight seasonal and trend contrastive losses in CoST-like models.

The intended contribution is to diagnose and mitigate “semantic imbalance” in time-series SSL, especially when weak periodic signals are suppressed by dominant trends. The paper reports forecasting gains and lower SDE on benchmark datasets, while also noting that directly regularizing SDE is less effective than using it for adaptive loss weighting.

### Strengths
1. **Addresses a plausible and practically relevant failure mode.**  
   The paper targets a meaningful issue in time-series SSL: dominant trend components can overwhelm weaker periodic structure. This is a real concern in forecasting and representation learning, and the paper’s motivation aligns with ICLR’s interest in understanding representation failures beyond benchmark gains.

2. **The plug-in framing is attractive for reuse.**  
   The method is presented as an add-on to existing frameworks like CoST, without requiring architectural changes. That is appealing from an adoption standpoint and potentially increases the method’s practical relevance.

3. **The diagnostic idea is simple and interpretable.**  
   SDE is conceptually easy to understand: it measures whether the embedding of a composite signal preserves the semantics of its parts via cosine similarity. ICLR typically values simple, interpretable probes when they illuminate hidden properties of learned representations.

4. **The paper attempts to separate diagnosis from intervention.**  
   A notable positive is that the authors do not assume the diagnostic metric is itself a good regularizer. They explicitly test direct SDE regularization and report that it is ineffective, which is a useful empirical lesson and suggests some methodological care.

5. **Focus on asymmetric semantic recoverability is interesting.**  
   The asymmetry between trend and periodic recoverability is a concrete hypothesis with an intuitive interpretation. If validated more rigorously, this could be a useful lens for analyzing decomposition-based time-series encoders.

### Weaknesses
1. **The novelty appears modest relative to existing frequency-aware and decomposition-based time-series methods.**  
   The paper builds heavily on CoST-style trend/seasonality decomposition and then adds an asymmetric weighting rule driven by a recovery metric. This feels more like a refinement of known decomposition + reweighting ideas than a clearly new learning paradigm, especially given prior frequency-aware and multi-view SSL work cited in the paper.

2. **The method is under-specified and raises reproducibility concerns.**  
   Key implementation details are missing or unclear: how exactly SDE is computed in practice, how decomposition is performed for real datasets, how the MLP fusion is trained, what values are used for the hyperparameters, and how the weights are stabilized to avoid negative or exploding scaling. For an ICLR submission, this level of ambiguity is a significant weakness.

3. **The theoretical justification is weak.**  
   SDE is introduced as an interpretability metric, but there is no analysis showing that low SDE corresponds to useful downstream representations, nor any optimization argument for why the proposed asymmetric weighting should reliably improve training. The leap from a diagnostic cosine measure to an adaptive objective is largely heuristic.

4. **The empirical evaluation is not sufficiently convincing as presented.**  
   The paper reports forecasting results, but the experimental section is incomplete and hard to assess rigorously. The comparison set is limited, there is little evidence of statistical significance or variance across runs, and the paper does not clearly demonstrate gains across a broad range of settings beyond the cited benchmarks.

5. **The paper does not adequately isolate the contribution of each component.**  
   The method combines multiple ideas: decomposition, SDE, MLP fusion, and asymmetry-aware reweighting. Although the authors mention an ablation where direct SDE regularization fails, it is not enough to disentangle which part actually drives improvements. An ICLR reviewer would likely expect cleaner ablations showing the marginal contribution of each module.

6. **Some claims are stronger than the evidence shown.**  
   The paper claims “consistent gains” and “superior robustness,” but the presented evidence does not convincingly establish robustness under semantic skew conditions beyond synthetic examples and a limited set of benchmarks. The relationship between the synthetic SDE study and real-world forecasting improvements remains somewhat speculative.

7. **The framing is somewhat confusing and terminology is inconsistent.**  
   The paper alternates between “Semantic Disentanglement Error” and “Semantic Separability Error,” and the exposition mixes diagnostic and optimization objectives in a way that obscures the core method. This makes the contribution harder to follow and weakens clarity.

### Novelty & Significance
**Novelty: Low to moderate.**  
The main idea is a reasonable combination of semantic recoverability scoring and asymmetric loss reweighting, but it does not appear to introduce a fundamentally new modeling principle. Much of the conceptual backbone is inherited from prior decomposition-based contrastive time-series learning, especially CoST.

**Significance: Moderate if the method is validated more thoroughly.**  
The problem is important, and a lightweight plug-in that improves weak-component preservation could be practically useful. However, based on the current presentation, the empirical and methodological support is not strong enough to meet ICLR’s usual acceptance bar for a clearly significant advance.

**Clarity: Mixed.**  
The high-level motivation is understandable, but the method description is inconsistent and sometimes difficult to parse. This would likely hinder reviewer confidence at ICLR.

**Reproducibility: Below ICLR expectations as written.**  
The paper lacks enough detail for straightforward reproduction, especially regarding decomposition, training setup, and hyperparameter choices.

### Suggestions for Improvement
1. **Provide a rigorous, end-to-end algorithm with all hyperparameters and implementation details.**  
   Include exact computation of SDE, decomposition method, weighting schedule, clamping or normalization of ∆, and training protocol. This is essential for ICLR-level reproducibility.

2. **Strengthen ablations substantially.**  
   Separate the effects of:  
   - decomposition alone,  
   - SDE computation alone,  
   - MLP fusion alone,  
   - asymmetric weighting alone,  
   - each hyperparameter choice.  
   Show whether the method still helps when any one component is removed.

3. **Add stronger baselines and statistical reporting.**  
   Compare against more recent representation-learning and decomposition-based forecasting methods, and report mean/std over multiple seeds with significance tests where appropriate.

4. **Clarify the relationship between SDE and downstream utility.**  
   Empirically test whether lower SDE correlates with better forecasting, classification, or anomaly detection performance across many settings. If this is the central claim, it should be validated directly.

5. **Improve the theoretical framing of the metric and weighting rule.**  
   Give a more principled justification for why cosine-based recoverability should be the right notion of “semantic balance,” and why the proposed linear reweighting should optimize it effectively.

6. **Clean up terminology and presentation.**  
   Use one consistent term for the metric, define all symbols once, and streamline the explanation of how the method differs from CoST. This would materially improve readability and reviewer trust.

7. **Test robustness beyond the current benchmark setting.**  
   Since the paper’s core claim concerns semantic imbalance, add experiments under controlled skew, varying noise, missingness, and different periodic/trend ratios, ideally across multiple datasets and downstream tasks.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a full comparison against stronger and more directly relevant baselines, especially CoST with standard decomposition variants, TS2Vec+frequency augmentations, and recent time-series SSL methods beyond TS2Vec/TNC. On ICLR standards, claiming a new mechanism is not convincing if it is only compared to a narrow baseline set.  
2. Add end-to-end downstream evaluations on standard representation-learning tasks beyond forecasting, especially classification and anomaly detection on widely used benchmarks. The paper claims a general representation benefit, but forecasting alone does not establish that the method improves semantic disentanglement broadly.  
3. Add ablations isolating each proposed ingredient: SDE score only, asymmetric weighting only, MLP fusion only, frequency/decomposition view only, and the combination. Without this, it is unclear which part actually drives the gains and whether the “plug-in mechanism” is doing anything novel.  
4. Add a comparison to simple loss reweighting and uncertainty-based weighting baselines. If performance improvements come from generic adaptive weighting rather than SDE-specific semantics, the core claim about semantic disentanglement is weakened.  
5. Add robustness experiments under controlled semantic skew and noise/interference settings on real and synthetic data. The paper’s main claim is about balancing weak vs. dominant components, so it needs evidence that gains persist when periodicity is weak, corrupted, or varies across regimes.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a validity analysis of SDE itself: correlate SDE with human-interpretable component recoverability, downstream performance, and linear probing of trend/seasonal information. Without showing SDE measures what it claims, the entire weighting mechanism is on uncertain ground.  
2. Analyze whether the method truly disentangles semantics or just changes loss scale. Show embedding geometry, component-specific retrieval/probing, and whether trend/seasonal information is actually more linearly separable after training.  
3. Report sensitivity to the SDE estimator and decomposition procedure, including different low-pass filters, window sizes, and decomposition choices. If the metric and training signal depend heavily on a particular decomposition heuristic, the method is not robust enough for ICLR.  
4. Analyze training dynamics of the adaptive weights over epochs and across datasets. The paper claims on-the-fly balancing, but does not show whether weights behave stably, saturate, or oscillate in ways that would undermine reproducibility or interpretability.  
5. Include significance tests or multiple-run variance. The reported improvements appear modest in several places, and ICLR reviewers will expect evidence that gains are reliable rather than seed-sensitive.

### Visualizations & Case Studies
1. Show before/after spectral plots and time-domain reconstructions for examples where the baseline suppresses weak periodicity and the method preserves it. This would directly reveal whether the approach actually recovers the claimed minor semantic component.  
2. Visualize the learned embedding space with trend-dominant and season-dominant samples, plus probes for how much each component is recoverable from the embedding. Without this, the semantic disentanglement claim remains abstract.  
3. Plot the evolution of SDE and adaptive weights during training on representative datasets. This would expose whether the method is genuinely balancing semantics or simply converging to a fixed heuristic weighting.  
4. Provide failure cases where the method does not help, especially on signals with ambiguous decomposition or weak periodic structure. ICLR reviewers will expect to know when the mechanism breaks, not only where it succeeds.  

### Obvious Next Steps
1. Validate the approach on more diverse and harder time-series domains, especially those with non-stationary, irregular, or multiscale structure. The current benchmark set is too narrow to justify broad claims about semantic imbalance.  
2. Test whether the method transfers to other SSL backbones besides CoST, since the paper claims plug-compatibility but only demonstrates one integration point.  
3. Replace heuristic decomposition assumptions with a principled, learnable decomposition module and compare it to the current low-pass/residual setup. If the method depends on hand-crafted separation, its practical contribution is limited.  
4. Evaluate computational overhead and stability relative to the baselines. A plug-in method for contrastive learning should show that the added metric and MLP do not materially hurt training cost or convergence.

# Final Consolidated Review
## Summary
This paper targets a real issue in contrastive time-series representation learning: dominant components like trend can suppress weaker but useful semantics such as seasonality. It proposes an interpretable recovery metric, SDE, and uses the resulting asymmetry to reweight CoST-style seasonal/trend contrastive losses, with the goal of improving balance between components.

## Strengths
- The paper identifies a plausible and practically relevant failure mode in time-series SSL: weak periodic structure being washed out by dominant trend. This is well-motivated and aligned with real forecasting settings.
- The diagnostic idea is simple and interpretable. SDE is meant to quantify whether a composite embedding preserves recoverable information from its constituent components, and the synthetic amplitude-ratio experiment does at least show the intended asymmetry trend.

## Weaknesses
- The method is under-specified and not reproducible enough. The paper uses inconsistent terminology for SDE, does not clearly specify how SDE is computed during training, and leaves ambiguous how the composite MLP, decomposition, and adaptive weighting interact end-to-end. This matters because the central contribution is a heuristic control signal, not a principled objective.
- The empirical support is too thin for the strength of the claims. The baselines are narrow relative to the paper’s scope, there are no reported variance estimates or significance tests, and the key result is not supported by a sufficiently broad ablation suite. In particular, the paper does not cleanly isolate the effect of decomposition, MLP fusion, and SDE-based reweighting.
- The core causal leap remains unproven: the paper shows that direct SDE regularization is ineffective, but then still uses SDE to drive adaptive loss weights without explaining why this should work. The link between the diagnostic and the optimization rule is largely heuristic.

## Nice-to-Haves
- A clearer end-to-end algorithm box with exact training steps, hyperparameters, and the precise computation of the asymmetry factor.
- More diagnostics showing whether lower SDE actually correlates with improved component recoverability or downstream performance.
- Sensitivity analyses for the decomposition filter, the weighting hyperparameters, and training stability over epochs.

## Novel Insights
The most interesting aspect of the paper is that it does not treat decomposition as a binary architectural choice, but instead tries to measure whether a learned representation is *asymmetrically recoverable* across semantic components, then feeds that imbalance back into optimization. That is a genuinely useful framing if validated further. However, the current version mostly demonstrates a heuristic that works on a limited set of benchmarks, rather than establishing a robust general principle for balanced time-series representation learning.

## Potentially Missed Related Work
- CoST — directly relevant as the main backbone the paper modifies, and the closest prior decomposition-based contrastive framework.
- Frequency-aware time-series SSL methods beyond CoST, including recent decomposition-oriented representation learning work — relevant because the paper’s claim depends on balancing trend/seasonality rather than just adding another weighting heuristic.

## Suggestions
- Provide a single, unambiguous algorithm describing how SDE is computed, when it is detached or updated, and how it drives the final loss.
- Add stronger ablations that separate decomposition, SDE measurement, fusion MLP, and adaptive weighting.
- Report mean/std over multiple runs and compare against more recent and more directly relevant time-series SSL and decomposition baselines.
- Include controlled skew experiments and component-recovery analyses to substantiate the claim that the method preserves weak semantics rather than merely reweighting losses.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
