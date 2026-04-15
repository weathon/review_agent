## Summary
This paper studies an overlooked Random Forest hyperparameter: the bootstrap rate (BR), including the unconventional regime \(BR>1\), i.e., drawing more than \(N\) samples with replacement per tree. Across 36 classification datasets and a grid of RF variants, the paper finds that \(BR>1\) is often selected as the best-performing setting in their search space, and it further proposes neighborhood-based \(k_l\) statistics to characterize when higher BR may help and to predict whether the best BR is \(\le 1\) or \(>1\).

## Strengths
- The paper asks a genuinely specific and underexplored question rather than rehashing standard RF tuning: whether **bootstrap rates above 1.0** can be useful. This is a concrete contribution because most practice and implementations assume \(BR \le 1\), and the paper explicitly investigates values up to 5.0.
- The empirical study is broader than the paper’s direct prior point of comparison in two important ways that are specific to this work: it evaluates **10 BR values spanning 0.2 to 5.0** and **18 RF configurations**, rather than only checking the standard regime plus a tiny excursion above 1.
- The paper surfaces an interesting empirical regularity that appears real from the reported results: **different datasets favor very different BR regimes**, and the atypical configuration \(RF(nf\_all)\) behaves qualitatively differently from the others. That exception is informative and suggests the paper is probing a nontrivial interaction between sampling diversity and split-feature randomness.
- Section 5 goes beyond a pure benchmark table by proposing **\(k_l\) neighborhood statistics** as dataset descriptors tied to the BR choice. Even if exploratory, this is a more substantive attempt at understanding the phenomenon than merely reporting a new tuning trick.

## Weaknesses

###: Fatal
- **The central empirical claim is not established with a clean evaluation protocol.**  
  The main headline is that \(BR>1\) “often yields better results” and can produce statistically significant improvements over standard \(BR \le 1\). But the paper’s comparison procedure does not directly test that claim. It first searches over **18 RF configurations × 10 BR values** and reports the winning pair per dataset (Table 1), then performs paired \(t\)-tests comparing the selected winner’s BR group against **all configurations in the opposite BR group**. This is a post-selection analysis and is not the same as comparing the best tuned \(BR>1\) model against the best tuned \(BR\le 1\) model on held-out data.  
  Moreover, the paper itself concedes a much weaker conclusion in Sec. 4: after applying significance thresholds, “**the number of datasets with the optimal solution involving \(BR \leq 1\) is roughly comparable to those with \(BR > 1\)**.” That directly undercuts the stronger abstract/introduction framing. Since this is the paper’s core claim, the evidential gap is serious.

### Major:
- **The claim that optimal BR is mainly a property of the dataset rather than RF hyperparameters is overstated relative to the design.**  
  The evidence for this claim is mostly qualitative: similar curve shapes across several configurations and three manually described pattern types. However, the hyperparameter sweep is limited to **one-at-a-time modifications from a default RF**, with no interaction study. That is not enough to support a strong statement of relative independence from RF hyperparameters. The paper also explicitly notes a major exception: \(RF(nf\_all)\) behaves differently, and pattern (c) mixes configurations preferring different BR regimes on the same dataset. The evidence supports a milder observation—many tested variants show qualitatively similar BR-response curves within a dataset—but not the stronger contribution claim repeated in the abstract and conclusion.
- **Section 5 provides exploratory correlational evidence, not a validated explanation of why high BR helps.**  
  The paper is strongest when it frames this section as “towards understanding,” but elsewhere it overstates what has been shown. The mechanistic story—high BR helps uniform datasets by increasing unique instances while preserving enough diversity, while low BR suppresses ambiguous/outlier effects—is plausible, but it is not directly tested through tree diversity, leaf purity, OOB behavior, repeated-instance counts, or sample-level analysis. Table 2’s raw correlations are also modest (the paper itself reports a maximum Spearman correlation of 0.330 for the overall best BR before feature engineering), so the explanatory evidence is limited.
- **The binary classifier result is too fragile to support a strong predictive claim.**  
  The classifier uses only **36 datasets** (or **24** in the filtered “undisputed labels” version) while drawing from **12,685 features** after engineered interactions. The paper does state that fold-wise feature selection is done inside leave-two-out CV, which is appropriate, so the basic protocol is not obviously leaking. Still, with such a tiny meta-dataset and such a huge candidate feature pool, the result is inherently unstable. The target labels are themselves derived from the same winner-selection process discussed above, and the stronger 88.81% result depends on filtering to datasets with \(p \le 0.01\) under the paper’s own unconventional significance procedure. This makes the predictive contribution interesting but preliminary rather than strong evidence of robust meta-learning.
- **Practical cost is acknowledged but not analyzed, despite being central for \(BR>1\).**  
  The paper correctly notes a “no free lunch” tradeoff: larger BR increases runtime because trees are trained on larger bootstrap samples. But since the proposal is effectively “try BR>1,” the omission of a cost/benefit analysis matters. A setting like \(BR=5\) can be attractive only if its accuracy gains are meaningful relative to its extra compute; otherwise the contribution risks being a brute-force effect rather than a practically compelling tuning insight.

### Minor
- **The statistical methodology in Sec. 4 is hard to interpret and weaker than standard alternatives.**  
  Using repeated 2-fold CV scores as if they were straightforward paired samples for \(t\)-testing is questionable, and taking the **maximum \(p\)-value** over multiple tests per dataset is unusual. Even leaving aside strict statistical orthodoxy, this procedure does not answer the comparison of greatest interest.
- **Table 1 hides practical effect sizes.**  
  The paper reports best accuracies and winning BRs, but not the direct per-dataset gap between the best \(BR>1\) and best \(BR\le 1\) models. Given 400 repeated CV results per configuration, statistical significance can coexist with tiny gains; the practical magnitude of improvement is therefore not sufficiently exposed.
- **The statement that boundary winners imply the true optimum may lie outside the tested BR range is speculative.**  
  The paper argues that because 0.2 and 5.0 frequently win, the optimum may often be even lower/higher. That is possible, but coarse-grid boundary wins alone do not support that inference.
- **Sensitivity of the neighborhood analysis is not checked.**  
  Section 5 relies on a specific construction: standardized continuous features, binary features mapped to \(\{-1,1\}\), and Manhattan distance. This is motivated, but no sensitivity analysis is reported, even though the interpretive claims hinge on this representation of local structure.

### Trivial
- None.

## Nice-to-Haves
- Add a compute-aware comparison, e.g., accuracy versus training cost for BR values, or a budget-matched comparison between higher BR and alternatives such as more trees.
- Report per-dataset effect sizes between the best tuned \(BR>1\) and best tuned \(BR\le 1\) settings, not just winner counts and \(p\)-values.
- Quantify the “dataset property” claim with an explicit agreement or variance-decomposition analysis instead of relying mainly on visual pattern inspection.
- Strengthen Sec. 5 with direct mechanistic diagnostics such as tree correlation/diversity, number of unique observations per bootstrap sample as BR varies, leaf purity, or the influence of ambiguous points.
- Provide uncertainty estimates for the meta-classifier accuracy and, if possible, validate on more datasets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Only classification tasks are evaluated; no regression.”**  
  Removed because this is scope creep. The paper consistently studies classification accuracy and does not claim regression results.
- **“The paper should also evaluate boosted trees / XGBoost / neural nets / other ensemble methods.”**  
  Removed as outside scope. The contribution is specifically about Random Forest bootstrap rate.
- **“Benchmarks are too old / mostly UCI / should use newer datasets.”**  
  Removed in this form. While one may wish for larger modern datasets, dismissing results because datasets are old is not by itself a sound criticism. The more valid concern is limited sample size and generalization of the meta-classifier, which is already captured above.
- **“Lack of theory” as a standalone fatal flaw.**  
  Weakened/removed as a primary weakness because this is an empirical paper, and a full formal theory is not required to make a useful contribution. The real issue is overclaiming explanation from correlational evidence, which is kept.
- **Generic strengths like ‘well-written’, ‘important topic’, ‘experiments are extensive’.**  
  Removed as too generic under the reviewing rules.

## Novel Insights
The most interesting synthesis across the paper and reviews is that the work contains a real empirical signal but packages it with claims stronger than the evidence supports. The observation that \(BR>1\) can sometimes be competitive or even best is credible and potentially useful, especially because the paper identifies a qualitatively different regime for \(nf\_all\), which hints that bootstrap size interacts with ensemble diversity in a structurally meaningful way. However, the current methodology is better viewed as **hypothesis-generating** than claim-settling: the paper has likely uncovered a nontrivial RF tuning phenomenon, but not yet demonstrated its prevalence or mechanism with the rigor expected for ICLR.

## Suggestions
- Redesign the main comparison around a **clean group-vs-group evaluation**: for each dataset/split, tune within \(BR\le 1\) and within \(BR>1\) separately, then compare those two tuned groups on held-out data.
- Replace or supplement the current significance analysis with an across-dataset paired comparison on per-dataset performance differences, and report **effect sizes**.
- Tone down the abstract and conclusion. As written, the paper’s strongest safe claim is closer to: **“In our search space, \(BR>1\) is often competitive and sometimes best, motivating it as a tunable option.”**
- Rephrase the “dataset property” claim to match the evidence unless a quantitative variance-decomposition analysis is added.
- Keep Sec. 5 explicitly exploratory unless direct mechanistic evidence is added.
- Add a practical discussion of **accuracy gains versus computational overhead**; this is particularly important if recommending BR values as large as 3–5.
- For the meta-classifier, provide uncertainty intervals and present it as a preliminary proof-of-concept rather than a mature predictive tool.

## Score and Decision
**Novelty:** moderate. The question is specific and underexplored, and the \(BR>1\) angle is genuinely interesting.  
**Technical soundness:** below the bar, mainly because the central comparison is not evaluated with a clean protocol.  
**Empirical support:** suggestive but not decisive; there is a real signal, but the analysis is not strong enough for the claims made.  
**Significance:** potentially meaningful for RF practice, but not yet established convincingly.  
**Clarity:** generally understandable, though some conclusions overreach the evidence.

Relative to the provided calibration examples, this paper is stronger than a low-end reject with no clear contribution, because it identifies a concrete and plausible empirical phenomenon. But it is weaker than an accept-level empirical systems/ML paper because the main claim is not rigorously validated. This places it in the **borderline-to-clear reject** range for ICLR.

MY FINAL SCORE: <pineapple>4.6</pineapple>
MY FINAL DECISION: <orange>Reject</orange>