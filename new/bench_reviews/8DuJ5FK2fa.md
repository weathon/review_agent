Now I have enough calibration data. Let me consolidate my review.

Key calibration anchors:
- **EvA** (Accept Poster, scores 6,6,6,8): Similar domain (spurious correlation mitigation, last-layer retraining), strong practical results but some concerns about baselines and claims.
- **GSR** (Accept Poster, scores 6,6,6,6): Similar domain (group robustness via last-layer retraining), theoretical contribution but some proof issues.
- **Severing Spurious Correlations with Data Pruning** (Accept Spotlight, scores 8,5,5,10): Same domain, very strong results, novel setting.
- **LaSAR** (Reject, scores 3,6,6,3,5): Similar domain (last-layer retraining for spurious correlation), weaker contribution.
- **Learn from Known Unknowns** (Reject, scores 3,5,3,5): Similar domain but with technical inconsistencies and unsupported claims.

Now let me carefully assess each criticism from the reviewers against the actual paper.

## Summary

EValS proposes a post-hoc method for improving robustness to spurious correlations without group annotations. It combines (1) loss-based sampling that selects equal numbers of high-loss and low-loss samples per class from an ERM model to create a balanced dataset for last-layer retraining, and (2) environment-based model selection using inferred environments (via EIL) where worst environment accuracy replaces worst group accuracy as a model selection criterion. The method achieves competitive performance on spurious correlation benchmarks without any group labels and introduces the Dominoes-CMF dataset for evaluating robustness to multiple spurious attributes.

## Strengths

- **Addresses a genuinely important practical problem**: Eliminating the need for group annotations at *every* stage including model selection is a meaningful advance over prior methods (JTT, AFR, DFR) that still require group labels for validation.
- **Strong empirical results on spurious correlation benchmarks**: EValS (with zero group annotations) achieves 88.4% on Waterbirds, 85.3% on CelebA, and 82.1% on UrbanCars, substantially improving over ERM (66.4%, 47.4%, 18.7%) and competing with methods that use group annotations. The UrbanCars result is notable since it contains an un-annotated spurious attribute.
- **Multi-shortcut evaluation with Dominoes-CMF**: The finding that methods using group annotations for a known attribute can increase reliance on unknown ones (Figure 4b), and that annotation-free methods paradoxically achieve better robustness to both known and unknown shortcuts, is an important and counterintuitive insight.
- **Conceptually simple and computationally efficient**: The method is purely post-hoc, requires only last-layer retraining, and doesn't need access to original training data or checkpoints—making it practically deployable.
- **Honest about limitations**: The paper explicitly acknowledges failure on class/attribute imbalance datasets (× for CivilComments and MultiNLI) and explains why EIL environments have negligible group shift in those settings.

## Weaknesses

### Fatal

None.

### Major

- **Scope of "group annotations not necessary" claim is overstated and contradicted by own evidence**: The abstract and introduction make sweeping claims ("group annotations are not necessary even for validation", "marks a new chapter in robustness"). However, EValS is not even applicable to 2 of 5 benchmarks (CivilComments, MultiNLI—marked × in Table 1), and the paper itself shows that EIL-based environment inference yields only 0.8–1.9% group shift for these datasets. The honest scope is: EValS works for spurious correlation settings where EIL can find meaningful environment splits. The claims should be consistently narrowed throughout the paper, particularly in the abstract and discussion, not just in a limitations paragraph. As stated, "marks a new chapter" and even the title's "without group annotation" universal claim are misleading given the method's demonstrated scope limitation.

- **No ablation separating loss-based sampling from environment-based model selection**: The method has two distinct components—loss-based sampling for creating D^Bal and environment-based (WEA) model selection for choosing k and other hyperparameters. Without independently varying each component, it is impossible to determine whether the performance gains come primarily from the sampling strategy, the validation strategy, or their combination. For instance, does loss-based sampling alone with a fixed k perform similarly? Does random subsampling with environment-based selection work nearly as well? This is critical because both components make distinct claims (loss-tails balance groups → environment-based validation eliminates group label needs), and neither is validated in isolation.

- **WEA as surrogate for WGA is empirically unvalidated**: The central claim that worst environment accuracy can replace worst group accuracy for model selection lacks direct empirical support. No scatter plots, correlation coefficients, or alignment analyses between WEA and WGA across hyperparameter settings are presented. This matters because EIL environments are noisy constructs from a shortcut-dominated ERM model, and the only quantitative evidence given is an average group shift of 28.7% (details in appendix). Without demonstrating that optimizing WEA actually tracks WGA well across the relevant hyperparameter space, the environment-based validation remains a hypothesis rather than a demonstrated mechanism.

### Minor

- **High variance for EValS on Waterbirds (88.4 ± 3.1)**: This is notably larger than DFR (92.9 ± 0.2) and EValS-GL (89.4 ± 0.3). The paper acknowledges variance in the discussion but does not diagnose whether this stems from environment inference instability, sensitivity to the choice of k, or random seed effects. Understanding the source would help practitioners assess reliability.

- **EValS outperforming EValS-GL on Dominoes-CMF is unexplained**: EValS (no group labels) outperforms EValS-GL (group labels for model selection) on this dataset, which is counterintuitive—more information should not hurt. The paper notes this without investigation. It could indicate that group-labeled validation causes overfitting to known groups or that EIL environments are a better surrogate, but a serious analysis is absent.

- **Theoretical analysis relies on strong, unvalidated assumptions**: Proposition 3.1 assumes logits follow a mixture of two Gaussians per class, and only proves *existence* of tail fractions achieving balance under certain conditions. It does not connect to the practical k-selection procedure (equal tails of size k) or validate the Gaussian assumption on real data. The theory is better described as motivation than proof, and this should be framed accordingly.

- **Dominoes-CMF is only tested with a single architecture (ResNet-18)**, while all other experiments use ResNet-50, limiting comparability across experiments. Additionally, the exact numerical results in Figure 4b are not reported in a table, making precise comparison difficult.

### Trivial

- The title promises the method works "without group annotation" universally, but the method is restricted to spurious correlation datasets.

## Nice-to-Haves

- Direct measurement of minority/majority proportions in D^Bal as a function of k across datasets, empirically validating whether loss-based sampling actually produces group-balanced data.
- Extending EValS to handle attribute/class imbalance shifts by developing alternative environment inference methods beyond EIL.
- Reporting average group accuracy alongside worst group accuracy to quantify the tradeoff the paper acknowledges in Section 5.

## Removed Points

- **Claim that EValS shouldn't be compared fairly with DFR because DFR uses group annotations**: The harsh critic argues that EVaLS is not superior to DFR and that comparisons underplay DFR's advantages. However, this is actually the *point*—EValS uses fewer annotations. Comparing a method with fewer assumptions against one with more is legitimate and in EValS's favor. The reviewer's concern that "near-optimal" characterization needs nuance is reasonable (moved to Major weaknesses about scope), but the raw DFR > EValS comparison is not itself a weakness since it's expected given DFR's extra annotation access.

- **Benchmark limitations from Waterbirds being "too easy"**: The human finder suggests Waterbirds is a limited/easy benchmark. While partially valid, the paper tests on 5 datasets including UrbanCars which is harder. The Waterbirds concern alone is not a substantive weakness.

- **Missing baselines for complete fairness**: Demands for additional baselines (random subsampling controls, JTT with environment selection, etc.) go beyond standard practice and would be ablation studies (already noted as a major weakness).

- **Formatting/presentation nitpicks**: Removed per rules.

## Novel Insights

The most genuinely novel observation across the reviews is that EValS (no group labels) outperforms EValS-GL (group labels for model selection) on the multi-shortcut Dominoes-CMF dataset. This suggests a counterintuitive principle: when some spurious attributes are unknown, group annotations for known attributes can *harm* robustness to unknown ones, making annotation-free approaches inherently more suitable for multi-shortcut settings. This deserves deeper investigation as it challenges the standard assumption that more supervision is always better.

## Suggestions

- **Narrow claims to match evidence**: Replace "marks a new chapter in robustness" and "group annotations are unnecessary even for validation" with accurate statements like "for spurious correlation settings where environment inference can capture group shifts, group annotations are unnecessary."
- **Add ablations**: Report (a) EValS with fixed k (no WEA tuning) vs. k selected by WEA, (b) loss-based sampling with group-labeled validation vs. environment-based validation, (c) random balanced subsampling vs. loss-based sampling, all with the same model selection criterion.
- **Validate WEA–WGA alignment**: For each hyperparameter setting (different k values), report both WEA on inferred environments and WGA on ground-truth groups, and show correlation/alignment for the three applicable datasets.
- **Investigate the EValS > EValS-GL result**: Analyze whether group-labeled validation causes overfitting to known groups at the expense of unknown ones on Dominoes-CMF.

## Score and Decision

**Calibration anchors**: EvA (Accept Poster, avg ~6.5) — similar domain, last-layer retraining for spurious correlation, comparable novelty but with concerns about overclaiming; GSR (Accept Poster, avg 6) — similar domain, group robustness via last-layer retraining; LaSAR (Reject, avg ~4.6) — similar domain but weaker methodology and unconvincing claims; Severing Spurious Correlations (Accept Spotlight, avg ~7) — stronger novelty and more comprehensive evaluation.

This paper makes a real contribution: eliminating group annotations for model selection in the spurious correlation setting is practically valuable and the multi-shortcut evaluation with Dominoes-CMF provides genuinely new insights. However, the overclaiming of scope (applying to settings the method demonstrably fails on), the lack of ablations, and the unvalidated WEA→WGA surrogate mechanism are significant issues. The paper is stronger than LaSAR (which had fundamental methodology issues and scores ~4.5) but has more overclaiming issues than EvA or GSR (which score ~6-6.5). The core idea is sound but the execution and claims need tightening.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>