## Summary
This paper proposes “Neurovectors,” a non-backpropagation method for tabular prediction in which each training instance is stored as a set of exact feature-value tokens, and inference retrieves candidate stored instances by token overlap, with an “energy” score used to break ties. The paper reports competitive results on a few small datasets and argues for very low computational cost via dictionary-based lookup and partial storage of only mispredicted examples.

The core idea is easy to understand and has some practical appeal as a lightweight retrieval-style learner. However, after checking the method against the paper text, the main concerns are substantive: the representation appears fundamentally brittle for continuous features, the evaluation is too limited and unstable for the breadth of the claims, and the efficiency analysis is not rigorous enough to support the strong computational conclusions.

## Strengths
- **The method is unusually transparent at the prediction level.** The model prediction is explicitly based on retrieving stored training instances whose `<feature_name, value>` tokens overlap with the query, and selecting among them using overlap count plus a simple historical reliability metric. This makes the decision process directly inspectable in a way that most tabular neural baselines are not.
- **The proposal does make a concrete algorithmic departure from standard backprop-trained tabular models.** The training rule in Section 3.4 only instantiates a new neurovector on failures and otherwise updates per-instance usage/success statistics; that is a specific retrieval/memory mechanism rather than just another MLP variant.
- **The paper surfaces an important practical tradeoff: lightweight prediction versus peak accuracy.** Even though the current evidence is not yet sufficient to validate the strong efficiency claims, the paper does articulate and experimentally explore the idea that a simpler memory-based tabular learner may be attractive when training/inference cost matters.
- **The model stores only mispredicted examples rather than all examples by construction.** This is a specific and potentially useful design choice, since Section 3.4 explicitly avoids creating neurovectors for correctly predicted samples, which could reduce memory relative to a full instance store in favorable cases.

## Weaknesses

###: Fatal
- **The paper’s central generalization mechanism is not convincing for continuous tabular data, and this directly threatens the core empirical claims.**  
  The method defines tokens as exact concatenations of feature name and feature value (Eq. 3: `τ_{j,l} = (name_feature_l + v_{j,l})`), and candidate retrieval depends on exact token matches. For datasets used in the paper such as Breast Cancer, Absenteeism, and Red Wine, many features are numerical/continuous. The paper does not describe any binning, rounding, tolerance rule, similarity kernel, or embedding that would let nearby numeric values match. Without such a mechanism, unseen test values will often share few or no exact tokens with stored neurovectors, making overlap-based retrieval brittle and potentially ineffective. This is not a side issue: exact matching is the backbone of the proposed algorithm. The paper repeatedly claims the method works “without any prior preprocessing,” which makes the absence of a numeric handling mechanism more consequential, not less.
- **The regression formulation is mathematically degenerate as written.**  
  Section 3.3 states: “for regression problems, a prediction is correct if the predicted value is identical to the current value, and therefore the MAE is 0.” Since the method predicts a stored target value from a selected neurovector, exact equality for continuous targets is generally measure-zero except in repeated labels. Yet Eq. (9) still uses `success(NV)` in the numerator, where success is based on this exact-match notion. This makes the regression “energy” definition very weakly motivated and likely uninformative in realistic continuous regression settings. Because one of the paper’s three main benchmark tasks is regression, this is a serious technical issue.

### Major:
- **The method is framed in neural/energy-based terms that are not supported by the actual algorithm.**  
  The paper presents Neurovectors as “a new neural network approach,” speaks of “energy propagation,” and positions the method against backprop-based neural learning. But the actual method is a retrieval-and-counting procedure over stored exact feature-value tokens with per-instance success/use statistics. There are no learned weights, no hidden representations, no propagation dynamics, and no energy minimization in the usual sense. The “energy” in Eqs. (8–9) is effectively a heuristic confidence score derived from success history. This mismatch matters because it inflates the apparent novelty and can mislead readers about what kind of contribution this is.
- **The empirical support is too narrow for the paper’s claims of effectiveness on tabular learning.**  
  The main experiments use only three small datasets with a single 60/20/20 split. For a paper making broad claims about a new tabular learning paradigm, this is not enough. The later comparison on Adult/Bank/Kick also does not rescue this, because it reveals much weaker performance and severe instability rather than robust competitiveness.
- **Table 4 exposes extreme instability that is not adequately addressed.**  
  The reported standard deviations for Neurovectors on Adult/Bank/Kick are 0.3096, 0.2881, and 0.1817, while other methods are around 1e-3. Even allowing for a typo possibility, the paper does not explain these values. If they are correct, they imply severe instability; if not, the results table is unreliable. In either case, this substantially weakens the credibility of the large-dataset evidence.
- **The efficiency claims are overstated relative to the evidence actually provided.**  
  Table 3 is based on hand-derived FLOP estimates rather than direct, unified runtime or memory measurements on the same hardware/software stack. The estimates for Neurovectors rely on simplified counts of hashing, dictionary search, and creation, while baseline costs are drawn from coarse formulas and assumptions. The paper also scales costs for some datasets simply by dataset-size ratios. This is not strong enough to support claims such as “four orders of magnitude less than tree-based ensemble methods and even six orders of magnitude less than neural networks.” At most, the paper suggests the method may be lightweight; it does not rigorously establish the magnitude of the claimed efficiency advantage.
- **The baseline setup is not strong enough to justify the comparative claims.**  
  The paper compares against RF, Gradient Boosting, SVC, and a simple 3-layer MLP with largely fixed configurations, but omits stronger standard tabular baselines from the main experiments. Since the paper’s own related work emphasizes modern tabular methods and strong boosting-based baselines, the main empirical section does not support claims of broad competitiveness against the field’s most relevant methods.

### Minor
- **There is no ablation establishing whether “energy” contributes meaningfully beyond plain overlap count.**  
  Since the prediction rule first maximizes `count(NV)` and only then uses energy as a tie-breaker, an ablation removing the energy term is needed to show whether this component is actually important.
- **Memory behavior is asserted rather than demonstrated.**  
  Section 3.5 argues that neurovector growth is sublinear because fewer new neurovectors are created over time, but this is not empirically characterized. Since storage is central to both scalability and efficiency, the paper should report neurovector growth curves and memory footprint.
- **The performance/efficiency tradeoff is not analyzed carefully enough on larger datasets.**  
  Table 4 shows Neurovectors trailing strong baselines by a noticeable margin on Adult/Bank/Kick. The paper argues this is acceptable because training times are lower, but that tradeoff is not quantified in a way that would let readers judge whether the accuracy loss is worthwhile in realistic applications.
- **Clarity suffers in several technical definitions.**  
  In particular, the candidate set definition in Eq. (4) and the discussion around count/energy are imprecise enough that the exact implementation behavior is harder to infer than it should be. This compounds the difficulty of understanding how the method behaves when exact token matches are sparse.

### Trivial
- None.

## Nice-to-Haves
- Add a simple numeric-feature handling mechanism (e.g., binning, tolerance windows, nearest-bin matching, or learned continuous similarity) and evaluate how sensitive results are to this choice.
- Report wall-clock training/inference time, memory footprint, and candidate-set sizes per query, rather than relying primarily on FLOP estimates.
- Include an ablation for: (i) overlap-only retrieval, (ii) overlap + energy tie-breaking, and (iii) storing all examples vs. storing only failure cases.
- Show robustness analyses under small perturbations/noise to continuous features.
- Use repeated splits or cross-validation with significance testing for the small-dataset results.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The related work omits foundational instance-based learning / k-NN / case-based reasoning.”**  
  This may be a fair scholarly positioning concern in general, but per instruction I am not including missing-related-work criticisms.
- **“The GitHub/code is anonymized / not independently reproducible / lacks environment specs.”**  
  Removed as a reproducibility nitpick and because the paper does provide a code link and public datasets.
- **“The use of Python dictionaries gives O(1) lookup, therefore inference is computationally efficient.”**  
  Removed as a strength because this is too generic and not sufficiently validated by the paper’s actual end-to-end measurements.
- **“Zero preprocessing pipeline” as an unqualified strength.**  
  Removed in strong form because the paper’s no-preprocessing claim is entangled with the exact-match brittleness problem on continuous features.
- **“Unfair comparison because the asymmetry favors the authors.”**  
  I kept only the valid part: the baselines are not strong enough for the claims made. I did not retain arguments that merely object to asymmetry in a way that could actually favor the baselines.

## Novel Insights
The most important synthesis is that the paper’s two headline advantages—general tabular applicability and extreme efficiency—are coupled to the same exact-match design choice, and that choice is also the source of the paper’s biggest weakness. Exact tokenization of raw feature values may indeed make the implementation simple and potentially cheap, but unless the model introduces a principled notion of similarity for numeric features, the retrieval mechanism collapses precisely on the kinds of real-valued tabular problems the paper evaluates. This means the paper is not just missing additional experiments; its current formulation leaves unresolved whether Neurovectors are a generally useful learner or only a brittle exact-memory system that works when values repeat enough.

## Suggestions
- Add a principled treatment for continuous features and targets. This is the highest-priority fix. At minimum, define binning/rounding/tolerance rules explicitly and show sensitivity analyses.
- Redefine regression energy so it depends on continuous error directly, rather than exact target equality.
- Reframe the contribution more accurately as a retrieval-/memory-based tabular learner unless a genuine neural or energy-based mechanism is introduced.
- Strengthen the evaluation with repeated trials on standardized tabular benchmark suites and report means/standard deviations consistently.
- Replace or complement FLOP estimates with direct runtime and memory profiling on the same hardware for all methods.
- Add ablations isolating the contributions of token overlap, energy tie-breaking, and failure-only storage.