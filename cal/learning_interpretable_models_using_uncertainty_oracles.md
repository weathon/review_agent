=== CALIBRATION EXAMPLE 31 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is directionally accurate: the paper is about learning interpretable models with an auxiliary uncertainty-based procedure. However, “uncertainty oracles” is not a standard term, and the title slightly overstates the conceptual novelty relative to what is, in effect, an uncertainty-guided reweighting/sampling scheme.
- The abstract clearly states the broad problem and the high-level method: learn a training distribution, model it with a Dirichlet process mixture, tune by Bayesian optimization, and use an uncertainty oracle for 1D projection. That part is understandable.
- The main concern is that the abstract makes very strong claims (“exhaustive experiments,” “significantly enhances,” “versatile,” “practically convenient,” “works across different feature spaces,” “may augment accuracies of fairly old techniques to be competitive with recent task-specialized techniques”) without giving enough context on the experimental protocol, statistical treatment, or scope. On ICLR standards, this reads as promotional unless the paper convincingly substantiates each claim with careful, fair comparisons and ablations.

### Introduction & Motivation
- The motivation is well-grounded: interpretable models often need to be small, and size–accuracy trade-offs are hard to tune with conventional hyperparameters. This is a real and relevant ICLR problem.
- The gap in prior work is identified reasonably: prior distribution-learning methods are referenced as specialized or limited, especially Ghose & Ravindran (2020). That said, the introduction sometimes frames the problem as if the only obstacle is the inability to search hyperparameters effectively, which is too narrow; there is also a question of whether reweighting/sampling training data actually preserves interpretability goals and generalizes across tasks.
- The contributions are clearly enumerated, but the framing often over-claims novelty. For instance, “model-agnostic” is used in a broad sense, yet the method still depends on a probabilistic oracle, uncertainty calibration, and specific model-training procedures. The paper also claims “only one hyperparameter” in practice, but later the method exposes a fairly large optimization/search space in Appendix A.3.
- The introduction under-discusses a central conceptual issue: why uncertainty scores from a separate oracle should be a good proxy for where small interpretable models need more data. That is the key scientific hypothesis, but it is not sufficiently justified up front.

### Method / Approach
- The method is described with decent algorithmic structure, and Algorithm 1 plus Appendices A.2–A.5 make the pipeline reproducible at a high level. The use of a 1D uncertainty projection, Beta mixture density, and BayesOpt-based parameter search is clear in outline.
- However, there are important logical gaps and ambiguities:
  - The choice of an “Infinite Beta Mixture Model” parameterized by sampled Beta priors is not fully justified mathematically. The paper describes sampling component parameters from Beta priors scaled by a fixed constant, but the resulting generative story is unusual and not clearly connected to a standard posterior inference objective.
  - It is not fully explained what optimization target BayesOpt is operating on with respect to the density model. The relation between sampled data subsets, component weights, and validation accuracy is intuitive, but the search space is large and the method feels heuristic rather than principled.
  - The use of the oracle’s uncertainty as a one-dimensional projection is a major assumption. The paper acknowledges this indirectly, but does not analyze when uncertainty is a poor proxy for “informational value,” especially if the oracle is miscalibrated or systematically uncertain in the wrong regions.
  - The “flattening” transformation in Appendix A.5 is particularly ad hoc. It may help optimization, but the paper does not demonstrate that it preserves the intended structure beyond empirical usefulness. The fact that the authors “perform this for all experiments, saving us the effort of assessing its need” is not a sufficient justification for a core preprocessing step.
- Edge cases and failure modes are under-discussed:
  - What happens when the oracle is weak or poorly calibrated?
  - What happens when uncertainty concentrates on noisy outliers rather than decision boundaries?
  - What if the interpretable model family cannot exploit the reweighted distribution effectively?
  - What if the original data are already near-optimal for the constrained model?
- For theoretical claims, there are no real theorems/proofs, so the issue is more about conceptual completeness than formal correctness. The paper does not provide guarantees on improvement or convergence, which is acceptable for an empirical paper, but then some of the broad claims should be softened.

### Experiments & Results
- The experiments do address the central claim that the learned distribution can improve small interpretable models. Table 1/4 show improvements across many datasets, model sizes, and oracle families, and Table 5 compares against the density-tree prior work. This is the strongest part of the paper.
- That said, ICLR-level scrutiny raises several concerns about whether the experiments fairly and fully support the claims:
  - The baseline comparisons are somewhat asymmetric. For the main result, the baseline appears to be the first-iteration model, and then improvements are reported relative to that. This is reasonable, but the baseline training protocol is not always as visible as the proposed method’s. More direct comparisons to standard training of the same interpretable model under identical tuning budgets would help.
  - The statistical protocol is somewhat unusual. The paper uses a t-test over five trials for validation selection, then reports test-set improvement if significance is met, otherwise sets improvement to zero. This procedure is not standard and may bias reported results toward “positive only when significant” while suppressing negative outcomes. It also mixes validation-based selection with test reporting in a way that is not fully transparent.
  - Table 1/4 reports averages over five runs, but there are no error bars in the main tables, and significance is relegated to the appendix. Given the magnitude of claims, confidence intervals or at least standard deviations would materially help.
  - The choice of evaluation datasets is broad and reasonable for tabular classification, but the dataset subsampling to 10,000 instances from each dataset is not fully justified. It may be fine, but the paper should explain whether this truncation changes the difficulty or interacts with the learned distribution.
  - The comparison to density trees is interesting, but the metric in Table 5 is custom (SDI and pct better). It is hard to interpret and not directly comparable to the raw improvements shown earlier.
  - The “competitiveness” experiments are provocative, especially cluster explanation and prototype-based classification, but they are not enough to support a general claim of outperforming “task-specialized techniques” in a broad sense. The selected tasks, datasets, and baselines matter a lot here.
- The ablation story is incomplete relative to the paper’s claims:
  - There is some evidence for smoothing (Appendix A.6), but not a full ablation showing the effect of the oracle choice, the DP mixture, BayesOpt, and the flattening step separately.
  - The paper claims one hyperparameter in practice, but the experiments clearly involve several implementation choices and search bounds; a sensitivity analysis would be important.
  - The claim that the technique works across different feature spaces is supported by one text-classification case, but this is only a single demonstration and should be framed as such.
- Overall, the empirical results are promising, but the presentation is somewhat too confident given the amount of heuristic machinery and the limited ablation on why each component matters.

### Writing & Clarity
- The paper is generally understandable, and the structure is clear: motivation, method, experiments, limitations. The figures and tables are helpful overall, especially the pipeline schematic and the result tables.
- The main clarity issue is not grammar but conceptual overload. The method combines many moving parts: oracle uncertainty, 1D projection, Beta mixtures, Dirichlet processes, smoothing, Bayesian optimization, and non-differentiable model training. The paper explains each piece, but the connection among them is sometimes hard to keep straight, especially around Algorithm 1 and the sampling procedure in Appendix A.2.
- Figure 2 is useful as a high-level overview, but the actual statistical meaning of the learned distribution is not immediately obvious from the figure alone.
- Tables 1, 4, and 5 are informative, but Table 5’s custom comparison metric is hard to parse, and the significance interpretation in Appendix A.8/A.10 requires careful reading to follow.
- The appendix is extensive, but some core ideas that matter for understanding the method, especially smoothing and the justification of parameter bounds, are buried there rather than summarized more cleanly in the main method section.

### Limitations & Broader Impact
- The paper does acknowledge runtime as a limitation, which is important and honest. It also notes a possible mitigation via different Bayesian optimization methods.
- However, the key limitations are broader than runtime:
  - Dependence on a strong calibrated oracle.
  - Sensitivity to the choice of uncertainty metric.
  - Heuristic nature of the distribution learning and smoothing.
  - Potential instability when the oracle’s uncertainty is noisy or uninformative.
  - Unknown behavior on domains where interpretability constraints are more semantically complex than depth/coefficients/prototypes.
- The broader impact discussion is minimal. Since the paper is about interpretable models in high-stakes settings, a more careful discussion of misuse risk is warranted. If the method improves small models, that is beneficial, but it could also produce overconfident interpretations if the oracle biases which regions are emphasized.
- The paper’s ethics statement is fine, but it does not address any substantive societal risks or deployment cautions.

### Overall Assessment
This paper proposes an interesting and potentially useful idea: use an oracle’s uncertainty to learn a reweighted training distribution that improves small interpretable models. The empirical results are broad and in several places compelling, especially for tabular classification and the density-tree comparison. However, the method is quite heuristic, the conceptual justification for the uncertainty-based 1D projection and the smoothing step is thin, and the experimental protocol leaves some unanswered questions about fairness, statistical reporting, and sensitivity to design choices. For ICLR, I think the contribution is promising but currently not fully convincing at the level of methodological clarity and evidential support that would justify a strong acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a model-agnostic method for improving the accuracy of small interpretable models by reweighting/sampling training data based on an oracle model’s uncertainty scores. The learned sampling distribution is modeled as a Dirichlet-process-based infinite Beta mixture and tuned with Bayesian optimization, with the claim that this enables better size-accuracy trade-offs for decision trees, linear models, and prototype-based methods across multiple datasets and even different feature spaces.

### Strengths
1. **Addresses a practically relevant problem for interpretability.**  
   The paper focuses on the common tension between model size and predictive accuracy for interpretable models, which is very much in scope for ICLR’s interest in practical ML methods. The motivation is well aligned with interpretable learning settings where compact models are desirable.

2. **Broad model-agnostic framing.**  
   A notable strength is that the method is presented as applicable to multiple interpretable model families and multiple notions of size (e.g., DT depth, number of non-zero coefficients, multivariate GBM size). The experiments include LPMs, DTs, and later prototype-based and clustering-related setups, suggesting versatility.

3. **Uses non-differentiable optimization.**  
   The Bayesian optimization formulation is a reasonable design choice for settings where the target training procedure is non-differentiable, such as CART. This is a useful practical contribution because many interpretable learners are not amenable to gradient-based tuning.

4. **Empirical validation across many datasets and settings.**  
   The paper evaluates on 13 public classification datasets and reports improvements across many size settings. It also includes comparison to a prior density-tree approach and some task-specific baselines for cluster explanation and prototype-based classification.

5. **Attempts to study additional properties and limitations.**  
   The appendix explores runtime, multivariate size constraints, smoothing of uncertainty scores, and differing feature spaces between oracle and target model. That extra analysis is helpful and suggests the authors thought about practical deployment.

### Weaknesses
1. **The central technical novelty is somewhat incremental and not clearly isolated.**  
   The core idea appears to be “learn a sampling distribution over training points using oracle uncertainty and optimize it with BayesOpt.” This is related to prior work on distribution learning / reweighting for small models, active learning, and importance sampling. The paper does not clearly separate what is fundamentally new beyond combining known ingredients: uncertainty-based projection, Beta mixture modeling, and BO. For ICLR, the bar for novelty is fairly high, and the conceptual advance feels limited.

2. **The method relies heavily on a strong oracle, which reduces the claimed generality.**  
   The paper says the oracle is only a “tool for dimensionality reduction,” but in practice the technique depends on training a highly accurate probabilistic oracle and using its uncertainty scores. This can be expensive and brittle, and the paper does not convincingly analyze what happens when the oracle is weaker, miscalibrated, or poorly matched to the target task. The reliance on calibrated probabilities is also nontrivial.

3. **Empirical protocol has several unclear or potentially questionable choices.**  
   The paper uses repeated trial selection and significance gating on validation performance, but the evaluation protocol is hard to parse and may be vulnerable to selection bias from many hyperparameter searches. The work also uses only 10,000 instances per dataset and mostly standard LIBSVM datasets, which limits evidence for robustness on more modern or challenging benchmarks.

4. **Baselines are not always strong enough or fully comparable.**  
   Some baselines are historical or implementation-specific, and several later claims are framed as “competitive” rather than clearly outperforming task-specific methods. For interpretability research, ICLR reviewers usually expect careful comparison against strong and directly relevant baselines, along with ablations. The paper does not sufficiently demonstrate that the observed gains come from the proposed distribution-learning mechanism rather than from tuning budget, sampling size, or oracle quality.

5. **Ablation and causal understanding are limited.**  
   The method has multiple interacting components: oracle uncertainty, DP Beta mixture, Bayesian optimization, smoothing/flattening, and uniform mixing via \(p_o\). The paper provides some qualitative evidence for smoothing, but it lacks a clean ablation showing the contribution of each component. It is therefore difficult to tell which part is responsible for the improvements.

6. **Reproducibility is only moderately strong.**  
   The paper states that code is provided, which is good, but many details remain fuzzy: exact oracle training setup, calibration procedure, BO settings, handling of random seeds, model-specific hyperparameter search spaces, and the full tuning budget across all datasets. The algorithm is also fairly complex, making faithful reproduction difficult without more explicit step-by-step implementation details.

7. **Significance of improvements is not always clear in context.**  
   Some reported relative gains are large, but relative improvement can exaggerate changes when the baseline is weak. The paper would be stronger if it reported absolute metrics, confidence intervals, and clearer comparisons of interpretability-performance trade-offs, not just improvements relative to a first iteration or baseline.

### Novelty & Significance
**Novelty: Moderate to low-moderate.** The paper combines uncertainty-based resampling, Dirichlet-process mixture modeling, and Bayesian optimization into a general framework for interpretable models. That combination is interesting, but the underlying idea is close to existing distribution-learning and sample-weighting approaches, and the paper does not convincingly show a major conceptual breakthrough.

**Significance: Moderate.** If the method is as broadly applicable and effective as claimed, it could be useful for practitioners who need compact decision trees or linear models with better accuracy. However, the dependence on a high-quality oracle, the complex optimization loop, and the limited ablation/analysis make it hard to judge whether this is an ICLR-level advance rather than a useful but incremental engineering contribution.

**Clarity: Moderate.** The paper is detailed, but the exposition is sometimes overcomplicated and the algorithmic story is not always crisp. The main idea can be understood, yet the many moving parts and repeated claims make it harder than necessary to assess the contribution.

**Reproducibility: Moderate.** Code availability helps, but the method has enough implementation choices that reproducing the reported gains may require substantial effort. More exact settings and ablations would improve confidence.

### Suggestions for Improvement
1. **Provide a rigorous ablation study.**  
   Separate the effects of oracle uncertainty, the DP/Beta mixture, BO, the smoothing transform, and the uniform mixing parameter \(p_o\). This is essential to show which component actually matters.

2. **Strengthen baseline comparisons.**  
   Compare against simpler and stronger alternatives such as direct sample reweighting by uncertainty, importance sampling, cost-sensitive training, active-learning-inspired selection, and tuned class-balanced subsampling. This would clarify whether the added modeling complexity is justified.

3. **Analyze oracle dependence more carefully.**  
   Report performance when the oracle is weaker, less calibrated, or intentionally mismatched. A sensitivity analysis over oracle quality would help assess whether the method is robust or only works with strong pretrained models.

4. **Report absolute performance and trade-off curves.**  
   In addition to relative improvement, include absolute F1, accuracy, and size metrics across sizes. Show full Pareto curves between model size and performance so the interpretability-accuracy trade-off is easier to judge.

5. **Simplify the method or justify its complexity.**  
   The current pipeline has many tunable parts. Consider whether a simpler formulation would capture most of the benefit, or explicitly justify why each layer of complexity is necessary.

6. **Expand evaluation beyond standard tabular benchmarks.**  
   Include at least one more modern or challenging domain where interpretability matters, and test whether the approach transfers beyond LIBSVM-style datasets. This would improve the significance claim.

7. **Clarify the optimization and reproducibility details.**  
   Provide exact BO settings, search spaces, random seed handling, calibration procedure, and any early-stopping or model-selection heuristics. Pseudocode should be supplemented with implementation-level details sufficient for faithful reproduction.

8. **Tighten the writing around the main contribution.**  
   The paper would benefit from a sharper statement of what is new relative to prior sample-reweighting or distribution-learning work, and what empirical evidence supports that claim.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons to strong, directly relevant baselines for “small interpretable models,” not just density trees and a few task-specific methods. For ICLR, the core claim that this method is broadly superior is not convincing without baselines like constrained/regularized logistic regression, sparse trees from modern methods, rule lists/sets, and simple reweighting or resampling schemes that optimize the same size-accuracy trade-off.

2. Add a full ablation that isolates each design choice: oracle uncertainty vs random scores, Beta-mixture sampling vs simpler kernel/density estimators, BayesOpt vs random search, and flattening vs no flattening. Without this, it is impossible to tell which part of the method actually drives the gains, or whether the whole pipeline is just an expensive resampling heuristic.

3. Add experiments showing performance under fixed compute budgets and wall-clock time matched against baselines. The method uses thousands of BO evaluations and can take close to an hour; without compute-normalized comparisons, the reported accuracy gains are not enough to support practical superiority.

4. Add sensitivity studies over the oracle choice and oracle quality. The method depends heavily on a separate probabilistic model, but the paper does not show how much performance varies when the oracle is weaker, miscalibrated, or trained on less data, which directly affects the method’s claimed robustness.

5. Add at least one modern, larger-scale benchmark and one non-tabular benchmark beyond the surname text case. The current datasets are mostly classical LIBSVM benchmarks, which is weak evidence for an ICLR claim of generality; the paper needs to show the method transfers beyond these standard tabular settings.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze why oracle uncertainty should identify instances worth oversampling, rather than just empirically asserting it. The method’s central assumption is that boundary uncertainty is a good 1D proxy for “informativeness,” but the paper never demonstrates this correlation or when it fails.

2. Quantify variance and statistical reliability properly. The paper reports five runs, but the selection protocol, repeated validation, and multiple comparisons make the reported gains hard to trust without confidence intervals, effect sizes, and a clearer correction strategy.

3. Analyze whether the method improves calibration, margin, or boundary complexity of the interpretable model. If the mechanism is really learning a better training distribution, the paper should show changes in decision boundary geometry or margin distribution, not only final F1.

4. Explain the relation to reweighting, active learning, and distillation more rigorously. Right now the method is positioned as distinct from all three, but the paper does not analyze whether it is effectively a form of sample weighting or query-by-uncertainty in disguise.

5. Show when the method fails. The paper mentions a few negative cases, but it does not analyze failure modes by class imbalance, dataset size, oracle miscalibration, or model family, which is necessary for trust at ICLR standards.

### Visualizations & Case Studies
1. Add before/after decision-boundary or tree-structure visualizations on several datasets, not just one toy example. Reviewers need to see whether the method truly simplifies or reshapes the learned boundary in a meaningful way, or whether it just overfits to a better sampled subset.

2. Add plots of the learned sampling density over oracle uncertainty, alongside the empirical uncertainty histogram and selected samples. This would reveal whether the optimizer is learning a sensible concentration near boundary cases or exploiting artifacts in the score distribution.

3. Add per-iteration optimization traces showing validation score, chosen hyperparameters, and sampled-set composition. Without this, it is unclear whether BO is actually searching a stable landscape or just finding noisy lucky draws.

4. Add case studies where the method hurts performance, including the resulting samples and model rules. This would expose whether the method collapses on certain dataset structures or oracle errors, which is critical for assessing robustness.

### Obvious Next Steps
1. Replace the expensive black-box BO loop with a simpler, cheaper optimization strategy or demonstrate that the complexity is actually necessary. At ICLR, a method that is materially slower than ordinary training needs a compelling efficiency story, not just a note that another optimizer may be faster.

2. Extend the method to end-to-end learned weighting or bilevel optimization as a principled alternative to heuristic sampling. The paper itself suggests this direction, and it is the most natural next step if the goal is to learn the training distribution rather than search over it.

3. Validate the method on more modern interpretable model classes and real deployment settings. The paper should show whether the same idea helps sparse GAMs, monotonic trees, rule lists, or constrained boosting under realistic compute and interpretability constraints.

4. Test robustness to oracle misspecification and distribution shift. If the method is meant to be model-agnostic, it needs to show that gains persist when the oracle is imperfect or when train/test distributions differ beyond the toy settings used here.

# Final Consolidated Review
## Summary
This paper proposes a model-agnostic procedure for improving small interpretable classifiers by learning a new training distribution guided by an auxiliary oracle’s uncertainty scores. Concretely, it projects examples to one dimension via oracle uncertainty, fits a Dirichlet-process Beta-mixture sampling distribution, and uses Bayesian optimization to tune the resulting resampling scheme for a size-constrained interpretable learner. The paper reports broad empirical gains on standard tabular benchmarks, plus additional comparisons to density-tree methods, cluster-explanation trees, prototype-based classifiers, and a text example with mismatched feature spaces.

## Strengths
- The core idea is practical and broadly applicable: use an oracle’s uncertainty to bias the training distribution toward informative regions, while keeping the target learner itself interpretable and size-constrained. The paper shows this can be plugged into multiple model families, including linear models, decision trees, GBMs, and prototype-based setups.
- The empirical section is extensive and not just confined to one toy setting. Results are reported across 13 public datasets, multiple model sizes, two oracle families, a comparison to the prior density-tree method, and additional task-specific experiments on clustering and prototype-based classification. The text-surname example also demonstrates that the oracle and interpretable model can operate on different feature spaces.

## Weaknesses
- The method is heavily heuristic and its main scientific claim is under-justified. The central assumption is that oracle uncertainty is a good 1D proxy for where a small interpretable model should concentrate training mass, but the paper does not really validate this premise beyond empirical success. As a result, the pipeline reads more like a clever resampling recipe than a principled learning method.
- The algorithm has many interacting moving parts, yet the paper does not isolate which ones matter. Oracle choice, DP/Beta-mixture modeling, Bayesian optimization, smoothing/flattening, and uniform mixing are all bundled together, with only limited evidence that each component is necessary. This makes it hard to tell whether the gains come from the proposed idea or from a costly black-box search over resampling distributions.
- The experimental protocol is not fully convincing. The use of only five runs, validation-based selection, and significance-gated reporting makes the reported improvements hard to interpret cleanly, especially when the main metric is relative improvement from a baseline or first iteration. The paper would benefit from absolute performance curves, error bars, and a clearer account of selection bias.
- Runtime is a real limitation, not a minor caveat. The paper acknowledges that the optimizer can take close to an hour for some settings, and the core loop uses thousands of BO evaluations. That is a serious practical cost for a method whose end goal is to produce small interpretable models, and the current mitigation story is only preliminary.

## Nice-to-Haves
- A stronger ablation study separating oracle uncertainty, the mixture model, BayesOpt, smoothing, and the uniform-sampling mix would make the method much easier to trust.
- Additional compute-matched comparisons against simpler resampling or weighting strategies would help clarify whether the extra modeling complexity is justified.
- Sensitivity to oracle quality and calibration would be useful, since the technique’s success plausibly depends on a fairly strong probabilistic oracle.

## Novel Insights
The most interesting aspect of the paper is not the Dirichlet process machinery itself, but the idea of using uncertainty as an intermediate representation for compressing a high-dimensional training set into a one-dimensional “informativeness” axis that can then drive sampling for any interpretable learner. That makes the approach more flexible than prior density-tree-based reweighting, and the cross-feature-space text experiment suggests the framework may transfer information between otherwise incompatible representations. At the same time, the results also hint that much of the benefit may come from a fairly generic boundary-focused resampling effect, with the proposed DP/BO apparatus mainly providing a way to search that effect rather than a deeper explanation of it.

## Potentially Missed Related Work
- Importance sampling / sample reweighting for learning with constrained models — relevant because the method is effectively a learned reweighting scheme, even if framed differently.
- Active learning / uncertainty sampling literature — relevant because the oracle-uncertainty projection is conceptually close to boundary-based acquisition.
- Modern sparse or constrained interpretable learners — relevant as stronger baselines for the “small model” claim, though not necessarily direct prior work the paper omitted.

## Suggestions
- Add a clean ablation table that toggles one component at a time: uncertainty projection, DP-mixture sampling, BayesOpt, flattening, and the original-uniform mixing term.
- Report absolute test metrics, confidence intervals, and Pareto-style size-vs-accuracy curves, not just relative improvement summaries.
- Include at least one compute-matched baseline such as direct uncertainty-based resampling, simple importance weighting, or random search over the same sampling family.
- Provide a sensitivity analysis over oracle calibration and oracle accuracy, since that is likely the main failure mode of the method.
- Tighten the methodological story: explicitly state whether the contribution is a principled learning method or a practical heuristic for searching over resampling distributions.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 0.0]
Average score: 3.3
Binary outcome: Reject
