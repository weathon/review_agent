=== CALIBRATION EXAMPLE 21 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the core idea: automatic neural feature selection. However, it slightly overpromises novelty by implying a broadly new paradigm, when the paper is more specifically a Gumbel-Sigmoid-based global-mask selector with a sparsity penalty.
- The abstract clearly states the problem, the proposed masking-plus-predictor setup, and the main claimed advantages: automatic determination of feature count, end-to-end training, and near-constant overhead.
- Several claims in the abstract are too strong relative to the evidence presented. In particular, “consistently outperforms both the classical and neural FS methods while selecting significantly fewer features” is not uniformly supported by the detailed tables. For example, in Table 2 some datasets improve, some match, and some degrade versus full-data baselines. The “nearly constant computational overhead regardless of input dimensionality” claim is also only empirically estimated later, not theoretically established.

### Introduction & Motivation
- The problem is well-motivated for high-dimensional tabular data, and the paper identifies a real pain point: choosing the number of selected features and retraining across budgets.
- The related-work gap is described reasonably well, especially the contrast between filter, wrapper, embedded, and neural methods. The motivation for a method that learns both the subset and its size is clear.
- The introduction does overstate the generality and practical impact. Claims like “maintains almost constant computational overhead regardless of the dimensionality” and “automatically determines the minimal subset of features sufficient for the downstream task” are stronger than what is actually justified. “Minimal” is especially difficult to support because the optimization only uses a sparsity penalty, not a minimality guarantee.
- The contribution list is coherent, but it somewhat conflates empirical success with methodological novelty. The method appears to be a straightforward differentiable mask selector with Gumbel-Sigmoid and an L1-style penalty on mask entries, so the main novelty bar at ICLR depends heavily on whether the empirical evidence is truly stronger and more general than prior neural FS work.

### Method / Approach
- The method is described at a high level, but several parts are not yet sufficiently precise for full reproducibility or for assessing correctness.
- The masking network in Section 3.2 is under-specified. It states that a randomly initialized embedding \(e \in \mathbb{R}^{D_e}\) is mapped to logits \(w=f_\phi(e)\), but the architecture of \(f\), the exact role of the embedding dimension, and why a single shared embedding suffices are not fully justified.
- A key conceptual issue is that the selector is global and dataset-level, not instance-specific. That is a valid design choice, but the paper should be more explicit that AutoNFS does not model sample-dependent feature relevance. This limits applicability on tasks where useful features vary across instances.
- The loss definition is missing or ambiguous in places. Section 3.3 defines \(L_{\text{select}} = \frac{1}{D}\sum_j m_j\), which is straightforward, but the text claims the model automatically finds the “minimal subset,” which is not guaranteed by a linear penalty alone. The penalty trades off sparsity and task loss; it does not solve exact subset minimization.
- The hard-thresholding rule in Section 3.5 is potentially problematic: using \(\sigma(w_i) > 0.5\) after training may not match the stochastic training dynamics, and it is unclear whether this is the actual selection criterion used in evaluation or merely a convenience. Since training uses Gumbel noise, the deterministic extraction rule should be justified carefully.
- The claim that computational cost is “nearly constant” is not mathematically established by the method. Since \(f_\phi(e)\) outputs a vector of size \(D\), the selector’s final layer itself scales with dimensionality. The paper’s empirical claim may still be meaningful, but “constant overhead” is too strong in an algorithmic sense.
- The use of Gumbel-Sigmoid for global feature selection is standard enough that the paper’s technical novelty seems limited unless there is an especially effective training scheme or a nontrivial empirical finding. As written, the method reads more like a combination of known components than a new algorithmic contribution.
- There are also some potential failure modes not discussed: sensitivity to \(\lambda\), temperature annealing, class imbalance, correlated features, and instability when multiple feature subsets are equally predictive.

### Experiments & Results
- The experiments do test the main claims to some extent: classification/regression benchmarks, corrupted-feature settings, metagenomic data, and a timing comparison.
- The benchmark design based on feature corruption is appropriate for testing whether a selector can ignore distractors. However, the paper should be clearer about whether all methods were allowed comparable feature-budget tuning. The statement that baselines “select the same number of features as were in the initial representation” is confusing: many listed baselines do not naturally output a variable number of features unless explicitly thresholded.
- Baselines are broadly reasonable and include several relevant neural and classical selectors. That said, the comparison is slightly imbalanced because some baselines are used as importance rankers and others as end-to-end selectors, and the paper does not clearly state how the final subset size was chosen for each method.
- A major concern is the lack of uncertainty reporting in the main benchmark tables. Tables 3–5 report only single scores and mean ranks. There are no standard deviations, confidence intervals, or significance tests across runs/datasets, which makes it hard to judge whether the differences are robust.
- The computational complexity experiment in Figure 4 is not fully convincing as currently presented. Fitting a power law to wall-clock time over a limited range of feature counts is a weak basis for claiming a stable complexity exponent, especially when the selector still has a \(D\)-dimensional output layer. The label “near-constant time scaling” may be an empirical observation, but it should not be read as a complexity guarantee.
- The metagenomic results are interesting and relevant, but the reporting is incomplete for a strong ICLR claim. Table 2 shows mixed performance: some datasets improve, some worsen, and the average gain is modest. The claim that AutoNFS “does not lead to deterioration of the results on average” is true for the averages shown, but it hides substantial per-dataset variability.
- The analyses in Figures 3 and 6 are useful, but they are not enough as ablations. Missing ablations that would materially affect the conclusions include:
  - removing the temperature annealing schedule,
  - replacing Gumbel-Sigmoid with a simpler continuous gate,
  - comparing against Hard Concrete / STG / Concrete Autoencoder / L0 regularization under the same evaluation protocol,
  - varying embedding size and selector network depth,
  - testing sensitivity to \(\lambda\),
  - evaluating whether the method still works when the number of truly relevant features is large or when the data contain strongly correlated redundant groups.
- The MNIST visualization is illustrative, but it is not central evidence for the method’s effectiveness on tabular feature selection.

### Writing & Clarity
- The paper is generally understandable, but several parts are conceptually muddled in ways that impede assessment of the contribution.
- The most important clarity issue is the distinction between “automatic” feature count discovery and “minimal subset” discovery. The paper repeatedly implies the latter without establishing it.
- The method section would benefit from a more explicit description of the selector architecture and the evaluation protocol for converting soft masks into final subsets.
- Figures and tables convey the intended message, but some key tables are hard to interpret because the baseline selection-size protocol is not clearly explained. This is especially important in Tables 1, 2, and 3–5, where performance and sparsity are intertwined.
- The prose sometimes attributes too much theoretical significance to an annealing heuristic. For example, the RL/exploration-exploitation analogy in Appendix B is interesting but does not materially help understanding the method.

### Limitations & Broader Impact
- The limitations are underdeveloped. The paper acknowledges almost none of the substantive limits of the method beyond the generic fact that there is a balance parameter \(\lambda\).
- Important limitations not addressed include:
  - the selector is global, not instance-specific;
  - performance depends on tuning \(\lambda\) and the temperature schedule;
  - correlated or interacting features may yield non-unique selections;
  - the method may not be suitable when different subpopulations require different features;
  - interpretability of selected features is still heuristic, especially under feature correlation.
- The ethics statement is too minimal for an ICLR submission discussing biological datasets and interpretability. Even if the method itself is benign, the authors should acknowledge possible misuse, domain-transfer risks, and the fact that feature selection can encode biases by excluding minority-relevant variables.
- The broader impact statement does not discuss failure modes or negative uses, which is a gap in ICLR’s current expectations.

### Overall Assessment
AutoNFS is a plausible and potentially useful neural feature-selection method, and the empirical sections suggest it can be competitive on several benchmarks while selecting fewer features. However, for ICLR the paper currently feels stronger on empirical packaging than on methodological depth. The core algorithm combines known ideas—Gumbel-Sigmoid gating, a sparsity penalty, and a task network—without fully establishing why it is meaningfully better than closely related neural FS methods or why its “automatic minimal subset” claim is justified. The experiments are interesting but not yet rigorous enough: uncertainty estimates are missing, baseline selection protocols are not fully transparent, and several ablations needed to isolate the contribution are absent. The contribution may still be worthwhile, but in its current form it does not quite clear the ICLR bar for a strongly compelling advance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces AutoNFS, a neural feature selection method that learns a global feature mask end-to-end using Gumbel-Sigmoid relaxation and a sparsity penalty. The central claim is that AutoNFS automatically determines both which features to keep and how many, while maintaining near-constant computational overhead with respect to input dimensionality. The method is evaluated on OpenML tabular benchmarks with synthetic corruption and on metagenomic datasets, where it reportedly outperforms several classical and neural baselines while selecting fewer features.

### Strengths
1. **Addresses an important and practical problem for tabular learning.**  
   The paper focuses on automatic feature selection for high-dimensional tabular data, where interpretability and efficiency matter. This is directly relevant to ICLR’s interest in scalable representation learning methods.

2. **Simple end-to-end formulation.**  
   AutoNFS is conceptually straightforward: a masking network produces feature logits, Gumbel-Sigmoid samples a mask, and a task network evaluates the masked input with a sparsity penalty. The training recipe is easy to understand and fits well within differentiable optimization paradigms.

3. **Attempts to avoid manual feature-budget tuning.**  
   A notable practical motivation is that users do not need to specify the number of selected features in advance. This is a useful direction compared with many feature selection methods that require a budget or repeated retraining.

4. **Broad experimental coverage.**  
   The paper tests on 11 benchmark datasets across three synthetic corruption settings and additionally on 24 metagenomic datasets. That breadth is a plus for an ICLR submission, since it goes beyond a single benchmark.

5. **Empirical evidence of sparsity plus performance.**  
   The reported results show that AutoNFS often matches or improves predictive scores while selecting substantially fewer features, especially in the metagenomic experiments and synthetic corruption scenarios.

6. **Includes some analysis beyond raw accuracy.**  
   The paper reports feature misselection errors, sensitivity to the sparsity coefficient, and qualitative MNIST visualizations. These analyses strengthen the story that the method is actually selecting a compact subset rather than merely behaving like another regularized predictor.

### Weaknesses
1. **The novelty appears incremental relative to prior differentiable sparsity methods.**  
   The core mechanism—continuous relaxation with a sparsity penalty and end-to-end training—is closely related to Concrete Autoencoders, Hard Concrete / L0 regularization, Stochastic Gates, and other neural feature selection approaches. The paper does not clearly isolate what is fundamentally new beyond using a global embedding plus Gumbel-Sigmoid and a cardinality penalty.

2. **The “automatic number of features” claim is not fully convincing.**  
   In practice, the method still depends on a user-chosen balance parameter λ, and the paper itself reports that λ strongly affects sparsity and accuracy. This means the number of selected features is not truly determined without external tuning; it is shifted from a budget hyperparameter to a regularization hyperparameter.

3. **Computational claims are overstated or insufficiently justified.**  
   The paper claims near-constant overhead with dimensionality, but the masking network produces D logits, and the loss includes all D mask entries. It may be more efficient than iterative wrapper methods, but “nearly constant” is hard to reconcile with an output layer scaling with feature dimension. The complexity section appears empirical and does not provide a rigorous derivation.

4. **Potential ambiguity about whether the selection is dataset-level or instance-level.**  
   The method learns a single global mask shared across all samples. That is a valid design choice, but the paper does not sufficiently discuss limitations: it may fail when relevant features vary across subpopulations or examples, which is a common case in tabular data.

5. **Experimental protocol lacks enough detail for full confidence.**  
   The paper says it “extended” a benchmark codebase, but important details seem under-specified: exact train/validation/test splitting, repeated runs, hyperparameter selection protocol per method, how λ was tuned across datasets, and whether all baselines were equally optimized. For ICLR, this level of rigor is important.

6. **Baseline comparison is not fully persuasive.**  
   Several baselines seem to be used in a way that may disadvantage them, e.g. the paper notes that many select the full set of features, but it is not always clear whether they were allowed comparable feature-budget tuning. ICLR reviewers will likely want a stronger and fairer comparison against tuned sparsity methods such as L0/L1-based neural selectors, Concrete Autoencoders, STG, and LassoNet.

7. **Evaluation focuses mostly on performance and sparsity, not statistical robustness.**  
   There are no clear significance tests, confidence intervals in the main tables, or stability analyses of selected features across seeds. Feature selection methods are often unstable, so this is a major missing piece.

8. **Clarity of the method description could be improved.**  
   Some equations and algorithmic details are presented ambiguously: for example, the use of a learnable embedding vector to generate the mask is unusual and not fully motivated, and the exact relation between the embedding, masking network, and per-feature logits could be clearer.

### Novelty & Significance
**Novelty: moderate to low.** The paper combines existing ideas—Gumbel-Sigmoid relaxation, end-to-end task loss, and sparsity regularization—into a practical feature selection pipeline. While the global-mask formulation and the “automatic feature count” framing are useful, the technical contribution appears evolutionary rather than a substantial algorithmic advance by ICLR standards.

**Significance: moderate.** If the empirical claims hold, the method could be practically useful for tabular and biological data where compact feature sets matter. However, the significance is reduced by limited novelty, somewhat overstated efficiency claims, and the fact that the sparsity-performance tradeoff still depends on tuning λ.

**Clarity: fair but uneven.** The high-level idea is understandable, but methodological and experimental details are not always precise enough, and some claims are stronger than the evidence provided.

**Reproducibility: moderate.** The paper mentions code availability and gives some hyperparameters, but the evaluation protocol is not sufficiently detailed to guarantee easy replication, especially regarding model selection, tuning, and baseline fairness.

### Suggestions for Improvement
1. **Clarify the actual novelty relative to Concrete Autoencoders, STG, Hard Concrete, and LassoNet.**  
   Add a direct conceptual and empirical comparison explaining what AutoNFS does that prior differentiable selectors do not, and why the global embedding-based parameterization matters.

2. **Tone down or rigorously justify the computational-efficiency claim.**  
   Provide a derivation of training/inference complexity and explain precisely in what sense overhead is “near constant.” If the claim is empirical, state it as such and compare wall-clock time under controlled settings.

3. **Treat λ as a first-class hyperparameter and report tuning protocol.**  
   Since λ determines sparsity, describe how it is selected for each dataset, whether it uses validation data, and how sensitive results are to λ. Consider reporting Pareto curves of accuracy vs. number of selected features.

4. **Add stronger reproducibility details.**  
   Include exact dataset splits, preprocessing, normalization, missing-value handling, seeds, number of runs, baseline hyperparameter grids, and selection criteria. ICLR reviewers typically expect this level of detail.

5. **Evaluate statistical significance and selection stability.**  
   Report confidence intervals or paired significance tests on performance, and measure how stable the selected feature sets are across random seeds. Stability is especially important for a feature selection paper.

6. **Strengthen the fairness of baseline comparisons.**  
   Ensure all baselines are tuned comparably and, where appropriate, allow feature-budget optimization for methods that need it. Also compare against more recent sparse neural selectors if not already included under the same tuning budget.

7. **Discuss limitations of global feature masks.**  
   Explicitly address cases where feature relevance is instance-dependent or subgroup-dependent. A brief discussion or an extension to instance-wise masks would improve the paper’s conceptual completeness.

8. **Improve presentation of the method.**  
   Make the architecture and optimization more explicit: define the role of the embedding vector, explain whether mask logits are dataset-specific parameters or learned through input statistics, and provide a cleaner algorithm summary with all loss terms spelled out.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add direct comparisons to the strongest modern tabular FS/selective modeling baselines on the same benchmarks, especially methods like STG, Concrete Autoencoders, INVASE, and recent tabular deep-learning selectors if they were not included. ICLR reviewers will not accept “state-of-the-art” claims without a competitive baseline set that includes both classic and neural feature selectors.

2. Evaluate sensitivity to the sparsity coefficient and annealing schedule with proper sweeps, not just a single default. The claim that the method “automatically determines the minimal subset” is not convincing if the selected subset size depends materially on an unreported or weakly justified choice of \(\lambda\), temperature decay, or initialization.

3. Report variance over many seeds on all main benchmarks, not just mean ranks or a few error bars. A feature-selection method can look strong on average while being unstable across runs, and ICLR will expect robustness evidence before trusting the selected subsets.

4. Add a fair compute comparison against baselines that includes wall-clock training time, memory use, and hyperparameter search cost under matched hardware. The “nearly constant overhead” claim is too strong without showing end-to-end cost against methods whose runtime may scale differently depending on implementation and tuning.

5. Include a stronger ablation against simpler alternatives: fixed top-\(k\) selection, deterministic sigmoid masks, hard-concrete gates, and removing the learned embedding. Without these, it is unclear whether the gains come from the proposed mechanism or just from using a sparse differentiable mask with an auxiliary predictor.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether the selected features are genuinely necessary, using stability and redundancy analyses across bootstrap resamples or dataset splits. The current “misselection” and feature-deletion analyses do not prove the method is identifying a reliable minimal subset rather than one of many equivalent subsets.

2. Analyze how selection interacts with correlated or synthetic interaction features. Because the paper claims to handle feature relationships and automatically find minimal sufficient subsets, it needs evidence on datasets where the relevant signal is spread across correlated variables or only appears through interactions.

3. Explain why the model is selecting a global mask rather than instance-specific masks, and when that assumption breaks. ICLR reviewers will likely question whether a single mask can handle heterogeneous tabular populations, especially on metagenomic data where feature relevance may vary across subtypes.

4. Provide a clearer theoretical or empirical account of what the objective is actually optimizing with the cardinality penalty. As written, it is not obvious that the learned thresholded mask corresponds to a principled minimum-feature solution rather than an arbitrary local optimum influenced by annealing.

5. Validate the “constant-time” scaling claim with a more careful complexity analysis that separates selector cost from downstream model cost. The paper currently mixes empirical fitting with algorithmic claims, which is not enough for a strong efficiency contribution.

### Visualizations & Case Studies
1. Show per-seed feature-selection stability plots for the same dataset, indicating overlap of selected subsets across runs. This would reveal whether AutoNFS learns a robust subset or merely one of many interchangeable solutions.

2. Add failure-case examples where AutoNFS selects the wrong features or under-selects and loses accuracy. A few such cases would make the method credible and would clarify the regimes where the proposed penalty/annealing breaks down.

3. Provide a rank/importance heatmap comparing AutoNFS-selected features to ground-truth or known proxy signals on synthetic data with planted informative features. This would directly test the claim that the method discovers the minimal sufficient set, rather than just a predictive subset.

4. Visualize the evolution of selected features across training for several representative datasets, not just MNIST and one metagenomic example. Reviewers need to see whether the selection converges smoothly, oscillates, or collapses prematurely in harder settings.

### Obvious Next Steps
1. Extend the method to instance-wise or group-wise feature selection and benchmark it against global selection. The current formulation is only for a single dataset-level mask, so its practical scope is narrower than the paper implies.

2. Test the method on larger and more diverse tabular benchmarks, including noisier and highly correlated real-world datasets beyond OpenML and metagenomics. ICLR will expect evidence that the approach generalizes beyond the specific benchmark suite used here.

3. Compare against standard model compression and structured sparsity methods as a selection baseline, not just feature-selection-specific methods. This is necessary to show the method is meaningfully better than generic sparse learning alternatives.

4. Add a principled stopping or thresholding rule for deciding the final feature count. Right now the paper relies on a learned penalty and a 0.5 threshold, but a publishable FS method should show how the final subset size is selected reliably in practice.

# Final Consolidated Review
## Summary
This paper proposes AutoNFS, a differentiable, global feature-selection model that learns a single dataset-level mask using Gumbel-Sigmoid sampling and a sparsity penalty, trained jointly with a downstream predictor. The authors evaluate it on corrupted OpenML benchmarks and metagenomic datasets, claiming that it often matches or exceeds baselines while using substantially fewer features.

## Strengths
- The paper targets a genuinely important problem: automatic feature selection for high-dimensional tabular data, where both sparsity and predictive performance matter. The method is conceptually simple and easy to understand: a mask network generates feature gates, and a task network scores the masked input end-to-end.
- The experiments are broader than average for this area, covering 11 benchmark datasets under three corruption settings plus 24 metagenomic datasets. The paper also includes analyses of selected-feature quality, sparsity/accuracy trade-offs, and an efficiency comparison, which helps substantiate the main story.

## Weaknesses
- The core method is only a modest recombination of known ingredients: Gumbel-Sigmoid relaxation, a task loss, and an \(\ell_1\)-style sparsity penalty. The paper does not convincingly establish a substantive algorithmic advance over closely related neural feature selectors such as Concrete Autoencoders, STG, Hard-Concrete/\(L_0\) methods, or LassoNet.
- The main “automatic minimal subset” claim is overstated. In practice the selected subset size still depends on the user-chosen balance parameter \(\lambda\) and the annealing schedule, and the objective only encourages sparsity—it does not guarantee a minimal sufficient feature set.
- The computational-efficiency claim is too strong as written. The selector still produces \(D\) logits and the reported “near-constant” scaling is based on empirical curve fitting rather than a rigorous complexity argument. At best, the paper shows that it can be much cheaper than iterative wrappers, not that the overhead is dimension-independent in an algorithmic sense.
- Experimental rigor is not yet strong enough for a convincing ICLR claim. The main tables report only point estimates and ranks, with no meaningful uncertainty summaries or stability analysis across seeds, and the paper does not fully clarify the tuning and feature-budget protocol for all baselines. That makes the reported gains hard to interpret.
- The method is limited to a single global mask shared across all samples, but the paper does not sufficiently discuss this restriction. This matters because many real tabular problems have heterogeneous feature relevance across subpopulations, where a global selector may be too crude.

## Nice-to-Haves
- A clearer direct comparison against the strongest modern neural selectors, with the same tuning budget and a transparent final-feature-count protocol.
- A Pareto-style analysis of accuracy versus number of selected features across \(\lambda\) values and random seeds.
- Additional ablations isolating the effect of Gumbel-Sigmoid, the learned embedding, and annealing versus simpler sparse-gating alternatives.

## Novel Insights
The most interesting aspect of the paper is not the gating mechanism itself, but the framing of feature selection as a learned global mask whose cardinality is discovered implicitly through training rather than specified a priori. That is practically appealing for tabular tasks, and the corruption benchmarks suggest the approach can indeed ignore distractors while remaining competitive. However, the current formulation appears to work more as a straightforward sparse differentiable selector than as a fundamentally new feature-selection principle, so the paper’s value is primarily in empirical convenience rather than deep methodological novelty.

## Potentially Missed Related Work
- Concrete Autoencoders — directly relevant differentiable feature selection via discrete relaxations.
- STG / Stochastic Gates — closely related sparsity-based neural feature selection.
- Hard Concrete / \(L_0\) regularization — important comparator for differentiable sparsity and cardinality control.
- INVASE — neural feature selection with a selection network and predictor, especially relevant for end-to-end masking approaches.
- LassoNet — strong baseline for hierarchical sparsity in tabular feature selection.

## Suggestions
- Add a rigorous baseline study against STG, Concrete Autoencoders, Hard-Concrete/\(L_0\), INVASE, and LassoNet under the same tuning and subset-selection protocol.
- Report seed-wise mean and variance for accuracy and selected-feature count, and include stability metrics for the chosen subsets.
- State clearly how \(\lambda\), temperature decay, and the final threshold are selected in practice, and show how sensitive the method is to each of them.
- Tone down the “minimal subset” and “near-constant complexity” language unless you can support it with stronger theory or much more careful empirical evidence.


# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0, 2.0, 4.0, 4.0]
Average score: 2.3
Binary outcome: Reject
