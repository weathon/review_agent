=== CALIBRATION EXAMPLE 40 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately conveys the application domain and the sequential planning focus. However, it under-specifies the core methodological novelty: the paper is less about “case-guided” planning in general and more specifically about a similarity-weighted implicit generative model plus ensemble MCTS.
- The abstract states the problem and method clearly enough, but several claims are too strong relative to the evidence shown in the paper. In particular:
  - “reduced resource consumption by up to 92% compared to established heuristics” is based on only a few representative real-world cases in Section 5.1, not a broad benchmark.
  - “demonstrating the superiority of its ensemble planner” is supported only by alignment against a synthetic deterministic baseline (Table 2), not by a direct comparison to stronger planning methods or ablations that isolate the ensemble’s effect.
- The abstract also frames IBMDP as a “Bayesian MDP” and “model-based RL” solution, but the paper’s own development is closer to a case-based posterior-predictive planner over historical analogs. That is a potentially valuable idea, but the abstract overstates the degree to which the paper establishes a new RL formalism.

### Introduction & Motivation
- The problem motivation is compelling and ICLR-relevant: sequential decision-making under severe data/simulator scarcity is important in drug discovery, and the setting is plausibly underexplored.
- The gap in prior work is stated, but the introduction overstates exclusivity. The paper claims that standard RL cannot operate because there is no explicit simulator or transition tuples. That is true for many model-learning approaches, but the broader literature on offline RL, Bayesian experimental design, POMDPs with latent models, and case-based planning is not fully engaged as a practical alternative space.
- The contributions are stated, but they are not always aligned with what is actually demonstrated:
  - “Evidence-adaptive dynamics” is mostly a re-weighting of historical cases as evidence accumulates.
  - “Similarity-weighted Bayesian belief mechanism” is conceptually reasonable, but the Bayesian interpretation depends on assumptions that are much stronger than the paper emphasizes.
  - “Robust ensemble MCTS despite non-stationary dynamics” is asserted, but the paper does not show that non-stationarity is the key reason ensemble voting helps.
- The introduction under-sells a key issue: the method depends critically on the historical case base being representative of the candidate’s latent outcome structure. This is mentioned later as a limitation, but it is central to whether the method is viable.

### Method / Approach
- The method is described with enough high-level structure to understand the intended pipeline: similarity weighting over historical compounds, posterior-predictive sampling, MCTS-DPW, ensemble voting, and a stop action.
- But there are several important logical gaps and ambiguities:
  - The paper alternates between treating the state as a partially observed belief state and as a concrete vector of observed assay outcomes. The POMDP analogy in Appendix A is useful, but the actual formal correspondence is not fully rigorous because the “hidden state” is a latent index over historical records rather than a true latent process governing the candidate.
  - The claim that the similarity weights are a “direct and principled Bayesian belief update” depends on a Gaussian likelihood interpretation. That interpretation is only approximate and requires independence assumptions across assays that are unlikely to hold in practice.
  - The transition model is sampled by copying assay outcomes from a single historical compound. This preserves empirical correlations, but it also means the model cannot generate anything outside the support of the case base. That is a major modeling limitation, not just an implementation detail.
  - The reward design is underspecified. In Section 2, the reward is essentially negative resource cost with zero at stop, while in the planner simulation a large penalty is introduced if feasibility is violated. But the optimization objective in Equation (3) is only loosely connected to the reward actually used in Algorithm 1.
  - The constrained objective in Equation (3) is not presented cleanly enough to verify exactly what is being optimized. The roles of uncertainty \(H(s)\), goal-likelihood \(L(s)\), terminal constraints, and per-step feasibility constraints need to be mathematically cleaner.
- There are also edge cases not discussed:
  - What happens when the similarity kernel becomes too sharp and all probability mass collapses onto one historical case?
  - What if the target \(g\) is available for only a small subset \(I_g\), making \(H\) and \(L\) noisy or biased?
  - How does the method behave when the candidate lies outside the support of the historical analogs?
- For the theoretical claims, the Appendix A correspondence to a POMDP is interesting but not fully convincing as a general equivalence. It is closer to a latent-case posterior-predictive model than a standard POMDP with a meaningful environment transition. The “convergence” discussion in A.6 is mostly qualified and does not establish a strong theoretical guarantee for the full algorithm.

### Experiments & Results
- The experiments partially test the paper’s claims, but not all of them at ICLR standards:
  - The real-world CNS assay case study demonstrates plausibility and practical value on a small number of representative compounds.
  - The synthetic benchmark is more appropriate for evaluating decision quality, but it mostly validates similarity-based estimation and stochastic exploration against a deterministic VI variant.
- The biggest concern is that the empirical evidence is too thin for the magnitude of the claims:
  - Section 5.1 reports four representative cases, not a comprehensive test over the 220-compound dataset. That is not enough to support claims like “up to 92% reduction” as a general result.
  - There are no confidence intervals, error bars, or statistical significance tests for any of the reported comparisons.
  - The synthetic results in Table 2 report only exact-match rates of first actions and top-2 coverage. This is informative, but it is not sufficient to establish that IBMDP produces better policies in a decision-theoretic sense, especially because the metric is very narrow.
- Baselines are limited:
  - The main real-world baseline is a rule-based heuristic. That is too weak for an ICLR paper making methodological claims about planning under uncertainty.
  - On the synthetic benchmark, VI-Theo and VI-Sim are reasonable controls, but they do not isolate the contribution of MCTS vs. similarity modeling vs. ensembling cleanly. A deterministic planner using the same implicit model is informative, but more ablations are needed.
- Missing ablations that would materially affect conclusions:
  - Without the ensemble voting, how much does performance change?
  - Without progressive widening, how much does search quality change?
  - Without the Bayesian reweighting, does a static similarity kernel perform similarly?
  - Sensitivity to \(\lambda_w\), \(\tau\), \(\epsilon\), \(N_e\), and rollout depth is only lightly discussed, not systematically evaluated.
- Dataset and metrics are appropriate in spirit, but the evaluation design is not strong enough by ICLR standards. The paper needs broader comparative experiments against more meaningful baselines and more robust aggregate metrics.

### Writing & Clarity
- The overall structure is understandable, but several methodological passages remain confusing in ways that impede evaluation of the contribution:
  - The distinction between the actual planning state, the latent prototype index, and the historical database record is not always stable across sections.
  - The exact form of the reward/objective and how it interacts with the state uncertainty and goal-likelihood constraints is difficult to reconstruct cleanly from the main text.
  - The appendix derivations are helpful in intent, but some theoretical transitions are hard to follow and appear to rely on unstated assumptions.
- Figures and tables are directionally useful:
  - Figure 2 communicates the real-world trade-off idea.
  - Table 2 summarizes the synthetic alignment comparison clearly.
  - Figure 3 illustrates ensemble voting qualitatively.
- But the tables and figures do not fully resolve the paper’s central questions, especially around how much each algorithmic component contributes. The visual presentation is acceptable, but the scientific story would benefit from more comprehensive experimental summaries.

### Limitations & Broader Impact
- The paper does acknowledge important limitations:
  - dependence on historical data coverage,
  - sensitivity to the similarity metric,
  - scalability concerns,
  - hyperparameter sensitivity.
- These are genuine and important. However, the limitations section misses several fundamental concerns:
  - The method cannot reliably propose novel strategies outside the support of historical data. That is a major limitation for discovery settings, where extrapolation matters.
  - The Bayesian/POMDP interpretation relies on assumptions that may not hold in real assay pipelines, especially conditional independence and Gaussian discrepancy modeling.
  - The method may reinforce historical biases in assay selection or compound representation if the case base is skewed.
- The broader impact discussion is reasonable and appropriately framed around reducing expensive in vivo testing. Still, there is a possible negative societal impact that is not discussed: if the method is deployed uncritically, it could entrench overconfident decision-making based on biased historical analogs, potentially leading to missed candidates or systematic exclusion of chemically novel regions.
- The ethical statement is balanced, but it would be stronger if it explicitly discussed model misuse or overreliance in high-stakes medicinal chemistry decisions.

### Overall Assessment
This is an interesting and potentially useful paper for sequential experimental planning in data-rich, simulator-poor settings, and the case-based posterior-predictive idea has real promise. That said, for ICLR the current submission is not yet at the bar for a strong methodological contribution because the empirical validation is too limited, the baselines are too weak, and several core theoretical claims are presented more strongly than the evidence supports. The paper’s central idea stands, but it needs a more rigorous treatment of the POMDP/Bayesian interpretation, stronger and broader experiments, and clearer ablations to demonstrate that the ensemble MCTS component adds value beyond a well-tuned similarity-based heuristic.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes IBMDP, a case-based planning framework for sequential assay selection in drug discovery when no explicit simulator or transition tuples \((s,a,s')\) are available. The key idea is to treat historical assay records as a nonparametric implicit generative model: similarity-weighted sampling over past compounds induces a belief update, and ensemble MCTS is used to plan assay batches under cost and confidence constraints.

### Strengths
1. **Addresses a practically important and underexplored setting.**  
   The paper targets sequential experimental planning in drug discovery where only historical assay outcomes are available, which is a real constraint in discovery workflows. This is a meaningful problem for ICLR, especially because it combines planning, uncertainty, and structured decision-making in a data-limited but data-rich setting.

2. **Clear attempt to unify case-based retrieval, Bayesian updating, and planning.**  
   The framework is conceptually coherent: similarity-weighted historical cases define an implicit posterior, which is then used inside MCTS-DPW. The appendix gives a detailed POMDP interpretation and a Bayes-filter-style update, which helps make the method more principled than a purely heuristic nearest-neighbor planner.

3. **Uses an ensemble to stabilize stochastic planning.**  
   The decision to run multiple MCTS planners and aggregate via majority vote is sensible given the stochasticity of case sampling and tree search. The reported Top-1/Top-2 improvements over a deterministic similarity-based value-iteration baseline suggest that the ensemble can improve robustness in practice.

4. **Includes both a real-world case study and a synthetic benchmark with a known optimum.**  
   The drug-discovery example demonstrates domain relevance, while the synthetic setup provides a controlled environment to compare against a theoretically optimal policy. This is a good experimental design choice in principle, since it avoids relying solely on heuristic baselines.

5. **The paper is relatively transparent about limitations.**  
   The limitations section acknowledges dependence on dataset coverage, similarity-metric assumptions, scalability issues, and hyperparameter sensitivity. That is a positive sign for scientific maturity, and it aligns with ICLR’s expectation that authors discuss failure modes.

### Weaknesses
1. **The empirical evaluation is not yet strong enough for ICLR’s acceptance bar.**  
   The real-world evaluation is essentially a small case study on four representative compounds, compared mainly to a rule-based heuristic. There is no broad benchmark suite, no statistical significance testing, and no comparison to strong modern baselines adapted to the setting. For ICLR, this is usually too narrow to support claims of general effectiveness.

2. **The comparison set is too weak and is argued away rather than addressed.**  
   The paper repeatedly claims that direct comparisons to RL/BO/POMDP baselines would be “unfair.” That may be partly true, but ICLR reviewers generally expect carefully adapted baselines, not just a declaration of non-comparability. The paper does include VI-Sim in the synthetic setting, but the real-world section still lacks meaningful algorithmic baselines beyond a heuristic.

3. **The method’s theoretical footing appears overclaimed relative to the actual assumptions.**  
   The paper states that the similarity weights constitute a principled Bayesian posterior, but this is only true under fairly specific modeling assumptions (e.g., independence, kernel likelihood interpretation, latent prototype model). In practice the method uses a heuristic similarity kernel over mixed QSAR predictions and assay outcomes, and the correspondence to true Bayesian inference is approximate rather than exact. The paper sometimes presents this as stronger than warranted.

4. **The paper lacks a convincing demonstration that the synthetic benchmark supports the main claim.**  
   The synthetic environment is linear, independent, and intentionally constructed to match the estimator. Under those assumptions, a similarity-based estimator converging to the true conditional variance is unsurprising. The 47% Top-1 alignment result is informative, but it is not enough to establish superiority in a broad sense, especially since the benchmark appears tailored to the method’s design.

5. **Reproducibility is incomplete despite the statement of reproducibility.**  
   The paper describes the framework and reports hyperparameters, but several details are still missing or ambiguous: exact implementation of MCTS-DPW, rollout heuristic, stopping criteria in practice, how thresholds are selected per task, how QSAR predictors are trained, and how representative the chosen compounds are. For a planning paper, these details matter a lot.

6. **The presentation sometimes overstates novelty and unfairness of alternatives.**  
   The repeated argument that related methods cannot be compared because they assume different inputs is not fully convincing. Many ICLR papers adapt standard tools to new settings; that is often part of the contribution. The paper would be stronger if it framed these differences constructively and showed adapted baselines rather than relying on rhetorical distancing.

7. **The real-world results are mostly descriptive rather than evaluative.**  
   Claims such as “up to 92% reduction in resource consumption” are impressive, but the paper does not show distribution over many compounds, uncertainty in estimates, or whether the same savings hold broadly. The reported examples may be cherry-picked representative cases, which makes it hard to judge the true average benefit.

### Novelty & Significance
**Novelty: Moderate.** The specific combination of similarity-weighted case-based dynamics, Bayesian-belief interpretation, and ensemble MCTS for assay planning is interesting, but it is somewhat incremental relative to existing ideas in kernel-based RL, case-based reasoning, Bayesian experimental design, and implicit models. The novelty lies more in the application formulation and integration than in a fundamentally new algorithmic primitive.

**Significance: Moderate to potentially high, but not yet convincingly demonstrated.** If validated on larger, more diverse discovery tasks with strong baselines, the framework could be practically valuable for sequential decision-making in simulator-poor scientific domains. At present, however, the evidence is insufficient for a strong ICLR acceptance case because the empirical validation is limited and the theoretical claims are broader than the demonstrated results.

**Clarity: Mixed.** The high-level story is understandable, and the appendix is detailed. But the paper sometimes obscures the core method behind repeated claims of Bayesian optimality and unfairness of comparison, and key algorithmic details are still not fully operationalized.

**Reproducibility: Moderate.** There is a reproducibility statement and a substantial appendix, but several essential experimental and implementation details remain under-specified.

### Suggestions for Improvement
1. **Add strong adapted baselines in the real-world setting.**  
   Compare against practical alternatives such as greedy uncertainty reduction, cost-sensitive acquisition functions, batch active learning, constrained Bayesian optimization variants, and simpler nearest-neighbor planners. Even if they require adaptation, that adaptation should be described explicitly.

2. **Evaluate on a larger and more diverse set of compounds/tasks.**  
   Replace or supplement the four-case study with a broader test set and report aggregate metrics, confidence intervals, and statistical significance. For ICLR, evidence of consistent gains matters more than a few illustrative wins.

3. **Clarify the exact probabilistic assumptions behind the Bayesian interpretation.**  
   State explicitly when the similarity weights are exact posteriors and when they are only a heuristic approximation. This would make the theory more credible and reduce the risk of overclaiming.

4. **Ablate the major design choices.**  
   Include ablations for: similarity normalization, feature weights, ensemble size, DPW, rollout policy, stopping thresholds, and use of QSAR predictions. This would show which components actually drive performance.

5. **Report more rigorous uncertainty and robustness analyses.**  
   Vary kernel bandwidths, dataset subsampling, and target sparsity; measure sensitivity to missing target labels and noisy assay values. Since the method is data-coverage dependent, robustness analysis is essential.

6. **Provide a clearer implementation protocol.**  
   Specify how QSAR models are trained, how similarity distances are computed in code, how actions are enumerated under batching constraints, and how rollout heuristics are defined. Pseudocode should be executable in spirit, not just conceptual.

7. **Tone down the “unfair comparison” framing and instead show adapted comparisons.**  
   ICLR reviewers will likely respond better to a constructive story: “existing methods need adaptation, so we adapted them and compared fairly.” That would strengthen the paper substantially.

8. **Demonstrate generalization beyond the presented drug-discovery tasks.**  
   If possible, add one non-drug-discovery sequential experimental design benchmark, or at least a broader pharmacokinetics/ADME suite, to support the claim of broad applicability.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add head-to-head comparisons against standard sequential decision baselines adapted to the same assay-planning setting: greedy information gain, myopic expected utility, uncertainty sampling, Thompson-style case-based sampling, and a simple supervised ranking policy. Without these, the claim that IBMDP materially improves planning is not convincing at ICLR’s bar.

2. Benchmark against strong nearest-neighbor / retrieval baselines that use the same historical database and QSAR inputs but do not use MCTS or Bayesian reweighting. Otherwise it is unclear whether gains come from the proposed planning method or just from case-based similarity scoring.

3. Run ablations for each core component: no ensemble voting, no DPW, deterministic rollout policy, uniform weights instead of similarity weights, no QSAR initialization, and no feasibility penalty. The paper’s main claim is that the full IBMDP stack is needed; right now that is not established.

4. Evaluate sensitivity and robustness across multiple real datasets, not just one CNS case study and one synthetic benchmark. ICLR expects evidence that the method generalizes beyond a single favorable domain, especially when the real-data result is only illustrated on a few representative compounds.

5. Include comparisons against offline RL / batch planning methods that can operate from logged data, such as conservative offline RL variants or fitted value iteration on a surrogate state abstraction. The “no simulator” framing does not remove the need to show the method beats the closest data-driven planning alternatives.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify uncertainty calibration: when the method says a compound is likely promising, how often is it correct? Without calibration analysis, the claimed “decision confidence” and thresholding on \(H\) and \(L\) are not trustworthy.

2. Analyze whether the similarity weights actually correspond to meaningful posterior behavior under distribution shift. The Bayesian/POMDP interpretation is central to the paper, but no empirical check shows the weights concentrate on chemically or biologically plausible analogs rather than artifacts of feature scaling.

3. Report variance across random seeds, especially for MCTS and the ensemble vote. The synthetic result of 47% top-1 alignment is not enough without confidence intervals, because ICLR reviewers will ask whether the gain over VI-Sim is stable or just seed-sensitive.

4. Study how performance changes with historical dataset coverage and missingness in \(G\). The method explicitly depends on representativeness of \(D\), yet the paper does not show when it breaks or how severe the effective sample-size problem becomes.

5. Provide a complexity/runtime analysis tied to practical throughput, not only asymptotic cost. One-hour per compound on a laptop is already potentially limiting; the paper needs to show scalability as \(|D|\), horizon, and batch size grow.

### Visualizations & Case Studies
1. Add full trajectory plots for several real compounds showing the sequence of assays, posterior weights over analogs, and how \(H(s_t)\) and \(L(s_t)\) evolve after each step. This would reveal whether the planner genuinely adapts or just follows a fixed heuristic path.

2. Show failure cases where IBMDP chooses an apparently cheap path that later underperforms, especially for compounds outside the historical support. This is necessary to test the paper’s central claim that similarity-guided planning is robust in simulator-free settings.

3. Visualize the ensemble vote distribution across actions, not just the final MLASP. Without this, it is unclear whether the ensemble is truly capturing multiple high-value plans or merely averaging unstable runs.

4. Add analog-neighborhood visualizations: for a few decision points, show the top contributing historical compounds and how their assay profiles justify the next recommended assay. This would make the “case-guided” claim believable rather than abstract.

### Obvious Next Steps
1. Extend the method to a broader suite of assay-planning tasks with different modality patterns and cost structures. ICLR expects a method claimed as general to demonstrate transfer beyond one CNS benchmark.

2. Add principled uncertainty quantification for the planner itself, not only the target property. A decision-making method should expose when its recommendations are unreliable under sparse or biased history.

3. Replace the ad hoc “unfair to compare” stance with a standardized benchmark protocol that adapts common offline planning and BO baselines to the same state/action abstraction. Without this, the contribution is hard to evaluate against ICLR norms for empirical rigor.

4. Test learned or domain-specific similarity metrics instead of a fixed variance-normalized Euclidean kernel. The entire framework hinges on similarity quality, so this is the most direct next step to improve credibility and performance.

# Final Consolidated Review
## Summary
This paper proposes IBMDP, a case-based planning framework for sequential assay selection in drug discovery when only historical assay outcomes are available and no explicit simulator or transition tuples exist. It combines similarity-weighted retrieval over historical compounds with a posterior-predictive sampling model, then uses ensemble MCTS-DPW and majority voting to plan batches of assays under cost and confidence constraints.

## Strengths
- The paper targets a genuinely important and underexplored problem: sequential experimental planning in a simulator-poor, data-rich drug discovery setting. The formulation is practical and well-motivated by real assay-selection workflows.
- The core idea is coherent and reasonably well developed: historical cases are reweighted by similarity, those weights are interpreted as an implicit belief state, and the resulting posterior-predictive model is used inside MCTS. The appendix gives a substantial formalization of this case-based belief update, which is more principled than a purely heuristic nearest-neighbor planner.
- The synthetic benchmark is a sensible choice in principle because it provides a ground-truth policy and lets the authors compare a stochastic ensemble planner against a deterministic similarity-based value-iteration baseline.
- The paper is transparent about several real limitations, including dependence on historical coverage, similarity-metric assumptions, scalability, and hyperparameter sensitivity.

## Weaknesses
- The empirical evaluation is far too thin for the strength of the claims. The real-world result is essentially a small case study on four representative compounds, compared mainly against a rule-based heuristic. That does not support broad claims like “up to 92% reduction” as a general performance result.
- The baselines are weak and the paper leans too hard on arguing that proper comparisons would be “unfair.” That may be partly true for exact off-the-shelf methods, but it does not remove the need for adapted, competitive baselines. As written, the evaluation does not convincingly show that IBMDP outperforms stronger sequential decision alternatives.
- The theoretical story is overclaimed relative to the assumptions. The Bayesian/POMDP interpretation is interesting, but it relies on strong modeling assumptions, including a latent-prototype view and conditional-independence/Gaussian-style likelihood assumptions that are unlikely to hold generally in assay pipelines. The paper presents this correspondence more strongly than the evidence warrants.
- The method is fundamentally support-limited: it can only sample outcomes from historical analogs and therefore cannot generate genuinely novel trajectories outside the case base. This is not a minor caveat; it is central to whether the method can be trusted in discovery settings where extrapolation matters.
- The synthetic benchmark is somewhat self-confirming. The environment is constructed to make the similarity-based estimator tractable, and the reported metric is narrow: exact first-action match and top-2 coverage. That is informative, but not enough to establish that the planner is broadly better in a decision-theoretic sense.

## Nice-to-Haves
- A fuller ablation study isolating the contribution of ensemble voting, DPW, similarity weighting, QSAR initialization, and feasibility penalties.
- More aggregate reporting on real data: confidence intervals, variance across seeds, and results over a larger set of compounds rather than a few illustrative examples.
- A clearer statement of when the Bayesian interpretation is exact versus approximate.

## Novel Insights
The main conceptual contribution is not a new RL primitive, but a useful reframing: planning can be done by treating a historical assay database as an implicit posterior over latent compound prototypes, then using that posterior to sample plausible futures during tree search. That is a legitimate and potentially useful idea for simulator-free scientific decision-making. However, the paper’s novelty is mostly in the integration and domain formulation rather than in a fundamentally new algorithmic ingredient, and the current evidence does not yet show that the ensemble MCTS machinery adds enough beyond a strong similarity-based heuristic to justify the stronger methodological framing.

## Potentially Missed Related Work
- Offline RL / batch decision-making methods — relevant because the paper argues that standard RL is inapplicable, but adapted offline planning baselines are still the natural comparison class.
- Bayesian experimental design and implicit design methods — relevant because the paper’s uncertainty-driven assay selection is close in spirit, even if the action/state abstraction differs.
- Kernel-based RL and case-based planning — relevant because the similarity-weighted transition model is closely related to these older ideas.
- Constrained Bayesian optimization / active learning for experiment selection — relevant as a nearby class of methods for sequential scientific decision-making under resource constraints.

## Suggestions
- Add adapted real-world baselines that operate on the same assay-planning abstraction, such as greedy uncertainty reduction, myopic expected utility, retrieval-only planners, and offline/batch planning surrogates.
- Expand the real-data evaluation beyond four examples and report aggregate statistics over many compounds, with variance across random seeds.
- Include ablations for the ensemble, DPW, similarity kernel, QSAR initialization, and thresholding logic.
- Make the probabilistic assumptions explicit: state clearly which parts are exact under the latent-prototype model and which are heuristic approximations.
- Show failure cases and out-of-support behavior, since robustness there is critical for any discovery application.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 4.0, 4.0]
Average score: 3.0
Binary outcome: Reject
