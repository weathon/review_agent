## Summary
DeLLMa (Decision-making Large Language Model assistant) is a framework that scaffolds LLM decision-making under uncertainty using classical decision theory. It decomposes the process into four steps: state enumeration (identifying latent factors via LLM), state forecasting (verbalized probabilities over factor values), utility function elicitation (pairwise LLM rankings fitted via Bradley-Terry), and expected utility maximization (Monte Carlo). The authors evaluate on two real-world domains—agricultural crop planning (USDA data) and stock selection—and demonstrate consistent improvements over prompting baselines and OpenAI o1 (zero-shot), while exposing interpretable intermediate reasoning traces.

---

## Strengths

- **Novel integration of decision theory with LLM inference-time scaling.** Rather than applying CoT or self-consistency generically, the paper constructs a theoretically grounded pipeline around the expected utility principle. This is a specific and actionable framing that is distinct from prior inference-time compute work, and fills a real gap between generic reasoning benchmarks and structured uncertainty quantification.

- **Bradley-Terry utility elicitation from LLM pairwise rankings is an inventive contribution.** Using overlapping minibatch rankings fed into a classic preference estimation model is a concrete and novel procedure that avoids asking the LLM for cardinal utilities directly (which is brittle). The variance-reduction trick of pairing the same sampled states with all actions is a well-designed engineering choice that meaningfully reduces noise.

- **Consistent multi-model and multi-domain improvements.** DeLLMa yields accuracy gains across GPT-4, Claude 3, and Gemini 1.5, and across two structurally different domains (natural language reports vs. tabular price data). Robustness across model families strengthens the generality claim.

- **DeLLMa substantially outperforms o1-preview at comparable cost.** Table 3 shows DeLLMa (at $0.09–$0.37/instance) achieving 73.3% vs. 33.3% on Agriculture and 64.2% vs. 35.0% on Stocks against o1-preview (zero-shot, ~$0.21/instance). Even granting the asymmetry of the comparison (see weaknesses), this is a strong empirical finding suggesting that specialized decomposition adds value over general-purpose inference-time reasoning.

- **Human-LLM utility agreement matches inter-annotator agreement.** Table 4 shows 65–68% human-LLM agreement versus 67.0% ± 6.3% inter-annotator agreement. Rather than interpreting this negatively, this is the right benchmark: if the task is genuinely hard for humans too, the elicited utility is performing as well as a human annotator could.

- **Baseline failure modes are diagnosed.** The paper provides concrete analysis (Appendix C.4, Figures 12/17/18) of why baselines fail—sentiment echoing, inability to reason counterfactually—which is a genuine contribution beyond just outperforming baselines.

---

## Weaknesses

### Fatal
None. The core empirical contributions stand and the methodological approximations, while imperfect, do not undermine the experimental claims.

### Major

- **Independence assumption over latent factors is unvalidated and potentially significant.** Section 3.2 factorizes the state distribution as $\prod_i \pi_i(\cdot|\mathcal{C})$, but factors like climate, yield, and price in agriculture (or economic conditions and individual stock growth in finance) are obviously correlated. The ablation in Table 2 compares DeLLMa to uniform/under/overspecified forecasts, but does *not* compare against a joint specification that accounts for correlations. Since the full expected utility computation depends on the correctness of this distribution, this is a central assumption that remains unquantified in its impact. The paper acknowledges it as a simplification but provides no sensitivity analysis.

- **Benchmark diversity and independence are limited.** Both environments use exactly 7 items and generate 120 instances by enumerating all subsets. These 120 instances are not independent—a model with a fixed preference ranking over the 7 fruits/stocks will have correlated performance across all subsets that include its preferred item. This inflates the effective sample count and limits what can be concluded about generalization. The paper's generalization claims would be substantially strengthened with additional domains or randomly sampled (non-exhaustive) decision instances.

- **Baseline comparisons are incomplete, particularly against o1.** The comparison to o1 is zero-shot only (Table 3). A structured prompt giving o1 the same decomposition scaffold (enumerate states, estimate probabilities, rank state-action pairs, compute expected utility) would test whether DeLLMa's gains stem from the decision-theoretic structure or simply from spending more compute tokens in a particular way. Without this, the claim that "specialized inference-time reasoning outperforms general-purpose reasoning models" is not fully established. Similarly, no baseline directly prompts the LLM to estimate expected utility numerically (a much simpler structured baseline than DeLLMa), making it hard to attribute gains to specific components.

- **Stock evaluation ground truth is noisy and potentially contaminated.** The target is a single month (December 2023) of returns for well-known tickers (NVDA, META, GME, etc.). A single month of returns for highly salient stocks is vulnerable both to stochastic noise (single-month returns are largely unpredictable) and to pretraining contamination (models likely have strong priors about these companies' trajectories). The paper acknowledges the leakage concern but does not resolve it. Multi-period aggregation or less salient assets would substantially improve the validity of this evaluation.

### Minor

- **Bradley-Terry scores used as cardinal utilities lack full justification.** Algorithm 2 returns Bradley-Terry preference strengths, which are ordinally grounded but not necessarily on a ratio scale. Averaging them in Eq. (3) as expected utilities assumes interval-scale properties. This is a reasonable heuristic that works empirically, but the paper would benefit from an explicit acknowledgment that this is an approximation and from discussing whether the results are sensitive to the scale of the elicited scores.

- **Verbalized probability mapping $\mathcal{V}$ is unjustified.** The mapping from {very likely, ..., very unlikely} to numeric values is applied throughout but its sensitivity is never analyzed. Different plausible mappings could shift the state distribution and thus the final decision. A brief sensitivity analysis or justification of the chosen mapping would substantially improve confidence in the forecasting component.

- **State forecasting ground-truth annotation is underspecified.** Table 1 reports ECE and NLL for forecast quality, but the latent factors are free-form strings generated by the LLM. The paper says ground truths were "manually annotated" but does not describe how annotators resolved cases where the LLM's factor definitions are abstract or action-entangled (e.g., "apple yield" as a state factor, which conflates state and action). This makes Table 1 difficult to interpret rigorously.

- **No confidence intervals or significance tests on primary results.** With 120 instances per environment, bootstrap confidence intervals are straightforward and would help distinguish genuine improvements from noise, particularly for the moderate gains seen in the state forecasting ablation (Table 2) and for the stock domain.

### Tiny

- **"Scaling laws" is an overstatement.** The linear trends in Figure 3 with respect to sample size and overlap percentage are informative hyperparameter studies, not scaling laws in any technical sense. This is a minor terminology issue.

- **Human evaluation includes paper authors as annotators.** The 5 external volunteers plus the paper authors constitute a small and potentially biased panel. The result is still useful, but the limitation should be noted more prominently.

---

## Nice-to-Haves

- **Factor correlation modeling.** Even a simple approach—asking the LLM to jointly assign probabilities over combinations of factor values for a small number of high-dependency factor pairs—could improve the fidelity of the state distribution without requiring full joint specification over all $\ell^k$ states.

- **Structured prompt for o1 comparison.** Providing o1 the same decision-theoretic scaffold and comparing its decomposed output to DeLLMa would tightly isolate whether the gains come from the pipeline structure vs. the underlying model, and would give a stronger argument for DeLLMa's design choices.

- **Failure mode analysis.** Characterizing instances where DeLLMa still fails—whether due to bad state enumeration, miscalibrated forecasts, or poor utility rankings—would provide actionable insight for future improvements and strengthen the paper's analytical depth.

- **Compute-vs-performance Pareto frontier.** An explicit plot of accuracy against total API cost across different sample sizes and overlap percentages would help practitioners calibrate DeLLMa for their budget and time constraints.

- **Sensitivity of results to the verbalized probability mapping.** Testing 2–3 reasonable alternative mappings $\mathcal{V}$ would either show robustness (positive) or reveal a hyperparameter that deserves more principled treatment.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Missing related works in preference learning/reward modeling"** (Critic): The paper cites the utility elicitation literature (Farquhar 1984) and LLM-for-ranking work (Lee et al., 2024; Qin et al., 2023). The missing-related-work criticism could invoke non-existent references and is excluded per guidelines.

- **High-stakes framing vs. narrow experiments** (Critic): The paper explicitly scopes itself to a proof-of-concept in its conclusion ("avails many possibilities") and defers portfolio construction to future work. Criticizing the absence of sequential/continuous-action domains beyond the paper's explicit scope is scope creep. The narrow evaluation is already captured as a major weakness about benchmark diversity, which is in scope.

- **"No end-to-end worked example"** (Spark Finder): Figure 3 (right) and Figure 4 (right) show complete decision trees with enumerated states, sampled latent factors, weights (posterior probabilities), and utilities for Agriculture and Stocks respectively. The criticism is factually incorrect for this paper.

- **"Unfair comparison where asymmetry favors the baseline" (o1 zero-shot)**: The o1 zero-shot comparison is *asymmetric in favor of DeLLMa* (more compute, structured pipeline), which is noted as a separate weakness about missing structured-prompt baselines. The criticism of "unfair comparison where asymmetry favors the baseline" does not apply here—the baseline is disadvantaged, not the authors' method, so comparing in this direction proves a stronger point.

- **Circular reliance on LLM for utility elicitation** (Spark Finder): The paper's key insight is that decomposing the task into explicit state-action pairwise comparisons makes the LLM's ranking task tractable even if direct decision-making is not. The empirical results support this design. The "circularity" concern conflates the hardness of the ranking sub-task with the hardness of the end-to-end decision. The concern is philosophically interesting but the empirical gains demonstrate it is not a practical failure mode.

- **"Contribution is diffuse"** (Critic): For a systems/empirical paper at ICLR, a clear framework + empirical validation across multiple models and domains constitutes a well-defined contribution. The claim is unfounded as a standalone criticism.

---

## Novel Insights

The most genuinely novel observation—partially surfaced by the Spark Finder—is that *the independence assumption in state forecasting may matter less than the utility elicitation module* for the final decision quality: Table 2 shows that Uniform, Underspecified, and Overspecified state priors perform similarly to calibrated DeLLMa forecasts on GPT-4 and Gemini 1.5, while DeLLMa-Naive (weaker utility elicitation) is consistently worse than DeLLMa-Pairs/Top1. This suggests that the Bradley-Terry–based utility elicitation—not the probabilistic forecasting of states—is the primary driver of performance, a conclusion the paper does not explicitly draw but which has important implications for where future work should focus: improving the utility elicitation mechanism (e.g., more robust ranking, cardinal alignment) may yield greater returns than refining the state forecasting pipeline.

---

## Suggestions

1. **Decompose the source of gains explicitly.** Run an oracle-state version of DeLLMa (replace LLM state forecasts with realized ground-truth states) and an oracle-utility version (replace Bradley-Terry with ground-truth utility rankings) to quantify each module's contribution. This would definitively settle whether improvements come from better uncertainty modeling or better preference elicitation.

2. **Test on at least one additional domain** with a different action structure—e.g., multi-attribute choices, where simple ranking is harder—to validate the framework beyond "pick one from k options."

3. **Add a structured-prompt o1 baseline.** Prompt o1 with explicit instructions to enumerate states, estimate probabilities, rank state-action pairs, and compute expected utility. This is the natural control for the Table 3 comparison.

4. **Analyze verbalized probability mapping sensitivity.** Try 2–3 alternative scale mappings (e.g., {0.9, 0.7, 0.55, 0.45, 0.3, 0.1} vs. the current mapping) and report whether results are stable. This is a cheap experiment that would substantially increase confidence in the forecasting step.

5. **Clarify the Bradley-Terry / cardinal utility relationship.** Add a brief discussion noting that BT scores are used as a proxy for cardinal utilities and that the validity of expected utility averaging depends on the monotone relationship between BT strengths and true utilities. Acknowledge this as a modeling assumption rather than a rigorous derivation.

6. **Report normalized utility (already in Appendix B) as a co-primary metric.** Accuracy against a single "optimal action" penalizes rational risk-averse choices and conflates decision quality with outcome luck. Normalized utility is a better reflection of decision quality under uncertainty and should appear in the main text.