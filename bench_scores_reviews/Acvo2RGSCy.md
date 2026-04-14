## Summary
DeLLMa (Decision-making Large Language Model assistant) proposes a four-step inference-time framework for LLM-based decision making under uncertainty: (1) latent state enumeration, (2) verbalized state forecasting, (3) utility elicitation via LLM-ranked pairwise preferences fit with a Bradley–Terry model, and (4) Monte Carlo expected-utility maximization. The framework is evaluated on 120-instance benchmarks in agricultural planning and stock selection, demonstrating consistent accuracy improvements over zero-shot, CoT, and self-consistency baselines across GPT-4, Claude 3, and Gemini 1.5, and outperforming o1-preview zero-shot by a wide margin.

---

## Strengths

- **Novel decision-theoretic scaffold for inference-time LLM reasoning.** While prior work scales inference compute via CoT, ToT, or self-consistency, DeLLMa is the first to explicitly decompose inference into state forecasting and utility elicitation guided by expected-utility maximization—a genuinely distinct contribution over reasoning-trace-based methods.

- **Consistent cross-model improvement.** DeLLMa yields accuracy gains across all three evaluated model families (GPT-4, Claude 3, Gemini 1.5), demonstrating the generality of the scaffolding rather than a quirk of one model's tendencies.

- **Inference-time compute scaling with meaningful signal.** Figure 3 shows monotonically increasing accuracy with both sample size and overlap percentage—an empirically grounded scaling property that is directly relevant to the inference-time compute literature.

- **Competitive with o1-preview at similar cost.** Table 3 shows that DeLLMa (GPT-4, $n=64$) achieves 73.3% and 64.2% vs. o1-preview's 33.3% and 35.0% on agriculture and stocks, at comparable per-instance cost. This is a striking empirical result that motivates task-specialized pipelines over general reasoning models.

- **Human evaluation of utility elicitation.** Rather than simply reporting end-to-end accuracy, the paper also evaluates the intermediate utility elicitation step via a 412-sample human agreement study, which is a meaningful attempt at component-level validation not common in prompting-framework papers.

---

## Weaknesses

### Fatal
None.

### Major

- **Evaluation conflates decision quality with realized outcome, undermining the theoretical framing.** The paper's primary metric is accuracy against the *ex-post* optimal action—the fruit or stock that actually turned out best. Decision theory, however, evaluates decisions against the *expected utility* given information available at decision time, not realized outcomes. A method that makes a well-reasoned decision with good beliefs can easily "fail" this metric if a low-probability event occurs, while a method that gets lucky with poor beliefs can "succeed." This is not a minor framing issue—it is a direct contradiction of the paper's own decision-theoretic justification. The normalized utility in Appendix B partially addresses this but is relegated to a secondary result. The paper should either (a) construct an evaluation where ground-truth expected utility under the true data-generating distribution is known, or (b) explicitly and prominently acknowledge that accuracy here measures *forecast quality* rather than *decision rationality*, and reframe claims accordingly.

- **Utility elicitation contribution is not isolated.** Both domains have analytically computable utilities from forecasted outcomes (price×yield, monthly return). A critical missing baseline is "forecast-and-pick": use the LLM-forecasted state distribution to compute expected value per action directly, without any Bradley–Terry ranking or utility elicitation. Without this, it is impossible to determine whether DeLLMa's gains come from the decision-theoretic scaffolding as a whole, from state forecasting alone, or specifically from the utility elicitation module presented as a core contribution. Table 2's ablations address state forecasting variants but not the removal of the elicitation step entirely.

### Minor

- **Independence assumption in state forecasting is unquantified.** The paper explicitly posits factor independence "for computational simplicity" (Algorithm 1). In both domains, the violated correlations are substantial (climate ↔ yield in agriculture; macroeconomic health ↔ individual stock growth in stocks). While the paper acknowledges this, there is no experiment or analysis showing how much this distorts downstream expected-utility estimates. Table 2 ablates forecast quality but not the factorization structure itself.

- **State forecasting ablation (Table 2) undermines the module's apparent importance.** For GPT-4 and Gemini 1.5, the uniform, underspecified, and overspecified forecast variants remain within 1–3% of full DeLLMa, while still substantially beating baselines. This raises the question of whether accurate state forecasting is doing meaningful work for these models, or whether the utility elicitation step largely compensates. The paper attributes this to "robustness," but an alternative interpretation—that the state forecasting step contributes little to GPT-4/Gemini performance—deserves explicit engagement.

- **State enumeration quality is not validated.** The entire pipeline's correctness depends on the latent factors $(f_1,\ldots,f_k)$ generated in §3.1 being relevant, non-redundant, and reasonably comprehensive. Yet this step receives no empirical validation. There is no measurement of factor relevance, coverage, or sensitivity to prompt phrasing. Because state space size is $\ell^k$, poor factor selection can compound combinatorially.

- **Calibration evaluation in Table 1 is underspecified.** The paper says "we manually annotate a set of ground truth values for states," but does not report how many forecast points were evaluated, who the annotators were, what constitutes a ground-truth value for a qualitative latent factor such as "climate," or whether annotations were made before or after seeing outcomes. These omissions make Table 1 difficult to interpret as reliable calibration evidence.

- **No confidence intervals or statistical significance reported.** Main results in Figures 2 and 4 report point estimates without variance. With 120 problem instances and stochastic LLM outputs, confidence intervals are warranted and standard.

- **Human evaluation does not strongly validate utility elicitation accuracy.** Table 4 shows LLM–human agreement of ~65–68%, while inter-annotator agreement is 67.0% ± 6.3%. The LLM matches human noise levels but does not demonstrably exceed them. This is encouraging as a lower bound but should not be presented as strong validation; it mainly confirms that the task is inherently ambiguous.

### Tiny

- **The verbalized-to-numeric probability mapping $\mathcal{V}$ is an important modeling choice** (the entire forecast posterior depends on it), but its exact values are deferred to the appendix and not discussed in the main text. The mapping should at minimum be summarized in §3.2, as readers need to know what "likely" maps to in order to evaluate the method.

- **Notation inconsistency between Eq. (1) and Eq. (3):** $U_{\mathcal{C}}(a)$ vs. $U_C(a)$.

- **Bradley–Terry scores are used as cardinal utilities**, but BT recovers a latent preference scale only up to monotone transformation, not as calibrated cardinal values. Since expected utility in Eq. (3) averages these scores across states, the cardinalness assumption matters and should be noted.

---

## Nice-to-Haves

- **Testing on a domain with multi-attribute or non-linear utilities.** Both current domains effectively reduce to "maximize a single numeric quantity." The utility elicitation module would be far more differentiated and informative in a domain involving risk aversion, competing objectives, or qualitative tradeoffs (e.g., medical triage with side-effect penalties), where a simple "expected value" strategy is demonstrably suboptimal.

- **Scalability analysis for larger action spaces.** The paper evaluates up to 7 actions. A discussion or experiment on how API cost and accuracy scale to 20–50 actions (e.g., a diversified portfolio) would help practitioners assess deployment feasibility. The authors do defer continuous/portfolio actions to future work, which is reasonable, but a cost-scaling analysis for discrete sets would be immediately useful.

- **Comparison with a richer o1 baseline.** The current o1 comparison uses zero-shot prompting. A version where o1 receives the same structured decision-theory chain (without full DeLLMa automation) would better isolate whether the gains come from the framework design versus DeLLMa's specific implementation details.

- **Per-factor calibration breakdown.** Table 1 reports aggregate ECE/NLL. Per-factor calibration plots would reveal which latent factors are systematically mis-forecasted and whether those failures drive downstream decision errors.

- **Correlation-aware forecasting experiment.** Even a brief experiment with joint prompting or a simple copula approximation for correlated factors would address the independence assumption more concretely and guide future framework extensions.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Critique that Bradley–Terry / utility elicitation is not novel because of RLHF/preference learning overlap.** The related work section does not need to more deeply engage with the RLHF literature to establish novelty; the contribution is the *integration* of BT-based elicitation into a decision-theoretic inference-time framework, not BT itself.

- **Criticism that "human-auditable" is not rigorously evaluated as a property.** The paper uses "human auditable" to refer to the transparency afforded by explicit intermediate artifacts (decision trees with states, weights, utilities), not a formally evaluated cognitive property. The claim is descriptive of mechanism, not a falsifiable empirical assertion. The concern is reasonable but asking for a formal HCI evaluation of auditability is out of scope for this systems paper.

- **Criticism that the o1 comparison is "unfair" because o1 is used zero-shot.** Per the review instructions, comparisons that are asymmetric in favor of the baseline (o1 is a more powerful model; using it zero-shot still arguably favors it given its built-in reasoning capabilities) should be discounted as a weakness. The comparison is informative and intentional.

- **Criticism that baselines underperforming random guessing undermines the evaluation.** This is actually an interesting finding that the paper discusses as a failure mode of prompting-based methods, not an evaluation flaw.

- **Critique that Figure 3's "linear performance trends" is too strong a claim for a "scaling law."** The paper uses "linear" to describe a qualitative shape observed in the figure, not to assert a formal scaling law. This is a language nitpick.

- **Critique of limited limitations section.** While a more explicit limitations section would improve the paper, this is a formatting/organization concern rather than a scientific flaw, especially given that the paper explicitly acknowledges the independence assumption, defers sequential and portfolio settings to future work, and scopes its contribution to single-step discrete-action problems.

---

## Novel Insights

The most genuinely novel insight surfaced across the three reviews—and not fully developed in the paper itself—is the **internal contradiction between the decision-theoretic framework and the evaluation protocol**. The paper frames DeLLMa as an expected-utility maximizer, yet measures performance against ex-post optimal outcomes. This is not merely a metric choice: it implies that the benchmarks are actually testing *forecast accuracy over a very narrow, retrospective outcome distribution*, not *decision quality under uncertainty* as claimed. A deeper insight follows: if both domains have ground-truth utility functions (price×yield, stock return), it becomes possible to construct a principled evaluation using historical empirical outcome distributions as the true posterior. That would allow direct measurement of how well DeLLMa approximates Bayes-optimal decisions, rather than how often it is lucky enough to pick the realized winner—and would make the decision-theoretic framing internally consistent. None of the reviews develop this constructively, but it points toward a more rigorous version of the benchmark.

---

## Suggestions

1. **Add a "forecast-and-pick" baseline** that computes expected value from the state forecast distribution without any Bradley–Terry ranking. This is straightforward to implement and is the minimal experiment needed to isolate the utility elicitation module's contribution.

2. **Reframe or relocate the evaluation metric discussion.** In the main text, explicitly acknowledge that "accuracy" measures whether the predicted decision matches the realized optimal, and discuss in what sense this is a proxy for (rather than a direct measure of) decision quality under uncertainty. Move normalized utility from the appendix to a co-equal main result.

3. **Report confidence intervals** (e.g., bootstrap 95% CIs) for all main accuracy figures, given the 120-instance evaluation size and stochastic outputs.

4. **Specify Table 1 calibration details in the main text**: number of annotated forecast points, annotation protocol, annotator identities (or that they are the authors), and how ground-truth values for qualitative factors were determined.

5. **Add a brief analysis of the independence assumption's impact**, e.g., by comparing forecast distributions from factorized sampling against joint prompting for a small number of instances, to quantify how much the approximation distorts the state distribution.

6. **State the verbalized-to-numeric mapping** $\mathcal{V}$ explicitly in the main text (even as a small table) for reproducibility, since it directly determines the shape of the forecast posterior.

---

## Evaluation on Key Axes

- **Originality:** Moderately high. Applying classical expected-utility maximization as an inference-time scaffold for LLMs is a genuinely novel framing, distinct from all prior reasoning methods. The specific combination of verbalized forecasting + Bradley–Terry utility elicitation + Monte Carlo EU maximization is original. However, each individual component is borrowed from existing work, and the novelty is in assembly rather than invention.

- **Importance of research question:** High. Decision making under uncertainty is a critical use case for LLMs, and the paper addresses a genuine gap: current inference-time methods are designed for deterministic reasoning, not EU maximization under a user-aligned utility.

- **Whether claims are well-supported:** Moderate. The main accuracy claims are supported for the two evaluated domains and multiple LLM backbones. However, the core theoretical claim—that DeLLMa maximizes expected utility—rests on an approximation chain (factorized independence, verbalized probabilities, BT cardinality) whose combined effect is not analyzed. The evaluation conflating decision quality with realized outcomes weakens the claim that the framework achieves rational decision making.

- **Soundness of experiments:** Moderate. The benchmarks are author-constructed and small (120 instances each, retrospective, narrow domain). Baselines are appropriately chosen for the comparison claims, but the missing forecast-only baseline is a significant gap. Statistical uncertainty is not reported. The stock experiment is particularly fragile given its single-month evaluation window and small, known-ticker action set.

- **Clarity of writing:** Good. The paper is well-organized, the four-step structure is easy to follow, and the algorithms are readable. Some overstatement in framing ("high-stakes," "human-auditable") slightly oversells the empirical grounding, but the technical exposition is clear.

- **Value to the research community:** Moderate-to-high. The framework provides a reusable scaffold for practitioners deploying LLMs in decision-support contexts, and the inference-time compute scaling results are directly actionable. The publicly available code and decision-tree visualization further increase practical value.

- **Contextualization relative to prior work:** Adequate but could be stronger. The paper appropriately situates itself relative to CoT/ToT/self-consistency and forecasting literature. The connection to utility elicitation and preference learning is acknowledged but not deeply developed. The paper is appropriately modest about not comparing to all possible agentic or tool-using pipelines.