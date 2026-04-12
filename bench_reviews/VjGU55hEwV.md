## Summary
This paper proposes RLIE, a hybrid framework for learning natural-language rules with LLMs and then combining them with an elastic-net logistic regression model. The method has four stages—rule generation, logistic weighting/selection, hard-example-driven iterative refinement, and evaluation of downstream inference strategies—and is tested on six HypoBench text classification tasks. The main empirical message is that using the learned weighted rule set directly via logistic regression works better than feeding rules and weights back into an LLM.

## Strengths
- **Clear hybrid decomposition with an explicit local/global split.** The paper does something more specific than “use an LLM plus a classifier”: the LLM is used for *local semantic operations* (generate rules; judge rule applicability with ternary outputs), while the logistic model handles *global aggregation, sparsity, and calibration*. This division is consistently formalized in Sections 2–3 and is a meaningful design choice for natural-language rule learning.
- **The evaluation is structured to probe how rules should actually be used, not just whether they can be generated.** The E1–E4 hierarchy in Section 3.4 is a useful experimental design: direct linear inference vs. LLM with rules only, rules+weights, and rules+weights+linear prediction. This is more informative than a single end-to-end accuracy number because it separates rule quality from the downstream inference mechanism.
- **The iterative refinement loop is concrete and tied to model errors.** Rather than regenerating rules blindly, RLIE mines high-error training examples using the current weighted model and uses them to revise the rule set. This is a sensible mechanism, and the Retweets case study does show a plausible refinement trajectory with improving training performance.
- **Empirically, the method appears competitive across several tasks.** In Table 1, RLIE is often among the strongest methods on the reported benchmarks, and Table 2 consistently supports the narrower claim that the learned linear combiner is more effective than re-injecting rule information into an LLM.

## Weaknesses
###: Fatal
- **The experimental setup contains a serious inconsistency about which LLMs were actually used.** Section 4.3 says: *“All experiments involving LLMs utilized gpt-4o-mini”*, but Table 1 reports baselines with **DeepSeek-V3** and RLIE with **Qwen3-Next-80B / Qwen3-235B / DeepSeek-V3**, and Table 2 reports **DeepSeek V3.2** and **Qwen3 235B**. This is not a minor wording issue; it affects the interpretation of the entire experimental section. As written, it is unclear whether:
  1. all LLM-mediated rule generation/judgment used gpt-4o-mini and the tables are mislabeled,
  2. different backbones were actually used for different methods,
  3. RLIE’s gains partly come from larger-capacity models rather than the framework itself.  
  Until this is resolved, the main comparison in Table 1 is not reliable enough to support the paper’s strongest empirical claims.

### Major:
- **The evidence for the broad conclusion about LLMs being poor at probabilistic integration is weaker than the paper claims.** Table 2 clearly shows that, under the authors’ prompting strategy, E1 (linear-only) outperforms E2–E4. That is a valid empirical result. However, the paper often escalates this into a broader claim that LLMs are generally “less reliable at fine-grained, controlled probabilistic integration” and uses this to motivate a general architectural principle. The prompts in Appendix E for E3/E4 are fairly lightweight natural-language instructions (“the weight’s magnitude reflects the pattern’s importance,” “use ... as reference”), not a particularly strong or structured test of numerical/probabilistic reasoning. So the data supports the narrower statement—*these prompting schemes underperform the linear combiner*—more strongly than the broader claim about intrinsic LLM limitations.
- **The scale of the evaluation is modest relative to the strength of the generalization/robustness claims.** Section 4.3 states fixed splits of 200 train / 200 val / 300 test per task. For six tasks this is enough for a proof-of-concept, but it is thin support for claims like “robust performance,” “generalizable superior performance,” and strong statements about reliable neuro-symbolic reasoning. The test sets are only 300 examples, and several method gaps are not huge. The paper reports means and standard deviations over repeats in Section 4.3, but the main result tables do not actually show deviations for Table 1, nor any significance analysis.
- **A key ablation is missing: how much does iterative refinement actually matter?** RLIE is presented as a four-stage framework, and iterative refinement is central to the method description. But there is no direct ablation comparing the full method against a simpler variant such as “single rule-generation round + logistic regression” using the same backbone and evaluation protocol. Without this, it is difficult to tell whether the gains come mainly from the local/global decomposition itself or specifically from the hard-example refinement loop.
- **The pruning strategy can conflict with the stated goal of learning a collaborative rule set.** In Section 3.3, when capacity is exceeded, rules are pruned by **individual accuracy on the validation set** before retraining the global combiner. This is a real methodological weakness: individually mediocre rules can still be valuable because they cover complementary subcases, while individually strong rules may be redundant. Since the paper emphasizes joint composition of rules as a central motivation, pruning by marginal accuracy is a somewhat mismatched heuristic.

### Minor
- **The treatment of ternary rule judgments is under-analyzed.** The local LLM returns \{-1, 0, +1\}, where 0 means abstain, and these values are used directly as logistic-regression features. This is reasonable, but the paper does not analyze whether abstention behaves symmetrically across classes, whether 0 is the best encoding, or whether abstention contributes materially to calibration or sparsity.
- **Interpretability claims are only partially substantiated.** The paper claims the learned rule sets are “more compact and semantically clearer” and support “knowledge discovery and human-AI consensus,” but the qualitative evidence is limited mainly to one case study in the appendix. More examples across tasks, or some human assessment of rule quality, would better support these interpretability claims.
- **Calibration is asserted more than demonstrated.** Calibration is an important part of the motivation for the probabilistic combiner, but the paper reports accuracy and macro-F1 only. If calibrated reasoning is a headline benefit, some direct calibration evidence would strengthen the case substantially.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled comparison where RLIE and the main baselines use the **same backbone model**, so the contribution of the framework is isolated from model capacity.
- Add an ablation comparing full RLIE to **generation + logistic regression without iterative refinement**.
- Include **calibration metrics** (e.g., ECE/reliability plots) for E1 vs. the LLM-based inference strategies.
- Provide **more qualitative rule examples** from at least two non-social-media tasks to support the claims about semantic clarity and auditability.
- Report **LLM call counts / cost / latency**, since RLIE requires repeated per-rule per-sample judgments and iterative refinement.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Criticism that baseline asymmetry is unfair because RLIE may be stronger.** If the experimental asymmetry favors the authors’ method, that is a legitimate concern only when it is tied to the concrete model-specification inconsistency above. Generic complaints about “unfair comparison” without that verified inconsistency were removed.
- **Requests for unrelated extra baselines based on external literature knowledge.** Suggestions such as adding specific missing related methods or claiming omission of certain prior work were removed because they rely on external completeness judgments not verifiable from the submission alone.
- **Pure reproducibility complaints about code release timing.** The paper states code will be released upon publication; under the review instructions, concerns centered on release status are not retained.
- **Formatting/style/prompt grammar nitpicks.** Issues such as minor English errors in Appendix E were removed as they do not materially affect the scientific evaluation.
- **Claim that coverage-threshold sensitivity is unstudied.** This criticism is not accurate: Appendix C/Table 4 does include a sensitivity analysis for the coverage threshold \(\gamma\), albeit on one dataset only.
- **Claim that the paper reports no variability at all.** Section 4.3 states experiments were repeated at least three times and mean/std were computed. The valid criticism is narrower: Table 1 does not actually display the stds or significance tests in the main results.

## Novel Insights
The paper’s strongest idea is not just “LLMs can generate rules,” but that natural-language rules may be most useful when LLMs are confined to **semantic interface tasks**—generation and local applicability judgments—while a classical model handles **global evidence aggregation**. That division is more compelling than the paper’s broader rhetoric about LLM limitations. At the same time, the current pruning heuristic reveals an internal tension: the method advocates collaborative rule sets, yet one of its main selection mechanisms still evaluates rules largely in isolation. This suggests the most promising next version of the work is not to replace the linear combiner with a more complex LLM prompt, but to improve the *set-level optimization* around rule retention, calibration, and interaction analysis.

## Suggestions
- **First, fix the model-usage inconsistency unambiguously.** State exactly which model is used for each role: rule generation, rule judgment, RLIE inference, and each baseline. Then align Tables 1–2 and Section 4.3 accordingly.
- **Tone down the broad claim about LLM limitations unless you test stronger prompting schemes.** Reframe the current conclusion as: under the evaluated prompting protocols, direct linear aggregation is more reliable than LLM-mediated aggregation.
- **Add the missing ablation:** initial rule generation + logistic regression, with and without iterative refinement.
- **Replace or supplement individual-accuracy pruning with a set-aware criterion,** e.g., pruning based on contribution after refitting, ablation-based importance, or elastic-net coefficients from the global model.
- **Show calibration evidence** if calibration is a core claimed benefit.
- **Expand qualitative analysis across datasets** so the interpretability claim is backed by more than a single appendix case study.