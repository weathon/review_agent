=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the core ingredients: rule generation, logistic regression, iterative refinement, and evaluation. However, “Rule Generation with Logistic Regression” slightly understates that the method is really an LLM-driven rule induction pipeline with logistic-regression-based global weighting.
- The abstract states the main problem and high-level method clearly, and it does mention the key empirical finding that direct weighted-rule inference works better than prompting the LLM with weights.
- One concern is that the abstract overreaches in phrasing like “superior performance” and “paving the way for more reliable neuro-symbolic reasoning systems” without clearly delimiting against what baselines and under what setting. The paper later shows the gains are mostly relative to LLM-based rule-generation baselines, not necessarily stronger than a simple finetuned model.

### Introduction & Motivation
- The motivation is solid: existing LLM-based hypothesis/rule generation methods often ignore rule interaction, and classical probabilistic rule learners do not naturally operate over natural language rules. That gap is meaningful and ICLR-relevant.
- The contribution statements are mostly clear, especially the emphasis on combining LLM-generated natural language rules with logistic regression and iterative refinement.
- The introduction does slightly over-claim novelty. The paper says it is “the first to explicitly combine LLMs with probabilistic methods to learn a set of weighted rules,” but the broader space includes prior neuro-symbolic and probabilistic rule-learning work, and the novelty here is more specific: LLM-generated natural language hypotheses plus probabilistic aggregation. That distinction should be stated more carefully.
- The framing would be stronger if it more directly articulated why this is not just “LLM hypothesis generation + classical calibration” but a genuine methodological contribution.

### Method / Approach
- The overall method is understandable and modular: initial rule generation, LLM-based ternary rule judgment, elastic-net logistic regression, iterative hard-example refinement, and then multiple inference modes at evaluation time.
- A key issue is reproducibility and ambiguity in the local rule-jurment pipeline. In Section 3.1, rule application is performed by prompting an LLM to output a ternary judgment \(z_{i,j}\in\{-1,0,+1\}\). But the paper does not clearly specify:
  - how the prompt enforces consistent semantics across datasets,
  - whether the LLM sees the label space during judgment,
  - how abstention is operationalized,
  - whether repeated calls are stable given the near-deterministic temperature setting.
- The coverage-based filtering criterion is under-justified. Coverage alone can keep broadly applicable but semantically weak rules, and can discard rare but highly predictive rules. The paper later uses validation accuracy to prune, but the interaction between coverage threshold \(\gamma\), rule capacity \(H\), and hard-example mining is not theoretically or empirically grounded enough.
- The logistic regression formulation is standard, but the exposition has a notable logical tension: the features are ternary rule judgments, yet the model is presented as if it learns “probabilistic weights of the rules for global selection and calibration.” It is not fully clear how abstentions are encoded in the linear model and whether they are treated as zeros or a separate indicator.
- The iterative refinement procedure is plausible, but the stopping criterion and selection strategy may bias the learned rules toward the validation set. Since hard examples are drawn from training error and the model is tuned on validation, the process risks repeated adaptive overfitting unless carefully isolated.
- A more serious issue is that the evaluation stage mixes three distinct use cases:
  1. using the logistic model directly,
  2. using the LLM to interpret rules,
  3. using the LLM to mimic probabilistic aggregation.
  These are not equally justified as “inference strategies,” and the paper’s main conclusion is that the LLM is poor at the latter two. That conclusion is reasonable, but the paper should more explicitly frame E2–E4 as diagnostic probes rather than recommended deployment strategies.
- The method would benefit from a clearer algorithm box and precise notation. As written, there are several places where the flow from candidate rule generation to final rule selection is understandable only after multiple passes.

### Experiments & Results
- The experiments do test the paper’s central claims: whether RLIE improves rule learning, whether iterative refinement helps, and whether injecting rule information back into the LLM is effective.
- The dataset choice is reasonable for a first evaluation: six real-world binary text tasks from HypoBench cover different linguistic phenomena and are aligned with hypothesis-generation settings.
- However, the baseline suite is incomplete for the strength of the claims being made at ICLR. The paper compares mainly against other LLM-based hypothesis/rule-generation methods, but does not include strong non-LLM baselines such as:
  - standard supervised classifiers on the same text inputs,
  - prompt-based chain-of-thought or structured prompting with the same backbone,
  - classical interpretable models using hand-engineered text features,
  - or a direct comparison to a simple finetuned encoder under comparable data regimes.
  This matters because the paper’s Table 1 shows a LoRA finetune baseline that is dramatically stronger on some tasks (e.g., Reviews and LLM Detect). That suggests RLIE is not a general accuracy leader on the benchmark, and the paper’s claims should be more carefully scoped.
- The main results are mixed. RLIE is competitive and often better than prior hypothesis-generation baselines, but Table 1 does not support the strongest version of the claim that it is broadly superior. On several tasks, the LoRA baseline is far better; on others, the gains over baselines are modest.
- Table 2 supports the conclusion that the linear combiner is the best way to use the learned rules, but the table presentation is confusing in the extracted text and, more importantly, the experimental design should clarify whether all LLM-inference strategies were evaluated with identical prompts, identical reasoning budget, and identical access to the rule set.
- The paper reports mean and standard deviation over at least three runs, which is good, but the statistical evidence remains thin. There are no significance tests, confidence intervals, or effect sizes, so it is hard to know how robust the modest differences are.
- The ablation coverage is incomplete. Material missing ablations include:
  - removing iterative refinement,
  - replacing elastic-net logistic regression with plain logistic regression or another selector,
  - varying rule capacity \(H\),
  - varying the hard-example size \(k\),
  - comparing coverage filtering versus no filtering,
  - and isolating the value of ternary abstention versus binary rule satisfaction.
- A particularly important missing analysis is whether the LLM-generated rules themselves are actually better after refinement, separate from the downstream logistic regression. The case study hints at this, but the main experimental section does not quantify it well.
- The choice of using a near-zero temperature for all LLM calls helps reproducibility, but it also raises questions about whether the observed behavior reflects the model’s best capability or just one deterministic prompt regime.

### Writing & Clarity
- The paper’s core idea is understandable, but the presentation is often harder to follow than necessary because the method, evaluation modes, and claims are interleaved across sections.
- Section 3 is the main clarity bottleneck. The pipeline is conceptually clean, yet the notation and step transitions are not always explicit enough to make the method fully reproducible without the appendix.
- The figures and tables are conceptually useful. Figure 1 helps organize the framework; the prompt figures in the appendix are especially valuable for understanding the experimental setup.
- The most important clarity issue is the distinction between “rule generation,” “rule judgment,” and “LLM inference with rules.” These are central to the paper, but the reader has to reconstruct the operational differences across Sections 3.1 and 3.4.
- The discussion is clearer than the method section and does a good job stating the main empirical takeaway: LLMs seem better as semantic generators/judges than as probabilistic aggregators.

### Limitations & Broader Impact
- The paper acknowledges some limitations indirectly, especially in the discussion about LLMs being unreliable for fine-grained probabilistic integration.
- Still, the paper misses several important limitations:
  - The framework depends heavily on an LLM for both rule generation and rule application, so it inherits LLM cost, latency, and variability.
  - The learned rules may be brittle outside the benchmark distribution, especially since they are derived from limited training samples and natural-language prompts.
  - The method’s gains are shown on relatively small, fixed-size splits; it is unclear how it scales with much larger datasets or noisier annotation regimes.
  - The approach may amplify biases present in the source data, especially in tasks involving deception, mental stress, or social media behavior.
- The broader impact statement is adequate but generic. A stronger discussion would address the risk of using such systems to infer sensitive attributes or to generate misleading “interpretable” explanations that may not be faithful to the true model behavior.

### Overall Assessment
RLIE is a thoughtful and potentially useful hybrid of LLM-based hypothesis generation with classical probabilistic rule aggregation, and the central empirical insight—that the linear combiner is more reliable than prompting the LLM to apply weights—is credible and interesting. That said, for ICLR’s standard, the paper’s main limitation is that the experimental evidence is not yet strong enough to support broad claims of superiority or generality: the baseline set is incomplete, the gains are modest in many settings, the finetuning baseline is much stronger on some tasks, and several important ablations are missing. The contribution stands as a solid systems-oriented step toward neuro-symbolic rule learning, but it needs a sharper positioning and more rigorous experimentation to fully meet the ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes RLIE, a four-stage framework that uses LLMs to generate natural-language rules, logistic regression with elastic-net regularization to weight and select them, iterative refinement on hard examples, and an evaluation suite that compares direct linear inference with several ways of prompting an LLM using the learned rules. The main empirical claim is that the weighted linear combiner is usually the best way to use the learned rules, while feeding rules, weights, or even the linear prediction back into an LLM often hurts performance.

### Strengths
1. **Clear and relevant problem framing for ICLR**
   - The paper tackles a timely neuro-symbolic question: how to combine LLM-generated natural-language rules with probabilistic aggregation for interpretable reasoning.
   - This is aligned with ICLR interests in LLM reasoning, interpretability, and hybrid neuro-symbolic methods.

2. **Reasonable and coherent overall framework**
   - The pipeline is logically structured: rule generation, weighting via logistic regression, hard-example-based refinement, and systematic evaluation.
   - The use of elastic net is sensible for compact rule selection and calibration, and the iterative refinement mechanism is a natural extension of error-driven hypothesis generation.

3. **Empirical comparison of inference strategies is useful**
   - The paper does more than compare against baselines; it studies several ways of using the learned rules at inference time.
   - The result that direct linear inference outperforms LLM-based re-prompting is practically informative and supports the paper’s central thesis about division of labor between LLMs and classical probabilistic models.

4. **Some evidence of reproducibility-conscious experimentation**
   - The authors specify datasets, fixed splits, repeated runs, deterministic low-temperature settings, and key hyperparameters.
   - The appendix includes prompts and sensitivity analysis for at least one threshold, which is helpful.

5. **Interpretability is a genuine advantage**
   - The learned outputs are natural-language rules with weights, and the case-study-style examples suggest the method can yield human-readable hypotheses.
   - This is a meaningful strength if the goal is auditable reasoning rather than pure predictive performance.

### Weaknesses
1. **Novelty is moderate relative to ICLR’s acceptance bar**
   - The core ingredients are not individually new: LLM-based hypothesis generation, iterative refinement, and logistic regression over binary rule activations are all established ideas.
   - The novelty is mainly in combining these pieces and evaluating prompt-based inference variants, but the conceptual leap appears incremental rather than strongly original.

2. **Limited evidence that the method is substantially better than strong baselines**
   - The paper reports improvements over a few LLM-based baselines, but the benchmark suite is narrow and appears limited to six tasks from one benchmark family.
   - Stronger non-LLM baselines or simpler alternatives are not thoroughly explored, so it is hard to judge whether RLIE is truly competitive beyond the chosen comparison set.

3. **The evaluation suggests the most interesting part may not be the LLM component**
   - The strongest result is that the linear classifier over LLM-generated rules works best, while prompting an LLM with rules and weights degrades performance.
   - This raises a concern that the paper’s main contribution is primarily a wrapper around standard logistic regression, rather than a deep advance in LLM reasoning or rule learning.

4. **Methodological ambiguity around rule satisfaction**
   - The paper says an LLM is used to judge whether each rule applies to each sample, producing ternary outputs (+1/0/-1), but does not fully specify how this judgment is made reliably or consistently across all examples.
   - This local rule-evaluation step is central to the method, yet the reliability, cost, and prompt sensitivity of these judgments are not deeply analyzed.

5. **Experimental scale is modest for ICLR**
   - Each task uses only 200 train / 200 validation / 300 test examples, which is fairly small.
   - Given the dependence on prompt quality and the benchmark’s limited scale, the results may not generalize well to broader settings.

6. **Some claims are overstated**
   - The paper states it is “the first to explicitly combine LLMs with probabilistic methods to learn a set of weighted rules,” which seems difficult to verify and may be broader than the evidence supports.
   - Claims about “superior overall performance” should be tempered because the improvements are not shown against a wide set of nontrivial baselines or across diverse domains.

7. **Clarity and consistency issues in the exposition**
   - The method description is understandable at a high level, but some notation and terminology are inconsistent (e.g., “linear regression” vs logistic regression in prompts and text).
   - The paper would benefit from a clearer explanation of how coverage, abstention, and rule accuracy interact during filtering and pruning.

### Novelty & Significance
**Novelty:** Moderate. RLIE combines known components in a fairly systematic way, and the layered evaluation of inference strategies is a useful contribution, but the method does not appear to introduce a fundamentally new learning principle.

**Clarity:** Moderate. The high-level pipeline is clear, but the paper would benefit from sharper formalization of the rule-judging process, refinement loop, and evaluation protocol. Some parts feel more like an engineering recipe than a fully specified algorithm.

**Reproducibility:** Moderately good on paper. The authors provide hyperparameters, prompts, fixed splits, and repeated runs, which is positive. However, the dependence on proprietary LLMs, prompt sensitivity, and the central but underspecified rule-judgment procedure make exact replication somewhat uncertain.

**Significance:** Moderate. The result that a transparent probabilistic combiner is preferable to re-prompting an LLM with learned rules is practically meaningful and consistent with ICLR interests in reliable reasoning. Still, the empirical scope and incremental novelty make the contribution less compelling than what is typically needed for a strong ICLR acceptance.

Overall, this looks like a solid but not yet strong-enough-for-ICLR paper: useful, well-motivated, and empirically informative, but likely below the typical ICLR bar for originality and breadth of validation.

### Suggestions for Improvement
1. **Strengthen the empirical baseline suite**
   - Compare against stronger and more varied baselines, including non-LLM interpretable models, simpler logistic-regression variants, and prompt-based methods that use the same candidate rules but different aggregation schemes.
   - Add ablations for each component: no iterative refinement, no coverage filtering, no elastic net, no LLM-based local judgment, and alternative rule selectors.

2. **Formalize and stress-test the rule-evaluation mechanism**
   - Provide a precise description of how the LLM decides rule applicability and abstention.
   - Report inter-run stability, prompt sensitivity, and cost for this step, since it is essential to the pipeline.

3. **Expand evaluation beyond one benchmark family**
   - Test on additional domains and datasets that differ more substantially in text style, noise, and label structure.
   - Include larger-scale experiments if possible, to show that the method is not tuned to a small benchmark regime.

4. **Improve analysis of why LLM prompting degrades**
   - Investigate failure modes of E2–E4 more deeply: does the model ignore weights, overfit to salient phrases, or get confused by instructions?
   - This would turn an interesting empirical observation into a stronger scientific insight.

5. **Clarify the algorithm and notation**
   - Separate logistic regression from linear regression terminology consistently.
   - Provide a concise algorithm box with inputs, outputs, and each step’s exact computation, especially for refinement and pruning.

6. **Temper the claims and sharpen the contribution**
   - Reframe the paper as a study of reliable rule aggregation for LLM-generated hypotheses, rather than emphasizing “first” or universal superiority.
   - Highlight the strongest contribution more explicitly: that classical probabilistic aggregation appears more reliable than asking an LLM to integrate weighted rules.

7. **Report efficiency and cost**
   - Since the method relies on repeated LLM calls for generation and rule judging, report total token usage, runtime, and relative cost.
   - This is important for assessing practical significance and reproducibility at ICLR standards.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add stronger non-LLM rule-learning baselines, especially classical probabilistic rule learners and sparse linear text models, because the current comparison is too narrow to support the claim that RLIE is a meaningful advance over existing rule-learning methods. On ICLR’s bar, you need to show improvement over the right category of methods, not just over a few LLM-driven hypothesis generators.

2. Add ablations that isolate each RLIE stage: rule generation only, logistic regression only, iterative refinement only, and coverage filtering only. Without these, it is not clear whether the reported gains come from the proposed framework or simply from using more data, more iterations, or a better prompt.

3. Add a direct comparison to standard text classifiers on the same datasets, such as linear models over TF-IDF, BERT-style encoders, or small instruction-tuned classifiers, because the current setup does not show that natural-language rules are competitive with simpler supervised baselines on these tasks. This is especially important since the paper’s core claim is about practical reasoning performance, not only interpretability.

4. Add cost/efficiency experiments: number of LLM calls, tokens, wall-clock time, and sensitivity to model size. The method’s value depends heavily on expensive repeated prompting, so without efficiency data the contribution may not be practically believable for ICLR reviewers.

5. Add robustness experiments across multiple random splits and a larger set of datasets or harder OOD settings. The current evaluation on six fixed splits is too small to justify claims of general robustness and generalizable inductive reasoning.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze why LLM-in-the-loop inference degrades when weights and model predictions are added. Right now the paper claims LLMs are unreliable for probabilistic integration, but it does not diagnose whether the failure is due to prompt format, token budget, calibration mismatch, or the LLM ignoring the numeric signal.

2. Quantify the contribution of rule quality versus rule diversity versus rule compactness. The paper claims the framework yields “high quality” and “compact” rules, but there is no analysis showing whether performance gains come from better rules, better selection, or simply fewer noisy rules.

3. Report calibration and confidence behavior of the logistic combiner, not just accuracy/F1. Since the method is explicitly probabilistic, calibration error and reliability curves are needed to support the claim that the linear combiner gives more robust inference.

4. Analyze error types before and after iterative refinement. The paper says hard-example mining improves the rule set, but it never shows which classes of mistakes are actually fixed, which remain, and whether refinement overfits validation errors.

5. Measure inter-iteration stability and rule set drift. For an iterative rule-generation method, ICLR reviewers would expect to know whether the learned rules converge, oscillate, or become redundant across rounds.

### Visualizations & Case Studies
1. Add a per-iteration learning curve showing validation performance, number of active rules, and rule weights across refinement rounds. This would reveal whether the method genuinely converges or just benefits from extra prompting.

2. Add a failure-case table for the LLM-based inference strategies, especially where adding weights or linear predictions makes the LLM worse than rules alone. This would expose whether the claimed limitation is systematic or anecdotal.

3. Add side-by-side examples of rules before and after refinement, including which hard examples triggered the change. Without this, the reader cannot tell whether refinement produces genuinely better hypotheses or superficial rephrasings.

4. Add a visualization of rule activation overlap and redundancy, such as a heatmap or clustering of rule judgments. This is necessary to show that the logistic regression is actually combining complementary rules rather than learning from near-duplicates.

### Obvious Next Steps
1. Extend the framework to a stronger probabilistic combiner than logistic regression and test whether the main claims survive. Since the paper itself suggests GAMs, factor graphs, or Bayesian variants, it should have at least one such comparison or prototype.

2. Evaluate whether the learned rules transfer across datasets or domains. A key promise of natural-language rules is reuse, but the paper never tests cross-task transfer or zero-shot portability of the learned rule set.

3. Add human evaluation of rule interpretability and usefulness. The paper repeatedly claims the rules are “clear” and “auditable,” but that claim is not substantiated by any human study.

4. Test whether the method remains effective when the LLM used for rule generation changes. If RLIE depends heavily on one specific model, then the framework is less general than the paper implies, and that should be made explicit.

# Final Consolidated Review
## Summary
This paper proposes RLIE, a four-stage framework for learning natural-language rules with an LLM, weighting and selecting them with elastic-net logistic regression, refining them on hard examples, and then evaluating multiple inference strategies. The most important empirical takeaway is that the learned linear combiner is usually the best way to use the rules, while prompting the LLM to incorporate rules, weights, or the linear prediction often degrades performance.

## Strengths
- The paper tackles a timely and relevant problem: how to combine LLM-generated natural-language hypotheses with probabilistic aggregation for interpretable reasoning. The overall pipeline is coherent and matches an important neuro-symbolic direction.
- The evaluation of inference strategies is genuinely informative. The finding that direct logistic-regression inference outperforms LLM-based re-prompting with rules and weights is a useful negative result and supports the paper’s “divide labor” argument.
- The outputs are interpretable in a meaningful way: compact rule sets with learned weights, plus a case study showing iterative refinement. That is a real advantage over black-box prompting if the goal is auditable reasoning.

## Weaknesses
- The empirical case is not strong enough to support the paper’s broader claims of superiority. Table 1 compares mainly against other LLM-based rule-generation methods, but a simple finetuned baseline is substantially stronger on some tasks, and the paper does not include the stronger supervised baselines needed to establish that RLIE is competitive as a general classifier.
- The local rule-judgment mechanism is central but underspecified. The method relies on prompting an LLM to assign ternary applicability labels to each rule-example pair, yet the paper does not thoroughly characterize prompt sensitivity, stability, cost, or how abstention is handled in a way that a reader can fully reproduce. This weakens confidence in the core pipeline.
- The ablation story is incomplete. The paper does not isolate the contribution of iterative refinement, coverage filtering, or elastic-net selection cleanly enough, so it is hard to tell how much each stage matters beyond the fact that the full system works reasonably well.
- The claims are a bit overextended relative to the evidence. The method is solidly positioned as a useful engineering framework, but statements about being “the first” or paving the way for more reliable neuro-symbolic reasoning are stronger than what the benchmark evidence supports.

## Nice-to-Haves
- A more explicit algorithm box with exact inputs/outputs for rule generation, rule judgment, pruning, and refinement would make the method easier to follow and replicate.
- Additional reporting of calibration, efficiency, and total LLM usage would improve practical interpretability of the framework.
- A deeper failure analysis of the LLM-based inference variants would help explain why adding weights or a linear prediction can hurt rather than help.

## Novel Insights
The most interesting insight here is not that LLMs can generate rules, but that they are comparatively unreliable when asked to do the global probabilistic aggregation over those rules. The paper’s experiments suggest a sharp division of labor: LLMs are best used for local semantic tasks such as proposing and judging candidate rules, while a simple calibrated linear model is more dependable for combining them. That is a credible and useful result, and arguably the main conceptual contribution of the work.

## Potentially Missed Related Work
- Logistic regression over rule features / logic regression — directly relevant to the weighting stage and worth positioning more precisely against classical probabilistic rule learning.
- Bayesian rule lists / sparse rule ensembles — relevant as classical interpretable rule-combination baselines, especially for the compactness and calibration claims.
- Prompt-based hypothesis generation and refinement papers such as IO Refinement and HypoGeniC — already cited and appropriate, but the paper should be clearer that RLIE extends this line with a probabilistic combiner rather than replacing it.

## Suggestions
- Add stronger non-LLM baselines, especially standard supervised text classifiers and classical interpretable linear/rule models, to show that RLIE is competitive beyond the current LLM-hypothesis-generation comparison set.
- Include ablations for each RLIE component: no iterative refinement, no coverage filtering, no elastic net, and binary vs ternary rule judgments.
- Report the number of LLM calls, token cost, and runtime, since the method’s practical value depends heavily on repeated prompting.
- Add a compact diagnostic study of the ternary rule-judgment step: prompt robustness, abstention rate, and whether judgments are stable across repeated runs.


# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
