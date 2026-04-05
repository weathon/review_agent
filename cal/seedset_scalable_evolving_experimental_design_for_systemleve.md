=== CALIBRATION EXAMPLE 44 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate in that the paper is about sequential experimental design for ethical testing, but it overstates “scalable” and “evolving” without the abstract/paper establishing a strong algorithmic novelty beyond a hierarchical GP plus an acquisition heuristic.
- The abstract does identify the three key ingredients: objective metrics, stakeholder preferences, and Bayesian experimental design. However, it is not very precise about what is actually new relative to prior preference BO / BOPE / active learning work.
- Several claims in the abstract are unsupported or at least not convincingly evidenced in the paper as written:
  - “the best” is asserted without clear statistical comparison or a comprehensive baseline suite.
  - “up to 2× optimal test candidates” and “1.25× improvement in coverage” are not clearly defined in the abstract and are hard to verify from the reported plots/metrics.
  - The abstract suggests broad generality, but the paper’s experiments remain fairly bespoke and rely heavily on handcrafted preference functions and LLM proxy criteria.

### Introduction & Motivation
- The motivation is strong at a high level: ethical evaluation of autonomous systems is important, and sample-efficient testing is a legitimate problem. This is well aligned with ICLR’s interest in machine learning methods for decision-making and uncertainty.
- The gap in prior work is only partially sharp. The paper argues that prior methods handle either objective metrics or subjective preferences, but it does not cleanly distinguish itself from existing preference BO / composite optimization / BOPE methods beyond a reparameterization of the problem and the use of a hierarchical surrogate.
- The introduction over-claims novelty in several places:
  - “the first framework of its kind to explicitly consider both objective and subjective ethical evaluation criteria” is too strong. The related work already cites preference learning, BOPE, and active learning over observables; the distinction here seems more about the application framing and LLM proxy use than a fundamentally new class of method.
  - It claims the dual dependency is “not explicitly acknowledged in prior works,” but the BOPE / composite optimization literature does address dependencies between outcomes and preferences to some extent.
- The contributions are stated clearly, but the third contribution (“novel joint acquisition criterion”) is not yet convincingly justified as a substantive theoretical advance over standard information gain plus preference optimization.

### Method / Approach
- The method description is not fully reproducible or internally crisp.
  - In Section 4.1, the hierarchical structure is described at a high level, but the exact probabilistic model is underspecified: what is the full generative story linking \(x \to y \to z\), what are the priors, and how are the two GPs coupled in training?
  - The paper alternates between calling the model a “hierarchical VGP,” “hierarchical GP,” and “variational Bayesian framework,” but it is not always clear what is actually variational versus what is standard GP regression.
- The acquisition function in Eq. (2) is the most important methodological piece, but it is not adequately justified:
  - \(V(x) = I(g_x; y) + \mathbb{E}[I(h_y; z) + \mathbb{E}[h_y]]\) is conceptually intuitive, but the derivation is not rigorous.
  - The final term appears to be an expectation of the latent utility, but the notation is ambiguous and the role of the “preference” term is not formally derived from a BED objective.
  - It is unclear whether the three terms are commensurate or whether any scaling/normalization is needed to prevent one term from dominating.
- Key assumptions need more scrutiny:
  - A1–A3 are very strong. In particular, A3 assumes the complete set of objective metrics is known a priori, which is often precisely what ethical evaluation struggles with in practice.
  - A2 assumes truthful, stationary preferences. That is standard in preference elicitation, but for “ethical testing” it is a major restriction and should be emphasized as such.
- The LLM-as-proxy component raises a significant methodological concern:
  - The paper says human evaluation is expensive, so it uses GPT-4o as an evaluator. But the paper’s main claims are about ethical testing of autonomous systems, and substituting an LLM for stakeholder judgment introduces a second model whose validity is not established.
  - The prompt-based proxy is task-specific and hard-coded to each domain, which limits generality.
- There are edge cases/failure modes not discussed:
  - What happens if objective metrics conflict strongly with the stated preference in ways not reflected by the handcrafted preference score?
  - How robust is the method when the preference function is non-transitive or noisy?
  - How does the method behave when the subjective criterion depends on variables not included among the “objective” observables?
- For the probabilistic modeling claims, the exposition does not fully support the complexity / inference claims. The scaling argument for VGPs is standard, but the paper does not give enough detail on inducing point choices, kernel settings, optimization, or training stability.

### Experiments & Results
- The experiments are relevant to the paper’s claims, but they do not fully validate them at the level expected for ICLR.
- Strengths:
  - The paper evaluates on multiple domains: power grid allocation, fire rescue, and optimal routing, plus a TravelMode case study and multi-stakeholder appendix.
  - There are ablations on acquisition terms and LLM specifications, which is good practice.
- Major concerns:
  - The evaluation is built around handcrafted “preference score” functions that are designed to match the intended criteria. This makes it hard to know whether the method is learning anything beyond the authors’ own scoring rule.
  - Because the ground truth preference is not available, the paper uses the same LLM-guided prompt structure and then validates with a handcrafted score function plus TrueSkill. This is a weak substitute for a real external metric.
  - The reported metrics do not convincingly establish that the method tests ethical alignment better than baselines in a principled way.
- Baselines:
  - Random and Single GP are reasonable.
  - VS-AL baselines from Keswani et al. are relevant, but the paper’s explanation of why they fail often seems to hinge on design choices the authors control (e.g., linear boundary assumptions) rather than a careful tuning/fair comparison.
  - BOPE is included in the appendix, but it is not integrated into the main experimental narrative strongly enough given that it is one of the closest prior methods.
- Missing ablations that would materially matter:
  - No clean ablation separating the value of hierarchy itself from the value of the acquisition rule.
  - No ablation on the LLM proxy versus a human oracle or a synthetic oracle with known noise.
  - No ablation on inducing-point count or GP kernel choice.
  - No sensitivity analysis on the handcrafted preference score definitions.
- Statistical reporting is insufficient for ICLR standards:
  - Five seeds is acceptable but modest.
  - Mean/std are reported in figures, but there is no evidence of significance testing or confidence intervals for the main claims.
  - The paper repeatedly claims “best” or “higher” performance without clearly quantifying effect sizes in the main text.
- The results may be cherry-picked in framing:
  - The main narrative emphasizes cases where the hierarchical model helps, but the paper also admits lower-dimensional settings where Single GP is competitive.
  - The TravelMode appendix appears more like a plausibility demo than a rigorous benchmark.

### Writing & Clarity
- The paper is often understandable at a high level, but several sections obscure the contribution.
- The main source of confusion is conceptual, not grammatical:
  - The exact distinction between objective metrics, subjective preferences, and latent utility is not consistently maintained.
  - Eq. (2) is central but not clearly parsed.
  - The connection between the LLM proxy and the preference GP is underexplained.
- Figures and tables:
  - Figures 3–6 are useful conceptually, but the axes/metrics are not always sufficiently explicit in the main text to fully interpret the claims.
  - Figure 1 gives a good overview.
  - Figure 8–10 are more like prompt templates than scientific figures; they are informative for reproducibility, but they do not strengthen the scientific argument.
- The paper’s overall clarity is weakened by the fact that the experiments are described in a very application-specific way, while the method is presented as general. More precise notation and a clearer formal problem statement would help substantially.

### Limitations & Broader Impact
- The limitations section acknowledges some real issues: scalability limits of sparse VGPs, stationary kernels, dependence on complete objective lists, and sensitivity to LLM prompts/context.
- However, it misses a few fundamental limitations:
  - The reliance on LLMs as “ethical proxies” is not just a practical limitation; it is a core validity issue. The paper does not adequately discuss the risk that the evaluator is misaligned with the intended stakeholder or encodes its own biases.
  - The framework assumes objective metrics are already known and measurable, which may exclude many ethically important dimensions.
  - Ethical evaluation is stakeholder-dependent and may involve disagreement, strategic behavior, or changing preferences; the paper’s stationary truthful preference assumption abstracts away the hardest parts.
- Broader impact discussion is thin. Since the method is framed around ethical testing of autonomous systems, it should discuss misuse risks:
  - The framework could be used to optimize systems toward appearing ethical under a surrogate evaluator rather than genuinely being so.
  - In safety-critical settings, over-reliance on proxy preferences could create false confidence.
- The ethics statement is limited but internally consistent: the paper does not involve human subjects directly. Still, because it is about ethical alignment, a more substantive discussion of deployment risks would be appropriate.

### Overall Assessment
SEED-SET is a timely and potentially useful idea: combining objective system metrics with preference modeling for sample-efficient ethical testing is a sensible direction, and the multi-domain case studies make the paper more compelling than a toy-method contribution. However, at ICLR’s bar, the paper currently falls short on methodological rigor and empirical validation. The central acquisition objective is not derived or justified tightly enough, the use of handcrafted preference scores and LLM proxies weakens the evidence for ethical alignment, and the baselines/ablations do not fully isolate the contribution of the hierarchical design. The paper is promising as an application-driven framework, but in its current form the novelty and correctness claims are not yet backed strongly enough for a confident acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SEED-SET, a Bayesian experimental design framework for system-level ethical testing of autonomous systems that jointly models objective metrics and subjective stakeholder preferences using a hierarchical variational Gaussian process (HVGP). It also introduces an acquisition strategy intended to balance exploration of objective space with exploitation of learned preferences, and validates the method on power-grid allocation, fire rescue, optimal routing, and a travel-mode case study.

### Strengths
1. The paper tackles an important and timely ICLR-relevant problem: how to evaluate ethical alignment of autonomous systems in a sample-efficient way under uncertainty and limited human feedback. The framing connects well to broader interests in trustworthy AI, preference learning, and decision-making under uncertainty.
2. The proposed hierarchical decomposition is conceptually appealing. Separating objective observables from subjective evaluations is a natural modeling choice for ethical assessment, and the paper gives a plausible rationale for why modeling preferences over outcomes rather than directly over latent scenario parameters can improve interpretability.
3. The method is scalable in spirit. Using sparse variational GPs and Bayesian experimental design is a reasonable technical direction for handling higher-dimensional scenario spaces and costly evaluations.
4. The experimental coverage is broad. The paper studies multiple domains and includes ablations on acquisition variants, stakeholder preferences, LLM evaluator settings, and BOPE-style baselines, which shows awareness of the need to stress-test the proposed pipeline.
5. The paper is transparent about several limitations, including reliance on stationary kernels, full knowledge of objective metrics, and sensitivity of LLM proxy evaluation. This is good practice and aligns with ICLR’s expectation of reflective discussion.

### Weaknesses
1. The core technical novelty is not fully convincing relative to prior work in preference learning and Bayesian optimization. The paper builds on established ideas from hierarchical modeling, pairwise preference elicitation, SVGPs, and BOPE-style preference exploration, but it is not always clear what is fundamentally new beyond combining them in a task-specific pipeline.
2. The objective of the acquisition function is under-specified and may not be theoretically grounded enough for an ICLR audience. Equation (2) is presented as a key contribution, but the derivation, assumptions, and optimization details are not clearly developed, making it difficult to assess whether the acquisition is principled or mainly heuristic.
3. Reproducibility is weaker than expected for ICLR. Although the paper mentions libraries and hardware, key experimental details are still unclear: exact kernel choices, inducing point selection, hyperparameter settings, number of iterations, budget allocation, prompt texts for all tasks, how baselines were tuned, and how many comparisons were used per run are not fully specified in the main text.
4. The evaluation is potentially undermined by the use of handcrafted “preference score” functions as stand-ins for ground truth. Since the method is evaluated largely against internally defined preferences and LLM proxy judgments, the reported gains may reflect alignment with the chosen scoring rule rather than genuine ethical improvement.
5. The use of LLMs as proxy evaluators is under-justified. The paper reports robustness checks over prompts/temperatures/models, but does not convincingly establish that the LLM outputs correspond to human ethical judgments, especially in high-stakes domains. This is a substantive concern for validity, not just an implementation detail.
6. Baseline comparison feels incomplete relative to the claims. The paper compares against random sampling, single GP, version-space active learning, and some BOPE variants, but does not clearly benchmark against stronger modern preference BO methods or ablation baselines that isolate the value of the hierarchical model from the acquisition function.
7. The experimental claims sometimes overreach the evidence. Statements such as “the first framework of its kind” and “our method performs the best” are strong, but the evidence is limited to a few simulated case studies with custom preference functions and proxy evaluators, so the generality of the claim is not established.
8. Clarity is uneven. The paper contains many conceptual repetitions, and several notation choices are hard to parse. For an ICLR submission, the exposition would benefit from a more precise problem statement, cleaner algorithm description, and a sharper separation between modeling, acquisition, and evaluation.

### Novelty & Significance
The paper has moderate novelty: the combination of hierarchical surrogate modeling, preference elicitation, and Bayesian experimental design for ethical testing is useful, but the individual ingredients are largely established. Its significance depends on whether the framework can be shown to provide a genuinely new and empirically validated capability beyond existing preference-based BO or active learning methods; as written, that case is suggestive but not yet fully persuasive. Against ICLR’s acceptance bar, this feels like a promising application-driven synthesis rather than a clearly breakthrough methodological advance.

### Suggestions for Improvement
1. Strengthen the technical contribution by formalizing the acquisition function more rigorously. Derive the objective, state what is being approximated, and explain why the joint MI-plus-preference criterion is preferable to simpler alternatives.
2. Add stronger ablations to isolate contributions. For example: HVGP without the new acquisition, acquisition with a flat GP instead of hierarchical modeling, and direct preference BO on outcomes versus the proposed two-stage model.
3. Expand comparison to more competitive baselines from preference optimization and Bayesian optimization with composite or multi-objective structure. The current baseline set is helpful but not sufficient to support broad superiority claims.
4. Improve evaluation validity by incorporating at least some human annotations or external expert judgments, even on a smaller scale, to calibrate the LLM proxy and the handcrafted preference scores.
5. Provide a more complete reproducibility package: exact prompts, hyperparameters, optimization budgets, inducing point counts, random seed protocol, baseline tuning procedures, and enough implementation detail to reproduce each case study.
6. Clarify the relationship between objective metrics and subjective preferences. It would help to state explicitly when the objective metrics are assumed known, when they are learned, and how uncertainty from the simulator versus uncertainty from stakeholder preference is propagated.
7. Tone down the “first framework” and “best” claims unless supported by broader evidence. More precise and conservative claims would be better aligned with ICLR standards.
8. Improve presentation by consolidating repeated background material and rewriting the method section around a single clean algorithmic pipeline, ideally with pseudocode and a clear complexity analysis.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong uncertainty-calibration and out-of-sample validation for the objective and subjective GPs on held-out scenarios. Right now the paper claims “sample-efficient ethical testing,” but it never shows that the learned surrogate is accurate enough to support those claims beyond the queried points.

2. Compare against more direct ICLR-relevant baselines for preference-based active learning and multi-objective Bayesian optimization, including BOPE variants on all main tasks and a tuned pairwise GP with the same acquisition budget. The current baseline set is too narrow to establish that the hierarchical design, rather than task-specific prompt engineering or modeling choices, drives the gains.

3. Run ablations that isolate the hierarchical decomposition itself: objective-only, subjective-only, direct end-to-end preference GP on x, and a non-hierarchical acquisition function using the same total model capacity. Without this, the central claim that hierarchical modeling is necessary is not convincing.

4. Evaluate sensitivity to budget, dimensionality, and label noise with full learning curves and confidence intervals across multiple budgets. ICLR reviewers will expect to know whether the method still works when the number of queries is smaller, the scenario dimension increases, or the preference oracle becomes noisier.

5. Replace or supplement LLM-proxy evaluations with human or at least human-validated judgments on a subset of scenarios. Because the main results rely on GPT-based preferences, the paper’s core ethical-testing claims are currently only as credible as the proxy itself.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how well the acquisition function balances exploration and exploitation, rather than only showing final preference scores. The paper claims a principled trade-off, but it never measures the information gain terms separately or shows that the combined criterion actually improves search behavior.

2. Analyze when the hierarchical factorization \(x \to y \to z\) is valid and when it fails. This matters because the method assumes subjective judgments depend on observables, but many ethical preferences may depend on latent scenario details not captured by the chosen metrics.

3. Provide calibration and reliability analysis for the LLM proxy across prompts, temperatures, and model variants on the same benchmark instances. The current robustness discussion is too thin for a method whose supervision signal is itself generated by an LLM.

4. Show statistical significance and effect sizes, not just mean curves, for all major comparisons. The paper makes strong claims about outperforming baselines, but without proper uncertainty analysis the improvement could be within run-to-run variability.

5. Analyze the dependence on handcrafted preference-score functions used as “ground truth” in evaluation. Since these scores define success, the paper needs to show that results are stable under alternative reasonable preference definitions, not just one tailored scoring rule.

### Visualizations & Case Studies
1. Add trajectory visualizations of the queried scenarios in both objective space and design space over time, including where the model believed the best ethical region was and where it actually sampled. This would reveal whether the acquisition rule is genuinely exploring informative trade-offs or just drifting toward one corner of the space.

2. Show failure cases where the method proposes scenarios that look optimal under the proxy but are ethically implausible or dominated under alternate criteria. ICLR reviewers will want evidence that the method does not overfit to the prompt or the proxy evaluator.

3. Visualize posterior uncertainty over the objective and subjective GPs before and after querying, especially on high-dimensional tasks. Without this, it is hard to tell whether the model is learning meaningful structure or simply fitting the sampled points.

4. Include a side-by-side case study comparing scenarios selected by SEED-SET, random search, and a strong preference-BO baseline, with the corresponding objective vectors and preference rationales. This would make it obvious whether the method is finding qualitatively better ethical test cases.

### Obvious Next Steps
1. Evaluate on at least one real human preference dataset or collect a small human study for the proxy-evaluator prompts. For ICLR, this is the clearest way to support the claim that the framework is meaningful for ethical testing rather than just prompt optimization.

2. Extend the method to settings where objectives are partially unknown or only partially observable. The current assumption that all objective metrics are known a priori is a major restriction that limits the generality of the contribution.

3. Add a principled uncertainty-aware selection rule for when to ask the LLM versus when to query a human. That would directly address the paper’s claimed goal of reducing human burden while preserving ethical fidelity.

4. Test the method on standard preference-optimization and active-learning benchmarks, not only bespoke simulation case studies. ICLR expects evidence that a proposed algorithm generalizes beyond custom-built domains.

# Final Consolidated Review
## Summary
This paper proposes SEED-SET, a Bayesian experimental design framework for system-level ethical testing that models objective system metrics and subjective stakeholder preferences in a hierarchical variational GP pipeline. The method is evaluated on several bespoke case studies in power-grid allocation, fire rescue, optimal routing, and a small travel-mode analysis, with an LLM used as a proxy preference oracle in place of human judgments.

## Strengths
- The problem framing is timely and relevant: sample-efficient ethical evaluation of autonomous systems under limited feedback is a genuine and important challenge, and the paper correctly connects it to Bayesian experimental design and preference learning.
- The hierarchical decomposition into objective observables and subjective preferences is conceptually sensible and interpretable. In domains like power allocation or rescue robotics, reasoning about preferences over measured outcomes is more natural than directly optimizing over latent scenario parameters.
- The paper does not stay at a toy level; it includes multiple application domains, acquisition ablations, LLM prompt/model/temperature ablations, and an appendix comparison to BOPE-style baselines. That breadth is helpful for stress-testing the proposed pipeline.

## Weaknesses
- The core novelty is modest relative to prior work in preference learning, BOPE, and composite optimization. The paper combines established ingredients—hierarchical surrogate modeling, pairwise preference elicitation, sparse VGPs, and Bayesian experimental design—but does not make a sufficiently strong case that the resulting method is more than a task-specific synthesis.
- The acquisition rule is under-justified. Equation (2) is presented as the central technical contribution, but the paper does not provide a rigorous derivation, clear normalization/scaling discussion, or a careful explanation of why this particular MI-plus-preference objective should be expected to outperform simpler alternatives.
- The empirical validation is weakened by the evaluation protocol. The paper repeatedly relies on handcrafted preference-score functions designed to match the intended criteria, then uses LLM proxy judgments on top of that. This makes it hard to tell whether SEED-SET is learning ethical structure or merely aligning to an internally specified scoring rule.
- The use of LLMs as evaluators is not sufficiently validated for the paper’s main ethical claims. Prompt/temperature/model ablations are useful, but they do not establish that GPT-4o-style judgments are faithful proxies for stakeholder ethics in high-stakes domains.
- Baseline coverage is not strong enough to support broad superiority claims. The paper includes some relevant baselines, but the strongest comparisons are incomplete, and there is no clean ablation that isolates the value of the hierarchical model from the value of the acquisition heuristic.
- The assumptions are very restrictive for a paper framed around ethical testing. In particular, requiring the complete set of objective metrics a priori and assuming truthful, stationary stakeholder preferences substantially narrows the scope of the problem the method can address.

## Nice-to-Haves
- A cleaner algorithmic presentation with pseudocode, explicit model factorization, and a clearer derivation of the acquisition objective.
- More direct calibration evidence for the LLM proxy and the learned GPs, ideally with a small human-annotated subset or held-out validation on preference data.
- Additional sensitivity analyses over budget, dimensionality, and noise level.

## Novel Insights
The most interesting idea here is not the individual components, but the attempt to couple two different uncertainty sources in a single sequential testing loop: uncertainty about objective outcomes and uncertainty about stakeholder utility. That is a plausible and potentially useful direction for ethical evaluation, especially when the objective landscape and the preference landscape are intertwined. However, the current implementation still reads more like a carefully engineered pipeline for bespoke benchmarks than a generally convincing new methodology for ethical testing.

## Potentially Missed Related Work
- BOPE / preference exploration work in Bayesian optimization — directly relevant because the paper’s acquisition and two-stage objective/preferences setup overlap substantially with this line of work.
- Preference-based active learning and composite Bayesian optimization — relevant for the same reason, especially for assessing whether the hierarchical modeling choice is truly necessary.
- None identified beyond those already cited for the specific ethical-testing framing.

## Suggestions
- Add a strict ablation study: hierarchical model vs flat preference GP vs objective-only vs subjective-only, all under the same query budget.
- Validate the proxy evaluator more directly, using at least a small human study or a human-verified benchmark subset.
- Report calibrated uncertainty and held-out predictive performance for both the objective GP and the subjective GP, not just final preference scores.
- Include stronger, more modern baselines from preference BO / BOPE, and tune them fairly.
- Tone down the “first” and “best” language unless supported by broader evidence than custom simulation tasks and proxy scores.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0]
Average score: 4.0
Binary outcome: Accept
