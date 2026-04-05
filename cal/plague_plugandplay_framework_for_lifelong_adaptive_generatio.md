=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals that the paper is about a plug-and-play framework for generating multi-turn jailbreaks. However, it is unusually aggressive in tone, and the acronym/title framing somewhat oversells novelty relative to the paper’s actual incremental modular composition of existing attack components.
- The abstract clearly states the problem, high-level method, and headline results. It also identifies the three phases (Primer, Planner, Finisher) and the claimed ASR gains.
- The main concern is that the abstract’s claims are stronger than the evidence later supports. In particular:
  - “state-of-the-art jailbreaking results” and “improving ASR by more than 30% across leading models” are broad claims, but the comparison setup is not uniformly apples-to-apples in the experiments section.
  - The headline numbers on o3 and Opus 4.1 depend on specific baselines and modified evaluation settings, and the paper itself later admits different finisher choices for some models.
  - The abstract frames PLAGUE as a novel framework, but the method largely orchestrates and recombines existing ideas (planning, reflection, backtracking, retrieval) rather than introducing a fundamentally new attack principle.

### Introduction & Motivation
- The problem is well-motivated in the narrow sense that multi-turn jailbreaks are an important and underexplored safety risk, and the paper correctly situates itself against single-turn and multi-turn prior work.
- The paper does identify a plausible gap: prior multi-turn attacks differ in their emphasis on planning, iterative refinement, or strategy libraries, but the authors argue there is no unifying framework that combines these components with continual adaptation.
- The contributions are stated, but not always with sufficient precision. The introduction repeatedly claims “first,” “state-of-the-art,” and “comprehensive” without clearly delimiting the scope of those claims.
- ICLR-level concern: the introduction under-addresses the scientific question of whether the framework is genuinely a new algorithmic contribution or mostly a meta-composition of existing attack modules. The paper’s contribution feels closer to a systems integration paper than a new conceptual advance, yet it does not make that distinction explicit.
- The tone occasionally over-claims. For example, the paper states PLAGUE “break[s] through the hardest safety-aligned models with ease,” which is stronger than the results can justify.

### Method / Approach
- The method is described in a mostly modular way, which is useful, but reproducibility is limited by missing operational details:
  - The exact prompts for planner/primer/finisher are deferred to appendices, but the parser artifact leaves those appendices inaccessible in the provided text, so the core attack procedure is not fully reproducible from the paper body.
  - Several critical design choices are underspecified: how the rubric scorer is calibrated, how scores are mapped to backtracking decisions, how summaries are generated, and how retrieval embeddings are updated over time.
- The logic of the three phases is understandable, but there are important ambiguities and some internal inconsistencies:
  - In Section 3.2, ASR is defined via a judge \(J\), while later the paper uses a rubric scorer \(R\), an evaluator \(J\), and StrongREJECT-based SRE interchangeably. The distinction between these roles is not cleanly maintained.
  - The paper says the final query is fed to the evaluator if the attack succeeds, otherwise the highest-scoring finisher round is used. This makes the attack-selection rule itself part of the evaluation, which can inflate ASR@K-style performance compared with a stricter end-to-end metric.
  - The rubric scorer thresholds are heuristic (7/10 for primer, 3/10 or 8/10 for finisher success), but there is no justification that these are robust across models or behaviors.
- Key assumptions are not fully justified:
  - That semantically similar goals imply similar jailbreak strategies.
  - That storing and reusing successful strategies across objectives improves generalization rather than causing overfitting to the HarmBench distribution.
  - That “lifelong learning” is the right conceptual lens; in practice this is retrieval-augmented prompt selection, not learning in the usual parameter-updating sense.
- Edge cases and failure modes are under-discussed:
  - What happens when the retrieved strategy is semantically similar but operationally harmful to the current objective?
  - How often does the planner produce plans that are too close to the target and therefore get filtered or refused early?
  - How sensitive are results to the threshold 0.6, the max of two retrieved strategies, and the summary mechanism?
- There are logical gaps in the derivation/procedure:
  - The paper claims the framework “disentangles the factors that increase attack success and relevance,” but there is no formal analysis showing that the components are independently responsible for the gains.
  - The “lifelong learning” component is not shown to produce true continual improvement over time beyond storing successful examples.

### Experiments & Results
- The experiments do test the stated claim that PLAGUE improves attack success against multiple target models under a fixed query budget.
- However, the fairness and interpretability of the comparisons are not fully convincing:
  - Several baselines are modified from their original implementations. For GOAT, the paper changes the evaluation environment, adds rubric checks after each round, disables history, and stops early on a high rubric score. Those changes may affect baseline strength in ways that are not obviously neutral.
  - For ActorBreaker, the paper limits to two actors. For Crescendo, the maximum turns are capped at six and explicit backtracking counts are removed. These may be reasonable harmonization choices, but the paper needs a more careful justification of how they preserve baseline intent.
  - The paper compares against only one single-turn baseline, AutoDAN-Turbo, even though the paper repeatedly claims superiority over both single-turn and multi-turn methods. That is acceptable for a multi-turn-centric paper, but the claim should be narrower.
- Missing ablations that would materially change conclusions:
  - No clean ablation of the lifelong retrieval memory itself versus simple in-context examples without memory retrieval.
  - No sensitivity analysis for the rubric thresholds, retrieval threshold, or number of retrieved strategies.
  - No ablation separating the effect of the planner from the effect of backtracking/reflection in a fully controlled manner across all target models.
  - No analysis of whether gains come from increased total effective query budget due to retries and backtracking.
- Statistical reporting is weak:
  - The paper says results are averaged over three runs, but does not provide confidence intervals, standard deviations, or significance tests for the main ASR/SRE results in Table 2.
  - Table 3 and Table 4 are single-point ablations without uncertainty estimates.
- The results generally support that the framework can improve attack success on the chosen benchmark, but several claims are overstated:
  - “More than 30% across leading models” is not uniformly supported across every model or metric.
  - Claims of “state-of-the-art” are hard to assess because the evaluation protocol differs across methods and because the paper mixes Bin-ASR and SRE in a way that is not always clean.
  - Figure 2’s claim that PLAGUE “scales linearly” is not convincingly established by the plotted points; it looks more like a quick rise and then plateau.
- Dataset and metric choice are broadly appropriate for a jailbreak paper:
  - HarmBench is a standard benchmark.
  - StrongREJECT is a reasonable complementary metric.
  - But the paper’s own statement that SRE and ASR are “interchangeable” is methodologically problematic, since they are not equivalent metrics.

### Writing & Clarity
- The paper is understandable at a high level, but several sections are hard to follow because the roles of the different components are overloaded.
- The clearest structural issue is the inconsistent use of notation and terminology:
  - \(R\) refers both to the Rubric Scorer and to the retrieval memory bank in different places.
  - “Evaluator,” “Rubric Scorer,” “J,” and “R” are not consistently separated.
  - The distinction between planner-generated steps, primer prompts, and finisher prompts could be much sharper.
- Figures and tables mostly convey the intended claims, but some are not fully informative:
  - Table 2 is useful, but its interpretation is complicated by the mixed metric reporting and the baseline modifications.
  - Table 3 is more convincing as an ablation, but it does not isolate all components cleanly.
  - Figure 2’s linear-scaling claim is not substantiated by the plotted trend.
  - Figure 3 suggests diversity differences, but the metric definition is not intuitive and the figure alone does not establish why diversity matters for the main claim.
- The paper’s main clarity weakness is that it often states conclusions before adequately explaining the experimental conditions that produce them.

### Limitations & Broader Impact
- The ethics statement acknowledges misuse and says the authors will release code and prompts for reproducibility. That is candid in one sense, but it does not really discuss limitations.
- Key limitations are missing or underdeveloped:
  - Dependence on benchmark-specific objectives from HarmBench.
  - Sensitivity to the quality of the attacker/evaluator models.
  - Possible brittleness of retrieval memory when the target distribution shifts.
  - The fact that the framework’s gains may partly come from evaluation-loop engineering rather than a fundamentally stronger attack principle.
- Broader impact discussion is incomplete for a paper of this kind. Since the method is explicitly designed to improve jailbreak capability, the paper should more directly discuss dual-use risk, how release could be gated, and what defensive value offsets that risk.
- A notable omission is discussion of whether the framework could also help evaluate or stress-test defenses in a controlled, safety-oriented setting, which would help justify publication at ICLR despite the obvious dual-use concerns.

### Overall Assessment
PLAGUE is a substantial and technically interesting systems paper for multi-turn jailbreak generation, and it likely does improve practical attack success on the reported benchmark under the chosen budget. However, for ICLR standards, the main weaknesses are conceptual and methodological: the contribution is more of a modular recombination of known ideas than a clearly novel learning framework; the experimental protocol is not fully fair or cleanly isolated; and the paper relies on strong claims that outpace the evidence in several places. I do think the contribution has value as an empirical red-teaming framework, but in its current form it does not quite meet the bar for a strong ICLR acceptance because the scientific novelty and evaluation rigor are not sufficiently compelling or cleanly established.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes PLAGUE, a modular, plug-and-play framework for generating multi-turn jailbreak attacks against black-box LLMs. The core idea is to separate attack generation into three phases—Planner, Primer, and Finisher—augmented with reflection, backtracking, summarization, and a retrieval-based lifelong memory of successful strategies. The authors claim substantial gains over prior multi-turn and single-turn jailbreak baselines on HarmBench, including strong reported attack success rates on several frontier models.

### Strengths
1. **Clear modular decomposition of multi-turn attack generation.**  
   The paper’s three-phase design (Planner/Primer/Finisher) is conceptually well-motivated and maps to practical concerns in multi-turn attack generation: plan initialization, contextual escalation, and final delivery. The ablation in Table 3 supports the claim that adding backtracking, reflection, planning, and strategy retrieval each contribute to improved performance.

2. **Ablation-oriented framing with multiple components evaluated.**  
   The paper does more than report a single method: it studies the impact of planner retrieval, reflection, backtracking, and module substitution (e.g., using Crescendo as a finisher). This aligns with ICLR reviewers’ preference for understanding which components matter rather than only presenting an end-to-end system.

3. **Attempts at efficiency analysis, not just success rate.**  
   The paper reports target-LLM calls, evaluator calls, and planner calls, and argues that PLAGUE is competitive in budget usage relative to baselines. This is valuable because multi-turn attacks are often evaluated only by success rate, without considering query cost.

4. **Evaluation on several strong models.**  
   The paper tests on multiple contemporary models, including OpenAI o3/o1, Claude Opus 4.1, Deepseek-R1, and Llama 3.3-70B, which is more informative than a narrow evaluation on a single target.

5. **Potentially useful conceptual insight for defensive evaluation.**  
   Regardless of the offensive framing, the paper identifies design factors that appear to improve multi-turn vulnerability exploitation: semantic plan initialization, context management, and strategy retrieval. These insights could help defensive researchers stress-test model robustness in a structured way.

### Weaknesses
1. **Limited novelty relative to prior agentic and lifelong-learning jailbreak work.**  
   The paper’s main ingredients—planning, reflection, retrieval of prior successful strategies, backtracking, and multi-turn escalation—are already present in some form across prior work cited in the paper (e.g., Crescendo, GOAT, ActorBreaker, AutoDAN-Turbo, AutoRedTeamer, Reflexion-style loops). PLAGUE appears more like a re-combination of known ideas than a clearly new algorithmic advance, which may be below ICLR’s novelty bar unless supported by deeper analysis or a more principled formulation.

2. **The empirical gains are hard to interpret due to evaluation ambiguities.**  
   The paper states that “SRE and ASR are used interchangeably,” which is concerning because StrongREJECT score and binary attack success rate are different metrics. This makes the headline numbers less cleanly interpretable and complicates comparison across methods. ICLR reviewers typically expect metric definitions and reporting to be precise, especially when claims of SOTA are made.

3. **Methodological fairness of baselines is not fully convincing.**  
   Several baselines are modified from their original implementations: GOAT’s evaluation environment is changed to invoke the rubric scorer every round, Crescendo’s backtracking is removed and turns are capped, ActorBreaker is limited to K=2, and AutoDAN-Turbo is re-run with different settings. Some adaptation is reasonable, but the paper does not fully justify that these comparisons remain faithful to the original methods. This weakens the strength of the SOTA claims.

4. **Reproducibility is only partially supported.**  
   The paper provides algorithm pseudocode and mentions prompts in an appendix, which helps. However, the practical reproducibility is limited by reliance on proprietary APIs, unspecified prompt details in the main text, custom rubric/evaluator behavior, and a retrieval memory whose contents are only loosely described. The exact conditions needed to replicate the reported results are not sufficiently transparent.

5. **Some claims appear overstated relative to the evidence.**  
   The paper describes PLAGUE as “state-of-the-art” and reports very high ASRs on resistant models, but the evidence is mostly benchmark-specific and centered on a fixed budget and specific prompts/targets. It is unclear how robust these gains are across different seeds, different benchmarks, or different refusal policies. For an ICLR paper, stronger evidence of generality would be expected.

6. **Defensive significance is underdeveloped.**  
   The paper is about attack generation, but the contribution to understanding model robustness is mostly asserted rather than deeply analyzed. There is little discussion of what specific safety failure modes are revealed beyond improved attack success. ICLR typically values insight, not only benchmark optimization.

7. **Writing and presentation occasionally obscure the core contribution.**  
   The paper is readable overall, but there is substantial repetition and promotional framing. Some statements blur methodological description with performance claims, and the relationship between the framework and the substituted baselines can be confusing. This reduces clarity of the actual technical contribution.

### Novelty & Significance
**Novelty: moderate to low.** The framework combines known attack-design motifs—planning, reflection, retrieval, backtracking, and modular prompt chaining—into a unified pipeline. That integration is potentially useful, but the paper does not yet convincingly establish a conceptual breakthrough or a principled new learning formulation.

**Significance: moderate.** If the results are accurate, the paper identifies a meaningful vulnerability in multi-turn safety alignment and shows that simple modular orchestration can outperform prior methods under fixed budgets. However, because the task is adversarial and the method is an incremental composition of existing ideas, the work may sit below the typical ICLR acceptance bar unless the authors can better justify novelty, evaluation fairness, and robustness of the claims.

**Clarity: fair.** The phase decomposition is understandable, and the pseudocode helps. But metric usage, baseline modifications, and the exact role of each module need clearer exposition.

**Reproducibility: fair to weak.** The paper includes pseudocode and mentions prompts/appendices, but the reliance on proprietary models and under-specified evaluation details make exact reproduction difficult.

### Suggestions for Improvement
1. **Tighten the evaluation protocol and metric reporting.**  
   Clearly separate binary ASR from StrongREJECT scores throughout the paper, and avoid using “ASR” and “SRE” interchangeably. Report each metric consistently, with confidence intervals or standard deviations across seeds/runs.

2. **Strengthen baseline fairness and transparency.**  
   For every baseline modification, explain why it is necessary and quantify its effect. Ideally, include both official baseline results and the authors’ normalized comparison setup so readers can assess whether gains come from PLAGUE or from altered evaluation conditions.

3. **Provide a more principled novelty claim.**  
   Explicitly articulate what is new beyond recombining prior attack components. For example, formalize the lifelong-memory retrieval mechanism, show that it generalizes across objective families, or derive why the three-phase decomposition improves multi-turn attack efficiency.

4. **Add robustness and transfer analyses.**  
   Test across more benchmarks, more random seeds, and more prompt categories. Show whether retrieved strategies transfer across semantically related but not identical objectives, and whether gains persist under different judge models.

5. **Improve reproducibility artifacts.**  
   Release exact prompts, retrieval thresholds, memory contents format, scoring rubrics, and a full run configuration table. If proprietary APIs prevent exact replication, provide a strong open-source surrogate setup and a detailed ablation suite.

6. **Deepen the defensive interpretation.**  
   Add analysis of which safety behaviors are most vulnerable to the planner/primer/finisher structure, and what defenders can learn from the failures. This would increase the paper’s scientific value for ICLR beyond attack optimization alone.

7. **Reduce promotional wording and focus on evidence.**  
   Rephrase broad superiority claims to reflect the exact experimental setting, budget, and models. A more restrained presentation would improve credibility and better match ICLR’s standards for balanced scientific reporting.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a strict, fully matched comparison against the strongest multi-turn baselines under identical budgets, attacker model, evaluator model, and stopping rules. Right now several baselines are modified asymmetrically (e.g., different backtracking, history, early stopping), so the claimed SOTA gains are not convincing under ICLR’s bar for fair empirical comparison.

2. Add robustness checks with alternative judges and human validation on a sizable subset. The paper relies heavily on LLM-as-judge and a modified StrongReject-style score, so the core ASR claims are not trustworthy unless shown to persist under independent evaluators and manual review.

3. Add ablations that isolate each proposed mechanism on every target model, not just a few selected cases. The paper claims Planner, Primer, Finisher, reflection, backtracking, and retrieval all matter, but the evidence is partial; without a complete component-by-component ablation, the framework contribution is not established.

4. Add cross-dataset evaluation beyond HarmBench. ICLR reviewers will expect the method to generalize; without testing on at least one additional jailbreak/red-teaming benchmark or held-out behavior set, the claim that PLAGUE is broadly effective is too narrow.

5. Add attack-budget sensitivity curves with fixed compute across all methods. The paper reports strong results at six turns, but not how performance changes as budget varies under the same total cost, which is essential to support the efficiency and “better within comparable budget” claim.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether gains come from the framework or from the stronger attacker/evaluator stack. Since PLAGUE combines Deepseek-R1, Qwen-based judging, retrieval memory, and reflection, the paper needs attribution analysis to show the framework itself—not just stronger base models or prompt engineering—drives the improvement.

2. Quantify judge sensitivity and calibration. The paper mixes binary-ASR, StrongReject-like score, and rubric thresholds, but does not show how often small prompt or threshold changes alter success rates; without this, the reported ASR may be brittle.

3. Analyze semantic drift and goal adherence over turns. The method’s central claim is that PLAGUE maintains relevance while escalating; the paper should measure this explicitly with progression/relevance metrics over dialogue turns rather than asserting it qualitatively.

4. Analyze memory retrieval quality and retrieval failure modes. The lifelong-learning claim depends on successful strategy reuse, but there is no evidence that retrieved examples are actually the right ones, nor whether retrieval ever hurts by transferring mismatched strategies.

5. Analyze transferability of learned strategies across target models and behaviors. If successful strategies are stored in memory, the paper should show whether they help on unseen models or only on near-duplicate goals; otherwise the “lifelong learning” contribution is overstated.

### Visualizations & Case Studies
1. Add turn-by-turn trajectories for successful and failed attacks, showing both the conversation and the rubric/judge scores. This would reveal whether PLAGUE genuinely progresses through the stages or simply succeeds via a strong final prompt.

2. Add retrieval case studies showing the query, retrieved past strategies, and why each retrieved example was selected. This is needed to verify that the memory module is doing meaningful work rather than random example recycling.

3. Add failure-mode examples for each phase: Planner failure, Primer drift, and Finisher refusal. Without these, it is impossible to see where the framework breaks and whether the claimed modularity is real.

4. Add side-by-side comparisons with baselines on the same target behavior. The paper would be much more convincing if reviewers could see how PLAGUE’s dialogue differs from Crescendo/GOAT/ActorBreaker on identical goals, not just aggregate scores.

### Obvious Next Steps
1. Provide a rigorous fairness-controlled benchmark suite with identical attacker, judge, budget, and stopping settings across all baselines. This should have been in the paper because it is the minimum needed for the SOTA claim to be credible at ICLR.

2. Release and evaluate a fully reproducible protocol with fixed seeds, repeated runs, and confidence intervals. The current three-run averages are not enough for a high-variance multi-turn setting, and stronger uncertainty reporting is necessary.

3. Test whether the framework still helps when the Planner or Finisher is replaced by weaker or non-jailbreak-specific modules. This would establish whether PLAGUE is a general framework or just a rebranding of strong existing attacks.

4. Evaluate on defenses designed for multi-turn interactions, not just target models alone. Since the paper’s core claim is about bypassing safety in conversation, it should demonstrate that PLAGUE remains effective against explicit multi-turn defenses and conversation-level monitoring.

# Final Consolidated Review
## Summary
PLAGUE proposes a three-stage black-box framework for multi-turn jailbreak generation: a Planner retrieves and composes successful strategies from memory, a Primer gradually builds adversarial context, and a Finisher attempts the final jailbreak while using reflection, backtracking, and summarization. The paper reports strong attack success on HarmBench across several recent models, and it is clear that the modular setup can improve attack efficiency and success relative to the authors’ chosen baselines.

## Strengths
- The paper has a coherent modular decomposition of multi-turn attack generation into planning, context-building, and final execution, and the ablation in Table 3 does suggest that adding backtracking, reflection, planning, and strategy retrieval each help.
- The evaluation is broader than a single-model demo: the authors test on multiple current frontier models and also include efficiency measurements in terms of target/evaluator/planner calls, which is better than reporting success rates alone.

## Weaknesses
- The novelty is limited: PLAGUE is mostly a recombination of already-known attack ingredients—planning, reflection, retrieval of prior successful strategies, backtracking, and iterative prompting—rather than a clearly new attack principle. The paper does not convincingly establish that this is more than a systems-level orchestration of existing methods.
- The empirical claims are hard to trust because the benchmark protocol is not cleanly controlled. Several baselines are modified from their official setups, and the paper’s own text blurs metric distinctions by saying SRE and ASR are used interchangeably. That makes the headline SOTA comparisons materially less convincing.
- The “lifelong learning” claim is overstated. What is implemented is retrieval-augmented prompt reuse with a memory of successful strategies, not learning in any parameter-updating or genuinely continual sense. The paper does not show strong evidence that this memory generalizes beyond the HarmBench-like settings used here.
- The ablations do not fully isolate the contribution of each component. The paper shows incremental gains in selected configurations, but it does not provide a complete attribution study across models or a sensitivity analysis for key heuristics such as retrieval threshold, rubric thresholds, or the number of retrieved strategies.

## Nice-to-Haves
- A strict, fully matched comparison against the strongest multi-turn baselines under identical attacker, judge, budget, and stopping settings would make the main claim substantially more credible.
- Additional robustness checks with alternative judges or human validation on a subset of examples would help verify that the reported ASR is not overly sensitive to the chosen evaluator prompt.

## Novel Insights
The most interesting idea in the paper is not a new jailbreak primitive but the attempt to structure multi-turn attacks as a lifecycle: initialize a plan from memory, progressively shape context, then finish with a goal-conditioned final step. That framing is genuinely useful as an analytical lens, and the ablation suggests that some attack strength comes from the interaction between components rather than any single prompt trick. However, the paper also reveals how easy it is for “framework” papers in this space to look stronger than they are when they lean on modified baselines, heuristic judges, and retrospective success selection.

## Potentially Missed Related Work
- AutoRedTeamer — relevant as another lifelong/agentic red-teaming framework with adaptation across attacks.
- Reflexion / agentic reflection work — relevant because PLAGUE’s reflection and feedback loop are conceptually close.
- EasyJailbreak — relevant as a unified jailbreak framework and a source of comparable modular attack design ideas.

## Suggestions
- Report binary-ASR and StrongREJECT-derived scores separately throughout, and stop using “ASR” and “SRE” interchangeably.
- Include official-baseline numbers alongside the authors’ harmonized comparison setup, so readers can see how much of the gain comes from PLAGUE versus protocol changes.
- Add a component-isolation ablation for Planner, Primer, Finisher, reflection, backtracking, and retrieval on every target model, with uncertainty estimates.
- Provide retrieval case studies and failure cases showing when memory helps and when it hurts.
- If the goal is to justify the “lifelong learning” framing, formalize the memory mechanism more carefully and demonstrate transfer of stored strategies to genuinely unseen objectives or models.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Accept
