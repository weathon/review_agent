=== CALIBRATION EXAMPLE 85 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
  - Yes, broadly. The title signals that this is a benchmark centered on interactive evaluation for text-to-SQL. “Dynamic interactions” is a fair descriptor of the core idea.
- Does the abstract clearly state the problem, method, and key results?
  - Mostly yes. It clearly states the gap in existing multi-turn benchmarks, the proposed environment, the two evaluation settings, and the task suite size.
  - The reported headline result that GPT-5 completes only 8.67% / 17.00% is clear and attention-grabbing.
- Are any claims in the abstract unsupported by the paper?
  - The abstract claims BIRD-INTERACT “restores missing realism” and that the two-stage user simulator “ensures” controlled responses without leakage. The paper provides some supporting evidence, but “ensures” is stronger than what is established. The simulator analysis in Section 6 supports robustness, but not absolute elimination of leakage or full realism.
  - The claim that the suite covers “the full CRUD spectrum” should be read carefully: the paper does include DML/DDL-like operations, but the extent to which each CRUD category is equally represented is not made explicit in the abstract.

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
  - Yes. The introduction makes a strong case that single-turn text-to-SQL benchmarks miss the iterative, stateful nature of real database assistance.
  - The distinction between static dialogue transcripts and dynamic interaction is well-motivated, and the critique of SELECT-only evaluation is important for ICLR-level interest.
- Are the contributions clearly stated and accurate?
  - The three contributions are clearly enumerated.
  - However, the introduction sometimes overstates novelty. The claim that BIRD-INTERACT is the first benchmark to jointly stress SQL generation, ambiguity resolution, and dynamic interaction with both users and environments is plausible but should be phrased more cautiously, given related work on MINT, tau-bench, WebShop, Spider 2.0, and interactive text-to-SQL datasets.
- Does the introduction over-claim or under-sell?
  - It over-claims in a few places, especially around “high-fidelity” realism and “restores missing realism.”
  - At the same time, it under-explains the evaluation target for ICLR readers: the benchmark contribution is clear, but the paper does not yet fully justify why the chosen interaction design is the right abstraction of real enterprise text-to-SQL workflows versus one plausible abstraction.

### Method / Approach
- Is the method clearly described and reproducible?
  - The overall pipeline is understandable: build on LIVESQLBENCH, inject ambiguities, add follow-up tasks, and evaluate with a function-driven simulator.
  - Reproducibility is better than average for a benchmark paper because the appendix includes prompts, action space, taxonomy, and test scripts.
  - Still, several core pieces remain under-specified in the main paper:
    - How exactly ambiguity injection is selected and balanced across tasks.
    - How annotators decide that a query is “unsolvable without clarification” but recoverable after clarification.
    - How the AST-based retrieval in the simulator selects the exact SQL fragment for LOC() in edge cases.
- Are key assumptions stated and justified?
  - Some are, but not all. The key assumption that the reference SQL can be used to derive clarifications is explicitly acknowledged in Appendix D, but this is also a major methodological choice that risks making the simulator partially oracle-like.
  - The assumption that “subsequent sub-tasks are released only after successful completion” is reasonable, but the impact of this gating on measuring downstream task difficulty is not fully analyzed.
- Are there logical gaps in the derivation or reasoning?
  - A major gap is the relationship between “ground-truth SQL fragments” and user-facing clarifications. If the user simulator is grounded in reference SQL, the benchmark may favor models that align with the annotators’ decomposition rather than models that can handle more open-ended interaction.
  - The budget formulas in Section 4 are plausible, but the rationale for the exact constants in \(B = B_{base} + 2m_{amb} + 2\lambda_{pat}\) is not justified beyond design choice.
  - The normalized reward definition in Appendix F is somewhat ad hoc. The 70/30 weighting and the different reward scales between c-Interact and a-Interact are reasonable design choices, but the paper does not establish that these correspond to user utility or are robust to alternative weights.
- Are there edge cases or failure modes not discussed?
  - Yes. Important ones include:
    - Ambiguity injection could produce unnatural or overly annotation-driven tasks.
    - The function-driven simulator may reduce leakage, but it may also reduce natural conversational variability, potentially underestimating the difficulty of real users.
    - State-dependent follow-ups are interesting, but the paper does not discuss failure cases where the agent’s earlier action changes the state in unintended ways, causing evaluation ambiguity.
    - For DM/DDL tasks, executable postconditions may still admit multiple valid solutions; the paper does not deeply discuss how equivalence classes are handled.
- For theoretical claims: are proofs correct and complete?
  - No theoretical claims are central here.

### Experiments & Results
- Do the experiments actually test the paper's claims?
  - Largely yes. The experiments do test whether models can handle dynamic, ambiguous, multi-turn text-to-SQL under both conversational and agentic settings.
  - Section 5.2 and Section 6 directly probe the main claims about communication, user simulator robustness, and action patterns.
- Are baselines appropriate and fairly compared?
  - The chosen model set is strong and includes frontier systems, which is appropriate for an ICLR benchmark paper.
  - The comparison to baseline user simulators in Section 6 is relevant.
  - However, some important baseline dimensions are missing or weakly covered:
    - There is no comparison to alternative interaction policies or planner architectures beyond prompting the same frontier models.
    - The paper does not compare against a simpler benchmark construction baseline, e.g., static multi-turn with the same underlying tasks but without dynamic simulator gating.
    - For simulator evaluation, the baseline prompt-based simulator is a reasonable baseline, but the paper could also compare against a rule-based or oracle-like controlled simulator to isolate which part of the two-stage pipeline matters most.
- Are there missing ablations that would materially change conclusions?
  - Yes, several:
    - Ablation of ambiguity types: how much does each of intent ambiguity, knowledge ambiguity, and environmental ambiguity contribute to difficulty?
    - Ablation of follow-up types: state-dependent follow-ups versus simple attribute modifications versus topic pivots.
    - Ablation of the two-stage simulator: parser-only vs generator-only vs full function-driven pipeline.
    - Ablation of grounding in reference SQL: how much of simulator quality comes from access to GT SQL segments?
    - Ablation of reward design: does the ranking of methods change under different reward weights?
  - These would materially strengthen the paper’s main claims.
- Are error bars / statistical significance reported?
  - Not really. The paper mostly reports single-run results because of API cost and deterministic decoding. That is understandable, but for a benchmark paper at ICLR, the lack of uncertainty estimates limits confidence in fine-grained comparisons, especially when some result differences are small.
  - Table 3 reports correlations with p-values, which is good, but the main benchmark results in Table 2 do not include confidence intervals or significance testing.
- Do the results support the claims made, or are they cherry-picked?
  - The results do support the broad claim that the benchmark is difficult.
  - However, some analytical claims feel selectively emphasized:
    - The memory grafting result is interesting, but it is a narrow probe and not enough to substantiate a broad conclusion about “communication schema.”
    - The interaction test-time scaling result is promising, but the evidence appears limited to a subset of models and LITE tasks.
    - The “balanced strategies outperform extremes” conclusion in Appendix J is plausible, but the analysis is correlational and not causal.
- Are datasets and evaluation metrics appropriate?
  - The datasets are appropriate for the benchmark goal.
  - The use of executable test cases for correctness is a strong choice.
  - That said, the “soft exact match” for BI queries and custom postcondition tests for DM queries create somewhat heterogeneous evaluation criteria. This is acceptable for a mixed benchmark, but the paper should discuss more explicitly how comparable the reported success rates are across these task families.

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
  - Yes, several parts are harder to follow than they should be:
    - Section 3.3 and Appendix N: the function-driven user simulator is important, but the exact flow from clarification question to action choice to grounded response is only partially clear.
    - Section 4: the budget constraints are introduced clearly enough, but the implications for model behavior are not fully explained.
    - Section 6: the simulator robustness evaluation is somewhat difficult to parse because the design of UserSim-Guard, the judge protocol, and the failure-rate metric are spread across main text and appendix.
  - The paper also occasionally mixes benchmark description, methodological justification, and experimental interpretation in ways that make the central contribution less crisp.
- Are figures and tables clear and informative?
  - Conceptually, yes, especially Figure 1, Figure 3, Table 2, Table 3, and Table 4.
  - But Figure 1 is overloaded and difficult to inspect as presented in the extracted text; the underlying idea is understandable, though the benchmark’s key interaction loop would benefit from a simpler schematic.
  - Table 2 is important, but the cumulative-success framing and the distinction between priority-question and follow-up-question metrics should be explained more prominently in the main text.
  - Table 4 is useful for positioning, though some columns are dense and the benchmark comparison would be stronger with a more concise summary of only the most relevant dimensions.

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
  - Only partially. The limitations section is quite brief relative to the ambition of the paper.
  - It notes that the work is centered on text-to-SQL and may generalize to other domains, but this is more future work than a limitation.
- Are there fundamental limitations they missed?
  - Yes:
    - The benchmark’s dependence on annotations derived from reference SQL may make it less open-ended than true user interactions.
    - The user simulator is still model-mediated, so evaluation may reflect simulator design choices as much as agent quality.
    - The benchmark appears to assume that ambiguity resolution can be formalized into a small taxonomy, which may not cover many real-world interactions.
    - The focus on PostgreSQL and executable tasks may limit transferability to heterogeneous enterprise stacks, where dialects, permissions, procedures, and side effects are more varied.
- Are there failure modes or negative societal impacts not discussed?
  - The paper does not discuss any major negative societal impacts, which is acceptable for a benchmark paper, but it could better acknowledge risks of over-relying on synthetic user simulators in production-assistant evaluations.
  - There is also a potential misuse concern: benchmarks that strongly optimize for clarification behavior may encourage models to over-ask questions or manipulate interaction budgets rather than genuinely helping users.

### Overall Assessment
BIRD-INTERACT is a substantive and timely benchmark contribution for ICLR: it tackles an important gap in text-to-SQL evaluation by moving from static, single-turn settings to dynamic multi-turn interaction with ambiguity, execution feedback, and state-dependent follow-ups. The benchmark design is ambitious, the execution-based evaluation is well-motivated, and the empirical findings convincingly show that current frontier models still struggle. That said, the paper’s main weakness is that several central design choices are under-justified: the simulator’s grounding in reference SQL, the specific reward/budget scheme, and the lack of ablations leave open how much of the difficulty is inherent versus benchmark-shaped. The main claims stand, but I would want stronger ablation evidence and a more careful discussion of simulator-induced bias before being fully confident in the benchmark as a general evaluation standard.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces BIRD-INTERACT, a new interactive text-to-SQL benchmark aimed at capturing realistic multi-turn database use with clarification, execution feedback, and follow-up queries. Its core contributions are an environment that combines a database, hierarchical knowledge base, and function-driven user simulator, plus two evaluation modes: a conversational protocol-guided setting and a more agentic tool-use setting.

### Strengths
1. The benchmark targets an important gap in text-to-SQL evaluation: real-world usage is often iterative and interactive, whereas many prior benchmarks are static or single-turn. This motivation is well aligned with current ICLR interest in agentic evaluation and realistic problem settings.
2. The benchmark appears substantially more comprehensive than prior interactive text-to-SQL datasets, covering ambiguities from user queries, knowledge, and environment state, and extending beyond read-only SQL to CRUD-style tasks. The paper also claims executable test cases for correctness, which is stronger than purely string-based evaluation.
3. The function-driven user simulator is a concrete and thoughtful design choice. The paper explicitly argues that naive LLM simulators can leak ground truth or behave inconsistently, and proposes a two-stage constrained simulator to mitigate this.
4. The paper includes multiple analyses beyond leaderboard-style results, including user simulator robustness, correlation with human behavior, action-distribution analysis, memory grafting, and interaction test-time scaling. This breadth is valuable for understanding the benchmark rather than only reporting scores.
5. The paper is clearly ambitious in scope and addresses a practically relevant problem for NLIDB systems. The inclusion of both BI and DM tasks broadens impact beyond conventional text-to-SQL benchmarks.
6. Reproducibility is considered seriously: the authors describe Docker-based execution, fixed decoding settings, prompt templates, released artifacts, and explicit model aliases. That is a positive sign for a benchmark paper.

### Weaknesses
1. The empirical results mainly demonstrate that current models perform poorly, but the paper does not yet show that BIRD-INTERACT changes model rankings or provides a strong methodological lesson beyond “interaction is hard.” For an ICLR benchmark paper, the significance is clearer if the benchmark yields actionable insights or reliably distinguishes methods in a way prior benchmarks do not.
2. The evaluation protocol appears quite complex, with multiple budgets, two interaction modes, two dataset variants, and a custom simulator. While this richness is interesting, it also raises concerns about comparability and whether results are sensitive to prompt design, simulator choices, or budget heuristics.
3. The user simulator design, although thoughtful, is still LLM-mediated and partially grounded in the reference SQL. The paper acknowledges leakage concerns, but the proposed guardrails may not fully eliminate bias from simulator access to ground truth fragments. This is a key issue because benchmark validity depends heavily on simulator fidelity.
4. The paper’s claims about human alignment and simulator robustness are promising but somewhat limited in scale. For example, the human comparison is reported on 100 sampled tasks, and the simulator-guard benchmark is newly constructed by the same authors. More external validation would strengthen confidence.
5. The paper’s novelty is somewhat mixed. Interactive text-to-SQL, ambiguity handling, and tool-using evaluation all have clear antecedents in CoSQL, SParC, MINT, WebShop-style agents, and recent enterprise text-to-SQL workflows. The main novelty is the combination and scaling, rather than a fundamentally new learning paradigm.
6. Clarity is uneven in places. The manuscript is conceptually clear at a high level, but the many custom terms, budgets, and simulator steps make the evaluation protocol hard to parse quickly. For ICLR reviewers, this can make it harder to assess whether the benchmark is principled versus heavily engineered.
7. The paper seems to rely on a large number of hand-designed annotation choices and prompt instructions, but it would benefit from a more explicit analysis of annotation failure modes, edge cases, and how often ambiguity injection may introduce unnatural tasks.
8. Significance is limited by the fact that this is primarily a benchmark paper rather than a new algorithmic method. That is acceptable at ICLR only if the benchmark is clearly high-impact and methodologically rigorous; here, the case is plausible but not yet fully airtight.

### Novelty & Significance
Novelty is moderate to strong at the benchmark-design level, mainly because the paper combines dynamic user interaction, environment exploration, executable task checking, and CRUD-style text-to-SQL in a single framework. However, the ingredients are individually somewhat familiar from prior interactive and agentic benchmarks, so the novelty lies more in integration and scale than in a new conceptual breakthrough.

Significance is potentially high if the benchmark is adopted broadly, since interactive database assistants are practically important and under-evaluated. Against ICLR’s standards, the paper is closest to an accept if the authors can better justify validity, reduce concern about simulator leakage, and demonstrate that the benchmark meaningfully advances research beyond existing interactive text-to-SQL settings.

### Suggestions for Improvement
1. Provide a stronger validity study for the benchmark itself, not just model performance. For example, compare BIRD-INTERACT outcomes against human success rates and against alternative benchmark formulations to show that the task difficulty and interaction structure are realistic.
2. Add a more explicit ablation study on the benchmark design choices: ambiguity injection, knowledge-chain breaking, state-dependent follow-ups, simulator guard, and budget constraints. This would clarify which components matter most.
3. Quantify simulator leakage risk more directly. For example, measure how often the simulator response can be reconstructed from the hidden SQL fragment, and compare against a purely symbolic or rule-based baseline where possible.
4. Improve transparency around annotation quality. Report inter-annotator agreement by ambiguity type, the percentage of tasks rejected or revised during annotation, and examples of failure cases or borderline annotations.
5. Evaluate more diverse agents or prompting strategies to show that the benchmark distinguishes not just between models, but between interaction policies. This would strengthen the benchmark’s utility for future research.
6. Simplify and standardize the evaluation protocol as much as possible, or at least isolate the minimum essential parts in the main paper. ICLR readers will benefit from a cleaner articulation of what is necessary versus optional in the benchmark.
7. Include a clearer discussion of how the benchmark avoids overfitting to the annotation schema or to the simulator prompts. This is especially important because the user simulator depends on constrained mappings from question types to actions.
8. If space permits, release a compact benchmark card summarizing task types, ambiguity types, state dependence, action space, budgets, and known limitations. This would greatly improve usability for the community.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong baselines beyond prompting frontier LLMs: a trained interactive text-to-SQL agent, a planner-executor variant, and at least one benchmark-adapted method from CoSQL/MINT/SWE-SQL. Without this, the claim that BIRD-INTERACT exposes a new capability gap is not convincing versus methods designed for interaction.

2. Evaluate on prior interactive text-to-SQL benchmarks under the same agentic setup and budget protocol. ICLR reviewers will expect evidence that the benchmark is not just hard, but measures something distinct from CoSQL/SParC/MINT under a fair comparison.

3. Run ablations that isolate the contribution of each benchmark component: ambiguity injection, dynamic environment/state dependency, follow-up tasks, and the user simulator guard/function-calling design. Without these, it is unclear which part actually drives difficulty and whether the benchmark’s complexity is necessary.

4. Add human-vs-model performance on the same tasks, not only correlation of simulator behavior. The paper’s core claim is about real interactive utility, so direct comparison to human task success and interaction cost is needed.

5. Report robustness across multiple random seeds or repeated runs for agentic settings. Since the paper relies on one-shot API evaluations with tool use and long interaction traces, single-run results are too unstable to support strong claims.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much of the failure comes from ambiguity resolution versus SQL generation versus environment/tool selection. The current error analysis is too coarse; without a breakdown, it is not clear whether the benchmark measures interaction or just harder SQL.

2. Analyze whether the two-stage user simulator changes the task distribution or makes tasks easier/harder compared with a natural-language simulator. This matters because the paper claims realism and fairness, but the simulator may be partially shaping the benchmark behavior.

3. Report budget sensitivity more rigorously in _a_-Interact and _c_-Interact, including the full curve over clarification budget, action budget, and debug opportunities. The current “ITS” discussion is too limited to support claims about scaling and resource tradeoffs.

4. Analyze contamination and leakage risks between LIVESQLBENCH-derived tasks, KB snippets, and the evaluated frontier models’ training data. For ICLR, benchmark papers need a clear argument that performance is not inflated or distorted by memorization.

5. Provide a principled comparison between normalized reward and success rate, including whether the reward design changes model ranking. Right now the metric choice could be steering the interpretation of what “better interaction” means.

### Visualizations & Case Studies
1. Show full interaction traces for success and failure cases in both _c_-Interact and _a_-Interact, especially where models ask the wrong clarification or waste budget. This would reveal whether the benchmark tests strategic interaction or just exposes brittle prompting.

2. Add a taxonomy-level confusion matrix showing which ambiguity types and follow-up types are most often misresolved. Without this, the paper does not demonstrate which interaction capabilities are genuinely missing.

3. Include case studies where a model succeeds only after environment exploration versus only after user clarification. This would expose whether the action space is actually being used intelligently or whether models collapse to one mode.

4. Visualize reward accumulation over turns and the point of failure/termination for representative tasks. ICLR reviewers will want to see whether the benchmark’s dynamics reward good planning or mainly punish length.

### Obvious Next Steps
1. Evaluate a learned policy that explicitly chooses between ask, execute, retrieve, and submit actions, rather than relying on generic frontier-model prompting. If the benchmark is meant to drive future work, this is the most direct proof that the setup is actionable.

2. Train or fine-tune a lightweight interactive agent on BIRD-INTERACT-LITE and transfer it to FULL. Without a learning baseline, the paper is mostly a benchmark release plus prompt evaluation.

3. Release and evaluate against a standardized leader-board protocol with fixed budgets and reproducible trajectories. For ICLR, benchmark papers need a clear, enforceable protocol so future results are comparable.

4. Test whether the benchmark can distinguish models with stronger planning from models with stronger SQL generation. That would validate the paper’s central claim that interaction skill is a separate capability, not just a byproduct of better code generation.

# Final Consolidated Review
## Summary
This paper introduces BIRD-INTERACT, an interactive text-to-SQL benchmark that extends prior static evaluation with dynamic user clarification, execution feedback, knowledge retrieval, and state-dependent follow-up tasks. The benchmark includes two evaluation modes — a protocol-guided conversational setting and a more agentic tool-use setting — and reports that even frontier models struggle badly on the resulting tasks.

## Strengths
- The paper tackles a genuinely important gap in text-to-SQL evaluation: real database assistance is iterative, ambiguous, and often depends on execution feedback or evolving user intent, while most prior benchmarks are still effectively single-turn or static-transcript settings.
- The benchmark is broader than many predecessors because it includes ambiguity from multiple sources (user query, knowledge base, and environment), supports both BI and DM-style tasks, and uses executable test cases rather than purely string-based comparison. The inclusion of a function-driven user simulator is also a concrete attempt to address leakage and inconsistency in LLM-based simulators.

## Weaknesses
- The benchmark design is heavily hand-engineered and partially oracle-driven. In particular, the user simulator is grounded in reference SQL fragments, and the paper does not convincingly quantify how much this simplifies the interaction relative to genuine user dialogue. This is a serious validity concern because the benchmark may end up measuring compliance with the annotation scheme more than true interactive reasoning.
- The evaluation protocol is complex, but the paper provides too little ablation evidence to justify the many design choices. There is no real isolation of the effect of ambiguity injection, knowledge-chain breaking, follow-up state dependence, simulator guarding, or the reward/budget scheme. Without these, it is unclear which components are essential and which are just adding benchmark-specific difficulty.
- The empirical section is thin for a benchmark of this ambition. Results mainly show that frontier models perform poorly, but there is little evidence that the benchmark clearly separates different interaction strategies or that the observed rankings are robust under alternative budgets, reward weights, or repeated runs.
- The paper’s strongest claims about realism and human alignment are not yet fully convincing. The human comparison is limited in scale, and the simulator validation is performed on an authors-constructed robustness set, so external validity remains uncertain.

## Nice-to-Haves
- A cleaner ablation suite comparing parser-only, generator-only, and full function-driven simulation; as well as removing specific ambiguity types or follow-up types one at a time.
- A stronger validity study showing how closely BIRD-INTERACT interaction traces match human behavior, not just simulator correlation or model failure rates.
- A more explicit sensitivity analysis over budget settings and reward weights, to show that conclusions do not depend on one particular scoring design.

## Novel Insights
The most interesting idea here is not merely “interactive text-to-SQL is hard,” but that difficulty emerges from a coupled system of ambiguity resolution, environment exploration, and stateful follow-up under resource constraints. The paper’s analyses suggest that frontier models often over-rely on direct execution or premature submission rather than strategic information gathering, which is a meaningful observation if it holds up beyond this benchmark. However, the same design also risks baking in a particular interaction script, so the core question is whether BIRD-INTERACT measures a general capability or a very specific style of benchmark compliance.

## Potentially Missed Related Work
- CoSQL / SParC — relevant prior static multi-turn text-to-SQL benchmarks, useful for positioning the dynamic contribution.
- MINT — relevant because it studies multi-turn interaction with tools and language feedback, though not specifically text-to-SQL.
- τ-bench — relevant as a recent benchmark for tool-agent-user interaction under controlled settings.
- SWE-SQL — relevant for stateful, real-world SQL issue solving and interaction-heavy evaluation.
- Spider 2.0 — relevant for enterprise-style, workflow-oriented text-to-SQL evaluation.

## Suggestions
- Add a focused ablation section that isolates the contribution of each benchmark component and simulator mechanism.
- Report budget sensitivity curves and, if feasible, repeated-run variance for the agentic setting.
- Include a direct comparison to at least one learned interactive policy or benchmark-adapted interactive baseline, rather than only prompting frontier LLMs.
- Provide a clearer analysis of simulator leakage risk, including how often the reference-SQL grounding could recover the user-facing clarification.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
