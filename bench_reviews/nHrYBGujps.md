## Summary
This paper introduces **BIRD-INTERACT**, a new benchmark for **dynamic, multi-turn text-to-SQL evaluation** that goes beyond static conversational transcripts and SELECT-only workloads. The benchmark combines executable databases, hierarchical knowledge bases, a function-driven user simulator, and two evaluation modes—protocol-guided (**c-Interact**) and agentic (**a-Interact**)—to test ambiguity resolution, debugging, state tracking, and follow-up reasoning across the full CRUD spectrum.

## Strengths
- **Substantially expands the scope of text-to-SQL evaluation beyond the prevailing static setup.** The benchmark does not just add dialogue turns; it explicitly combines ambiguity resolution, execution feedback, follow-up tasks, and database state changes. This is evidenced by the benchmark construction: each task has an ambiguous first sub-task, a stateful follow-up, executable test cases, and support for DML/DDL as well as BI-style queries.
- **The function-driven simulator is a concrete technical contribution, not just infrastructure glue.** The two-stage design—first mapping a model question into `AMB`, `LOC`, or `UNA`, then generating constrained responses—directly targets a known failure mode of LLM user simulators: leakage and unfairness. The paper does provide supporting evidence here: on UserSim-Guard, the proposed simulator reduces failures on unanswerable questions dramatically relative to single-pass baselines, and the simulator-to-human alignment analysis reports notably higher correlation than the baseline simulator.
- **The benchmark exposes an important capability gap that static text-to-SQL benchmarks can hide.** The headline results are specific and informative: even strong frontier models remain far from robust on these tasks, and there is a substantial drop from priority subtasks to follow-up subtasks, consistent with the intended challenge of maintaining context and handling evolving user intent.
- **The two evaluation settings are meaningfully differentiated.** `c-Interact` probes structured conversational clarification under a fixed protocol, while `a-Interact` tests tool use and planning under explicit budget constraints. This creates a useful decomposition of assistant-style versus agentic behavior rather than a single monolithic score.
- **Some of the analysis goes beyond leaderboard reporting.** In particular, the interaction test-time scaling experiment is a genuinely useful observation: increasing interaction opportunities improves performance, supporting the claim that interaction budget matters and that these tasks are not purely impossible for current models.

## Weaknesses

### Major:
- **The core reported task success rates remain partially confounded by the benchmark’s simulator/parsing layer, and the paper does not sufficiently quantify this effect.**  
  The evaluation depends critically on the two-stage user simulator, whose first stage maps free-form clarification questions into discrete actions (`AMB`, `LOC`, `UNA`). If this parser rejects a semantically valid clarification or routes it incorrectly, the evaluated model can fail despite asking a reasonable question. The paper does validate simulator robustness in Section 6, but that validation is indirect relative to the main benchmark outcomes: it measures classification-style performance on UserSim-Guard and correlation with humans on 100 sampled tasks, rather than how often benchmark failures are attributable to simulator misclassification in end-to-end runs. Since the paper’s central claims lean heavily on very low absolute success rates, a more direct attribution analysis is needed to separate **model interaction failure** from **simulator interpretation failure**.

- **Some of the causal claims in the analysis are stronger than the evidence warrants.**  
  The clearest case is the **memory grafting** analysis. The paper states that supplying GPT-5 with ambiguity-resolution histories from stronger communicators shows that “communication effectiveness often determines success” and suggests GPT-5 has a communication deficiency. The experiment does support the narrower claim that **better acquired interaction history materially helps downstream SQL success**. But it does **not by itself isolate “communication skill”** from several other possibilities, including better ambiguity coverage, better state acquisition, or better task decomposition being handed to the model. In other words, the evidence supports that interaction history quality matters; it does not cleanly identify which behavioral faculty is lacking.

- **The action-distribution interpretation in `a-Interact` over-attributes behavior to model bias without adequately disentangling the benchmark’s imposed cost structure.**  
  The paper notes that models overuse `submit` and `ask` relative to environment exploration and interprets this as evidence of trial-and-error or pretraining bias. But the benchmark itself imposes a nontrivial action economy (`execute` cost 1, `ask` cost 2, `submit` cost 3, many retrieval actions cost 0.5–1), and the total budget is formulaically tied to annotated ambiguities. This means action frequencies are shaped not just by model tendencies but by the benchmark’s own incentives. The descriptive analysis is still useful, but the stronger interpretation—especially claims about intrinsic bias or architectural tendency—needs either a sensitivity analysis over cost schemes or a more cautious framing.

- **Empirical support for comparative claims is limited by single-run evaluation.**  
  The paper explicitly states: “conducting single runs due to cost,” with deterministic decoding (`temperature=0`, `top_p=1`). Deterministic decoding reduces one source of randomness, but the benchmark is still an interactive, long-horizon setup involving multiple tools, conditional branching, and an LLM-driven simulator. Small perturbations can alter trajectories and binary end-task outcomes. This does not invalidate the broad conclusion that the benchmark is hard, but it weakens fine-grained comparative claims between models and interaction modes. At minimum, variance estimates on a representative subset would strengthen confidence in model ranking and in several interpretive claims.

### Minor
- **The realism claim is directionally right but somewhat overstated.**  
  The benchmark is clearly more realistic than static transcript-based text-to-SQL evaluation. However, the simulator remains constrained by annotated ambiguities and GT-grounded clarification sources. The paper itself acknowledges a pragmatic choice in Appendix D:  
  > “to avoid cases where certain ambiguities lack explicit annotations, the simulator is additionally provided with the reference SQL … This pragmatic design choice enhances evaluation reliability.”  
  This is reasonable for benchmark control, but it also means the interaction setting is still more structured and cleaner than real user behavior. The benchmark is best described as a **controlled approximation to interactive database work**, not as full restoration of real-world interaction realism.

- **There is no human upper-bound baseline on task completion.**  
  The paper includes human-related validation for simulator alignment and dataset quality, but not a direct human-expert performance baseline on the benchmark tasks themselves. Without that, the interpretation of “how hard” the benchmark is lacks an upper anchor: the low model scores show difficulty, but not how close or far these systems are from competent expert performance under the same protocol and budgets.

- **Error analysis is too coarse to fully support the paper’s strongest bottleneck claim.**  
  The paper reports that “over 80% of the errors were caused by incomplete ambiguity resolution,” but this bucket is broad. It would be more convincing to separate: failure to detect ambiguity, poor clarification wording, simulator rejection, incorrect use of retrieved clarification, SQL synthesis failure after successful clarification, and follow-up state-tracking failure. As written, the analysis suggests the likely bottleneck but does not localize it sharply enough.

### Trivial
- **The normalized reward weighting (70/30) is only lightly motivated in the main narrative.**  
  Since some interpretation relies on divergences between online success rate and offline reward, a brief sensitivity analysis or stronger justification would improve confidence that conclusions are not overly metric-dependent.

## Nice-to-Haves
- Add a **human expert baseline** on a representative subset under the same budgets and interfaces.
- Provide an **oracle clarification experiment** where gold clarifications are supplied upfront, to cleanly separate interaction failure from SQL generation failure.
- Include a **cost-sensitivity analysis** for `a-Interact`, varying action prices or budget formulas to test whether action-distribution conclusions are robust.
- Expand the error taxonomy to separate ambiguity detection, clarification formulation, simulator routing, SQL generation, and follow-up state tracking.
- Add side-by-side traces of **strong vs. weak interaction trajectories** on the same task, annotated at the point of divergence.
- Report subset-level variance or repeated-run stability for a representative portion of the benchmark.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about release status / existence / verifiability of cited models, tools, or benchmarks.** The paper cites these resources; per review policy, such criticisms are not valid.
- **Pure reproducibility complaints about missing prompts or configuration details.** The paper already states that prompts are provided in Appendix R and lists key decoding settings in Appendix I.3.
- **Generic criticism that the benchmark is “synthetic” because ambiguities are injected.** This is partly true in the literal sense, but the paper is explicit that it converts single-turn tasks into interactive ones via controlled ambiguity injection, and it backs this with annotation protocols, quality control, and human quality checks. The valid concern is not that it is synthetic per se, but that ecological validity remains only partially established.
- **Complaints about missing related work in adjacent dynamic-agent domains.** Without external verification, these are not reliable grounds for criticism here.
- **Resource-intensity as a core weakness.** The paper’s API-based evaluation is expensive, but this is common for modern benchmark studies of frontier models and is not itself a substantive flaw in the benchmark design.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is likely **the benchmark and simulator design itself**, whereas some of its strongest behavioral conclusions are still one step ahead of the evidence. The results convincingly show that dynamic text-to-SQL under ambiguity, debugging, and state dependence is much harder than static text-to-SQL. What is less fully established is *why* models fail: the current experiments suggest interaction bottlenecks, but do not yet cleanly disentangle failures of ambiguity detection, question formulation, simulator routing, and downstream SQL reasoning. In that sense, the paper has already built an impactful stress test, but its explanatory story would benefit from tighter causal isolation.

## Suggestions
- Add an **oracle-clarification ablation** and compare it directly to standard interaction to quantify how much failure comes from acquiring the right information versus using it.
- Audit a sample of failed episodes for **simulator-routing errors** (`AMB/LOC/UNA`) and report how often valid clarification attempts are rejected or misclassified.
- Reframe the **memory grafting** conclusion more conservatively: it shows the value of high-quality interaction history, not a clean diagnosis of “communication skill.”
- Temper or support the **action-bias** claims in `a-Interact` with a sensitivity study over action costs and budgets.
- Include at least a **small repeated-run or subset variance analysis** to support comparative statements between models.
- Provide a **human task-performance baseline** on a subset to contextualize benchmark difficulty.